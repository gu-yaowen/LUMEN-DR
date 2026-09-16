"""Cold-start training and evaluation script for the AutoDR model.

This script trains AutoDR under entity-level cold-start settings (``cold_drug``
or ``cold_disease``) using the predefined cold-start split files produced by
the LMDDA project protocol.

Run from the ``AutoDR/code/`` directory::

    python cold_start_train.py --dataset Ldataset --split_mode cold_drug
    python cold_start_train.py --dataset Fdataset --split_mode cold_disease

Key design notes
----------------
* AutoDR's ``original_interactions`` (a.k.a. ``truth_label``) has shape
  ``(n_diseases, n_drugs)``  --  ``args.n_diseases, args.n_drugs = interactions.shape``.
* The cold-start split files encode pairs as
  ``pair_idx = drug_idx * n_diseases + disease_idx`` over a matrix of shape
  ``(n_drugs, n_diseases)``.  ``cold_start_utils.build_masks_from_splits``
  therefore returns ``(n_drugs, n_diseases)`` masks.  We simply transpose
  them (``.T``) to obtain the ``(n_diseases, n_drugs)`` orientation expected
  by AutoDR's ``BatchManager``.
* ``drug_adj`` / ``disease_adj`` are intrinsic similarity graphs (TopK on the
  full similarity matrices) and therefore include **all** entities -- they are
  identical across folds.  Only ``train_adj`` (built inside the ``BatchManager``
  from the fold-specific train mask) changes per fold.
* The original AutoDR code only has train / test.  Here we add a **validation**
  ``BatchManager`` and select the best checkpoint by validation AUROC (early
  stopping with a patience window), then evaluate on the test pairs.
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
from torch.optim.lr_scheduler import CyclicLR
from torch_scatter.scatter import scatter_add
from sklearn import metrics

# ---------------------------------------------------------------------------
# Path setup.
#   _SCRIPT_DIR  -> .../repos/AutoDR/code   (for loader / model / utils)
#   _REPOS_DIR   -> .../repos               (for cold_start_utils, splits, results)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AUTODR_DIR = os.path.dirname(_SCRIPT_DIR)
_REPOS_DIR = os.path.dirname(_AUTODR_DIR)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _REPOS_DIR)

from loader import data_preparation, BatchManager
from model import AutoDR
from utils import setup_seed, co_ratio_deg_disease_sc, co_ratio_deg_disease_sc2
import cold_start_utils

# Map AutoDR dataset names to the names used in the cold-start split files.
# Ldataset shares its data with Bdataset, so the split file is named Bdataset.
_DATASET_SPLIT_MAP = {
    'Ldataset': 'Bdataset',
    'Cdataset': 'Cdataset',
    'Fdataset': 'Fdataset',
}


def parse_args():
    parser = argparse.ArgumentParser(description="AutoDR cold-start training.")
    parser.add_argument('--dataset', type=str, default='Fdataset',
                        choices=['Ldataset', 'Cdataset', 'Fdataset'],
                        help='Dataset name (Ldataset is mapped to Bdataset for split files).')
    parser.add_argument('--split_mode', type=str, default='cold_drug',
                        choices=['cold_drug', 'cold_disease'],
                        help='Cold-start split mode.')
    parser.add_argument('--epochs', type=int, default=110, help='Max number of epochs.')
    parser.add_argument('--batch_size', type=int, default=1024 * 5, help='Batch size.')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--disease_TopK', type=int, default=4)
    parser.add_argument('--drug_TopK', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Used only by data_preparation fallback (ignored for cold-start).')
    parser.add_argument('--dim', type=int, default=64, help='Embedding size.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--trend_coeff', type=float, default=1,
                        help='Coefficient of attention.')
    parser.add_argument('--n_hops', type=int, default=2)
    parser.add_argument('--aggr', type=str, default='mean')
    parser.add_argument('--layer_sizes', nargs='?', default=[64, 64, 64])
    parser.add_argument('--bt_coeff', type=float, default=0.01)
    parser.add_argument('--all_bt_coeff', type=float, default=0.2)
    parser.add_argument('--patience', type=int, default=20,
                        help='Early-stopping patience (epochs without val AUC improvement).')
    return parser.parse_args()


def evaluate(model, manager):
    """Evaluate the model on a (val/test) BatchManager.

    Returns
    -------
    scores : np.ndarray
        Predicted probabilities for every pair in the manager (in batch order).
    labels : np.ndarray
        Ground-truth labels aligned with ``scores``.
    """
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for batch in manager.iter_batch():
            score, label = model.generate(batch)
            scores.append(score.cpu().detach().numpy())
            labels.append(label)
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    return scores, labels


def setup_model_graph(model, args):
    """Pre-compute sparse graphs, edge indices and trend tensors.

    Replicates the setup performed in the original ``main.py``.  The bipartite
    graph (``adj_mat``) is derived from the fold-specific ``train_adj``, while
    the similarity graph (``adj_mat2``) is derived from the intrinsic
    disease / drug adjacency matrices which include **all** entities.
    """
    adj_sp_norm = model.adj_mat
    adj_sp_norm2 = model.adj_mat2

    edge_index, edge_weight = adj_sp_norm._indices(), adj_sp_norm._values()
    edge_index2, edge_weight2 = adj_sp_norm2._indices(), adj_sp_norm2._values()

    model.adj_sp_norm = adj_sp_norm.to(args.device)
    model.edge_index = edge_index.to(args.device)
    model.edge_weight = edge_weight.to(args.device)

    model.adj_sp_norm2 = adj_sp_norm2.to(args.device)
    model.edge_index2 = edge_index2.to(args.device)
    model.edge_weight2 = edge_weight2.to(args.device)

    row, col = edge_index
    row2, col2 = edge_index2

    trend = co_ratio_deg_disease_sc(adj_sp_norm, edge_index, args)
    trend2 = co_ratio_deg_disease_sc2(adj_sp_norm2, edge_index2, args)

    norm_now = scatter_add(
        trend, col, dim=0, dim_size=args.n_diseases + args.n_drugs)[col]
    norm_now2 = scatter_add(
        trend2, col2, dim=0, dim_size=args.n_diseases + args.n_drugs)[col2]

    trend = args.trend_coeff * trend / norm_now + edge_weight
    trend2 = args.trend_coeff * trend2 / norm_now2 + edge_weight2

    model.trend = trend.to(args.device)
    model.trend2 = trend2.to(args.device)


def main():
    args = parse_args()
    setup_seed(args.seed)

    # Output / split directories (absolute; equivalent to ``../../`` from
    # the ``AutoDR/code/`` working directory).
    results_dir = os.path.join(_REPOS_DIR, 'cold_start_results')
    splits_dir = os.path.join(_REPOS_DIR, 'cold_start_splits')
    os.makedirs(results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load AutoDR data once.
    #    disease_adj / drug_adj are intrinsic (include ALL entities);
    #    original_interactions is (n_diseases, n_drugs);
    #    pos_weight is the global class-imbalance ratio.
    #    The train/test masks returned by data_preparation are NOT used --
    #    we build our own cold-start masks below.
    # ------------------------------------------------------------------
    print(f'Loading data for dataset={args.dataset} ...')
    disease_adj, drug_adj, original_interactions, _, _, pos_weight = data_preparation(args)
    n_diseases = args.n_diseases
    n_drugs = args.n_drugs
    print(f'  n_diseases={n_diseases}, n_drugs={n_drugs}, '
          f'positives={int(original_interactions.sum())}, pos_weight={pos_weight:.4f}')

    # ------------------------------------------------------------------
    # 2. Load cold-start splits (map dataset name for the split file).
    # ------------------------------------------------------------------
    split_dataset = _DATASET_SPLIT_MAP.get(args.dataset, args.dataset)
    folds = cold_start_utils.load_cold_start_splits(split_dataset, args.split_mode, splits_dir)
    print(f'Loaded {len(folds)} cold-start folds '
          f'(split file: {split_dataset}_{args.split_mode}_splits.json).')

    fold_aurocs, fold_auprs = [], []
    all_true_pooled, all_score_pooled = [], []

    # ------------------------------------------------------------------
    # 3. Fold loop.
    # ------------------------------------------------------------------
    for fold_num, fold_data in enumerate(folds):
        fold_id = fold_num + 1  # 1-based fold numbering for consistency
        print(f'\n{"=" * 70}')
        print(f'Fold {fold_id} / {len(folds)}  ({split_dataset} | {args.split_mode})')
        print(f'{"=" * 70}')

        # Build masks in the (n_drugs, n_diseases) convention used by
        # cold_start_utils, then transpose to (n_diseases, n_drugs) which is
        # AutoDR's convention (original_interactions[disease, drug]).
        train_mask_rd, val_mask_rd, test_mask_rd = cold_start_utils.build_masks_from_splits(
            fold_data, n_drugs, n_diseases)
        train_mask = train_mask_rd.T  # (n_diseases, n_drugs)
        val_mask = val_mask_rd.T
        test_mask = test_mask_rd.T

        print(f'  train pairs={int(train_mask.sum())}, '
              f'val pairs={int(val_mask.sum())}, '
              f'test pairs={int(test_mask.sum())}')

        # BatchManagers.
        #   train type -> builds train_adj from train associations only and
        #                 iterates over ALL pairs (labels masked outside train).
        #   test  type -> iterates only over the masked pairs (used for val/test).
        train_manager = BatchManager((train_mask, original_interactions),
                                     args.batch_size, 'train')
        val_manager = BatchManager((val_mask, original_interactions),
                                   args.batch_size, 'test')
        test_manager = BatchManager((test_mask, original_interactions),
                                    args.batch_size, 'test')

        train_adj = train_manager.train_adj

        # Build model and pre-compute graph / trend tensors.
        model = AutoDR(args, (train_manager, train_adj, disease_adj, drug_adj,
                              pos_weight)).to(args.device)
        setup_model_graph(model, args)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
        lr_scheduler = CyclicLR(optimizer, base_lr=0.1 * args.lr, max_lr=args.lr,
                                step_size_up=20, mode='exp_range', gamma=0.995,
                                cycle_momentum=False)

        # --------------------------------------------------------------
        # 4. Train with early stopping on validation AUROC.
        # --------------------------------------------------------------
        best_val_auc = -1.0
        best_state = None
        epochs_no_improve = 0
        for epoch in range(args.epochs):
            model.train()
            for batch in train_manager.iter_batch(shuffle=True):
                loss, _ = model.forward(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

            # Validate.
            val_scores, val_labels = evaluate(model, val_manager)
            try:
                val_auroc = metrics.roc_auc_score(y_true=val_labels,
                                                  y_score=val_scores)
                val_aupr = metrics.average_precision_score(y_true=val_labels,
                                                           y_score=val_scores)
            except ValueError:
                # Only one class present in val -- skip selection this epoch.
                val_auroc = -1.0
                val_aupr = -1.0

            improved = val_auroc > best_val_auc
            if improved:
                best_val_auc = val_auroc
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            print(f'  Epoch {epoch + 1:3d}/{args.epochs}  '
                  f'val_auroc={val_auroc:.5f}  val_aupr={val_aupr:.5f}  '
                  f'best_val_auroc={best_val_auc:.5f}'
                  f'{"  *" if improved else ""}')

            if epochs_no_improve >= args.patience:
                print(f'  Early stopping at epoch {epoch + 1} '
                      f'(no val AUC improvement for {args.patience} epochs).')
                break

        # --------------------------------------------------------------
        # 5. Evaluate best checkpoint on the test set (ALL test pairs).
        # --------------------------------------------------------------
        if best_state is not None:
            model.load_state_dict(best_state)
        test_scores, test_labels = evaluate(model, test_manager)

        test_auroc = metrics.roc_auc_score(y_true=test_labels, y_score=test_scores)
        test_aupr = metrics.average_precision_score(y_true=test_labels, y_score=test_scores)
        fold_aurocs.append(test_auroc)
        fold_auprs.append(test_aupr)
        print(f'  >> Fold {fold_id} test_auroc={test_auroc:.5f}  '
              f'test_aupr={test_aupr:.5f}  (best_val_auroc={best_val_auc:.5f})')

        # --------------------------------------------------------------
        # 6. Save per-fold predictions.
        #    Collect drug_id / disease_id from the test batches so that they
        #    stay aligned with test_scores (batch[0]=disease, batch[1]=drug).
        # --------------------------------------------------------------
        fold_drug_ids, fold_disease_ids = [], []
        for batch in test_manager.iter_batch():
            fold_drug_ids.extend(list(batch[1]))     # drug_id
            fold_disease_ids.extend(list(batch[0]))  # disease_id

        fold_predictions = []
        for i in range(len(test_scores)):
            fold_predictions.append({
                'drug_id': int(fold_drug_ids[i]),
                'disease_id': int(fold_disease_ids[i]),
                'true_label': int(test_labels[i]),
                'pred_prob': float(test_scores[i]),
            })

        fold_file = os.path.join(
            results_dir,
            f'autodr_coldstart_{split_dataset}_{args.split_mode}_fold{fold_id}.json')
        with open(fold_file, 'w') as f:
            json.dump(fold_predictions, f)
        print(f'  Fold predictions saved to {fold_file} ({len(fold_predictions)} pairs)')

        all_true_pooled.extend(test_labels.tolist())
        all_score_pooled.extend(test_scores.tolist())

    # ------------------------------------------------------------------
    # 7. Pooled metrics + formal results + pooled predictions.
    # ------------------------------------------------------------------
    all_true_pooled = np.array(all_true_pooled)
    all_score_pooled = np.array(all_score_pooled)

    pooled_auroc = metrics.roc_auc_score(y_true=all_true_pooled, y_score=all_score_pooled)
    pooled_aupr = metrics.average_precision_score(y_true=all_true_pooled, y_score=all_score_pooled)

    print(f'\n{"=" * 70}')
    print(f'Cold-start summary  ({args.dataset} | {args.split_mode})')
    print(f'{"=" * 70}')
    print(f'  Pooled AUROC = {pooled_auroc:.5f}   Pooled AUPR = {pooled_aupr:.5f}')
    print(f'  Per-fold AUROC: {np.mean(fold_aurocs):.5f} +/- {np.std(fold_aurocs):.5f}')
    print(f'  Per-fold AUPR : {np.mean(fold_auprs):.5f} +/- {np.std(fold_auprs):.5f}')
    for i, (a, p) in enumerate(zip(fold_aurocs, fold_auprs)):
        print(f'    fold {i}: auroc={a:.5f}  aupr={p:.5f}')

    # Formal results JSON.
    results_file = os.path.join(
        results_dir,
        f'formal_results_autodr_coldstart_{split_dataset}_{args.split_mode}.json')
    cold_start_utils.save_results_json(
        model_name='AutoDR', dataset=split_dataset, split_mode=args.split_mode,
        fold_aurocs=fold_aurocs, fold_auprs=fold_auprs,
        all_true=all_true_pooled, all_score=all_score_pooled,
        filepath=results_file,
        extra_info={
            'pooled_auroc': pooled_auroc,
            'pooled_aupr': pooled_aupr,
            'epochs': args.epochs,
            'patience': args.patience,
            'seed': args.seed,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'dim': args.dim,
            'weight_decay': args.weight_decay,
        })
    print(f'\nFormal results saved to {results_file}')

    # Pooled predictions JSON.
    pooled_file = os.path.join(
        results_dir,
        f'autodr_coldstart_{split_dataset}_{args.split_mode}_pooled.json')
    cold_start_utils.save_pooled_predictions(
        model_name='AutoDR', dataset=split_dataset, split_mode=args.split_mode,
        all_true=all_true_pooled, all_score=all_score_pooled,
        filepath=pooled_file)
    print(f'Pooled predictions saved to {pooled_file}')


if __name__ == '__main__':
    main()
