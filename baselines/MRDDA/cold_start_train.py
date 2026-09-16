"""Cold-start training script for the MRDDA model.

Usage:
    cd MRDDA && python cold_start_train.py --dataset Bdataset --split_mode cold_drug

This script trains MRDDA under cold-start evaluation (cold_drug or cold_disease)
using pre-computed fold splits.  For each fold it:

  1. Builds train/val/test masks from the cold-start split.
  2. Loads the heterogeneous graph via ``load()`` from ``load_data.py``.
  3. Removes val+test positive association edges via ``remove_graph()``.
  4. Obtains node features and metapaths, then runs metapath2vec (``m2v``).
  5. Trains the model with early stopping based on validation AUC.
  6. Evaluates on ALL test pairs and saves per-fold predictions.
  7. Saves pooled predictions and a formal results JSON.

The graph retains ALL entities (drugs and diseases); only association edges for
val+test positive pairs are removed.  Drug-drug / disease-disease similarity
edges remain untouched.  The model predicts the full drug x disease matrix and
test-pair predictions are extracted afterwards.
"""
import os
import sys
import copy
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
import torch as th
from warnings import simplefilter
from sklearn.metrics import roc_auc_score, average_precision_score

# Import from existing MRDDA modules -----------------------------------------
from model import Model
from load_data import load, remove_graph
from utils import get_metrics_auc, set_seed, m2v, get_metrics

# Locate the repos directory (parent of this script's directory) so that the
# shared cold-start utility module can be imported regardless of the working
# directory from which the script is launched.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPOS_DIR = os.path.dirname(_SCRIPT_DIR)
if _REPOS_DIR not in sys.path:
    sys.path.insert(0, _REPOS_DIR)

from cold_start_utils import (
    load_cold_start_splits,
    build_masks_from_splits,
    get_test_pairs,
    save_fold_predictions,
    save_results_json,
    save_pooled_predictions,
)

# Output / split directories (equivalent to ../cold_start_results and
# ../cold_start_splits when running from inside the MRDDA folder).
_RESULTS_DIR = os.path.join(_REPOS_DIR, 'cold_start_results')
_SPLITS_DIR = os.path.join(_REPOS_DIR, 'cold_start_splits')


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def parse_args():
    """Parse command-line arguments.

    A dedicated parser is used (instead of importing ``args`` from ``args.py``)
    because ``args.py`` calls ``parse_args()`` at import time, which would
    conflict with the ``--split_mode`` flag introduced by this script.  The
    hyper-parameters mirror ``args.py`` with the modifications requested for
    cold-start training: epoch=1000, patience=200, lr=0.001,
    weight_decay=5e-4, hidden_feats=128, dropout=0.2.
    """
    parser = argparse.ArgumentParser(
        description='MRDDA cold-start training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str,
                        choices=['Bdataset', 'Cdataset', 'Fdataset'],
                        default='Bdataset',
                        help='Dataset to use.')
    parser.add_argument('--split_mode', type=str,
                        choices=['cold_drug', 'cold_disease'],
                        default='cold_drug',
                        help='Cold-start split mode.')
    parser.add_argument('--seed', default=42, type=int,
                        help='Global random seed.')
    # Training hyper-parameters
    parser.add_argument('--epoch', default=1000, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='Learning rate.')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay.')
    parser.add_argument('--patience', default=200, type=int,
                        help='Early-stopping patience (epochs without val AUC improvement).')
    # Model hyper-parameters
    parser.add_argument('--hidden_feats', default=128, type=int,
                        help='Hidden feature dimension.')
    parser.add_argument('--num_heads', default=5, type=int,
                        help='Number of GAT attention heads.')
    parser.add_argument('--dropout', default=0.2, type=float,
                        help='Dropout rate.')
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_dda_matrix(dataset):
    """Load the drug-disease association matrix.

    Bdataset is stored as a CSV (header-less); Cdataset and Fdataset are
    stored as ``.mat`` files containing the ``didr`` variable.
    """
    if dataset == 'Bdataset':
        df = pd.read_csv('./dataset/Bdataset/Bdataset_baseline.csv',
                         header=None).values
    elif dataset in ('Cdataset', 'Fdataset'):
        m = sio.loadmat(f'./dataset/{dataset}/{dataset}.mat')
        df = m['didr'].T
    else:
        raise ValueError(f'Unknown dataset: {dataset}')
    return df


def get_feature_and_metapath(dataset, g):
    """Return the node-feature dict and metapath list for a given dataset.

    Mirrors the logic in ``main.py`` but only covers the three datasets
    supported by the cold-start evaluation (B, C, F).
    """
    if dataset == 'Bdataset':
        feature = {
            'drug': g.nodes['drug'].data['h'],
            'disease': g.nodes['disease'].data['h'],
            'protein': g.nodes['protein'].data['h'],
        }
        metapath = ['disease_drug', 'drug_protein', 'protein_drug', 'drug_disease']
    else:  # Cdataset, Fdataset
        feature = {
            'drug': g.nodes['drug'].data['h'],
            'disease': g.nodes['disease'].data['h'],
        }
        metapath = ['drug_disease', 'disease_drug']
    return feature, metapath


# --------------------------------------------------------------------------- #
# Main training routine
# --------------------------------------------------------------------------- #
def train_cold_start(args):
    simplefilter(action='ignore', category=FutureWarning)
    print(args)

    device = th.device('cpu')
    print('Training on CPU')

    # --- Load DDA matrix -------------------------------------------------- #
    df = load_dda_matrix(args.dataset)
    n_drugs, n_diseases = df.shape
    print(f'Dataset: {args.dataset}, drugs={n_drugs}, diseases={n_diseases}')

    # --- Load cold-start splits ------------------------------------------ #
    folds = load_cold_start_splits(args.dataset, args.split_mode,
                                   splits_dir=_SPLITS_DIR)
    print(f'Loaded {len(folds)} folds from cold-start splits ({args.split_mode})')

    os.makedirs(_RESULTS_DIR, exist_ok=True)
    set_seed(args.seed)

    all_true, all_score = [], []
    fold_aurocs, fold_auprs = [], []

    for fold_idx, fold_data in enumerate(folds):
        fold_num = fold_idx + 1
        print(f'\n{"=" * 60}')
        print(f'Fold {fold_num}/{len(folds)}')
        print(f'{"=" * 60}')

        # (a) Build train/val/test masks from the split ------------------- #
        train_mask, val_mask, test_mask = build_masks_from_splits(
            fold_data, n_drugs, n_diseases)

        # Convert boolean masks to index tuples for tensor indexing.
        mask_train = np.where(train_mask)
        mask_train = (tuple(mask_train[0]), tuple(mask_train[1]))
        mask_val = np.where(val_mask)
        mask_val = (tuple(mask_val[0]), tuple(mask_val[1]))
        mask_test = np.where(test_mask)
        mask_test = (tuple(mask_test[0]), tuple(mask_test[1]))

        train_pos_count = int(df[mask_train].sum())
        train_neg_count = len(mask_train[0]) - train_pos_count
        test_pos_count = int(df[mask_test].sum())
        test_neg_count = len(mask_test[0]) - test_pos_count

        print(f'Train: {len(mask_train[0])} '
              f'(pos={train_pos_count}, neg={train_neg_count})')
        print(f'Val:   {len(mask_val[0])}')
        print(f'Test:  {len(mask_test[0])} '
              f'(pos={test_pos_count}, neg={test_neg_count})')

        label = th.tensor(df).float().to(device)

        # (c) Collect val+test positive pairs for edge removal ------------ #
        # Only positive association edges exist in the graph, so we filter
        # to pairs where df[d, dis] == 1 before calling remove_graph().
        val_pairs = list(zip(np.where(val_mask)[0].tolist(),
                             np.where(val_mask)[1].tolist()))
        test_pairs_all = list(zip(np.where(test_mask)[0].tolist(),
                                  np.where(test_mask)[1].tolist()))
        test_pos_arr = np.array([[d, dis] for d, dis in test_pairs_all
                                 if df[d, dis] == 1])
        val_pos_arr = np.array([[d, dis] for d, dis in val_pairs
                                if df[d, dis] == 1])
        if len(val_pos_arr) > 0 and len(test_pos_arr) > 0:
            all_holdout_pos = np.vstack([test_pos_arr, val_pos_arr])
        elif len(val_pos_arr) > 0:
            all_holdout_pos = val_pos_arr
        else:
            all_holdout_pos = test_pos_arr

        # (b) Load heterogeneous graph and remove holdout edges ----------- #
        # The graph still includes ALL entities; we only remove association
        # edges for val+test positive pairs.  Drug-drug / disease-disease
        # similarity edges remain.
        g = load(args.dataset)
        if len(all_holdout_pos) > 0:
            g = remove_graph(g, all_holdout_pos).to(device)
        else:
            g = g.to(device)

        # (d) Get features and metapaths ---------------------------------- #
        feature, metapath = get_feature_and_metapath(args.dataset, g)

        # (e) Run metapath2vec -------------------------------------------- #
        drug_emb, disease_emb = m2v(g, metapath)
        drug_emb = drug_emb.to(device)
        disease_emb = disease_emb.to(device)

        # Build model ----------------------------------------------------- #
        model = Model(etypes=g.etypes, ntypes=g.ntypes,
                      in_feats=feature['drug'].shape[1],
                      hidden_feats=args.hidden_feats,
                      num_heads=args.num_heads,
                      dropout=args.dropout)
        model.to(device)

        optimizer = th.optim.Adam(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
        optim_scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=50, min_lr=1e-5)
        criterion = th.nn.BCEWithLogitsLoss(
            pos_weight=th.tensor(train_neg_count / max(train_pos_count, 1)))
        print(f'Loss pos weight: {train_neg_count / max(train_pos_count, 1):.3f}')

        # (f) Training with early stopping based on validation AUC -------- #
        best_val_auc = 0.0
        best_model_state = None
        patience_counter = 0
        patience = args.patience

        for epoch in range(1, args.epoch + 1):
            # --- train step --- #
            model.train()
            score = model(g, feature, drug_emb, disease_emb)
            loss = criterion(score[mask_train].cpu().flatten(),
                             label[mask_train].cpu().flatten())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- validation (eval mode, no dropout/batchnorm mismatch) --- #
            model.eval()
            with th.no_grad():
                eval_score = model(g, feature, drug_emb, disease_emb)
                eval_pred = th.sigmoid(eval_score)
                val_auc, val_aupr = get_metrics_auc(
                    label[mask_val].cpu().detach().numpy(),
                    eval_pred[mask_val].cpu().detach().numpy())

            optim_scheduler.step(val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch} '
                      f'(val AUC not improved for {patience} epochs)')
                break

            if epoch % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch} Loss: {loss.item():.4f}; '
                      f'Val AUC {val_auc:.4f} (best={best_val_auc:.4f}); '
                      f'LR: {current_lr:.6f}')

        # (g) Evaluate on test set (ALL test pairs) using best model ----- #
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        model.eval()
        with th.no_grad():
            # The model predicts the FULL drug x disease matrix; we then
            # extract only the test-pair predictions.
            pred = th.sigmoid(model(g, feature, drug_emb, disease_emb))
            pred_np = pred.cpu().detach().numpy()

            test_labels = label[mask_test].cpu().numpy()
            test_preds = pred_np[mask_test]

            auroc = roc_auc_score(test_labels, test_preds)
            aupr = average_precision_score(test_labels, test_preds)

            fold_aurocs.append(auroc)
            fold_auprs.append(aupr)
            all_true.extend(test_labels.tolist())
            all_score.extend(test_preds.tolist())

            # (h) Save per-fold predictions -------------------------------- #
            test_pairs = get_test_pairs(test_mask, df)
            fold_pred_file = os.path.join(
                _RESULTS_DIR,
                f'mrdda_coldstart_{args.dataset}_{args.split_mode}'
                f'_fold{fold_num}.json')
            n_saved = save_fold_predictions(test_pairs, test_preds,
                                            fold_pred_file)
            print(f'Fold {fold_num} Test AUROC: {auroc:.4f}, AUPR: {aupr:.4f} '
                  f'({n_saved} predictions saved to {fold_pred_file})')

    # ------------------------------------------------------------------ #
    # (5) Save formal results and pooled predictions
    # ------------------------------------------------------------------ #
    overall_auc, overall_aupr, acc, f1, pre, rec, spe = get_metrics(
        np.array(all_true), np.array(all_score))

    extra_info = {
        'overall_auc': float(overall_auc),
        'overall_aupr': float(overall_aupr),
        'accuracy': float(acc),
        'f1': float(f1),
        'precision': float(pre),
        'recall': float(rec),
        'specificity': float(spe),
        'hyperparameters': {
            'epoch': args.epoch,
            'patience': args.patience,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'hidden_feats': args.hidden_feats,
            'num_heads': args.num_heads,
            'dropout': args.dropout,
            'seed': args.seed,
        },
    }

    results_file = os.path.join(
        _RESULTS_DIR,
        f'formal_results_mrdda_coldstart_{args.dataset}_{args.split_mode}.json')
    results = save_results_json(
        'MRDDA', args.dataset, args.split_mode,
        fold_aurocs, fold_auprs, all_true, all_score,
        results_file, extra_info=extra_info)
    print(f'\nResults saved to {results_file}')

    pooled_file = os.path.join(
        _RESULTS_DIR,
        f'mrdda_coldstart_{args.dataset}_{args.split_mode}_pooled.json')
    save_pooled_predictions('MRDDA', args.dataset, args.split_mode,
                            all_true, all_score, pooled_file)
    print(f'Pooled predictions saved to {pooled_file}')

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print(f'\n{"=" * 60}')
    print(f'MRDDA Cold-Start on {args.dataset} ({args.split_mode}) - '
          f'Results ({len(folds)}-fold CV)')
    print(f'  Pooled AUROC: {results["pooled_auroc"]:.4f}')
    print(f'  Pooled AUPR:  {results["pooled_aupr"]:.4f}')
    print(f'  Per-fold AUROC: {[f"{a:.4f}" for a in fold_aurocs]}')
    print(f'  Per-fold AUPR:  {[f"{a:.4f}" for a in fold_auprs]}')
    print(f'  Overall AUC: {overall_auc:.4f}, AUPR: {overall_aupr:.4f}')
    print(f'  Acc: {acc:.4f}, F1: {f1:.4f}, Pre: {pre:.4f}, '
          f'Rec: {rec:.4f}, Spe: {spe:.4f}')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    args = parse_args()
    train_cold_start(args)
