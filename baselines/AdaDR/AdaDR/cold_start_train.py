"""Cold-start training script for the AdaDR model.

Usage:
    cd AdaDR/AdaDR && python cold_start_train.py --dataset Ldataset --split_mode cold_drug
    cd AdaDR/AdaDR && python cold_start_train.py --dataset Cdataset --split_mode cold_disease

This script trains AdaDR under cold-start evaluation (cold_drug or cold_disease)
using pre-computed fold splits.  For each fold it:

  1. Builds train/val/test DataFrames from the cold-start split pair indices.
  2. Builds enc_graph from TRAIN pairs only (association graph).
  3. Builds train/val/test dec_graphs from the corresponding pairs.
  4. Trains the model with early stopping based on validation AUROC.
  5. Evaluates on ALL test pairs and saves per-fold predictions.
  6. Saves pooled predictions and a formal results JSON.

The drug_graph and disease_graph (similarity-based KNN graphs) are intrinsic
and include ALL entities -- they do not change per fold.  Only the enc_graph
(association graph) changes per fold, containing only training associations.

Dataset name mapping:
    Ldataset -> Bdataset (for split files)
    Cdataset -> Cdataset
    Fdataset -> Fdataset
"""
import os
import sys
import copy
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
import torch as th
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

from model import Net
from data import DrugDataLoader, _paths
from utils import common_loss, setup_seed

# --------------------------------------------------------------------------- #
# Locate the repos directory so that the shared cold-start utility module can
# be imported regardless of the working directory.
# _SCRIPT_DIR = .../repos/AdaDR/AdaDR
# _REPOS_DIR  = .../repos
# --------------------------------------------------------------------------- #
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPOS_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _REPOS_DIR not in sys.path:
    sys.path.insert(0, _REPOS_DIR)

from cold_start_utils import (
    load_cold_start_splits,
    save_fold_predictions,
    save_results_json,
    save_pooled_predictions,
)

_RESULTS_DIR = os.path.join(_REPOS_DIR, 'cold_start_results')
_SPLITS_DIR = os.path.join(_REPOS_DIR, 'cold_start_splits')

# AdaDR data_name -> cold-start split file name
_DATASET_SPLIT_MAP = {
    'Ldataset': 'Bdataset',
    'Cdataset': 'Cdataset',
    'Fdataset': 'Fdataset',
}


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def parse_args():
    """Parse command-line arguments.

    Hyper-parameters mirror ``drug_train.py`` with the addition of
    ``--dataset``, ``--split_mode``, and ``--patience`` for cold-start
    evaluation.
    """
    parser = argparse.ArgumentParser(
        description='AdaDR cold-start training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['Ldataset', 'Cdataset', 'Fdataset'],
                        help='Dataset name (Ldataset corresponds to Bdataset).')
    parser.add_argument('--split_mode', type=str, required=True,
                        choices=['cold_drug', 'cold_disease'],
                        help='Cold-start split mode.')
    parser.add_argument('--seed', default=125, type=int,
                        help='Global random seed.')
    parser.add_argument('--device', default=-1, type=int,
                        help='Running device. E.g. --device 0 for GPU 0, '
                             '--device -1 for CPU.')
    # Model hyper-parameters (same as drug_train.py)
    parser.add_argument('--model_activation', type=str, default='tanh')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--gcn_agg_units', type=int, default=840)
    parser.add_argument('--gcn_agg_accum', type=str, default='sum')
    parser.add_argument('--gcn_out_units', type=int, default=75)
    parser.add_argument('--gcn_agg_norm_symm', type=bool, default=True)
    parser.add_argument('--nhid1', type=int, default=500)
    parser.add_argument('--nhid2', type=int, default=75)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--share_param', default=True, action='store_true')
    parser.add_argument('--num_neighbor', type=int, default=1)
    parser.add_argument('--beta', type=float, default=0.01)
    # Training hyper-parameters
    parser.add_argument('--train_max_iter', type=int, default=4000)
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_valid_interval', type=int, default=100)
    parser.add_argument('--train_lr', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=10,
                        help='Early-stopping patience (number of validation '
                             'intervals without val AUROC improvement).')
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# ColdStartDataLoader -- lightweight loader that only builds intrinsic graphs
# and features, skipping the cross-validation graph building.
# --------------------------------------------------------------------------- #
class ColdStartDataLoader(DrugDataLoader):
    """Lightweight data loader for cold-start evaluation.

    Unlike ``DrugDataLoader``, this class does **not** build the
    cross-validation ``data_cv`` graphs.  It only loads the raw data
    (association matrix and similarity features), builds the intrinsic
    similarity graphs (drug_graph, disease_graph), and generates the node
    features.  Per-fold enc_graph and dec_graphs are built separately using
    the inherited ``_generate_enc_graph`` and ``_generate_dec_graph`` methods.
    """

    def __init__(self, name, device, symm=True, k=2):
        self._name = name
        self._device = device
        self._symm = symm
        self.num_neighbor = k
        print("Starting processing {} ...".format(self._name))
        self._dir = _paths[self._name]

        # Load raw data (association matrix + similarity features)
        self._load_raw_data(self._dir, self._name)

        # possible_rel_values for binary association data
        self.possible_rel_values = np.array([0.0, 1.0], dtype=np.float32)

        # Build intrinsic similarity graphs (include ALL entities)
        self.drug_graph, self.disease_graph = self._generate_feat_graph()
        self._generate_feat()

    def _load_raw_data(self, file_path, data_name):
        """Load association matrix and similarity features.

        Stores ``self.association_matrix``, ``self.drug_sim_features``,
        ``self.disease_sim_features``, ``self._num_drug``, and
        ``self._num_disease``.
        """
        if data_name in ['Gdataset', 'Cdataset', 'Fdataset']:
            data = sio.loadmat(file_path)
            self.association_matrix = data['didr'].T
            self.disease_sim_features = data['disease']
            self.drug_sim_features = data['drug']
        elif data_name == 'Ldataset':
            self.association_matrix = np.loadtxt(
                os.path.join(file_path, 'drug_dis.csv'), delimiter=",")
            self.disease_sim_features = np.loadtxt(
                os.path.join(file_path, 'dis_sim.csv'), delimiter=",")
            self.drug_sim_features = np.loadtxt(
                os.path.join(file_path, 'drug_sim.csv'), delimiter=",")
        elif data_name == 'lrssl':
            df = pd.read_csv(os.path.join(file_path, 'drug_dis.txt'),
                             index_col=0, delimiter='\t')
            self.association_matrix = df.values
            self.disease_sim_features = pd.read_csv(
                os.path.join(file_path, 'dis_sim.txt'),
                index_col=0, delimiter='\t').values
            self.drug_sim_features = pd.read_csv(
                os.path.join(file_path, 'drug_sim.txt'),
                index_col=0, delimiter='\t').values
        else:
            raise ValueError(f'Unknown dataset: {data_name}')

        self._num_drug = self.association_matrix.shape[0]
        self._num_disease = self.association_matrix.shape[1]
        print(f"  {data_name}: {self._num_drug} drugs, "
              f"{self._num_disease} diseases")


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def pairs_to_dataframe(pair_indices, n_diseases, association_matrix):
    """Convert flat pair indices to a DataFrame.

    Pair index = drug_idx * n_diseases + disease_idx

    Returns a DataFrame with columns: drug_id, disease_id, values
    where values are the association matrix entries (0.0 or 1.0).
    """
    pair_indices = np.array(pair_indices)
    drug_ids = pair_indices // n_diseases
    disease_ids = pair_indices % n_diseases
    values = association_matrix[drug_ids, disease_ids].astype(np.float32)
    return pd.DataFrame({
        'drug_id': drug_ids.astype(np.int64),
        'disease_id': disease_ids.astype(np.int64),
        'values': values,
    })


def build_fold_graphs(dataset, train_df, val_df, test_df):
    """Build enc_graph and dec_graphs for a single fold.

    Parameters
    ----------
    dataset : ColdStartDataLoader
        Provides _generate_enc_graph, _generate_dec_graph, _generate_pair_value.
    train_df : pd.DataFrame
        Training pairs (drug_id, disease_id, values).
    val_df : pd.DataFrame
        Validation pairs.
    test_df : pd.DataFrame
        Test pairs.

    Returns
    -------
    dict with keys:
        enc_graph        -- heterogeneous association graph (TRAIN pairs only)
        train_dec_graph  -- bipartite dec graph for training pairs
        train_truths     -- FloatTensor of training labels
        val_dec_graph    -- bipartite dec graph for validation pairs
        val_truths       -- FloatTensor of validation labels
        test_dec_graph   -- bipartite dec graph for test pairs
        test_truths      -- FloatTensor of test labels
    """
    # Convert DataFrames to (pairs, values)
    train_pairs, train_values = dataset._generate_pair_value(train_df)
    val_pairs, val_values = dataset._generate_pair_value(val_df)
    test_pairs, test_values = dataset._generate_pair_value(test_df)

    # enc_graph from TRAIN pairs only (add_support=True for normalisation)
    enc_graph = dataset._generate_enc_graph(
        train_pairs, train_values, add_support=True)

    # dec_graphs for train / val / test
    train_dec_graph = dataset._generate_dec_graph(train_pairs)
    val_dec_graph = dataset._generate_dec_graph(val_pairs)
    test_dec_graph = dataset._generate_dec_graph(test_pairs)

    return {
        'enc_graph': enc_graph,
        'train_dec_graph': train_dec_graph,
        'train_truths': th.FloatTensor(train_values),
        'val_dec_graph': val_dec_graph,
        'val_truths': th.FloatTensor(val_values),
        'test_dec_graph': test_dec_graph,
        'test_truths': th.FloatTensor(test_values),
    }


def evaluate_on_graph(args, model, enc_graph, dec_graph, truths,
                      drug_graph, drug_feat, drug_sim_feat,
                      dis_graph, dis_feat, dis_sim_feat):
    """Evaluate the model on a given enc/dec graph pair.

    Returns
    -------
    auroc : float
    aupr : float
    y_true : np.ndarray
    y_score : np.ndarray
    """
    model.eval()
    with th.no_grad():
        pred_ratings, _, _, _, _ = model(
            enc_graph.int().to(args.device),
            dec_graph.int().to(args.device),
            drug_graph, drug_sim_feat, drug_feat,
            dis_graph, dis_sim_feat, dis_feat)

    y_score = pred_ratings.view(-1).cpu().numpy()
    y_true = truths.cpu().numpy()

    auroc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)

    return auroc, aupr, y_true, y_score


def train_fold(args, dataset, fold_graphs, fold_num):
    """Train AdaDR on one fold with early stopping on validation AUROC.

    Parameters
    ----------
    args : Namespace
        Parsed arguments (mutated: src_in_units, dst_in_units, fdim_drug,
        fdim_disease, rating_vals are set from dataset).
    dataset : ColdStartDataLoader
    fold_graphs : dict
        Output of ``build_fold_graphs``.
    fold_num : int
        1-based fold number (for logging).

    Returns
    -------
    model : Net
        Best model (selected by validation AUROC).
    """
    # Set model dimensions from dataset
    args.src_in_units = dataset.drug_feature_shape[1]
    args.dst_in_units = dataset.disease_feature_shape[1]
    args.fdim_drug = dataset.drug_feature_shape[0]
    args.fdim_disease = dataset.disease_feature_shape[0]
    args.rating_vals = dataset.possible_rel_values

    # Move intrinsic data to device
    drug_graph = dataset.drug_graph.to(args.device)
    dis_graph = dataset.disease_graph.to(args.device)
    drug_sim_feat = th.FloatTensor(dataset.drug_sim_features).to(args.device)
    dis_sim_feat = th.FloatTensor(dataset.disease_sim_features).to(args.device)
    drug_feat, dis_feat = dataset.drug_feature, dataset.disease_feature

    # Build model
    model = Net(args=args)
    model = model.to(args.device)
    rel_loss = nn.BCEWithLogitsLoss()
    optimizer = th.optim.Adam(model.parameters(), lr=args.train_lr)

    # Prepare training data
    train_gt_ratings = fold_graphs['train_truths'].to(args.device)
    train_enc_graph = fold_graphs['enc_graph'].int().to(args.device)
    train_dec_graph = fold_graphs['train_dec_graph'].int().to(args.device)

    # Early stopping state
    best_val_auroc = 0.0
    best_state = None
    patience_counter = 0

    print(f"  Fold {fold_num}: Start training "
          f"(max_iter={args.train_max_iter}, lr={args.train_lr}) ...")

    for iter_idx in range(1, args.train_max_iter + 1):
        model.train()
        Two_Stage = False

        pred_ratings, drug_out, drug_sim_out, dis_out, dis_sim_out = \
            model(train_enc_graph, train_dec_graph,
                  drug_graph, drug_sim_feat, drug_feat,
                  dis_graph, dis_sim_feat, dis_feat, Two_Stage)

        pred_ratings = pred_ratings.squeeze(-1)

        # Contrastive (common) loss + BCE loss
        loss_com_drug = common_loss(drug_out, drug_sim_out)
        loss_com_dis = common_loss(dis_out, dis_sim_out)
        loss = rel_loss(pred_ratings, train_gt_ratings) + \
            args.beta * loss_com_dis + args.beta * loss_com_drug

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.train_grad_clip)
        optimizer.step()

        # Validate periodically
        if iter_idx % args.train_valid_interval == 0:
            val_auroc, val_aupr, _, _ = evaluate_on_graph(
                args, model, fold_graphs['enc_graph'],
                fold_graphs['val_dec_graph'], fold_graphs['val_truths'],
                drug_graph, drug_feat, drug_sim_feat,
                dis_graph, dis_feat, dis_sim_feat)

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            print(f"    Iter={iter_idx}, loss={loss.item():.4f}, "
                  f"val_AUROC={val_auroc:.4f}, val_AUPR={val_aupr:.4f}, "
                  f"best_val_AUROC={best_val_auroc:.4f}, "
                  f"patience={patience_counter}/{args.patience}")

            if patience_counter >= args.patience:
                print(f"    Early stopping at iter {iter_idx}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"  Fold {fold_num}: Training done. "
          f"Best val AUROC={best_val_auroc:.4f}")
    return model


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    print(args)

    args.device = th.device(args.device) if args.device >= 0 \
        else th.device('cpu')
    setup_seed(args.seed)

    # Map dataset name for split files
    split_dataset = _DATASET_SPLIT_MAP[args.dataset]

    # Create output directory
    os.makedirs(_RESULTS_DIR, exist_ok=True)

    # Load cold-start splits
    folds = load_cold_start_splits(
        split_dataset, args.split_mode, splits_dir=_SPLITS_DIR)
    print(f"Loaded {len(folds)} folds for "
          f"{split_dataset}_{args.split_mode}")

    # Load AdaDR dataset (intrinsic graphs + features + association matrix)
    dataset = ColdStartDataLoader(
        args.dataset, args.device,
        symm=args.gcn_agg_norm_symm,
        k=args.num_neighbor)
    print("AdaDR dataset loaded (intrinsic graphs and features).")

    n_drugs = dataset.num_drug
    n_diseases = dataset.num_disease
    association_matrix = dataset.association_matrix

    # Move intrinsic data to device (reused across all folds)
    drug_graph = dataset.drug_graph.to(args.device)
    dis_graph = dataset.disease_graph.to(args.device)
    drug_sim_feat = th.FloatTensor(dataset.drug_sim_features).to(args.device)
    dis_sim_feat = th.FloatTensor(dataset.disease_sim_features).to(args.device)
    drug_feat, dis_feat = dataset.drug_feature, dataset.disease_feature

    all_true, all_score = [], []
    fold_aurocs, fold_auprs = [], []

    for fold_idx, fold_data in enumerate(folds):
        fold_num = fold_idx + 1
        print(f"\n{'=' * 60}")
        print(f"Fold {fold_num}/{len(folds)}")
        print(f"{'=' * 60}")

        # --- Build DataFrames from split pair indices -------------------- #
        train_df = pairs_to_dataframe(
            fold_data['train_idx'], n_diseases, association_matrix)
        val_df = pairs_to_dataframe(
            fold_data['val_idx'], n_diseases, association_matrix)
        test_df = pairs_to_dataframe(
            fold_data['test_idx'], n_diseases, association_matrix)

        train_pos = int((train_df['values'] == 1).sum())
        val_pos = int((val_df['values'] == 1).sum())
        test_pos = int((test_df['values'] == 1).sum())
        print(f"Train: {len(train_df)} "
              f"(pos={train_pos}, neg={len(train_df) - train_pos})")
        print(f"Val:   {len(val_df)} "
              f"(pos={val_pos}, neg={len(val_df) - val_pos})")
        print(f"Test:  {len(test_df)} "
              f"(pos={test_pos}, neg={len(test_df) - test_pos})")

        # --- Build per-fold graphs --------------------------------------- #
        fold_graphs = build_fold_graphs(dataset, train_df, val_df, test_df)

        # --- Train with early stopping ----------------------------------- #
        model = train_fold(args, dataset, fold_graphs, fold_num)

        # --- Evaluate on test set ---------------------------------------- #
        test_auroc, test_aupr, y_true, y_score = evaluate_on_graph(
            args, model, fold_graphs['enc_graph'],
            fold_graphs['test_dec_graph'], fold_graphs['test_truths'],
            drug_graph, drug_feat, drug_sim_feat,
            dis_graph, dis_feat, dis_sim_feat)

        fold_aurocs.append(test_auroc)
        fold_auprs.append(test_aupr)
        all_true.extend(y_true.tolist())
        all_score.extend(y_score.tolist())

        print(f"  Fold {fold_num} Test: "
              f"AUROC={test_auroc:.4f}, AUPR={test_aupr:.4f}")

        # --- Save per-fold predictions ----------------------------------- #
        fold_pred_file = os.path.join(
            _RESULTS_DIR,
            f'adadr_coldstart_{split_dataset}_{args.split_mode}'
            f'_fold{fold_num}.json')

        # Build test pairs list (drug_id, disease_id, true_label)
        test_pairs = []
        for idx in range(len(y_true)):
            test_pairs.append({
                'drug_id': int(test_df.iloc[idx]['drug_id']),
                'disease_id': int(test_df.iloc[idx]['disease_id']),
                'true_label': int(y_true[idx]),
            })
        n_saved = save_fold_predictions(
            test_pairs, y_score, fold_pred_file)
        print(f"  {n_saved} predictions saved to {fold_pred_file}")

    # ------------------------------------------------------------------ #
    # Save formal results and pooled predictions
    # ------------------------------------------------------------------ #
    extra_info = {
        'hyperparameters': {
            'train_max_iter': args.train_max_iter,
            'train_lr': args.train_lr,
            'dropout': args.dropout,
            'beta': args.beta,
            'num_neighbor': args.num_neighbor,
            'layers': args.layers,
            'gcn_out_units': args.gcn_out_units,
            'gcn_agg_units': args.gcn_agg_units,
            'nhid1': args.nhid1,
            'nhid2': args.nhid2,
            'seed': args.seed,
            'patience': args.patience,
            'train_valid_interval': args.train_valid_interval,
        },
    }

    results_file = os.path.join(
        _RESULTS_DIR,
        f'formal_results_adadr_coldstart_'
        f'{split_dataset}_{args.split_mode}.json')
    results = save_results_json(
        'AdaDR', split_dataset, args.split_mode,
        fold_aurocs, fold_auprs, all_true, all_score,
        results_file, extra_info=extra_info)
    print(f"\nResults saved to {results_file}")

    pooled_file = os.path.join(
        _RESULTS_DIR,
        f'adadr_coldstart_{split_dataset}_{args.split_mode}_pooled.json')
    save_pooled_predictions(
        'AdaDR', split_dataset, args.split_mode,
        all_true, all_score, pooled_file)
    print(f"Pooled predictions saved to {pooled_file}")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print(f"\n{'=' * 60}")
    print(f"AdaDR Cold-Start on {split_dataset} ({args.split_mode}) - "
          f"Results ({len(folds)}-fold CV)")
    print(f"  Pooled AUROC: {results['pooled_auroc']:.4f}")
    print(f"  Pooled AUPR:  {results['pooled_aupr']:.4f}")
    print(f"  Per-fold AUROC: {[f'{a:.4f}' for a in fold_aurocs]}")
    print(f"  Per-fold AUPR:  {[f'{a:.4f}' for a in fold_auprs]}")
    print(f"  AUROC mean +/- std: {results['auroc_mean']:.4f} "
          f"+/- {results['auroc_std']:.4f}")
    print(f"  AUPR  mean +/- std: {results['aupr_mean']:.4f} "
          f"+/- {results['aupr_std']:.4f}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
