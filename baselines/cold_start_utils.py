"""Common utilities for cold-start evaluation.

This module provides functions to load cold-start split files and convert
pair-level indices to (drug_idx, disease_idx) pairs and masks.
"""
import json
import os
import numpy as np


def load_cold_start_splits(dataset, split_mode, splits_dir='../cold_start_splits'):
    """Load cold-start split file for a given dataset and split mode.

    Parameters
    ----------
    dataset : str
        Dataset name: 'Bdataset', 'Cdataset', 'Fdataset'
    split_mode : str
        'cold_drug' or 'cold_disease'
    splits_dir : str
        Directory containing the split JSON files

    Returns
    -------
    list of dict
        Each dict has keys: 'fold', 'train_idx', 'val_idx', 'test_idx'
        where indices are pair-level (flattened from N_drug x N_disease matrix)
    """
    split_file = os.path.join(splits_dir, f'{dataset}_{split_mode}_splits.json')
    with open(split_file, 'r') as f:
        data = json.load(f)
    return data['folds']


def pair_idx_to_coordinates(pair_indices, n_diseases):
    """Convert flat pair indices to (drug_idx, disease_idx) tuples.

    The pair index is computed as: pair_idx = drug_idx * n_diseases + disease_idx

    Parameters
    ----------
    pair_indices : list or array
        Flat pair indices
    n_diseases : int
        Number of diseases in the dataset

    Returns
    -------
    drug_indices : np.ndarray
    disease_indices : np.ndarray
    """
    pair_indices = np.array(pair_indices)
    drug_indices = pair_indices // n_diseases
    disease_indices = pair_indices % n_diseases
    return drug_indices, disease_indices


def build_masks_from_splits(fold_data, n_drugs, n_diseases):
    """Build train/val/test boolean masks from cold-start split data.

    Parameters
    ----------
    fold_data : dict
        Contains 'train_idx', 'val_idx', 'test_idx' (pair-level indices)
    n_drugs : int
    n_diseases : int

    Returns
    -------
    train_mask : np.ndarray of shape (n_drugs, n_diseases), bool
    val_mask : np.ndarray of shape (n_drugs, n_diseases), bool
    test_mask : np.ndarray of shape (n_drugs, n_diseases), bool
    """
    train_mask = np.zeros((n_drugs, n_diseases), dtype=bool)
    val_mask = np.zeros((n_drugs, n_diseases), dtype=bool)
    test_mask = np.zeros((n_drugs, n_diseases), dtype=bool)

    for idx in fold_data['train_idx']:
        d = idx // n_diseases
        dis = idx % n_diseases
        train_mask[d, dis] = True

    for idx in fold_data['val_idx']:
        d = idx // n_diseases
        dis = idx % n_diseases
        val_mask[d, dis] = True

    for idx in fold_data['test_idx']:
        d = idx // n_diseases
        dis = idx % n_diseases
        test_mask[d, dis] = True

    return train_mask, val_mask, test_mask


def get_test_pairs(test_mask, labels):
    """Extract test pairs (drug_idx, disease_idx, label) from test mask.

    Parameters
    ----------
    test_mask : np.ndarray of shape (n_drugs, n_diseases), bool
    labels : np.ndarray of shape (n_drugs, n_diseases)
        Binary association matrix

    Returns
    -------
    list of dict with keys: drug_id, disease_id, true_label
    """
    test_drugs, test_diseases = np.where(test_mask)
    return [{'drug_id': int(d), 'disease_id': int(dis),
             'true_label': int(labels[d, dis])}
            for d, dis in zip(test_drugs, test_diseases)]


def save_fold_predictions(test_pairs, pred_scores, filepath):
    """Save per-fold test predictions to JSON.

    Parameters
    ----------
    test_pairs : list of dict
        Each dict has 'drug_id', 'disease_id', 'true_label'
    pred_scores : np.ndarray
        Predicted probabilities for each test pair
    filepath : str
        Output JSON file path
    """
    predictions = []
    for i, pair in enumerate(test_pairs):
        predictions.append({
            'drug_id': pair['drug_id'],
            'disease_id': pair['disease_id'],
            'true_label': pair['true_label'],
            'pred_prob': float(pred_scores[i])
        })
    with open(filepath, 'w') as f:
        json.dump(predictions, f)
    return len(predictions)


def save_results_json(model_name, dataset, split_mode, fold_aurocs, fold_auprs,
                      all_true, all_score, filepath, extra_info=None):
    """Save formal results JSON with pooled and per-fold metrics."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    pooled_auroc = float(roc_auc_score(all_true, all_score))
    pooled_aupr = float(average_precision_score(all_true, all_score))

    results = {
        'model': model_name,
        'dataset': dataset,
        'split_mode': split_mode,
        'pooled_auroc': pooled_auroc,
        'pooled_aupr': pooled_aupr,
        'auroc_mean': float(np.mean(fold_aurocs)),
        'auroc_std': float(np.std(fold_aurocs)),
        'aupr_mean': float(np.mean(fold_auprs)),
        'aupr_std': float(np.std(fold_auprs)),
        'fold_aurocs': [float(a) for a in fold_aurocs],
        'fold_auprs': [float(a) for a in fold_auprs],
        'n_folds': len(fold_aurocs),
    }
    if extra_info:
        results.update(extra_info)

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    return results


def save_pooled_predictions(model_name, dataset, split_mode, all_true, all_score, filepath):
    """Save pooled predictions (all folds combined)."""
    pooled = {
        'model': model_name,
        'dataset': dataset,
        'split_mode': split_mode,
        'n_samples': len(all_true),
        'true_labels': [int(t) for t in all_true],
        'pred_probs': [float(s) for s in all_score]
    }
    with open(filepath, 'w') as f:
        json.dump(pooled, f)
