import os
import json
import copy
import numpy as np
import pandas as pd
import torch as th
from warnings import simplefilter
from model import Model
from load_data import load, remove_graph
from utils import get_metrics_auc, set_seed, m2v, get_metrics
from args import args
import scipy.io as sio
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold


def build_splits(labels, n_splits, seed):
    """Build deterministic pair-level train/validation/test folds locally."""
    flat_indices = np.arange(labels.size)
    flat_labels = labels.reshape(-1)
    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_val_idx, test_idx in outer.split(flat_indices, flat_labels):
        inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + 777)
        train_rel, val_rel = next(inner.split(train_val_idx, flat_labels[train_val_idx]))
        train_idx = train_val_idx[train_rel]
        val_idx = train_val_idx[val_rel]
        train_pairs = [(idx // labels.shape[1], idx % labels.shape[1]) for idx in train_idx]
        val_pairs = [(idx // labels.shape[1], idx % labels.shape[1]) for idx in val_idx]
        test_pairs = [(idx // labels.shape[1], idx % labels.shape[1]) for idx in test_idx]
        folds.append({'train': train_pairs, 'val': val_pairs, 'test': test_pairs})
    return folds


def get_feature_and_metapath(dataset, g):
    """Get feature dict and metapath based on dataset."""
    if dataset == 'Kdataset':
        feature = {'drug': g.nodes['drug'].data['h'],
                   'disease': g.nodes['disease'].data['h'],
                   'protein': g.nodes['protein'].data['h'],
                   'gene': g.nodes['gene'].data['h'],
                   'pathway': g.nodes['pathway'].data['h']}
        metapath = ['disease_drug', 'drug_protein', 'protein_drug', 'drug_disease']
    elif dataset == 'Bdataset':
        feature = {'drug': g.nodes['drug'].data['h'],
                   'disease': g.nodes['disease'].data['h'],
                   'protein': g.nodes['protein'].data['h']}
        metapath = ['disease_drug', 'drug_protein', 'protein_drug', 'drug_disease']
    else:
        feature = {'drug': g.nodes['drug'].data['h'],
                   'disease': g.nodes['disease'].data['h']}
        metapath = ['drug_disease', 'disease_drug']
    return feature, metapath


def train():
    simplefilter(action='ignore', category=FutureWarning)
    print(args)

    device = th.device('cpu')
    print('Training on CPU')

    # load DDA data matrix
    if args.dataset in ['Kdataset', 'Bdataset']:
        df = pd.read_csv('./dataset/{}/{}_baseline.csv'.format(args.dataset, args.dataset), header=None).values
    elif args.dataset in ['Cdataset', 'Fdataset']:
        m = sio.loadmat('./dataset/{}/{}.mat'.format(args.dataset, args.dataset))
        df = m['didr'].T
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')

    n_drugs, n_diseases = df.shape
    print(f'Dataset: {args.dataset}, drugs={n_drugs}, diseases={n_diseases}')

    # Generate deterministic pair-level folds locally. Historical split archives
    # are intentionally not included in this repository.
    folds = build_splits(df, args.nfold, args.seed)
    print(f'Generated {len(folds)} local stratified folds')

    set_seed(args.seed)

    all_true, all_score = [], []
    fold_aurocs, fold_auprs = [], []

    for fold_idx, fold_data in enumerate(folds):
        fold_num = fold_idx + 1
        print(f'\n{"="*60}')
        print(f'Fold {fold_num}/{len(folds)}')
        print(f'{"="*60}')

        train_pairs = fold_data['train']
        val_pairs = fold_data['val']
        test_pairs = fold_data['test']

        # Build masks - val and test are both masked out from training
        mask_label = np.ones(df.shape)
        for d, dis in test_pairs:
            mask_label[d, dis] = 0
        for d, dis in val_pairs:
            mask_label[d, dis] = 0
        mask_train = np.where(mask_label == 1)
        mask_train = (tuple(mask_train[0]), tuple(mask_train[1]))

        # Build val mask (only val pairs)
        val_mask = np.zeros(df.shape)
        for d, dis in val_pairs:
            val_mask[d, dis] = 1
        mask_val = np.where(val_mask == 1)
        mask_val = (tuple(mask_val[0]), tuple(mask_val[1]))

        # Build test mask (only test pairs)
        test_mask_arr = np.zeros(df.shape)
        for d, dis in test_pairs:
            test_mask_arr[d, dis] = 1
        mask_test_only = np.where(test_mask_arr == 1)
        mask_test_only = (tuple(mask_test_only[0]), tuple(mask_test_only[1]))

        # Count pos/neg in train
        train_pos_count = int(df[mask_train].sum())
        train_neg_count = len(mask_train[0]) - train_pos_count
        test_pos_count = int(df[mask_test_only].sum())
        test_neg_count = len(mask_test_only[0]) - test_pos_count

        print(f'Train: {len(mask_train[0])} (pos={train_pos_count}, neg={train_neg_count})')
        print(f'Val:   {len(mask_val[0])}')
        print(f'Test:  {len(mask_test_only[0])} (pos={test_pos_count}, neg={test_neg_count})')

        label = th.tensor(df).float().to(device)

        # Get test positive pairs for graph edge removal (both val and test positives)
        test_pos_arr = np.array([[d, dis] for d, dis in test_pairs if df[d, dis] == 1])
        val_pos_arr = np.array([[d, dis] for d, dis in val_pairs if df[d, dis] == 1])
        all_holdout_pos = np.vstack([test_pos_arr, val_pos_arr]) if len(val_pos_arr) > 0 else test_pos_arr

        # Load graph and remove holdout edges (val + test positives)
        g = load(args.dataset)
        if len(all_holdout_pos) > 0:
            g = remove_graph(g, all_holdout_pos).to(device)
        else:
            g = g.to(device)

        feature, metapath = get_feature_and_metapath(args.dataset, g)

        # Build model
        drug_emb, disease_emb = m2v(g, metapath)
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

        # Training with early stopping based on VALIDATION AUC
        best_val_auc = 0
        best_model_state = None
        patience_counter = 0
        patience = args.patience

        for epoch in range(1, args.epoch + 1):
            model.train()
            score = model(g, feature, drug_emb, disease_emb)
            pred = th.sigmoid(score)
            loss = criterion(score[mask_train].cpu().flatten(),
                             label[mask_train].cpu().flatten())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with th.no_grad():
                # Compute FRESH predictions in eval mode for validation
                # (fix: previously used train-mode pred which has dropout/batchnorm mismatch)
                eval_score = model(g, feature, drug_emb, disease_emb)
                eval_pred = th.sigmoid(eval_score)
                val_auc, val_aupr = get_metrics_auc(
                    label[mask_val].cpu().detach().numpy(),
                    eval_pred[mask_val].cpu().detach().numpy())

            # ReduceLROnPlateau monitors val_auc
            optim_scheduler.step(val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            # Check early stopping every epoch (not every 100)
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch} (val AUC not improved for {patience} epochs)')
                break

            if epoch % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch} Loss: {loss.item():.4f}; Val AUC {val_auc:.4f} (best={best_val_auc:.4f}); LR: {current_lr:.6f}')

        # Load best model and evaluate on test set
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        model.eval()
        with th.no_grad():
            pred = th.sigmoid(model(g, feature, drug_emb, disease_emb))
            pred_np = pred.cpu().detach().numpy()

            # Evaluate on TEST set only
            test_labels = label[mask_test_only].cpu().numpy()
            test_preds = pred_np[mask_test_only]

            auroc = roc_auc_score(test_labels, test_preds)
            aupr = average_precision_score(test_labels, test_preds)

            fold_aurocs.append(auroc)
            fold_auprs.append(aupr)
            all_true.extend(test_labels.tolist())
            all_score.extend(test_preds.tolist())

            # Save per-fold test predictions (drug_id, disease_id, true_label, pred_prob)
            fold_predictions = []
            test_drug_ids, test_disease_ids = mask_test_only
            for i in range(len(test_drug_ids)):
                fold_predictions.append({
                    'drug_id': int(test_drug_ids[i]),
                    'disease_id': int(test_disease_ids[i]),
                    'true_label': int(test_labels[i]),
                    'pred_prob': float(test_preds[i])
                })
            pred_file = f'../mrdda_predictions_{args.dataset}_fold{fold_num}.json'
            with open(pred_file, 'w') as f:
                json.dump(fold_predictions, f)
            print(f'Fold {fold_num} Test AUROC: {auroc:.4f}, AUPR: {aupr:.4f} (predictions saved to {pred_file})')

    # Pooled metrics
    pooled_auroc = roc_auc_score(all_true, all_score)
    pooled_aupr = average_precision_score(all_true, all_score)
    overall_auc, overall_aupr, acc, f1, pre, rec, spe = get_metrics(
        np.array(all_true), np.array(all_score))

    print(f'\n{"="*60}')
    print(f'MRDDA on {args.dataset} - Results ({len(folds)}-fold CV)')
    print(f'  Pooled AUROC: {pooled_auroc:.4f}')
    print(f'  Pooled AUPR:  {pooled_aupr:.4f}')
    print(f'  Per-fold AUROC: {[f"{a:.4f}" for a in fold_aurocs]}')
    print(f'  Per-fold AUPR:  {[f"{a:.4f}" for a in fold_auprs]}')
    print(f'  Overall AUC: {overall_auc:.4f}, AUPR: {overall_aupr:.4f}')
    print(f'  Acc: {acc:.4f}, F1: {f1:.4f}, Pre: {pre:.4f}, Rec: {rec:.4f}, Spe: {spe:.4f}')
    print(f'{"="*60}')

    results = {
        'model': 'MRDDA',
        'dataset': args.dataset,
        'auroc': float(pooled_auroc),
        'aupr': float(pooled_aupr),
        'fold_aurocs': [float(a) for a in fold_aurocs],
        'fold_auprs': [float(a) for a in fold_auprs],
        'overall_auc': float(overall_auc),
        'overall_aupr': float(overall_aupr),
        'accuracy': float(acc),
        'f1': float(f1),
        'precision': float(pre),
        'recall': float(rec),
        'specificity': float(spe)
    }
    result_file = f'../formal_results_mrdda_{args.dataset}.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {result_file}')

    # Save pooled test predictions (all folds combined)
    pooled_predictions = {
        'model': 'MRDDA',
        'dataset': args.dataset,
        'n_samples': len(all_true),
        'true_labels': [int(t) for t in all_true],
        'pred_probs': [float(s) for s in all_score]
    }
    pooled_pred_file = f'../mrdda_predictions_{args.dataset}_pooled.json'
    with open(pooled_pred_file, 'w') as f:
        json.dump(pooled_predictions, f)
    print(f'Pooled predictions saved to {pooled_pred_file}')


if __name__ == '__main__':
    train()
