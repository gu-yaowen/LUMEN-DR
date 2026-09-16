import argparse
from loader import *
from torch.optim.lr_scheduler import CyclicLR
from model import *
from sklearn import metrics
from utils import *
from torch_scatter.scatter import scatter_add
import sys

sys.path.append(".")

def parse_args():
    parser = argparse.ArgumentParser(description="Run AutoDR.")
    parser.add_argument('--dataset', nargs='?', default='Fdataset', help='Choose a dataset:[Fdataset, Cdataset, LRSSL]')
    parser.add_argument('--epochs', type=int, default=110, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=1024*5, help='Batch size.')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--disease_TopK', type=int, default=4)
    parser.add_argument('--drug_TopK', type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument("--model", type=str, default="AutoDR", help="backbone")
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument('--trend_coeff', type=float, default=1, help='coefficient of attention')
    parser.add_argument("--n_hops", type=int, default=2)
    parser.add_argument("--aggr", type=str, default='mean')
    parser.add_argument("--layer_sizes", nargs='?', default=[64, 64, 64])
    parser.add_argument('--bt_coeff', type=float, default=0.01, help='learning rate')
    parser.add_argument('--all_bt_coeff', type=float, default=0.2, help='learning rate')

    return parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

args = parse_args()

if __name__ == "__main__":
    import json
    avg_auroc, avg_aupr = [], []
    all_true_pooled, all_score_pooled = [], []
    for i in range(args.num_trials):
        setup_seed(i)
        disease_adj, drug_adj, original_interactions, all_train_mask, all_test_mask, pos_weight = data_preparation(args)
        all_scores, all_labels = [], []
        print(f'+++++++++++++++This is {i + 1}-th 10 fold validation.+++++++++++++++')
        for fold_num in range(len(all_train_mask)):
            print(f'---------------This is {fold_num + 1}-th fold validation.---------------')

            # dataset splitting
            train_manager, test_manager = data_split(args, all_train_mask[fold_num], all_test_mask[fold_num],
                                                     original_interactions)
            train_adj = train_manager.train_adj

            """build model"""
            model = AutoDR(args, (train_manager, train_adj, disease_adj, drug_adj, pos_weight)).to(args.device)

            """define optimizer"""
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            lr_scheduler = CyclicLR(optimizer, base_lr=0.1 * args.lr, max_lr=args.lr, step_size_up=20,
                                    mode="exp_range", gamma=0.995, cycle_momentum=False)

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

            cal_trend = co_ratio_deg_disease_sc
            cal_trend2 = co_ratio_deg_disease_sc2

            trend = cal_trend(
                adj_sp_norm, edge_index, args)
            trend2 = cal_trend2(
                adj_sp_norm2, edge_index2, args)

            norm_now = scatter_add(
                trend, col, dim=0, dim_size=args.n_diseases + args.n_drugs)[col]
            norm_now2 = scatter_add(
                trend2, col2, dim=0, dim_size=args.n_diseases + args.n_drugs)[col2]

            trend = args.trend_coeff * trend / norm_now + edge_weight
            trend2 = args.trend_coeff * trend2 / norm_now2 + edge_weight2

            model.trend = (trend).to(args.device)
            model.trend2 = (trend2).to(args.device)

            for epoch in range(args.epochs):
                model.train()
                loss_list = []
                for batch in train_manager.iter_batch(shuffle=True):
                    loss, _ = model.forward(batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                model.eval()
                scores, labels = [], []
                for batch in test_manager.iter_batch():
                    score, label = model.generate(batch)
                    scores.append(score.cpu().detach().numpy())
                    labels.append(label)
                loss_sum = np.sum(loss_list)
                scores = np.concatenate(scores)
                labels = np.concatenate(labels)
                aupr = metrics.average_precision_score(y_true=labels, y_score=scores)
                auroc = metrics.roc_auc_score(y_true=labels, y_score=scores)
                print(f'Epoch: {epoch + 1}, auroc: {auroc}, aupr: {aupr}')
                if (epoch + 1) == args.epochs:
                    all_scores.append(scores)
                    all_labels.append(labels)
                    # Collect drug_id and disease_id for prediction saving
                    fold_drug_ids, fold_disease_ids = [], []
                    for batch in test_manager.iter_batch():
                        fold_drug_ids.extend(list(batch[1]))  # drug_id
                        fold_disease_ids.extend(list(batch[0]))  # disease_id
                    # Save per-fold predictions
                    fold_predictions = []
                    for idx in range(len(scores)):
                        fold_predictions.append({
                            'drug_id': int(fold_drug_ids[idx]),
                            'disease_id': int(fold_disease_ids[idx]),
                            'true_label': int(labels[idx]),
                            'pred_prob': float(scores[idx])
                        })
                    pred_file = f'../autodr_predictions_{args.dataset}_fold{fold_num + 1}.json'
                    with open(pred_file, 'w') as f:
                        json.dump(fold_predictions, f)
                    print(f'Fold {fold_num + 1} predictions saved to {pred_file}')
                    all_true_pooled.extend(labels.tolist())
                    all_score_pooled.extend(scores.tolist())
        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)
        aupr = metrics.average_precision_score(y_true=all_labels, y_score=all_scores)
        auroc = metrics.roc_auc_score(y_true=all_labels, y_score=all_scores)
        avg_auroc.append(auroc)
        avg_aupr.append(aupr)
        print(f'------------------------------------------------------------------------')
        print(f"{i + 1}-th 10 cv auroc：{auroc:.5f}")
        print(f"{i + 1}-th 10 cv auroc：{aupr:.5f}")
    print(f'------------------------------------------------------------------------')
    print(f"auroc：{np.mean(avg_auroc):.5f}, std：{np.std(avg_auroc):.5f}")
    print(f"aupr：{np.mean(avg_aupr):.5f}, std：{np.std(avg_aupr):.5f}")
    
    results = {'model': 'AutoDR', 'dataset': args.dataset,
               'auroc': float(np.mean(avg_auroc)), 'aupr': float(np.mean(avg_aupr)),
               'auroc_std': float(np.std(avg_auroc)), 'aupr_std': float(np.std(avg_aupr))}
    with open(f'../formal_results_autodr_{args.dataset}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to ../formal_results_autodr_{args.dataset}.json")

    # Save pooled predictions (all folds combined)
    pooled_predictions = {
        'model': 'AutoDR',
        'dataset': args.dataset,
        'n_samples': len(all_true_pooled),
        'true_labels': [int(t) for t in all_true_pooled],
        'pred_probs': [float(s) for s in all_score_pooled]
    }
    pooled_pred_file = f'../autodr_predictions_{args.dataset}_pooled.json'
    with open(pooled_pred_file, 'w') as f:
        json.dump(pooled_predictions, f)
    print(f'Pooled predictions saved to {pooled_pred_file}')