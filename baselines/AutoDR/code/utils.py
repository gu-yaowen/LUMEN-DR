import random
import numpy as np
import torch

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def co_ratio_deg_disease_sc(adj_sp_norm, edge_index, args):
    disease_drug_graph = adj_sp_norm.to_dense(
    )[:args.n_diseases, args.n_diseases:].cpu()
    disease_drug_graph[disease_drug_graph > 0] = 1

    edge_weight = torch.zeros(
        (args.n_diseases + args.n_drugs, args.n_diseases + args.n_drugs))

    for i in range(args.n_drugs):
        diseases = disease_drug_graph[:, i].nonzero().squeeze(-1)

        drugs = disease_drug_graph[diseases]
        disease_disease_cap = torch.matmul(drugs, drugs.t())

        sc_simi = (disease_disease_cap / ((drugs.sum(dim=1) *
                                     drugs.sum(dim=1).unsqueeze(-1))**0.5)).mean(dim=1)

        edge_weight[diseases, i + args.n_diseases] = sc_simi

    for i in range(args.n_diseases):
        drugs = disease_drug_graph[i, :].nonzero().squeeze(-1)

        diseases = disease_drug_graph[:, drugs].t()
        drug_drug_cap = torch.matmul(diseases, diseases.t())

        sc_simi = (drug_drug_cap / ((diseases.sum(dim=1) *
                                     diseases.sum(dim=1).unsqueeze(-1))**0.5)).mean(dim=1)

        edge_weight[drugs + args.n_diseases, i] = sc_simi

    edge_weight = edge_weight[edge_index[0], edge_index[1]]

    return edge_weight

def co_ratio_deg_disease_sc2(adj_sp_norm, edge_index, args):
    disease_disease_graph = adj_sp_norm.to_dense(
    )[:args.n_diseases, :args.n_diseases].cpu()
    disease_disease_graph[disease_disease_graph > 0] = 1

    drug_drug_graph = adj_sp_norm.to_dense(
    )[args.n_diseases:, args.n_diseases:].cpu()
    drug_drug_graph[drug_drug_graph > 0] = 1

    edge_weight = torch.zeros(
        (args.n_diseases + args.n_drugs, args.n_diseases + args.n_drugs))

    for i in range(args.n_diseases):
        diseases = disease_disease_graph[:, i].nonzero().squeeze(-1)

        diseases2 = disease_disease_graph[diseases]
        disease_disease_cap = torch.matmul(diseases2, diseases2.t())

        sc_simi = (disease_disease_cap / ((diseases2.sum(dim=1) *
                                     diseases2.sum(dim=1).unsqueeze(-1))**0.5)).mean(dim=1)

        edge_weight[diseases, i] = sc_simi

    for i in range(args.n_drugs):
        drugs = drug_drug_graph[:, i].nonzero().squeeze(-1)

        drugs2 = drug_drug_graph[drugs]
        drug_drug_cap = torch.matmul(drugs2, drugs2.t())

        sc_simi = (drug_drug_cap / ((drugs2.sum(dim=1) *
                                     drugs2.sum(dim=1).unsqueeze(-1))**0.5)).mean(dim=1)

        edge_weight[drugs + args.n_diseases, i + args.n_diseases] = sc_simi

    edge_weight = edge_weight[edge_index[0], edge_index[1]]

    return edge_weight
