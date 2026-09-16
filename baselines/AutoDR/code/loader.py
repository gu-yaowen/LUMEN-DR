import math
import random
import numpy as np
import scipy.io as scio
from sklearn.model_selection import KFold
import pandas as pd

def get_disease_sim_Matrix(disease_similarity, disease_disease_topk):
    disease_disease_Matrix = np.zeros((disease_similarity.shape[0], disease_similarity.shape[0]), np.float32)
    disease_sim_Matrix = disease_similarity
    for disease_num0 in range(disease_similarity.shape[0]):
        disease_sim = {}
        for disease_num1 in range(disease_similarity.shape[0]):
            if disease_num0 == disease_num1:
                continue
            disease_sim[disease_num1] = disease_sim_Matrix[disease_num0][disease_num1]
        sorted_disease_list = sorted(disease_sim.items(), key=lambda d: d[1], reverse=True)
        for i in range(min(disease_disease_topk, len(sorted_disease_list))):
            disease_num1 = sorted_disease_list[i][0]
            disease_disease_Matrix[disease_num0][disease_num1] = disease_sim_Matrix[disease_num0][disease_num1]

    return disease_disease_Matrix


def get_drug_sim_Matrix(drug_similarity, drug_drug_topk):
    drug_drug_Matrix = np.zeros((drug_similarity.shape[0], drug_similarity.shape[0]), np.float32)
    drug_sim_Matrix = drug_similarity
    for drug_num0 in range(drug_similarity.shape[0]):
        drug_sim = {}
        for drug_num1 in range(drug_similarity.shape[0]):
            if drug_num0 == drug_num1:
                continue
            drug_sim[drug_num1] = drug_sim_Matrix[drug_num0][drug_num1]
        sorted_drug_list = sorted(drug_sim.items(), key=lambda d: d[1], reverse=True)
        for i in range(min(drug_drug_topk, len(sorted_drug_list))):
            drug_num1 = sorted_drug_list[i][0]
            drug_drug_Matrix[drug_num0][drug_num1] = drug_sim_Matrix[drug_num0][drug_num1]

    return drug_drug_Matrix


def load_mat(filepath):
    mat = scio.loadmat(filepath)
    drug_sim = mat["drug"].astype(np.float64)
    disease_sim = mat["disease"].astype(np.float64)
    drug_name = mat["Wrname"].reshape(-1)
    drug_num = len(drug_name)
    disease_name = mat["Wdname"].reshape(-1)
    disease_num = len(disease_name)
    interactions = mat["didr"]
    return drug_sim, disease_sim, drug_name, drug_num, disease_name, disease_num, interactions

def load_lrssl(filepath, reduce=True):
    """ C drug:658, disease:409 association:2520 (False 2353)
        PREDICT drug:593, disease:313 association:1933 (Fdataset)
        LRSSL drug: 763, disease:681, association:3051
    """
    drug_chemical = pd.read_csv(filepath + "lrssl_simmat_dc_chemical.txt", sep="\t", index_col=0)
    drug_dataset = pd.read_csv(filepath + "lrssl_simmat_dc_domain.txt", sep="\t", index_col=0)
    drug_go = pd.read_csv(filepath + "lrssl_simmat_dc_go.txt", sep="\t", index_col=0)
    disease_sim = pd.read_csv(filepath + "lrssl_simmat_dg.txt", sep="\t", index_col=0)
    if reduce:
        drug_sim = (drug_chemical+drug_dataset+drug_go)/3
    else:
        drug_sim = drug_chemical
    drug_disease = pd.read_csv(filepath + "lrssl_admat_dgc.txt", sep="\t", index_col=0).T
    drug_disease = drug_disease.T
    rr = drug_sim.to_numpy(dtype=np.float32)
    rd = drug_disease.to_numpy(dtype=np.float32)
    dd = disease_sim.to_numpy(dtype=np.float32)
    rname = drug_sim.columns.to_numpy()
    dname = disease_sim.columns.to_numpy()
    drug_sim = rr.astype(np.float64)
    disease_sim = dd.astype(np.float64)
    drug_name = rname.reshape(-1)
    drug_num = len(drug_name)
    disease_name = dname.reshape(-1)
    disease_num = len(disease_name)
    interactions = rd.T
    return drug_sim, disease_sim, drug_name, drug_num, disease_name, disease_num, interactions

def load_Ldataset(filepath):
    """drug:598, disease:269 association:18416
    """
    disease_sim = pd.read_csv(filepath + "dis_sim.csv", header=None).to_numpy(dtype=np.float32)
    interactions = pd.read_csv(filepath + "drug_dis.csv", header=None).to_numpy(dtype=np.float32)
    drug_sim = pd.read_csv(filepath + "drug_sim.csv", header=None).to_numpy(dtype=np.float32)
    disease_name = np.arange(disease_sim.shape[0])
    drug_name = np.arange(drug_sim.shape[0])
    disease_num = disease_sim.shape[0]
    drug_num = drug_sim.shape[1]
    return drug_sim, disease_sim, drug_name, drug_num, disease_name, disease_num, interactions.T

def data_preparation(args):

    path = "../dataset/" + args.dataset + "/"
    assert args.dataset in ["Cdataset", "Fdataset", "Ldataset", "LRSSL"]
    if args.dataset in ['Fdataset', 'Cdataset']:
        drug_sim, disease_sim, drug_name, drug_num, disease_name, disease_num, interactions = load_mat(
                                                                                                path + args.dataset + ".mat")
    elif args.dataset == "LRSSL":
        drug_sim, disease_sim, drug_name, drug_num, disease_name, disease_num, interactions = load_lrssl(path)
    else:
        drug_sim, disease_sim, drug_name, drug_num, disease_name, disease_num, interactions = load_Ldataset(path)

    args.n_diseases, args.n_drugs = interactions.shape
    
    # Use the original deterministic KFold fallback. The historical shared
    # split archive is intentionally omitted from this repository.
    kfold = KFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    pos_row, pos_col = np.nonzero(interactions)
    neg_row, neg_col = np.nonzero(1 - interactions)
    assert len(pos_row) + len(neg_row) == np.prod(interactions.shape)
    train_data, test_data = [], []
    for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kfold.split(pos_row),
                                                                            kfold.split(neg_row)):
        train_mask = np.zeros_like(interactions, dtype="bool")
        test_mask = np.zeros_like(interactions, dtype="bool")
        train_pos_edge = np.stack([pos_row[train_pos_idx], pos_col[train_pos_idx]])
        train_neg_edge = np.stack([neg_row[train_neg_idx], neg_col[train_neg_idx]])
        test_pos_edge = np.stack([pos_row[test_pos_idx], pos_col[test_pos_idx]])
        test_neg_edge = np.stack([neg_row[test_neg_idx], neg_col[test_neg_idx]])
        train_edge = np.concatenate([train_pos_edge, train_neg_edge], axis=1)
        test_edge = np.concatenate([test_pos_edge, test_neg_edge], axis=1)
        train_mask[train_edge[0], train_edge[1]] = True
        test_mask[test_edge[0], test_edge[1]] = True
        train_data.append(train_mask)
        test_data.append(test_mask)

    disease_disease_sim_Matrix = get_disease_sim_Matrix(disease_sim, args.disease_TopK)
    drug_drug_sim_Matrix = get_drug_sim_Matrix(drug_sim, args.drug_TopK)
    truth_label = interactions
    pos_num = interactions.sum()
    neg_num = np.prod(interactions.shape) - pos_num
    pos_weight = neg_num / pos_num

    return disease_disease_sim_Matrix, drug_drug_sim_Matrix, truth_label, train_data, test_data, pos_weight


class BatchManager(object):

    def __init__(self, data, batch_size, type):

        disease_input, drug_input, labels = [], [], []

        if type == "train":
            train_mask, truth_label = data
            disease_drug_Adj = np.zeros((truth_label.shape[0], truth_label.shape[1]))
            disease_drug_Adj[train_mask] = truth_label[train_mask]
            self.train_adj = disease_drug_Adj
            indices = np.indices(train_mask.shape)
            disease_indices = indices[0].flatten()
            drug_indices = indices[1].flatten()
            disease_input.extend(disease_indices)
            drug_input.extend(drug_indices)
            labels_masked = np.where(train_mask, truth_label, 0).flatten()
            labels.extend(labels_masked)
            num_batch = int(math.ceil(len(disease_input) / batch_size))
            self.batch_data = list()
            for i in range(num_batch):
                input_disease = disease_input[i * batch_size: (i + 1) * batch_size]
                input_drug = drug_input[i * batch_size: (i + 1) * batch_size]
                label = labels[i * batch_size: (i + 1) * batch_size]
                self.batch_data.append([input_disease, input_drug, label])
        elif type == "test":
            test_mask, truth_label = data
            indices = np.where(test_mask)
            disease_input.extend(indices[0])
            drug_input.extend(indices[1])
            labels.extend(truth_label[indices])
            num_batch = int(math.ceil(len(disease_input) / batch_size))
            self.batch_data = list()
            for i in range(num_batch):
                input_disease = disease_input[i * batch_size: (i + 1) * batch_size]
                input_drug = drug_input[i * batch_size: (i + 1) * batch_size]
                label = labels[i * batch_size: (i + 1) * batch_size]
                self.batch_data.append([input_disease, input_drug, label])
        self.len_data = len(self.batch_data)

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]


def data_split(args, train_mask, test_mask, original_interactions):
    train_data = (train_mask, original_interactions)
    test_data = (test_mask, original_interactions)
    train_manager = BatchManager(train_data, args.batch_size, "train")
    test_manager = BatchManager(test_data, args.batch_size, "test")

    return train_manager, test_manager
