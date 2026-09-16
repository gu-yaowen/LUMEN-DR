import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_scatter import scatter
from scipy.sparse import coo_matrix, lil_matrix
import scipy.sparse as sp


class GraphConv_CA(nn.Module):

    """
    Automatic Collaborative Learning
    """

    def __init__(self, args):
        super(GraphConv_CA, self).__init__()
        self.args = args

    def forward(self, embed, edge_index, trend):
        agg_embed = embed
        embs = [embed]

        row, col = edge_index

        for hop in range(self.args.n_hops):
            out = agg_embed[row] * (trend).unsqueeze(-1)
            agg_embed = scatter(out, col, dim=0, dim_size=self.args.n_diseases + self.args.n_drugs, reduce='add')
            embs.append(agg_embed)

        embs = torch.stack(embs, dim=1)

        return embs


class AutoDR(nn.Module):
    def __init__(self, args, dataset):
        super(AutoDR, self).__init__()
        train_manager, train_adj, disease_adj, drug_adj, pos_weight = dataset
        self.args = args
        self.n_diseases = args.n_diseases
        self.n_drugs = args.n_drugs
        self.interaction_matrix = coo_matrix(train_adj).astype(np.float32)
        self.disease_adj = disease_adj
        self.drug_adj = drug_adj
        self.pos_weight = pos_weight
        self.device = args.device
        self.adj_mat = self.getSparseGraph()
        self.adj_mat2 = self.getSparseGraph2()
        self._init_weight()
        self.gcn = self._init_model()

        ##################################
        self.bn = nn.BatchNorm1d(self.args.dim, affine=False)

        ##################################
        layers = []
        embs = str(self.args.dim) + '-' + str(self.args.dim) + '-' + str(self.args.dim)
        sizes = [self.args.dim] + list(map(int, embs.split('-')))
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.embeds = nn.Parameter(initializer(torch.empty(
            self.args.n_diseases + self.args.n_drugs, self.args.dim)))
        self.embeds2 = nn.Parameter(initializer(torch.empty(
            self.args.n_diseases + self.args.n_drugs, self.args.dim)))

    def _init_model(self):
        return GraphConv_CA(self.args)

    def getSparseGraph(self):
        adj_mat = sp.dok_matrix((self.n_diseases + self.n_drugs, self.n_diseases + self.n_drugs), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_diseases, self.n_diseases:] = R
        adj_mat[self.n_diseases:, :self.n_diseases] = R.T
        adj_mat = adj_mat.todok()
        rowsum = np.array(adj_mat.sum(axis=1))
        rowsum[rowsum == 0.] = 1.
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        Graph = Graph.coalesce().to(self.device)
        return Graph

    def getSparseGraph2(self):
        adj_mat = sp.dok_matrix((self.n_diseases + self.n_drugs, self.n_diseases + self.n_drugs), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R1 = lil_matrix(self.disease_adj)
        R2 = lil_matrix(self.drug_adj)
        adj_mat[:self.n_diseases, :self.n_diseases] = R1
        adj_mat[self.n_diseases:, self.n_diseases:] = R2
        adj_mat = adj_mat.todok()
        rowsum = np.array(adj_mat.sum(axis=1))
        rowsum[rowsum == 0.] = 1.
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        Graph = Graph.coalesce().to(self.device)
        return Graph

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def forward(self, batch=None):
        diseases, drugs, labels = batch

        embs = self.gcn(self.embeds, self.edge_index, self.trend)
        embs2 = self.gcn(self.embeds2, self.edge_index2, self.trend2)

        embs = self.pooling(embs)
        embs2 = self.pooling(embs2)

        disease_gcn_embs, drug_gcn_embs = embs[:
                                            self.args.n_diseases], embs[self.args.n_diseases:]
        disease_gcn_embs2, drug_gcn_embs2 = embs2[:
                                               self.args.n_diseases], embs2[self.args.n_diseases:]

        #####################################
        batch_disease_gcn_embs1, batch_disease_gcn_embs2 = disease_gcn_embs[diseases], disease_gcn_embs2[diseases]
        batch_drug_gcn_embs1, batch_drug_gcn_embs2 = drug_gcn_embs[drugs], drug_gcn_embs2[drugs]

        disease_final_emb = disease_gcn_embs + disease_gcn_embs2
        drug_final_emb = drug_gcn_embs + drug_gcn_embs2

        batch_disease_all_embeddings = disease_final_emb[diseases]
        batch_drug_all_embeddings = drug_final_emb[drugs]

        scores = torch.mul(batch_disease_all_embeddings, batch_drug_all_embeddings).sum(dim=1)
        labels = torch.FloatTensor(labels)

        BCE_loss = self.bce_loss_fn(torch.sigmoid(scores), labels)

        #####################
        bt_loss_disease = self.bt(batch_disease_gcn_embs1, batch_disease_gcn_embs2)
        bt_loss_drug = self.bt(batch_drug_gcn_embs1, batch_drug_gcn_embs2)
        bt_loss = bt_loss_disease + bt_loss_drug

        ####################
        mom_loss = self.loss_fn(batch_disease_gcn_embs1, batch_disease_gcn_embs2) / 2 + self.loss_fn(batch_drug_gcn_embs1,  batch_drug_gcn_embs2) / 2

        loss = BCE_loss + bt_loss * self.args.all_bt_coeff + mom_loss * 0.1

        return loss, torch.sigmoid(scores)

    def pooling(self, embeddings):

        if self.args.aggr == 'mean':
            return embeddings.mean(dim=1)
        elif self.args.aggr == 'sum':
            return embeddings.sum(dim=1)
        elif self.args.aggr == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:
            return embeddings[:, -1, :]

    def generate(self, batch=None):
        _, _, labels = batch
        _, scores = self.forward(batch)

        return scores, labels

    def bce_loss_fn(self, predict, label):
        predict = predict.reshape(-1)
        label = label.reshape(-1)
        weight = self.pos_weight * label + 1 - label
        loss = F.binary_cross_entropy(input=predict.to(self.device), target=label.to(self.device), weight=weight.to(self.device))
        return loss

    @staticmethod
    def off_diagonal(x):

        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def bt(self, x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)

        user_e = self.projector(x)
        item_e = self.projector(y)
        c = self.bn(user_e).T @ self.bn(item_e)
        c.div_(user_e.size()[0])

        # sum the cross-correlation matrix
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().div(self.args.dim)
        off_diag = self.off_diagonal(c).pow_(2).sum().div(self.args.dim)
        bt = on_diag + self.args.bt_coeff * off_diag
        return bt

    def loss_fn(self, p, z):  # cosine similarity
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()