import dgl
import torch as th
import numpy as np
import pandas as pd
import scipy.io as sio


def load(dataset):

    if dataset == 'Bdataset':
        return load_Bdataset()
    if dataset == 'Kdataset':
        return load_Kdataset()
    if dataset == 'Cdataset':
        return load_Cdataset()
    if dataset == 'Fdataset':
        return load_Fdataset()

def load_Kdataset():
    """Load the heterogeneous network of Kdataset.
    """

    drug_drug = pd.read_csv('./dataset/Kdataset/drug_drug_baseline.csv', header=None).values
    drug_sim = drug_drug
    for i in range(len(drug_drug)):
        sorted_idx = np.argpartition(drug_drug[i], 15)
        drug_drug[i, sorted_idx[-15:]] = 1
    drug_drug = pd.DataFrame(np.array(np.where(drug_drug == 1)).T, columns=['Drug1', 'Drug2'])
    protein_protein = pd.read_csv('./dataset/Kdataset/interactions/protein_protein.csv')
    gene_gene = pd.read_csv('./dataset/Kdataset/interactions/gene_gene.csv')
    pathway_pathway = pd.read_csv('./dataset/Kdataset/interactions/pathway_pathway.csv')
    disease_disease = pd.read_csv('./dataset/Kdataset/disease_disease_baseline.csv', header=None).values
    disease_sim = disease_disease
    for i in range(len(disease_disease)):
        sorted_idx = np.argpartition(disease_disease[i], 15)
        disease_disease[i, sorted_idx[-15:]] = 1
    disease_disease = pd.DataFrame(np.array(np.where(disease_disease == 1)).T, columns=['Disease1', 'Disease2'])
    drug_protein = pd.read_csv('./dataset/Kdataset/associations/drug_protein.csv')
    protein_gene = pd.read_csv('./dataset/Kdataset/associations/protein_gene.csv')
    gene_pathway = pd.read_csv('./dataset/Kdataset/associations/gene_pathway.csv')
    pathway_disease = pd.read_csv('./dataset/Kdataset/associations/pathway_disease.csv')
    drug_disease = pd.read_csv('./dataset/Kdataset/associations/Kdataset.csv')
    graph_data = {
        ('drug', 'drug_drug', 'drug'): (th.tensor(drug_drug['Drug1'].values),  # dtype=th.float
                                        th.tensor(drug_drug['Drug2'].values)),
        ('drug', 'drug_protein', 'protein'): (th.tensor(drug_protein['Drug'].values),
                                              th.tensor(drug_protein['Protein'].values)),
        ('protein', 'protein_drug', 'drug'): (th.tensor(drug_protein['Protein'].values),
                                              th.tensor(drug_protein['Drug'].values)),
        ('protein', 'protein_protein', 'protein'): (th.tensor(protein_protein['Protein1'].values),
                                                    th.tensor(protein_protein['Protein2'].values)),
        ('protein', 'protein_gene', 'gene'): (th.tensor(protein_gene['Protein'].values),
                                              th.tensor(protein_gene['Gene'].values)),
        ('gene', 'gene_protein', 'protein'): (th.tensor(protein_gene['Gene'].values),
                                              th.tensor(protein_gene['Protein'].values)),
        ('gene', 'gene_gene', 'gene'): (th.tensor(gene_gene['Gene1'].values),
                                        th.tensor(gene_gene['Gene2'].values)),
        ('gene', 'gene_pathway', 'pathway'): (th.tensor(gene_pathway['Gene'].values),
                                              th.tensor(gene_pathway['Pathway'].values)),
        ('pathway', 'pathway_gene', 'gene'): (th.tensor(gene_pathway['Pathway'].values),
                                              th.tensor(gene_pathway['Gene'].values)),
        ('pathway', 'pathway_pathway', 'pathway'): (th.tensor(pathway_pathway['Pathway1'].values),
                                                    th.tensor(pathway_pathway['Pathway2'].values)),
        ('pathway', 'pathway_disease', 'disease'): (th.tensor(pathway_disease['Pathway'].values),
                                                    th.tensor(pathway_disease['Disease'].values)),
        ('disease', 'disease_pathway', 'pathway'): (th.tensor(pathway_disease['Disease'].values),
                                                    th.tensor(pathway_disease['Pathway'].values)),
        ('disease', 'disease_disease', 'disease'): (th.tensor(disease_disease['Disease1'].values),
                                                    th.tensor(disease_disease['Disease2'].values)),
        ('drug', 'drug_disease', 'disease'): (th.tensor(drug_disease['Drug'].values),
                                              th.tensor(drug_disease['Disease'].values)),
        ('disease', 'disease_drug', 'drug'): (th.tensor(drug_disease['Disease'].values),
                                              th.tensor(drug_disease['Drug'].values)),
    }
    g = dgl.heterograph(graph_data)
    drug_feature = np.hstack((drug_sim, np.zeros((g.num_nodes('drug'), g.num_nodes('disease')))))
    dis_feature = np.hstack((np.zeros((g.num_nodes('disease'), g.num_nodes('drug'))), disease_sim))
    g.nodes['drug'].data['h'] = th.from_numpy(drug_feature).to(th.float32)
    g.nodes['disease'].data['h'] = th.from_numpy(dis_feature).to(th.float32)
    g.nodes['protein'].data['h'] = th.zeros((g.num_nodes('protein'), drug_feature.shape[1])).to(th.float32)
    g.nodes['gene'].data['h'] = th.zeros((g.num_nodes('gene'), drug_feature.shape[1])).to(th.float32)
    g.nodes['pathway'].data['h'] = th.zeros((g.num_nodes('pathway'), drug_feature.shape[1])).to(th.float32)
    return g

def load_Bdataset():
    """Load the heterogeneous network of Bdataset.
    """

    drug_drug = pd.read_csv('./dataset/Bdataset/drug_drug_baseline.csv', header=None).values
    drug_sim = drug_drug.copy()
    for i in range(len(drug_drug)):
        sorted_idx = np.argpartition(drug_drug[i], 15)
        drug_drug[i, sorted_idx[-15:]] = 1
    drug_drug = pd.DataFrame(np.array(np.where(drug_drug == 1)).T, columns=['Drug1', 'Drug2'])
    protein_protein = pd.read_csv('./dataset/Bdataset/interactions/protein_protein.csv')
    disease_disease = pd.read_csv('./dataset/Bdataset/disease_disease_baseline.csv', header=None).values
    disease_sim = disease_disease.copy()
    for i in range(len(disease_disease)):
        sorted_idx = np.argpartition(disease_disease[i], 15)
        disease_disease[i, sorted_idx[-15:]] = 1
    disease_disease = pd.DataFrame(np.array(np.where(disease_disease == 1)).T, columns=['Disease1', 'Disease2'])
    drug_protein = pd.read_csv('./dataset/Bdataset/associations/drug_protein.csv')
    drug_disease = pd.read_csv('./dataset/Bdataset/associations/Bdataset.csv')
    # protein_disease = pd.read_csv('./dataset/Bdataset/associations/protein_disease.csv')
    graph_data = {
        ('drug', 'drug_drug', 'drug'): (th.tensor(drug_drug['Drug1'].values),
                                        th.tensor(drug_drug['Drug2'].values)),
        ('drug', 'drug_protein', 'protein'): (th.tensor(drug_protein['Drug'].values),
                                              th.tensor(drug_protein['Protein'].values)),
        ('protein', 'protein_drug', 'drug'): (th.tensor(drug_protein['Protein'].values),
                                              th.tensor(drug_protein['Drug'].values)),
        ('protein', 'protein_protein', 'protein'): (th.tensor(protein_protein['Protein1'].values),
                                                    th.tensor(protein_protein['Protein2'].values)),
        ('disease', 'disease_disease', 'disease'): (th.tensor(disease_disease['Disease1'].values),
                                                    th.tensor(disease_disease['Disease2'].values)),
        ('drug', 'drug_disease', 'disease'): (th.tensor(drug_disease['Drug'].values),
                                              th.tensor(drug_disease['Disease'].values)),
        ('disease', 'disease_drug', 'drug'): (th.tensor(drug_disease['Disease'].values),
                                              th.tensor(drug_disease['Drug'].values)),
    }
    g = dgl.heterograph(graph_data)
    drug_feature = np.hstack((drug_sim, np.zeros((g.num_nodes('drug'), g.num_nodes('disease')))))
    dis_feature = np.hstack((np.zeros((g.num_nodes('disease'), g.num_nodes('drug'))), disease_sim))
    g.nodes['drug'].data['h'] = th.from_numpy(drug_feature).to(th.float32)
    g.nodes['disease'].data['h'] = th.from_numpy(dis_feature).to(th.float32)
    g.nodes['protein'].data['h'] = th.zeros((g.num_nodes('protein'), drug_feature.shape[1])).to(th.float32)
    return g  

def remove_graph(g, test_id):
    """Delete the drug-disease associations which belong to test set
    from heterogeneous network.
    """

    test_drug_id = test_id[:, 0]
    test_dis_id = test_id[:, 1]
    edges_id = g.edge_ids(th.tensor(test_drug_id, dtype=th.int64),
                          th.tensor(test_dis_id, dtype=th.int64),
                          etype=('drug', 'drug_disease', 'disease'))
    g = dgl.remove_edges(g, edges_id, etype=('drug', 'drug_disease', 'disease'))
    edges_id = g.edge_ids(th.tensor(test_dis_id, dtype=th.int64),
                          th.tensor(test_drug_id, dtype=th.int64),
                          etype=('disease', 'disease_drug', 'drug'))
    g = dgl.remove_edges(g, edges_id, etype=('disease', 'disease_drug', 'drug'))
    return g


def load_Cdataset():
    """Load the heterogeneous network of Cdataset.
    """

    data = sio.loadmat('./dataset/Cdataset/Cdataset.mat')
    drug_disease = data['didr'].T
    disease_disease = data['disease']
    drug_drug = data['drug']
    drug_sim = drug_drug.copy()
    for i in range(len(drug_drug)):
        sorted_idx = np.argpartition(drug_drug[i], 15)
        drug_drug[i, sorted_idx[-15:]] = 1
    drug_drug = pd.DataFrame(np.array(np.where(drug_drug == 1)).T, columns=['Drug1', 'Drug2'])
    disease_sim = disease_disease.copy()
    for i in range(len(disease_disease)):
        sorted_idx = np.argpartition(disease_disease[i], 15)
        disease_disease[i, sorted_idx[-15:]] = 1
    disease_disease = pd.DataFrame(np.array(np.where(disease_disease == 1)).T, columns=['Disease1', 'Disease2'])
    drug_disease = pd.DataFrame(np.array(np.where(drug_disease == 1)).T, columns=['Drug', 'Disease'])
    graph_data = {
        ('drug', 'drug_drug', 'drug'): (th.tensor(drug_drug['Drug1'].values),
                                        th.tensor(drug_drug['Drug2'].values)),
        ('disease', 'disease_disease', 'disease'): (th.tensor(disease_disease['Disease1'].values),
                                                    th.tensor(disease_disease['Disease2'].values)),
        ('drug', 'drug_disease', 'disease'): (th.tensor(drug_disease['Drug'].values),
                                              th.tensor(drug_disease['Disease'].values)),
        ('disease', 'disease_drug', 'drug'): (th.tensor(drug_disease['Disease'].values),
                                              th.tensor(drug_disease['Drug'].values)),
    }
    g = dgl.heterograph(graph_data)
    drug_feature = np.hstack((drug_sim, np.zeros((g.num_nodes('drug'), g.num_nodes('disease')))))
    dis_feature = np.hstack((np.zeros((g.num_nodes('disease'), g.num_nodes('drug'))), disease_sim))
    g.nodes['drug'].data['h'] = th.from_numpy(drug_feature).to(th.float32)
    g.nodes['disease'].data['h'] = th.from_numpy(dis_feature).to(th.float32)
    return g


def load_Fdataset():
    """Load the heterogeneous network of Fdataset.
    """

    data = sio.loadmat('./dataset/Fdataset/Fdataset.mat')
    drug_disease = data['didr'].T
    disease_disease = data['disease']
    drug_drug = data['drug']
    drug_sim = drug_drug.copy()
    for i in range(len(drug_drug)):
        sorted_idx = np.argpartition(drug_drug[i], 15)
        drug_drug[i, sorted_idx[-15:]] = 1
    drug_drug = pd.DataFrame(np.array(np.where(drug_drug == 1)).T, columns=['Drug1', 'Drug2'])
    disease_sim = disease_disease.copy()
    for i in range(len(disease_disease)):
        sorted_idx = np.argpartition(disease_disease[i], 15)
        disease_disease[i, sorted_idx[-15:]] = 1
    disease_disease = pd.DataFrame(np.array(np.where(disease_disease == 1)).T, columns=['Disease1', 'Disease2'])
    drug_disease = pd.DataFrame(np.array(np.where(drug_disease == 1)).T, columns=['Drug', 'Disease'])
    graph_data = {
        ('drug', 'drug_drug', 'drug'): (th.tensor(drug_drug['Drug1'].values),
                                        th.tensor(drug_drug['Drug2'].values)),
        ('disease', 'disease_disease', 'disease'): (th.tensor(disease_disease['Disease1'].values),
                                                    th.tensor(disease_disease['Disease2'].values)),
        ('drug', 'drug_disease', 'disease'): (th.tensor(drug_disease['Drug'].values),
                                              th.tensor(drug_disease['Disease'].values)),
        ('disease', 'disease_drug', 'drug'): (th.tensor(drug_disease['Disease'].values),
                                              th.tensor(drug_disease['Drug'].values)),
    }
    g = dgl.heterograph(graph_data)
    drug_feature = np.hstack((drug_sim, np.zeros((g.num_nodes('drug'), g.num_nodes('disease')))))
    dis_feature = np.hstack((np.zeros((g.num_nodes('disease'), g.num_nodes('drug'))), disease_sim))
    g.nodes['drug'].data['h'] = th.from_numpy(drug_feature).to(th.float32)
    g.nodes['disease'].data['h'] = th.from_numpy(dis_feature).to(th.float32)
    return g
