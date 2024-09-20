import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pca import pca
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from utlis import softmax, reduce_dimensions, cal_ratio
from sklearn.metrics.pairwise import euclidean_distances

def add_geneset_human_pancreas(dataset, gene_embeddings,gene_names, method):
    csv_data = pd.read_csv('marker_genes/'+dataset+".csv")
    markergene = csv_data['Cell marker'].values
    common_genes = np.intersect1d(markergene, gene_names)
    common_genes_indices = np.concatenate([np.where(markergene == gene)[0] for gene in common_genes])
    markergene = markergene[common_genes_indices]
    cellname = csv_data['Cell name'].values[common_genes_indices]

    ## Set selected cell type/group
    all_ct = np.unique(cellname)
    selected_ct = np.array(['acinar', 'beta', 'ductal', 'macrophage','endothelial'], dtype=object)
    gs_dict_new = {k: markergene[cellname == k] for k in all_ct}
    gs_dict_comb_subset = {k: markergene[cellname == k] for k in selected_ct}

    ## Get mutually exclusive sets
    gs_dict_excl = {}
    for gs_name, gs_genes in gs_dict_comb_subset.items():
        for gs_name2, gs_genes2 in gs_dict_new.items():
            if gs_name != gs_name2:
                gs_genes = np.setdiff1d(gs_genes, gs_genes2)
        gs_dict_excl[gs_name] = gs_genes

    X_plot, dim_labels = reduce_dimensions(gene_embeddings, method='PCA', reduced_dimension=2)
    df_plot = pd.concat([pd.DataFrame(X_plot), pd.DataFrame(gene_names)], axis=1)
    df_plot.columns = dim_labels + ["Label"]
    df_plot.index = gene_names
    df_plot['Geneset'] = 'None'
    gs_dict_in = gs_dict_excl
    for gs_, genes in gs_dict_in.items():
        df_plot['Geneset'].loc[np.isin(df_plot.index, genes)] = gs_

    ratio,L, celltype = cal_ratio(gene_embeddings, df_plot)
    print(method, " inter/intra ratio: ", ratio)

    df_plot = df_plot[df_plot.Geneset != 'None']
    return df_plot, dim_labels, L, celltype

if __name__=="__main__":
    data = sc.read_h5ad("../../gene&cell_emb_human_pancreas_scHetG")  # gene embeddings obtained by scHetG, stored in data.varm[‘feat’]
    SIMBA_data = sc.read_h5ad("../../gene&cell_emb_human_pancreas_SIMBA")  # This data is only used to take the names of both scHetG and SIMBA computes the gene embedding

    data_genes = data.var_names
    simba_genes = SIMBA_data.obs_names
    common_genes = np.intersect1d(data_genes, simba_genes)
    data_genes_indices = np.concatenate([np.where(data_genes == gene)[0] for gene in common_genes])
    simba_genes_indices = np.concatenate([np.where(simba_genes == gene)[0] for gene in common_genes])


    ####
    gene_embeddings = data.varm['feat']
    gene_embeddings = softmax(data,adata_ref=data.obsm['feat'], adata_query=gene_embeddings)
    gene_names = data.var_names
    gene_embeddings = gene_embeddings[data_genes_indices]
    gene_names = gene_names[data_genes_indices]
    df_plot, dim_labels,L3, celltype = add_geneset_human_pancreas("human pancreas", gene_embeddings, gene_names, "scHetG")
    df_plot.insert(loc=4, column='gene', value=LabelEncoder().fit_transform(df_plot.Geneset.values).astype(np.int_))


    all_known_marker_names = df_plot[df_plot.Geneset.values == "beta"].index
    sub_known_marker_names =df_plot[df_plot.Geneset.values == "beta"].index[10:15]
    print("Known marker gene names:", sub_known_marker_names.values)
    all_genes_names = data.var_names
    all_known_marker_index = [np.where(all_genes_names == elem)[0][0] for elem in all_known_marker_names]
    sub_known_marker_index = [np.where(all_genes_names == elem)[0][0] for elem in sub_known_marker_names]

    all_genes_embeddings = data.varm['feat']

    distance_matrix = np.square(euclidean_distances(all_genes_embeddings, all_genes_embeddings))

    sub_known_embeddings = all_genes_embeddings[sub_known_marker_index]

    distance_matrix_subset = euclidean_distances(all_genes_embeddings, sub_known_embeddings)

    weights = np.ones(sub_known_embeddings.shape[0])  # 权重为1
    weighted_distances = distance_matrix_subset @ weights
    weighted_distances = -np.exp(-weighted_distances)

    num_closest = 5

    closest_indices = np.array([item for item in np.argsort(weighted_distances) if item not in set(sub_known_marker_index)])[:num_closest]
    intersection_count = len(np.intersect1d(closest_indices, all_known_marker_index))
    print("Identification of marker genes:",all_genes_names[np.intersect1d(closest_indices, all_known_marker_index)].values)

    print("Number of correctly identified marker genes:", intersection_count)