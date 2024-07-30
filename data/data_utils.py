import scanpy as sc
import numpy as np
import dgl
import torch
from model.utils import add_degree
from sklearn.utils.extmath import randomized_svd
from scipy.sparse import csr_matrix, find
from sklearn.neighbors import KDTree

def preprocess(adata, common_genes=True,filter_min_counts=True, size_factors=True, normalize_input=False, logtrans_input=True):

    if filter_min_counts:
        if common_genes:
            sc.pp.filter_genes(adata, min_cells=3)
            sc.pp.filter_cells(adata, min_genes=200)
        else:
            sc.pp.filter_cells(adata, min_genes=200)

    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['sz_factor'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['sz_factor'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    sc.tl.pca(adata, n_comps=50, svd_solver='arpack')

    return adata


def _knn(X_ref,
         X_query=None,
         k=20,
         leaf_size=40,  #leaf_size越小越精确，但时间越长
         metric='euclidean'):
    """Calculate K nearest neigbors for each row.
    """
    if X_query is None:
        X_query = X_ref.copy()
    kdt = KDTree(X_ref, leaf_size=leaf_size, metric=metric)
    kdt_d, kdt_i = kdt.query(X_query, k=k, return_distance=True)
    # kdt_i = kdt_i[:, 1:]  # exclude the point itself
    # kdt_d = kdt_d[:, 1:]  # exclude the point itself
    sp_row = np.repeat(np.arange(kdt_i.shape[0]), kdt_i.shape[1])
    sp_col = kdt_i.flatten()
    sp_conn = np.repeat(1, len(sp_row))
    sp_dist = kdt_d.flatten()
    mat_conn_ref_query = csr_matrix(
        (sp_conn, (sp_row, sp_col)),
        shape=(X_query.shape[0], X_ref.shape[0])).T
    mat_dist_ref_query = csr_matrix(
        (sp_dist, (sp_row, sp_col)),
        shape=(X_query.shape[0], X_ref.shape[0])).T
    return mat_conn_ref_query, mat_dist_ref_query


def mnn_edges_svd(adata_ref,
                adata_query,
                n_components=10,
                random_state=0,
                layer=None,
                k=10,
                metric='euclidean',
                leaf_size=40):
    X_ref = adata_ref.X
    X_query = adata_query.X
    print('Performing randomized SVD ...')
    mat = np.dot(X_ref, X_query.T)
    U, Sigma, VT = randomized_svd(mat,
                                  n_components=n_components,
                                  random_state=random_state)
    svd_data = np.vstack((U, VT.T))
    X_svd_ref = svd_data[:U.shape[0], :]
    X_svd_query = svd_data[-VT.shape[1]:, :]
    X_svd_ref = X_svd_ref / (X_svd_ref ** 2).sum(-1, keepdims=True) ** 0.5
    X_svd_query = X_svd_query / (X_svd_query ** 2).sum(-1, keepdims=True) ** 0.5
    print('Searching for mutual nearest neighbors ...')
    knn_conn_ref_query, knn_dist_ref_query = _knn(
        X_ref=X_svd_ref,
        X_query=X_svd_query,
        k=k,
        leaf_size=leaf_size,
        metric=metric)
    knn_conn_query_ref, knn_dist_query_ref = _knn(
        X_ref=X_svd_query,
        X_query=X_svd_ref,
        k=k,
        leaf_size=leaf_size,
        metric=metric)

    sum_conn_ref_query = knn_conn_ref_query + knn_conn_query_ref.T
    id_x, id_y, values = find(sum_conn_ref_query > 1)
    print(f'{len(id_x)} mnn edges based on svd are selected')

    return id_x, id_y


def mnn_edges_pca(adata_ref,
                adata_query,
                n_components=25,
                random_state=42,
                k=20,
                metric='euclidean',
                leaf_size=40):
    X_ref = adata_ref.obsm['X_pca']
    X_query = adata_query.obsm['X_pca']

    X_pca_ref = X_ref
    X_pca_query = X_query
    #X_pca_ref = X_ref/(X_ref ** 2).sum(-1, keepdims=True) ** 0.5
    #X_pca_query = X_query/(X_query ** 2).sum(-1, keepdims=True) ** 0.5

    print('Searching for mutual nearest neighbors ...')
    knn_conn_ref_query, knn_dist_ref_query = _knn(
        X_ref=X_pca_ref,
        X_query=X_pca_query,
        k=k,
        leaf_size=leaf_size,
        metric=metric)
    knn_conn_query_ref, knn_dist_query_ref = _knn(
        X_ref=X_pca_query,
        X_query=X_pca_ref,
        k=k,
        leaf_size=leaf_size,
        metric=metric)

    sum_conn_ref_query = knn_conn_ref_query + knn_conn_query_ref.T
    id_x, id_y, values = find(sum_conn_ref_query > 1)
    print(f'{len(id_x)} mnn edges based on pca are selected')

    return id_x, id_y




def knn_edges(adata_ref,
              mnn_id_x, mnn_id_y,
              k_to_m_ratio=0.5,
              k=20,
              metric='euclidean',
              leaf_size=40):
    batch_cell = np.delete(np.arange(adata_ref.n_obs), np.unique(np.concatenate((mnn_id_x, mnn_id_y))))
    if batch_cell.shape[0] == 0:
        return np.array([]).astype('int'), np.array([]).astype('int'), 0, batch_cell,batch_cell,batch_cell
    X_ref = adata_ref.obsm['X_pca'][batch_cell]
    X_pca_ref = X_ref
    # X_pca_ref = X_ref/(X_ref ** 2).sum(-1, keepdims=True) ** 0.5
    # X_pca_query = X_query/(X_query ** 2).sum(-1, keepdims=True) ** 0.5
    knn_conn_ref_query, knn_dist_ref_query = _knn(
        X_ref=X_pca_ref,
        k=k)
    non_zero_elem = find(knn_conn_ref_query)
    row_indices = non_zero_elem[0]
    col_indices = non_zero_elem[1]
    if int(k_to_m_ratio*len(mnn_id_x)) > len(row_indices):
        #print("knn2")
        return knn_edges2(adata_ref=adata_ref,
                          mnn_id_x=mnn_id_x, mnn_id_y=mnn_id_y,
                          k_to_m_ratio=k_to_m_ratio,
                          k=k
                          )

    else:
        random_indices = np.random.choice(len(row_indices), int(k_to_m_ratio*len(mnn_id_x)), replace=False)
        random_row_indices = row_indices[random_indices]
        random_col_indices = col_indices[random_indices]

        #print(f'{len( random_row_indices)} knn edges based on pca are selected')
        knn_id_x = batch_cell[random_row_indices]
        knn_id_y = batch_cell[random_col_indices]

        return knn_id_x, knn_id_y, 0, batch_cell,row_indices,col_indices

def knn_edges2(adata_ref,
              mnn_id_x, mnn_id_y,
              k_to_m_ratio=0.5,
              k=20,
              metric='euclidean',
              leaf_size=40):
    batch_cell = np.delete(np.arange(adata_ref.n_obs), np.unique(np.concatenate((mnn_id_x, mnn_id_y))))
    X_ref = adata_ref.obsm['X_pca'][batch_cell]
    X_pca_ref = X_ref
    # X_pca_ref = X_ref/(X_ref ** 2).sum(-1, keepdims=True) ** 0.5
    # X_pca_query = X_query/(X_query ** 2).sum(-1, keepdims=True) ** 0.5
    knn_conn_ref_query, knn_dist_ref_query = _knn(
        X_ref=X_pca_ref,
        X_query=adata_ref.obsm['X_pca'],
        k=k)
    non_zero_elem = find(knn_conn_ref_query)
    row_indices = non_zero_elem[0]
    col_indices = non_zero_elem[1]

    random_indices = np.random.choice(len(row_indices), int(k_to_m_ratio * len(mnn_id_x)), replace=False)

    random_row_indices = row_indices[random_indices]
    random_col_indices = col_indices[random_indices]

    #print(f'{len( random_row_indices)} knn edges based on pca are selected')
    knn_id_x = batch_cell[random_row_indices]
    knn_id_y = np.arange(adata_ref.n_obs)[random_col_indices]

    return knn_id_x, knn_id_y, 1,batch_cell,row_indices,col_indices


def precluster(adata, resolution=0.2, use_rep=None):
    sc.pp.neighbors(adata,n_neighbors=25, use_rep=use_rep, metric='euclidean')
    sc.tl.louvain(adata, resolution=resolution, key_added='precluster', random_state=0)
    return adata

def precluster_negative_paris(adata,batch_num, mnn_id_x,knn_id_x):
    scale = np.cumsum(batch_num)
    precluster = adata.obs['precluster'].values
    id_x = np.concatenate((mnn_id_x, knn_id_x))
    negative_pair = np.zeros(len(id_x))

    for i, row_index in enumerate(id_x):
        if row_index < scale[0]:
            nonzero_indices = np.where((np.isin(precluster[:scale[0]], precluster[row_index])) == 0)[0]
        elif row_index >= scale[-2]:
            nonzero_indices = np.where((np.isin(precluster[scale[-2]:], precluster[row_index])) == 0)[0] + scale[-2]
        else:
            order = np.where(scale > row_index)[0][0]
            nonzero_indices = np.where((np.isin(precluster[scale[order-1]:scale[order]], precluster[row_index])) == 0)[0] + scale[order-1]

        negative_pair[i] = np.random.choice(nonzero_indices)

    return negative_pair




def make_graph(single_adata,n_batch, raw_exp=False, highly_variable=None, common_genes=True, val_ratio=0.02):

    num_nodes_dict = {}
    exp_train_dict = {}
    exp_value = []
    exp_dec_graph = []
    exp_val_graph = []
    exp_val_value = []
    unexp_edges = []

    for i in range(n_batch):
        X = single_adata[i].X.toarray()
        num_cells, num_genes = X.shape
        num_nodes_dict.update({'cell'+str(i+1): num_cells, 'gene': num_genes})


        gene_factor = np.max(X, axis=0, keepdims=True)
        single_adata[i].var['gene_factor'] = gene_factor.reshape(-1)
        exp_cell, exp_gene = np.where(X > 0)
        if common_genes==True:
            unexp_edges.append(np.where(X == 0))
        else:
            un_edges = np.where(X == 0)
            temp = np.isin(un_edges[0], np.arange(highly_variable[i].shape[0])[highly_variable[i].values])
            unexp_edges.append((un_edges[0][temp], un_edges[1][temp]))

        idx = np.arange(len(exp_cell))
        np.random.shuffle(idx)
        num_valid = int(np.ceil(len(exp_cell) * val_ratio))
        idx_train, idx_valid = idx[num_valid:], idx[:num_valid]
        idx_train.sort()
        idx_valid.sort()

        exp_valid_cell, exp_valid_gene = exp_cell[idx_valid], exp_gene[idx_valid]
        exp_train_cell, exp_train_gene = exp_cell[idx_train], exp_gene[idx_train]
        exp_train_dict.update({
            ('cell'+str(i+1), 'exp'+str(i+1), 'gene'): (exp_train_cell, exp_train_gene),
            ('gene', 'reverse-exp'+str(i+1), 'cell'+str(i+1)): (exp_train_gene, exp_train_cell)
                               })
        exp_dec_graph.append(
            dgl.heterograph({('cell'+str(i+1), 'exp'+str(i+1), 'gene'): (exp_train_cell, exp_train_gene)},
                                         num_nodes_dict={'cell'+str(i+1): num_cells, 'gene': num_genes}))
        exp_val_graph.append(
            dgl.heterograph({('cell'+str(i+1), 'exp'+str(i+1), 'gene'): (exp_valid_cell, exp_valid_gene)},
                                         num_nodes_dict={'cell'+str(i+1): num_cells, 'gene': num_genes}))

        if raw_exp:
            if common_genes:
                if highly_variable[0] == None:
                    exp_value.append(single_adata[i].raw.X[exp_train_cell, exp_train_gene].reshape(-1, 1))
                    exp_val_value.append(single_adata[i].raw.X[exp_valid_cell, exp_valid_gene].reshape(-1, 1))

                    exp_dec_graph[i].nodes['cell'+str(i+1)].data['sz_factor'] = torch.Tensor(single_adata[i].obs['sz_factor']).reshape(-1, 1)
                    exp_dec_graph[i].nodes['gene'].data['ge_factor'] = torch.Tensor(gene_factor).reshape(-1, 1)

                else:
                    exp_value.append(single_adata[i].raw[:, highly_variable].X[exp_train_cell, exp_train_gene].reshape(-1, 1))
                    exp_val_value.append(single_adata[i].raw[:, highly_variable].X[exp_valid_cell, exp_valid_gene].reshape(-1, 1))

                    exp_dec_graph[i].nodes['cell' + str(i + 1)].data['sz_factor'] = torch.Tensor(single_adata[i].obs['sz_factor']).reshape(-1, 1)
                    exp_dec_graph[i].nodes['gene'].data['ge_factor'] = torch.Tensor(gene_factor).reshape(-1, 1)
            else:
                if highly_variable[0][0] == None:
                    exp_value.append(single_adata[i].raw.X[exp_train_cell, exp_train_gene].reshape(-1, 1))
                    exp_val_value.append(single_adata[i].raw.X[exp_valid_cell, exp_valid_gene].reshape(-1, 1))

                    exp_dec_graph[i].nodes['cell' + str(i + 1)].data['sz_factor'] = torch.Tensor(
                        single_adata[i].obs['sz_factor']).reshape(-1, 1)
                    exp_dec_graph[i].nodes['gene'].data['ge_factor'] = torch.Tensor(gene_factor).reshape(-1, 1)

                else:
                    exp_value.append(
                        single_adata[i].raw[:, highly_variable[-1]].X[exp_train_cell, exp_train_gene].reshape(-1, 1))
                    exp_val_value.append(
                        single_adata[i].raw[:, highly_variable[-1]].X[exp_valid_cell, exp_valid_gene].reshape(-1, 1))

                    exp_dec_graph[i].nodes['cell' + str(i + 1)].data['sz_factor'] = torch.Tensor(
                        single_adata[i].obs['sz_factor']).reshape(-1, 1)
                    exp_dec_graph[i].nodes['gene'].data['ge_factor'] = torch.Tensor(gene_factor).reshape(-1, 1)

        else:
            X = X / gene_factor
            exp_value.append(X[exp_train_cell, exp_train_gene].reshape(-1, 1))
            exp_val_value.append(X[exp_valid_cell, exp_valid_gene].reshape(-1, 1))

    exp_enc_graph = dgl.heterograph(exp_train_dict, num_nodes_dict=num_nodes_dict)
    add_degree(exp_enc_graph, ['exp'+str(i+1) for i in range(n_batch)])



    return single_adata, exp_value, exp_enc_graph, exp_dec_graph, exp_val_graph, exp_val_value, unexp_edges