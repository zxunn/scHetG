from scipy.spatial import distance
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def preprocess(adata, filter_min_counts=True, size_factors=True, normalize_input=False, logtrans_input=True):
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.filter_cells(adata, min_genes=200)

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['cs_factor'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['cs_factor'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    if normalize_input:
        sc.pp.scale(adata)

    return adata

def softmax(data,adata_ref,
            adata_query,
            T=0.5,
            n_top=None,
            percentile=0):
    """Softmax-based transformation

    This will transform query data to reference-comparable data

    Parameters
    ----------
    adata_ref: `AnnData`
        Reference anndata.
    adata_query: `list`
        Query anndata objects
    T: `float`
        Temperature parameter.
        It controls the output probability distribution.
        When T goes to inf, it becomes a discrete uniform distribution,
        each query becomes the average of reference;
        When T goes to zero, softargmax converges to arg max,
        each query is approximately the best of reference.
    n_top: `float` (default: None)
        The number of top reference entities to include when transforming
        entities in query data.
        It is used to filter out low-probability reference entities.
        All reference entities used by default.
    percentile: `int` (default: 0)
        Percentile (0-100) of reference entities to exclude when transforming
        entities in query data.
        Only valid when `n_top` is not specified.
        It is used to filter out low-probability reference entities.
        All reference entities used by default.

    Returns
    -------
    updates `adata_query` with the following field.
    softmax: `array_like` (`.layers['softmax']`)
        Store #observations × #dimensions softmax transformed data matrix.
    """

    #scores_ref_query = np.matmul(adata_ref, adata_query.T)
    #scores_ref_query = data.X.toarray()
    scores_ref_query = data.X
    # data2 = data.copy()
    # data2.X = data.obsm['edge']
    # data2 = preprocess(data2)
    # scores_ref_query=data2.X

    # avoid overflow encountered
    scores_ref_query = scores_ref_query - scores_ref_query.max()
    scores_softmax = np.exp(scores_ref_query/T) / \
        (np.exp(scores_ref_query/T).sum(axis=0))[None, :]
    if n_top is None:
        thresh = np.percentile(scores_softmax, q=percentile, axis=0)
    else:
        thresh = (np.sort(scores_softmax, axis=0)[::-1, :])[n_top-1, ]
    mask = scores_softmax < thresh[None, :]
    scores_softmax[mask] = 0
    # rescale to make scores add up to 1
    scores_softmax = scores_softmax/scores_softmax.sum(axis=0, keepdims=1)
    X_query = np.dot(scores_softmax.T, adata_ref)
    #adata_query.layers['softmax'] = X_query
    return X_query

def reduce_dimensions(X, reduced_dimension = 2, method = 'PCA', tsne_min = 50, match_dims = True,**kwargs):
    """
    Reduce dimensioanlity of X using dimensionality reduction method.
    """
    if X.shape[-1] > reduced_dimension:
        if method == "PCA":
            pca = PCA(n_components = reduced_dimension,**kwargs)
            pca.fit(X)
            X_pca = pca.transform(X)
            X_reduced = X_pca[:,:reduced_dimension]
            dim_name = 'PC'
        elif method == "tSNE":
            if X.shape[-1] > tsne_min:
                X,_ = reduce_dimensions(X, reduced_dimension = tsne_min, method = 'PCA')
            tsne = TSNE(n_components = reduced_dimension, init = 'pca', **kwargs)
            X_tsne = tsne.fit_transform(X)
            X_reduced = X_tsne[:,:2]
            dim_name = 't-SNE'
        elif method == 'UMAP':
            if X.shape[-1] > tsne_min:
                X,_ = reduce_dimensions(X, reduced_dimension = tsne_min, method = 'PCA')
            X_reduced = umap.UMAP(**kwargs).fit_transform(X)
            X_reduced = X_reduced[:,:2]
            dim_name = 'UMAP'
        else:
            raise Exception('{} is not a valid DR method (PCA,tSNE,UMAP)'.format(method))
    else:
        if X.shape[-1] < reduced_dimension and match_dims:
            dim_diff = reduced_dimension - X.shape[-1]
            X_reduced = np.concatenate([X, np.zeros([len(X),dim_diff])], axis = 1)
        else:
            X_reduced = X
        dim_name = "Dim"
    dim_labels = ["{} {}".format(dim_name, ii+1) for ii in range(reduced_dimension)]
    return X_reduced, dim_labels


def cal_ratio(gene_embeddings, df_plot):
    G=gene_embeddings[df_plot.Geneset != 'None']
    celltype =df_plot.Geneset[df_plot.Geneset != 'None'].values
    sorted_index = np.argsort(celltype)
    celltype=celltype[sorted_index]
    G=G[sorted_index]

    S = np.exp(-np.square(distance.cdist(G, G, 'euclidean')))
    differences = celltype[1:] != celltype[:-1]
    indices = np.where(differences)[0] + 1
    indices = np.append(0, indices)
    indices = np.append(indices, G.shape[0])

    num_ctype = len(np.unique(celltype))
    L=np.zeros((num_ctype,num_ctype))

    for p in range(1,num_ctype+1):
        for q in range(1,num_ctype+1):
            if p == q:
                L[p-1,q-1] = 1 / (np.square(indices[p] - indices[p-1]) - (indices[p] - indices[p-1]))
                temp_s=S[indices[p-1]:indices[p],indices[p-1]:indices[p]]
                sum_s = temp_s.sum()-np.trace(temp_s)
                #sum_s = sum_s/2
                if sum_s == 0:
                    sum_s = np.amax(temp_s[~np.eye(temp_s.shape[0], dtype=bool)])
                    #sum_s = 0.00001
                L[p-1, q-1] = L[p-1,q-1]*sum_s
            else:
                L[p-1, q-1]=1 / ((indices[p] - indices[p-1]) * (indices[q] - indices[q-1]))
                temp_s = S[indices[p-1]:indices[p],indices[q-1]:indices[q]]
                sum_s = temp_s.sum()
                L[p-1, q-1] = L[p-1, q-1] * sum_s

    diagonal_L = np.diag(L)
    L_new = L / diagonal_L[:, np.newaxis]
    ratio = 1/(num_ctype*(num_ctype-1)) * (L_new.sum()-np.trace(L_new))

    return ratio