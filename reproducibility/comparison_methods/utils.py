import scanpy as sc
import numpy as np
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
import pandas as pd


def silhouette(adata, group_key, embed, metric='euclidean', scale=True):
    """
    wrapper for sklearn silhouette function values range from [-1, 1] with 1 being an ideal fit, 0 indicating
    overlapping clusters and -1 indicating misclassified cells
    :param group_key: key in adata.obs of cell labels
    :param embed: embedding key in adata.obsm, default: 'X_pca'
    """
    if embed not in adata.obsm.keys():
        print(adata.obsm.keys())
        raise KeyError(f'{embed} not in obsm')
    asw = metrics.silhouette_score(
        X=adata.obsm[embed],
        labels=adata.obs[group_key],
        metric=metric
    )
    if scale:
        asw = (asw + 1)/2
    return asw


def silhouette_batch(adata, batch_key, group_key, embed, metric='euclidean',
                     verbose=True, scale=True):
    """
    Silhouette score of batch labels subsetted for each group.
    params:
        batch_key: batches to be compared against
        group_key: group labels to be subsetted by e.g. cell type
        embed: name of column in adata.obsm
        metric: see sklearn silhouette score
    returns:
        all scores: absolute silhouette scores per group label
        group means: if `mean=True`
    """
    if embed not in adata.obsm.keys():
        print(adata.obsm.keys())
        raise KeyError(f'{embed} not in obsm')

    sil_all = pd.DataFrame(columns=['group', 'silhouette_score'])

    for group in adata.obs[group_key].unique():
        adata_group = adata[adata.obs[group_key] == group]
        n_batches = adata_group.obs[batch_key].nunique()
        if (n_batches == 1) or (n_batches == adata_group.shape[0]):
            continue
        sil_per_group = metrics.silhouette_samples(adata_group.obsm[embed], adata_group.obs[batch_key],
                                                           metric=metric)
        # take only absolute value
        sil_per_group = [abs(i) for i in sil_per_group]
        if scale:
            # scale s.t. highest number is optimal
            sil_per_group = [1 - i for i in sil_per_group]
        d = pd.DataFrame({'group': [group] * len(sil_per_group), 'silhouette_score': sil_per_group})
        sil_all = sil_all.append(d)
    sil_all = sil_all.reset_index(drop=True)
    sil_means = sil_all.groupby('group').mean()

    if verbose:
        print(f'mean silhouette per cell: {sil_means}')
    return sil_all, sil_means


def silhouette_coeff_ASW_single(adata, embed='feat', c_type='louvain'):
    np.random.seed(2023)
    asw_batch = metrics.silhouette_score(adata.obsm[embed], adata.obs['batch'])
    asw_celltype = metrics.silhouette_score(adata.obsm[embed], adata.obs[c_type])
    min_val = -1
    max_val = 1
    asw_batch_norm = abs(asw_batch)
    asw_celltype_norm = (asw_celltype - min_val) / (max_val - min_val)

    fscoreASW = (2 * (1 - asw_batch_norm) * (asw_celltype_norm)) / (1 - asw_batch_norm + asw_celltype_norm)
    asw = 0.8 * (1 - asw_batch_norm) + 0.2 * asw_celltype_norm

    BASW = 1 - asw_batch_norm
    CASW = asw_celltype_norm
    F_Score = fscoreASW
    ASW = asw
    return F_Score, BASW, CASW, ASW



def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = y_true.astype(np.float64).astype(np.int64)
    y_pred = y_pred.astype(np.float64).astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind,col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size

#原来为def calculate_metric(pred, label):
def calculate_metric(label, pred):
    labels_name = np.unique(label)
    label_dict = {name:i for i,name in enumerate(labels_name)}
    label = np.array([label_dict[x] for x in label])
    acc = np.round(cluster_acc(label, pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(label, pred), 5)
    ari = np.round(metrics.adjusted_rand_score(label, pred), 5)

    return acc, nmi, ari

def compute_pairwise_distances(x, y):
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


def _gaussian_kernel_matrix(x, y, device):
    sigmas = torch.FloatTensor(
        [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]).to(device)
    dist = compute_pairwise_distances(x, y)
    beta = 1. / (2. * sigmas[:, None])
    s = - beta.mm(dist.reshape((1, -1)))
    result = torch.sum(torch.exp(s), dim=0)
    return result


def _maximum_mean_discrepancy(x, y, device=torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')):  # Function to calculate MMD value
    cost = torch.mean(_gaussian_kernel_matrix(x, x, device))
    cost += torch.mean(_gaussian_kernel_matrix(y, y, device))
    cost -= 2.0 * torch.mean(_gaussian_kernel_matrix(x, y, device))
    cost = torch.sqrt(cost ** 2 + 1e-9)
    if cost.data.item() < 0:
        cost = torch.FloatTensor([0.0]).to(device)

    return cost

def cl_crition(x1,x2,anchors,postives,negatives=None,margin=0.5,device = torch.device('cuda')):
    X= torch.cat((x1,x2), dim=0)
    dis_ap = torch.norm(X[anchors]-X[postives], p=2, dim=1)
    if negatives[0] == None:
        cl_loss = torch.mean(torch.maximum(dis_ap, torch.zeros(len(anchors)).to(device)))
    else:
        dis_an = torch.norm(X[anchors]-X[negatives], p=2, dim=1)
        cl_loss = torch.mean(torch.maximum(dis_ap-dis_an+margin, torch.zeros(len(anchors)).to(device)))
    return cl_loss


def calculate_ber(label, pred, batch):
    labels_name = np.unique(label)
    label_dict = {name:i for i,name in enumerate(labels_name)}
    label = np.array([label_dict[x] for x in label])
    y_true = np.array(label)
    y_pred = np.array(pred)
    y_true = y_true.astype(np.float64).astype(np.int64)
    y_pred = y_pred.astype(np.float64).astype(np.int64)

    assert y_pred.size == y_true.size
    K = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((K, K), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind,col_ind = linear_sum_assignment(w.max() - w)

    for i in range(len(label)):
        y_pred[i] = col_ind[y_pred[i]]

    batch_name = np.unique(batch)
    B = len(batch_name)
    batch_dict = {name: i for i, name in enumerate(batch_name)}
    batch = np.array([batch_dict[x] for x in batch])
    batch = batch.astype(np.float64).astype(np.int64)

    pred_mx = np.zeros((K, B))
    true_mx = np.zeros((K, B))
    for k in range(K):
        for b in range(B):
            pred_mx[k, b] = np.sum((y_pred == k) & (batch == b))
            true_mx[k, b] = np.sum((y_true == k) & (batch == b))

    ber = np.sum(pred_mx * np.log((pred_mx+1) / (true_mx+1)))

    return ber


