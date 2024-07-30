import scanpy as sc
import numpy as np
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn


class ZINBLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mean, disp, pi, x=None, eps=1e-10):
        if x is None:
            zero_nb = torch.pow(disp / (disp + mean + eps), disp)
            zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
            return zero_case.mean()

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)

        return nb_case.mean()

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


def calculate_metric(label, pred):
    labels_name = np.unique(label)
    label_dict = {name:i for i,name in enumerate(labels_name)}
    label = np.array([label_dict[x] for x in label])
    acc = np.round(cluster_acc(label, pred), 5)
    nmi = np.round(metrics.normalized_mutual_info_score(label, pred), 5)
    ari = np.round(metrics.adjusted_rand_score(label, pred), 5)

    return acc, nmi, ari


def cl_crition(x, anchors,postives,negatives=None,margin=0.5,device = torch.device('cuda')):
    X = torch.cat(x, dim=0)
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


def louvain(adata, resolution=None, use_rep='feat'):
    sc.pp.neighbors(adata,n_neighbors=15, use_rep=use_rep, metric='euclidean')
    sc.tl.louvain(adata, resolution=resolution,random_state=0)
    return adata

def leiden(adata, resolution=None, use_rep='feat'):
    sc.pp.neighbors(adata, n_neighbors=15,use_rep=use_rep)
    sc.tl.leiden(adata, resolution=resolution)
    return adata