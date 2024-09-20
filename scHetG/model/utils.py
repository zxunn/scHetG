import torch
import numpy as np
import scipy.sparse as sp
from torch.nn import functional as F

def identity_mapping(x):
    return x

activation_map = {
    'relu' : F.relu,
    'leaky' : F.leaky_relu,
    'selu' : F.selu,
    'sigmoid' : F.sigmoid,
    'tanh' : F.tanh,
    'none' : identity_mapping
}

def sparse_to_torch(sp,device):
    coo = sp.tocoo()
    values = torch.FloatTensor(coo.data)
    indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
    shape = coo.shape

    return torch.sparse.FloatTensor(indices, values, torch.Size(shape),)

def get_degree_inv(graph, edge_types, reverse):
    if reverse:
        degree = sum(graph.in_degrees(etype = str(etype)) for etype in edge_types)
    else:
        degree = sum(graph.out_degrees(etype = str(etype)) for etype in edge_types)

    degree_inv = torch.pow(degree, -0.5)
    degree_inv[torch.isinf(degree_inv)] = 0.

    n_dots = len(degree_inv)
    idx = [list(range(n_dots)),list(range(n_dots))]

    return torch.sparse_coo_tensor(idx, degree_inv, (n_dots, n_dots))


def get_adjacency(graph, edge_types, ckey = 'cell', gkey = 'gene'):
    n_cells, n_genes = graph.num_nodes(ckey), graph.num_nodes(gkey)

    adj_uv = torch.sparse_coo_tensor(size = (n_cells, n_genes))
    for etype in edge_types:
        adj_uv += graph.adjacency_matrix(etype = str(etype))

    return adj_uv.to(graph.device)

def degree_noramlization(graph, edge_types, ckey = 'cell', gkey = 'gene'):
    adj_uv = get_adjacency(graph, edge_types, ckey, gkey)
    degree_u = get_degree_inv(graph, edge_types, False)
    degree_v = get_degree_inv(graph, edge_types, True)

    normed_adj_u = torch.sparse.mm(degree_u, adj_uv)
    normed_adj_u = torch.sparse.mm(normed_adj_u, degree_v)

    normed_adj_v = torch.sparse.mm(degree_v, adj_uv.t())
    normed_adj_v = torch.sparse.mm(normed_adj_v, degree_u)

    return normed_adj_u, normed_adj_v

def add_degree(graph, edge_types, symmetric = True):
    def _calc_norm(x):
        x = x.numpy().astype('float32')
        x[x == 0.] = np.inf
        x = torch.FloatTensor(1. / np.sqrt(x))

        return x.unsqueeze(1)

    gene_ci = []
    gene_cj = []

    for i in range(len(edge_types)):
        cell_ci = []
        cell_cj = []

        cell_ci.append(graph['reverse-exp'+str(i+1)].in_degrees())
        cell_cj.append(graph['exp'+str(i+1)].out_degrees())

        gene_ci.append(graph[f'{edge_types[i]}'].in_degrees())
        if symmetric:
            gene_cj.append(graph[f'reverse-{edge_types[i]}'].out_degrees())

        cell_ci = _calc_norm(sum(cell_ci))

        if symmetric:
            cell_cj = _calc_norm(sum(cell_cj))

        graph.nodes['cell'+str(i+1)].data.update({'ci': cell_ci, 'cj': cell_cj})


    gene_ci = _calc_norm(sum(gene_ci))
    if symmetric:
        gene_cj = _calc_norm(sum(gene_cj))

    graph.nodes['gene'].data.update({'ci': gene_ci, 'cj': gene_cj})


