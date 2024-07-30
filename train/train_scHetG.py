#!/usr/bin/env python
# coding: utf-8
import dgl
import torch
from torch import nn, optim
import numpy as np
from model.scHetG import scHetG
from data.data_utils import make_graph, mnn_edges_svd, knn_edges, precluster, precluster_negative_paris, mnn_edges_pca
from train.utils import calculate_metric, ZINBLoss, cl_crition, calculate_ber
import gc
import anndata as ad
from itertools import combinations
import math

def train_scHetG(adata, batch_name,
          n_clusters=None,
          cl_type=None,
          n_layers=3,
          feats_dim=128,
          drop_out=0.3,
          gamma=1,
          decoder='Dot',
          lr=0.01,
          iteration=2000,
          early_stop_epoch=50,
          log_interval=10,
          resolution=1,
          sample_rate=1.,
          learnable_w=False,
          highly_variable=None,
          common_genes=True,
          recon_ratio=1.,
          cl_ratio=1.,
          mnn_components=10,
          mnn_k=10,
          k_to_m_ratio=0.5,
          knn_k=20,
          margin=0.5,
          resolution_l=1.,
          resolution_preclus=0.2
          ):

    n_batch = len(batch_name)
    single_adata = []
    for i in range(n_batch):
        single_adata.append(adata[adata.obs['batch'].values == batch_name[i]])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert decoder in ['Dot', 'ZINB'], "Please choose decoder in ['Dot','ZINB']"
    assert sample_rate <= 1, "Please set 0<sample_rate<=1"

    ####################   Prepare data for training   ####################
    cell_type = np.array(adata.obs[cl_type].values)

    raw_exp = True if decoder == 'ZINB' else False
    if n_clusters is None:
        if cell_type:
            n_clusters = len(np.unique(cell_type))
        else:
            raise Exception('Please input number of clusters or set cell type information in adata.obs[cl_type]')

    single_adata, exp_value, exp_enc_graph, exp_graph, exp_val_graph, exp_val_value, unexp_edges = make_graph(single_adata, n_batch, raw_exp, highly_variable,common_genes)

    n_pos_edges = []
    n_neg_edges = []
    n_cells = []

    for i in range(n_batch):
        n_pos_edges.append(int(sample_rate * len(exp_value[i])))
        n_neg_edges.append(int(sample_rate * len(unexp_edges[i][0])))

        exp_value[i] = torch.tensor(exp_value[i], device=device)
        exp_val_value[i] = torch.tensor(exp_val_value[i], device=device)
        exp_graph[i] = exp_graph[i].to(device)
        exp_val_graph[i] = exp_val_graph[i].to(device)

        n_cells.append(single_adata[i].X.shape[0])

    n_genes = adata.n_vars
    exp_enc_graph = exp_enc_graph.to(device)

    #######################   Prepare models   #######################

    model = scHetG(n_layers=n_layers,
                     n_cells=n_cells,
                     n_genes=n_genes,
                     drop_out=drop_out,
                     decoder=decoder,
                     feats_dim=feats_dim,
                     learnable_weight=learnable_w).to(device)
    # print(model)
    model = model.to(device)

    if decoder in ['Dot']:
        criterion = nn.MSELoss()
    else:
        criterion = ZINBLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)


    #######################   Record values   #######################
    all_loss = []
    all_loss_val = []

    count_cl_loss = 0

    stop_flag = False
    #######################   Start training model   #######################
    print(f"Start training on {device}...")
    all_unexp_index = []
    unexp_dec_graph = []
    exp_dec_graph = []
    exp_dec_value = []
    all_index = []

    for i in range(n_batch):
        all_unexp_index.append(np.arange(len(unexp_edges[i][0])))
        unexp_sample_index = np.random.choice(all_unexp_index[i], n_neg_edges[i])
        unexp_sample = (unexp_edges[i][0][unexp_sample_index], unexp_edges[i][1][unexp_sample_index])
        unexp_dec_graph.append(dgl.heterograph({('cell'+str(i+1), 'exp'+str(i+1), 'gene'): unexp_sample},
                                           num_nodes_dict={'cell'+str(i+1): n_cells[i], 'gene': n_genes}).to(device))

        if sample_rate == 1:
            exp_dec_graph.append(exp_graph[i])
            exp_dec_value.append(exp_value[i])
            if decoder == 'ZINB':
                exp_dec_graph[i].nodes['cell'+str(i+1)].data['sz_factor'] = exp_graph[i].nodes['cell'+str(i+1)].data['sz_factor']
                exp_dec_graph[i].nodes['gene'].data['ge_factor'] = exp_graph[i].nodes['gene'].data['ge_factor']
        else:
            all_index.append(np.arange(len(exp_value[i])))

        single_adata[i] = precluster(single_adata[i], resolution=resolution_preclus)
        print("precluster adata"+str(i+1)+":", len(single_adata[i].obs['precluster'].unique()))
    adata = ad.concat(single_adata, merge="same")

    del single_adata

    # Constructing Mnns
    if n_batch<=5:
        combinations_list = list(combinations(batch_name, 2))  # 有n(n-1)/2个组合
    else:
        n_group = math.ceil((n_batch/5))
        combinations_list = list(combinations(batch_name[:5], 2))
        reference_batch_name = [batch_name[:5][n_cells[:5].index(max(n_cells[:5]))]]
        for group in range(1, n_group):
            if group < n_group-1:
                combinations_list = combinations_list + list(combinations(batch_name[group*5:(group+1)*5], 2))
                reference_batch_name.append(batch_name[group*5:(group+1)*5][n_cells[group*5:(group+1)*5].index(max(n_cells[group*5:(group+1)*5]))])
            else:
                combinations_list = combinations_list + list(combinations(batch_name[group * 5:], 2))
                reference_batch_name.append(batch_name[group * 5:][n_cells[group * 5:].index(max(n_cells[group * 5:]))])
        combinations_list = combinations_list + list(combinations(reference_batch_name, 2))

    adata_batch1 = adata[adata.obs['batch'].values == combinations_list[0][0]]
    adata_batch2 = adata[adata.obs['batch'].values == combinations_list[0][1]]
    if common_genes:
        mnn_id_x, mnn_id_y = mnn_edges_svd(adata_batch1, adata_batch2, n_components=mnn_components, k=mnn_k)
    else:
        mnn_id_x, mnn_id_y = mnn_edges_pca(adata_batch1, adata_batch2, n_components=mnn_components, k=mnn_k)
    batch1_order, batch2_order = int(combinations_list[0][0][-1]), int(combinations_list[0][1][-1])
    mnn_id_x, mnn_id_y = mnn_id_x + sum(n_cells[:(batch1_order - 1)]), mnn_id_y + sum(n_cells[:(batch2_order - 1)])

    for i in range(1, len(combinations_list)):
        adata_batch1 = adata[adata.obs['batch'].values == combinations_list[i][0]]
        adata_batch2 = adata[adata.obs['batch'].values == combinations_list[i][1]]
        if common_genes:
            mnn_id_x_, mnn_id_y_ = mnn_edges_svd(adata_batch1, adata_batch2, n_components=mnn_components, k=mnn_k)
        else:
            mnn_id_x_, mnn_id_y_ = mnn_edges_pca(adata_batch1, adata_batch2, n_components=mnn_components, k=mnn_k)
        batch1_order, batch2_order = int(combinations_list[i][0][-1]), int(combinations_list[i][1][-1])
        mnn_id_x_, mnn_id_y_ = mnn_id_x_ + sum(n_cells[:(batch1_order - 1)]), mnn_id_y_ + sum(
            n_cells[:(batch2_order - 1)])
        mnn_id_x, mnn_id_y = np.concatenate((mnn_id_x, mnn_id_x_)), np.concatenate((mnn_id_y, mnn_id_y_))

    # Modify the parameter knn_k to ensure that the program runs properly
    batch_cell = np.delete(np.arange(adata.n_obs), np.unique(np.concatenate((mnn_id_x, mnn_id_y))))
    if batch_cell.shape[0]*knn_k > 0.5 * len(mnn_id_x):
        knn_id_x, knn_id_y, knn_type, batch_cell, row_indices, col_indices = knn_edges(adata, mnn_id_x, mnn_id_y,
                                                                                       k_to_m_ratio=k_to_m_ratio,
                                                                                       k=knn_k)
    else:
        if batch_cell.shape[0] < knn_k * 2:
            knn_k = batch_cell.shape[0]
        else:
            knn_k = knn_k * 2
            k_to_m_ratio = k_to_m_ratio * 5
        knn_id_x, knn_id_y, knn_type, batch_cell, row_indices, col_indices = knn_edges(adata, mnn_id_x, mnn_id_y,
                                                                                       k_to_m_ratio=k_to_m_ratio,
                                                                                       k=knn_k)

    ridge = [None] * n_batch

    for iter_idx in range(iteration):
        if iter_idx % 10 == 0:
            # Updating the triplets
            if row_indices.shape[0] != 0:
                if knn_type == 0:
                    random_indices = np.random.choice(len(row_indices), int(k_to_m_ratio * len(mnn_id_x)), replace=False)
                    random_row_indices = row_indices[random_indices]
                    random_col_indices = col_indices[random_indices]
                    knn_id_x = batch_cell[random_row_indices]
                    knn_id_y = batch_cell[random_col_indices]
                else:
                    random_indices = np.random.choice(len(row_indices), int(k_to_m_ratio * len(mnn_id_x)), replace=False)
                    random_row_indices = row_indices[random_indices]
                    random_col_indices = col_indices[random_indices]
                    knn_id_x = batch_cell[random_row_indices]
                    knn_id_y = np.arange(adata.n_obs)[random_col_indices]
                anchors, postives = np.concatenate((mnn_id_x, knn_id_x)), np.concatenate((mnn_id_y, knn_id_y))
                negatives = precluster_negative_paris(adata, n_cells, mnn_id_x, knn_id_x)
            else:
                knn_id_x, knn_id_y = [], []
                anchors, postives = mnn_id_x, mnn_id_y
                negatives = precluster_negative_paris(adata, n_cells, mnn_id_x, np.array([]).astype('int'))

        if sample_rate != 1:
            exp_dec_value = []
            exp_dec_graph = []
            for i in range(n_batch):
                exp_sample_index = np.random.choice(all_index[i], n_pos_edges[i])
                exp_dec_value.append(exp_value[i][exp_sample_index])
                exp_sample = (exp_graph[i].edges()[0][exp_sample_index], exp_graph[i].edges()[1][exp_sample_index])

                exp_dec_graph.append(dgl.heterograph({('cell'+str(i+1), 'exp'+str(i+1), 'gene'): exp_sample},
                                                 num_nodes_dict={'cell'+str(i+1): n_cells[i], 'gene': n_genes})
                                     )

                if decoder == 'ZINB':
                    exp_dec_graph[i].nodes['cell'+str(i+1)].data['sz_factor'] = exp_graph[i].nodes['cell'+str(i+1)].data['sz_factor']
                    exp_dec_graph[i].nodes['gene'].data['ge_factor'] = exp_graph[i].nodes['gene'].data['ge_factor']

        if decoder == 'ZINB':
            for i in range(n_batch):
                unexp_dec_graph[i].nodes['cell'+str(i+1)].data['sz_factor'] = exp_graph[i].nodes['cell'+str(i+1)].data['sz_factor']
                unexp_dec_graph[i].nodes['gene'].data['ge_factor'] = exp_graph[i].nodes['gene'].data['ge_factor']
                exp_val_graph[i].nodes['cell'+str(i+1)].data['sz_factor'] = exp_graph[i].nodes['cell'+str(i+1)].data['sz_factor']
                exp_val_graph[i].nodes['gene'].data['ge_factor'] = exp_graph[i].nodes['gene'].data['ge_factor']


        pred_exp, pred_unexp = model(exp_enc_graph, exp_dec_graph, unexp_dec_graph)


        if decoder in ['Dot']:
            loss_exp = sum([criterion(pred_exp[i].type(torch.float64),exp_dec_value[i]) for i in range(n_batch)])
            loss_unexp = sum([criterion(pred_unexp[i],torch.zeros_like(pred_unexp[i]).to(device)) for i in range(n_batch)])

        else:
            loss_exp = sum([criterion(pred_exp[i][0], pred_exp[i][1], pred_exp[i][2], exp_dec_value[i]) for i in range(n_batch)])
            loss_unexp = sum([criterion(pred_unexp[i][0], pred_unexp[i][1], pred_unexp[i][2]) for i in range(n_batch)])


        # TODO: check the reg_loss and hyper-parameter

        reg_loss = (1/2) * sum([model.cell_feature[i].norm(2).pow(2) for i in range(n_batch)] + [model.gene_feature.norm(2).pow(2)])/float(adata.n_obs + n_genes)

        loss = loss_exp + gamma * loss_unexp + 0.0001*reg_loss

        if decoder == 'ZINB':
            ridge = sum([torch.square(pred_exp[i][2]).mean() + torch.square(pred_unexp[i][2]).mean() for i in range(n_batch)])
            loss = recon_ratio*loss + 1e-3 * ridge
        else:
            loss = recon_ratio * loss

        cl_loss = cl_crition(model.emb_cell, anchors, postives, negatives, margin)

        count_cl_loss += cl_loss.item()

        loss = loss + cl_ratio * cl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_loss.append(loss.item())

        # validating
        model.eval()
        with torch.no_grad():
            pred_val_exp = model(exp_enc_graph, exp_val_graph)
        model.train()

        if decoder in ['Dot']:
            loss_val = sum([criterion(pred_val_exp[i].type(torch.float64), exp_val_value[i]) for i in range(n_batch)])
        else:
            loss_val = sum([criterion(pred_val_exp[i][0], pred_val_exp[i][1], pred_val_exp[i][2], exp_val_value[i]) for i in range(n_batch)])

        all_loss_val.append(loss_val.item())

        if (iter_idx+1) % log_interval == 0:

            print("[{}/{}-iter]" .format(iter_idx+1, iteration))
            unexp_dec_graph = []
            for i in range(n_batch):
                unexp_sample_index = np.random.choice(all_unexp_index[i], n_neg_edges[i])
                unexp_sample = (unexp_edges[i][0][unexp_sample_index], unexp_edges[i][1][unexp_sample_index])
                unexp_dec_graph.append(
                    dgl.heterograph({('cell' + str(i + 1), 'exp' + str(i + 1), 'gene'): unexp_sample},
                                    num_nodes_dict={'cell' + str(i+1): n_cells[i], 'gene': n_genes}).to(device))

            if decoder in ['Dot']:
                if (iter_idx+1) >= early_stop_epoch and recon_ratio*loss_val+0.0001*reg_loss + cl_ratio*cl_loss >= loss:
                    print("Reach stop critrion!")
                    stop_flag = True
            else:
                if (iter_idx+1) >= early_stop_epoch and recon_ratio*loss_val+0.0001*reg_loss+1e-3 * ridge + cl_ratio*cl_loss >= loss:
                    print("Reach stop critrion!")
                    stop_flag = True

            torch.cuda.empty_cache()

        if (iter_idx+1) == iteration or stop_flag or (iter_idx+1) % log_interval == 0:

            adata.obsm['feat'] = np.concatenate([model.emb_cell[i].data.cpu().numpy() for i in range(n_batch)], axis=0)
            adata.varm['feat'] = model.emb_gene.data.cpu().numpy()

            if stop_flag:
                break


    # Visualization of the training process
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 10))
    plt.subplot(111)
    plt.plot(np.arange(len(all_loss)), all_loss, label='train loss')
    plt.plot(np.arange(len(all_loss)), all_loss_val, label='valid loss')
    plt.legend()
    plt.title("loss")
    plt.show()
    plt.savefig('Metric.png', dpi=800)
    plt.show()

    print("model.weights:", model.weights)
    del model
    # del all_exp
    gc.collect()
    torch.cuda.empty_cache()

    return adata
