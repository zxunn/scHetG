import numpy as np
import scanpy as sc
import torch
import anndata as ad
import pandas as pd
from scHetG_master import train_scHetG
from scHetG_master import preprocess
import random
import warnings
warnings.filterwarnings('ignore')
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from scipy.sparse import csr_matrix
from time import time
import scipy.sparse
from memory_profiler import memory_usage
from scHetG_master import louvain, calculate_metric, calculate_ber
import scib
import harmonypy as hm

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


for dataset in ['mouse_pancreas', 'mouse_atlas','human_pancreas','human_pancreas_2','human_lung','human_heart','mouse_brain']:

    print('----------------data: {} ----------------- '.format(dataset))

    if dataset == 'mouse_pancreas':
        csv_data = pd.read_csv('../../datasets/mouse_pancreas/GSM2230761_mouse1_umifm_counts.csv')
        barcode = csv_data['barcode'].values
        assigned_cluster = csv_data['assigned_cluster'].values
        gene_expression = csv_data.iloc[:, 3:].values
        adata1 = ad.AnnData(X=gene_expression, obs={'barcode': barcode, 'celltype': assigned_cluster})
        csv_data = pd.read_csv('../../datasets/mouse_pancreas/GSM2230762_mouse2_umifm_counts.csv')
        barcode = csv_data['barcode'].values
        assigned_cluster = csv_data['assigned_cluster'].values
        gene_expression = csv_data.iloc[:, 3:].values
        adata2 = ad.AnnData(X=gene_expression, obs={'barcode': barcode, 'celltype': assigned_cluster})
        adata = [adata1, adata2]
        adata = ad.concat(adata, merge="same")
        adata.obs['batch'] = np.concatenate((np.array(['batch1'] * adata1.n_obs), np.array(['batch2'] * adata2.n_obs)), axis=0)
        adata.var_names = csv_data.columns[3:]
        batch_name = np.array(['batch1', 'batch2'], dtype=object)

    elif dataset == 'mouse_atlas':
        adata1 = ad.read_h5ad('../../datasets/mouse_atlas/rna_seq_mi.h5ad')
        adata2 = ad.read_h5ad('../../datasets/mouse_atlas/rna_seq_sm.h5ad')
        adata = [adata1, adata2]
        adata = ad.concat(adata, merge="same")
        adata.obs['batch'] = np.concatenate((np.array(['batch1']*adata1.n_obs), np.array(['batch2']*adata2.n_obs)), axis=0)
        batch_name = np.array(['batch1', 'batch2'], dtype=object)

    elif dataset == 'mouse_brain':
        adata = sc.read_h5ad("../../datasets/mouse_brain/sub_mouse_brain.h5ad")
        adata.obs.rename(columns={'BATCH': 'batch'}, inplace=True)
        adata1 = adata[adata.obs['batch'].values == 'batch1']
        adata2 = adata[adata.obs['batch'].values == 'batch2']
        adata = ad.concat([adata1, adata2], merge='same')
        batch_name = np.array(['batch1', 'batch2'], dtype=object)

    elif dataset == 'human_pancreas_2':
        adata1 = ad.read_h5ad('../../datasets/human_pancreas_2/rna_seq_baron.h5ad')
        adata2 = ad.read_h5ad('../../datasets/human_pancreas_2/rna_seq_segerstolpe.h5ad')
        adata = [adata1, adata2]
        adata = ad.concat(adata, merge="same")
        adata.obs.rename(columns={'cell_type1': 'celltype'}, inplace=True)
        adata.obs['batch'] = np.concatenate((np.array(['batch1']*adata1.n_obs), np.array(['batch2']*adata2.n_obs)), axis=0)
        batch_name = np.array(['batch1', 'batch2'], dtype=object)

    elif dataset == 'human_pancreas':
        adata = sc.read("../../datasets/human_pancreas/human_pancreas.h5ad")
        batch_name = np.array(['human1', 'human2', 'human3', 'human4'], dtype=object)

    elif dataset == 'human_lung':
        adata = sc.read_h5ad("../../datasets/human_lung/human_lung_marker.h5ad")
        adata.obs['batch'].replace('muc3843', 'batch1', inplace=True)
        adata.obs['batch'].replace('muc4658', 'batch2', inplace=True)
        adata.obs['batch'].replace('muc5103', 'batch3', inplace=True)
        adata.obs['batch'].replace('muc5104', 'batch4', inplace=True)
        batch_name = np.array(['batch1', 'batch2', 'batch3', 'batch4'], dtype=object)


    elif dataset == 'human_heart':
        adata = sc.read("../../datasets/human_heart/healthy_human_heart.h5ad")
        adata.obs.rename(columns={'sampleID': 'batch'}, inplace=True)
        unique_sampleIDs = adata.obs['batch'].values.unique()[-10:]
        adata = adata[adata.obs['batch'].isin(unique_sampleIDs)]
        batch_name = unique_sampleIDs
        for i in range(len(batch_name)):
            adata.obs['batch'].replace(batch_name[i], 'batch'+str(i+1), inplace=True)
        batch_name=np.array(['batch1', 'batch2', 'batch3','batch4','batch5','batch6','batch7','batch8','batch9','batch10'])

    start = time()

    adata.X = adata.X.astype(np.float64)
    adata = preprocess(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor='seurat')
    highly_variable = adata.var['highly_variable']
    adata = adata[:, adata.var['highly_variable']]
    n_clusters = len(np.unique(adata.obs['celltype'].values))
    n_batch = len(np.unique(adata.obs['batch'].values))
    print(adata, "\n", "n_batch:", n_batch, "\n", "n_clusters:", n_clusters)

    if isinstance(adata.X, scipy.sparse.csr_matrix) or isinstance(adata.X, scipy.sparse._csc.csc_matrix):
        print("Sparsity: ", np.where(adata.X.todense() == 0)[0].shape[0] / (adata.X.todense().shape[0] * adata.X.todense().shape[1]))
    else:
        print("Sparsity: ", np.where(adata.X == 0)[0].shape[0] / (adata.X.shape[0] * adata.X.shape[1]))

    #training
    adata = train_scHetG(adata, batch_name, n_clusters=n_clusters, cl_type='celltype', use_graph=True,
                         feats_dim=64, lr=0.15, n_layers=2,
                         drop_out=0.1, decoder='Dot', gamma=1,
                         log_interval=10, iteration=200, early_stop_epoch=50,
                         sample_rate=0.1, learnable_w=True, highly_variable=highly_variable, common_genes=True,
                         recon_ratio=1., cl_ratio=1.,
                         mnn_components=10, mnn_k=10,
                         k_to_m_ratio=0.1, knn_k=5,
                         margin=1.5, resolution_l=0.5, resolution_preclus=0.2)

    mem_used = memory_usage(-1, interval=.1, timeout=1)
    print("elapsed memory:", max(mem_used))
    end = time()
    print('elapsed{:.2f} seconds'.format(end - start))

    adata.obs['celltype'] = adata.obs['celltype'].astype('category')
    print(adata)

    # louvain
    adata = louvain(adata, resolution=0.5, use_rep='feat')
    y_pred_l = np.array(adata.obs['louvain'])
    print('Number of clusters identified by Louvain is {}'.format(len(np.unique(y_pred_l))))
    cell_type = adata.obs['celltype'].values
    acc, nmi, ari = calculate_metric(cell_type, y_pred_l)
    ber = calculate_ber(np.array(cell_type), np.array(y_pred_l), adata.obs['batch'])
    print("ACC:", acc)
    print("NMI:", nmi)
    print("ARI:", ari)
    print("BER:", ber)

    gc = scib.me.graph_connectivity(adata, label_key="celltype")
    print("GC:", gc)

    iLISI = hm.compute_lisi(adata.obsm['feat'], adata.obs, ['batch'],30)
    iLISI = (iLISI).mean()
    cLISI = hm.compute_lisi(adata.obsm['feat'], adata.obs, ['celltype'],30)
    cLISI = (cLISI).mean()

    print("iLISI:", iLISI)
    print("cLISI:", cLISI)

    sc.tl.umap(adata)
    sc.pl.umap(adata, color=['batch', 'celltype', 'louvain'], save="scHetG_dot"+dataset+".pdf")
    adata.write_h5ad("scHetG_dot" + dataset + ari.astype('str')+".h5ad")




