import bbknn
import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
from time import time
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80)
import harmonypy as hm
import scib
from memory_profiler import memory_usage

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
        adata1 = ad.read_h5ad('./datasets/human_pancreas_2/rna_seq_baron.h5ad')
        adata2 = ad.read_h5ad('./datasets/human_pancreas_2/rna_seq_segerstolpe.h5ad')
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
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_genes=200)
    #adata.X=np.array(adata.X.todense())
    sc.pp.pca(adata)
    bbknn.bbknn(adata, batch_key='batch')
    sc.tl.umap(adata)

    sc.tl.leiden(adata, resolution=0.4)

    adata.X = np.asarray(adata.X.todense())
    bbknn.ridge_regression(adata, batch_key=['batch'], confounder_key=['leiden'])
    #adata.X=np.asarray(adata.X)
    sc.pp.pca(adata)
    bbknn.bbknn(adata, batch_key='batch')
    sc.tl.umap(adata)

    adata.obsm['X_latent'] = adata.obsm['X_pca']

    mem_used = memory_usage(-1, interval=.1, timeout=1)
    print("elapsed memory:", max(mem_used))
    end = time()
    print('elapsed{:.2f} seconds'.format(end - start))

    sc.tl.louvain(adata, resolution=0.5)
    adata.write_h5ad("BBKNN_"+dataset+ ".h5ad")