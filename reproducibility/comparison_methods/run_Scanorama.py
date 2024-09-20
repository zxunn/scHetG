import numpy as np
import pandas as pd
import scanpy as sc
import scanorama
from time import time
import anndata as ad
import harmonypy as hm
import scipy.sparse
import scanpy.external as sce
from memory_profiler import memory_usage
#from util.utils_B import set_seed
#from memory_profiler import profile

# # 1. Running 10 sampled data ---------------------------------------------------------
data_dir = '../../datasets/simulate/'
seed = [0, 1, 4, 6, 7, 8, 10, 11, 100, 111]
"""
for i in seed:
    set_seed(2023)
    d = np.load(data_dir + 'sampling/' + str(i) + '_' + 'simulate_raw.npz', allow_pickle=True)

    adata = ad.AnnData(d['X_latent'])
    adata.obs['celltype'] = np.array(d['celltype'])
    adata.obs['batch'] = np.array(d['batch'])
    adata.obs['condition'] = np.array(d['condition'])

    adata.obs['celltype'] = adata.obs['celltype'].astype('category')
    adata.obs['batch'] = adata.obs['batch'].astype('category')
    adata.obs['condition'] = adata.obs['condition'].astype('category')

    adata_corr = []
    genes_ = []
    all_batch = list(set(adata.obs['batch']))
    for b in all_batch:
        adata_corr.append(adata.X[adata.obs['batch'] == b, :])
        genes_.append(adata.var_names)

    integrated, corrected, genes = scanorama.correct(adata_corr, genes_, return_dimred=True)

    scanorama_res = np.concatenate(integrated)
    inted = pd.DataFrame(scanorama_res)
    adata_inted = ad.AnnData(inted, obs=adata.obs, dtype='float64')
    adata_inted.obsm['X_latent'] = adata_inted.X
    adata_inted.obs['celltype'] = np.array(adata.obs['celltype'])
    adata_inted.obs['batch'] = np.array(adata.obs['batch'])
    adata_inted.obs['condition'] = np.array(adata.obs['condition'])

    adata_inted.write_h5ad(data_dir + 'sampling/' + str(i) + '_' + 'simulate_scanorama.h5ad')
"""
# # 2. Running Complete data by scanorama -------------------------------------------------------
#@profile
def my_func(adata):
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.recipe_zheng17(adata)
    sc.tl.pca(adata)
    sce.pp.scanorama_integrate(adata, 'batch')
    # #set_seed(2023)
    # if isinstance(adata.X, scipy.sparse.csr_matrix) or isinstance(adata.X, scipy.sparse._csc.csc_matrix):
    #     ho = hm.run_harmony(adata.X.todense(), adata.obs, ['batch'])
    # else:
    #     ho = hm.run_harmony(adata.X.astype(np.float64), adata.obs, ['batch'])
    # #ho = hm.run_harmony(adata.X.toarray(), adata.obs, ['batch'])
    return adata


#set_seed(2023)
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
    adata_corrd = my_func(adata)

    mem_used = memory_usage(-1, interval=.1, timeout=1)
    print("elapsed memory:", max(mem_used))
    end = time()
    print('elapsed{:.2f} seconds'.format(end - start))
    #import matplotlib
    #matplotlib.use('TkAgg')
    sc.pp.neighbors(adata_corrd,use_rep='X_scanorama')
    sc.tl.louvain(adata_corrd, resolution=0.5)

    adata_corrd.write_h5ad("Scanorama_scanpy_"+dataset+".h5ad")

