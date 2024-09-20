import os              
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import simba as si
#from util.utils_B import set_seed
from time import time
from memory_profiler import profile
from baseline.utils import calculate_ber
import shutil
from scipy.sparse import csr_matrix
os.environ['PYTHONHASHSEED'] = '0'
#os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from memory_profiler import memory_usage


#from IPython.display import set_matplotlib_formats
#set_matplotlib_formats('retina')
#@profile
def my_func2(adata_CG_mi, adata_CG_sm, dataset):
    workdir = 'result_'+dataset
    si.settings.set_workdir(workdir)
    si.settings.set_figure_params(dpi=80,
                                  style='white',
                                  fig_size=[5, 5],
                                  rc={'image.cmap': 'viridis'})

    si.pp.filter_genes(adata_CG_mi,min_n_cells=3)
    si.pp.cal_qc_rna(adata_CG_mi)
    si.pp.normalize(adata_CG_mi,method='lib_size')
    si.pp.log_transform(adata_CG_mi)
    si.pp.select_variable_genes(adata_CG_mi, n_top_genes=3000)
    si.tl.discretize(adata_CG_mi,n_bins=5)
    si.pp.filter_genes(adata_CG_sm,min_n_cells=3)
    si.pp.cal_qc_rna(adata_CG_sm)
    si.pp.normalize(adata_CG_sm,method='lib_size')
    si.pp.log_transform(adata_CG_sm)
    si.pp.select_variable_genes(adata_CG_sm, n_top_genes=3000)
    si.tl.discretize(adata_CG_sm,n_bins=5)
    adata_CmiCsm = si.tl.infer_edges(adata_CG_mi, adata_CG_sm, n_components=20, k=20)
    si.tl.trim_edges(adata_CmiCsm, cutoff=0.5)
    si.tl.gen_graph(list_CG=[adata_CG_mi, adata_CG_sm],
                    list_CC=[adata_CmiCsm],
                    copy=False,
                    use_highly_variable=True,
                    dirname='graph0')
    si.settings.pbg_params
    #si.tl.pbg_train(auto_wd=True, save_wd=True, output='model')
    # modify parameters
    dict_config = si.settings.pbg_params.copy()
    # dict_config['wd'] = 0.026209
    dict_config['workers'] = 12

    ## start training
    si.tl.pbg_train(pbg_params=dict_config, auto_wd=True, save_wd=True, output='model')
    #si.settings.pbg_params = dict_config.copy()
    # load in graph ('graph0') info
    si.load_graph_stats()
    # load in model info for ('graph0')
    si.load_pbg_config()
    # load in graph ('graph0') info
    si.load_graph_stats(path='./'+workdir+'/pbg/graph0/')
    # load in model info for ('graph0')
    si.load_pbg_config(path='./'+workdir+'/pbg/graph0/model/')
    dict_adata = si.read_embedding()
    adata_C = dict_adata['C']  # embeddings for cells
    adata_C2 = dict_adata['C2']  # embeddings for C2
    adata_G = dict_adata['G']  # embeddings for genes
    adata_C.obs['celltype'] = adata_CG_mi[adata_C.obs_names, :].obs['celltype'].copy().values
    adata_C2.obs['celltype'] = adata_CG_sm[adata_C2.obs_names, :].obs['celltype'].copy().values

    #adata_C.write_h5ad("../data/adata_C.h5ad")
    #adata_C2.write_h5ad("../data/adata_C2.h5ad")
    #adata_G.write_h5ad("../data/adata_G.h5ad")

    adata_all_CG = si.tl.embed(adata_ref=adata_C,list_adata_query=[adata_C2,adata_G],use_precomputed=False)
    adata_all_CG.obs['entity_anno'] = ""
    adata_all_CG.obs.loc[adata_C.obs_names, 'entity_anno'] = adata_all_CG.obs.loc[adata_C.obs_names, 'celltype'].tolist()
    adata_all_CG.obs.loc[adata_C2.obs_names, 'entity_anno'] = adata_all_CG.obs.loc[adata_C2.obs_names, 'celltype'].tolist()
    adata_all_CG.obs.loc[adata_G.obs_names, 'entity_anno'] = 'gene'
    del adata_all_CG.obs['celltype']
    #adata_all_CG.obs.rename(columns={'entity_anno': 'celltype'}, inplace=True)

    shutil.rmtree(workdir)
    return adata_all_CG


def my_func3(adata1, adata2,adata3, dataset):
    workdir = 'result_'+dataset
    si.settings.set_workdir(workdir)
    si.settings.set_figure_params(dpi=80,
                                  style='white',
                                  fig_size=[5, 5],
                                  rc={'image.cmap': 'viridis'})
    si.pp.filter_genes(adata1,min_n_cells=3)
    si.pp.cal_qc_rna(adata1)
    si.pp.normalize(adata1,method='lib_size')
    si.pp.log_transform(adata1)
    si.pp.select_variable_genes(adata1, n_top_genes=3000)
    si.tl.discretize(adata1,n_bins=5)

    si.pp.filter_genes(adata2,min_n_cells=3)
    si.pp.cal_qc_rna(adata2)
    si.pp.normalize(adata2,method='lib_size')
    si.pp.log_transform(adata2)
    si.pp.select_variable_genes(adata2, n_top_genes=3000)
    si.tl.discretize(adata2,n_bins=5)

    si.pp.filter_genes(adata3,min_n_cells=3)
    si.pp.cal_qc_rna(adata3)
    si.pp.normalize(adata3,method='lib_size')
    si.pp.log_transform(adata3)
    si.pp.select_variable_genes(adata3, n_top_genes=3000)
    si.tl.discretize(adata3,n_bins=5)


    adata_C1C2 = si.tl.infer_edges(adata1, adata2, n_components=15, k=15)
    adata_C1C3 = si.tl.infer_edges(adata1, adata3, n_components=15, k=15)


    si.tl.gen_graph(list_CG=[adata1,adata2,adata3],
                    list_CC=[adata_C1C2,adata_C1C3],
                    copy=False,
                    dirname='graph0')
    si.settings.pbg_params
    # modify parameters
    dict_config = si.settings.pbg_params.copy()
    # dict_config['wd'] = 0.00477
    dict_config['workers'] = 12

    ## start training
    si.tl.pbg_train(pbg_params=dict_config, auto_wd=True, save_wd=True, output='model')

    # load in graph ('graph0') info
    si.load_graph_stats()
    # load in model info for ('graph0')
    si.load_pbg_config()

    # load in graph ('graph0') info
    si.load_graph_stats(path='./result_'+dataset+'/pbg/graph0/')
    # load in model info for ('graph0')
    si.load_pbg_config(path='./result_'+dataset+'/pbg/graph0/model/')

    dict_adata = si.read_embedding()
    adata_C = dict_adata['C']  # embeddings for cells from data baron
    adata_C2 = dict_adata['C2']  # embeddings for cells from data segerstolpe
    adata_C3 = dict_adata['C3']  # embeddings for cells from data muraro
    adata_G = dict_adata['G']  # embeddings for genes

    adata_C.obs['celltype'] = adata1[adata_C.obs_names, :].obs['celltype'].copy().values
    adata_C2.obs['celltype'] = adata2[adata_C2.obs_names, :].obs['celltype'].copy().values
    adata_C3.obs['celltype'] = adata3[adata_C3.obs_names, :].obs['celltype'].copy().values


    adata_all_CG = si.tl.embed(adata_ref=adata_C,
                            list_adata_query=[adata_C2, adata_C3, adata_G],
                            use_precomputed=False)
    adata_all_CG.obs['entity_anno'] = ""
    adata_all_CG.obs.loc[adata_C.obs_names, 'entity_anno'] = adata_all_CG.obs.loc[adata_C.obs_names, 'celltype'].tolist()
    adata_all_CG.obs.loc[adata_C2.obs_names, 'entity_anno'] = adata_all_CG.obs.loc[adata_C2.obs_names, 'celltype'].tolist()
    adata_all_CG.obs.loc[adata_C3.obs_names, 'entity_anno'] = adata_all_CG.obs.loc[adata_C3.obs_names, 'celltype'].tolist()
    adata_all_CG.obs.loc[adata_G.obs_names, 'entity_anno'] = 'gene'
    del adata_all_CG.obs['celltype']
    #adata_all_CG.obs.rename(columns={'entity_anno': 'celltype'}, inplace=True)

    shutil.rmtree(workdir)
    return adata_all_CG


def my_func4(adata1, adata2,adata3, adata4, dataset):
    workdir = 'result_'+dataset
    si.settings.set_workdir(workdir)
    si.settings.set_figure_params(dpi=80,
                                  style='white',
                                  fig_size=[5, 5],
                                  rc={'image.cmap': 'viridis'})
    si.pp.filter_genes(adata1,min_n_cells=3)
    si.pp.cal_qc_rna(adata1)
    si.pp.normalize(adata1,method='lib_size')
    si.pp.log_transform(adata1)
    si.pp.select_variable_genes(adata1, n_top_genes=3000)
    si.tl.discretize(adata1,n_bins=5)

    si.pp.filter_genes(adata2,min_n_cells=3)
    si.pp.cal_qc_rna(adata2)
    si.pp.normalize(adata2,method='lib_size')
    si.pp.log_transform(adata2)
    si.pp.select_variable_genes(adata2, n_top_genes=3000)
    si.tl.discretize(adata2,n_bins=5)

    si.pp.filter_genes(adata3,min_n_cells=3)
    si.pp.cal_qc_rna(adata3)
    si.pp.normalize(adata3,method='lib_size')
    si.pp.log_transform(adata3)
    si.pp.select_variable_genes(adata3, n_top_genes=3000)
    si.tl.discretize(adata3,n_bins=5)

    si.pp.filter_genes(adata4,min_n_cells=3)
    si.pp.cal_qc_rna(adata4)
    si.pp.normalize(adata4,method='lib_size')
    si.pp.log_transform(adata4)
    si.pp.select_variable_genes(adata4, n_top_genes=3000)
    si.tl.discretize(adata4,n_bins=5)

    adata_C1C2 = si.tl.infer_edges(adata1, adata2, n_components=15, k=15)
    adata_C1C3 = si.tl.infer_edges(adata1, adata3, n_components=15, k=15)
    adata_C1C4 = si.tl.infer_edges(adata1, adata4, n_components=15, k=15)


    si.tl.gen_graph(list_CG=[adata1,adata2,adata3,adata4],
                    list_CC=[adata_C1C2,adata_C1C3,adata_C1C4],
                    copy=False,
                    dirname='graph0')
    si.settings.pbg_params
    # modify parameters
    dict_config = si.settings.pbg_params.copy()
    # dict_config['wd'] = 0.00477
    dict_config['workers'] = 12

    ## start training
    si.tl.pbg_train(pbg_params=dict_config, auto_wd=True, save_wd=True, output='model')

    # load in graph ('graph0') info
    si.load_graph_stats()
    # load in model info for ('graph0')
    si.load_pbg_config()

    # load in graph ('graph0') info
    si.load_graph_stats(path='./result_'+dataset+'/pbg/graph0/')
    # load in model info for ('graph0')
    si.load_pbg_config(path='./result_'+dataset+'/pbg/graph0/model/')

    dict_adata = si.read_embedding()
    adata_C = dict_adata['C']  # embeddings for cells from data baron
    adata_C2 = dict_adata['C2']  # embeddings for cells from data segerstolpe
    adata_C3 = dict_adata['C3']  # embeddings for cells from data muraro
    adata_C4 = dict_adata['C4']  # embeddings for cells from data wang
    adata_G = dict_adata['G']  # embeddings for genes

    adata_C.obs['celltype'] = adata1[adata_C.obs_names.astype(str), :].obs['celltype'].copy().values
    adata_C2.obs['celltype'] = adata2[adata_C2.obs_names.astype(str), :].obs['celltype'].copy().values
    adata_C3.obs['celltype'] = adata3[adata_C3.obs_names.astype(str), :].obs['celltype'].copy().values
    adata_C4.obs['celltype'] = adata4[adata_C4.obs_names.astype(str), :].obs['celltype'].copy().values

    adata_all_CG = si.tl.embed(adata_ref=adata_C,
                            list_adata_query=[adata_C2, adata_C3, adata_C4, adata_G],
                            use_precomputed=False)
    adata_all_CG.obs['entity_anno'] = ""
    adata_all_CG.obs.loc[adata_C.obs_names, 'entity_anno'] = adata_all_CG.obs.loc[
        adata_C.obs_names, 'celltype'].tolist()
    adata_all_CG.obs.loc[adata_C2.obs_names, 'entity_anno'] = adata_all_CG.obs.loc[
        adata_C2.obs_names, 'celltype'].tolist()
    adata_all_CG.obs.loc[adata_C3.obs_names, 'entity_anno'] = adata_all_CG.obs.loc[
        adata_C3.obs_names, 'celltype'].tolist()
    adata_all_CG.obs.loc[adata_C4.obs_names, 'entity_anno'] = adata_all_CG.obs.loc[
        adata_C4.obs_names, 'celltype'].tolist()
    adata_all_CG.obs.loc[adata_G.obs_names, 'entity_anno'] = 'gene'

    adata_all_CG.obs.head()

    del adata_all_CG.obs['celltype']
    #adata_all_CG.obs.rename(columns={'entity_anno': 'celltype'}, inplace=True)
    shutil.rmtree(workdir)
    return adata_all_CG


def my_func5(adata1, adata2,adata3, adata4, adata5, dataset):
    workdir = 'result_'+dataset
    si.settings.set_workdir(workdir)
    si.settings.set_figure_params(dpi=80,
                                  style='white',
                                  fig_size=[5, 5],
                                  rc={'image.cmap': 'viridis'})
    si.pp.filter_genes(adata1,min_n_cells=3)
    si.pp.cal_qc_rna(adata1)
    si.pp.normalize(adata1,method='lib_size')
    si.pp.log_transform(adata1)
    si.pp.select_variable_genes(adata1, n_top_genes=3000)
    si.tl.discretize(adata1,n_bins=5)

    si.pp.filter_genes(adata2,min_n_cells=3)
    si.pp.cal_qc_rna(adata2)
    si.pp.normalize(adata2,method='lib_size')
    si.pp.log_transform(adata2)
    si.pp.select_variable_genes(adata2, n_top_genes=3000)
    si.tl.discretize(adata2,n_bins=5)

    si.pp.filter_genes(adata3,min_n_cells=3)
    si.pp.cal_qc_rna(adata3)
    si.pp.normalize(adata3,method='lib_size')
    si.pp.log_transform(adata3)
    si.pp.select_variable_genes(adata3, n_top_genes=3000)
    si.tl.discretize(adata3,n_bins=5)

    si.pp.filter_genes(adata4,min_n_cells=3)
    si.pp.cal_qc_rna(adata4)
    si.pp.normalize(adata4,method='lib_size')
    si.pp.log_transform(adata4)
    si.pp.select_variable_genes(adata4, n_top_genes=3000)
    si.tl.discretize(adata4,n_bins=5)

    si.pp.filter_genes(adata5,min_n_cells=3)
    si.pp.cal_qc_rna(adata5)
    si.pp.normalize(adata5,method='lib_size')
    si.pp.log_transform(adata5)
    si.pp.select_variable_genes(adata5, n_top_genes=3000)
    si.tl.discretize(adata5,n_bins=5)

    adata_C1C2 = si.tl.infer_edges(adata1, adata2, n_components=15, k=15)
    adata_C1C3 = si.tl.infer_edges(adata1, adata3, n_components=15, k=15)
    adata_C1C4 = si.tl.infer_edges(adata1, adata4, n_components=15, k=15)
    adata_C1C5 = si.tl.infer_edges(adata1, adata5, n_components=15, k=15)

    si.tl.gen_graph(list_CG=[adata1,adata2,adata3,adata4,adata5],
                    list_CC=[adata_C1C2,adata_C1C3,adata_C1C4,adata_C1C5],
                    copy=False,
                    dirname='graph0')
    si.settings.pbg_params
    # modify parameters
    dict_config = si.settings.pbg_params.copy()
    # dict_config['wd'] = 0.00477
    dict_config['workers'] = 12

    ## start training
    si.tl.pbg_train(pbg_params=dict_config, auto_wd=True, save_wd=True, output='model')

    # load in graph ('graph0') info
    si.load_graph_stats()
    # load in model info for ('graph0')
    si.load_pbg_config()

    # load in graph ('graph0') info
    si.load_graph_stats(path='./result_'+dataset+'/pbg/graph0/')
    # load in model info for ('graph0')
    si.load_pbg_config(path='./result_'+dataset+'/pbg/graph0/model/')

    dict_adata = si.read_embedding()
    adata_C = dict_adata['C']  # embeddings for cells from data baron
    adata_C2 = dict_adata['C2']  # embeddings for cells from data segerstolpe
    adata_C3 = dict_adata['C3']  # embeddings for cells from data muraro
    adata_C4 = dict_adata['C4']  # embeddings for cells from data wang
    adata_C5 = dict_adata['C5']  # embeddings for cells from data xin
    adata_G = dict_adata['G']  # embeddings for genes

    adata_C.obs['celltype'] = adata1[adata_C.obs_names, :].obs['celltype'].copy().values
    adata_C2.obs['celltype'] = adata2[adata_C2.obs_names, :].obs['celltype'].copy().values
    adata_C3.obs['celltype'] = adata3[adata_C3.obs_names, :].obs['celltype'].copy().values
    adata_C4.obs['celltype'] = adata4[adata_C4.obs_names, :].obs['celltype'].copy().values
    adata_C5.obs['celltype'] = adata5[adata_C5.obs_names, :].obs['celltype'].copy().values

    adata_all = si.tl.embed(adata_ref=adata_C,
                            list_adata_query=[adata_C2, adata_C3, adata_C4, adata_C5],
                            use_precomputed=False)
    adata_all.obs.rename(columns={'entity_group': 'batch'}, inplace=True)
    shutil.rmtree(workdir)
    return adata_all

def my_func10(adata1, adata2,adata3, adata4, adata5,adata6,adata7,adata8,adata9,adata10, dataset):
    workdir = 'result_'+dataset
    si.settings.set_workdir(workdir)
    si.settings.set_figure_params(dpi=80,
                                  style='white',
                                  fig_size=[5, 5],
                                  rc={'image.cmap': 'viridis'})
    si.pp.filter_genes(adata1,min_n_cells=3)
    si.pp.cal_qc_rna(adata1)
    si.pp.normalize(adata1,method='lib_size')
    si.pp.log_transform(adata1)
    si.pp.select_variable_genes(adata1, n_top_genes=3000)
    si.tl.discretize(adata1,n_bins=5)

    si.pp.filter_genes(adata2,min_n_cells=3)
    si.pp.cal_qc_rna(adata2)
    si.pp.normalize(adata2,method='lib_size')
    si.pp.log_transform(adata2)
    si.pp.select_variable_genes(adata2, n_top_genes=3000)
    si.tl.discretize(adata2,n_bins=5)

    si.pp.filter_genes(adata3,min_n_cells=3)
    si.pp.cal_qc_rna(adata3)
    si.pp.normalize(adata3,method='lib_size')
    si.pp.log_transform(adata3)
    si.pp.select_variable_genes(adata3, n_top_genes=3000)
    si.tl.discretize(adata3,n_bins=5)

    si.pp.filter_genes(adata4,min_n_cells=3)
    si.pp.cal_qc_rna(adata4)
    si.pp.normalize(adata4,method='lib_size')
    si.pp.log_transform(adata4)
    si.pp.select_variable_genes(adata4, n_top_genes=3000)
    si.tl.discretize(adata4,n_bins=5)

    si.pp.filter_genes(adata5,min_n_cells=3)
    si.pp.cal_qc_rna(adata5)
    si.pp.normalize(adata5,method='lib_size')
    si.pp.log_transform(adata5)
    si.pp.select_variable_genes(adata5, n_top_genes=3000)
    si.tl.discretize(adata5,n_bins=5)

    si.pp.filter_genes(adata6,min_n_cells=3)
    si.pp.cal_qc_rna(adata6)
    si.pp.normalize(adata6,method='lib_size')
    si.pp.log_transform(adata6)
    si.pp.select_variable_genes(adata6, n_top_genes=3000)
    si.tl.discretize(adata6,n_bins=5)

    si.pp.filter_genes(adata7,min_n_cells=3)
    si.pp.cal_qc_rna(adata7)
    si.pp.normalize(adata7,method='lib_size')
    si.pp.log_transform(adata7)
    si.pp.select_variable_genes(adata7, n_top_genes=3000)
    si.tl.discretize(adata7,n_bins=5)

    si.pp.filter_genes(adata8,min_n_cells=3)
    si.pp.cal_qc_rna(adata8)
    si.pp.normalize(adata8,method='lib_size')
    si.pp.log_transform(adata8)
    si.pp.select_variable_genes(adata8, n_top_genes=3000)
    si.tl.discretize(adata8,n_bins=5)

    si.pp.filter_genes(adata9,min_n_cells=3)
    si.pp.cal_qc_rna(adata9)
    si.pp.normalize(adata9,method='lib_size')
    si.pp.log_transform(adata9)
    si.pp.select_variable_genes(adata9, n_top_genes=3000)
    si.tl.discretize(adata9,n_bins=5)

    si.pp.filter_genes(adata10,min_n_cells=3)
    si.pp.cal_qc_rna(adata10)
    si.pp.normalize(adata10,method='lib_size')
    si.pp.log_transform(adata10)
    si.pp.select_variable_genes(adata10, n_top_genes=3000)
    si.tl.discretize(adata10,n_bins=5)

    adata_C1C2 = si.tl.infer_edges(adata1, adata2, n_components=15, k=15)
    adata_C1C3 = si.tl.infer_edges(adata1, adata3, n_components=15, k=15)
    adata_C1C4 = si.tl.infer_edges(adata1, adata4, n_components=15, k=15)
    adata_C1C5 = si.tl.infer_edges(adata1, adata5, n_components=15, k=15)
    adata_C1C6 = si.tl.infer_edges(adata1, adata6, n_components=15, k=15)
    adata_C1C7 = si.tl.infer_edges(adata1, adata7, n_components=15, k=15)
    adata_C1C8 = si.tl.infer_edges(adata1, adata8, n_components=15, k=15)
    adata_C1C9 = si.tl.infer_edges(adata1, adata9, n_components=15, k=15)
    adata_C1C10 = si.tl.infer_edges(adata1, adata10, n_components=15, k=15)

    si.tl.gen_graph(list_CG=[adata1,adata2,adata3,adata4,adata5,adata6,adata7,adata8,adata9,adata10],
                    list_CC=[adata_C1C2,adata_C1C3,adata_C1C4,adata_C1C5,adata_C1C6,adata_C1C7,adata_C1C8,adata_C1C9,adata_C1C10],
                    copy=False,
                    dirname='graph0')
    si.settings.pbg_params
    # modify parameters
    dict_config = si.settings.pbg_params.copy()
    # dict_config['wd'] = 0.00477
    dict_config['workers'] = 12

    ## start training
    si.tl.pbg_train(pbg_params=dict_config, auto_wd=True, save_wd=True, output='model')

    # load in graph ('graph0') info
    si.load_graph_stats()
    # load in model info for ('graph0')
    si.load_pbg_config()

    # load in graph ('graph0') info
    si.load_graph_stats(path='./result_'+dataset+'/pbg/graph0/')
    # load in model info for ('graph0')
    si.load_pbg_config(path='./result_'+dataset+'/pbg/graph0/model/')

    dict_adata = si.read_embedding()
    adata_C = dict_adata['C']  # embeddings for cells from data baron
    adata_C2 = dict_adata['C2']  # embeddings for cells from data segerstolpe
    adata_C3 = dict_adata['C3']  # embeddings for cells from data muraro
    adata_C4 = dict_adata['C4']  # embeddings for cells from data wang
    adata_C5 = dict_adata['C5']  # embeddings for cells from data xin
    adata_C6 = dict_adata['C6']  # embeddings for cells from data xin
    adata_C7 = dict_adata['C7']  # embeddings for cells from data xin
    adata_C8 = dict_adata['C8']  # embeddings for cells from data xin
    adata_C9 = dict_adata['C9']  # embeddings for cells from data xin
    adata_C10 = dict_adata['C10']  # embeddings for cells from data xin
    adata_G = dict_adata['G']  # embeddings for genes

    adata_C.obs['celltype'] = adata1[adata_C.obs_names, :].obs['celltype'].copy().values
    adata_C2.obs['celltype'] = adata2[adata_C2.obs_names, :].obs['celltype'].copy().values
    adata_C3.obs['celltype'] = adata3[adata_C3.obs_names, :].obs['celltype'].copy().values
    adata_C4.obs['celltype'] = adata4[adata_C4.obs_names, :].obs['celltype'].copy().values
    adata_C5.obs['celltype'] = adata5[adata_C5.obs_names, :].obs['celltype'].copy().values
    adata_C6.obs['celltype'] = adata6[adata_C6.obs_names, :].obs['celltype'].copy().values
    adata_C7.obs['celltype'] = adata7[adata_C7.obs_names, :].obs['celltype'].copy().values
    adata_C8.obs['celltype'] = adata8[adata_C8.obs_names, :].obs['celltype'].copy().values
    adata_C9.obs['celltype'] = adata9[adata_C9.obs_names, :].obs['celltype'].copy().values
    adata_C10.obs['celltype'] = adata10[adata_C10.obs_names, :].obs['celltype'].copy().values

    adata_all_CG = si.tl.embed(adata_ref=adata_C,
                            list_adata_query=[adata_C2, adata_C3, adata_C4, adata_C5,adata_C6,adata_C7,adata_C8,adata_C9,adata_C10,adata_G],
                            use_precomputed=False)

    adata_all_CG.obs['entity_anno'] = ""
    adata_all_CG.obs.loc[adata_C.obs_names, 'entity_anno'] = adata_all_CG.obs.loc[
        adata_C.obs_names, 'celltype'].tolist()
    adata_all_CG.obs.loc[adata_C2.obs_names, 'entity_anno'] = adata_all_CG.obs.loc[
        adata_C2.obs_names, 'celltype'].tolist()
    adata_all_CG.obs.loc[adata_C3.obs_names, 'entity_anno'] = adata_all_CG.obs.loc[
        adata_C3.obs_names, 'celltype'].tolist()
    adata_all_CG.obs.loc[adata_C4.obs_names, 'entity_anno'] = adata_all_CG.obs.loc[
        adata_C4.obs_names, 'celltype'].tolist()
    adata_all_CG.obs.loc[adata_C5.obs_names, 'entity_anno'] = adata_all_CG.obs.loc[
        adata_C5.obs_names, 'celltype'].tolist()
    adata_all_CG.obs.loc[adata_C6.obs_names, 'entity_anno'] = adata_all_CG.obs.loc[
        adata_C6.obs_names, 'celltype'].tolist()
    adata_all_CG.obs.loc[adata_C7.obs_names, 'entity_anno'] = adata_all_CG.obs.loc[
        adata_C7.obs_names, 'celltype'].tolist()
    adata_all_CG.obs.loc[adata_C8.obs_names, 'entity_anno'] = adata_all_CG.obs.loc[
        adata_C8.obs_names, 'celltype'].tolist()
    adata_all_CG.obs.loc[adata_C9.obs_names, 'entity_anno'] = adata_all_CG.obs.loc[
        adata_C9.obs_names, 'celltype'].tolist()
    adata_all_CG.obs.loc[adata_C10.obs_names, 'entity_anno'] = adata_all_CG.obs.loc[
        adata_C10.obs_names, 'celltype'].tolist()
    adata_all_CG.obs.loc[adata_G.obs_names, 'entity_anno'] = 'gene'
    del adata_all_CG.obs['celltype']
    #adata_all_CG.obs.rename(columns={'entity_anno': 'celltype'}, inplace=True)
    shutil.rmtree(workdir)
    return adata_all_CG




if __name__ == '__main__':
#set_seed(2023)
    for dataset in ['human heart']:
        print('----------------real data: {} ----------------- '.format(dataset))
        if dataset == 'bct':
            adata = sc.read("../data/bct_raw.h5ad")
            adata.obs.rename(columns={'BATCH': 'batch'}, inplace=True)
            batch_name = np.array(['vis', 'spk', 'wal'], dtype=object)
            adata1 = adata[adata.obs['batch'].values == 'vis']
            adata2 = adata[adata.obs['batch'].values == 'spk']
            adata3 = adata[adata.obs['batch'].values == 'wal']
            #n_clusters = len(np.unique(adata.obs['celltype'].values))
            #print(adata1, "\n", adata2, "\n", "n_clusters:", n_clusters)

        elif dataset == 'lung':
            adata = sc.read_h5ad("../data/multiple_batches/Lung_two_species_raw.h5ad")
            adata1 = adata[adata.obs['BATCH'].values == 'human']
            adata2 = adata[adata.obs['BATCH'].values == 'mouse']

        elif dataset == 'mouse pancreas':
            # csv_data = pd.read_csv('../data/GSM2230761_mouse1_umifm_counts.csv')
            # barcode = csv_data['barcode'].values
            # assigned_cluster = csv_data['assigned_cluster'].values
            # gene_expression = csv_data.iloc[:, 3:].values
            # adata1 = ad.AnnData(X=gene_expression, obs={'barcode': barcode, 'celltype': assigned_cluster})
            # adata1.obs_names = csv_data['Unnamed: 0'].values
            # adata1.var_names = csv_data.columns[3:].values
            # csv_data = pd.read_csv('../data/GSM2230762_mouse2_umifm_counts.csv')
            # barcode = csv_data['barcode'].values
            # assigned_cluster = csv_data['assigned_cluster'].values
            # gene_expression = csv_data.iloc[:, 3:].values
            # adata2 = ad.AnnData(X=gene_expression, obs={'barcode': barcode, 'celltype': assigned_cluster})
            # adata2.obs_names = csv_data['Unnamed: 0'].values
            # adata2.var_names = csv_data.columns[3:].values
            adata = sc.read_h5ad("../scHetG_github/mouse_pancreas.h5ad")
            batch_name = np.array(['batch1', 'batch2'], dtype=object)
            adata.obs['batch'].replace('batch1', 0, inplace=True)
            adata.obs['batch'].replace('batch2', 1, inplace=True)
            adata1 = adata[adata.obs['batch'] == 0]
            adata2 = adata[adata.obs['batch'] == 1]

        elif dataset == 'mouse brain':
            adata = sc.read_h5ad("../data/mousebrain.h5ad")
            adata.obs.rename(columns={'BATCH': 'batch'}, inplace=True)
            adata1 = adata[adata.obs['batch'].values == 'batch1']
            adata2 = adata[adata.obs['batch'].values == 'batch2']

        elif dataset == 'human lung':
            # adata = sc.read_h5ad("../data/human_lung_marker.h5ad")
            # adata.X = csr_matrix(adata.X)
            # adata.obs_names = [f"Cell_{i:d}" for i in range(adata.n_obs)]
            # adata1 = adata[adata.obs['batch'].values == 'muc3843']
            # adata2 = adata[adata.obs['batch'].values == 'muc4658']
            # adata3 = adata[adata.obs['batch'].values == 'muc5103']
            # adata4 = adata[adata.obs['batch'].values == 'muc5104']
            adata = sc.read_h5ad("../scHetG_github/human_lung.h5ad")
            adata.obs['batch'].replace('batch1', 0, inplace=True)
            adata.obs['batch'].replace('batch2', 1, inplace=True)
            adata.obs['batch'].replace('batch3', 2, inplace=True)
            adata.obs['batch'].replace('batch4', 3, inplace=True)
            adata1 = adata[adata.obs['batch'] == 0]
            adata2 = adata[adata.obs['batch'] == 1]
            adata3 = adata[adata.obs['batch'] == 2]
            adata4 = adata[adata.obs['batch'] == 3]


        elif dataset == 'mouse atlas':
            adata1 = ad.read_h5ad('../data/rna_seq_mi.h5ad')
            adata2 = ad.read_h5ad('../data/rna_seq_sm.h5ad')



        elif dataset == 'human pancreas 2':
            # adata1 = ad.read_h5ad('../data/rna_seq_baron.h5ad')
            # adata2 = ad.read_h5ad('../data/rna_seq_segerstolpe.h5ad')
            # adata1.obs.rename(columns={'cell_type1': 'celltype'}, inplace=True)
            # adata2.obs.rename(columns={'cell_type1': 'celltype'}, inplace=True)
            adata = sc.read_h5ad("../scHetG_github/human_pancreas_2.h5ad")
            adata.obs['batch'].replace('batch1', 0, inplace=True)
            adata.obs['batch'].replace('batch2', 1, inplace=True)

            adata1 = adata[adata.obs['batch'] == 0]
            adata2 = adata[adata.obs['batch'] == 1]

        elif dataset == 'human pancreas 4':
            # adata = sc.read("../data/human.h5ad")
            # adata.X = csr_matrix(adata.X)
            # adata1 = adata[adata.obs['batch'].values == 'human1']
            # adata2 = adata[adata.obs['batch'].values == 'human2']
            # adata3 = adata[adata.obs['batch'].values == 'human3']
            # adata4 = adata[adata.obs['batch'].values == 'human4']
            adata = sc.read_h5ad("../scHetG_github/human_pancreas.h5ad")
            adata.obs['batch'].replace('human1', 0, inplace=True)
            adata.obs['batch'].replace('human2', 1, inplace=True)
            adata.obs['batch'].replace('human3', 2, inplace=True)
            adata.obs['batch'].replace('human4', 3, inplace=True)
            adata1 = adata[adata.obs['batch'] == 0]
            adata2 = adata[adata.obs['batch'] == 1]
            adata3 = adata[adata.obs['batch'] == 2]
            adata4 = adata[adata.obs['batch'] == 3]

        elif dataset == 'human heart':
            adata = sc.read("../data/Healthy_human_heart_adata.h5ad")
            adata.obs.rename(columns={'sampleID': 'batch'}, inplace=True)
            unique_sampleIDs = adata.obs['batch'].values.unique()[-10:]
            adata = adata[adata.obs['batch'].isin(unique_sampleIDs)]
            batch_name = unique_sampleIDs
            adata1 = adata[adata.obs['batch'].values == batch_name[0]]
            adata2 = adata[adata.obs['batch'].values == batch_name[1]]
            adata3 = adata[adata.obs['batch'].values == batch_name[2]]
            adata4 = adata[adata.obs['batch'].values == batch_name[3]]
            adata5 = adata[adata.obs['batch'].values == batch_name[4]]
            adata6 = adata[adata.obs['batch'].values == batch_name[5]]
            adata7 = adata[adata.obs['batch'].values == batch_name[6]]
            adata8 = adata[adata.obs['batch'].values == batch_name[7]]
            adata9 = adata[adata.obs['batch'].values == batch_name[8]]
            adata10 = adata[adata.obs['batch'].values == batch_name[9]]

        elif dataset == 'simulation':
            adata = sc.read_h5ad("../data/simulation2.h5ad")
            adata.obs.rename(columns={'Batch': 'batch'}, inplace=True)
            adata.obs.rename(columns={'Group': 'celltype'}, inplace=True)
            batch_name = np.array(['Batch1', 'Batch2', 'Batch3'])
            adata1 = adata[adata.obs['batch'].values == batch_name[0]]
            adata2 = adata[adata.obs['batch'].values == batch_name[1]]
            adata3 = adata[adata.obs['batch'].values == batch_name[2]]


        start = time()

        # adata = my_func2(adata1,adata2,dataset)
        # adata = my_func3(adata1,adata2,adata3, dataset)
        adata = my_func4(adata1,adata2,adata3,adata4, dataset)
        # adata = my_func10(adata1, adata2,adata3, adata4, adata5,adata6,adata7,adata8,adata9,adata10, dataset)


        # adata_c = adata[adata.obs['entity_anno'] != 'gene'] # when you don't need the gene embeddings
        adata_c = adata # when you need the gene embeddings

        umap_embeddings = adata_c.X
        inted = pd.DataFrame(umap_embeddings)
        adata_inted = ad.AnnData(inted, obs=adata_c.obs, dtype='float64')
        adata_inted.obsm['X_latent'] = adata_c.X
        adata_inted.obs['celltype'] = np.array(adata_c.obs['entity_anno'])
        adata_inted.obs['batch'] = np.array(adata_c.obs['id_dataset'])


        mem_used = memory_usage(-1, interval=.1, timeout=1)
        print("elapsed memory:", max(mem_used))
        end = time()
        print('elapsed{:.2f} seconds'.format(end - start))

        sc.pp.neighbors(adata_inted)
        sc.tl.louvain(adata_inted,resolution=0.5)
        #adata_inted.write_h5ad("SIMBA_"+dataset + ".h5ad")

        # If used for gene downstream analysis, it is recommended to change the storage filename for easy identification.
        # adata.write_h5ad("gene&cell_emb_"+dataset+"_SIMBA.h5ad")


