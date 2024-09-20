# scHetG

![overview](/overview.png "overview")

scHetG is a structure-preserved scRNA-seq data integration approach using heterogeneous graph neural network. By establishing a heterogeneous graph that represents the interactions between multiple batches of cells and genes, and combining a heterogeneous graph neural network with contrastive learning, scHetG concurrently obtained cell and gene embeddings with structural information. 

The sources of all preprocessed data used in this work are available at [https://drive.google.com/drive/folders/1OCN6UmUsM98CpsecpbmQZsXmS0HKcB4k?usp=drive_link.](https://drive.google.com/drive/folders/1Ar8-n-HOEOlK-7A82oEHBItEr0-JhIbJ?usp=sharing)

## demo
The reproducibility file provides the experimental reproduction process, including ablation, comparison_methods, integration_common_genes(datasets with common genes between batches), integration_distinct_genes(dataset with distinct genes between batches).
scHetG is implemented in Python 3.9. Below we describe how to integrate datasets with common genes:

1.Dataset preparation
The dataset storage format is annadata ```adata```, including adata.obs['batch']. Note that the batch tag needs to be changed to ‘batch1, batch2, batch3, batch4,....’. Similarly for sequential form starting from 1.

2.Preprocessing
  ```
  adata.X = adata.X.astype(np.float64)
  adata = preprocess(adata), the preprocess() function is imported from scHetG.data.data_utils.py
  adata = preprocess(adata)
  sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor='seurat')
  highly_variable = adata.var['highly_variable']
  adata = adata[:, adata.var['highly_variable']]
  ```

3.Train
```
  adata = train_scHetG(adata, batch_name, n_clusters=n_clusters, cl_type='celltype',use_graph=True,
                      feats_dim=64, lr=0.15, n_layers=2,
                      drop_out=0.1, decoder='ZINB', gamma=1,
                      log_interval=10, iteration=200,
                      sample_rate=0.1, learnable_w=True, highly_variable=highly_variable,common_genes=True,
                      recon_ratio=1., cl_ratio=1.,
                      mnn_components=10, mnn_k=10,
                      k_to_m_ratio=0.1, knn_k=5,
                      margin=1.5, resolution_l=0.5, resolution_preclus=0.2)
```

The output adata contains the cell embeddings in ```adata.obsm['feat']``` and the gene embeddings in ```adata.varm['feat']```. The embeddings can be used as input of other downstream analyses.

3.Gene downstream analysis
```
  1.python gene_downstream_analysis/gene_co_expression_human_lung.py           This step obtains the gene embedding image and calculates the ratio between inter-cell-type similarity and intra-cell-type similarity.
  2.python gene_downstream_analysis/find_markers_human_lung.py                 This step use to find novel marker genes.
```

