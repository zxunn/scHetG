# scHetG

![overview](/overview.png "overview")

scHetG is a structure-preserved scRNA-seq data integration approach using heterogeneous graph neural network. By establishing a heterogeneous graph that represents the interactions between multiple batches of cells and genes, and combining a heterogeneous graph neural network with contrastive learning, scHetG concurrently obtained cell and gene embeddings with structural information. 

The sources of all preprocessed data used in this work are available at [https://drive.google.com/drive/folders/1OCN6UmUsM98CpsecpbmQZsXmS0HKcB4k?usp=drive_link.](https://drive.google.com/drive/folders/1Ar8-n-HOEOlK-7A82oEHBItEr0-JhIbJ?usp=sharing)

## demo

scHetG is implemented in Python 3.9, the code execution adheres to the subsequent steps:

1.```python scHetG.py```

Prior to execution, it is necessary to set either ```common_genes = True``` or ```common_genes = False``` to determine whether to integrate each batch of datasets within common genes or distinct genes. 

The output adata contains the cell embeddings in ```adata.obsm['feat']``` and the gene embeddings in ```adata.varm['feat']```. The embeddings can be used as input of other downstream analyses.

2.```python scHetG/co_expression/gene_co_expression_human_lung.py```

This step obtains the gene embedding image and calculates the ratio between inter-cell-type similarity and intra-cell-type similarity.
