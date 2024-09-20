#setwd("D:/0AALRJ/scRNA-seq/VAE/scDisco/experiences/simulate")
rm(list=ls())
library(SeuratDisk)
library(Seurat)
library(stringr)
library(aricode)
library(rhdf5)
library(Matrix)

# ### 2. Running Complete data by seurat -------------------------------------------------
data_dir = '../../datasets/human_lung/'
# # prepare data
Convert(paste0(data_dir, "human_lung.h5ad"), dest="h5seurat",
         assay = "RNA",
         overwrite=F)
#seurat_object <- LoadH5Seurat(paste0(data_dir , "human_lung.h5seurat"))
seurat_object <- LoadH5Seurat(paste0(data_dir , "human_lung.h5seurat"),meta.data = FALSE, misc = FALSE)
meta_data <- read.table(paste0(data_dir, "metadata_human_lung.tsv"),sep="\t",header=T,row.names=1)
seurat_object <- AddMetaData(seurat_object, metadata = meta_data)
seurat_object = UpdateSeuratObject(seurat_object)

simulate.list <- SplitObject(seurat_object, split.by = "batch")
# 
# # normalize and identify variable features for each dataset independently
simulate.list <- lapply(X = simulate.list, FUN = function(x) {
   x <- NormalizeData(x)
   x <- FindVariableFeatures(x)})

# # run Seurat
ptm <- proc.time()
simulate.anchors <- FindIntegrationAnchors(object.list = simulate.list, dims = 1:20)
simulate.combined <- IntegrateData(anchorset = simulate.anchors)
simulate.combined <- ScaleData(simulate.combined, verbose = FALSE)
simulate.combined <- RunPCA(simulate.combined, verbose = FALSE)
xpca = simulate.combined@reductions[["pca"]]@cell.embeddings
time = proc.time()-ptm
print(time)
# 
simulate.combined <- FindNeighbors(simulate.combined)
simulate.combined <- FindClusters(simulate.combined, resolution = 0.5)
clust = simulate.combined@meta.data[["seurat_clusters"]]
write.csv(clust, paste0(data_dir, "human_lung_seurat_clust.csv"))
# 
results = data.frame(xpca,
                      simulate.combined@meta.data[["celltype"]],
                      simulate.combined@meta.data[["batch"]])
#                      simulate.combined@meta.data[["condition"]]
#                      )
write.csv(results, paste0(data_dir, "human_lung_seurat.csv"))

print(memory.profile())

gc_output <- gc()

total_last_column <- sum(gc_output[, ncol(gc_output)])

print(paste("max used   (Mb):", total_last_column))

