import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pca import pca
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from utlis import softmax, reduce_dimensions, cal_ratio

def add_geneset_human_pancreas(dataset, gene_embeddings,gene_names, method):
    csv_data = pd.read_csv('marker_genes/'+dataset+".csv")
    markergene = csv_data['Cell marker'].values
    common_genes = np.intersect1d(markergene, gene_names)
    common_genes_indices = np.concatenate([np.where(markergene == gene)[0] for gene in common_genes])
    markergene = markergene[common_genes_indices]
    cellname = csv_data['Cell name'].values[common_genes_indices]

    ## Set selected cell type/group
    all_ct = np.unique(cellname)
    selected_ct = np.array(['acinar', 'beta', 'ductal', 'macrophage','endothelial'], dtype=object)
    gs_dict_new = {k: markergene[cellname == k] for k in all_ct}
    gs_dict_comb_subset = {k: markergene[cellname == k] for k in selected_ct}

    ## Get mutually exclusive sets
    gs_dict_excl = {}
    for gs_name, gs_genes in gs_dict_comb_subset.items():
        for gs_name2, gs_genes2 in gs_dict_new.items():
            if gs_name != gs_name2:
                gs_genes = np.setdiff1d(gs_genes, gs_genes2)
        gs_dict_excl[gs_name] = gs_genes

    X_plot, dim_labels = reduce_dimensions(gene_embeddings, method='PCA', reduced_dimension=2)
    df_plot = pd.concat([pd.DataFrame(X_plot), pd.DataFrame(gene_names)], axis=1)
    df_plot.columns = dim_labels + ["Label"]
    df_plot.index = gene_names
    df_plot['Geneset'] = 'None'
    gs_dict_in = gs_dict_excl
    for gs_, genes in gs_dict_in.items():
        df_plot['Geneset'].loc[np.isin(df_plot.index, genes)] = gs_

    ratio,L, celltype = cal_ratio(gene_embeddings, df_plot)
    print(method, " inter/intra ratio: ", ratio)

    df_plot = df_plot[df_plot.Geneset != 'None']
    return df_plot, dim_labels, L, celltype


data = sc.read_h5ad("../../gene&cell_emb_human_pancreas_scHetG") # gene embeddings obtained by scHetG
SIMBA_data = sc.read_h5ad("../../gene&cell_emb_human_pancreas_SIMBA") # gene embeddings obtained by SIMBA
data_genes = data.var_names
simba_genes = SIMBA_data.obs_names
common_genes = np.intersect1d(data_genes, simba_genes)
data_genes_indices = np.concatenate([np.where(data_genes == gene)[0] for gene in common_genes])
simba_genes_indices = np.concatenate([np.where(simba_genes == gene)[0] for gene in common_genes])

####
colors = ['#4878D0', 'gold', '#6ACC64', '#D65F5F', 'lightpink', 'teal', 'navy', 'darkorchid', 'orangered', '#FF6600']
fig, ax = plt.subplots(1, 3, constrained_layout=True, figsize=(30,8))
legend_labels = ['acinar', 'beta', 'ductal', 'macrophage','endothelial']
legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8) for color in
                  colors]
model = pca(n_components=64)
X_reduction = model.fit_transform(data.X.T)
gene_embeddings, gene_names = X_reduction['PC'].values, data.var_names
gene_embeddings = gene_embeddings[data_genes_indices]
gene_names = gene_names[data_genes_indices]

df_plot, dim_labels,L1, celltype = add_geneset_human_pancreas("human pancreas", gene_embeddings, gene_names, "normalized data")
df_plot.insert(loc=4, column='gene', value=LabelEncoder().fit_transform(df_plot.Geneset.values).astype(np.int_))
ax0 = sns.scatterplot(x=dim_labels[0], y=dim_labels[1], hue='Geneset', data=df_plot, legend=False,palette=colors,
                      edgecolor='black', linewidth=0.2, s=50, ax=ax[0])
ax0.set_title("normalized data", fontsize=36)
ax0.xaxis.set_visible(False)
ax0.yaxis.set_visible(False)


gene_embeddings = SIMBA_data.X
gene_names = SIMBA_data.obs_names
gene_embeddings = gene_embeddings[simba_genes_indices]
gene_names = gene_names[simba_genes_indices]
df_plot, dim_labels,L2, celltype = add_geneset_human_pancreas("human pancreas", gene_embeddings, gene_names, "SIMBA")
df_plot.insert(loc=4, column='gene', value=LabelEncoder().fit_transform(df_plot.Geneset.values).astype(np.int_))
ax1 = sns.scatterplot(x=dim_labels[0], y=dim_labels[1], hue='Geneset', data=df_plot, legend=False, palette=colors,
                      edgecolor='black', linewidth=0.2, s=50, ax=ax[1])
ax1.set_title("SIMBA", fontsize=36)
ax1.xaxis.set_visible(False)
ax1.yaxis.set_visible(False)

gene_embeddings = data.varm['feat']
gene_embeddings = softmax(data,adata_ref=data.obsm['feat'], adata_query=gene_embeddings)
gene_names = data.var_names
gene_embeddings = gene_embeddings[data_genes_indices]
gene_names = gene_names[data_genes_indices]
df_plot, dim_labels,L3, celltype = add_geneset_human_pancreas("human pancreas", gene_embeddings, gene_names, "scHetG")
df_plot.insert(loc=4, column='gene', value=LabelEncoder().fit_transform(df_plot.Geneset.values).astype(np.int_))
ax2 = sns.scatterplot(x=dim_labels[0], y=dim_labels[1], hue='Geneset', data=df_plot, legend=False, palette=colors,
                      edgecolor='black', linewidth=0.2, s=50, ax=ax[2])
ax2.set_title("scHetG", fontsize=36)
ax2.xaxis.set_visible(False)
ax2.yaxis.set_visible(False)

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['legend.fontsize'] = 28
fig.legend(legend_handles, legend_labels, loc='right',frameon=False,markerscale=2.)
plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.savefig('figure_co/human_pancreas.png', format='png', dpi=900)
plt.show()


########heat graph

df1 = pd.DataFrame(L1, index=np.unique(celltype), columns=np.unique(celltype))
df2 = pd.DataFrame(L2, index=np.unique(celltype), columns=np.unique(celltype))
df3 = pd.DataFrame(L3, index=np.unique(celltype), columns=np.unique(celltype))


fig, axes = plt.subplots(1, 3, figsize=(24, 8))


sns.heatmap(df1, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[0])
axes[0].set_title('normalized data')
axes[0].tick_params(axis='x', rotation=0)

sns.heatmap(df2, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[1])
axes[1].set_title('SIMBA')
axes[1].tick_params(axis='x', rotation=0)

sns.heatmap(df3, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[2])
axes[2].set_title('scHetG')
axes[2].tick_params(axis='x', rotation=0)


plt.tight_layout()
plt.savefig('heat_graph/human_pancreas.png', format='png', dpi=900)

plt.show()