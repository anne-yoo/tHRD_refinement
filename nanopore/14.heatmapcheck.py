#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
#%%
trans = pd.read_csv('/home/jiye/jiye/nanopore/FINALDATA/matched_transcript_TPM.txt', sep='\t', index_col=0)

#%%
df = trans
# normal sample과 tumor sample의 column 인덱스 설정
normal_cols = list(range(0, df.shape[1], 2))  # 0,2,4,6...
tumor_cols = list(range(1, df.shape[1], 2))   # 1,3,5,7...

# 각 row에서 normal sample에서 발현량이 0 초과인 샘플 개수 계산
normal_expr_count = (df.iloc[:,normal_cols] > 0).sum(axis=1)
tumor_expr_count = (df.iloc[:,tumor_cols] > 0).sum(axis=1)

# 15명 이상 발현되는 row만 필터링
filtered_df = df[(normal_expr_count >= 15) | (tumor_expr_count >= 15)]

#%%
filtered_df.to_csv('/home/jiye/jiye/nanopore/FINALDATA/filtered_matched_transcript_TPM.txt', sep='\t', index=True)
#%%
gene291.to_csv('/home/jiye/jiye/nanopore/FINALDATA/289_gene_TPM.txt', sep='\t', index=True)
trans291.to_csv('/home/jiye/jiye/nanopore/FINALDATA/289_transcript_TPM.txt', sep='\t', index=True)


#%%
trans = pd.read_csv('/home/jiye/jiye/nanopore/FINALDATA/137_transcript_TPM.txt', sep='\t', index_col=0)
trans = trans[trans.apply(lambda x: (x != 0).sum(), axis=1) >= trans.shape[1]*0.2]
tumor = trans.iloc[:,1::2]
tumor.index = tumor.index.str.split('-',1).str[0]
filterlist = tumor.index.to_list()
novellist = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/TSSanalysis/sqt_output_classification.txt', sep='\t')
novellist = novellist[(novellist['structural_category']=='novel_in_catalog') |(novellist['structural_category']=='novel_not_in_catalog') ]['isoform'].to_list()

#%%
dut = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/isoformswitchanalyzer/DUTresult.txt', sep='\t')
det = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/DESeq2/deseq2_det.txt', sep='\t')
det['genename'] = det['transcript_id'].str.split('-',1).str[1]
det['isoform_id'] = det['transcript_id'].str.split('-',1).str[0]


#%%
####^^#####################################################
crcgene = ['GNAS','RPS27A','ALDOA','TSPAN1','NME2','IL32']
#translist = list(set(tss[tss['gene_name'].isin(crcgene)]['transcript_id']))
#translist = list(set(tss[tss['within_CAGE_peak']==False]['transcript_id']))
trans = list(dut['isoform_id'])

glist = list(set(list(det.loc[(np.abs(det['log2FoldChange'])>1.5) & (det['padj']<0.05),:]['isoform_id'])))
#glist = list(set(list(det.loc[(det['log2FoldChange']>1.5) & (det['padj']<0.05),:]['isoform_id'])))

#translist = trans
#translist = list(set(trans).intersection(set(glist)))
translist = list(set(glist).intersection(set(novellist)))
#translist = list(set(translist).intersection(set(filterlist)))
#####^^####################################################


expr_matrix = tumor[tumor.index.isin(translist)]
expr_matrix.columns = expr_matrix.columns.str[:-2]
expr_matrix = expr_matrix.dropna()

#%%
#########&&#####################################
# tumor = trans.iloc[:,1::2]
# normal = trans.iloc[:,0::2]
# tumor = tumor[tumor.apply(lambda x: (x != 0).sum(), axis=1) >= tumor.shape[1]*0.2]
# normal = normal[normal.apply(lambda x: (x != 0).sum(), axis=1) >= normal.shape[1]*0.2]

# tumorlist = tumor.index
# normallist = normal.index

# results = set(tumorlist)-set(normallist)
# tsdf = tumor.loc[list(results),:]

# expr_matrix = tsdf
# expr_matrix.columns = expr_matrix.columns.str[:-2]
#########&&#####################################

#%%
from scipy.stats.mstats import winsorize
import numpy as np

# Apply Winsorization row-wise
def winsorize_row(row, lower, upper):
    return winsorize(row, limits=(lower, upper))

lower_limit = 0.05
upper_limit = 0.05

# Apply row-wise Winsorization
df_winsorized = expr_matrix.apply(lambda row: winsorize_row(row, lower_limit, upper_limit), axis=1)
#df_winsorized = expr_matrix

# Convert the result back to a DataFrame
df_winsorized = pd.DataFrame(
    np.array(df_winsorized.tolist()),  # Convert list of arrays back to 2D array
    index=expr_matrix.index,
    columns=expr_matrix.columns
)

# df_winsorized = df_winsorized.loc[df_winsorized.std(axis=1).nlargest(300).index] #########top100 std##############
# df_winsorized.index = df_winsorized.index.str.split('-',1).str[0]

# tumor = trans.iloc[:,1::2]
# tumor.index = tumor.index.str.split('-',1).str[0]
# expr_matrix = tumor[tumor.index.isin(translist)]
# expr_matrix.columns = expr_matrix.columns.str[:-2]
# merged = pd.concat([df_winsorized,expr_matrix], axis=0)

# merged = merged.drop_duplicates()

scaler = StandardScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_winsorized.T).T, index=df_winsorized.index, columns=df_winsorized.columns)
df_normalized = df_normalized.dropna()
# Replace NaN with 0
df_normalized.fillna(0, inplace=True)

# Replace Inf/-Inf with a large finite value (or 0)
df_normalized.replace([np.inf, -np.inf], 0, inplace=True)
#df_normalized = pd.DataFrame(scaler.fit_transform(merged.T).T, index=merged.index, columns=merged.columns)


#%%

metadata = pd.read_csv("/home/jiye/jiye/nanopore/FINALDATA/137_clinicaldata_final.txt", sep='\t', index_col=0)
# Ensure appropriate data types
metadata['sex'] = metadata['sex'].astype('category')
metadata['MSI_status'] = metadata['MSI_status'].astype('category')
metadata['KRAS_mut'] = metadata['KRAS_mut'].astype('category')
#metadata['lymphatic_invasion'] = metadata['lymphatic_invasion'].astype('category')
#metadata['venous_invasion'] = metadata['venous_invasion'].astype('category')
#metadata['perineural_invasion'] = metadata['perineural_invasion'].astype('category')
metadata['stage'] = metadata['stage'].astype('category')
metadata['CMS'] = metadata['CMS'].astype('category')
metadata['Metastasis'] = metadata['Mstage'].apply(lambda x: 'No Metastasis' if x == 'M0' else 'Metastasis')


# Define color mappings for metadata annotations
colormap = {
    'sex': {"Male": "#ABC9FF", "Female": "#FF8A8A"},
    'MSI_status': {"MSS": "#FCE5B5", "MSI-H": "#DB5415", "MSI-L": "#E1880D"},
    'KRAS_mut': {"WT": "#FFFFFF", "MT": "#4F2A00"},
    #'venous_invasion': {0: "#FAF7F0", 1: "#AB886D"},
    #'perineural_invasion': {0: "#FAF7F0", 1: "#AB886D"},
    ##'lymphatic_invasion': {0: "#FAF7F0", 1: "#AB886D"},
    'stage': {1: "#F1F1E8", 2: "#90C96E", 3: "#219304", 4: "#206A5D"},
    'CMS': {'CMS1': "#77C919", 'CMS2': "#4B64EE", 'CMS3': "#A02D63", 'CMS4': "#E36922"},
    'Metastasis': {'Metastasis': "#DD5757", 'No Metastasis': "#A2B3FF"}
}

# Convert metadata categories to colors
col_colors = metadata.apply(
    lambda col: col.map(colormap[col.name]) if col.name in colormap else col,
    axis=0
)
col_colors = col_colors[['CMS','MSI_status','KRAS_mut','Metastasis','stage']]
os_min, os_max = metadata['OS'].min(), metadata['OS'].max()
os_cmap = plt.cm.gray
col_colors['OS'] = metadata['OS'].map(lambda x: os_cmap((x - os_min) / (os_max - os_min)))

df_normalized.fillna(0, inplace=True)  # Replace NaN with 0
df_normalized.replace([np.inf, -np.inf], 0, inplace=True)

# Create the clustermap
g = sns.clustermap(
    df_normalized ,
    method='ward',  # Clustering method
    cmap='vlag',  # Heatmap color scheme
    row_cluster=True,  # Enable row clustering
    col_cluster=True,  # Enable column clustering
    col_colors=col_colors,
    yticklabels=False,xticklabels=False,
    #z_score=0,
    figsize=(10, 8),
    cbar_pos=(0.02, 0.8, 0.03, 0.18),  # Adjust the colorbar position
    dendrogram_ratio=(0.1, 0.1),  # Adjust dendrogram sizes (row, column)
    colors_ratio=0.03,
    center=0,  # Center the colormap at 0
    vmax=5,    # Set maximum limit
    vmin=-5,
)
ax = g.ax_heatmap
ax.set_ylabel("")

from matplotlib.patches import Patch

# Define the legend creation function
def create_legend(ax, category, color_dict, title, position):
    """
    Create a legend for a specific metadata category.
    """
    legend_patches = [Patch(color=color, label=label) for label, color in color_dict.items()]
    ax.figure.legend(
        handles=legend_patches,
        title=title,
        loc="center",
        bbox_to_anchor=position,  # Position of the legend
        bbox_transform=plt.gcf().transFigure,
        
    )


# Define the categories and positions for the legends
categories = ['CMS', 'MSI_status', 'KRAS_mut', 'Metastasis', 'stage']
positions = [(1.05, 0.85), (1.05, 0.7), (1.05, 0.59), (1.05, 0.49), (1.05, 0.37)]  # Adjust positions as needed

# Loop through each category and create its legend
for category, position in zip(categories, positions):
    create_legend(g.ax_heatmap, category, colormap[category], title=category, position=position)


plt.show()



# %%
####^^ N=4 ######
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.preprocessing import StandardScaler

# Perform hierarchical clustering on columns (samples)
linkage_matrix = linkage(df_normalized.T, method='ward') 

# Define the number of clusters for samples
n_clusters = 5
cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

# Add cluster labels to metadata
metadata['Cluster'] = cluster_labels

# Define colors for clusters
cluster_colors = {1: "#DB3939", 2: "#43DB43", 3: "#3434DB", 4: "#FFFF00", 5: "#C212BF"}
metadata['Cluster'] = metadata['Cluster'].map(cluster_colors)

# Create color annotations for the heatmap
col_colors['Cluster'] = metadata[['Cluster']]



# Create the clustermap
g = sns.clustermap(
    df_normalized,
    row_cluster=True,  # Cluster rows
    col_cluster=True,  # Cluster columns
    col_colors=col_colors,
    cmap="vlag",  # Heatmap color scheme
    method="ward",  # Linkage method
    z_score=0,
    xticklabels=False,
    yticklabels=False,
    figsize=(9, 8),
    cbar_pos=(0.02, 0.8, 0.03, 0.18),  # Adjust the colorbar position
    dendrogram_ratio=(0.1, 0.1),  # Adjust dendrogram sizes (row, column)
    colors_ratio=0.03,
    center=0,  # Center the colormap at 0
    vmax=3,    # Set maximum limit
    vmin=-3,
)

ax = g.ax_heatmap
ax.set_ylabel("")

from matplotlib.patches import Patch

# Define the legend creation function
def create_legend(ax, category, color_dict, title, position):
    """
    Create a legend for a specific metadata category.
    """
    legend_patches = [Patch(color=color, label=label) for label, color in color_dict.items()]
    ax.figure.legend(
        handles=legend_patches,
        title=title,
        loc="center",
        bbox_to_anchor=position,  # Position of the legend
        bbox_transform=plt.gcf().transFigure,
        
    )


# Define the categories and positions for the legends
categories = ['CMS', 'MSI_status', 'KRAS_mut', 'Metastasis', 'stage']
positions = [(1.05, 0.85), (1.05, 0.7), (1.05, 0.59), (1.05, 0.49), (1.05, 0.37)]  # Adjust positions as needed

# Loop through each category and create its legend
for category, position in zip(categories, positions):
    create_legend(g.ax_heatmap, category, colormap[category], title=category, position=position)

#plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/figures/WHOLEdet_dut_intersect_heatmap.pdf',bbox_inches='tight', dpi=300)

plt.show()


# Add row cluster labels from the dendrogram
row_linkage_matrix = g.dendrogram_row.linkage  # Access the row linkage from the heatmap
n_row_clusters = 2  # Define the number of row clusters (adjust as needed)
row_cluster_labels = fcluster(row_linkage_matrix, n_row_clusters, criterion='maxclust')

# Map row cluster labels to colors
row_cluster_color_map = {1: "red", 2: "blue"}  # Define colors for clusters
row_cluster_colors = pd.Series(row_cluster_labels, index=df_normalized.index).map(row_cluster_color_map)

# Save transcripts for each cluster
cluster_transcripts = {}
for cluster, color in row_cluster_color_map.items():
    cluster_transcripts[color] = df_normalized[row_cluster_colors == color].index.tolist()
    print(f"Transcripts in {color} cluster: {cluster_transcripts[color]}")


# %%
###^^ Quick GO #####
transet = cluster_transcripts['red']
glist = list(set(det[det['isoform_id'].isin(transet)]['genename']))

import gseapy as gp
enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                gene_sets=[
                    #'GO_Biological_Process_2023',
                            #'Reactome_2022',
                            'GO_Biological_Process_2018',
                            ], 
                organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                outdir=None, # don't write to disk
)

#%%
file = enr.results.sort_values(by=['Adjusted P-value']) 
file['Adjusted P-value'] = -np.log10(file['Adjusted P-value'])
file = file[file['Adjusted P-value']>3]
#%%










#%%
####^^survival plot#####

import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

# Initialize the Kaplan-Meier fitter
kmf = KaplanMeierFitter()

# Prepare the data
metadata['Event'] = metadata['survival'].apply(lambda x: 1 if x == 'Dead' else 0)  # Replace with actual column indicating event occurrence (e.g., death = 1, alive = 0)

survival = metadata[['Cluster','Event','OS']]
survival = survival.dropna()

from lifelines.statistics import multivariate_logrank_test

# Perform the log-rank test for all clusters
results = multivariate_logrank_test(
    survival['OS'],  # Overall survival times
    survival['Cluster'],  # Cluster labels
    survival['Event']  # Event occurrence
)

# Get the p-value
p_value = results.p_value
print(f"Log-rank test p-value: {p_value:.4f}")

# Plot Kaplan-Meier curves
plt.figure(figsize=(8, 6))

for cluster, color in cluster_colors.items():
    # Filter the survival DataFrame for the current cluster
    cluster_data = survival[survival['Cluster'] == color]

    # Check if the current cluster has valid data
    if cluster_data.empty:
        print(f"Cluster {cluster} has no data.")
        continue

    # Fit the Kaplan-Meier curve
    kmf.fit(durations=cluster_data['OS'], event_observed=cluster_data['Event'], label=f"Cluster {cluster}")

    # Plot the survival curve
    kmf.plot_survival_function(ci_show=False, color=color)
    
    print(f"Cluster {cluster}: {len(cluster_data)} samples")
    
plt.text(
    x=0.7, y=0.1,  # Position the text (adjust as needed)
    s=f"p-value = {p_value:.4e}",
    fontsize=12,
    transform=plt.gca().transAxes  # Use axes coordinates for placement
)

# Add the p-value to the plot
plt.title('OS',fontsize=12)
plt.xlabel("Time (OS)", fontsize=12)
plt.ylabel("Survival Probability", fontsize=12)
plt.legend(title="Clusters", fontsize=10, loc="best")
plt.tight_layout()
#plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/isoformswitchanalyzer/DUT_cluster_OS_survival.pdf',bbox_inches='tight', dpi=300)
plt.show()

# %%
####^^^ boxplot check ###########

check = metadata
check['mean TPM'] = df_winsorized.loc[cluster_transcripts['red'],:].mean()
check['lymphatic_invasion'] = check['lymphatic_invasion'].astype(str)
check['perineural_invasion'] = check['perineural_invasion'].astype(str)
check['venous_invasion'] = check['venous_invasion'].astype(str)

plt.figure(figsize=(5, 7))
ax = sns.boxplot(data=check, x='venous_invasion', y='mean TPM', showfliers=True, palette='Set2',)
#plt.xticks(rotation=45)

# sns.stripplot(x='tss_overlap', y='dPSI', data=sig, color='grey', alpha=0.7, jitter=True, s=4, order=["Overlap", "No Overlap"])
# #plt.axhline(0, color='red', linestyle='--', linewidth=1) 
from statannot import add_stat_annotation
add_stat_annotation(ax, data=check, x='venous_invasion', y='mean TPM',
                    box_pairs=[('0', '1')], 
                    comparisons_correction=None,
                    test='Mann-Whitney',  text_format='star', loc='inside', fontsize=15) # Reference line at dPSI = 0

#plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/isoformswitchanalyzer/DUT_TPM_CMS.pdf',bbox_inches='tight', dpi=300)
plt.show()

# %%
tumor = trans.iloc[:,1::2]
normal = trans.iloc[:,0::2]
tumor = tumor[tumor.apply(lambda x: (x != 0).sum(), axis=1) >= tumor.shape[1]*0.2]
normal = normal[normal.apply(lambda x: (x != 0).sum(), axis=1) >= normal.shape[1]*0.2]

tumorlist = tumor.index
normallist = normal.index

results = set(tumorlist)-set(normallist)
tsdf = tumor.loc[list(results),:]

# %%
check = metadata
check['mean TPM'] = list(tsdf.loc[cluster_transcripts['red'],:].mean())
check['lymphatic_invasion'] = check['lymphatic_invasion'].astype(str)
check['perineural_invasion'] = check['perineural_invasion'].astype(str)
check['venous_invasion'] = check['perineural_invasion'].astype(str)

plt.figure(figsize=(5, 7))
ax = sns.boxplot(data=check, x='CMS', y='mean TPM', showfliers=False, palette='Set2',)
#plt.xticks(rotation=45)
# sns.stripplot(x='tss_overlap', y='dPSI', data=sig, color='grey', alpha=0.7, jitter=True, s=4, order=["Overlap", "No Overlap"])
# #plt.axhline(0, color='red', linestyle='--', linewidth=1) 
from statannot import add_stat_annotation
add_stat_annotation(ax, data=check, x='CMS', y='mean TPM',
                    box_pairs=[('CMS3', 'CMS4'),('CMS1', 'CMS3'),('CMS2', 'CMS4'),], 
                    comparisons_correction=None, # Adjust the position of the horizontal line
                    test='Mann-Whitney',  text_format='star', loc='outside', fontsize=15) # Reference line at dPSI = 0
sns.despine()
#plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/isoformswitchanalyzer/TSI_TPM_CMS.pdf',bbox_inches='tight', dpi=300)
plt.show()
# %%
