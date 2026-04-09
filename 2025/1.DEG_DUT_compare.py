#%%
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.multitest as ssm
import scipy as sp
import pickle
import sys
import re
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.stats.multitest import multipletests
from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42  
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'Helvetica'

plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 13,    # x축 틱 라벨 글꼴 크기

'ytick.labelsize': 13,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 13,
'legend.title_fontsize': 13, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})
sns.set_style("ticks")

# %%
#######^^^^ DUT gene vs. gene ############

AR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/DEGDUT/AR_stable_DUT_Wilcoxon.txt', sep='\t', index_col=0)
IR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/DEGDUT/IR_stable_DUT_Wilcoxon.txt', sep='\t', index_col=0)
AR_dut_var = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/DEGDUT/AR_variable_DUT_Wilcoxon.txt', sep='\t', index_col=0)
IR_dut_var = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/DEGDUT/IR_variable_DUT_Wilcoxon.txt', sep='\t', index_col=0)
AR_deg = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/DEGDUT/AR_Wilcoxon_DEGresult_FC.txt', sep='\t', index_col=0)
IR_deg = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/DEGDUT/IR_Wilcoxon_DEGresult_FC.txt', sep='\t', index_col=0)

#%%
AR_dut_1 = set(AR_dut.loc[(AR_dut['p_value']<0.05) & (np.abs(AR_dut['log2FC'])>1.5)]['Gene Symbol'])
AR_dut_2 = set(AR_dut_var.loc[(AR_dut_var['p_value']<0.05) & (np.abs(AR_dut_var['log2FC'])>1.5)]['Gene Symbol'])
IR_dut_1 = set(IR_dut.loc[(IR_dut['p_value']<0.05) & (np.abs(IR_dut['log2FC'])>1.5)]['Gene Symbol'])
IR_dut_2= set(IR_dut_var.loc[(IR_dut_var['p_value']<0.05) & (np.abs(IR_dut_var['log2FC'])>1.5)]['Gene Symbol'])
AR_dut_list = AR_dut_1.union(AR_dut_2)
IR_dut_list = IR_dut_1.union(IR_dut_2)

AR_deg_list = set(AR_deg.loc[(AR_deg['p_value']<0.05) & (np.abs(AR_deg['log2FC'])>1)]['Gene Symbol'])
IR_deg_list = set(IR_deg.loc[(IR_deg['p_value']<0.05) & (np.abs(IR_deg['log2FC'])>1)]['Gene Symbol'])

#%%

# %%
from venn import venn
sets = {'AR gDUT': AR_dut_list,
        'IR gDUT': IR_dut_list,
        'AR DEG': AR_deg_list,
        'IR DEG': IR_deg_list}

# sets = {'AR gDUT': AR_dut_list,
#         'AR DEG': AR_deg_list,
#         }

# sets = {'IR gDUT': IR_dut_list,
#         'IR DEG': IR_deg_list,
#         }

sets = {'AR variable gDUT': AR_dut_2,
        'IR variable gDUT': IR_dut_2,
        }

# sets = {'AR DEG': AR_deg_list,
#         'IR DEG': IR_deg_list,
#         }
fig, ax = plt.subplots(figsize=(5, 5))  # Adjust size here
venn(sets, ax=ax)
plt.show()

# %% 

#####^^ UMAP + HBDSCAN for DUT filtering + clustering ##########
import umap.umap_ as umap
#import hdbscan ####USE py39 environment@!! 

ARdutlist = AR_dut.loc[(AR_dut['p_value']<0.05) & (np.abs(AR_dut['log2FC'])>1.5)].index.to_list()
IRdutlist = IR_dut.loc[(IR_dut['p_value']<0.05) & (np.abs(IR_dut['log2FC'])>1.5)].index.to_list()

TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0)
sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
ARlist = list(set(sampleinfo.loc[(sampleinfo['response']==1),'sample_id']))
IRlist = list(set(sampleinfo.loc[(sampleinfo['response']==0),'sample_id']))
TU = TU.iloc[:,1::2] ## pre TU
TU.columns = TU.columns.str[:-4]
ARpre = TU.loc[ARdutlist,ARlist]
IRpre = TU.loc[IRdutlist,IRlist]


# %%
def umap_hdbscan_transcript_clustering(data, dataset_name):
    # UMAP dimensionality reduction for transcripts
    reducer = umap.UMAP(n_neighbors=50, min_dist=0.1, n_components=2, random_state=42)
    umap_embedding = reducer.fit_transform(data)  # No need to transpose, rows = transcripts

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=20, metric='euclidean')
    clusters = clusterer.fit_predict(umap_embedding)

    # Visualization of clusters
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        x=umap_embedding[:, 0],
        y=umap_embedding[:, 1],
        c=clusters,  # Use cluster labels for coloring
        cmap='viridis',
        s=50,
        alpha=0.8
    )
    plt.colorbar(scatter, label="Cluster Label")
    plt.title(f"UMAP + HDBSCAN Clustering for Transcripts ({dataset_name})")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.show()

    return umap_embedding, clusters

print("Processing AR transcript dataset...")
ar_umap, ar_clusters = umap_hdbscan_transcript_clustering(ARpre, "AR")

print("Processing IR transcript dataset...")
ir_umap, ir_clusters = umap_hdbscan_transcript_clustering(IRpre, "IR")

AR_cluster_counts = pd.Series(ar_clusters).value_counts()
IR_cluster_counts = pd.Series(ir_clusters).value_counts()

print(AR_cluster_counts)
print(IR_cluster_counts)

#%%
# Step 1: Add clustering results to the ARpre DataFrame
print(f"Number of transcripts in ARpre: {len(ARpre)}")
print(f"Number of clusters in ar_clusters: {len(ar_clusters)}")

# Add cluster labels
ARpre['Cluster'] = ar_clusters

# Step 2: Extract gene names from transcript identifiers
ARpre['Gene'] = ARpre.index.str.split("-",n=1).str[1]

# Step 3: Group by cluster and get unique gene names
cluster_genes = (
    ARpre.groupby('Cluster')['Gene']
    .apply(lambda x: list(set(x)))  # Get unique gene names
    .to_dict()
)

# Print the number of genes in each cluster
for cluster, genes in cluster_genes.items():
    print(f"Cluster {cluster}: {len(genes)} genes")

# Step 4: Perform GO enrichment analysis
#from gprofiler import GProfiler

# gp = GProfiler(return_dataframe=True)
# enrichment_results = {}

# for cluster, genes in cluster_genes.items():
#     # Run GO enrichment for the gene list of each cluster
#     results = gp.profile(organism='hsapiens', query=genes)
#     results = results[results['source']=='GO:BP']
#     enrichment_results[cluster] = results
#     results = results.reset_index()
#     repair = results[results["name"].str.contains("repair", case=False)]

#     # Print the top enriched GO terms for the cluster
#     print("repairterms: ", repair)
#     print(f"\nCluster {cluster} - Top Enriched Terms:")
#     print(results[['name', 'p_value', 'term_size']].head(10))  # Show top 5 terms

#check = enrichment_results[1].sort_values('p_value')
#%%


# %%
###^^^^^ AR DUT clustering ########
TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0)
TU.columns = TU.columns.str[:-4]
preTU = TU.iloc[:,1::2]
postTU = TU.iloc[:,0::2]
deltaTU = postTU - preTU
AR_delta = deltaTU.loc[ARdutlist,ARlist]

majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = majorminor[majorminor['type']=='major']['gene_ENST'].to_list()

AR_delta = AR_delta.loc[AR_delta.index.isin(majorlist),:]

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import hdbscan
import gseapy as gp

# Step 1: UMAP + HDBSCAN Clustering
def cluster_and_visualize(data, dataset_name):
    # Remove non-numeric columns (e.g., 'Gene' and 'Cluster' if present)
    numeric_data = data.select_dtypes(include=[np.number])

# UMAP dimensionality reduction
    reducer = umap.UMAP(n_neighbors=50, min_dist=0.1, n_components=2, random_state=42)
    umap_embedding = reducer.fit_transform(numeric_data)

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric='euclidean')
    clusters = clusterer.fit_predict(umap_embedding)

    # Add clustering and UMAP coordinates back to the data
    data['Cluster'] = clusters
    data['UMAP1'], data['UMAP2'] = umap_embedding[:, 0], umap_embedding[:, 1]

    # Visualization
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='UMAP1', y='UMAP2', hue='Cluster', palette='viridis', data=data, s=50, alpha=0.8
    )
    plt.title(f"UMAP + HDBSCAN Clustering for {dataset_name} Transcripts")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(title="Cluster", loc="best", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.show()

    return data, clusters

# Step 3: Perform clustering on AR DUT delta TU data
# Assume 'ar_delta_TU' has rows as transcripts and columns as delta values for each sample
print("Clustering AR delta TU data...")
AR_delta['Gene'] = AR_delta.index.str.split("-", n=1).str[1]  # Extract gene names from transcript IDs
ar_clustered, ar_clusters = cluster_and_visualize(AR_delta, "AR")

# Function to run GO enrichment analysis
def run_go_enrichment(cluster_genes, cluster_name):
    # Use 'GO_Biological_Process_2021' for enrichment analysis
    results = gp.enrichr(
        gene_list=cluster_genes,
        gene_sets='GO_Biological_Process_2018',
        organism='Human',
        outdir=None
    )
    # Filter results by adjusted p-value and sort by adjusted p-value
    filtered_results = results.results[results.results['Adjusted P-value'] < 0.01]
    filtered_results = filtered_results.sort_values(by='Adjusted P-value')
    
    # Display top enriched terms
    print(f"\nCluster {cluster_name} - Top Enriched Terms (p < 0.01):")
    top_results = filtered_results[['Term', 'Adjusted P-value', 'Genes']].head(15)
    print(top_results)
    return filtered_results

# Step 1: Group transcripts by cluster and extract unique gene names
# Assuming 'ar_clustered' contains 'Cluster' and 'Gene' columns
cluster_genes = (
    ar_clustered.groupby('Cluster')['Gene']
    .apply(lambda x: list(set(x)))  # Get unique gene names for each cluster
    .to_dict()
)

# Step 2: Run enrichment for each cluster
enrichment_results = {}
for cluster, genes in cluster_genes.items():
    if len(genes) > 0:  # Proceed if the cluster has genes
        result = run_go_enrichment(genes, cluster)
        enrichment_results[cluster] = result

# Step 3: Identify clusters enriched for target terms
target_terms = ['DNA repair', 'cell cycle']  # Specify your target GO terms
highlight_clusters = [
    cluster for cluster, results in enrichment_results.items()
    if any(term in results['Term'].values for term in target_terms)
]

print("Clusters enriched for target terms:", highlight_clusters)

# Step 4: Visualize clusters and highlight those with target terms
plt.figure(figsize=(10, 8))
highlight_mask = ar_clustered['Cluster'].isin(highlight_clusters)

# Create a scatterplot to visualize highlighted clusters
sns.scatterplot(
    x='UMAP1', y='UMAP2', hue=highlight_mask, data=ar_clustered,
    palette={True: 'red', False: 'gray'}, s=50, alpha=0.8
)
plt.title("Highlighted Clusters with Target GO Terms")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(title='Target Cluster', loc='best')
plt.show()


# %%
###^^^^^ AR DUT filtering (1) variance ##########

ARdut_preTU = preTU.loc[ARdutlist,:]
transcript_variances = ARdut_preTU.var(axis=1)
top_percent_threshold = transcript_variances.quantile(0.50)
filtered_data = ARdut_preTU[transcript_variances >= top_percent_threshold]
filtered_data['Gene'] = filtered_data.index.str.split("-", n=1).str[1]
#filtered_data = filtered_data.loc[filtered_data.index.isin(majorlist),:]
glist = list(set(filtered_data['Gene']))
results = gp.enrichr(
        gene_list=glist,
        gene_sets='GO_Biological_Process_2021',
        organism='Human',
        outdir=None
    )
# Filter results by adjusted p-value and sort by adjusted p-value
filtered_results = results.results[results.results['Adjusted P-value'] < 0.01]
filtered_results = filtered_results.sort_values(by='Adjusted P-value')

# %%
###^^^^^ AR DUT filtering (2) fold change ##########
ARdutdf = AR_dut.loc[(AR_dut['p_value']<0.05) & (np.abs(AR_dut['log2FC'])>1.5)]
ARdutdf = ARdutdf.loc[ARdutdf.index.isin(majorlist),:]
fc = ARdutdf['log2FC']
fc = fc[~np.isinf(fc)]

IRdutdf = IR_dut.loc[(IR_dut['p_value']<0.05) & (np.abs(IR_dut['log2FC'])>1.5)]
IRdutdf = IRdutdf.loc[~IRdutdf.index.isin(majorlist),:]
fc = IRdutdf['log2FC']
fc = fc[~np.isinf(fc)]
# %%
ARpremajor = ARpre.loc[ARpre.index.isin(majorlist),:]
IRpremajor = IRpre.loc[IRpre.index.isin(majorlist),:]

ARpreminor = ARpre.loc[~ARpre.index.isin(majorlist),:]
IRpreminor= IRpre.loc[~IRpre.index.isin(majorlist),:]

# %%
input = pd.DataFrame()
ARdutdf = AR_dut.loc[(AR_dut['p_value']<0.05) & (np.abs(AR_dut['log2FC'])>1.5)]
input['preTU'] = ARpre.mean(axis=1)
input['log2FC'] = ARdutdf['log2FC']
input.index = ARdutdf.index
input = pd.merge(input,majorminor,left_index=True, right_on='gene_ENST')
#input = input[input['type']=='major']
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=input,
    x='preTU',
    y='log2FC',
    hue='type',
    palette={'major': 'blue', 'minor': 'orange'},
    s=100, alpha=0.7
)
plt.xscale('log')
plt.title('AR')
# %%
input = pd.DataFrame()
IRdutdf = IR_dut.loc[(IR_dut['p_value']<0.05) & (np.abs(IR_dut['log2FC'])>1.5)]
input['preTU'] = IRpre.mean(axis=1)
input['log2FC'] = IRdutdf['log2FC']
input.index = IRdutdf.index
input = pd.merge(input,majorminor,left_index=True, right_on='gene_ENST')
input = input[input['type']=='major']

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=input,
    x='preTU',
    y='log2FC',
    hue='type',
    palette={'major': 'blue', 'minor': 'orange'},
    s=100, alpha=0.7
)
plt.xscale('log')
plt.title('IR')

# %%
preTU = TU.iloc[:,1::2] ## pre TU
ARpre = preTU.loc[~preTU.index.isin(majorlist),ARlist]
IRpre = preTU.loc[~preTU.index.isin(majorlist),IRlist]

# %%
##^^ CR patients check ##########
val = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_TU.txt', sep='\t', index_col=0)
val = val.apply(pd.to_numeric, errors='coerce')
valinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_validation_clinicalinfo.txt', sep='\t', index_col=0)
valinfo = valinfo.loc[list(val.columns),:]
valinfo['finalresponse'] = 'x'
valinfo.loc[(valinfo['ongoing']==1) | (valinfo['ongoing']==2) | (valinfo['ongoing']==4),'finalresponse'] = 'CR'
valinfo.loc[(valinfo['ongoing']==0) & (valinfo['response']==1), 'finalresponse'] = 'AR'
valinfo.loc[(valinfo['ongoing']==3) & (valinfo['response']==1), 'finalresponse'] = 'AR'
valinfo.loc[(valinfo['response']==0), 'finalresponse'] = 'IR'



#%%
typelist = ['CR','AR','IR']
fig = pd.DataFrame()
for t in typelist:
    CRlist = valinfo[valinfo['finalresponse']==t].index.to_list()
    CRpre = val.loc[IRdutlist,CRlist]
    CRpremajor = CRpre.loc[CRpre.index.isin(majorlist),:]
    CRpreminor = CRpre.loc[~CRpre.index.isin(majorlist),:]
    print('from dis cohort IR DUTs, ind cohort pre TU of ',t,':',CRpremajor.mean(axis=1).median())
    fig[t] = CRpreminor.mean(axis=1)

fig = pd.melt(fig, var_name='sample', value_name='pre TU')

from statannotations.Annotator import Annotator

ax = sns.boxplot(data=fig, x='sample', y='pre TU', showfliers=False)

annotator = Annotator(ax, [('IR', 'AR')], data=fig, x='sample', y='pre TU',
                      order=['CR', 'AR', 'IR'])
annotator.configure(test='Mann-Whitney', text_format='simple', loc='inside', fontsize=12)
annotator.apply_and_annotate()

# %%
####^^ AR vs. IR pre/post major/minor 

ARdutlist = AR_dut.loc[(AR_dut['p_value']<0.05) & (np.abs(AR_dut['log2FC'])>1.5)].index.to_list()
IRdutlist = IR_dut.loc[(IR_dut['p_value']<0.05) & (np.abs(IR_dut['log2FC'])>1.5)].index.to_list()

TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost_80_transcript_TU.txt', sep='\t', index_col=0)
sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
ARlist = list(set(sampleinfo.loc[(sampleinfo['response']==1),'sample_id']))
IRlist = list(set(sampleinfo.loc[(sampleinfo['response']==0),'sample_id']))
preTU = TU.iloc[:,1::2] ## pre TU
postTU = TU.iloc[:,0::2]
preTU.columns = preTU.columns.str[:-4]
postTU.columns = postTU.columns.str[:-4]

majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/majorlist.txt', sep='\t')
majorlist = majorminor['Transcript-Gene'].to_list()
minorlist = TU.index[~TU.index.isin(majorlist)].to_list()
#%%
from statannotations.Annotator import Annotator

###^^ major ######
ARpre = preTU.loc[ARdutlist,ARlist]
IRpre = preTU.loc[IRdutlist,IRlist]
ARpost = postTU.loc[ARdutlist,ARlist]
IRpost = postTU.loc[IRdutlist,IRlist]

ARpre = ARpre.loc[ARpre.index.isin(majorlist),:]
IRpre = IRpre.loc[IRpre.index.isin(majorlist),:]
ARpost = ARpost.loc[ARpost.index.isin(majorlist),:]
IRpost = IRpost.loc[IRpost.index.isin(majorlist),:]

ARpre_mean = ARpre.mean(axis=1)
IRpre_mean = IRpre.mean(axis=1)
ARpost_mean = ARpost.mean(axis=1)
IRpost_mean = IRpost.mean(axis=1)

df_long = pd.DataFrame({
    "Value": pd.concat([ARpre_mean, IRpre_mean, ARpost_mean, IRpost_mean], ignore_index=True),
    "Time": (["pre"] * len(ARpre_mean)) + (["pre"] * len(IRpre_mean)) + 
            (["post"] * len(ARpost_mean)) + (["post"] * len(IRpost_mean)),
    "Group": (["AR"] * len(ARpre_mean)) + (["IR"] * len(IRpre_mean)) + 
             (["AR"] * len(ARpost_mean)) + (["IR"] * len(IRpost_mean))
})

# Boxplot 그리기
plt.figure(figsize=(6, 5))
ax = sns.boxplot(x="Time", y="Value", hue="Group", data=df_long, palette={"AR": "#FFCC29", "IR": "#81B214"}, showfliers=False)
ax.set_xlabel('')
ax.set_ylabel('TU')
#plt.yticks(np.arange(0,0.41,0.1))
# # Add statistical annotation
pairs = pairs = [(("pre", "AR"), ("pre", "IR")), (("post", "AR"), ("post", "IR")),(("pre", "AR"), ("post", "AR")),(("pre", "IR"), ("post", "IR"))]

annot = Annotator(ax, pairs, data=df_long, x="Time", hue="Group", y="Value")
annot.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=1, comparisons_correction="Bonferroni")
annot.apply_and_annotate()

# 제목 설정
plt.title("major DUT")
#plt.title("major transcripts")
plt.legend(title="Group")  
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
sns.despine()

# plt.legend().remove()
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/figures/majorDUT_boxplot.pdf', dpi=300, bbox_inches='tight') 
plt.show()

# %%
###^^ nonmajor #####

ARpre = preTU.loc[ARdutlist,ARlist]
IRpre = preTU.loc[IRdutlist,IRlist]
ARpost = postTU.loc[ARdutlist,ARlist]
IRpost = postTU.loc[IRdutlist,IRlist]

# ARpre = preTU.loc[:,ARlist]
# IRpre = preTU.loc[:,IRlist]
# ARpost = postTU.loc[:,ARlist]
# IRpost = postTU.loc[:,IRlist]

ARpre = ARpre.loc[~ARpre.index.isin(majorlist),:]
IRpre = IRpre.loc[~IRpre.index.isin(majorlist),:]
ARpost = ARpost.loc[~ARpost.index.isin(majorlist),:]
IRpost = IRpost.loc[~IRpost.index.isin(majorlist),:]

ARpre_mean = ARpre.mean(axis=1)
IRpre_mean = IRpre.mean(axis=1)
ARpost_mean = ARpost.mean(axis=1)
IRpost_mean = IRpost.mean(axis=1)

df_long = pd.DataFrame({
    "Value": pd.concat([ARpre_mean, IRpre_mean, ARpost_mean, IRpost_mean], ignore_index=True),
    "Time": (["pre"] * len(ARpre_mean)) + (["pre"] * len(IRpre_mean)) + 
            (["post"] * len(ARpost_mean)) + (["post"] * len(IRpost_mean)),
    "Group": (["AR"] * len(ARpre_mean)) + (["IR"] * len(IRpre_mean)) + 
             (["AR"] * len(ARpost_mean)) + (["IR"] * len(IRpost_mean))
})

# Boxplot 그리기
plt.figure(figsize=(4, 5))
ax = sns.violinplot(x="Time", y="Value", hue="Group", data=df_long, palette={"AR": "#FFCC29", "IR": "#81B214"}, split=True, fill=False, inner='quarts', gap=0.1)
ax.set_xlabel('')
ax.set_ylabel('TU')
plt.ylim([-0.02,0.3])
#plt.yticks(np.arange(0,0.2,0.05))


pairs = pairs = [(("pre", "AR"), ("pre", "IR")), (("post", "AR"), ("post", "IR")),(("pre", "AR"), ("post", "AR")),(("pre", "IR"), ("post", "IR"))]

# annot = Annotator(ax, pairs, data=df_long, x="Time", hue="Group", y="Value")
# annot.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=1, comparisons_correction="Bonferroni")
# annot.apply_and_annotate()


# 제목 설정
plt.title("non-major DUT")
#plt.title("non-major transcripts")
plt.legend(title="Group")  
sns.despine()
sns.move_legend(
    ax, "lower center",
    bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
)
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/nonmajorDUT_violonplot.pdf', dpi=300, bbox_inches='tight') 
plt.show()

# %%
###^^^ major ridgeplot?########
ARpre = preTU.loc[ARdutlist,ARlist]
IRpre = preTU.loc[IRdutlist,IRlist]
ARpost = postTU.loc[ARdutlist,ARlist]
IRpost = postTU.loc[IRdutlist,IRlist]

ARpre_DUT = ARpre.loc[ARpre.index.isin(majorlist),:]
IRpre_DUT = IRpre.loc[IRpre.index.isin(majorlist),:]
ARpost_DUT = ARpost.loc[ARpost.index.isin(majorlist),:]
IRpost_DUT = IRpost.loc[IRpost.index.isin(majorlist),:]

ARpre_DUT_mean = ARpre_DUT.mean(axis=1)
IRpre_DUT_mean = IRpre_DUT.mean(axis=1)
ARpost_DUT_mean = ARpost_DUT.mean(axis=1)
IRpost_DUT_mean = IRpost_DUT.mean(axis=1)

ARpre = preTU.loc[:,ARlist]
IRpre = preTU.loc[:,IRlist]
ARpost = postTU.loc[:,ARlist]
IRpost = postTU.loc[:,IRlist]

ARpre_whole = ARpre.loc[ARpre.index.isin(majorlist),:]
IRpre_whole = IRpre.loc[IRpre.index.isin(majorlist),:]
ARpost_whole = ARpost.loc[ARpost.index.isin(majorlist),:]
IRpost_whole = IRpost.loc[IRpost.index.isin(majorlist),:]

ARpre_whole_mean = ARpre_whole.mean(axis=1)
IRpre_whole_mean = IRpre_whole.mean(axis=1)
ARpost_whole_mean = ARpost_whole.mean(axis=1)
IRpost_whole_mean = IRpost_whole.mean(axis=1)

# Create DataFrame for plotting
df_AR_whole = pd.DataFrame({
    'TU': pd.concat([ARpre_whole_mean, ARpost_whole_mean]),
    'Condition': ['Pre'] * len(ARpre_whole_mean) + ['Post'] * len(ARpost_whole_mean)
})

df_IR_whole = pd.DataFrame({
    'TU': pd.concat([IRpre_whole_mean, IRpost_whole_mean]),
    'Condition': ['Pre'] * len(IRpre_whole_mean) + ['Post'] * len(IRpost_whole_mean)
})

df_AR_DUT = pd.DataFrame({
    'TU': pd.concat([ARpre_DUT_mean, ARpost_DUT_mean]),
    'Condition': ['Pre'] * len(ARpre_DUT_mean) + ['Post'] * len(ARpost_DUT_mean)
})

df_IR_DUT = pd.DataFrame({
    'TU': pd.concat([IRpre_DUT_mean, IRpost_DUT_mean]),
    'Condition': ['Pre'] * len(IRpre_DUT_mean) + ['Post'] * len(IRpost_DUT_mean)
})


# Set up the figure
fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)

# Define color palettes for pre vs. post
palette_AR = {"Pre": "#FFCC29", "Post": "#FFAE4B"}
palette_IR = {"Pre": "#81B214", "Post": "#517662"}

# Plot for AR (All Transcripts)
sns.kdeplot(data=df_AR_whole, x='TU', hue='Condition', fill=True, palette=palette_AR, ax=axes[0,0])
axes[0,0].set_title("AR (All Transcripts)")

# Plot for IR (All Transcripts)
sns.kdeplot(data=df_IR_whole, x='TU', hue='Condition', fill=True, palette=palette_IR, ax=axes[0,1])
axes[0,1].set_title("IR (All Transcripts)")

# Plot for AR (DUT Only)
sns.kdeplot(data=df_AR_DUT, x='TU', hue='Condition', fill=True, palette=palette_AR, ax=axes[1,0])
axes[1,0].set_title("AR (DUT Only)")

# Plot for IR (DUT Only)
sns.kdeplot(data=df_IR_DUT, x='TU', hue='Condition', fill=True, palette=palette_IR, ax=axes[1,1])
axes[1,1].set_title("IR (DUT Only)")

for ax in axes.flatten():
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(0,6)
# Adjust layout
plt.tight_layout()
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/figures/major_dist_ridgeplot.pdf', dpi=300, bbox_inches='tight') 
plt.show()

# %%
###^^^ non-major ridgeplot?########
ARpre = preTU.loc[ARdutlist,ARlist]
IRpre = preTU.loc[IRdutlist,IRlist]
ARpost = postTU.loc[ARdutlist,ARlist]
IRpost = postTU.loc[IRdutlist,IRlist]

ARpre_DUT = ARpre.loc[~ARpre.index.isin(majorlist),:]
IRpre_DUT = IRpre.loc[~IRpre.index.isin(majorlist),:]
ARpost_DUT = ARpost.loc[~ARpost.index.isin(majorlist),:]
IRpost_DUT = IRpost.loc[~IRpost.index.isin(majorlist),:]

ARpre_DUT_mean = ARpre_DUT.mean(axis=1)
IRpre_DUT_mean = IRpre_DUT.mean(axis=1)
ARpost_DUT_mean = ARpost_DUT.mean(axis=1)
IRpost_DUT_mean = IRpost_DUT.mean(axis=1)

ARpre = preTU.loc[:,ARlist]
IRpre = preTU.loc[:,IRlist]
ARpost = postTU.loc[:,ARlist]
IRpost = postTU.loc[:,IRlist]

ARpre_whole = ARpre.loc[~ARpre.index.isin(majorlist),:]
IRpre_whole = IRpre.loc[~IRpre.index.isin(majorlist),:]
ARpost_whole = ARpost.loc[~ARpost.index.isin(majorlist),:]
IRpost_whole = IRpost.loc[~IRpost.index.isin(majorlist),:]

ARpre_whole_mean = ARpre_whole.mean(axis=1)
IRpre_whole_mean = IRpre_whole.mean(axis=1)
ARpost_whole_mean = ARpost_whole.mean(axis=1)
IRpost_whole_mean = IRpost_whole.mean(axis=1)

# Create DataFrame for plotting
df_AR_whole = pd.DataFrame({
    'TU': pd.concat([ARpre_whole_mean, ARpost_whole_mean]),
    'Condition': ['Pre'] * len(ARpre_whole_mean) + ['Post'] * len(ARpost_whole_mean)
})

df_IR_whole = pd.DataFrame({
    'TU': pd.concat([IRpre_whole_mean, IRpost_whole_mean]),
    'Condition': ['Pre'] * len(IRpre_whole_mean) + ['Post'] * len(IRpost_whole_mean)
})

df_AR_DUT = pd.DataFrame({
    'TU': pd.concat([ARpre_DUT_mean, ARpost_DUT_mean]),
    'Condition': ['Pre'] * len(ARpre_DUT_mean) + ['Post'] * len(ARpost_DUT_mean)
})

df_IR_DUT = pd.DataFrame({
    'TU': pd.concat([IRpre_DUT_mean, IRpost_DUT_mean]),
    'Condition': ['Pre'] * len(IRpre_DUT_mean) + ['Post'] * len(IRpost_DUT_mean)
})


# Set up the figure
fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True, sharey=True)

# Define color palettes for pre vs. post
palette_AR = {"Pre": "#FFCC29", "Post": "#FFAE4B"}
palette_IR = {"Pre": "#81B214", "Post": "#517662"}

# Plot for AR (All Transcripts)
sns.kdeplot(data=df_AR_whole, x='TU', hue='Condition', fill=True, palette=palette_AR, ax=axes[0,0])
axes[0,0].set_title("AR (All Transcripts)")

# Plot for IR (All Transcripts)
sns.kdeplot(data=df_IR_whole, x='TU', hue='Condition', fill=True, palette=palette_IR, ax=axes[0,1])
axes[0,1].set_title("IR (All Transcripts)")

# Plot for AR (DUT Only)
sns.kdeplot(data=df_AR_DUT, x='TU', hue='Condition', fill=True, palette=palette_AR, ax=axes[1,0])
axes[1,0].set_title("AR (DUT Only)")

# Plot for IR (DUT Only)
sns.kdeplot(data=df_IR_DUT, x='TU', hue='Condition', fill=True, palette=palette_IR, ax=axes[1,1])
axes[1,1].set_title("IR (DUT Only)")

for ax in axes.flatten():
    ax.set_xlim(-0.05, 0.52)
    ax.set_ylim(0,18)
    
# plt.title('minor transcripts')
# Adjust layout
plt.tight_layout()
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/figures/minor_dist_ridgeplot.pdf', dpi=300, bbox_inches='tight') 
plt.show()

#%%
###^^^^ major transcript rank ######

###^^ major ######
ARpre = preTU.loc[:,ARlist]
IRpre = preTU.loc[:,IRlist]
ARpost = postTU.loc[:,ARlist]
IRpost = postTU.loc[:,IRlist]


def compute_rank(df):
    """
    Compute transcript rank within each gene for each sample.
    - df: DataFrame with transcripts as index and samples as columns
    - Assumes transcript names are in format 'ENSTxxxx-Gene'
    """
    df = df.copy()
    df['Gene'] = df.index.str.split('-', n=1).str[1]  # Extract gene name
    ranked_df = df.groupby("Gene").rank(method="average", ascending=False)  # Rank within gene
    return ranked_df  # Drop gene column to keep only ranks

# Compute ranks for each dataset
ARpre_rank = compute_rank(ARpre)
IRpre_rank = compute_rank(IRpre)
ARpost_rank = compute_rank(ARpost)
IRpost_rank = compute_rank(IRpost)

from scipy.stats import mannwhitneyu

ARmajordutlist = list(set(ARdutlist).intersection(set(majorlist)))
IRmajordutlist = list(set(IRdutlist).intersection(set(majorlist)))

# Filter DUTs only
ARpre_DUT = ARpre_rank.loc[ARmajordutlist,:]
ARpost_DUT = ARpost_rank.loc[ARmajordutlist,:]
IRpre_DUT = IRpre_rank.loc[IRmajordutlist,:]
IRpost_DUT = IRpost_rank.loc[IRmajordutlist,:]

deltaRank_AR = ARpre_DUT - ARpost_DUT #상승정도
deltaRank_IR = IRpre_DUT - IRpost_DUT

melted_df_AR = pd.DataFrame({"Group": "AR", "deltaRank": deltaRank_AR.mean(axis=1)}).reset_index()
melted_df_IR = pd.DataFrame({"Group": "IR", "deltaRank": deltaRank_IR.mean(axis=1)}).reset_index()
melted_df = pd.concat([melted_df_AR, melted_df_IR])

# Swarmplot
from statannotations.Annotator import Annotator

plt.figure(figsize=(4, 5))
plt.axhline(0, linestyle="--", color="grey", alpha=0.6)  # Baseline reference
ax = sns.boxplot(x="Group", y="deltaRank", data=melted_df, palette={"AR": "#FFCC29", "IR": "#81B214"}, showfliers=False)
plt.ylabel("Major DUT Rank Increase")
plt.xlabel("")

pairs = pairs = [("AR","IR")]
annot = Annotator(ax, pairs, data=melted_df, x="Group", y="deltaRank")
annot.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=1, comparisons_correction="Bonferroni")
annot.apply_and_annotate()
sns.despine()
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/figures/majorDUTrankincrease.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%
ARsubset = melted_df_AR[melted_df_AR['deltaRank']>0]
ARsubset['Gene'] = ARsubset['Transcript-Gene'].str.split("-",n=1).str[1]
ARgenelist = list(set(ARsubset['Gene']))

import gseapy as gp

# enr = gp.enrichr(gene_list=ARgenelist, # or "./tests/data/gene_list.txt",
#                 gene_sets=['GO_Biological_Process_2021'], 
#                 organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
#                 outdir=None, # don't write to disk
# )

# enrresult = enr.results.sort_values(by=['Adjusted P-value']) 
# glist = list(ARsubset['Gene'])

#%%
import gseapy as gp

# Pull gene lists for each GO term
go_results = gp.get_library("GO_Biological_Process_2021", organism="Human")
go_hallmark = gp.get_library(name="MSigDB_Hallmark_2020", organism="Human")

geneset1 = ["double-strand break repair via homologous recombination (GO:0000724)"] #,"double-strand break repair via homologous recombination (GO:0000724)"
geneset2 = ["replication fork processing (GO:0031297)", "replication fork protection (GO:0048478)"]
geneset3 = ["positive regulation of Wnt signaling pathway (GO:0030177)"]
geneset3_1 = ['PI3K/AKT/mTOR  Signaling']
geneset4 = [ "cell cycle G2/M phase transition (GO:0044839)","DNA damage checkpoint signaling (GO:0000077)","DNA integrity checkpoint signaling (GO:0031570)"] #"DNA damage checkpoint signaling (GO:0000077)" #"DNA damage checkpoint signaling (GO:0000077)"] #"DNA damage response, signal transduction by p53 class mediator (GO:0030330)",,"DNA integrity checkpoint signaling (GO:0031570)","signal transduction in response to DNA damage (GO:0042770)"

geneset1 = {term: go_results[term] for term in geneset1 if term in go_results}
geneset2 = {term: go_results[term] for term in geneset2 if term in go_results}
geneset3 = {term: go_results[term] for term in geneset3 if term in go_results}
geneset3_1 = {term: go_hallmark[term] for term in geneset3_1 if term in go_hallmark}
geneset4 = {term: go_results[term] for term in geneset4 if term in go_results}


geneset1 = sorted(set(gene for genes in geneset1.values() for gene in genes))
geneset2 = sorted(set(gene for genes in geneset2.values() for gene in genes))
geneset3 = sorted(set(gene for genes in geneset3.values() for gene in genes))
geneset3_1 = sorted(set(gene for genes in geneset3_1.values() for gene in genes))
geneset4 = sorted(set(gene for genes in geneset4.values() for gene in genes))

geneset3 = list(set(geneset3).union(set(geneset3_1)))

# %%
ARsubset = melted_df_AR[melted_df_AR['deltaRank']>0]
ARsubset['Gene'] = ARsubset['Transcript-Gene'].str.split("-",n=1).str[1]
ARgenelist = list(set(ARsubset['Gene']))
glist = list(ARsubset['Gene'])

finalgenelist = []
l = [geneset1,geneset3,geneset4]
for geneset in l:
    finalgenelist = finalgenelist + list(set(geneset).intersection(set(glist)))

tmp = ARsubset[ARsubset['Gene'].isin(finalgenelist)]
tlist = list(tmp['Transcript-Gene'])
# %%
with open('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARup_tlist.txt', 'w') as f:
    for line in tlist:
        f.write(f"{line}\n")
# %%
