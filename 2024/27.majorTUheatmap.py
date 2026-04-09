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
import os
import re
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import plotly.express as px
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.stats.multitest import multipletests


sns.set(font = 'Arial')
sns.set_style("whitegrid")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 13,    # x축 틱 라벨 글꼴 크기

'ytick.labelsize': 13,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 13,
'legend.title_fontsize': 13, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})
sns.set_style("white")

#%%
### make dataframe of delta TU (pre -> post)
TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0)
majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = majorminor[majorminor['type']=='major']['gene_ENST'].to_list()
minorlist = majorminor[majorminor['type']=='minor']['gene_ENST'].to_list()


#%%
dut_r = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t')
dut_r = dut_r[(dut_r['p_value']<0.05)&(np.abs(dut_r['log2FC'])>1.5)]
dut_nr = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/nonresponder_stable_DUT_Wilcoxon.txt', sep='\t')
dut_nr = dut_nr[(dut_nr['p_value']<0.05)&(np.abs(dut_nr['log2FC'])>1.5)]

from matplotlib_venn import venn2
vd2 = venn2([set(dut_r['gene_ENST']),set(dut_nr['gene_ENST'])],set_labels=('AR', 'IR'),set_colors=('#F8CF6A','#74B2D4'), alpha=0.8)

#%%


dutlist = set(dut_r['gene_ENST']).union(set(dut_nr['gene_ENST']))

######** include only DUT #@####
df = TU[TU.index.isin(dutlist)]
#*##############################

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo_new.txt', sep='\t')

sampleinfo['Rgroup'] = sampleinfo['response'].apply(lambda x: 'AR' if x == 1 else 'IR')
sampleinfo['brca'] = sampleinfo['BRCAmut'].apply(lambda x: 'MT' if x == 1 else 'WT')

colors_group = { "AR":"#FF9595", "IR":"#7CB5FF"} 
col_colors_group = sampleinfo['Rgroup'].map(colors_group)

colormap = {
    'group': {"pre_R": "#FFB7B7", "post_R":"#FF8A8A","pre_NR":"#9AC6FF", "post_NR":"#5DA3FF"},
    'line_binary': {"FL":"#F1F1E8", "N-FL": "#90C96E"},
    'BRCAmut': {1: "#FFFFFF", 1: "#898989"},
    'survival':{0:"#898989",1:"#FFFFFF"},
}

# Convert metadata categories to colors
col_colors = sampleinfo.apply(
    lambda col: col.map(colormap[col.name]) if col.name in colormap else col,
    axis=0
)
col_colors = col_colors[['group','line_binary','BRCAmut','survival']]
os_min, os_max = sampleinfo['PFI'].min(), sampleinfo['PFI'].max()
os_cmap = plt.cm.gray
col_colors['PFI'] = sampleinfo['PFI'].map(lambda x: os_cmap(1 - (x - os_min) / (os_max - os_min))) #smaller white bigger black
col_colors.index = sampleinfo['sample_full']
#%%
majordf = df[df.index.isin(majorlist)]
minordf = df[df.index.isin(minorlist)]
majordf['transcript'] = 'major'
minordf['transcript'] = 'minor'
dfconcat = pd.concat([majordf,minordf],axis=0)
dfatd = dfconcat.iloc[:,0::2]
dfbfd = dfconcat.iloc[:,1::2]
finaldf = pd.concat([dfbfd,dfatd],axis=1)

##^^ only AR ########
ARlist = sampleinfo[sampleinfo['Rgroup']=='AR']['sample_id']
IRlist = sampleinfo[sampleinfo['Rgroup']=='IR']['sample_id']

#%%
from matplotlib.pyplot import gcf
from sklearn.preprocessing import StandardScaler

with open('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/featureselection/Reactome2022_featurelist.pkl', 'rb') as file:
    featurelist_reactome = pickle.load(file)

with open('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/featureselection/BP2018_featurelist.pkl', 'rb') as file:
    featurelist_bp = pickle.load(file)

featurelist = set(featurelist_bp).union(set(featurelist_reactome))


#* draw seaborn clustermap: sample type AR, I
col_colors_treatment = ["#939B62"] * 40 + ["#436420"] * 40
row_colors_major = ["#939B62"] * 2154 + ["#436420"] * 26558

input = majordf # major vs. minor
input = input.loc[input.index.isin(featurelist),:] ##only DNArepair genes
ARinput = input[[col for col in input.columns if any(keyword in col for keyword in ARlist)]] 
IRinput = input[[col for col in input.columns if any(keyword in col for keyword in IRlist)]] 
ARinput = pd.concat([ARinput.iloc[:,1::2],ARinput.iloc[:,0::2]], axis=1)
IRinput = pd.concat([IRinput.iloc[:,1::2],IRinput.iloc[:,0::2]], axis=1)
finalinput = pd.concat([ARinput,IRinput],axis=1)
#finalinput = ARinput

#finalinput = finalinput.loc[finalinput.std(axis=1).nlargest(2000).index] ##largest std
#finalinput = finalinput.loc[finalinput.std(axis=1)>0.01]

scaler =  MinMaxScaler(feature_range=(-1, 1))
df_normalized = (finalinput.T - finalinput.mean(axis=1)) / finalinput.std(axis=1)
df_normalized = df_normalized.T

# Convert metadata categories to colors
metadata = sampleinfo.loc[sampleinfo['sample_full'].isin(finalinput.columns),:]
#metadata['group'] = metadata['group'].replace('post_NR', 'post_R')

col_colors = metadata.apply(
    lambda col: col.map(colormap[col.name]) if col.name in colormap else col,
    axis=0
)
col_colors = col_colors[['group','BRCAmut','survival']]
os_min, os_max = metadata['PFI'].min(), metadata['PFI'].max()
os_cmap = plt.cm.gray
col_colors['PFI'] = sampleinfo['PFI'].map(lambda x: os_cmap(1 - (x - os_min) / (os_max - os_min)))
col_colors.index = metadata['sample_full']

row_linkage_matrix = g.dendrogram_row.linkage  # Access the row linkage from the heatmap
n_row_clusters = 5  # Define the number of row clusters (adjust as needed)
row_cluster_labels = fcluster(row_linkage_matrix, n_row_clusters, criterion='maxclust')

# Map row cluster labels to colors
row_cluster_color_map = {1: "red", 2: "blue", 3:"green", 4:"yellow", 5:"white"}  # Define colors for clusters
row_cluster_colors = pd.Series(row_cluster_labels, index=df_normalized.index).map(row_cluster_color_map)

# Save transcripts for each cluster
cluster_transcripts = {}
for cluster, color in row_cluster_color_map.items():
    cluster_transcripts[color] = df_normalized[row_cluster_colors == color].index.tolist()
    print(f"Transcripts in {color} cluster: {cluster_transcripts[color]}")

g = sns.clustermap(
    df_normalized ,
    method='ward',  # Clustering method
    cmap='vlag',  # Heatmap color scheme
    row_cluster=True,  # Enable row clustering
    col_cluster=True,  # Enable column clustering
    col_colors=col_colors,
    row_colors=row_cluster_colors,
    yticklabels=False,
    xticklabels=True,
    #z_score=0,
    #cbar=False,
    figsize=(10, 7),
    cbar_pos=(-0.11, 0.3, 0.03, 0.2),  # Adjust the colorbar position
    dendrogram_ratio=(0.1, 0.1),  # Adjust dendrogram sizes (row, column)
    colors_ratio=0.03,
    center=0,  # Center the colormap at 0
    vmax=1,    # Set maximum limit
    vmin=-1,
)
from matplotlib.patches import Patch

g.ax_heatmap.set_ylabel("")
plt.show()

# %%
transet = cluster_transcripts['yellow']
glist = [item.split('-',1)[1] for item in transet]

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

file = enr.results.sort_values(by=['Adjusted P-value']) 
file['Adjusted P-value'] = -np.log10(file['Adjusted P-value'])
file = file[file['Adjusted P-value']>1]
print(file)
# %%
