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

# %%
###^^ (1) Whole major TU heatmap ###

### make dataframe of delta TU (pre -> post)
TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0)
TU['Gene Symbol'] = TU.index.str.split("-",2).str[1]


dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t')

###**####
dut = dut[(dut['p_value']<0.05)&(np.abs(dut['log2FC'])>1)]
check = dut
#dut = dut[dut['log2FC']!=np.inf]
###**####

dutlist = set(dut['gene_ENST'])

enr = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202401_analysis/DUT/responder_stable_Wilcoxon_GOenrichment_FC_1.txt', sep='\t')

######** include only DUT #@####
TU = TU[TU.index.isin(dutlist)]
#*##############################

enr = enr[(enr["Term"].str.contains("GO:0006302", case=False))|(enr["Term"].str.contains("GO:0006974", case=False))]
dnarepairgenes = [gene for sublist in enr['Genes'].str.split(';') for gene in sublist]
dnarepairgenes = set(dnarepairgenes)

TU = TU[TU['Gene Symbol'].isin(dnarepairgenes)]

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')

major = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/final_202306_discovery_major_TU.txt', sep='\t', index_col=0)
minor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/final_202306_discovery_minor_TU.txt', sep='\t', index_col=0)

majortrans = major.index.tolist()
minortrans = minor.index.tolist()

###^ only AR samples ###
ARlist = list(sampleinfo.loc[sampleinfo['response']==1,'sample_full'])
TU = TU.loc[:,ARlist]
sampleinfo = sampleinfo.loc[sampleinfo['sample_full'].isin(ARlist),:]
##^#####################

###^ only 1L samples ###
#ARlist = list(sampleinfo.loc[sampleinfo['line_binary']=='FL','sample_full'])
#TU = TU.loc[:,ARlist]
#sampleinfo = sampleinfo.loc[sampleinfo['sample_full'].isin(ARlist),:]
##^#####################



####** all transcript vs. only major transcripts ######
majorTU = TU[TU.index.isin(majortrans)]
#TU['type'] = TU.index.to_series().apply(lambda x: 'major' if x in majortrans else ('minor' if x in minortrans else None))
#majorTU = TU
####***################################################

df_bfD = majorTU[[col for col in majorTU.columns if '-bfD' in col]]
df_atD = majorTU[[col for col in majorTU.columns if '-atD' in col]]

df_bfD.columns = df_bfD.columns.str[:-4]
df_atD.columns = df_atD.columns.str[:-4]

delta_df = df_atD.subtract(df_bfD)

####**** only certain genes ######
#delta_df = delta_df.loc[(row for row in delta_df.index if 'BRIP1' in row), :]
####****##########################

sampleinfo['Rgroup'] = sampleinfo['response'].apply(lambda x: 'AR' if x == 1 else 'IR')
sampleinfo['brca'] = sampleinfo['BRCAmut'].apply(lambda x: 'MT' if x == 1 else 'WT')

sampleinfo = sampleinfo.iloc[0::2,:]

colors_group = { "AR":"#FF9595", "IR":"#7CB5FF"} 
col_colors_group = sampleinfo['Rgroup'].map(colors_group)

colors_brca = { "MT":"#FF7B54", "WT":"#939B62"} 
col_colors_brca = sampleinfo['brca'].map(colors_brca)



#%%
from matplotlib.pyplot import gcf

#* draw seaborn clustermap: sample type AR, I

plt.figure(figsize=(8,11))
g = sns.clustermap(delta_df, col_colors=[col_colors_group], cmap="RdBu_r" , figsize=(8,11), 
                #vmin=-2,vmax=2, center=0,
                method='average', #single, complete, average, weighted, ward
                metric='euclidean', #cityblock, minkowski, euclidean, cosine, correlation, hamming, jaccard, chebyshev, canberra, dice, rogerstanimoto, russellrao, sokalsneath
                linewidths=0, xticklabels=True, yticklabels=False, col_cluster=True,row_cluster=True, standard_scale=1)

g.fig.subplots_adjust(right=0.7)
g.ax_cbar.set_position((0.73, .2, .03, .4))

for group in sampleinfo['Rgroup'].unique():
    g.ax_col_dendrogram.bar(0, 0, color=colors_group[group],
                            label=group, linewidth=0)

g.ax_col_dendrogram.legend(title='sample type', ncol=2, bbox_to_anchor=(0.2, 0.98), bbox_transform=gcf().transFigure) 

ax = g.ax_heatmap
# ax.set_xlabel("Samples")
ax.set_ylabel("")
#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202401_analysis/heatmap/majorgroup_whole.pdf", bbox_inches="tight")
plt.show()

# %%
#* draw seaborn clustermap: BRCA MT, WT

plt.figure(figsize=(8,11))

g = sns.clustermap(delta_df, col_colors=[col_colors_brca], cmap="RdBu_r" , figsize=(8,11), 
                #vmin=-1,vmax=1, center=0,
                method='average', #single, complete, average, weighted, ward
                metric='euclidean', #cityblock, minkowski, euclidean, cosine, correlation, hamming, jaccard, chebyshev, canberra, dice, rogerstanimoto, russellrao, sokalsneath
                linewidths=0, xticklabels=True, yticklabels=False, col_cluster=True,row_cluster=True, standard_scale=1)

g.fig.subplots_adjust(right=0.7)
g.ax_cbar.set_position((0.73, .2, .03, .4))

for group in sampleinfo['brca'].unique():
    g.ax_col_dendrogram.bar(0, 0, color=colors_brca[group],
                            label=group, linewidth=0)

g.ax_col_dendrogram.legend(title='BRCA mutation', ncol=2, bbox_to_anchor=(0.2, 0.98), bbox_transform=gcf().transFigure) 
ax = g.ax_heatmap
# ax.set_xlabel("Samples")
ax.set_ylabel("")

#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202401_analysis/heatmap/majorbrca_whole.pdf", bbox_inches="tight")
plt.show()

#%%
# %%
####^^ cluster comparison ####

new_sample = sampleinfo[['sample_id','sample_full']]
#new_sample['cluster'] = [1,1,0,0,1,1,1,0,1,1,1,0,1,0,0,0,0] ###IR
new_sample['cluster'] = [1,1,1,0,1,1,1,1,1,1,1,0,1,0,1,1,0,0,1,0,1,0,0]
new_sample['cluster'] = new_sample['cluster'].replace(1,"up")
new_sample['cluster'] = new_sample['cluster'].replace(0,"down")

colors_cluster = { "up":"#5465FF", "down":"#BC67C6"} 
col_colors_cluster = new_sample['cluster'].map(colors_cluster)

plt.figure(figsize=(8,5))

g = sns.clustermap(delta_df, col_colors=[col_colors_cluster], cmap="RdBu_r" , figsize=(8,10), 
                #vmin=-1,vmax=1, center=0,
                method='average', #single, complete, average, weighted, ward
                metric='euclidean', #cityblock, minkowski, euclidean, cosine, correlation, hamming, jaccard, chebyshev, canberra, dice, rogerstanimoto, russellrao, sokalsneath
                linewidths=0, xticklabels=True, yticklabels=False, col_cluster=True,row_cluster=False, standard_scale=1)

g.fig.subplots_adjust(right=0.7)
g.ax_cbar.set_position((0.73, .2, .03, .4))

for group in new_sample['cluster'].unique():
    g.ax_col_dendrogram.bar(0, 0, color=colors_cluster[group],
                            label=group, linewidth=0)

g.ax_col_dendrogram.legend(title='cluster', ncol=2, bbox_to_anchor=(0.2, 0.98), bbox_transform=gcf().transFigure) 
ax = g.ax_heatmap
# ax.set_xlabel("Samples")
ax.set_ylabel("")

#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202401_analysis/heatmap/majorbrca_whole.pdf", bbox_inches="tight")
plt.show()

#%%
###^ PFS comparison ####
new_sample['interval'] = sampleinfo['interval']
new_sample['status'] = True

T = new_sample["interval"]
E = new_sample["status"]
plt.hist(T, bins = 50)
plt.show()

from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()
kmf.fit(durations = T, event_observed = E)
#kmf.plot_survival_function()

ax = plt.subplot(111)
sns.set_style('whitegrid')
m = (new_sample["cluster"] == 'up')
kmf.fit(durations = T[m], event_observed = E[m], label = "up")
kmf.plot_survival_function(ax = ax, color="#5465FF")

kmf.fit(T[~m], event_observed = E[~m], label = "down")
kmf.plot_survival_function(ax = ax, at_risk_counts = False, color="#BC67C6")
plt.xlabel('treatment interval')


#%%























# %%
###^^ (2) only AR samples, DNA repair major TU heatmap: pre-post ###

TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0)
TU['Gene Symbol'] = TU.index.str.split("-",2).str[1]

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')

dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t')
enr = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202401_analysis/DUT/responder_stable_Wilcoxon_GOenrichment_FC_1.txt', sep='\t')

###**####
dut = dut[(dut['p_value']<0.05)&(dut['log2FC']>1)]
#dut = dut[dut['log2FC']!=np.inf]
###**####

dutlist = set(dut['gene_ENST'])

######** include only DUT #@####
TU = TU[TU.index.isin(dutlist)]
#*##############################
enr = enr[(enr["Term"].str.contains("GO:0006302", case=False))|(enr["Term"].str.contains("GO:0006974", case=False))]
dnarepairgenes = [gene for sublist in enr['Genes'].str.split(';') for gene in sublist]
dnarepairgenes = set(dnarepairgenes)

finaldf = TU[TU['Gene Symbol'].isin(dnarepairgenes)]


##*#only AR ###
ARlist = list(sampleinfo.loc[sampleinfo['response']==1,'sample_full'])
finaldf = finaldf.loc[:,ARlist]

ARinfo = sampleinfo.loc[sampleinfo['sample_full'].isin(ARlist),:]

major = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/final_202306_discovery_major_TU.txt', sep='\t', index_col=0)
minor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/final_202306_discovery_minor_TU.txt', sep='\t', index_col=0)

majortrans = major.index.tolist()
minortrans = minor.index.tolist()

####** all transcript vs. only major transcripts ######
majorTU = finaldf[finaldf.index.isin(majortrans)]
#finaldf['type'] = finaldf.index.to_series().apply(lambda x: 'major' if x in majortrans else ('minor' if x in minortrans else None))
####**#################################################

#%%

major_pre = majorTU[[col for col in majorTU.columns if '-bfD' in col]]
major_post = majorTU[[col for col in majorTU.columns if '-atD' in col]]

finalTU = pd.concat([major_pre,major_post],axis=1)

finalTU = finalTU.loc[:, (finalTU != 0).any(axis=0)]
finalcol = finalTU.columns


####**** only certain genes ######
#finalTU = finalTU.loc[(row for row in finalTU.index if 'BRIP1' in row), :]
####****##########################


sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
sampleinfo = sampleinfo.loc[sampleinfo['sample_full'].isin(ARlist),:]

sampleinfo['Rgroup'] = sampleinfo['response'].apply(lambda x: 'AR' if x == 1 else 'IR')
sampleinfo['brca'] = sampleinfo['BRCAmut'].apply(lambda x: 'MT' if x == 1 else 'WT')


sample_pre = sampleinfo.iloc[1::2,:]
sample_post = sampleinfo.iloc[0::2,:]

final_sample = pd.concat([sample_pre,sample_post],axis=0)
final_sample = final_sample[final_sample['sample_full'].isin(finalcol)]

colors_group = { "pre":"#FF9595", "post":"#7CB5FF"} 
col_colors_group = final_sample['treatment'].map(colors_group)

colors_brca = { "MT":"#FF7B54", "WT":"#939B62"} 
col_colors_brca = final_sample['brca'].map(colors_brca)

from matplotlib.pyplot import gcf

#colors_transcript = {"major":"#20722F", "minor":"#C37C02"}
#row_colors = finaldf['type'].map(colors_transcript)

#%%
#* draw seaborn clustermap: group pre, post
plt.figure(figsize=(8,11))
g = sns.clustermap(finalTU, col_colors=[col_colors_group],  cmap="RdBu_r" , figsize=(8,11), 
                #vmin=-2,vmax=2, center=0,
                
                method='weighted', #single, complete, average, weighted, ward
                metric='hamming', #cityblock, minkowski, euclidean, cosine, correlation, hamming, jaccard, chebyshev, canberra, dice, rogerstanimoto, russellrao, sokalsneath
                linewidths=0, xticklabels=True, yticklabels=True, col_cluster=True, row_cluster=True, standard_scale=1)

g.fig.subplots_adjust(right=0.7)
g.ax_cbar.set_position((0.73, .2, .03, .4))

for group in ['pre','post']:
    g.ax_col_dendrogram.bar(0, 0, color=colors_group[group],
                            label=group, linewidth=0)

g.ax_col_dendrogram.legend(title='sample type', ncol=2, bbox_to_anchor=(0.2, 0.98), bbox_transform=gcf().transFigure) 

ax = g.ax_heatmap
# ax.set_xlabel("Samples")
ax.set_ylabel("")
#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202401_analysis/heatmap/majorgroup_whole.pdf", bbox_inches="tight")
plt.show()

# %%
#* draw seaborn clustermap: BRCA MT, WT

plt.figure(figsize=(8,5))

g = sns.clustermap(finalTU, col_colors=[col_colors_brca], cmap="RdBu_r" , figsize=(8,10), 
                #vmin=-1,vmax=1, center=0,
                method='single', #single, complete, average, weighted, ward
                metric='hamming', #cityblock, minkowski, euclidean, cosine, correlation, hamming, jaccard, chebyshev, canberra, dice, rogerstanimoto, russellrao, sokalsneath
                linewidths=0, xticklabels=False, yticklabels=True, col_cluster=True,row_cluster=False, standard_scale=1)

g.fig.subplots_adjust(right=0.7)
g.ax_cbar.set_position((0.73, .2, .03, .4))

for group in sampleinfo['brca'].unique():
    g.ax_col_dendrogram.bar(0, 0, color=colors_brca[group],
                            label=group, linewidth=0)

g.ax_col_dendrogram.legend(title='BRCA mutation', ncol=2, bbox_to_anchor=(0.2, 0.98), bbox_transform=gcf().transFigure) 
ax = g.ax_heatmap
# ax.set_xlabel("Samples")
ax.set_ylabel("")

#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202401_analysis/heatmap/majorbrca_whole.pdf", bbox_inches="tight")
plt.show()




# %%
