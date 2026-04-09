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


#%%
TPM = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM_symbol.txt', sep='\t', index_col=0)
TPM = TPM.set_index('Gene Symbol')
pre = TPM.iloc[:,1::2]
post = TPM.iloc[:,0::2]

TPM.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/cybersort/80_discovery_gene_exp_TPM_symbol.txt', sep='\t', index=True)
pre.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/cybersort/pre_OV_TPM.txt', sep='\t', index=True)
post.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/cybersort/post_OV_TPM.txt', sep='\t', index=True)


# %%
enr = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202401_analysis/DUT/responder_stable_Wilcoxon_GOenrichment_FC_15.txt', sep='\t')

data = enr
data = data[data["Adjusted P-value"] <= 0.01]
data = data[(data["Term"].str.contains("DNA repair", case=False))]

dnarepairgenes = [gene for sublist in data['Genes'].str.split(';') for gene in sublist]
dnarepairgenes = set(dnarepairgenes)


# %%
tu = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_TU.txt', sep='\t', index_col=0)
# %%
dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t')
# %%
sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
responder = sampleinfo.loc[(sampleinfo['response']==1),'sample_full'].to_list()
nonresponder = sampleinfo.loc[(sampleinfo['response']==0),'sample_full'].to_list()


# %%
figdf = tu.loc[:,tu.columns.isin(responder)]
#figdf = tu
trans = 'MSTRG.80592.1-SMARCAD1'
figdf = figdf.loc[(tu.index == trans) ,:]
# | (tu.index == 'MSTRG.49834.247-BRIP1')
figdf = figdf.T
figdf.columns = ['TU']
figdf['treatment'] = 'pre'
figdf.iloc[0::2,1] = 'post'

figdf['sampleid'] = figdf.index.str[:-4]

##^ boxplot ###

plt.figure(figsize=(5,8))
sns.set_theme(style='ticks', palette='pastel', font_scale=1)

pal = ['#6196A6','#A4CE95']
ax = sns.boxplot(y='TU', x='treatment', data=figdf, order=['pre','post'],
            showmeans=False, 
            #meanprops = {"marker":"o", "color":"blue","markeredgecolor": "blue", "markersize":"10"}
            palette=pal)

from statannot import add_stat_annotation
add_stat_annotation(ax, data=figdf, x='treatment', y='TU',
                    box_pairs=[("pre", "post")],
                    test='Wilcoxon', 
                    comparisons_correction=None,text_format='simple', loc='inside')

for p in figdf['sampleid'].unique():
    subset = figdf[figdf['sampleid']==p]
    x = [list(subset['treatment'])[1], list(subset['treatment'])[0]]
    y = [list(subset['TU'])[1], list(subset['TU'])[0]]
    plt.plot(x,y, marker="o", markersize=4, color='grey', linestyle="--", linewidth=0.7)
plt.title(trans)

# %%


