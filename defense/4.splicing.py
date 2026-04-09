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
###################*   
events=['SE','A5','A3','MX','RI'] 
suppa = []

###file download###
for i in range(5):
    psi_suppa = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/suppa2/suppa_'+events[i]+'_variable_10.ioe', sep='\t', index_col=0)
    suppa.append(psi_suppa.shape[0])

#%%
events=['SE','A5SS','A3SS','MXE','RI'] 
rmats = []
for i in range(5):
    psi_rmats = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/rMATS/output_gencode/'+events[i]+'.MATS.JC.txt', sep='\t', index_col=0)
    rmats.append(psi_rmats.shape[0])
# %%
# Data
values1 = [37459, 28674, 31102, 4781, 21190]
values2 = [152881, 6913, 10458, 31751, 5371]
categories = ['SE', 'A5', 'A3', 'MX', 'RI']

####^^^ Barplot for counting events #####
df = pd.DataFrame({
    'Category': categories * 2,
    'Value': values1 + values2,
    'Type': ['SUPPA2']*len(categories) + ['rMATs']*len(categories)
})

# Plotting with the updated legend
plt.figure(figsize=(4,2))
sns.set(style='whitegrid')
plt.figure(figsize=(10, 6))
sns.barplot(x='Value', y='Category', hue='Type', data=df, palette='Blues')
plt.xlabel('number of events')
plt.ylabel('event category')
plt.legend(title='Methods')
plt.tight_layout()
plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/fig1c_suppa2_rmats.pdf", bbox_inches="tight")
plt.show()






#%%
#######^^ Manual dpsi analysis: SUPPA2 ##########
ensembl = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/annotation/ensg2symbol.dict', sep='\t', header=None)
ensembl.columns = ['gene id', 'ensembl gene symbol']

mergedmw = pd.DataFrame()

events=['SE','A5','A3','MX','RI'] 

for k in range(5):
    pre = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/suppa2/pre-suppa_'+events[k]+'_variable_10.ioe.psi', sep='\t', index_col=0)
    post = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/suppa2/post-suppa_'+events[k]+'_variable_10.ioe.psi', sep='\t', index_col=0)
    pre['null_count'] = pre.isnull().sum(axis=1)
    post['null_count'] = post.isnull().sum(axis=1)
    
    #*mannwhitney
    mw = pd.DataFrame()
    mw.index = pre.index
    mw['pval'] = 100
    mw['d_psi'] = 100
    mw['event'] = events[k]
    for i in range(post.shape[0]):
        if (pre.iloc[i,-1] != 40) & (post.iloc[i,-1] != 40):
            list_pre = pre.iloc[i,:pre.shape[1]-1]
            list_post = post.iloc[i,:post.shape[1]-1]
            if set(list_pre) != set(list_post):
                mw.iloc[i,0] = stats.mannwhitneyu(list_pre, list_post, alternative='two-sided')[1]
            d_psi = (list_post.mean() - list_pre.mean())
            mw.iloc[i,1] = d_psi
        
    mw['gene id'] = list(mw.index.str.split(';',1).str[0])
    mw['p_adj'] = multipletests(mw['pval'], method='fdr_bh')[1]
    mw = pd.merge(mw,ensembl,how='inner',left_on='gene id', right_on='gene id')
    
    mergedmw = pd.concat([mergedmw, mw])
mergedmw.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/suppa2/dpsi_analysis/MW_dpsi_5events.txt', index=False, sep='\t')




#%%
#######^^ Manual dpsi analysis: rMATs ##########

mergedmw = pd.DataFrame()

events=['SE','A5SS','A3SS','MXE','RI'] 

for k in range(5):
    psi = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/rMATS/output_gencode/'+events[k]+'.MATS.JC.txt', sep='\t', index_col=0)
    pre = []
    for row in psi['IncLevel1']:
        row_values = []
        for val in row.split(','):
            if val == 'NA':
                row_values.append(np.nan)
            else:
                row_values.append(float(val))
        pre.append(row_values)
    pre = pd.DataFrame(pre)
    
    post = []
    for row in psi['IncLevel2']:
        row_values = []
        for val in row.split(','):
            if val == 'NA':
                row_values.append(np.nan)
            else:
                row_values.append(float(val))
        post.append(row_values)
    post = pd.DataFrame(post)
    
    pre['null_count'] = pre.isnull().sum(axis=1)
    post['null_count'] = post.isnull().sum(axis=1)
    
    #*mannwhitney
    mw = pd.DataFrame()
    mw.index = psi.index
    mw['pval'] = 100
    mw['d_psi'] = 100
    mw['event'] = events[k]
    for i in range(post.shape[0]):
        if (pre.iloc[i,-1] != 40) & (post.iloc[i,-1] != 40):
            list_pre = pre.iloc[i,:pre.shape[1]-1]
            list_post = post.iloc[i,:post.shape[1]-1]
            if set(list_pre) != set(list_post):
                mw.iloc[i,0] = stats.mannwhitneyu(list_pre, list_post, alternative='two-sided')[1]
            d_psi = (list_post.mean() - list_pre.mean())
            mw.iloc[i,1] = d_psi
    mw['p_adj'] = multipletests(mw['pval'], method='fdr_bh')[1]
    mw['gene id'] = psi['GeneID']
    mw['gene symbol'] = psi['geneSymbol']
    mergedmw = pd.concat([mergedmw, mw])

mergedmw.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/rMATS/dpsi_analysis/MW_dpsi_5events.txt', sep='\t', index=False)



# %%
####^ Significant Events #####
suppa_sig = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/suppa2/dpsi_analysis/MW_dpsi_5events.txt', sep='\t')
rmats_sig = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/rMATS/dpsi_analysis/MW_dpsi_5events.txt', sep='\t')
suppa_sig = suppa_sig[(suppa_sig['pval']<0.05) & (suppa_sig['d_psi']>0.01)]
rmats_sig = rmats_sig[(rmats_sig['pval']<0.05) & (rmats_sig['d_psi']>0.01)]

rmats_count = []
suppa_count = []

events=['SE','A5','A3','MX','RI'] 
events2 = ['SE','A5SS','A3SS','MXE','RI'] 
for i in range(5):
    rmats_count.append(rmats_sig[rmats_sig['event']==events2[i]].shape[0])
    suppa_count.append(suppa_sig[suppa_sig['event']==events[i]].shape[0])


# %%
# Data
values1 = [6927, 4381, 4279, 1692, 2258]
values2 = [29740, 1254, 1824, 3676, 698]

categories = ['SE', 'A5', 'A3', 'MX', 'RI']

####^^^ Barplot for counting events #####
df = pd.DataFrame({
    'Category': categories * 2,
    'Value': values1 + values2,
    'Type': ['SUPPA2']*len(categories) + ['rMATs']*len(categories)
})

# Plotting with the updated legend
plt.figure(figsize=(4,2))
sns.set(style='whitegrid')
plt.figure(figsize=(10, 6))
sns.barplot(x='Value', y='Category', hue='Type', data=df, palette='Blues')
plt.xlabel('number of differential splicing events')
plt.ylabel('event category')
plt.legend(title='Methods')
plt.tight_layout()
plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/fig1c_sig_suppa2_rmats.pdf", bbox_inches="tight")
plt.show()

# %%
####^^^^^^ SUPPA2 #########
values1 = [37459, 28674, 31102, 4781, 21190]
values2 = [6927, 4381, 4279, 1692, 2258]

categories = ['SE', 'A5', 'A3', 'MX', 'RI']

####^^^ Barplot for counting events #####
df = pd.DataFrame({
    'Category': categories * 2,
    'Value': values1 + values2,
    'Type': ['Whole']*len(categories) + ['Significant']*len(categories)
})

# Plotting with the updated legend
plt.figure(figsize=(4,2))
sns.set(style='whitegrid')
plt.figure(figsize=(10, 6))
sns.barplot(x='Value', y='Category', hue='Type', data=df, palette='Blues')
plt.xlabel('number of events')
plt.ylabel('event category')
plt.legend(title='Methods')
plt.tight_layout()
plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/fig1c_suppa2.pdf", bbox_inches="tight")
plt.show()

# %%
####^^^^^^ rMATs #########
values1 = [152881, 6913, 10458, 31751, 5371]
values2 = [29740, 1254, 1824, 3676, 698]

categories = ['SE', 'A5', 'A3', 'MX', 'RI']

####^^^ Barplot for counting events #####
df = pd.DataFrame({
    'Category': categories * 2,
    'Value': values1 + values2,
    'Type': ['Whole']*len(categories) + ['Significant']*len(categories)
})

# Plotting with the updated legend
plt.figure(figsize=(4,2))
sns.set(style='whitegrid')
plt.figure(figsize=(10, 6))
sns.barplot(x='Value', y='Category', hue='Type', data=df, palette='Blues')
plt.xlabel('number of events')
plt.ylabel('event category')
plt.legend(title='Methods')
plt.tight_layout()
plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/fig1c_rmats.pdf", bbox_inches="tight")
plt.show()

# %%
#######^^^^make rMATs psi final dataframe ######
events=['SE','A5SS','A3SS','MXE','RI'] 

wholedf = pd.DataFrame()
for k in range(5):
    psi = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/rMATS/output_gencode/'+events[k]+'.MATS.JC.txt', sep='\t', index_col=0)
    pre = []
    for row in psi['IncLevel1']:
        row_values = []
        for val in row.split(','):
            if val == 'NA':
                row_values.append(np.nan)
            else:
                row_values.append(float(val))
        pre.append(row_values)
    pre = pd.DataFrame(pre)
    
    post = []
    for row in psi['IncLevel2']:
        row_values = []
        for val in row.split(','):
            if val == 'NA':
                row_values.append(np.nan)
            else:
                row_values.append(float(val))
        post.append(row_values)
    post = pd.DataFrame(post)
    
    prepost = pd.concat([pre,post], axis=1)
    prepost['event'] = events[k]
    wholedf = pd.concat([wholedf,prepost])
    print(prepost.shape[0])
# %%
wholedf.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/rMATS/dpsi_analysis/whole_psi.txt',sep='\t',index=False)

# %%
