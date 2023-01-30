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

#%%
#** download file: pval / pre mean TPM / post mean TPM <- mean per gene
TPMdata = pd.read_csv('/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/readcount/final_data/MW_result/MW_whole_pre-post.txt',sep='\t')
TPMdata.columns = ['gene','pval','pre_mean','post_mean']

# %%
TPMdata['D_mean'] = TPMdata['post_mean'] - TPMdata['pre_mean']
TPMdata = TPMdata[TPMdata['pval']>0.05]
sns.histplot(data=TPMdata, x='D_mean')

# %%
sns.histplot(data=TPMdata, x='pval')
# %%
TPMdata.quantile([0.25,0.75])['D_mean']
# %%
##* splicing gene where??
splicing_gene = pd.read_csv('/home/hyeongu/DATA5/hyeongu/GSEA/gmt/splicing_related_genes.txt', header=None)
splicing_gene.columns = ['gene']
sp_data = pd.merge(TPMdata,splicing_gene, on='gene',how='inner')

#%%
fig, ax = plt.subplots()
sns.histplot(data=TPMdata, x='D_mean', ax=ax, color='green', alpha=0.6)
sns.histplot(data=sp_data, x='D_mean', ax=ax, color='red',alpha=0.6)
plt.xlim([-100,100])

# %%
sp_data.quantile([0.25,0.75])['D_mean']

# %%
finaldata = TPMdata.loc[(TPMdata['D_mean']> -2.803527) & (TPMdata['D_mean']<2.983549),:]

# %%
