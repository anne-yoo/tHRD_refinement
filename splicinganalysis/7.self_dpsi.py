#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.stats import pearsonr
import random
from scipy.stats import ttest_ind
from scipy.stats import ranksums

# %% #^ transcript id - gene id library
mstrg = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/annotation/mstrg2symbol.txt', sep='\t')
ensembl = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/annotation/ensg2symbol.dict', sep='\t', header=None)
mstrg = mstrg[['qry_gene_id','ref_gene_id']]
mstrg.columns = ['gene id', 'mstrg gene symbol']
ensembl.columns = ['gene id', 'ensembl gene symbol']

# %%
events=['A3','A5','AF','AL','MX','RI','SE'] #A3, A5

pre = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/pre-suppa_'+events[2]+'_variable_10.ioe.psi', sep='\t', index_col=0)
post = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/post-suppa_'+events[2]+'_variable_10.ioe.psi', sep='\t', index_col=0)



# %%
pre['null_count'] = pre.isnull().sum(axis=1)
post['null_count'] = post.isnull().sum(axis=1)

# %%
mw = pd.DataFrame()
mw.index = pre.index
mw['pval'] = 5
#%%

mergedmw = pd.DataFrame()
for k in range(7):
    pre = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/pre-suppa_'+events[k]+'_variable_10.ioe.psi', sep='\t', index_col=0)
    post = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/post-suppa_'+events[k]+'_variable_10.ioe.psi', sep='\t', index_col=0)
    pre['null_count'] = pre.isnull().sum(axis=1)
    post['null_count'] = post.isnull().sum(axis=1)
    
    #*mannwhitney
    mw = pd.DataFrame()
    mw.index = pre.index
    mw['pval'] = 5
    mw['d_psi'] = 0
    for i in range(post.shape[0]):
        if (pre.iloc[i,-1] == 0) & (post.iloc[i,-1] == 0):
            list_pre = pre.iloc[i,:pre.shape[1]-1]
            list_post = post.iloc[i,:post.shape[1]-1]
            if set(list_pre) != set(list_post):
                mw.iloc[i,0] = mannwhitneyu(list_pre, list_post, alternative='two-sided')[1]
            d_psi = abs((list_post.mean() - list_pre.mean()))
            mw.iloc[i,1] = d_psi
    con = (mw['pval']<0.05) & (mw['d_psi']>=0.1)
    filtered_mw = mw.loc[con,:]

    #* transcript id + gene id extracting
    filtered_mw['gene id'] = filtered_mw.index.str.split(";",1).str[0]
    filtered_mw['transcript id'] = filtered_mw.index

    mw_mstrg = pd.merge(filtered_mw, mstrg, how='left', left_on = 'gene id', right_on='gene id')
    mw_mstrg = mw_mstrg.drop_duplicates(subset=['pval','d_psi','gene id','transcript id'])
    mw_ensembl = pd.merge(mw_mstrg, ensembl, how='left', left_on = 'gene id', right_on = 'gene id')
    finaldf = mw_ensembl.drop_duplicates(subset=['pval','d_psi','gene id','transcript id'])
    finaldf['event'] = events[k]

    mergedmw = pd.concat([mergedmw, finaldf])





# %%
mergedmw.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/MW_psi_7events.txt', index=False, sep='\t')


# %%
for k in range(7):
    pre = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/pre-suppa_'+events[k]+'_variable_10.ioe.psi', sep='\t', index_col=0)
    post = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/post-suppa_'+events[k]+'_variable_10.ioe.psi', sep='\t', index_col=0)
    pre['null_count'] = pre.isnull().sum(axis=1)
    post['null_count'] = post.isnull().sum(axis=1)
    
    #*mannwhitney
    mw = pd.DataFrame()
    mw.index = pre.index
    mw['pval'] = 5
    mw['d_psi'] = 0
    for i in range(post.shape[0]):
        if (pre.iloc[i,-1] == 0) & (post.iloc[i,-1] == 0):
            list_pre = pre.iloc[i,:pre.shape[1]-1]
            list_post = post.iloc[i,:post.shape[1]-1]
            if set(list_pre) != set(list_post):
                mw.iloc[i,0] = mannwhitneyu(list_pre, list_post, alternative='two-sided')[1]
            d_psi = abs((list_post.mean() - list_pre.mean()))
            mw.iloc[i,1] = d_psi
            cal_mw = mw[mw['pval']==5]

    print(events[k], cal_mw.shape[0])
# %%
