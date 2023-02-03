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
from scipy.stats import kruskal
# %%
#####* DSG analysis with Kruskal-Wallis test #####
#^ transcript id - gene id library
mstrg = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/annotation/mstrg2symbol.txt', sep='\t')
ensembl = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/annotation/ensg2symbol.dict', sep='\t', header=None)
mstrg = mstrg[['qry_gene_id','ref_gene_id']]
mstrg.columns = ['gene id', 'mstrg gene symbol']
ensembl.columns = ['gene id', 'ensembl gene symbol']

events=['A3','A5','AF','AL','MX','RI','SE'] #A3, A5

pre = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/strict/pre-suppa_'+events[2]+'_strict.ioe.psi', sep='\t', index_col=0)
post = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/strict/post-suppa_'+events[2]+'_strict.ioe.psi', sep='\t', index_col=0)

pre['null_count'] = pre.isnull().sum(axis=1)
post['null_count'] = post.isnull().sum(axis=1)

mw = pd.DataFrame()
mw.index = pre.index
mw['pval'] = 5

mergedmw = pd.DataFrame()

for k in range(7):
    pre = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/strict/pre-suppa_'+events[k]+'_strict.ioe.psi', sep='\t', index_col=0)
    post = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/strict/post-suppa_'+events[k]+'_strict.ioe.psi', sep='\t', index_col=0)
    pre['null_count'] = pre.isnull().sum(axis=1)
    post['null_count'] = post.isnull().sum(axis=1)
    
    #* Kruskal-Wallis
    mw = pd.DataFrame()
    mw.index = pre.index
    mw['pval'] = 5
    mw['d_psi'] = 0
    for i in range(post.shape[0]):
        if (pre.iloc[i,-1] == 0) & (post.iloc[i,-1] == 0):
            list_pre = pre.iloc[i,:pre.shape[1]-1]
            list_post = post.iloc[i,:post.shape[1]-1]
            if set(list_pre) != set(list_post):
                mw.iloc[i,0] = kruskal(list_pre, list_post)[1]
            d_psi = list_post.mean() - list_pre.mean()
            mw.iloc[i,1] = d_psi
    # con = (mw['pval']<0.01) & (mw['d_psi']>0.1) #####pval filtering#####
    # filtered_mw = mw.loc[con,:]
    filtered_mw = mw
    #* transcript id + gene id extracting
    filtered_mw['gene id'] = filtered_mw.index.str.split(";",1).str[0]
    filtered_mw['transcript id'] = filtered_mw.index

    mw_mstrg = pd.merge(filtered_mw, mstrg, how='left', left_on = 'gene id', right_on='gene id')
    mw_mstrg = mw_mstrg.drop_duplicates(subset=['pval','d_psi','gene id','transcript id'])
    mw_ensembl = pd.merge(mw_mstrg, ensembl, how='left', left_on = 'gene id', right_on = 'gene id')
    finaldf = mw_ensembl.drop_duplicates(subset=['pval','d_psi','gene id','transcript id'])
    finaldf['event'] = events[k]

    mergedmw = pd.concat([mergedmw, finaldf])

#%%
#%%
##^ DTU gene list
forgenelist = mergedmw[['pval','d_psi','gene id','mstrg gene symbol','event','transcript id']]
forgenelist = forgenelist.dropna()
forgenelist = forgenelist[forgenelist['pval']!=5]
forgenelist = forgenelist[forgenelist['mstrg gene symbol']!='-']
dtu_genelist = pd.DataFrame(set(forgenelist['mstrg gene symbol']))
dtu_genelist.columns = ['dtu_genelist']

#%%
#* save DSG gene list
# dtu_genelist.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/suppa2/dtu_genelist.csv', index=False)
forgenelist.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/suppa2/DSGinfo_beforefiltering.csv', index=False, sep='\t')



# %%
##^ DTU gene -> psi dataframe for heatmap!
mergedpsi = pd.DataFrame()
for k in range(7):
    pre = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/strict/pre-suppa_'+events[k]+'_strict.ioe.psi', sep='\t', index_col=0)
    post = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/strict/post-suppa_'+events[k]+'_strict.ioe.psi', sep='\t', index_col=0)
    prepost = pd.concat([pre,post],axis=1)
    prepost['transcript id'] = prepost.index
    prepost['event'] = events[k]
    mergedpsi = pd.concat([mergedpsi,prepost])

formerge = forgenelist[['transcript id','event']]
finalpsi = pd.merge(mergedpsi,formerge, on=['transcript id','event'], how='inner')
finalpsi = finalpsi.set_index(['transcript id'])
# %%
finalpsi.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/suppa2/dtu_psi_forheatmap.csv', index=True)
# %%








#%%
#####* DEG analysis with Kruskal-Wallis test #####
tpm = pd.read_csv('/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/pre_post_TU/final_data/230106_new_final_pre_post_samples_TU_input.txt', sep='\t', index_col=0)

# remove transcript-gene unmatched transcripts
tpm = tpm[tpm['target_gene']!= '-']

# sample name change
sample = tpm.columns.tolist()# %%
samples = sample
for i in range(int(len(samples)/2)):
    i = i*2
    if "atD" in samples[i]:
        samples[i+1] = samples[i+1][:10]+"-bfD"
    elif "bfD" in samples[i]:
        samples[i+1] = samples[i+1][:10]+"-atD"
    elif "atD" in samples[i+1]:
        samples[i] = samples[i][:10]+"-bfD"
    elif "bfD" in samples[i+1]:
        samples[i] = samples[i][:10]+"-atD"

tpm.columns = samples
try:
    tpm = tpm.drop(["SV-OV-P080-atD","SV-OV-P250-atD","SV-OV-P055-atD",\
            "SV-OV-P143-atD","SV-OV-P137-atD","SV-OV-P134-atD",\
            "SV-OV-P174-bfD","SV-OV-P164-atD"], axis=1)
except:
    

    pass

rawcount = tpm

# rawcount = pd.read_csv('/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/readcount/final_data/230106_final_pre_post_samples_raw_counts.txt', sep='\t', index_col=0)
# rawcount = rawcount.drop(["SV-OV-P080-atD","SV-OV-P250-atD","SV-OV-P055-atD",\
                # "SV-OV-P143-atD","SV-OV-P137-atD","SV-OV-P134-atD",\
                # "SV-OV-P174-bfD","SV-OV-P164-atD"], axis=1)
# %%
pregroup = rawcount.loc[:, rawcount.columns.str.contains('atD')]
postgroup = rawcount.loc[:, rawcount.columns.str.contains('bfD')]
pregroup['null_count'] = pregroup.isnull().sum(axis=1)
postgroup['null_count'] = postgroup.isnull().sum(axis=1)

kw = pd.DataFrame()
kw.index = pregroup.index
kw['pval'] = 5
kw['d_exp'] = 0
for i in range(postgroup.shape[0]):
    if (pregroup.iloc[i,-1] == 0) & (postgroup.iloc[i,-1] == 0):
        list_pre = pregroup.iloc[i,:pregroup.shape[1]-1]
        list_post = postgroup.iloc[i,:postgroup.shape[1]-1]
        if set(list_pre) != set(list_post):
            kw.iloc[i,0] = kruskal(list_pre, list_post)[1]
        d_exp = abs((list_post.mean() - list_pre.mean()))
        kw.iloc[i,1] = d_exp
cond = (kw['pval']<0.01)
filtered_kw = kw.loc[cond,:]

# %%
filtered_kw_tpm = filtered_kw
filtered_kw_tpm['gene_ENST'] = filtered_kw_tpm.index
filtered_kw_tpm['gene symbol'] = filtered_kw_tpm.index.str.split("-",1).str[1]
# %%
filtered_kw_tpm = filtered_kw_tpm[filtered_kw_tpm['pval']<0.001]
deg_genelist = pd.DataFrame(set(filtered_kw_tpm['gene symbol']))
deg_genelist.columns = ['deg_genelist']
#%%
deg_genelist.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/DESeq2/KW_deg_genelist_withTPM.csv', index=False)
# %%
deg_transcriptlist = filtered_kw_tpm[['gene_ENST']]
inputtpm = rawcount
deg_tpm_forheatmap = pd.merge(inputtpm, deg_transcriptlist, left_index=True, right_index=True, how='inner')
deg_tpm_forheatmap = deg_tpm_forheatmap.set_index(['gene_ENST'])
deg_tpm_forheatmap = deg_tpm_forheatmap.drop(['target_gene'],axis=1)

#%%
####* save heatmap input of KW TPM DEG ####
deg_tpm_forheatmap.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/DESeq2/KW_TPM_DEG_forheatmap.csv', index=True)

#%%
#####* DESeq extract significant DEG genelist #####
deseq = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/DESeq2/sortedDEGresult_symbol.csv')
sig_pval = deseq[deseq['padj']<0.05]


##^ compare with KW result!!

# %%
kw_set = set(filtered_kw_tpm['gene symbol'])
deseq_set = set(sig_pval['gene symbol'])
inter = kw_set.intersection(deseq_set)
dtu_set = set(dtu_genelist['dtu_genelist'])
# %%
deg_deseq2_genelist = pd.DataFrame(deseq_set)
deg_deseq2_genelist.columns = ['deg_genelist']
deg_deseq2_genelist.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/DESeq2/DESeq2_deg_genelist.csv', index=False)

# 
