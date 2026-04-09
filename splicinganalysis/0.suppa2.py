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
import random

# %%
#tpm = pd.read_csv('/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/pre_post_TU/final_data/230106_new_final_pre_post_samples_TU_input.txt', sep='\t', index_col=0)
# info = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost_80_clinicalinfo.txt', sep='\t')
# info = info[info['purpose']=='maintenance']
# arlist = info[info['response']==1]['sample_full'].to_list()
# irlist = info[info['response']==0]['sample_full'].to_list()

transexp = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/merged_83_transcript_TPM.txt', sep='\t', index_col=0)
val_clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2025clinical/83_POLO_clinicalinfo.txt', sep='\t', index_col=0)
res_list = val_clin[(val_clin['PFS']>=730)].index.to_list()
nonres_list = val_clin[(val_clin['recur']==1)&(val_clin['PFS']<365)].index.to_list()

res_tpm = transexp.loc[:, transexp.columns.isin(res_list)]
nonres_tpm = transexp.loc[:, transexp.columns.isin(nonres_list)]




#%%
tpm = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_80_transcript_TPM.txt', sep='\t', index_col=0)
#minimum_samples = int(tpm.shape[1] * 0.2) ########## 20% threshold
#tpm = tpm[tpm.apply(lambda x: (x != 0).sum(), axis=1) >= minimum_samples]

tpm = tpm.loc[:,tpm.columns.isin(arlist)]
normal = tpm.iloc[:,0::2]
tumor = tpm.iloc[:,1::2]
normal.index = normal.index.str.split("-",1).str[0]
tumor.index = tumor.index.str.split("-",1).str[0]

#normal = normal.drop(['gene_name'], axis=1)
#tumor = tumor.drop(['gene_name'], axis=1)

normal.to_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/suppaoutput/onlymaintenance/AR/AR_post_TPM.txt', sep='\t', index=True)
tumor.to_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/suppaoutput/onlymaintenance/AR/AR_pre_TPM.txt', sep='\t', index=True)

#%% 
#^ change sample id (atD / bfD)
tpm = tpm.drop(['target_gene'], axis=1)
samples = tpm.columns.to_list()
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

tpm.index = tpm.index.str.split("-",1).str[0]

# %%
pre = [x for x in samples if 'bfD' in x ]
post = [x for x in samples if 'atD' in x ]

pre_tu = tpm.filter(pre)
post_tu = tpm.filter(post)

#%%
pre_tu.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/strict/pre_TPM.txt', sep='\t', index=True)
post_tu.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/strict/post_TPM.txt', sep='\t', index=True)

# %%
test_pre = pre_tu.iloc[:1000,:]
test_post = post_tu.iloc[:1000,:]

test_pre.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/test_pre_TPM.txt', sep='\t')
test_post.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/test_post_TPM.txt', sep='\t')
# %%
pre_psi = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/pre-suppa_SE_variable_10.ioe.psi', sep='\t', index_col=0)
post_psi = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/post-suppa_SE_variable_10.ioe.psi', sep='\t', index_col=0)
ioe = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/suppa_SE_variable_10.ioe', sep='\t', index_col=0)

test_pre_psi = pre_psi.iloc[:1000,:]
test_post_psi = post_psi.iloc[:1000,:]
test_ioe = ioe.iloc[:1000,:]
# %%
test_pre_psi.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/test_pre_psi.psi', sep='\t')
test_post_psi.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/test_post_psi.psi', sep='\t')
test_ioe.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/test_ioe.ioe', sep='\t')


# %%
