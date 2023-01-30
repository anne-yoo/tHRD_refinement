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
tpm = pd.read_csv('/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/pre_post_TU/final_data/230106_new_final_pre_post_samples_TU_input.txt', sep='\t', index_col=0)
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
