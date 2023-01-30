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

#%% #** data download
TUdata = pd.read_csv('//home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/pre_post_TU/final_data/new_final_pre_post_samples_TU_input.txt', sep='\t', index_col=0)
clinical = pd.read_csv('/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/clinical_info_gHRD/processed_info_with_originalID.txt', sep="\t", index_col="GID")

#** remove transcript-gene unmatched transcripts
TUdata = TUdata[TUdata['target_gene']!= '-']

#%% #** sample name change
sample = TUdata.columns.tolist()# %%
samples = sample.copy()
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

TUdata.columns = samples
try:
    TUdata = TUdata.drop(["SV-OV-P080-atD","SV-OV-P250-atD","SV-OV-P055-atD",\
            "SV-OV-P143-atD","SV-OV-P137-atD","SV-OV-P134-atD",\
            "SV-OV-P174-bfD","SV-OV-P164-atD"], axis=1)
except:
    

    pass

#%% #* only maintenance samples
wholesample = pd.DataFrame(TUdata.columns, columns=['sample'])
maintenance = clinical[clinical['OM/OS']=='Maintenance']
mtsample = pd.DataFrame(maintenance.index)
mtsample['GID'] = mtsample['GID'].str.replace('T','P')
mtsample['GID'] = mtsample['GID'].str.replace('F','P')
mtsample['GID'] = mtsample['GID'].str[:10]
mtsample = mtsample.drop_duplicates(subset=['GID'])
wholesample['GID'] = wholesample['sample'].str[:-4]

onlymt = pd.merge(wholesample, mtsample, on='GID', how='inner')

mtsamplelist = onlymt['sample'].tolist()

TUdata = TUdata[mtsamplelist]



# %% #** transcript filtering
tu = TUdata.copy()
# tu = tu[(tu==0).astype(int).sum(axis=1) <= 10]   # 11명 이상의 샘플에서 발현이 있는 경우
# tu = tu.T[(tu==0).sum() < tu.shape[0]*0.5].T # Abnormal samples filtering
# tu["gene"] = tu.index.str.split("-",1).str[1]



#%% ##* only transcripts with stable expression
TPMdata = pd.read_csv('/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/readcount/final_data/MW_result/MW_whole_pre-post.txt',sep='\t')
TPMdata.columns = ['gene','pval','pre_mean','post_mean']
TPMdata['D_mean'] = TPMdata['post_mean'] - TPMdata['pre_mean']
# TPMdata = TPMdata[TPMdata['pval']>0.05]
# finaldata = TPMdata.loc[(TPMdata['D_mean']> -2.803527) & (TPMdata['D_mean']<2.983549),:] #0.25~0.75 quantile 

finaldata = TPMdata[TPMdata['pval']<=0.05]

tu = tu.reset_index(level=0)
tu['gene'] = tu['gene_ENST'].str.split("-",1).str[1]
mergedtu = pd.merge(tu,finaldata, on='gene', how='inner')
mergedtu = mergedtu.set_index('gene_ENST')


tu_data = mergedtu.iloc[:,:-5]



#%% ##^ sample info = metadata
sampleinfo = pd.DataFrame(tu_data.columns, columns=['samples'])
sampleinfo['group'] = 'pre'

sampleinfo.loc[sampleinfo['samples'].str.contains("atD"),'group'] = "post"
#%% ##^ geneinfo
geneinfo = pd.DataFrame(columns=['transcript','gene'])
geneinfo["gene"] = tu_data.index.str.split("-",1).str[1]
geneinfo["transcript"] = tu_data.index
geneinfo = geneinfo.set_index("transcript", drop=False)


# %%

sampleinfo.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/variable_mt_prepostmetadata.csv',index=False)
tu_data.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/variable_mt_prepostTUdata.csv',index=True)
geneinfo.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/variable_mt_prepostgeneinfo.csv',index=True)




#############&#######################################################################################


# %%
# ########*** satuRn results ########
saturn= pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/variable_mt_prepostDTUresult.csv', sep=',')
saturn.rename(columns = {"Unnamed: 0": "transcript"}, inplace = True)
saturn['gene'] = saturn['transcript'].str.split("-",1).str[1]
saturn_pvalsorted = saturn.sort_values(by=['pval']) #empirical.pvalue-sorted

# %%
saturn_pvalsorted.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/satuRn_results/variable_mt_DTUresult.csv')

# %%
newsaturn00 = saturn_pvalsorted[saturn_pvalsorted['pval']<0.05]

# %%
