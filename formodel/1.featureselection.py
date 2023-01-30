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
TUdata = pd.read_csv('/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/pre_post_TU/final_data/whole_TU.txt', sep='\t', index_col=0)

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
#%%
sampleinfo = pd.DataFrame(TUdata.columns, columns=['samples'])
sampleinfo['group'] = 'pre'

sampleinfo.loc[sampleinfo['samples'].str.contains("atD"),'group'] = "post"

# %% 
# #** sample filtering
tu = TUdata.copy()
tu = tu[(tu==0).astype(int).sum(axis=1) <= 10]   # 90% 이상의 샘플에서 발현이 있는 경우
tu = tu.T[(tu==0).sum() < tu.shape[0]*0.5].T # Abnormal samples filtering

#%%
geneinfo = pd.DataFrame(columns=['transcript','gene'])
geneinfo["gene"] = tu.index.str.split("-",1).str[1]
geneinfo["transcript"] = tu.index
geneinfo = geneinfo.set_index("transcript", drop=False)
# %%

sampleinfo.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/prepostmetadata.csv',index=False)
tu.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/prepostTUdata.csv',index=True)

# %%
geneinfo.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/prepostgeneinfo.csv',index=True)


# %%
########*** satuRn results ########
saturn= pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/prepostDTUresult.csv', sep=',')
saturn.rename(columns = {"Unnamed: 0": "transcript"}, inplace = True)
saturn['gene'] = saturn['transcript'].str.split("-",1).str[1]
saturn_pvalsorted = saturn.sort_values(by=['pval']) #empirical.pvalue-sorted

saturn_pvalsorted.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/whole_DTU_result.csv',index=False)

# %%
########** vegf/repair-related_gene #####
repair = pd.read_csv(
    "/home/hyeongu/DATA5/hyeongu/GSEA/gmt/merged_GO_NER.txt",
    sep="\t", header=None
)
vegf = pd.read_csv(
    "/home/hyeongu/DATA5/hyeongu/GSEA/gmt/vegf_repair-related_fgf.txt",
    sep="\t", header=None
)
repair.columns=['gene']
vegf.columns=['gene']
geneterms = pd.merge(repair, vegf, on='gene', how='outer')

#%%

saturn_pval_genesorted = pd.merge(saturn_pvalsorted, geneterms, how='inner', on='gene')

#%%
saturn_005 = saturn_pval_genesorted[saturn_pval_genesorted['empirical_pval']<0.05]



# %%
# saturn_005.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/filtered_DTU_result.csv',index=False)

# %%
