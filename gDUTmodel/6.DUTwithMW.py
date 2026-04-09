#%%
#! this code is.....
## 20230518
"""
1) R pre vs. NR pre: stable / variable groups MW로 DUT 찾기
2) GO enrichment 돌려서 5.DUT_GObubbleplot.py 돌릴 수 있는 인풋 만들기
"""

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
import matplotlib.cm as cm
from matplotlib.pyplot import gcf

# %%
#^ 1) R pre vs. NR pre: stable / variable groups MW로 DUT 찾기


#* data download
minor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/discovery_minorTU.txt',  sep='\t', index_col=0)
major = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/discovery_majorTU.txt',  sep='\t', index_col=0)
group_info = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/clinicaldata.txt', sep="\t", index_col="GID")

wholeTU = pd.concat([minor,major])

texp = wholeTU
#* clinicaldata processing (056 -> 105)
group_info = group_info[["OM/OS", "ORR", "drug", "interval"]]
group_info.columns = ["OM/OS", "group", "drug", "interval"]
group_info["drug"] = group_info["drug"].str.replace("Olapairb","Olaparib")
group_info["drug"] = group_info["drug"].str.replace("Olapairib","Olaparib")
group_info["drug"] = group_info["drug"].str.replace("olaparib","Olaparib")
group_info["GID"] = group_info.index.str.replace("F","T").str[:10]
group_info["GID"] = group_info["GID"].str.replace("T","P")
group_info["GID"] = group_info["GID"].str.replace("SV-OV-P056", "SV-OV-P105")
group_info = group_info.dropna()
group_info = group_info.drop("drug", axis=1)
group_info = group_info.drop_duplicates()
group_info = group_info.set_index("GID")

clinical = group_info

#* remove transcript-gene unmatched transcripts
texp['target_gene'] = texp.index.str.split("-",1).str[1]
TUdata = texp[texp['target_gene']!= '-']

#* sample name change
sample = TUdata.columns.tolist()# %%
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

TUdata.columns = samples
try:
    TUdata = TUdata.drop(["SV-OV-P080-atD","SV-OV-P250-atD","SV-OV-P055-atD",\
            "SV-OV-P143-atD","SV-OV-P137-atD","SV-OV-P134-atD",\
            "SV-OV-P174-bfD","SV-OV-P164-atD"], axis=1)
except:
    pass

#* filter with 60%
TUdata = TUdata.drop(['target_gene'],axis=1)
TUdata = TUdata[(TUdata>0).astype(int).sum(axis=1) > TUdata.shape[1]*0.6]

#* TU_R/TU_NR 나누기
wholesample = pd.DataFrame(TUdata.columns, columns=['sample'])
res_int = clinical[clinical['group']==1]

mtsample = pd.DataFrame(res_int.index)
wholesample['GID'] = wholesample['sample'].str[:-4]

onlymt = pd.merge(wholesample, mtsample, on='GID', how='inner')

mtsamplelist = onlymt['sample'].tolist()

TU_R = TUdata[mtsamplelist]

nonres = clinical[clinical['group']==0]

nrsample = pd.DataFrame(nonres.index)
wholesample['GID'] = wholesample['sample'].str[:-4]

onlynr = pd.merge(wholesample, nrsample, on='GID', how='inner')

nrsamplelist = onlynr['sample'].tolist()

TU_NR = TUdata[nrsamplelist]

TU_R = TU_R.filter(regex='-bfD')
TU_NR = TU_NR.filter(regex='-bfD')



TPMdata = pd.read_csv('/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/readcount/final_data/MW_result/MW_whole_pre-post.txt',sep='\t')

TPMdata.columns = ['gene','pval','pre_mean','post_mean']
TPMdata['D_mean'] = TPMdata['post_mean'] - TPMdata['pre_mean']
TU_R['gene'] = TU_R.index.str.split("-",1).str[1]
TU_NR['gene'] = TU_NR.index.str.split("-",1).str[1]

##&######################################################################
stable = TPMdata[TPMdata['pval']>0.05] #^stable
#variable = TPMdata[TPMdata['pval']<0.05] #^variable
##&######################################################################

TU_R = TU_R.reset_index(level=0) #R_pre
TU_NR = TU_NR.reset_index(level=0) #NR_pre

merged_R = pd.merge(TU_R,stable, on='gene', how='inner')
merged_R = merged_R.set_index(['gene_ENST'])
merged_R = merged_R.drop(columns=['gene','pval','pre_mean','post_mean','D_mean'])

merged_NR = pd.merge(TU_NR,stable, on='gene', how='inner')
merged_NR = merged_NR.set_index(['gene_ENST'])
merged_NR = merged_NR.drop(columns=['gene','pval','pre_mean','post_mean','D_mean'])


#* mannwhitney

from scipy.stats import mannwhitneyu

num = merged_R.shape[0]
mw_test = pd.DataFrame()
mw_test['gene_ENST'] = merged_R.index.to_list()
mw_test['p-val'] = 1

for i in range(num):
    list_a = merged_R.iloc[i,:]
    list_b = merged_NR.iloc[i,:]
    if all(item == 0 for item in list_a) or all(item == 0 for item in list_b) :
        mw_test.iloc[i,1] = 1
    else:
        mw_test.iloc[i,1] = mannwhitneyu(list_a, list_b)[1]
    # print(mannwhitneyu(list_a, list_b)[1])

final = mw_test[mw_test['p-val']<0.05]
# %%
final.to_csv("/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/MW/stable_RpreNRpreMWDUT.csv",index=False)
# %%















#%%
#^ 2) GO enrichment 돌려서 5.DUT_GObubbleplot.py 돌릴 수 있는 인풋 만들기
import gseapy as gp

results = pd.read_csv("/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/MW/variable_RpreNRpreMWDUT.csv")

results['gene'] = results['gene_ENST'].str.split("-",1).str[1]

pcut = results[results['p-val']<0.05]['gene']
pcut = pcut.drop_duplicates()
glist = pcut.squeeze().str.strip().to_list()

enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                gene_sets=['GO_Biological_Process_2021'], 
                organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                outdir=None, # don't write to disk
)

enrresult = enr.results.sort_values(by=['Adjusted P-value']) 

# %%
##* file saving
enrresult.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/MW/variable_RpreNRpreMWDUT_GOenrichment.csv', index=False)

# %%
