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
rawdata = pd.read_csv('/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/pre_post_TU/final_data/230106_new_final_pre_post_samples_TU_input.txt', sep='\t', index_col=0)
group_info = pd.read_csv('/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/clinical_info_gHRD/processed_info_with_originalID.txt', sep="\t", index_col="GID")

group_info = group_info[["OM/OS", "ORR", "drug", "interval"]]
group_info.columns = ["OM/OS", "group", "drug", "interval"]
group_info["drug"] = group_info["drug"].str.replace("Olapairb","Olaparib")
group_info["drug"] = group_info["drug"].str.replace("Olapairib","Olaparib")
group_info["drug"] = group_info["drug"].str.replace("olaparib","Olaparib")
group_info = group_info.dropna()
group_info = group_info.drop("drug", axis=1)
group_info["GID"] = group_info.index.str.replace("F","T").str[:10]
group_info["GID"] = group_info["GID"].str.replace("T","P")
group_info = group_info.drop_duplicates()
group_info = group_info.set_index("GID")

clinical = group_info

#** remove transcript-gene unmatched transcripts
TUdata = rawdata[rawdata['target_gene']!= '-']

#** sample name change
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



TUdata = TUdata.drop(['target_gene'],axis=1)

TUdata = TUdata[(TUdata>0).astype(int).sum(axis=1) > TUdata.shape[1]*0.8]

#%%

#* sample filtering
wholesample = pd.DataFrame(TUdata.columns, columns=['sample'])

###&&&&################################################
res_int = clinical[clinical['group']==1]
###&&&&#################################################

mtsample = pd.DataFrame(res_int.index)
wholesample['GID'] = wholesample['sample'].str[:-4]

onlymt = pd.merge(wholesample, mtsample, on='GID', how='inner')

mtsamplelist = onlymt['sample'].tolist()

TUdata = TUdata[mtsamplelist]


##* only transcripts with stable expression
tu = TUdata.copy()

##&#####################################################################
TPMdata = pd.read_csv('/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/readcount/final_data/MW_result/MW_whole_pre-post.txt',sep='\t')
##&#####################################################################

TPMdata.columns = ['gene','pval','pre_mean','post_mean']
TPMdata['D_mean'] = TPMdata['post_mean'] - TPMdata['pre_mean']

##&######################################################################
# finaldata = TPMdata[TPMdata['pval']>0.05] #^stable
# finaldata = TPMdata.loc[(abs(TPMdata['D_mean'])<abs(TPMdata['D_mean']).quantile(0.75)),:] 

finaldata = TPMdata[TPMdata['pval']<0.05] #^variable
# finaldata = TPMdata.loc[(abs(TPMdata['D_mean'])>abs(TPMdata['D_mean']).quantile(0.25)) ,:] 

##&######################################################################

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

sampleinfo.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/satuRn_inputs/new/variable_R_metadata.csv',index=False)
tu_data.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/satuRn_inputs/new/variable_R_TUdata.csv',index=True)
geneinfo.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/satuRn_inputs/new/variable_R_geneinfo.csv',index=True)




#############&#######################################################################################


# %%
# ########*** satuRn results ########
saturn= pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/satuRn_results/newtmp/stable_NR_DTUresult.csv', sep=',')
saturn.rename(columns = {"Unnamed: 0": "transcript"}, inplace = True)
saturn['gene'] = saturn['transcript'].str.split("-",1).str[1]
saturn_pvalsorted = saturn.sort_values(by=['pval']) 

saturn_pvalsorted.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/satuRn_results/new/stable_NR_DTUresult.csv', index=False)

# %%
newsaturn00 = saturn_pvalsorted[saturn_pvalsorted['pval']<0.05]

# %%
