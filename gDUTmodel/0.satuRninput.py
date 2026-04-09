#%%
#! this code is.....
##1) 기존의 샘플에서 105번(=056번, Responder)을 추가하여 pre vs. post새턴을 돌리기 위한 R-stable, NR-stable, R-variable, NR-variable 데이터셋을 만들기
##1-1) 기존의 샘플에서 105번(=056번, Responder)을 추가하여 pre vs. post새턴 돌린 결과 GO enrichment test
##2) validation set(n=130)에서 R vs. NR 새턴 돌리기위한 input 만들기

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
#^ 1) 기존의 샘플에서 105번(=056번, Responder)을 추가하여 pre vs. post새턴을 돌리기 위한 R-stable, NR-stable, R-variable, NR-variable 데이터셋을 만들기

#* data download
texp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/discovery_transcript_exp.txt',  sep='\t', index_col=0)
group_info = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/clinicaldata.txt', sep="\t", index_col="GID")

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

#* 60% 이상 filtering
TUdata = TUdata.drop(['target_gene'],axis=1)
TUdata = TUdata[(TUdata>0).astype(int).sum(axis=1) > TUdata.shape[1]*0.6]

#* sample filtering: R / NR

wholesample = pd.DataFrame(TUdata.columns, columns=['sample'])

###&&&&################################################
res_int = clinical[clinical['group']==1]
###&&&&#################################################

mtsample = pd.DataFrame(res_int.index)
wholesample['GID'] = wholesample['sample'].str[:-4]

onlymt = pd.merge(wholesample, mtsample, on='GID', how='inner')

mtsamplelist = onlymt['sample'].tolist()

TUdata = TUdata[mtsamplelist]


#* gene filtering: stable / variable (only with p-value)

tu = TUdata.copy()

TPMdata = pd.read_csv('/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/readcount/final_data/MW_result/MW_whole_pre-post.txt',sep='\t')

TPMdata.columns = ['gene','pval','pre_mean','post_mean']
TPMdata['D_mean'] = TPMdata['post_mean'] - TPMdata['pre_mean']

##&######################################################################
finaldata = TPMdata[TPMdata['pval']>0.05] #^stable
#finaldata = TPMdata[TPMdata['pval']<0.05] #^variable
##&######################################################################

tu = tu.reset_index(level=0)
tu['gene'] = tu['gene_ENST'].str.split("-",1).str[1]
mergedtu = pd.merge(tu,finaldata, on='gene', how='inner')
mergedtu = mergedtu.set_index('gene_ENST')


tu_data = mergedtu.iloc[:,:-5]

#* sample info = metadata
sampleinfo = pd.DataFrame(tu_data.columns, columns=['samples'])
sampleinfo['group'] = 'pre'
sampleinfo.loc[sampleinfo['samples'].str.contains("atD"),'group'] = "post"

#* geneinfo
geneinfo = pd.DataFrame(columns=['transcript','gene'])
geneinfo["gene"] = tu_data.index.str.split("-",1).str[1]
geneinfo["transcript"] = tu_data.index
geneinfo = geneinfo.set_index("transcript", drop=False)


# %%
#* input data export
path = '/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/input/discovery/'
#sampleinfo.to_csv(path+'stable_R_metadata.csv',index=False)
#tu_data.to_csv(path +'stable_R_inputdata.csv',index=True)
#geneinfo.to_csv(path+'stable_R_geneinfo.csv',index=True)



# %%
#^ 1-1) 기존의 샘플에서 105번(=056번, Responder)을 추가하여 pre vs. post새턴 돌린 결과 enrichment test

import gseapy as gp
###*enrichR GO enrichment analysis

filelist = ['stable_NR','stable_R','variable_NR','variable_R']
i=3
path2 = '/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/output/discovery/'

results = pd.read_csv(path2+filelist[i]+'_satuRnresult.csv')

results.rename(columns = {"Unnamed: 0": "transcript"}, inplace = True)
results['gene'] = results['transcript'].str.split("-",1).str[1]

#pcut = results[results['pval']<0.05]['gene']
pcut = results[results['empirical_pval']<0.05]['gene']

pcut = pcut.drop_duplicates()
glist = pcut.squeeze().str.strip().to_list()

enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                gene_sets=['GO_Biological_Process_2021'], 
                organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                outdir=None, # don't write to disk
)

enrresult = enr.results.sort_values(by=['Adjusted P-value']) 

##* file saving
#enrresult.to_csv(path2+filelist[i]+'_GOenrichment.csv', index=False)

#%%
#* GO enrichment result check

file = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/output/discovery/stable_NR_GOenrichment.csv')
file = file[file['Adjusted P-value']<=0.1]
# %%





















#%%
##^ 20230517: R_pre vs. NR_pre 추가로 satuRn input 생성하기 

#* data download
texp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/discovery_transcript_exp.txt',  sep='\t', index_col=0)
group_info = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/clinicaldata.txt', sep="\t", index_col="GID")

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

#* 60% 이상 filtering
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

final_data = pd.concat([TU_R,TU_NR],axis=1)


TPMdata = pd.read_csv('/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/readcount/final_data/MW_result/MW_whole_pre-post.txt',sep='\t')

TPMdata.columns = ['gene','pval','pre_mean','post_mean']
TPMdata['D_mean'] = TPMdata['post_mean'] - TPMdata['pre_mean']
final_data['gene'] = final_data.index.str.split("-",1).str[1]

##&######################################################################
#stable = TPMdata[TPMdata['pval']>0.05] #^stable
variable = TPMdata[TPMdata['pval']<0.05] #^variable
##&######################################################################

final_data = final_data.reset_index(level=0)
mergedtu = pd.merge(final_data,variable, on='gene', how='inner')

mergedtu = mergedtu.set_index(['gene_ENST'])
inputdata = mergedtu.drop(columns=['gene','pval','pre_mean','post_mean','D_mean'])

geneinfo = pd.DataFrame(inputdata.index)
geneinfo.columns = ['transcript']
geneinfo['gene'] = geneinfo['transcript'].str.split("-",1).str[1]

sampleinfo = pd.DataFrame(inputdata.columns)
sampleinfo.columns = ['samples']
sampleinfo['group'] = 'R_pre'
sampleinfo.iloc[11:,1] = 'NR_pre'

#%%
#* input data export
path = '/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/input/discovery/RpreNRpre/'
sampleinfo.to_csv(path+'variable_metadata.csv',index=False)
inputdata.to_csv(path +'variable_inputdata.csv',index=True)
geneinfo.to_csv(path+'variable_geneinfo.csv',index=False)
# %%


#^
#^ 1-1) 20230517: R_pre vs. NR_pre 추가로 satuRn 돌린거 enrichment test

import gseapy as gp
###*enrichR GO enrichment analysis

filelist = ['stable','variable']
i=1
path2 = '/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/output/discovery/RpreNRpre/'

results = pd.read_csv(path2+filelist[i]+'_satuRnresult.csv')

results.rename(columns = {"Unnamed: 0": "transcript"}, inplace = True)
results['gene'] = results['transcript'].str.split("-",1).str[1]

#pcut = results[results['pval']<0.05]['gene']
pcut = results[results['empirical_pval']<0.05]['gene']
pcut = pcut.drop_duplicates()
glist = pcut.squeeze().str.strip().to_list()

enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                gene_sets=['GO_Biological_Process_2021'], 
                organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                outdir=None, # don't write to disk
)

enrresult = enr.results.sort_values(by=['Adjusted P-value']) 

##* file saving
#enrresult.to_csv(path2+filelist[i]+'_GOenrichment.csv', index=False)

#%%
#* GO enrichment result check

file = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/output/discovery/stable_NR_GOenrichment.csv')
file = file[file['Adjusted P-value']<=0.1]