#%%
#! this code is.....

"""
20230603

1. R pre vs. NR pre
2. R post vs. NR post
3. R pre vs. R post
4. NR pre vs. NR post

네 가지 경우에 대한 DESeq2을 돌려서 DEG 리스트를 쫙 뽑아놓고, 각각 stable/variable 나눠서 satuRn 다시 돌리기..!!!

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


#%%
##** make dataset for DESeq2 input to get stable / variable genes of 1~4 cases 

countdata = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/DESeq2/countdata.csv', sep = ",", index_col=0)

# clinical data
group_info = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/clinicaldata.txt', sep="\t", index_col="GID")


# clinicaldata processing (056 -> 105)
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

mTU = countdata

wholesample = pd.DataFrame(mTU.columns, columns=['sample'])
##res
res = clinical[clinical['group']==1]
mtsample = pd.DataFrame(res.index)
wholesample['GID'] = wholesample['sample'].str[:-4]
onlymt = pd.merge(wholesample, mtsample, on='GID', how='inner')
mtsamplelist = onlymt['sample'].tolist()
major_res = mTU[mtsamplelist]
##nonres
nonres = clinical[clinical['group']==0]
mtsample = pd.DataFrame(nonres.index)
wholesample['GID'] = wholesample['sample'].str[:-4]
onlymt = pd.merge(wholesample, mtsample, on='GID', how='inner')
mtsamplelist = onlymt['sample'].tolist()
major_nonres = mTU[mtsamplelist]

#then split into R_pre, R_post, NR_pre, NR_post
major_R_pre = major_res.loc[:, major_res.columns.str.contains('bfD')]
major_info_1 = pd.DataFrame(major_R_pre.columns)
major_info_1['group'] = 'R_pre'

major_R_post = major_res.loc[:, major_res.columns.str.contains('atD')]
major_info_2 = pd.DataFrame(major_R_post.columns)
major_info_2['group'] = 'R_post'

major_NR_pre = major_nonres.loc[:, major_nonres.columns.str.contains('bfD')]
major_info_3 = pd.DataFrame(major_NR_pre.columns)
major_info_3['group'] = 'NR_pre'

major_NR_post = major_nonres.loc[:, major_nonres.columns.str.contains('atD')]
major_info_4 = pd.DataFrame(major_NR_post.columns)
major_info_4['group'] = 'NR_post'

#concat 4 data
final_major = pd.concat([major_R_pre,major_R_post,major_NR_pre,major_NR_post],axis=1)
final_major_info = pd.concat([major_info_1,major_info_2,major_info_3,major_info_4])
final_major_info.columns = ['samples','group']


#^ filter out only NR pre vs. NR post (acquired control)
inputmeta = final_major_info[ (final_major_info['group'] == 'NR_pre') | (final_major_info['group'] == 'NR_post') ]
inputsamples = list(inputmeta['samples'])
inputcount = final_major[inputsamples]

# save data
# inputmeta.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/DESeq2/resistance/comp_acquired_metadata.csv',index=False)
# inputcount.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/DESeq2/resistance/comp_acquired_countdata.csv',index=True)

#^ filter out only R post vs. NR post (innate control)
inputmeta2 = final_major_info[ (final_major_info['group'] == 'R_post') | (final_major_info['group'] == 'NR_post') ]
inputsamples2 = list(inputmeta2['samples'])
inputcount2 = final_major[inputsamples2]

#save data
# inputmeta2.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/DESeq2/resistance/comp_innate_metadata.csv',index=False)
# inputcount2.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/DESeq2/resistance/comp_innate_countdata.csv',index=True)








#%%
##* satuRn input generation

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

res = clinical[clinical['group']==1]
nonres = clinical[clinical['group']==0]

rsample = pd.DataFrame(res.index)
nrsample = pd.DataFrame(nonres.index)
wholesample['GID'] = wholesample['sample'].str[:-4]

rtmp = pd.merge(wholesample, rsample, on='GID', how='inner')
nrtmp = pd.merge(wholesample, nrsample, on='GID', how='inner')

rTU = TUdata[list(rtmp['sample'])]
nrTU = TUdata[list(nrtmp['sample'])]

rTU_pre = rTU.loc[:, rTU.columns.str.contains('bfD')]
rTU_post = rTU.loc[:, rTU.columns.str.contains('atD')]
nrTU_pre = nrTU.loc[:, nrTU.columns.str.contains('bfD')]
nrTU_post = nrTU.loc[:, nrTU.columns.str.contains('atD')]

rTU_pre_info = pd.DataFrame(rTU_pre.columns)
rTU_pre_info['group'] = 'R_pre'

rTU_post_info = pd.DataFrame(rTU_post.columns)
rTU_post_info['group'] = 'R_post'

nrTU_pre_info = pd.DataFrame(nrTU_pre.columns)
nrTU_pre_info['group'] = 'NR_pre'

nrTU_post_info = pd.DataFrame(nrTU_post.columns)
nrTU_post_info['group'] = 'NR_post'


final_TU = pd.concat([rTU_pre,rTU_post,nrTU_pre,nrTU_post],axis=1)
final_TU_info = pd.concat([rTU_pre_info,rTU_post_info,nrTU_pre_info,nrTU_post_info])
final_TU_info.columns = ['sample','group']

#%%
acquired = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/DESeq2/resistance/acquired_DEG_result.csv',index_col=0)
comp_acquired = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/DESeq2/resistance/comp_acquired_DEG_result.csv',index_col=0)
innate = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/DESeq2/resistance/innate_DEG_result.csv',index_col=0)
comp_innate = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/DESeq2/resistance/comp_innate_DEG_result.csv',index_col=0)

acquired_stable = acquired[acquired['pvalue']>0.05].index.to_list()
acquired_variable = acquired[acquired['pvalue']<0.05].index.to_list()

comp_acquired_stable = comp_acquired[comp_acquired['pvalue']>0.05].index.to_list()
comp_acquired_variable = comp_acquired[comp_acquired['pvalue']<0.05].index.to_list()

innate_stable = innate[innate['pvalue']>0.05].index.to_list()
innate_variable = innate[innate['pvalue']<0.05].index.to_list()

comp_innate_stable = comp_innate[comp_innate['pvalue']>0.05].index.to_list()
comp_innate_variable = comp_innate[comp_innate['pvalue']<0.05].index.to_list()

#%%

##^ change 8 genesets into ensg ids

ensgtosymbol = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/ensgidtosymbol.dict', sep='\t', header=None)
ensgtosymbol.columns = ['id','symbol']

a_s = pd.DataFrame(acquired_stable, columns=['id'])
a_s = pd.merge(a_s, ensgtosymbol, on='id', how='inner')
a_s = list(a_s['symbol'])

ca_s = pd.DataFrame(comp_acquired_stable, columns=['id'])
ca_s = pd.merge(ca_s, ensgtosymbol, on='id', how='inner')
ca_s = list(ca_s['symbol'])

i_s = pd.DataFrame(innate_stable, columns=['id'])
i_s = pd.merge(i_s, ensgtosymbol, on='id', how='inner')
i_s = list(i_s['symbol'])

ci_s = pd.DataFrame(comp_innate_stable, columns=['id'])
ci_s = pd.merge(ci_s, ensgtosymbol, on='id', how='inner')
ci_s = list(ci_s['symbol'])

a_v = pd.DataFrame(acquired_variable, columns=['id'])
a_v = pd.merge(a_v, ensgtosymbol, on='id', how='inner')
a_v = list(a_v['symbol'])

ca_v = pd.DataFrame(comp_acquired_variable, columns=['id'])
ca_v = pd.merge(ca_v, ensgtosymbol, on='id', how='inner')
ca_v = list(ca_v['symbol'])

i_v = pd.DataFrame(innate_variable, columns=['id'])
i_v = pd.merge(i_v, ensgtosymbol, on='id', how='inner')
i_v = list(i_v['symbol'])

ci_v = pd.DataFrame(comp_innate_variable, columns=['id'])
ci_v = pd.merge(ci_v, ensgtosymbol, on='id', how='inner')
ci_v = list(ci_v['symbol'])

# %%
final_TU['gene'] = final_TU.index.str.split("-",1).str[1]
#%%

####^ save satuRn input

path = '/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/input/discovery/resistance/gDUT/'

###acquired_stable
d1_meta = final_TU_info[(final_TU_info['group'] == 'R_pre') | (final_TU_info['group'] == 'R_post') ]
d1_sample = list(d1_meta['sample'])
d1_count = final_TU[final_TU['gene'].isin(a_s)][d1_sample]
d1_geneinfo = pd.DataFrame({'transcript':d1_count.index})
d1_geneinfo['gene'] = d1_geneinfo['transcript'].str.split("-",1).str[1]

d1_count.to_csv(path+'acquired_stable_inputdata.csv',index=True)
d1_meta.to_csv(path+'acquired_stable_metadata.csv',index=False)
d1_geneinfo.to_csv(path+'acquired_stable_geneinfo.csv',index=False)

#%%
###acquired_variable
d1_meta = final_TU_info[(final_TU_info['group'] == 'R_pre') | (final_TU_info['group'] == 'R_post') ]
d1_sample = list(d1_meta['sample'])
d1_count = final_TU[final_TU['gene'].isin(a_v)][d1_sample]
d1_geneinfo = pd.DataFrame({'transcript':d1_count.index})
d1_geneinfo['gene'] = d1_geneinfo['transcript'].str.split("-",1).str[1]

d1_count.to_csv(path+'acquired_variable_inputdata.csv',index=True)
d1_meta.to_csv(path+'acquired_variable_metadata.csv',index=False)
d1_geneinfo.to_csv(path+'acquired_variable_geneinfo.csv',index=False)

#%%
###comp_acquired_variable
d1_meta = final_TU_info[(final_TU_info['group'] == 'NR_pre') | (final_TU_info['group'] == 'NR_post') ]
d1_sample = list(d1_meta['sample'])
d1_count = final_TU[final_TU['gene'].isin(ca_v)][d1_sample]
d1_geneinfo = pd.DataFrame({'transcript':d1_count.index})
d1_geneinfo['gene'] = d1_geneinfo['transcript'].str.split("-",1).str[1]

d1_count.to_csv(path+'comp_acquired_variable_inputdata.csv',index=True)
d1_meta.to_csv(path+'comp_acquired_variable_metadata.csv',index=False)
d1_geneinfo.to_csv(path+'comp_acquired_variable_geneinfo.csv',index=False)

# %%
###comp_acquired_stable
d1_meta = final_TU_info[(final_TU_info['group'] == 'NR_pre') | (final_TU_info['group'] == 'NR_post') ]
d1_sample = list(d1_meta['sample'])
d1_count = final_TU[final_TU['gene'].isin(ca_s)][d1_sample]
d1_geneinfo = pd.DataFrame({'transcript':d1_count.index})
d1_geneinfo['gene'] = d1_geneinfo['transcript'].str.split("-",1).str[1]

d1_count.to_csv(path+'comp_acquired_stable_inputdata.csv',index=True)
d1_meta.to_csv(path+'comp_acquired_stable_metadata.csv',index=False)
d1_geneinfo.to_csv(path+'comp_acquired_stable_geneinfo.csv',index=False)

# %%
###innate_stable
d1_meta = final_TU_info[(final_TU_info['group'] == 'R_pre') | (final_TU_info['group'] == 'NR_pre') ]
d1_sample = list(d1_meta['sample'])
d1_count = final_TU[final_TU['gene'].isin(i_s)][d1_sample]
d1_geneinfo = pd.DataFrame({'transcript':d1_count.index})
d1_geneinfo['gene'] = d1_geneinfo['transcript'].str.split("-",1).str[1]

d1_count.to_csv(path+'innate_stable_inputdata.csv',index=True)
d1_meta.to_csv(path+'innate_stable_metadata.csv',index=False)
d1_geneinfo.to_csv(path+'innate_stable_geneinfo.csv',index=False)

# %%
###innate_variable
d1_meta = final_TU_info[(final_TU_info['group'] == 'R_pre') | (final_TU_info['group'] == 'NR_pre') ]
d1_sample = list(d1_meta['sample'])
d1_count = final_TU[final_TU['gene'].isin(i_v)][d1_sample]
d1_geneinfo = pd.DataFrame({'transcript':d1_count.index})
d1_geneinfo['gene'] = d1_geneinfo['transcript'].str.split("-",1).str[1]

d1_count.to_csv(path+'innate_variable_inputdata.csv',index=True)
d1_meta.to_csv(path+'innate_variable_metadata.csv',index=False)
d1_geneinfo.to_csv(path+'innate_variable_geneinfo.csv',index=False)

# %%
###comp_innate_variable
d1_meta = final_TU_info[(final_TU_info['group'] == 'R_post') | (final_TU_info['group'] == 'NR_post') ]
d1_sample = list(d1_meta['sample'])
d1_count = final_TU[final_TU['gene'].isin(ci_v)][d1_sample]
d1_geneinfo = pd.DataFrame({'transcript':d1_count.index})
d1_geneinfo['gene'] = d1_geneinfo['transcript'].str.split("-",1).str[1]

d1_count.to_csv(path+'comp_innate_variable_inputdata.csv',index=True)
d1_meta.to_csv(path+'comp_innate_variable_metadata.csv',index=False)
d1_geneinfo.to_csv(path+'comp_innate_variable_geneinfo.csv',index=False)

# %%
###comp_innate_stable
d1_meta = final_TU_info[(final_TU_info['group'] == 'R_post') | (final_TU_info['group'] == 'NR_post') ]
d1_sample = list(d1_meta['sample'])
d1_count = final_TU[final_TU['gene'].isin(ci_s)][d1_sample]
d1_geneinfo = pd.DataFrame({'transcript':d1_count.index})
d1_geneinfo['gene'] = d1_geneinfo['transcript'].str.split("-",1).str[1]

d1_count.to_csv(path+'comp_innate_stable_inputdata.csv',index=True)
d1_meta.to_csv(path+'comp_innate_stable_metadata.csv',index=False)
d1_geneinfo.to_csv(path+'comp_innate_stable_geneinfo.csv',index=False)
# %%








#%%
#######^ GO enrichment test
import gseapy as gp

filelist = ['acquired_stable','acquired_variable','comp_acquired_stable','comp_acquired_variable','innate_stable','innate_variable','comp_innate_stable','comp_innate_variable']

for i in range(len(filelist)):
        
    path2 = '/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/output/discovery/resistance/'

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
    print(filelist[i] , len(pcut))
    ##* file saving
    #enrresult.to_csv(path2+filelist[i]+'_GOenrichment.csv', index=False)
# %%
