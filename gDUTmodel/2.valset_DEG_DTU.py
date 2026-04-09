#%%
#! this code is.....
##1) validation set(n=130)에서 28 transcripts: DTU MW with TU file
##1-1) 1번 결과 boxplot?
##2) validation set(n=130)에서 23 gene: DEG MW with gene expression file
##3) make SatuRn input file with 28 transcripts
##4) make DeSeq2 input file with 23 genes
""
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
from scipy.stats import mannwhitneyu

#%%
#^ 1) validation set(n=130)에서 28 transcripts: DTU MW with TU file
TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/validation_TU.txt',sep='\t')
TU.rename(columns = {"Unnamed: 0": "transcript_id"}, inplace = True)

#*final feature list (from TOP FI)
f_list = ['ENST00000416441.2-TAOK2','MSTRG.67243.76-CHEK2','MSTRG.77497.409-ATRIP','MSTRG.21485.29-RAD52','MSTRG.77574.12-ATRIP','MSTRG.36711.1-HERC2','MSTRG.110686.21-LYN','MSTRG.49834.288-BRIP1','MSTRG.58505.24-NABP1','MSTRG.97014.1-ASCC3','MSTRG.7347.9-KDM4A','MSTRG.15065.1-MGMT','MSTRG.18408.16-DDB1','ENST00000521492.1-POLB','MSTRG.47598.2-RBBP8','MSTRG.12799.1-ZNF365','MSTRG.94452.1-KDM1B','ENST00000344836.4-USP7','MSTRG.49509.1-RECQL5','ENST00000409193.1-TSN','MSTRG.94444.7-KDM1B','MSTRG.118485.21-INIP','MSTRG.34618.2-NRDE2','MSTRG.42895.315-TAOK2','MSTRG.110699.1-LYN','MSTRG.92444.181-ERCC2','MSTRG.15059.1-MGMT','MSTRG.15443.1-MMS19']
f_list = pd.DataFrame(f_list)
f_list.columns = ['transcript_id']

f_tu = pd.merge(f_list,TU, on='transcript_id')

sample = f_tu.columns
sample = sample.str.replace('-T','-P')
f_tu.columns = sample

clinical = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/modified_clinicaldata.txt', index_col="GID")
res = clinical[clinical['group']==1]
nonres = clinical[clinical['group']==0]

res = list(res.index)
nonres = list(nonres.index)

f_tu = f_tu.set_index('transcript_id')

tu_res = f_tu[np.intersect1d(f_tu.columns, res)]
tu_nonres = f_tu[np.intersect1d(f_tu.columns, nonres)]
tu_res = tu_res.astype(float)
tu_nonres = tu_nonres.astype(float)


mw = pd.DataFrame()
mw.index = f_tu.index
mw['pval'] = 5

for i in range(f_tu.shape[0]):
    list_res = np.float_(list(tu_res.iloc[i,:]))
    list_nonres = np.float_(list(tu_nonres.iloc[i,:]))
    mw.iloc[i,0] = mannwhitneyu(list_res, list_nonres, alternative='less')[1]
    

#mw.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/modelinterpretation/28features_response_dtu.csv', index=True)

mod_mw = mw
mod_mw['pval'] = -1*(np.log10(mod_mw['pval']))

#* make df for fig generation
tu_res['type'] = 'responder'
tu_nonres['type'] = 'non-responder'

res_forfig = tu_res[['type']]
nonres_forfig = tu_nonres[['type']]
res_forfig['mean'] = tu_res.mean(axis=1)
nonres_forfig['mean'] = tu_nonres.mean(axis=1)
res_forfig['median'] = tu_res.median(axis=1)
nonres_forfig['median'] = tu_nonres.median(axis=1)

forfig = pd.concat([res_forfig,nonres_forfig])
forfig['transcript_id'] = forfig.index

#%%
forfig['mean_compare'] = 'low'
forfig['median_compare'] = 'low'
for i in range(28):
    if forfig.iloc[i,1] < forfig.iloc[i+28,1]:
        forfig.iloc[i,4] = 'high'
    if forfig.iloc[i,2] < forfig.iloc[i+28,2]:
        forfig.iloc[i,5] = 'high'
        
        
# %%
#^ 1-1) barplot! r vs. nr median TU
df = forfig
col = {"responder":"#559EE7", "non-responder":"#D5DBE0"}
sns.set(rc = {'figure.figsize':(8,6)}, style='whitegrid')
g = sns.catplot(x="median", y="transcript_id", hue="type", kind="bar", data=df, height=8, aspect=1, orient="horizontal", palette=col)

# set the x-axis label and title
plt.xlabel('Median TU')
plt.title('Responder vs. Non-Responder Median TU')

tmp = forfig.iloc[:28,:]
high_non_responder_median = tmp[tmp['median_compare']=='low']['transcript_id']
                                
ylabels = g.ax.get_yticklabels()

for label in ylabels:
    transcript_id = label.get_text()
    if transcript_id in high_non_responder_median:
        label.set_color('red')
plt.show()


# %%#
##^2) validation set(n=130)에서 23 gene: DEG MW with gene expression file

forfig['gene'] = forfig['transcript_id'].str.split('-',1).str[1]
genelist = list(set(forfig['gene']))

geneexp = pd.read_csv('/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/readcount/230405_62genes_readcounts.txt',sep='\t', index_col=0)
exp = geneexp.loc[genelist]
sample2 = exp.columns
sample2 = sample2.str.replace('-T','-P')
exp.columns = sample2
# %%
exp_res = exp[np.intersect1d(exp.columns, res)]
exp_nonres = exp[np.intersect1d(exp.columns, nonres)]
exp_res = exp_res.astype(float)
exp_nonres = exp_nonres.astype(float)
exp_res['type'] = 'responder'
exp_nonres['type'] = 'non-responder'

res_forfig_exp = exp_res[['type']]
nonres_forfig_exp = exp_nonres[['type']]
res_forfig_exp['mean'] = exp_res.mean(axis=1)
nonres_forfig_exp['mean'] = exp_nonres.mean(axis=1)
res_forfig_exp['median'] = exp_res.median(axis=1)
nonres_forfig_exp['median'] = exp_nonres.median(axis=1)

forfig_exp = pd.concat([res_forfig_exp,nonres_forfig_exp])
forfig_exp['gene'] = forfig_exp.index

forfig_exp['mean_compare'] = 'low'
forfig_exp['median_compare'] = 'low'
for i in range(23):
    if forfig_exp.iloc[i,1] < forfig.iloc[i+23,1]:
        forfig.iloc[i,4] = 'high'
    if forfig.iloc[i,2] < forfig.iloc[i+23,2]:
        forfig.iloc[i,5] = 'high'
# %%
#^ 2-1) barplot! r vs. nr median gene exp

df = forfig_exp
col = {"responder":"#559EE7", "non-responder":"#D5DBE0"}
sns.set(rc = {'figure.figsize':(8,6)}, style='whitegrid')
g = sns.catplot(x="mean", y="gene", hue="type", kind="bar", data=df, height=8, aspect=1, orient="horizontal", palette=col)

# set the x-axis label and title
plt.xlabel('Mean Expression')
plt.title('Responder vs. Non-Responder Mean Gene Exp')

tmp = forfig_exp.iloc[:23,:]
high_non_responder_median = tmp[tmp['mean_compare']=='low']['gene']
                                
ylabels = g.ax.get_yticklabels()

for label in ylabels:
    transcript_id = label.get_text()
    if transcript_id in high_non_responder_median:
        label.set_color('red')
plt.show()
# %%
