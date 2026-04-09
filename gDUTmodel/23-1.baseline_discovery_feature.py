#%%
#! this code is.....
"""
<Discovery Model initial features>
20231004
1. FDR < 0.1 DNA repair gene의 모든 major / minor transcript 비교 (mean/median)
2. FDR < 0.1 DNA repair gene 중 stable DUT만 major / minor transcript 비교  (mean/median)

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
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import plotly.express as px
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.stats.multitest import multipletests


sns.set(font = 'Arial')
sns.set_style("whitegrid")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# %%
dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/baseline/stable_DUT_MW.txt', sep='\t')
enr = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_Wilcoxon_GOenrichment.txt', sep='\t')

dut.columns = ['transcript','pval','Gene Symbol']
dut = dut[dut['pval']<0.05]
dutlist = dut['transcript'].tolist()

transexp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt',sep='\t', index_col=0)
#filtered_trans = transexp.loc[(transexp > 0).sum(axis=1) >= transexp.shape[1]*0.3]
transexp['Gene Symbol'] = transexp.index.str.split("-",1).str[1]
transexp = transexp[transexp['Gene Symbol']!='-']

data = enr
data = data[data["Adjusted P-value"] <= 0.01]
# data = data[(data["Term"].str.contains("repair")) | (data["Term"].str.contains("DNA damage")) 
#                 | (data["Term"].str.contains("DNA metabolic")) | (data["Term"].str.contains("(GO:0006260)"))
#                 | (data["Term"].str.contains("DNA duplex unwinding"))]

#data = data[(data["Term"].str.contains("repair", case=False)) | (data["Term"].str.contains("DNA Damage", case=False))]
data = data[(data["Term"].str.contains("repair", case=False))]

dnarepairgenes = [gene for sublist in data['Genes'].str.split(';') for gene in sublist]
dnarepairgenes = set(dnarepairgenes)

#%%

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.csv', sep=',')
transexp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt',sep='\t', index_col=0)
#filtered_trans = transexp.loc[(transexp > 0).sum(axis=1) >= transexp.shape[1]*0.3]
transexp['Gene Symbol'] = transexp.index.str.split("-",1).str[1]
transexp = transexp[transexp['Gene Symbol']!='-']
repairexp = transexp[transexp['Gene Symbol'].isin(dnarepairgenes)]

dut_TU = repairexp[repairexp.index.isin(dutlist)]
df_dut = dut_TU.iloc[:,:-1]

responder = sampleinfo[sampleinfo['response']==1]['sample_full']
nonresponder = sampleinfo[sampleinfo['response']==0]['sample_full']
responder = responder.to_list()
nonresponder = nonresponder.to_list()

R_df_dut = df_dut[responder]
NR_df_dut = df_dut[nonresponder]

R_df_pre = R_df_dut[R_df_dut.columns[1::2]]
NR_df_pre = NR_df_dut[NR_df_dut.columns[1::2]]


major = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/final_202306_discovery_major_TU.txt', sep='\t', index_col=0)
minor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/final_202306_discovery_minor_TU.txt', sep='\t', index_col=0)

majortrans = major.index.tolist()
minortrans = minor.index.tolist()

R_df_pre['type'] = R_df_pre.index.to_series().apply(lambda x: 'major' if x in majortrans else ('minor' if x in minortrans else None))
# R_df_whole['type'] = R_df_whole.index.to_series().apply(lambda x: 'major' if x in majortrans else ('minor' if x in minortrans else None))
NR_df_pre['type'] = NR_df_pre.index.to_series().apply(lambda x: 'major' if x in majortrans else ('minor' if x in minortrans else None))
# NR_df_whole['type'] = NR_df_whole.index.to_series().apply(lambda x: 'major' if x in majortrans else ('minor' if x in minortrans else None))





#%%
################^^^^^^^^ R_df_dut ##############

filelist = [R_df_pre, NR_df_pre]
namelist = ['R_df_pre','NR_df_pre']
namelist2 = ['Responder(pre)','Nonresponder(pre)']
typelist = ['major','minor']


#####^^^ R pre vs. NR pre
df_bfD = R_df_pre[R_df_pre['type']=='major']
df_atD = NR_df_pre[NR_df_pre['type']=='major']

mean_bfD = list(df_bfD.mean(axis=1))
mean_atD = list(df_atD.mean(axis=1))
meandict = {'pre': mean_bfD, 'post': mean_atD}
meandf = pd.DataFrame(meandict)
meandf.index = df_bfD.index


plt.figure(figsize=(4,6))
sns.set_style("white")

col1 = ["#9AE584","#FFDF8F"]

##
figureinput = meandf.melt(var_name='Treatment', value_name='Usage')
##


plt.ylim(0,0.35)
ax = sns.boxplot(x="Treatment", y="Usage", data=figureinput,  palette=col1, showfliers=False)
sns.stripplot(x='Treatment', y='Usage', data=figureinput, palette=col1, s=9)

from statannot import add_stat_annotation

add_stat_annotation(ax, data=figureinput, x='Treatment', y='Usage',
            box_pairs=[("pre", "post")],
            test='Wilcoxon',  text_format='simple', loc='outside')
ax.set_xticklabels(['R pre','NR pre'])
plt.figtext(0.5, 1.005, namelist2[i] + ' - ' + typelist[0]+' transcript exp', ha='center', va='center', fontsize=13)
#plt.title(namelist2[i] + ' - ' + typelist[j]+' transcript exp', fontsize=13, y=3)
sns.despine()

#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/boxplot/baseline/EXP_mean_"+namelist[i]+"_"+typelist[j]+".pdf", bbox_inches="tight")
plt.show()



# %%
####^ count major transcripts which usage went up after PARPi treatment: R vs. NR ####

count_list = []

R_treatment = df_bfD.iloc[:,:-1]
NR_treatment = df_atD.iloc[:,:-1]

for i in range(0,R_treatment.shape[1]):
    R_t = R_treatment.iloc[:,i]
    NR_t = NR_treatment.iloc[:,i]
    
    count = sum(NR_t > R_t)
    count_list.append(count)
    
count_df = pd.DataFrame({'sample': 'sample',
                        'Count': count_list})

# Create the swarm plot
plt.figure(figsize=(4,6))

sns.set(font_scale=1.1)
sns.set_style("whitegrid")
col2 = ["#2B9C09","#F2BA1F"]
col1 = ["#9AE584","#FFDF8F"]
ax = sns.boxplot(x='sample', y='Count', data=count_df, palette=col1)
sns.stripplot(x='sample', y='Count', data=count_df, palette=col2, s=9)
plt.xlabel('')
plt.ylabel('# increased in post (major TU)', fontsize=13)

#add_stat_annotation(ax, data=count_df, x='sample', y='Count',
                    # box_pairs=[("Responder", "Non-Responder")],
                    # test='Mann-Whitney',  text_format='simple', loc='outside')

#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/boxplot/001_increasedmajorTU.pdf", bbox_inches="tight")


plt.show()







# %%
