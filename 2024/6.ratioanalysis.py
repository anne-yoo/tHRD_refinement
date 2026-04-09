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
ARdut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t')
IRdut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/nonresponder_stable_DUT_Wilcoxon.txt', sep='\t')
ratio = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/majorminor/major2minorratio.txt', sep='\t')
sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_TU.txt', sep='\t')

major = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
#%%
majorlist = major[major['type']=='major']
# %%
responder = sampleinfo.loc[(sampleinfo['response']==1),'sample_full'].to_list()
nonresponder = sampleinfo.loc[(sampleinfo['response']==0),'sample_full'].to_list()

###* AR ####
gDUTlist = list(set(ARdut.loc[(ARdut['p_value']<0.05) & (ARdut['log2FC']>1.5), 'Gene Symbol']))
DUTlist = list(ARdut.loc[(ARdut['p_value']<0.05) & (ARdut['log2FC']>1.5), 'gene_ENST'])

###################################################################
#enr = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUT/responder_stable_DUT_GOenrichment.txt', sep='\t')
enr = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUT/AR_stable_distDUT_GOenrichment.txt', sep='\t')
###################################################################

enr = enr[(enr["Term"].str.contains("repair", case=False)) | (enr["Term"].str.contains("DNA Damage", case=False))]
enr = enr[enr["Adjusted P-value"]<=0.01]
genelist = [gene for sublist in enr['Genes'].str.split(';') for gene in sublist]
genelist = list(set(genelist))
enrmajorlist = majorlist[majorlist['genename'].isin(genelist)]['gene_ENST']
enrmajordutlist = set(enrmajorlist).intersection(set(DUTlist))

filtered_TU = TU[TU['gene_ENST'].isin(enrmajordutlist)]
#filtered_TU.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/model/enrmajordutTU.txt', sep='\t', index=False)
#%%
###* IR ####
gDUTlist = list(set(IRdut.loc[(IRdut['p_value']<0.05) & (IRdut['log2FC']>1.5), 'Gene Symbol']))
DUTlist = list(ARdut.loc[(IRdut['p_value']<0.05) & (IRdut['log2FC']>1.5), 'gene_ENST'])
enr = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202401_analysis/DUT/nonresponder_stable_Wilcoxon_GOenrichment_FC_15.txt', sep='\t')
#enr = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUT/IR_stable_distDUT_GOenrichment.txt', sep='\t')

enr = enr[(enr["Term"].str.contains("repair", case=False)) | (enr["Term"].str.contains("DNA Damage", case=False))]
enr = enr[enr["Adjusted P-value"]<=0.01]
genelist = [gene for sublist in enr['Genes'].str.split(';') for gene in sublist]
genelist = list(set(genelist))
enrmajorlist = majorlist[majorlist['genename'].isin(genelist)]['gene_ENST']
enrmajordutlist = set(enrmajorlist).intersection(set(DUTlist))

#%%
###^^^ Major Sum #########################
new_df = TU.set_index(['gene_ENST'])
###################### every major in gDUT / only major DUT in gDUT ###############
new_df = new_df[new_df.index.isin(enrmajordutlist)]
###################################################################################

####################responder / nonresponder############################
new_df = new_df[nonresponder]
########################################################################

## summation of majot TUs in same gene ##
new_df['gene'] = new_df.index.str.split("-",1).str[1]
new_df = new_df.reset_index()
new_df = new_df.groupby('gene').sum()
new_df = new_df.reset_index().set_index('gene')

df_bfD = new_df[[col for col in new_df.columns if '-bfD' in col]]
df_atD = new_df[[col for col in new_df.columns if '-atD' in col]]

mean_bfD = list(df_bfD.mean(axis=1))
mean_atD = list(df_atD.mean(axis=1))
meandict = {'pre': mean_bfD, 'post': mean_atD}
meandf = pd.DataFrame(meandict)
meandf.index = new_df.index

#%%
plt.figure(figsize=(4,6))
sns.set_style("white")

col1 = ["#70A8F2","#F36464"]

tmplist = meandf.index.to_list()
tmplist = tmplist*2

##
figureinput = meandf.melt(var_name='Treatment', value_name='TU')
figureinput['gene'] = tmplist
##



ax = sns.boxplot(x="Treatment", y="TU", data=figureinput,  palette=col1, showfliers=False)
plt.ylim(0,0.16)

from statannot import add_stat_annotation

add_stat_annotation(ax, data=figureinput, x='Treatment', y='TU',
            box_pairs=[("pre", "post")],
            test='Wilcoxon', text_format='simple', loc='outside', #comparisons_correction=None,
            )
#plt.figtext(0.5, 1.005, namelist2[i] + ' - ' + typelist[j]+' transcript exp', ha='center', va='center', fontsize=13)
#plt.title(namelist2[i] + ' - ' + typelist[j]+' transcript exp', fontsize=13, y=3)
medians = figureinput.groupby(['Treatment'])['TU'].median()
medians = medians.iloc[::-1]
medians = medians.round(decimals=5)
vertical_offset = figureinput['TU'].median() * 0.8

for xtick in ax.get_xticks():
    print(xtick)
    ax.text(xtick, medians[xtick] + vertical_offset, medians[xtick], 
            horizontalalignment='center',size='medium',color='w',weight='semibold')
    
sns.despine()















#%%
###^^^ gDUT gene expression #########
exp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM_symbol.txt', sep='\t')
exp = exp.set_index(['Gene Symbol'])
exp = exp.loc[exp.index.isin(genelist),:]
####################responder / nonresponder############################
new_df = exp[responder]
########################################################################

## summation of majot TUs in same gene ##


df_bfD = new_df[[col for col in new_df.columns if '-bfD' in col]]
df_atD = new_df[[col for col in new_df.columns if '-atD' in col]]

mean_bfD = list(df_bfD.mean(axis=1))
mean_atD = list(df_atD.mean(axis=1))
meandict = {'pre': mean_bfD, 'post': mean_atD}
meandf = pd.DataFrame(meandict)
meandf.index = new_df.index


plt.figure(figsize=(4,6))
sns.set_style("white")

col1 = ["#70A8F2","#F36464"]

tmplist = meandf.index.to_list()
tmplist = tmplist*2

##
figureinput = meandf.melt(var_name='Treatment', value_name='TPM')
figureinput['gene'] = tmplist
##



ax = sns.boxplot(x="Treatment", y="TPM", data=figureinput,  palette=col1, showfliers=False)
plt.ylim(0,150)

from statannot import add_stat_annotation

add_stat_annotation(ax, data=figureinput, x='Treatment', y='TPM',
            box_pairs=[("pre", "post")],
            test='Wilcoxon', text_format='simple', loc='outside', #comparisons_correction=None,
            )
#plt.figtext(0.5, 1.005, namelist2[i] + ' - ' + typelist[j]+' transcript exp', ha='center', va='center', fontsize=13)
#plt.title(namelist2[i] + ' - ' + typelist[j]+' transcript exp', fontsize=13, y=3)
medians = figureinput.groupby(['Treatment'])['TPM'].median()
medians = medians.iloc[::-1]
medians = medians.round(decimals=5)
vertical_offset = figureinput['TPM'].median() * 0.2

for xtick in ax.get_xticks():
    print(xtick)
    ax.text(xtick, medians[xtick] + vertical_offset, medians[xtick], 
            horizontalalignment='center',size='medium',color='w',weight='semibold')
    
sns.despine()
#%%
















#%%
###^^^ Functionality Ratio ###########


new_df = ratio.set_index(['gene'])
new_df = new_df[new_df.index.isin(genelist)]
####################responder / nonresponder############################
new_df = new_df[responder]
########################################################################

df_bfD = new_df[[col for col in new_df.columns if '-bfD' in col]]
df_atD = new_df[[col for col in new_df.columns if '-atD' in col]]

mean_bfD = list(df_bfD.mean(axis=1))
mean_atD = list(df_atD.mean(axis=1))
meandict = {'pre': mean_bfD, 'post': mean_atD}
meandf = pd.DataFrame(meandict)
meandf.index = new_df.index

plt.figure(figsize=(4,6))
sns.set_style("white")

col1 = ["#70A8F2","#F36464"]

tmplist = meandf.index.to_list()
tmplist = tmplist*2

##
figureinput = meandf.melt(var_name='Treatment', value_name='ratio')
figureinput['gene'] = tmplist
##


ax = sns.boxplot(x="Treatment", y="ratio", data=figureinput,  palette=col1, showfliers=False)
plt.ylim(0,1.65)

from statannot import add_stat_annotation

add_stat_annotation(ax, data=figureinput, x='Treatment', y='ratio',
            box_pairs=[("pre", "post")],
            test='Wilcoxon', text_format='simple', loc='outside', #comparisons_correction=None,
            )
#plt.figtext(0.5, 1.005, namelist2[i] + ' - ' + typelist[j]+' transcript exp', ha='center', va='center', fontsize=13)
#plt.title(namelist2[i] + ' - ' + typelist[j]+' transcript exp', fontsize=13, y=3)
medians = figureinput.groupby(['Treatment'])['ratio'].median()
medians = medians.iloc[::-1]
medians = medians.round(decimals=5)
vertical_offset = figureinput['ratio'].median() * 0.3

for xtick in ax.get_xticks():
    print(xtick)
    ax.text(xtick, medians[xtick] + vertical_offset, medians[xtick], 
            horizontalalignment='center',size='medium',color='w',weight='semibold')
    
sns.despine()

#%%
for p in figureinput['gene'].unique():
    subset = figureinput[figureinput['gene']==p]
    x = [list(subset['Treatment'])[1], list(subset['Treatment'])[0]]
    y = [list(subset['ratio'])[1], list(subset['ratio'])[0]]
    plt.plot(x,y, marker="o", markersize=4, color='grey', linestyle="--", linewidth=0.7)
plt.title('xx')

# %%
