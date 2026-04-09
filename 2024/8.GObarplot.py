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
import plotly.express as px
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.stats.multitest import multipletests
import textwrap


sns.set(font = 'Arial')
sns.set_style("whitegrid")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#%%
##### DO enrichR ! #######
import gseapy as gp

results = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_analysis/AR_stable_DUT_Wilcoxon_001.txt', sep='\t')
#results = results[(results['p_value']<0.01)]

### up-regulated vs. down-regulated ??###
results = results[(results['p_value']<0.05) & (np.abs(results['log2FC'])>1.5)]

pcut = results[['gene_name']]
pcut = pcut.drop_duplicates()
glist = pcut.squeeze().str.strip().to_list()
print(len(glist))
#%%
enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                gene_sets=['Reactome_2022',
                           #'Reactome_2022'
                           #'GO_Biological_Process_2018','GO_Biological_Process_2023','Reactome_2022'
                           ], 
                organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                outdir=None, # don't write to disk
)

enrresult = enr.results.sort_values(by=['Adjusted P-value']) 


# ########### enrichR output file #########
# file = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/rMATS/individual/enrichr/onlyIR_enrichr.txt', sep='\t')
# #########################################

#%%
file = enrresult
def string_fraction_to_float(fraction_str):
    numerator, denominator = fraction_str.split('/')
    return float(numerator) / float(denominator)

file['per'] = file['Overlap'].apply(string_fraction_to_float)

file = file.sort_values('Adjusted P-value')
#file = file.sort_values(by='Combined Score', ascending=False)
##remove mitochondrial ##
file['Term'] = file['Term'].str.rsplit(" ",1).str[0]
#file = file[~file['Term'].str.contains('mitochondrial')]
#file = file.iloc[:10,:]
file = file[file['Adjusted P-value']<0.1]
file['Adjusted P-value'] = -np.log10(file['Adjusted P-value'])

#%%
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 12,    # x축 틱 라벨 글꼴 크기
'ytick.labelsize': 12,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 13,
'legend.title_fontsize': 13, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})
plt.figure(figsize=(7,14))
sns.set_style("whitegrid")
scatter = sns.scatterplot(
    data=file, x='Adjusted P-value', y='Term', hue='per', palette='coolwarm', edgecolor=None, legend=False, s=80
)
plt.xlabel('-log10(FDR)')
plt.ylabel('')
#plt.yticks(fontsize=13)
#plt.xscale('log')  # Log scale for better visualization

# Expanding the plot layout to make room for GO term labels
plt.gcf().subplots_adjust(left=0.4)
#plt.gcf().subplots_adjust(right=0.8)

# Creating color bar
norm = plt.Normalize(file['per'].min(), file['per'].max())
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
sm.set_array([])

# Displaying color bar
cbar = plt.colorbar(sm)
#cbar.set_label('Overlap Percentage')A
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/fig1/GO_DEG_AR_fc1.pdf', dpi=300, bbox_inches='tight')
plt.show()
## %%

# %%
