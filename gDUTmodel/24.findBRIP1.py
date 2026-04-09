#%%
#! this code is.....
"""
FIND BRIP1 !!!!!!!!!!!!!
SAVE BRIP1 !!!!!!!!!!!!!
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
#####^^^ Paired Wilcoxon DEG ####
#candi = ["SV-OV-P040-bfD","SV-OV-P040-atD","SV-OV-P221-bfD","SV-OV-P221-atD","YUHS-180-bfD", "YUHS-180-atD"]
candi = []

geneexp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM_symbol.txt',sep='\t', index_col=0)
geneexp = geneexp.drop(candi, axis=1, inplace=False)
genedf = geneexp.iloc[:,:-1]
f_gene = genedf.loc[(genedf > 0).sum(axis=1) >= genedf.shape[1]*0.9]

f_gene['Gene Symbol'] = geneexp[['Gene Symbol']]

transexp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_transcript_exp.txt',sep='\t', index_col=0)
transexp = transexp.drop(candi, axis=1, inplace=False)

filtered_trans = transexp.loc[(transexp > 0).sum(axis=1) >= transexp.shape[1]*0.3]
filtered_trans['Gene Symbol'] = filtered_trans.index.str.split("-",1).str[1]
filtered_trans = filtered_trans[filtered_trans['Gene Symbol']!= '-']

transdf = filtered_trans.iloc[:,:-1]
genedf = f_gene.iloc[:,:-1]

geneexp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM_symbol.txt',sep='\t', index_col=0)
sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.csv', sep=',')

responder = sampleinfo[sampleinfo['response']==1]['sample_full']
responder = responder.tolist()
responder = [x for x in responder if x not in candi]
resgene = genedf[responder]


deg_pval = []
for index, row in resgene.iterrows():
    pre_samples = row[1::2].values  # Even-indexed columns are pre-treatment samples
    post_samples = row[::2].values  # Odd-indexed columns are post-treatment samples
    
    # Perform the paired Wilcoxon test and store the p-value
    w, p = stats.wilcoxon(post_samples, pre_samples)
    deg_pval.append(p)
    
# Adjust for multiple testing using the Benjamini-Hochberg procedure
p_adjusted = multipletests(deg_pval, method='fdr_bh')[1]

# Create a new DataFrame with geneid and respective p-values
result_df = pd.DataFrame({
    'p_value': deg_pval,
})
result_df.index = f_gene.index
genesym = f_gene[['Gene Symbol']]
result_df = pd.merge(result_df,genesym, how='inner', left_index=True, right_index=True)

degs = result_df[result_df['p_value'] < 0.05]

## FC
avg_pre = resgene.iloc[:, 1::2].mean(axis=1)
avg_post = resgene.iloc[:, ::2].mean(axis=1)

# Calculate the fold change as the log2 ratio of average post-treatment to pre-treatment expression
fold_change = np.log2(avg_post / avg_pre)
result_df['log2FC'] = fold_change


nonDEGlist = set(result_df[result_df['p_value'] > 0.05]['Gene Symbol'])

####^ stable genes: DUT ####
stable_trans = filtered_trans[filtered_trans['Gene Symbol'].isin(nonDEGlist)]
stable_trans = stable_trans[responder]

stable_dut_pval = []

for index, row in stable_trans.iterrows():
    pre_samples = row[1::2].values  # Even-indexed columns are pre-treatment samples
    post_samples = row[::2].values  # Odd-indexed columns are post-treatment samples
    
    # Perform the paired Wilcoxon test and store the p-value
    w, p = stats.wilcoxon(post_samples, pre_samples)
    stable_dut_pval.append(p)
    

# Create a new DataFrame with geneid and respective p-values
stable_result = pd.DataFrame({
    'p_value': stable_dut_pval,
})
stable_result.index = stable_trans.index
stable_result['Gene Symbol'] = stable_result.index.str.split("-",1).str[1]

# %%






#%%%%%%%
#########^^^^ visualize BRIP1 ############
transexp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_transcript_exp.txt',sep='\t', index_col=0)
filtered_trans = transexp
#filtered_trans = transexp.loc[(transexp > 0).sum(axis=1) >= transexp.shape[1]*0.5]
filtered_trans['Gene Symbol'] = filtered_trans.index.str.split("-",1).str[1]
filtered_trans = filtered_trans[filtered_trans['Gene Symbol']!= '-']

brip1_trans = filtered_trans[filtered_trans['Gene Symbol']=='BRIP1']
# %%
brip1 = brip1_trans.iloc[:,:-1]
brip1 = brip1.astype(bool).astype(int)


grid = sns.clustermap(brip1,cmap="mako",xticklabels=1, yticklabels=1, figsize=(20,10), dendrogram_ratio=0.03)
grid.cax.set_visible(False)

# %%
filtered_trans = transexp.loc[(transexp > 0).sum(axis=1) >= transexp.shape[1]*0.4]
filtered_trans['Gene Symbol'] = filtered_trans.index.str.split("-",1).str[1]
filtered_trans = filtered_trans[filtered_trans['Gene Symbol']!= '-']
#filtered_trans = filtered_trans[filtered_trans['Gene Symbol']== 'BRIP1']
translist = filtered_trans.index
#print(translist)
# %%
