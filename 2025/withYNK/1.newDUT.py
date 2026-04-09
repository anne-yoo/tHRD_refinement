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
import re
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.stats.multitest import multipletests
from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42  
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'Helvetica'

plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 13,    # x축 틱 라벨 글꼴 크기

'ytick.labelsize': 13,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 13,
'legend.title_fontsize': 13, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})
sns.set_style("ticks")

# %%
####^^^ validation cohort check ########
val_df = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost_80_transcript_TU.txt', sep='\t', index_col=0)
val_gene = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost_80_gene_TPM.txt', sep='\t', index_col=0)
val_clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost_80_clinicalinfo.txt', sep='\t', index_col=0)
val_df = val_df.apply(pd.to_numeric, errors='coerce')
#%%

vallist = list(val_clin.index)
val_df = val_df.loc[:,vallist]
val_gene = val_gene.loc[genesymbol.index,vallist]

##### maintenance / salvage ######
inputlist = val_clin[(val_clin['OM/OS']=='maintenance')].index.to_list()
val_df =  val_df.iloc[:-2,val_df.columns.isin(inputlist)]
val_gene =  val_gene.iloc[:,val_gene.columns.isin(inputlist)]
val_gene = pd.merge(val_gene,genesymbol,left_index=True,right_index=True)
val_clin = val_clin.loc[val_clin.index.isin(inputlist),:]
##### only CR + IR ######

# %%
##^^ DEG ##############
r_list = val_clin[(val_clin['response']==1)].index.to_list()
nr_list = val_clin[(val_clin['response']==0)].index.to_list()

f_gene = val_gene.iloc[:,:-1]
genesym = val_gene[['Gene Symbol']]

deg_pval = []
r_df = val_gene.loc[:,r_list]
nr_df = val_gene.loc[:,nr_list]

for i in range(r_df.shape[0]):
    r_samples = r_df.iloc[i,:]
    nr_samples = nr_df.iloc[i,:]
    w, p = stats.mannwhitneyu(r_samples, nr_samples)
    deg_pval.append(p)

# Create a new DataFrame with geneid and respective p-values
result_df = pd.DataFrame({
    'p_value': deg_pval,
})
result_df.index = f_gene.index
result_df = pd.merge(result_df,genesym, how='inner', left_index=True, right_index=True)

### add FC ###

avg_r = r_df.mean(axis=1)
avg_nr = nr_df.mean(axis=1)

# Calculate the fold change as the log2 ratio of average post-treatment to pre-treatment expression
fold_change = np.log2(avg_r / avg_nr)
result_df['log2FC'] = fold_change

result_df.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_DEGDUT/83_DEGresult_FC.txt', sep='\t')

#%%
##^^^ DUT #############

degresult = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_DEGDUT/83_DEGresult_FC.txt', sep='\t')
DEGlist = set(degresult[degresult['p_value']<0.05]['Gene Symbol'])
nonDEGlist = set(degresult[degresult['p_value'] >= 0.05]['Gene Symbol'])
r_df = val_df.loc[:,r_list]
nr_df = val_df.loc[:,nr_list]

r_df['Gene Symbol'] = r_df.index.str.split("-",n=1).str[-1]
nr_df['Gene Symbol'] = nr_df.index.str.split("-",n=1).str[-1]


variable_NR = nr_df[nr_df['Gene Symbol'].isin(DEGlist)]
variable_R = r_df[r_df['Gene Symbol'].isin(DEGlist)]
stable_NR = nr_df[nr_df['Gene Symbol'].isin(nonDEGlist)]
stable_R = r_df[r_df['Gene Symbol'].isin(nonDEGlist)]

variable_NR = variable_NR.iloc[:,:-1]
variable_R = variable_R.iloc[:,:-1]
stable_NR = stable_NR.iloc[:,:-1]
stable_R = stable_R.iloc[:,:-1]

variable_dut_pval = []
for i in range(len(variable_NR.index)):
    NR_samples = variable_NR.iloc[i,:].values 
    R_samples = variable_R.iloc[i,:].values 
    
    w, p = stats.mannwhitneyu(NR_samples, R_samples)
    variable_dut_pval.append(p)

# Create a new DataFrame with geneid and respective p-values
variable_result = pd.DataFrame({
    'p_value':variable_dut_pval,
})
variable_result.index = variable_NR.index
variable_result['Gene Symbol'] = variable_result.index.str.split("-",n=1).str[-1]

##### FC #####
avg_NR = variable_NR.mean(axis=1)
avg_R = variable_R.mean(axis=1)

fold_change = np.log2(avg_R / avg_NR)
variable_result['log2FC'] = fold_change

#### stable
stable_dut_pval = []
for i in range(len(stable_NR.index)):
    NR_samples = stable_NR.iloc[i,:].values 
    R_samples = stable_R.iloc[i,:].values 
    
    w, p = stats.mannwhitneyu(NR_samples, R_samples)
    stable_dut_pval.append(p)

# Create a new DataFrame with geneid and respective p-values
stable_result = pd.DataFrame({
    'p_value': stable_dut_pval,
})
stable_result.index = stable_NR.index
stable_result['Gene Symbol'] = stable_result.index.str.split("-",n=1).str[-1]

##### FC #####
avg_NR = stable_NR.mean(axis=1)
avg_R = stable_R.mean(axis=1)

fold_change = np.log2(avg_R / avg_NR)
stable_result['log2FC'] = fold_change

variable_result.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_DEGDUT/83_variable_DUT_FC.txt', sep='\t')
stable_result.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_DEGDUT/83_salvage_DUT_FC.txt', sep='\t')

# %%
