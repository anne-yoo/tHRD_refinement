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
sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo_new.txt', sep='\t')
ARlist = list(set(sampleinfo.loc[(sampleinfo['response']==1),'sample_id']))
IRlist = list(set(sampleinfo.loc[(sampleinfo['response']==0),'sample_id']))

TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0)
TU.columns = TU.columns.str[:-4]
preTU = TU.iloc[:,1::2]
postTU = TU.iloc[:,0::2]

ARpre = preTU.loc[:,ARlist]
IRpre = preTU.loc[:,IRlist]
ARpre['Gene Symbol'] = ARpre.index.str.split("-",n=1).str[-1]
IRpre['Gene Symbol'] = IRpre.index.str.split("-",n=1).str[-1]

degresult = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DEG/baseline/MW_DEGresult.txt', sep='\t')
DEGlist = set(degresult[degresult['p_value']<0.05]['Gene Symbol'])
nonDEGlist = set(degresult[degresult['p_value'] > 0.05]['Gene Symbol'])

variable_IRpre = IRpre[IRpre['Gene Symbol'].isin(DEGlist)]
variable_ARpre = ARpre[ARpre['Gene Symbol'].isin(DEGlist)]
stable_IRpre = IRpre[IRpre['Gene Symbol'].isin(nonDEGlist)]
stable_ARpre = ARpre[ARpre['Gene Symbol'].isin(nonDEGlist)]


#%%
variable_IRpre = variable_IRpre.iloc[:,:-1]
variable_ARpre =variable_ARpre.iloc[:,:-1]
stable_IRpre = stable_IRpre.iloc[:,:-1]
stable_ARpre = stable_ARpre.iloc[:,:-1]

#### variable 
variable_dut_pval = []
for i in range(len(variable_ARpre.index)):
    IR_samples = variable_IRpre.iloc[i,:].values 
    AR_samples = variable_ARpre.iloc[i,:].values 
    
    w, p = stats.mannwhitneyu(IR_samples, AR_samples)
    variable_dut_pval.append(p)


# Create a new DataFrame with geneid and respective p-values
variable_result = pd.DataFrame({
    'p_value':variable_dut_pval,
})
variable_result.index = variable_IRpre.index
variable_result['Gene Symbol'] = variable_result.index.str.split("-",n=1).str[-1]

##### FC #####
avg_IR = variable_IRpre.mean(axis=1)
avg_AR = variable_ARpre.mean(axis=1)

fold_change = np.log2(avg_AR / avg_IR)
variable_result['log2FC'] = fold_change

#### stable
stable_dut_pval = []
for i in range(len(stable_ARpre.index)):
    IR_samples = stable_IRpre.iloc[i,:].values 
    AR_samples = stable_ARpre.iloc[i,:].values 
    
    w, p = stats.mannwhitneyu(IR_samples, AR_samples)
    stable_dut_pval.append(p)

# Create a new DataFrame with geneid and respective p-values
stable_result = pd.DataFrame({
    'p_value':stable_dut_pval,
})
stable_result.index = stable_IRpre.index
stable_result['Gene Symbol'] = stable_result.index.str.split("-",n=1).str[-1]

##### FC #####
avg_IR = stable_IRpre.mean(axis=1)
avg_AR = stable_ARpre.mean(axis=1)

fold_change = np.log2(avg_AR / avg_IR)
stable_result['log2FC'] = fold_change

#%%
#stable_result.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/baseline/stable_DUT_MW_FC.txt', sep='\t', index=True)
#variable_result.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/baseline/variable_DUT_MW_FC.txt', sep='\t', index=True)
