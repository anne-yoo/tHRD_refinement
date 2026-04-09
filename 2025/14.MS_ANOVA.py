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
AR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t', index_col=0)
IR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUT/nonresponder_stable_DUT_Wilcoxon.txt', sep='\t', index_col=0)

ARdutlist = AR_dut.loc[(AR_dut['p_value']<0.05) & (np.abs(AR_dut['log2FC'])>1.5)].index.to_list()
IRdutlist = IR_dut.loc[(IR_dut['p_value']<0.05) & (np.abs(IR_dut['log2FC'])>1.5)].index.to_list()

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo_new.txt', sep='\t')
ARlist = list(set(sampleinfo.loc[(sampleinfo['response']==1),'sample_id']))
IRlist = list(set(sampleinfo.loc[(sampleinfo['response']==0),'sample_id']))

TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0)
TU.columns = TU.columns.str[:-4]
preTU = TU.iloc[:,1::2]
postTU = TU.iloc[:,0::2]

ARpre = preTU.loc[ARdutlist,ARlist]
ARpost = postTU.loc[ARdutlist,ARlist]
IRpre = preTU.loc[IRdutlist,IRlist]
IRpost = postTU.loc[IRdutlist,IRlist]

majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = majorminor[majorminor['type']=='major']['gene_ENST'].to_list()
minorlist = majorminor[majorminor['type']=='minor']['gene_ENST'].to_list()

ARpremajor = ARpre.loc[ARpre.index.isin(majorlist),:]
ARpostmajor = ARpost.loc[ARpost.index.isin(majorlist),:]
IRpremajor = IRpre.loc[IRpre.index.isin(majorlist),:]
IRpostmajor = IRpost.loc[IRpost.index.isin(majorlist),:]

# %%
####^^^ AR S/M ANOVA with major DUTs ##########################

clin_df = sampleinfo.copy()

# 1. ΔTU 계산: AR_post - AR_pre (transcript x sample)
AR_delta = ARpostmajor - ARpremajor

# 2. Sample-level ΔTU 평균 계산
AR_delta_mean = AR_delta.mean(axis=0).reset_index()
AR_delta_mean.columns = ['sample_id', 'delta_TU_mean']

# 3. 임상 정보 merge
ar_clin = clin_df[clin_df['sample_id'].isin(AR_delta.columns)]
ar_clin['sample_id'] = ar_clin['sample_id'].astype(str)
merged = AR_delta_mean.merge(ar_clin[['sample_id', 'purpose']], on='sample_id')
merged = merged.iloc[1::2,:]
# 4. Linear model: ΔTU_mean ~ purpose
from statsmodels.formula.api import ols

model = ols('delta_TU_mean ~ C(purpose)', data=merged).fit()
print(model.summary())


# %%
###^^ validation cohort ##########################
hrr_core_genes = ['GEN1', 'BARD1', 'RAD50', 'SHFM1', 'XRCC2', 'NBN', 'MUS81', 'MRE11A', 'RAD52', 'BRCA2', 'XRCC3', 'RAD51C', 'RAD51D', 'TP53BP1', 'BLM', 'SLX1A', 'PALB2', 'TOP3A', 'BRCA1', 'EME1', 'BRIP1', 'RBBP8']

val_df = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/116_validation_TU_group.txt', sep='\t', index_col=0)
val_gene = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/116_validation_gene_TPM_symbol_group.txt', sep='\t', index_col=0)
val_clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_validation_clinicalinfo_new.txt', sep='\t', index_col=0)
val_df = val_df.iloc[:-1,:-1]
val_df = val_df.apply(pd.to_numeric, errors='coerce')
vallist = list(val_df.columns)
val_clin = val_clin.loc[vallist,:]
genesym = val_gene.iloc[:,-1]
val_gene.index = val_gene.iloc[:,-1]
val_gene = val_gene.iloc[:-1,:-1]
val_gene = val_gene.apply(pd.to_numeric, errors='coerce')

majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = majorminor[majorminor['type']=='major']['gene_ENST'].to_list()

druginfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/validation_sampleinfo.txt', sep='\t', index_col=0)
druginfo = druginfo['drug']

val_clin = val_clin.merge(druginfo, left_index=True, right_index=True, how='left')

from scipy.stats import spearmanr

val_gene = val_gene.loc[:,val_clin.index]
val_df = val_df.loc[:,val_clin.index]

val_clin['group'] = val_clin['type'].replace({'CR': 'R', 'IR': 'NR','AR': 'R'})
val_majortu = val_df.loc[val_df.index.isin(majorlist),:]

def get_subset_by_genes(df, genelist):
    return df[df.index.map(lambda x: x.split('-')[-1] in genelist)]

val_majortu = get_subset_by_genes(val_majortu, hrr_core_genes)
val_clin['sample_id'] = val_clin.index.astype(str)
val_clin.index.names = ['SAMPLE']
# sample x transcript → sample x TU_mean
val_majortu_mean = val_majortu.mean(axis=0)  # axis=0 → 각 column = sample별 평균
val_majortu_mean = val_majortu_mean.reset_index()
val_majortu_mean.columns = ['sample_id', 'TU_mean']
val_clin['sample_id'] = val_clin.index.astype(str)  # 보장
val_clin = val_clin.rename(columns={'group': 'Response', 'OM/OS': 'Cohort'})
merged = val_majortu_mean.merge(val_clin[['sample_id', 'Response', 'Cohort']], on='sample_id')
from statsmodels.formula.api import ols
model = ols('TU_mean ~ C(Response) * C(Cohort)', data=merged).fit()
print(model.summary())

# %%
