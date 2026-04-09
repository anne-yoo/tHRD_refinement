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
'legend.fontsize': 12,
'legend.title_fontsize': 12, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})
sns.set_style("ticks")

# %%
#####^^ Nanopore Make TU file #########
df = pd.read_csv("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/quantification/LR_TMM_transcript.txt", sep='\t', index_col=0)
#df["gene_id"] = df.index.str.split("-", n=1).str[-1]
#df.index = df.index.str.split("-", n=1).str[0]
tpm =  pd.read_csv("/home/jiye/jiye/nanopore/HG_JH_check/mixSRLR/quantification_v2/LR_283_transcript_TPM.txt", sep='\t', index_col=0)
tpm["gene_id"] = tpm.index.str.split("-", n=1).str[-1]
tpm.index = tpm.index.str.split("-", n=1).str[0]
tpm = tpm [['gene_id']]

# 수치형 column만 따로 추출 (즉, 샘플들)
df = df.merge(tpm, left_index=True, right_index=True)

#%%
value_cols = df.columns.difference(["gene_id"])

# gene별 총합 계산 (원래와 동일한 row 수로 확장됨)
gene_sum = df.groupby("gene_id")[value_cols].transform("sum")

# usage 계산
usage_df = df[value_cols] / gene_sum
usage_df.index = df.index

# %%
usage_df['gene_id'] = usage_df.index.str.split("-", n=1).str[-1]

# %%
#gene = pd.read_csv("/home/jiye/jiye/nanopore/FINALDATA/matched_gene_TPM.txt", sep='\t', index_col=0)
# %%
usage_df = usage_df.drop(columns=['gene_id'])
# %%
usage_df = usage_df.fillna(0)
usage_df.to_csv("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/quantification/LR_238_transcript_TU_fillna_fromTMM.txt", sep='\t', index=True)
# %%
