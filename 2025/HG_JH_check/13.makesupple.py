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
from scipy.stats import pearsonr
from scipy.stats import linregress

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
sr_sample = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/SR_283_transcript_coverage.txt', sep='\t', index_col=0)
#%%
sr_read = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/LRstats/Illumina/mapped_counts.txt', sep='\t', index_col=0, header=None)
lr_read = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/LRstats/mapped_counts.txt', sep='\t', index_col=0, header=None)

# %%
samplelist = sr_sample.columns.tolist()
sr_read.index = sr_read.index.str[:-3]
lr_read.index = lr_read.index.str[:-3]
sr = sr_read.loc[samplelist, :]
lr = lr_read.loc[samplelist, :]
# %%
sr.columns = ['mapped_reads']
lr.columns = ['mapped_reads']

# %%
plt.figure(figsize=(6,5))
sns.histplot(lr['mapped_reads'], bins=100, label='Short-read', alpha=0.8)
sns.despine()
plt.xlabel('Log10 Mapped reads')
plt.show()


plt.figure(figsize=(6,5))
sns.histplot(np.log10(sr['mapped_reads']), bins=100, label='Short-read', alpha=0.8)
sns.despine()
plt.xlabel('Log10 Mapped reads')
plt.show()
# %%

newdf = pd.merge(sr, lr, left_index=True, right_index=True, suffixes=('_SR', '_LR'))
# %%
newdf.to_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/LRstats/SR_LR_mapped_reads.txt', sep='\t')
# %%
