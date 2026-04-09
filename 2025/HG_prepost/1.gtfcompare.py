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
merged_info = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/filtered_transcripts_with_gene_info.tsv',sep='\t')
merged_cov5_info = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/cov5_filtered_transcripts_with_gene_info.tsv',sep='\t')
merged_cov5_stranded_info = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/stranded_cov5_filtered_transcripts_with_gene_info.tsv',sep='\t')

#%%

merged_info = merged_info.dropna()
merged_cov5_info = merged_cov5_info.dropna()
v19_info = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO/filtered_transcripts_with_gene_info.tsv', sep='\t')
v19_major = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/gencode_majorminorlist.txt', sep='\t')
merged_major = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_80_majorminorlist.txt',sep='\t')


# %%
tpm = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_80_transcript_TPM.txt', sep='\t', index_col=0)
# %%
import pandas as pd

# v19_major: dataframe with columns ['transcriptid','genename','Transcript-Gene','type']
# tpm_df: expression matrix indexed by 'Transcript-Gene'

# align the annotation to the TPM matrix
annot = merged_major[['Transcript-Gene', 'type']].set_index('Transcript-Gene')
tpm_with_type = tpm.join(annot, how='inner')

# Drop the 'type' column so only TPM columns remain
tpm_only = tpm_with_type.iloc[:, :-1]

major_means = tpm_only[tpm_with_type['type'] == 'major'].mean(axis=0)
minor_means = tpm_only[tpm_with_type['type'] == 'minor'].mean(axis=0)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

sns.kdeplot(major_means, fill=True, alpha=0.4, linewidth=2, label="Mean Major TPM per sample")
sns.kdeplot(minor_means, fill=True, alpha=0.4, linewidth=2, label="Mean Minor TPM per sample")

plt.xscale("log")
plt.xlabel("Mean TPM per sample")
plt.ylabel("Density")
plt.title("Mean TPM: stringtie merged ver.")
plt.legend()
plt.tight_layout()
sns.despine()
plt.show()


# %%
import matplotlib.ticker as ticker

df = pd.DataFrame({
    "TPM": pd.concat([pd.Series(major_means), pd.Series(minor_means)], ignore_index=True),
    "type": ["major"] * len(major_means) + ["minor"] * len(minor_means)
})

plt.figure(figsize=(6,6))
ticks = [1, 10, 100]

ax = sns.violinplot(data=df, x="type", y="TPM", cut=0)
ax.set_yscale("log")
ax.set_ylabel('log10 TPM')
ax.set_yticks(ticks)
ax.set_yticklabels([str(t) for t in ticks])

plt.title("Mean TPM: stringtie merged ver.")
sns.despine()
plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# count values
counts = merged_major['type'].value_counts()

plt.figure(figsize=(4,5))
sns.barplot(x=counts.index, y=counts.values)

plt.xlabel("Transcript type")
plt.ylabel("Count")
plt.title("Number of Major vs Minor transcripts")
plt.tight_layout()
plt.show()

# %%
