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
from scipy.stats import ranksums
from statsmodels.stats.multitest import multipletests

idcheck = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/merged_83_filtered_transcripts_with_gene_info.tsv', sep='\t')
idcheck = idcheck[['mstrg_gene_id','gene_name']].drop_duplicates()
df = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/rMATS/rmats_out_con1/SE.MATS.JCEC.txt', sep='\t')

def calc_psi(ijc_str, sjc_str):
    ijc = np.array(list(map(float, ijc_str.split(","))))
    sjc = np.array(list(map(float, sjc_str.split(","))))
    denom = ijc + sjc
    psi = np.where(denom > 0, ijc / denom, np.nan)
    return psi

psi_1 = df.apply(
    lambda r: calc_psi(r["IJC_SAMPLE_1"], r["SJC_SAMPLE_1"]),
    axis=1
)

psi_2 = df.apply(
    lambda r: calc_psi(r["IJC_SAMPLE_2"], r["SJC_SAMPLE_2"]),
    axis=1
)
pvals = []
delta_psi = []

for p1, p2 in zip(psi_1, psi_2):
    v1 = p1[~np.isnan(p1)]
    v2 = p2[~np.isnan(p2)]

    # 최소 샘플 수 필터 (중요)
    if len(v1) < 5 or len(v2) < 5:
        pvals.append(np.nan)
        delta_psi.append(np.nan)
        continue

    stat, p = ranksums(v1, v2)
    pvals.append(p)
    delta_psi.append(np.nanmean(v1) - np.nanmean(v2))

pvals = np.array(pvals)
valid = ~np.isnan(pvals)

fdr = np.full_like(pvals, np.nan, dtype=float)
fdr[valid] = multipletests(pvals[valid], method="fdr_bh")[1]

out = df.loc[:, [
    "ID", "GeneID", "geneSymbol", "chr", "strand",
    "exonStart_0base", "exonEnd"
]].copy()

out["deltaPSI"] = delta_psi
out["pvalue_custom"] = pvals
out["fdr_custom"] = fdr

out = pd.merge(out,idcheck,left_on='GeneID', right_on='mstrg_gene_id')

# %%
check = out.dropna().sort_values("pvalue_custom")
# %%
from scipy.stats import mannwhitneyu

df = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/rMATS/rmats_out_con1/RI.MATS.JCEC.txt', sep='\t')
def parse_inc_level(s):
    if pd.isna(s):
        return np.array([])
    vals = []
    for x in s.split(","):
        if x == "NA":
            vals.append(np.nan)
        else:
            vals.append(float(x))
    return np.array(vals)

psi_1 = df["IncLevel1"].apply(parse_inc_level)
psi_2 = df["IncLevel2"].apply(parse_inc_level)

pvals = []
delta_psi = []

for v1, v2 in zip(psi_1, psi_2):
    v1 = v1[~np.isnan(v1)]
    v2 = v2[~np.isnan(v2)]

    # 최소 샘플 필터 (중요)
    if len(v1) < 5 or len(v2) < 5:
        pvals.append(np.nan)
        delta_psi.append(np.nan)
        continue

    stat, p = mannwhitneyu(v1, v2, alternative="two-sided")
    pvals.append(p)
    delta_psi.append(np.mean(v1) - np.mean(v2))

pvals = np.array(pvals)
valid = ~np.isnan(pvals)

fdr = np.full_like(pvals, np.nan, dtype=float)
fdr[valid] = multipletests(pvals[valid], method="fdr_bh")[1]

out = df.loc[:, [
    "ID", "GeneID", "geneSymbol", "chr", "strand",
    "riExonStart_0base", "riExonEnd"
]].copy()

out["deltaPSI_custom"] = delta_psi
out["pvalue_custom"] = pvals
out["fdr_custom"] = fdr

out = pd.merge(out,idcheck,left_on='GeneID', right_on='mstrg_gene_id')
# %%
check = out.dropna().sort_values("pvalue_custom")

# %%
