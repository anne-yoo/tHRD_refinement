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

#%%
tmap_path = "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/gffcompare/gffcompare.SR_238_merged.gtf.tmap"
tmap = pd.read_csv(tmap_path, sep="\t")
matched = tmap[tmap['class_code'] == '='][['qry_id', 'ref_id']]
matched.columns = ['sr_id', 'lr_id']  # for clarity

cpat_df = pd.read_csv("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/CPAT/SR/SR_cpat.ORF_prob.best.tsv", sep="\t", index_col=0)
cpc_df = pd.read_csv("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/CPC2/SR/cpc2output.txt",   sep="\t", index_col=0)
cpat_nc = cpat_df[cpat_df['Coding_prob']<0.364].index.to_list()
cpc_nc = cpc_df[cpc_df['label']=='noncoding'].index.to_list()
nc_trans = set(cpat_nc).union(set(cpc_nc))

transinfo = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/transcriptinfo_SR.txt', sep='\t', index_col=0)

length = transinfo['length_kb']*1000

# Step 1. match 정보에서 LR 기준으로 groupby
matched_grouped = matched.groupby('lr_id')['sr_id'].apply(list)

# Step 2. LR transcript들에 대해 SR expression 합산
collapsed_SR_data = []
collapsed_LR_data = []
collapsed_lengths = []
collapsed_exon = []
collapsed_cov = []
valid_lr_ids = []

tmm = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/SR_transcript_TMM.txt', sep='\t', index_col=0)
LRtmm = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/LR_transcript_TMM.txt', sep='\t', index_col=0)

cov = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/SR_283_transcript_coverage.txt', sep='\t', index_col=0)
cov.index = cov.index.str.split('-',n=1).str[0]

for lr_id, sr_list in matched_grouped.items():
    sr_in_data = [s for s in sr_list if s in tmm.index]
    if len(sr_in_data) == 0 or lr_id not in LRtmm.index:
        continue
    sr_sum = tmm.loc[sr_in_data].sum(axis=0)
    lr_expr = LRtmm.loc[lr_id]
    collapsed_SR_data.append(sr_sum)
    collapsed_LR_data.append(lr_expr)
    valid_lr_ids.append(lr_id)
    
    sr_in_data = [s for s in sr_list if s in cov.index]
    if len(sr_in_data) == 0 or lr_id not in LRtmm.index:
        continue
    sr_sum = cov.loc[sr_in_data].sum(axis=0)
    collapsed_cov.append(sr_sum)

    # transcript length: SR transcript 중 가장 긴 것으로 사용 (또는 첫 번째)
    sr_lengths = transinfo.loc[[s for s in sr_list if s in transinfo.index], 'length_kb']
    sr_lengths = sr_lengths*1000
    if not sr_lengths.empty:
        collapsed_lengths.append(sr_lengths.max())  # or mean(), min()
    
    sr_exon = transinfo.loc[[s for s in sr_list if s in transinfo.index], 'exon_count']
    if not sr_exon.empty:
        collapsed_exon.append(sr_exon.max()) 

# Step 3. DataFrame으로 변환
SR_data = pd.DataFrame(collapsed_SR_data, index=valid_lr_ids)
LR_data = pd.DataFrame(collapsed_LR_data, index=valid_lr_ids)
SR_cov = pd.DataFrame(collapsed_cov, index=valid_lr_ids)
trans_lengths = pd.Series(collapsed_lengths, index=valid_lr_ids)
exon_counts = pd.Series(collapsed_exon, index=valid_lr_ids)

#%%
##^^ histogram check ########
import matplotlib.pyplot as plt
import seaborn as sns

# 시각화 스타일
sns.set_style("whitegrid")

# 히스토그램 그리기
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Exon count histogram
sns.histplot(exon_counts,kde=False, ax=axes[0], color='#0072B2')
axes[0].set_title("Exon Count Distribution")
axes[0].set_xlabel("Exon Count")
axes[0].set_xlim([0,50])
axes[0].set_ylabel("Number of Transcripts")

# Transcript length histogram
sns.histplot(trans_lengths, kde=False, ax=axes[1], color='#D55E00')
axes[1].set_xlim([0,20000])
axes[1].set_title("Transcript Length Distribution")
axes[1].set_xlabel("Transcript Length (bp)")
axes[1].set_ylabel("Number of Transcripts")

plt.tight_layout()
plt.show()

# %%
####^^only transcripts with at least five exons and mean coverage ≥1 across at least 10% of samples. #####

import matplotlib.pyplot as plt

# 필터 조건 범위
exon_cutoffs = [0, 1, 2, 3, 4, 5, 10, 20]
length_cutoffs = [0, 300, 600, 900, 1200, 1800, 2400, 3600]
coverage_cutoffs = [0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]  # min_coverage=1, fraction만 바뀜

# 결과 저장용
results = {
    'exon': [],
    'length': [],
    'coverage': []
}

# exon cutoff별 transcript 개수
for cutoff in exon_cutoffs:
    passed = exon_counts[exon_counts >= cutoff].index
    results['exon'].append(len(passed))

# length cutoff별 transcript 개수
for cutoff in length_cutoffs:
    passed = trans_lengths[trans_lengths >= cutoff].index
    results['length'].append(len(passed))

# coverage cutoff별 transcript 개수 (min_coverage = 1 고정)
def filter_by_coverage(coverage_df, min_coverage, min_fraction):
    required_count = int(coverage_df.shape[1] * min_fraction)
    passing_counts = (coverage_df >= min_coverage).sum(axis=1)
    return coverage_df.index[passing_counts >= required_count]

for cutoff in coverage_cutoffs:
    passed = filter_by_coverage(SR_cov, min_coverage=1, min_fraction=cutoff)
    results['coverage'].append(len(passed))

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(exon_cutoffs, results['exon'], marker='o', linestyle='-')
axes[0].set_title("Exon Count Filter")
axes[0].set_xlabel("Exon ≥ cutoff")
axes[0].set_ylabel("# Transcripts")

axes[1].plot(length_cutoffs, results['length'], marker='o', linestyle='-')
axes[1].set_title("Transcript Length Filter")
axes[1].set_xlabel("Length ≥ cutoff (bp)")

axes[2].plot([int(c * 100) for c in coverage_cutoffs], results['coverage'], marker='o', linestyle='-')
axes[2].set_title("Coverage Filter")
axes[2].set_xlabel("≥1 coverage in ≥% samples")
axes[2].set_xticks([int(c * 100) for c in coverage_cutoffs])

plt.tight_layout()
plt.show()

#%%
cpat_df = pd.read_csv("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/CPAT/SR/SR_cpat.ORF_prob.best.tsv", sep="\t", index_col=0)
cpc_df = pd.read_csv("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/CPC2/SR/cpc2output.txt",   sep="\t", index_col=0)
cpat_nc = cpat_df[cpat_df['Coding_prob']<0.364].index.to_list()
cpc_nc = cpc_df[cpc_df['label']=='noncoding'].index.to_list()
nc_trans = set(cpat_nc).union(set(cpc_nc))

# %%
###^^^ regression plot ######

from scipy.stats import pearsonr

exon_cutoffs = [0, 1, 2, 3, 4, 5, 10, 20]
coverage_cutoffs = [0.01, 0.03, 0.05, 0.1]

fig, axes = plt.subplots(len(exon_cutoffs), len(coverage_cutoffs), figsize=(20, 24), sharey=True)
all_results = []

for i, exon_cut in enumerate(exon_cutoffs):
    passing_exon = exon_counts[exon_counts >= exon_cut].index

    for j, cov_frac in enumerate(coverage_cutoffs):
        passing_cov = SR_cov[(SR_cov >= 1).sum(axis=1) >= int(SR_cov.shape[1] * cov_frac)].index
        valid_tx = passing_exon.intersection(passing_cov)

        plot_records = []

        for coding_label, is_coding in [('coding', True), ('noncoding', False)]:
            tx_ids = [tx for tx in valid_tx if (tx not in nc_trans) == is_coding]
            SR_sub = SR_data.loc[SR_data.index.intersection(tx_ids)]
            LR_sub = LR_data.loc[LR_data.index.intersection(tx_ids)]

            for sample in SR_sub.columns:
                sr_vals = SR_sub[sample]
                lr_vals = LR_sub[sample]
                mask = sr_vals.notna() & lr_vals.notna()
                if mask.sum() > 2:
                    r, p = pearsonr(sr_vals[mask], lr_vals[mask])
                    group = 'Normal' if '-N' in sample else 'Tumor'
                    plot_records.append({
                        'Group': group,
                        'Sample': sample,
                        'Pearson_r': r,
                        'P_value': p,
                        'CodingStatus': coding_label,
                        'ExonCut': exon_cut,
                        'CovCut': cov_frac
                    })

            # Whole
            for sample in SR_sub.columns:
                sr_vals = SR_sub[sample]
                lr_vals = LR_sub[sample]
                mask = sr_vals.notna() & lr_vals.notna()
                if mask.sum() > 2:
                    r, p = pearsonr(sr_vals[mask], lr_vals[mask])
                    plot_records.append({
                        'Group': 'Whole',
                        'Sample': sample,
                        'Pearson_r': r,
                        'P_value': p,
                        'CodingStatus': coding_label,
                        'ExonCut': exon_cut,
                        'CovCut': cov_frac
                    })

        all_results.extend(plot_records)

        ax = axes[i, j]
        if plot_records:
            df = pd.DataFrame(plot_records)
            sns.violinplot(data=df, x='Group', y='Pearson_r', hue='CodingStatus', order=['Whole','Normal','Tumor'],
                           split=True, inner='box', ax=ax, palette='Set2')
        ax.set_title(f"exon≥{exon_cut}, cov≥{int(cov_frac*100)}% (n={len(valid_tx)})")
        if j == 0:
            ax.set_ylabel("Pearson r")
        else:
            ax.set_ylabel("")
        ax.set_xlabel("")

plt.tight_layout()
plt.show()

# %%

##^^^ threshold 별 regression

def compute_samplewise_r_for_thresholds(SR_data, LR_data, trans_lengths, thresholds):
    all_r = []
    all_group = []

    for threshold in thresholds:
        selected = trans_lengths[trans_lengths >= threshold].index
        sr_sub = SR_data.loc[SR_data.index.intersection(selected)]
        lr_sub = LR_data.loc[LR_data.index.intersection(selected)]

        for sample in sr_sub.columns:
            sr_vals = sr_sub[sample]
            lr_vals = lr_sub[sample]
            valid_idx = sr_vals.index.intersection(lr_vals.index)
            sr_vals = sr_vals.loc[valid_idx]
            lr_vals = lr_vals.loc[valid_idx]
            mask = ~sr_vals.isna() & ~lr_vals.isna()

            if mask.sum() > 2:
                r, _ = pearsonr(sr_vals[mask], lr_vals[mask])
                all_r.append(r)
                all_group.append(f">= {threshold}bp")

    return pd.DataFrame({'Pearson_r': all_r, 'Length_Threshold': all_group})

thresholds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

df_r = compute_samplewise_r_for_thresholds(SR_data, LR_data, exon_counts, thresholds)

plt.figure(figsize=(14, 5))
sns.boxplot(data=df_r, x='Length_Threshold', y='Pearson_r', color='white', showfliers=False)
#plt.title("Sample-wise SR vs. LR Pearson correlation by length threshold")
plt.xticks(rotation=45)
sns.despine()
plt.tight_layout()
plt.show()


# 각 구간별 transcript 개수 계산
exon_counts_dic = {
    f">= {thr}bp": (exon_counts >= thr).sum()
    for thr in thresholds
}

# DataFrame으로 변환
count_df = pd.DataFrame({
    "Exon_Threshold": list(exon_counts_dic.keys()),
    "Transcript_Count": list(exon_counts_dic.values())
})

# 시각화
plt.figure(figsize=(14, 5))
sns.barplot(data=count_df, x="Exon_Threshold", y="Transcript_Count", color="gray")
plt.xticks(rotation=45)
plt.title("Number of Transcripts at Each Exon Threshold")
plt.xlabel("Exon Count Threshold")
plt.ylabel("Transcript Count")
sns.despine()
plt.tight_layout()
plt.show()


# %%
