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


#%%
tmap_path = "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/gffcompare/gffcompare.SR_238_merged.gtf.tmap"
tmap = pd.read_csv(tmap_path, sep="\t")
matched = tmap[['qry_id', 'ref_id']]
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

def filter_by_coverage(coverage_df, min_coverage=1, min_fraction=0.1):
    num_samples = coverage_df.shape[1]
    required_count = int(num_samples * min_fraction)

    # 각 transcript별로 coverage ≥ min_coverage 인 sample 수 세기
    passing_counts = (coverage_df >= min_coverage).sum(axis=1)

    # 조건을 만족하는 transcript ID만 선택
    filtered = coverage_df.index[passing_counts >= required_count]
    return filtered

cov = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/SR_283_transcript_coverage.txt', sep='\t', index_col=0)
info = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/transcriptinfo_SR.txt', sep='\t', index_col=0)

filtered_cov = filter_by_coverage(cov, min_coverage=3, min_fraction=0.1)
filtered_exon= info[info['exon_count'] >= 3].index

both_trans = set(filtered_cov.str.split("-", n=1).str[0]).intersection(set(filtered_exon))

tmm = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/SR_transcript_TMM.txt', sep='\t', index_col=0)
finaltmm = tmm
LRtmm = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/LR_transcript_TMM.txt', sep='\t', index_col=0)

cov = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/SR_283_transcript_coverage.txt', sep='\t', index_col=0)
cov.index = cov.index.str.split('-',n=1).str[0]

for lr_id, sr_list in matched_grouped.items():
    sr_in_data = [s for s in sr_list if s in finaltmm.index]
    if len(sr_in_data) == 0 or lr_id not in LRtmm.index:
        continue
    sr_sum = finaltmm.loc[sr_in_data].sum(axis=0)
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

# %%
#^^^ correlation + matching percent barplot ######
# === 매칭 정보 준비 ===
def classify_class(code):
    if code == '=':
        return 'Full'
    elif code == 'c':
        return 'Partial'
    else:
        return 'Unmatched'

tmap['match_type'] = tmap['class_code'].apply(classify_class)
tx_to_match = tmap.set_index('qry_id')['match_type'].to_dict()

# === 샘플별 바플롯 + r값 계산 ===
result_rows = []

for sample in SR_data.columns:
    sr_vals = SR_data[sample]
    lr_vals = LR_data[sample]

    # 둘 다 발현량 > 0인 transcript만 사용
    detected = sr_vals[(sr_vals > 0) & (lr_vals > 0)]
    idx = detected.index

    # match type 계산
    match_counts = {'Full': 0, 'Partial': 0, 'Unmatched': 0}
    for tx in idx:
        match = tx_to_match.get(tx, 'Unmatched')
        match_counts[match] += 1

    # Pearson r 계산
    x = sr_vals.loc[idx]
    y = lr_vals.loc[idx]
    if len(idx) > 2:
        r, _ = pearsonr(np.log2(x), np.log2(y))
    else:
        r = np.nan

    # 결과 저장
    result_rows.append({
        'Sample': sample,
        'Full': match_counts['Full'],
        'Partial': match_counts['Partial'],
        'Unmatched': match_counts['Unmatched'],
        'Pearson_r': r,
        'Group': 'Tumor' if '-T' in sample or 'T' in sample.split('-')[-1] else 'Normal'
    })

# === 데이터프레임 정리 ===
plot_df = pd.DataFrame(result_rows)
normal_df = plot_df[plot_df['Group'] == 'Normal'].sort_values('Pearson_r', ascending=False)
tumor_df = plot_df[plot_df['Group'] == 'Tumor'].sort_values('Pearson_r', ascending=False)
plot_df = pd.concat([normal_df, tumor_df]).reset_index(drop=True)
plot_df['x'] = range(len(plot_df))
nt_split = len(normal_df)

#%%
fig, ax1 = plt.subplots(figsize=(16, 5))

# Stacked barplot
ax1.bar(plot_df['x'], plot_df['Full'], color="#F9E400", label="Full match", width=1.0, edgecolor="#F9E400")
ax1.bar(plot_df['x'], plot_df['Partial'], bottom=plot_df['Full'], color="#178C84", label="Partial match", width=1.0, edgecolor="#178C84")
ax1.bar(plot_df['x'], plot_df['Unmatched'], bottom=plot_df['Full'] + plot_df['Partial'],
        color="#220039", label="Unmatched", width=1.0, edgecolor="#220039")

ax1.set_ylabel("Count of matching transcripts")
ax1.set_xticks([])
ax1.set_xlim(-0.5, len(plot_df) - 0.5)
ax1.legend(loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.3, 1.1))

# Correlation lineplot
ax2 = ax1.twinx()
ax2.plot(plot_df["x"], plot_df["Pearson_r"], color="#FF8031", label="Pearson r", linewidth=2, alpha=0.7, linestyle=':')
ax2.set_ylabel("Pearson r")
ax2.set_ylim(0, 1)

# 경계선
ax1.axvline(x=nt_split - 0.5, color='white', linestyle='--', lw=2)

sns.despine()
plt.tight_layout()
plt.savefig("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/figures/samplewiseLRfiltered_barplot.pdf", bbox_inches='tight', dpi=300)
plt.show()

# %%
##^^ proportion ################
# 전체 수로 나누어서 비율로 변환
plot_df['Total'] = plot_df[['Full', 'Partial', 'Unmatched']].sum(axis=1)
plot_df['Full_prop'] = plot_df['Full'] / plot_df['Total']
plot_df['Partial_prop'] = plot_df['Partial'] / plot_df['Total']
plot_df['Unmatched_prop'] = plot_df['Unmatched'] / plot_df['Total']

fig, ax1 = plt.subplots(figsize=(16, 5))

# Stacked barplot (비율 기반)
ax1.bar(plot_df['x'], plot_df['Full_prop'], color="#F9E400", label="Full match", width=1.0, edgecolor="#F9E400")
ax1.bar(plot_df['x'], plot_df['Partial_prop'], bottom=plot_df['Full_prop'], color="#178C84", label="Partial match", width=1.0, edgecolor="#178C84")
ax1.bar(plot_df['x'], plot_df['Unmatched_prop'], bottom=plot_df['Full_prop'] + plot_df['Partial_prop'],
        color="#220039", label="Unmatched", width=1.0, edgecolor="#220039")

ax1.set_ylabel("Proportion of matching transcripts")
ax1.set_xticks([])
ax1.set_xlim(-0.5, len(plot_df) - 0.5)
ax1.set_ylim(0, 1)
ax1.legend(loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.3, 1.1))

# Correlation lineplot
ax2 = ax1.twinx()
ax2.plot(plot_df["x"], plot_df["Pearson_r"], color="#FF8031", label="Pearson r", linewidth=2, alpha=0.7, linestyle=':')
ax2.set_ylabel("Pearson r")
ax2.set_ylim(0, 1)

# 경계선
ax1.axvline(x=nt_split - 0.5, color='white', linestyle='--', lw=2)
ax1.axhline(y=0.4, color='white', linestyle='-', lw=1, alpha=0.5)

sns.despine()
plt.tight_layout()
plt.savefig("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/figures/samplewiseLRfiltered_barplot_proportion.pdf", bbox_inches='tight', dpi=300)
plt.show()

# %%
