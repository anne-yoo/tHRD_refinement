#%%
from sklearn.preprocessing import MinMaxScaler
def parse_trmap_file(trmap_path):
    records = []
    with open(trmap_path, 'r') as f:
        current_qry = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # e.g., >MSTRG.15.1 GL000194.1:52950-115195 . 52950-115195
                current_qry = line.split()[0][1:]  # remove '>'
            elif line and not line.startswith("#"):
                parts = line.split("\t")
                if len(parts) >= 7 and current_qry:
                    record = {
                        "qry_id": current_qry,
                        "class_code": parts[0],
                        "ref_chr": parts[1],
                        "ref_strand": parts[2],
                        "ref_start": int(parts[3]),
                        "ref_end": int(parts[4]),
                        "ref_id": parts[5],
                        "ref_exons": parts[6]
                    }
                    records.append(record)
    return pd.DataFrame(records)

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
from sklearn.preprocessing import MinMaxScaler#%%
from sklearn.preprocessing import MinMaxScaler

def parse_trmap_file(trmap_path):
    records = []
    with open(trmap_path, 'r') as f:
        current_qry = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # e.g., >MSTRG.15.1 GL000194.1:52950-115195 . 52950-115195
                current_qry = line.split()[0][1:]  # remove '>'
            elif line and not line.startswith("#"):
                parts = line.split("\t")
                if len(parts) >= 7 and current_qry:
                    record = {
                        "qry_id": current_qry,
                        "class_code": parts[0],
                        "ref_chr": parts[1],
                        "ref_strand": parts[2],
                        "ref_start": int(parts[3]),
                        "ref_end": int(parts[4]),
                        "ref_id": parts[5],
                        "ref_exons": parts[6]
                    }
                    records.append(record)
    return pd.DataFrame(records)

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.stats.multitest import multipletests
from matplotlib import rcParams
from scipy.stats import pearsonr
from scipy.stats import linregress
from collections import defaultdict


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
SR_tmm = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/quantification/SR_transcript_GeTMM.txt', sep='\t', index_col=0)
LR_tmm = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/quantification/LR_TMM_transcript.txt', sep='\t', index_col=0)

from sklearn.preprocessing import MinMaxScaler
def parse_trmap_file(trmap_path):
    records = []
    with open(trmap_path, 'r') as f:
        current_qry = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # e.g., >MSTRG.15.1 GL000194.1:52950-115195 . 52950-115195
                current_qry = line.split()[0][1:]  # remove '>'
            elif line and not line.startswith("#"):
                parts = line.split("\t")
                if len(parts) >= 7 and current_qry:
                    record = {
                        "qry_id": current_qry,
                        "class_code": parts[0],
                        "ref_chr": parts[1],
                        "ref_strand": parts[2],
                        "ref_start": int(parts[3]),
                        "ref_end": int(parts[4]),
                        "ref_id": parts[5],
                        "ref_exons": parts[6]
                    }
                    records.append(record)
    return pd.DataFrame(records)

# %%
trmap = parse_trmap_file("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/STAR_SR_LR_trmap.txt")

#%%
###^^ sample마다 SR/LR 둘 다 filtering ############

cov = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/quantification/SR_238_transcript_coverage.txt', sep='\t', index_col=0)
info = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/SR_transcriptinfo.txt', sep='\t', index_col=0)

def filter_by_coverage(coverage_df, min_coverage=1, min_fraction=0.1):
    num_samples = coverage_df.shape[1]
    required_count = int(num_samples * min_fraction)

    # 각 transcript별로 coverage ≥ min_coverage 인 sample 수 세기
    passing_counts = (coverage_df >= min_coverage).sum(axis=1)

    # 조건을 만족하는 transcript ID만 선택
    filtered = coverage_df.index[passing_counts >= required_count]
    return filtered

filtered_cov = filter_by_coverage(cov, min_coverage=1, min_fraction=0.1)
filtered_exon = info[info['exon_count'] >= 2].index

both_trans = set(filtered_cov.str.split("-", n=1).str[0]).intersection(set(filtered_exon))

# 2. Class priority dictionary
priority = {'=': 1, 'c': 2, 'k': 3, 'm': 4, 'n': 5, 'j': 6, 'e': 7, 'o': 8, 's': 9, 'x': 10, 'i': 11, 'y': 12, 'p': 13, 'r': 14, 'u': 15}

# 3. Sample-wise class count collector
result_list = []

for sample in SR_tmm.columns:
    sr_expr = SR_tmm[sample]
    lr_expr = LR_tmm[sample]
    
    sr_detected = sr_expr[sr_expr > 0].index
    lr_detected = lr_expr[lr_expr > 0].index
    
    # #####exon + coverage 추가 필터링 #####
    sr_detected = set(sr_detected).intersection(both_trans)  #filtered_exon / both_trans
    # ###################################

    # 1. trmap에서 SR, LR 발현 > 0 인 경우만 필터링
    trmap_sample = trmap[trmap['qry_id'].isin(sr_detected) & trmap['ref_id'].isin(lr_detected)].copy()
    
    # 2. 우선순위 적용: qry_id별로 가장 높은 priority class_code만 남기기
    def get_priority(code):
        return priority.get(code, 999)  # Unranked class_codes get lowest priority

    trmap_sample['priority'] = trmap_sample['class_code'].apply(get_priority)
    trmap_best = trmap_sample.sort_values('priority').drop_duplicates('qry_id', keep='first')

    # 3. 매치되지 않은 SR transcript는 Unmatched로 분류
    matched_qry_ids = set(trmap_best['qry_id'])
    unmatched_qry_ids = set(sr_detected) - matched_qry_ids

    # 4. class count
    class_count = defaultdict(int)

    for code in trmap_best['class_code']:
        class_count[code] += 1

    class_count['Unmatched'] = len(unmatched_qry_ids)

    # 5. 결과 저장
    row = {'Sample': sample}
    for code in ['=', 'c', 'k', 'm', 'n', 'j', 'e', 'o', 's','x','i','y','p','r','u','Unmatched']:
        row[code] = class_count[code]
    result_list.append(row)

# 6. Make final DataFrame
class_df = pd.DataFrame(result_list)
class_df.set_index("Sample", inplace=True)


#%%
###^^ add correlation values #############

from scipy.stats import pearsonr

SR_data_dict = {}
LR_data_dict = {}

for sample in SR_tmm.columns:
    sr_expr = SR_tmm[sample]
    lr_expr = LR_tmm[sample]

    # (1) SR transcript: 발현 > 0 & coverage+exon 조건 만족
    sr_detected = sr_expr[sr_expr > 0].index
    sr_detected = set(sr_detected).intersection(both_trans)
    lr_detected = lr_expr[lr_expr > 0].index
    
    # (2) trmap 필터링
# 1. trmap에서 SR, LR 발현 > 0 인 경우만 필터링
    trmap_sample = trmap[trmap['qry_id'].isin(sr_detected) & trmap['ref_id'].isin(lr_detected)].copy()
    
    def get_priority(code):
        return priority.get(code, 999)  # Unranked class_codes get lowest priority
    trmap_sample['priority'] = trmap_sample['class_code'].apply(get_priority)
    trmap_best = trmap_sample.sort_values('priority').drop_duplicates('qry_id', keep='first')

    # (3) '=' class만 필터
    trmap_eq = trmap_best[trmap_best['class_code'] == '='].copy()

    # (4) SR/LR 추출
    sr_ids = trmap_eq['qry_id'].tolist()
    lr_ids = trmap_eq['ref_id'].tolist()

    # 해당 샘플에서 LR 발현량 0인 transcript 제거
    lr_values = lr_expr.loc[lr_ids]
    nonzero_mask = lr_values > 0

    sr_ids_filtered = [sr for sr, keep in zip(sr_ids, nonzero_mask) if keep]
    lr_ids_filtered = [lr for lr, keep in zip(lr_ids, nonzero_mask) if keep]

    if sr_ids_filtered and lr_ids_filtered:
        sr_vals = SR_tmm.loc[sr_ids_filtered, sample].to_frame()
        lr_vals = LR_tmm.loc[lr_ids_filtered, sample].to_frame()
        sr_vals.index = sr_ids_filtered
        lr_vals.index = sr_ids_filtered  # SR과 동일한 index로 맞춤

        SR_data_dict[sample] = sr_vals
        LR_data_dict[sample] = lr_vals

r_detailed = []

for sample in SR_data_dict:
    sr_df = SR_data_dict[sample]
    lr_df = LR_data_dict[sample]

    # 샘플별 대응 관계: index = SR transcript, value = 대응되는 LR transcript
    trmap_sample = dict(zip(sr_df.index, lr_df.index))

    # 전체 SR transcript 사용
    sr_ids = sr_df.index.tolist()
    lr_ids = [trmap_sample[tx] for tx in sr_ids if tx in trmap_sample]

    # transcript 수 > 2개일 때만 계산
    if len(sr_ids) > 2 and len(lr_ids) == len(sr_ids):
        x = sr_df.loc[sr_ids].values.flatten()
        y = lr_df.loc[lr_ids].values.flatten()

        x_log = np.log2(x)
        y_log = np.log2(y)

        if len(x_log) > 2:
            r, p = pearsonr(x_log, y_log)
            group = 'Normal' if '-N' in sample else 'Tumor'
            r_detailed.append({
                'Sample': sample,
                'r': r,
                'p': p,
                'Group': group
            })

r_df_long = pd.DataFrame(r_detailed)
r_df_whole = r_df_long.copy()


#%%
# ^^=== Plotting ===
r_df_whole = r_df_long.set_index("Sample")
sample_order = class_df.sort_values('=', ascending=False).index.tolist()
class_df_sorted = class_df.loc[sample_order]
cor_df_sorted = r_df_whole.loc[r_df_whole.index.intersection(sample_order)]
cor_df_sorted = cor_df_sorted.reindex(sample_order)

# Plot
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14, 8), 
    gridspec_kw={'height_ratios': [4, 1]}, 
    sharex=True,
    constrained_layout=True
)

type_color = {'=': "#68A5FA",'c':"#52D623",'k':"#B771E5",'m':"#FF7F24",'n':"#CC5900",'j':"#FFD93D",'e':"#FF7383",'o':"#C87CF7",'s':"#FF0000",'x':"#89CE61",'i':"#00FFFF",'y':"#0000FF",'p':"#800080",'r':"#006200",'u':"#808080",'Unmatched': "#FF6B6B"}


# type_color = {'=': "#A0D81E",'c':"#FA5893",'k':"#ebdc97",'m':"#f5bebe",'n':"#2281C1",'j':"#f88434",'e':"#FF7383",'o':"#C87CF7",'s':"#FF0000",'x':"#FF00FF",'i':"#00FFFF",'y':"#0000FF",'p':"#800080",'r':"#008000",'u':"#808080",'Unmatched': "#39A5DF"}

class_df_sorted.plot(
    kind='bar',
    stacked=True,
    figsize=(14, 5),
    
    color=type_color,  # 여기!
    width=1,
    edgecolor='none',
    ax=ax1
)

ax1.set_ylabel("")
ax1.set(xticklabels=[])
ax1.set_xticks([])
ax1.legend(
    loc="upper left",       
    bbox_to_anchor=(1.03, 1),  
    borderaxespad=0,
    frameon=False
)

# 2. 아래쪽: Pearson correlation
ax2.plot(
    cor_df_sorted.index, 
    cor_df_sorted['r'], 
    color='black', marker='o', markersize=2.5, linewidth=0.4 # width=1, edgecolor='none'
)
ax2.set_ylabel("")
ax2.set_xticks([])
ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax2.set_xlabel("Sample")
ax2.set_ylim(0.39, 0.61)
ax2.set_yticks([0.4, 0.5, 0.6])


# Layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.08)
sns.despine()
#plt.savefig("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/figures/SR_LR_both_barplot.pdf", bbox_inches='tight', dpi=300)
plt.show()

#%%
####^^ mean count for each class #######
classes = ['=', 'c', 'Unmatched']

# 1–3) Mean count and mean proportion per class
counts_mean = class_df[classes].mean()

total_per_sample = class_df.sum(axis=1).replace(0, np.nan)
proportions = class_df[classes].div(total_per_sample, axis=0)
proportions_mean = proportions.mean()

summary = pd.DataFrame({
    'count_mean': counts_mean,
    'proportion_mean': proportions_mean
}).loc[classes]

print(summary)

# 4) Mean Pearson r
r_mean = r_df_whole['r'].mean()
print("\nmean_r:", r_mean)

#%%
#%%
#####^^^^^^^ regression boxplot ##########################
from collections import defaultdict
# -------------------------------------------------------
# coding / noncoding classification
# -------------------------------------------------------
cpat_df = pd.read_csv(
    "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/quantification/CPAT/cpat.ORF_prob.best.tsv",
    sep="\t", index_col=0
)
cpc_df = pd.read_csv(
    "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/quantification/CPC2/cpc2output.txt",
    sep="\t", index_col=0
)

cpat_nc = cpat_df[cpat_df['Coding_prob'] < 0.364].index.to_list()
cpc_nc = cpc_df[cpc_df['label'] == 'noncoding'].index.to_list()
nc_trans = set(cpat_nc).union(set(cpc_nc))

r_detailed = []

# SR_data_dict, LR_data_dict는 위 filtering 코드에서 이미 채워진 상태
for sample in SR_data_dict:
    sr_df = SR_data_dict[sample]
    lr_df = LR_data_dict[sample]

    # 샘플별 대응 관계: index = SR transcript, value = 대응되는 LR transcript
    trmap_sample = dict(zip(sr_df.index, lr_df.index))

    for tx_type in ['coding', 'noncoding']:
        if tx_type == 'noncoding':
            sr_ids = [tx for tx in sr_df.index if tx in nc_trans]
        else:
            sr_ids = [tx for tx in sr_df.index if tx not in nc_trans]

        # 그에 대응되는 LR transcript id
        lr_ids = [trmap_sample[tx] for tx in sr_ids if tx in trmap_sample]

        # transcript 수 > 2개일 때만 계산
        if len(sr_ids) > 2 and len(lr_ids) == len(sr_ids):
            x = sr_df.loc[sr_ids].values.flatten()
            y = lr_df.loc[lr_ids].values.flatten()

            x_log = np.log2(x)
            y_log = np.log2(y)

            if len(x_log) > 2:
                r, p = pearsonr(x_log, y_log)
                group = 'Normal' if '-N' in sample else 'Tumor'
                r_detailed.append({
                    'Sample': sample,
                    'r': r,
                    'p': p,
                    'TranscriptType': tx_type,
                    'Group': group
                })

r_df_long = pd.DataFrame(r_detailed)
r_df_whole = r_df_long.copy()
r_df_whole['Group'] = 'Whole'

# 세 그룹 결합
r_df_all = pd.concat([r_df_whole, r_df_long], ignore_index=True)

# -------------------------------------------------------
# Boxplot + p-value annotation
# -------------------------------------------------------
plt.figure(figsize=(6, 5))
ax = sns.boxplot(
    data=r_df_all,
    x='Group',
    y='r',
    hue='TranscriptType',
    palette={'coding': '#FF8F00', 'noncoding': '#AF47D2'},
)

ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    borderaxespad=0,
    frameon=False
)

from statannotations.Annotator import Annotator

pairs = [
    (("Whole", "coding"), ("Whole", "noncoding")),
    (("Normal", "coding"), ("Normal", "noncoding")),
    (("Tumor", "coding"), ("Tumor", "noncoding")),
]

annotator = Annotator(ax, pairs, data=r_df_all, x="Group", y="r", hue="TranscriptType")
annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=1)
annotator.apply_and_annotate()

plt.ylim(0.26, 0.64)
plt.ylabel("Pearson r")
plt.xlabel('')
sns.despine()
plt.tight_layout()
plt.savefig(
    "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/figures/final_sampler_SRLRboth_cnc.pdf",
    bbox_inches='tight', dpi=300
)
plt.show()


#%%








# %%
####^^ LR은 전체로: FINAL !!!!!!!!!! #################
#%%
from sklearn.preprocessing import MinMaxScaler
def parse_trmap_file(trmap_path):
    records = []
    with open(trmap_path, 'r') as f:
        current_qry = None
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # e.g., >MSTRG.15.1 GL000194.1:52950-115195 . 52950-115195
                current_qry = line.split()[0][1:]  # remove '>'
            elif line and not line.startswith("#"):
                parts = line.split("\t")
                if len(parts) >= 7 and current_qry:
                    record = {
                        "qry_id": current_qry,
                        "class_code": parts[0],
                        "ref_chr": parts[1],
                        "ref_strand": parts[2],
                        "ref_start": int(parts[3]),
                        "ref_end": int(parts[4]),
                        "ref_id": parts[5],
                        "ref_exons": parts[6]
                    }
                    records.append(record)
    return pd.DataFrame(records)

# %%

cov = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/quantification/SR_238_transcript_coverage.txt', sep='\t', index_col=0)
info = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/SR_transcriptinfo.txt', sep='\t', index_col=0)


def filter_by_coverage(coverage_df, min_coverage=1, min_fraction=0.1):
    num_samples = coverage_df.shape[1]
    required_count = int(num_samples * min_fraction)

    # 각 transcript별로 coverage ≥ min_coverage 인 sample 수 세기
    passing_counts = (coverage_df >= min_coverage).sum(axis=1)

    # 조건을 만족하는 transcript ID만 선택
    filtered = coverage_df.index[passing_counts >= required_count]
    return filtered

filtered_cov = filter_by_coverage(cov, min_coverage=1, min_fraction=0.1)
filtered_exon = info[info['exon_count'] >= 2].index

both_trans = set(filtered_cov.str.split("-", n=1).str[0]).intersection(set(filtered_exon))

trmap = parse_trmap_file("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/STAR_SR_LR_trmap.txt")

SR_tmm = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/quantification/SR_transcript_GeTMM.txt', sep='\t', index_col=0)
LR_tmm = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/quantification/LR_TMM_transcript.txt', sep='\t', index_col=0)

SR_tmm.index = SR_tmm.index.str.split("-", n=1).str[0]
LR_tmm.index = LR_tmm.index.str.split("-", n=1).str[0]


priority = {'=': 1, 'c': 2, 'k': 3, 'm': 4, 'n': 5, 'j': 6, 'e': 7, 'o': 8, 's': 9, 'x': 10, 'i': 11, 'y': 12, 'p': 13, 'r': 14, 'u': 15}

# 샘플 개수 기반으로 전역적으로 발현된 LR transcript 필터링
lr_detected_binary = (LR_tmm > 0).astype(int)
lr_detected_freq = lr_detected_binary.sum(axis=1) / LR_tmm.shape[1]
valid_lr_transcripts = set(lr_detected_freq[lr_detected_binary.sum(axis=1) >= 24].index) ##### 1명 이상??

from collections import defaultdict

# 결과 저장 리스트
result_list = []

for sample in SR_tmm.columns:
    sr_expr = SR_tmm[sample]
    lr_expr = LR_tmm[sample]
    
    sr_detected = sr_expr[sr_expr > 0].index
    lr_detected = valid_lr_transcripts  # 고정된 ref list 사용

    #####exon + coverage 추가 필터링 #####
    sr_detected = set(sr_detected).intersection(both_trans)  #filtered_exon / both_trans
    ###################################
    
    # trmap 필터링
    trmap_sample = trmap[trmap['qry_id'].isin(sr_detected) & trmap['ref_id'].isin(lr_detected)].copy()
    
    # 우선순위 정렬 및 최고 등급 선택
    def get_priority(code):
        return priority.get(code, 999)
    
    trmap_sample['priority'] = trmap_sample['class_code'].apply(get_priority)
    trmap_best = trmap_sample.sort_values('priority').drop_duplicates('qry_id', keep='first')
    
    # Unmatched 계산
    matched_qry_ids = set(trmap_best['qry_id'])
    unmatched_qry_ids = set(sr_detected) - matched_qry_ids
    
    # 클래스 카운트
    class_count = defaultdict(int)
    for code in trmap_best['class_code']:
        class_count[code] += 1
    class_count['Unmatched'] = len(unmatched_qry_ids)
    
    # 결과 저장
    row = {'Sample': sample}
    for code in ['=', 'c', 'k', 'm', 'n', 'j', 'e', 'o', 's','x','i','y','p','r','u','Unmatched']:
        row[code] = class_count[code]
    result_list.append(row)

# 최종 데이터프레임
class_df = pd.DataFrame(result_list)
class_df.set_index("Sample", inplace=True)

#%%
###^^ add correlation values #############

from scipy.stats import pearsonr

SR_data_dict = {}
LR_data_dict = {}

for sample in SR_tmm.columns:
    sr_expr = SR_tmm[sample]
    lr_expr = LR_tmm[sample]

    # (1) SR transcript: 발현 > 0 & coverage+exon 조건 만족
    sr_detected = sr_expr[sr_expr > 0].index
    sr_detected = set(sr_detected).intersection(both_trans)
    
    # (2) trmap 필터링
    trmap_sample = trmap[
        trmap['qry_id'].isin(sr_detected) &
        trmap['ref_id'].isin(valid_lr_transcripts)
    ].copy()
    
    def get_priority(code):
        return priority.get(code, 999)  # Unranked class_codes get lowest priority
    trmap_sample['priority'] = trmap_sample['class_code'].apply(get_priority)
    trmap_best = trmap_sample.sort_values('priority').drop_duplicates('qry_id', keep='first')

    # (3) '=' class만 필터
    trmap_eq = trmap_best[trmap_best['class_code'] == '='].copy()

    # (4) SR/LR 추출
    sr_ids = trmap_eq['qry_id'].tolist()
    lr_ids = trmap_eq['ref_id'].tolist()

    # 해당 샘플에서 LR 발현량 0인 transcript 제거
    lr_values = lr_expr.loc[lr_ids]
    nonzero_mask = lr_values > 0

    sr_ids_filtered = [sr for sr, keep in zip(sr_ids, nonzero_mask) if keep]
    lr_ids_filtered = [lr for lr, keep in zip(lr_ids, nonzero_mask) if keep]

    if sr_ids_filtered and lr_ids_filtered:
        sr_vals = SR_tmm.loc[sr_ids_filtered, sample].to_frame()
        lr_vals = LR_tmm.loc[lr_ids_filtered, sample].to_frame()
        sr_vals.index = sr_ids_filtered
        lr_vals.index = sr_ids_filtered  # SR과 동일한 index로 맞춤

        SR_data_dict[sample] = sr_vals
        LR_data_dict[sample] = lr_vals

r_detailed = []

for sample in SR_data_dict:
    sr_df = SR_data_dict[sample]
    lr_df = LR_data_dict[sample]

    # 샘플별 대응 관계: index = SR transcript, value = 대응되는 LR transcript
    trmap_sample = dict(zip(sr_df.index, lr_df.index))

    # 전체 SR transcript 사용
    sr_ids = sr_df.index.tolist()
    lr_ids = [trmap_sample[tx] for tx in sr_ids if tx in trmap_sample]

    # transcript 수 > 2개일 때만 계산
    if len(sr_ids) > 2 and len(lr_ids) == len(sr_ids):
        x = sr_df.loc[sr_ids].values.flatten()
        y = lr_df.loc[lr_ids].values.flatten()

        x_log = np.log2(x)
        y_log = np.log2(y)

        if len(x_log) > 2:
            r, p = pearsonr(x_log, y_log)
            group = 'Normal' if '-N' in sample else 'Tumor'
            r_detailed.append({
                'Sample': sample,
                'r': r,
                'p': p,
                'Group': group
            })

r_df_long = pd.DataFrame(r_detailed)
r_df_whole = r_df_long.copy()



#%%
# ^^=== Plotting ===
r_df_whole = r_df_long.set_index("Sample")
sample_order = class_df.sort_values('=', ascending=False).index.tolist()
class_df_sorted = class_df.loc[sample_order]
cor_df_sorted = r_df_whole.loc[r_df_whole.index.intersection(sample_order)]
cor_df_sorted = cor_df_sorted.reindex(sample_order)

# Plot
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14, 8), 
    gridspec_kw={'height_ratios': [4, 1]}, 
    sharex=True,
    constrained_layout=True
)

type_color = {'=': "#68A5FA",'c':"#52D623",'k':"#B771E5",'m':"#FF7F24",'n':"#CC5900",'j':"#FFD93D",'e':"#FF7383",'o':"#C87CF7",'s':"#FF0000",'x':"#89CE61",'i':"#00FFFF",'y':"#0000FF",'p':"#800080",'r':"#006200",'u':"#808080",'Unmatched': "#FF6B6B"}


# type_color = {'=': "#A0D81E",'c':"#FA5893",'k':"#ebdc97",'m':"#f5bebe",'n':"#2281C1",'j':"#f88434",'e':"#FF7383",'o':"#C87CF7",'s':"#FF0000",'x':"#FF00FF",'i':"#00FFFF",'y':"#0000FF",'p':"#800080",'r':"#008000",'u':"#808080",'Unmatched': "#39A5DF"}

class_df_sorted.plot(
    kind='bar',
    stacked=True,
    figsize=(14, 5),
    
    color=type_color,  # 여기!
    width=1,
    edgecolor='none',
    ax=ax1
)

ax1.set_ylabel("")
ax1.set(xticklabels=[])
ax1.set_xticks([])
ax1.legend(
    loc="upper left",       
    bbox_to_anchor=(1.03, 1),  
    borderaxespad=0,
    frameon=False
)

# 2. 아래쪽: Pearson correlation
ax2.plot(
    cor_df_sorted.index, 
    cor_df_sorted['r'], 
    color='black', marker='o', markersize=2.5, linewidth=0.4 # width=1, edgecolor='none'
)
ax2.set_ylabel("")
ax2.set_xlabel("Sample")
ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax2.set_ylim(0.39, 0.65)
ax2.set_yticks([0.4, 0.5, 0.6])


# Layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.08)
sns.despine()
plt.savefig("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/figures/SR_LRwhole_barplot_zero.pdf", bbox_inches='tight', dpi=300)
plt.show()

#%%
####^^ mean count for each class #######
classes = ['=', 'c', 'Unmatched']

# 1–3) Mean count and mean proportion per class
counts_mean = class_df[classes].mean()

total_per_sample = class_df.sum(axis=1).replace(0, np.nan)
proportions = class_df[classes].div(total_per_sample, axis=0)
proportions_mean = proportions.mean()

summary = pd.DataFrame({
    'count_mean': counts_mean,
    'proportion_mean': proportions_mean
}).loc[classes]

print(summary)

# 4) Mean Pearson r
r_mean = r_df_whole['r'].mean()
print("\nmean_r:", r_mean)


#%%
#####^^^^^^^ coding/noncoding boxplot ##########################
from collections import defaultdict

SR_data_dict = {}
LR_data_dict = {}

cpat_df = pd.read_csv("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/quantification/CPAT/cpat.ORF_prob.best.tsv", sep="\t", index_col=0)
cpc_df = pd.read_csv("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/quantification/CPC2/cpc2output.txt",   sep="\t", index_col=0)
cpat_nc = cpat_df[cpat_df['Coding_prob']<0.364].index.to_list()
cpc_nc = cpc_df[cpc_df['label']=='noncoding'].index.to_list()
nc_trans = set(cpat_nc).union(set(cpc_nc))
nc_trans = {}

for sample in SR_tmm.columns:
    sr_expr = SR_tmm[sample]
    lr_expr = LR_tmm[sample]

    # (1) SR transcript: 발현 > 0 & coverage+exon 조건 만족
    sr_detected = sr_expr[sr_expr > 0].index
    sr_detected = set(sr_detected).intersection(both_trans)
    
    # (2) trmap 필터링
    trmap_sample = trmap[
        trmap['qry_id'].isin(sr_detected) &
        # ~trmap['qry_id'].isin(nc_trans) &
        trmap['ref_id'].isin(valid_lr_transcripts)
    ].copy()

    trmap_sample['priority'] = trmap_sample['class_code'].apply(get_priority)
    trmap_best = trmap_sample.sort_values('priority').drop_duplicates('qry_id', keep='first')

    # (3) '=' class만 필터
    trmap_eq = trmap_best[trmap_best['class_code'] == '='].copy()

    # (4) SR/LR 추출
    sr_ids = trmap_eq['qry_id'].tolist()
    lr_ids = trmap_eq['ref_id'].tolist()

    # 해당 샘플에서 LR 발현량 0인 transcript 제거
    lr_values = lr_expr.loc[lr_ids]
    nonzero_mask = lr_values > 0

    sr_ids_filtered = [sr for sr, keep in zip(sr_ids, nonzero_mask) if keep]
    lr_ids_filtered = [lr for lr, keep in zip(lr_ids, nonzero_mask) if keep]

    if sr_ids_filtered and lr_ids_filtered:
        sr_vals = SR_tmm.loc[sr_ids_filtered, sample].to_frame()
        lr_vals = LR_tmm.loc[lr_ids_filtered, sample].to_frame()
        sr_vals.index = sr_ids_filtered
        lr_vals.index = sr_ids_filtered  # SR과 동일한 index로 맞춤

        SR_data_dict[sample] = sr_vals
        LR_data_dict[sample] = lr_vals


sample_groups = {
    'Whole': list(SR_data_dict.keys()),
    'Normal': [s for s in SR_data_dict if '-N' in s],
    'Tumor':  [s for s in SR_data_dict if '-T' in s]
}

r_results = []

for group, sample_list in sample_groups.items():
    for sample in sample_list:
        if sample in SR_data_dict and sample in LR_data_dict:
            x = SR_data_dict[sample].values.flatten()
            y = LR_data_dict[sample].values.flatten()
            if len(x) > 2:
                x_log = np.log2(x)
                y_log = np.log2(y)
                r, p = pearsonr(x_log, y_log)
                r_results.append({
                    'Sample': sample,
                    'r': r,
                    'p': p,
                    'N': len(x),
                    'Group': group
                })

r_df = pd.DataFrame(r_results)

plt.figure(figsize=(4,5))
sns.boxplot(data=r_df, x='Group', y='r', palette={'Whole':'#525252','Normal':'#366CB0','Tumor':'#F0027F'}, showfliers=False, width=0.8)
sns.stripplot(data=r_df, x='Group', y='r', palette={'Whole':'#525252','Normal':'#366CB0','Tumor':'#F0027F'}, alpha=0.2, jitter=0.2, size=3)
plt.xlabel('')

# sns.boxplot(data=r_df[r_df['Group']=='Whole'], x='Group', y='r', palette={'Whole':'#525252'}, showfliers=False)
# sns.stripplot(data=r_df[r_df['Group']=='Whole'], x='Group', y='r', palette={'Whole':'#525252','Normal':'#366CB0'}, alpha=0.2, jitter=0.2, size=3)
# plt.xlabel('')
# plt.xticks([])

plt.ylabel('Pearson r')
plt.ylim(0.42,0.63)
#plt.title("Pearson r by Group")
sns.despine()
plt.tight_layout()
plt.savefig("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/figures/final_sampler_SRLRboth.pdf", bbox_inches='tight', dpi=300)
plt.show()

########^^^^^^ coding/noncoding box plot#######################
cpat_df = pd.read_csv("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/quantification/CPAT/cpat.ORF_prob.best.tsv", sep="\t", index_col=0)
cpc_df = pd.read_csv("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/quantification/CPC2/cpc2output.txt",   sep="\t", index_col=0)
cpat_nc = cpat_df[cpat_df['Coding_prob']<0.364].index.to_list()
cpc_nc = cpc_df[cpc_df['label']=='noncoding'].index.to_list()
nc_trans = set(cpat_nc).union(set(cpc_nc))

r_detailed = []

for sample in SR_data_dict:
    sr_df = SR_data_dict[sample]
    lr_df = LR_data_dict[sample]

    # 샘플별로 대응 관계: index = SR transcript, value = 대응되는 LR transcript
    trmap_sample = dict(zip(sr_df.index, lr_df.index))

    for tx_type in ['coding', 'noncoding']:
        if tx_type == 'noncoding':
            sr_ids = [tx for tx in sr_df.index if tx in nc_trans]
        else:
            sr_ids = [tx for tx in sr_df.index if tx not in nc_trans]

        # 그에 대응되는 LR transcript id
        lr_ids = [trmap_sample[tx] for tx in sr_ids if tx in trmap_sample]

        # transcript 수 > 2개일 때만 계산
        if len(sr_ids) > 2 and len(lr_ids) == len(sr_ids):
            x = sr_df.loc[sr_ids].values.flatten()
            y = lr_df.loc[lr_ids].values.flatten()

            x_log = np.log2(x)
            y_log = np.log2(y)

            if len(x_log) > 2:
                r, p = pearsonr(x_log, y_log)
                group = 'Normal' if '-N' in sample else 'Tumor'
                r_detailed.append({
                    'Sample': sample,
                    'r': r,
                    'p': p,
                    'TranscriptType': tx_type,
                    'Group': group
                })

r_df_long = pd.DataFrame(r_detailed)
r_df_whole = r_df_long.copy()
r_df_whole['Group'] = 'Whole'

# 세 그룹 결합
r_df_all = pd.concat([r_df_whole,r_df_long], ignore_index=True)

plt.figure(figsize=(6,5))
ax = sns.boxplot(
    data=r_df_all,
    x='Group',
    y='r',
    hue='TranscriptType',
    #split=True,
    palette={'coding': '#FF8F00', 'noncoding': '#AF47D2'},
)

ax.legend(
    loc="upper left",       
    bbox_to_anchor=(1.02, 1),  
    borderaxespad=0,
    frameon=False
)
from statannotations.Annotator import Annotator


pairs = [
    (("Whole", "coding"), ("Whole", "noncoding")),
    (("Normal", "coding"), ("Normal", "noncoding")),
    (("Tumor", "coding"), ("Tumor", "noncoding")),
]

annotator = Annotator(ax, pairs, data=r_df_all, x="Group", y="r", hue="TranscriptType")
annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=1)
annotator.apply_and_annotate()


plt.ylim(0.26, 0.64)
plt.ylabel("Pearson r")
plt.xlabel('')
sns.despine()
plt.tight_layout()
plt.savefig("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/figures/final_sampler_SRLRboth_cnc.pdf.pdf", bbox_inches='tight', dpi=300)
plt.show()



#%%
#######********** no nonzero mask (including zero values) ###

###^^ add correlation values #############

from scipy.stats import pearsonr

SR_data_dict = {}
LR_data_dict = {}

for sample in SR_tmm.columns:
    sr_expr = SR_tmm[sample]
    lr_expr = LR_tmm[sample]

    # (1) SR transcript: 발현 > 0 & coverage+exon 조건 만족
    sr_detected = sr_expr[sr_expr > 0].index
    sr_detected = set(sr_detected).intersection(both_trans)
    
    # (2) trmap 필터링
    trmap_sample = trmap[
        trmap['qry_id'].isin(sr_detected) &
        trmap['ref_id'].isin(valid_lr_transcripts)
    ].copy()
    
    def get_priority(code):
        return priority.get(code, 999)  # Unranked class_codes get lowest priority
    trmap_sample['priority'] = trmap_sample['class_code'].apply(get_priority)
    trmap_best = trmap_sample.sort_values('priority').drop_duplicates('qry_id', keep='first')

    # (3) '=' class만 필터
    trmap_eq = trmap_best[trmap_best['class_code'] == '='].copy()

    # (4) SR/LR 추출 (nonzero mask 제거)
    sr_ids = trmap_eq['qry_id'].tolist()
    lr_ids = trmap_eq['ref_id'].tolist()

    if sr_ids and lr_ids:
        sr_vals = SR_tmm.loc[sr_ids, sample].to_frame()
        lr_vals = LR_tmm.loc[lr_ids, sample].to_frame()
        sr_vals.index = sr_ids
        lr_vals.index = sr_ids  # SR과 동일한 index로 맞춤

        SR_data_dict[sample] = sr_vals
        LR_data_dict[sample] = lr_vals

r_detailed = []

for sample in SR_data_dict:
    sr_df = SR_data_dict[sample]
    lr_df = LR_data_dict[sample]

    # 샘플별 대응 관계: index = SR transcript, value = 대응되는 LR transcript
    trmap_sample = dict(zip(sr_df.index, lr_df.index))

    # 전체 SR transcript 사용
    sr_ids = sr_df.index.tolist()
    lr_ids = [trmap_sample[tx] for tx in sr_ids if tx in trmap_sample]

    # transcript 수 > 2개일 때만 계산
    if len(sr_ids) > 2 and len(lr_ids) == len(sr_ids):
        x = sr_df.loc[sr_ids].values.flatten()
        y = lr_df.loc[lr_ids].values.flatten()

        # log2(x + 0.5) 변환
        x_log = np.log2(x + 0.5)
        y_log = np.log2(y + 0.5)

        if len(x_log) > 2:
            r, p = pearsonr(x_log, y_log)
            group = 'Normal' if '-N' in sample else 'Tumor'
            r_detailed.append({
                'Sample': sample,
                'r': r,
                'p': p,
                'Group': group
            })

r_df_long = pd.DataFrame(r_detailed)
r_df_whole = r_df_long.copy()

#%%
# ^^=== Plotting ===
r_df_whole = r_df_long.set_index("Sample")
sample_order = class_df.sort_values('=', ascending=False).index.tolist()
class_df_sorted = class_df.loc[sample_order]
cor_df_sorted = r_df_whole.loc[r_df_whole.index.intersection(sample_order)]
cor_df_sorted = cor_df_sorted.reindex(sample_order)

# Plot
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14, 8), 
    gridspec_kw={'height_ratios': [4, 1]}, 
    sharex=True,
    constrained_layout=True
)

type_color = {'=': "#68A5FA",'c':"#52D623",'k':"#B771E5",'m':"#FF7F24",'n':"#CC5900",'j':"#FFD93D",'e':"#FF7383",'o':"#C87CF7",'s':"#FF0000",'x':"#89CE61",'i':"#00FFFF",'y':"#0000FF",'p':"#800080",'r':"#006200",'u':"#808080",'Unmatched': "#FF6B6B"}


# type_color = {'=': "#A0D81E",'c':"#FA5893",'k':"#ebdc97",'m':"#f5bebe",'n':"#2281C1",'j':"#f88434",'e':"#FF7383",'o':"#C87CF7",'s':"#FF0000",'x':"#FF00FF",'i':"#00FFFF",'y':"#0000FF",'p':"#800080",'r':"#008000",'u':"#808080",'Unmatched': "#39A5DF"}

class_df_sorted.plot(
    kind='bar',
    stacked=True,
    figsize=(14, 5),
    
    color=type_color,  # 여기!
    width=1,
    edgecolor='none',
    ax=ax1
)

ax1.set_ylabel("")
ax1.set(xticklabels=[])
ax1.set_xticks([])
ax1.legend(
    loc="upper left",       
    bbox_to_anchor=(1.03, 1),  
    borderaxespad=0,
    frameon=False
)

# 2. 아래쪽: Pearson correlation
ax2.plot(
    cor_df_sorted.index, 
    cor_df_sorted['r'], 
    color='black', marker='o', markersize=2.5, linewidth=0.4 # width=1, edgecolor='none'
)
ax2.set_ylabel("")
ax2.set_xlabel("Sample")
ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax2.set_ylim(0.39, 0.65)
ax2.set_yticks([0.4, 0.5, 0.6])


# Layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.08)
sns.despine()
#plt.savefig("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/figures/SR_LRwhole_barplot_withzero.pdf", bbox_inches='tight', dpi=300)
plt.show()

#%%
####^^^ plotting with 4 classes ###################


group_map = {
    '=': "Complete match",
    'c': "Partial intron match",
    'k': "Partial intron match",
    'm': "Partial intron match",
    'n': "Partial intron match",
    'j': "Partial intron match",
    'e': "Same strand overlap without junction match",
    'o': "Same strand overlap without junction match"
}
class_groups = class_df_sorted.copy()
class_groups = class_groups.groupby(
    class_groups.columns.map(lambda x: group_map.get(x, "Not matched")),
    axis=1
).sum()

group_colors = {
    "Complete match": "#569CFE",  # 파랑
    "Partial intron match": "#FFD93D",  # 노랑
    "Same strand overlap without junction match": "#DA50EC",  # 분홍
    "Not matched": "#595959"  # 회색
}


order = [
    "Complete match",
    "Partial intron match",
    "Same strand overlap without junction match",
    "Not matched"
]

# Plot
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14, 8), 
    gridspec_kw={'height_ratios': [4, 1]}, 
    sharex=True,
    constrained_layout=True
)

class_groups = class_groups.reindex(columns=order)

class_groups.plot(
    kind='bar',
    stacked=True,
    figsize=(14, 5),
    color=[group_colors[c] for c in class_groups.columns],
    width=1,
    edgecolor='none',
    ax=ax1
)

ax1.set_ylabel("")
ax1.set(xticklabels=[])
ax1.set_xticks([])

# legend 순서 고정
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(
    handles,
    labels,
    loc="upper left",
    bbox_to_anchor=(1.03, 1),
    borderaxespad=0,
    frameon=False
)

# 2. 아래쪽: Pearson correlation
ax2.plot(
    cor_df_sorted.index, 
    cor_df_sorted['r'], 
    color='black', marker='o', markersize=2.5, linewidth=0.4 # width=1, edgecolor='none'
)
ax2.set_ylabel("")
ax2.set_xlabel("Sample")
ax2.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax2.set_ylim(0.39, 0.65)
ax2.set_yticks([0.4, 0.5, 0.6])


# Layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.08)
sns.despine()
#plt.savefig("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/figures/SR_LRwhole_barplot_withzero_4classes.pdf", bbox_inches='tight', dpi=300)
plt.show()

#%%
####** 20251009 supple############################
#=== Plot only Complete + Partial intron match (normalized to 100%) ===

# 1. 샘플 순서 동일
sample_order = class_df.sort_values('=', ascending=False).index.tolist()
class_df_sorted = class_df.loc[sample_order]

# 2. complete match + partial intron match만 남기기 (=, c, k, m, n, j)
subset_cols = ['=', 'c', 'k', 'm', 'n', 'j']
subset_df = class_df_sorted[subset_cols].copy()

# 3. 각 샘플별 합계로 나누어 비율 계산 (100%로 정규화)
subset_df = subset_df.div(subset_df.sum(axis=1), axis=0) * 100  # <-- 비율(%)

# 4. 색상 정의
type_color = {
    '=': "#68A5FA",  # complete match
    'c': "#52D623",
    'k': "#B771E5",
    'm': "#FF7F24",
    'n': "#CC5900",
    'j': "#FFD93D"
}

# 5. 퍼센티지 기반 stacked barplot
fig, ax = plt.subplots(figsize=(12, 4))
subset_df.plot(
    kind='bar',
    stacked=True,
    color=[type_color[c] for c in subset_df.columns],
    width=1,
    edgecolor='none',
    ax=ax
)

ax.set_ylabel("Proportion (%)")
ax.set_xlabel("Sample")
ax.set(xticklabels=[])
ax.set_xticks([])

# y축 0~100, tick label 퍼센티지로
ax.set_ylim(0, 100)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_yticklabels([f"{y}%" for y in ax.get_yticks()])

ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.03, 1),
    borderaxespad=0,
    frameon=False,
    title="Class"
)

sns.despine()
plt.tight_layout()
plt.savefig("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/figures/SR_LR_partial_complete_ratio_barplot_percent.pdf", bbox_inches='tight', dpi=300)
plt.show()

#%%
####^^ mean count for each class #######
classes = ['=', 'c', 'Unmatched']

# 1–3) Mean count and mean proportion per class
counts_mean = class_df[classes].mean()

total_per_sample = class_df.sum(axis=1).replace(0, np.nan)
proportions = class_df[classes].div(total_per_sample, axis=0)
proportions_mean = proportions.mean()

summary = pd.DataFrame({
    'count_mean': counts_mean,
    'proportion_mean': proportions_mean
}).loc[classes]

print(summary)

# 4) Mean Pearson r
r_mean = r_df_whole['r'].mean()
print("\nmean_r:", r_mean)

#%%
#####^^^^^^^ coding/noncoding boxplot ##########################
from collections import defaultdict

SR_data_dict = {}
LR_data_dict = {}

cpat_df = pd.read_csv(
    "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/quantification/CPAT/cpat.ORF_prob.best.tsv",
    sep="\t", index_col=0
)
cpc_df = pd.read_csv(
    "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/quantification/CPC2/cpc2output.txt",
    sep="\t", index_col=0
)
cpat_nc = cpat_df[cpat_df['Coding_prob'] < 0.364].index.to_list()
cpc_nc = cpc_df[cpc_df['label'] == 'noncoding'].index.to_list()
nc_trans = set(cpat_nc).union(set(cpc_nc))
nc_trans = {}   # coding/noncoding 구분은 아래 violin plot에서 사용

for sample in SR_tmm.columns:
    sr_expr = SR_tmm[sample]
    lr_expr = LR_tmm[sample]

    # (1) SR transcript: 발현 > 0 & coverage+exon 조건 만족
    sr_detected = sr_expr[sr_expr > 0].index
    sr_detected = set(sr_detected).intersection(both_trans)
    
    # (2) trmap 필터링
    trmap_sample = trmap[
        trmap['qry_id'].isin(sr_detected) &
        trmap['ref_id'].isin(valid_lr_transcripts)
    ].copy()

    trmap_sample['priority'] = trmap_sample['class_code'].apply(get_priority)
    trmap_best = trmap_sample.sort_values('priority').drop_duplicates('qry_id', keep='first')

    # (3) '=' class만 필터
    trmap_eq = trmap_best[trmap_best['class_code'] == '='].copy()

    # (4) SR/LR 추출 (nonzero mask 제거)
    sr_ids = trmap_eq['qry_id'].tolist()
    lr_ids = trmap_eq['ref_id'].tolist()

    if sr_ids and lr_ids:
        sr_vals = SR_tmm.loc[sr_ids, sample].to_frame()
        lr_vals = LR_tmm.loc[lr_ids, sample].to_frame()
        sr_vals.index = sr_ids
        lr_vals.index = sr_ids  # SR과 동일한 index로 맞춤

        SR_data_dict[sample] = sr_vals
        LR_data_dict[sample] = lr_vals


sample_groups = {
    'Whole': list(SR_data_dict.keys()),
    'Normal': [s for s in SR_data_dict if '-N' in s],
    'Tumor':  [s for s in SR_data_dict if '-T' in s]
}

#%%
r_results = []

for group, sample_list in sample_groups.items():
    for sample in sample_list:
        if sample in SR_data_dict and sample in LR_data_dict:
            x = SR_data_dict[sample].values.flatten()
            y = LR_data_dict[sample].values.flatten()
            if len(x) > 2:
                x_log = np.log2(x + 0.5)
                y_log = np.log2(y + 0.5)
                r, p = pearsonr(x_log, y_log)
                r_results.append({
                    'Sample': sample,
                    'r': r,
                    'p': p,
                    'N': len(x),
                    'Group': group
                })

r_df = pd.DataFrame(r_results)

plt.figure(figsize=(4,5))
sns.boxplot(
    data=r_df, x='Group', y='r',
    palette={'Whole':'#525252','Normal':'#366CB0','Tumor':'#F0027F'},
    showfliers=False, width=0.8
)
sns.stripplot(
    data=r_df, x='Group', y='r',
    palette={'Whole':'#525252','Normal':'#366CB0','Tumor':'#F0027F'},
    alpha=0.2, jitter=0.2, size=3
)
plt.xlabel('')
plt.ylabel('Pearson r')
plt.ylim(0.46, 0.66)
sns.despine()
plt.tight_layout()
plt.savefig(
    "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/figures/final_sampler_SR_LRwhole_boxplot_zero.pdf",
    bbox_inches='tight', dpi=300
)
plt.show()

#%%
########^^^^^^ violin plot (coding / noncoding) #######################
cpat_df = pd.read_csv(
    "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/quantification/CPAT/cpat.ORF_prob.best.tsv",
    sep="\t", index_col=0
)
cpc_df = pd.read_csv(
    "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/quantification/CPC2/cpc2output.txt",
    sep="\t", index_col=0
)
cpat_nc = cpat_df[cpat_df['Coding_prob'] < 0.364].index.to_list()
cpc_nc = cpc_df[cpc_df['label'] == 'noncoding'].index.to_list()
nc_trans = set(cpat_nc).union(set(cpc_nc))

r_detailed = []

for sample in SR_data_dict:
    sr_df = SR_data_dict[sample]
    lr_df = LR_data_dict[sample]

    # 샘플별 대응 관계: index = SR transcript, value = 대응되는 LR transcript
    trmap_sample = dict(zip(sr_df.index, lr_df.index))

    for tx_type in ['coding', 'noncoding']:
        if tx_type == 'noncoding':
            sr_ids = [tx for tx in sr_df.index if tx in nc_trans]
        else:
            sr_ids = [tx for tx in sr_df.index if tx not in nc_trans]

        # 대응되는 LR transcript id
        lr_ids = [trmap_sample[tx] for tx in sr_ids if tx in trmap_sample]

        if len(sr_ids) > 2 and len(lr_ids) == len(sr_ids):
            x = sr_df.loc[sr_ids].values.flatten()
            y = lr_df.loc[lr_ids].values.flatten()

            x_log = np.log2(x + 0.5)
            y_log = np.log2(y + 0.5)

            if len(x_log) > 2:
                r, p = pearsonr(x_log, y_log)
                group = 'Normal' if '-N' in sample else 'Tumor'
                r_detailed.append({
                    'Sample': sample,
                    'r': r,
                    'p': p,
                    'TranscriptType': tx_type,
                    'Group': group
                })

r_df_long = pd.DataFrame(r_detailed)
r_df_whole = r_df_long.copy()
r_df_whole['Group'] = 'Whole'

# 세 그룹 결합
r_df_all = pd.concat([r_df_whole, r_df_long], ignore_index=True)

#%%
plt.figure(figsize=(6,5))
ax = sns.boxplot(
    data=r_df_all,
    x='Group',
    y='r',
    hue='TranscriptType',
    palette={'coding': '#FF8F00', 'noncoding': '#AF47D2'},
)

ax.legend(
    loc="upper left",       
    bbox_to_anchor=(1.02, 1),
    borderaxespad=0,
    frameon=False
)

from statannotations.Annotator import Annotator
pairs = [
    (("Whole", "coding"), ("Whole", "noncoding")),
    (("Normal", "coding"), ("Normal", "noncoding")),
    (("Tumor", "coding"), ("Tumor", "noncoding")),
]
annotator = Annotator(ax, pairs, data=r_df_all, x="Group", y="r", hue="TranscriptType")
annotator.configure(test='t-test_ind', text_format='star', loc='inside', verbose=1)
annotator.apply_and_annotate()

plt.ylim(0.36, 0.67)
plt.ylabel("Pearson r")
plt.xlabel('')
sns.despine()
plt.tight_layout()
plt.savefig(
    "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/figures/final_sampler_SRLRwhole_cnc_zero.pdf",
    bbox_inches='tight', dpi=300
)
plt.show()


# %%
####^^^ regression line plot ###############

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import linregress

def plot_sr_lr_regression_from_dicts(SR_dict, LR_dict, sample_list, title, inputcolor, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    x_all, y_all = [], []

    for sample in sample_list:
        if sample in SR_dict and sample in LR_dict:
            x_vals = SR_dict[sample].values.flatten()
            y_vals = LR_dict[sample].values.flatten()

            # 0/NaN 필터
            valid = (~np.isnan(x_vals)) & (~np.isnan(y_vals)) & (y_vals > 0)
            x = np.log2(x_vals[valid])
            y = np.log2(y_vals[valid])

            if len(x) > 2:
                # scatter
                ax.scatter(x, y, alpha=0.1, color='gray', s=2)

                # 회귀선 (연하게)
                slope, intercept, *_ = linregress(x, y)
                x_line = np.array([x.min(), x.max()])
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, color=inputcolor, alpha=0.05, lw=1)

                # flatten 대상에 추가
                x_all.extend(x)
                y_all.extend(y)

    # 전체 flatten된 회귀선 (진하게)
    if len(x_all) > 2:
        x_all = np.array(x_all)
        y_all = np.array(y_all)
        slope, intercept, r, p, stderr = linregress(x_all, y_all)
        x_line = np.linspace(x_all.min(), x_all.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color=inputcolor, lw=1.5)

        print(f"[{title}] N = {len(x_all)}, r = {r:.3f}, p = {p:.1e}")
        text = f"$r$ = {r:.2f}\n$p$ = {p:.1e}"
        ax.text(-0.5 * x_all.max(), 0.82 * y_all.max(), text, fontsize=11, color='black')

    ax.set_title(title)
    ax.set_ylabel("")
    ax.set_xlabel("Illumina log2(TMM)")
    sns.despine()

# 샘플 그룹 정의
sample_groups = {
    "Whole": list(SR_data_dict.keys()),
    "Normal": [s for s in SR_data_dict if '-N' in s],
    "Tumor":  [s for s in SR_data_dict if '-T' in s]
}

# 플롯
sns.set_style("ticks")
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

plot_sr_lr_regression_from_dicts(SR_data_dict, LR_data_dict, sample_groups["Whole"],  title="Whole",  inputcolor='#000000', ax=axes[0])
plot_sr_lr_regression_from_dicts(SR_data_dict, LR_data_dict, sample_groups["Normal"], title="Normal", inputcolor='#3765A8', ax=axes[1])
plot_sr_lr_regression_from_dicts(SR_data_dict, LR_data_dict, sample_groups["Tumor"],  title="Tumor",  inputcolor='#E04A8F', ax=axes[2])

axes[0].set_ylabel("Nanopore log2(TMM)")
fig.subplots_adjust(wspace=0.1)
plt.savefig("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/figures/final_sampler_bothSRLR_exon_cov_regression_TMM.pdf", bbox_inches='tight', dpi=300)
plt.show()

# %%
####^^ PCA ###########################

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def run_pca_and_plot(expr_df, title, inputcolor_dict=None):
    # 1. 샘플 × transcript로 전치
    df_t = expr_df.T

    # 2. 결측값/0 제거 없이 진행 (전처리 시 선택 가능)
    X_scaled = StandardScaler().fit_transform(df_t)

    # 3. PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 4. 시각화를 위한 메타 정보 추가
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['Sample'] = df_t.index
    pca_df['Group'] = pca_df['Sample'].apply(lambda x: 'Tumor' if '-T' in x else 'Normal')

    # 5. Plot
    plt.figure(figsize=(6,5))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Group', palette=inputcolor_dict, s=40)
    plt.title(f"{title}: PCA by Sample (SR vs LR)")
    plt.legend(title='Group')
    sns.despine()
    plt.tight_layout()
    plt.show()

    return pca_df

# trmap 필터링 (전 샘플 공통 기준 적용)
trmap_filtered = trmap[
    trmap['qry_id'].isin(both_trans) &
    trmap['ref_id'].isin(valid_lr_transcripts)
].copy()

trmap_filtered['priority'] = trmap_filtered['class_code'].apply(priority.get)
trmap_best = trmap_filtered.sort_values('priority').drop_duplicates('qry_id', keep='first')

# 최종 transcript 리스트
selected_sr_tx = trmap_best['qry_id'].tolist()
selected_lr_tx = trmap_best['ref_id'].tolist()

SR_pca_input = SR_tmm.loc[selected_sr_tx].copy()
LR_pca_input = LR_tmm.loc[selected_lr_tx].copy()

# index 통일
SR_pca_input.index = selected_sr_tx
LR_pca_input.index = selected_sr_tx  # ref_id → qry_id 기준으로 맞춤

colors = {'Tumor': '#E04A8F', 'Normal': '#3765A8'}

# PCA 함수는 동일하게 사용
pca_df_sr = run_pca_and_plot(SR_pca_input, "Illumina", inputcolor_dict=colors)
pca_df_lr = run_pca_and_plot(LR_pca_input, "Nanopore", inputcolor_dict=colors)


import umap
sns.set_style("whitegrid")
# 예시: TPM log2 변환 + NaN 처리
def preprocess(data, min_detect=24):
    data = np.log2(data + 1)
    filt = (data > 0).sum(axis=1) >= min_detect  # 최소 N개 샘플에서 발현된 transcript만
    return data.loc[filt]

# SR/LR 정제
SR_proc = preprocess(SR_tmm)
LR_proc = preprocess(LR_tmm)

# Transpose: PCA는 (sample × feature) 형태여야 함
pca_sr = PCA(n_components=2)
SR_pca = pca_sr.fit_transform(SR_proc.T)
print("SR explained variance ratio:", pca_sr.explained_variance_ratio_)

# Long-read
pca_lr = PCA(n_components=2)
LR_pca = pca_lr.fit_transform(LR_proc.T)
print("LR explained variance ratio:", pca_lr.explained_variance_ratio_)

# Optional: UMAP도 가능
# SR_umap = umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=42).fit_transform(SR_proc.T)
# LR_umap = umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=42).fit_transform(LR_proc.T)

# Sample metadata (e.g., tumor/normal)
sample_meta = pd.DataFrame({
    "sample": SR_proc.columns,
    "condition": ["Tumor" if "T" in s else "Normal" for s in SR_proc.columns],  # 필요시 수정
})

#%%
# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, emb, title in zip(axes, [SR_pca, LR_pca], ["Illumina", "Nanopore"]):
    df_plot = pd.DataFrame(emb, columns=["PC1", "PC2"])
    df_plot["condition"] = sample_meta["condition"].values
    sns.scatterplot(data=df_plot, x="PC1", y="PC2", hue="condition", ax=ax, s=50, palette=["#4B74B6", "#E33C89"])
    ax.set_title(f"{title} PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="")
sns.despine()
plt.savefig("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/figures/PCA_normal_tumor.pdf", bbox_inches='tight', dpi=300)
plt.tight_layout()

# %%
######^^^^^^^ length별 barplot##############

type_color = {'=': "#68A5FA",'c':"#52D623",'k':"#B771E5",'m':"#FF7F24",'n':"#CC5900",'j':"#FFD93D",'e':"#FF7383",'o':"#C87CF7",'s':"#FF0000",'x':"#89CE61",'i':"#00FFFF",'y':"#0000FF",'p':"#800080",'r':"#006200",'u':"#808080",'Unmatched': "#FF6B6B"}

covlist = set(filtered_cov.str.split("-", n=1).str[0])

# 2. trmap 우선순위 기반 best match 결정
trmap_filtered = trmap[
    trmap['qry_id'].isin(covlist) &
    trmap['ref_id'].isin(valid_lr_transcripts)
].copy()

trmap_filtered['priority'] = trmap_filtered['class_code'].apply(priority.get)
trmap_best = trmap_filtered.sort_values('priority').drop_duplicates('qry_id', keep='first')

# 3. match label 부여 (priority 기반 class code)
match_class = trmap_best.set_index('qry_id')['class_code'].to_dict()
transcripts = list(covlist)  # 기준은 숏리드 transcript
match_labels = [match_class.get(tx, 'Unmatched') for tx in transcripts]

match_labels_dict = {tx: match_class.get(tx, 'Unmatched') for tx in transcripts}

# 2. info 서브셋
df_plot = info.loc[info.index.isin(transcripts)].copy()

# 3. MatchType 정확히 매핑
df_plot['MatchType'] = df_plot.index.map(match_labels_dict)
df_plot['length_bp'] = df_plot['length_kb'] * 1000


match_order = list(type_color.keys())

# 2. Categorical로 순서 지정
df_plot['MatchType'] = pd.Categorical(df_plot['MatchType'], categories=match_order[::-1], ordered=True)

#######################
df_plot = df_plot[df_plot['exon_count']>=1]
#######################


# 6. 그리기
plt.figure(figsize=(8, 5))
sns.set_style("white")
ax = sns.histplot(data=df_plot, x="exon_count", hue="MatchType", bins = np.linspace(1, 31, 31), multiple="stack", palette=type_color, linewidth=0,  )

plt.xlabel("exon count")
plt.ylabel("transcript count")
ax.legend_.set_title(None)
#plt.xticks([3,6,9,12,15,18,21,24,27,30])
plt.xticks([2,5,8,11,14,17,20,23,26,29])
plt.xlim(1, 32)
ax.get_legend().remove()
sns.despine()
plt.tight_layout()
#plt.savefig("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/figures/exoncount_histogram_fromexon2_proportion.pdf", bbox_inches='tight', dpi=300)
plt.show()

#%%
# ######^^^^^^^ exon count threshold barplot##############

# 1. Threshold 리스트 정의
thresholds = list(range(1, 31, 1))  # 예: 1,3,5,...,29

# 2. MatchType별 count 저장
threshold_data = []

for thres in thresholds:
    df_filtered = df_plot[df_plot['exon_count'] >= thres]
    count_by_type = df_filtered['MatchType'].value_counts()
    for match_type, count in count_by_type.items():
        threshold_data.append({
            'Threshold': thres,
            'MatchType': match_type,
            'Count': count
        })

# 3. DataFrame 생성
df_thresh = pd.DataFrame(threshold_data)

# 4. 순서 지정 (역순)
df_thresh['MatchType'] = pd.Categorical(df_thresh['MatchType'], categories=match_order[::-1], ordered=True)
df_thresh = df_thresh[df_thresh['Threshold']>=2]

df_pivot = df_thresh.pivot(index='Threshold', columns='MatchType', values='Count').fillna(0)
df_pivot = df_pivot[match_order[::-1]]  # 컬럼 순서 정렬

# 5. 그리기
plt.figure(figsize=(14, 4))
bottom = np.zeros(len(df_pivot))

for match_type in reversed(df_pivot.columns):
    plt.bar(df_pivot.index, df_pivot[match_type], bottom=bottom, color=type_color[match_type], label=match_type)
    bottom += df_pivot[match_type].values

plt.xlabel("Exon count threshold (≥)")
plt.ylabel("Transcript count")
plt.xticks(thresholds)
plt.tight_layout()
sns.despine()
plt.legend().remove()
plt.margins(x=0)  
#plt.savefig("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/figures/exon2/exon_threshold_barplot.pdf", bbox_inches='tight', dpi=300)
plt.show()


# %%
ggg = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_validation_clinicalinfo_new.txt', sep='\t', index_col=0)
ggg = ggg[ggg['ongoing']!=3]

# %%
aa = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/LRstats/mapped_counts.txt', sep='\t', index_col=0, header=None)
aa.columns = ['cnt']
sns.histplot(aa['cnt'], bins=100, kde=True)


# %%
####^^ exon 1 filtering ###############
df_exon1 = df_plot[(df_plot['exon_count'] == 1) & (df_plot['MatchType']=='c')]
df_eql = df_plot[df_plot['MatchType']=='c']
exon1_list = df_exon1.index.tolist()
trmap_best_eql = trmap_best[trmap_best['class_code']=='c']
trmap_best_exon1 = trmap_best_eql[trmap_best_eql['qry_id'].isin(exon1_list)]
# %%
df_exon1.index.to_series().str.extract(r'^(MSTRG|ENST)')[0].value_counts()


# %%
sns.histplot(df_exon1['length_bp'],bins=100, kde=True)

# %%
# 1) 개수 → 비율 계산
# 2. info 서브셋
df_plot = info.loc[info.index.isin(transcripts)].copy()

# 3. MatchType 정확히 매핑
df_plot['MatchType'] = df_plot.index.map(match_labels_dict)
df_plot['length_bp'] = df_plot['length_kb'] * 1000


match_order = list(type_color.keys())

# 2. Categorical로 순서 지정
df_plot['MatchType'] = pd.Categorical(df_plot['MatchType'], categories=match_order[::-1], ordered=True)

#######################
df_plot = df_plot[df_plot['exon_count']>=1]
#######################

df_counts = (
    df_plot
    .groupby(['exon_count', 'MatchType'], as_index=False)
    .size()
    .rename(columns={'size': 'count'})
)
df_counts['proportion'] = df_counts['count'] / df_counts.groupby('exon_count')['count'].transform('sum')

# 2) 스택 순서(legend 순서와 맞추기)
stack_order = ['=', 'c','k','m','n','j','e','o','s','x','i','y','p','r','u','Unmatched']
stack_order = [c for c in stack_order if c in df_counts['MatchType'].unique()]  # 존재하는 것만

# 3) wide 형태로 피벗
pivot = (df_counts
        .pivot(index='exon_count', columns='MatchType', values='proportion')
        .fillna(0.0)
        .reindex(columns=stack_order)
        .sort_index())

# 4) 실제 스택 그리기
import numpy as np
plt.figure(figsize=(8,4))
sns.set_style("white")

x = pivot.index.values
bottom = np.zeros_like(x, dtype=float)

for code in pivot.columns:
    y = pivot[code].values
    plt.bar(x, y, bottom=bottom, color=type_color.get(code, 'gray'), width=0.8, linewidth=0, label=code)
    bottom += y

plt.xlabel("exon count")
plt.ylabel("proportion")
#plt.xticks([1,3,6,9,12,15,18,21,24,27,30])
plt.xlim(0.5, 30.5)
plt.xticks(np.arange(1, 31, 1))
sns.despine()
plt.tight_layout()
#plt.savefig("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/figures/exoncount_histogram_proportion_fromexon1.pdf", bbox_inches='tight', dpi=300)
plt.show()
# 필요하면 범례 켜기
# plt.legend(title=None, bbox_to_anchor=(1.02, 1), loc="upper left")

# %%
# --- 1) count pivot까지는 그대로 ---
df_pivot = df_thresh.pivot(index='Threshold', columns='MatchType', values='Count').fillna(0)
df_pivot = df_pivot.reindex(columns=match_order[::-1])  # 컬럼 순서 고정

# --- 2) 각 threshold(행)별 합이 1이 되도록 비율 변환 ---
df_prop = df_pivot.div(df_pivot.sum(axis=1), axis=0).fillna(0)

# --- 3) 100% stacked barplot ---
plt.figure(figsize=(8, 4))
bottom = np.zeros(len(df_prop))

# 스택 순서: 위에서부터 match_order (역순으로 쌓을 거면 for도 역순 주의)
for mt in df_prop.columns:  # 이미 match_order[::-1]로 정렬했음
    vals = df_prop[mt].values
    plt.bar(df_prop.index, vals, bottom=bottom, color=type_color.get(mt, 'gray'), label=mt, linewidth=0)
    bottom += vals

plt.xlabel("Exon count threshold (≥)")
plt.ylabel("Proportion")
plt.xticks(thresholds)
plt.ylim(0, 1)
plt.margins(x=0)
sns.despine()
plt.legend().remove()
plt.tight_layout()
plt.savefig("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/figures/exon2/exoncount_histogram_exonthreshold_proportion.pdf", bbox_inches='tight', dpi=300)
plt.show()

# %%
import pandas as pd
tmp = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/mixSRLR/quantification_v2/transcript_count_matrix.csv', sep=',', index_col=0)
tmp = tmp.fillna(0)
tmp.to_csv('/home/jiye/jiye/nanopore/HG_JH_check/mixSRLR/quantification_v2/transcript_count_matrix_fillna.csv', sep=',', index=True)

# %%
###^^^ +++++ exon histogram 4 classes ##############
# 1. class code → 그룹 매핑
code_to_group = {
    '=': 'Complete match',
    'c': 'Partial intron match', 'k': 'Partial intron match',
    'm': 'Partial intron match', 'n': 'Partial intron match',
    'j': 'Partial intron match',
    'e': 'Same strand overlap without junction match',
    'o': 'Same strand overlap without junction match',
    's': 'Not matched', 'x': 'Not matched', 'i': 'Not matched', 'y': 'Not matched',
    'p': 'Not matched', 'r': 'Not matched', 'u': 'Not matched',
    'Unmatched': 'Not matched'
}

group_colors = {
    "Complete match": "#569CFE",  # 파랑
    "Partial intron match": "#FFD93D",  # 노랑
    "Same strand overlap without junction match": "#DA50EC",  # 분홍
    "Not matched": "#595959"  # 회색
    }

# 2. 그룹 매핑 추가
df_plot = info.loc[info.index.isin(transcripts)].copy()
df_plot['MatchType'] = df_plot.index.map(match_labels_dict)
df_plot['Group'] = df_plot['MatchType'].map(code_to_group)
df_plot['length_bp'] = df_plot['length_kb'] * 1000

df_plot = df_plot[df_plot['exon_count'] >= 1]

# 3. 그룹별 집계
df_counts = (
    df_plot
    .groupby(['exon_count', 'Group'], as_index=False)
    .size()
    .rename(columns={'size': 'count'})
)
df_counts['proportion'] = df_counts['count'] / df_counts.groupby('exon_count')['count'].transform('sum')

# 4. wide 형태로 피벗
stack_order = list(group_colors.keys())
pivot = (df_counts
         .pivot(index='exon_count', columns='Group', values='proportion')
         .fillna(0.0)
         .reindex(columns=stack_order)
         .sort_index())

# 5. 스택 그리기
import numpy as np
plt.figure(figsize=(8,4))
sns.set_style("white")

x = pivot.index.values
bottom = np.zeros_like(x, dtype=float)

for group in pivot.columns:
    y = pivot[group].values
    plt.bar(x, y, bottom=bottom, color=group_colors[group], width=0.8, linewidth=0, label=group)
    bottom += y

plt.xlabel("exon count")
plt.ylabel("proportion")
plt.xlim(0.5, 30.5)
plt.xticks(np.arange(1, 31, 1))
sns.despine()
plt.tight_layout()
plt.savefig("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/figures/exoncount_histogram_proportion_4grouped.pdf", bbox_inches='tight', dpi=300)
plt.show()


# %%
