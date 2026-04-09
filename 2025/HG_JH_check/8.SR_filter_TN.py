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

# %%
####^^only transcripts with at least five exons and mean coverage ≥1 across at least 10% of samples. #####

cov = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/SR_283_transcript_coverage.txt', sep='\t', index_col=0)
info = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/transcriptinfo_SR.txt', sep='\t', index_col=0)

def filter_by_coverage(coverage_df, min_coverage=1, min_fraction=0.1):
    num_samples = coverage_df.shape[1]
    required_count = int(num_samples * min_fraction)

    # 각 transcript별로 coverage ≥ min_coverage 인 sample 수 세기
    passing_counts = (coverage_df >= min_coverage).sum(axis=1)

    # 조건을 만족하는 transcript ID만 선택
    filtered = coverage_df.index[passing_counts >= required_count]
    return filtered

cov_normal = cov[[s for s in cov.columns if '-N' in s]]
cov_tumor = cov[[s for s in cov.columns if '-T' in s]]

filtered_cov_normal = filter_by_coverage(cov_normal, min_coverage=1, min_fraction=0.1)
filtered_cov_tumor = filter_by_coverage(cov_tumor, min_coverage=1, min_fraction=0.1)
filtered_cov = filter_by_coverage(cov, min_coverage=1, min_fraction=0.1)

filtered_exon= info[info['exon_count'] >= 3].index

both_trans_normal = set(filtered_cov_normal.str.split("-", n=1).str[0]).intersection(set(filtered_exon))
both_trans_tumor = set(filtered_cov_tumor.str.split("-", n=1).str[0]).intersection(set(filtered_exon))
both_trans = set(filtered_cov.str.split("-", n=1).str[0]).intersection(set(filtered_exon))

#%%

tmm = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/SR_transcript_TMM.txt', sep='\t', index_col=0)
tmm_normal = tmm[[s for s in tmm.columns if '-N' in s]]
tmm_tumor = tmm[[s for s in tmm.columns if '-T' in s]]

finaltmm_normal = tmm_normal.loc[tmm_normal.index.isin(both_trans_normal),]
finaltmm_tumor = tmm_tumor.loc[tmm_tumor.index.isin(both_trans_tumor),:]
finaltmm = tmm.loc[tmm.index.isin(both_trans),:]

LRtmm = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/LR_transcript_TMM.txt', sep='\t', index_col=0)
LRtmm_normal = LRtmm[[s for s in LRtmm.columns if '-N' in s]]
LRtmm_tumor = LRtmm[[s for s in LRtmm.columns if '-T' in s]]

#^^ tmap
tmap_path = "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/gffcompare/gffcompare.SR_238_merged.gtf.tmap"
tmap = pd.read_csv(tmap_path, sep="\t")
matched = tmap[tmap['class_code'] == '='][['qry_id', 'ref_id']]
matched.columns = ['sr_id', 'lr_id'] 

#%%

####^^ 1. before QC and after QC ###################

def load_transcript_gene_map(tpm_path):
    df = pd.read_csv(tpm_path, sep="\t", index_col=0)

    # Step 1: transcript ID만 남기기
    transcript_ids = df.index.str.split("-", n=1).str[0]
    gene_names = df.index.str.split("-", n=1).str[-1]

    # Step 2: gene_map: transcript_id → gene_name
    gene_map = pd.Series(gene_names.values, index=transcript_ids)

    # Step 3: TPM index도 transcript_id로
    df.index = transcript_ids

    return gene_map, df

def parse_strand_info(gtf_path):
    strand_map = {}
    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"): continue
            fields = line.strip().split("\t")
            if fields[2] != "transcript": continue
            attr = fields[8]
            tid = attr.split('transcript_id "')[1].split('"')[0]
            strand = fields[6]
            strand_map[tid] = strand
    return pd.Series(strand_map)

def apply_filters(tpm_path, count_path, gtf_path):
    # Load gene name and TPM
    gene_map, tpm = load_transcript_gene_map(tpm_path)

    # Load read count
    count_df = pd.read_csv(count_path, index_col=0)

    # Load strand info
    strand_map = parse_strand_info(gtf_path)

    all_transcripts = count_df.index
    result = []

    # Step 1: Before filtering
    current = set(all_transcripts)
    result.append(len(current))

    # Step 2: Strand filter
    strand_ok = strand_map[strand_map.isin(["+", "-"])].index
    current = current.intersection(strand_ok)
    result.append(len(current))

    # Step 3: Gene type filter (exclude Ig and TCR)
    gene_filtered = gene_map[~gene_map.str.contains("IG[HKL]|TR[ABDG]")].index
    current = current.intersection(gene_filtered)
    result.append(len(current))

    # Step 4: Read count filter (95% 샘플에서 count < 10 → 제거)
    filtered_counts = count_df.loc[list(current)]
    mask = (filtered_counts < 10).sum(axis=1)>= int(filtered_counts.shape[1] * 0.95)
    current = filtered_counts[~mask].index
    result.append(len(current))

    # Step 5: Gene annotation filter (GENCODE-known → ENST로 시작하는 애들만)
    annotated = gene_map[~gene_map.str.startswith("MSTRG")].index
    current = current.intersection(annotated)
    result.append(len(current))

    return result, current

# 파일 경로
tpm_file = {
    "SR": "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/SR_283_transcript_TPM.txt",
    "LR": "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/LR_283_transcript_TPM.txt"
}

gtf_file = {
    "SR": "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/SR_238_merged.gtf",
    "LR": "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/LR_238_merged.gtf"
}

labels = [
    "Before filtering",
    "Strand filter",
    "Gene type filter",
    "Read count filter",
    "Gene annotation filter"
]

count_file = {
    "SR": "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/SR_transcript_count_matrix.csv",
    "LR": "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/LR_transcript_count_matrix.csv"
}

sr_values, sr_filtered = apply_filters(tpm_file["SR"], count_file["SR"], gtf_file["SR"])
lr_values, lr_filtered = apply_filters(tpm_file["LR"], count_file["LR"], gtf_file["LR"])



# %%
###^^ coding / noncoding ########

cpat_df = pd.read_csv("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/CPAT/SR/SR_cpat.ORF_prob.best.tsv", sep="\t", index_col=0)
cpc_df = pd.read_csv("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/CPC2/SR/cpc2output.txt",   sep="\t", index_col=0)
cpat_nc = cpat_df[cpat_df['Coding_prob']<0.364].index.to_list()
cpc_nc = cpc_df[cpc_df['label']=='noncoding'].index.to_list()
nc_trans = set(cpat_nc).union(set(cpc_nc))

inputdata = finaltmm_normal
inputLRdata = LRtmm_normal
inputLRdata = inputLRdata[inputLRdata.mean(axis=1)!=0]
SR_input = inputdata
SR_input = SR_input.loc[SR_input.index.isin(matched['sr_id']),:]  

####
matched_filtered = matched[matched['lr_id'].isin(inputLRdata.index)]
matched_filtered = matched_filtered[matched_filtered['sr_id'].isin(SR_input.index)]

LR_input = inputLRdata.loc[inputLRdata.index.isin(matched_filtered['lr_id']),:]
matched_filtered = matched_filtered[matched_filtered['lr_id'].isin(LR_input.index)]
SR_input = SR_input.loc[SR_input.index.isin(matched_filtered['sr_id']),:]




# %%
#^^ violin plot ################
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# SR_data_filt = SR_data.loc[SR_data.index.intersection(valid_tx)]
# LR_data_filt = LR_data.loc[LR_data.index.intersection(valid_tx)]

SR_data_filt = SR_input
LR_data_filt = LR_input

# transcript 별 coding 여부 라벨 만들기
coding_status = pd.Series("coding", index=SR_data_filt.index)
coding_status.loc[coding_status.index.intersection(nc_trans)] = "noncoding"

# 샘플 그룹 정의
sample_groups = {
    "Whole": SR_data_filt.columns,
    "Normal": [s for s in SR_data_filt.columns if '-N' in s],
    "Tumor": [s for s in SR_data_filt.columns if '-T' in s]
}

from scipy.stats import pearsonr
coding_status = pd.Series(
    data=['noncoding' if tx in nc_trans else 'coding' for tx in SR_data_filt.index],
    index=SR_data_filt.index
)

from scipy.stats import pearsonr

def compute_samplewise_r_fast(SR_df, LR_df, coding_status, group_name):
    result = []

    for sample in SR_df.columns:
        sr_vals = SR_df[sample]
        lr_vals = LR_df[sample]
        mask = sr_vals.notna() & lr_vals.notna()
        sr_vals = sr_vals[mask]
        lr_vals = lr_vals[mask]
        labels = coding_status[mask]

        for label in ['coding', 'noncoding']:
            idx = labels[labels == label].index
            if len(idx) > 2:
                r, p = pearsonr(sr_vals.loc[idx], lr_vals.loc[idx])
                result.append({
                    'Sample': sample,
                    'Group': group_name,
                    'CodingStatus': label,
                    'Pearson_r': r,
                    'p_value': p
                })
    return pd.DataFrame(result)

dfs = []
for group, cols in sample_groups.items():
    df = compute_samplewise_r_fast(SR_data_filt[cols], LR_data_filt[cols], coding_status, group)
    dfs.append(df)

r_samp_df = pd.concat(dfs, ignore_index=True)
r_samp_df['-log10(p)'] = -np.log10(r_samp_df['p_value'])
r_samp_df['Group'] = pd.Categorical(r_samp_df['Group'], categories=['Whole', 'Normal', 'Tumor'], ordered=True)

# Pearson r violin plot
plt.figure(figsize=(10, 5))
sns.violinplot(
    data=r_samp_df,
    x='Group', y='Pearson_r',
    hue='CodingStatus',
    split=True, inner='box', palette='Set2'
)
plt.title("Per-sample Pearson r by Coding Status")
plt.ylabel("Pearson r")
plt.tight_layout()
plt.show()

# -log10(p) violin plot
plt.figure(figsize=(10, 5))
sns.violinplot(
    data=r_samp_df,
    x='Group', y='-log10(p)',
    hue='CodingStatus',
    split=True, inner='box', palette='Set2'
)
plt.title("Per-sample -log10(p) by Coding Status")
plt.ylabel("-log10(p-value)")
plt.tight_layout()
plt.show()

# %%
