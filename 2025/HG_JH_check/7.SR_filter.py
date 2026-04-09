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

filtered_cov = filter_by_coverage(cov, min_coverage=1, min_fraction=0.1)
filtered_exon= info[info['exon_count'] >= 3].index

both_trans = set(filtered_cov.str.split("-", n=1).str[0]).intersection(set(filtered_exon))

tmm = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/SR_transcript_TMM.txt', sep='\t', index_col=0)
finaltmm = tmm.loc[tmm.index.isin(both_trans),:]

tpm = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/SR_283_transcript_TPM.txt', sep='\t', index_col=0)
tpm.index = tpm.index.str.split("-", n=1).str[0]  # transcript_id로 변경
finaltpm = tpm.loc[tpm.index.isin(both_trans),:]

# %%
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

plt.figure(figsize=(6, 6))
sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.plot(labels, sr_values, marker='o', label='Short-read')
plt.xticks(rotation=45, ha="right")
plt.ylabel("Number of remaining transcripts")
plt.xlabel("Filtering step")
plt.grid(True, axis='y')
plt.legend()
plt.tight_layout()
sns.despine()
#plt.savefig('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/figures/SR_filtering.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%
print(len(set(sr_filtered).intersection(set(both_trans))))
# %%
tmap_path = "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/gffcompare/gffcompare.SR_238_merged.gtf.tmap"
tmap = pd.read_csv(tmap_path, sep="\t")
matched = tmap[tmap['class_code'] == '='][['qry_id', 'ref_id']]
matched.columns = ['sr_id', 'lr_id']  # for clarity

LRtpm = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/LR_283_transcript_TPM.txt', sep='\t', index_col=0)
LRtpm.index = LRtpm.index.str.split("-", n=1).str[0] 
LRtmm = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/LR_transcript_GeTMM.txt', sep='\t', index_col=0)

#%%
##^^ 2. noncoding whole / T / N #############
cpat_df = pd.read_csv("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/CPAT/SR/SR_cpat.ORF_prob.best.tsv", sep="\t", index_col=0)
cpc_df = pd.read_csv("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/CPC2/SR/cpc2output.txt",   sep="\t", index_col=0)
cpat_nc = cpat_df[cpat_df['Coding_prob']<0.364].index.to_list()
cpc_nc = cpc_df[cpc_df['label']=='noncoding'].index.to_list()
nc_trans = set(cpat_nc).union(set(cpc_nc))

filtertpm = tpm.loc[tpm.index.isin(sr_filtered),:]
filtertpm = finaltpm
filtertpm = tpm[tpm.mean(axis=1)!=0]
filtertpm = finaltmm[finaltmm.index.str.startswith("ENST")]
SR_input = filtertpm.loc[~filtertpm.index.isin(nc_trans),:] #tmm vs. finaltmm / coding vs. noncoding
SR_input = SR_input.loc[SR_input.index.isin(matched['sr_id']),:]  # only matched transcripts

####
matched_filtered = matched.set_index('sr_id').loc[SR_input.index]
ordered_lr_ids = matched_filtered['lr_id'].values
LR_input = LRtmm.loc[ordered_lr_ids]
####

from scipy.stats import linregress

def mean_ci(values):
    import numpy as np
    mean = np.mean(values)
    se = np.std(values, ddof=1) / np.sqrt(len(values))
    ci = 1.96 * se  # 95% CI
    return mean, ci

def plot_sr_lr_regression(SR_input, LR_input, title, inputcolor, xlim=(0, 1000), ylim=(0, 3000),ax=None):
    slopes = []
    intercepts = []
    r_values = []
    p_values = []

    for sample in SR_input.columns:
        x = SR_input[sample].values
        y = LR_input[sample].values
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() > 10:
            x_, y_ = x[mask], y[mask]
            ax.scatter(x_, y_, alpha=0.1, color='gray', s=2)
            slope, intercept, r, p, stderr = linregress(x_, y_)
            slopes.append(slope)
            intercepts.append(intercept)
            r_values.append(r)
            p_values.append(p)
            ax.plot(x_, slope * x_ + intercept, color=inputcolor, alpha=0.05, lw=1)

    if slopes:
        mean_slope, ci_slope = mean_ci(slopes)
        mean_intercept, ci_intercept = mean_ci(intercepts)
        mean_r, ci_r = mean_ci(r_values)
        mean_p = np.mean(p_values)

        max_x = xlim[1]
        ax.plot([0, max_x],
                [mean_intercept, mean_slope * max_x + mean_intercept],
                color=inputcolor, lw=1.6)

        text = f"$r$ = {mean_r:.2f} ± {ci_r:.2f}" #\n$p$ = {mean_p:.1e}
        ax.text(xlim[1] * 0.7, ylim[1] * 0.8, text, fontsize=11, color='black')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_xlabel("SR transcript exp")
    sns.despine()

sns.set_style("ticks")
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig.suptitle("coding transcripts", fontsize=14)

plot_sr_lr_regression(SR_input, LR_input, title="Whole", ax=axes[0], inputcolor='#095083')
plot_sr_lr_regression(SR_input.iloc[:, ::2], LR_input.iloc[:, ::2], title="Normal", ax=axes[1], inputcolor='#F2A900')
plot_sr_lr_regression(SR_input.iloc[:, 1::2], LR_input.iloc[:, 1::2], title="Tumor", ax=axes[2], inputcolor='#11710A')
axes[0].set_ylabel("LR transcript exp")
fig.subplots_adjust(wspace=0.1)
#plt.savefig('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/figures/beforefiltering_regression_coding_TPM.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%
#^^ violin plot ################3
def get_r_and_p_values(SR_input, LR_input):
    r_values = []
    p_values = []
    for sample in SR_input.columns:
        x = SR_input[sample].values
        y = LR_input[sample].values
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() > 10:
            x_, y_ = x[mask], y[mask]
            result = linregress(x_, y_)
            r_values.append(result.rvalue)
            p_values.append(result.pvalue)
    return r_values, p_values

# coding transcript
coding_idx = finaltmm.index.difference(nc_trans)
matched_sr_ids = matched['sr_id']

# coding 중 matched된 것만
coding_matched_idx = coding_idx.intersection(matched_sr_ids)

SR_coding = finaltmm.loc[coding_matched_idx]
matched_coding = matched.set_index('sr_id').loc[coding_matched_idx]
LR_coding = LRtmm.loc[matched_coding['lr_id'].values]

# noncoding transcript
noncoding_idx = finaltmm.index.intersection(nc_trans)
noncoding_matched_idx = noncoding_idx.intersection(matched_sr_ids)

SR_noncoding = finaltmm.loc[noncoding_matched_idx]
matched_noncoding = matched.set_index('sr_id').loc[noncoding_matched_idx]
LR_noncoding = LRtmm.loc[matched_noncoding['lr_id'].values]


# r값 계산
def r_df_from_inputs(SR, LR, label):
    r1, p1 = get_r_and_p_values(SR, LR)
    r2, p2 = get_r_and_p_values(SR.iloc[:, ::2], LR.iloc[:, ::2])
    r3, p3 = get_r_and_p_values(SR.iloc[:, 1::2], LR.iloc[:, 1::2])
    
    rs = r1 + r2 + r3
    ps = p1 + p2 + p3
    groups = (["Whole"] * len(r1)) + (["Normal"] * len(r2)) + (["Tumor"] * len(r3))
    transcript_type = [label] * len(rs)

    return pd.DataFrame({
        "r": rs,
        "-log10(p)": -np.log10(ps),
        "Group": groups,
        "Transcript": transcript_type
    })


df_coding = r_df_from_inputs(SR_coding, LR_coding, "Coding")
df_noncoding = r_df_from_inputs(SR_noncoding, LR_noncoding, "Noncoding")
df_all = pd.concat([df_coding, df_noncoding], ignore_index=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

sns.violinplot(data=df_all, x="Group", y="r", hue="Transcript",
               split=True, inner="box", palette="Paired", ax=axes[0])
axes[0].set_title("Pearson r")
axes[0].set_xlabel("")
axes[0].legend_.remove()

sns.violinplot(data=df_all, x="Group", y="-log10(p)", hue="Transcript",
               split=True, inner="box", palette="Paired", ax=axes[1])
axes[1].set_title("-log10(p-value)")
axes[1].set_xlabel("")
axes[1].legend(title="Transcript", loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)

sns.despine()
plt.tight_layout()
#plt.savefig('violin_r_and_logp.pdf', dpi=300, bbox_inches='tight')
plt.show()



#%%
def plot_kde_triplet(sr_df, lr_df, labels=["Whole", "Normal", "Tumor"], xlim=(0, 500), ylim=(0, 2000)):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    fig.suptitle("noncoding transcripts", fontsize=16)

    for i, (ax, label) in enumerate(zip(axs, labels)):
        if label == "Whole":
            sr_vals = sr_df.values.flatten()
            lr_vals = lr_df.values.flatten()
        elif label == "Normal":
            sr_vals = sr_df.iloc[:, ::2].values.flatten()
            lr_vals = lr_df.iloc[:, ::2].values.flatten()
        elif label == "Tumor":
            sr_vals = sr_df.iloc[:, 1::2].values.flatten()
            lr_vals = lr_df.iloc[:, 1::2].values.flatten()

        mask = ~np.isnan(sr_vals) & ~np.isnan(lr_vals)
        x = sr_vals[mask]
        y = lr_vals[mask]

        # KDE plot
        sns.kdeplot(x=x, y=y, cmap="viridis", fill=True, ax=ax, levels=100, thresh=1e-5)

        # 대각선 y = x
        ax.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], ls="--", color="white", linewidth=2)

        # 상관계수
        r, p = pearsonr(x, y)
        ax.text(xlim[0]+100, ylim[1]*0.7, f"$r$ = {r:.2f}\n$p$ = {p:.1e}", color='white', fontsize=12)

        ax.set_title(label)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("SR")
        if i == 0:
            ax.set_ylabel("LR")

    plt.show()

plot_kde_triplet(SR_input, LR_input)

#plt.savefig('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/figures/kde_noncoding.pdf', dpi=300, bbox_inches='tight')
#plt.show()

#%%
###^^ 3. cancer gene ###########

def plot_sr_lr_regression(SR_input, LR_input, title, xlim=(0, 300), ylim=(0, 300),ax=None):
    slopes = []
    intercepts = []
    r_values = []
    p_values = []

    for sample in SR_input.columns:
        x = SR_input[sample].values
        y = LR_input[sample].values
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() > 10:
            x_, y_ = x[mask], y[mask]
            ax.scatter(x_, y_, alpha=0.1, color='gray', s=2)
            slope, intercept, r, p, stderr = linregress(x_, y_)
            slopes.append(slope)
            intercepts.append(intercept)
            r_values.append(r)
            p_values.append(p)
            ax.plot(x_, slope * x_ + intercept, color='green', alpha=0.05, lw=1)

    if slopes:
        mean_slope, ci_slope = mean_ci(slopes)
        mean_intercept, ci_intercept = mean_ci(intercepts)
        mean_r, ci_r = mean_ci(r_values)
        mean_p = np.mean(p_values)

        max_x = xlim[1]
        ax.plot([0, max_x],
                [mean_intercept, mean_slope * max_x + mean_intercept],
                color='#278309', lw=1.6)

        text = f"$r$ = {mean_r:.2f} ± {ci_r:.2f}\n$p$ = {mean_p:.1e}"
        ax.text(xlim[1] * 0.7, ylim[1] * 0.8, text, fontsize=11, color='black')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_xlabel("SR transcript exp")
    sns.despine()
    
genelist = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/cancergenelist.txt', sep="\t", index_col=0)
genelist = genelist[(genelist['CPG']=='Yes') ] #& (genelist['TSG']=='Yes')
gene_map, _ = load_transcript_gene_map(tpm_file["SR"])
translist = gene_map[gene_map.isin(genelist.index)].index

####
SR_input = finaltmm.loc[finaltmm.index.isin(translist),:]
#SR_input = SR_input.loc[SR_input.index.isin(nc_trans),:] ##coding vs. noncoding
# SR_input = finaltmm
####
####
matched_filtered = matched.set_index('sr_id').loc[SR_input.index]
ordered_lr_ids = matched_filtered['lr_id'].values
LR_input = LRtmm.loc[ordered_lr_ids]
####
sns.set_style("ticks")
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig.suptitle("CPG", fontsize=14)

plot_sr_lr_regression(SR_input, LR_input, title="Whole", ax=axes[0])
plot_sr_lr_regression(SR_input.iloc[:, ::2], LR_input.iloc[:, ::2], title="Normal", ax=axes[1])
plot_sr_lr_regression(SR_input.iloc[:, 1::2], LR_input.iloc[:, 1::2], title="Tumor", ax=axes[2])
axes[0].set_ylabel("LR transcript exp")
fig.subplots_adjust(wspace=0.1)
plt.savefig('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/figures/regression_CPG_all.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%

#%%
from matplotlib.colors import SymLogNorm

# define pairs
pairs = [
    ("Whole", SR_input.values.flatten(), LR_input.values.flatten()),
    ("Normal", SR_input.iloc[:, ::2].values.flatten(), LR_input.iloc[:, ::2].values.flatten()),
    ("Tumor", SR_input.iloc[:, 1::2].values.flatten(), LR_input.iloc[:, 1::2].values.flatten()),
]

fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
hist2d_list = []

for ax, (title, x_raw, y_raw) in zip(axs, pairs):
    mask = ~np.isnan(x_raw) & ~np.isnan(y_raw)
    x = x_raw[mask]
    y = y_raw[mask]
    
    r, p = pearsonr(x, y)
    
    h = ax.hist2d(x, y, bins=100, norm=SymLogNorm(linthresh=1), cmap='viridis')
    hist2d_list.append(h)  # Save to access the colorbar image
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 2000)
    ax.set_title(title)
    ax.set_xlabel("SR")
    ax.text(30, 1500, f"$r$ = {r:.2f}\n$p$ = {p:.1e}", fontsize=12, color='white')

axs[0].set_ylabel("LR")

# Colorbar 설정은 마지막 plot 기준으로 하나만 표시
cbar = fig.colorbar(hist2d_list[-1][3], ax=axs, location='right', pad=0.02)
cbar.set_label("Count")

fig.suptitle("cancer gene", fontsize=14)
plt.savefig('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/figures/hist2d_cancergene.pdf', dpi=300, bbox_inches='tight')
plt.show()


# %%
#^^^^^^^ stacked bar plot for proportions of expressed transcripts by classification

import matplotlib.pyplot as plt

def calculate_proportions(tpm_df, classification_dict):
    category_map = {
        'Full match': set(),
        'Partial match': set(),
        'Unmatched': set()
    }
    for tid, label in classification_dict.items():
        if label in category_map:
            category_map[label].add(tid)
    
    sample_proportions = {
        'Full match': [],
        'Partial match': [],
        'Unmatched': []
    }
    
    for sample in tpm_df.columns:
        sample_tpm = tpm_df[sample]
        expressed = sample_tpm[sample_tpm > 0].index
        
        total_expressed = len(expressed)
        #print(total_expressed)
        if total_expressed == 0:
            for key in sample_proportions:
                sample_proportions[key].append(0)
            continue
        
        for category in ['Full match', 'Partial match', 'Unmatched']:
            n = len(set(expressed).intersection(category_map[category]))
            sample_proportions[category].append(n / total_expressed)
    
    return sample_proportions

def plot_stacked_bar(proportions_dict):
    full = np.array(proportions_dict['Full match'])
    partial = np.array(proportions_dict['Partial match'])
    unmatch = np.array(proportions_dict['Unmatched'])

    # 1. Full match 기준 내림차순 정렬 인덱스
    sort_idx = np.argsort(-full)

    # 2. 정렬된 데이터
    # full_sorted = full[sort_idx]
    # partial_sorted = partial[sort_idx]
    # unmatch_sorted = unmatch[sort_idx]
    
    full_sorted = full
    partial_sorted = partial
    unmatch_sorted = unmatch
    

    samples = np.arange(len(full_sorted))  # x축 위치

    sns.set_style("white")
    plt.figure(figsize=(12, 4))
    plt.bar(samples, full_sorted, label='Full match', color='#FFDE4B',
            edgecolor='#FFFFFF', width=1.0, align='edge', linewidth=0.1)
    plt.bar(samples, partial_sorted, bottom=full_sorted, label='Partial match',
            edgecolor='#FFFFFF', color='#238B88', width=1.0, align='edge', linewidth=0.1)
    bottom2 = full_sorted + partial_sorted
    plt.bar(samples, unmatch_sorted, bottom=bottom2, label='Unmatched',
            edgecolor='#FFFFFF', color='#2C003E', width=1.0, align='edge', linewidth=0.1)

    plt.xlabel('Samples')
    plt.ylabel('Proportion of expressed transcripts')
    plt.xticks([])
    plt.ylim(0, 1)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0], ['0%', '25%', '50%', '75%', '100%'])
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=3)
    plt.tight_layout()
    sns.despine()
    plt.savefig('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/figures/notfiltered_proportions_ver2_unsorted.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    
transcript_match_dict = {}

for tid, class_code in zip(tmap['qry_id'], tmap['class_code']):
    if class_code == '=':
        transcript_match_dict[tid] = 'Full match'
    elif class_code == 'c':
        transcript_match_dict[tid] = 'Partial match'
    else:
        transcript_match_dict[tid] = 'Unmatched'

# Step 3: TPM matrix and proportions

proportions = calculate_proportions(tmm, transcript_match_dict) #tmm.loc[tmm.index.isin(sr_filtered),:]

# Step 4: plot
plot_stacked_bar(proportions)

# %%
