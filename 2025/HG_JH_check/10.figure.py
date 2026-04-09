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
finaltmm = tmm.loc[tmm.index.isin(both_trans),:]
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


#%%
sr_to_lr = matched.set_index('sr_id')['lr_id'].to_dict()
nc_lr_ids = set(sr_to_lr[sr] for sr in nc_trans if sr in sr_to_lr)

# %%
###^^^ regression plot ##############

def plot_sr_lr_regression_flattened(SR_input, LR_input, title, inputcolor, xlim=(-100, 30000), ylim=(-100, 30000), ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # flatten한 전체 값에서 회귀선 및 r 계산 (rechecked 방식)
    x_all = SR_input.values.flatten()
    y_all = LR_input.values.flatten()
    mask = ~np.isnan(x_all) & ~np.isnan(y_all)
    x_all = np.log2(x_all[mask]+1/2)
    y_all = np.log2(y_all[mask]+1/2)

    # 샘플별 scatter (회색 점)
    for sample in SR_input.columns:
        x = SR_input[sample].values
        y = LR_input[sample].values
        sample_mask = ~np.isnan(x) & ~np.isnan(y)
        if sample_mask.sum() > 2:
            x_, y_ = np.log2(x[sample_mask]+1/2), np.log2(y[sample_mask]+1/2)
            ax.scatter(x_, y_, alpha=0.1, color='gray', s=2)

            # 샘플별 회귀선 추가 (투명도 높임)
            slope, intercept, *_ = linregress(x_, y_)
            x_line = np.array([x_.min(), x_.max()])
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, color=inputcolor, alpha=0.05, lw=1)


    # 전체 회귀선 그리기
    if len(x_all) > 2:
        slope, intercept, r, p, stderr = linregress(x_all, y_all)
        max_x = xlim[1]
        #ax.plot([0, max_x], [intercept, slope * max_x + intercept], color=inputcolor, lw=1.2)
        print(f"[{title}] N = {len(x_all)}, r = {r:.3f}, p = {p:.1e}")

        # 텍스트로 r, p 표시
        text = f"$r$ = {r:.2f}\n$p$ = {p:.1e}"
        ax.text(15 * 0.7, 15 * 0.75, text, fontsize=11, color='black')

    ax.set_title(title)
    ax.set_xlabel("Illumina")
    ax.set_ylabel("Nanopore")
    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)
    sns.despine()


sns.set_style("ticks")
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

#fig.suptitle("noncoding transcripts", fontsize=14)

SR_data_filt = SR_data.loc[~SR_data.index.isin(nc_lr_ids)]
LR_data_filt = LR_data.loc[~LR_data.index.isin(nc_lr_ids)]
# SR_data_filt = SR_data
# LR_data_filt = LR_data

###### zero LR transcript 제거 ########
all_zero_lr = LR_data_filt[(LR_data_filt == 0).all(axis=1)].index
LR_data_filt = LR_data_filt.drop(index=all_zero_lr)
SR_data_filt = SR_data_filt.drop(index=all_zero_lr)
###### zero LR transcript 제거 ########

sample_groups = {
    "Whole": SR_data_filt.columns,
    "Normal": [s for s in SR_data_filt.columns if '-N' in s],
    "Tumor": [s for s in SR_data_filt.columns if '-T' in s]
}

plot_sr_lr_regression_flattened(SR_data_filt[sample_groups["Whole"]], LR_data_filt[sample_groups["Whole"]], title="Whole", ax=axes[0], inputcolor='#000000')
plot_sr_lr_regression_flattened(SR_data_filt[sample_groups["Normal"]], LR_data_filt[sample_groups["Normal"]], title="Normal", ax=axes[1], inputcolor='#3765A8')
plot_sr_lr_regression_flattened(SR_data_filt[sample_groups["Tumor"]],  LR_data_filt[sample_groups["Tumor"]],  title="Tumor",  ax=axes[2], inputcolor='#E04A8F')

axes[0].set_ylabel("Nanopore")
fig.subplots_adjust(wspace=0.1)
#plt.savefig("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/figures/noLRfilter_SR_LR_regression_noncoding.pdf", bbox_inches='tight', dpi=300)
plt.show()

#%%
###^^ check #######
coding_status = pd.Series("coding", index=SR_data_filt.index)
coding_status.loc[coding_status.index.intersection(nc_lr_ids)] = "noncoding"

# 샘플 그룹 정의
sample_groups = {
    "Whole": SR_data_filt.columns,
    "Normal": [s for s in SR_data_filt.columns if '-N' in s],
    "Tumor": [s for s in SR_data_filt.columns if '-T' in s]
}

for group, cols in sample_groups.items():
    x = SR_data_filt[cols].values.flatten()
    y = LR_data_filt[cols].values.flatten()
    mask = ~np.isnan(x) & ~np.isnan(y)
    r, p = pearsonr(x[mask], y[mask])
    print(len(x[mask]), len(y[mask]))
    print(f"[rechecked] {group} Pearson r = {r:.3f}")

# normal_trans = set(SR_data_filt[sample_groups['Normal']].dropna(how='all').index)
# tumor_trans = set(SR_data_filt[sample_groups['Tumor']].dropna(how='all').index)
# common_trans = normal_trans.intersection(tumor_trans)

# print(f"Normal transcripts: {len(normal_trans)}")
# print(f"Tumor transcripts: {len(tumor_trans)}")
# print(f"Shared transcripts: {len(common_trans)}")

# %%
#####^^ violin plot ############
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

SR_data_filt = SR_data
LR_data_filt = LR_data

###### zero LR transcript 제거 ########
all_zero_lr = LR_data_filt[(LR_data_filt == 0).all(axis=1)].index
LR_data_filt = LR_data_filt.drop(index=all_zero_lr)
SR_data_filt = SR_data_filt.drop(index=all_zero_lr)
###### zero LR transcript 제거 ########

# transcript 별 coding 여부 라벨 만들기
coding_status = pd.Series("coding", index=SR_data_filt.index)
coding_status.loc[coding_status.index.intersection(nc_lr_ids)] = "noncoding"

# 샘플 그룹 정의
sample_groups = {
    "Whole": SR_data_filt.columns,
    "Normal": [s for s in SR_data_filt.columns if '-N' in s],
    "Tumor": [s for s in SR_data_filt.columns if '-T' in s]
}

from scipy.stats import pearsonr
coding_status = pd.Series(
    data=['noncoding' if tx in nc_lr_ids else 'coding' for tx in SR_data_filt.index],
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
                r, p = pearsonr(np.log2(sr_vals.loc[idx]+1/2), np.log2(lr_vals.loc[idx]+1/2))
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
# r_samp_df['Group'] = pd.Categorical(r_samp_df['Group'], categories=['Whole', 'Normal', 'Tumor'], ordered=True)

# Pearson r violin plot
plt.figure(figsize=(5, 6)) #(7,5)
# ax = sns.violinplot(
#     data=r_samp_df,
#     x='Group', y='Pearson_r',
#     hue='CodingStatus', palette={'coding': '#FF8F00', 'noncoding': '#AF47D2'},
#     split=True, inner='box'
# )

r_samp_df = r_samp_df[r_samp_df['Group']=='Whole']  # 'Whole' 그룹만 필터링
ax = sns.boxplot(
    data=r_samp_df,
    x='Group', y='Pearson_r',
    hue='CodingStatus', palette={'coding': '#FF8F00', 'noncoding': '#AF47D2'}, showfliers=False,
)
plt.ylabel("Pearson r")
plt.xlabel("")
ax.legend_.set_title(None)
ax.legend(
    loc="upper left",       
    bbox_to_anchor=(1.02, 1),  
    borderaxespad=0,
    frameon=False
)
plt.tight_layout()
sns.despine()
#plt.savefig("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/figures/noLRfilter_SR_LR_boxplot_onlywhole.pdf", bbox_inches='tight', dpi=300)
plt.show()

group_medians = r_samp_df.groupby(['Group', 'CodingStatus'])['Pearson_r'].median()

print("▶️ Group-wise medians (Pearson r):")
print(group_medians)

# # -log10(p) violin plot
# plt.figure(figsize=(10, 5))
# sns.violinplot(
#     data=r_samp_df,
#     x='Group', y='-log10(p)',
#     hue='CodingStatus',
#     split=True, inner='box', palette='Set2'
# )
# plt.title("Per-sample -log10(p) by Coding Status")
# plt.ylabel("-log10(p-value)")
# plt.tight_layout()
# plt.show()


#%%
# %%
###^^^ correlation heatmap ###################
sr_samples = SR_data.columns
lr_samples = LR_data.columns

r_matrix = pd.DataFrame(index=sr_samples, columns=lr_samples, dtype=float)

for sr_sample in sr_samples:
    for lr_sample in lr_samples:
        x = SR_data[sr_sample]
        y = LR_data[lr_sample]
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() > 2:
            r, _ = pearsonr(x[mask], y[mask])
            r_matrix.loc[sr_sample, lr_sample] = r

plt.figure(figsize=(10, 8))
sns.heatmap(r_matrix, annot=False, cmap="coolwarm", square=True, cbar_kws={"label": "Pearson r"})
plt.title("All Pairwise SR vs LR Correlation")
plt.xlabel("LR Samples")
plt.ylabel("SR Samples")
plt.tight_layout()
plt.show()

# %%
####^^ length distribution ###########
import seaborn as sns
import matplotlib.pyplot as plt
# 2. class_code → match type으로 매핑
def classify(code):
    if code == '=':
        return 'Full match'
    elif code == 'c':
        return 'Partial match'
    else:
        return 'Unmatched'

tmap['match_type'] = tmap['class_code'].apply(classify)

# 3. 필요한 컬럼만 추출하고 info와 매칭
matched = tmap[['qry_id', 'match_type']].copy()
matched = matched.rename(columns={'qry_id': 'transcript_id'})

# 4. info와 merge
merged = info.merge(matched, left_index=True, right_on='transcript_id', how='inner')
merged['length_kb'] = merged['length_kb']*1000

custom_palette = {
    "Full match": "#FFDE4B",
    "Partial match": "#238B88",
    "Unmatched": "#2C003E"
}

# 5. 히스토그램 그리기 (stacked barplot)
plt.figure(figsize=(10, 6))
ax = sns.histplot(data=merged, x="length_kb", hue="match_type", bins=2000, multiple="stack", palette=custom_palette, edgecolor='white', linewidth=0.3)
plt.xlabel("Transcript length (bp)")
plt.ylabel("Count")
plt.xlim(0, 15000) 
#plt.title("Transcript Length Distribution by Match Type")
ax.legend_.set_title(None)
plt.tight_layout()
sns.despine()
#plt.savefig("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/figures/length_distribution.pdf", bbox_inches='tight', dpi=300)
plt.show()


# %%
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

tpm_file = {
    "SR": "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/SR_283_transcript_TPM.txt",
    "LR": "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/LR_283_transcript_TPM.txt"
}

#%%
#######^^^^^^^^ Cancer gene subset #################

genelist = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/cancergenelist.txt', sep="\t", index_col=0)
genelist = genelist[(genelist['Oncogene']=='Yes')]
gene_map, _ = load_transcript_gene_map(tpm_file["SR"])
translist = gene_map[gene_map.isin(genelist.index)].index

SR_data_filt = SR_data.loc[SR_data.index.isin(translist)]

coding_status = pd.Series("coding", index=SR_data_filt.index)
coding_status.loc[coding_status.index.intersection(nc_lr_ids)] = "noncoding"

from scipy.stats import pearsonr
coding_status = pd.Series(
    data=['noncoding' if tx in nc_lr_ids else 'coding' for tx in SR_data_filt.index],
    index=SR_data_filt.index
)

LR_data_filt = LR_data.loc[LR_data.index.isin(SR_data_filt.index)]

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

plt.figure(figsize=(9, 5))

# Boxplot
ax = sns.boxplot(
    data=r_samp_df,
    x='Group', y='Pearson_r',
    hue='CodingStatus',
    palette={'coding': '#FFDE4A', 'noncoding': '#1BA39A'},
    showcaps=True, fliersize=0,  # 이상치 점 제거 (stripplot에 따로 표시되므로)
    boxprops=dict(alpha=1),   # box 약간 투명하게
    linewidth=1
)

# Stripplot (위에 점 겹쳐서 표현)
sns.stripplot(
    data=r_samp_df,
    x='Group', y='Pearson_r',
    hue='CodingStatus',
    palette={'coding': '#FFD723', 'noncoding': '#08736C'},
    dodge=True, jitter=True, alpha=1, size=3, linewidth=0
)

# 범례 정리 (두 번 그려져서 중복됨)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:2], labels[:2], title=None)

plt.ylabel("Pearson r")
plt.xlabel("")
plt.tight_layout()
sns.despine()
plt.title("Oncogene")
#plt.savefig("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/figures/pearsonr_violinplot_oncogene.pdf", bbox_inches='tight', dpi=300)
plt.show()


# %%
######^^^^^^^^ Clustering ###############
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

sns.set_style("whitegrid")
# 예시: TPM log2 변환 + NaN 처리
def preprocess(data, min_detect=24):
    data = np.log2(data + 1)
    filt = (data > 0).sum(axis=1) >= min_detect  # 최소 N개 샘플에서 발현된 transcript만
    return data.loc[filt]

# SR/LR 정제
SR_proc = preprocess(tmm)
LR_proc = preprocess(LRtmm)

# Transpose: PCA는 (sample × feature) 형태여야 함
SR_pca = PCA(n_components=2).fit_transform(SR_proc.T)
LR_pca = PCA(n_components=2).fit_transform(LR_proc.T)

# Optional: UMAP도 가능
SR_umap = umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=42).fit_transform(SR_proc.T)
LR_umap = umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=42).fit_transform(LR_proc.T)

# Sample metadata (e.g., tumor/normal)
sample_meta = pd.DataFrame({
    "sample": SR_proc.columns,
    "condition": ["Tumor" if "T" in s else "Normal" for s in SR_proc.columns],  # 필요시 수정
})

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
plt.tight_layout()
plt.savefig("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/figures/pca_T_N.pdf", bbox_inches='tight', dpi=300)
plt.show()

# %%
import umap
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# UMAP 임베딩
sr_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(SR_proc.T)
lr_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(LR_proc.T)

# plot용 데이터프레임
df_sr = pd.DataFrame(sr_umap, columns=['UMAP1', 'UMAP2'])
df_sr['condition'] = ['Tumor' if 'T' in s else 'Normal' for s in SR_proc.columns]

df_lr = pd.DataFrame(lr_umap, columns=['UMAP1', 'UMAP2'])
df_lr['condition'] = ['Tumor' if 'T' in s else 'Normal' for s in LR_proc.columns]

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.scatterplot(data=df_sr, x='UMAP1', y='UMAP2', hue='condition', ax=axes[0],
                palette=["#11710A", "#F2A900"], s=50)
axes[0].set_title("Short-read UMAP")
axes[0].set_xlabel("UMAP1")
axes[0].set_ylabel("UMAP2")
axes[0].legend(title='')

sns.scatterplot(data=df_lr, x='UMAP1', y='UMAP2', hue='condition', ax=axes[1], palette=["#4B74B6", "#E33C89"], s=50)
axes[1].set_title("Long-read UMAP")
axes[1].set_xlabel("UMAP1")
axes[1].set_ylabel("UMAP2")
axes[1].legend(title='')

plt.tight_layout()
sns.despine()
plt.savefig("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/figures/umap_T_N.pdf", bbox_inches='tight', dpi=300)
plt.show()

# %%
#%%
#^^^ correlation + matching percent barplot ######

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import seaborn as sns

# === 샘플별 Pearson r 계산 ===
def compute_samplewise_r(SR_df, LR_df):
    result = []
    for sample in SR_df.columns:
        x = SR_df[sample].values
        y = LR_df[sample].values
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() > 2:
            r, p = pearsonr(x[mask], y[mask])

        else:
            r, p = np.nan, np.nan
        result.append({'Sample': sample, 'Pearson_r': r})
    return pd.DataFrame(result)


# Step 1: Noncoding transcript만 필터링
SR_data_filt = SR_data
LR_data_filt = LR_data

###### zero LR transcript 제거 ########
# all_zero_lr = LR_data_filt[(LR_data_filt == 0).all(axis=1)].index
# LR_data_filt = LR_data_filt.drop(index=all_zero_lr)
# SR_data_filt = SR_data_filt.drop(index=all_zero_lr)
###### zero LR transcript 제거 ########

# Step 2: 매칭 정보 class_code → match type
def classify_class(code):
    if code == '=':
        return 'Full'
    elif code == 'c':
        return 'Partial'
    else:
        return 'Unmatched'

tmap['match_type'] = tmap['class_code'].apply(classify_class)

# Step 3: transcript → match type 딕셔너리 생성
tx_to_match = tmap.set_index('qry_id')['match_type'].to_dict()

# # Step 4: 각 샘플별 match type 비율 계산
# match_prop_df = []
# for sample in tmm.columns:
#     tx_vals = tmm[sample]
#     detected_tx = tx_vals[tx_vals > 0].index
    
#     match_counts = {'Full': 0, 'Partial': 0, 'Unmatched': 0}
    
#     for tx in detected_tx:
#         match = tx_to_match.get(tx, 'Unmatched')
#         match_counts[match] += 1
#     total = sum(match_counts.values())
#     match_prop_df.append({
#         'Sample': sample,
#         'Full': match_counts['Full']/total, #/ total if total else 0,
#         'Partial': match_counts['Partial']/ total,# / total if total else 0,
#         'Unmatched': match_counts['Unmatched']/ total,# / total if total else 0
#     })

# match_df = pd.DataFrame(match_prop_df)

match_prop_df = []
lentx = []
lensr = []

for sample in SR_data.columns:
    sr_vals = SR_data[sample]
    lr_vals = LR_data[sample]

    # 둘 다 발현량 > 0인 transcript만 사용
    detected_tx = sr_vals[(sr_vals > 0) & (lr_vals > 0)].index
    detected_sr = sr_vals[(sr_vals > 0)].index

    match_counts = {'Full': 0, 'Partial': 0, 'Unmatched': 0}
    for tx in detected_tx:
        match = tx_to_match.get(tx, 'Unmatched')
        match_counts[match] += 1

    total = sum(match_counts.values())
    match_prop_df.append({
        'Sample': sample,
        'Full': match_counts['Full'] / total if total else 0,
        'Partial': match_counts['Partial'] / total if total else 0,
        'Unmatched': match_counts['Unmatched'] / total if total else 0,
    })
    
    lentx.append(len(detected_tx))
    lensr.append(len(detected_sr))

match_df = pd.DataFrame(match_prop_df)

# Step 5: 샘플별 correlation 계산
r_df = compute_samplewise_r(SR_data_filt, LR_data_filt)


# Step 6: Group 정보 없이 sample 이름 기반으로 분류
merged = pd.merge(match_df, r_df, on='Sample')
merged['Group'] = merged['Sample'].apply(lambda s: 'Tumor' if 'T' in s else 'Normal')

# Step 7: Normal → Tumor 정렬 + Pearson r 내림차순 정렬
normal_df = merged[merged['Group'] == 'Normal'].sort_values('Pearson_r', ascending=False)
tumor_df = merged[merged['Group'] == 'Tumor'].sort_values('Pearson_r', ascending=False)
plot_df = pd.concat([normal_df, tumor_df]).reset_index(drop=True)
plot_df['x'] = range(len(plot_df))
nt_split = len(normal_df)

# === 시각화 ===
fig, ax1 = plt.subplots(figsize=(16, 5))

# Stacked barplot
ax1.bar(plot_df['x'], plot_df['Full'], color="#F9E400", label="Full match", width=1.0, edgecolor="#F9E400")
ax1.bar(plot_df['x'], plot_df['Partial'], bottom=plot_df['Full'], color="#178C84", label="Partial match", width=1.0, edgecolor="#178C84")
ax1.bar(plot_df['x'], plot_df['Unmatched'], bottom=plot_df['Full'] + plot_df['Partial'],
        color="#220039", label="Unmatched", width=1.0, edgecolor="#220039")

ax1.set_ylabel("Proportion of expressed transcripts")
ax1.set_xlabel("")
#ax1.set_ylim(0, 1)
ax1.set_xlim(-0.5, len(plot_df) - 0.5)
plt.yticks([0, 0.25, 0.5, 0.75, 1.0], ['0%', '25%', '50%', '75%', '100%'])
ax1.set_xticks([])
ax1.legend(loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0.31, 1.1))

# # Correlation lineplot
# ax2 = ax1.twinx()
# ax2.plot(plot_df["x"], plot_df["Pearson_r"], color="#FF8031", label="Pearson r", linewidth=2, alpha=0.7, linestyle=':')
# ax2.set_ylabel("Pearson r")
# ax2.set_ylim(0, 1)

# # 경계선
# #ax1.axvline(x=nt_split - 0.5, color='white', linestyle='--', lw=2)
# ax1.axhline(y=0.4, color='white', linestyle='-', lw=1, alpha=0.5)

sns.despine()
plt.tight_layout()
plt.show()



# %%
####^^ SQANTI3 #############################
sqantioutput = pd.read_csv("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/sqantioutput/LRsqanti_classification.txt", sep="\t")

# Step 1: long-read 전체 → category별 카운트
lr_category = sqantioutput.set_index('isoform')['structural_category']
cat_total = lr_category.value_counts()

# Step 2: full match된 long-read transcript ID
fullmatch_lr = tmap[tmap['class_code'] == '=']['ref_id'].unique()

# Step 3: 이 중 실제 SQANTI에 존재하는 것만 필터
matched_lr = [tx for tx in fullmatch_lr if tx in lr_category]

# Step 4: matched 된 transcript들의 카테고리 카운트
cat_matched = pd.Series([lr_category[tx] for tx in matched_lr]).value_counts()

# Step 5: join and fill
cat_df = pd.DataFrame({
    'Total': cat_total,
    'Matched': cat_matched
}).fillna(0)

cat_df['Unmatched'] = cat_df['Total'] - cat_df['Matched']
cat_df[['Matched', 'Unmatched']] = cat_df[['Matched', 'Unmatched']].astype(int)

# Step 6: stacked barplot용 long-form
cat_long = cat_df[['Matched', 'Unmatched']].reset_index().melt(
    id_vars='index', var_name='MatchStatus', value_name='Count'
).rename(columns={'index': 'Category'})

plt.figure(figsize=(10, 6))
sns.barplot(data=cat_long, x='Category', y='Count', hue='MatchStatus')
plt.xticks(rotation=45, ha='right')
plt.ylabel("Number of LR transcripts")
plt.title("Short-read full match coverage per LR category (SQANTI)")
plt.legend(title="SR match")
plt.tight_layout()
plt.show()


# %%
# Step 1. SR transcript 중 발현량 > 0 인 애들 (모든 샘플 합산)
detected_sr = tmm[tmm > 0].stack().index.get_level_values(0).unique()

# Step 2. gffcompare에서 full match인 애들 중 위와 겹치는 것만
full_match = tmap[tmap['class_code'] == '=']
matched_sr = full_match[full_match['qry_id'].isin(detected_sr)]

# Step 3. 대응되는 LR transcript들
matched_lr = matched_sr['ref_id'].unique()

# Step 4. 이 중 SQANTI 결과에 있는 것만 필터
sqanti_cat = sqantioutput.set_index('isoform')['structural_category']
matched_lr = [tx for tx in matched_lr if tx in sqanti_cat]

# Step 5. category 카운트
from collections import Counter
category_counts = Counter([sqanti_cat[tx] for tx in matched_lr])

# Step 6. pandas로 보기 좋게
cat_df = pd.DataFrame.from_dict(category_counts, orient='index', columns=['Count']).sort_values('Count', ascending=False)
print(cat_df)

# %%
# Step 1: 발현된 SR transcript 중 MSTRG만
detected_sr = tmm[tmm > 0].stack().index.get_level_values(0).unique()
detected_mstrg = [tx for tx in detected_sr if tx.startswith('MSTRG')]

# Step 2: gffcompare에서 full match된 것 중 MSTRG만
full_match = tmap[tmap['class_code'] == '=']
matched_sr_mstrg = full_match[full_match['qry_id'].isin(detected_mstrg)]

# Step 3: 대응되는 LR transcript ID들
matched_lr_mstrg = matched_sr_mstrg['ref_id'].unique()

# Step 4: SQANTI category lookup
sqanti_cat = sqantioutput.set_index('isoform')['structural_category']
matched_lr_mstrg = [tx for tx in matched_lr_mstrg if tx in sqanti_cat]

# Step 5: 카테고리 카운트
from collections import Counter
category_counts = Counter([sqanti_cat[tx] for tx in matched_lr_mstrg])

# Step 6: pandas dataframe으로
cat_df_mstrg = pd.DataFrame.from_dict(category_counts, orient='index', columns=['Count']).sort_values('Count', ascending=False)
print(cat_df_mstrg)

# %%
LRcov = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/LR_283_transcript_coverage.txt', sep='\t', index_col=0)
filtered_cov_LR = filter_by_coverage(LRcov, min_coverage=1, min_fraction=0.1)

# %%
