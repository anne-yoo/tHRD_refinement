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
# %%
"""
"DNA repair (GO:0006281)"
"base-excision repair (GO:0006284)"
"nucleotide-excision repair (GO:0006289)"
"mismatch repair (GO:0006298)"
"double-strand break repair via homologous recombination (GO:0000724)"
"double-strand break repair via nonhomologous end joining (GO:0006303)"
"""

with open('/home/jiye/jiye/copycomparison/gDUTresearch/202408_analysis/expHRD_comparison/expHRD_356_genes.txt', 'r') as file:
    explist = file.read()
explist = list(explist.strip("'").split("', '"))
idmatch = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/116_validation_gene_TPM_symbol_group.txt', sep='\t')
idmatch = idmatch[['Geneid','Gene Symbol']]
idmatch['Geneid'] = idmatch['Geneid'].str.split('.').str[0]
idmatch = idmatch.loc[idmatch['Geneid'].isin(explist),:]
finalexp = list(idmatch['Gene Symbol'])

hrr_core_genes = ['GEN1', 'BARD1', 'RAD50', 'SHFM1', 'XRCC2', 'NBN', 'MUS81', 'MRE11A', 'RAD52', 'BRCA2', 'XRCC3', 'RAD51C', 'RAD51D', 'TP53BP1', 'BLM', 'SLX1A', 'PALB2', 'TOP3A', 'BRCA1', 'EME1', 'BRIP1', 'RBBP8']

import gseapy as gp
go_results = gp.get_library("GO_Biological_Process_2021", organism="Human")

dnarepair = go_results["DNA repair (GO:0006281)"]
ber = go_results["base-excision repair (GO:0006284)"]
ner = go_results["nucleotide-excision repair (GO:0006289)"]
mmr = go_results["mismatch repair (GO:0006298)"]
hrr = go_results["double-strand break repair via homologous recombination (GO:0000724)"]
nhej = go_results["double-strand break repair via nonhomologous end joining (GO:0006303)"]

genelist = [ner,mmr,hrr,nhej,hrr_core_genes]

# %%

val_df = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/116_validation_TU_group.txt', sep='\t', index_col=0)
val_gene = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/116_validation_gene_TPM_symbol_group.txt', sep='\t', index_col=0)
val_clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_validation_clinicalinfo_new.txt', sep='\t', index_col=0)
val_df = val_df.iloc[:-1,:-1]
val_df = val_df.apply(pd.to_numeric, errors='coerce')
vallist = list(val_df.columns)
val_clin = val_clin.loc[vallist,:]
genesym = val_gene.iloc[:,-1]
val_gene.index = val_gene.iloc[:,-1]
val_gene = val_gene.iloc[:-1,:-1]
val_gene = val_gene.apply(pd.to_numeric, errors='coerce')

majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = majorminor[majorminor['type']=='major']['gene_ENST'].to_list()

druginfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/validation_sampleinfo.txt', sep='\t', index_col=0)
druginfo = druginfo['drug']

val_clin = val_clin.merge(druginfo, left_index=True, right_index=True, how='left')

from scipy.stats import spearmanr

val_gene = val_gene.loc[:,val_clin.index]
val_df = val_df.loc[:,val_clin.index]

olaparib_list = val_clin[val_clin['drug']=='Olaparib'].index.to_list()
niraparib_list = val_clin[val_clin['drug']=='Niraparib'].index.to_list()

main_list = val_clin[val_clin['OM/OS']=='maintenance'].index.to_list()
sal_list = val_clin[val_clin['OM/OS']=='salvage'].index.to_list()

main_gene = val_gene.loc[:,main_list]
sal_gene = val_gene.loc[:,sal_list]

main_tu = val_df.loc[:,main_list]
sal_tu = val_df.loc[:,sal_list]

main_clin = val_clin.loc[main_list,:]
sal_clin = val_clin.loc[sal_list,:]

# %%
gene_set_names = ['NER', 'MMR', 'HRR', 'NHEJ','HRR core genes']
from scipy.stats import mannwhitneyu

# ====== Plot ======
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
axes = axes.flatten()

for i, (genes, name) in enumerate(zip(genelist, gene_set_names)):
    ##################
    gene_df = main_gene
    tu_df = main_tu
    clin_df = main_clin.copy()
    ##################
    clin_df['group'] = clin_df['type'].replace({'CR': 'R', 'IR': 'NR', 'AR': 'R'})
    group_info = clin_df['group']
    r_samples = group_info[group_info == 'R'].index
    nr_samples = group_info[group_info == 'NR'].index

    # --- Subplot 1: Gene exp vs. PFS ---
    valid_genes = list(set(genes) & set(gene_df.index))
    gene_exp = gene_df.loc[valid_genes].mean(axis=0)
    pfs = clin_df['interval']

    ax1 = axes[i]
    sns.regplot(x=pfs, y=gene_exp, ax=ax1, scatter_kws={'s': 20}, line_kws={"color": "black"}, color='orange')
    r1, p1 = spearmanr(pfs, gene_exp)
    ax1.set_title(f"{name}\nGene Exp: r={r1:.2f}, p={p1:.3f}")
    ax1.set_xlabel("Treatment Duration")
    ax1.set_ylabel("Gene exp")

    # --- Subplot 2: TU vs. PFS ---
    major_transcripts = [t for t in majorlist if t.split('-')[-1] in genes]
    valid_trans = tu_df.index.intersection(major_transcripts)
    tu_values = tu_df.loc[valid_trans].mean(axis=0)

    ax2 = axes[i + 5]
    sns.regplot(x=pfs, y=tu_values, ax=ax2, scatter_kws={'s': 20}, line_kws={"color": "black"}, color='green')
    r2, p2 = spearmanr(pfs, tu_values)
    ax2.set_title(f"{name}\nMajor TU: r={r2:.2f}, p={p2:.3f}")
    ax2.set_xlabel("Treatment Duration")
    ax2.set_ylabel("Major TU")

    # --- Subplot 3: Volcano Plot ---
    deg_data = []
    for g in valid_genes:
        r_vals = gene_df.loc[g, r_samples]
        nr_vals = gene_df.loc[g, nr_samples]
        stat, pval = mannwhitneyu(r_vals, nr_vals, alternative='two-sided')
        fc = r_vals.mean() / (nr_vals.mean() + 1e-6)
        deg_data.append({'id': g, 'log2FC': np.log2(fc), 'pval': pval, 'type': 'gene'})

    dut_data = []
    for t in valid_trans:
        r_vals = tu_df.loc[t, r_samples]
        nr_vals = tu_df.loc[t, nr_samples]
        stat, pval = mannwhitneyu(r_vals, nr_vals, alternative='two-sided')
        fc = r_vals.mean() / (nr_vals.mean() + 1e-6)
        dut_data.append({'id': t, 'log2FC': np.log2(fc), 'pval': pval, 'type': 'transcript'})

    df_plot = pd.DataFrame(deg_data + dut_data)
    df_plot['-log10(pval)'] = -np.log10(df_plot['pval'])

    ax3 = axes[i + 10]
    sns.scatterplot(
        data=df_plot, x='log2FC', y='-log10(pval)', hue='type',
        palette={'gene': 'orange', 'transcript': 'green'},
        ax=ax3, s=30, alpha=0.5
    )
    ax3.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax3.set_title(f"{name}")
    ax3.set_xlabel("log2 FC (R / NR)")
    ax3.set_ylabel("-log10(p-value)")
    ax3.get_legend().remove()
    ax3.set_xlim(-5.2, 5.2)

plt.tight_layout()
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/PFScorr_volcano_maintenance.pdf', dpi=300, bbox_inches='tight')
plt.show()


# %%
###^^^^^^^^^ salvage R vs. NR boxplot ###############

stable_result = pd.read_csv(f'/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/salvagestable_DUT_FC.txt', sep='\t', index_col=0)
dutlist = stable_result.loc[
    (stable_result['p_value'] < 0.05) & (np.abs(stable_result['log2FC']) > 1.5)
].index.to_list()

from statannotations.Annotator import Annotator
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# 준비
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

# R/NR 정보
clin_df = sal_clin.copy()
clin_df['group'] = clin_df['type'].replace({'CR': 'R', 'IR': 'NR', 'AR': 'R'})
group_df = clin_df[['group']]

# loop
for i, (genes, name) in enumerate(zip(genelist, gene_set_names)):

    # === (1) term에 속하는 모든 major TU ===
    term_major = [t for t in majorlist if t.split('-')[-1] in genes]
    mean_expr_all = sal_tu.loc[sal_tu.index.intersection(term_major)].mean(axis=0)
    ct = len(sal_tu.index.intersection(term_major))
    df_all = pd.DataFrame({'Sample': mean_expr_all.index,
                           'MeanTU': mean_expr_all.values})
    df_all = df_all.merge(group_df, left_on='Sample', right_index=True)

    ax1 = axes[i]
    sns.boxplot(data=df_all, x='group', y='MeanTU', ax=ax1, palette={'R': '#FF8286', 'NR': '#77CDFF'},showfliers=False)
    sns.stripplot(data=df_all, x='group', y='MeanTU', ax=ax1, color='black', size=4, jitter=0.1, alpha=0.7)

    pairs = [('R', 'NR')]
    annotator = Annotator(ax1, pairs, data=df_all, x='group', y='MeanTU')
    annotator.configure(test='Mann-Whitney', text_format='simple', loc='inside')
    annotator.apply_and_annotate()

    ax1.set_title(f"{name} (n={ct})") # (n={ct})
    ax1.set_ylabel("mean TU")
    ax1.set_xlabel("")

    # === (2) term-major ∩ DUT ===
    term_major_dut = list(set(term_major) & set(dutlist))
    ax2 = axes[i + 5]
    ct2 = len(term_major_dut)

    if len(term_major_dut) == 0:
        ax2.axis('off')
        ax2.set_title(f"{name} (No overlap)")
        continue

    mean_expr_dut = sal_tu.loc[sal_tu.index.intersection(term_major_dut)].mean(axis=0)

    df_dut = pd.DataFrame({'Sample': mean_expr_dut.index,
                           'MeanTU': mean_expr_dut.values})
    df_dut = df_dut.merge(group_df, left_on='Sample', right_index=True)

    sns.boxplot(data=df_dut, x='group', y='MeanTU', ax=ax2, palette={'R': '#FF8286', 'NR': '#77CDFF'}, showfliers=False)
    sns.stripplot(data=df_dut, x='group', y='MeanTU', ax=ax2, color='black', size=4, jitter=0.1, alpha=0.7)

    annotator2 = Annotator(ax2, pairs, data=df_dut, x='group', y='MeanTU')
    annotator2.configure(test='Mann-Whitney', text_format='simple', loc='inside')
    annotator2.apply_and_annotate()

    ax2.set_title(f"{name} (n={ct2})") # (n={ct2})
    ax2.set_ylabel("mean TU")
    ax2.set_xlabel("")
    
    plt.tight_layout()  # 먼저 auto layout 조정
    fig.subplots_adjust(hspace=0.3, wspace=0.3) 
    plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/salvage_major_DUT_boxplot_count.pdf', dpi=300, bbox_inches='tight')


# %%
#^^^^ check for maintenance rescue #########################
val_df = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/116_validation_TU_group.txt', sep='\t', index_col=0)
val_gene = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/116_validation_gene_TPM_symbol_group.txt', sep='\t', index_col=0)
val_clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_validation_clinicalinfo_new.txt', sep='\t', index_col=0)
val_df = val_df.iloc[:-1,:-1]
val_df = val_df.apply(pd.to_numeric, errors='coerce')
vallist = list(val_df.columns)
val_clin = val_clin.loc[vallist,:]
genesym = val_gene.iloc[:,-1]
val_gene.index = val_gene.iloc[:,-1]
val_gene = val_gene.iloc[:-1,:-1]
val_gene = val_gene.apply(pd.to_numeric, errors='coerce')

majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = majorminor[majorminor['type']=='major']['gene_ENST'].to_list()

druginfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/validation_sampleinfo.txt', sep='\t', index_col=0)
druginfo = druginfo['drug']

val_clin = val_clin.merge(druginfo, left_index=True, right_index=True, how='left')


genelist_tmp = [nhej,hrr,hrr_core_genes]
gene_set_names_tmp = ['NHEJ','HRR','HRR core gene']
topduts = []
for i, (genes, name) in enumerate(zip(genelist_tmp, gene_set_names_tmp)):
    term_major = [t for t in majorlist if t.split('-')[-1] in genes]
    term_major_dut = list(set(term_major) & set(dutlist))
    #term_major_dut = term_major
    topduts = topduts + term_major_dut
topduts = list(set(topduts))

pred_proba = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/HGmodel/116_HGmodel_proba.txt', sep='\t')
val_clin['tHRD'] = list(pred_proba['pred_HRD'])
val_clin['group'] = val_clin['type'].replace({'CR': 'R', 'IR': 'NR', 'AR': 'R'})

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from sklearn.preprocessing import MinMaxScaler
    
val_clin['4group'] = val_clin['OM/OS'] + '_' + val_clin['group']
val_clin['BRCAmut'] = val_clin['BRCAmut'].astype(int)

# 2. 정렬 순서 기준
group_order = ['salvage_R', 'salvage_NR', 'maintenance_R', 'maintenance_NR']
sample_order = []

#################################
# val_clin = val_clin[val_clin['drug']=='Niraparib']  # Olaparib만 필터링
###################################




for group in group_order:
    for brca in [1, 0]:
        subset = val_clin[(val_clin['4group'] == group) & (val_clin['BRCAmut'] == brca)]
        subset_sorted = subset.sort_values(by='interval', ascending=True)
        sample_order.extend(subset_sorted.index.tolist())

###########################################
# # 1. 그룹 라벨 생성
# val_clin['4group'] = val_clin['OM/OS'] + '_' + val_clin['group']  # e.g. 'salvage_R'

# # 2. 정렬 순서 정의
# group_order = ['salvage_R', 'salvage_NR', 'maintenance_R', 'maintenance_NR']

# # 3. 각 그룹 내에서 interval 오름차순 정렬
# sample_order = []
# for group in group_order:
#     subset = val_clin[val_clin['4group'] == group]
#     subset_sorted = subset.sort_values(by='interval', ascending=True)
#     sample_order.extend(subset_sorted.index.tolist())

#############################################3

# === palette 설정 ===
fourgroup_palette = {
    'salvage_R': '#FF8286',
    'salvage_NR': '#77CDFF',
    'maintenance_R': '#FF9C9F',
    'maintenance_NR': '#90D6FF'
}
brca_palette = {1: '#EABDE6', 0: '#FFFFFF'}
drug_palette = {'Olaparib': '#C0E218', 'Niraparib': '#F9F871'}
gray_cmap = plt.get_cmap('Greys')

# === continuous 값 스케일링 ===
val_clin['tHRD_scaled'] = MinMaxScaler().fit_transform(val_clin[['tHRD']])
val_clin['gHRD_scaled'] = MinMaxScaler().fit_transform(val_clin[['gHRDscore']])

# === 컬러 스트립 행 만들기 ===
def map_color(series, palette, default='#FFFFFF'):
    mapped = series.map(palette)
    return mapped.fillna(default).values

def safe_graymap(v, fallback=(0.9, 0.9, 0.9, 1.0)):  # 밝은 회색
    return gray_cmap(v) if not pd.isna(v) else fallback

color_rows = [
    map_color(val_clin.loc[sample_order, '4group'], fourgroup_palette),
    map_color(val_clin.loc[sample_order, 'BRCAmut'], brca_palette),
    #map_color(val_clin.loc[sample_order, 'OM/OS'], om_palette),
    #map_color(val_clin.loc[sample_order, 'group'], group_palette),
    map_color(val_clin.loc[sample_order, 'drug'], drug_palette),
    [safe_graymap(v) for v in val_clin.loc[sample_order, 'gHRD_scaled']]
,
    [gray_cmap(v) for v in val_clin.loc[sample_order, 'tHRD_scaled']],
]


# mean_tu_row = inputdf[sample_order].mean()

# mean_tu_scaled = MinMaxScaler().fit_transform(mean_tu_row.values.reshape(-1,1)).flatten()
# mean_tu_colors = [white_to_red(v) for v in mean_tu_scaled]


# color_rows.append(mean_tu_colors)


# === 색상 문자열 → RGB tuple 변환 ===
import matplotlib.colors as mcolors

def to_rgb_strict(c):
    """모든 색상을 RGB 3채널 float tuple로 통일"""
    try:
        # hex string → RGB
        if isinstance(c, str):
            return mcolors.to_rgb(c)
        # RGBA tuple → RGB
        elif isinstance(c, (tuple, list)) and len(c) == 4:
            return tuple(c[:3])
        # 이미 RGB tuple
        elif isinstance(c, (tuple, list)) and len(c) == 3:
            return tuple(c)
        else:
            raise ValueError(f"Unknown color format: {c}")
    except Exception as e:
        print(f"Color conversion failed for: {c} → using white")
        return (1.0, 1.0, 1.0)  # fallback: white


from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler

# === 그림 그리기 ===
row_labels = ['Group', 'BRCAmut', 'Drug',
              'gHRD (scaled)', 'tHRD (scaled)', ]

# === 여러 gene set에 대해 mean TU 색상 줄 추가 ===
from matplotlib.colors import LinearSegmentedColormap
white_to_red = LinearSegmentedColormap.from_list("white_green", ['#FFFFFF', '#00AA00'])

# 이 부분만 genelist별로 확장

genelist_tmp = [ner,hrr,nhej,hrr_core_genes]
gene_set_names_tmp = ['NER','HRR','NHEJ','HRR core gene']


for geneset, name in zip(genelist_tmp, gene_set_names_tmp):
    term_major = [t for t in majorlist if t.split('-')[-1] in geneset]
    term_major_dut = list(set(term_major) & set(dutlist))
    #term_major_dut = term_major
    # 해당 gene set의 TU row (샘플 평균)
    tu_df_subset = val_df.loc[val_df.index.intersection(term_major_dut), sample_order]
    print(name, val_df.index.intersection(term_major_dut))
    if tu_df_subset.shape[0] == 0:
        print(f"[!] No matched DUTs for {name}")
        row_colors = [(1.0, 1.0, 1.0)] * len(sample_order)  # white
    else:
        cmap = cm.get_cmap('Reds')  # 또는 'inferno', 'viridis', etc.
        row = tu_df_subset.mean(axis=0)
        row_scaled = MinMaxScaler().fit_transform(row.values.reshape(-1,1)).flatten()
        row_colors = [cmap(v) for v in row_scaled]

    color_rows.append(row_colors)
    row_labels.append(f"{name}")

# 예시 컬러맵 (흰색 → 파란색)
from matplotlib.colors import LinearSegmentedColormap
white_to_blue = LinearSegmentedColormap.from_list("white_blue", ['#FFFFFF', '#00AA00'])

# === 투약 기간 스케일링 및 색상 변환 ===
val_clin['duration_scaled'] = MinMaxScaler().fit_transform(
    val_clin[['interval']]  # <- 실제 컬럼명으로 바꿔야 함!
)

duration_colors = [white_to_blue(v) for v in val_clin.loc[sample_order, 'duration_scaled']]

# === 컬러 스트립 및 레이블에 추가 ===
color_rows.insert(5, duration_colors)  # 원하는 위치로 조정 가능
row_labels.insert(5, 'Treatment Duration')

# 1. 팔레트 지정
ongoing_palette = {
    0: '#90B2D6',  # PD 투약 종료 (resistance)
    1: '#FBF3C1', # ongoing
    2: '#EBA7A7', #NED
    3: '#7b7b7b',
    4: '#FFFFFF' # other reason
}

# 2. 컬러 행 생성
ongoing_colors = map_color(val_clin.loc[sample_order, 'ongoing'], ongoing_palette, default='#ffffff')

# 3. 위치 지정해서 추가 (예: 마지막에)
color_rows.insert(6,ongoing_colors)
row_labels.insert(6,"Ongoing (0–4)")

# ####### 위에 DUT들 합집합 ################################
# merged_df = val_df.loc[topduts, sample_order]
# cmap = cm.get_cmap('Reds')  # 또는 'inferno', 'viridis', etc.
# row2 = merged_df.mean(axis=0)
# row2_scaled = MinMaxScaler().fit_transform(row2.values.reshape(-1,1)).flatten()
# row2_colors = [cmap(v) for v in row2_scaled]
# color_rows.insert(13, row2_colors)  # 원하는 위치로 조정 가능
# row_labels.insert(13, 'merged')
# ####### 위에 DUT들 합집합 ################################


# --- 모든 row를 동일한 길이로 갖는 RGB 튜플 배열로 변환 ---
rgb_array = np.array([
    [to_rgb_strict(c) for c in row]
    for row in color_rows
], dtype=float)

fig, ax = plt.subplots(figsize=(18, 4))
ax.imshow(rgb_array, aspect='auto')
ax.set_yticks(np.arange(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=10)
ax.set_xticks([])
ax.set_xlabel("")
ax.set_title("", fontsize=14)
plt.tight_layout()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/majorTUmean_heatmap_genesets.pdf', dpi=300, bbox_inches='tight')
plt.show()
#%%

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# treatment duration 범위에 맞는 범례 컬러바
norm = Normalize(vmin=val_clin['interval'].min(), vmax=val_clin['interval'].max())
sm = ScalarMappable(norm=norm, cmap=white_to_blue)

# 별도 figure or subplot에 추가
fig, ax = plt.subplots(figsize=(6, 1))
fig.colorbar(sm, cax=ax, orientation='horizontal', label='Treatment Duration')
plt.show()
#%%
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor=color, edgecolor='black', label=f'Ongoing = {k}')
    for k, color in ongoing_palette.items()
]

fig, ax = plt.subplots(figsize=(5, 1))
ax.legend(handles=legend_elements, loc='center', ncol=5, frameon=False)
ax.axis('off')
plt.title("Ongoing Status")
plt.tight_layout()
plt.show()

#%%
### simple check ######
atmp = val_clin.loc[sample_order]
atmp['meantu'] = row
tmp = atmp[atmp['OM/OS']=='maintenance']
#sns.scatterplot(data=tmp, x='interval', y='meantu', hue='BRCAmut')
# tmp = tmp[(tmp['meantu']>0.1) & (tmp['interval']<360)]
tmp = tmp[tmp['BRCAmut']==0]
print(tmp.shape)


from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, precision_recall_curve, auc, classification_report
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Define true labels and TU scores
y = (tmp["interval"] >= 360).astype(int)     # 1 = Responder, 0 = Non-responder
score = 1 - tmp["meantu"]                    # Lower TU = more likely responder

# 2. Compute ROC and find best threshold using Youden's J
fpr, tpr, thresholds = roc_curve(y, score)
youden_j = tpr - fpr
best_idx = np.argmax(youden_j)
best_threshold = thresholds[best_idx]
optimal_tu_cutoff = 1 - best_threshold        # Convert back to TU scale

# 3. Predict using optimal threshold
y_pred = (score >= best_threshold).astype(int)

# 4. Confusion matrix and accuracy
cm = confusion_matrix(y, y_pred)
acc = accuracy_score(y, y_pred)
report = classification_report(y, y_pred, output_dict=True)

# 5. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y, score)
pr_auc = auc(recall, precision)

# 6. Print outputs
print("Optimal TU threshold (cutoff):", round(optimal_tu_cutoff, 3))
print("Accuracy:", round(acc, 3))
print("Confusion Matrix:\n", cm)
print("\nClassification Report:")
print(pd.DataFrame(report).T)
print("\nPR AUC:", round(pr_auc, 3))

# %%
###################^^^ SIMPLE RANDOMFOREST #########################################
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, accuracy_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Binary label
X = val_df.loc[topduts].T.copy()
#X['BRCAmut'] = val_clin['BRCAmut']
X['OM/OS'] = val_clin['OM/OS']
X['group'] = val_clin['group']
X.loc[(val_clin['interval'] < 360) & (val_clin['group'] == 'R') & (val_clin['ongoing'] != 2), 'group'] = 'NR'

X = X.dropna(subset=['group'])  # Ensure 'group' has no NaNs
from sklearn.preprocessing import LabelEncoder

# Label encoding for 'OM/OS'
le = LabelEncoder()
X['OM/OS'] = le.fit_transform(X['OM/OS'].astype(str))  # 혹시 NaN이 있다면 .astype(str)로 안전하게 처리

y = X['group'].map({'NR': 0, 'R': 1})

# Step 2: Features and split
X_feat = X.drop(columns=['group'])

X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size=0.3, random_state=42, stratify=y)

from sklearn.model_selection import RandomizedSearchCV

# Step 1: Set up parameter grid
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Step 2: Run RandomizedSearchCV
clf = RandomForestClassifier(random_state=42)
search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=30, 
                            cv=5, scoring='f1', random_state=42, n_jobs=1, verbose=1)

search.fit(X_train, y_train)

# Step 3: Evaluate best model
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"[Tuned] Accuracy: {acc:.3f}")

# ROC AUC
roc_auc = roc_auc_score(y_test, y_proba)
print(f"[Tuned] ROC AUC: {roc_auc:.3f}")

# PR AUC
pr_auc = average_precision_score(y_test, y_proba)
print(f"[Tuned] PR AUC: {pr_auc:.3f}")

# Step 4: Plot updated ROC + PR + Confusion Matrix
# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Tuned)')
plt.legend()
plt.show()

# PR Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure()
plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Tuned)')
plt.legend()
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Tuned)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Best parameters
print("Best Params:", search.best_params_)

importances = best_model.feature_importances_
features = X_feat.columns

# Create a DataFrame for sorting
feat_imp_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot top N important features (e.g., top 20)
top_n = 20
plt.figure(figsize=(8, top_n * 0.4))
sns.barplot(data=feat_imp_df.head(top_n), x='Importance', y='Feature', palette='viridis')
plt.title(f'Top {top_n} Feature Importances (Random Forest)')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()



# %%
3#####^^ venn diagram check
from venn import venn
stable_result = pd.read_csv(f'/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/salvagevariable_DUT_FC.txt', sep='\t', index_col=0)
saldutlist = stable_result.loc[
    (stable_result['p_value'] < 0.05) & (np.abs(stable_result['log2FC']) > 1.5)
].index.to_list()

stable_result = pd.read_csv(f'/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/maintenancevariable_DUT_FC.txt', sep='\t', index_col=0)
maindutlist = stable_result.loc[
    (stable_result['p_value'] < 0.05) & (np.abs(stable_result['log2FC']) > 1.5)
].index.to_list()


sets = {'salvage': set(saldutlist)&set(majorlist),
        'maintenance': set(maindutlist)&set(majorlist),}

fig, ax = plt.subplots(figsize=(5, 5))  # Adjust size here
venn(sets, ax=ax)
plt.show()

# %%
######^^ GO enrichment
import gseapy as gp

stable_result = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/salvagestable_DUT_FC.txt', sep='\t', index_col=0)
#glist = list(DEGlist)
glist = list(set(stable_result[(stable_result['p_value']<0.05) & (np.abs(stable_result['log2FC'])>1.5)]['Gene Symbol']))
#glist = list(set(variable_result[(variable_result['p_value']<0.05) & (np.abs(variable_result['log2FC']>1.5))]['Gene Symbol']))
enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                gene_sets=['GO_Biological_Process_2021',
                        #'Reactome_2022'
                        #'GO_Biological_Process_2018','GO_Biological_Process_2023','Reactome_2022'
                        ], 
                organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                outdir=None, # don't write to disk
)

enrresult = enr.results.sort_values(by=['Adjusted P-value']) 
file = enrresult
def string_fraction_to_float(fraction_str):
    numerator, denominator = fraction_str.split('/')
    return float(numerator) / float(denominator)

file['per'] = file['Overlap'].apply(string_fraction_to_float)

file = file.sort_values('Adjusted P-value')
#file = file.sort_values(by='Combined Score', ascending=False)
##remove mitochondrial ##
#file['Term'] = file['Term'].str.rsplit(" ",1).str[0]
#file = file[~file['Term'].str.contains('mitochondrial')]
#file = file.iloc[:10,:]
file = file[file['Adjusted P-value']<0.01]
file = file.iloc[:30,:]
file['Adjusted P-value'] = -np.log10(file['Adjusted P-value'])

plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 12,    # x축 틱 라벨 글꼴 크기
'ytick.labelsize': 12,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 13,
'legend.title_fontsize': 13, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})
sns.set_style("ticks")

fig, ax = plt.subplots(figsize=(6, 6))

# Create a horizontal bar plot
bars = ax.barh(file['Term'], file['Adjusted P-value'], color='#154674')
ax.set_xlabel('-log10(adjp)')

# Adjust y-axis tick labels: set font size and ensure labels are fully visible
ax.set_yticklabels(file['Term'], fontsize=10)

# Invert y-axis to have the lowest p-value at the top
ax.invert_yaxis()

# Adjust subplot parameters to give more room for y-axis labels
plt.subplots_adjust(left=0.65)  # Adjust the left margin as needed
sns.despine()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/DEG_downregulated_GOplot.pdf', dpi=300, bbox_inches='tight')
plt.show()
# %%
