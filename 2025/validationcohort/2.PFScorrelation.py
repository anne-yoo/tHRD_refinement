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

genelist = [dnarepair,ber,ner,mmr,hrr,nhej]
genelist_2 = [finalexp,hrr_core_genes]


#%%
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

########################
#val_clin = val_clin[val_clin['drug']=='Niraparib']
########################

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

#%%
genelist = [dnarepair,ber,ner,mmr,hrr,nhej,finalexp,hrr_core_genes]
gene_set_names = ['DNA Repair', 'BER', 'NER', 'MMR', 'HRR', 'NHEJ','expHRD', 'HRR core genes']

# ====== Plot ======
fig, axes = plt.subplots(2, 8, figsize=(32, 8))
axes = axes.flatten()

for i, (genes, name) in enumerate(zip(genelist, gene_set_names)):
    
    gene_df = val_gene
    tu_df = val_df
    clin_df = val_clin
    
    # --- Gene expression 평균 ---
    valid_genes = list(set(genes) & set(val_gene.index))
    gene_exp = gene_df.loc[valid_genes].mean(axis=0)

    # --- Major TU 평균 ---
    # majorlist 중 해당 gene에 속하는 transcript만 필터
    major_transcripts = [t for t in majorlist if t.split('-')[-1] in genes]
    tu_values = tu_df.loc[tu_df.index.intersection(major_transcripts)].mean(axis=0)

    # --- PFS ---
    pfs = clin_df['interval']

    # --- Subplot 1 (Major TU vs PFS) ---
    ax1 = axes[i]
    sns.regplot(x=pfs, y=gene_exp, ax=ax1, scatter_kws={'s': 20}, line_kws={"color": "black"}, color='green')
    r1, p1 = spearmanr(pfs, gene_exp)
    ax1.set_title(f"{name}\nr={r1:.2f}, p={p1:.3f}")
    ax1.set_xlabel("PFS")
    ax1.set_ylabel("Gene exp")

    # --- Subplot 2 (Gene exp vs PFS) ---
    ax2 = axes[i+8]
    sns.regplot(x=pfs, y=tu_values, ax=ax2, scatter_kws={'s': 20}, line_kws={"color": "black"}, color='green')
    r2, p2 = spearmanr(pfs, tu_values)
    ax2.set_title(f"{name}\nr={r2:.2f}, p={p2:.3f}")
    ax2.set_xlabel("PFS")
    ax2.set_ylabel("Major TU")

plt.tight_layout()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/PFScorr_whole_niraparib.pdf', dpi=300, bbox_inches='tight')
plt.show()




# %%
#####^^ heatmap #####################

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
pred_proba = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/HGmodel/116_HGmodel_proba.txt', sep='\t')
val_clin['tHRD'] = pred_proba['pred_HRD']
val_clin['group'] = val_clin['type'].replace({'CR': 'R', 'IR': 'NR','AR': 'R'})
val_clin['BRCAmut'] = val_clin['BRCAmut'].replace({1: 'MT', 0: 'WT'})


# ===== 1. Filter to major transcripts =====
genesubset = dnarepair
major_transcripts = [t for t in majorlist if t.split('-')[-1] in genesubset]
heatmat = val_df.loc[val_df.index.intersection(major_transcripts),sal_list]
## only salvage

# ===== 2. Prepare column annotations =====
# Only keep relevant annotation columns and align sample order
annot_cols = ['OM/OS', 'gHRDscore', 'tHRD', 'group', 'BRCAmut', 'drug']
col_anno = val_clin.loc[heatmat.columns, annot_cols]

# Convert categorical to category dtype if needed
cat_cols = ['OM/OS', 'group', 'BRCAmut','drug']
for col in cat_cols:
    col_anno[col] = col_anno[col].astype('category')

# ===== 3. Color palettes for annotations =====
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

# Define colors manually or use seaborn palettes
anno_colors = {
    'OM/OS': {'maintenance': '#FFD700', 'salvage': '#20B2AA'},
    'group': {'R': '#FF6F61', 'NR': '#88B04B'},
    'BRCAmut': {'MT': '#E69F00', 'WT': '#56B4E9'},
    'drug': {'Olaparib': '#D55E00', 'Niraparib': '#009E73'},
}
cont_cols = ['gHRDscore', 'tHRD']

col_anno_cat = col_anno[cat_cols].copy()
for col in cat_cols:
    col_anno_cat[col] = col_anno_cat[col].astype(str)  # 문자열로 바꿔주기 (safe for replace)

# 색상 매핑
col_colors = col_anno_cat.copy()
for col in cat_cols:
    col_colors[col] = col_colors[col].map(anno_colors[col])

#%%

# clustermap 그리기
g = sns.clustermap(
    heatmat,
    cmap="vlag",
    row_cluster=True,
    col_cluster=True,
    xticklabels=False,
    yticklabels=False,
    figsize=(12, 15),
    method="ward",     # ← 여기에서 method 설정
    metric="euclidean",# 거리 메트릭도 함께 설정 가능
    col_colors=col_colors,
    z_score=1 # sample x annotation 형태여야 함
)

# Optional: Add colorbars for continuous annotations
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Add legends manually
handles = []
for col, color_map in anno_colors.items():
    for label, color in color_map.items():
        handles.append(Patch(facecolor=color, label=f"{col}: {label}"))

g.ax_col_dendrogram.legend(
    handles=handles,
    title="Annotations",
    loc="center left",
    bbox_to_anchor=(1.0, 0.5)
)

plt.show()

#%%













# %%
#########^^^^^UMAP ########################

from sklearn.decomposition import PCA
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns

# ---- (1) TU 데이터 준비 ----
# 모든 major transcript 사용
full_data = val_df.loc[val_df.index.isin(majorlist)]

# 특정 gene list 사용
def get_subset_by_genes(df, genelist):
    return df[df.index.map(lambda x: x.split('-')[-1] in genelist)]

# ---- (2) 샘플 기준 전치 + z-score ----
from scipy.stats import zscore
def preprocess_for_embedding(df):
    df_z = df.apply(zscore, axis=1).dropna()
    return df_z.T  # sample x feature matrix

# ---- (3) 임베딩 함수 ----
def run_embedding(data, method='pca', n_components=2):
    if method == 'pca':
        model = PCA(n_components=n_components, random_state=0)
    elif method == 'umap':
        model = umap.UMAP(n_components=n_components, random_state=0)
    else:
        raise ValueError("Only 'pca' or 'umap' supported")
    return model.fit_transform(data)

# ---- (4) 시각화 함수 ----
def plot_embedding(embedding, labels, title):
    df_plot = pd.DataFrame(embedding, columns=['Dim1', 'Dim2'])
    df_plot['Group'] = labels.values
    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=df_plot, x='Dim1', y='Dim2', hue='Group', palette='Set1', s=60)
    plt.title(title)
    plt.legend(title='OM/OS')
    plt.tight_layout()
    plt.show()

# X_full = preprocess_for_embedding(full_data)
# embed_full = run_embedding(X_full, method='umap')
# plot_embedding(embed_full, val_clin.loc[X_full.index, 'OM/OS'], title="UMAP: All Major Transcripts")


subset_data = get_subset_by_genes(val_df.loc[val_df.index.isin(majorlist)], hrr_core_genes)
X_subset = preprocess_for_embedding(subset_data)
embed_subset = run_embedding(X_subset, method='umap')
plot_embedding(embed_subset, val_clin.loc[X_subset.index, 'OM/OS'], title="UMAP: HRR core genes")



# %%
# UMAP 결과 그대로 사용 (예: embed_full or embed_subset)
X_plot = pd.DataFrame(embed_subset, columns=['Dim1', 'Dim2'], index=X_subset.index)

# 그룹 정의
meta = val_clin.loc[X_plot.index, ['OM/OS', 'group']]
X_plot['Group4'] = meta['OM/OS'] + '_' + meta['group']
X_plot['Group4'] = X_plot['Group4'].replace({
    'maintenance_R': 'maint_R',
    'maintenance_NR': 'maint_NR',
    'salvage_R': 'salv_R',
    'salvage_NR': 'salv_NR'
})

# 시각화
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=X_plot,
    x='Dim1', y='Dim2',
    hue='Group4',
    palette={'maint_R': '#1f77b4', 'maint_NR': '#aec7e8', 'salv_R': '#d62728', 'salv_NR': '#ff9896'},
    s=60
)
plt.title("UMAP: 4-group comparison (OM/OS + Response)")
plt.legend(title='Group')
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator

# 4x2 subplot
fig, axes = plt.subplots(2, 4, figsize=(18, 12))
axes = axes.flatten()

palette = {"R": "#FF9E9E", "NR": "#769CFF"}

for i, (genes, name) in enumerate(zip(genelist, gene_set_names)):
    # 1. 해당 gene set에 속한 major transcript만 필터링
    major_subset = [t for t in majorlist if t.split('-')[-1] in genes]
    tu_data = val_df.loc[val_df.index.intersection(major_subset)]

    # 2. sample별 평균 TU 계산
    tu_mean = tu_data.mean(axis=0)  # index: sample

    # 3. metadata 붙이기
    df_plot = val_clin.loc[tu_mean.index, ['OM/OS', 'group']].copy()
    df_plot['mean_TU'] = tu_mean

    # 4. boxplot 그리기
    ax = axes[i]
    sns.boxplot(data=df_plot, x='OM/OS', y='mean_TU', hue='group', ax=ax, palette=palette)

    # 5. annotation 설정
    pairs = [
        (("maintenance", "R"), ("maintenance", "NR")),
        (("salvage", "R"), ("salvage", "NR"))
    ]
    annotator = Annotator(ax, pairs, data=df_plot, x='OM/OS', y='mean_TU', hue='group')
    annotator.configure(test='Mann-Whitney', text_format='simple', loc='inside', verbose=0)
    annotator.apply_and_annotate()

    ax.set_title(name)
    ax.set_xlabel("")
    ax.set_ylabel("")


plt.tight_layout()
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/repairgroup_MS_boxplot.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
