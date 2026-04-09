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
'legend.fontsize': 12,
'legend.title_fontsize': 12, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})
sns.set_style("ticks")
# %%

DDRlist = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/DDR_genelist_whole.txt', sep='\t')
DDRcorelist = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/DDR_genelist_core.txt', sep='\t')

from collections import OrderedDict

rename_map_1 = {'Nucleotide Excision Repair (NER) - includes TC-NER and GC-NER': 'Nucleotide Excision Repair (NER)'}
rename_map_2 = {'Nucleotide Excision Repair (NER, including TC-NER and GC-NER))': 'Nucleotide Excision Repair (NER)', 
                'Homologous Recomination (HR)': 'Homologous Recombination (HR)',}

ddr_gene = {}

for col in DDRlist.columns:
    # NaN 제거하고 리스트로 변환
    genes = DDRlist[col].dropna().tolist()
    ddr_gene[col] = genes

ddr_genelist = OrderedDict()
for k, v in ddr_gene.items(): 
    new_k = rename_map_1.get(k, k)
    ddr_genelist[new_k] = v

ddr_coregene = {}

for col in DDRcorelist.columns:
    # NaN 제거하고 리스트로 변환
    genes = DDRcorelist[col].dropna().tolist()
    ddr_coregene[col] = genes

ddr_coregenelist = OrderedDict()
for k, v in ddr_coregene.items():
    new_k = rename_map_2.get(k, k)
    ddr_coregenelist[new_k] = v

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
###^^^ t-SNE #################################
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

clin_df = val_clin.copy()
tu_df = val_df.copy()
major_tu = tu_df.loc[tu_df.index.isin(majorlist)]

# 2. transpose → samples x features
X = major_tu.T

# 3. scaling
X_scaled = StandardScaler().fit_transform(X)

from sklearn.decomposition import PCA
import umap.umap_ as umap

# 1. PCA 선처리
# pca = PCA(n_components=50, random_state=42)
# X_pca = pca.fit_transform(X_scaled)

# 2. UMAP 적용
reducer = umap.UMAP(random_state=42) #n_neighbors=15, min_dist=0.1, metric='euclidean', 
X_umap = reducer.fit_transform(X_scaled)

# 3. 시각화 준비
umap_df = pd.DataFrame(X_umap, columns=['UMAP1', 'UMAP2'], index=X.index)
clin_df['group'] = clin_df['type'].replace({'CR': 'R', 'IR': 'NR', 'AR': 'R'})
umap_df = umap_df.merge(clin_df, left_index=True, right_index=True)
umap_df['BRCA'] = umap_df['BRCAmut'].map({0: 'BRCAwt', 1: 'BRCAmt'})

#%%
# 4. 플롯


sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='line_binary', style='OM/OS', s=80, alpha=0.9, palette='YlOrRd') #YlOrRd
plt.title('UMAP of TU space (no PCA)')
plt.tight_layout()
plt.show()

# %%
#^^^ NMF #################################
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


clin_df = main_clin.copy()
tu_df = main_tu.copy()
brca_wt = clin_df[clin_df['BRCAmut'] == 0]
brca_mt = clin_df[clin_df['BRCAmut'] == 1]

# TU dataframe도 맞춰서 subset
tu_wt = tu_df.loc[tu_df.index.isin(majorlist), brca_wt.index]
tu_mt = tu_df.loc[tu_df.index.isin(majorlist), brca_mt.index]

from scipy.stats import ks_2samp

def ks_test_tu(tu_subdf, clin_subdf):
    tu_1L = tu_subdf.loc[:, clin_subdf[clin_subdf['line_binary'] == 'FL'].index]
    tu_2L = tu_subdf.loc[:, clin_subdf[clin_subdf['line_binary'] != 'FL'].index]
    
    results = []
    for tx in tu_subdf.index:
        vals_1L = tu_1L.loc[tx]
        vals_2L = tu_2L.loc[tx]
        stat, pval = ks_2samp(vals_1L, vals_2L)
        results.append((tx, stat, pval))
    return pd.DataFrame(results, columns=['Transcript', 'KS_stat', 'pval'])

df_wt = ks_test_tu(tu_wt, brca_wt)
df_mt = ks_test_tu(tu_mt, brca_mt)
df_all = ks_test_tu(tu_df.loc[tu_df.index.isin(majorlist), :], clin_df)

sig_wt = df_wt[df_wt['pval'] < 0.05]
sig_mt = df_mt[df_mt['pval'] < 0.05]
sig_all = df_all[df_all['pval'] < 0.05]

#%%
df_wt['FDR'] = multipletests(df_wt['pval'], method='fdr_bh')[1]
df_mt['FDR'] = multipletests(df_mt['pval'], method='fdr_bh')[1]
df_all['FDR'] = multipletests(df_all['pval'], method='fdr_bh')[1]

sig_wt = df_wt[df_wt['FDR'] < 0.1]
sig_mt = df_mt[df_mt['FDR'] < 0.1]
sig_all = df_all[df_all['FDR'] < 0.1]

clin_df = val_clin.copy()
tu_df = val_df.copy()
major_tu = tu_df.loc[tu_df.index.isin(majorlist)]
major_tu =  major_tu.loc[major_tu.index.isin(sig_wt['Transcript'])]  # KS test 통과한 transcript만 사용
# TU matrix: sample × transcript (e.g., tu_df)
X = major_tu.T.values  # numpy array로 변환
X = normalize(X, axis=1)  # 정규화 (optional)

# NMF 클러스터 개수 설정
k = 3  # 또는 2, 4 등 실험

nmf = NMF(n_components=k, init='random', random_state=42, max_iter=500)
W = nmf.fit_transform(X)  # sample × k
H = nmf.components_       # k × transcript

# 가장 높은 weight 기준으로 샘플에 클러스터 label 지정
cluster_labels = W.argmax(axis=1)

# 결과 시각화 (UMAP + 클러스터)
import umap
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(X)

#%%
plot_df = pd.DataFrame({
    'UMAP1': embedding[:,0],
    'UMAP2': embedding[:,1],
    'Cluster': cluster_labels,
})
plot_df.index = clin_df.index
plot_df = plot_df.merge(clin_df, left_index=True, right_index=True, how='left')
plot_df['Group_OM_BRCA'] = plot_df['OM/OS'].astype(str) + '_' + plot_df['BRCAmut'].astype(str)

plt.figure(figsize=(6,5))
ax = sns.scatterplot(data=plot_df, x='UMAP1', y='UMAP2', hue='Cluster', style='Group_OM_BRCA', palette='Set2') #YlOrRd
plt.title('NMF-based Clustering of TU Space')
plt.tight_layout()
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

plt.show()

# %%
