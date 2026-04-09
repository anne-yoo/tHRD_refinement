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
AR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t', index_col=0)
IR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUT/nonresponder_stable_DUT_Wilcoxon.txt', sep='\t', index_col=0)

ARdutlist = AR_dut.loc[(AR_dut['p_value']<0.05) & (np.abs(AR_dut['log2FC'])>1.5)].index.to_list()
IRdutlist = IR_dut.loc[(IR_dut['p_value']<0.05) & (np.abs(IR_dut['log2FC'])>1.5)].index.to_list()

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo_new.txt', sep='\t')
ARlist = list(set(sampleinfo.loc[(sampleinfo['response']==1),'sample_id']))
IRlist = list(set(sampleinfo.loc[(sampleinfo['response']==0),'sample_id']))

TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0)
TU.columns = TU.columns.str[:-4]
preTU = TU.iloc[:,1::2]
postTU = TU.iloc[:,0::2]

geneexp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_gene_exp_TPM_symbol.txt', sep='\t', index_col=-1)
geneexp.drop(columns=['Unnamed: 0'], inplace=True)
geneexp.columns = geneexp.columns.str[:-4]
pregene = geneexp.iloc[:,1::2]
postgene = geneexp.iloc[:,0::2]

ARpregene = pregene.loc[:,ARlist]
ARpostgene = postgene.loc[:,ARlist]
IRpregene = pregene.loc[:,IRlist]
IRpostgene = postgene.loc[:,IRlist]

ARpre = preTU.loc[ARdutlist,ARlist]
ARpost = postTU.loc[ARdutlist,ARlist]
IRpre = preTU.loc[IRdutlist,IRlist]
IRpost = postTU.loc[IRdutlist,IRlist]

majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = majorminor[majorminor['type']=='major']['gene_ENST'].to_list()
minorlist = majorminor[majorminor['type']=='minor']['gene_ENST'].to_list()

ARpremajor = ARpre.loc[ARpre.index.isin(majorlist),:]
ARpostmajor = ARpost.loc[ARpost.index.isin(majorlist),:]
IRpremajor = IRpre.loc[IRpre.index.isin(majorlist),:]
IRpostmajor = IRpost.loc[IRpost.index.isin(majorlist),:]

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

parp_genelist = {
    "Homologous Recombination": ddr_coregenelist['Homologous Recombination (HR)'],
    
    "ATR-CHK1-WEE1 Pathway": [
        "ATR", "ATRIP", "RPA1", "RPA2", "RPA3", "CHEK1",
        "CDC25A", "CDC25B", "CDC25C", "CDKN1A", "WEE1",
        "CDK1", "CDK2", "ATM", "CHEK2", "TP53", "TP53BP1",
        "TOPBP1", "ETAA1"
    ],
    "PI3K-AKT-mTOR Pathway": [
        "PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG", "PIK3R5", "PIK3R6",
        "AKT1", "AKT2", "AKT3", "PDK1", "MTOR", "RICTOR", "RPTOR",
        "MLST8", "RPS6KB1", "MAPKAP1", "PRR5", "PRR5L"
    ],
    
    "Drug Efflux Pump": [
        "ABCB1", "ABCG2", "ABCC1", "ABCC2", "ABCC3", "ABCC4", "ABCC5"
    ],

    "Replication Fork Stabilization": [
        "PTIP", "SMARCAL1", "EZH2", "MUS81", "ZRANB3", "HLTF", "WRN",
        "TIM1", "TIPIN", "CLASPIN", "AND1", "MRE11A", "FANCD2",
        "BOD1L", "SETD1A", "WRNIP1"
    ]
}

# %%
##^^ AR pre vs. IR pre #####

baseline = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/baseline/stable_DUT_MW_FC.txt', sep='\t', index_col=0)
baseline = baseline[(baseline['p_value']<0.05) & (np.abs(baseline['log2FC'])>1.5)] 
dutlist = list(set(baseline.index) & set(majorlist))




clin_df = sampleinfo.iloc[1::2,:]
clin_df.set_index('sample_id', inplace=True)
clin_df['group'] = clin_df['response'].map({0: 'IR', 1: 'AR'})


clin_df = clin_df[(clin_df['purpose']=='maintenance')&(clin_df['BRCAmut']==0)] #&(clin_df['line_binary']=='N-FL')
#clin_df = clin_df[(clin_df['purpose']=='salvage')]
tu_df = preTU.loc[preTU.index.isin(majorlist),clin_df.index]

fig, axes = plt.subplots(1, 5, figsize=(20, 4))  # 5x1 subplot
gene_sets = list(parp_genelist.items())

for i, (name, genes) in enumerate(gene_sets):
    ax = axes[i]

    # (1) 해당 gene set에 속하는 major transcript 추출
    term_major = [t for t in majorlist if t.split('-')[-1] in genes]
    valid_trans = tu_df.index.intersection(term_major)

    # (2) 해당 transcript들의 mean TU 계산 (sample별)
    mean_tu = tu_df.loc[valid_trans].mean(axis=0) 
    print(valid_trans, name)

    # (3) scatter plot용 데이터프레임 생성
    df_plot = pd.DataFrame({
        'Sample': mean_tu.index,
        'MeanTU': mean_tu.values,
        'interval': clin_df.loc[mean_tu.index, 'interval'],
        'BRCAmut': clin_df.loc[mean_tu.index, 'BRCAmut'],
        'group': clin_df['group']
    }) 

    # (4) plot
    sns.scatterplot(
        data=df_plot, x='interval', y='MeanTU',
        #hue='Response', palette={0: '#94D0F2', 1: '#F0989B'},
        hue='group', palette={'IR': '#81B214', 'AR': '#FFCC29'},
        ##style='BRCAmut', markers={0: 'o', 1: 'X'},
        ax=ax,
        s=100
    )

    ax.set_title(f"{name}")
    ax.set_xlabel("Treatment Duration")
    ax.set_ylabel("Mean major TU")
    
    ax.get_legend().remove()

plt.tight_layout()
plt.show()

# %%
###^^ boxplot + stripplot: major DUT ###########

clin_df = sampleinfo.iloc[1::2,:]
clin_df.set_index('sample_id', inplace=True)
clin_df = clin_df[clin_df['response']==1]

premajor = ARpremajor
postmajor = ARpostmajor

from statannotations.Annotator import Annotator

fig, axes = plt.subplots(1, 5, figsize=(25, 6)) #plt.subplots(1, 5, figsize=(25, 6)) #plt.subplots(1, 9, figsize=(36, 5))
axes = axes.flatten()

for i, (pathway, genelist) in enumerate(parp_genelist.items()):
    if i >= 5:
        break
    
    # 해당 gene set의 major transcript 추출
    term_major = [t for t in majorlist if t.split('-')[-1] in genelist]
    matching = list(set(term_major) & set(premajor.index) & set(postmajor.index))
    if not matching:
        axes[i].set_title(f"{pathway} (No data)")
        axes[i].axis("off")
        continue
    print(len(matching), pathway)
    if len(matching) < 1:
        axes[i].set_title(f"{pathway} (Not enough data)")
        axes[i].axis("off")
        continue
    # pre/post 평균 계산
    pre_mean = premajor.loc[matching].mean()
    post_mean = postmajor.loc[matching].mean()

    df = pd.DataFrame({'Pre': pre_mean, 'Post': post_mean})
    df['Sample'] = df.index
    df = df.merge(clin_df[['purpose', 'BRCAmut']], left_on='Sample', right_index=True)

    # melt for seaborn
    df_melt = df.melt(id_vars=['Sample', 'purpose', 'BRCAmut'], var_name='Timepoint', value_name='TU')

    ax = axes[i]
    sns.boxplot(data=df_melt, x='Timepoint', y='TU', ax=ax, color='lightgray', showfliers=True, zorder=1, palette={"Pre": "#FFCC29", "Post": "#FFAE4B"}, ) #{"Pre": "#81B214", "Post": "#5F8805"} {"Pre": "#FFCC29", "Post": "#FFAE4B"}

    # annotate
    pairs = [('Pre', 'Post')]
    annot = Annotator(ax, pairs, data=df_melt, x='Timepoint', y='TU')
    annot.configure(test='Wilcoxon', text_format='star', loc='inside')
    annot.apply_and_annotate()

    #scatter + line for matched samples
    for _, row in df.iterrows():
        ax.plot(['Pre', 'Post'], [row['Pre'], row['Post']], color='gray', alpha=0.5)
        ax.scatter(['Pre', 'Post'], [row['Pre'], row['Post']],
                   c='grey',
                   #c='tab:green' if row['purpose'] == 'maintenance' else 'tab:orange',
                   marker='o' if row['BRCAmut'] else 's',
                   edgecolor='black', s=30, linewidth=0.6)

    ax.set_title(pathway)
    ax.set_xlabel("")
    ax.set_ylabel("Mean TU")

plt.tight_layout()
plt.show()

#%%
###^^ boxplot + stripplot: gene exp ###########

clin_df = sampleinfo.iloc[1::2,:]
clin_df.set_index('sample_id', inplace=True)
clin_df = clin_df[clin_df['response']==1]

premajor = ARpregene
postmajor = ARpostgene

from statannotations.Annotator import Annotator

fig, axes = plt.subplots(1, 5, figsize=(25, 6))
axes = axes.flatten()

for i, (pathway, genelist) in enumerate(parp_genelist.items()):
    if i >= 5:
        break

    # 해당 gene set의 major transcript 추출
    term_major = genelist
    matching = list(set(genelist) & set(premajor.index) & set(postmajor.index))
    if not matching:
        axes[i].set_title(f"{pathway} (No data)")
        axes[i].axis("off")
        continue

    # pre/post 평균 계산
    pre_mean = premajor.loc[matching].mean()
    post_mean = postmajor.loc[matching].mean()

    df = pd.DataFrame({'Pre': pre_mean, 'Post': post_mean})
    df['Sample'] = df.index
    df = df.merge(clin_df[['purpose', 'BRCAmut']], left_on='Sample', right_index=True)

    # melt for seaborn
    df_melt = df.melt(id_vars=['Sample', 'purpose', 'BRCAmut'], var_name='Timepoint', value_name='TU')

    ax = axes[i]
    sns.boxplot(data=df_melt, x='Timepoint', y='TU', ax=ax, color='lightgray', showfliers=False, zorder=1, palette={"Pre": "#FFCC29", "Post": "#FFAE4B"}, ) #{"Pre": "#81B214", "Post": "#5F8805"}

    # annotate
    pairs = [('Pre', 'Post')]
    annot = Annotator(ax, pairs, data=df_melt, x='Timepoint', y='TU')
    annot.configure(test='Wilcoxon', text_format='star', loc='inside')
    annot.apply_and_annotate()

    #scatter + line for matched samples
    for _, row in df.iterrows():
        ax.plot(['Pre', 'Post'], [row['Pre'], row['Post']], color='gray', alpha=0.5)
        ax.scatter(['Pre', 'Post'], [row['Pre'], row['Post']],
                   c='grey',
                   #c='tab:green' if row['purpose'] == 'maintenance' else 'tab:orange',
                   marker='o' if row['BRCAmut'] else 's',
                   edgecolor='black', s=30, linewidth=0.6)

    ax.set_title(pathway)
    ax.set_xlabel("")
    ax.set_ylabel("gene exp")

plt.tight_layout()
plt.show()

#%%
###^^ AR post vs. IR pre #######################
ARpre = preTU.loc[:,ARlist]
ARpost = postTU.loc[:,ARlist]
ARpremajor = ARpre.loc[ARpre.index.isin(majorlist),:]
ARpostmajor = ARpost.loc[ARpost.index.isin(majorlist),:]

IRpre = preTU.loc[:,IRlist]
IRpost = postTU.loc[:,IRlist]
IRpremajor = IRpre.loc[IRpre.index.isin(majorlist),:]
IRpostmajor = IRpost.loc[IRpost.index.isin(majorlist),:]    

shared = list(set(ARdutlist) & set(ARpre.index) & set(ARpost.index) & set(IRpre.index) & set(IRpost.index))

# 데이터프레임 구성
df_all = pd.DataFrame({
    'Transcript': shared * 4,
    'TU': pd.concat([
        ARpre.loc[shared].mean(axis=1),
        ARpost.loc[shared].mean(axis=1),
        IRpre.loc[shared].mean(axis=1),
        IRpost.loc[shared].mean(axis=1)
    ]).values,
    'Group': ['AR Pre'] * len(shared) + ['AR Post'] * len(shared) +
             ['IR Pre'] * len(shared) + ['IR Post'] * len(shared)
})

plt.figure(figsize=(8, 5))
sns.boxplot(data=df_all, x='Group', y='TU', showfliers=False, palette={"AR Pre": "#FFCC29", "AR Post": "#FFAE4B", "IR Pre": "#81B214", "IR Post": "#5F8805"})

from statannotations.Annotator import Annotator
pairs = [('AR Pre', 'AR Post'), ('IR Pre', 'IR Post'),('AR Pre', 'IR Pre')]
annotator = Annotator(plt.gca(), pairs, data=df_all, x='Group', y='TU')
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
annotator.apply_and_annotate()

plt.ylabel("TU")
plt.xlabel("")
plt.tight_layout()
plt.title("AR major DUTs: TU distribution")
sns.despine()
plt.show()

#%%
fig, axes = plt.subplots(1, len(parp_genelist), figsize=(5 * len(parp_genelist), 5))

for i, (term, gene_set) in enumerate(parp_genelist.items()):
    term_trans = [t for t in ARdutlist if t.split('-')[-1] in gene_set]
    shared_trans = list(set(term_trans) & set(ARpre.index) & set(ARpost.index) & set(IRpre.index) & set(IRpost.index))

    if len(shared_trans) == 0:
        axes[i].axis('off')
        axes[i].set_title(f"{term} (no data)")
        continue

    df = pd.DataFrame({
        'Transcript': shared_trans * 4,
        'TU': pd.concat([
            ARpre.loc[shared_trans].mean(axis=1),
            ARpost.loc[shared_trans].mean(axis=1),
            IRpre.loc[shared_trans].mean(axis=1),
            IRpost.loc[shared_trans].mean(axis=1)
        ]).values,
        'Group': ['AR Pre'] * len(shared_trans) + ['AR Post'] * len(shared_trans) +
                 ['IR Pre'] * len(shared_trans) + ['IR Post'] * len(shared_trans)
    })

    ax = axes[i]
    sns.boxplot(data=df, x='Group', y='TU', ax=ax, palette={"AR Pre": "#FFCC29", "AR Post": "#FFAE4B", "IR Pre": "#81B214", "IR Post": "#5F8805"}, showfliers=False)
    #sns.stripplot(data=df, x='Group', y='TU', ax=ax, color='black', size=3, jitter=True, alpha=0.6)

    pairs = [('AR Pre', 'AR Post'), ('IR Pre', 'IR Post'),('AR Pre', 'IR Pre')]
    annot = Annotator(ax, pairs, data=df, x='Group', y='TU')
    annot.configure(test='Mann-Whitney', text_format='star', loc='inside')
    annot.apply_and_annotate()

    ax.set_title(term)
    ax.set_ylabel("Mean TU")
    ax.set_xlabel("")

plt.tight_layout()
plt.show()

#%%
###^^ treatment interval ##########
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# 결과 시각화를 위한 설정
sns.set(style="whitegrid")
plt.figure(figsize=(5 * len(parp_genelist), 4))

# AR 그룹만 추출
ar_clin = clin_df[clin_df['response'] == 1]
samples = ar_clin.index
delta_tu = ARpostmajor[samples] - ARpremajor[samples]# 시각화
sns.set(style="whitegrid")
plt.figure(figsize=(5 * len(parp_genelist), 4))

for i, (pathway, gene_list) in enumerate(parp_genelist.items()):
    # 해당 pathway에 해당하는 ARdutlist transcript 추출
    matching_transcripts = [t for t in ARdutlist if t.split('-')[-1] in gene_list]
    matching_transcripts = [t for t in matching_transcripts if t in delta_tu.index]

    if len(matching_transcripts) < 2:
        continue  # 너무 적으면 스킵

    # 샘플별 ΔTU 평균 계산
    tu_mean = delta_tu.loc[matching_transcripts].mean()

    # 상관계수 계산
    x = tu_mean.values
    y = ar_clin.loc[tu_mean.index, 'interval'].values
    r, p = spearmanr(x, y)
    title = f"{pathway}\nR = {r:.2f}, p = {p:.3g}"

    # Plot
    ax = plt.subplot(1, len(parp_genelist), i + 1)
    sns.regplot(x=x, y=y, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("ΔTU (mean)")
    ax.set_ylabel("Interval")

plt.tight_layout()
plt.show()
# %%
