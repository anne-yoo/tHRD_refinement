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
val_df = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/116_validation_TU_group.txt', sep='\t', index_col=0)
val_gene = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/116_validation_gene_TPM_symbol_group.txt', sep='\t', index_col=0)
val_clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2025clinical/112_validation_clinicalinfo_new.txt', sep='\t', index_col=0)
val_df = val_df.iloc[:-1,:-1]
val_df = val_df.apply(pd.to_numeric, errors='coerce')

genesymbol =  pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM_symbol.txt', sep='\t', index_col=0)
genesymbol = pd.DataFrame(genesymbol.iloc[:,-1])
genesymbol.columns = ['genesymbol']
vallist = list(val_clin.index)
val_df = val_df.loc[:,vallist]
val_gene = val_gene.apply(pd.to_numeric, errors='coerce')
val_gene = val_gene.iloc[:-1,:]
val_gene = val_gene.loc[:,vallist]
val_gene = pd.merge(val_gene,genesymbol,left_index=True,right_index=True, how='left')
val_gene.set_index('genesymbol', inplace=True)

majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = majorminor[majorminor['type']=='major']['gene_ENST'].to_list()

from scipy.stats import spearmanr

olaparib_list = val_clin[val_clin['drug']=='Olaparib'].index.to_list()
niraparib_list = val_clin[val_clin['drug']=='Niraparib'].index.to_list()

main_list = val_clin[val_clin['setting']=='maintenance'].index.to_list()
sal_list = val_clin[val_clin['setting']=='salvage'].index.to_list()

main_gene = val_gene.loc[:,main_list]
sal_gene = val_gene.loc[:,sal_list]

main_tu = val_df.loc[:,main_list]
sal_tu = val_df.loc[:,sal_list]

main_clin = val_clin.loc[main_list,:]
sal_clin = val_clin.loc[sal_list,:]

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


# import gseapy as gp
# go_results = gp.get_library("GO_Biological_Process_2021", organism="Human")
# GO_terms = list(go_results.keys())

# transcript_genes = val_df.index.to_series().str.split("-", n=1).str[-1]
# transcript_to_gene = pd.Series(transcript_genes.values, index=val_df.index)

# def get_transcripts_for_group(go_terms, go_results, transcript_to_gene):
#     selected_transcripts = set()
#     for go_term in go_terms:
#         genes = go_results.get(go_term, [])
#         transcripts = transcript_to_gene[transcript_to_gene.isin(genes)].index
#         selected_transcripts.update(transcripts)
#     return list(selected_transcripts)

# wnt = ['positive regulation of Wnt signaling pathway (GO:0030177)', ]
# nfkb = ['positive regulation of phosphatidylinositol 3-kinase signaling (GO:0014068)']
# pik = ['positive regulation of phosphatidylinositol 3-kinase signaling (GO:0014068)']

# parp_genelist = {
#     "Wnt Signaling": go_results['positive regulation of Wnt signaling pathway (GO:0030177)'],
#     "NF-kB Signaling": go_results['positive regulation of phosphatidylinositol 3-kinase signaling (GO:0014068)'],
#     "PI3K-AKT-mTOR Pathway": go_results['positive regulation of phosphatidylinositol 3-kinase signaling (GO:0014068)'],
# }

#%%


####^^^^^^^^^^^^ PFS Plot ################################
from scipy.stats import mannwhitneyu
fig, axes = plt.subplots(2, 9, figsize=(36, 8))  # 9개 gene set 기준 (3, 9, figsize=(36, 12)) (2, 5, figsize=(20, 8)
axes = axes.flatten()

for i, (name, genes) in enumerate(ddr_coregenelist.items()):
    
    ##################
    clin_df = main_clin[(main_clin['line'] != '1L') & (main_clin['BRCAmt'] == 0)] ##(main_clin['line'] == '1L') &
    gene_df = main_gene.loc[:,clin_df.index]
    tu_df = main_tu.loc[:,clin_df.index]
    ##################
    
    ##################
    # clin_df = sal_clin
    # gene_df = sal_gene
    # tu_df = sal_tu
    ##################
    
    group_info = clin_df['response']
    r_samples = group_info[group_info == 1].index
    nr_samples = group_info[group_info == 0].index

    # --- Subplot 1: Gene exp vs. PFS ---
    valid_genes = list(set(genes) & set(gene_df.index))
    gene_exp = gene_df.loc[valid_genes].mean(axis=0)
    pfs = clin_df['PFS']

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

    ax2 = axes[i + 9] #5/9
    sns.regplot(x=pfs, y=tu_values, ax=ax2, scatter_kws={'s': 20}, line_kws={"color": "black"}, color='green')
    r2, p2 = spearmanr(pfs, tu_values)
    ax2.set_title(f"{name}\nMajor TU: r={r2:.2f}, p={p2:.3f}")
    ax2.set_xlabel("Treatment Duration")
    ax2.set_ylabel("Major TU")
    
        
    ax0 = axes[i]
    if i == 0:
        ax0.set_ylabel("mean gene exp")
    else:
        ax0.set_ylabel("")

    # 예: all major TU (2행)
    ax1 = axes[i + 9] #5/9
    if i == 0:
        ax1.set_ylabel("mean TU")
    else:
        ax1.set_ylabel("")
        

plt.tight_layout()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/figures/116_val/PFScorr_main_BRCAmt_ddrcoregenes.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%
for name, genes in parp_genelist.items():
    print(f"\n{name}")
    
    # major transcript 리스트 중 이 gene set에 해당하는 것만
    major_transcripts = [t for t in majorlist if t.split('-')[-1] in genes]
    valid_trans = tu_df.index.intersection(major_transcripts)
    
    for t in valid_trans:
        tu_vals = tu_df.loc[t]
        corr, pval = spearmanr(pfs, tu_vals)
        print(f"{t}: r = {corr:.2f}, p = {pval:.3g}")


# %%
###^^ salvage boxplot ######

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from statannotations.Annotator import Annotator

# salvage cohort 기준
clin_df = sal_clin.copy()
gene_df = sal_gene.loc[:, clin_df.index]
tu_df   = sal_tu.loc[:, clin_df.index]

fig, axes = plt.subplots(2, 9, figsize=(36, 8))
axes = axes.flatten()

for i, (name, genes) in enumerate(ddr_coregenelist.items()):
    group_info = clin_df['response']  # 0/1 binary
    r_samples = group_info[group_info == 1].index
    nr_samples = group_info[group_info == 0].index

    # -------------------------------
    # (1) Gene-level expression
    # -------------------------------
    valid_genes = list(set(genes) & set(gene_df.index))
    if len(valid_genes) > 0:
        gene_exp = gene_df.loc[valid_genes].mean(axis=0)
        plot_df = clin_df.copy()
        plot_df["gene_exp"] = gene_exp

        ax1 = axes[i]
        sns.boxplot(x="response", y="gene_exp", data=plot_df, palette="Set2", ax=ax1)
        sns.stripplot(x="response", y="gene_exp", data=plot_df,
                    color="black", size=3, jitter=True, ax=ax1)

        # annotation
        pairs = [(0, 1)]
        annotator = Annotator(ax1, pairs, data=plot_df,
                            x="response", y="gene_exp")
        annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
        annotator.apply_and_annotate()

        ax1.set_title(f"{name} (Gene exp)")
        ax1.set_xlabel("response")
        if i == 0:
            ax1.set_ylabel("Mean gene exp")
        else:
            ax1.set_ylabel("")

    # -------------------------------
    # (2) Transcript-level TU
    # -------------------------------
    major_transcripts = [t for t in majorlist if t.split('-')[-1] in genes]
    valid_trans = tu_df.index.intersection(major_transcripts)
    if len(valid_trans) > 0:
        tu_values = tu_df.loc[valid_trans].mean(axis=0)
        plot_df2 = clin_df.copy()
        plot_df2["tu_values"] = tu_values

        ax2 = axes[i + 9]
        sns.boxplot(x="response", y="tu_values", data=plot_df2, palette="Set3", ax=ax2)
        sns.stripplot(x="response", y="tu_values", data=plot_df2,
                    color="black", size=3, jitter=True, ax=ax2)

        # annotation
        pairs = [(0, 1)]
        annotator = Annotator(ax2, pairs, data=plot_df2,
                            x="response", y="tu_values")
        annotator.configure(test='Mann-Whitney', text_format='simple', loc='inside')
        annotator.apply_and_annotate()

        ax2.set_title(f"{name} (Major TU)")
        ax2.set_xlabel("response")
        if i == 0:
            ax2.set_ylabel("mean TU")
        else:
            ax2.set_ylabel("")

plt.tight_layout()
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/figures/116_val/boxplot_sal_ddrcoregenes.pdf', dpi=300, bbox_inches='tight')
plt.show()


# %%
