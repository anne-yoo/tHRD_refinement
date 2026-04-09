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
###^^ cohort scatterplot ####################
plt.figure(figsize=(8, 6))
clin_df = sal_clin.copy()
clin_df['status'] = clin_df['ongoing'].replace({1: 'Ongoing', 0: 'PD', 2: 'NED', 4: 'Others'})
ax = sns.scatterplot(data=clin_df, x='interval', y='gHRDscore', hue='response', style='status', s=100, palette='Set2')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.xlabel('Treatment Duration (days)')
plt.ylabel('gHRD Score')
plt.title('salvage cohort')
plt.show()

plt.figure(figsize=(5, 6))
clin_df = val_clin.copy()
count_df = clin_df.groupby(['line', 'OM/OS']).size().unstack(fill_value=0)

# Stacked barplot
count_df.plot(kind='bar', stacked=True, figsize=(5, 5), color={'maintenance':'#7B9E34','salvage':'#B1693D'},width=0.85)
plt.xlabel("")
plt.ylabel("count")
plt.legend(title="OM/OS")
plt.tight_layout()
sns.despine()
plt.xticks(rotation=0)
plt.show()

tu_df = val_df.loc[val_df.index.isin(majorlist),:]
tu_df = val_gene.copy()
clin = val_clin.copy()
group_info = []
for sample in tu_df.columns:
    if sample not in clin.index:
        continue
    omos = clin.loc[sample, 'OM/OS']
    brca = clin.loc[sample, 'BRCAmut']
    
    if omos == 'maintenance' and brca == 1:
        group_info.append(('maintenance BRCAmt', sample))
    elif omos == 'maintenance' and brca == 0:
        group_info.append(('maintenance BRCAwt', sample))
    elif omos == 'salvage':
        group_info.append(('salvage', sample))

# Step 2: group 컬럼 만들기
group_df = pd.DataFrame(group_info, columns=['Group', 'sample_id'])

# Step 3: TU 값을 long-form으로 melt
tu_long = tu_df.T.reset_index().melt(id_vars='sample_id', var_name='Transcript', value_name='TU')

# Step 4: merge with group info
merged = tu_long.merge(group_df, on='sample_id')

# Step 5: violinplot 그리기
plt.figure(figsize=(6, 6))
sns.boxplot(data=merged, showfliers=False, x='Group', y='TU', palette={'maintenance BRCAmt':'#B8E165','maintenance BRCAwt':'#839D4F','salvage':'#B1693D'})
plt.title('gene expression')
plt.ylabel("TPM")
plt.xlabel("")
plt.xticks(rotation=20)
plt.tight_layout()
sns.despine()
plt.show()

#%%
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
####^^^^^^^^^^^^ PFS Plot ################################
from scipy.stats import mannwhitneyu
fig, axes = plt.subplots(2, 9, figsize=(36, 8))  # 9개 gene set 기준 (3, 9, figsize=(36, 12)) (2, 5, figsize=(20, 8)
axes = axes.flatten()

for i, (name, genes) in enumerate(ddr_coregenelist.items()):
    
    ##################
    clin_df = main_clin[ (main_clin['BRCAmut'] == 0)] ##(main_clin['line_binary'] == 'N-FL') &
    gene_df = main_gene.loc[:,clin_df.index]
    tu_df = main_tu.loc[:,clin_df.index]
    ##################
    
    ##################
    # clin_df = sal_clin
    # gene_df = sal_gene
    # tu_df = sal_tu
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

    ax2 = axes[i + 9]
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
    ax1 = axes[i + 9]
    if i == 0:
        ax1.set_ylabel("mean TU")
    else:
        ax1.set_ylabel("")
        

    # # --- Subplot 3: Volcano Plot ---
    # deg_data = []
    # for g in valid_genes:
    #     r_vals = gene_df.loc[g, r_samples]
    #     nr_vals = gene_df.loc[g, nr_samples]
    #     stat, pval = mannwhitneyu(r_vals, nr_vals, alternative='two-sided')
    #     fc = r_vals.mean() / (nr_vals.mean() + 1e-6)
    #     deg_data.append({'id': g, 'log2FC': np.log2(fc), 'pval': pval, 'type': 'gene'})

    # dut_data = []
    # for t in valid_trans:
    #     r_vals = tu_df.loc[t, r_samples]
    #     nr_vals = tu_df.loc[t, nr_samples]
    #     stat, pval = mannwhitneyu(r_vals, nr_vals, alternative='two-sided')
    #     fc = r_vals.mean() / (nr_vals.mean() + 1e-6)
    #     dut_data.append({'id': t, 'log2FC': np.log2(fc), 'pval': pval, 'type': 'transcript'})

    # df_plot = pd.DataFrame(deg_data + dut_data)
    # df_plot['-log10(pval)'] = -np.log10(df_plot['pval'])

    # ax3 = axes[i + 10]
    # sns.scatterplot(
    #     data=df_plot, x='log2FC', y='-log10(pval)', hue='type',
    #     palette={'gene': 'orange', 'transcript': 'green'},
    #     ax=ax3, s=30, alpha=0.5
    # )
    # ax3.axvline(0, color='gray', linestyle='--', linewidth=1)
    # ax3.set_title(f"{name}")
    # ax3.set_xlabel("log2 FC (R / NR)")
    # ax3.set_ylabel("-log10(p-value)")
    # ax3.get_legend().remove()
    # ax3.set_xlim(-5.2, 5.2)
    
    # print(f"{name} - {df_plot[(df_plot['pval'] < 0.05) & (df_plot['log2FC'] < -1) & (df_plot['type']=='transcript')].shape[0]} significant genes/transcripts")

plt.tight_layout()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/PFScorr_volcano_maintenance_N-FL_BRCAwt_DDRcoregenes.pdf', dpi=300, bbox_inches='tight')
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
###^^^^ DUT comparison (sample level)################################
from statannotations.Annotator import Annotator

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


stable_result = pd.read_csv(f'/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/maintenancestable_DUT_FC.txt', sep='\t', index_col=0)
dutlist = stable_result.loc[
    (stable_result['p_value'] < 0.05) & (np.abs(stable_result['log2FC']) > 1.5)
].index.to_list()



# R/NR 정보
######################################################
# clin_df = main_clin[(main_clin['line_binary']!='FL') & (main_clin['BRCAmut'] == 0)]
# tu_df = main_tu.loc[:,clin_df.index]

clin_df = main_clin[(main_clin['BRCAmut'] == 1)]
gene_df = main_gene.loc[:,clin_df.index]
tu_df = main_tu.loc[:,clin_df.index]

# clin_df = sal_clin
# gene_df = sal_gene
# tu_df = sal_tu
######################################################

clin_df['group'] = clin_df['type'].replace({'CR': 'R', 'IR': 'NR', 'AR': 'R'})
group_df = clin_df[['group']]

fig, axes = plt.subplots(2, 9, figsize=(16,8)) #(16,8) (30,10)
axes = axes.flatten()
# gene set 순회
for i, (name, genes) in enumerate(ddr_coregenelist.items()):

    # === (1) term에 속하는 모든 major TU ===
    term_major = [t for t in majorlist if t.split('-')[-1] in genes]
    mean_expr_all = tu_df.loc[tu_df.index.intersection(term_major)].mean(axis=0)
    ct = len(tu_df.index.intersection(term_major))

    df_all = pd.DataFrame({'Sample': mean_expr_all.index,
                           'MeanTU': mean_expr_all.values})
    df_all = df_all.merge(group_df, left_on='Sample', right_index=True)

    ax1 = axes[i]
    sns.boxplot(data=df_all, x='group', y='MeanTU', ax=ax1, palette={'R': '#F0989B', 'NR': '#94D0F2'}, showfliers=False)
    sns.stripplot(data=df_all, x='group', y='MeanTU', ax=ax1, color='black', size=4, jitter=0.1, alpha=0.7)

    pairs = [('R', 'NR')]
    annotator = Annotator(ax1, pairs, data=df_all, x='group', y='MeanTU')
    annotator.configure(test='Mann-Whitney', text_format='simple', loc='inside')
    annotator.apply_and_annotate()

    ax1.set_title(f"{name}")  #(n={ct})
    ax1.set_ylabel("mean TU")
    ax1.set_xlabel("")

    # === (2) term-major ∩ DUT ===
    term_major_dut = list(set(term_major) & set(dutlist))
    ax2 = axes[i + 9]
    ct2 = len(term_major_dut)

    if ct2 == 0:
        ax2.axis('off')
        ax2.set_title(f"{name}")
        continue

    mean_expr_dut = tu_df.loc[tu_df.index.intersection(term_major_dut)].mean(axis=0)

    df_dut = pd.DataFrame({'Sample': mean_expr_dut.index,
                           'MeanTU': mean_expr_dut.values})
    df_dut = df_dut.merge(group_df, left_on='Sample', right_index=True)

    sns.boxplot(data=df_dut, x='group', y='MeanTU', ax=ax2, palette={'R': '#F0989B', 'NR': '#94D0F2'}, showfliers=False)
    sns.stripplot(data=df_dut, x='group', y='MeanTU', ax=ax2, color='black', size=4, jitter=0.1, alpha=0.7)

    annotator2 = Annotator(ax2, pairs, data=df_dut, x='group', y='MeanTU')
    annotator2.configure(test='Mann-Whitney', text_format='simple', loc='inside')
    annotator2.apply_and_annotate()

    ax2.set_title(f"{name}")  #(n={ct2})
    ax2.set_ylabel("mean TU")
    ax2.set_xlabel("")

# layout 정리
plt.tight_layout()
fig.subplots_adjust(hspace=0.2, wspace=0.4)
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/boxplot_maintenance_BRCAwt_DDRcoregenes.pdf', dpi=300, bbox_inches='tight')
plt.show()



#%%=
####^^ DUT boxplot with gene expreession ############

#######################################################
stable_result = pd.read_csv(f'/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/maintenance_BRCAwt_stable_DUT_FC.txt', sep='\t', index_col=0)
dutlist = stable_result.loc[
    (stable_result['p_value'] < 0.05) & (np.abs(stable_result['log2FC']) > 1.5)
].index.to_list()


clin_df = main_clin[(main_clin['BRCAmut'] == 0)] #&(main_clin['line_binary']!='FL') 
gene_df = main_gene.loc[:,clin_df.index]
tu_df = main_tu.loc[:,clin_df.index]
######################################################

clin_df['group'] = clin_df['type'].replace({'CR': 'R', 'IR': 'NR', 'AR': 'R'})
group_df = clin_df[['group']]

fig, axes = plt.subplots(3, len(ddr_coregenelist), figsize=(40,15)) #(40,15) #(25, 15)
axes = axes.flatten()  # for 1D indexing
for i, (name, genes) in enumerate(ddr_coregenelist.items()):

    valid_genes = list(set(genes) & set(gene_df.index))
    ct0 = len(valid_genes)
    if ct0 == 0:
        axes[i].axis('off')
        axes[i].set_title(f"{name}")
    else:
        mean_expr_gene = gene_df.loc[valid_genes].mean(axis=0)
        df_gene = pd.DataFrame({'Sample': mean_expr_gene.index,
                                'MeanExpr': mean_expr_gene.values})
        df_gene = df_gene.merge(group_df, left_on='Sample', right_index=True)

        ax0 = axes[i]  # ← 1st row: gene expression
        sns.boxplot(data=df_gene, x='group', y='MeanExpr', ax=ax0, palette={'R': '#F0989B', 'NR': '#94D0F2'}, showfliers=False)
        sns.stripplot(data=df_gene, x='group', y='MeanExpr', ax=ax0, color='black', size=4, jitter=0.1, alpha=0.7)

        annotator0 = Annotator(ax0, [('R', 'NR')], data=df_gene, x='group', y='MeanExpr')
        annotator0.configure(test='Mann-Whitney', text_format='simple', loc='inside')
        annotator0.apply_and_annotate()

        ax0.set_title(f"{name}")
        ax0.set_ylabel("mean gene exp")
        ax0.set_xlabel("")

    # === (2nd row) all major TU ===
    term_major = [t for t in majorlist if t.split('-')[-1] in genes]
    mean_expr_all = tu_df.loc[tu_df.index.intersection(term_major)].mean(axis=0)
    ct = len(term_major)

    df_all = pd.DataFrame({'Sample': mean_expr_all.index,
                           'MeanTU': mean_expr_all.values})
    df_all = df_all.merge(group_df, left_on='Sample', right_index=True)

    ax1 = axes[i + 9]
    sns.boxplot(data=df_all, x='group', y='MeanTU', ax=ax1, palette={'R': '#F0989B', 'NR': '#94D0F2'}, showfliers=False)
    sns.stripplot(data=df_all, x='group', y='MeanTU', ax=ax1, color='black', size=4, jitter=0.1, alpha=0.7)

    annotator1 = Annotator(ax1, [('R', 'NR')], data=df_all, x='group', y='MeanTU')
    annotator1.configure(test='Mann-Whitney', text_format='simple', loc='inside')
    annotator1.apply_and_annotate()

    ax1.set_title(f"{name}")
    ax1.set_ylabel("mean TU")
    ax1.set_xlabel("")

    # === (3rd row) major ∩ DUT ===
    term_major_dut = list(set(term_major) & set(dutlist))
    ct2 = len(term_major_dut)
    ax2 = axes[i + 18]

    if ct2 == 0:
        ax2.axis('off')
        ax2.set_title(f"{name}")
    else:
        mean_expr_dut = tu_df.loc[tu_df.index.intersection(term_major_dut)].mean(axis=0)
        df_dut = pd.DataFrame({'Sample': mean_expr_dut.index,
                               'MeanTU': mean_expr_dut.values})
        df_dut = df_dut.merge(group_df, left_on='Sample', right_index=True)

        sns.boxplot(data=df_dut, x='group', y='MeanTU', ax=ax2, palette={'R': '#F0989B', 'NR': '#94D0F2'}, showfliers=False)
        sns.stripplot(data=df_dut, x='group', y='MeanTU', ax=ax2, color='black', size=4, jitter=0.1, alpha=0.7)

        annotator2 = Annotator(ax2, [('R', 'NR')], data=df_dut, x='group', y='MeanTU')
        annotator2.configure(test='Mann-Whitney', text_format='simple', loc='inside')
        annotator2.apply_and_annotate()

        ax2.set_title(f"{name}") # ({ct2})
        ax2.set_ylabel("mean TU")
        ax2.set_xlabel("")
    
    ax0 = axes[i]
    if i == 0:
        ax0.set_ylabel("mean gene exp")
    else:
        ax0.set_ylabel("")

    # 예: all major TU (2행)
    ax1 = axes[i + 9]
    if i == 0:
        ax1.set_ylabel("mean TU")
    else:
        ax1.set_ylabel("")

    # 예: major ∩ DUT TU (3행)
    ax2 = axes[i + 18]
    if i == 0:
        ax2.set_ylabel("mean TU")
    else:
        ax2.set_ylabel("")


#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/boxplot_maintenance_BRCAwt_2L_3row_PARPcoregenes.pdf', dpi=300, bbox_inches='tight')
plt.show()


#%%
###^^ Scatterplot for each sample  * 5 pathways ##########################################
clin_df = main_clin[(main_clin['line_binary']!='FL') ]
tu_df = main_tu.loc[:,clin_df.index]

fig, axes = plt.subplots(1, 5, figsize=(20, 4))  # 5x1 subplot
gene_sets = list(parp_genelist.items())

for i, (name, genes) in enumerate(gene_sets):
    ax = axes[i]

    # (1) 해당 gene set에 속하는 major transcript 추출
    term_major = [t for t in majorlist if t.split('-')[-1] in genes]
    valid_trans = tu_df.index.intersection(term_major)

    # (2) 해당 transcript들의 mean TU 계산 (sample별)
    mean_tu = tu_df.loc[valid_trans].mean(axis=0)  # index: sample

    # (3) scatter plot용 데이터프레임 생성
    df_plot = pd.DataFrame({
        'Sample': mean_tu.index,
        'MeanTU': mean_tu.values,
        'interval': clin_df.loc[mean_tu.index, 'interval'],
        'BRCAmut': clin_df.loc[mean_tu.index, 'BRCAmut'].astype(str)  # hue 색상 처리를 위해 str
    }) 

    # (4) plot
    sns.scatterplot(
        data=df_plot, x='interval', y='MeanTU',
        hue='BRCAmut', palette={'0': '#94D0F2', '1': '#F0989B'},
        ax=ax
    )

    ax.set_title(f"{name}")
    ax.set_xlabel("Treatment Duration")
    ax.set_ylabel("Mean major TU")
    ax.legend(title='BRCAmut')

plt.tight_layout()
plt.show()

# %%
###^^ Time-dependent feature extraction ################################

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

print(f"BRCAwt: {len(sig_wt)} significant major transcripts before correction")
print(f"BRCAmt: {len(sig_mt)} significant major transcripts before correction")
print(f"All: {len(sig_all)} significant major transcripts before correction")

#%%
from statsmodels.stats.multitest import multipletests

# df_wt['FDR'] = multipletests(df_wt['pval'], method='fdr_bh')[1]
# df_mt['FDR'] = multipletests(df_mt['pval'], method='fdr_bh')[1]
# df_all['FDR'] = multipletests(df_all['pval'], method='fdr_bh')[1]

# sig_wt = df_wt[df_wt['FDR'] < 0.1]
# sig_mt = df_mt[df_mt['FDR'] < 0.1]
# sig_all = df_all[df_all['FDR'] < 0.1]

# print(f"BRCAwt: {len(sig_wt)} significant major transcripts after correction")
# print(f"BRCAmt: {len(sig_mt)} significant major transcripts after correction")
# print(f"All: {len(sig_all)} significant major transcripts after correction")

# %%
# %%
###^^^^ maintenance 1L BRCAmt vs. BRCAwt ################################

clin_df = main_clin.loc[(main_clin['line_binary'] == 'N-FL')& (main_clin['BRCAmut'] == 0),: ] #& (main_clin['BRCAmut'] == 0)
# clin_df = main_clin.copy()
# clin_df = clin_df[clin_df['type']!='IR']
# clin_df = clin_df[((clin_df['interval'] > 540) & (clin_df['BRCAmut']==0)) | ((clin_df['interval'] < 360)  )]
#clin_df = clin_df[((clin_df['interval'] > 540) ) | ((clin_df['interval'] < 360) & (clin_df['ongoing'] == 0) )]
gene_df = main_gene.loc[:,clin_df.index]
tu_df = main_tu.loc[:,clin_df.index]

genes = ddr_coregenelist['Homologous Recombination (HR)'] 
# filteredgenelist = sig_all['Transcript'].str.split('-',n=1).str[-1].to_list()

# finalgenes = set(genes).intersection(set(filteredgenelist))
# genes = ['GEN1', 'BARD1', 'RAD50', 'SHFM1', 'XRCC2', 'NBN', 'MUS81', 'MRE11A', 'RAD52', 'BRCA2', 'XRCC3', 'RAD51C', 'RAD51D', 'TP53BP1', 'BLM', 'SLX1A', 'PALB2', 'TOP3A', 'BRCA1', 'EME1', 'BRIP1', 'RBBP8']
term_major = [t for t in majorlist if t.split('-')[-1] in genes]
#tlist = term_major
tlist = list(set(term_major) & set(sig_all['Transcript']))

clin_df['status'] = clin_df['ongoing'].replace({1: 'ongoing', 0: 'PD', 2: 'NED'})

mean_expr_all = tu_df.loc[tu_df.index.intersection(tlist)].mean(axis=0)
clin_df['HR major TU'] = mean_expr_all.values
ax = sns.scatterplot(clin_df, x='interval', y='HR major TU', hue='BRCAmut', style='status', s=100, palette='Set1')
sns.regplot(clin_df, x='interval', y='HR major TU', ax=ax, scatter_kws={'s': 20}, line_kws={"color": "black"}, color='green')
r2, p2 = spearmanr(clin_df['interval'], clin_df['HR major TU'])
ax.set_title(f"Major TU: r={r2:.2f}, p={p2:.3f}")
plt.show()

###
clin_df = sal_clin.copy()
tu_df = sal_tu.copy()
clin_df['status'] = clin_df['ongoing'].replace({1: 'ongoing', 0: 'PD', 2: 'NED'})

mean_expr_all = tu_df.loc[tu_df.index.intersection(tlist)].mean(axis=0)
clin_df['HR major TU'] = mean_expr_all.values
ax = sns.scatterplot(clin_df, x='interval', y='HR major TU', hue='BRCAmut', style='line', s=100, palette='Set1')
sns.regplot(clin_df, x='interval', y='HR major TU', ax=ax, scatter_kws={'s': 20}, line_kws={"color": "black"}, color='green')
r2, p2 = spearmanr(clin_df['interval'], clin_df['HR major TU'])
ax.set_title(f"Major TU: r={r2:.2f}, p={p2:.3f}")
plt.show()


# for i in range(len(tlist)):
#     if tlist[i] not in tu_df.index:
#         print(f"{tlist[i]} is not in the TU dataframe.")
#         continue
#     mean_expr_all = tu_df.loc[tlist[i],:]
#     clin_df['HR major TU'] = mean_expr_all.values
    
#     sns.scatterplot(clin_df, x='interval', y='HR major TU', hue='BRCAmut', style='status', s=100, palette='Set1')
#     plt.title(tlist[i])
#     plt.show()

#%%
clin_df = main_clin.loc[(main_clin['line_binary'] == 'N-FL')& (main_clin['BRCAmut'] == 0),: ] #& (main_clin['BRCAmut'] == 0)
# clin_df = main_clin.copy()
# clin_df = clin_df[clin_df['type']!='IR']
# clin_df = clin_df[((clin_df['interval'] > 500) & (clin_df['ongoing']!=0)) | ((clin_df['interval'] < 200) & (clin_df['ongoing'] != 2)  )]
#clin_df = clin_df[(clin_df['interval'] > 540)  | ((clin_df['interval'] < 360) & (clin_df['ongoing'] != 2) )]
gene_df = main_gene.loc[:,clin_df.index]
tu_df = main_tu.loc[:,clin_df.index]
tlist = term_major
tlist = list(set(term_major) & set(sig_all['Transcript']))

clin_df = sal_clin.copy()
tu_df = sal_tu.copy()

for i in range(len(tlist)):
    if tlist[i] not in tu_df.index:
        print(f"{tlist[i]} is not in the TU dataframe.")
        continue
    mean_expr_all = tu_df.loc[tlist[i],:]
    clin_df['HR major TU'] = mean_expr_all.values
    #ax = sns.regplot(clin_df, x='interval', y='HR major TU',  scatter_kws={'s': 20}, line_kws={"color": "black"}, color='green')
    r2, p2 = spearmanr(clin_df['interval'], clin_df['HR major TU'])
    # ax.set_title(tlist[i] + f"Major TU: r={r2:.2f}, p={p2:.3f}")
    # plt.show()
    print(f"{tlist[i]}: r={r2:.2f}, p={p2:.3f}")

# %%

#%%
#mean_expr_all = tu_df.loc[tu_df.index.intersection(tlist)].mean(axis=0)
mean_expr_all = tu_df.loc[tlist[3],:]

clin_df['HR major TU'] = mean_expr_all.values
clin_df['group'] = clin_df['type'].replace({'CR': 'R', 'IR': 'NR', 'AR': 'R'})
clin_df['status'] = clin_df['ongoing'].replace({1: 'ongoing', 0: 'PD', 2: 'NED'})

# import statsmodels.api as sm

# # X: 독립변수들 (HR major TU + BRCA mutation status)
# X = clin_df['HR major TU']
# X = sm.add_constant(X)

# # y: 종속변수 (투약 기간)
# y = clin_df['interval']

# # OLS 회귀 모델 피팅
# model = sm.OLS(y, X).fit()

# # 결과 출력
# print(model.summary())

# ax = sns.scatterplot(clin_df, x='interval', y='gHRDscore', hue='BRCAmut', style='status', s=100, palette='Set1')
# sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
# plt.show()
# ax2 = sns.scatterplot(clin_df, x='interval', y='HR major TU', hue='BRCAmut', style='status', s=100, palette='Set1')
# sns.move_legend(ax2, "upper left", bbox_to_anchor=(1, 1))


# %%
fig, axes = plt.subplots(3, 9, figsize=(36, 12))  # 10개 gene set 기준
axes = axes.flatten()
from scipy.stats import mannwhitneyu
for i, (name, genes) in enumerate(ddr_coregenelist.items()):
    
    ##################
    clin_df = main_clin[(main_clin['line_binary'] == 'N-FL') & (main_clin['BRCAmut'] == 0)]
    gene_df = main_gene.loc[:,clin_df.index]
    tu_df = main_tu.loc[:,clin_df.index]
    ##################
    
    ##################
    # clin_df = sal_clin
    # gene_df = sal_gene
    # tu_df = sal_tu
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
    time_trans = sig_wt['Transcript']
    
    valid_trans = tu_df.index.intersection(list(set(major_transcripts).intersection(set(time_trans))))
    tu_values = tu_df.loc[valid_trans].mean(axis=0)

    ax2 = axes[i + 9]
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

    ax3 = axes[i + 18]
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
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/PFScorr_volcano_maintenance_N-FL_BRCAwt_DDRcoregenes.pdf', dpi=300, bbox_inches='tight')
plt.show()


#%%
##########^^ Salvage clustering ##################
stable_result = pd.read_csv(f'/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/salvagestable_DUT_FC.txt', sep='\t', index_col=0)
dutlist = stable_result.loc[
    (stable_result['p_value'] < 0.05) & (np.abs(stable_result['log2FC']) > 1.5)
].index.to_list()

clin_df = main_clin[(main_clin['line_binary'] == 'N-FL') & (main_clin['BRCAmut'] == 0)]
tu_df = main_tu.loc[:,clin_df.index]

# clin_df = sal_clin
# tu_df = sal_tu
clin_df['group'] = clin_df['type'].replace({'CR': 'R', 'IR': 'NR', 'AR': 'R'})
group_df = clin_df[['group']]

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
from sklearn.preprocessing import StandardScaler


genelist = {
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

# 1. 모든 major transcript 수집
all_genes = sum(genelist.values(), [])
major_transcripts = [t for t in majorlist if t.split('-')[-1] in all_genes]
#valid_transcripts = tu_df.index.intersection(list(set(major_transcripts).intersection(set(dutlist))))
valid_transcripts = tu_df.index.intersection(major_transcripts)

# 2. TU data subset
tu_sub = tu_df.loc[valid_transcripts]

# 3. sample 정렬 및 group color mapping
sample_order = tu_sub.columns
group_series = clin_df.loc[sample_order, 'group']
group_colors = group_series.map({'R': '#F0989B', 'NR': '#94D0F2'})

# 4. 정규화 (optional but helpful for clustering)
tu_scaled = pd.DataFrame(
    StandardScaler().fit_transform(tu_sub.T),  # clustering은 보통 sample x feature 기준
    index=tu_sub.columns,
    columns=tu_sub.index
).T
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
# 5. clustermap
# 2. linkage 계산 (same as clustermap)
row_linkage = linkage(pdist(tu_scaled, metric='euclidean'), method='ward')

# 3. 원하는 클러스터 개수 또는 거리 기준으로 잘라내기
# 예: 상위 덩어리 하나만 뽑고 싶을 때 클러스터 2개로 자르기
clusters = fcluster(row_linkage, t=4, criterion='maxclust')

# 4. 클러스터 번호가 1인 transcript만 선택 (보통 dendrogram 기준 위쪽 덩어리)
selected_cluster = 1
selected_transcripts = tu_scaled.index[clusters == selected_cluster]

# 5. 필터링된 transcript로 다시 heatmap
tu_selected = tu_scaled.loc[selected_transcripts]

sns.clustermap(
    tu_selected,
    col_colors=group_colors,
    method='ward',
    #metric='correlation',
    cmap='vlag',
    figsize=(6, 6),
    xticklabels=False,
    yticklabels=True,
)
plt.show()


# %%
###^^ Salvage vs. Maintenance 양쪽에서 다 유의한 애들 예시 보여주기##########################

from scipy.stats import mannwhitneyu, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator


sal_clin['group'] = sal_clin['type'].replace({'CR': 'R', 'IR': 'NR', 'AR': 'R'})

main_clin_fil = main_clin[(main_clin['line_binary'] != 'FL') & (main_clin['BRCAmut'] == 0)]
main_tu_fil = main_tu.loc[:, main_clin_fil.index]

from scipy.stats import mannwhitneyu, spearmanr
from statannotations.Annotator import Annotator
import seaborn as sns
import matplotlib.pyplot as plt

for pathway, gene_list in parp_genelist.items():
    term_majors = [t for t in majorlist if t.split('-')[-1] in gene_list]

    for t_id in term_majors:
        gene = t_id.split('-')[-1]
        plotted = False

        # === (1) salvage: R vs. NR boxplot ===
        if t_id in sal_tu.index:
            r_vals = sal_tu.loc[t_id, sal_clin[sal_clin['group'] == 'R'].index]
            nr_vals = sal_tu.loc[t_id, sal_clin[sal_clin['group'] == 'NR'].index]
            stat, pval_sal = mannwhitneyu(r_vals, nr_vals, alternative='two-sided')

            if pval_sal < 0.05:
                df_sal = pd.DataFrame({
                    'TU': sal_tu.loc[t_id],
                    'group': sal_clin['group']
                })

                plt.figure(figsize=(5, 6))
                ax = sns.boxplot(data=df_sal, x='group', y='TU',
                                 palette={'R': '#F0989B', 'NR': '#94D0F2'}, showfliers=False)
                sns.stripplot(data=df_sal, x='group', y='TU', color='black', size=4, jitter=0.1, alpha=0.6)

                annot = Annotator(ax, [('R', 'NR')], data=df_sal, x='group', y='TU')
                annot.configure(test='Mann-Whitney', text_format='star', loc='inside')
                annot.apply_and_annotate()

                plt.title(f"{pathway} - {gene} (p={pval_sal:.3g})")
                plt.ylabel("Transcript Usage")
                plt.xlabel("")
                plt.tight_layout()
                plt.show()
                plotted = True

        # === (2) main: interval vs TU regplot ===
        if t_id in main_tu_fil.index:
            corr, pval_main = spearmanr(main_tu_fil.loc[t_id], main_clin_fil['interval'])
            if pval_main < 0.05:
                df_main = pd.DataFrame({
                    'TU': main_tu_fil.loc[t_id],
                    'interval': main_clin_fil['interval']
                })

                plt.figure(figsize=(5, 5))
                sns.regplot(data=df_main, x='interval', y='TU',
                            scatter_kws={'s': 30}, line_kws={"color": "black"}, color='#6EA8FE')
                plt.title(f"{pathway} - {gene} (r={corr:.2f}, p={pval_main:.3g})")
                plt.ylabel("Transcript Usage")
                plt.xlabel("Treatment Duration")
                plt.tight_layout()
                plt.show()
                plotted = True




# %%
tmp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2025clinical/250801_116_clinical_tmp_UKto01.txt', sep='\t', index_col=0)
tmp = tmp[tmp['ongoing']!=3]
# %%
import numpy as np
def assign_finalresponse(row):
    if row['OM/OS'] == 'maintenance':
        try:
            interval = float(row['interval'])
        except:
            return 'no data'

        if row['BRCAmut'] == 1:
            if row['line'] == '1L':
                return int(interval >= 540)
            else:
                return int(interval >= 360)
        else:  # BRCA wt
            if row['line'] == '1L':
                return int(interval >= 360)
            else:
                return int(interval >= 180)
    else:
        return row['response']

tmp['finalresponse'] = tmp.apply(assign_finalresponse, axis=1)

# %%
pred_proba = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/HGmodel/116_HGmodel_proba.txt', sep='\t', index_col=1)
tmp = pd.read_csv("/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2025clinical/250801_116_clinical_tmp_UKto01.txt", sep='\t', index_col=0)
tmp['tHRD'] = pred_proba['pred_HRD']
def assign_finalresponse(row):
    if row['OM/OS'] == 'maintenance':
        try:
            interval = float(row['interval'])
        except:
            return 'no data'

        if row['BRCAmut'] == 1:
            if row['line'] == '1L':
                return int(interval >= 540)
            else:
                return int(interval >= 360)
        else:  # BRCA wt
            if row['line'] == '1L':
                return int(interval >= 360)
            else:
                return int(interval >= 180)
    else:
        return row['response']

druginfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/validation_sampleinfo.txt', sep='\t', index_col=0)
druginfo = druginfo['drug']

# 0:PD
# 1:ongoing
# 2:NED
# 4:반응성과상관없이다른항암을위해hold
# 5:부작용

tmp['finalresponse'] = tmp.apply(assign_finalresponse, axis=1)
tmp = tmp[tmp['ongoing']!=3]
finaltmp = tmp[['line','finalresponse','OM/OS','BRCAmut','tHRD','gHRDscore','interval','ongoing_new']]
finaltmp.columns = ['line', 'response', 'OM/OS', 'BRCAmt', 'tHRDscore', 'gHRDscore', 'interval','stopreason']
finaltmp['interval'] = finaltmp['interval'].astype(int)
finaltmp['stopreason'] = finaltmp['stopreason'].astype(int)
finaltmp = finaltmp.merge(druginfo, left_index=True, right_index=True, how='left')
finaltmp['drug'] = finaltmp['drug'].fillna('Olaparib')
finaltmp.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2025clinical/2508_116_clinical_response_tmp.txt', sep='\t', index=True)
# %%
original = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_validation_clinicalinfo_new.txt', sep='\t', index_col=0)
original['finalresponse'] = original.apply(assign_finalresponse, axis=1)


# %%
