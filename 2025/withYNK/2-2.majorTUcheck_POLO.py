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
# val_df = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/POLO/83_POLO_TU.txt', sep='\t', index_col=0)
val_gene = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/merged_83_gene_TPM.txt', sep='\t', index_col=0)
val_clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2025clinical/83_POLO_clinicalinfo.txt', sep='\t', index_col=0)
tpm_df = pd.read_csv("/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/merged_83_transcript_TPM.txt", sep="\t", index_col=0)
tpm_df = tpm_df.drop(columns=['gene_name'])  # gene_name 컬럼 제거

tpm = tpm_df.copy()

# --- sample sorting (bfD / atD pairs) ---
sorted_cols = sorted(
    tpm.columns,
    key=lambda x: (re.sub(r"-(bfD|atD)$", "", x), "bfD" in x, "atD" in x)
)
tpm = tpm[sorted_cols]

# --- gene name extract ---
gene_names = tpm.index.str.split("-", n=1).str[-1]
tpm["gene_name"] = gene_names

###############filtering###################
n_samples = tpm.shape[1]
frac_nonzero = (tpm != 0).sum(axis=1) / n_samples
tpm = tpm[frac_nonzero >= 0.20]
###############filtering###################

# --- gene-level TPM sum ---
gene_tpm = tpm.groupby("gene_name").sum(numeric_only=True)

# --- align gene sum to each transcript ---
gene_sum_aligned = gene_tpm.loc[tpm["gene_name"]].set_index(tpm.index)
gene_sum_aligned.columns = tpm.columns[:-1]

# --- ratio calculation (no replacement, let 0/0 produce NaN) ---
val_df = tpm.iloc[:, :-1] / gene_sum_aligned


#%%
val_df = val_df.apply(pd.to_numeric, errors='coerce')
# genesymbol =  pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM_symbol.txt', sep='\t', index_col=0)
# genesymbol = pd.DataFrame(genesymbol.iloc[:,-1])
# genesymbol.columns = ['genesymbol']

# vallist = list(val_clin.index)
# val_df = val_df.loc[:,vallist]
# val_gene = val_gene.loc[genesymbol.index,vallist]
# val_gene.index = genesymbol.loc[val_gene.index,'genesymbol']

val_clin['drug'] = 'Niraparib'

majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/merged_83_cov5_majorminorlist.txt', sep='\t')
majorlist = majorminor[majorminor['type']=='major']['Transcript-Gene'].to_list()
minorlist = majorminor[majorminor['type']=='minor']['Transcript-Gene'].to_list()

sqanti = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/sqantioutput/sqanti_classification.txt', sep='\t')
sqanti.dropna(axis=1, how='all', inplace=True)

df_cat = sqanti[['isoform','structural_category','subcategory','within_CAGE_peak','coding']]
df_cat['isoform_clean'] = df_cat['isoform'].astype(str).str.split('-', n=1).str[0]
majorlist_set = set([x.split('-', 1)[0] for x in majorlist])
coding_set = set(sqanti[sqanti['coding']=='coding']['isoform'])
majorlist_set = majorlist_set.intersection(coding_set)
df_cat['major'] = df_cat['isoform'].isin(majorlist_set)
####filter#####
valid_cat = {"full-splice_match", "novel_in_catalog"}
df_cat['major'] = df_cat['major'] & df_cat['structural_category'].isin(valid_cat)
df_cat.set_index('isoform', inplace=True)
###############

group1 = df_cat[df_cat['major']==True]['isoform_clean'].to_list()
group2 = df_cat[(df_cat['major']==False)&(df_cat['coding']=='coding')]['isoform_clean'].to_list()
group3 = df_cat[(df_cat['major']==False)&(df_cat['coding']=='non_coding')]['isoform_clean'].to_list()

group1_list = majorminor[majorminor['transcriptid'].isin(group1)]['Transcript-Gene'].to_list()
group2_list = majorminor[majorminor['transcriptid'].isin(group2)]['Transcript-Gene'].to_list()
group3_list = majorminor[majorminor['transcriptid'].isin(group3)]['Transcript-Gene'].to_list()

from scipy.stats import spearmanr

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

# -----------------------------
# 2. Make gene-level matrices
# -----------------------------

gene_map = pd.Series([x.split("-",1)[-1] for x in tpm_df.index], index=tpm_df.index)

# -----------------------------
# 2. Make gene-level matrices
# -----------------------------
# (i) Total TPM (all transcripts)

expr_total = tpm_df.groupby(gene_map).sum()

# (ii) Major TPM (filter first → then sum)
expr_major = tpm_df.loc[tpm_df.index.isin(group1_list)] #*major/minor
gene_map_major = gene_map.loc[expr_major.index]
major_exp_df = expr_major.groupby(gene_map_major).sum()
tu_major = val_df.loc[val_df.index.isin(group1_list)] #*major/minor

####^^^^^^^^^^^^ PFS Plot ################################
from scipy.stats import mannwhitneyu
fig, axes = plt.subplots(3, 10, figsize=(40, 12))  # 9개 gene set 기준 (3, 9, figsize=(36, 12)) (2, 5, figsize=(20, 8)
axes = axes.flatten()

for i, (name, genes) in enumerate(ddr_genelist.items()):
    
    ##################
    clin_df = main_clin[ (main_clin['BRCAmut'] == 0)] ##(main_clin['line_binary'] == 'N-FL') &
    gene_df = main_gene.loc[:,clin_df.index]
    tu_df = main_tu.loc[:,clin_df.index]
    major_exp_sub = major_exp_df.loc[:, clin_df.index]
    major_tu_sub = tu_major.loc[:, clin_df.index]
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
    major_transcripts = [t for t in group1_list if t.split('-')[-1] in genes] #*major/minor
    valid_trans = tu_df.index.intersection(major_transcripts)
    tu_values = tu_df.loc[valid_trans].mean(axis=0)

    # ax2 = axes[i + 9] #5/9
    # sns.regplot(x=pfs, y=tu_values, ax=ax2, scatter_kws={'s': 20}, line_kws={"color": "black"}, color='green')
    # r2, p2 = spearmanr(pfs, tu_values)
    # ax2.set_title(f"{name}\nMajor TU: r={r2:.2f}, p={p2:.3f}")
    # ax2.set_xlabel("Treatment Duration")
    # ax2.set_ylabel("Major TU")
    
        
    ax0 = axes[i]
    if i == 0:
        ax0.set_ylabel("mean gene exp")
    else:
        ax0.set_ylabel("")

    # 예: all major TU (2행)
    ax1 = axes[i + 10] #5/9
    if i == 0:
        ax1.set_ylabel("mean TU")
    else:
        ax1.set_ylabel("")
        
    # --- (3) Major Exp Sum vs PFS ---
    valid_genes_major = list(set(genes) & set(major_exp_sub.index))
    major_exp = major_exp_sub.loc[valid_genes_major].sum(axis=0)

    ax3 = axes[i + 20]
    sns.regplot(x=pfs, y=major_exp, ax=ax3,
                scatter_kws={'s':20}, line_kws={"color":"black"}, color='steelblue')
    r3, p3 = spearmanr(pfs, major_exp)
    ax3.set_title(f"{name}\nMajor Exp Sum: r={r3:.2f}, p={p3:.3f}")
    ax3.set_xlabel("PFS")
    ax3.set_ylabel("Major Exp Sum" if i==0 else "")


plt.tight_layout()
#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/PFScorrelation_regressionplot.pdf", dpi=300)
plt.show()


# %%
####^^ gene exp / major exp sum : coxPH KM plot ##########################

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test

# --- Import libraries ---
# (Assuming pandas, numpy, plt, sns are already imported)


# --- Survival variables (identified from user's example code) ---
TIME_COL = 'PFS'
EVENT_COL = 'recur' # 1=event, 0=censored


# --- Prepare Figure for KM Plots ---
# Create a 4x5 grid as requested
n_rows = 4
n_cols = 5
fig_km, axes_km = plt.subplots(n_rows, n_cols, 
                               figsize=(6 * n_cols, 5 * n_rows), # (30, 20)
                               squeeze=False, 
                               facecolor='white')

print("========= CoxPH & Kaplan-Meier Analysis Results =========")

n_sets = len(ddr_genelist)
if n_sets > 10:
    print(f"WARNING: ddr_genelist has {n_sets} items, but layout is fixed to 4x5 for 10 items. Only the first 10 will be plotted.")

for i, (name, genes) in enumerate(ddr_genelist.items()):
    
    if i >= 10:
        break
        
    ##################
    # Filter data (same as original code)
    clin_df = main_clin[ (main_clin['BRCAmut'] == 0)] 
    gene_df = main_gene.loc[:,clin_df.index]
    major_exp_sub = major_exp_df.loc[:, clin_df.index]
    major_tu_sub = tu_major.loc[:, clin_df.index]
    ##################

    # --- Create DataFrame for analysis ---
    try:
        survival_df = clin_df[[TIME_COL, EVENT_COL]].copy()
    except KeyError:
        print(f"!!! ERROR: '{TIME_COL}' or '{EVENT_COL}' column not found in main_clin. !!!")
        break

    # 1. Calculate Mean Gene Exp
    valid_genes = list(set(genes) & set(gene_df.index))
    survival_df['mean_gene_exp'] = gene_df.loc[valid_genes].mean(axis=0)

    # 2. Calculate Mean Major Exp
    valid_genes_major = list(set(genes) & set(major_exp_sub.index))
    
    survival_df['mean_major_exp'] = major_exp_sub.loc[valid_genes_major].mean(axis=0) #** major_tu_sub vs. major_expsub
    # major_transcripts = [t for t in majorlist if t.split('-')[-1] in genes] #*major/minor
    # valid_trans = tu_df.index.intersection(major_transcripts)
    # survival_df['mean_major_exp'] = major_tu_sub.loc[valid_trans].mean(axis=0)
    
    # Drop samples with NA in essential columns
    analysis_df = survival_df.dropna(subset=[TIME_COL, EVENT_COL, 'mean_gene_exp', 'mean_major_exp'])

    print(f"\n--- Results for Gene Set: {name} (Plot {i+1}/10) ---")

    # --- Calculate plot coordinates for the 4x5 grid ---
    plot_col = i % n_cols             # Col: 0, 1, 2, 3, 4, 0, 1, 2, 3, 4
    plot_row_offset = (i // n_cols) * 2 # Row offset: 0 (for i=0-4), 2 (for i=5-9)

    # --- 1. Analysis: Mean Gene Exp ---
    
    var_to_analyze_1 = 'mean_gene_exp'
    ax1 = axes_km[plot_row_offset, plot_col] 
    kmf = KaplanMeierFitter()

    # Group data by median for KM plot
    median_val_1 = analysis_df[var_to_analyze_1].median()
    analysis_df['group_1'] = (analysis_df[var_to_analyze_1] > median_val_1).astype(int) 

    high_data_1 = analysis_df[analysis_df['group_1'] == 1]
    low_data_1 = analysis_df[analysis_df['group_1'] == 0]

    # Fit and plot KM curves
    kmf.fit(low_data_1[TIME_COL], event_observed=low_data_1[EVENT_COL], label=f"Low (n={len(low_data_1)})")
    kmf.plot_survival_function(ax=ax1, ci_show=False, color='blue', show_censors=True)
    kmf.fit(high_data_1[TIME_COL], event_observed=high_data_1[EVENT_COL], label=f"High (n={len(high_data_1)})")
    kmf.plot_survival_function(ax=ax1, ci_show=False, color='red', show_censors=True)

    # Log-rank test (High vs Low)
    lr_results_1 = logrank_test(high_data_1[TIME_COL], low_data_1[TIME_COL], 
                                high_data_1[EVENT_COL], low_data_1[EVENT_COL])
    p_val_gene_km = lr_results_1.p_value

    # --- ⬇️ TITLE MODIFIED HERE ⬇️ ---
    ax1.set_title(f"{name} - Mean Gene Exp", fontsize=16)
    # --- ⬆️ TITLE MODIFIED HERE ⬆️ ---
    
    ax1.set_xlabel(TIME_COL)
    if plot_col == 0: 
        ax1.set_ylabel("Survival Probability")

    # CoxPH Model (using continuous variable)
    cph_gene = CoxPHFitter()
    cph_gene_model = cph_gene.fit(analysis_df[[TIME_COL, EVENT_COL, var_to_analyze_1]], 
                                  duration_col=TIME_COL, event_col=EVENT_COL)
    summary_1 = cph_gene_model.summary.loc[var_to_analyze_1]
    hr_1, ci_low_1, ci_high_1, p_val_gene_cox = summary_1[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]
    
    # Add CoxPH results as annotation
    ax1.text(
        0.05, 0.05,
        f"CoxPH (continuous):\n"
        f"HR = {hr_1:.2f} (95% CI {ci_low_1:.2f}–{ci_high_1:.2f})\n"
        f"Log-rank p = {p_val_gene_km:.3f}\n" # Log-rank p-value moved here
        f"Cox p = {p_val_gene_cox:.3e}",
        transform=ax1.transAxes, fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, boxstyle="round")
    )
    
    # Print significant results to console
    print(f"[Mean Gene Exp] KM Log-rank p = {p_val_gene_km:.4f}")
    print(f"[Mean Gene Exp] CoxPH p = {p_val_gene_cox:.4f} | HR = {hr_1:.3f}")
    if p_val_gene_cox < 0.05:
        print("  -> SIGNIFICANT (CoxPH)")
        print(cph_gene_model.summary)


    # --- 2. Analysis: Mean Major Exp ---
    
    var_to_analyze_2 = 'mean_major_exp'
    ax2 = axes_km[plot_row_offset + 1, plot_col] 
    kmf = KaplanMeierFitter() 

    # Group data by median for KM plot
    median_val_2 = analysis_df[var_to_analyze_2].median()
    analysis_df['group_2'] = (analysis_df[var_to_analyze_2] > median_val_2).astype(int) 

    high_data_2 = analysis_df[analysis_df['group_2'] == 1]
    low_data_2 = analysis_df[analysis_df['group_2'] == 0]

    # Fit and plot KM curves
    kmf.fit(low_data_2[TIME_COL], event_observed=low_data_2[EVENT_COL], label=f"Low (n={len(low_data_2)})")
    kmf.plot_survival_function(ax=ax2, ci_show=False, color='blue', show_censors=True)
    kmf.fit(high_data_2[TIME_COL], event_observed=high_data_2[EVENT_COL], label=f"High (n={len(high_data_2)})")
    kmf.plot_survival_function(ax=ax2, ci_show=False, color='red', show_censors=True)

    # Log-rank test (High vs Low)
    lr_results_2 = logrank_test(high_data_2[TIME_COL], low_data_2[TIME_COL], 
                                high_data_2[EVENT_COL], low_data_2[EVENT_COL])
    p_val_major_km = lr_results_2.p_value

    # --- ⬇️ TITLE MODIFIED HERE ⬇️ ---
    ax2.set_title(f"{name} - Mean Major Exp", fontsize=16)
    # --- ⬆️ TITLE MODIFIED HERE ⬆️ ---

    ax2.set_xlabel(TIME_COL)
    if plot_col == 0: 
        ax2.set_ylabel("Survival Probability")

    # CoxPH Model (using continuous variable)
    cph_major = CoxPHFitter()
    cph_major_model = cph_major.fit(analysis_df[[TIME_COL, EVENT_COL, var_to_analyze_2]], 
                                    duration_col=TIME_COL, event_col=EVENT_COL)
    summary_2 = cph_major_model.summary.loc[var_to_analyze_2]
    hr_2, ci_low_2, ci_high_2, p_val_major_cox = summary_2[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]]

    # Add CoxPH results as annotation
    ax2.text(
        0.05, 0.05,
        f"CoxPH (continuous):\n"
        f"HR = {hr_2:.2f} (95% CI {ci_low_2:.2f}–{ci_high_2:.2f})\n"
        f"Log-rank p = {p_val_major_km:.3f}\n" # Log-rank p-value moved here
        f"Cox p = {p_val_major_cox:.3e}",
        transform=ax2.transAxes, fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, boxstyle="round")
    )
    
    # Print significant results to console
    print(f"[Mean Major Exp] KM Log-rank p = {p_val_major_km:.4f}")
    print(f"[Mean Major Exp] CoxPH p = {p_val_major_cox:.4f} | HR = {hr_2:.3f}")
    if p_val_major_cox < 0.05:
        print("  -> SIGNIFICANT (CoxPH)")
        print(cph_major_model.summary)

print("\n=======================================================")

# Clean up any empty plots if n_sets < 10
for i in range(n_sets, n_cols * (n_rows // 2)):
    plot_col = i % n_cols
    plot_row_offset = (i // n_cols) * 2
    axes_km[plot_row_offset, plot_col].axis('off')
    axes_km[plot_row_offset + 1, plot_col].axis('off')

# Adjust layout and display the plot
plt.tight_layout(pad=2.0)
#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/dnarepair_KMPlot.pdf", dpi=300)
plt.show()

# %%
