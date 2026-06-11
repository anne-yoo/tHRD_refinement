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
import os
import matplotlib
import gseapy as gp
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.stats.multitest import multipletests
from matplotlib import rcParams
from statannotations.Annotator import Annotator
from statannot import add_stat_annotation
from matplotlib_venn import venn2
from pathlib import Path
from matplotlib import font_manager

rcParams['pdf.fonttype'] = 42  
rcParams['ps.fonttype'] = 42
for arial_font_path in [
    "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/arialbd.ttf",
]:
    if Path(arial_font_path).exists():
        font_manager.fontManager.addfont(arial_font_path)

plt.rcParams["font.family"] = "Arial"
rcParams['font.family'] = 'Arial'

plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 13,    # x축 틱 라벨 글꼴 크기
'ytick.labelsize': 13,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 13,
'legend.title_fontsize': 13, # 범례 글꼴 크기|
'figure.titlesize': 15    # figure 제목 글꼴 크기
})
sns.set_style("ticks")

# %%
# %%
####^^ (1-1) group 1 증가 GO enrichment######
sqanti = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/sqantioutput/sqanti_hg19_classification.txt', sep='\t')
sqanti.dropna(axis=1, how='all', inplace=True)

AR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/whole_AR_stable_DUT_Wilcoxon_delta_withna.txt', sep='\t', index_col=0)
IR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/whole_IR_stable_DUT_Wilcoxon_delta_withna.txt', sep='\t', index_col=0)
baseline_dut = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/whole_baseline_ARpre_vs_IRpre_stable_DUT_MannWhitney_delta_withna.txt', sep='\t', index_col=0)
ARdeg = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/whole_AR_Wilcoxon_DEGresult_FC.txt', sep='\t')
IRdeg = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/whole_IR_Wilcoxon_DEGresult_FC.txt', sep='\t')
# AR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/maintenance/AR_stable_DUT_Wilcoxon_delta_withna.txt', sep='\t', index_col=0) #^Only maintenance
# IR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/maintenance/IR_stable_DUT_Wilcoxon_delta_withna.txt', sep='\t', index_col=0) #^Only maintenance

ARdutlist = AR_dut.loc[(AR_dut['p_value']<0.05) & (np.abs(AR_dut['delta_TU'])>0.05)].index.to_list()
IRdutlist = IR_dut.loc[(IR_dut['p_value']<0.05) & (np.abs(IR_dut['delta_TU'])>0.05)].index.to_list()
baseline_dutlist = baseline_dut.loc[(baseline_dut['p_value']<0.05) & (np.abs(baseline_dut['delta_TU'])>0.05)].index.to_list()

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost_80_clinicalinfo.txt', sep='\t', index_col=0)
sampleinfo_full = sampleinfo.copy()
#sampleinfo = sampleinfo[sampleinfo['purpose']=='maintenance'] #^ Only maintenance

ARlist = list(set(sampleinfo.loc[(sampleinfo['response']==1),'sample_full']))
IRlist = list(set(sampleinfo.loc[(sampleinfo['response']==0),'sample_full']))

transexp = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_80_transcript_TPM.txt',sep='\t', index_col=0)
transexp = transexp.iloc[:,:-1]
transexp = transexp.loc[(transexp > 0).sum(axis=1) >= 8] #20% 이상에서는 나오긴 해야됨 ...

#transexp = transexp.loc[(transexp > 0).sum(axis=1) >= transexp.shape[1]*0.3]
transexp["gene"] = transexp.index.str.split("-", n=1).str[-1]
gene_sum = transexp.groupby("gene").transform("sum")
filtered_trans = transexp.iloc[:, :-1].div(gene_sum)
TU = filtered_trans.copy()

mainlist =(list(set(sampleinfo[sampleinfo['purpose']=='maintenance']['sample_full'])))
sallist = (list(set(sampleinfo_full[sampleinfo_full['purpose']=='salvage']['sample_full'])))

main_TU = TU[mainlist]
sal_TU = TU[sallist]

main_TU = main_TU.sort_index(axis=1)
main_TU.columns = main_TU.columns.str[:-4]

sal_TU = sal_TU.sort_index(axis=1)
sal_TU.columns = sal_TU.columns.str[:-4]

majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_80_majorminorlist.txt',sep='\t')
majorlist = majorminor[majorminor['type']=='major']['Transcript-Gene'].to_list()
minorlist = majorminor[majorminor['type']=='minor']['Transcript-Gene'].to_list()

sampleinfo = sampleinfo.iloc[::2,:]

proteincoding = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM_symbol.txt', sep='\t', index_col=0)
proteincodinglist = proteincoding['Gene Symbol'].to_list()

TU = filtered_trans.copy()
#TU = TU.loc[:,TU.columns.isin(mainlist)] #^Only maintenance
TU = TU[TU.index.str.split("-", n=1).str[-1].isin(proteincodinglist)]
TU.columns = TU.columns.str[:-4] 

# 인덱스 정리 (Transcript ID만 남김)
TU.index = TU.index.str.split("-", n=1).str[0]

# Pre/Post 분리
preTU = TU.iloc[:, 1::2] 
postTU = TU.iloc[:, 0::2]

preTU = preTU.fillna(0)
postTU = postTU.fillna(0)

# ---------------------------------------------------------
# 2. 샘플 분리 및 리스트 전처리
# ---------------------------------------------------------
# (1) 샘플 분리
ar_samples = sampleinfo[sampleinfo['response'] == 1].index.intersection(preTU.columns)
ir_samples = sampleinfo[sampleinfo['response'] == 0].index.intersection(preTU.columns)

# (2) DUT List 전처리 (ID만 추출)
ar_isoforms_clean = [x.split('-', 1)[0] for x in ARdutlist]
ir_isoforms_clean = [x.split('-', 1)[0] for x in IRdutlist]

# ---------------------------------------------------------
# 3. 데이터 집계 (Aggregation): 샘플별 평균 계산
# ---------------------------------------------------------
# Coding 여부(True/False)와 Response(AR/IR)에 따라 반복문을 돌며 평균을 계산합니다.

# df_cat의 isoform 컬럼도 ID만 남도록 정리 필요 (안전장치)
df_cat = sqanti[['isoform','structural_category','subcategory','within_CAGE_peak','coding']].copy()
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

class1 = majorminor[majorminor['transcriptid'].isin(group1)]['Transcript-Gene'].to_list()
class2 = majorminor[majorminor['transcriptid'].isin(group2)]['Transcript-Gene'].to_list()
class3 = majorminor[majorminor['transcriptid'].isin(group3)]['Transcript-Gene'].to_list()

geneexp = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_80_gene_TPM.txt', sep='\t', index_col=0)
# %%
SF_genes = [
    'ACIN1', 'AGGF1', 'ALYREF', 'AQR', 'ARGLU1', 'BAG2', 'BCAS1', 'BCAS2', 'BUB3', 'BUD13',
    'BUD31', 'C17orf85', 'C19orf43', 'C1orf55', 'C1QBP', 'C9orf78', 'CACTIN', 'CCAR1', 'CCDC12', 'CCDC130',
    'CCDC75', 'CCDC94', 'CD2BP2', 'CDC40', 'CDC5L', 'CDK10', 'CDK11A', 'CDK12', 'CELF1', 'CELF2',
    'CELF3', 'CELF4', 'CELF5', 'CELF6', 'CFAP20', 'CHERP', 'CIRBP', 'CLASRP', 'CLK1', 'CLK2',
    'CLK3', 'CLK4', 'CLNS1A', 'CPSF6', 'CRNKL1', 'CSN3', 'CTNNBL1', 'CWC15', 'CWC22', 'CWC25',
    'CWC27', 'CXorf56', 'DDX1', 'DDX17', 'DDX18', 'DDX19A', 'DDX19B', 'DDX20', 'DDX21', 'DDX23',
    'DDX26B', 'DDX27', 'DDX39A', 'DDX39B', 'DDX3X', 'DDX3Y', 'DDX41', 'DDX42', 'DDX46', 'DDX5',
    'DDX50', 'DDX6', 'DGCR14', 'DHX15', 'DHX16', 'DHX30', 'DHX34', 'DHX35', 'DHX36', 'DHX38',
    'DHX40', 'DHX57', 'DHX8', 'DHX9', 'DNAJC6', 'DNAJC8', 'EEF1A1', 'EFTUD2', 'EIF2S2', 'EIF3A',
    'EIF4A3', 'ELAVL1', 'ELAVL2', 'ELAVL3', 'ELAVL4', 'FAM32A', 'FAM50A', 'FAM50B', 'FAM58A', 'FMR1',
    'FRA10AC1', 'FRG1', 'FUBP1', 'FUBP3', 'FUS', 'GEMIN2', 'GEMIN5', 'GNB2L1', 'GPATCH1', 'GPATCH3',
    'GPATCH8', 'GPKOW', 'GRSF1', 'HNRNPA0', 'HNRNPA1', 'HNRNPA2B1', 'HNRNPA3', 'HNRNPAB', 'HNRNPC', 'HNRNPCL1',
    'HNRNPD', 'HNRNPDL', 'HNRNPF', 'HNRNPH1', 'HNRNPH2', 'HNRNPH3', 'HNRNPK', 'HNRNPL', 'HNRNPLL', 'HNRNPM',
    'HNRNPR', 'HNRNPU', 'HNRNPUL1', 'HNRNPUL2', 'HSPA1A', 'HSPA1B', 'HSPA5', 'HSPA8', 'HSPB1', 'HTATSF1',
    'IGF2BP3', 'IK', 'ILF2', 'ILF3', 'INTS1', 'INTS3', 'INTS4', 'INTS5', 'INTS6', 'INTS7',
    'ISY1', 'JUP', 'KHDRBS1', 'KHDRBS3', 'KHSRP', 'KIAA1429', 'KIAA1967', 'KIN', 'LENG1', 'LOC649330',
    'LSM1', 'LSM10', 'LSM2', 'LSM3', 'LSM4', 'LSM5', 'LSM6', 'LSM7', 'NAA38', 'LSMD1',
    'LUC7L', 'LUC7L2', 'LUC7L3', 'MAGOH', 'MATR3', 'MBNL1', 'MBNL2', 'MBNL3', 'MFAP1', 'MFSD11',
    'MOV10', 'MSI1', 'MSI2', 'MYEF2', 'NCBP1', 'NCBP2', 'NELFE', 'NKAP', 'NONO', 'NOSIP',
    'NOVA1', 'NOVA2', 'NRIP2', 'NSRP1', 'NUDT21', 'NUMA1', 'PABPC1', 'PAXBP1', 'PCBP1', 'PCBP2',
    'PCBP3', 'PCBP4', 'PDCD7', 'PHF5A', 'PLRG1', 'PNN', 'PPIE', 'PPIG', 'PPIH', 'PPIL1',
    'PPIL2', 'PPIL3', 'PPIL4', 'PPM1G', 'PPP1CA', 'PPP1R8', 'PPWD1', 'PQBP1', 'PRCC', 'PRMT5',
    'PRPF18', 'PRPF19', 'PRPF3', 'PRPF31', 'PRPF38A', 'PRPF38B', 'PRPF39', 'PRPF4', 'PRPF40A', 'PRPF40B',
    'PRPF4B', 'PRPF6', 'PRPF8', 'PSEN1', 'PSIP1', 'PTBP1', 'PTBP2', 'PTBP3', 'PUF60', 'QKI',
    'RALY', 'RALYL', 'RAVER1', 'RAVER2', 'RBBP6', 'RBFOX2', 'RBM10', 'RBM14', 'RBM15', 'RBM15B',
    'RBM17', 'RBM22', 'RBM23', 'RBM25', 'RBM26', 'RBM27', 'RBM3', 'RBM39', 'RBM4', 'RBM42',
    'RBM45', 'RBM47', 'RBM4B', 'RBM5', 'RBM7', 'RBM8A', 'RBMS1', 'RBMX', 'RBMX2', 'RBMXL1',
    'RBMXL2', 'RNF113A', 'RNF20', 'RNF213', 'RNF34', 'RNF40', 'RNPC3', 'RNPS1', 'RNU1-1', 'RNU2-1',
    'RNU4-1', 'RNU5A-1', 'RNU6-1', 'SAP18', 'SAP30BP', 'SART1', 'SEC31B', 'SF1', 'SF3A1', 'SF3A2',
    'SF3A3', 'SF3B1', 'SF3B2', 'SF3B3', 'SF3B4', 'SF3B5', 'SF3B6', 'SFPQ', 'SKIV2L2', 'SLU7',
    'SMN1', 'SMNDC1', 'SMU1', 'SNIP1', 'SNRNP200', 'SNRNP25', 'SNRNP27', 'SNRNP35', 'SNRNP40', 'SNRNP48',
    'SNRNP70', 'SNRPA', 'SNRPA1', 'SNRPB', 'SNRPB2', 'SNRPC', 'SNRPD1', 'SNRPD2', 'SNRPD3', 'SNRPE',
    'SNRPF', 'SNRPG', 'SNRPN', 'NHP2L1', 'SNURF', 'SNW1', 'SPEN', 'SREK1', 'SRPK1', 'SRPK2',
    'SRPK3', 'SRRM1', 'SRRM2', 'SRRT', 'SRSF1', 'SRSF10', 'SRSF11', 'SRSF12', 'SRSF2', 'SRSF3',
    'SRSF4', 'SRSF5', 'SRSF6', 'SRSF7', 'SRSF8', 'SRSF9', 'SSB', 'SUGP1', 'SYF2', 'SYNCRIP',
    'TAF15', 'TCERG1', 'TFIP11', 'THOC1', 'THOC2', 'THOC3', 'THOC5', 'THOC6', 'THOC7', 'THRAP3',
    'TIA1', 'TIAL1', 'TNPO1', 'TOE1', 'TOP1MT', 'TOPORS', 'TRA2A', 'TRA2B', 'TRIM24', 'TTC14',
    'TXNL4A', 'U2AF1', 'U2AF1L4', 'U2AF2', 'U2SURP', 'UBL5', 'USP39', 'WBP11', 'WBP4', 'WDR77',
    'WDR83', 'WTAP', 'XAB2', 'YBX1', 'YBX3', 'ZC3H11A', 'ZC3H13', 'ZC3H18', 'ZC3H4', 'ZC3HAV1',
    'ZCCHC10', 'ZCCHC8', 'ZCRB1', 'ZFR', 'ZMAT2', 'ZMAT5', 'ZMYM3', 'ZNF131', 'ZNF207', 'ZNF326',
    'ZNF346', 'ZNF830', 'ZRSR1', 'ZRSR2'
]
OUT_DIR = "/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/2605figs"
AR_COLOR = "#FF8D29"
IR_COLOR = "#8AC509"

# %%
######^^ (1) whole gene vs. SF gene correlation kdeplot ########
OUT_DIR = "/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/2605figs"
AR_COLOR = "#FF8D29"
IR_COLOR = "#8AC509"


def prepare_prepost_gene_matrices(raw_geneexp):
    pre_gene = raw_geneexp.loc[:, raw_geneexp.columns.str.endswith("-bfD")].copy()
    post_gene = raw_geneexp.loc[:, raw_geneexp.columns.str.endswith("-atD")].copy()

    pre_gene.columns = pre_gene.columns.str.replace("-bfD", "", regex=False)
    post_gene.columns = post_gene.columns.str.replace("-atD", "", regex=False)

    common_samples = pre_gene.columns.intersection(post_gene.columns)
    pre_gene = pre_gene.loc[:, common_samples].sort_index(axis=1)
    post_gene = post_gene.loc[:, common_samples].sort_index(axis=1)
    return pre_gene, post_gene


def compute_prepost_sample_corr(pre_data, post_data, feature_list, samples, method="spearman"):
    samples_valid = [sample for sample in samples if sample in pre_data.columns and sample in post_data.columns]
    features_valid = [feature for feature in feature_list if feature in pre_data.index and feature in post_data.index]

    if len(samples_valid) == 0 or len(features_valid) < 2:
        return None, samples_valid, features_valid

    pre_sub = pre_data.loc[features_valid, samples_valid]
    post_sub = post_data.loc[features_valid, samples_valid]

    pre_labels = [f"{sample} (pre)" for sample in samples_valid]
    post_labels = [f"{sample} (post)" for sample in samples_valid]

    combined = pd.concat([pre_sub.T, post_sub.T], axis=0)
    combined.index = pre_labels + post_labels
    corr_all = combined.T.corr(method=method)
    corr_matrix = corr_all.loc[pre_labels, post_labels]
    return corr_matrix, samples_valid, features_valid


def flatten_corr_values(corr_matrix):
    if corr_matrix is None or corr_matrix.empty:
        return np.array([])
    values = corr_matrix.to_numpy().reshape(-1)
    return values[~np.isnan(values)]


def plot_groupwise_prepost_corr_rho_kde(
    pre_gene,
    post_gene,
    features,
    ar_samples,
    ir_samples,
    save_stem,
    figsize=(4.5, 4),
):
    ar_corr, ar_samples_valid, features_valid = compute_prepost_sample_corr(
        pre_gene, post_gene, features, ar_samples
    )
    ir_corr, ir_samples_valid, _ = compute_prepost_sample_corr(
        pre_gene, post_gene, features, ir_samples
    )

    ar_rho = flatten_corr_values(ar_corr)
    ir_rho = flatten_corr_values(ir_corr)
    palette = {"AR": AR_COLOR, "IR": IR_COLOR}
    x_min, x_max = -0.25, 1.0

    fig, ax = plt.subplots(figsize=figsize)
    for group_name, rho_values in (("AR", ar_rho), ("IR", ir_rho)):
        if rho_values.size == 0:
            continue

        unique_values = np.unique(np.round(rho_values, 12))
        if unique_values.size >= 2:
            sns.kdeplot(
                rho_values,
                fill=True,
                common_norm=False,
                alpha=0.28,
                linewidth=1.5,
                color=palette[group_name],
                ax=ax,
                clip=(x_min, x_max),
                warn_singular=False,
                label=group_name,
            )
            sns.kdeplot(
                rho_values,
                fill=False,
                common_norm=False,
                linewidth=2.0,
                color=palette[group_name],
                ax=ax,
                clip=(x_min, x_max),
                warn_singular=False,
            )
        else:
            ax.axvline(rho_values[0], color=palette[group_name], linewidth=2.0, alpha=0.85, label=group_name)

        median_rho = float(np.median(rho_values))
        ax.axvline(median_rho, color=palette[group_name], linestyle="--", linewidth=1.3, alpha=0.9)

    ar_median = float(np.median(ar_rho)) if ar_rho.size else np.nan
    ir_median = float(np.median(ir_rho)) if ir_rho.size else np.nan
    ax.text(
        0.05,
        0.45,
        f"AR median $\\rho$ = {ar_median:.2f}\nIR median $\\rho$ = {ir_median:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
    )

    ax.axvline(0, color="gray", linestyle="-", linewidth=1.0, alpha=0.25, zorder=0)
    ax.set_xlim(x_min, x_max)
    ax.set_xlabel("Spearman $\\rho$")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="upper left")
    sns.despine(trim=True)
    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return {
        "pdf": pdf_path,
        "png": png_path,
        "ar_median": ar_median,
        "ir_median": ir_median,
        "ar_rho_n": int(ar_rho.size),
        "ir_rho_n": int(ir_rho.size),
        "ar_sample_n": len(ar_samples_valid),
        "ir_sample_n": len(ir_samples_valid),
        "feature_n": len(features_valid),
    }


pre_gene, post_gene = prepare_prepost_gene_matrices(geneexp)
common_samples = pre_gene.columns.intersection(post_gene.columns)
sampleinfo_corr = sampleinfo.loc[sampleinfo.index.intersection(common_samples)].copy()
ar_samples_corr = sorted(sampleinfo_corr.loc[sampleinfo_corr["response"] == 1].index.tolist())
ir_samples_corr = sorted(sampleinfo_corr.loc[sampleinfo_corr["response"] == 0].index.tolist())

min_expressed_samples = int(np.ceil(geneexp.shape[1] * 0.2))
expressed_genes = geneexp.index[(geneexp > 0).sum(axis=1) >= min_expressed_samples].tolist()
sf_features = sorted(set(SF_genes) & set(pre_gene.index) & set(post_gene.index))

all_gene_result = plot_groupwise_prepost_corr_rho_kde(
    pre_gene=pre_gene,
    post_gene=post_gene,
    features=expressed_genes,
    ar_samples=ar_samples_corr,
    ir_samples=ir_samples_corr,
    save_stem="prepost_spearman_rho_kde_all_genes_expr20pct",
)
sf_gene_result = plot_groupwise_prepost_corr_rho_kde(
    pre_gene=pre_gene,
    post_gene=post_gene,
    features=sf_features,
    ar_samples=ar_samples_corr,
    ir_samples=ir_samples_corr,
    save_stem="prepost_spearman_rho_kde_SF_genes",
)

print(
    "All genes:",
    f"features={all_gene_result['feature_n']}",
    f"AR median rho={all_gene_result['ar_median']:.4f}",
    f"IR median rho={all_gene_result['ir_median']:.4f}",
    f"saved={all_gene_result['pdf']}",
)
print(
    "SF genes:",
    f"features={sf_gene_result['feature_n']}",
    f"AR median rho={sf_gene_result['ar_median']:.4f}",
    f"IR median rho={sf_gene_result['ir_median']:.4f}",
    f"saved={sf_gene_result['pdf']}",
)

# %%
######^^ (1-1) SF gene mean delta gene expression boxplot ########
def compute_sf_gene_mean_delta_expression(pre_gene, post_gene, sf_features, ar_samples, ir_samples):
    features_valid = sorted(set(sf_features) & set(pre_gene.index) & set(post_gene.index))
    sample_groups = {
        "AR": [sample for sample in ar_samples if sample in pre_gene.columns and sample in post_gene.columns],
        "IR": [sample for sample in ir_samples if sample in pre_gene.columns and sample in post_gene.columns],
    }

    records = []
    for group_name, samples_valid in sample_groups.items():
        if len(samples_valid) == 0 or len(features_valid) == 0:
            continue

        delta_expr = post_gene.loc[features_valid, samples_valid] - pre_gene.loc[features_valid, samples_valid]
        mean_delta_expr = delta_expr.mean(axis=1, skipna=True)
        for gene_name, mean_delta in mean_delta_expr.dropna().items():
            records.append(
                {
                    "gene": gene_name,
                    "group": group_name,
                    "mean_delta_gene_expression": mean_delta,
                    "sample_n": len(samples_valid),
                }
            )

    return pd.DataFrame(records)


def get_sf_gene_delta_wilcoxon_pvalue(plot_df):
    if plot_df.empty:
        return np.nan, 0

    paired_df = plot_df.pivot(index="gene", columns="group", values="mean_delta_gene_expression")
    if not {"AR", "IR"}.issubset(paired_df.columns):
        return np.nan, 0

    paired_df = paired_df.dropna(subset=["AR", "IR"])
    if paired_df.shape[0] < 2:
        return np.nan, int(paired_df.shape[0])
    if np.allclose(paired_df["AR"].values, paired_df["IR"].values):
        return 1.0, int(paired_df.shape[0])

    try:
        p_value = stats.wilcoxon(paired_df["AR"], paired_df["IR"]).pvalue
    except ValueError:
        p_value = np.nan
    return p_value, int(paired_df.shape[0])


def sf_gene_delta_pvalue_to_star(p_value):
    if pd.isna(p_value):
        return "n/a"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def add_sf_gene_delta_stat_annotation(ax, p_value):
    text = sf_gene_delta_pvalue_to_star(p_value)
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    if y_range == 0:
        y_range = 1

    line_y = y_max + y_range * 0.04
    line_h = y_range * 0.03
    ax.plot([0, 0, 1, 1], [line_y, line_y + line_h, line_y + line_h, line_y], color="black", linewidth=1.0)
    ax.text(
        0.5,
        line_y + line_h,
        text,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="black",
    )
    ax.set_ylim(y_min, line_y + line_h + y_range * 0.10)


def plot_sf_gene_mean_delta_expression_boxplot(pre_gene, post_gene, sf_features, ar_samples, ir_samples, figsize=(4, 4)):
    plot_df = compute_sf_gene_mean_delta_expression(pre_gene, post_gene, sf_features, ar_samples, ir_samples)
    palette = {"AR": AR_COLOR, "IR": IR_COLOR}
    order = ["AR", "IR"]
    wilcoxon_p, paired_gene_n = get_sf_gene_delta_wilcoxon_pvalue(plot_df)

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=plot_df,
        x="group",
        y="mean_delta_gene_expression",
        order=order,
        palette=palette,
        showfliers=False,
        linewidth=1.0,
        ax=ax,
    )
    add_sf_gene_delta_stat_annotation(ax, wilcoxon_p)
    ax.axhline(0, color="gray", linestyle="-", linewidth=1.0, alpha=0.35, zorder=0)
    ax.set_xlabel("")
    ax.set_ylabel("Mean $\\Delta$ SF gene exp")
    ax.grid(axis="y", alpha=0.18)
    sns.despine(ax=ax, trim=False)
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    tsv_path = os.path.join(OUT_DIR, "SF_gene_mean_delta_gene_expression_AR_IR.tsv")
    pdf_path = os.path.join(OUT_DIR, "SF_gene_mean_delta_gene_expression_boxplot_AR_IR.pdf")
    png_path = os.path.join(OUT_DIR, "SF_gene_mean_delta_gene_expression_boxplot_AR_IR.png")
    plot_df.to_csv(tsv_path, sep="\t", index=False)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return {
        "pdf": pdf_path,
        "png": png_path,
        "tsv": tsv_path,
        "gene_n": int(plot_df["gene"].nunique()) if not plot_df.empty else 0,
        "paired_gene_n": paired_gene_n,
        "wilcoxon_p": wilcoxon_p,
    }


sf_gene_delta_result = plot_sf_gene_mean_delta_expression_boxplot(
    pre_gene=pre_gene,
    post_gene=post_gene,
    sf_features=sf_features,
    ar_samples=ar_samples_corr,
    ir_samples=ir_samples_corr,
    figsize=(4, 4),
)
print(
    "SF gene mean delta gene expression:",
    f"genes={sf_gene_delta_result['gene_n']}",
    f"paired_genes={sf_gene_delta_result['paired_gene_n']}",
    f"Wilcoxon p={sf_gene_delta_result['wilcoxon_p']:.4g}",
    f"saved={sf_gene_delta_result['pdf']}",
)

# %%
######^^ (1-2) Top AR SF DEG pre/post gene expression boxplot ########
def select_top_ar_sf_deg_genes(ar_deg_df, sf_features, expression_genes, top_n=4, p_cutoff=0.05):
    gene_col = "gene_name" if "gene_name" in ar_deg_df.columns else ar_deg_df.columns[0]
    p_col = "p_value" if "p_value" in ar_deg_df.columns else "pval"
    fc_col = "log2FC" if "log2FC" in ar_deg_df.columns else None

    work_df = ar_deg_df.copy()
    work_df["gene"] = work_df[gene_col].astype(str)
    work_df[p_col] = pd.to_numeric(work_df[p_col], errors="coerce")
    if fc_col is not None:
        work_df[fc_col] = pd.to_numeric(work_df[fc_col], errors="coerce")
        work_df["abs_log2FC"] = work_df[fc_col].abs()
    else:
        work_df["abs_log2FC"] = 0

    candidate_mask = (
        work_df["gene"].isin(sf_features)
        & work_df["gene"].isin(expression_genes)
        & work_df[p_col].notna()
    )
    candidate_df = work_df.loc[candidate_mask].copy()
    deg_df = candidate_df.loc[candidate_df[p_col] < p_cutoff].copy()
    if deg_df.shape[0] < top_n:
        deg_df = candidate_df.copy()

    top_df = (
        deg_df
        .sort_values([p_col, "abs_log2FC"], ascending=[True, False])
        .drop_duplicates("gene")
        .head(top_n)
        .reset_index(drop=True)
    )
    keep_cols = ["gene", p_col, "abs_log2FC"] + ([fc_col] if fc_col is not None else [])
    keep_cols = list(dict.fromkeys(keep_cols))
    return top_df["gene"].tolist(), top_df.loc[:, keep_cols]


def build_top_ar_sf_deg_expression_long_df(pre_gene, post_gene, top_genes, ar_samples, ir_samples):
    sample_groups = {
        "AR": ar_samples,
        "IR": ir_samples,
    }
    records = []
    for gene_name in top_genes:
        for group_name, sample_list in sample_groups.items():
            samples_valid = [
                sample
                for sample in sample_list
                if sample in pre_gene.columns and sample in post_gene.columns
            ]
            if len(samples_valid) == 0:
                continue

            paired_df = pd.DataFrame(
                {
                    "pre": pd.Series(pre_gene.loc[gene_name, samples_valid], dtype="float64"),
                    "post": pd.Series(post_gene.loc[gene_name, samples_valid], dtype="float64"),
                }
            ).dropna()
            if paired_df.empty:
                continue

            paired_df = np.log2(paired_df + 1)
            for sample_name, row in paired_df.iterrows():
                records.append(
                    {
                        "gene": gene_name,
                        "sample": sample_name,
                        "response_group": group_name,
                        "time": "pre",
                        "condition": f"{group_name} pre",
                        "expression": row["pre"],
                    }
                )
                records.append(
                    {
                        "gene": gene_name,
                        "sample": sample_name,
                        "response_group": group_name,
                        "time": "post",
                        "condition": f"{group_name} post",
                        "expression": row["post"],
                    }
                )
    return pd.DataFrame(records)


def get_top_ar_sf_deg_annotation_pairs(plot_df, gene_order):
    pairs = []
    for gene_name in gene_order:
        gene_df = plot_df.loc[plot_df["gene"] == gene_name]
        for group_name in ["AR", "IR"]:
            pre_label = f"{group_name} pre"
            post_label = f"{group_name} post"
            condition_counts = gene_df["condition"].value_counts()
            if condition_counts.get(pre_label, 0) >= 2 and condition_counts.get(post_label, 0) >= 2:
                pairs.append(((gene_name, pre_label), (gene_name, post_label)))
    return pairs


def plot_top_ar_sf_deg_prepost_expression_boxplot(
    pre_gene,
    post_gene,
    ar_deg_df,
    sf_features,
    ar_samples,
    ir_samples,
    selected_genes=None,
    figsize=(8, 4),
):
    if selected_genes is None:
        gene_order, selected_deg_df = select_top_ar_sf_deg_genes(
            ar_deg_df=ar_deg_df,
            sf_features=sf_features,
            expression_genes=pre_gene.index.intersection(post_gene.index),
            top_n=4,
        )
    else:
        expression_genes = set(pre_gene.index).intersection(post_gene.index)
        gene_order = [gene for gene in selected_genes if gene in expression_genes]
        gene_col = "gene_name" if "gene_name" in ar_deg_df.columns else ar_deg_df.columns[0]
        p_col = "p_value" if "p_value" in ar_deg_df.columns else "pval"
        fc_col = "log2FC" if "log2FC" in ar_deg_df.columns else None
        selected_deg_df = ar_deg_df.copy()
        selected_deg_df["gene"] = selected_deg_df[gene_col].astype(str)
        selected_deg_df = selected_deg_df.loc[selected_deg_df["gene"].isin(gene_order)].copy()
        if fc_col is not None:
            selected_deg_df[fc_col] = pd.to_numeric(selected_deg_df[fc_col], errors="coerce")
            selected_deg_df["abs_log2FC"] = selected_deg_df[fc_col].abs()
        selected_deg_df[p_col] = pd.to_numeric(selected_deg_df[p_col], errors="coerce")
        selected_deg_df["gene_order"] = pd.Categorical(selected_deg_df["gene"], categories=gene_order, ordered=True)
        selected_deg_df = selected_deg_df.sort_values("gene_order").drop(columns=["gene_order"]).reset_index(drop=True)

    plot_df = build_top_ar_sf_deg_expression_long_df(
        pre_gene=pre_gene,
        post_gene=post_gene,
        top_genes=gene_order,
        ar_samples=ar_samples,
        ir_samples=ir_samples,
    )
    hue_order = ["AR pre", "AR post", "IR pre", "IR post"]
    palette = {
        "AR pre": "#FDD49E",
        "AR post": "#F28E2B",
        "IR pre": "#C7E9C0",
        "IR post": "#5AAE61",
    }
    pairs = get_top_ar_sf_deg_annotation_pairs(plot_df, gene_order)

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(
        data=plot_df,
        x="gene",
        y="expression",
        hue="condition",
        order=gene_order,
        hue_order=hue_order,
        palette=palette,
        showfliers=False,
        linewidth=1.0,
        width=0.78,
        ax=ax,
    )

    if pairs:
        annotator = Annotator(
            ax,
            pairs,
            data=plot_df,
            x="gene",
            y="expression",
            hue="condition",
            order=gene_order,
            hue_order=hue_order,
        )
        annotator.configure(
            test="Wilcoxon",
            text_format="star",
            loc="inside",
            verbose=0,
            line_height=0.02,
            line_width=1.0,
            text_offset=1,
        )
        annotator.apply_and_annotate()

    ax.set_xlabel("")
    ax.set_ylabel("log2(TPM + 1)")
    ax.grid(axis="y", alpha=0.18)
    ax.legend('', frameon=False, loc="upper left")
    sns.despine(ax=ax, trim=False)
    fig.subplots_adjust(left=0.08, right=0.83, top=0.86, bottom=0.16)

    os.makedirs(OUT_DIR, exist_ok=True)
    selected_path = os.path.join(OUT_DIR, "top4_AR_SF_DEG_selected.tsv")
    long_path = os.path.join(OUT_DIR, "top4_AR_SF_DEG_prepost_expression_long.tsv")
    pdf_path = os.path.join(OUT_DIR, "top4_AR_SF_DEG_prepost_expression_boxplot.pdf")
    png_path = os.path.join(OUT_DIR, "top4_AR_SF_DEG_prepost_expression_boxplot.png")
    selected_deg_df.to_csv(selected_path, sep="\t", index=False)
    plot_df.to_csv(long_path, sep="\t", index=False)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return {
        "pdf": pdf_path,
        "png": png_path,
        "selected": selected_path,
        "long": long_path,
        "genes": gene_order,
        "annotation_pair_n": len(pairs),
    }


top_ar_sf_deg_expr_result = plot_top_ar_sf_deg_prepost_expression_boxplot(
    pre_gene=pre_gene,
    post_gene=post_gene,
    ar_deg_df=ARdeg,
    sf_features=sf_features,
    ar_samples=ar_samples_corr,
    ir_samples=ir_samples_corr,
    selected_genes=["DDX1", "PRPF6", "HNRNPF","RBM47"],
    figsize=(8, 4),
)
print(
    "Top AR SF DEG pre/post expression:",
    f"genes={','.join(top_ar_sf_deg_expr_result['genes'])}",
    f"annotation_pairs={top_ar_sf_deg_expr_result['annotation_pair_n']}",
    f"saved={top_ar_sf_deg_expr_result['pdf']}",
)

# %%
#####^^ (2) DEG GO enrichment ########
def get_upregulated_deg_genes(deg_df, log2fc_cutoff=1, p_cutoff=0.05):
    gene_col = "gene_name" if "gene_name" in deg_df.columns else deg_df.columns[0]
    p_col = "p_value" if "p_value" in deg_df.columns else "pval"
    fc_col = "log2FC"

    work_df = deg_df.copy()
    work_df[p_col] = pd.to_numeric(work_df[p_col], errors="coerce")
    work_df[fc_col] = pd.to_numeric(work_df[fc_col], errors="coerce")

    gene_series = work_df[gene_col] if gene_col in work_df.columns else work_df.index.to_series()
    mask = (work_df[fc_col] > log2fc_cutoff) & (work_df[p_col] < p_cutoff)
    genes = (
        gene_series.loc[mask]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )
    return genes


def run_deg_pathway_enrichr(
    gene_list,
    label,
    gene_sets=None,
    save_tag="Reactome2022_GOBP2021",
):
    if len(gene_list) == 0:
        print(f"{label}: no input genes for enrichment")
        return pd.DataFrame()

    if gene_sets is None:
        gene_sets = ["Reactome_2022", "GO_Biological_Process_2021"]

    enr = gp.enrichr(
        gene_list=gene_list,
        gene_sets=gene_sets,
        organism="human",
        outdir=None,
    )
    enrresult = enr.results.sort_values(by=["Adjusted P-value"]).copy()
    result_path = os.path.join(
        OUT_DIR,
        f"{label}_DEG_log2FCgt1_pval005_{save_tag}_enrichr.tsv",
    )
    enrresult.to_csv(result_path, sep="\t", index=False)
    return enrresult


def wrap_term(term, width=74):
    words = str(term).split()
    lines = []
    current = ""
    for word in words:
        candidate = word if current == "" else f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def plot_deg_pathway_fdr_bar_with_terms(
    enrichr_df,
    gene_count,
    label,
    color,
    fdr_cutoff=0.1,
    top_n=10,
    figsize=(7, 4),
    save_tag="Reactome2022_GOBP2021",
    include_gene_set_label=True,
):
    os.makedirs(OUT_DIR, exist_ok=True)
    pdf_path = os.path.join(OUT_DIR, f"{label}_DEG_log2FCgt1_pval005_{save_tag}_top10_FDRbar.pdf")
    png_path = os.path.join(OUT_DIR, f"{label}_DEG_log2FCgt1_pval005_{save_tag}_top10_FDRbar.png")

    if enrichr_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        ax.text(0.02, 0.52, f"No enrichment result\nInput genes: {gene_count}", ha="left", va="center", fontsize=10)
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        return pdf_path

    plot_df = enrichr_df.copy()
    plot_df["FDR"] = pd.to_numeric(plot_df["Adjusted P-value"], errors="coerce")
    plot_df = plot_df.dropna(subset=["FDR"])
    plot_df = plot_df.loc[plot_df["FDR"] < fdr_cutoff].sort_values("FDR").head(top_n).copy()

    if plot_df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        ax.text(
            0.02,
            0.52,
            f"No significant terms (FDR < {fdr_cutoff})\nInput genes: {gene_count}",
            ha="left",
            va="center",
            fontsize=10,
        )
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        return pdf_path

    plot_df["Term_clean"] = (
        plot_df["Term"]
        .astype(str)
        .str.replace(r"\s+R-HSA-\d+.*$", "", regex=True)
    )
    if include_gene_set_label and "Gene_set" in plot_df.columns:
        plot_df["Gene_set_short"] = (
            plot_df["Gene_set"]
            .astype(str)
            .replace(
                {
                    "Reactome_2022": "Reactome",
                    "GO_Biological_Process_2021": "GO BP",
                }
            )
        )
    else:
        plot_df["Gene_set_short"] = ""
    plot_df["Term_display"] = np.where(
        plot_df["Gene_set_short"].eq(""),
        plot_df["Term_clean"],
        plot_df["Term_clean"] + " [" + plot_df["Gene_set_short"] + "]",
    )
    plot_df["neg_log10_FDR"] = -np.log10(plot_df["FDR"].clip(lower=np.nextafter(0, 1)))

    y_pos = np.arange(plot_df.shape[0])
    fig, (ax_bar, ax_text) = plt.subplots(
        1,
        2,
        figsize=figsize,
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 4.9], "wspace": 0.025},
    )

    ax_bar.barh(y_pos, plot_df["neg_log10_FDR"], color=color, alpha=0.9)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([])
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("-log10(FDR)", fontsize=10)
    ax_bar.grid(axis="x", alpha=0.2)
    ax_bar.tick_params(axis="y", length=0)
    sns.despine(ax=ax_bar, left=True)

    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(ax_bar.get_ylim())
    ax_text.axis("off")
    for y, (_, row) in zip(y_pos, plot_df.iterrows()):
        ax_text.text(
            0.0,
            y,
            wrap_term(row["Term_display"]),
            ha="left",
            va="center",
            fontsize=10,
        )

    fig.subplots_adjust(left=0.035, right=0.995, top=0.995, bottom=0.14)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return pdf_path


ar_deg_up_genes = get_upregulated_deg_genes(ARdeg, log2fc_cutoff=1, p_cutoff=0.05)
ir_deg_up_genes = get_upregulated_deg_genes(IRdeg, log2fc_cutoff=1, p_cutoff=0.05)

DEG_BAR_COLOR = "#1B365D"

ar_deg_pathway_result = run_deg_pathway_enrichr(ar_deg_up_genes, "AR")
ir_deg_pathway_result = run_deg_pathway_enrichr(ir_deg_up_genes, "IR")

ar_deg_pathway_fig = plot_deg_pathway_fdr_bar_with_terms(
    ar_deg_pathway_result,
    gene_count=len(ar_deg_up_genes),
    label="AR",
    color=DEG_BAR_COLOR,
    fdr_cutoff=0.1,
    top_n=10,
)
ir_deg_pathway_fig = plot_deg_pathway_fdr_bar_with_terms(
    ir_deg_pathway_result,
    gene_count=len(ir_deg_up_genes),
    label="IR",
    color=DEG_BAR_COLOR,
    fdr_cutoff=0.1,
    top_n=10,
)

print(f"AR upregulated DEG genes: {len(ar_deg_up_genes)}, saved={ar_deg_pathway_fig}")
print(f"IR upregulated DEG genes: {len(ir_deg_up_genes)}, saved={ir_deg_pathway_fig}")

# Reactome-only version saved separately.
ar_deg_reactome_result = run_deg_pathway_enrichr(
    ar_deg_up_genes,
    "AR",
    gene_sets=["Reactome_2022"],
    save_tag="Reactome2022_only",
)
ir_deg_reactome_result = run_deg_pathway_enrichr(
    ir_deg_up_genes,
    "IR",
    gene_sets=["Reactome_2022"],
    save_tag="Reactome2022_only",
)

ar_deg_reactome_fig = plot_deg_pathway_fdr_bar_with_terms(
    ar_deg_reactome_result,
    gene_count=len(ar_deg_up_genes),
    label="AR",
    color=DEG_BAR_COLOR,
    fdr_cutoff=0.1,
    top_n=10,
    save_tag="Reactome2022_only",
    include_gene_set_label=False,
)
ir_deg_reactome_fig = plot_deg_pathway_fdr_bar_with_terms(
    ir_deg_reactome_result,
    gene_count=len(ir_deg_up_genes),
    label="IR",
    color=DEG_BAR_COLOR,
    fdr_cutoff=0.1,
    top_n=10,
    save_tag="Reactome2022_only",
    include_gene_set_label=False,
)

print(f"AR Reactome-only DEG GO barplot saved={ar_deg_reactome_fig}")
print(f"IR Reactome-only DEG GO barplot saved={ir_deg_reactome_fig}")

# %%
#####^^ (3) ConsensusOV pre-post subtype transition ########
CONSENSUSOV_AR_PATH = "/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/ConsensusOV_AR.txt"
CONSENSUSOV_IR_PATH = "/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/ConsensusOV_IR.txt"


def load_consensusov_transition(path, group):
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip().str.replace('"', '', regex=False)

    if {"sample_2", "sample_1"}.issubset(df.columns):
        df = df.rename(columns={"sample_2": "pre", "sample_1": "post"})
    else:
        df = df.rename(columns={df.columns[0]: "pre", df.columns[1]: "post"})

    df["pre"] = df["pre"].astype(str).str.replace("_consensus", "", regex=False)
    df["post"] = df["post"].astype(str).str.replace("_consensus", "", regex=False)
    df["group"] = group
    df["sample_id"] = [f"{group}_{idx + 1:02d}" for idx in range(df.shape[0])]
    return df[["sample_id", "group", "pre", "post"]]


def draw_alluvial_ribbon(ax, x0, x1, y0_bottom, y0_top, y1_bottom, y1_top, color, alpha=0.58):
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    curve = (x1 - x0) * 0.55
    verts = [
        (x0, y0_bottom),
        (x0 + curve, y0_bottom),
        (x1 - curve, y1_bottom),
        (x1, y1_bottom),
        (x1, y1_top),
        (x1 - curve, y1_top),
        (x0 + curve, y0_top),
        (x0, y0_top),
        (x0, y0_bottom),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    patch = PathPatch(
        Path(verts, codes),
        facecolor=color,
        edgecolor="white",
        linewidth=0.45,
        alpha=alpha,
        zorder=1,
    )
    ax.add_patch(patch)


def get_consensusov_stack_bounds(values, categories, total_n, gap=0.035):
    available_height = 1 - gap * (len(categories) - 1)
    y_top = 1.0
    bounds = {}
    for category in categories:
        height = (values.get(category, 0) / total_n) * available_height
        y_bottom = y_top - height
        bounds[category] = (y_bottom, y_top)
        y_top = y_bottom - gap
    return bounds, available_height


def plot_single_consensusov_alluvial(ax, group_df, group_label, category_order, category_colors):
    from matplotlib.patches import Rectangle

    group_df = group_df.copy()
    group_df["changed"] = group_df["pre"] != group_df["post"]
    total_n = group_df.shape[0]

    pre_counts = group_df["pre"].value_counts().to_dict()
    post_counts = group_df["post"].value_counts().to_dict()
    left_bounds, available_height = get_consensusov_stack_bounds(pre_counts, category_order, total_n)
    right_bounds, _ = get_consensusov_stack_bounds(post_counts, category_order, total_n)

    flow_counts = (
        group_df
        .groupby(["pre", "post"])
        .size()
        .reset_index(name="count")
    )
    flow_counts["pre"] = pd.Categorical(flow_counts["pre"], categories=category_order, ordered=True)
    flow_counts["post"] = pd.Categorical(flow_counts["post"], categories=category_order, ordered=True)

    left_offsets = {category: left_bounds[category][0] for category in category_order}
    right_offsets = {category: right_bounds[category][0] for category in category_order}
    left_segments = {}
    right_segments = {}

    for _, row in flow_counts.sort_values(["pre", "post"]).iterrows():
        pre = str(row["pre"])
        post = str(row["post"])
        height = (row["count"] / total_n) * available_height
        left_segments[(pre, post)] = (left_offsets[pre], left_offsets[pre] + height)
        left_offsets[pre] += height

    for _, row in flow_counts.sort_values(["post", "pre"]).iterrows():
        pre = str(row["pre"])
        post = str(row["post"])
        height = (row["count"] / total_n) * available_height
        right_segments[(pre, post)] = (right_offsets[post], right_offsets[post] + height)
        right_offsets[post] += height

    x_left, x_right = 0.20, 0.80
    bar_width = 0.085
    for _, row in flow_counts.sort_values("count", ascending=True).iterrows():
        pre = str(row["pre"])
        post = str(row["post"])
        y0_bottom, y0_top = left_segments[(pre, post)]
        y1_bottom, y1_top = right_segments[(pre, post)]
        draw_alluvial_ribbon(
            ax,
            x_left,
            x_right,
            y0_bottom,
            y0_top,
            y1_bottom,
            y1_top,
            color=category_colors.get(pre, "#777777"),
            alpha=0.50 if pre == post else 0.66,
        )

    for side, bounds, x in [("pre", left_bounds, x_left), ("post", right_bounds, x_right)]:
        for category, (y_bottom, y_top) in bounds.items():
            if y_top <= y_bottom:
                continue
            ax.add_patch(
                Rectangle(
                    (x - bar_width / 2, y_bottom),
                    bar_width,
                    y_top - y_bottom,
                    facecolor=category_colors.get(category, "#777777"),
                    edgecolor="white",
                    linewidth=0.8,
                    alpha=0.96,
                    zorder=3,
                )
            )
            if y_top - y_bottom > 0.055:
                text_x = x - 0.050 if side == "pre" else x + 0.050
                ha = "right" if side == "pre" else "left"
                ax.text(
                    text_x,
                    (y_bottom + y_top) / 2,
                    category,
                    ha=ha,
                    va="center",
                    fontsize=8,
                )

    ax.text(0.5, 1.045, group_label, ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.text(x_left, -0.06, "pre", ha="center", va="top", fontsize=9)
    ax.text(x_right, -0.06, "post", ha="center", va="top", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.09, 1.08)
    ax.axis("off")


def plot_consensusov_alluvial(ar_df, ir_df, figsize=(5, 4)):
    transition_df = pd.concat([ar_df, ir_df], axis=0, ignore_index=True)
    transition_df["changed"] = transition_df["pre"] != transition_df["post"]
    transition_df["change_label"] = np.where(transition_df["changed"], "Changed", "Stable")
    transition_df["transition"] = transition_df["pre"] + " -> " + transition_df["post"]

    preferred_order = ["PRO", "MES", "IMR", "DIF"]
    observed_categories = pd.Index(transition_df[["pre", "post"]].to_numpy().ravel()).dropna().unique().tolist()
    category_order = [category for category in preferred_order if category in observed_categories]
    category_order += sorted([category for category in observed_categories if category not in category_order])
    category_colors = {
        "PRO": "#B279A2",
        "MES": "#4E79A7",
        "IMR": "#76B7B2",
        "DIF": "#F28E2B",
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.03},
    )
    plot_single_consensusov_alluvial(axes[0], ar_df, "AR", category_order, category_colors)
    plot_single_consensusov_alluvial(axes[1], ir_df, "IR", category_order, category_colors)

    os.makedirs(OUT_DIR, exist_ok=True)
    sample_path = os.path.join(OUT_DIR, "ConsensusOV_prepost_alluvial_AR_IR_samples.tsv")
    transition_df.to_csv(sample_path, sep="\t", index=False)
    transition_count_path = os.path.join(OUT_DIR, "ConsensusOV_prepost_alluvial_AR_IR_transition_counts.tsv")
    transition_df.groupby(["group", "pre", "post"]).size().reset_index(name="count").to_csv(
        transition_count_path,
        sep="\t",
        index=False,
    )

    pdf_path = os.path.join(OUT_DIR, "ConsensusOV_prepost_alluvial_AR_IR.pdf")
    png_path = os.path.join(OUT_DIR, "ConsensusOV_prepost_alluvial_AR_IR.png")
    fig.subplots_adjust(left=0.015, right=0.985, top=0.91, bottom=0.12, wspace=0.03)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return pdf_path


consensusov_ar = load_consensusov_transition(CONSENSUSOV_AR_PATH, "AR")
consensusov_ir = load_consensusov_transition(CONSENSUSOV_IR_PATH, "IR")
consensusov_alluvial_fig = plot_consensusov_alluvial(consensusov_ar, consensusov_ir, figsize=(5, 4))
print(f"ConsensusOV alluvial figure saved={consensusov_alluvial_fig}")

# %%
###^^ (4) dPSI event count  ##########
ar_psi = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/suppaoutput/AR/MW_psi_7events.txt',sep='\t')
ir_psi = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/suppaoutput/IR/MW_psi_7events.txt',sep='\t')


def count_dpsi_event_direction(psi_df, event_col="event", dpsi_col="d_psi"):
    event_order = ["MX", "AL", "AF", "SE", "RI", "A5", "A3"]
    work_df = psi_df.copy()
    work_df[dpsi_col] = pd.to_numeric(work_df[dpsi_col], errors="coerce")
    work_df = work_df.dropna(subset=[event_col, dpsi_col]).copy()
    work_df = work_df.loc[work_df[dpsi_col] != 0].copy()
    work_df["Direction"] = np.where(work_df[dpsi_col] < 0, "Pre", "Post")

    counts = (
        work_df
        .groupby([event_col, "Direction"])
        .size()
        .unstack(fill_value=0)
        .reindex(event_order)
        .fillna(0)
        .astype(int)
    )
    for direction in ["Pre", "Post"]:
        if direction not in counts.columns:
            counts[direction] = 0
    return counts[["Pre", "Post"]]


def plot_dpsi_event_count_panel(count_df, title, ax, x_limit, colors):
    y_pos = np.arange(count_df.shape[0])
    pre_values = -count_df["Pre"].values
    post_values = count_df["Post"].values

    ax.barh(
        y_pos,
        pre_values,
        color=colors["Pre"],
        edgecolor="#505050",
        linewidth=0.45,
        height=0.78,
        label="Pre",
    )
    ax.barh(
        y_pos,
        post_values,
        color=colors["Post"],
        edgecolor="#505050",
        linewidth=0.45,
        height=0.78,
        label="Post",
    )

    ax.axvline(0, color="#2A2A2A", linewidth=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(count_df.index)
    ax.invert_yaxis()
    ax.set_xlim(-x_limit, x_limit)
    ax.set_title(title, fontweight="bold", fontsize=12, pad=6)
    ax.set_xlabel("Number of Events")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{int(abs(value))}"))
    ax.grid(axis="x", alpha=0.14)
    ax.legend(frameon=False, loc="lower right")
    sns.despine(ax=ax)


def plot_dpsi_event_count(ar_psi, ir_psi, figsize=(8, 4)):
    ar_counts = count_dpsi_event_direction(ar_psi)
    ir_counts = count_dpsi_event_direction(ir_psi)
    max_count = max(
        ar_counts[["Pre", "Post"]].to_numpy().max(),
        ir_counts[["Pre", "Post"]].to_numpy().max(),
    )
    x_limit = int(np.ceil(max_count / 50) * 50) if max_count > 0 else 10

    palette = {
        "AR pre": "#FDD49E",
        "AR post": "#F28E2B",
        "IR pre": "#C7E9C0",
        "IR post": "#5AAE61",
    }
    ar_colors = {"Pre": palette["AR pre"], "Post": palette["AR post"]}
    ir_colors = {"Pre": palette["IR pre"], "Post": palette["IR post"]}

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    plot_dpsi_event_count_panel(ar_counts, "AR", axes[0], x_limit, ar_colors)
    plot_dpsi_event_count_panel(ir_counts, "IR", axes[1], x_limit, ir_colors)

    dpsi_count_df = pd.concat(
        [
            ar_counts.assign(Group="AR"),
            ir_counts.assign(Group="IR"),
        ]
    ).reset_index(names="event")
    os.makedirs(OUT_DIR, exist_ok=True)
    dpsi_count_path = os.path.join(OUT_DIR, "dPSI_event_count_pre_post_AR_IR.tsv")
    dpsi_count_df.to_csv(dpsi_count_path, sep="\t", index=False)

    pdf_path = os.path.join(OUT_DIR, "dPSI_event_count_pre_post_AR_IR.pdf")
    png_path = os.path.join(OUT_DIR, "dPSI_event_count_pre_post_AR_IR.png")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.16, wspace=0.1)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return pdf_path


dpsi_event_count_fig = plot_dpsi_event_count(ar_psi, ir_psi, figsize=(8, 4))
print(f"dPSI event count figure saved={dpsi_event_count_fig}")

# %%
###^^ (5) RI dPSI distribution KDE ##########
def get_ri_dpsi_values(psi_df, event_col="event", dpsi_col="d_psi"):
    ri_df = psi_df.loc[psi_df[event_col] == "RI"].copy()
    return pd.to_numeric(ri_df[dpsi_col], errors="coerce").dropna().values


def plot_ri_dpsi_distribution(ar_psi, ir_psi, figsize=(4.5, 4)):
    ar_ri_dpsi = get_ri_dpsi_values(ar_psi)
    ir_ri_dpsi = get_ri_dpsi_values(ir_psi)
    palette = {"AR": AR_COLOR, "IR": IR_COLOR}
    x_min, x_max = -0.7, 0.7

    fig, ax = plt.subplots(figsize=figsize)
    for group_name, dpsi_values in (("AR", ar_ri_dpsi), ("IR", ir_ri_dpsi)):
        if dpsi_values.size == 0:
            continue

        unique_values = np.unique(np.round(dpsi_values, 12))
        if unique_values.size >= 2:
            sns.kdeplot(
                dpsi_values,
                fill=True,
                common_norm=False,
                alpha=0.28,
                linewidth=1.5,
                color=palette[group_name],
                ax=ax,
                
                warn_singular=False,
                label=group_name,
            )
            sns.kdeplot(
                dpsi_values,
                fill=False,
                common_norm=False,
                linewidth=2.0,
                color=palette[group_name],
                ax=ax,
                
                warn_singular=False,
            )
        else:
            ax.axvline(dpsi_values[0], color=palette[group_name], linewidth=2.0, alpha=0.85, label=group_name)

        median_dpsi = float(np.median(dpsi_values))
        ax.axvline(median_dpsi, color=palette[group_name], linestyle="--", linewidth=1.3, alpha=0.9)

    ax.axvline(0, color="gray", linestyle="-", linewidth=1.0, alpha=0.25, zorder=0)
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(np.arange(-0.7, 0.71, 0.35))
    #ax.set_ylim(0, 4.)
    ax.set_xlabel("RI $\\Delta$PSI (Post - Pre)")
    ax.set_ylabel("Density")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, loc="upper left")
    sns.despine(ax=ax, trim=False)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.98, bottom=0.15)

    os.makedirs(OUT_DIR, exist_ok=True)
    pdf_path = os.path.join(OUT_DIR, "RI_dPSI_distribution_kde_AR_IR.pdf")
    png_path = os.path.join(OUT_DIR, "RI_dPSI_distribution_kde_AR_IR.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return pdf_path


ri_dpsi_distribution_fig = plot_ri_dpsi_distribution(ar_psi, ir_psi, figsize=(6, 4))
print(f"RI dPSI distribution figure saved={ri_dpsi_distribution_fig}")

# %%
###^^ (5-1) RI PSI pre/post distribution KDE ##########
RI_PSI_PATHS = {
    "AR": {
        "pre": "/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/suppaoutput/AR/AR_pre-suppaevent_RI_variable_10.ioe.psi",
        "post": "/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/suppaoutput/AR/AR_post-suppaevent_RI_variable_10.ioe.psi",
    },
    "IR": {
        "pre": "/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/suppaoutput/IR/IR_pre-suppaevent_RI_variable_10.ioe.psi",
        "post": "/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/suppaoutput/IR/IR_post-suppaevent_RI_variable_10.ioe.psi",
    },
}


def read_suppa_psi_table(psi_path):
    with open(psi_path) as handle:
        sample_cols = handle.readline().rstrip("\n").split("\t")
    return pd.read_csv(
        psi_path,
        sep="\t",
        names=["event_id"] + sample_cols,
        skiprows=1,
        na_values=["nan", "NaN"],
    )


def get_ri_event_ids(psi_df, event_col="event", event_id_col="event_id"):
    return set(psi_df.loc[psi_df[event_col] == "RI", event_id_col].dropna().astype(str))


def get_ri_psi_values(psi_path, event_ids=None):
    psi_df = read_suppa_psi_table(psi_path)
    if event_ids is not None:
        psi_df = psi_df.loc[psi_df["event_id"].astype(str).isin(event_ids)].copy()

    psi_values = psi_df.drop(columns=["event_id"]).apply(pd.to_numeric, errors="coerce")
    psi_values = psi_values.to_numpy().reshape(-1)
    psi_values = psi_values[~np.isnan(psi_values)]
    return psi_values


def plot_ri_psi_prepost_distribution(ar_psi, ir_psi, figsize=(8, 4), fill_alpha=0.35):
    palette = {
        "AR pre": "#FDD49E",
        "AR post": "#F28E2B",
        "IR pre": "#C7E9C0",
        "IR post": "#5AAE61",
    }
    event_ids_by_group = {
        "AR": get_ri_event_ids(ar_psi),
        "IR": get_ri_event_ids(ir_psi),
    }
    psi_values_by_group = {
        group_name: {
            time_label: get_ri_psi_values(
                psi_path,
                event_ids=event_ids_by_group[group_name],
            )
            for time_label, psi_path in time_paths.items()
        }
        for group_name, time_paths in RI_PSI_PATHS.items()
    }

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    for ax, group_name in zip(axes, ["AR", "IR"]):
        for time_label in ["pre", "post"]:
            label = f"{group_name} {time_label}"
            psi_values = psi_values_by_group[group_name][time_label]
            if psi_values.size == 0:
                continue

            unique_values = np.unique(np.round(psi_values, 12))
            if unique_values.size >= 2:
                sns.kdeplot(
                    psi_values,
                    fill=True,
                    common_norm=False,
                    alpha=fill_alpha,
                    linewidth=1.4,
                    color=palette[label],
                    ax=ax,
                    clip=(0, 1),
                    cut=0,
                    warn_singular=False,
                    label=label,
                )
                sns.kdeplot(
                    psi_values,
                    fill=False,
                    common_norm=False,
                    linewidth=2.0,
                    color=palette[label],
                    ax=ax,
                    clip=(0, 1),
                    cut=0,
                    warn_singular=False,
                )
            else:
                ax.axvline(psi_values[0], color=palette[label], linewidth=2.0, alpha=0.85, label=label)

            median_psi = float(np.median(psi_values))
            ax.axvline(median_psi, color=palette[label], linestyle="--", linewidth=1.3, alpha=0.9)

        ax.set_title(group_name, fontweight="bold", fontsize=12, pad=6)
        ax.set_xlim(0, 1)
        ax.set_xticks(np.arange(0, 1.01, 0.25))
        ax.set_xlabel("PSI")
        ax.grid(alpha=0.18)
        ax.legend(frameon=False, loc="upper left")
        sns.despine(ax=ax, trim=False)

    axes[0].set_ylabel("Density")
    axes[1].set_ylabel("")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.15, wspace=0.12)

    os.makedirs(OUT_DIR, exist_ok=True)
    pdf_path = os.path.join(OUT_DIR, "RI_PSI_prepost_distribution_kde_AR_IR.pdf")
    png_path = os.path.join(OUT_DIR, "RI_PSI_prepost_distribution_kde_AR_IR.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return pdf_path


ri_psi_prepost_distribution_fig = plot_ri_psi_prepost_distribution(ar_psi, ir_psi, figsize=(8, 4))
print(f"RI PSI pre/post distribution figure saved={ri_psi_prepost_distribution_fig}")

# %%
###^^ (6) Shared and group-specific dPSI events by event type ##########
def get_event_sets_by_type(psi_df, event_col="event", event_id_col="event_id"):
    event_sets = {}
    for event_type, group_df in psi_df.dropna(subset=[event_col, event_id_col]).groupby(event_col):
        event_sets[event_type] = set(group_df[event_id_col].astype(str))
    return event_sets


def build_event_overlap_counts(ar_psi, ir_psi, event_col="event", event_id_col="event_id"):
    event_order = ["SE", "RI", "AF", "AL", "A3", "A5", "MX"]
    ar_sets = get_event_sets_by_type(ar_psi, event_col=event_col, event_id_col=event_id_col)
    ir_sets = get_event_sets_by_type(ir_psi, event_col=event_col, event_id_col=event_id_col)
    observed_events = sorted(set(ar_sets.keys()) | set(ir_sets.keys()))
    event_order = [event for event in event_order if event in observed_events] + [
        event for event in observed_events if event not in event_order
    ]

    records = []
    for event_type in event_order:
        ar_set = ar_sets.get(event_type, set())
        ir_set = ir_sets.get(event_type, set())
        shared = ar_set & ir_set
        records.append(
            {
                "event": event_type,
                "AR-only": len(ar_set - ir_set),
                "shared": len(shared),
                "IR-only": len(ir_set - ar_set),
            }
        )
    return pd.DataFrame(records)


def plot_event_overlap_stacked_bar(ar_psi, ir_psi, figsize=(5, 4)):
    overlap_df = build_event_overlap_counts(ar_psi, ir_psi)
    colors = {
        "AR-only": AR_COLOR,
        "shared": "#5B6F95",
        "IR-only": IR_COLOR,
    }

    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(overlap_df.shape[0])
    x_pos = np.arange(overlap_df.shape[0])
    for category in ["AR-only", "shared", "IR-only"]:
        ax.bar(
            x_pos,
            overlap_df[category].values,
            bottom=bottom,
            color=colors[category],
            edgecolor="#404040",
            linewidth=0.45,
            width=0.72,
            label=category,
        )
        bottom += overlap_df[category].values

    ax.set_xticks(x_pos)
    ax.set_xticklabels(overlap_df["event"])
    ax.set_ylabel("Number of Events")
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.16)
    ax.legend(frameon=False, loc="upper right")
    sns.despine(ax=ax)

    os.makedirs(OUT_DIR, exist_ok=True)
    count_path = os.path.join(OUT_DIR, "dPSI_event_overlap_AR_only_shared_IR_only.tsv")
    overlap_df.to_csv(count_path, sep="\t", index=False)

    pdf_path = os.path.join(OUT_DIR, "dPSI_event_overlap_AR_only_shared_IR_only.pdf")
    png_path = os.path.join(OUT_DIR, "dPSI_event_overlap_AR_only_shared_IR_only.png")
    fig.subplots_adjust(left=0.15, right=0.98, top=0.96, bottom=0.14)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return pdf_path


dpsi_event_overlap_fig = plot_event_overlap_stacked_bar(ar_psi, ir_psi, figsize=(5, 4))
print(f"dPSI event overlap figure saved={dpsi_event_overlap_fig}")

# %%
###^^ (7) ATM gene expression and AR transcript usage ##########
def safe_paired_wilcoxon(pre_values, post_values):
    pre_values = pd.Series(pre_values, dtype="float64")
    post_values = pd.Series(post_values, dtype="float64")
    paired = pd.concat([pre_values, post_values], axis=1).dropna()
    if paired.shape[0] < 2:
        return np.nan
    if np.allclose(paired.iloc[:, 0].values, paired.iloc[:, 1].values):
        return 1.0
    try:
        return stats.wilcoxon(paired.iloc[:, 0], paired.iloc[:, 1]).pvalue
    except ValueError:
        return np.nan


def pvalue_to_star(p_value):
    if pd.isna(p_value) or p_value >= 0.05:
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    return "*"


def prepare_atm_ar_tu_data(filtered_trans, sampleinfo, gene_name="STAT3", min_tu=0.05, min_sample_frac=0.2):
    atm_tu = filtered_trans.loc[filtered_trans.index.str.split("-", n=1).str[-1] == gene_name].copy()
    pre_tu = atm_tu.loc[:, atm_tu.columns.str.endswith("-bfD")].copy()
    post_tu = atm_tu.loc[:, atm_tu.columns.str.endswith("-atD")].copy()
    pre_tu.columns = pre_tu.columns.str.replace("-bfD", "", regex=False)
    post_tu.columns = post_tu.columns.str.replace("-atD", "", regex=False)

    ar_samples = sorted(sampleinfo.loc[sampleinfo["response"] == 1].index.tolist())
    paired_samples = [sample for sample in ar_samples if sample in pre_tu.columns and sample in post_tu.columns]
    pre_tu = pre_tu.loc[:, paired_samples].fillna(0)
    post_tu = post_tu.loc[:, paired_samples].fillna(0)

    expressed_n = (pd.concat([pre_tu, post_tu], axis=1) >= min_tu).sum(axis=1)
    min_expressed_n = int(np.ceil(len(paired_samples) * 2 * min_sample_frac))
    keep_transcripts = expressed_n.loc[expressed_n >= min_expressed_n].index.tolist()

    pre_tu = pre_tu.loc[keep_transcripts]
    post_tu = post_tu.loc[keep_transcripts]

    stats_records = []
    for transcript in keep_transcripts:
        pre_values = pre_tu.loc[transcript, paired_samples]
        post_values = post_tu.loc[transcript, paired_samples]
        stats_records.append(
            {
                "Transcript-Gene": transcript,
                "transcript_id": transcript.split("-", 1)[0],
                "gene": gene_name,
                "mean_pre_TU": pre_values.mean(),
                "mean_post_TU": post_values.mean(),
                "sem_pre_TU": pre_values.sem(),
                "sem_post_TU": post_values.sem(),
                "wilcoxon_p": safe_paired_wilcoxon(pre_values, post_values),
                "detected_AR_timepoints": int(expressed_n.loc[transcript]),
            }
        )

    transcript_stats = pd.DataFrame(stats_records)
    if not transcript_stats.empty:
        transcript_stats = transcript_stats.sort_values(
            ["mean_pre_TU", "mean_post_TU"],
            ascending=False,
        ).reset_index(drop=True)
        transcript_stats["label"] = [f"t{i + 1}" for i in range(transcript_stats.shape[0])]
    return pre_tu, post_tu, paired_samples, transcript_stats


def prepare_atm_gene_expression(geneexp, sampleinfo, gene_name="ATM"):
    pre_gene_df, post_gene_df = prepare_prepost_gene_matrices(geneexp)
    ar_samples = sorted(sampleinfo.loc[sampleinfo["response"] == 1].index.tolist())
    paired_samples = [sample for sample in ar_samples if sample in pre_gene_df.columns and sample in post_gene_df.columns]
    pre_values = np.log2(pre_gene_df.loc[gene_name, paired_samples].astype(float) + 1)
    post_values = np.log2(post_gene_df.loc[gene_name, paired_samples].astype(float) + 1)
    long_df = pd.DataFrame(
        {
            "sample": paired_samples + paired_samples,
            "time": ["pre"] * len(paired_samples) + ["post"] * len(paired_samples),
            "expression": np.concatenate([pre_values.values, post_values.values]),
        }
    )
    return long_df, pre_values, post_values


def plot_atm_ar_expression_tu(geneexp, filtered_trans, sampleinfo, figsize_height=4):
    gene_name = "RAD51AP1"
    ar_pre_color = "#FDD49E"
    ar_post_color = "#F28E2B"

    gene_long_df, gene_pre, gene_post = prepare_atm_gene_expression(geneexp, sampleinfo, gene_name=gene_name)
    pre_tu, post_tu, paired_samples, transcript_stats = prepare_atm_ar_tu_data(
        filtered_trans,
        sampleinfo,
        gene_name=gene_name,
        min_tu=0.05,
        min_sample_frac=0.2,
    )

    n_transcripts = max(transcript_stats.shape[0], 1)
    fig_width = max(10.0, 3.4 + n_transcripts * 0.78)
    fig = plt.figure(figsize=(fig_width, figsize_height))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, max(3.4, n_transcripts * 0.52)], wspace=0.20)
    ax_gene = fig.add_subplot(gs[0, 0])
    ax_tu = fig.add_subplot(gs[0, 1])

    sns.boxplot(
        data=gene_long_df,
        x="time",
        y="expression",
        order=["pre", "post"],
        palette={"pre": ar_pre_color, "post": ar_post_color},
        width=0.7,
        fliersize=0,
        linewidth=1.0,
        ax=ax_gene,
    )
    ax_gene.set_xlim(-0.55, 1.55)
    gene_y = gene_long_df["expression"].max()
    gene_pad = max(gene_long_df["expression"].max() - gene_long_df["expression"].min(), 0.5) * 0.12
    ax_gene.plot([0, 0, 1, 1], [gene_y + gene_pad * 0.35, gene_y + gene_pad * 0.7, gene_y + gene_pad * 0.7, gene_y + gene_pad * 0.35], color="#303030", linewidth=0.8)
    ax_gene.text(0.5, gene_y + gene_pad * 0.78, "n.s.", ha="center", va="bottom", fontsize=9)
    ax_gene.set_ylim(gene_long_df["expression"].min() - gene_pad, gene_y + gene_pad * 1.45)
    ax_gene.set_xticklabels(["pre", "post"])
    ax_gene.set_xlabel("")
    ax_gene.set_ylabel("log2(TPM + 1)")
    ax_gene.set_title(gene_name, fontsize=12, fontweight="bold", pad=8)
    sns.despine(ax=ax_gene)

    if transcript_stats.empty:
        ax_tu.axis("off")
        ax_tu.text(0.5, 0.5, "No ATM transcript passed filter", ha="center", va="center", fontsize=10)
    else:
        x_pos = np.arange(transcript_stats.shape[0])
        bar_width = 0.34
        ax_tu.bar(
            x_pos - bar_width / 2,
            transcript_stats["mean_pre_TU"],
            yerr=transcript_stats["sem_pre_TU"],
            width=bar_width,
            color=ar_pre_color,
            edgecolor="#404040",
            linewidth=0.45,
            capsize=2.0,
            error_kw={"ecolor": "#303030", "elinewidth": 0.8, "capthick": 0.8},
            label="pre",
        )
        ax_tu.bar(
            x_pos + bar_width / 2,
            transcript_stats["mean_post_TU"],
            yerr=transcript_stats["sem_post_TU"],
            width=bar_width,
            color=ar_post_color,
            edgecolor="#404040",
            linewidth=0.45,
            capsize=2.0,
            error_kw={"ecolor": "#303030", "elinewidth": 0.8, "capthick": 0.8},
            label="post",
        )

        ymax = max(
            (transcript_stats["mean_pre_TU"] + transcript_stats["sem_pre_TU"].fillna(0)).max(),
            (transcript_stats["mean_post_TU"] + transcript_stats["sem_post_TU"].fillna(0)).max(),
            0.05,
        )
        for idx, row in transcript_stats.iterrows():
            star = pvalue_to_star(row["wilcoxon_p"])
            if star:
                y_star = max(
                    row["mean_pre_TU"] + (row["sem_pre_TU"] if pd.notna(row["sem_pre_TU"]) else 0),
                    row["mean_post_TU"] + (row["sem_post_TU"] if pd.notna(row["sem_post_TU"]) else 0),
                ) + ymax * 0.07
                ax_tu.text(idx, y_star, star, ha="center", va="bottom", fontsize=13, fontweight="bold")

        ax_tu.set_xticks(x_pos)
        ax_tu.set_xticklabels(transcript_stats["label"], rotation=0)
        ax_tu.set_ylabel("TU")
        ax_tu.set_xlabel("")
        ax_tu.set_ylim(0, min(1.05, ymax * 1.28))
        ax_tu.grid(axis="y", alpha=0.16)
        ax_tu.legend(frameon=False, loc="upper right")
        sns.despine(ax=ax_tu)

    os.makedirs(OUT_DIR, exist_ok=True)
    mapping_path = os.path.join(OUT_DIR, "RAD51AP1_AR_transcript_TU_label_mapping.tsv")
    transcript_stats.to_csv(mapping_path, sep="\t", index=False)

    pdf_path = os.path.join(OUT_DIR, "RAD51AP1_AR_gene_expression_transcript_TU_prepost.pdf")
    png_path = os.path.join(OUT_DIR, "RAD51AP1_AR_gene_expression_transcript_TU_prepost.png")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.17)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return pdf_path


rad51ap1_ar_tu_fig = plot_atm_ar_expression_tu(geneexp, filtered_trans, sampleinfo, figsize_height=4)
print(f"RAD51AP1 AR transcript usage figure saved={rad51ap1_ar_tu_fig}")

# %%
###^^ (8) AR/IR DUT and DUT gene counts ##########
def count_unique_genes_from_transcript_gene(dut_list):
    return (
        pd.Series(dut_list, dtype="object")
        .dropna()
        .astype(str)
        .str.split("-", n=1)
        .str[-1]
        .nunique()
    )


def plot_ar_ir_count_bar(count_df, y_col, ylabel, save_stem, figsize=(2.5, 4), order=None, palette=None):
    if order is None:
        order = ["AR", "IR"]
    if palette is None:
        palette = {"AR": "#FF952C", "IR": "#5AB862"}
    y_max = 4500
    if count_df[y_col].max() >= y_max:
        y_max = int(np.ceil(count_df[y_col].max() / 500) * 500) + 500
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=count_df,
        x="Group",
        y=y_col,
        order=order,
        palette=palette,
        edgecolor=None,
        linewidth=0.6,
        ax=ax,
        width=0.6
    )

    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, y_max)
    ax.set_yticks(np.arange(0, y_max + 1, 1000))
    ax.grid(axis="y", alpha=0.16)
    sns.despine(ax=ax)

    os.makedirs(OUT_DIR, exist_ok=True)
    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return pdf_path


dut_count_df = pd.DataFrame(
    {
        "Group": ["AR", "IR"],
        "DUT_count": [len(ARdutlist), len(IRdutlist)],
        "DUT_gene_count": [
            count_unique_genes_from_transcript_gene(ARdutlist),
            count_unique_genes_from_transcript_gene(IRdutlist),
        ],
    }
)

os.makedirs(OUT_DIR, exist_ok=True)
dut_count_df.to_csv(os.path.join(OUT_DIR, "AR_IR_DUT_and_DUT_gene_counts.tsv"), sep="\t", index=False)

dut_count_fig = plot_ar_ir_count_bar(
    dut_count_df,
    y_col="DUT_count",
    ylabel="Number of DUTs",
    save_stem="AR_IR_DUT_count_barplot",
    figsize=(2.5, 4),
)
dut_gene_count_fig = plot_ar_ir_count_bar(
    dut_count_df,
    y_col="DUT_gene_count",
    ylabel="Number of DUT genes",
    save_stem="AR_IR_DUT_gene_count_barplot",      
    figsize=(2.5, 4),
)

print(f"DUT count figure saved={dut_count_fig}")
print(f"DUT gene count figure saved={dut_gene_count_fig}")

# %%
###^^ (8-1) Class1/Class3 AR/IR DUT mean TU and volcano ##########
CLASS13_PANEL_SPECS = [
    ("PRT", class1),
    ("PST", class3),
]
DUT_GROUP_SPECS = {
    "AR": {
        "dut_df": AR_dut,
        "dut_list": ARdutlist,
        "samples": ar_samples,
        "colors": ["#FDD49E", "#F28E2B"],
    },
    "IR": {
        "dut_df": IR_dut,
        "dut_list": IRdutlist,
        "samples": ir_samples,
        "colors": ["#C7E9C0", "#5AAE61"],
    },
}


def extract_class13_sample_id(sample_col, pre_label="bfD", post_label="atD"):
    sample_col = str(sample_col)
    return sample_col.replace(f"-{pre_label}", "").replace(f"-{post_label}", "")


def compute_class_dut_sample_mean_tu(
    filtered_trans,
    transcript_gene_list,
    sample_list,
    pre_label="bfD",
    post_label="atD",
):
    tx_list = sorted(set(transcript_gene_list).intersection(filtered_trans.index))
    sample_set = {str(sample) for sample in sample_list}
    pre_cols = [
        col for col in filtered_trans.columns
        if str(col).endswith(f"-{pre_label}") and extract_class13_sample_id(col, pre_label, post_label) in sample_set
    ]
    post_cols = [
        col for col in filtered_trans.columns
        if str(col).endswith(f"-{post_label}") and extract_class13_sample_id(col, pre_label, post_label) in sample_set
    ]

    if len(tx_list) == 0 or len(pre_cols) == 0 or len(post_cols) == 0:
        return pd.DataFrame(columns=["Sample", "Pre", "Post"])

    tu_df = filtered_trans.loc[tx_list, pre_cols + post_cols].copy()
    pre_mean = tu_df[pre_cols].mean(axis=0, skipna=True)
    post_mean = tu_df[post_cols].mean(axis=0, skipna=True)

    pre_df = pd.DataFrame({"SampleCol": pre_mean.index, "Pre": pre_mean.values})
    post_df = pd.DataFrame({"SampleCol": post_mean.index, "Post": post_mean.values})
    pre_df["Sample"] = pre_df["SampleCol"].map(lambda x: extract_class13_sample_id(x, pre_label, post_label))
    post_df["Sample"] = post_df["SampleCol"].map(lambda x: extract_class13_sample_id(x, pre_label, post_label))
    paired_df = pd.merge(
        pre_df[["Sample", "Pre"]],
        post_df[["Sample", "Post"]],
        on="Sample",
        how="inner",
    ).dropna(subset=["Pre", "Post"])
    return paired_df


def paired_mean_tu_to_long_df(paired_df, row_label, group_label):
    if paired_df.empty:
        return pd.DataFrame(columns=["Sample", "ClassLabel", "Group", "Time", "MeanTU"])
    return paired_df.melt(
        id_vars="Sample",
        value_vars=["Pre", "Post"],
        var_name="Time",
        value_name="MeanTU",
    ).assign(ClassLabel=row_label, Group=group_label)


def add_wilcoxon_annotator(ax, plot_df, x_col, y_col, pairs, order=None, hue_col=None, hue_order=None, loc="inside"):
    if plot_df.empty or len(pairs) == 0:
        return

    annotator_kwargs = {
        "data": plot_df,
        "x": x_col,
        "y": y_col,
        "order": order,
    }
    if hue_col is not None:
        annotator_kwargs["hue"] = hue_col
        annotator_kwargs["hue_order"] = hue_order

    annotator = Annotator(ax, pairs, **annotator_kwargs)
    annotator.configure(
        test="Wilcoxon",
        text_format="star",
        loc=loc,
        verbose=0,
        line_height=0.02,
        line_width=1.0,
        text_offset=0,
    )
    annotator.apply_and_annotate()


def plot_class13_ar_ir_dut_mean_tu_boxplot(filtered_trans, figsize=(6, 6)):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    long_records = []
    summary_records = []
    panel_data = {}

    for row_idx, (row_label, class_tx) in enumerate(CLASS13_PANEL_SPECS):
        row_values = []
        for col_idx, group_label in enumerate(["AR", "IR"]):
            group_spec = DUT_GROUP_SPECS[group_label]
            class_dut_list = sorted(set(group_spec["dut_list"]).intersection(class_tx))
            paired_df = compute_class_dut_sample_mean_tu(
                filtered_trans=filtered_trans,
                transcript_gene_list=class_dut_list,
                sample_list=group_spec["samples"],
            )
            panel_data[(row_idx, col_idx)] = (class_dut_list, paired_df)
            if not paired_df.empty:
                row_values.extend(paired_df["Pre"].tolist())
                row_values.extend(paired_df["Post"].tolist())
                long_records.append(paired_mean_tu_to_long_df(paired_df, row_label, group_label))

            p_value = np.nan
            if paired_df.shape[0] >= 2 and not np.allclose(paired_df["Pre"].values, paired_df["Post"].values):
                try:
                    p_value = stats.wilcoxon(paired_df["Pre"], paired_df["Post"]).pvalue
                except ValueError:
                    p_value = np.nan
            elif paired_df.shape[0] >= 2:
                p_value = 1.0
            summary_records.append(
                {
                    "ClassLabel": row_label,
                    "Group": group_label,
                    "DUT_transcript_count": len(class_dut_list),
                    "Sample_count": paired_df.shape[0],
                    "Mean_Pre": paired_df["Pre"].mean() if not paired_df.empty else np.nan,
                    "Mean_Post": paired_df["Post"].mean() if not paired_df.empty else np.nan,
                    "Wilcoxon_p": p_value,
                }
            )

        if len(row_values) == 0:
            y_min, y_max = 0, 1
        else:
            y_min = float(np.nanmin(row_values))
            y_max = float(np.nanmax(row_values))
            y_pad = 0.05 if np.isclose(y_min, y_max) else (y_max - y_min) * 0.18
            y_min -= y_pad
            y_max += y_pad

        for col_idx, group_label in enumerate(["AR", "IR"]):
            ax = axes[row_idx, col_idx]
            group_spec = DUT_GROUP_SPECS[group_label]
            class_dut_list, paired_df = panel_data[(row_idx, col_idx)]

            if paired_df.empty:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=11)
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["Pre", "Post"])
                ax.set_ylim(y_min, y_max)
                ax.set_ylabel("Mean TU")
                if col_idx == 1:
                    ax.set_ylabel("")
                sns.despine(ax=ax)
                continue

            plot_df = paired_mean_tu_to_long_df(paired_df, row_label, group_label)
            sns.boxplot(
                data=plot_df,
                x="Time",
                y="MeanTU",
                order=["Pre", "Post"],
                palette=group_spec["colors"],
                width=0.70,
                showfliers=False,
                linewidth=1.1,
                ax=ax,
            )
            for _, sample_row in paired_df.iterrows():
                ax.plot(
                    [0, 1],
                    [sample_row["Pre"], sample_row["Post"]],
                    color="#8C8C8C",
                    alpha=0.65,
                    linewidth=0.9,
                    marker="o",
                    markersize=2.4,
                    markerfacecolor="#8C8C8C",
                    markeredgewidth=0,
                    zorder=3,
                )

            ax.set_ylim(y_min, y_max)
            add_wilcoxon_annotator(
                ax=ax,
                plot_df=plot_df,
                x_col="Time",
                y_col="MeanTU",
                pairs=[("Pre", "Post")] if paired_df.shape[0] >= 2 else [],
                order=["Pre", "Post"],
                loc="inside",
            )
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("")
            ax.set_ylabel("Mean TU")
            if col_idx == 1:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)
            if row_idx == 0:
                ax.set_yticks([0.2, 0.4, 0.6])
            ax.grid(axis="y", alpha=0.16)
            sns.despine(ax=ax, trim=False)

    fig.subplots_adjust(left=0.12, right=0.98, top=0.97, bottom=0.08, hspace=0.16, wspace=0.12)

    os.makedirs(OUT_DIR, exist_ok=True)
    long_df = pd.concat(long_records, ignore_index=True) if long_records else pd.DataFrame()
    summary_df = pd.DataFrame(summary_records)
    long_path = os.path.join(OUT_DIR, "Class13_AR_IR_DUT_meanTU_prepost_long.tsv")
    summary_path = os.path.join(OUT_DIR, "Class13_AR_IR_DUT_meanTU_prepost_summary.tsv")
    pdf_path = os.path.join(OUT_DIR, "Class13_AR_IR_DUT_meanTU_prepost_boxplot.pdf")
    png_path = os.path.join(OUT_DIR, "Class13_AR_IR_DUT_meanTU_prepost_boxplot.png")
    long_df.to_csv(long_path, sep="\t", index=False)
    summary_df.to_csv(summary_path, sep="\t", index=False)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return pdf_path


def build_class_dut_volcano_df(dut_df, class_tx, p_cutoff=0.05, delta_cutoff=0.05):
    work_df = dut_df.loc[dut_df.index.intersection(class_tx)].copy()
    work_df["Transcript-Gene"] = work_df.index.astype(str)
    work_df["p_value"] = pd.to_numeric(work_df["p_value"], errors="coerce")
    work_df["delta_TU"] = pd.to_numeric(work_df["delta_TU"], errors="coerce")
    work_df = work_df.dropna(subset=["p_value", "delta_TU"]).copy()
    work_df["neglog10_p"] = -np.log10(work_df["p_value"].clip(lower=np.nextafter(0, 1)))
    work_df["Regulation"] = "NS"
    work_df.loc[(work_df["p_value"] < p_cutoff) & (work_df["delta_TU"] < -delta_cutoff), "Regulation"] = "Down"
    work_df.loc[(work_df["p_value"] < p_cutoff) & (work_df["delta_TU"] > delta_cutoff), "Regulation"] = "Up"
    return work_df


def plot_class13_ar_ir_dut_volcano(figsize=(6, 6), p_cutoff=0.05, delta_cutoff=0.05):
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    volcano_records = []
    count_records = []
    color_map = {"NS": "#BDBDBD", "Down": "#4C78A8", "Up": "#E45756"}
    all_delta = []
    all_neglog = []
    panel_data = {}

    for row_idx, (row_label, class_tx) in enumerate(CLASS13_PANEL_SPECS):
        for col_idx, group_label in enumerate(["AR", "IR"]):
            group_spec = DUT_GROUP_SPECS[group_label]
            volcano_df = build_class_dut_volcano_df(
                dut_df=group_spec["dut_df"],
                class_tx=class_tx,
                p_cutoff=p_cutoff,
                delta_cutoff=delta_cutoff,
            )
            volcano_df["ClassLabel"] = row_label
            volcano_df["Group"] = group_label
            panel_data[(row_idx, col_idx)] = volcano_df
            volcano_records.append(volcano_df)
            all_delta.extend(volcano_df["delta_TU"].tolist())
            all_neglog.extend(volcano_df["neglog10_p"].tolist())
            count_records.append(
                {
                    "ClassLabel": row_label,
                    "Group": group_label,
                    "Downregulated_DUT_n": int((volcano_df["Regulation"] == "Down").sum()),
                    "Upregulated_DUT_n": int((volcano_df["Regulation"] == "Up").sum()),
                    "Tested_transcript_n": volcano_df.shape[0],
                }
            )

    x_abs = max(np.nanmax(np.abs(all_delta)) if all_delta else delta_cutoff * 2, delta_cutoff * 4)
    x_abs = float(np.ceil(x_abs / 0.05) * 0.05)
    y_max = 5

    for row_idx, (row_label, _) in enumerate(CLASS13_PANEL_SPECS):
        for col_idx, group_label in enumerate(["AR", "IR"]):
            ax = axes[row_idx, col_idx]
            volcano_df = panel_data[(row_idx, col_idx)]

            for regulation in ["NS", "Down", "Up"]:
                subset_df = volcano_df.loc[volcano_df["Regulation"] == regulation]
                ax.scatter(
                    subset_df["delta_TU"],
                    subset_df["neglog10_p"],
                    s=5,
                    color=color_map[regulation],
                    alpha=0.40 if regulation == "NS" else 0.78,
                    linewidths=0,
                )

            down_n = int((volcano_df["Regulation"] == "Down").sum())
            up_n = int((volcano_df["Regulation"] == "Up").sum())
            ax.text(0.08, 0.92, f"n={down_n}", transform=ax.transAxes, ha="left", va="top", color="black", fontsize=10, fontweight="bold")
            ax.text(0.92, 0.92, f"n={up_n}", transform=ax.transAxes, ha="right", va="top", color="black", fontsize=10, fontweight="bold")
            ax.axvline(-delta_cutoff, color="#707070", linestyle="--", linewidth=0.8, alpha=0.65)
            ax.axvline(delta_cutoff, color="#707070", linestyle="--", linewidth=0.8, alpha=0.65)
            ax.axhline(-np.log10(p_cutoff), color="#707070", linestyle="--", linewidth=0.8, alpha=0.65)
            ax.set_xlim(-x_abs, x_abs)
            ax.set_ylim(0, y_max)
            ax.set_xlabel("$\\Delta$ TU" if row_idx == 1 else "")
            ax.set_ylabel("$-log_{10}(p)$")
            ax.set_yticks(np.arange(0, y_max + 0.1, 1))
            if col_idx == 1:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)
            else:
                ax.tick_params(axis="y", labelleft=True)
            ax.tick_params(axis="x", labelbottom=True)
            ax.grid(alpha=0.12)
            sns.despine(ax=ax, trim=False)

    fig.subplots_adjust(left=0.12, right=0.98, top=0.97, bottom=0.08, hspace=0.16, wspace=0.12)

    os.makedirs(OUT_DIR, exist_ok=True)
    all_volcano_df = pd.concat(volcano_records, ignore_index=True) if volcano_records else pd.DataFrame()
    count_df = pd.DataFrame(count_records)
    volcano_path = os.path.join(OUT_DIR, "Class13_AR_IR_DUT_volcano_values.tsv")
    count_path = os.path.join(OUT_DIR, "Class13_AR_IR_DUT_volcano_counts.tsv")
    pdf_path = os.path.join(OUT_DIR, "Class13_AR_IR_DUT_volcano.pdf")
    png_path = os.path.join(OUT_DIR, "Class13_AR_IR_DUT_volcano.png")
    all_volcano_df.to_csv(volcano_path, sep="\t", index=False)
    count_df.to_csv(count_path, sep="\t", index=False)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return pdf_path


class13_mean_tu_fig = plot_class13_ar_ir_dut_mean_tu_boxplot(filtered_trans, figsize=(5, 5))
class13_volcano_fig = plot_class13_ar_ir_dut_volcano(figsize=(5, 5))
print(f"Class13 AR/IR DUT mean TU boxplot saved={class13_mean_tu_fig}")
print(f"Class13 AR/IR DUT volcano saved={class13_volcano_fig}")

#%% 
###^^ (9) BRCAmt/wt AR DUT comparison #####
BRCAmt_AR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/BRCAmt_AR_stable_DUT_Wilcoxon_delta_withna.txt', sep='\t', index_col=0)
BRCAwt_AR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/BRCAwt_AR_stable_DUT_Wilcoxon_delta_withna.txt', sep='\t', index_col=0)

BRCAmt_ARdutlist = BRCAmt_AR_dut.loc[
    (BRCAmt_AR_dut["p_value"] < 0.05) & (np.abs(BRCAmt_AR_dut["delta_TU"]) > 0.05)
].index.to_list()
BRCAwt_ARdutlist = BRCAwt_AR_dut.loc[
    (BRCAwt_AR_dut["p_value"] < 0.05) & (np.abs(BRCAwt_AR_dut["delta_TU"]) > 0.05)
].index.to_list()

brca_ar_dut_count_df = pd.DataFrame(
    {
        "Group": ["BRCAwt", "BRCAmt"],
        "DUT_count": [len(BRCAwt_ARdutlist), len(BRCAmt_ARdutlist)],
        "DUT_gene_count": [
            count_unique_genes_from_transcript_gene(BRCAwt_ARdutlist),
            count_unique_genes_from_transcript_gene(BRCAmt_ARdutlist),
        ],
    }
)

brca_order = ["BRCAwt", "BRCAmt"]
brca_palette = {"BRCAwt": "#E6E6E6", "BRCAmt": "#D73027"}
brca_ar_dut_count_df.to_csv(
    os.path.join(OUT_DIR, "BRCAwt_BRCAmt_AR_DUT_and_DUT_gene_counts.tsv"),
    sep="\t",
    index=False,
)

brca_dut_count_fig = plot_ar_ir_count_bar(
    brca_ar_dut_count_df,
    y_col="DUT_count",
    ylabel="Number of DUTs",
    save_stem="BRCAwt_BRCAmt_AR_DUT_count_barplot",
    figsize=(2.5, 4),
    order=brca_order,
    palette=brca_palette,
)
brca_dut_gene_count_fig = plot_ar_ir_count_bar(
    brca_ar_dut_count_df,
    y_col="DUT_gene_count",
    ylabel="Number of DUT genes",
    save_stem="BRCAwt_BRCAmt_AR_DUT_gene_count_barplot",
    figsize=(2.5, 4),
    order=brca_order,
    palette=brca_palette,
)

print(f"BRCAwt/BRCAmt AR DUT count figure saved={brca_dut_count_fig}")
print(f"BRCAwt/BRCAmt AR DUT gene count figure saved={brca_dut_gene_count_fig}")

#%%
###^^ (10) BRCAwt/mt AR DUT Class 1/2/3 mean TU boxplot and GO enrichment #####
def safe_paired_wilcoxon(pre_values, post_values):
    pre_values = pd.Series(pre_values, dtype="float64")
    post_values = pd.Series(post_values, dtype="float64")
    paired = pd.concat([pre_values, post_values], axis=1).dropna()
    if paired.shape[0] < 2:
        return np.nan
    if np.allclose(paired.iloc[:, 0].values, paired.iloc[:, 1].values):
        return 1.0
    try:
        return stats.wilcoxon(paired.iloc[:, 0], paired.iloc[:, 1]).pvalue
    except ValueError:
        return np.nan


def wrap_term_for_brca(term, width=74):
    words = str(term).split()
    lines = []
    current = ""
    for word in words:
        candidate = word if current == "" else f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def extract_sample_id(sample_col, pre_label="bfD", post_label="atD"):
    sample_col = str(sample_col)
    return (
        sample_col
        .replace(f"-{pre_label}", "")
        .replace(f"-{post_label}", "")
    )


def compute_paired_sample_meanTU(
    filtered_trans,
    transcript_list,
    sample_list,
    fillna=False,
    pre_label="bfD",
    post_label="atD",
):
    tx_list = sorted(set(transcript_list).intersection(filtered_trans.index))
    if len(tx_list) == 0:
        return pd.DataFrame(columns=["PairID", "Pre", "Post"])

    sample_list = [str(sample) for sample in sample_list]
    pre_cols = [
        col for col in filtered_trans.columns
        if str(col).endswith(f"-{pre_label}") and extract_sample_id(col, pre_label, post_label) in sample_list
    ]
    post_cols = [
        col for col in filtered_trans.columns
        if str(col).endswith(f"-{post_label}") and extract_sample_id(col, pre_label, post_label) in sample_list
    ]

    if len(pre_cols) == 0 or len(post_cols) == 0:
        return pd.DataFrame(columns=["PairID", "Pre", "Post"])

    tu_df = filtered_trans.loc[tx_list, pre_cols + post_cols].copy()
    if fillna:
        tu_df = tu_df.fillna(0)

    mean_pre = tu_df[pre_cols].mean(axis=0, skipna=True)
    mean_post = tu_df[post_cols].mean(axis=0, skipna=True)

    pre_df = pd.DataFrame({"SampleCol": mean_pre.index, "Pre": mean_pre.values})
    post_df = pd.DataFrame({"SampleCol": mean_post.index, "Post": mean_post.values})
    pre_df["PairID"] = pre_df["SampleCol"].map(lambda x: extract_sample_id(x, pre_label, post_label))
    post_df["PairID"] = post_df["SampleCol"].map(lambda x: extract_sample_id(x, pre_label, post_label))

    paired = pd.merge(
        pre_df[["PairID", "Pre"]],
        post_df[["PairID", "Post"]],
        on="PairID",
        how="inner",
    ).dropna(subset=["Pre", "Post"])
    return paired


def p_to_star_with_ns(p_value):
    if pd.isna(p_value):
        return "ns"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def add_pairwise_stat_annotation(ax, x1, x2, y, h, text, fontsize=11):
    ax.plot([x1, x1, x2, x2], [y + h, y + 2 * h, y + 2 * h, y + h], lw=1.2, c="black")
    ax.text(
        (x1 + x2) / 2,
        y + 2 * h,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color="black",
        fontweight="bold" if text != "ns" else "normal",
    )


def plot_brca_meanTU_box_grid_by_class(
    filtered_trans,
    brcawt_dutlist,
    brcamt_dutlist,
    class1,
    class2,
    class3,
    brcawt_samples,
    brcamt_samples,
    fillna=False,
    pre_label="bfD",
    post_label="atD",
    save_stem="BRCAwt_BRCAmt_AR_Class123_DUT_boxplot",
):
    class_groups = [class1, class2, class3]
    panel_specs = [
        ("BRCAwt", brcawt_dutlist, brcawt_samples, ["#F0F0F0", "#9A9A9A"]),
        ("BRCAmt", brcamt_dutlist, brcamt_samples, ["#F6B0AA", "#D73027"]),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(8, 10))
    summary_records = []

    for row_idx, class_tx in enumerate(class_groups):
        row_vals = []
        row_paired = {}

        for label, dutlist, sample_list, _ in panel_specs:
            tx_list = sorted(set(dutlist).intersection(class_tx))
            paired_df = compute_paired_sample_meanTU(
                filtered_trans=filtered_trans,
                transcript_list=tx_list,
                sample_list=sample_list,
                fillna=fillna,
                pre_label=pre_label,
                post_label=post_label,
            )
            row_paired[label] = (tx_list, paired_df)
            if not paired_df.empty:
                row_vals.extend(paired_df["Pre"].tolist())
                row_vals.extend(paired_df["Post"].tolist())

        if len(row_vals) == 0:
            y_min, y_max = 0, 1
        else:
            y_min = np.nanmin(row_vals)
            y_max = np.nanmax(row_vals)
            margin = 0.05 if np.isclose(y_min, y_max) else (y_max - y_min) * 0.08
            y_min -= margin
            y_max += margin

        for col_idx, (label, dutlist, sample_list, colors) in enumerate(panel_specs):
            ax = axes[row_idx, col_idx]
            tx_list, paired_df = row_paired[label]

            if paired_df.empty:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=12)
                ax.set_title(f"{label} AR DUT (Class {row_idx + 1})", fontsize=13, fontweight="bold")
                ax.set_ylim(y_min, y_max)
                ax.set_xlabel("")
                ax.set_ylabel("Mean TU", fontsize=13)
                sns.despine(ax=ax)
                continue

            long_df = paired_df.melt(
                id_vars="PairID",
                value_vars=["Pre", "Post"],
                var_name="Time",
                value_name="MeanTU",
            )

            for _, r in paired_df.iterrows():
                ax.plot(
                    [0, 1],
                    [r["Pre"], r["Post"]],
                    color="gray",
                    alpha=0.3,
                    linewidth=1,
                    zorder=1,
                )

            sns.boxplot(
                data=long_df,
                x="Time",
                y="MeanTU",
                order=["Pre", "Post"],
                palette=colors,
                width=0.8,
                fliersize=0,
                linewidth=1.4,
                ax=ax,
            )
            sns.stripplot(
                data=long_df,
                x="Time",
                y="MeanTU",
                order=["Pre", "Post"],
                color="gray",
                alpha=0.75,
                size=4,
                jitter=0.08,
                ax=ax,
                zorder=2,
            )

            pval = safe_paired_wilcoxon(paired_df["Pre"], paired_df["Post"])
            star = p_to_star_with_ns(pval)
            yr = y_max - y_min
            ymax_panel = np.nanmax(paired_df[["Pre", "Post"]].to_numpy())
            line_y = ymax_panel + yr * 0.015
            h = yr * 0.01
            add_pairwise_stat_annotation(ax, 0, 1, line_y, h, star, fontsize=11)

            ax.set_title(f"{label} AR DUT (Class {row_idx + 1})", fontsize=13, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("Mean TU", fontsize=13)
            ax.set_ylim(y_min, y_max + yr * 0.08)
            sns.despine(ax=ax)

            summary_records.append(
                {
                    "Group": label,
                    "Class": f"Class{row_idx + 1}",
                    "DUT_transcript_count": len(tx_list),
                    "Sample_count": paired_df.shape[0],
                    "Mean_Pre": paired_df["Pre"].mean(),
                    "Mean_Post": paired_df["Post"].mean(),
                    "Wilcoxon_p": pval,
                }
            )

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    summary_df = pd.DataFrame(summary_records)
    summary_path = os.path.join(OUT_DIR, f"{save_stem}_summary.tsv")
    summary_df.to_csv(summary_path, sep="\t", index=False)
    return pdf_path


brca_numeric = pd.to_numeric(sampleinfo["BRCAmut"], errors="coerce")
response_numeric = pd.to_numeric(sampleinfo["response"], errors="coerce")
BRCAwt_AR_samples = sorted(
    sampleinfo.loc[(response_numeric == 1) & (brca_numeric == 0)].index.intersection(preTU.columns).tolist()
)
BRCAmt_AR_samples = sorted(
    sampleinfo.loc[(response_numeric == 1) & (brca_numeric == 1)].index.intersection(preTU.columns).tolist()
)

print("\n===== BRCAwt/BRCAmt AR samples used for Class123 DUT boxplot =====")
print(f"BRCAwt AR samples: {len(BRCAwt_AR_samples)}")
print(f"BRCAmt AR samples: {len(BRCAmt_AR_samples)}")

brca_class_dut_count_records = []
for group_label, dutlist in [
    ("BRCAwt", BRCAwt_ARdutlist),
    ("BRCAmt", BRCAmt_ARdutlist),
]:
    brca_class_dut_count_records.append(
        {
            "Group": group_label,
            "Class": "All",
            "DUT_count": len(dutlist),
            "DUT_gene_count": count_unique_genes_from_transcript_gene(dutlist),
        }
    )
    for class_label, class_tx in [
        ("Class1", class1),
        ("Class2", class2),
        ("Class3", class3),
    ]:
        class_dutlist = sorted(set(dutlist).intersection(class_tx))
        brca_class_dut_count_records.append(
            {
                "Group": group_label,
                "Class": class_label,
                "DUT_count": len(class_dutlist),
                "DUT_gene_count": count_unique_genes_from_transcript_gene(class_dutlist),
            }
        )

brca_class_dut_count_df = pd.DataFrame(brca_class_dut_count_records)
brca_class_dut_count_df.to_csv(
    os.path.join(OUT_DIR, "BRCAwt_BRCAmt_AR_Class123_DUT_counts.tsv"),
    sep="\t",
    index=False,
)
print("\n===== BRCAwt/BRCAmt AR Class123 DUT counts =====")
print(brca_class_dut_count_df.to_string(index=False))

brca_class_boxplot_fig = plot_brca_meanTU_box_grid_by_class(
    filtered_trans=filtered_trans,
    brcawt_dutlist=BRCAwt_ARdutlist,
    brcamt_dutlist=BRCAmt_ARdutlist,
    class1=class1,
    class2=class2,
    class3=class3,
    brcawt_samples=BRCAwt_AR_samples,
    brcamt_samples=BRCAmt_AR_samples,
    fillna=False,
    pre_label="bfD",
    post_label="atD",
    save_stem="BRCAwt_BRCAmt_AR_Class123_DUT_boxplot",
)
print(f"BRCAwt/BRCAmt AR Class123 DUT boxplot saved={brca_class_boxplot_fig}")


def dut_gene_list_from_transcripts(transcript_list):
    return sorted(
        pd.Series(transcript_list, dtype="object")
        .dropna()
        .astype(str)
        .str.split("-", n=1)
        .str[-1]
        .drop_duplicates()
        .tolist()
    )


def run_dut_top5_enrichment(gene_list, label, gene_sets=None, fdr_cutoff=None):
    if gene_sets is None:
        gene_sets = ["GO_Biological_Process_2021", "Reactome_2022"]

    result_path = os.path.join(OUT_DIR, f"{label}_GOBP_Reactome_enrichr.tsv")
    if len(gene_list) == 0:
        empty_df = pd.DataFrame()
        empty_df.to_csv(result_path, sep="\t", index=False)
        print(f"{label}: no input genes for GO enrichment")
        return empty_df

    try:
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=gene_sets,
            organism="human",
            outdir=None,
        )
        enr_df = enr.results.sort_values("Adjusted P-value").copy()
        if fdr_cutoff is not None:
            enr_df["Adjusted P-value"] = pd.to_numeric(enr_df["Adjusted P-value"], errors="coerce")
            enr_df = enr_df.loc[enr_df["Adjusted P-value"] < fdr_cutoff].copy()
    except Exception as exc:
        print(f"{label}: GO enrichment failed: {exc}")
        enr_df = pd.DataFrame()

    enr_df.to_csv(result_path, sep="\t", index=False)
    return enr_df


def clean_enrichr_term(term):
    return (
        str(term)
        .replace("_", " ")
    )


def plot_dut_top5_enrichment_bar(enrichr_df, gene_count, label, title, color, top_n=5, fdr_cutoff=None):
    pdf_path = os.path.join(OUT_DIR, f"{label}_top5_GOBP_Reactome_barplot.pdf")
    png_path = os.path.join(OUT_DIR, f"{label}_top5_GOBP_Reactome_barplot.png")

    if enrichr_df.empty:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.axis("off")
        if fdr_cutoff is None:
            message = f"No enrichment result\nInput genes: {gene_count}"
        else:
            message = f"No enriched terms with FDR < {fdr_cutoff}\nInput genes: {gene_count}"
        ax.text(0.02, 0.52, message, ha="left", va="center", fontsize=10)
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        return pdf_path

    plot_df = enrichr_df.copy()
    plot_df["FDR"] = pd.to_numeric(plot_df["Adjusted P-value"], errors="coerce")
    if fdr_cutoff is not None:
        plot_df = plot_df.loc[plot_df["FDR"] < fdr_cutoff].copy()
    plot_df = plot_df.dropna(subset=["FDR"]).sort_values("FDR").head(top_n).copy()
    if plot_df.empty:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.axis("off")
        if fdr_cutoff is None:
            message = f"No valid enrichment terms\nInput genes: {gene_count}"
        else:
            message = f"No enriched terms with FDR < {fdr_cutoff}\nInput genes: {gene_count}"
        ax.text(0.02, 0.52, message, ha="left", va="center", fontsize=10)
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        return pdf_path

    plot_df["Term_clean"] = (
        plot_df["Term"]
        .astype(str)
        .str.replace(r"\s+R-HSA-\d+.*$", "", regex=True)
        .str.replace(r"\s+GO:\d+.*$", "", regex=True)
        .map(clean_enrichr_term)
    )
    if "Gene_set" in plot_df.columns:
        plot_df["Gene_set_short"] = (
            plot_df["Gene_set"]
            .astype(str)
            .replace({"GO_Biological_Process_2021": "GO BP", "Reactome_2022": "Reactome"})
        )
        plot_df["Term_display"] = plot_df["Term_clean"] + " [" + plot_df["Gene_set_short"] + "]"
    else:
        plot_df["Term_display"] = plot_df["Term_clean"]
    plot_df["neg_log10_FDR"] = -np.log10(plot_df["FDR"].clip(lower=np.nextafter(0, 1)))

    y_pos = np.arange(plot_df.shape[0])
    fig, (ax_bar, ax_text) = plt.subplots(
        1,
        2,
        figsize=(7, 3.6),
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 4.9], "wspace": 0.025},
    )

    ax_bar.barh(y_pos, plot_df["neg_log10_FDR"], color=color, alpha=0.9)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([])
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("-log10(FDR)", fontsize=10)
    ax_bar.grid(axis="x", alpha=0.2)
    ax_bar.tick_params(axis="y", length=0)
    sns.despine(ax=ax_bar, left=True)

    ax_text.set_xlim(0, 1)
    ax_text.set_ylim(ax_bar.get_ylim())
    ax_text.axis("off")
    for y, (_, row) in zip(y_pos, plot_df.iterrows()):
        ax_text.text(
            0.0,
            y,
            wrap_term_for_brca(row["Term_display"], width=74),
            ha="left",
            va="center",
            fontsize=10,
        )

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.99)
    fig.subplots_adjust(left=0.045, right=0.995, top=0.88, bottom=0.15)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return pdf_path


brca_go_specs = [
    (
        "BRCAwt_AR_upregulated_Class1_DUT",
        "BRCAwt AR Upregulated Class 1 DUT",
        sorted(set(BRCAwt_AR_dut.loc[(BRCAwt_AR_dut["p_value"] < 0.05) & (BRCAwt_AR_dut["delta_TU"] > 0.05)].index).intersection(class1)),
        "#9A9A9A",
    ),
    (
        "BRCAwt_AR_downregulated_Class3_DUT",
        "BRCAwt AR Downregulated Class 3 DUT",
        sorted(set(BRCAwt_AR_dut.loc[(BRCAwt_AR_dut["p_value"] < 0.05) & (BRCAwt_AR_dut["delta_TU"] < -0.05)].index).intersection(class3)),
        "#5F5F5F",
    ),
    (
        "BRCAmt_AR_upregulated_Class1_DUT",
        "BRCAmt AR Upregulated Class 1 DUT",
        sorted(set(BRCAmt_AR_dut.loc[(BRCAmt_AR_dut["p_value"] < 0.05) & (BRCAmt_AR_dut["delta_TU"] > 0.05)].index).intersection(class1)),
        "#D73027",
    ),
    (
        "BRCAmt_AR_downregulated_Class3_DUT",
        "BRCAmt AR Downregulated Class 3 DUT",
        sorted(set(BRCAmt_AR_dut.loc[(BRCAmt_AR_dut["p_value"] < 0.05) & (BRCAmt_AR_dut["delta_TU"] < -0.05)].index).intersection(class3)),
        "#7F1D1D",
    ),
]

brca_go_summary = []
for save_label, title, transcript_list, color in brca_go_specs:
    genes = dut_gene_list_from_transcripts(transcript_list)
    gene_input_path = os.path.join(OUT_DIR, f"{save_label}_GO_input_genes.txt")
    pd.Series(genes, name="gene").to_csv(gene_input_path, sep="\t", index=False)

    enr_df = run_dut_top5_enrichment(genes, save_label)
    fig_path = plot_dut_top5_enrichment_bar(
        enr_df,
        gene_count=len(genes),
        label=save_label,
        title=title,
        color=color,
        top_n=5,
    )
    brca_go_summary.append(
        {
            "Label": save_label,
            "Transcript_count": len(transcript_list),
            "Gene_count": len(genes),
            "Figure": fig_path,
        }
    )
    print(f"{title}: transcripts={len(transcript_list)}, genes={len(genes)}, saved={fig_path}")

pd.DataFrame(brca_go_summary).to_csv(
    os.path.join(OUT_DIR, "BRCAwt_BRCAmt_AR_Class1up_Class3down_GO_summary.tsv"),
    sep="\t",
    index=False,
)

# %%
#########^^^^ (11) ARDUT to BRCAwt/BRCAmt pre samples ###################
def parse_treatment_line_num(line_value):
    match = re.search(r"(\d+)", str(line_value))
    if match is None:
        return np.nan
    return float(match.group(1))


def get_pre_sample_mean_tu(filtered_trans, transcript_list, samples, pre_label="bfD"):
    tx_list = sorted(set(transcript_list).intersection(filtered_trans.index))
    records = []
    for sample in samples:
        pre_col = f"{sample}-{pre_label}"
        if pre_col not in filtered_trans.columns:
            continue
        mean_tu = filtered_trans.loc[tx_list, pre_col].mean(skipna=True) if len(tx_list) > 0 else np.nan
        if pd.notna(mean_tu):
            records.append({"Sample": sample, "MeanTU": mean_tu})
    return pd.DataFrame(records)


def mannwhitney_pvalue_if_possible(group_a, group_b):
    group_a = pd.Series(group_a, dtype="float64").dropna()
    group_b = pd.Series(group_b, dtype="float64").dropna()
    if len(group_a) < 2 or len(group_b) < 2:
        return np.nan
    try:
        return stats.mannwhitneyu(group_a, group_b, alternative="two-sided").pvalue
    except ValueError:
        return np.nan


def format_pvalue_label(p_value):
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return f"p={p_value:.1e}"
    return f"p={p_value:.3f}"


def add_ar_ir_pre_pvalue_annotation(ax, x_center, values_a, values_b, p_value):
    if pd.isna(p_value):
        return
    values = pd.concat(
        [pd.Series(values_a, dtype="float64"), pd.Series(values_b, dtype="float64")],
        ignore_index=True,
    ).dropna()
    if values.empty:
        return

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    y = values.max() + y_range * 0.035
    h = y_range * 0.018
    x1 = x_center - 0.20
    x2 = x_center + 0.20
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", lw=1.0)
    ax.text(
        x_center,
        y + h * 1.15,
        format_pvalue_label(p_value),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    if y + h * 3 > y_max:
        ax.set_ylim(y_min, y + h * 3)


def build_ar_ir_pre_mean_tu_df(dut_specs, comparison_specs):
    all_records = []
    pvalue_records = []
    sample_count_records = []

    for subset_label, subset_samples in comparison_specs:
        subset_samples = set(subset_samples)
        subset_ar_samples = sorted(set(ar_samples).intersection(subset_samples))
        subset_ir_samples = sorted(set(ir_samples).intersection(subset_samples))
        sample_count_records.append(
            {
                "Subset": subset_label,
                "AR_pre_sample_count": len(subset_ar_samples),
                "IR_pre_sample_count": len(subset_ir_samples),
            }
        )

        for dut_label, transcript_list in dut_specs:
            ar_df = get_pre_sample_mean_tu(filtered_trans, transcript_list, subset_ar_samples)
            ir_df = get_pre_sample_mean_tu(filtered_trans, transcript_list, subset_ir_samples)

            if not ar_df.empty:
                ar_df["Condition"] = "AR pre"
                ar_df["DUT_set"] = dut_label
                ar_df["Subset"] = subset_label
                all_records.extend(ar_df.to_dict("records"))
            if not ir_df.empty:
                ir_df["Condition"] = "IR pre"
                ir_df["DUT_set"] = dut_label
                ir_df["Subset"] = subset_label
                all_records.extend(ir_df.to_dict("records"))

            p_value = mannwhitney_pvalue_if_possible(ar_df["MeanTU"], ir_df["MeanTU"])
            pvalue_records.append(
                {
                    "Subset": subset_label,
                    "DUT_set": dut_label,
                    "AR_pre_sample_count": ar_df.shape[0],
                    "IR_pre_sample_count": ir_df.shape[0],
                    "MannWhitney_p": p_value,
                }
            )

    return (
        pd.DataFrame(all_records),
        pd.DataFrame(pvalue_records),
        pd.DataFrame(sample_count_records),
    )


def plot_ar_ir_pre_dut_mean_tu_by_subset(
    mean_df,
    pvalue_df,
    save_stem,
    subset_order=None,
    suptitle="AR pre vs IR pre using AR-derived DUT sets",
    figsize=None,
):
    if subset_order is None:
        subset_order = ["All", "BRCAwt & >=2L", "BRCAmt or 1L"]
    dut_order = ["Up Class1", "Down Class3"]
    condition_order = ["AR pre", "IR pre"]
    condition_palette = {"AR pre": "#FDD49E", "IR pre": "#C7E9C0"}

    if figsize is None:
        figsize = (4 * len(subset_order), 4.2)
    fig, axes = plt.subplots(1, len(subset_order), figsize=figsize, sharey=False)
    axes = np.atleast_1d(axes)
    for ax, subset_label in zip(axes, subset_order):
        plot_df = mean_df.loc[mean_df["Subset"] == subset_label].copy()
        if plot_df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
            ax.set_title(subset_label, fontsize=12, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("Mean TU")
            sns.despine(ax=ax)
            continue

        sns.boxplot(
            data=plot_df,
            x="DUT_set",
            y="MeanTU",
            hue="Condition",
            order=dut_order,
            hue_order=condition_order,
            palette=condition_palette,
            showfliers=False,
            width=0.72,
            linewidth=1.2,
            ax=ax,
        )
        sns.stripplot(
            data=plot_df,
            x="DUT_set",
            y="MeanTU",
            hue="Condition",
            order=dut_order,
            hue_order=condition_order,
            dodge=True,
            palette={"AR pre": "#8C8C8C", "IR pre": "#8C8C8C"},
            alpha=0.72,
            size=3.5,
            jitter=0.08,
            ax=ax,
        )

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

        for x_idx, dut_label in enumerate(dut_order):
            p_row = pvalue_df.loc[
                (pvalue_df["Subset"] == subset_label) & (pvalue_df["DUT_set"] == dut_label)
            ]
            if p_row.empty:
                continue
            ar_values = plot_df.loc[
                (plot_df["DUT_set"] == dut_label) & (plot_df["Condition"] == "AR pre"),
                "MeanTU",
            ]
            ir_values = plot_df.loc[
                (plot_df["DUT_set"] == dut_label) & (plot_df["Condition"] == "IR pre"),
                "MeanTU",
            ]
            add_ar_ir_pre_pvalue_annotation(
                ax,
                x_center=x_idx,
                values_a=ar_values,
                values_b=ir_values,
                p_value=p_row["MannWhitney_p"].iloc[0],
            )

        ax.set_title(subset_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Mean TU")
        ax.set_xticks(range(len(dut_order)))
        ax.set_xticklabels(["AR up\nClass1", "AR down\nClass3"])
        ax.grid(axis="y", alpha=0.16)
        sns.despine(ax=ax)

    legend_handles = [
        matplotlib.patches.Patch(facecolor=condition_palette["AR pre"], edgecolor="#707070", label="AR pre"),
        matplotlib.patches.Patch(facecolor=condition_palette["IR pre"], edgecolor="#707070", label="IR pre"),
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        fontsize=10,
    )
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return pdf_path


AR_up_class1_DUT = sorted(
    set(AR_dut.loc[(AR_dut["p_value"] < 0.05) & (AR_dut["delta_TU"] > 0.05)].index).intersection(class1)
)
AR_down_class3_DUT = sorted(
    set(AR_dut.loc[(AR_dut["p_value"] < 0.05) & (AR_dut["delta_TU"] < -0.05)].index).intersection(class3)
)

sample_meta = sampleinfo.loc[~sampleinfo.index.duplicated(keep="first")].copy()
sample_meta["BRCAmut_numeric"] = pd.to_numeric(sample_meta["BRCAmut"], errors="coerce")
sample_meta["line_num"] = sample_meta["line"].map(parse_treatment_line_num)
valid_pre_samples = set(preTU.columns)

all_pre_samples = sorted(sample_meta.index.intersection(valid_pre_samples))
brcawt_2l_samples = sorted(
    sample_meta.loc[
        (sample_meta["BRCAmut_numeric"] == 0) & (sample_meta["line_num"] >= 2)
    ].index.intersection(valid_pre_samples)
)
brcamt_or_1l_samples = sorted(
    sample_meta.loc[
        (sample_meta["BRCAmut_numeric"] == 1) | (sample_meta["line_num"] == 1)
    ].index.intersection(valid_pre_samples)
)

ar_ir_pre_dut_specs = [
    ("Up Class1", AR_up_class1_DUT),
    ("Down Class3", AR_down_class3_DUT),
]
ar_ir_pre_comparison_specs = [
    ("All", all_pre_samples),
    ("BRCAwt & >=2L", brcawt_2l_samples),
    ("BRCAmt or 1L", brcamt_or_1l_samples),
]

print("\n===== AR-derived DUT sets applied to AR pre vs IR pre =====")
print(f"AR upregulated Class1 DUT: {len(AR_up_class1_DUT)} transcripts, {count_unique_genes_from_transcript_gene(AR_up_class1_DUT)} genes")
print(f"AR downregulated Class3 DUT: {len(AR_down_class3_DUT)} transcripts, {count_unique_genes_from_transcript_gene(AR_down_class3_DUT)} genes")

ar_ir_pre_mean_df, ar_ir_pre_pvalue_df, ar_ir_pre_sample_count_df = build_ar_ir_pre_mean_tu_df(
    ar_ir_pre_dut_specs,
    ar_ir_pre_comparison_specs,
)

print("\n===== AR pre vs IR pre sample counts by subset =====")
print(ar_ir_pre_sample_count_df.to_string(index=False))
print("\n===== AR pre vs IR pre Mann-Whitney p-values =====")
print(ar_ir_pre_pvalue_df.to_string(index=False))

ar_ir_pre_mean_df.to_csv(
    os.path.join(OUT_DIR, "ARderived_upClass1_downClass3_ARpre_vs_IRpre_meanTU_by_sample.tsv"),
    sep="\t",
    index=False,
)
ar_ir_pre_pvalue_df.to_csv(
    os.path.join(OUT_DIR, "ARderived_upClass1_downClass3_ARpre_vs_IRpre_pvalues.tsv"),
    sep="\t",
    index=False,
)
ar_ir_pre_sample_count_df.to_csv(
    os.path.join(OUT_DIR, "ARderived_upClass1_downClass3_ARpre_vs_IRpre_sample_counts.tsv"),
    sep="\t",
    index=False,
)

ar_ir_pre_boxplot_fig = plot_ar_ir_pre_dut_mean_tu_by_subset(
    ar_ir_pre_mean_df,
    ar_ir_pre_pvalue_df,
    save_stem="ARderived_upClass1_downClass3_ARpre_vs_IRpre_meanTU_boxplot",
)
print(f"AR-derived DUT AR pre vs IR pre boxplot saved={ar_ir_pre_boxplot_fig}")

# %%
#########^^^^ (11-1) Mean of sample mean TU from AR pre vs IR pre boxplots #####
def summarize_ar_ir_pre_mean_tu_for_barplot(mean_df):
    if mean_df.empty:
        return pd.DataFrame(
            columns=[
                "Subset",
                "DUT_set",
                "Condition",
                "Mean_of_sample_mean_TU",
                "SEM",
                "Sample_count",
            ]
        )

    summary_df = (
        mean_df
        .groupby(["Subset", "DUT_set", "Condition"], as_index=False)
        .agg(
            Mean_of_sample_mean_TU=("MeanTU", "mean"),
            SD=("MeanTU", "std"),
            Sample_count=("MeanTU", "count"),
        )
    )
    summary_df["SEM"] = summary_df["SD"] / np.sqrt(summary_df["Sample_count"])
    summary_df["SEM"] = summary_df["SEM"].fillna(0)
    return summary_df


def plot_ar_ir_pre_mean_tu_summary_bar(
    summary_df,
    save_stem,
    subset_order=None,
    suptitle="Mean of sample mean TU from AR pre vs IR pre boxplots",
    figsize=None,
):
    if subset_order is None:
        subset_order = ["All", "BRCAwt & >=2L", "BRCAmt or 1L"]
    dut_order = ["Up Class1", "Down Class3"]
    condition_order = ["AR pre", "IR pre"]
    condition_palette = {"AR pre": "#FDD49E", "IR pre": "#C7E9C0"}

    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")

    if figsize is None:
        figsize = (3.5 * len(subset_order), 3.8)
    fig, axes = plt.subplots(1, len(subset_order), figsize=figsize, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, subset_label in zip(axes, subset_order):
        plot_df = summary_df.loc[summary_df["Subset"] == subset_label].copy()
        if plot_df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
            ax.set_title(subset_label, fontsize=12, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("Mean TU")
            sns.despine(ax=ax)
            continue

        sns.barplot(
            data=plot_df,
            x="DUT_set",
            y="Mean_of_sample_mean_TU",
            hue="Condition",
            order=dut_order,
            hue_order=condition_order,
            palette=condition_palette,
            edgecolor="#555555",
            linewidth=0.4,
            errorbar=None,
            ax=ax,
        )

        for patch, (_, row) in zip(ax.patches, plot_df.set_index(["DUT_set", "Condition"]).loc[
            [(dut, cond) for dut in dut_order for cond in condition_order if (dut, cond) in plot_df.set_index(["DUT_set", "Condition"]).index]
        ].reset_index().iterrows()):
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_height()
            sem = row["SEM"]
            if pd.notna(sem) and sem > 0:
                ax.errorbar(
                    x,
                    y,
                    yerr=sem,
                    color="#333333",
                    capsize=2,
                    linewidth=0.8,
                    fmt="none",
                    zorder=3,
                )

        ax.set_title(subset_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Mean TU")
        ax.set_xticks(range(len(dut_order)))
        ax.set_xticklabels(["AR up\nClass1", "AR down\nClass3"])
        ax.grid(axis="y", alpha=0.16)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        sns.despine(ax=ax)

    legend_handles = [
        matplotlib.patches.Patch(facecolor=condition_palette["AR pre"], edgecolor="#707070", label="AR pre"),
        matplotlib.patches.Patch(facecolor=condition_palette["IR pre"], edgecolor="#707070", label="IR pre"),
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        fontsize=10,
    )
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return pdf_path


ar_ir_pre_mean_summary_df = summarize_ar_ir_pre_mean_tu_for_barplot(ar_ir_pre_mean_df)
ar_ir_pre_mean_summary_df.to_csv(
    os.path.join(OUT_DIR, "ARderived_upClass1_downClass3_ARpre_vs_IRpre_meanTU_barplot_summary.tsv"),
    sep="\t",
    index=False,
)

print("\n===== Mean of sample mean TU from AR pre vs IR pre boxplots =====")
print(ar_ir_pre_mean_summary_df.to_string(index=False))

ar_ir_pre_mean_summary_bar_fig = plot_ar_ir_pre_mean_tu_summary_bar(
    ar_ir_pre_mean_summary_df,
    save_stem="ARderived_upClass1_downClass3_ARpre_vs_IRpre_meanTU_summary_barplot",
)
print(f"AR-derived DUT AR pre vs IR pre mean summary barplot saved={ar_ir_pre_mean_summary_bar_fig}")

# %%
#########^^^^ (11-2) ARDUT to BRCAwt/BRCAmt pre samples without line split #####
brcawt_pre_samples = sorted(
    sample_meta.loc[sample_meta["BRCAmut_numeric"] == 0].index.intersection(valid_pre_samples)
)
brcamt_pre_samples = sorted(
    sample_meta.loc[sample_meta["BRCAmut_numeric"] == 1].index.intersection(valid_pre_samples)
)
ar_ir_pre_brca_subset_order = ["All", "BRCAwt", "BRCAmt"]
ar_ir_pre_brca_comparison_specs = [
    ("All", all_pre_samples),
    ("BRCAwt", brcawt_pre_samples),
    ("BRCAmt", brcamt_pre_samples),
]


def plot_ar_ir_pre_dut_mean_tu_brca_subset(mean_df, pvalue_df, save_stem):
    subset_order = ["All", "BRCAwt", "BRCAmt"]
    dut_order = ["Up Class1", "Down Class3"]
    condition_order = ["AR pre", "IR pre"]
    condition_palette = {"AR pre": "#FDD49E", "IR pre": "#C7E9C0"}

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=False)
    for ax, subset_label in zip(axes, subset_order):
        plot_df = mean_df.loc[mean_df["Subset"] == subset_label].copy()
        if plot_df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
            ax.set_title(subset_label, fontsize=12, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("Mean TU")
            sns.despine(ax=ax)
            continue

        sns.boxplot(
            data=plot_df,
            x="DUT_set",
            y="MeanTU",
            hue="Condition",
            order=dut_order,
            hue_order=condition_order,
            palette=condition_palette,
            showfliers=False,
            width=0.72,
            linewidth=1.2,
            ax=ax,
        )
        sns.stripplot(
            data=plot_df,
            x="DUT_set",
            y="MeanTU",
            hue="Condition",
            order=dut_order,
            hue_order=condition_order,
            dodge=True,
            palette={"AR pre": "#8C8C8C", "IR pre": "#8C8C8C"},
            alpha=0.72,
            size=3.5,
            jitter=0.08,
            ax=ax,
        )

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

        for x_idx, dut_label in enumerate(dut_order):
            p_row = pvalue_df.loc[
                (pvalue_df["Subset"] == subset_label) & (pvalue_df["DUT_set"] == dut_label)
            ]
            if p_row.empty:
                continue
            ar_values = plot_df.loc[
                (plot_df["DUT_set"] == dut_label) & (plot_df["Condition"] == "AR pre"),
                "MeanTU",
            ]
            ir_values = plot_df.loc[
                (plot_df["DUT_set"] == dut_label) & (plot_df["Condition"] == "IR pre"),
                "MeanTU",
            ]
            add_ar_ir_pre_pvalue_annotation(
                ax,
                x_center=x_idx,
                values_a=ar_values,
                values_b=ir_values,
                p_value=p_row["MannWhitney_p"].iloc[0],
            )

        ax.set_title(subset_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Mean TU")
        ax.set_xticks(range(len(dut_order)))
        ax.set_xticklabels(["AR up\nClass1", "AR down\nClass3"])
        ax.grid(axis="y", alpha=0.16)
        sns.despine(ax=ax)

    legend_handles = [
        matplotlib.patches.Patch(facecolor=condition_palette["AR pre"], edgecolor="#707070", label="AR pre"),
        matplotlib.patches.Patch(facecolor=condition_palette["IR pre"], edgecolor="#707070", label="IR pre"),
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        fontsize=10,
    )
    fig.suptitle("AR pre vs IR pre using AR-derived DUT sets by BRCA status", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return pdf_path


def plot_ar_ir_pre_mean_tu_summary_brca_bar(summary_df, save_stem):
    subset_order = ["All", "BRCAwt", "BRCAmt"]
    dut_order = ["Up Class1", "Down Class3"]
    condition_order = ["AR pre", "IR pre"]
    condition_palette = {"AR pre": "#FDD49E", "IR pre": "#C7E9C0"}

    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), sharey=True)
    for ax, subset_label in zip(axes, subset_order):
        plot_df = summary_df.loc[summary_df["Subset"] == subset_label].copy()
        if plot_df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=11)
            ax.set_title(subset_label, fontsize=12, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("Mean TU")
            sns.despine(ax=ax)
            continue

        sns.barplot(
            data=plot_df,
            x="DUT_set",
            y="Mean_of_sample_mean_TU",
            hue="Condition",
            order=dut_order,
            hue_order=condition_order,
            palette=condition_palette,
            edgecolor="#555555",
            linewidth=0.4,
            errorbar=None,
            ax=ax,
        )

        plot_indexed = plot_df.set_index(["DUT_set", "Condition"])
        ordered_rows = plot_indexed.loc[
            [(dut, cond) for dut in dut_order for cond in condition_order if (dut, cond) in plot_indexed.index]
        ].reset_index()
        for patch, (_, row) in zip(ax.patches, ordered_rows.iterrows()):
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_height()
            sem = row["SEM"]
            if pd.notna(sem) and sem > 0:
                ax.errorbar(
                    x,
                    y,
                    yerr=sem,
                    color="#333333",
                    capsize=2,
                    linewidth=0.8,
                    fmt="none",
                    zorder=3,
                )

        ax.set_title(subset_label, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Mean TU")
        ax.set_xticks(range(len(dut_order)))
        ax.set_xticklabels(["AR up\nClass1", "AR down\nClass3"])
        ax.grid(axis="y", alpha=0.16)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        sns.despine(ax=ax)

    legend_handles = [
        matplotlib.patches.Patch(facecolor=condition_palette["AR pre"], edgecolor="#707070", label="AR pre"),
        matplotlib.patches.Patch(facecolor=condition_palette["IR pre"], edgecolor="#707070", label="IR pre"),
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        fontsize=10,
    )
    fig.suptitle("Mean of sample mean TU by BRCA status", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return pdf_path


ar_ir_pre_brca_mean_df, ar_ir_pre_brca_pvalue_df, ar_ir_pre_brca_sample_count_df = build_ar_ir_pre_mean_tu_df(
    ar_ir_pre_dut_specs,
    ar_ir_pre_brca_comparison_specs,
)

print("\n===== AR pre vs IR pre sample counts by BRCA status (line ignored) =====")
print(ar_ir_pre_brca_sample_count_df.to_string(index=False))
print("\n===== AR pre vs IR pre Mann-Whitney p-values by BRCA status (line ignored) =====")
print(ar_ir_pre_brca_pvalue_df.to_string(index=False))

ar_ir_pre_brca_mean_df.to_csv(
    os.path.join(OUT_DIR, "ARderived_upClass1_downClass3_ARpre_vs_IRpre_BRCAwt_BRCAmt_meanTU_by_sample.tsv"),
    sep="\t",
    index=False,
)
ar_ir_pre_brca_pvalue_df.to_csv(
    os.path.join(OUT_DIR, "ARderived_upClass1_downClass3_ARpre_vs_IRpre_BRCAwt_BRCAmt_pvalues.tsv"),
    sep="\t",
    index=False,
)
ar_ir_pre_brca_sample_count_df.to_csv(
    os.path.join(OUT_DIR, "ARderived_upClass1_downClass3_ARpre_vs_IRpre_BRCAwt_BRCAmt_sample_counts.tsv"),
    sep="\t",
    index=False,
)

ar_ir_pre_brca_boxplot_fig = plot_ar_ir_pre_dut_mean_tu_brca_subset(
    ar_ir_pre_brca_mean_df,
    ar_ir_pre_brca_pvalue_df,
    save_stem="ARderived_upClass1_downClass3_ARpre_vs_IRpre_BRCAwt_BRCAmt_meanTU_boxplot",
)
print(f"AR-derived DUT AR pre vs IR pre BRCAwt/BRCAmt boxplot saved={ar_ir_pre_brca_boxplot_fig}")

ar_ir_pre_brca_mean_summary_df = summarize_ar_ir_pre_mean_tu_for_barplot(ar_ir_pre_brca_mean_df)
ar_ir_pre_brca_mean_summary_df.to_csv(
    os.path.join(OUT_DIR, "ARderived_upClass1_downClass3_ARpre_vs_IRpre_BRCAwt_BRCAmt_meanTU_barplot_summary.tsv"),
    sep="\t",
    index=False,
)

print("\n===== Mean of sample mean TU by BRCA status (line ignored) =====")
print(ar_ir_pre_brca_mean_summary_df.to_string(index=False))

ar_ir_pre_brca_mean_summary_bar_fig = plot_ar_ir_pre_mean_tu_summary_brca_bar(
    ar_ir_pre_brca_mean_summary_df,
    save_stem="ARderived_upClass1_downClass3_ARpre_vs_IRpre_BRCAwt_BRCAmt_meanTU_summary_barplot",
)
print(f"AR-derived DUT AR pre vs IR pre BRCAwt/BRCAmt mean summary barplot saved={ar_ir_pre_brca_mean_summary_bar_fig}")

# %%
#########^^^^ (12) AR-derived Class1/Class3 pre mean TU sample barplots #####
def build_sorted_sample_mean_bar_df(transcript_list, class_label, sort_ascending=True):
    records = []
    for group_label, sample_list in [
        ("AR", sorted(ar_samples)),
        ("IR", sorted(ir_samples)),
    ]:
        mean_df = get_pre_sample_mean_tu(filtered_trans, transcript_list, sample_list)
        if mean_df.empty:
            continue
        mean_df["Group"] = group_label
        mean_df["Class"] = class_label
        mean_df = mean_df.sort_values("MeanTU", ascending=sort_ascending).reset_index(drop=True)
        mean_df["Group_rank"] = np.arange(1, mean_df.shape[0] + 1)
        records.extend(mean_df.to_dict("records"))

    if len(records) == 0:
        return pd.DataFrame(columns=["Sample", "MeanTU", "Group", "Class", "Group_rank"])
    return pd.DataFrame(records)


def plot_sample_mean_tu_barplot(bar_df, class_label, title, save_stem):
    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")

    if bar_df.empty:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.axis("off")
        ax.text(0.02, 0.52, f"No data for {class_label}", ha="left", va="center", fontsize=11)
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        return pdf_path

    plot_df = pd.concat(
        [
            bar_df.loc[bar_df["Group"] == "AR"],
            bar_df.loc[bar_df["Group"] == "IR"],
        ],
        ignore_index=True,
    )
    plot_df["Sample_ordered"] = pd.Categorical(
        plot_df["Sample"],
        categories=plot_df["Sample"].tolist(),
        ordered=True,
    )

    palette = {"AR": "#FDD49E", "IR": "#C7E9C0"}
    fig_width = max(9.5, plot_df.shape[0] * 0.28)
    fig, ax = plt.subplots(figsize=(fig_width, 4.0))
    sns.barplot(
        data=plot_df,
        x="Sample_ordered",
        y="MeanTU",
        hue="Group",
        dodge=False,
        palette=palette,
        edgecolor="#555555",
        linewidth=0.3,
        ax=ax,
    )

    ar_n = int((plot_df["Group"] == "AR").sum())
    ir_n = int((plot_df["Group"] == "IR").sum())
    if ar_n > 0 and ir_n > 0:
        ax.axvline(ar_n - 0.5, color="#303030", linewidth=0.8)
        y_top = ax.get_ylim()[1]
        ax.text((ar_n - 1) / 2, y_top * 1.015, "AR", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(ar_n + (ir_n - 1) / 2, y_top * 1.015, "IR", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylim(0, y_top * 1.10)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=18)
    ax.set_xlabel("")
    ax.set_ylabel("Mean TU")
    ax.set_xticks(np.arange(plot_df.shape[0]))
    ax.set_xticklabels(plot_df["Sample"], rotation=90, ha="center", fontsize=7)
    ax.grid(axis="y", alpha=0.16)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return pdf_path


class1_sample_bar_df = build_sorted_sample_mean_bar_df(
    AR_up_class1_DUT,
    class_label="Class1",
    sort_ascending=True,
)
class3_sample_bar_df = build_sorted_sample_mean_bar_df(
    AR_down_class3_DUT,
    class_label="Class3",
    sort_ascending=False,
)
sample_bar_df = pd.concat([class1_sample_bar_df, class3_sample_bar_df], ignore_index=True)
sample_bar_df.to_csv(
    os.path.join(OUT_DIR, "ARderived_Class1_Class3_pre_meanTU_sample_barplot_values.tsv"),
    sep="\t",
    index=False,
)

print("\n===== AR-derived Class1/Class3 pre mean TU sample barplots =====")
print(f"Class1 samples: AR={(class1_sample_bar_df['Group'] == 'AR').sum()}, IR={(class1_sample_bar_df['Group'] == 'IR').sum()} (sorted low to high)")
print(f"Class3 samples: AR={(class3_sample_bar_df['Group'] == 'AR').sum()}, IR={(class3_sample_bar_df['Group'] == 'IR').sum()} (sorted high to low)")

class1_sample_bar_fig = plot_sample_mean_tu_barplot(
    class1_sample_bar_df,
    class_label="Class1",
    title="AR upregulated Class1 DUT pre mean TU by sample",
    save_stem="ARderived_upClass1_pre_meanTU_sample_barplot",
)
class3_sample_bar_fig = plot_sample_mean_tu_barplot(
    class3_sample_bar_df,
    class_label="Class3",
    title="AR downregulated Class3 DUT pre mean TU by sample",
    save_stem="ARderived_downClass3_pre_meanTU_sample_barplot",
)

print(f"Class1 sample barplot saved={class1_sample_bar_fig}")
print(f"Class3 sample barplot saved={class3_sample_bar_fig}")

# %%
#########^^^^ (13) Validation BRCAwt >=2L target transcript TU boxplot and survival #####
VALIDATION_CLIN_PATH = "/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/112_PARPi_clinicalinfo.txt"
VALIDATION_TRANSCRIPT_TPM_PATH = "/home/jiye/jiye/copycomparison/GENCODEquant/SEV_pre/111_pre/forval_111_transcript_TPM.txt"
VALIDATION_TARGET_TRANSCRIPTS = [
    {
        "transcript_gene": "ENST00000303538.8-PDS5A",
        "transcript_id": "ENST00000303538.8",
        "gene": "PDS5A",
        "target_class": "Class1",
    },
    {
        "transcript_gene": "MSTRG.219936.19-RBBP8",
        "transcript_id": "MSTRG.219936.19",
        "gene": "RBBP8",
        "target_class": "Class3",
    },
]
VALIDATION_RESPONSE_ORDER = ["CR", "AR", "IR"]
VALIDATION_RESPONSE_PALETTE = {"CR": "#58C1EE", "AR": AR_COLOR, "IR": IR_COLOR}
VALIDATION_PLOT_PAIRS = [("CR", "AR"), ("CR", "IR"), ("AR", "IR")]


def safe_filename_label(value):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


def validation_response_group(row):
    response = pd.to_numeric(row.get("response"), errors="coerce")
    recur = pd.to_numeric(row.get("recur"), errors="coerce")
    if response == 1 and recur == 0:
        return "CR"
    if response == 1 and recur == 1:
        return "AR"
    if response == 0:
        return "IR"
    return np.nan


def format_survival_pvalue(p_value):
    if pd.isna(p_value):
        return "NA"
    return f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.3f}"


def get_validation_expression_samples(transcript_tpm_path):
    with open(transcript_tpm_path) as handle:
        sample_cols = handle.readline().rstrip("\n").split("\t")[1:]
    if sample_cols and sample_cols[-1] == "gene_name":
        sample_cols = sample_cols[:-1]
    return pd.Index(sample_cols)


def build_validation_clin_202602(clin_path, transcript_tpm_path):
    expression_samples = get_validation_expression_samples(transcript_tpm_path)
    clin = pd.read_csv(clin_path, sep="\t", index_col=0)
    clin = clin.loc[clin.index.isin(expression_samples), :].copy()
    clin = clin[(clin["line"] != "1L") & (pd.to_numeric(clin["BRCAmt"], errors="coerce") == 0)].copy()
    clin["group"] = clin.apply(validation_response_group, axis=1)
    clin["line_num"] = clin["line"].map(parse_treatment_line_num)
    clin["PFS"] = pd.to_numeric(clin["PFS"], errors="coerce")
    clin["recur"] = pd.to_numeric(clin["recur"], errors="coerce")
    return clin


def build_validation_tpm_tu_202602(transcript_tpm_path, clin):
    val_tpm = pd.read_csv(transcript_tpm_path, sep="\t", index_col=0)
    val_tpm = val_tpm.apply(pd.to_numeric, errors="coerce")
    val_tpm = val_tpm.loc[(val_tpm > 0).sum(axis=1) >= 15]
    val_tpm = val_tpm.loc[:, val_tpm.columns.isin(clin.index)]
    val_tpm["gene"] = val_tpm.index.str.split("-", n=1).str[-1]
    gene_sum = val_tpm.groupby("gene").transform("sum")
    val_tu = val_tpm.iloc[:, :-1].div(gene_sum)
    return val_tpm, val_tu


def match_validation_transcript_key(transcript_gene, val_index):
    val_index = pd.Index(val_index.astype(str))
    if transcript_gene in val_index:
        return transcript_gene

    tx_id = str(transcript_gene).split("-", 1)[0]
    matched = val_index[val_index.str.split("-", n=1).str[0] == tx_id]
    if len(matched) == 0:
        return None
    return matched[0]


def build_validation_brcawt_ge2l_target_tu_df(clin_df, val_tu, target_specs):
    use_samples = clin_df.index[
        clin_df["group"].isin(VALIDATION_RESPONSE_ORDER)
        & clin_df["PFS"].notna()
        & clin_df["recur"].notna()
        & clin_df.index.isin(val_tu.columns)
    ].tolist()

    records = []
    missing_targets = []
    for target in target_specs:
        transcript_gene = target["transcript_gene"]
        matched_transcript = match_validation_transcript_key(transcript_gene, val_tu.index)
        if matched_transcript is None:
            missing_targets.append(transcript_gene)
            continue

        for sample in use_samples:
            records.append(
                {
                    "Transcript-Gene": transcript_gene,
                    "Transcript": target["transcript_id"],
                    "Gene": target["gene"],
                    "Class": target["target_class"],
                    "Sample": sample,
                    "TU": val_tu.loc[matched_transcript, sample],
                    "ResponseGroup": clin_df.loc[sample, "group"],
                    "line": clin_df.loc[sample, "line"],
                    "line_num": clin_df.loc[sample, "line_num"],
                    "BRCAmt": pd.to_numeric(clin_df.loc[sample, "BRCAmt"], errors="coerce"),
                    "PFS": clin_df.loc[sample, "PFS"],
                    "recur": clin_df.loc[sample, "recur"],
                    "ValidationMatchedTranscript": matched_transcript,
                }
            )

    if missing_targets:
        print("Missing validation target transcripts:", ", ".join(missing_targets))

    return pd.DataFrame(records)


def kruskal_pvalue_if_possible(plot_df, group_col="ResponseGroup", value_col="TU", order=None):
    if order is None:
        order = VALIDATION_RESPONSE_ORDER
    grouped_values = [
        pd.to_numeric(plot_df.loc[plot_df[group_col] == group, value_col], errors="coerce").dropna()
        for group in order
    ]
    grouped_values = [values for values in grouped_values if len(values) >= 2]
    if len(grouped_values) < 2:
        return np.nan
    try:
        return stats.kruskal(*grouped_values).pvalue
    except ValueError:
        return np.nan


def plot_validation_target_tu_boxplot(target_df, target, save_stem):
    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")
    os.makedirs(OUT_DIR, exist_ok=True)

    plot_df = target_df.copy()
    plot_df["ResponseGroup"] = pd.Categorical(
        plot_df["ResponseGroup"],
        categories=VALIDATION_RESPONSE_ORDER,
        ordered=True,
    )

    fig, ax = plt.subplots(figsize=(3.8, 4.2))
    if plot_df.empty:
        ax.axis("off")
        ax.text(0.02, 0.52, "No validation samples", ha="left", va="center", fontsize=11)
    else:
        sns.boxplot(
            data=plot_df,
            x="ResponseGroup",
            y="TU",
            order=VALIDATION_RESPONSE_ORDER,
            palette=VALIDATION_RESPONSE_PALETTE,
            showfliers=False,
            width=0.62,
            linewidth=1.2,
            ax=ax,
        )
        sns.stripplot(
            data=plot_df,
            x="ResponseGroup",
            y="TU",
            order=VALIDATION_RESPONSE_ORDER,
            color="#4C4C4C",
            alpha=0.75,
            size=4,
            jitter=0.14,
            ax=ax,
        )

        counts = plot_df["ResponseGroup"].value_counts().reindex(VALIDATION_RESPONSE_ORDER).fillna(0).astype(int)
        ax.set_xticklabels(VALIDATION_RESPONSE_ORDER)

        valid_pairs = [
            pair
            for pair in VALIDATION_PLOT_PAIRS
            if counts.get(pair[0], 0) >= 2 and counts.get(pair[1], 0) >= 2
        ]
        if valid_pairs:
            annotator = Annotator(
                ax,
                valid_pairs,
                data=plot_df,
                x="ResponseGroup",
                y="TU",
                order=VALIDATION_RESPONSE_ORDER,
            )
            annotator.configure(
                test="Mann-Whitney",
                text_format="full",
                loc="inside",
                verbose=0,
                pvalue_format_string="{:.2f}",
                line_height=0.02,
                line_width=1.0,
                text_offset=2,
            )
            annotator.apply_and_annotate()

        ax.set_xlabel("group")
        ax.set_ylabel("TU")
        ax.set_title(
            target["transcript_gene"],
            fontsize=10,
            fontweight="normal",
        )
        ax.grid(axis="y", alpha=0.18)
        sns.despine(ax=ax)

    fig.tight_layout()
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return pdf_path


def plot_validation_target_tu_survival(target_df, target, save_stem, title_suffix):
    from lifelines import CoxPHFitter
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")
    os.makedirs(OUT_DIR, exist_ok=True)

    survival_df = target_df.copy()
    survival_df["TU"] = pd.to_numeric(survival_df["TU"], errors="coerce")
    survival_df["PFS"] = pd.to_numeric(survival_df["PFS"], errors="coerce")
    survival_df["recur"] = pd.to_numeric(survival_df["recur"], errors="coerce")
    survival_df = survival_df.dropna(subset=["TU", "PFS", "recur"]).copy()

    fig, ax = plt.subplots(figsize=(4.5, 4.1))
    if survival_df.empty or survival_df["TU"].nunique() < 2:
        ax.axis("off")
        ax.text(0.02, 0.52, "Not enough TU variation for KM plot", ha="left", va="center", fontsize=11)
    else:
        median_tu = survival_df["TU"].median()
        survival_df["TU_group"] = np.where(survival_df["TU"] >= median_tu, "High", "Low")

        low_df = survival_df.loc[survival_df["TU_group"] == "Low"].copy()
        high_df = survival_df.loc[survival_df["TU_group"] == "High"].copy()

        if low_df.empty or high_df.empty:
            ax.axis("off")
            ax.text(0.02, 0.52, "Median split created an empty group", ha="left", va="center", fontsize=11)
        else:
            kmf = KaplanMeierFitter()
            kmf.fit(low_df["PFS"], event_observed=low_df["recur"], label="Low")
            kmf.plot_survival_function(ax=ax, ci_show=False, show_censors=True, color="#2B6CB0", linewidth=2)

            kmf.fit(high_df["PFS"], event_observed=high_df["recur"], label="High")
            kmf.plot_survival_function(ax=ax, ci_show=False, show_censors=True, color="#C94137", linewidth=2)

            logrank_p = logrank_test(
                high_df["PFS"],
                low_df["PFS"],
                event_observed_A=high_df["recur"],
                event_observed_B=low_df["recur"],
            ).p_value

            cox_text = "HR = NA\nCox p = NA"
            try:
                cox_df = survival_df[["PFS", "recur", "TU_group"]].copy()
                cox_df["High_vs_Low"] = cox_df["TU_group"].eq("High").astype(int)
                cox_df = cox_df.drop(columns="TU_group")
                cph = CoxPHFitter()
                cph.fit(cox_df, duration_col="PFS", event_col="recur")
                hr = cph.summary.loc["High_vs_Low", "exp(coef)"]
                ci_low = cph.summary.loc["High_vs_Low", "exp(coef) lower 95%"]
                ci_high = cph.summary.loc["High_vs_Low", "exp(coef) upper 95%"]
                cox_p = cph.summary.loc["High_vs_Low", "p"]
                cox_text = (
                    f"HR = {hr:.2f} ({ci_low:.2f}-{ci_high:.2f})\n"
                    f"Cox p = {format_survival_pvalue(cox_p)}"
                )
            except Exception as exc:
                print(f"{target['transcript_gene']} CoxPH skipped: {exc}")

            ax.text(
                0.67,
                0.57,
                f"{cox_text}\nlog-rank p = {format_survival_pvalue(logrank_p)}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
            )
            ax.set_ylim(-0.03, 1.03)
            ax.set_xlabel("PFS")
            ax.set_ylabel("Survival probability")
            ax.grid(alpha=0.18)
            ax.legend(frameon=False, loc="upper right", fontsize=9)
            sns.despine(ax=ax)

    ax.set_title(
        target["transcript_gene"],
        fontsize=10,
        fontweight="normal",
    )
    fig.tight_layout()
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return pdf_path


clin = build_validation_clin_202602(
    VALIDATION_CLIN_PATH,
    VALIDATION_TRANSCRIPT_TPM_PATH,
)
val_tpm, val_tu = build_validation_tpm_tu_202602(
    VALIDATION_TRANSCRIPT_TPM_PATH,
    clin,
)

ARdut_forval = AR_dut.loc[
    (AR_dut["p_value"] < 0.05) & (AR_dut["delta_TU"] < -0.05)
].index.to_list()
ARgroup1dut = set(ARdut_forval).intersection(set(class3))
val_exp = val_tpm.loc[val_tpm.index.isin(ARgroup1dut), val_tpm.columns.isin(clin.index)]
val_dut = val_tu.loc[val_tu.index.isin(ARgroup1dut), val_tu.columns.isin(clin.index)]

validation_target_tu_df = build_validation_brcawt_ge2l_target_tu_df(
    clin,
    val_tu,
    VALIDATION_TARGET_TRANSCRIPTS,
)
validation_target_tu_df.to_csv(
    os.path.join(OUT_DIR, "validation_BRCAwt_ge2L_target_transcript_TU_values.tsv"),
    sep="\t",
    index=False,
)

print("\n===== Validation BRCAwt >=2L target transcript TU sample counts =====")
if validation_target_tu_df.empty:
    print("No validation target TU records found")
else:
    print(
        validation_target_tu_df.groupby(["Transcript-Gene", "ResponseGroup"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=VALIDATION_RESPONSE_ORDER, fill_value=0)
        .to_string()
    )

validation_target_figure_paths = []
for validation_target in VALIDATION_TARGET_TRANSCRIPTS:
    target_key = validation_target["transcript_gene"]
    target_safe = safe_filename_label(target_key)
    target_plot_df = validation_target_tu_df.loc[
        validation_target_tu_df["Transcript-Gene"] == target_key
    ].copy()

    boxplot_path = plot_validation_target_tu_boxplot(
        target_plot_df,
        validation_target,
        save_stem=f"validation_BRCAwt_ge2L_{target_safe}_TU_boxplot_CR_AR_IR",
    )
    km_all_path = plot_validation_target_tu_survival(
        target_plot_df,
        validation_target,
        save_stem=f"validation_BRCAwt_ge2L_{target_safe}_TU_medianKM_CR_AR_IR",
        title_suffix="BRCAwt >=2L validation: CR+AR+IR",
    )
    km_ar_ir_path = plot_validation_target_tu_survival(
        target_plot_df.loc[target_plot_df["ResponseGroup"].isin(["AR", "IR"])].copy(),
        validation_target,
        save_stem=f"validation_BRCAwt_ge2L_{target_safe}_TU_medianKM_AR_IR_only",
        title_suffix="BRCAwt >=2L validation: AR+IR only",
    )

    validation_target_figure_paths.extend([boxplot_path, km_all_path, km_ar_ir_path])

print("\n===== Validation target TU figures saved =====")
for figure_path in validation_target_figure_paths:
    print(figure_path)

# %%
#######^^^ (14) ordered DUT: GO and survival analysis #####
def get_prepost_cols_for_samples(samples, pre_label="bfD", post_label="atD"):
    sample_set = set(map(str, samples))
    pre_cols = [
        col for col in filtered_trans.columns
        if str(col).endswith(f"-{pre_label}") and extract_sample_id(col, pre_label, post_label) in sample_set
    ]
    post_cols = [
        col for col in filtered_trans.columns
        if str(col).endswith(f"-{post_label}") and extract_sample_id(col, pre_label, post_label) in sample_set
    ]
    return pre_cols, post_cols


def compute_discovery_ordered_dut_table(
    transcript_list,
    class_label,
    expected_direction,
    ar_samples,
    ir_samples,
    pre_label="bfD",
    post_label="atD",
):
    tx_list = sorted(set(transcript_list).intersection(filtered_trans.index))
    if len(tx_list) == 0:
        return pd.DataFrame(
            columns=[
                "Transcript-Gene",
                "Class",
                "Expected_direction",
                "AR_pre",
                "IR_pre",
                "IR_post",
                "AR_post",
                "Ordered",
                "In_AR_DUT",
                "In_IR_DUT",
            ]
        )

    ar_pre_cols, ar_post_cols = get_prepost_cols_for_samples(ar_samples, pre_label, post_label)
    ir_pre_cols, ir_post_cols = get_prepost_cols_for_samples(ir_samples, pre_label, post_label)
    tu_sub = filtered_trans.loc[tx_list].apply(pd.to_numeric, errors="coerce")

    mean_df = pd.DataFrame(index=tx_list)
    mean_df["AR_pre"] = tu_sub[ar_pre_cols].mean(axis=1, skipna=True) if ar_pre_cols else np.nan
    mean_df["IR_pre"] = tu_sub[ir_pre_cols].mean(axis=1, skipna=True) if ir_pre_cols else np.nan
    mean_df["IR_post"] = tu_sub[ir_post_cols].mean(axis=1, skipna=True) if ir_post_cols else np.nan
    mean_df["AR_post"] = tu_sub[ar_post_cols].mean(axis=1, skipna=True) if ar_post_cols else np.nan

    if expected_direction == "increasing":
        ordered_mask = (
            (mean_df["AR_pre"] < mean_df["IR_pre"])
            & (mean_df["IR_pre"] < mean_df["IR_post"])
            & (mean_df["IR_post"] < mean_df["AR_post"])
        )
        mean_df["Step1_IRpre_minus_ARpre"] = mean_df["IR_pre"] - mean_df["AR_pre"]
        mean_df["Step2_IRpost_minus_IRpre"] = mean_df["IR_post"] - mean_df["IR_pre"]
        mean_df["Step3_ARpost_minus_IRpost"] = mean_df["AR_post"] - mean_df["IR_post"]
    elif expected_direction == "decreasing":
        ordered_mask = (
            (mean_df["AR_pre"] > mean_df["IR_pre"])
            & (mean_df["IR_pre"] > mean_df["IR_post"])
            & (mean_df["IR_post"] > mean_df["AR_post"])
        )
        mean_df["Step1_IRpre_minus_ARpre"] = mean_df["AR_pre"] - mean_df["IR_pre"]
        mean_df["Step2_IRpost_minus_IRpre"] = mean_df["IR_pre"] - mean_df["IR_post"]
        mean_df["Step3_ARpost_minus_IRpost"] = mean_df["IR_post"] - mean_df["AR_post"]
    else:
        raise ValueError("expected_direction must be 'increasing' or 'decreasing'")

    mean_df["Class"] = class_label
    mean_df["Expected_direction"] = expected_direction
    mean_df["Ordered"] = ordered_mask.fillna(False)
    mean_df["Min_order_step"] = mean_df[
        ["Step1_IRpre_minus_ARpre", "Step2_IRpost_minus_IRpre", "Step3_ARpost_minus_IRpost"]
    ].min(axis=1)
    mean_df["In_AR_DUT"] = mean_df.index.isin(ARdutlist)
    mean_df["In_IR_DUT"] = mean_df.index.isin(IRdutlist)
    mean_df["Gene"] = pd.Series(mean_df.index, index=mean_df.index).astype(str).str.split("-", n=1).str[-1]
    mean_df = mean_df.reset_index().rename(columns={"index": "Transcript-Gene"})
    return mean_df


ar_ir_dut_union = sorted(set(ARdutlist) | set(IRdutlist))
class1_ar_ir_union = sorted(set(ar_ir_dut_union).intersection(class1))
class3_ar_ir_union = sorted(set(ar_ir_dut_union).intersection(class3))

ordered_class1_table = compute_discovery_ordered_dut_table(
    class1_ar_ir_union,
    class_label="Class1",
    expected_direction="increasing",
    ar_samples=ar_samples,
    ir_samples=ir_samples,
)
ordered_class3_table = compute_discovery_ordered_dut_table(
    class3_ar_ir_union,
    class_label="Class3",
    expected_direction="decreasing",
    ar_samples=ar_samples,
    ir_samples=ir_samples,
)
ordered_dut_table = pd.concat([ordered_class1_table, ordered_class3_table], ignore_index=True)
ordered_dut_table.to_csv(
    os.path.join(OUT_DIR, "Discovery_ARIR_union_ordered_Class1_Class3_DUT_group_mean_trend.tsv"),
    sep="\t",
    index=False,
)

ordered_class1_dut = ordered_class1_table.loc[
    ordered_class1_table["Ordered"], "Transcript-Gene"
].tolist()
ordered_class3_dut = ordered_class3_table.loc[
    ordered_class3_table["Ordered"], "Transcript-Gene"
].tolist()
ordered_ar_class1_dut = sorted(set(ordered_class1_dut).intersection(AR_up_class1_DUT))
ordered_ar_class3_dut = sorted(set(ordered_class3_dut).intersection(AR_down_class3_DUT))

def summarize_ordered_dut_origin(transcript_list, class_label, expected_order, union_count):
    tx_set = set(transcript_list)
    ar_set = set(ARdutlist)
    ir_set = set(IRdutlist)
    ar_origin = tx_set.intersection(ar_set)
    ir_origin = tx_set.intersection(ir_set)
    both_origin = ar_origin.intersection(ir_origin)
    ar_only_origin = ar_origin.difference(ir_set)
    ir_only_origin = ir_origin.difference(ar_set)
    ordered_ar_set = tx_set.intersection(set(AR_up_class1_DUT if class_label == "Class1" else AR_down_class3_DUT))

    return {
        "Class": class_label,
        "Expected_order": expected_order,
        "AR_IR_union_DUT_count": union_count,
        "Ordered_DUT_count": len(tx_set),
        "Ordered_DUT_gene_count": count_unique_genes_from_transcript_gene(list(tx_set)),
        "Ordered_from_AR_DUT_count": len(ar_origin),
        "Ordered_from_IR_DUT_count": len(ir_origin),
        "Ordered_from_both_AR_IR_DUT_count": len(both_origin),
        "Ordered_from_AR_only_DUT_count": len(ar_only_origin),
        "Ordered_from_IR_only_DUT_count": len(ir_only_origin),
        "Ordered_AR_DUT_count": len(ordered_ar_set),
        "Ordered_AR_DUT_gene_count": count_unique_genes_from_transcript_gene(list(ordered_ar_set)),
    }


ordered_dut_summary_df = pd.DataFrame(
    [
        summarize_ordered_dut_origin(
            ordered_class1_dut,
            class_label="Class1",
            expected_order="AR_pre < IR_pre < IR_post < AR_post",
            union_count=len(class1_ar_ir_union),
        ),
        summarize_ordered_dut_origin(
            ordered_class3_dut,
            class_label="Class3",
            expected_order="AR_pre > IR_pre > IR_post > AR_post",
            union_count=len(class3_ar_ir_union),
        ),
    ]
)
ordered_dut_summary_df.to_csv(
    os.path.join(OUT_DIR, "Discovery_ARIR_union_ordered_Class1_Class3_DUT_summary.tsv"),
    sep="\t",
    index=False,
)

print("\n===== Discovery AR/IR union ordered Class1/Class3 DUT =====")
print(ordered_dut_summary_df.to_string(index=False))

ordered_go_specs = [
    (
        "Discovery_ARIR_union_ordered_Class1_DUT",
        "Discovery AR/IR union ordered Class 1 DUT",
        ordered_class1_dut,
        AR_COLOR,
    ),
    (
        "Discovery_ARIR_union_ordered_Class3_DUT",
        "Discovery AR/IR union ordered Class 3 DUT",
        ordered_class3_dut,
        IR_COLOR,
    ),
]


def filter_enrichr_result_by_fdr(enrichr_df, fdr_cutoff=0.1):
    if enrichr_df.empty or "Adjusted P-value" not in enrichr_df.columns:
        return enrichr_df
    filtered_df = enrichr_df.copy()
    filtered_df["Adjusted P-value"] = pd.to_numeric(filtered_df["Adjusted P-value"], errors="coerce")
    return filtered_df.loc[filtered_df["Adjusted P-value"] < fdr_cutoff].copy()


ordered_go_summary = []
for save_label, title, transcript_list, color in ordered_go_specs:
    genes = dut_gene_list_from_transcripts(transcript_list)
    gene_input_path = os.path.join(OUT_DIR, f"{save_label}_GO_input_genes.txt")
    pd.Series(genes, name="gene").to_csv(gene_input_path, sep="\t", index=False)

    enr_df = run_dut_top5_enrichment(genes, save_label)
    enr_df = filter_enrichr_result_by_fdr(enr_df, fdr_cutoff=0.1)
    enr_df.to_csv(os.path.join(OUT_DIR, f"{save_label}_GOBP_Reactome_enrichr.tsv"), sep="\t", index=False)
    fig_path = plot_dut_top5_enrichment_bar(
        enr_df,
        gene_count=len(genes),
        label=save_label,
        title=title,
        color=color,
        top_n=5,
    )
    ordered_go_summary.append(
        {
            "Label": save_label,
            "Transcript_count": len(transcript_list),
            "Gene_count": len(genes),
            "Figure": fig_path,
        }
    )
    print(f"{title}: transcripts={len(transcript_list)}, genes={len(genes)}, saved={fig_path}")

pd.DataFrame(ordered_go_summary).to_csv(
    os.path.join(OUT_DIR, "Discovery_ARIR_union_ordered_Class1_Class3_DUT_GO_summary.tsv"),
    sep="\t",
    index=False,
)


def match_validation_transcript_list(transcript_list, val_index):
    columns = ["Transcript-Gene", "ValidationMatchedTranscript", "Matched"]
    records = []
    used_validation_keys = set()
    for transcript_gene in transcript_list:
        matched = match_validation_transcript_key(transcript_gene, val_index)
        if matched is None:
            records.append(
                {
                    "Transcript-Gene": transcript_gene,
                    "ValidationMatchedTranscript": np.nan,
                    "Matched": False,
                }
            )
            continue
        if matched in used_validation_keys:
            continue
        used_validation_keys.add(matched)
        records.append(
            {
                "Transcript-Gene": transcript_gene,
                "ValidationMatchedTranscript": matched,
                "Matched": True,
            }
        )
    return pd.DataFrame(records, columns=columns)


def build_validation_clin_brcawt_only_202602(clin_path, transcript_tpm_path):
    expression_samples = get_validation_expression_samples(transcript_tpm_path)
    clin = pd.read_csv(clin_path, sep="\t", index_col=0)
    clin = clin.loc[clin.index.isin(expression_samples), :].copy()
    clin = clin[pd.to_numeric(clin["BRCAmt"], errors="coerce") == 0].copy()
    clin["group"] = clin.apply(validation_response_group, axis=1)
    clin["PFS"] = pd.to_numeric(clin["PFS"], errors="coerce")
    clin["recur"] = pd.to_numeric(clin["recur"], errors="coerce")
    return clin


def build_validation_dut_score_df(clin_df, val_tu, score_specs):
    use_val_tu = val_tu.loc[:, val_tu.columns.isin(clin_df.index)].apply(pd.to_numeric, errors="coerce")
    use_samples = clin_df.index[
        clin_df["group"].isin(VALIDATION_RESPONSE_ORDER)
        & clin_df["PFS"].notna()
        & clin_df["recur"].notna()
        & clin_df.index.isin(use_val_tu.columns)
    ].tolist()

    score_records = []
    match_frames = []
    summary_records = []
    score_columns = [
        "Sample",
        "Class",
        "Score_set",
        "DUT_score",
        "ResponseGroup",
        "BRCAmt",
        "PFS",
        "recur",
        "Matched_feature_count",
    ]
    for class_label, score_label, transcript_list in score_specs:
        match_df = match_validation_transcript_list(transcript_list, use_val_tu.index)
        match_df["Class"] = class_label
        match_df["Score_set"] = score_label
        match_frames.append(match_df)

        matched_features = match_df.loc[
            match_df["Matched"], "ValidationMatchedTranscript"
        ].dropna().astype(str).tolist()

        summary_records.append(
            {
                "Class": class_label,
                "Score_set": score_label,
                "Discovery_transcript_count": len(transcript_list),
                "Validation_matched_transcript_count": len(matched_features),
                "Validation_sample_count": len(use_samples),
            }
        )

        if len(matched_features) == 0:
            continue

        score_series = use_val_tu.loc[matched_features, use_samples].mean(axis=0, skipna=True)
        for sample, score in score_series.dropna().items():
            score_records.append(
                {
                    "Sample": sample,
                    "Class": class_label,
                    "Score_set": score_label,
                    "DUT_score": score,
                    "ResponseGroup": clin_df.loc[sample, "group"],
                    "BRCAmt": pd.to_numeric(clin_df.loc[sample, "BRCAmt"], errors="coerce"),
                    "PFS": clin_df.loc[sample, "PFS"],
                    "recur": clin_df.loc[sample, "recur"],
                    "Matched_feature_count": len(matched_features),
                }
            )

    score_df = pd.DataFrame(score_records, columns=score_columns)
    match_df = pd.concat(match_frames, ignore_index=True) if match_frames else pd.DataFrame()
    summary_df = pd.DataFrame(summary_records)
    return score_df, match_df, summary_df


VALIDATION_SCORE_LABEL_DISPLAY = {
    "Full_AR_DUT_score": "Full AR DUT score",
    "Ordered_DUT_score": "Ordered DUT score\n(AR+IR union)",
    "Ordered_AR_DUT_score": "Ordered AR DUT score",
}


def plot_validation_dut_score_survival_panel(score_df, ax, class_label, score_label, cohort_label):
    from lifelines import CoxPHFitter
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    survival_df = score_df.copy()
    survival_df["DUT_score"] = pd.to_numeric(survival_df["DUT_score"], errors="coerce")
    survival_df["PFS"] = pd.to_numeric(survival_df["PFS"], errors="coerce")
    survival_df["recur"] = pd.to_numeric(survival_df["recur"], errors="coerce")
    survival_df = survival_df.dropna(subset=["DUT_score", "PFS", "recur"]).copy()

    stats_record = {
        "Class": class_label,
        "Score_set": score_label,
        "Cohort": cohort_label,
        "N": survival_df.shape[0],
        "Low_N": np.nan,
        "High_N": np.nan,
        "Median_score": np.nan,
        "HR_High_vs_Low": np.nan,
        "HR_CI95_low": np.nan,
        "HR_CI95_high": np.nan,
        "Cox_p": np.nan,
        "Logrank_p": np.nan,
    }

    if survival_df.empty or survival_df["DUT_score"].nunique() < 2:
        ax.axis("off")
        ax.text(0.02, 0.52, "Not enough score variation for KM plot", ha="left", va="center", fontsize=11)
    else:
        median_score = survival_df["DUT_score"].median()
        survival_df["Score_group"] = np.where(survival_df["DUT_score"] >= median_score, "High", "Low")
        low_df = survival_df.loc[survival_df["Score_group"] == "Low"].copy()
        high_df = survival_df.loc[survival_df["Score_group"] == "High"].copy()

        stats_record["Low_N"] = low_df.shape[0]
        stats_record["High_N"] = high_df.shape[0]
        stats_record["Median_score"] = median_score

        if low_df.empty or high_df.empty:
            ax.axis("off")
            ax.text(0.02, 0.52, "Median split created an empty group", ha="left", va="center", fontsize=11)
        else:
            kmf = KaplanMeierFitter()
            kmf.fit(low_df["PFS"], event_observed=low_df["recur"], label=f"Low (n={low_df.shape[0]})")
            kmf.plot_survival_function(ax=ax, ci_show=False, show_censors=True, color="#2B6CB0", linewidth=2)

            kmf.fit(high_df["PFS"], event_observed=high_df["recur"], label=f"High (n={high_df.shape[0]})")
            kmf.plot_survival_function(ax=ax, ci_show=False, show_censors=True, color="#C94137", linewidth=2)

            logrank_p = logrank_test(
                high_df["PFS"],
                low_df["PFS"],
                event_observed_A=high_df["recur"],
                event_observed_B=low_df["recur"],
            ).p_value
            stats_record["Logrank_p"] = logrank_p

            cox_text = "HR = NA\nCox p = NA"
            try:
                cox_df = survival_df[["PFS", "recur", "Score_group"]].copy()
                cox_df["High_vs_Low"] = cox_df["Score_group"].eq("High").astype(int)
                cox_df = cox_df.drop(columns="Score_group")
                cph = CoxPHFitter()
                cph.fit(cox_df, duration_col="PFS", event_col="recur")
                hr = cph.summary.loc["High_vs_Low", "exp(coef)"]
                ci_low = cph.summary.loc["High_vs_Low", "exp(coef) lower 95%"]
                ci_high = cph.summary.loc["High_vs_Low", "exp(coef) upper 95%"]
                cox_p = cph.summary.loc["High_vs_Low", "p"]
                stats_record["HR_High_vs_Low"] = hr
                stats_record["HR_CI95_low"] = ci_low
                stats_record["HR_CI95_high"] = ci_high
                stats_record["Cox_p"] = cox_p
                cox_text = (
                    f"HR = {hr:.2f} ({ci_low:.2f}-{ci_high:.2f})\n"
                    f"Cox p = {format_survival_pvalue(cox_p)}"
                )
            except Exception as exc:
                print(f"{class_label} {score_label} {cohort_label} CoxPH skipped: {exc}")

            ax.text(
                0.61,
                0.56,
                f"{cox_text}\nlog-rank p = {format_survival_pvalue(logrank_p)}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
            )
            ax.set_ylim(-0.03, 1.03)
            ax.set_xlabel("PFS")
            ax.set_ylabel("Survival probability")
            ax.grid(alpha=0.18)
            ax.legend(frameon=False, loc="upper right", fontsize=9)
            sns.despine(ax=ax)

    ax.set_title(
        f"{VALIDATION_SCORE_LABEL_DISPLAY.get(score_label, score_label)}\n{cohort_label}",
        fontsize=10,
        fontweight="normal",
    )
    return stats_record


def plot_validation_dut_score_survival_grid(score_df, class_label, save_stem):
    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")
    os.makedirs(OUT_DIR, exist_ok=True)

    score_order = ["Full_AR_DUT_score", "Ordered_DUT_score", "Ordered_AR_DUT_score"]
    cohort_specs = [
        ("CR+AR+IR", ["CR", "AR", "IR"]),
        ("AR+IR only", ["AR", "IR"]),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(8.8, 10.6), sharey=True)
    stats_records = []
    for row_idx, score_label in enumerate(score_order):
        for col_idx, (cohort_label, response_groups) in enumerate(cohort_specs):
            ax = axes[row_idx, col_idx]
            panel_df = score_df.loc[
                (score_df["Class"] == class_label)
                & (score_df["Score_set"] == score_label)
                & (score_df["ResponseGroup"].isin(response_groups))
            ].copy()
            stats_record = plot_validation_dut_score_survival_panel(
                panel_df,
                ax=ax,
                class_label=class_label,
                score_label=score_label,
                cohort_label=cohort_label,
            )
            stats_record["Figure"] = pdf_path
            stats_records.append(stats_record)

    fig.suptitle(f"{class_label} BRCAwt validation PFS by DUT score", fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.subplots_adjust(top=0.91)
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return pdf_path, stats_records


validation_score_specs = [
    ("Class1", "Full_AR_DUT_score", AR_up_class1_DUT),
    ("Class1", "Ordered_DUT_score", ordered_class1_dut),
    ("Class1", "Ordered_AR_DUT_score", ordered_ar_class1_dut),
    ("Class3", "Full_AR_DUT_score", AR_down_class3_DUT),
    ("Class3", "Ordered_DUT_score", ordered_class3_dut),
    ("Class3", "Ordered_AR_DUT_score", ordered_ar_class3_dut),
]


def summarize_validation_score_median_split_concordance(score_df):
    score_order = ["Full_AR_DUT_score", "Ordered_DUT_score", "Ordered_AR_DUT_score"]
    cohort_specs = [
        ("CR+AR+IR", ["CR", "AR", "IR"]),
        ("AR+IR only", ["AR", "IR"]),
    ]
    records = []

    for class_label in ["Class1", "Class3"]:
        class_df = score_df.loc[score_df["Class"] == class_label].copy()
        for cohort_label, response_groups in cohort_specs:
            cohort_df = class_df.loc[class_df["ResponseGroup"].isin(response_groups)].copy()
            score_wide = cohort_df.pivot_table(
                index="Sample",
                columns="Score_set",
                values="DUT_score",
                aggfunc="first",
            )

            for idx, score_a in enumerate(score_order):
                for score_b in score_order[idx + 1:]:
                    if score_a not in score_wide.columns or score_b not in score_wide.columns:
                        continue

                    pair_df = score_wide[[score_a, score_b]].dropna()
                    if pair_df.empty:
                        continue

                    median_a = pair_df[score_a].median()
                    median_b = pair_df[score_b].median()
                    group_a = pd.Series(
                        np.where(pair_df[score_a] >= median_a, "High", "Low"),
                        index=pair_df.index,
                    )
                    group_b = pd.Series(
                        np.where(pair_df[score_b] >= median_b, "High", "Low"),
                        index=pair_df.index,
                    )
                    high_a = set(group_a.index[group_a == "High"])
                    high_b = set(group_b.index[group_b == "High"])
                    high_union = high_a.union(high_b)

                    records.append(
                        {
                            "Class": class_label,
                            "Cohort": cohort_label,
                            "Score_A": score_a,
                            "Score_B": score_b,
                            "N_common": pair_df.shape[0],
                            "Median_A": median_a,
                            "Median_B": median_b,
                            "High_A_count": len(high_a),
                            "High_B_count": len(high_b),
                            "High_group_overlap_count": len(high_a.intersection(high_b)),
                            "High_group_jaccard": (
                                len(high_a.intersection(high_b)) / len(high_union)
                                if len(high_union) > 0 else np.nan
                            ),
                            "Median_split_same_count": int((group_a == group_b).sum()),
                            "Median_split_same_fraction": float((group_a == group_b).mean()),
                            "Median_split_identical": bool((group_a == group_b).all()),
                            "Spearman_rho": pair_df[score_a].corr(pair_df[score_b], method="spearman"),
                            "Pearson_r": pair_df[score_a].corr(pair_df[score_b], method="pearson"),
                        }
                    )

    return pd.DataFrame(records)


validation_brcawt_clin = build_validation_clin_brcawt_only_202602(
    VALIDATION_CLIN_PATH,
    VALIDATION_TRANSCRIPT_TPM_PATH,
)
_, validation_brcawt_val_tu = build_validation_tpm_tu_202602(
    VALIDATION_TRANSCRIPT_TPM_PATH,
    validation_brcawt_clin,
)

validation_dut_score_df, validation_dut_score_match_df, validation_dut_score_feature_summary_df = build_validation_dut_score_df(
    validation_brcawt_clin,
    validation_brcawt_val_tu,
    validation_score_specs,
)
validation_dut_score_df.to_csv(
    os.path.join(OUT_DIR, "validation_BRCAwt_fullAR_vs_orderedUnion_vs_orderedAR_DUT_scores.tsv"),
    sep="\t",
    index=False,
)
validation_dut_score_match_df.to_csv(
    os.path.join(OUT_DIR, "validation_BRCAwt_fullAR_vs_orderedUnion_vs_orderedAR_DUT_score_transcript_matching.tsv"),
    sep="\t",
    index=False,
)
validation_dut_score_feature_summary_df.to_csv(
    os.path.join(OUT_DIR, "validation_BRCAwt_fullAR_vs_orderedUnion_vs_orderedAR_DUT_score_feature_summary.tsv"),
    sep="\t",
    index=False,
)

print("\n===== Validation BRCAwt full AR DUT vs ordered DUT score features =====")
print(validation_dut_score_feature_summary_df.to_string(index=False))

validation_score_concordance_df = summarize_validation_score_median_split_concordance(validation_dut_score_df)
validation_score_concordance_df.to_csv(
    os.path.join(OUT_DIR, "validation_BRCAwt_DUT_score_median_split_concordance.tsv"),
    sep="\t",
    index=False,
)

print("\n===== Validation DUT score median split concordance =====")
print(validation_score_concordance_df.to_string(index=False))

validation_score_survival_stats = []
validation_score_survival_figures = []
for class_label in ["Class1", "Class3"]:
    fig_path, stats_records = plot_validation_dut_score_survival_grid(
        validation_dut_score_df,
        class_label=class_label,
        save_stem=f"validation_BRCAwt_{class_label}_fullAR_vs_orderedUnion_vs_orderedAR_DUT_score_survival_grid",
    )
    validation_score_survival_figures.append(fig_path)
    validation_score_survival_stats.extend(stats_records)

validation_score_survival_stats_df = pd.DataFrame(validation_score_survival_stats)
validation_score_survival_stats_df.to_csv(
    os.path.join(OUT_DIR, "validation_BRCAwt_fullAR_vs_orderedUnion_vs_orderedAR_DUT_score_survival_stats.tsv"),
    sep="\t",
    index=False,
)

print("\n===== Validation full AR DUT vs ordered DUT score survival stats =====")
print(validation_score_survival_stats_df.to_string(index=False))
print("\n===== Validation full AR DUT vs ordered DUT score survival figures saved =====")
for figure_path in validation_score_survival_figures:
    print(figure_path)

# %%
