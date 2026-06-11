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

OUT_DIR = "/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/2605figs"
AR_COLOR = "#FF8D29"
IR_COLOR = "#8AC509"

# %%
#######^^ (1-2) Validation clinical covariate forest plot ########
from lifelines import CoxPHFitter

VALIDATION_FOREST_CLIN_PATH = "/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/112_PARPi_clinicalinfo.txt"
VALIDATION_FOREST_TPM_HEADER_PATH = "/home/jiye/jiye/copycomparison/GENCODEquant/SEV_pre/111_pre/forval_111_gene_TPM.txt"
VALIDATION_FOREST_GHRD_CUTOFF = 42
VALIDATION_FOREST_COVARIATES = [
    {
        "Column": "BRCAmt_binary",
        "Variable": "BRCAmt",
        "Forest_label": "BRCAmt",
        "Comparison": "mutant vs wildtype",
        "Reference": "BRCAwt",
    },
    {
        "Column": "Line_ge2_binary",
        "Variable": "Line binary",
        "Forest_label": "Line >=2L",
        "Comparison": ">=2L vs 1L",
        "Reference": "1L",
    },
    {
        "Column": "gHRD_high_binary",
        "Variable": "gHRD",
        "Forest_label": f"gHRDscore >= {VALIDATION_FOREST_GHRD_CUTOFF}",
        "Comparison": f">={VALIDATION_FOREST_GHRD_CUTOFF} vs <{VALIDATION_FOREST_GHRD_CUTOFF}",
        "Reference": f"gHRDscore < {VALIDATION_FOREST_GHRD_CUTOFF}",
    },
    {
        "Column": "Drug_Niraparib_binary",
        "Variable": "Drug",
        "Forest_label": "Drug: Niraparib",
        "Comparison": "Niraparib vs Olaparib",
        "Reference": "Olaparib",
    },
    {
        "Column": "Purpose_salvage_binary",
        "Variable": "Purpose",
        "Forest_label": "Purpose: salvage",
        "Comparison": "salvage vs maintenance",
        "Reference": "maintenance",
    },
]
VALIDATION_FOREST_SUBGROUPS = [
    ("CR", "CR", "CR"),
    ("ARIR", "AR/IR", "ARIR"),
]


def format_validation_forest_pvalue(p_value):
    if pd.isna(p_value):
        return "NA"
    return f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.3f}"


def prepare_validation_forest_clinical_df():
    clin_df = pd.read_csv(VALIDATION_FOREST_CLIN_PATH, sep="\t", index_col=0)
    val_header_df = pd.read_csv(VALIDATION_FOREST_TPM_HEADER_PATH, sep="\t", index_col=0, nrows=0)
    clin_df = clin_df.loc[clin_df.index.intersection(val_header_df.columns)].copy()

    model_df = clin_df.copy()
    model_df["PFS"] = pd.to_numeric(model_df["PFS"], errors="coerce")
    model_df["recur"] = pd.to_numeric(model_df["recur"], errors="coerce")
    model_df["BRCAmt_binary"] = pd.to_numeric(model_df["BRCAmt"], errors="coerce")

    line_number = pd.to_numeric(
        model_df["line"].astype(str).str.extract(r"(\d+)")[0],
        errors="coerce",
    )
    model_df["Line_ge2_binary"] = np.nan
    model_df.loc[line_number.notna(), "Line_ge2_binary"] = (
        line_number.loc[line_number.notna()] >= 2
    ).astype(int)

    ghrd_score = pd.to_numeric(model_df["gHRDscore"], errors="coerce")
    model_df["gHRD_high_binary"] = np.nan
    model_df.loc[ghrd_score.notna(), "gHRD_high_binary"] = (
        ghrd_score.loc[ghrd_score.notna()] >= VALIDATION_FOREST_GHRD_CUTOFF
    ).astype(int)

    drug_lower = model_df["drug"].astype(str).str.lower()
    model_df["Drug_Niraparib_binary"] = np.nan
    model_df.loc[drug_lower.eq("olaparib"), "Drug_Niraparib_binary"] = 0
    model_df.loc[drug_lower.eq("niraparib"), "Drug_Niraparib_binary"] = 1

    setting_lower = model_df["setting"].astype(str).str.lower()
    model_df["Purpose_salvage_binary"] = np.nan
    model_df.loc[setting_lower.eq("maintenance"), "Purpose_salvage_binary"] = 0
    model_df.loc[setting_lower.eq("salvage"), "Purpose_salvage_binary"] = 1

    response_numeric = pd.to_numeric(model_df["response"], errors="coerce")
    recur_numeric = pd.to_numeric(model_df["recur"], errors="coerce")
    model_df["Response_group"] = "Other"
    model_df.loc[(response_numeric == 1) & (recur_numeric == 0), "Response_group"] = "CR"
    model_df.loc[
        ((response_numeric == 1) & (recur_numeric == 1)) | (response_numeric == 0),
        "Response_group",
    ] = "ARIR"

    return model_df


def run_validation_forest_multivariable_cox(model_df):
    covariate_cols = [spec["Column"] for spec in VALIDATION_FOREST_COVARIATES]
    cox_df = model_df[["PFS", "recur", *covariate_cols]].copy()
    for col in ["PFS", "recur", *covariate_cols]:
        cox_df[col] = pd.to_numeric(cox_df[col], errors="coerce")
    cox_df = cox_df.dropna(subset=["PFS", "recur", *covariate_cols])
    cox_df = cox_df.loc[cox_df["recur"].isin([0, 1])].copy()
    cox_df["recur"] = cox_df["recur"].astype(int)

    base_info = {
        "Model_type": "multivariable CoxPH",
        "Model_sample_count": int(cox_df.shape[0]),
        "Model_event_count": int(cox_df["recur"].sum()) if not cox_df.empty else 0,
        "gHRD_cutoff": VALIDATION_FOREST_GHRD_CUTOFF,
    }
    records = []

    if cox_df.empty or cox_df["recur"].sum() == 0:
        status = "skipped_no_valid_samples_or_events"
        cph_summary = pd.DataFrame()
    elif any(cox_df[col].nunique(dropna=True) < 2 for col in covariate_cols):
        status = "skipped_nonvariable_covariate"
        cph_summary = pd.DataFrame()
    else:
        try:
            cph = CoxPHFitter()
            cph.fit(cox_df, duration_col="PFS", event_col="recur")
            cph_summary = cph.summary.copy()
            status = "ok"
        except Exception as exc:
            cph_summary = pd.DataFrame()
            status = f"cox_failed: {exc}"

    for spec in VALIDATION_FOREST_COVARIATES:
        col = spec["Column"]
        positive_count = int(cox_df[col].eq(1).sum()) if col in cox_df else 0
        reference_count = int(cox_df[col].eq(0).sum()) if col in cox_df else 0
        if not cph_summary.empty and col in cph_summary.index:
            hr = float(cph_summary.loc[col, "exp(coef)"])
            ci_low = float(cph_summary.loc[col, "exp(coef) lower 95%"])
            ci_high = float(cph_summary.loc[col, "exp(coef) upper 95%"])
            p_value = float(cph_summary.loc[col, "p"])
        else:
            hr = ci_low = ci_high = p_value = np.nan

        records.append(
            {
                **base_info,
                "Variable": spec["Variable"],
                "Forest_label": spec["Forest_label"],
                "Comparison": spec["Comparison"],
                "Reference": spec["Reference"],
                "Covariate_column": col,
                "Positive_count": positive_count,
                "Reference_count": reference_count,
                "HR": hr,
                "CI95_low": ci_low,
                "CI95_high": ci_high,
                "P_value": p_value,
                "Status": status,
            }
        )

    return pd.DataFrame(records), cox_df


def plot_validation_forest(
    summary_df,
    save_stem,
    title="Clinical covariates",
    effect_col="HR",
    effect_label="HR",
    x_label="Hazard ratio",
):
    plot_df = summary_df.dropna(subset=[effect_col, "CI95_low", "CI95_high"]).copy()
    plot_df = plot_df.loc[
        (plot_df[effect_col] > 0) & (plot_df["CI95_low"] > 0) & (plot_df["CI95_high"] > 0)
    ]
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    if plot_df.empty:
        ax.text(0.5, 0.5, "No valid CoxPH result", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.set_axis_off()
    else:
        y_positions = np.arange(plot_df.shape[0])
        for y_pos in y_positions:
            if y_pos % 2 == 0:
                ax.axhspan(y_pos - 0.5, y_pos + 0.5, color="#F6F6F6", zorder=0)

        xerr = np.vstack(
            [
                plot_df[effect_col].to_numpy(dtype=float) - plot_df["CI95_low"].to_numpy(dtype=float),
                plot_df["CI95_high"].to_numpy(dtype=float) - plot_df[effect_col].to_numpy(dtype=float),
            ]
        )
        ax.errorbar(
            plot_df[effect_col],
            y_positions,
            xerr=xerr,
            fmt="o",
            color="#2F6F9F",
            ecolor="#333333",
            elinewidth=1.2,
            capsize=3.5,
            markersize=6.5,
            markeredgecolor="#333333",
            markeredgewidth=0.6,
            zorder=3,
        )
        ax.axvline(1, color="#777777", linestyle="--", linewidth=1.0, alpha=0.65, zorder=1)

        xmin = max(plot_df["CI95_low"].min() * 0.72, 0.05)
        xmax = plot_df["CI95_high"].max() * 1.35
        ax.set_xscale("log")
        ax.set_xlim(xmin, xmax)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(plot_df["Forest_label"])
        ax.set_xlabel(x_label)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax.grid(axis="x", alpha=0.16, linewidth=0.7, zorder=0)
        ax.tick_params(axis="y", length=0, pad=7)
        sns.despine(ax=ax)
        ax.text(
            1.03,
            1.02,
            f"{effect_label} (95% CI), p",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            clip_on=False,
        )
        for y_pos, row in zip(y_positions, plot_df.to_dict("records")):
            ax.text(
                1.03,
                y_pos,
                f"{row[effect_col]:.2f} ({row['CI95_low']:.2f}-{row['CI95_high']:.2f}), "
                f"p={format_validation_forest_pvalue(row['P_value'])}",
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="left",
                fontsize=9,
                color="#1F1F1F",
                clip_on=False,
            )

    fig.subplots_adjust(right=0.68)
    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return {"pdf": pdf_path, "png": png_path}


os.makedirs(OUT_DIR, exist_ok=True)
validation_forest_model_df = prepare_validation_forest_clinical_df()
validation_forest_summary_df, validation_forest_cox_input_df = run_validation_forest_multivariable_cox(
    validation_forest_model_df
)
validation_forest_summary_path = os.path.join(
    OUT_DIR,
    "VAL_clinical_covariates_multivariable_Cox_forest_summary.tsv",
)
validation_forest_input_path = os.path.join(
    OUT_DIR,
    "VAL_clinical_covariates_multivariable_Cox_model_input.tsv",
)
validation_forest_summary_df.to_csv(validation_forest_summary_path, sep="\t", index=False)
validation_forest_cox_input_df.reset_index().rename(columns={"index": "Sample"}).to_csv(
    validation_forest_input_path,
    sep="\t",
    index=False,
)
validation_forest_paths = plot_validation_forest(
    validation_forest_summary_df,
    save_stem="VAL_clinical_covariates_multivariable_Cox_forest_plot",
    title="Clinical covariates",
)

validation_forest_subgroup_summaries = []
validation_forest_subgroup_inputs = []
validation_forest_subgroup_figures = []
for subgroup_id, subgroup_label, response_group in VALIDATION_FOREST_SUBGROUPS:
    subgroup_model_df = validation_forest_model_df.loc[
        validation_forest_model_df["Response_group"] == response_group
    ].copy()
    subgroup_summary_df, subgroup_cox_input_df = run_validation_forest_multivariable_cox(
        subgroup_model_df
    )
    subgroup_summary_df.insert(0, "Subgroup", subgroup_id)
    subgroup_summary_df.insert(1, "Subgroup_label", subgroup_label)
    subgroup_cox_input_df = subgroup_cox_input_df.copy()
    subgroup_cox_input_df.insert(0, "Subgroup", subgroup_id)

    subgroup_summary_path = os.path.join(
        OUT_DIR,
        f"VAL_{subgroup_id}_clinical_covariates_multivariable_Cox_forest_summary.tsv",
    )
    subgroup_input_path = os.path.join(
        OUT_DIR,
        f"VAL_{subgroup_id}_clinical_covariates_multivariable_Cox_model_input.tsv",
    )
    subgroup_summary_df.to_csv(subgroup_summary_path, sep="\t", index=False)
    subgroup_cox_input_df.reset_index().rename(columns={"index": "Sample"}).to_csv(
        subgroup_input_path,
        sep="\t",
        index=False,
    )
    subgroup_paths = plot_validation_forest(
        subgroup_summary_df,
        save_stem=f"VAL_{subgroup_id}_clinical_covariates_multivariable_Cox_forest_plot",
        title=f"{subgroup_label} clinical covariates",
    )
    validation_forest_subgroup_summaries.append(subgroup_summary_df)
    validation_forest_subgroup_inputs.append(subgroup_cox_input_df.reset_index().rename(columns={"index": "Sample"}))
    validation_forest_subgroup_figures.append(
        {
            "Subgroup": subgroup_id,
            "Subgroup_label": subgroup_label,
            "Summary_tsv": subgroup_summary_path,
            "Model_input_tsv": subgroup_input_path,
            **subgroup_paths,
        }
    )

validation_forest_subgroup_summary_df = pd.concat(
    validation_forest_subgroup_summaries,
    ignore_index=True,
)
validation_forest_subgroup_input_df = pd.concat(
    validation_forest_subgroup_inputs,
    ignore_index=True,
)
validation_forest_subgroup_figure_df = pd.DataFrame(validation_forest_subgroup_figures)
validation_forest_subgroup_summary_path = os.path.join(
    OUT_DIR,
    "VAL_CR_ARIR_clinical_covariates_multivariable_Cox_forest_summary.tsv",
)
validation_forest_subgroup_input_path = os.path.join(
    OUT_DIR,
    "VAL_CR_ARIR_clinical_covariates_multivariable_Cox_model_input.tsv",
)
validation_forest_subgroup_figure_path = os.path.join(
    OUT_DIR,
    "VAL_CR_ARIR_clinical_covariates_multivariable_Cox_forest_figure_paths.tsv",
)
validation_forest_subgroup_summary_df.to_csv(
    validation_forest_subgroup_summary_path,
    sep="\t",
    index=False,
)
validation_forest_subgroup_input_df.to_csv(
    validation_forest_subgroup_input_path,
    sep="\t",
    index=False,
)
validation_forest_subgroup_figure_df.to_csv(
    validation_forest_subgroup_figure_path,
    sep="\t",
    index=False,
)

print("\n===== Validation clinical covariate multivariable Cox forest plot =====")
print(validation_forest_summary_df.to_string(index=False))
print(f"Validation forest Cox model input saved={validation_forest_input_path}")
print(f"Validation forest summary saved={validation_forest_summary_path}")
print(f"Validation forest plot saved={validation_forest_paths['pdf']}")
print("\n===== Validation CR/ARIR clinical covariate multivariable Cox forest plots =====")
print(validation_forest_subgroup_summary_df.to_string(index=False))
print(f"Validation CR/ARIR forest summary saved={validation_forest_subgroup_summary_path}")
print(f"Validation CR/ARIR forest model input saved={validation_forest_subgroup_input_path}")
print(f"Validation CR/ARIR forest figure paths saved={validation_forest_subgroup_figure_path}")

# %%
#######^^ (1-3) Discovery survival clinical covariate forest plot ########
DISCOVERY_SURVIVAL_FOREST_CLIN_PATH = "/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost_80_clinicalinfo.txt"
DISCOVERY_SURVIVAL_FOREST_COVARIATES = [
    {
        "Column": "AR_binary",
        "Variable": "Response",
        "Forest_label": "Response: AR",
        "Comparison": "AR vs IR",
        "Reference": "IR",
    },
    {
        "Column": "BRCAmut_binary",
        "Variable": "BRCAmut",
        "Forest_label": "BRCAmut",
        "Comparison": "mutant vs wildtype",
        "Reference": "BRCAwt",
    },
    {
        "Column": "Line_NFL_binary",
        "Variable": "Line binary",
        "Forest_label": "Line: N-FL",
        "Comparison": "N-FL vs FL",
        "Reference": "FL",
    },
    {
        "Column": "Drug_Niraparib_binary",
        "Variable": "Drug",
        "Forest_label": "Drug: Niraparib",
        "Comparison": "Niraparib vs Olaparib",
        "Reference": "Olaparib",
    },
    {
        "Column": "Purpose_salvage_binary",
        "Variable": "Purpose",
        "Forest_label": "Purpose: salvage",
        "Comparison": "salvage vs maintenance",
        "Reference": "maintenance",
    },
]


def prepare_discovery_survival_forest_clinical_df():
    clin_df = pd.read_csv(DISCOVERY_SURVIVAL_FOREST_CLIN_PATH, sep="\t", index_col=0)
    clin_df.columns = clin_df.columns.str.strip()
    object_cols = clin_df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        clin_df[col] = clin_df[col].astype(str).str.strip()

    clin_df = clin_df.loc[~clin_df.index.duplicated(keep="first")].copy()
    model_df = clin_df.copy()
    model_df["AR_binary"] = pd.to_numeric(model_df["response"], errors="coerce")
    model_df["PFI"] = pd.to_numeric(model_df["PFI"], errors="coerce")
    model_df["survival"] = pd.to_numeric(model_df["survival"], errors="coerce")
    model_df["BRCAmut_binary"] = pd.to_numeric(model_df["BRCAmut"], errors="coerce")

    line_binary = model_df["line_binary"].astype(str).str.upper()
    model_df["Line_NFL_binary"] = np.nan
    model_df.loc[line_binary.eq("FL"), "Line_NFL_binary"] = 0
    model_df.loc[line_binary.eq("N-FL"), "Line_NFL_binary"] = 1

    drug_lower = model_df["drug"].astype(str).str.lower()
    model_df["Drug_Niraparib_binary"] = np.nan
    model_df.loc[drug_lower.eq("olaparib"), "Drug_Niraparib_binary"] = 0
    model_df.loc[drug_lower.eq("niraparib"), "Drug_Niraparib_binary"] = 1

    purpose_lower = model_df["purpose"].astype(str).str.lower()
    model_df["Purpose_salvage_binary"] = np.nan
    model_df.loc[purpose_lower.eq("maintenance"), "Purpose_salvage_binary"] = 0
    model_df.loc[purpose_lower.eq("salvage"), "Purpose_salvage_binary"] = 1

    return model_df


def run_discovery_survival_forest_cox(model_df):
    covariate_cols = [spec["Column"] for spec in DISCOVERY_SURVIVAL_FOREST_COVARIATES]
    cox_df = model_df[["PFI", "survival", *covariate_cols]].copy()
    for col in ["PFI", "survival", *covariate_cols]:
        cox_df[col] = pd.to_numeric(cox_df[col], errors="coerce")
    cox_df = cox_df.dropna(subset=["PFI", "survival", *covariate_cols])
    cox_df = cox_df.loc[cox_df["survival"].isin([0, 1])].copy()
    cox_df["survival"] = cox_df["survival"].astype(int)

    base_info = {
        "Model_type": "multivariable CoxPH",
        "Outcome": "PFI/survival",
        "Model_sample_count": int(cox_df.shape[0]),
        "Model_event_count": int(cox_df["survival"].sum()) if not cox_df.empty else 0,
        "AR_count": int(cox_df["AR_binary"].eq(1).sum()) if "AR_binary" in cox_df else 0,
        "IR_count": int(cox_df["AR_binary"].eq(0).sum()) if "AR_binary" in cox_df else 0,
    }
    records = []

    if cox_df.empty or cox_df["survival"].sum() == 0:
        status = "skipped_no_valid_samples_or_events"
        cph_summary = pd.DataFrame()
    elif any(cox_df[col].nunique(dropna=True) < 2 for col in covariate_cols):
        status = "skipped_nonvariable_covariate"
        cph_summary = pd.DataFrame()
    else:
        try:
            cph = CoxPHFitter()
            cph.fit(cox_df, duration_col="PFI", event_col="survival")
            cph_summary = cph.summary.copy()
            status = "ok"
        except Exception as exc:
            cph_summary = pd.DataFrame()
            status = f"cox_failed: {exc}"

    for spec in DISCOVERY_SURVIVAL_FOREST_COVARIATES:
        col = spec["Column"]
        positive_count = int(cox_df[col].eq(1).sum()) if col in cox_df else 0
        reference_count = int(cox_df[col].eq(0).sum()) if col in cox_df else 0
        if not cph_summary.empty and col in cph_summary.index:
            hr = float(cph_summary.loc[col, "exp(coef)"])
            ci_low = float(cph_summary.loc[col, "exp(coef) lower 95%"])
            ci_high = float(cph_summary.loc[col, "exp(coef) upper 95%"])
            p_value = float(cph_summary.loc[col, "p"])
        else:
            hr = ci_low = ci_high = p_value = np.nan

        records.append(
            {
                **base_info,
                "Variable": spec["Variable"],
                "Forest_label": spec["Forest_label"],
                "Comparison": spec["Comparison"],
                "Reference": spec["Reference"],
                "Covariate_column": col,
                "Positive_count": positive_count,
                "Reference_count": reference_count,
                "HR": hr,
                "CI95_low": ci_low,
                "CI95_high": ci_high,
                "P_value": p_value,
                "Status": status,
            }
        )

    return pd.DataFrame(records), cox_df


discovery_survival_forest_model_df = prepare_discovery_survival_forest_clinical_df()
discovery_survival_forest_summary_df, discovery_survival_forest_input_df = run_discovery_survival_forest_cox(
    discovery_survival_forest_model_df
)
discovery_survival_forest_summary_path = os.path.join(
    OUT_DIR,
    "DISC_survival_clinical_covariates_multivariable_Cox_forest_summary.tsv",
)
discovery_survival_forest_input_path = os.path.join(
    OUT_DIR,
    "DISC_survival_clinical_covariates_multivariable_Cox_model_input.tsv",
)
discovery_survival_forest_summary_df.to_csv(
    discovery_survival_forest_summary_path,
    sep="\t",
    index=False,
)
discovery_survival_forest_input_df.reset_index().rename(columns={"index": "Sample"}).to_csv(
    discovery_survival_forest_input_path,
    sep="\t",
    index=False,
)
discovery_survival_forest_paths = plot_validation_forest(
    discovery_survival_forest_summary_df,
    save_stem="DISC_survival_clinical_covariates_multivariable_Cox_forest_plot",
    title="Discovery survival clinical covariates",
)

print("\n===== Discovery survival clinical covariate multivariable Cox forest plot =====")
print(discovery_survival_forest_summary_df.to_string(index=False))
print(f"Discovery survival forest Cox input saved={discovery_survival_forest_input_path}")
print(f"Discovery survival forest summary saved={discovery_survival_forest_summary_path}")
print(f"Discovery survival forest plot saved={discovery_survival_forest_paths['pdf']}")

# %%
#######^^ (2) AR DUT + IR DUT union 2D scatter ########
CONDITION_COLORS = {
    "AR_pre": "#FDD49E",
    "AR_post": "#F28E2B",
    "IR_pre": "#C7E9C0",
    "IR_post": "#5AAE61",
}
CONDITION_ORDER = ["AR_pre", "AR_post", "IR_pre", "IR_post"]


def clean_transcript_ids(transcript_gene_list):
    return {str(x).split("-", 1)[0] for x in transcript_gene_list}


def build_class13_union_sample_scatter_df(pre_tu, post_tu, class1_features, class3_features, ar_sample_ids, ir_sample_ids):
    rows = []
    sample_specs = [
        ("AR_pre", pre_tu, ar_sample_ids, "AR", "pre"),
        ("AR_post", post_tu, ar_sample_ids, "AR", "post"),
        ("IR_pre", pre_tu, ir_sample_ids, "IR", "pre"),
        ("IR_post", post_tu, ir_sample_ids, "IR", "post"),
    ]

    for condition, matrix, sample_ids, response_group, timepoint in sample_specs:
        valid_samples = [sample for sample in sample_ids if sample in matrix.columns]
        for sample in valid_samples:
            rows.append(
                {
                    "Sample": sample,
                    "Condition": condition,
                    "Response": response_group,
                    "Timepoint": timepoint,
                    "Class3_mean_TU": matrix.loc[class3_features, sample].mean(),
                    "Class1_mean_TU": matrix.loc[class1_features, sample].mean(),
                }
            )

    return pd.DataFrame(rows)


def padded_axis_limits(values, floor=0.0, ceiling=1.0):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if values.empty:
        return floor, ceiling

    vmin = float(values.min())
    vmax = float(values.max())
    span = max(vmax - vmin, 0.01)
    return max(floor, vmin - span * 0.08), min(ceiling, vmax + span * 0.08)


def add_interval_to_scatter_df(scatter_df, sampleinfo_df):
    interval_by_sample = pd.to_numeric(sampleinfo_df["interval"], errors="coerce")
    if interval_by_sample.index.has_duplicates:
        interval_by_sample = interval_by_sample.groupby(level=0).first()

    scatter_df = scatter_df.copy()
    scatter_df["Interval"] = scatter_df["Sample"].map(interval_by_sample)
    return scatter_df


def build_interval_norm(interval_values):
    interval_values = pd.to_numeric(pd.Series(interval_values), errors="coerce").dropna()
    if interval_values.empty:
        return matplotlib.colors.Normalize(vmin=0, vmax=1)

    vmin = float(interval_values.min())
    vmax = float(interval_values.max())
    if vmin == vmax:
        vmin -= 0.5
        vmax += 0.5
    return matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)


def draw_interval_prepost_arrows(ax, scatter_df, interval_norm, interval_cmap):
    arrow_count = 0
    for sample, sample_df in scatter_df.groupby("Sample"):
        pre_df = sample_df[sample_df["Timepoint"] == "pre"]
        post_df = sample_df[sample_df["Timepoint"] == "post"]
        if pre_df.empty or post_df.empty:
            continue

        pre_row = pre_df.iloc[0]
        post_row = post_df.iloc[0]
        interval = pre_row["Interval"] if pd.notna(pre_row["Interval"]) else post_row["Interval"]
        if pd.isna(interval):
            arrow_color = "#777777"
        else:
            arrow_color = interval_cmap(interval_norm(float(interval)))

        ax.annotate(
            "",
            xy=(post_row["Class1_mean_TU"], post_row["Class3_mean_TU"]),
            xytext=(pre_row["Class1_mean_TU"], pre_row["Class3_mean_TU"]),
            arrowprops={
                "arrowstyle": "->",
                "color": arrow_color,
                "alpha": 0.8,
                "lw": 1.25,
                "mutation_scale": 10,
                "shrinkA": 6,
                "shrinkB": 6,
            },
            zorder=2,
        )
        arrow_count += 1
    return arrow_count


def add_interval_colorbar(fig, ax, interval_norm, interval_cmap):
    interval_sm = matplotlib.cm.ScalarMappable(norm=interval_norm, cmap=interval_cmap)
    interval_sm.set_array([])
    cbar = fig.colorbar(interval_sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Treatment interval (days)")
    return cbar


def plot_class13_sample_scatter(
    scatter_df,
    class1_count,
    class3_count,
    title,
    save_stem,
    class1_label="Class1",
    class3_label="Class3",
    show_arrows=False,
):
    fig, ax = plt.subplots(figsize=(6, 5) if show_arrows else (6, 5))
    arrow_count = 0
    if show_arrows:
        arrow_count = draw_interval_prepost_arrows(
            ax,
            scatter_df,
            interval_norm=INTERVAL_NORM,
            interval_cmap=INTERVAL_CMAP,
        )

    for condition in CONDITION_ORDER:
        plot_df = scatter_df[scatter_df["Condition"] == condition]
        if plot_df.empty:
            continue

        ax.scatter(
            plot_df["Class1_mean_TU"],
            plot_df["Class3_mean_TU"],
            s=54,
            color=CONDITION_COLORS[condition],
            edgecolor="#333333",
            linewidth=0.45,
            alpha=0.9,
            label=f"{condition} (n={len(plot_df)})",
            zorder=3,
        )

    ax.set_xlabel(f"{class1_label} mean TU\n(AR/IR union DUT, n={class1_count})")
    ax.set_ylabel(f"{class3_label} mean TU\n(AR/IR union DUT, n={class3_count})")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(padded_axis_limits(scatter_df["Class1_mean_TU"]))
    ax.set_ylim(padded_axis_limits(scatter_df["Class3_mean_TU"]))
    ax.grid(alpha=0.18, linewidth=0.7)
    ax.legend(frameon=False, loc="best")
    sns.despine(ax=ax)
    if show_arrows:
        add_interval_colorbar(fig, ax, INTERVAL_NORM, INTERVAL_CMAP)
    fig.tight_layout()

    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    return {"pdf": pdf_path, "png": png_path, "arrow_count": arrow_count}


INTERVAL_CMAP = plt.get_cmap("viridis")
INTERVAL_NORM = build_interval_norm(sampleinfo["interval"])

ar_ir_dut_union_clean = clean_transcript_ids(ARdutlist) | clean_transcript_ids(IRdutlist)
class1_clean = clean_transcript_ids(class1)
class3_clean = clean_transcript_ids(class3)

class1_ar_ir_union = sorted(ar_ir_dut_union_clean & class1_clean & set(preTU.index) & set(postTU.index))
class3_ar_ir_union = sorted(ar_ir_dut_union_clean & class3_clean & set(preTU.index) & set(postTU.index))

if len(class1_ar_ir_union) == 0:
    raise ValueError("No Class1 AR/IR union DUT transcripts were found in preTU/postTU.")
if len(class3_ar_ir_union) == 0:
    raise ValueError("No Class3 AR/IR union DUT transcripts were found in preTU/postTU.")

class13_union_scatter_df = build_class13_union_sample_scatter_df(
    pre_tu=preTU,
    post_tu=postTU,
    class1_features=class1_ar_ir_union,
    class3_features=class3_ar_ir_union,
    ar_sample_ids=ar_samples,
    ir_sample_ids=ir_samples,
)
class13_union_scatter_df = class13_union_scatter_df.dropna(subset=["Class3_mean_TU", "Class1_mean_TU"])
class13_union_scatter_df = add_interval_to_scatter_df(class13_union_scatter_df, sampleinfo)

os.makedirs(OUT_DIR, exist_ok=True)
class13_union_scatter_df.to_csv(
    os.path.join(OUT_DIR, "ARIR_union_Class1_Class3_meanTU_2D_scatter_points.tsv"),
    sep="\t",
    index=False,
)

class13_union_feature_summary = pd.DataFrame(
    [
        {"Class": "Class1", "AR_IR_union_DUT_count": len(class1_ar_ir_union)},
        {"Class": "Class3", "AR_IR_union_DUT_count": len(class3_ar_ir_union)},
    ]
)
class13_union_feature_summary.to_csv(
    os.path.join(OUT_DIR, "ARIR_union_Class1_Class3_meanTU_2D_scatter_feature_summary.tsv"),
    sep="\t",
    index=False,
)

union_scatter_no_arrow_paths = plot_class13_sample_scatter(
    scatter_df=class13_union_scatter_df,
    class1_count=len(class1_ar_ir_union),
    class3_count=len(class3_ar_ir_union),
    title="AR/IR union DUT mean TU by sample",
    save_stem="ARIR_union_Class1x_Class3y_meanTU_2D_scatter_no_arrows",
    class1_label="Class1",
    class3_label="Class3",
    show_arrows=False,
)
union_scatter_arrow_paths = plot_class13_sample_scatter(
    scatter_df=class13_union_scatter_df,
    class1_count=len(class1_ar_ir_union),
    class3_count=len(class3_ar_ir_union),
    title="AR/IR union DUT mean TU by sample",
    save_stem="ARIR_union_Class1x_Class3y_meanTU_2D_scatter_with_interval_arrows",
    class1_label="Class1",
    class3_label="Class3",
    show_arrows=True,
)

print("\n===== AR/IR union Class1+Class3 mean TU 2D scatter =====")
print(class13_union_feature_summary.to_string(index=False))
print(f"Pre-to-post arrows drawn={union_scatter_arrow_paths['arrow_count']}")
print(f"Scatter point table saved={os.path.join(OUT_DIR, 'ARIR_union_Class1_Class3_meanTU_2D_scatter_points.tsv')}")
print(f"Scatter no-arrow figure saved={union_scatter_no_arrow_paths['pdf']}")
print(f"Scatter interval-arrow figure saved={union_scatter_arrow_paths['pdf']}")

# %%
#######^^ (3) AR DUT + IR DUT union MCT-only 2D scatter ########
def compute_mct_transcript_table(pre_tu, post_tu, transcript_ids, class_label, ar_sample_ids, ir_sample_ids):
    ar_pre_cols = [sample for sample in ar_sample_ids if sample in pre_tu.columns]
    ar_post_cols = [sample for sample in ar_sample_ids if sample in post_tu.columns]
    ir_pre_cols = [sample for sample in ir_sample_ids if sample in pre_tu.columns]
    ir_post_cols = [sample for sample in ir_sample_ids if sample in post_tu.columns]

    valid_transcripts = sorted(set(transcript_ids) & set(pre_tu.index) & set(post_tu.index))
    if len(valid_transcripts) == 0:
        return pd.DataFrame(
            columns=[
                "Transcript",
                "Class",
                "AR_pre_mean_TU",
                "IR_pre_mean_TU",
                "IR_post_mean_TU",
                "AR_post_mean_TU",
                "MCT",
            ]
        )

    mct_df = pd.DataFrame(index=valid_transcripts)
    mct_df["AR_pre_mean_TU"] = pre_tu.loc[valid_transcripts, ar_pre_cols].mean(axis=1)
    mct_df["IR_pre_mean_TU"] = pre_tu.loc[valid_transcripts, ir_pre_cols].mean(axis=1)
    mct_df["IR_post_mean_TU"] = post_tu.loc[valid_transcripts, ir_post_cols].mean(axis=1)
    mct_df["AR_post_mean_TU"] = post_tu.loc[valid_transcripts, ar_post_cols].mean(axis=1)

    if class_label == "Class1":
        mct_df["MCT"] = (
            (mct_df["AR_pre_mean_TU"] < mct_df["IR_pre_mean_TU"])
            & (mct_df["IR_pre_mean_TU"] < mct_df["IR_post_mean_TU"])
            & (mct_df["IR_post_mean_TU"] < mct_df["AR_post_mean_TU"])
        )
        mct_df["MCT_order"] = "AR_pre < IR_pre < IR_post < AR_post"
    elif class_label == "Class3":
        mct_df["MCT"] = (
            (mct_df["AR_pre_mean_TU"] > mct_df["IR_pre_mean_TU"])
            & (mct_df["IR_pre_mean_TU"] > mct_df["IR_post_mean_TU"])
            & (mct_df["IR_post_mean_TU"] > mct_df["AR_post_mean_TU"])
        )
        mct_df["MCT_order"] = "AR_pre > IR_pre > IR_post > AR_post"
    else:
        raise ValueError("class_label must be 'Class1' or 'Class3'.")

    transcript_gene_map = (
        majorminor.drop_duplicates("transcriptid")
        .set_index("transcriptid")["Transcript-Gene"]
    )
    mct_df["Transcript-Gene"] = pd.Series(mct_df.index, index=mct_df.index).map(transcript_gene_map)
    mct_df["Transcript-Gene"] = mct_df["Transcript-Gene"].fillna(pd.Series(mct_df.index, index=mct_df.index))
    mct_df["Gene"] = mct_df["Transcript-Gene"].astype(str).str.split("-", n=1).str[-1]
    mct_df["Class"] = class_label
    mct_df = mct_df.reset_index().rename(columns={"index": "Transcript"})
    return mct_df[
        [
            "Transcript",
            "Transcript-Gene",
            "Gene",
            "Class",
            "MCT_order",
            "MCT",
            "AR_pre_mean_TU",
            "IR_pre_mean_TU",
            "IR_post_mean_TU",
            "AR_post_mean_TU",
        ]
    ]


class1_mct_table = compute_mct_transcript_table(
    pre_tu=preTU,
    post_tu=postTU,
    transcript_ids=class1_ar_ir_union,
    class_label="Class1",
    ar_sample_ids=ar_samples,
    ir_sample_ids=ir_samples,
)
class3_mct_table = compute_mct_transcript_table(
    pre_tu=preTU,
    post_tu=postTU,
    transcript_ids=class3_ar_ir_union,
    class_label="Class3",
    ar_sample_ids=ar_samples,
    ir_sample_ids=ir_samples,
)
mct_transcript_table = pd.concat([class1_mct_table, class3_mct_table], ignore_index=True)
mct_transcript_table.to_csv(
    os.path.join(OUT_DIR, "ARIR_union_Class1_Class3_MCT_transcripts.tsv"),
    sep="\t",
    index=False,
)

class1_mct = class1_mct_table.loc[class1_mct_table["MCT"], "Transcript"].tolist()
class3_mct = class3_mct_table.loc[class3_mct_table["MCT"], "Transcript"].tolist()

if len(class1_mct) == 0:
    raise ValueError("No Class1 MCT transcripts were found in the AR/IR union DUT set.")
if len(class3_mct) == 0:
    raise ValueError("No Class3 MCT transcripts were found in the AR/IR union DUT set.")

mct_scatter_df = build_class13_union_sample_scatter_df(
    pre_tu=preTU,
    post_tu=postTU,
    class1_features=class1_mct,
    class3_features=class3_mct,
    ar_sample_ids=ar_samples,
    ir_sample_ids=ir_samples,
)
mct_scatter_df = mct_scatter_df.dropna(subset=["Class3_mean_TU", "Class1_mean_TU"])
mct_scatter_df = add_interval_to_scatter_df(mct_scatter_df, sampleinfo)
mct_scatter_df.to_csv(
    os.path.join(OUT_DIR, "ARIR_union_Class1_Class3_MCT_meanTU_2D_scatter_points.tsv"),
    sep="\t",
    index=False,
)

mct_feature_summary = pd.DataFrame(
    [
        {
            "Class": "Class1",
            "AR_IR_union_DUT_count": len(class1_ar_ir_union),
            "MCT_count": len(class1_mct),
            "MCT_order": "AR_pre < IR_pre < IR_post < AR_post",
        },
        {
            "Class": "Class3",
            "AR_IR_union_DUT_count": len(class3_ar_ir_union),
            "MCT_count": len(class3_mct),
            "MCT_order": "AR_pre > IR_pre > IR_post > AR_post",
        },
    ]
)
mct_feature_summary.to_csv(
    os.path.join(OUT_DIR, "ARIR_union_Class1_Class3_MCT_meanTU_2D_scatter_feature_summary.tsv"),
    sep="\t",
    index=False,
)

mct_scatter_no_arrow_paths = plot_class13_sample_scatter(
    scatter_df=mct_scatter_df,
    class1_count=len(class1_mct),
    class3_count=len(class3_mct),
    title="MCT-only AR/IR union DUT mean TU by sample",
    save_stem="ARIR_union_Class1x_Class3y_MCT_meanTU_2D_scatter_no_arrows",
    class1_label="Class1 MCT",
    class3_label="Class3 MCT",
    show_arrows=False,
)
mct_scatter_arrow_paths = plot_class13_sample_scatter(
    scatter_df=mct_scatter_df,
    class1_count=len(class1_mct),
    class3_count=len(class3_mct),
    title="MCT-only AR/IR union DUT mean TU by sample",
    save_stem="ARIR_union_Class1x_Class3y_MCT_meanTU_2D_scatter_with_interval_arrows",
    class1_label="Class1 MCT",
    class3_label="Class3 MCT",
    show_arrows=True,
)

print("\n===== AR/IR union Class1+Class3 MCT-only mean TU 2D scatter =====")
print(mct_feature_summary.to_string(index=False))
print(f"Pre-to-post arrows drawn={mct_scatter_arrow_paths['arrow_count']}")
print(f"MCT transcript table saved={os.path.join(OUT_DIR, 'ARIR_union_Class1_Class3_MCT_transcripts.tsv')}")
print(f"MCT scatter point table saved={os.path.join(OUT_DIR, 'ARIR_union_Class1_Class3_MCT_meanTU_2D_scatter_points.tsv')}")
print(f"MCT scatter no-arrow figure saved={mct_scatter_no_arrow_paths['pdf']}")
print(f"MCT scatter interval-arrow figure saved={mct_scatter_arrow_paths['pdf']}")

# %%
#######^^ (4) MCT 2D scatter with KMeans k=3 cluster markers ########
CLUSTER_MARKERS = {
    1: "o",
    2: "s",
    3: "^",
}


def add_kmeans_clusters_to_scatter_df(scatter_df, k=3, random_state=0):
    clustered_df = scatter_df.copy()
    coord_cols = ["Class1_mean_TU", "Class3_mean_TU"]
    coords = clustered_df[coord_cols].apply(pd.to_numeric, errors="coerce")
    valid_mask = coords.notna().all(axis=1)
    if valid_mask.sum() < k:
        raise ValueError(f"Need at least {k} valid points for KMeans clustering.")

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=50)
    raw_labels = kmeans.fit_predict(coords.loc[valid_mask])

    center_df = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=["Class1_centroid", "Class3_centroid"],
    )
    center_df["Raw_cluster"] = center_df.index
    center_df = center_df.sort_values(["Class1_centroid", "Class3_centroid"]).reset_index(drop=True)
    raw_to_cluster = {
        int(row["Raw_cluster"]): cluster_id
        for cluster_id, row in enumerate(center_df.to_dict("records"), start=1)
    }

    clustered_df["MCT_KMeans3_cluster"] = np.nan
    clustered_df.loc[valid_mask, "MCT_KMeans3_cluster"] = [
        raw_to_cluster[int(label)] for label in raw_labels
    ]
    clustered_df["MCT_KMeans3_cluster"] = clustered_df["MCT_KMeans3_cluster"].astype("Int64")

    center_df["MCT_KMeans3_cluster"] = center_df["Raw_cluster"].map(raw_to_cluster)
    center_df = center_df.sort_values("MCT_KMeans3_cluster").reset_index(drop=True)

    cluster_counts = (
        clustered_df.dropna(subset=["MCT_KMeans3_cluster"])
        .groupby("MCT_KMeans3_cluster")
        .size()
        .rename("Point_count")
    )
    condition_counts = (
        clustered_df.dropna(subset=["MCT_KMeans3_cluster"])
        .pivot_table(
            index="MCT_KMeans3_cluster",
            columns="Condition",
            values="Sample",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(columns=CONDITION_ORDER, fill_value=0)
    )
    center_df = center_df.set_index("MCT_KMeans3_cluster")
    center_df["Point_count"] = cluster_counts
    center_df = center_df.join(condition_counts)
    center_df = center_df.reset_index()
    center_df["Marker"] = center_df["MCT_KMeans3_cluster"].map(CLUSTER_MARKERS)
    return clustered_df, center_df


def plot_mct_clustered_scatter(
    scatter_df,
    class1_count,
    class3_count,
    save_stem,
    show_arrows=False,
):
    fig, ax = plt.subplots(figsize=(6.2, 5.1))
    arrow_count = 0
    if show_arrows:
        arrow_count = draw_interval_prepost_arrows(
            ax,
            scatter_df,
            interval_norm=INTERVAL_NORM,
            interval_cmap=INTERVAL_CMAP,
        )

    for cluster_id in sorted(CLUSTER_MARKERS):
        marker = CLUSTER_MARKERS[cluster_id]
        cluster_df = scatter_df[scatter_df["MCT_KMeans3_cluster"] == cluster_id]
        for condition in CONDITION_ORDER:
            plot_df = cluster_df[cluster_df["Condition"] == condition]
            if plot_df.empty:
                continue

            ax.scatter(
                plot_df["Class1_mean_TU"],
                plot_df["Class3_mean_TU"],
                s=64,
                marker=marker,
                color=CONDITION_COLORS[condition],
                edgecolor="#333333",
                linewidth=0.55,
                alpha=0.92,
                zorder=3,
            )

    ax.set_xlabel(f"Class1 MCT mean TU\n(AR/IR union DUT, n={class1_count})")
    ax.set_ylabel(f"Class3 MCT mean TU\n(AR/IR union DUT, n={class3_count})")
    ax.set_title("MCT-only AR/IR union DUT KMeans k=3", fontsize=13, fontweight="bold")
    ax.set_xlim(padded_axis_limits(scatter_df["Class1_mean_TU"]))
    ax.set_ylim(padded_axis_limits(scatter_df["Class3_mean_TU"]))
    ax.grid(alpha=0.18, linewidth=0.7)
    sns.despine(ax=ax)

    condition_handles = [
        matplotlib.lines.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CONDITION_COLORS[condition],
            markeredgecolor="#333333",
            markersize=7,
            linestyle="None",
            label=condition,
        )
        for condition in CONDITION_ORDER
    ]
    cluster_handles = [
        matplotlib.lines.Line2D(
            [0],
            [0],
            marker=CLUSTER_MARKERS[cluster_id],
            color="none",
            markerfacecolor="#F5F5F5",
            markeredgecolor="#333333",
            markersize=7,
            linestyle="None",
            label=f"Cluster {cluster_id}",
        )
        for cluster_id in sorted(CLUSTER_MARKERS)
    ]
    condition_legend = ax.legend(
        handles=condition_handles,
        frameon=False,
        loc="upper left",
        title="Condition",
        fontsize=9,
        title_fontsize=9,
    )
    ax.add_artist(condition_legend)
    ax.legend(
        handles=cluster_handles,
        frameon=False,
        loc="lower right",
        title="KMeans",
        fontsize=9,
        title_fontsize=9,
    )

    if show_arrows:
        add_interval_colorbar(fig, ax, INTERVAL_NORM, INTERVAL_CMAP)
    fig.tight_layout()

    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return {"pdf": pdf_path, "png": png_path, "arrow_count": arrow_count}


mct_clustered_scatter_df, mct_cluster_summary_df = add_kmeans_clusters_to_scatter_df(
    mct_scatter_df,
    k=3,
    random_state=0,
)
mct_clustered_scatter_df.to_csv(
    os.path.join(OUT_DIR, "ARIR_union_Class1x_Class3y_MCT_KMeans3_scatter_points.tsv"),
    sep="\t",
    index=False,
)
mct_cluster_summary_df.to_csv(
    os.path.join(OUT_DIR, "ARIR_union_Class1x_Class3y_MCT_KMeans3_cluster_summary.tsv"),
    sep="\t",
    index=False,
)

mct_cluster_no_arrow_paths = plot_mct_clustered_scatter(
    scatter_df=mct_clustered_scatter_df,
    class1_count=len(class1_mct),
    class3_count=len(class3_mct),
    save_stem="ARIR_union_Class1x_Class3y_MCT_KMeans3_scatter_no_arrows",
    show_arrows=False,
)
mct_cluster_arrow_paths = plot_mct_clustered_scatter(
    scatter_df=mct_clustered_scatter_df,
    class1_count=len(class1_mct),
    class3_count=len(class3_mct),
    save_stem="ARIR_union_Class1x_Class3y_MCT_KMeans3_scatter_with_interval_arrows",
    show_arrows=True,
)

print("\n===== MCT-only mean TU KMeans k=3 scatter =====")
print(mct_cluster_summary_df.to_string(index=False))
print(f"MCT KMeans point table saved={os.path.join(OUT_DIR, 'ARIR_union_Class1x_Class3y_MCT_KMeans3_scatter_points.tsv')}")
print(f"MCT KMeans cluster summary saved={os.path.join(OUT_DIR, 'ARIR_union_Class1x_Class3y_MCT_KMeans3_cluster_summary.tsv')}")
print(f"MCT KMeans no-arrow figure saved={mct_cluster_no_arrow_paths['pdf']}")
print(f"MCT KMeans interval-arrow figure saved={mct_cluster_arrow_paths['pdf']}")

# %%
#######^^ (5) MCT KMeans cluster composition stacked barplot ########
STACKED_CONDITION_ORDER = ["AR_pre", "IR_pre", "IR_post", "AR_post"]


def plot_cluster_condition_stacked_bar(cluster_summary_df, save_stem):
    plot_df = cluster_summary_df.copy()
    x_labels = [f"Cluster {cluster}" for cluster in plot_df["MCT_KMeans3_cluster"]]
    bottoms = np.zeros(plot_df.shape[0], dtype=float)

    fig, ax = plt.subplots(figsize=(6, 4.0))
    for condition in STACKED_CONDITION_ORDER:
        values = plot_df[condition].to_numpy(dtype=float)
        bars = ax.bar(
            x_labels,
            values,
            bottom=bottoms,
            color=CONDITION_COLORS[condition],
            edgecolor="#333333",
            linewidth=0.5,
            label=condition,
        )

        for bar, value, bottom in zip(bars, values, bottoms):
            if value <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bottom + value / 2,
                f"{int(value)}",
                ha="center",
                va="center",
                fontsize=9,
                color="#1F1F1F",
            )
        bottoms += values

    for x_pos, total in enumerate(bottoms):
        ax.text(
            x_pos,
            total + max(bottoms) * 0.025,
            f"n={int(total)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel("Sample-condition count")
    ax.set_xlabel("MCT KMeans cluster")
    ax.set_title("Condition composition by MCT KMeans cluster", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(bottoms) * 1.16)
    ax.grid(axis="y", alpha=0.18, linewidth=0.7)
    ax.legend([], frameon=False)
    sns.despine(ax=ax)
    fig.tight_layout()

    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return {"pdf": pdf_path, "png": png_path}


mct_cluster_stacked_bar_paths = plot_cluster_condition_stacked_bar(
    mct_cluster_summary_df,
    save_stem="ARIR_union_Class1x_Class3y_MCT_KMeans3_condition_stacked_barplot",
)

print("\n===== MCT KMeans cluster condition stacked barplot =====")
print(mct_cluster_summary_df[["MCT_KMeans3_cluster", *STACKED_CONDITION_ORDER]].to_string(index=False))
print(f"MCT KMeans stacked barplot saved={mct_cluster_stacked_bar_paths['pdf']}")

# %%
#######^^ (6) AR/IR union Class1/Class3 condition mean TU lineplot with CI ########
LINE_CLASS_COLORS = {
    "Class1": "#56B4E9",
    "Class3": "#8E63C7",
}


def mean_ci95(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    n = values.size
    if n == 0:
        return np.nan, np.nan, 0
    mean_value = float(np.mean(values))
    if n == 1:
        return mean_value, 0.0, 1

    sem = stats.sem(values, nan_policy="omit")
    ci95 = float(stats.t.ppf(0.975, n - 1) * sem) if np.isfinite(sem) else 0.0
    return mean_value, ci95, int(n)


def style_stripplot_point_edges(ax, collection_start, edgecolor="#333333", linewidth=0.45):
    for collection in ax.collections[collection_start:]:
        if hasattr(collection, "set_edgecolor"):
            collection.set_edgecolor(edgecolor)
        if hasattr(collection, "set_linewidths"):
            collection.set_linewidths(linewidth)


def summarize_condition_mean_tu(scatter_df):
    records = []
    class_specs = [
        ("Class1", "Class1_mean_TU"),
        ("Class3", "Class3_mean_TU"),
    ]

    for class_label, value_col in class_specs:
        for condition in STACKED_CONDITION_ORDER:
            values = scatter_df.loc[scatter_df["Condition"] == condition, value_col]
            mean_value, ci95, sample_count = mean_ci95(values)
            records.append(
                {
                    "Class": class_label,
                    "Condition": condition,
                    "Mean_TU": mean_value,
                    "CI95": ci95,
                    "Sample_count": sample_count,
                }
            )
    return pd.DataFrame(records)


def plot_condition_mean_tu_lineplot(summary_df, save_stem, title):
    x_positions = np.arange(len(STACKED_CONDITION_ORDER), dtype=float)
    x_labels = ["AR pre", "IR pre", "IR post", "AR post"]

    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    for class_label in ["Class1", "Class3"]:
        plot_df = (
            summary_df[summary_df["Class"] == class_label]
            .set_index("Condition")
            .reindex(STACKED_CONDITION_ORDER)
            .reset_index()
        )

        ax.errorbar(
            x_positions,
            plot_df["Mean_TU"],
            yerr=plot_df["CI95"],
            color=LINE_CLASS_COLORS[class_label],
            marker="o",
            markersize=6,
            markeredgecolor="#333333",
            markeredgewidth=0.6,
            linewidth=2.0,
            capsize=4,
            elinewidth=1.2,
            label=class_label,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Mean TU")
    ax.set_xlabel("Condition")
    ax.grid(axis="y", alpha=0.18, linewidth=0.7)
    ax.legend(frameon=False, loc="best")
    sns.despine(ax=ax)
    fig.tight_layout()

    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return {"pdf": pdf_path, "png": png_path}


union_condition_mean_tu_summary_df = summarize_condition_mean_tu(class13_union_scatter_df)
union_condition_mean_tu_summary_df.to_csv(
    os.path.join(OUT_DIR, "ARIR_union_Class1_Class3_condition_meanTU_lineplot_summary.tsv"),
    sep="\t",
    index=False,
)
union_condition_mean_tu_lineplot_paths = plot_condition_mean_tu_lineplot(
    union_condition_mean_tu_summary_df,
    save_stem="ARIR_union_Class1_Class3_condition_meanTU_lineplot_CI",
    title="AR/IR union DUT mean TU across conditions",
)

print("\n===== AR/IR union Class1/Class3 condition mean TU lineplot with CI =====")
print(union_condition_mean_tu_summary_df.to_string(index=False))
print(f"AR/IR union condition mean TU summary saved={os.path.join(OUT_DIR, 'ARIR_union_Class1_Class3_condition_meanTU_lineplot_summary.tsv')}")
print(f"AR/IR union condition mean TU lineplot saved={union_condition_mean_tu_lineplot_paths['pdf']}")

# %%
#######^^ (7) AR/IR union Class1 minus Class3 condition mean TU lineplot with CI ########
def summarize_condition_delta_tu(scatter_df, point_save_stem):
    delta_df = scatter_df.copy()
    delta_df["Class1_minus_Class3_mean_TU"] = delta_df["Class1_mean_TU"] - delta_df["Class3_mean_TU"]
    delta_df.to_csv(
        os.path.join(OUT_DIR, f"{point_save_stem}.tsv"),
        sep="\t",
        index=False,
    )

    records = []
    for condition in STACKED_CONDITION_ORDER:
        values = delta_df.loc[delta_df["Condition"] == condition, "Class1_minus_Class3_mean_TU"]
        mean_value, ci95, sample_count = mean_ci95(values)
        records.append(
            {
                "Metric": "Class1_minus_Class3",
                "Condition": condition,
                "Mean_delta_TU": mean_value,
                "CI95": ci95,
                "Sample_count": sample_count,
            }
        )
    return pd.DataFrame(records)


def plot_condition_delta_tu_lineplot(summary_df, save_stem, title):
    x_positions = np.arange(len(STACKED_CONDITION_ORDER), dtype=float)
    x_labels = ["AR pre", "IR pre", "IR post", "AR post"]
    plot_df = summary_df.set_index("Condition").reindex(STACKED_CONDITION_ORDER).reset_index()

    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    ax.errorbar(
        x_positions,
        plot_df["Mean_delta_TU"],
        yerr=plot_df["CI95"],
        color="#2F6F9F",
        marker="o",
        markersize=6,
        markeredgecolor="#333333",
        markeredgewidth=0.6,
        linewidth=2.0,
        capsize=4,
        elinewidth=1.2,
        label="Class1 - Class3",
    )

    ax.axhline(0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.55)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Class1 - Class3 mean TU")
    ax.set_xlabel("Condition")
    ax.grid(axis="y", alpha=0.18, linewidth=0.7)
    ax.legend(frameon=False, loc="best")
    sns.despine(ax=ax)
    fig.tight_layout()

    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return {"pdf": pdf_path, "png": png_path}


union_condition_delta_tu_summary_df = summarize_condition_delta_tu(
    class13_union_scatter_df,
    point_save_stem="ARIR_union_Class1_minus_Class3_condition_deltaTU_points",
)
union_condition_delta_tu_summary_df.to_csv(
    os.path.join(OUT_DIR, "ARIR_union_Class1_minus_Class3_condition_deltaTU_lineplot_summary.tsv"),
    sep="\t",
    index=False,
)
union_condition_delta_tu_lineplot_paths = plot_condition_delta_tu_lineplot(
    union_condition_delta_tu_summary_df,
    save_stem="ARIR_union_Class1_minus_Class3_condition_deltaTU_lineplot_CI",
    title="AR/IR union DUT Class1 - Class3 mean TU",
)

print("\n===== AR/IR union Class1 - Class3 condition mean TU lineplot with CI =====")
print(union_condition_delta_tu_summary_df.to_string(index=False))
print(f"AR/IR union Class1 - Class3 point table saved={os.path.join(OUT_DIR, 'ARIR_union_Class1_minus_Class3_condition_deltaTU_points.tsv')}")
print(f"AR/IR union Class1 - Class3 summary saved={os.path.join(OUT_DIR, 'ARIR_union_Class1_minus_Class3_condition_deltaTU_lineplot_summary.tsv')}")
print(f"AR/IR union Class1 - Class3 lineplot saved={union_condition_delta_tu_lineplot_paths['pdf']}")

# %%
#######^^ (8) AR Class1/Class3 DUT discovery condition mean TU lineplot with CI ########
DISCOVERY_AR_CLASS_LABELS = {
    "Class1": "AR PRT",
    "Class3": "AR PST",
}
DIRECTIONAL_AR_CLASS_LABELS = {
    "Class1": "Upregulated PRT",
    "Class3": "Downregulated PST",
}


def plot_discovery_ar_condition_mean_tu_lineplot(summary_df, save_stem, title, class_label_map):
    x_positions = np.arange(len(STACKED_CONDITION_ORDER), dtype=float)
    x_labels = ["AR pre", "IR pre", "IR post", "AR post"]

    fig, ax = plt.subplots(figsize=(5.2, 4.1))
    for class_label in ["Class1", "Class3"]:
        plot_df = (
            summary_df[summary_df["Class"] == class_label]
            .set_index("Condition")
            .reindex(STACKED_CONDITION_ORDER)
            .reset_index()
        )

        ax.errorbar(
            x_positions,
            plot_df["Mean_TU"],
            yerr=plot_df["CI95"],
            color=LINE_CLASS_COLORS[class_label],
            marker="o",
            markersize=6,
            markeredgecolor="#333333",
            markeredgewidth=0.6,
            linewidth=2.0,
            capsize=4,
            elinewidth=1.2,
            label=class_label_map.get(class_label, class_label),
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Mean TU")
    ax.set_xlabel("Condition")
    ax.grid(axis="y", alpha=0.18, linewidth=0.7)
    ax.legend(frameon=False, loc="best")
    sns.despine(ax=ax)
    fig.tight_layout()

    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return {"pdf": pdf_path, "png": png_path}


def build_mean_tu_lineplot_overlay_summary(
    primary_summary_df,
    comparison_summary_df,
    primary_feature_set,
    comparison_feature_set,
    primary_label_map,
    comparison_label_map,
):
    primary_df = primary_summary_df.copy()
    primary_df["Lineplot_feature_set"] = primary_feature_set
    primary_df["Legend_label"] = primary_df["Class"].map(primary_label_map)

    comparison_df = comparison_summary_df.copy()
    comparison_df["Lineplot_feature_set"] = comparison_feature_set
    comparison_df["Legend_label"] = comparison_df["Class"].map(comparison_label_map)

    return pd.concat([primary_df, comparison_df], ignore_index=True, sort=False)


def zscore_numeric(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce")
    mean_value = values.mean()
    std_value = values.std(ddof=0)
    if pd.isna(std_value) or np.isclose(std_value, 0):
        return pd.Series(np.nan, index=values.index)
    return (values - mean_value) / std_value


def add_resistance_state_score(scatter_df):
    score_df = scatter_df.copy()
    score_df["upregulated_AR_PRT_mean_TU"] = score_df["Class1_mean_TU"]
    score_df["downregulated_AR_PST_mean_TU"] = score_df["Class3_mean_TU"]
    score_df["z_upregulated_AR_PRT_mean_TU"] = zscore_numeric(score_df["upregulated_AR_PRT_mean_TU"])
    score_df["z_downregulated_AR_PST_mean_TU"] = zscore_numeric(score_df["downregulated_AR_PST_mean_TU"])
    score_df["Resistance_state_score"] = (
        score_df["z_upregulated_AR_PRT_mean_TU"]
        - score_df["z_downregulated_AR_PST_mean_TU"]
    )
    return score_df


def summarize_resistance_state_score(score_df):
    records = []
    for condition in STACKED_CONDITION_ORDER:
        values = pd.to_numeric(
            score_df.loc[score_df["Condition"] == condition, "Resistance_state_score"],
            errors="coerce",
        ).dropna()
        mean_value, ci95, sample_count = mean_ci95(values)
        records.append(
            {
                "Condition": condition,
                "Mean_resistance_state_score": mean_value,
                "CI95": ci95,
                "Median_resistance_state_score": float(values.median()) if not values.empty else np.nan,
                "Q1": float(values.quantile(0.25)) if not values.empty else np.nan,
                "Q3": float(values.quantile(0.75)) if not values.empty else np.nan,
                "Sample_count": sample_count,
            }
        )
    return pd.DataFrame(records)


def add_resistance_score_statannotations(ax, plot_df, x_col, order, pairs):
    valid_pairs = []
    for left, right in pairs:
        left_values = plot_df.loc[plot_df[x_col] == left, "Resistance_state_score"].dropna()
        right_values = plot_df.loc[plot_df[x_col] == right, "Resistance_state_score"].dropna()
        if not left_values.empty and not right_values.empty:
            valid_pairs.append((left, right))

    if not valid_pairs:
        return

    annotator = Annotator(
        ax,
        valid_pairs,
        data=plot_df,
        x=x_col,
        y="Resistance_state_score",
        order=order,
    )
    annotator.configure(
        test="Mann-Whitney",
        text_format="star",
        loc="inside",
        verbose=0,
    )
    annotator.apply_and_annotate()


def plot_resistance_state_score_boxplot(score_df, save_stem, title):
    plot_df = score_df.dropna(subset=["Condition", "Resistance_state_score"]).copy()
    x_labels = ["AR pre", "IR pre", "IR post", "AR post"]
    condition_palette = {condition: CONDITION_COLORS[condition] for condition in STACKED_CONDITION_ORDER}

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        data=plot_df,
        x="Condition",
        y="Resistance_state_score",
        order=STACKED_CONDITION_ORDER,
        palette=condition_palette,
        width=0.58,
        fliersize=0,
        linewidth=1.0,
        boxprops={"edgecolor": "#333333", "alpha": 0.78},
        whiskerprops={"color": "#333333", "linewidth": 1.0},
        capprops={"color": "#333333", "linewidth": 1.0},
        medianprops={"color": "#111111", "linewidth": 1.4},
        ax=ax,
    )
    point_collection_start = len(ax.collections)
    sns.stripplot(
        data=plot_df,
        x="Condition",
        y="Resistance_state_score",
        order=STACKED_CONDITION_ORDER,
        palette=condition_palette,
        jitter=0.18,
        size=4.3,
        edgecolor="#333333",
        alpha=0.88,
        ax=ax,
    )
    for collection in ax.collections[point_collection_start:]:
        collection.set_edgecolor("#333333")
        collection.set_linewidths(0.45)
    add_resistance_score_statannotations(
        ax,
        plot_df,
        x_col="Condition",
        order=STACKED_CONDITION_ORDER,
        pairs=[
            ("AR_pre", "IR_pre"),
            ("AR_pre", "IR_post"),
            ("AR_pre", "AR_post"),
        ],
    )

    ax.axhline(0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.55)
    ax.set_xticks(np.arange(len(STACKED_CONDITION_ORDER)))
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Resistance-state score")
    ax.set_xlabel("Condition")
    ax.grid(axis="y", alpha=0.18, linewidth=0.7)
    sns.despine(ax=ax)
    fig.tight_layout()

    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return {"pdf": pdf_path, "png": png_path}


ar_dut_clean = clean_transcript_ids(ARdutlist)
class1_ar_dut = sorted(ar_dut_clean & class1_clean & set(preTU.index) & set(postTU.index))
class3_ar_dut = sorted(ar_dut_clean & class3_clean & set(preTU.index) & set(postTU.index))

if len(class1_ar_dut) == 0:
    raise ValueError("No Class1 AR DUT transcripts were found in preTU/postTU.")
if len(class3_ar_dut) == 0:
    raise ValueError("No Class3 AR DUT transcripts were found in preTU/postTU.")

ar_dut_scatter_df = build_class13_union_sample_scatter_df(
    pre_tu=preTU,
    post_tu=postTU,
    class1_features=class1_ar_dut,
    class3_features=class3_ar_dut,
    ar_sample_ids=ar_samples,
    ir_sample_ids=ir_samples,
)
ar_dut_scatter_df = ar_dut_scatter_df.dropna(subset=["Class3_mean_TU", "Class1_mean_TU"])
ar_dut_scatter_df = add_interval_to_scatter_df(ar_dut_scatter_df, sampleinfo)
ar_dut_scatter_df.to_csv(
    os.path.join(OUT_DIR, "AR_DUT_Class1_Class3_discovery_condition_meanTU_points.tsv"),
    sep="\t",
    index=False,
)

ar_dut_feature_summary_df = pd.DataFrame(
    [
        {
            "Class": "Class1",
            "Legend_label": DISCOVERY_AR_CLASS_LABELS["Class1"],
            "AR_DUT_count": len(class1_ar_dut),
        },
        {
            "Class": "Class3",
            "Legend_label": DISCOVERY_AR_CLASS_LABELS["Class3"],
            "AR_DUT_count": len(class3_ar_dut),
        },
    ]
)
ar_dut_feature_summary_df.to_csv(
    os.path.join(OUT_DIR, "AR_DUT_Class1_Class3_discovery_condition_meanTU_feature_summary.tsv"),
    sep="\t",
    index=False,
)

ar_dut_condition_mean_tu_summary_df = summarize_condition_mean_tu(ar_dut_scatter_df)
ar_dut_condition_mean_tu_summary_df["Legend_label"] = ar_dut_condition_mean_tu_summary_df["Class"].map(
    DISCOVERY_AR_CLASS_LABELS
)
ar_dut_condition_mean_tu_summary_df.to_csv(
    os.path.join(OUT_DIR, "AR_DUT_Class1_Class3_discovery_condition_meanTU_lineplot_summary.tsv"),
    sep="\t",
    index=False,
)
ar_dut_condition_mean_tu_lineplot_paths = plot_discovery_ar_condition_mean_tu_lineplot(
    ar_dut_condition_mean_tu_summary_df,
    save_stem="AR_DUT_Class1_Class3_discovery_condition_meanTU_lineplot_CI",
    title="Discovery AR DUT mean TU across conditions",
    class_label_map=DISCOVERY_AR_CLASS_LABELS,
)

print("\n===== AR Class1/Class3 DUT discovery condition mean TU lineplot with CI =====")
print(ar_dut_feature_summary_df.to_string(index=False))
print(ar_dut_condition_mean_tu_summary_df.to_string(index=False))
print(f"AR DUT discovery condition mean TU point table saved={os.path.join(OUT_DIR, 'AR_DUT_Class1_Class3_discovery_condition_meanTU_points.tsv')}")
print(f"AR DUT discovery condition mean TU summary saved={os.path.join(OUT_DIR, 'AR_DUT_Class1_Class3_discovery_condition_meanTU_lineplot_summary.tsv')}")
print(f"AR DUT discovery condition mean TU lineplot saved={ar_dut_condition_mean_tu_lineplot_paths['pdf']}")

# Keep AR-upregulated variables available for downstream paired/directional analyses.
ar_up_dut_clean = clean_transcript_ids(
    AR_dut.loc[(AR_dut["p_value"] < 0.05) & (AR_dut["delta_TU"] > 0.05)].index
)
class1_ar_up = sorted(ar_up_dut_clean & class1_clean & set(preTU.index) & set(postTU.index))
class3_ar_up = sorted(ar_up_dut_clean & class3_clean & set(preTU.index) & set(postTU.index))

if len(class1_ar_up) == 0:
    raise ValueError("No Class1 AR-upregulated DUT transcripts were found in preTU/postTU.")
if len(class3_ar_up) == 0:
    raise ValueError("No Class3 AR-upregulated DUT transcripts were found in preTU/postTU.")

ar_up_scatter_df = build_class13_union_sample_scatter_df(
    pre_tu=preTU,
    post_tu=postTU,
    class1_features=class1_ar_up,
    class3_features=class3_ar_up,
    ar_sample_ids=ar_samples,
    ir_sample_ids=ir_samples,
)
ar_up_scatter_df = ar_up_scatter_df.dropna(subset=["Class3_mean_TU", "Class1_mean_TU"])
ar_up_scatter_df = add_interval_to_scatter_df(ar_up_scatter_df, sampleinfo)

ar_down_dut_clean = clean_transcript_ids(
    AR_dut.loc[(AR_dut["p_value"] < 0.05) & (AR_dut["delta_TU"] < -0.05)].index
)
class3_ar_down = sorted(ar_down_dut_clean & class3_clean & set(preTU.index) & set(postTU.index))

if len(class3_ar_down) == 0:
    raise ValueError("No Class3 AR-downregulated DUT transcripts were found in preTU/postTU.")

ar_directional_scatter_df = build_class13_union_sample_scatter_df(
    pre_tu=preTU,
    post_tu=postTU,
    class1_features=class1_ar_up,
    class3_features=class3_ar_down,
    ar_sample_ids=ar_samples,
    ir_sample_ids=ir_samples,
)
ar_directional_scatter_df = ar_directional_scatter_df.dropna(subset=["Class3_mean_TU", "Class1_mean_TU"])
ar_directional_scatter_df = add_interval_to_scatter_df(ar_directional_scatter_df, sampleinfo)
ar_directional_scatter_df.to_csv(
    os.path.join(OUT_DIR, "AR_directional_Class1_up_Class3_down_discovery_condition_meanTU_points.tsv"),
    sep="\t",
    index=False,
)

ar_directional_lineplot_feature_summary_df = pd.DataFrame(
    [
        {
            "Class": "Class1",
            "Legend_label": DIRECTIONAL_AR_CLASS_LABELS["Class1"],
            "AR_directional_DUT": "AR_up",
            "Transcript_count": len(class1_ar_up),
        },
        {
            "Class": "Class3",
            "Legend_label": DIRECTIONAL_AR_CLASS_LABELS["Class3"],
            "AR_directional_DUT": "AR_down",
            "Transcript_count": len(class3_ar_down),
        },
    ]
)
ar_directional_lineplot_feature_summary_df.to_csv(
    os.path.join(OUT_DIR, "AR_directional_Class1_up_Class3_down_discovery_condition_meanTU_feature_summary.tsv"),
    sep="\t",
    index=False,
)

ar_directional_lineplot_summary_df = summarize_condition_mean_tu(ar_directional_scatter_df)
ar_directional_lineplot_summary_df["Legend_label"] = ar_directional_lineplot_summary_df["Class"].map(
    DIRECTIONAL_AR_CLASS_LABELS
)
ar_directional_lineplot_summary_df.to_csv(
    os.path.join(OUT_DIR, "AR_directional_Class1_up_Class3_down_discovery_condition_meanTU_lineplot_summary.tsv"),
    sep="\t",
    index=False,
)
ar_directional_condition_mean_tu_lineplot_paths = plot_discovery_ar_condition_mean_tu_lineplot(
    ar_directional_lineplot_summary_df,
    save_stem="AR_directional_Class1_up_Class3_down_discovery_condition_meanTU_lineplot_CI",
    title="Discovery directional AR DUT mean TU across conditions",
    class_label_map=DIRECTIONAL_AR_CLASS_LABELS,
)

resistance_state_score_df = add_resistance_state_score(ar_directional_scatter_df)
resistance_state_score_df.to_csv(
    os.path.join(OUT_DIR, "AR_directional_resistance_state_score_points.tsv"),
    sep="\t",
    index=False,
)
resistance_state_score_summary_df = summarize_resistance_state_score(resistance_state_score_df)
resistance_state_score_summary_df.to_csv(
    os.path.join(OUT_DIR, "AR_directional_resistance_state_score_boxplot_summary.tsv"),
    sep="\t",
    index=False,
)
resistance_state_score_boxplot_paths = plot_resistance_state_score_boxplot(
    resistance_state_score_df,
    save_stem="AR_directional_resistance_state_score_boxplot",
    title="",
)

print("\n===== Upregulated AR PRT/downregulated AR PST discovery condition mean TU lineplot with CI =====")
print(ar_directional_lineplot_feature_summary_df.to_string(index=False))
print(ar_directional_lineplot_summary_df.to_string(index=False))
print(f"Directional AR DUT discovery condition mean TU point table saved={os.path.join(OUT_DIR, 'AR_directional_Class1_up_Class3_down_discovery_condition_meanTU_points.tsv')}")
print(f"Directional AR DUT discovery condition mean TU summary saved={os.path.join(OUT_DIR, 'AR_directional_Class1_up_Class3_down_discovery_condition_meanTU_lineplot_summary.tsv')}")
print(f"Directional AR DUT discovery condition mean TU lineplot saved={ar_directional_condition_mean_tu_lineplot_paths['pdf']}")
print("\n===== Resistance-state score boxplot =====")
print(resistance_state_score_summary_df.to_string(index=False))
print(f"Resistance-state score point table saved={os.path.join(OUT_DIR, 'AR_directional_resistance_state_score_points.tsv')}")
print(f"Resistance-state score summary saved={os.path.join(OUT_DIR, 'AR_directional_resistance_state_score_boxplot_summary.tsv')}")
print(f"Resistance-state score boxplot saved={resistance_state_score_boxplot_paths['pdf']}")

# %%
#######^^ (8-0) Order-correlation filtered AR PRT/PST lineplot and score ########
ORDERCORR8_P_CUTOFF = 0.05
ORDERCORR8_MIN_SAMPLES = 12
ORDERCORR8_CLASS_LABELS = {
    "Class1": "State-ordered PRT",
    "Class3": "State-ordered PST",
}
ORDERCORR8_VALIDATION_GROUP_ORDER = ["AR", "IR"]
ORDERCORR8_VALIDATION_GROUP_COLORS = {
    "AR": AR_COLOR,
    "IR": IR_COLOR,
}


def plot_discovery_state_ordered_plus_directional_lineplot(
    state_ordered_summary_df,
    directional_summary_df,
    save_stem,
    title,
):
    x_positions = np.arange(len(STACKED_CONDITION_ORDER), dtype=float)
    x_labels = ["AR pre", "IR pre", "IR post", "AR post"]
    line_specs = [
        (state_ordered_summary_df, "Class1", ORDERCORR8_CLASS_LABELS["Class1"], "-", "o", 2.0, 1.0),
        (state_ordered_summary_df, "Class3", ORDERCORR8_CLASS_LABELS["Class3"], "-", "o", 2.0, 1.0),
        (directional_summary_df, "Class1", DIRECTIONAL_AR_CLASS_LABELS["Class1"], "--", "s", 1.6, 0.82),
        (directional_summary_df, "Class3", DIRECTIONAL_AR_CLASS_LABELS["Class3"], "--", "s", 1.6, 0.82),
    ]

    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    for summary_df, class_label, legend_label, linestyle, marker, linewidth, alpha in line_specs:
        plot_df = (
            summary_df[summary_df["Class"] == class_label]
            .set_index("Condition")
            .reindex(STACKED_CONDITION_ORDER)
            .reset_index()
        )
        ax.errorbar(
            x_positions,
            plot_df["Mean_TU"],
            yerr=plot_df["CI95"],
            color=LINE_CLASS_COLORS[class_label],
            marker=marker,
            markersize=5.6,
            markeredgecolor="#333333",
            markeredgewidth=0.55,
            linewidth=linewidth,
            linestyle=linestyle,
            capsize=3.5,
            elinewidth=1.0,
            alpha=alpha,
            label=legend_label,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Mean TU")
    ax.set_xlabel("Condition")
    ax.grid(axis="y", alpha=0.18, linewidth=0.7)
    ax.legend(frameon=False, loc="best", fontsize=10.5)
    sns.despine(ax=ax)
    fig.tight_layout()

    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return {"pdf": pdf_path, "png": png_path}


def build_ordercorr8_transcript_table(transcript_ids, class_label):
    condition_specs = [
        ("AR_pre", 1, preTU, ar_samples),
        ("IR_pre", 2, preTU, ir_samples),
        ("IR_post", 3, postTU, ir_samples),
        ("AR_post", 4, postTU, ar_samples),
    ]
    transcript_gene_map = (
        majorminor.drop_duplicates("transcriptid")
        .set_index("transcriptid")["Transcript-Gene"]
    )
    records = []

    for transcript_id in sorted(set(transcript_ids) & set(preTU.index) & set(postTU.index)):
        order_values = []
        tu_values = []
        record = {
            "Transcript": transcript_id,
            "Class": class_label,
            "Transcript-Gene": transcript_gene_map.get(transcript_id, transcript_id),
        }
        record["Gene"] = str(record["Transcript-Gene"]).split("-", 1)[-1]

        for condition, order_rank, matrix, sample_ids in condition_specs:
            valid_samples = [sample for sample in sample_ids if sample in matrix.columns]
            values = pd.to_numeric(matrix.loc[transcript_id, valid_samples], errors="coerce").dropna()
            record[f"{condition}_mean_TU"] = float(values.mean()) if not values.empty else np.nan
            record[f"{condition}_n"] = int(values.shape[0])
            order_values.extend([order_rank] * values.shape[0])
            tu_values.extend(values.to_numpy(dtype=float).tolist())

        if (
            len(tu_values) >= ORDERCORR8_MIN_SAMPLES
            and pd.Series(tu_values).nunique(dropna=True) > 1
            and pd.Series(order_values).nunique(dropna=True) > 1
        ):
            rho, p_value = stats.spearmanr(order_values, tu_values)
        else:
            rho, p_value = np.nan, np.nan

        if class_label == "Class1":
            selected = pd.notna(rho) and pd.notna(p_value) and rho > 0 and p_value < ORDERCORR8_P_CUTOFF
            record["Target_order"] = "AR_pre < IR_pre < IR_post < AR_post"
        elif class_label == "Class3":
            selected = pd.notna(rho) and pd.notna(p_value) and rho < 0 and p_value < ORDERCORR8_P_CUTOFF
            record["Target_order"] = "AR_pre > IR_pre > IR_post > AR_post"
        else:
            raise ValueError("class_label must be 'Class1' or 'Class3'.")

        record["Spearman_rho_TU_vs_order1234"] = float(rho) if pd.notna(rho) else np.nan
        record["Spearman_p_value"] = float(p_value) if pd.notna(p_value) else np.nan
        record["Total_n"] = int(len(tu_values))
        record["Selected_ordercorr"] = bool(selected)
        records.append(record)

    return pd.DataFrame(records)


def summarize_ordercorr8_features(class1_table, class3_table):
    rows = []
    for class_label, table in [("Class1", class1_table), ("Class3", class3_table)]:
        selected = table.loc[table["Selected_ordercorr"]].copy()
        rows.append(
            {
                "Class": class_label,
                "Legend_label": ORDERCORR8_CLASS_LABELS[class_label],
                "Candidate_count": int(table.shape[0]),
                "Selected_count": int(selected.shape[0]),
                "Selection_method": f"Spearman order p < {ORDERCORR8_P_CUTOFF}",
                "Direction": "rho > 0" if class_label == "Class1" else "rho < 0",
                "P_value_cutoff": ORDERCORR8_P_CUTOFF,
                "Min_samples": ORDERCORR8_MIN_SAMPLES,
                "Target_order": table["Target_order"].iloc[0] if not table.empty else "",
            }
        )
    return pd.DataFrame(rows)


def add_ordercorr8_resistance_state_score(scatter_df):
    score_df = scatter_df.copy()
    score_df["ordercorr_AR_PRT_mean_TU"] = score_df["Class1_mean_TU"]
    score_df["ordercorr_AR_PST_mean_TU"] = score_df["Class3_mean_TU"]
    score_df["z_ordercorr_AR_PRT_mean_TU"] = zscore_numeric(score_df["ordercorr_AR_PRT_mean_TU"])
    score_df["z_ordercorr_AR_PST_mean_TU"] = zscore_numeric(score_df["ordercorr_AR_PST_mean_TU"])
    score_df["Resistance_state_score"] = (
        score_df["z_ordercorr_AR_PRT_mean_TU"]
        - score_df["z_ordercorr_AR_PST_mean_TU"]
    )
    return score_df


def ordercorr8_match_validation_rows(transcript_ids, val_index):
    txid_to_val_rows = {}
    for val_row in pd.Index(val_index).astype(str):
        tx_id = val_row.split("-", 1)[0]
        txid_to_val_rows.setdefault(tx_id, []).append(val_row)

    matched_rows = []
    for transcript_id in transcript_ids:
        matched_rows.extend(txid_to_val_rows.get(str(transcript_id).split("-", 1)[0], []))
    return list(dict.fromkeys(matched_rows))


def ordercorr8_prepare_validation_tu(transcript_tpm_path):
    val_transcript_tpm = pd.read_csv(transcript_tpm_path, sep="\t", index_col=0)
    if "gene_name" in val_transcript_tpm.columns:
        val_transcript_tpm = val_transcript_tpm.drop(columns=["gene_name"])
    val_transcript_tpm = val_transcript_tpm.apply(pd.to_numeric, errors="coerce").fillna(0)
    val_gene = val_transcript_tpm.index.to_series().astype(str).str.split("-", n=1).str[-1]
    val_gene_sum = val_transcript_tpm.groupby(val_gene).transform("sum").replace(0, np.nan)
    return val_transcript_tpm.div(val_gene_sum).fillna(0)


def ordercorr8_prepare_validation_clinical(val_tu_df):
    val_clin = pd.read_csv(
        "/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/112_PARPi_clinicalinfo.txt",
        sep="\t",
        index_col=0,
    )
    val_clin = val_clin.loc[val_clin.index.intersection(val_tu_df.columns)].copy()
    val_clin["group"] = "i"
    val_clin.loc[(val_clin["response"] == 1) & (val_clin["recur"] == 1), "group"] = "AR"
    val_clin.loc[val_clin["response"] == 0, "group"] = "IR"
    val_clin.loc[(val_clin["response"] == 1) & (val_clin["recur"] == 0), "group"] = "CR"
    return val_clin


def build_ordercorr8_validation_point_df(val_tu_df, val_clin_df, class1_features, class3_features):
    class1_rows = ordercorr8_match_validation_rows(class1_features, val_tu_df.index)
    class3_rows = ordercorr8_match_validation_rows(class3_features, val_tu_df.index)
    if len(class1_rows) == 0:
        raise ValueError("No order-corr AR PRT transcripts matched the validation TU matrix.")
    if len(class3_rows) == 0:
        raise ValueError("No order-corr AR PST transcripts matched the validation TU matrix.")

    plot_clin = val_clin_df.loc[val_clin_df["group"].isin(ORDERCORR8_VALIDATION_GROUP_ORDER)].copy()
    common_samples = plot_clin.index.intersection(val_tu_df.columns)
    plot_clin = plot_clin.loc[common_samples].copy()

    class1_mean = val_tu_df.loc[class1_rows, plot_clin.index].mean(axis=0)
    class3_mean = val_tu_df.loc[class3_rows, plot_clin.index].mean(axis=0)
    point_df = pd.DataFrame(
        {
            "Sample": plot_clin.index,
            "group": plot_clin["group"].values,
            "Class1_mean_TU": class1_mean.reindex(plot_clin.index).values,
            "Class3_mean_TU": class3_mean.reindex(plot_clin.index).values,
        }
    )
    match_summary_df = pd.DataFrame(
        [
            {
                "Class": "Class1",
                "Legend_label": ORDERCORR8_CLASS_LABELS["Class1"],
                "Discovery_selected_count": len(class1_features),
                "Validation_matched_row_count": len(class1_rows),
            },
            {
                "Class": "Class3",
                "Legend_label": ORDERCORR8_CLASS_LABELS["Class3"],
                "Discovery_selected_count": len(class3_features),
                "Validation_matched_row_count": len(class3_rows),
            },
        ]
    )
    return point_df, match_summary_df


def summarize_ordercorr8_validation_group_mean_tu(point_df):
    records = []
    for class_label, value_col in [("Class1", "Class1_mean_TU"), ("Class3", "Class3_mean_TU")]:
        for group in ORDERCORR8_VALIDATION_GROUP_ORDER:
            values = point_df.loc[point_df["group"] == group, value_col]
            mean_value, ci95, sample_count = mean_ci95(values)
            records.append(
                {
                    "Class": class_label,
                    "Legend_label": ORDERCORR8_CLASS_LABELS[class_label],
                    "group": group,
                    "Mean_TU": mean_value,
                    "CI95": ci95,
                    "Sample_count": sample_count,
                }
            )
    return pd.DataFrame(records)


def summarize_ordercorr8_validation_score(score_df):
    records = []
    for group in ORDERCORR8_VALIDATION_GROUP_ORDER:
        values = pd.to_numeric(
            score_df.loc[score_df["group"] == group, "Resistance_state_score"],
            errors="coerce",
        ).dropna()
        mean_value, ci95, sample_count = mean_ci95(values)
        records.append(
            {
                "group": group,
                "Mean_resistance_state_score": mean_value,
                "CI95": ci95,
                "Median_resistance_state_score": float(values.median()) if not values.empty else np.nan,
                "Q1": float(values.quantile(0.25)) if not values.empty else np.nan,
                "Q3": float(values.quantile(0.75)) if not values.empty else np.nan,
                "Sample_count": sample_count,
            }
        )
    return pd.DataFrame(records)


def plot_ordercorr8_validation_lineplot(
    summary_df,
    save_stem,
    title,
    comparison_summary_df=None,
    comparison_label_map=None,
):
    x_positions = np.arange(len(ORDERCORR8_VALIDATION_GROUP_ORDER), dtype=float)
    line_specs = [
        (summary_df, "Class1", ORDERCORR8_CLASS_LABELS["Class1"], "-", "o", 2.0, 1.0),
        (summary_df, "Class3", ORDERCORR8_CLASS_LABELS["Class3"], "-", "o", 2.0, 1.0),
    ]
    if comparison_summary_df is not None:
        comparison_label_map = comparison_label_map or {}
        line_specs.extend(
            [
                (
                    comparison_summary_df,
                    "Class1",
                    comparison_label_map.get("Class1", "Class1"),
                    "--",
                    "s",
                    1.6,
                    0.82,
                ),
                (
                    comparison_summary_df,
                    "Class3",
                    comparison_label_map.get("Class3", "Class3"),
                    "--",
                    "s",
                    1.6,
                    0.82,
                ),
            ]
        )

    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    for summary_source_df, class_label, legend_label, linestyle, marker, linewidth, alpha in line_specs:
        plot_df = (
            summary_source_df[summary_source_df["Class"] == class_label]
            .set_index("group")
            .reindex(ORDERCORR8_VALIDATION_GROUP_ORDER)
            .reset_index()
        )
        ax.errorbar(
            x_positions,
            plot_df["Mean_TU"],
            yerr=plot_df["CI95"],
            color=LINE_CLASS_COLORS[class_label],
            marker=marker,
            markersize=5.6,
            markeredgecolor="#333333",
            markeredgewidth=0.55,
            linewidth=linewidth,
            linestyle=linestyle,
            capsize=3.5,
            elinewidth=1.0,
            alpha=alpha,
            label=legend_label,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(ORDERCORR8_VALIDATION_GROUP_ORDER)
    ax.set_ylabel("Mean TU")
    ax.set_xlabel("Group")
    ax.grid(axis="y", alpha=0.18, linewidth=0.7)
    ax.legend(frameon=False, loc="best", fontsize=10.5)
    sns.despine(ax=ax)
    fig.tight_layout()

    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return {"pdf": pdf_path, "png": png_path}


def plot_ordercorr8_validation_score_boxplot(score_df, save_stem, title):
    plot_df = score_df.dropna(subset=["group", "Resistance_state_score"]).copy()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        data=plot_df,
        x="group",
        y="Resistance_state_score",
        order=ORDERCORR8_VALIDATION_GROUP_ORDER,
        palette=ORDERCORR8_VALIDATION_GROUP_COLORS,
        width=0.58,
        fliersize=0,
        linewidth=1.0,
        boxprops={"edgecolor": "#333333", "alpha": 0.78},
        whiskerprops={"color": "#333333", "linewidth": 1.0},
        capprops={"color": "#333333", "linewidth": 1.0},
        medianprops={"color": "#111111", "linewidth": 1.4},
        ax=ax,
    )
    point_collection_start = len(ax.collections)
    sns.stripplot(
        data=plot_df,
        x="group",
        y="Resistance_state_score",
        order=ORDERCORR8_VALIDATION_GROUP_ORDER,
        palette=ORDERCORR8_VALIDATION_GROUP_COLORS,
        jitter=0.18,
        size=4.3,
        edgecolor="#333333",
        alpha=0.88,
        ax=ax,
    )
    for collection in ax.collections[point_collection_start:]:
        collection.set_edgecolor("#333333")
        collection.set_linewidths(0.45)
    add_resistance_score_statannotations(
        ax,
        plot_df,
        x_col="group",
        order=ORDERCORR8_VALIDATION_GROUP_ORDER,
        pairs=[("AR", "IR")],
    )
    ax.axhline(0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.55)
    ax.set_ylabel("Resistance-state score")
    ax.set_xlabel("Group")
    ax.grid(axis="y", alpha=0.18, linewidth=0.7)
    sns.despine(ax=ax)
    fig.tight_layout()

    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return {"pdf": pdf_path, "png": png_path}


ordercorr8_class1_table = build_ordercorr8_transcript_table(class1_ar_up, "Class1")
ordercorr8_class3_table = build_ordercorr8_transcript_table(class3_ar_down, "Class3")
ordercorr8_transcript_table = pd.concat(
    [ordercorr8_class1_table, ordercorr8_class3_table],
    ignore_index=True,
)
ordercorr8_feature_summary_df = summarize_ordercorr8_features(
    ordercorr8_class1_table,
    ordercorr8_class3_table,
)
ordercorr8_class1_features = ordercorr8_class1_table.loc[
    ordercorr8_class1_table["Selected_ordercorr"],
    "Transcript",
].tolist()
ordercorr8_class3_features = ordercorr8_class3_table.loc[
    ordercorr8_class3_table["Selected_ordercorr"],
    "Transcript",
].tolist()

if len(ordercorr8_class1_features) == 0:
    raise ValueError("No order-correlation selected AR PRT transcripts were found.")
if len(ordercorr8_class3_features) == 0:
    raise ValueError("No order-correlation selected AR PST transcripts were found.")

ordercorr8_transcript_table.to_csv(
    os.path.join(OUT_DIR, "AR_DUT_PRT_PST_ordercorr1234_transcripts.tsv"),
    sep="\t",
    index=False,
)
ordercorr8_feature_summary_df.to_csv(
    os.path.join(OUT_DIR, "AR_DUT_PRT_PST_ordercorr1234_feature_summary.tsv"),
    sep="\t",
    index=False,
)

ordercorr8_discovery_scatter_df = build_class13_union_sample_scatter_df(
    pre_tu=preTU,
    post_tu=postTU,
    class1_features=ordercorr8_class1_features,
    class3_features=ordercorr8_class3_features,
    ar_sample_ids=ar_samples,
    ir_sample_ids=ir_samples,
)
ordercorr8_discovery_scatter_df = ordercorr8_discovery_scatter_df.dropna(
    subset=["Class3_mean_TU", "Class1_mean_TU"]
)
ordercorr8_discovery_scatter_df = add_interval_to_scatter_df(ordercorr8_discovery_scatter_df, sampleinfo)
ordercorr8_discovery_scatter_df.to_csv(
    os.path.join(OUT_DIR, "AR_DUT_PRT_PST_ordercorr1234_discovery_condition_meanTU_points.tsv"),
    sep="\t",
    index=False,
)
ordercorr8_discovery_summary_df = summarize_condition_mean_tu(ordercorr8_discovery_scatter_df)
ordercorr8_discovery_summary_df["Legend_label"] = ordercorr8_discovery_summary_df["Class"].map(
    ORDERCORR8_CLASS_LABELS
)
ordercorr8_discovery_summary_df.to_csv(
    os.path.join(OUT_DIR, "AR_DUT_PRT_PST_ordercorr1234_discovery_condition_meanTU_lineplot_summary.tsv"),
    sep="\t",
    index=False,
)
ordercorr8_discovery_overlay_summary_df = build_mean_tu_lineplot_overlay_summary(
    primary_summary_df=ordercorr8_discovery_summary_df,
    comparison_summary_df=ar_directional_lineplot_summary_df,
    primary_feature_set="State_ordered",
    comparison_feature_set="AR_directional_Class1_up_Class3_down",
    primary_label_map=ORDERCORR8_CLASS_LABELS,
    comparison_label_map=DIRECTIONAL_AR_CLASS_LABELS,
)
ordercorr8_discovery_overlay_summary_df.to_csv(
    os.path.join(
        OUT_DIR,
        "AR_DUT_PRT_PST_ordercorr1234_plus_directional_discovery_condition_meanTU_lineplot_summary.tsv",
    ),
    sep="\t",
    index=False,
)
ordercorr8_discovery_lineplot_paths = plot_discovery_state_ordered_plus_directional_lineplot(
    state_ordered_summary_df=ordercorr8_discovery_summary_df,
    directional_summary_df=ar_directional_lineplot_summary_df,
    save_stem="AR_DUT_PRT_PST_ordercorr1234_discovery_condition_meanTU_lineplot_CI",
    title="Discovery order-corr AR PRT/PST mean TU",
)

ordercorr8_discovery_score_df = add_ordercorr8_resistance_state_score(ordercorr8_discovery_scatter_df)
ordercorr8_discovery_score_df.to_csv(
    os.path.join(OUT_DIR, "AR_DUT_PRT_PST_ordercorr1234_discovery_resistance_state_score_points.tsv"),
    sep="\t",
    index=False,
)
ordercorr8_discovery_score_summary_df = summarize_resistance_state_score(ordercorr8_discovery_score_df)
ordercorr8_discovery_score_summary_df.to_csv(
    os.path.join(OUT_DIR, "AR_DUT_PRT_PST_ordercorr1234_discovery_resistance_state_score_summary.tsv"),
    sep="\t",
    index=False,
)
ordercorr8_discovery_score_boxplot_paths = plot_resistance_state_score_boxplot(
    ordercorr8_discovery_score_df,
    save_stem="AR_DUT_PRT_PST_ordercorr1234_discovery_resistance_state_score_boxplot",
    title="Discovery order-corr resistance-state score",
)

ordercorr8_val_tu = ordercorr8_prepare_validation_tu(
    "/home/jiye/jiye/copycomparison/GENCODEquant/SEV_pre/111_pre/forval_111_transcript_TPM.txt"
)
ordercorr8_val_clin = ordercorr8_prepare_validation_clinical(ordercorr8_val_tu)
ordercorr8_validation_point_df, ordercorr8_validation_match_summary_df = build_ordercorr8_validation_point_df(
    ordercorr8_val_tu,
    ordercorr8_val_clin,
    ordercorr8_class1_features,
    ordercorr8_class3_features,
)
ordercorr8_directional_validation_point_df, ordercorr8_directional_validation_match_summary_df = (
    build_ordercorr8_validation_point_df(
        ordercorr8_val_tu,
        ordercorr8_val_clin,
        class1_ar_up,
        class3_ar_down,
    )
)
ordercorr8_directional_validation_match_summary_df["Legend_label"] = (
    ordercorr8_directional_validation_match_summary_df["Class"].map(DIRECTIONAL_AR_CLASS_LABELS)
)
ordercorr8_directional_validation_match_summary_df["Feature_set"] = (
    "AR_directional_Class1_up_Class3_down"
)
ordercorr8_validation_point_df.to_csv(
    os.path.join(OUT_DIR, "VAL_noCR_AR_DUT_PRT_PST_ordercorr1234_group_meanTU_points.tsv"),
    sep="\t",
    index=False,
)
ordercorr8_directional_validation_point_df.to_csv(
    os.path.join(OUT_DIR, "VAL_noCR_AR_directional_Class1_up_Class3_down_group_meanTU_points.tsv"),
    sep="\t",
    index=False,
)
ordercorr8_validation_match_summary_df.to_csv(
    os.path.join(OUT_DIR, "VAL_noCR_AR_DUT_PRT_PST_ordercorr1234_matching_summary.tsv"),
    sep="\t",
    index=False,
)
ordercorr8_directional_validation_match_summary_df.to_csv(
    os.path.join(OUT_DIR, "VAL_noCR_AR_directional_Class1_up_Class3_down_matching_summary.tsv"),
    sep="\t",
    index=False,
)
ordercorr8_validation_summary_df = summarize_ordercorr8_validation_group_mean_tu(
    ordercorr8_validation_point_df
)
ordercorr8_directional_validation_summary_df = summarize_ordercorr8_validation_group_mean_tu(
    ordercorr8_directional_validation_point_df
)
ordercorr8_directional_validation_summary_df["Legend_label"] = (
    ordercorr8_directional_validation_summary_df["Class"].map(DIRECTIONAL_AR_CLASS_LABELS)
)
ordercorr8_validation_summary_df.to_csv(
    os.path.join(OUT_DIR, "VAL_noCR_AR_DUT_PRT_PST_ordercorr1234_group_meanTU_lineplot_summary.tsv"),
    sep="\t",
    index=False,
)
ordercorr8_directional_validation_summary_df.to_csv(
    os.path.join(OUT_DIR, "VAL_noCR_AR_directional_Class1_up_Class3_down_group_meanTU_lineplot_summary.tsv"),
    sep="\t",
    index=False,
)
ordercorr8_validation_overlay_summary_df = build_mean_tu_lineplot_overlay_summary(
    primary_summary_df=ordercorr8_validation_summary_df,
    comparison_summary_df=ordercorr8_directional_validation_summary_df,
    primary_feature_set="State_ordered",
    comparison_feature_set="AR_directional_Class1_up_Class3_down",
    primary_label_map=ORDERCORR8_CLASS_LABELS,
    comparison_label_map=DIRECTIONAL_AR_CLASS_LABELS,
)
ordercorr8_validation_overlay_summary_df.to_csv(
    os.path.join(
        OUT_DIR,
        "VAL_noCR_AR_DUT_PRT_PST_ordercorr1234_plus_directional_group_meanTU_lineplot_summary.tsv",
    ),
    sep="\t",
    index=False,
)
ordercorr8_validation_lineplot_paths = plot_ordercorr8_validation_lineplot(
    ordercorr8_validation_summary_df,
    save_stem="VAL_noCR_AR_DUT_PRT_PST_ordercorr1234_group_meanTU_lineplot_CI",
    title="Validation order-corr AR PRT/PST mean TU",
    comparison_summary_df=ordercorr8_directional_validation_summary_df,
    comparison_label_map=DIRECTIONAL_AR_CLASS_LABELS,
)

ordercorr8_validation_score_df = add_ordercorr8_resistance_state_score(ordercorr8_validation_point_df)
ordercorr8_validation_score_df.to_csv(
    os.path.join(OUT_DIR, "VAL_noCR_AR_DUT_PRT_PST_ordercorr1234_resistance_state_score_points.tsv"),
    sep="\t",
    index=False,
)
ordercorr8_validation_score_summary_df = summarize_ordercorr8_validation_score(
    ordercorr8_validation_score_df
)
ordercorr8_validation_score_summary_df.to_csv(
    os.path.join(OUT_DIR, "VAL_noCR_AR_DUT_PRT_PST_ordercorr1234_resistance_state_score_summary.tsv"),
    sep="\t",
    index=False,
)
ordercorr8_validation_score_boxplot_paths = plot_ordercorr8_validation_score_boxplot(
    ordercorr8_validation_score_df,
    save_stem="VAL_noCR_AR_DUT_PRT_PST_ordercorr1234_resistance_state_score_boxplot",
    title="Validation order-corr resistance-state score",
)

print("\n===== Order-correlation filtered AR PRT/PST discovery and validation plots =====")
print(ordercorr8_feature_summary_df.to_string(index=False))
print(ordercorr8_discovery_summary_df.to_string(index=False))
print(ordercorr8_discovery_score_summary_df.to_string(index=False))
print(ordercorr8_validation_match_summary_df.to_string(index=False))
print(ordercorr8_validation_summary_df.to_string(index=False))
print(ordercorr8_validation_score_summary_df.to_string(index=False))
print(f"Order-corr transcript table saved={os.path.join(OUT_DIR, 'AR_DUT_PRT_PST_ordercorr1234_transcripts.tsv')}")
print(f"Order-corr discovery lineplot saved={ordercorr8_discovery_lineplot_paths['pdf']}")
print(f"Order-corr discovery resistance score boxplot saved={ordercorr8_discovery_score_boxplot_paths['pdf']}")
print(f"Order-corr validation lineplot saved={ordercorr8_validation_lineplot_paths['pdf']}")
print(f"Order-corr validation resistance score boxplot saved={ordercorr8_validation_score_boxplot_paths['pdf']}")

# %%
#######^^ (8-1) Validation cohort no-CR BRCA-stratified AR/IR plots ########
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

VALIDATION8_GROUP_ORDER = ["AR", "IR"]
VALIDATION8_GROUP_COLORS = {
    "AR": AR_COLOR,
    "IR": IR_COLOR,
}
VALIDATION8_DIRECTIONAL_CLASS_LABELS = {
    "Class1": "PRT",
    "Class3": "PST",
}
VALIDATION8_BRCA_COHORTS = [
    ("All", "All", None),
    ("BRCAmt", "BRCAmt", 1),
    ("BRCAwt", "BRCAwt", 0),
]
VALIDATION8_SURVIVAL_PALETTE = {
    "High resistance-state score": "#EF463A",
    "Low resistance-state score": "#18A0DA",
}


def validation8_prepare_tu_from_transcript_tpm(transcript_tpm_path, min_detected_samples=None):
    val_transcript_tpm = pd.read_csv(transcript_tpm_path, sep="\t", index_col=0)
    if "gene_name" in val_transcript_tpm.columns:
        val_transcript_tpm = val_transcript_tpm.drop(columns=["gene_name"])

    val_transcript_tpm = val_transcript_tpm.apply(pd.to_numeric, errors="coerce").fillna(0)
    if min_detected_samples is not None:
        val_transcript_tpm = val_transcript_tpm.loc[
            (val_transcript_tpm > 0).sum(axis=1) >= min_detected_samples
        ]
    val_gene = val_transcript_tpm.index.to_series().astype(str).str.split("-", n=1).str[-1]
    val_gene_sum = val_transcript_tpm.groupby(val_gene).transform("sum").replace(0, np.nan)
    return val_transcript_tpm.div(val_gene_sum).fillna(0)


def validation8_prepare_clinical_info(val_tu_df):
    val_clin = pd.read_csv(
        "/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/112_PARPi_clinicalinfo.txt",
        sep="\t",
        index_col=0,
    )
    val_clin = val_clin.loc[val_clin.index.intersection(val_tu_df.columns)].copy()
    val_clin["BRCAmt_numeric"] = pd.to_numeric(val_clin["BRCAmt"], errors="coerce")
    val_clin["group"] = "i"
    val_clin.loc[(val_clin["response"] == 1) & (val_clin["recur"] == 1), "group"] = "AR"
    val_clin.loc[val_clin["response"] == 0, "group"] = "IR"
    val_clin.loc[(val_clin["response"] == 1) & (val_clin["recur"] == 0), "group"] = "CR"
    return val_clin


def validation8_match_transcript_ids_to_rows(transcript_ids, val_index):
    txid_to_val_rows = {}
    for val_row in pd.Index(val_index).astype(str):
        tx_id = val_row.split("-", 1)[0]
        txid_to_val_rows.setdefault(tx_id, []).append(val_row)

    matched_rows = []
    for transcript_id in transcript_ids:
        matched_rows.extend(txid_to_val_rows.get(str(transcript_id).split("-", 1)[0], []))

    return list(dict.fromkeys(matched_rows))


def validation8_match_transcript_ids_to_first_unique_rows(transcript_ids, val_index):
    val_index = pd.Index(val_index).astype(str)
    val_rows = val_index.tolist()
    val_row_set = set(val_rows)
    txid_to_first_val_row = {}
    for val_row in val_rows:
        tx_id = val_row.split("-", 1)[0]
        txid_to_first_val_row.setdefault(tx_id, val_row)

    used_rows = set()
    matched_rows = []
    for transcript_id in transcript_ids:
        transcript_id = str(transcript_id)
        if transcript_id in val_row_set:
            matched_row = transcript_id
        else:
            tx_id = transcript_id.split("-", 1)[0]
            matched_row = txid_to_first_val_row.get(tx_id)
        if matched_row is None or matched_row in used_rows:
            continue
        used_rows.add(matched_row)
        matched_rows.append(matched_row)
    return matched_rows


def validation8_build_group_mean_tu_df(
    val_tu_df,
    val_clin_df,
    class1_features,
    class3_features,
    cohort_id,
    cohort_label,
    brca_value,
    feature_set_label,
):
    class1_rows = validation8_match_transcript_ids_to_rows(class1_features, val_tu_df.index)
    class3_rows = validation8_match_transcript_ids_to_rows(class3_features, val_tu_df.index)
    if len(class1_rows) == 0:
        raise ValueError(f"No Class1 transcripts matched validation TU for {feature_set_label}.")
    if len(class3_rows) == 0:
        raise ValueError(f"No Class3 transcripts matched validation TU for {feature_set_label}.")

    if brca_value is None:
        plot_clin = val_clin_df.copy()
    else:
        plot_clin = val_clin_df.loc[val_clin_df["BRCAmt_numeric"] == brca_value].copy()
    plot_clin = plot_clin.loc[plot_clin["group"].isin(VALIDATION8_GROUP_ORDER)]
    common_samples = plot_clin.index.intersection(val_tu_df.columns)
    plot_clin = plot_clin.loc[common_samples].copy()

    class1_mean = val_tu_df.loc[class1_rows, plot_clin.index].mean(axis=0)
    class3_mean = val_tu_df.loc[class3_rows, plot_clin.index].mean(axis=0)
    point_df = pd.DataFrame(
        {
            "Cohort": cohort_id,
            "Cohort_label": cohort_label,
            "Feature_set": feature_set_label,
            "Sample": plot_clin.index,
            "group": plot_clin["group"].values,
            "PFS": plot_clin["PFS"].values,
            "recur": plot_clin["recur"].values,
            "BRCAmt_numeric": plot_clin["BRCAmt_numeric"].values,
            "Class1_mean_TU": class1_mean.reindex(plot_clin.index).values,
            "Class3_mean_TU": class3_mean.reindex(plot_clin.index).values,
        }
    )
    match_summary_df = pd.DataFrame(
        [
            {
                "Cohort": cohort_id,
                "Feature_set": feature_set_label,
                "Class": "Class1",
                "Discovery_transcript_count": len(class1_features),
                "Validation_matched_row_count": len(class1_rows),
            },
            {
                "Cohort": cohort_id,
                "Feature_set": feature_set_label,
                "Class": "Class3",
                "Discovery_transcript_count": len(class3_features),
                "Validation_matched_row_count": len(class3_rows),
            },
        ]
    )
    return point_df, match_summary_df


def validation8_build_single_class_mean_tu_df(
    val_tu_df,
    val_clin_df,
    feature_ids,
    class_label,
    value_col,
    cohort_id,
    cohort_label,
    brca_value,
    feature_set_label,
    match_mode="all_rows",
):
    if match_mode == "first_unique":
        matched_rows = validation8_match_transcript_ids_to_first_unique_rows(feature_ids, val_tu_df.index)
    elif match_mode == "all_rows":
        matched_rows = validation8_match_transcript_ids_to_rows(feature_ids, val_tu_df.index)
    else:
        raise ValueError("match_mode must be 'all_rows' or 'first_unique'.")
    if len(matched_rows) == 0:
        raise ValueError(f"No {class_label} transcripts matched validation TU for {feature_set_label}.")

    if brca_value is None:
        plot_clin = val_clin_df.copy()
    else:
        plot_clin = val_clin_df.loc[val_clin_df["BRCAmt_numeric"] == brca_value].copy()
    plot_clin = plot_clin.loc[plot_clin["group"].isin(VALIDATION8_GROUP_ORDER)]
    common_samples = plot_clin.index.intersection(val_tu_df.columns)
    plot_clin = plot_clin.loc[common_samples].copy()

    mean_tu = val_tu_df.loc[matched_rows, plot_clin.index].mean(axis=0)
    point_df = pd.DataFrame(
        {
            "Cohort": cohort_id,
            "Cohort_label": cohort_label,
            "Feature_set": feature_set_label,
            "Sample": plot_clin.index,
            "group": plot_clin["group"].values,
            "PFS": plot_clin["PFS"].values,
            "recur": plot_clin["recur"].values,
            "BRCAmt_numeric": plot_clin["BRCAmt_numeric"].values,
            value_col: mean_tu.reindex(plot_clin.index).values,
        }
    )
    match_summary_df = pd.DataFrame(
        [
            {
                "Cohort": cohort_id,
                "Feature_set": feature_set_label,
                "Class": class_label,
                "Match_mode": match_mode,
                "Discovery_transcript_count": len(feature_ids),
                "Validation_matched_row_count": len(matched_rows),
            }
        ]
    )
    return point_df, match_summary_df


def validation8_summarize_group_mean_tu(point_df):
    records = []
    for class_label, value_col in [("Class1", "Class1_mean_TU"), ("Class3", "Class3_mean_TU")]:
        for group in VALIDATION8_GROUP_ORDER:
            values = point_df.loc[point_df["group"] == group, value_col]
            mean_value, ci95, sample_count = mean_ci95(values)
            records.append(
                {
                    "Cohort": point_df["Cohort"].iloc[0],
                    "Cohort_label": point_df["Cohort_label"].iloc[0],
                    "Feature_set": point_df["Feature_set"].iloc[0],
                    "Class": class_label,
                    "group": group,
                    "Mean_TU": mean_value,
                    "CI95": ci95,
                    "Sample_count": sample_count,
                }
            )
    return pd.DataFrame(records)


def validation8_plot_group_mean_tu_lineplot(
    summary_df,
    save_stem,
    title,
    class_label_map,
    comparison_summary_df=None,
    comparison_label_map=None,
):
    x_positions = np.arange(len(VALIDATION8_GROUP_ORDER), dtype=float)
    line_specs = [
        (summary_df, "Class1", class_label_map.get("Class1", "Class1"), "-", "o", 2.0, 1.0),
        (summary_df, "Class3", class_label_map.get("Class3", "Class3"), "-", "o", 2.0, 1.0),
    ]
    if comparison_summary_df is not None:
        comparison_label_map = comparison_label_map or {}
        line_specs.extend(
            [
                (
                    comparison_summary_df,
                    "Class1",
                    comparison_label_map.get("Class1", "Class1"),
                    "--",
                    "s",
                    1.6,
                    0.82,
                ),
                (
                    comparison_summary_df,
                    "Class3",
                    comparison_label_map.get("Class3", "Class3"),
                    "--",
                    "s",
                    1.6,
                    0.82,
                ),
            ]
        )

    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    for summary_source_df, class_label, legend_label, linestyle, marker, linewidth, alpha in line_specs:
        plot_df = (
            summary_source_df[summary_source_df["Class"] == class_label]
            .set_index("group")
            .reindex(VALIDATION8_GROUP_ORDER)
            .reset_index()
        )
        ax.errorbar(
            x_positions,
            plot_df["Mean_TU"],
            yerr=plot_df["CI95"],
            color=LINE_CLASS_COLORS[class_label],
            marker=marker,
            markersize=5.6,
            markeredgecolor="#333333",
            markeredgewidth=0.55,
            linewidth=linewidth,
            linestyle=linestyle,
            capsize=3.5,
            elinewidth=1.0,
            alpha=alpha,
            label=legend_label,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(VALIDATION8_GROUP_ORDER)
    ax.set_ylabel("Mean TU")
    ax.set_xlabel("Group")
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.18, linewidth=0.7)
    ax.legend(frameon=False, loc="best", fontsize=10.5)
    sns.despine(ax=ax)
    fig.tight_layout()

    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return {"pdf": pdf_path, "png": png_path}


def validation8_summarize_resistance_state_score(score_df):
    records = []
    for group in VALIDATION8_GROUP_ORDER:
        values = pd.to_numeric(
            score_df.loc[score_df["group"] == group, "Resistance_state_score"],
            errors="coerce",
        ).dropna()
        mean_value, ci95, sample_count = mean_ci95(values)
        records.append(
            {
                "Cohort": score_df["Cohort"].iloc[0],
                "Cohort_label": score_df["Cohort_label"].iloc[0],
                "Feature_set": score_df["Feature_set"].iloc[0],
                "group": group,
                "Mean_resistance_state_score": mean_value,
                "CI95": ci95,
                "Median_resistance_state_score": float(values.median()) if not values.empty else np.nan,
                "Q1": float(values.quantile(0.25)) if not values.empty else np.nan,
                "Q3": float(values.quantile(0.75)) if not values.empty else np.nan,
                "Sample_count": sample_count,
            }
        )
    return pd.DataFrame(records)


def validation8_plot_resistance_state_score_boxplot(score_df, save_stem, title):
    plot_df = score_df.dropna(subset=["group", "Resistance_state_score"]).copy()

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(
        data=plot_df,
        x="group",
        y="Resistance_state_score",
        order=VALIDATION8_GROUP_ORDER,
        palette=VALIDATION8_GROUP_COLORS,
        width=0.58,
        fliersize=0,
        linewidth=1.0,
        boxprops={"edgecolor": "#333333", "alpha": 0.78},
        whiskerprops={"color": "#333333", "linewidth": 1.0},
        capprops={"color": "#333333", "linewidth": 1.0},
        medianprops={"color": "#111111", "linewidth": 1.4},
        ax=ax,
    )
    point_collection_start = len(ax.collections)
    sns.stripplot(
        data=plot_df,
        x="group",
        y="Resistance_state_score",
        order=VALIDATION8_GROUP_ORDER,
        palette=VALIDATION8_GROUP_COLORS,
        jitter=0.18,
        size=4.3,
        edgecolor="#333333",
        alpha=0.88,
        ax=ax,
    )
    for collection in ax.collections[point_collection_start:]:
        collection.set_edgecolor("#333333")
        collection.set_linewidths(0.45)
    add_resistance_score_statannotations(
        ax,
        plot_df,
        x_col="group",
        order=VALIDATION8_GROUP_ORDER,
        pairs=[("AR", "IR")],
    )

    ax.axhline(0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.55)
    ax.set_xticks(np.arange(len(VALIDATION8_GROUP_ORDER)))
    ax.set_xticklabels(VALIDATION8_GROUP_ORDER)
    ax.set_ylabel("Resistance-state score")
    ax.set_xlabel("Group")
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.18, linewidth=0.7)
    sns.despine(ax=ax)
    fig.tight_layout()

    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return {"pdf": pdf_path, "png": png_path}


def validation8_format_survival_pvalue(p_value):
    if pd.isna(p_value):
        return "NA"
    return f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.3f}"


def validation8_run_resistance_state_score_survival(score_df, save_stem, title):
    plot_df = score_df.copy()
    plot_df["PFS"] = pd.to_numeric(plot_df["PFS"], errors="coerce")
    plot_df["recur"] = pd.to_numeric(plot_df["recur"], errors="coerce")
    plot_df["Resistance_state_score"] = pd.to_numeric(
        plot_df["Resistance_state_score"],
        errors="coerce",
    )
    plot_df = plot_df.dropna(subset=["PFS", "recur", "Resistance_state_score"])
    plot_df = plot_df.loc[plot_df["recur"].isin([0, 1])].copy()
    plot_df["recur"] = plot_df["recur"].astype(int)

    base_record = {
        "Cohort": score_df["Cohort"].iloc[0] if not score_df.empty else "",
        "Cohort_label": score_df["Cohort_label"].iloc[0] if not score_df.empty else "",
        "Feature_set": score_df["Feature_set"].iloc[0] if not score_df.empty else "",
        "Survival_score": "Resistance_state_score",
    }

    if plot_df.empty or plot_df["Resistance_state_score"].nunique(dropna=True) < 2:
        record = {
            **base_record,
            "Sample_count": int(plot_df.shape[0]),
            "High_count": 0,
            "Low_count": 0,
            "Median_score": np.nan,
            "HR_high_vs_low": np.nan,
            "HR_CI95_low": np.nan,
            "HR_CI95_high": np.nan,
            "Cox_p": np.nan,
            "Logrank_p": np.nan,
            "Figure_pdf": "",
            "Figure_png": "",
            "Status": "skipped_no_valid_or_variable_score",
        }
        point_out_df = plot_df.copy()
        for key, value in base_record.items():
            point_out_df[key] = value
        return record, point_out_df

    median_score = plot_df["Resistance_state_score"].median()
    plot_df["Resistance_score_group"] = np.where(
        plot_df["Resistance_state_score"] >= median_score,
        "High resistance-state score",
        "Low resistance-state score",
    )
    plot_df["High_vs_Low"] = plot_df["Resistance_score_group"].eq(
        "High resistance-state score"
    ).astype(int)
    high_df = plot_df.loc[plot_df["Resistance_score_group"] == "High resistance-state score"].copy()
    low_df = plot_df.loc[plot_df["Resistance_score_group"] == "Low resistance-state score"].copy()

    hr = ci_low = ci_high = cox_p = np.nan
    cox_status = "ok"
    if plot_df["High_vs_Low"].nunique() == 2 and plot_df["recur"].sum() > 0:
        try:
            cox_df = plot_df[["PFS", "recur", "High_vs_Low"]].copy()
            cph = CoxPHFitter()
            cph.fit(cox_df, duration_col="PFS", event_col="recur")
            hr = float(cph.summary.loc["High_vs_Low", "exp(coef)"])
            ci_low = float(cph.summary.loc["High_vs_Low", "exp(coef) lower 95%"])
            ci_high = float(cph.summary.loc["High_vs_Low", "exp(coef) upper 95%"])
            cox_p = float(cph.summary.loc["High_vs_Low", "p"])
        except Exception as exc:
            cox_status = f"cox_failed: {exc}"
    else:
        cox_status = "cox_skipped_insufficient_score_groups_or_no_events"

    logrank_p = np.nan
    logrank_status = "ok"
    try:
        logrank_result = logrank_test(
            high_df["PFS"],
            low_df["PFS"],
            event_observed_A=high_df["recur"],
            event_observed_B=low_df["recur"],
        )
        logrank_p = float(logrank_result.p_value)
    except Exception as exc:
        logrank_status = f"logrank_failed: {exc}"

    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    kmf = KaplanMeierFitter()
    for score_group in ["High resistance-state score", "Low resistance-state score"]:
        mask = plot_df["Resistance_score_group"].eq(score_group)
        if not mask.any():
            continue
        kmf.fit(
            plot_df.loc[mask, "PFS"],
            event_observed=plot_df.loc[mask, "recur"],
            label=f"{score_group} (n={int(mask.sum())})",
        )
        kmf.plot_survival_function(
            ax=ax,
            color=VALIDATION8_SURVIVAL_PALETTE[score_group],
            ci_show=False,
        )

    hr_text = "NA" if pd.isna(hr) else f"{hr:.2f} ({ci_low:.2f}-{ci_high:.2f})"
    ax.text(
        0.55,
        0.48,
        f"HR = {hr_text}\n"
        f"Cox p = {validation8_format_survival_pvalue(cox_p)}\n"
        f"log-rank p = {validation8_format_survival_pvalue(logrank_p)}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
    )
    ax.set_xlabel("PFS")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper right")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    record = {
        **base_record,
        "Sample_count": int(plot_df.shape[0]),
        "High_count": int(high_df.shape[0]),
        "Low_count": int(low_df.shape[0]),
        "Median_score": float(median_score),
        "HR_high_vs_low": hr,
        "HR_CI95_low": ci_low,
        "HR_CI95_high": ci_high,
        "Cox_p": cox_p,
        "Logrank_p": logrank_p,
        "Figure_pdf": pdf_path,
        "Figure_png": png_path,
        "Status": "; ".join([cox_status, logrank_status]),
    }
    return record, plot_df


def validation8_run_mean_tu_survival(
    point_df,
    score_col,
    class_label,
    feature_id,
    feature_label,
    save_stem,
    title,
):
    plot_df = point_df.copy()
    plot_df["PFS"] = pd.to_numeric(plot_df["PFS"], errors="coerce")
    plot_df["recur"] = pd.to_numeric(plot_df["recur"], errors="coerce")
    plot_df[score_col] = pd.to_numeric(plot_df[score_col], errors="coerce")
    plot_df = plot_df.dropna(subset=["PFS", "recur", score_col])
    plot_df = plot_df.loc[plot_df["recur"].isin([0, 1])].copy()
    plot_df["recur"] = plot_df["recur"].astype(int)

    base_record = {
        "Cohort": point_df["Cohort"].iloc[0] if not point_df.empty else "",
        "Cohort_label": point_df["Cohort_label"].iloc[0] if not point_df.empty else "",
        "Feature_set": point_df["Feature_set"].iloc[0] if not point_df.empty else "",
        "Class": class_label,
        "Feature_id": feature_id,
        "Feature_label": feature_label,
        "Survival_score": score_col,
    }

    if plot_df.empty or plot_df[score_col].nunique(dropna=True) < 2:
        record = {
            **base_record,
            "Sample_count": int(plot_df.shape[0]),
            "High_count": 0,
            "Low_count": 0,
            "Median_score": np.nan,
            "HR_high_vs_low": np.nan,
            "HR_CI95_low": np.nan,
            "HR_CI95_high": np.nan,
            "Cox_p": np.nan,
            "Logrank_p": np.nan,
            "Figure_pdf": "",
            "Figure_png": "",
            "Status": "skipped_no_valid_or_variable_score",
        }
        point_out_df = plot_df.copy()
        for key, value in base_record.items():
            point_out_df[key] = value
        return record, point_out_df

    high_label = f"High {feature_label}"
    low_label = f"Low {feature_label}"
    median_score = plot_df[score_col].median()
    plot_df["Mean_TU_score_group"] = np.where(
        plot_df[score_col] >= median_score,
        high_label,
        low_label,
    )
    plot_df["High_vs_Low"] = plot_df["Mean_TU_score_group"].eq(high_label).astype(int)
    high_df = plot_df.loc[plot_df["Mean_TU_score_group"] == high_label].copy()
    low_df = plot_df.loc[plot_df["Mean_TU_score_group"] == low_label].copy()

    hr = ci_low = ci_high = cox_p = np.nan
    cox_status = "ok"
    if plot_df["High_vs_Low"].nunique() == 2 and plot_df["recur"].sum() > 0:
        try:
            cox_df = plot_df[["PFS", "recur", "High_vs_Low"]].copy()
            cph = CoxPHFitter()
            cph.fit(cox_df, duration_col="PFS", event_col="recur")
            hr = float(cph.summary.loc["High_vs_Low", "exp(coef)"])
            ci_low = float(cph.summary.loc["High_vs_Low", "exp(coef) lower 95%"])
            ci_high = float(cph.summary.loc["High_vs_Low", "exp(coef) upper 95%"])
            cox_p = float(cph.summary.loc["High_vs_Low", "p"])
        except Exception as exc:
            cox_status = f"cox_failed: {exc}"
    else:
        cox_status = "cox_skipped_insufficient_score_groups_or_no_events"

    logrank_p = np.nan
    logrank_status = "ok"
    try:
        logrank_result = logrank_test(
            high_df["PFS"],
            low_df["PFS"],
            event_observed_A=high_df["recur"],
            event_observed_B=low_df["recur"],
        )
        logrank_p = float(logrank_result.p_value)
    except Exception as exc:
        logrank_status = f"logrank_failed: {exc}"

    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    kmf = KaplanMeierFitter()
    for score_group, color in [(high_label, "#EF463A"), (low_label, "#18A0DA")]:
        mask = plot_df["Mean_TU_score_group"].eq(score_group)
        if not mask.any():
            continue
        kmf.fit(
            plot_df.loc[mask, "PFS"],
            event_observed=plot_df.loc[mask, "recur"],
            label=f"{score_group.split(' ', 1)[0]} (n={int(mask.sum())})",
        )
        kmf.plot_survival_function(
            ax=ax,
            color=color,
            ci_show=False,
        )

    hr_text = "NA" if pd.isna(hr) else f"{hr:.2f} ({ci_low:.2f}-{ci_high:.2f})"
    ax.text(
        0.52,
        0.46,
        f"HR = {hr_text}\n"
        f"Cox p = {validation8_format_survival_pvalue(cox_p)}\n"
        f"log-rank p = {validation8_format_survival_pvalue(logrank_p)}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("PFS")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    record = {
        **base_record,
        "Sample_count": int(plot_df.shape[0]),
        "High_count": int(high_df.shape[0]),
        "Low_count": int(low_df.shape[0]),
        "Median_score": float(median_score),
        "HR_high_vs_low": hr,
        "HR_CI95_low": ci_low,
        "HR_CI95_high": ci_high,
        "Cox_p": cox_p,
        "Logrank_p": logrank_p,
        "Figure_pdf": pdf_path,
        "Figure_png": png_path,
        "Status": "; ".join([cox_status, logrank_status]),
    }

    point_out_df = plot_df.copy()
    for key, value in base_record.items():
        point_out_df[key] = value
    return record, point_out_df


def validation8_save_no_cr_brca_outputs():
    validation8_transcript_tpm_path = (
        "/home/jiye/jiye/copycomparison/GENCODEquant/SEV_pre/111_pre/forval_111_transcript_TPM.txt"
    )
    val_tu_df = validation8_prepare_tu_from_transcript_tpm(validation8_transcript_tpm_path)
    legacy_val_tu_df = validation8_prepare_tu_from_transcript_tpm(
        validation8_transcript_tpm_path,
        min_detected_samples=15,
    )
    val_clin_df = validation8_prepare_clinical_info(val_tu_df)

    point_tables = []
    summary_tables = []
    match_tables = []
    score_point_tables = []
    score_summary_tables = []
    survival_records = []
    survival_point_tables = []
    mean_tu_survival_records = []
    mean_tu_survival_point_tables = []
    class_source_survival_records = []
    class_source_survival_point_tables = []
    class_source_match_tables = []
    figure_records = []
    class3_ar_dut_ordercorr_table = build_ordercorr8_transcript_table(class3_ar_dut, "Class3")
    class3_ar_dut_ordercorr_features = class3_ar_dut_ordercorr_table.loc[
        class3_ar_dut_ordercorr_table["Selected_ordercorr"],
        "Transcript",
    ].tolist()

    for cohort_id, cohort_label, brca_value in VALIDATION8_BRCA_COHORTS:
        ar_point_df, ar_match_df = validation8_build_group_mean_tu_df(
            val_tu_df=val_tu_df,
            val_clin_df=val_clin_df,
            class1_features=class1_ar_up,
            class3_features=class3_ar_down,
            cohort_id=cohort_id,
            cohort_label=cohort_label,
            brca_value=brca_value,
            feature_set_label="AR_directional_Class1_up_Class3_down",
        )
        ar_summary_df = validation8_summarize_group_mean_tu(ar_point_df)
        ar_paths = validation8_plot_group_mean_tu_lineplot(
            summary_df=ar_summary_df,
            save_stem=f"VAL_noCR_{cohort_id}_AR_directional_Class1_up_Class3_down_group_meanTU_lineplot_CI",
            title=cohort_label,
            class_label_map=VALIDATION8_DIRECTIONAL_CLASS_LABELS,
        )
        directional_score_df = add_resistance_state_score(ar_point_df)
        directional_score_summary_df = validation8_summarize_resistance_state_score(
            directional_score_df
        )
        directional_score_paths = validation8_plot_resistance_state_score_boxplot(
            score_df=directional_score_df,
            save_stem=f"VAL_noCR_{cohort_id}_PRT_PST_resistance_state_score_boxplot",
            title=cohort_label,
        )
        directional_survival_record, directional_survival_point_df = (
            validation8_run_resistance_state_score_survival(
                score_df=directional_score_df,
                save_stem=f"VAL_noCR_{cohort_id}_PRT_PST_resistance_state_score_survival",
                title=f"Validation {cohort_label} PRT/PST resistance-state score survival",
            )
        )

        state_ordered_point_df, state_ordered_match_df = validation8_build_group_mean_tu_df(
            val_tu_df=val_tu_df,
            val_clin_df=val_clin_df,
            class1_features=ordercorr8_class1_features,
            class3_features=ordercorr8_class3_features,
            cohort_id=cohort_id,
            cohort_label=cohort_label,
            brca_value=brca_value,
            feature_set_label="AR_directional_ordercorr1234",
        )
        state_ordered_summary_df = validation8_summarize_group_mean_tu(state_ordered_point_df)
        state_ordered_paths = validation8_plot_group_mean_tu_lineplot(
            summary_df=state_ordered_summary_df,
            save_stem=f"VAL_noCR_{cohort_id}_AR_directional_ordercorr1234_group_meanTU_lineplot_CI",
            title=cohort_label,
            class_label_map=ORDERCORR8_CLASS_LABELS,
            comparison_summary_df=ar_summary_df,
            comparison_label_map=DIRECTIONAL_AR_CLASS_LABELS,
        )

        state_ordered_score_df = add_ordercorr8_resistance_state_score(state_ordered_point_df)
        state_ordered_score_summary_df = validation8_summarize_resistance_state_score(
            state_ordered_score_df
        )
        state_ordered_score_paths = validation8_plot_resistance_state_score_boxplot(
            score_df=state_ordered_score_df,
            save_stem=f"VAL_noCR_{cohort_id}_AR_directional_ordercorr1234_resistance_state_score_boxplot",
            title=cohort_label,
        )
        state_ordered_survival_record, state_ordered_survival_point_df = (
            validation8_run_resistance_state_score_survival(
                score_df=state_ordered_score_df,
                save_stem=f"VAL_noCR_{cohort_id}_AR_directional_ordercorr1234_resistance_state_score_survival",
                title=f"Validation {cohort_label} state-ordered resistance-state score survival",
            )
        )

        mean_tu_survival_specs = [
            (
                ar_point_df,
                "Class1_mean_TU",
                "Class1",
                "PRT",
                "PRT",
            ),
            (
                ar_point_df,
                "Class3_mean_TU",
                "Class3",
                "PST",
                "PST",
            ),
            (
                state_ordered_point_df,
                "Class1_mean_TU",
                "Class1",
                "State_ordered_PRT",
                "State-ordered PRT",
            ),
            (
                state_ordered_point_df,
                "Class3_mean_TU",
                "Class3",
                "State_ordered_PST",
                "State-ordered PST",
            ),
        ]
        mean_tu_survival_plot_records = []
        for (
            feature_point_df,
            score_col,
            class_label,
            feature_id,
            feature_label,
        ) in mean_tu_survival_specs:
            mean_tu_survival_record, mean_tu_survival_point_df = validation8_run_mean_tu_survival(
                point_df=feature_point_df,
                score_col=score_col,
                class_label=class_label,
                feature_id=feature_id,
                feature_label=feature_label,
                save_stem=f"VAL_noCR_{cohort_id}_{feature_id}_meanTU_median_score_survival",
                title=f"{feature_label} ({cohort_label})",
            )
            mean_tu_survival_records.append(mean_tu_survival_record)
            mean_tu_survival_plot_records.append(mean_tu_survival_record)
            if not mean_tu_survival_point_df.empty:
                mean_tu_survival_point_tables.append(mean_tu_survival_point_df)

        class_source_survival_specs = [
            (
                "Class1_deltaTU_gt0_DUT",
                "Delta TU > 0.05 Class1 DUT",
                class1_ar_up,
                val_tu_df,
                "all_rows",
                "Class1",
                "Class1_mean_TU",
            ),
            (
                "Class1_all_DUT",
                "All Class1 DUT",
                class1_ar_dut,
                val_tu_df,
                "all_rows",
                "Class1",
                "Class1_mean_TU",
            ),
            (
                "Class3_deltaTU_lt0_DUT",
                "Delta TU < -0.05 Class3 DUT",
                class3_ar_down,
                val_tu_df,
                "all_rows",
                "Class3",
                "Class3_mean_TU",
            ),
            (
                "Class3_deltaTU_lt0_DUT_legacy_202605",
                "Delta TU < -0.05 Class3 DUT legacy",
                class3_ar_down,
                legacy_val_tu_df,
                "first_unique",
                "Class3",
                "Class3_mean_TU",
            ),
            (
                "Class3_all_DUT",
                "All Class3 DUT",
                class3_ar_dut,
                val_tu_df,
                "all_rows",
                "Class3",
                "Class3_mean_TU",
            ),
            (
                "Class3_order_filtered_DUT",
                "Order-filtered Class3 DUT",
                class3_ar_dut_ordercorr_features,
                val_tu_df,
                "all_rows",
                "Class3",
                "Class3_mean_TU",
            ),
        ]
        class_source_survival_plot_records = []
        for (
            feature_id,
            feature_label,
            feature_ids,
            feature_val_tu_df,
            match_mode,
            class_label,
            value_col,
        ) in class_source_survival_specs:
            class_point_df, class_match_df = validation8_build_single_class_mean_tu_df(
                val_tu_df=feature_val_tu_df,
                val_clin_df=val_clin_df,
                feature_ids=feature_ids,
                class_label=class_label,
                value_col=value_col,
                cohort_id=cohort_id,
                cohort_label=cohort_label,
                brca_value=brca_value,
                feature_set_label=feature_id,
                match_mode=match_mode,
            )
            class_survival_record, class_survival_point_df = validation8_run_mean_tu_survival(
                point_df=class_point_df,
                score_col=value_col,
                class_label=class_label,
                feature_id=feature_id,
                feature_label=feature_label,
                save_stem=f"VAL_noCR_{cohort_id}_{feature_id}_meanTU_median_score_survival",
                title=f"{feature_label} ({cohort_label})",
            )
            class_source_match_tables.append(class_match_df)
            class_source_survival_records.append(class_survival_record)
            class_source_survival_plot_records.append(class_survival_record)
            if not class_survival_point_df.empty:
                class_source_survival_point_tables.append(class_survival_point_df)

        point_tables.extend([ar_point_df, state_ordered_point_df])
        summary_tables.extend([ar_summary_df, state_ordered_summary_df])
        match_tables.extend([ar_match_df, state_ordered_match_df])
        score_point_tables.extend([directional_score_df, state_ordered_score_df])
        score_summary_tables.extend(
            [directional_score_summary_df, state_ordered_score_summary_df]
        )
        survival_records.extend(
            [directional_survival_record, state_ordered_survival_record]
        )
        if not directional_survival_point_df.empty:
            survival_point_tables.append(directional_survival_point_df)
        if not state_ordered_survival_point_df.empty:
            survival_point_tables.append(state_ordered_survival_point_df)
        figure_records.extend(
            [
                {"Cohort": cohort_id, "Plot": "PRT_PST_lineplot", **ar_paths},
                {"Cohort": cohort_id, "Plot": "State_ordered_lineplot", **state_ordered_paths},
                {
                    "Cohort": cohort_id,
                    "Plot": "PRT_PST_resistance_state_score_boxplot",
                    **directional_score_paths,
                },
                {
                    "Cohort": cohort_id,
                    "Plot": "PRT_PST_resistance_state_score_survival",
                    "pdf": directional_survival_record["Figure_pdf"],
                    "png": directional_survival_record["Figure_png"],
                },
                {
                    "Cohort": cohort_id,
                    "Plot": "State_ordered_resistance_state_score_boxplot",
                    **state_ordered_score_paths,
                },
                {
                    "Cohort": cohort_id,
                    "Plot": "State_ordered_resistance_state_score_survival",
                    "pdf": state_ordered_survival_record["Figure_pdf"],
                    "png": state_ordered_survival_record["Figure_png"],
                },
            ]
        )
        figure_records.extend(
            [
                {
                    "Cohort": cohort_id,
                    "Plot": f"{record['Feature_id']}_mean_TU_survival",
                    "pdf": record["Figure_pdf"],
                    "png": record["Figure_png"],
                }
                for record in mean_tu_survival_plot_records
            ]
        )
        figure_records.extend(
            [
                {
                    "Cohort": cohort_id,
                    "Plot": f"{record['Feature_id']}_source_check_survival",
                    "pdf": record["Figure_pdf"],
                    "png": record["Figure_png"],
                }
                for record in class_source_survival_plot_records
            ]
        )

    point_table = pd.concat(point_tables, ignore_index=True)
    summary_table = pd.concat(summary_tables, ignore_index=True)
    match_table = pd.concat(match_tables, ignore_index=True)
    score_point_table = pd.concat(score_point_tables, ignore_index=True)
    score_summary_table = pd.concat(score_summary_tables, ignore_index=True)
    survival_summary_table = pd.DataFrame(survival_records)
    if survival_point_tables:
        survival_point_table = pd.concat(survival_point_tables, ignore_index=True)
    else:
        survival_point_table = pd.DataFrame()
    mean_tu_survival_summary_table = pd.DataFrame(mean_tu_survival_records)
    if mean_tu_survival_point_tables:
        mean_tu_survival_point_table = pd.concat(mean_tu_survival_point_tables, ignore_index=True)
    else:
        mean_tu_survival_point_table = pd.DataFrame()
    class_source_survival_summary_table = pd.DataFrame(class_source_survival_records)
    if class_source_survival_point_tables:
        class_source_survival_point_table = pd.concat(
            class_source_survival_point_tables,
            ignore_index=True,
        )
    else:
        class_source_survival_point_table = pd.DataFrame()
    if class_source_match_tables:
        class_source_match_table = pd.concat(class_source_match_tables, ignore_index=True)
    else:
        class_source_match_table = pd.DataFrame()
    figure_table = pd.DataFrame(figure_records)

    output_tables = {
        "points": (
            point_table,
            os.path.join(OUT_DIR, "VAL_noCR_All_BRCAmt_BRCAwt_PRT_PST_directional_and_ordercorr1234_group_meanTU_points.tsv"),
        ),
        "summary": (
            summary_table,
            os.path.join(OUT_DIR, "VAL_noCR_All_BRCAmt_BRCAwt_PRT_PST_directional_and_ordercorr1234_group_meanTU_summary.tsv"),
        ),
        "matching": (
            match_table,
            os.path.join(OUT_DIR, "VAL_noCR_All_BRCAmt_BRCAwt_PRT_PST_directional_and_ordercorr1234_matching_summary.tsv"),
        ),
        "score_points": (
            score_point_table,
            os.path.join(OUT_DIR, "VAL_noCR_All_BRCAmt_BRCAwt_PRT_PST_directional_and_ordercorr1234_resistance_state_score_points.tsv"),
        ),
        "score_summary": (
            score_summary_table,
            os.path.join(OUT_DIR, "VAL_noCR_All_BRCAmt_BRCAwt_PRT_PST_directional_and_ordercorr1234_resistance_state_score_summary.tsv"),
        ),
        "figures": (
            figure_table,
            os.path.join(OUT_DIR, "VAL_noCR_All_BRCAmt_BRCAwt_PRT_PST_directional_and_ordercorr1234_resistance_figure_paths.tsv"),
        ),
        "survival_summary": (
            survival_summary_table,
            os.path.join(OUT_DIR, "VAL_noCR_All_BRCAmt_BRCAwt_PRT_PST_directional_and_ordercorr1234_resistance_state_score_survival_summary.tsv"),
        ),
        "survival_points": (
            survival_point_table,
            os.path.join(OUT_DIR, "VAL_noCR_All_BRCAmt_BRCAwt_PRT_PST_directional_and_ordercorr1234_resistance_state_score_survival_points.tsv"),
        ),
        "mean_tu_survival_summary": (
            mean_tu_survival_summary_table,
            os.path.join(OUT_DIR, "VAL_noCR_All_BRCAmt_BRCAwt_PRT_PST_State_ordered_meanTU_survival_summary.tsv"),
        ),
        "mean_tu_survival_points": (
            mean_tu_survival_point_table,
            os.path.join(OUT_DIR, "VAL_noCR_All_BRCAmt_BRCAwt_PRT_PST_State_ordered_meanTU_survival_points.tsv"),
        ),
        "class_source_survival_summary": (
            class_source_survival_summary_table,
            os.path.join(OUT_DIR, "VAL_noCR_All_BRCAmt_BRCAwt_Class1_Class3_source_comparison_meanTU_survival_summary.tsv"),
        ),
        "class_source_survival_points": (
            class_source_survival_point_table,
            os.path.join(OUT_DIR, "VAL_noCR_All_BRCAmt_BRCAwt_Class1_Class3_source_comparison_meanTU_survival_points.tsv"),
        ),
        "class_source_matching": (
            class_source_match_table,
            os.path.join(OUT_DIR, "VAL_noCR_All_BRCAmt_BRCAwt_Class1_Class3_source_comparison_matching_summary.tsv"),
        ),
        "class3_source_order_filter_table": (
            class3_ar_dut_ordercorr_table,
            os.path.join(OUT_DIR, "VAL_Class3_all_DUT_ordercorr1234_source_check_transcripts.tsv"),
        ),
    }
    for table, path in output_tables.values():
        table.to_csv(path, sep="\t", index=False)

    print("\n===== Validation no-CR BRCA-stratified AR/IR plots =====")
    print(match_table.to_string(index=False))
    print(summary_table.to_string(index=False))
    print(score_summary_table.to_string(index=False))
    print(survival_summary_table.to_string(index=False))
    print(mean_tu_survival_summary_table.to_string(index=False))
    print(class_source_survival_summary_table.to_string(index=False))
    for output_name, (_, path) in output_tables.items():
        print(f"Validation {output_name} saved={path}")

    return output_tables, figure_table


validation8_output_tables, validation8_figure_table = validation8_save_no_cr_brca_outputs()
if os.environ.get("FIG4_STOP_AFTER_8_1") == "1":
    sys.exit(0)

# %%
#######^^ (8-2) Discovery AR resistance-state score change vs interval ########
def build_discovery_ar_resistance_delta_interval_df(score_df):
    ar_score_df = score_df.loc[score_df["Condition"].isin(["AR_pre", "AR_post"])].copy()
    score_wide = ar_score_df.pivot_table(
        index="Sample",
        columns="Condition",
        values="Resistance_state_score",
        aggfunc="mean",
    )
    interval_by_sample = ar_score_df.groupby("Sample")["Interval"].first()

    delta_df = pd.DataFrame(index=score_wide.index)
    delta_df["Interval"] = pd.to_numeric(interval_by_sample.reindex(delta_df.index), errors="coerce")
    delta_df["AR_pre_resistance_state_score"] = score_wide.get("AR_pre")
    delta_df["AR_post_resistance_state_score"] = score_wide.get("AR_post")
    delta_df["Delta_resistance_state_score"] = (
        delta_df["AR_post_resistance_state_score"]
        - delta_df["AR_pre_resistance_state_score"]
    )

    for value_col, output_col in [
        ("upregulated_AR_PRT_mean_TU", "Delta_upregulated_AR_PRT_mean_TU"),
        ("downregulated_AR_PST_mean_TU", "Delta_downregulated_AR_PST_mean_TU"),
        ("z_upregulated_AR_PRT_mean_TU", "Delta_z_upregulated_AR_PRT_mean_TU"),
        ("z_downregulated_AR_PST_mean_TU", "Delta_z_downregulated_AR_PST_mean_TU"),
    ]:
        value_wide = ar_score_df.pivot_table(
            index="Sample",
            columns="Condition",
            values=value_col,
            aggfunc="mean",
        )
        delta_df[output_col] = value_wide.get("AR_post") - value_wide.get("AR_pre")

    delta_df = delta_df.reset_index()
    return delta_df


def summarize_discovery_ar_resistance_delta_interval(delta_df):
    plot_df = delta_df.dropna(subset=["Interval", "Delta_resistance_state_score"]).copy()
    records = [
        {
            "Metric": "sample_count_total",
            "N": int(delta_df.shape[0]),
            "Statistic": np.nan,
            "P_value": np.nan,
        },
        {
            "Metric": "sample_count_used_for_correlation",
            "N": int(plot_df.shape[0]),
            "Statistic": np.nan,
            "P_value": np.nan,
        },
    ]

    if (
        plot_df.shape[0] >= 3
        and plot_df["Interval"].nunique(dropna=True) > 1
        and plot_df["Delta_resistance_state_score"].nunique(dropna=True) > 1
    ):
        pearson_r, pearson_p = stats.pearsonr(
            plot_df["Interval"],
            plot_df["Delta_resistance_state_score"],
        )
        spearman_rho, spearman_p = stats.spearmanr(
            plot_df["Interval"],
            plot_df["Delta_resistance_state_score"],
        )
        slope, intercept, slope_r, slope_p, slope_stderr = stats.linregress(
            plot_df["Interval"],
            plot_df["Delta_resistance_state_score"],
        )
    else:
        pearson_r, pearson_p = np.nan, np.nan
        spearman_rho, spearman_p = np.nan, np.nan
        slope, intercept, slope_r, slope_p, slope_stderr = np.nan, np.nan, np.nan, np.nan, np.nan

    records.extend(
        [
            {
                "Metric": "Pearson_interval_vs_delta_score",
                "N": int(plot_df.shape[0]),
                "Statistic": float(pearson_r) if pd.notna(pearson_r) else np.nan,
                "P_value": float(pearson_p) if pd.notna(pearson_p) else np.nan,
            },
            {
                "Metric": "Spearman_interval_vs_delta_score",
                "N": int(plot_df.shape[0]),
                "Statistic": float(spearman_rho) if pd.notna(spearman_rho) else np.nan,
                "P_value": float(spearman_p) if pd.notna(spearman_p) else np.nan,
            },
            {
                "Metric": "Linear_regression_slope",
                "N": int(plot_df.shape[0]),
                "Statistic": float(slope) if pd.notna(slope) else np.nan,
                "P_value": float(slope_p) if pd.notna(slope_p) else np.nan,
            },
            {
                "Metric": "Linear_regression_intercept",
                "N": int(plot_df.shape[0]),
                "Statistic": float(intercept) if pd.notna(intercept) else np.nan,
                "P_value": np.nan,
            },
            {
                "Metric": "Linear_regression_slope_stderr",
                "N": int(plot_df.shape[0]),
                "Statistic": float(slope_stderr) if pd.notna(slope_stderr) else np.nan,
                "P_value": np.nan,
            },
        ]
    )
    return pd.DataFrame(records)


def plot_discovery_ar_resistance_delta_interval(delta_df, summary_df, save_stem):
    plot_df = delta_df.dropna(subset=["Interval", "Delta_resistance_state_score"]).copy()
    pearson_row = summary_df.loc[summary_df["Metric"] == "Pearson_interval_vs_delta_score"]
    spearman_row = summary_df.loc[summary_df["Metric"] == "Spearman_interval_vs_delta_score"]

    pearson_text = "Pearson r=NA, p=NA"
    spearman_text = "Spearman rho=NA, p=NA"
    if not pearson_row.empty and pd.notna(pearson_row.iloc[0]["Statistic"]):
        pearson_text = (
            f"Pearson r={pearson_row.iloc[0]['Statistic']:.2f}, "
            f"p={pearson_row.iloc[0]['P_value']:.3g}"
        )
    if not spearman_row.empty and pd.notna(spearman_row.iloc[0]["Statistic"]):
        spearman_text = (
            f"Spearman rho={spearman_row.iloc[0]['Statistic']:.2f}, "
            f"p={spearman_row.iloc[0]['P_value']:.3g}"
        )

    fig, ax = plt.subplots(figsize=(6, 4))
    if (
        plot_df.shape[0] >= 3
        and plot_df["Interval"].nunique(dropna=True) > 1
        and plot_df["Delta_resistance_state_score"].nunique(dropna=True) > 1
    ):
        sns.regplot(
            data=plot_df,
            x="Interval",
            y="Delta_resistance_state_score",
            scatter_kws={
                "s": 54,
                "color": AR_COLOR,
                "edgecolor": "#333333",
                "linewidths": 0.45,
                "alpha": 0.9,
            },
            line_kws={"color": "#B85C00", "linewidth": 2.0},
            ci=95,
            ax=ax,
        )
    else:
        ax.scatter(
            plot_df["Interval"],
            plot_df["Delta_resistance_state_score"],
            s=54,
            color=AR_COLOR,
            edgecolor="#333333",
            linewidth=0.45,
            alpha=0.9,
        )

    ax.axhline(0, color="#666666", linewidth=1.0, linestyle="--", alpha=0.55)
    ax.set_xlabel("Treatment interval (days)")
    ax.set_ylabel("Delta resistance-state score\n(AR post - AR pre)")
    ax.set_title("", fontsize=13, fontweight="bold")
    ax.text(
        0.04,
        0.96,
        f"n={plot_df.shape[0]}\n{pearson_text}\n{spearman_text}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
    )
    ax.grid(alpha=0.18, linewidth=0.7)
    sns.despine(ax=ax)
    fig.tight_layout()

    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return {"pdf": pdf_path, "png": png_path}


discovery_ar_resistance_delta_interval_df = build_discovery_ar_resistance_delta_interval_df(
    resistance_state_score_df
)
discovery_ar_resistance_delta_interval_df.to_csv(
    os.path.join(OUT_DIR, "Discovery_AR_resistance_state_score_delta_vs_interval_points.tsv"),
    sep="\t",
    index=False,
)
discovery_ar_resistance_delta_interval_summary_df = summarize_discovery_ar_resistance_delta_interval(
    discovery_ar_resistance_delta_interval_df
)
discovery_ar_resistance_delta_interval_summary_df.to_csv(
    os.path.join(OUT_DIR, "Discovery_AR_resistance_state_score_delta_vs_interval_summary.tsv"),
    sep="\t",
    index=False,
)
discovery_ar_resistance_delta_interval_plot_paths = plot_discovery_ar_resistance_delta_interval(
    discovery_ar_resistance_delta_interval_df,
    discovery_ar_resistance_delta_interval_summary_df,
    save_stem="Discovery_AR_resistance_state_score_delta_vs_interval_scatter",
)

print("\n===== Discovery AR resistance-state score change vs treatment interval =====")
print(discovery_ar_resistance_delta_interval_summary_df.to_string(index=False))
print(f"Discovery AR resistance score delta point table saved={os.path.join(OUT_DIR, 'Discovery_AR_resistance_state_score_delta_vs_interval_points.tsv')}")
print(f"Discovery AR resistance score delta interval summary saved={os.path.join(OUT_DIR, 'Discovery_AR_resistance_state_score_delta_vs_interval_summary.tsv')}")
print(f"Discovery AR resistance score delta interval plot saved={discovery_ar_resistance_delta_interval_plot_paths['pdf']}")

# %%
#######^^ (9) AR-upregulated DUT paired Class1 vs Class3 Wilcoxon tests ########
EXPECTED_PAIRWISE_COMPARISON = {
    "AR_pre": {
        "Expected_relation": "Class3 > Class1",
        "Expected_p_column": "p_Class3_greater_Class1",
        "Expected_p_note": "p < 0.05 expected",
    },
    "IR_pre": {
        "Expected_relation": "Class3 = Class1",
        "Expected_p_column": "p_two_sided",
        "Expected_p_note": "p approx 1 expected",
    },
    "IR_post": {
        "Expected_relation": "Class1 > Class3",
        "Expected_p_column": "p_Class1_greater_Class3",
        "Expected_p_note": "p < 0.05 expected",
    },
    "AR_post": {
        "Expected_relation": "Class1 >> Class3",
        "Expected_p_column": "p_Class1_greater_Class3",
        "Expected_p_note": "p < 0.001 expected",
    },
}


def safe_paired_wilcoxon(class1_values, class3_values, alternative):
    paired_df = pd.DataFrame(
        {
            "Class1": pd.to_numeric(pd.Series(class1_values), errors="coerce"),
            "Class3": pd.to_numeric(pd.Series(class3_values), errors="coerce"),
        }
    ).dropna()
    if paired_df.empty:
        return np.nan, np.nan

    diff = paired_df["Class1"] - paired_df["Class3"]
    if np.allclose(diff.to_numpy(dtype=float), 0):
        return 0.0, 1.0

    result = stats.wilcoxon(
        paired_df["Class1"],
        paired_df["Class3"],
        alternative=alternative,
        zero_method="wilcox",
    )
    return float(result.statistic), float(result.pvalue)


def paired_class1_class3_wilcoxon_by_condition(scatter_df):
    records = []
    for condition in STACKED_CONDITION_ORDER:
        plot_df = scatter_df.loc[scatter_df["Condition"] == condition].copy()
        class1_values = plot_df["Class1_mean_TU"]
        class3_values = plot_df["Class3_mean_TU"]
        diff = pd.to_numeric(class1_values, errors="coerce") - pd.to_numeric(class3_values, errors="coerce")
        diff = diff.dropna()

        stat_two, p_two = safe_paired_wilcoxon(class1_values, class3_values, alternative="two-sided")
        stat_c1_gt, p_c1_gt = safe_paired_wilcoxon(class1_values, class3_values, alternative="greater")
        stat_c3_gt, p_c3_gt = safe_paired_wilcoxon(class1_values, class3_values, alternative="less")

        expected = EXPECTED_PAIRWISE_COMPARISON[condition]
        p_values = {
            "p_two_sided": p_two,
            "p_Class1_greater_Class3": p_c1_gt,
            "p_Class3_greater_Class1": p_c3_gt,
        }
        expected_p = p_values[expected["Expected_p_column"]]

        mean_diff = float(diff.mean()) if not diff.empty else np.nan
        median_diff = float(diff.median()) if not diff.empty else np.nan
        observed_relation = (
            "Class1 > Class3"
            if mean_diff > 0
            else "Class3 > Class1"
            if mean_diff < 0
            else "Class1 = Class3"
        )

        records.append(
            {
                "Condition": condition,
                "N_pairs": int(diff.shape[0]),
                "Class1_mean": float(pd.to_numeric(class1_values, errors="coerce").mean()),
                "Class3_mean": float(pd.to_numeric(class3_values, errors="coerce").mean()),
                "Mean_Class1_minus_Class3": mean_diff,
                "Median_Class1_minus_Class3": median_diff,
                "Observed_relation_by_mean": observed_relation,
                "Expected_relation": expected["Expected_relation"],
                "Expected_p_note": expected["Expected_p_note"],
                "Wilcoxon_stat_two_sided": stat_two,
                "p_two_sided": p_two,
                "Wilcoxon_stat_Class1_greater_Class3": stat_c1_gt,
                "p_Class1_greater_Class3": p_c1_gt,
                "Wilcoxon_stat_Class3_greater_Class1": stat_c3_gt,
                "p_Class3_greater_Class1": p_c3_gt,
                "Expected_p_column": expected["Expected_p_column"],
                "Expected_direction_p_value": expected_p,
            }
        )
    return pd.DataFrame(records)


ar_up_class1_vs_class3_wilcoxon_df = paired_class1_class3_wilcoxon_by_condition(ar_up_scatter_df)
ar_up_class1_vs_class3_wilcoxon_df.to_csv(
    os.path.join(OUT_DIR, "AR_up_DUT_Class1_vs_Class3_paired_Wilcoxon_by_condition.tsv"),
    sep="\t",
    index=False,
)

print("\n===== AR-upregulated DUT paired Class1 vs Class3 Wilcoxon by condition =====")
print(ar_up_class1_vs_class3_wilcoxon_df.to_string(index=False))
print(f"AR-up paired Wilcoxon table saved={os.path.join(OUT_DIR, 'AR_up_DUT_Class1_vs_Class3_paired_Wilcoxon_by_condition.tsv')}")

# %%
#######^^ (10) AR-directional DUT paired Class1-up vs Class3-down Wilcoxon tests ########
ar_down_dut_clean = {
    str(transcript_gene).split("-", 1)[0]
    for transcript_gene in AR_dut.loc[
        (AR_dut["p_value"] < 0.05) & (AR_dut["delta_TU"] < -0.05)
    ].index
}
class3_ar_down = sorted(ar_down_dut_clean & class3_clean & set(preTU.index) & set(postTU.index))

if len(class3_ar_down) == 0:
    raise ValueError("No Class3 AR-downregulated DUT transcripts were found in preTU/postTU.")

ar_directional_scatter_df = build_class13_union_sample_scatter_df(
    pre_tu=preTU,
    post_tu=postTU,
    class1_features=class1_ar_up,
    class3_features=class3_ar_down,
    ar_sample_ids=ar_samples,
    ir_sample_ids=ir_samples,
)
ar_directional_scatter_df = ar_directional_scatter_df.dropna(subset=["Class3_mean_TU", "Class1_mean_TU"])
ar_directional_scatter_df = add_interval_to_scatter_df(ar_directional_scatter_df, sampleinfo)
ar_directional_scatter_df.to_csv(
    os.path.join(OUT_DIR, "AR_directional_Class1_up_Class3_down_condition_meanTU_points.tsv"),
    sep="\t",
    index=False,
)

ar_directional_feature_summary_df = pd.DataFrame(
    [
        {"Class": "Class1", "AR_directional_DUT": "AR_up", "Transcript_count": len(class1_ar_up)},
        {"Class": "Class3", "AR_directional_DUT": "AR_down", "Transcript_count": len(class3_ar_down)},
    ]
)
ar_directional_feature_summary_df.to_csv(
    os.path.join(OUT_DIR, "AR_directional_Class1_up_Class3_down_feature_summary.tsv"),
    sep="\t",
    index=False,
)

ar_directional_condition_mean_tu_summary_df = summarize_condition_mean_tu(ar_directional_scatter_df)
ar_directional_condition_mean_tu_summary_df.to_csv(
    os.path.join(OUT_DIR, "AR_directional_Class1_up_Class3_down_condition_meanTU_summary.tsv"),
    sep="\t",
    index=False,
)

ar_directional_class1_vs_class3_wilcoxon_df = paired_class1_class3_wilcoxon_by_condition(
    ar_directional_scatter_df
)
ar_directional_class1_vs_class3_wilcoxon_df.to_csv(
    os.path.join(OUT_DIR, "AR_directional_Class1_up_Class3_down_paired_Wilcoxon_by_condition.tsv"),
    sep="\t",
    index=False,
)

print("\n===== AR-directional DUT paired Class1-up vs Class3-down Wilcoxon by condition =====")
print(ar_directional_feature_summary_df.to_string(index=False))
print(ar_directional_condition_mean_tu_summary_df.to_string(index=False))
print(ar_directional_class1_vs_class3_wilcoxon_df.to_string(index=False))
print(f"AR-directional point table saved={os.path.join(OUT_DIR, 'AR_directional_Class1_up_Class3_down_condition_meanTU_points.tsv')}")
print(f"AR-directional paired Wilcoxon table saved={os.path.join(OUT_DIR, 'AR_directional_Class1_up_Class3_down_paired_Wilcoxon_by_condition.tsv')}")

# %%
###^^^ validation #######################
newcohort = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_pre/111_pre/forval_111_gene_TPM.txt', sep='\t', index_col=0, nrows=0)
clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/112_PARPi_clinicalinfo.txt', sep='\t', index_col=0)
clin = clin.loc[clin.index.isin(newcohort.columns),:]
clin["BRCAmt_numeric"] = pd.to_numeric(clin["BRCAmt"], errors="coerce")
clin['group'] = 'i'
clin.loc[(clin['response']==1)&(clin['recur']==1),'group'] = 'AR'
clin.loc[(clin['response']==0),'group'] = 'IR'
clin.loc[(clin['response']==1)&(clin['recur']==0),'group'] = 'CR'

# %%
#######^^ (11) New cohort AR-directional Class1/Class3 BRCA-stratified group mean TU lineplot ########
NEWCOHORT_GROUP_ORDERS = {
    "withCR": ["CR", "AR", "IR"],
    "noCR": ["AR", "IR"],
}
NEWCOHORT_BRCA_COHORTS = [
    ("BRCAall", "BRCA all"),
    ("BRCAmt", "BRCAmt"),
    ("BRCAwt", "BRCAwt"),
]


def prepare_validation_tu_from_transcript_tpm(transcript_tpm_path):
    val_transcript_tpm = pd.read_csv(transcript_tpm_path, sep="\t", index_col=0)
    if "gene_name" in val_transcript_tpm.columns:
        val_transcript_tpm = val_transcript_tpm.drop(columns=["gene_name"])

    val_transcript_tpm = val_transcript_tpm.apply(pd.to_numeric, errors="coerce").fillna(0)
    val_gene = val_transcript_tpm.index.to_series().astype(str).str.split("-", n=1).str[-1]
    val_gene_sum = val_transcript_tpm.groupby(val_gene).transform("sum").replace(0, np.nan)
    val_tu = val_transcript_tpm.div(val_gene_sum).fillna(0)
    return val_tu


def match_transcript_ids_to_validation_rows(transcript_ids, val_index):
    txid_to_val_rows = {}
    for val_row in pd.Index(val_index).astype(str):
        tx_id = val_row.split("-", 1)[0]
        txid_to_val_rows.setdefault(tx_id, []).append(val_row)

    matched_rows = []
    for transcript_id in transcript_ids:
        matched_rows.extend(txid_to_val_rows.get(str(transcript_id).split("-", 1)[0], []))

    return list(dict.fromkeys(matched_rows))


def subset_clin_by_brca_cohort(clin_df, cohort_id):
    if cohort_id == "BRCAall":
        return clin_df.copy()
    if cohort_id == "BRCAmt":
        return clin_df.loc[clin_df["BRCAmt_numeric"] == 1].copy()
    if cohort_id == "BRCAwt":
        return clin_df.loc[clin_df["BRCAmt_numeric"] == 0].copy()
    raise ValueError(f"Unknown BRCA cohort: {cohort_id}")


def build_newcohort_group_mean_tu_df(
    val_tu_df,
    clin_df,
    class1_rows,
    class3_rows,
    group_order,
    cohort_id,
    cohort_label,
):
    common_samples = clin_df.index.intersection(val_tu_df.columns)
    plot_clin = clin_df.loc[common_samples].copy()
    plot_clin = plot_clin.loc[plot_clin["group"].isin(group_order)]

    class1_mean = val_tu_df.loc[class1_rows, plot_clin.index].mean(axis=0)
    class3_mean = val_tu_df.loc[class3_rows, plot_clin.index].mean(axis=0)
    point_df = pd.DataFrame(
        {
            "Cohort": cohort_id,
            "Cohort_label": cohort_label,
            "Sample": plot_clin.index,
            "group": plot_clin["group"].values,
            "Class1_mean_TU": class1_mean.reindex(plot_clin.index).values,
            "Class3_mean_TU": class3_mean.reindex(plot_clin.index).values,
        }
    )
    point_df["Class1_minus_Class3_mean_TU"] = point_df["Class1_mean_TU"] - point_df["Class3_mean_TU"]
    return point_df


def summarize_newcohort_group_mean_tu(point_df, group_order, cohort_id, cohort_label):
    records = []
    for class_label, value_col in [("Class1", "Class1_mean_TU"), ("Class3", "Class3_mean_TU")]:
        for group in group_order:
            values = point_df.loc[point_df["group"] == group, value_col]
            mean_value, ci95, sample_count = mean_ci95(values)
            records.append(
                {
                    "Cohort": cohort_id,
                    "Cohort_label": cohort_label,
                    "Class": class_label,
                    "group": group,
                    "Mean_TU": mean_value,
                    "CI95": ci95,
                    "Sample_count": sample_count,
                }
            )
    return pd.DataFrame(records)


def build_newcohort_brca_panel_tables(val_tu_df, clin_df, class1_features, class3_features, group_order):
    class1_rows = match_transcript_ids_to_validation_rows(class1_features, val_tu_df.index)
    class3_rows = match_transcript_ids_to_validation_rows(class3_features, val_tu_df.index)
    if len(class1_rows) == 0:
        raise ValueError("No Class1 AR-up transcripts matched the validation TU matrix.")
    if len(class3_rows) == 0:
        raise ValueError("No Class3 AR-down transcripts matched the validation TU matrix.")

    match_summary_df = pd.DataFrame(
        [
            {
                "Class": "Class1",
                "Discovery_direction": "AR_up",
                "Discovery_transcript_count": len(class1_features),
                "Validation_matched_row_count": len(class1_rows),
            },
            {
                "Class": "Class3",
                "Discovery_direction": "AR_down",
                "Discovery_transcript_count": len(class3_features),
                "Validation_matched_row_count": len(class3_rows),
            },
        ]
    )

    point_dfs = []
    summary_dfs = []
    cohort_sample_records = []
    for cohort_id, cohort_label in NEWCOHORT_BRCA_COHORTS:
        cohort_clin = subset_clin_by_brca_cohort(clin_df, cohort_id)
        point_df = build_newcohort_group_mean_tu_df(
            val_tu_df=val_tu_df,
            clin_df=cohort_clin,
            class1_rows=class1_rows,
            class3_rows=class3_rows,
            group_order=group_order,
            cohort_id=cohort_id,
            cohort_label=cohort_label,
        )
        summary_df = summarize_newcohort_group_mean_tu(
            point_df,
            group_order=group_order,
            cohort_id=cohort_id,
            cohort_label=cohort_label,
        )
        point_dfs.append(point_df)
        summary_dfs.append(summary_df)

        group_counts = point_df["group"].value_counts().reindex(group_order, fill_value=0)
        for group, sample_count in group_counts.items():
            cohort_sample_records.append(
                {
                    "Cohort": cohort_id,
                    "Cohort_label": cohort_label,
                    "group": group,
                    "Sample_count": int(sample_count),
                }
            )

    point_table = pd.concat(point_dfs, ignore_index=True)
    summary_table = pd.concat(summary_dfs, ignore_index=True)
    cohort_sample_summary_df = pd.DataFrame(cohort_sample_records)
    return point_table, summary_table, match_summary_df, cohort_sample_summary_df


def plot_newcohort_brca_panel_lineplot(summary_df, group_order, save_stem, title):
    x_positions = np.arange(len(group_order), dtype=float)
    fig, axes = plt.subplots(
        1,
        len(NEWCOHORT_BRCA_COHORTS),
        figsize=(3.7 * len(NEWCOHORT_BRCA_COHORTS), 4.0),
        sharey=True,
    )
    if len(NEWCOHORT_BRCA_COHORTS) == 1:
        axes = [axes]

    for ax, (cohort_id, cohort_label) in zip(axes, NEWCOHORT_BRCA_COHORTS):
        cohort_summary_df = summary_df[summary_df["Cohort"] == cohort_id]
        for class_label in ["Class1", "Class3"]:
            plot_df = (
                cohort_summary_df[cohort_summary_df["Class"] == class_label]
                .set_index("group")
                .reindex(group_order)
                .reset_index()
            )
            ax.errorbar(
                x_positions,
                plot_df["Mean_TU"],
                yerr=plot_df["CI95"],
                color=LINE_CLASS_COLORS[class_label],
                marker="o",
                markersize=5.5,
                markeredgecolor="#333333",
                markeredgewidth=0.5,
                linewidth=2.0,
                capsize=3.5,
                elinewidth=1.1,
                label=class_label,
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(group_order)
        ax.set_xlabel("Group")
        ax.grid(axis="y", alpha=0.18, linewidth=0.7)
        sns.despine(ax=ax)

    axes[0].set_ylabel("Mean TU")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
    )
    fig.tight_layout()

    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    return {"pdf": pdf_path, "png": png_path}


def save_newcohort_brca_panel_outputs(version_id, group_order, title):
    (
        point_df,
        summary_df,
        match_summary_df,
        cohort_sample_summary_df,
    ) = build_newcohort_brca_panel_tables(
        val_tu_df=val_tu_newcohort,
        clin_df=clin,
        class1_features=class1_ar_up,
        class3_features=class3_ar_down,
        group_order=group_order,
    )

    save_stem = f"newcohort_{version_id}_BRCAsubsets_AR_directional_Class1_up_Class3_down_group_meanTU_panel_lineplot_CI"
    point_path = os.path.join(
        OUT_DIR,
        f"newcohort_{version_id}_BRCAsubsets_AR_directional_Class1_up_Class3_down_group_meanTU_points.tsv",
    )
    summary_path = os.path.join(
        OUT_DIR,
        f"newcohort_{version_id}_BRCAsubsets_AR_directional_Class1_up_Class3_down_group_meanTU_lineplot_summary.tsv",
    )
    match_path = os.path.join(
        OUT_DIR,
        f"newcohort_{version_id}_BRCAsubsets_AR_directional_Class1_up_Class3_down_matching_summary.tsv",
    )
    cohort_sample_path = os.path.join(
        OUT_DIR,
        f"newcohort_{version_id}_BRCAsubsets_group_sample_counts.tsv",
    )

    point_df.to_csv(point_path, sep="\t", index=False)
    summary_df.to_csv(summary_path, sep="\t", index=False)
    match_summary_df.to_csv(match_path, sep="\t", index=False)
    cohort_sample_summary_df.to_csv(cohort_sample_path, sep="\t", index=False)
    figure_paths = plot_newcohort_brca_panel_lineplot(
        summary_df=summary_df,
        group_order=group_order,
        save_stem=save_stem,
        title=title,
    )

    print(f"\n===== New cohort {version_id} BRCA-panel AR-directional Class1-up/Class3-down group mean TU lineplot =====")
    print(match_summary_df.to_string(index=False))
    print(cohort_sample_summary_df.to_string(index=False))
    print(summary_df.to_string(index=False))
    print(f"New cohort point table saved={point_path}")
    print(f"New cohort lineplot summary saved={summary_path}")
    print(f"New cohort BRCA panel lineplot saved={figure_paths['pdf']}")

    return {
        "points": point_path,
        "summary": summary_path,
        "matching": match_path,
        "cohort_sample_counts": cohort_sample_path,
        **figure_paths,
    }


val_tu_newcohort = prepare_validation_tu_from_transcript_tpm(
    "/home/jiye/jiye/copycomparison/GENCODEquant/SEV_pre/111_pre/forval_111_transcript_TPM.txt"
)
newcohort_withCR_brca_panel_paths = save_newcohort_brca_panel_outputs(
    version_id="withCR",
    group_order=NEWCOHORT_GROUP_ORDERS["withCR"],
    title="New cohort AR-directional DUT mean TU",
)
newcohort_noCR_brca_panel_paths = save_newcohort_brca_panel_outputs(
    version_id="noCR",
    group_order=NEWCOHORT_GROUP_ORDERS["noCR"],
    title="New cohort AR-directional DUT mean TU (no CR)",
)

# %%
#######^^ (12) Order-correlation DUT selection and validation survival ########
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test

ORDER_CORR_P_CUTOFF = 0.05
ORDER_CORR_MIN_SAMPLES = 12
ORDER_CORR_CONDITION_SPECS = [
    ("AR_pre", 1, preTU, ar_samples),
    ("IR_pre", 2, preTU, ir_samples),
    ("IR_post", 3, postTU, ir_samples),
    ("AR_post", 4, postTU, ar_samples),
]
VALIDATION_SURVIVAL_GROUP_SETS = [
    ("ARIR", ["AR", "IR"], "AR+IR"),
    ("ARIRCR", ["AR", "IR", "CR"], "AR+IR+CR"),
]
VALIDATION_SURVIVAL_BRCA_SETS = [
    ("BRCAall", "All", None),
    ("BRCAwt", "BRCAwt", 0),
]
ORDER_CORR_SURVIVAL_PALETTE = {
    "High DUT score": "#EF463A",
    "Low DUT score": "#18A0DA",
}


def format_ordercorr_survival_pvalue(p_value):
    if pd.isna(p_value):
        return "NA"
    return f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.3f}"


def build_order_correlation_table(transcript_ids, class_label):
    records = []
    transcript_gene_map = (
        majorminor.drop_duplicates("transcriptid")
        .set_index("transcriptid")["Transcript-Gene"]
    )

    for transcript_id in sorted(set(transcript_ids) & set(preTU.index) & set(postTU.index)):
        order_values = []
        tu_values = []
        record = {
            "Transcript": transcript_id,
            "Class": class_label,
            "Transcript-Gene": transcript_gene_map.get(transcript_id, transcript_id),
        }
        record["Gene"] = str(record["Transcript-Gene"]).split("-", 1)[-1]

        for condition, order_rank, matrix, sample_ids in ORDER_CORR_CONDITION_SPECS:
            valid_samples = [sample for sample in sample_ids if sample in matrix.columns]
            values = pd.to_numeric(matrix.loc[transcript_id, valid_samples], errors="coerce").dropna()
            record[f"{condition}_mean_TU"] = float(values.mean()) if not values.empty else np.nan
            record[f"{condition}_n"] = int(values.shape[0])
            order_values.extend([order_rank] * values.shape[0])
            tu_values.extend(values.to_numpy(dtype=float).tolist())

        if (
            len(tu_values) >= ORDER_CORR_MIN_SAMPLES
            and pd.Series(tu_values).nunique(dropna=True) > 1
            and pd.Series(order_values).nunique(dropna=True) > 1
        ):
            rho, p_value = stats.spearmanr(order_values, tu_values)
        else:
            rho, p_value = np.nan, np.nan

        record["Spearman_rho_TU_vs_order1234"] = float(rho) if pd.notna(rho) else np.nan
        record["Spearman_p_value"] = float(p_value) if pd.notna(p_value) else np.nan
        record["Total_n"] = int(len(tu_values))

        if class_label == "Class1":
            order_score = record["Spearman_rho_TU_vs_order1234"]
            selected = (
                pd.notna(order_score)
                and pd.notna(record["Spearman_p_value"])
                and order_score > 0
                and record["Spearman_p_value"] < ORDER_CORR_P_CUTOFF
            )
            record["Target_order"] = "AR_pre < IR_pre < IR_post < AR_post"
        elif class_label == "Class3":
            order_score = -record["Spearman_rho_TU_vs_order1234"]
            selected = (
                pd.notna(order_score)
                and pd.notna(record["Spearman_p_value"])
                and order_score > 0
                and record["Spearman_p_value"] < ORDER_CORR_P_CUTOFF
            )
            record["Target_order"] = "AR_pre > IR_pre > IR_post > AR_post"
        else:
            raise ValueError("class_label must be 'Class1' or 'Class3'.")

        record["Directional_order_score"] = float(order_score) if pd.notna(order_score) else np.nan
        record["Selected_ordercorr"] = bool(selected)
        records.append(record)

    return pd.DataFrame(records)


def summarize_ordercorr_features(class1_table, class3_table):
    rows = []
    for class_label, table in [("Class1", class1_table), ("Class3", class3_table)]:
        selected = table[table["Selected_ordercorr"]].copy()
        rows.append(
            {
                "Class": class_label,
                "Candidate_count": int(table.shape[0]),
                "Selected_count": int(selected.shape[0]),
                "Selection_method": f"directional Spearman p < {ORDER_CORR_P_CUTOFF}",
                "P_value_cutoff": ORDER_CORR_P_CUTOFF,
                "Rho_cutoff": "none",
                "Min_samples": ORDER_CORR_MIN_SAMPLES,
                "Median_directional_order_score_selected": (
                    float(selected["Directional_order_score"].median())
                    if not selected.empty
                    else np.nan
                ),
                "Target_order": table["Target_order"].iloc[0] if not table.empty else "",
            }
        )
    return pd.DataFrame(rows)


class1_ordercorr_table = build_order_correlation_table(class1_ar_ir_union, "Class1")
class3_ordercorr_table = build_order_correlation_table(class3_ar_ir_union, "Class3")
ordercorr_transcript_table = pd.concat(
    [class1_ordercorr_table, class3_ordercorr_table],
    ignore_index=True,
)
ordercorr_summary_df = summarize_ordercorr_features(class1_ordercorr_table, class3_ordercorr_table)

ordercorr_transcript_path = os.path.join(
    OUT_DIR,
    "ARIR_union_Class1_Class3_ordercorr_transcripts.tsv",
)
ordercorr_summary_path = os.path.join(
    OUT_DIR,
    "ARIR_union_Class1_Class3_ordercorr_feature_summary.tsv",
)
ordercorr_transcript_table.to_csv(ordercorr_transcript_path, sep="\t", index=False)
ordercorr_summary_df.to_csv(ordercorr_summary_path, sep="\t", index=False)

class1_ordercorr = class1_ordercorr_table.loc[
    class1_ordercorr_table["Selected_ordercorr"],
    "Transcript",
].tolist()
class3_ordercorr = class3_ordercorr_table.loc[
    class3_ordercorr_table["Selected_ordercorr"],
    "Transcript",
].tolist()

print("\n===== AR/IR union Class1/Class3 order-correlation DUT selection =====")
print(ordercorr_summary_df.to_string(index=False))
print(f"Order-correlation transcript table saved={ordercorr_transcript_path}")
print(f"Order-correlation feature summary saved={ordercorr_summary_path}")


def make_validation_ordercorr_survival_df(val_tu_df, clin_df, feature_ids, group_order, brca_value):
    matched_rows = match_transcript_ids_to_validation_rows(feature_ids, val_tu_df.index)
    if len(matched_rows) == 0:
        return pd.DataFrame(), matched_rows

    plot_clin = clin_df.copy()
    if brca_value is not None:
        plot_clin = plot_clin.loc[plot_clin["BRCAmt_numeric"] == brca_value].copy()

    common_samples = plot_clin.index.intersection(val_tu_df.columns)
    plot_df = plot_clin.loc[common_samples, ["PFS", "recur", "group", "BRCAmt_numeric"]].copy()
    plot_df = plot_df.loc[plot_df["group"].isin(group_order)]

    if plot_df.empty:
        return pd.DataFrame(), matched_rows

    dut_score = (
        val_tu_df.loc[matched_rows, plot_df.index]
        .apply(pd.to_numeric, errors="coerce")
        .mean(axis=0)
        .rename("mean_DUT_TU")
    )
    plot_df = plot_df.join(dut_score)
    plot_df["PFS"] = pd.to_numeric(plot_df["PFS"], errors="coerce")
    plot_df["recur"] = pd.to_numeric(plot_df["recur"], errors="coerce")
    plot_df = plot_df.dropna(subset=["PFS", "recur", "mean_DUT_TU"])
    plot_df = plot_df.loc[plot_df["recur"].isin([0, 1])].copy()
    plot_df["recur"] = plot_df["recur"].astype(int)
    return plot_df, matched_rows


def run_validation_ordercorr_survival(
    val_tu_df,
    clin_df,
    feature_ids,
    class_label,
    group_set_id,
    group_order,
    group_set_label,
    brca_id,
    brca_label,
    brca_value,
):
    plot_df, matched_rows = make_validation_ordercorr_survival_df(
        val_tu_df=val_tu_df,
        clin_df=clin_df,
        feature_ids=feature_ids,
        group_order=group_order,
        brca_value=brca_value,
    )

    base_record = {
        "Class": class_label,
        "Group_set": group_set_id,
        "Group_set_label": group_set_label,
        "BRCA_cohort": brca_id,
        "BRCA_label": brca_label,
        "Discovery_selected_count": int(len(feature_ids)),
        "Validation_matched_row_count": int(len(matched_rows)),
    }

    if plot_df.empty or plot_df["mean_DUT_TU"].nunique(dropna=True) < 2:
        record = {
            **base_record,
            "Sample_count": int(plot_df.shape[0]),
            "High_count": 0,
            "Low_count": 0,
            "Median_score": np.nan,
            "HR_high_vs_low": np.nan,
            "HR_CI95_low": np.nan,
            "HR_CI95_high": np.nan,
            "Cox_p": np.nan,
            "Logrank_p": np.nan,
            "Figure_pdf": "",
            "Figure_png": "",
            "Status": "skipped_no_valid_or_variable_score",
        }
        return record, plot_df

    median_score = plot_df["mean_DUT_TU"].median()
    plot_df["DUT_score_group"] = np.where(
        plot_df["mean_DUT_TU"] >= median_score,
        "High DUT score",
        "Low DUT score",
    )
    plot_df["High_vs_Low"] = plot_df["DUT_score_group"].eq("High DUT score").astype(int)
    high_df = plot_df.loc[plot_df["DUT_score_group"] == "High DUT score"].copy()
    low_df = plot_df.loc[plot_df["DUT_score_group"] == "Low DUT score"].copy()

    hr = ci_low = ci_high = cox_p = np.nan
    cox_status = "ok"
    if plot_df["High_vs_Low"].nunique() == 2 and plot_df["recur"].sum() > 0:
        try:
            cox_df = plot_df[["PFS", "recur", "High_vs_Low"]].copy()
            cph = CoxPHFitter()
            cph.fit(cox_df, duration_col="PFS", event_col="recur")
            hr = float(cph.summary.loc["High_vs_Low", "exp(coef)"])
            ci_low = float(cph.summary.loc["High_vs_Low", "exp(coef) lower 95%"])
            ci_high = float(cph.summary.loc["High_vs_Low", "exp(coef) upper 95%"])
            cox_p = float(cph.summary.loc["High_vs_Low", "p"])
        except Exception as exc:
            cox_status = f"cox_failed: {exc}"
    else:
        cox_status = "cox_skipped_insufficient_score_groups_or_no_events"

    logrank_p = np.nan
    logrank_status = "ok"
    try:
        logrank_result = logrank_test(
            high_df["PFS"],
            low_df["PFS"],
            event_observed_A=high_df["recur"],
            event_observed_B=low_df["recur"],
        )
        logrank_p = float(logrank_result.p_value)
    except Exception as exc:
        logrank_status = f"logrank_failed: {exc}"

    save_stem = (
        f"VAL_ordercorr_{group_set_id}_{brca_id}_{class_label}_"
        "DUT_median_score_survival"
    )
    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    kmf = KaplanMeierFitter()
    for score_group in ["High DUT score", "Low DUT score"]:
        mask = plot_df["DUT_score_group"].eq(score_group)
        if not mask.any():
            continue
        kmf.fit(
            plot_df.loc[mask, "PFS"],
            event_observed=plot_df.loc[mask, "recur"],
            label=f"{score_group} (n={int(mask.sum())})",
        )
        kmf.plot_survival_function(
            ax=ax,
            color=ORDER_CORR_SURVIVAL_PALETTE[score_group],
            ci_show=False,
        )

    hr_text = "NA" if pd.isna(hr) else f"{hr:.2f} ({ci_low:.2f}-{ci_high:.2f})"
    ax.text(
        0.58,
        0.50,
        f"HR = {hr_text}\n"
        f"Cox p = {format_ordercorr_survival_pvalue(cox_p)}\n"
        f"log-rank p = {format_ordercorr_survival_pvalue(logrank_p)}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
    )
    ax.set_title(f"{class_label} order-corr DUT ({group_set_label}, {brca_label})")
    ax.set_xlabel("PFS")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper right")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    record = {
        **base_record,
        "Sample_count": int(plot_df.shape[0]),
        "High_count": int(high_df.shape[0]),
        "Low_count": int(low_df.shape[0]),
        "Median_score": float(median_score),
        "HR_high_vs_low": hr,
        "HR_CI95_low": ci_low,
        "HR_CI95_high": ci_high,
        "Cox_p": cox_p,
        "Logrank_p": logrank_p,
        "Figure_pdf": pdf_path,
        "Figure_png": png_path,
        "Status": "; ".join([cox_status, logrank_status]),
    }

    point_df = plot_df.reset_index().rename(columns={"index": "Sample"})
    for key, value in base_record.items():
        point_df[key] = value
    return record, point_df


ordercorr_survival_records = []
ordercorr_survival_point_tables = []
for class_label, selected_features in [
    ("Class1", class1_ordercorr),
    ("Class3", class3_ordercorr),
]:
    for group_set_id, group_order, group_set_label in VALIDATION_SURVIVAL_GROUP_SETS:
        for brca_id, brca_label, brca_value in VALIDATION_SURVIVAL_BRCA_SETS:
            summary_record, point_df = run_validation_ordercorr_survival(
                val_tu_df=val_tu_newcohort,
                clin_df=clin,
                feature_ids=selected_features,
                class_label=class_label,
                group_set_id=group_set_id,
                group_order=group_order,
                group_set_label=group_set_label,
                brca_id=brca_id,
                brca_label=brca_label,
                brca_value=brca_value,
            )
            ordercorr_survival_records.append(summary_record)
            if not point_df.empty:
                ordercorr_survival_point_tables.append(point_df)

ordercorr_survival_summary_df = pd.DataFrame(ordercorr_survival_records)
ordercorr_survival_summary_path = os.path.join(
    OUT_DIR,
    "VAL_ordercorr_Class1_Class3_DUT_median_score_survival_summary.tsv",
)
ordercorr_survival_summary_df.to_csv(
    ordercorr_survival_summary_path,
    sep="\t",
    index=False,
)

if ordercorr_survival_point_tables:
    ordercorr_survival_points_df = pd.concat(ordercorr_survival_point_tables, ignore_index=True)
else:
    ordercorr_survival_points_df = pd.DataFrame()
ordercorr_survival_points_path = os.path.join(
    OUT_DIR,
    "VAL_ordercorr_Class1_Class3_DUT_median_score_survival_points.tsv",
)
ordercorr_survival_points_df.to_csv(
    ordercorr_survival_points_path,
    sep="\t",
    index=False,
)

print("\n===== Validation survival for order-correlation Class1/Class3 DUT scores =====")
print(ordercorr_survival_summary_df.to_string(index=False))
print(f"Validation order-correlation survival summary saved={ordercorr_survival_summary_path}")
print(f"Validation order-correlation survival points saved={ordercorr_survival_points_path}")

# %%
#######^^ (13) PARPi-related GO BP 2021 membership among order-correlation DUTs ########
PARPI_GO_BP2021_TERM_GROUPS = {
    "Homologous_recombination": [
        "double-strand break repair via homologous recombination (GO:0000724)",
        "regulation of double-strand break repair via homologous recombination (GO:0010569)",
    ],
    "DNA_repair_DSB": [
        "DNA repair (GO:0006281)",
        "double-strand break repair (GO:0006302)",
        "cellular response to DNA damage stimulus (GO:0006974)",
    ],
    "Replication_fork": [
        "replication fork processing (GO:0031297)",
        "replication fork protection (GO:0048478)",
    ],
    "DNA_damage_checkpoint": [
        "DNA damage checkpoint signaling (GO:0000077)",
        "DNA integrity checkpoint signaling (GO:0031570)",
        "DNA damage response, signal transduction by p53 class mediator (GO:0030330)",
        "signal transduction in response to DNA damage (GO:0042770)",
        "cell cycle checkpoint signaling (GO:0000075)",
    ],
    "Cell_cycle": [
        "cell cycle G2/M phase transition (GO:0044839)",
        "G2/M transition of mitotic cell cycle (GO:0000086)",
        "regulation of cell cycle (GO:0051726)",
        "mitotic cell cycle phase transition (GO:0044772)",
    ],
}
PARPI_GO_BP2021_LIBRARY = "GO_Biological_Process_2021"


def load_go_bp2021_library():
    try:
        return gp.get_library(PARPI_GO_BP2021_LIBRARY, organism="Human")
    except Exception as exc:
        fallback_paths = [
            "/home/jiye/jiye/copycomparison/OC_transcriptome/GO_Biological_Process_2021.txt",
            "/home/jiye/jiye/copycomparison/GENCODEquant/GO_Biological_Process_2021.txt",
        ]
        for fallback_path in fallback_paths:
            if not os.path.exists(fallback_path):
                continue
            parsed_library = {}
            with open(fallback_path) as handle:
                for line in handle:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 3:
                        continue
                    parsed_library[parts[0]] = parts[2:]
            if parsed_library:
                return parsed_library
        raise RuntimeError(f"Could not load {PARPI_GO_BP2021_LIBRARY}.") from exc


def build_parpi_go_term_panel(go_library):
    panel_rows = []
    missing_rows = []
    for category, terms in PARPI_GO_BP2021_TERM_GROUPS.items():
        for term in terms:
            if term in go_library:
                genes = sorted({str(gene).upper() for gene in go_library[term] if pd.notna(gene)})
                panel_rows.append(
                    {
                        "Category": category,
                        "Term": term,
                        "Term_gene_count": len(genes),
                        "Term_genes": ";".join(genes),
                    }
                )
            else:
                missing_rows.append({"Category": category, "Term": term})
    return pd.DataFrame(panel_rows), pd.DataFrame(missing_rows)


def match_selected_ordercorr_transcripts_to_parpi_go(selected_table, go_library, term_panel_df):
    term_specs = []
    for row in term_panel_df.to_dict("records"):
        term_genes = {str(gene).upper() for gene in go_library[row["Term"]] if pd.notna(gene)}
        term_specs.append(
            {
                "Category": row["Category"],
                "Term": row["Term"],
                "Term_gene_count": row["Term_gene_count"],
                "Genes": term_genes,
            }
        )

    match_rows = []
    for row in selected_table.to_dict("records"):
        gene = str(row["Gene"])
        gene_upper = gene.upper()
        for spec in term_specs:
            if gene_upper not in spec["Genes"]:
                continue
            match_rows.append(
                {
                    "Class": row["Class"],
                    "Transcript": row["Transcript"],
                    "Transcript-Gene": row["Transcript-Gene"],
                    "Gene": gene,
                    "Spearman_rho_TU_vs_order1234": row["Spearman_rho_TU_vs_order1234"],
                    "Spearman_p_value": row["Spearman_p_value"],
                    "Directional_order_score": row["Directional_order_score"],
                    "Category": spec["Category"],
                    "Term": spec["Term"],
                    "Term_gene_count": spec["Term_gene_count"],
                }
            )

    match_df = pd.DataFrame(match_rows)
    if match_df.empty:
        transcript_df = pd.DataFrame()
        term_summary_df = pd.DataFrame()
        category_summary_df = pd.DataFrame()
        overall_summary_df = pd.DataFrame(
            [
                {
                    "Class": class_label,
                    "Selected_ordercorr_transcript_count": int(
                        selected_table.loc[selected_table["Class"] == class_label, "Transcript"].nunique()
                    ),
                    "PARPi_GO_matched_transcript_count": 0,
                    "PARPi_GO_matched_gene_count": 0,
                }
                for class_label in ["Class1", "Class3"]
            ]
        )
        return match_df, transcript_df, term_summary_df, category_summary_df, overall_summary_df

    match_df = match_df.sort_values(["Class", "Category", "Gene", "Transcript", "Term"]).reset_index(drop=True)
    transcript_df = (
        match_df.groupby(["Class", "Transcript", "Transcript-Gene", "Gene"], as_index=False)
        .agg(
            Spearman_rho_TU_vs_order1234=("Spearman_rho_TU_vs_order1234", "first"),
            Spearman_p_value=("Spearman_p_value", "first"),
            Directional_order_score=("Directional_order_score", "first"),
            PARPi_GO_category_count=("Category", lambda x: len(set(x))),
            PARPi_GO_term_count=("Term", lambda x: len(set(x))),
            PARPi_GO_categories=("Category", lambda x: ";".join(sorted(set(x)))),
            PARPi_GO_terms=("Term", lambda x: ";".join(sorted(set(x)))),
        )
        .sort_values(["Class", "Gene", "Transcript"])
        .reset_index(drop=True)
    )

    term_summary_df = (
        match_df.groupby(["Class", "Category", "Term", "Term_gene_count"], as_index=False)
        .agg(
            Matched_gene_count=("Gene", lambda x: len(set(x))),
            Matched_transcript_count=("Transcript", lambda x: len(set(x))),
            Matched_genes=("Gene", lambda x: ";".join(sorted(set(x)))),
            Matched_transcripts=("Transcript-Gene", lambda x: ";".join(sorted(set(x)))),
        )
        .sort_values(["Class", "Category", "Term"])
        .reset_index(drop=True)
    )

    category_summary_df = (
        match_df.groupby(["Class", "Category"], as_index=False)
        .agg(
            Matched_gene_count=("Gene", lambda x: len(set(x))),
            Matched_transcript_count=("Transcript", lambda x: len(set(x))),
            Matched_genes=("Gene", lambda x: ";".join(sorted(set(x)))),
            Matched_transcripts=("Transcript-Gene", lambda x: ";".join(sorted(set(x)))),
        )
        .sort_values(["Class", "Category"])
        .reset_index(drop=True)
    )

    overall_rows = []
    for class_label in ["Class1", "Class3"]:
        class_selected = selected_table[selected_table["Class"] == class_label]
        class_matches = match_df[match_df["Class"] == class_label]
        overall_rows.append(
            {
                "Class": class_label,
                "Selected_ordercorr_transcript_count": int(class_selected["Transcript"].nunique()),
                "Selected_ordercorr_gene_count": int(class_selected["Gene"].nunique()),
                "PARPi_GO_matched_transcript_count": int(class_matches["Transcript"].nunique()),
                "PARPi_GO_matched_gene_count": int(class_matches["Gene"].nunique()),
                "PARPi_GO_matched_genes": ";".join(sorted(set(class_matches["Gene"]))),
                "PARPi_GO_matched_transcripts": ";".join(sorted(set(class_matches["Transcript-Gene"]))),
            }
        )
    overall_summary_df = pd.DataFrame(overall_rows)
    return match_df, transcript_df, term_summary_df, category_summary_df, overall_summary_df


go_bp2021_library = load_go_bp2021_library()
parpi_go_term_panel_df, parpi_go_missing_terms_df = build_parpi_go_term_panel(go_bp2021_library)
selected_ordercorr_transcript_table = ordercorr_transcript_table[
    ordercorr_transcript_table["Selected_ordercorr"]
].copy()

(
    parpi_go_match_df,
    parpi_go_transcript_df,
    parpi_go_term_summary_df,
    parpi_go_category_summary_df,
    parpi_go_overall_summary_df,
) = match_selected_ordercorr_transcripts_to_parpi_go(
    selected_table=selected_ordercorr_transcript_table,
    go_library=go_bp2021_library,
    term_panel_df=parpi_go_term_panel_df,
)

parpi_go_term_panel_path = os.path.join(
    OUT_DIR,
    "PARPi_related_GO_BP2021_term_panel.tsv",
)
parpi_go_missing_terms_path = os.path.join(
    OUT_DIR,
    "PARPi_related_GO_BP2021_missing_terms.tsv",
)
parpi_go_match_path = os.path.join(
    OUT_DIR,
    "ARIR_union_ordercorr_selected_PARPi_GO_BP2021_transcript_term_matches.tsv",
)
parpi_go_transcript_path = os.path.join(
    OUT_DIR,
    "ARIR_union_ordercorr_selected_PARPi_GO_BP2021_transcripts.tsv",
)
parpi_go_term_summary_path = os.path.join(
    OUT_DIR,
    "ARIR_union_ordercorr_selected_PARPi_GO_BP2021_term_summary.tsv",
)
parpi_go_category_summary_path = os.path.join(
    OUT_DIR,
    "ARIR_union_ordercorr_selected_PARPi_GO_BP2021_category_summary.tsv",
)
parpi_go_overall_summary_path = os.path.join(
    OUT_DIR,
    "ARIR_union_ordercorr_selected_PARPi_GO_BP2021_overall_summary.tsv",
)

parpi_go_term_panel_df.to_csv(parpi_go_term_panel_path, sep="\t", index=False)
parpi_go_missing_terms_df.to_csv(parpi_go_missing_terms_path, sep="\t", index=False)
parpi_go_match_df.to_csv(parpi_go_match_path, sep="\t", index=False)
parpi_go_transcript_df.to_csv(parpi_go_transcript_path, sep="\t", index=False)
parpi_go_term_summary_df.to_csv(parpi_go_term_summary_path, sep="\t", index=False)
parpi_go_category_summary_df.to_csv(parpi_go_category_summary_path, sep="\t", index=False)
parpi_go_overall_summary_df.to_csv(parpi_go_overall_summary_path, sep="\t", index=False)

print("\n===== PARPi-related GO BP 2021 membership among order-correlation DUTs =====")
print(parpi_go_overall_summary_df.to_string(index=False))
if not parpi_go_category_summary_df.empty:
    print(parpi_go_category_summary_df[["Class", "Category", "Matched_gene_count", "Matched_transcript_count"]].to_string(index=False))
else:
    print("No selected order-correlation DUTs matched the PARPi-related GO BP 2021 term panel.")
if not parpi_go_missing_terms_df.empty:
    print("Missing GO BP 2021 terms:")
    print(parpi_go_missing_terms_df.to_string(index=False))
print(f"PARPi GO BP 2021 term panel saved={parpi_go_term_panel_path}")
print(f"PARPi GO BP 2021 transcript-term matches saved={parpi_go_match_path}")
print(f"PARPi GO BP 2021 matched transcript list saved={parpi_go_transcript_path}")
print(f"PARPi GO BP 2021 term summary saved={parpi_go_term_summary_path}")
print(f"PARPi GO BP 2021 category summary saved={parpi_go_category_summary_path}")

# %%
#######^^ (14) Order-correlation DUT enrichment, pathway score bias, and baseline direction checks ########
ORDERCORR_ENRICHMENT_LIBRARIES = ["GO_Biological_Process_2021", "Reactome_2022"]
ORDERCORR_ENRICHMENT_FDR_CUTOFF = 0.01
ORDERCORR_PATHWAY_BIAS_CATEGORIES = ["Cell_cycle", "Homologous_recombination"]
ORDERCORR_PATHWAY_BIAS_N_PERMUTATIONS = 10000
ORDERCORR_PATHWAY_BIAS_RANDOM_SEED = 20260529


def run_ordercorr_selected_enrichment(selected_table, gene_sets, fdr_cutoff=0.01):
    all_enrichr_results = []
    enrichment_summary_rows = []
    for class_label in ["Class1", "Class3"]:
        class_selected = selected_table[selected_table["Class"] == class_label].copy()
        genes = sorted({str(gene).upper() for gene in class_selected["Gene"].dropna()})
        gene_input_path = os.path.join(
            OUT_DIR,
            f"ARIR_union_ordercorr_selected_{class_label}_enrichment_input_genes.txt",
        )
        pd.Series(genes, name="Gene").to_csv(gene_input_path, sep="\t", index=False)

        base_summary = {
            "Class": class_label,
            "Selected_transcript_count": int(class_selected["Transcript"].nunique()),
            "Selected_gene_count": int(len(genes)),
            "Gene_input_path": gene_input_path,
        }

        if len(genes) == 0:
            for gene_set in gene_sets:
                enrichment_summary_rows.append(
                    {
                        **base_summary,
                        "Gene_set": gene_set,
                        "Tested_term_count": 0,
                        "Significant_term_count_FDR001": 0,
                        "Significant_term_count_FDR010": 0,
                        "Top_term": "",
                        "Top_adjusted_p_value": np.nan,
                        "Top_overlap": "",
                        "Status": "skipped_no_genes",
                    }
                )
            continue

        try:
            enr = gp.enrichr(
                gene_list=genes,
                gene_sets=gene_sets,
                organism="Human",
                outdir=None,
            )
            enrichr_df = enr.results.copy()
            enrichr_df["Adjusted P-value"] = pd.to_numeric(
                enrichr_df["Adjusted P-value"],
                errors="coerce",
            )
            enrichr_df = enrichr_df.sort_values(["Gene_set", "Adjusted P-value", "Term"]).reset_index(drop=True)
            enrichr_df["Class"] = class_label
            enrichr_df["Selected_transcript_count"] = int(class_selected["Transcript"].nunique())
            enrichr_df["Selected_gene_count"] = int(len(genes))
            all_enrichr_results.append(enrichr_df)

            for gene_set in gene_sets:
                gene_set_df = enrichr_df[enrichr_df["Gene_set"] == gene_set].copy()
                sig001_df = gene_set_df[gene_set_df["Adjusted P-value"] < fdr_cutoff]
                sig010_df = gene_set_df[gene_set_df["Adjusted P-value"] < 0.10]
                if gene_set_df.empty:
                    top_row = {}
                else:
                    top_row = gene_set_df.iloc[0].to_dict()
                enrichment_summary_rows.append(
                    {
                        **base_summary,
                        "Gene_set": gene_set,
                        "Tested_term_count": int(gene_set_df.shape[0]),
                        "Significant_term_count_FDR001": int(sig001_df.shape[0]),
                        "Significant_term_count_FDR010": int(sig010_df.shape[0]),
                        "Top_term": top_row.get("Term", ""),
                        "Top_adjusted_p_value": top_row.get("Adjusted P-value", np.nan),
                        "Top_overlap": top_row.get("Overlap", ""),
                        "Status": "ok",
                    }
                )
        except Exception as exc:
            for gene_set in gene_sets:
                enrichment_summary_rows.append(
                    {
                        **base_summary,
                        "Gene_set": gene_set,
                        "Tested_term_count": 0,
                        "Significant_term_count_FDR001": 0,
                        "Significant_term_count_FDR010": 0,
                        "Top_term": "",
                        "Top_adjusted_p_value": np.nan,
                        "Top_overlap": "",
                        "Status": f"enrichr_failed: {exc}",
                    }
                )

    if all_enrichr_results:
        all_enrichr_df = pd.concat(all_enrichr_results, ignore_index=True)
    else:
        all_enrichr_df = pd.DataFrame()
    enrichment_summary_df = pd.DataFrame(enrichment_summary_rows)
    sig_enrichr_df = (
        all_enrichr_df[all_enrichr_df["Adjusted P-value"] < fdr_cutoff].copy()
        if not all_enrichr_df.empty
        else pd.DataFrame()
    )
    return all_enrichr_df, sig_enrichr_df, enrichment_summary_df


def permutation_pvalue_mean_difference(pathway_values, nonpathway_values, n_permutations=10000, random_seed=0):
    pathway_values = pd.to_numeric(pd.Series(pathway_values), errors="coerce").dropna().to_numpy(dtype=float)
    nonpathway_values = pd.to_numeric(pd.Series(nonpathway_values), errors="coerce").dropna().to_numpy(dtype=float)
    if pathway_values.size == 0 or nonpathway_values.size == 0:
        return np.nan, np.nan

    observed_diff = float(pathway_values.mean() - nonpathway_values.mean())
    pooled = np.concatenate([pathway_values, nonpathway_values])
    pathway_n = pathway_values.size
    rng = np.random.default_rng(random_seed)
    perm_diffs = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        shuffled = rng.permutation(pooled)
        perm_diffs[i] = shuffled[:pathway_n].mean() - shuffled[pathway_n:].mean()
    perm_p = (np.sum(perm_diffs >= observed_diff) + 1) / (n_permutations + 1)
    return observed_diff, float(perm_p)


def build_ordercorr_pathway_bias_table(selected_table, category_match_df):
    selected_for_bias = selected_table.copy()
    selected_for_bias["Neg_log10_Spearman_p_value"] = -np.log10(
        pd.to_numeric(selected_for_bias["Spearman_p_value"], errors="coerce").clip(lower=np.nextafter(0, 1))
    )

    category_members = {}
    if category_match_df is not None and not category_match_df.empty:
        for category in ORDERCORR_PATHWAY_BIAS_CATEGORIES:
            category_members[category] = set(
                category_match_df.loc[category_match_df["Category"] == category, "Transcript"].astype(str)
            )
    else:
        category_members = {category: set() for category in ORDERCORR_PATHWAY_BIAS_CATEGORIES}

    rows = []
    metric_specs = [
        ("Directional_order_score", "directional Spearman r"),
        ("Neg_log10_Spearman_p_value", "-log10 Spearman p"),
    ]
    for class_label in ["Class1", "Class3"]:
        class_df = selected_for_bias[selected_for_bias["Class"] == class_label].copy()
        for category in ORDERCORR_PATHWAY_BIAS_CATEGORIES:
            members = category_members.get(category, set())
            class_df["Pathway_member"] = class_df["Transcript"].astype(str).isin(members)
            pathway_df = class_df[class_df["Pathway_member"]].copy()
            nonpathway_df = class_df[~class_df["Pathway_member"]].copy()

            for metric_col, metric_label in metric_specs:
                pathway_values = pd.to_numeric(pathway_df[metric_col], errors="coerce").dropna()
                nonpathway_values = pd.to_numeric(nonpathway_df[metric_col], errors="coerce").dropna()
                if pathway_values.empty or nonpathway_values.empty:
                    mw_stat, mw_p = np.nan, np.nan
                else:
                    mw_res = stats.mannwhitneyu(
                        pathway_values,
                        nonpathway_values,
                        alternative="greater",
                    )
                    mw_stat, mw_p = float(mw_res.statistic), float(mw_res.pvalue)

                observed_diff, perm_p = permutation_pvalue_mean_difference(
                    pathway_values,
                    nonpathway_values,
                    n_permutations=ORDERCORR_PATHWAY_BIAS_N_PERMUTATIONS,
                    random_seed=ORDERCORR_PATHWAY_BIAS_RANDOM_SEED
                    + (0 if class_label == "Class1" else 100000)
                    + ORDERCORR_PATHWAY_BIAS_CATEGORIES.index(category) * 1000
                    + (0 if metric_col == "Directional_order_score" else 100),
                )
                rows.append(
                    {
                        "Class": class_label,
                        "Category": category,
                        "Metric": metric_label,
                        "Pathway_transcript_count": int(pathway_df["Transcript"].nunique()),
                        "Nonpathway_transcript_count": int(nonpathway_df["Transcript"].nunique()),
                        "Pathway_mean": float(pathway_values.mean()) if not pathway_values.empty else np.nan,
                        "Nonpathway_mean": float(nonpathway_values.mean()) if not nonpathway_values.empty else np.nan,
                        "Pathway_median": float(pathway_values.median()) if not pathway_values.empty else np.nan,
                        "Nonpathway_median": float(nonpathway_values.median()) if not nonpathway_values.empty else np.nan,
                        "Observed_mean_difference_pathway_minus_nonpathway": observed_diff,
                        "MannWhitneyU_stat": mw_stat,
                        "MannWhitneyU_p_pathway_greater": mw_p,
                        "Permutation_p_pathway_mean_greater": perm_p,
                        "Permutation_count": ORDERCORR_PATHWAY_BIAS_N_PERMUTATIONS,
                        "Pathway_transcripts": ";".join(sorted(pathway_df["Transcript-Gene"].astype(str).unique())),
                    }
                )

    return selected_for_bias, pd.DataFrame(rows)


def safe_mannwhitneyu(ar_values, ir_values, alternative):
    ar_values = pd.to_numeric(pd.Series(ar_values), errors="coerce").dropna()
    ir_values = pd.to_numeric(pd.Series(ir_values), errors="coerce").dropna()
    if ar_values.empty or ir_values.empty:
        return np.nan, np.nan
    if ar_values.nunique() <= 1 and ir_values.nunique() <= 1 and ar_values.iloc[0] == ir_values.iloc[0]:
        return np.nan, 1.0
    result = stats.mannwhitneyu(ar_values, ir_values, alternative=alternative)
    return float(result.statistic), float(result.pvalue)


def build_arpre_irpre_direction_check(selected_table):
    rows = []
    ar_pre_cols = [sample for sample in ar_samples if sample in preTU.columns]
    ir_pre_cols = [sample for sample in ir_samples if sample in preTU.columns]

    for row in selected_table.to_dict("records"):
        transcript_id = str(row["Transcript"])
        if transcript_id not in preTU.index:
            continue

        ar_pre_values = pd.to_numeric(preTU.loc[transcript_id, ar_pre_cols], errors="coerce").dropna()
        ir_pre_values = pd.to_numeric(preTU.loc[transcript_id, ir_pre_cols], errors="coerce").dropna()
        ar_pre_mean = float(ar_pre_values.mean()) if not ar_pre_values.empty else np.nan
        ir_pre_mean = float(ir_pre_values.mean()) if not ir_pre_values.empty else np.nan

        if row["Class"] == "Class1":
            expected_relation = "AR_pre < IR_pre"
            directional_delta = ir_pre_mean - ar_pre_mean
            satisfied = pd.notna(directional_delta) and directional_delta > 0
            mw_stat, mw_p = safe_mannwhitneyu(ar_pre_values, ir_pre_values, alternative="less")
        elif row["Class"] == "Class3":
            expected_relation = "AR_pre > IR_pre"
            directional_delta = ar_pre_mean - ir_pre_mean
            satisfied = pd.notna(directional_delta) and directional_delta > 0
            mw_stat, mw_p = safe_mannwhitneyu(ar_pre_values, ir_pre_values, alternative="greater")
        else:
            continue

        rows.append(
            {
                "Class": row["Class"],
                "Transcript": transcript_id,
                "Transcript-Gene": row["Transcript-Gene"],
                "Gene": row["Gene"],
                "Expected_relation": expected_relation,
                "AR_pre_mean_TU": ar_pre_mean,
                "IR_pre_mean_TU": ir_pre_mean,
                "Directional_baseline_delta": directional_delta,
                "Baseline_expected_relation_satisfied": bool(satisfied),
                "AR_pre_n": int(ar_pre_values.shape[0]),
                "IR_pre_n": int(ir_pre_values.shape[0]),
                "Baseline_MannWhitneyU_stat": mw_stat,
                "Baseline_directional_MannWhitneyU_p": mw_p,
                "Baseline_directional_MannWhitneyU_p_lt005": bool(pd.notna(mw_p) and mw_p < 0.05),
                "Spearman_rho_TU_vs_order1234": row["Spearman_rho_TU_vs_order1234"],
                "Spearman_p_value": row["Spearman_p_value"],
                "Directional_order_score": row["Directional_order_score"],
            }
        )

    direction_df = pd.DataFrame(rows)
    if direction_df.empty:
        return direction_df, pd.DataFrame()

    summary_df = (
        direction_df.groupby("Class", as_index=False)
        .agg(
            Selected_transcript_count=("Transcript", "nunique"),
            Baseline_expected_relation_satisfied_count=("Baseline_expected_relation_satisfied", "sum"),
            Baseline_directional_MW_p_lt005_count=("Baseline_directional_MannWhitneyU_p_lt005", "sum"),
            Median_directional_baseline_delta=("Directional_baseline_delta", "median"),
            Mean_directional_baseline_delta=("Directional_baseline_delta", "mean"),
        )
    )
    summary_df["Baseline_expected_relation_satisfied_fraction"] = (
        summary_df["Baseline_expected_relation_satisfied_count"]
        / summary_df["Selected_transcript_count"]
    )
    summary_df["Baseline_directional_MW_p_lt005_fraction"] = (
        summary_df["Baseline_directional_MW_p_lt005_count"]
        / summary_df["Selected_transcript_count"]
    )
    return direction_df, summary_df


ordercorr_enrichr_all_df, ordercorr_enrichr_sig_df, ordercorr_enrichment_summary_df = run_ordercorr_selected_enrichment(
    selected_ordercorr_transcript_table,
    gene_sets=ORDERCORR_ENRICHMENT_LIBRARIES,
    fdr_cutoff=ORDERCORR_ENRICHMENT_FDR_CUTOFF,
)
ordercorr_enrichr_all_path = os.path.join(
    OUT_DIR,
    "ARIR_union_ordercorr_selected_GOBP2021_Reactome_enrichr_all.tsv",
)
ordercorr_enrichr_sig_path = os.path.join(
    OUT_DIR,
    "ARIR_union_ordercorr_selected_GOBP2021_Reactome_enrichr_FDR001.tsv",
)
ordercorr_enrichment_summary_path = os.path.join(
    OUT_DIR,
    "ARIR_union_ordercorr_selected_GOBP2021_Reactome_enrichment_summary.tsv",
)
ordercorr_enrichr_all_df.to_csv(ordercorr_enrichr_all_path, sep="\t", index=False)
ordercorr_enrichr_sig_df.to_csv(ordercorr_enrichr_sig_path, sep="\t", index=False)
ordercorr_enrichment_summary_df.to_csv(ordercorr_enrichment_summary_path, sep="\t", index=False)

ordercorr_pathway_bias_input_df, ordercorr_pathway_bias_summary_df = build_ordercorr_pathway_bias_table(
    selected_ordercorr_transcript_table,
    parpi_go_match_df,
)
ordercorr_pathway_bias_input_path = os.path.join(
    OUT_DIR,
    "ARIR_union_ordercorr_selected_pathway_bias_input_transcripts.tsv",
)
ordercorr_pathway_bias_summary_path = os.path.join(
    OUT_DIR,
    "ARIR_union_ordercorr_selected_CellCycle_HRR_score_bias_tests.tsv",
)
ordercorr_pathway_bias_input_df.to_csv(ordercorr_pathway_bias_input_path, sep="\t", index=False)
ordercorr_pathway_bias_summary_df.to_csv(ordercorr_pathway_bias_summary_path, sep="\t", index=False)

ordercorr_arpre_irpre_direction_df, ordercorr_arpre_irpre_summary_df = build_arpre_irpre_direction_check(
    selected_ordercorr_transcript_table
)
ordercorr_arpre_irpre_direction_path = os.path.join(
    OUT_DIR,
    "ARIR_union_ordercorr_selected_ARpre_IRpre_direction_check.tsv",
)
ordercorr_arpre_irpre_summary_path = os.path.join(
    OUT_DIR,
    "ARIR_union_ordercorr_selected_ARpre_IRpre_direction_summary.tsv",
)
ordercorr_arpre_irpre_satisfied_path = os.path.join(
    OUT_DIR,
    "ARIR_union_ordercorr_selected_ARpre_IRpre_direction_satisfied_transcripts.tsv",
)
ordercorr_arpre_irpre_direction_df.to_csv(ordercorr_arpre_irpre_direction_path, sep="\t", index=False)
ordercorr_arpre_irpre_summary_df.to_csv(ordercorr_arpre_irpre_summary_path, sep="\t", index=False)
if not ordercorr_arpre_irpre_direction_df.empty:
    ordercorr_arpre_irpre_direction_df[
        ordercorr_arpre_irpre_direction_df["Baseline_expected_relation_satisfied"]
    ].to_csv(ordercorr_arpre_irpre_satisfied_path, sep="\t", index=False)
else:
    pd.DataFrame().to_csv(ordercorr_arpre_irpre_satisfied_path, sep="\t", index=False)

print("\n===== Order-correlation selected DUT enrichment (GO BP 2021 + Reactome) =====")
print(ordercorr_enrichment_summary_df.to_string(index=False))
print(f"Order-correlation enrichment all terms saved={ordercorr_enrichr_all_path}")
print(f"Order-correlation enrichment FDR<0.01 terms saved={ordercorr_enrichr_sig_path}")

print("\n===== Cell cycle / HRR order-correlation score bias tests =====")
print(ordercorr_pathway_bias_summary_df.to_string(index=False))
print(f"Cell cycle / HRR score bias table saved={ordercorr_pathway_bias_summary_path}")

print("\n===== AR_pre vs IR_pre direction among order-correlation selected DUTs =====")
print(ordercorr_arpre_irpre_summary_df.to_string(index=False))
print(f"AR_pre vs IR_pre direction check saved={ordercorr_arpre_irpre_direction_path}")
print(f"AR_pre vs IR_pre satisfied transcript list saved={ordercorr_arpre_irpre_satisfied_path}")

# %%
#######^^ (15) Validation survival for ARpre/IRpre direction-satisfied order-correlation DUTs ########
def run_validation_baseline_direction_satisfied_survival(
    val_tu_df,
    clin_df,
    feature_ids,
    feature_table_df,
    class_label,
    group_set_id,
    group_order,
    group_set_label,
    brca_id,
    brca_label,
    brca_value,
):
    plot_df, matched_rows = make_validation_ordercorr_survival_df(
        val_tu_df=val_tu_df,
        clin_df=clin_df,
        feature_ids=feature_ids,
        group_order=group_order,
        brca_value=brca_value,
    )

    base_record = {
        "Class": class_label,
        "Feature_source": "ordercorr_and_ARpre_IRpre_direction_satisfied",
        "Group_set": group_set_id,
        "Group_set_label": group_set_label,
        "BRCA_cohort": brca_id,
        "BRCA_label": brca_label,
        "Discovery_direction_satisfied_count": int(len(feature_ids)),
        "Validation_matched_row_count": int(len(matched_rows)),
    }

    if plot_df.empty or plot_df["mean_DUT_TU"].nunique(dropna=True) < 2:
        record = {
            **base_record,
            "Sample_count": int(plot_df.shape[0]),
            "High_count": 0,
            "Low_count": 0,
            "Median_score": np.nan,
            "HR_high_vs_low": np.nan,
            "HR_CI95_low": np.nan,
            "HR_CI95_high": np.nan,
            "Cox_p": np.nan,
            "Logrank_p": np.nan,
            "Figure_pdf": "",
            "Figure_png": "",
            "Status": "skipped_no_valid_or_variable_score",
        }
        return record, plot_df

    median_score = plot_df["mean_DUT_TU"].median()
    plot_df["DUT_score_group"] = np.where(
        plot_df["mean_DUT_TU"] >= median_score,
        "High DUT score",
        "Low DUT score",
    )
    plot_df["High_vs_Low"] = plot_df["DUT_score_group"].eq("High DUT score").astype(int)
    high_df = plot_df.loc[plot_df["DUT_score_group"] == "High DUT score"].copy()
    low_df = plot_df.loc[plot_df["DUT_score_group"] == "Low DUT score"].copy()

    hr = ci_low = ci_high = cox_p = np.nan
    cox_status = "ok"
    if plot_df["High_vs_Low"].nunique() == 2 and plot_df["recur"].sum() > 0:
        try:
            cox_df = plot_df[["PFS", "recur", "High_vs_Low"]].copy()
            cph = CoxPHFitter()
            cph.fit(cox_df, duration_col="PFS", event_col="recur")
            hr = float(cph.summary.loc["High_vs_Low", "exp(coef)"])
            ci_low = float(cph.summary.loc["High_vs_Low", "exp(coef) lower 95%"])
            ci_high = float(cph.summary.loc["High_vs_Low", "exp(coef) upper 95%"])
            cox_p = float(cph.summary.loc["High_vs_Low", "p"])
        except Exception as exc:
            cox_status = f"cox_failed: {exc}"
    else:
        cox_status = "cox_skipped_insufficient_score_groups_or_no_events"

    logrank_p = np.nan
    logrank_status = "ok"
    try:
        logrank_result = logrank_test(
            high_df["PFS"],
            low_df["PFS"],
            event_observed_A=high_df["recur"],
            event_observed_B=low_df["recur"],
        )
        logrank_p = float(logrank_result.p_value)
    except Exception as exc:
        logrank_status = f"logrank_failed: {exc}"

    save_stem = (
        f"VAL_ordercorr_baseline_direction_satisfied_{group_set_id}_{brca_id}_{class_label}_"
        "DUT_median_score_survival"
    )
    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    kmf = KaplanMeierFitter()
    for score_group in ["High DUT score", "Low DUT score"]:
        mask = plot_df["DUT_score_group"].eq(score_group)
        if not mask.any():
            continue
        kmf.fit(
            plot_df.loc[mask, "PFS"],
            event_observed=plot_df.loc[mask, "recur"],
            label=f"{score_group} (n={int(mask.sum())})",
        )
        kmf.plot_survival_function(
            ax=ax,
            color=ORDER_CORR_SURVIVAL_PALETTE[score_group],
            ci_show=False,
        )

    hr_text = "NA" if pd.isna(hr) else f"{hr:.2f} ({ci_low:.2f}-{ci_high:.2f})"
    ax.text(
        0.58,
        0.50,
        f"HR = {hr_text}\n"
        f"Cox p = {format_ordercorr_survival_pvalue(cox_p)}\n"
        f"log-rank p = {format_ordercorr_survival_pvalue(logrank_p)}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
    )
    ax.set_title(f"{class_label} baseline-direction DUT ({group_set_label}, {brca_label})")
    ax.set_xlabel("PFS")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper right")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    record = {
        **base_record,
        "Sample_count": int(plot_df.shape[0]),
        "High_count": int(high_df.shape[0]),
        "Low_count": int(low_df.shape[0]),
        "Median_score": float(median_score),
        "HR_high_vs_low": hr,
        "HR_CI95_low": ci_low,
        "HR_CI95_high": ci_high,
        "Cox_p": cox_p,
        "Logrank_p": logrank_p,
        "Figure_pdf": pdf_path,
        "Figure_png": png_path,
        "Status": "; ".join([cox_status, logrank_status]),
    }

    point_df = plot_df.reset_index().rename(columns={plot_df.index.name or "index": "Sample"})
    for key, value in base_record.items():
        point_df[key] = value
    feature_gene_map = (
        feature_table_df.drop_duplicates("Transcript")
        .set_index("Transcript")["Transcript-Gene"]
        .to_dict()
    )
    point_df["Feature_transcripts"] = ";".join(
        str(feature_gene_map.get(feature_id, feature_id))
        for feature_id in feature_ids
    )
    return record, point_df


if ordercorr_arpre_irpre_direction_df.empty:
    baseline_direction_satisfied_feature_df = pd.DataFrame()
else:
    baseline_direction_satisfied_feature_df = ordercorr_arpre_irpre_direction_df.loc[
        ordercorr_arpre_irpre_direction_df["Baseline_expected_relation_satisfied"]
    ].copy()

baseline_direction_satisfied_feature_summary_df = (
    baseline_direction_satisfied_feature_df.groupby("Class", as_index=False)
    .agg(
        Direction_satisfied_transcript_count=("Transcript", "nunique"),
        Direction_satisfied_gene_count=("Gene", "nunique"),
        Median_directional_baseline_delta=("Directional_baseline_delta", "median"),
        Median_directional_order_score=("Directional_order_score", "median"),
    )
    if not baseline_direction_satisfied_feature_df.empty
    else pd.DataFrame(
        columns=[
            "Class",
            "Direction_satisfied_transcript_count",
            "Direction_satisfied_gene_count",
            "Median_directional_baseline_delta",
            "Median_directional_order_score",
        ]
    )
)

baseline_direction_satisfied_feature_path = os.path.join(
    OUT_DIR,
    "ARIR_union_ordercorr_selected_ARpre_IRpre_direction_satisfied_features_for_survival.tsv",
)
baseline_direction_satisfied_feature_summary_path = os.path.join(
    OUT_DIR,
    "ARIR_union_ordercorr_selected_ARpre_IRpre_direction_satisfied_feature_summary_for_survival.tsv",
)
baseline_direction_satisfied_feature_df.to_csv(
    baseline_direction_satisfied_feature_path,
    sep="\t",
    index=False,
)
baseline_direction_satisfied_feature_summary_df.to_csv(
    baseline_direction_satisfied_feature_summary_path,
    sep="\t",
    index=False,
)

baseline_direction_survival_records = []
baseline_direction_survival_point_tables = []
for class_label in ["Class1", "Class3"]:
    feature_ids = baseline_direction_satisfied_feature_df.loc[
        baseline_direction_satisfied_feature_df["Class"] == class_label,
        "Transcript",
    ].drop_duplicates().astype(str).tolist()

    for group_set_id, group_order, group_set_label in VALIDATION_SURVIVAL_GROUP_SETS:
        for brca_id, brca_label, brca_value in VALIDATION_SURVIVAL_BRCA_SETS:
            summary_record, point_df = run_validation_baseline_direction_satisfied_survival(
                val_tu_df=val_tu_newcohort,
                clin_df=clin,
                feature_ids=feature_ids,
                feature_table_df=baseline_direction_satisfied_feature_df,
                class_label=class_label,
                group_set_id=group_set_id,
                group_order=group_order,
                group_set_label=group_set_label,
                brca_id=brca_id,
                brca_label=brca_label,
                brca_value=brca_value,
            )
            baseline_direction_survival_records.append(summary_record)
            if not point_df.empty:
                baseline_direction_survival_point_tables.append(point_df)

baseline_direction_survival_summary_df = pd.DataFrame(baseline_direction_survival_records)
baseline_direction_survival_summary_path = os.path.join(
    OUT_DIR,
    "VAL_ordercorr_baseline_direction_satisfied_Class1_Class3_DUT_median_score_survival_summary.tsv",
)
baseline_direction_survival_summary_df.to_csv(
    baseline_direction_survival_summary_path,
    sep="\t",
    index=False,
)

if baseline_direction_survival_point_tables:
    baseline_direction_survival_points_df = pd.concat(
        baseline_direction_survival_point_tables,
        ignore_index=True,
    )
else:
    baseline_direction_survival_points_df = pd.DataFrame()
baseline_direction_survival_points_path = os.path.join(
    OUT_DIR,
    "VAL_ordercorr_baseline_direction_satisfied_Class1_Class3_DUT_median_score_survival_points.tsv",
)
baseline_direction_survival_points_df.to_csv(
    baseline_direction_survival_points_path,
    sep="\t",
    index=False,
)

print("\n===== Validation survival for ARpre/IRpre direction-satisfied order-correlation DUT scores =====")
print(baseline_direction_satisfied_feature_summary_df.to_string(index=False))
print(baseline_direction_survival_summary_df.to_string(index=False))
print(f"Baseline-direction satisfied feature table saved={baseline_direction_satisfied_feature_path}")
print(f"Baseline-direction satisfied survival summary saved={baseline_direction_survival_summary_path}")
print(f"Baseline-direction satisfied survival points saved={baseline_direction_survival_points_path}")

# %%
#######^^ (16) PARPi-related GO BP 2021 membership after ARpre/IRpre direction filter ########
(
    baseline_direction_parpi_go_match_df,
    baseline_direction_parpi_go_transcript_df,
    baseline_direction_parpi_go_term_summary_df,
    baseline_direction_parpi_go_category_summary_df,
    baseline_direction_parpi_go_overall_summary_df,
) = match_selected_ordercorr_transcripts_to_parpi_go(
    selected_table=baseline_direction_satisfied_feature_df,
    go_library=go_bp2021_library,
    term_panel_df=parpi_go_term_panel_df,
)

baseline_direction_parpi_go_match_path = os.path.join(
    OUT_DIR,
    "ARIR_union_ordercorr_baseline_direction_satisfied_PARPi_GO_BP2021_transcript_term_matches.tsv",
)
baseline_direction_parpi_go_transcript_path = os.path.join(
    OUT_DIR,
    "ARIR_union_ordercorr_baseline_direction_satisfied_PARPi_GO_BP2021_transcripts.tsv",
)
baseline_direction_parpi_go_term_summary_path = os.path.join(
    OUT_DIR,
    "ARIR_union_ordercorr_baseline_direction_satisfied_PARPi_GO_BP2021_term_summary.tsv",
)
baseline_direction_parpi_go_category_summary_path = os.path.join(
    OUT_DIR,
    "ARIR_union_ordercorr_baseline_direction_satisfied_PARPi_GO_BP2021_category_summary.tsv",
)
baseline_direction_parpi_go_overall_summary_path = os.path.join(
    OUT_DIR,
    "ARIR_union_ordercorr_baseline_direction_satisfied_PARPi_GO_BP2021_overall_summary.tsv",
)

baseline_direction_parpi_go_match_df.to_csv(
    baseline_direction_parpi_go_match_path,
    sep="\t",
    index=False,
)
baseline_direction_parpi_go_transcript_df.to_csv(
    baseline_direction_parpi_go_transcript_path,
    sep="\t",
    index=False,
)
baseline_direction_parpi_go_term_summary_df.to_csv(
    baseline_direction_parpi_go_term_summary_path,
    sep="\t",
    index=False,
)
baseline_direction_parpi_go_category_summary_df.to_csv(
    baseline_direction_parpi_go_category_summary_path,
    sep="\t",
    index=False,
)
baseline_direction_parpi_go_overall_summary_df.to_csv(
    baseline_direction_parpi_go_overall_summary_path,
    sep="\t",
    index=False,
)

print("\n===== PARPi-related GO BP 2021 membership after ARpre/IRpre direction filter =====")
print(baseline_direction_parpi_go_overall_summary_df.to_string(index=False))
if not baseline_direction_parpi_go_category_summary_df.empty:
    print(
        baseline_direction_parpi_go_category_summary_df[
            ["Class", "Category", "Matched_gene_count", "Matched_transcript_count"]
        ].to_string(index=False)
    )
else:
    print("No baseline-direction satisfied DUTs matched the PARPi-related GO BP 2021 term panel.")
print(f"Baseline-direction PARPi GO BP 2021 transcript-term matches saved={baseline_direction_parpi_go_match_path}")
print(f"Baseline-direction PARPi GO BP 2021 matched transcript list saved={baseline_direction_parpi_go_transcript_path}")
print(f"Baseline-direction PARPi GO BP 2021 term summary saved={baseline_direction_parpi_go_term_summary_path}")
print(f"Baseline-direction PARPi GO BP 2021 category summary saved={baseline_direction_parpi_go_category_summary_path}")

# %%
#######^^ (17) Validation survival for manually selected PARPi-related transcripts ########
USER_PROVIDED_TRANSCRIPT_GROUPS = {
    "Class1": [
        "ENST00000262643.3-CCNE1",
        "ENST00000347343.2-AURKA",
        "ENST00000260810.5-TOPBP1",
        "ENST00000352618.4-RAD51AP1",
    ],
    "Class3": [
        "MSTRG.219936.19-RBBP8",
        "MSTRG.203888.31-FANCA",
        "MSTRG.107505.6-RECQL",
    ],
}


def build_user_provided_feature_table(transcript_groups):
    rows = []
    for class_label, transcript_genes in transcript_groups.items():
        for transcript_gene in transcript_genes:
            transcript_gene = str(transcript_gene)
            transcript_id = transcript_gene.split("-", 1)[0]
            gene = transcript_gene.split("-", 1)[-1] if "-" in transcript_gene else ""
            rows.append(
                {
                    "Class": class_label,
                    "Transcript": transcript_id,
                    "Transcript-Gene": transcript_gene,
                    "Gene": gene,
                }
            )
    return pd.DataFrame(rows)


def run_validation_user_provided_transcript_survival(
    val_tu_df,
    clin_df,
    feature_ids,
    feature_table_df,
    class_label,
    group_set_id,
    group_order,
    group_set_label,
    brca_id,
    brca_label,
    brca_value,
):
    plot_df, matched_rows = make_validation_ordercorr_survival_df(
        val_tu_df=val_tu_df,
        clin_df=clin_df,
        feature_ids=feature_ids,
        group_order=group_order,
        brca_value=brca_value,
    )

    base_record = {
        "Class": class_label,
        "Feature_source": "user_provided_PARPi_related_transcripts",
        "Group_set": group_set_id,
        "Group_set_label": group_set_label,
        "BRCA_cohort": brca_id,
        "BRCA_label": brca_label,
        "User_provided_transcript_count": int(len(feature_ids)),
        "Validation_matched_row_count": int(len(matched_rows)),
    }

    if plot_df.empty or plot_df["mean_DUT_TU"].nunique(dropna=True) < 2:
        record = {
            **base_record,
            "Sample_count": int(plot_df.shape[0]),
            "High_count": 0,
            "Low_count": 0,
            "Median_score": np.nan,
            "HR_high_vs_low": np.nan,
            "HR_CI95_low": np.nan,
            "HR_CI95_high": np.nan,
            "Cox_p": np.nan,
            "Logrank_p": np.nan,
            "Figure_pdf": "",
            "Figure_png": "",
            "Status": "skipped_no_valid_or_variable_score",
        }
        return record, plot_df

    median_score = plot_df["mean_DUT_TU"].median()
    plot_df["DUT_score_group"] = np.where(
        plot_df["mean_DUT_TU"] >= median_score,
        "High DUT score",
        "Low DUT score",
    )
    plot_df["High_vs_Low"] = plot_df["DUT_score_group"].eq("High DUT score").astype(int)
    high_df = plot_df.loc[plot_df["DUT_score_group"] == "High DUT score"].copy()
    low_df = plot_df.loc[plot_df["DUT_score_group"] == "Low DUT score"].copy()

    hr = ci_low = ci_high = cox_p = np.nan
    cox_status = "ok"
    if plot_df["High_vs_Low"].nunique() == 2 and plot_df["recur"].sum() > 0:
        try:
            cox_df = plot_df[["PFS", "recur", "High_vs_Low"]].copy()
            cph = CoxPHFitter()
            cph.fit(cox_df, duration_col="PFS", event_col="recur")
            hr = float(cph.summary.loc["High_vs_Low", "exp(coef)"])
            ci_low = float(cph.summary.loc["High_vs_Low", "exp(coef) lower 95%"])
            ci_high = float(cph.summary.loc["High_vs_Low", "exp(coef) upper 95%"])
            cox_p = float(cph.summary.loc["High_vs_Low", "p"])
        except Exception as exc:
            cox_status = f"cox_failed: {exc}"
    else:
        cox_status = "cox_skipped_insufficient_score_groups_or_no_events"

    logrank_p = np.nan
    logrank_status = "ok"
    try:
        logrank_result = logrank_test(
            high_df["PFS"],
            low_df["PFS"],
            event_observed_A=high_df["recur"],
            event_observed_B=low_df["recur"],
        )
        logrank_p = float(logrank_result.p_value)
    except Exception as exc:
        logrank_status = f"logrank_failed: {exc}"

    save_stem = (
        f"VAL_user_provided_PARPi_transcripts_{group_set_id}_{brca_id}_{class_label}_"
        "DUT_median_score_survival"
    )
    pdf_path = os.path.join(OUT_DIR, f"{save_stem}.pdf")
    png_path = os.path.join(OUT_DIR, f"{save_stem}.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    kmf = KaplanMeierFitter()
    for score_group in ["High DUT score", "Low DUT score"]:
        mask = plot_df["DUT_score_group"].eq(score_group)
        if not mask.any():
            continue
        kmf.fit(
            plot_df.loc[mask, "PFS"],
            event_observed=plot_df.loc[mask, "recur"],
            label=f"{score_group} (n={int(mask.sum())})",
        )
        kmf.plot_survival_function(
            ax=ax,
            color=ORDER_CORR_SURVIVAL_PALETTE[score_group],
            ci_show=False,
        )

    hr_text = "NA" if pd.isna(hr) else f"{hr:.2f} ({ci_low:.2f}-{ci_high:.2f})"
    ax.text(
        0.58,
        0.50,
        f"HR = {hr_text}\n"
        f"Cox p = {format_ordercorr_survival_pvalue(cox_p)}\n"
        f"log-rank p = {format_ordercorr_survival_pvalue(logrank_p)}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
    )
    ax.set_title(f"{class_label} user-provided transcripts ({group_set_label}, {brca_label})")
    ax.set_xlabel("PFS")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="upper right")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    record = {
        **base_record,
        "Sample_count": int(plot_df.shape[0]),
        "High_count": int(high_df.shape[0]),
        "Low_count": int(low_df.shape[0]),
        "Median_score": float(median_score),
        "HR_high_vs_low": hr,
        "HR_CI95_low": ci_low,
        "HR_CI95_high": ci_high,
        "Cox_p": cox_p,
        "Logrank_p": logrank_p,
        "Figure_pdf": pdf_path,
        "Figure_png": png_path,
        "Status": "; ".join([cox_status, logrank_status]),
    }

    point_df = plot_df.reset_index()
    if "index" in point_df.columns:
        point_df = point_df.rename(columns={"index": "Sample"})
    for key, value in base_record.items():
        point_df[key] = value
    feature_gene_map = (
        feature_table_df.drop_duplicates("Transcript")
        .set_index("Transcript")["Transcript-Gene"]
        .to_dict()
    )
    point_df["Feature_transcripts"] = ";".join(
        str(feature_gene_map.get(feature_id, feature_id))
        for feature_id in feature_ids
    )
    return record, point_df


user_provided_feature_df = build_user_provided_feature_table(USER_PROVIDED_TRANSCRIPT_GROUPS)
user_provided_feature_path = os.path.join(
    OUT_DIR,
    "user_provided_PARPi_related_transcripts_for_survival.tsv",
)
user_provided_feature_df.to_csv(user_provided_feature_path, sep="\t", index=False)

user_provided_survival_records = []
user_provided_survival_point_tables = []
for class_label in ["Class1", "Class3"]:
    feature_ids = user_provided_feature_df.loc[
        user_provided_feature_df["Class"] == class_label,
        "Transcript",
    ].drop_duplicates().astype(str).tolist()

    for group_set_id, group_order, group_set_label in VALIDATION_SURVIVAL_GROUP_SETS:
        for brca_id, brca_label, brca_value in VALIDATION_SURVIVAL_BRCA_SETS:
            summary_record, point_df = run_validation_user_provided_transcript_survival(
                val_tu_df=val_tu_newcohort,
                clin_df=clin,
                feature_ids=feature_ids,
                feature_table_df=user_provided_feature_df,
                class_label=class_label,
                group_set_id=group_set_id,
                group_order=group_order,
                group_set_label=group_set_label,
                brca_id=brca_id,
                brca_label=brca_label,
                brca_value=brca_value,
            )
            user_provided_survival_records.append(summary_record)
            if not point_df.empty:
                user_provided_survival_point_tables.append(point_df)

user_provided_survival_summary_df = pd.DataFrame(user_provided_survival_records)
user_provided_survival_summary_path = os.path.join(
    OUT_DIR,
    "VAL_user_provided_PARPi_transcripts_Class1_Class3_DUT_median_score_survival_summary.tsv",
)
user_provided_survival_summary_df.to_csv(
    user_provided_survival_summary_path,
    sep="\t",
    index=False,
)

if user_provided_survival_point_tables:
    user_provided_survival_points_df = pd.concat(user_provided_survival_point_tables, ignore_index=True)
else:
    user_provided_survival_points_df = pd.DataFrame()
user_provided_survival_points_path = os.path.join(
    OUT_DIR,
    "VAL_user_provided_PARPi_transcripts_Class1_Class3_DUT_median_score_survival_points.tsv",
)
user_provided_survival_points_df.to_csv(
    user_provided_survival_points_path,
    sep="\t",
    index=False,
)

print("\n===== Validation survival for user-provided PARPi-related transcripts =====")
print(user_provided_feature_df.to_string(index=False))
print(user_provided_survival_summary_df.to_string(index=False))
print(f"User-provided transcript table saved={user_provided_feature_path}")
print(f"User-provided transcript survival summary saved={user_provided_survival_summary_path}")
print(f"User-provided transcript survival points saved={user_provided_survival_points_path}")

# %%
