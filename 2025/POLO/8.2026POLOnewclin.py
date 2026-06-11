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
import requests
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

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.plotting import add_at_risk_counts
import matplotlib as mpl
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

#%%
BASE_DIR = Path("/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38")
OUTDIR = Path("/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/2605figs")
OUTDIR.mkdir(parents=True, exist_ok=True)

val_clin = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/POLO_83_newdata2026.txt', sep='\t', index_col=0)
geneexp = pd.read_csv(BASE_DIR / "merged_83_gene_TPM.txt", sep="\t", index_col=0)
transexp = pd.read_csv(BASE_DIR / "merged_83_transcript_TPM.txt", sep="\t", index_col=0)

majorminor = pd.read_csv(BASE_DIR / "merged_83_cov5_majorminorlist.txt", sep="\t")
sqanti = pd.read_csv(BASE_DIR / "sqantioutput/sqanti_classification.txt", sep="\t")
sqanti.dropna(axis=1, how="all", inplace=True)

# Expression matrices use PL-OV-* IDs, so all downstream analysis is indexed by sampleid.
clin_df = val_clin.copy()
clin_df["sampleid"] = clin_df["sampleid"].astype(str).str.strip()
clin_df = clin_df.set_index("sampleid", drop=False)
clin_df["RECUR"] = pd.to_numeric(clin_df["RECUR"], errors="coerce")
clin_df["PFS_month"] = pd.to_numeric(clin_df["PFS_month"], errors="coerce")
clin_df["BRCAmut_numeric"] = pd.to_numeric(clin_df["BRCAmut"], errors="coerce")
clin_df["BRCAmut_group"] = clin_df["BRCAmut_numeric"].map({1.0: "BRCAmut", 0.0: "BRCAwt"})
clin_df["Group_HRD"] = clin_df["Group_HRD"].astype(str).str.strip()
clin_df.loc[clin_df["Group_HRD"].isin(["", "nan", "NaN", "Not determined"]), "Group_HRD"] = np.nan

print("Clinical shape:", clin_df.shape)
print("Gene expression shape:", geneexp.shape)
print("Transcript expression shape:", transexp.shape)
print("Clinical index is sampleid:", clin_df.index[:5].tolist())
print("Group_HRD counts:")
print(clin_df["Group_HRD"].value_counts(dropna=False))
print("BRCAmut counts:")
print(clin_df["BRCAmut_group"].value_counts(dropna=False))

#%%
def clean_filename(text):
    text = re.sub(r"[^\w\-.]+", "_", str(text))
    return text.strip("_")


def save_current_fig(name):
    path = OUTDIR / name
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")


def get_common_samples(*objects):
    common = None
    for obj in objects:
        idx = pd.Index(obj)
        common = idx if common is None else common.intersection(idx)
    return common


def plot_survival_by_group(
    data,
    group_col,
    title,
    time_col="PFS_month",
    event_col="RECUR",
    group_order=None,
    colors=None,
    xlabel="PFS (months)",
    ylabel="PFS Probability",
    save_name=None,
):
    plot_df = data[[time_col, event_col, group_col]].copy()
    plot_df[time_col] = pd.to_numeric(plot_df[time_col], errors="coerce")
    plot_df[event_col] = pd.to_numeric(plot_df[event_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[time_col, event_col, group_col])

    if group_order is None:
        labels = sorted(plot_df[group_col].dropna().unique().tolist())
    else:
        plot_df = plot_df[plot_df[group_col].isin(group_order)]
        labels = [x for x in group_order if x in set(plot_df[group_col])]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    kmf = KaplanMeierFitter()
    fitters = []

    if colors is None:
        palette = sns.color_palette("Set2", n_colors=max(len(labels), 1))
        colors = dict(zip(labels, palette))

    if len(labels) == 0:
        ax.text(0.5, 0.5, "No valid groups", transform=ax.transAxes, ha="center", va="center")
    else:
        for label in labels:
            idx = plot_df[group_col] == label
            if idx.sum() == 0:
                continue
            kmf.fit(
                durations=plot_df.loc[idx, time_col],
                event_observed=plot_df.loc[idx, event_col],
                label=f"{label} (n={idx.sum()})",
            )
            kmf.plot_survival_function(
                ax=ax,
                ci_show=False,
                color=colors.get(label, "gray"),
                linewidth=2,
                show_censors=True,
            )
            fitters.append(kmf)

    stat_text = ""
    if len(labels) == 2:
        first, second = labels
        idx_first = plot_df[group_col] == first
        try:
            lr = logrank_test(
                plot_df.loc[idx_first, time_col],
                plot_df.loc[~idx_first, time_col],
                event_observed_A=plot_df.loc[idx_first, event_col],
                event_observed_B=plot_df.loc[~idx_first, event_col],
            )
            cox_df = plot_df[[time_col, event_col, group_col]].copy()
            cox_df["group_binary"] = (cox_df[group_col] == first).astype(int)
            cph = CoxPHFitter()
            cph.fit(cox_df[[time_col, event_col, "group_binary"]], duration_col=time_col, event_col=event_col)
            summary = cph.summary.loc["group_binary"]
            hr = summary["exp(coef)"]
            ci_low = summary["exp(coef) lower 95%"]
            ci_high = summary["exp(coef) upper 95%"]
            stat_text = f"HR ({first} vs {second}) = {hr:.2f} ({ci_low:.2f}-{ci_high:.2f})\nLog-rank p = {lr.p_value:.4f}"
        except Exception as exc:
            stat_text = f"Statistics failed: {exc}"
    elif len(labels) > 2:
        try:
            lr = multivariate_logrank_test(plot_df[time_col], plot_df[group_col], plot_df[event_col])
            stat_text = f"Overall log-rank p = {lr.p_value:.4f}"
        except Exception as exc:
            stat_text = f"Statistics failed: {exc}"
    elif len(labels) == 1:
        stat_text = "Only one group present; log-rank/Cox not run"

    if stat_text:
        ax.text(
            0.05,
            0.12,
            stat_text,
            transform=ax.transAxes,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False)
    sns.despine(ax=ax)
    plt.tight_layout()

    if save_name is not None:
        save_current_fig(save_name)
    plt.show()

    return plot_df


def find_pathway(score_df, target):
    if target in score_df.index:
        return target

    target_lower = target.lower()
    exact = [term for term in score_df.index if str(term).lower() == target_lower]
    if exact:
        return exact[0]

    contains = [term for term in score_df.index if target_lower in str(term).lower()]
    if contains:
        return contains[0]

    raise KeyError(f"Cannot find pathway: {target}")

#%%
# 1. Group_HRD survival plot using RECUR and PFS_month.
plot_survival_by_group(
    clin_df,
    group_col="Group_HRD",
    group_order=["Yes", "No"],
    colors={"Yes": "#3396D3", "No": "#DA4343"},
    title="PFS by Group_HRD",
    save_name="01_survival_Group_HRD.pdf",
)

#%%
# 2. BRCAmut survival plot using RECUR and PFS_month.
plot_survival_by_group(
    clin_df,
    group_col="BRCAmut_group",
    group_order=["BRCAmut", "BRCAwt"],
    colors={"BRCAmut": "#DA4343", "BRCAwt": "#3396D3"},
    title="PFS by BRCAmut",
    save_name="02_survival_BRCAmut.pdf",
)

#%%
# 3. PFS distribution boxplot.
pfs_df = clin_df[["PFS_month", "RECUR"]].copy()
pfs_df = pfs_df.dropna(subset=["PFS_month", "RECUR"])
pfs_df["RECUR_label"] = pfs_df["RECUR"].map({0.0: "No recur", 1.0: "Recur"})

fig, ax = plt.subplots(figsize=(4.2, 5))
sns.boxplot(
    data=pfs_df,
    x="RECUR_label",
    y="PFS_month",
    order=["No recur", "Recur"],
    width=0.6,
    fliersize=0,
    palette={"No recur": "#3396D3", "Recur": "#DA4343"},
    ax=ax,
)
sns.stripplot(
    data=pfs_df,
    x="RECUR_label",
    y="PFS_month",
    order=["No recur", "Recur"],
    color="black",
    jitter=True,
    size=5,
    alpha=0.35,
    ax=ax,
)

groups = [pfs_df.loc[pfs_df["RECUR_label"] == label, "PFS_month"] for label in ["No recur", "Recur"]]
if all(len(g) > 0 for g in groups):
    u_stat, pval = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
    ax.text(
        0.5,
        0.95,
        f"Mann-Whitney p = {pval:.4f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )

ax.set_xlabel("")
ax.set_ylabel("PFS (months)")
ax.set_title("PFS distribution by recurrence", fontsize=13, fontweight="bold")
sns.despine(ax=ax)
plt.tight_layout()
save_current_fig("03_PFS_month_boxplot_by_RECUR.pdf")
plt.show()

#%%
# 4-1. Prepare functional transcript expression and whole gene expression.
coding = sqanti[["isoform", "coding", "structural_category"]]
tx_meta = pd.merge(majorminor, coding, left_on="transcriptid", right_on="isoform", how="left")

functional_tx = tx_meta[
    (tx_meta["type"] == "major")
    & (tx_meta["coding"] == "coding")
    & (tx_meta["structural_category"].isin(["full-splice_match", "novel_in_catalog"]))
]["Transcript-Gene"].dropna().unique().tolist()

tx_counts = transexp.drop(columns=["gene_name"], errors="ignore").copy()
tx_counts = tx_counts.apply(pd.to_numeric, errors="coerce").fillna(0)
tx_counts_filtered = tx_counts.loc[tx_counts.index.intersection(functional_tx)].copy()

tx_to_gene_map = tx_meta.drop_duplicates("Transcript-Gene").set_index("Transcript-Gene")["genename"]
tx_counts_filtered["genename"] = tx_counts_filtered.index.map(tx_to_gene_map)
tx_counts_filtered = tx_counts_filtered.dropna(subset=["genename"])
functional_gene_exp = tx_counts_filtered.groupby("genename").sum()

whole_gene_exp = geneexp.copy()
whole_gene_exp = whole_gene_exp.apply(pd.to_numeric, errors="coerce").fillna(0)
whole_gene_exp = whole_gene_exp.groupby(whole_gene_exp.index).sum()

common_samples = get_common_samples(clin_df.index, functional_gene_exp.columns, whole_gene_exp.columns)
clin_ssgsea = clin_df.loc[common_samples].copy()
functional_gene_exp = functional_gene_exp.loc[:, common_samples]
whole_gene_exp = whole_gene_exp.loc[:, common_samples]

expr_functional_log = np.log2(functional_gene_exp + 0.1)
expr_gene_log = np.log2(whole_gene_exp + 0.1)

print(f"Functional transcript-derived genes: {functional_gene_exp.shape[0]}")
print(f"Whole gene expression genes: {whole_gene_exp.shape[0]}")
print(f"Common samples for ssGSEA: {len(common_samples)}")

#%%
import requests

# 4-2. Run ssGSEA for MSigDB Hallmark and GOBP2021.
def load_enrichr_library(gene_sets):
    gmt_path = OUTDIR / f"{clean_filename(gene_sets)}.gmt"

    if gmt_path.exists():
        text = gmt_path.read_text()
    else:
        url = f"https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={gene_sets}"
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        text = response.text
        gmt_path.write_text(text)

    library = {}
    for line in text.splitlines():
        entries = line.strip().split("\t")
        if len(entries) < 3:
            continue
        term = entries[0]
        genes = [gene.split(",")[0] for gene in entries[2:] if gene]
        library[term] = genes

    print(f"{gene_sets}: loaded {len(library)} gene sets")
    return library


def run_ssgsea(expr_matrix, gene_sets, label):
    gene_sets_dict = load_enrichr_library(gene_sets)
    res = gp.ssgsea(
        data=expr_matrix,
        gene_sets=gene_sets_dict,
        outdir=None,
        sample_norm_method="rank",
        permutation_num=0,
    )
    score = res.res2d.pivot(index="Term", columns="Name", values="ES")
    score = score.apply(pd.to_numeric, errors="coerce")
    score = score.loc[:, score.columns.intersection(clin_ssgsea.index)]
    score.to_csv(OUTDIR / f"ssGSEA_{label}.txt", sep="\t")
    print(f"{label}: {score.shape}")
    return score


score_func_hallmark = run_ssgsea(expr_functional_log, "MSigDB_Hallmark_2020", "functional_transcript_MSigDB_Hallmark_2020")
score_gene_hallmark = run_ssgsea(expr_gene_log, "MSigDB_Hallmark_2020", "whole_gene_MSigDB_Hallmark_2020")
score_func_gobp = run_ssgsea(expr_functional_log, "GO_Biological_Process_2021", "functional_transcript_GO_Biological_Process_2021")
score_gene_gobp = run_ssgsea(expr_gene_log, "GO_Biological_Process_2021", "whole_gene_GO_Biological_Process_2021")

#%%
# 4-3. Cox screening and top HR barplot helpers.
def cox_pathway_screen(score_df, clin, label, time_col="PFS_month", event_col="RECUR"):
    common = clin.index.intersection(score_df.columns)
    results = []

    for term in score_df.index:
        vals = pd.to_numeric(score_df.loc[term, common], errors="coerce")
        tmp = clin.loc[common, [time_col, event_col]].copy()
        tmp["score"] = vals.reindex(common).values
        tmp[time_col] = pd.to_numeric(tmp[time_col], errors="coerce")
        tmp[event_col] = pd.to_numeric(tmp[event_col], errors="coerce")
        tmp = tmp.dropna()

        if tmp["score"].nunique() < 2 or tmp.shape[0] < 5:
            continue

        cutoff = tmp["score"].median()
        tmp["group"] = (tmp["score"] > cutoff).astype(int)
        if tmp["group"].nunique() < 2:
            continue

        try:
            cph = CoxPHFitter()
            cph.fit(tmp[[time_col, event_col, "group"]], duration_col=time_col, event_col=event_col)
            summary = cph.summary.loc["group"]

            idx_high = tmp["group"] == 1
            lr = logrank_test(
                tmp.loc[idx_high, time_col],
                tmp.loc[~idx_high, time_col],
                event_observed_A=tmp.loc[idx_high, event_col],
                event_observed_B=tmp.loc[~idx_high, event_col],
            )

            results.append({
                "Pathway": term,
                "Metric": label,
                "HR": summary["exp(coef)"],
                "CI_low": summary["exp(coef) lower 95%"],
                "CI_high": summary["exp(coef) upper 95%"],
                "pval": summary["p"],
                "logrank_p": lr.p_value,
                "median_cutoff": cutoff,
                "High_n": int(idx_high.sum()),
                "Low_n": int((~idx_high).sum()),
            })
        except Exception as exc:
            print(f"Skipped {term}: {exc}")

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        res_df["FDR"] = multipletests(res_df["pval"], method="fdr_bh")[1]
        res_df = res_df.sort_values("pval")
        res_df.to_csv(OUTDIR / f"cox_pathway_screen_{clean_filename(label)}.txt", sep="\t", index=False)

    print(f"{label}: {res_df.shape[0]} tested pathways")
    return res_df


def plot_top_hr_barplot(res_df, title, save_name, top_n=10):
    if res_df.empty:
        print(f"No Cox results for {title}")
        return None

    df = res_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["HR", "pval"]).copy()
    df = df[df["HR"] > 1].sort_values("pval").head(top_n)
    if df.empty:
        df = res_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["HR", "pval"]).sort_values("HR", ascending=False).head(top_n)

    df["log2HR"] = np.log2(df["HR"])
    df["neglog10p"] = -np.log10(df["pval"].clip(lower=1e-300))
    df = df.sort_values("log2HR")

    fig_height = max(4.5, 0.38 * len(df) + 1.8)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))

    cmap = plt.cm.Reds
    norm = mpl.colors.Normalize(vmin=df["neglog10p"].min(), vmax=df["neglog10p"].max())
    ax.barh(df["Pathway"], df["log2HR"], color=cmap(norm(df["neglog10p"])))
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("log2(HR), High vs Low ssGSEA score")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=13, fontweight="bold")

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("-log10(Cox p)")

    sns.despine(ax=ax)
    plt.tight_layout()
    save_current_fig(save_name)
    plt.show()

    return df

#%%
# 4-4. MSigDB Hallmark top HR barplots.
cox_func_hallmark = cox_pathway_screen(score_func_hallmark, clin_ssgsea, "Functional transcript - MSigDB Hallmark")
cox_gene_hallmark = cox_pathway_screen(score_gene_hallmark, clin_ssgsea, "Whole gene - MSigDB Hallmark")

top_func_hallmark = plot_top_hr_barplot(
    cox_func_hallmark,
    "Top HR pathways: Functional transcript / MSigDB Hallmark",
    "04_topHR_functional_transcript_MSigDB_Hallmark.pdf",
)
top_gene_hallmark = plot_top_hr_barplot(
    cox_gene_hallmark,
    "Top HR pathways: Whole gene / MSigDB Hallmark",
    "05_topHR_whole_gene_MSigDB_Hallmark.pdf",
)

#%%
# 4-5. GOBP2021 top HR barplots.
cox_func_gobp = cox_pathway_screen(score_func_gobp, clin_ssgsea, "Functional transcript - GOBP2021")
cox_gene_gobp = cox_pathway_screen(score_gene_gobp, clin_ssgsea, "Whole gene - GOBP2021")

top_func_gobp = plot_top_hr_barplot(
    cox_func_gobp,
    "Top HR pathways: Functional transcript / GOBP2021",
    "06_topHR_functional_transcript_GOBP2021.pdf",
)
top_gene_gobp = plot_top_hr_barplot(
    cox_gene_gobp,
    "Top HR pathways: Whole gene / GOBP2021",
    "07_topHR_whole_gene_GOBP2021.pdf",
)

#%%
# 5-6. Target pathway survival plot helper.
def plot_pathway_score_survival(score_df, clin, term, metric_label, save_prefix):
    pathway = find_pathway(score_df, term)
    common = clin.index.intersection(score_df.columns)
    vals = pd.to_numeric(score_df.loc[pathway, common], errors="coerce")

    tmp = clin.loc[common, ["PFS_month", "RECUR"]].copy()
    tmp["Pathway_score"] = vals.reindex(common).values
    tmp = tmp.dropna(subset=["PFS_month", "RECUR", "Pathway_score"])
    tmp["Score_group"] = np.where(tmp["Pathway_score"] > tmp["Pathway_score"].median(), "High", "Low")

    print(metric_label)
    print(pathway)
    print(tmp["Score_group"].value_counts())

    return plot_survival_by_group(
        tmp,
        group_col="Score_group",
        group_order=["High", "Low"],
        colors={"High": "#DA4343", "Low": "#3396D3"},
        title=f"{metric_label}\n{pathway}",
        save_name=f"{save_prefix}_{clean_filename(metric_label)}.pdf",
    )

#%%
# 5. MSigDB Hallmark score -> DNA Repair survival plot.
hallmark_target = "DNA Repair"

hallmark_dna_repair_func_surv = plot_pathway_score_survival(
    score_func_hallmark,
    clin_ssgsea,
    hallmark_target,
    "Functional transcript - Hallmark DNA Repair",
    "08_survival_Hallmark_DNA_Repair",
)
hallmark_dna_repair_gene_surv = plot_pathway_score_survival(
    score_gene_hallmark,
    clin_ssgsea,
    hallmark_target,
    "Whole gene - Hallmark DNA Repair",
    "08_survival_Hallmark_DNA_Repair",
)

#%%
# 6. GOBP2021 score -> double-strand break repair via homologous recombination survival plot.
gobp_target = "double-strand break repair via homologous recombination (GO:0000724)"

gobp_hrr_func_surv = plot_pathway_score_survival(
    score_func_gobp,
    clin_ssgsea,
    gobp_target,
    "Functional transcript - GOBP2021 HRR",
    "09_survival_GOBP2021_HRR",
)
gobp_hrr_gene_surv = plot_pathway_score_survival(
    score_gene_gobp,
    clin_ssgsea,
    gobp_target,
    "Whole gene - GOBP2021 HRR",
    "09_survival_GOBP2021_HRR",
)

#%%
print(f"All result files are written to: {OUTDIR}")
