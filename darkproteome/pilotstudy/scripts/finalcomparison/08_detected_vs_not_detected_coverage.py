#!/usr/bin/env python3
"""Plot detected-vs-not-detected sample-level coverage features."""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from fc_common import (
    FIG_DIR,
    GROUP_COLORS,
    INPUT_DIR,
    NEGATIVE_GROUP,
    PLOT_GROUP_ORDER,
    POSITIONS,
    PILOT_DIR,
    archive_script,
    bh_adjust,
    bw_values,
    clean_text,
    discover_bigwigs,
    import_pybigwig,
    pvalue_label,
    plot_group_from_group,
    read_master,
    start_centered_window,
    vector_mean_normalize,
)


OUT_DIR = FIG_DIR / "detected_vs_not_detected"
PDF_DIR = OUT_DIR / "pdf"
PNG_DIR = OUT_DIR / "png"
TABLE_DIR = OUT_DIR / "tables"
LOG_DIR = OUT_DIR / "logs"

DETECTION_TABLE = INPUT_DIR / "tables" / "pancreas.translated_orfs.sample_level.tsv"
COVERAGE_TABLE = FIG_DIR / "tables" / "four_group_coverage_features.sample_level.tsv"
ORF_LEVEL_COVERAGE_TABLE = FIG_DIR / "tables" / "four_group_coverage_features.orf_level.tsv"
TRANSCRIPT_EXPRESSION_TABLE = FIG_DIR / "tables" / "four_group_transcript_expression_features.tsv"
MASTER_TABLE = FIG_DIR / "tables" / "four_group_orf_metadata.tsv"

COMBINED_PDF = PDF_DIR / "Fig_detected_vs_not_detected_coverage_features_full_no_variance.pdf"
COMBINED_PNG = PNG_DIR / "Fig_detected_vs_not_detected_coverage_features_full_no_variance.png"
STATS_OUT = TABLE_DIR / "detected_vs_not_detected_coverage_statistics_full_no_variance.tsv"
METAPLOT_RAW_OUT = TABLE_DIR / "detected_vs_not_detected_metaplot_raw_overlay.tsv"
METAPLOT_NORMALIZED_OUT = TABLE_DIR / "detected_vs_not_detected_metaplot_vector_normalized_overlay.tsv"
EXPRESSION_STATS_OUT = TABLE_DIR / "detected_vs_not_detected_transcript_expression_statistics.tsv"
README_OUT = OUT_DIR / "README_detected_vs_not_detected_coverage.txt"
METAPLOT_LOG = LOG_DIR / "detected_vs_not_detected_metaplot.log"
EXPRESSION_LOG = LOG_DIR / "detected_vs_not_detected_transcript_expression.log"

RAW_METAPLOT_PDF = PDF_DIR / "Fig_detected_vs_not_detected_metaplot_raw_overlay.pdf"
NORM_METAPLOT_PDF = PDF_DIR / "Fig_detected_vs_not_detected_metaplot_vector_normalized_overlay.pdf"
TPM_PDF = PDF_DIR / "Fig_detected_vs_not_detected_TPM.pdf"
TPM_VARIANCE_PDF = PDF_DIR / "Fig_detected_vs_not_detected_TPM_variance.pdf"
TU_PDF = PDF_DIR / "Fig_detected_vs_not_detected_TU.pdf"
EXPRESSION_COMBINED_PDF = PDF_DIR / "Fig_detected_vs_not_detected_transcript_expression_combined.pdf"

STATUS_ORDER = ["not_detected", "detected"]
STATUS_PALETTE = {
    "not_detected": "#D0D0D0",
    "detected": "#2C7FB8",
}
METAPLOT_COLORS = GROUP_COLORS.copy()
METAPLOT_COLORS[NEGATIVE_GROUP] = "#D62728"

FEATURES = [
    {
        "feature_id": "mean_ORF_coverage",
        "source_columns": ["mean_ORF_coverage", "mean_coverage"],
        "label": "Mean ORF coverage",
        "definition": "Mean CPM coverage across ORF genomic bases for one ORF-sample pair.",
        "individual_pdf": "coverage_mean_ORF_coverage_detected_vs_not_detected.pdf",
    },
    {
        "feature_id": "ORF_upstream_coverage_ratio",
        "source_columns": ["ORF_upstream_coverage_ratio", "orf_upstream_coverage_ratio"],
        "label": "ORF/upstream coverage ratio",
        "definition": "Mean ORF coverage divided by mean upstream 200 nt coverage for one ORF-sample pair.",
        "individual_pdf": "coverage_ORF_upstream_coverage_ratio_detected_vs_not_detected.pdf",
    },
    {
        "feature_id": "start_peak_coverage",
        "source_columns": ["start_peak_coverage"],
        "label": "Start peak coverage",
        "definition": "Mean start-centered CPM coverage from -5 to +5 around the translation start.",
        "individual_pdf": "coverage_start_peak_coverage_detected_vs_not_detected.pdf",
    },
    {
        "feature_id": "start_peak_ratio",
        "source_columns": ["start_peak_ratio"],
        "label": "Start peak ratio",
        "definition": "Start peak coverage divided by mean start-centered window coverage.",
        "individual_pdf": "coverage_start_peak_ratio_detected_vs_not_detected.pdf",
    },
    {
        "feature_id": "upstream_slope",
        "source_columns": ["upstream_slope"],
        "label": "Upstream slope",
        "definition": "Linear slope of start-centered coverage over -100 to 0.",
        "individual_pdf": "coverage_upstream_slope_detected_vs_not_detected.pdf",
    },
    {
        "feature_id": "AUC_upstream",
        "source_columns": ["AUC_upstream"],
        "label": "AUC upstream",
        "definition": "Sum of start-centered coverage over -100 to 0.",
        "individual_pdf": "coverage_AUC_upstream_detected_vs_not_detected.pdf",
    },
    {
        "feature_id": "AUC_downstream",
        "source_columns": ["AUC_downstream"],
        "label": "AUC downstream",
        "definition": "Sum of start-centered coverage over 0 to +100.",
        "individual_pdf": "coverage_AUC_downstream_detected_vs_not_detected.pdf",
    },
]


def ensure_dirs() -> None:
    for path in [OUT_DIR, PDF_DIR, PNG_DIR, TABLE_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def read_tsv(path: Path, required: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing required column(s) in {path}: {', '.join(missing)}")
    return df


def detected_pairs() -> set[tuple[str, str]]:
    detection = read_tsv(DETECTION_TABLE, required=["ORF_id", "sample"])
    return set(zip(detection["ORF_id"].astype(str), detection["sample"].astype(str)))


def add_detection_status(df: pd.DataFrame, pairs: set[tuple[str, str]]) -> pd.DataFrame:
    out = df.copy()
    if "plot_group" not in out.columns:
        if "group" not in out.columns:
            raise SystemExit("Input table needs plot_group or group column for detection-status labelling.")
        out["plot_group"] = out["group"].map(plot_group_from_group)
    if "sample" not in out.columns:
        raise SystemExit("Input table needs sample column for detection-status labelling.")
    out["plot_group"] = out["plot_group"].map(lambda x: plot_group_from_group(clean_text(x)))
    out["detected"] = [
        0 if row["plot_group"] == NEGATIVE_GROUP else int((str(row["ORF_id"]), str(row["sample"])) in pairs)
        for _, row in out.iterrows()
    ]
    out["detection_status"] = np.where(out["detected"].eq(1), "detected", "not_detected")
    out["detection_status"] = pd.Categorical(out["detection_status"], categories=STATUS_ORDER, ordered=True)
    out["plot_group"] = pd.Categorical(out["plot_group"], categories=PLOT_GROUP_ORDER, ordered=True)
    return out


def resolve_feature_columns(df: pd.DataFrame) -> Dict[str, str]:
    mapping = {}
    for spec in FEATURES:
        for candidate in spec["source_columns"]:
            if candidate in df.columns:
                mapping[spec["feature_id"]] = candidate
                break
        if spec["feature_id"] not in mapping:
            raise SystemExit(
                f"Coverage feature column not found for {spec['feature_id']}; "
                f"tried {', '.join(spec['source_columns'])}. Available columns: {', '.join(df.columns)}"
            )
    return mapping


def standardize_coverage_table() -> pd.DataFrame:
    df = read_tsv(COVERAGE_TABLE, required=["ORF_id", "sample"])
    feature_map = resolve_feature_columns(df)
    for feature_id, source_col in feature_map.items():
        df[feature_id] = pd.to_numeric(df[source_col], errors="coerce")
    return add_detection_status(df, detected_pairs())


def mannwhitney_by_group(df: pd.DataFrame, feature_id: str) -> pd.DataFrame:
    rows = []
    for group in PLOT_GROUP_ORDER:
        if group == NEGATIVE_GROUP:
            rows.append(
                {
                    "feature": feature_id,
                    "plot_group": group,
                    "test": "mannwhitney_u",
                    "n_not_detected": int(df[df["plot_group"].eq(group) & df["detection_status"].eq("not_detected")][feature_id].notna().sum()),
                    "n_detected": 0,
                    "not_detected_median": np.nan,
                    "detected_median": np.nan,
                    "effect_size_rank_biserial": np.nan,
                    "statistic": np.nan,
                    "pvalue": np.nan,
                    "skipped_reason": "CPAT-negative noncoding is always not_detected",
                }
            )
            continue
        sub = df[df["plot_group"].eq(group)].copy()
        not_detected = sub.loc[sub["detection_status"].eq("not_detected"), feature_id].dropna().astype(float)
        detected = sub.loc[sub["detection_status"].eq("detected"), feature_id].dropna().astype(float)
        if len(not_detected) and len(detected):
            result = stats.mannwhitneyu(not_detected, detected, alternative="two-sided")
            u_stat = float(result.statistic)
            pvalue = float(result.pvalue)
            rank_biserial = (2.0 * u_stat / (len(not_detected) * len(detected))) - 1.0
            skipped = ""
        else:
            u_stat = np.nan
            pvalue = np.nan
            rank_biserial = np.nan
            skipped = "missing detected or not_detected values"
        rows.append(
            {
                "feature": feature_id,
                "plot_group": group,
                "test": "mannwhitney_u",
                "n_not_detected": int(len(not_detected)),
                "n_detected": int(len(detected)),
                "not_detected_median": float(not_detected.median()) if len(not_detected) else np.nan,
                "detected_median": float(detected.median()) if len(detected) else np.nan,
                "effect_size_rank_biserial": rank_biserial,
                "statistic": u_stat,
                "pvalue": pvalue,
                "skipped_reason": skipped,
            }
        )
    out = pd.DataFrame(rows)
    out["padj_bh_all_tests"] = bh_adjust(out["pvalue"])
    return out


def add_manual_group_annotations(ax, df: pd.DataFrame, feature_id: str, stats_df: pd.DataFrame) -> None:
    ymin, ymax = ax.get_ylim()
    if not np.isfinite(ymax):
        ymax = 1.0
    span = ymax - ymin
    if span <= 0:
        span = max(abs(ymax), 1.0)
    step = span * 0.06
    ax.set_ylim(ymin, ymax + step * 3.4)
    lookup = {group: idx for idx, group in enumerate(PLOT_GROUP_ORDER)}
    for _, row in stats_df.dropna(subset=["pvalue"]).iterrows():
        group = row["plot_group"]
        if group not in lookup or group == NEGATIVE_GROUP:
            continue
        group_values = pd.to_numeric(df.loc[df["plot_group"].astype(str).eq(group), feature_id], errors="coerce").dropna()
        if group_values.empty:
            continue
        x_center = lookup[group]
        x1 = x_center - 0.20
        x2 = x_center + 0.20
        local_top = min(float(group_values.quantile(0.98)), ymax)
        y = local_top + step * 0.8
        ax.plot([x1, x1, x2, x2], [y, y + step * 0.25, y + step * 0.25, y], color="black", lw=0.8)
        ax.text(
            (x1 + x2) / 2,
            y + step * 0.28,
            pvalue_label(row["pvalue"]),
            ha="center",
            va="bottom",
            fontsize=8,
            clip_on=True,
        )


def annotate_detected_vs_not(ax, df: pd.DataFrame, feature_id: str, stats_df: pd.DataFrame) -> None:
    valid_stats = stats_df.dropna(subset=["pvalue"])
    valid_stats = valid_stats[valid_stats["plot_group"].ne(NEGATIVE_GROUP)]
    if valid_stats.empty:
        return
    pairs = [((row["plot_group"], "not_detected"), (row["plot_group"], "detected")) for _, row in valid_stats.iterrows()]
    pvalues = [row["pvalue"] for _, row in valid_stats.iterrows()]
    try:
        from statannotations.Annotator import Annotator

        annotator = Annotator(
            ax,
            pairs,
            data=df,
            x="plot_group",
            y=feature_id,
            hue="detection_status",
            order=PLOT_GROUP_ORDER,
            hue_order=STATUS_ORDER,
        )
        annotator.configure(test=None, text_format="simple", loc="inside", verbose=0)
        annotator.set_pvalues_and_annotate(pvalues)
    except Exception:
        add_manual_group_annotations(ax, df, feature_id, stats_df)


def plot_feature(ax, df: pd.DataFrame, spec: Dict[str, object], stats_df: pd.DataFrame, *, title: str | None = None) -> None:
    feature_id = str(spec["feature_id"])
    work = df.dropna(subset=[feature_id, "plot_group", "detection_status"]).copy()
    if work.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{spec['label']}\nnot available", ha="center", va="center")
        return
    sns.boxplot(
        data=work,
        x="plot_group",
        y=feature_id,
        hue="detection_status",
        order=PLOT_GROUP_ORDER,
        hue_order=STATUS_ORDER,
        palette=STATUS_PALETTE,
        showfliers=False,
        linewidth=1.1,
        ax=ax,
    )
    ax.set_title(title or str(spec["label"]), pad=18)
    ax.set_xlabel("")
    ax.set_ylabel(str(spec["label"]))
    ax.tick_params(axis="x", rotation=25)
    annotate_detected_vs_not(ax, work, feature_id, stats_df[stats_df["feature"].eq(feature_id)])
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:2], labels[:2], title="Detection status", fontsize=8)


def save_pdf_png(fig, pdf_path: Path, png_path: Path | None = None) -> None:
    fig.tight_layout(pad=2.6, h_pad=4.0, w_pad=2.2)
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Wrote {pdf_path}")
    if png_path is not None:
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        print(f"Wrote {png_path}")
    plt.close(fig)


def line_label(group: str, status: str) -> str:
    return f"{group} ({status})"


def metaplot_status_order(group: str) -> list[str]:
    return ["not_detected"] if group == NEGATIVE_GROUP else ["detected", "not_detected"]


def update_metaplot_aggregate(aggregate: dict, key: tuple[str, str], vector: np.ndarray) -> None:
    vector = np.asarray(vector, dtype=float)
    if key not in aggregate:
        aggregate[key] = {
            "sum": np.zeros(len(POSITIONS), dtype=float),
            "count": np.zeros(len(POSITIONS), dtype=float),
            "n_vectors": 0,
        }
    finite = np.isfinite(vector)
    aggregate[key]["sum"][finite] += vector[finite]
    aggregate[key]["count"][finite] += 1
    aggregate[key]["n_vectors"] += 1


def aggregate_to_long(aggregate: dict) -> pd.DataFrame:
    rows = []
    for (group, status), values in aggregate.items():
        count = values["count"]
        mean = np.divide(
            values["sum"],
            count,
            out=np.full(len(POSITIONS), np.nan, dtype=float),
            where=count > 0,
        )
        for pos, value, n_pos in zip(POSITIONS, mean, count):
            rows.append(
                {
                    "plot_group": group,
                    "detection_status": status,
                    "position": int(pos),
                    "mean_coverage": value,
                    "n_vectors": int(values["n_vectors"]),
                    "n_nonmissing_at_position": int(n_pos),
                }
            )
    return pd.DataFrame(rows)


def build_metaplot_tables(pairs: set[tuple[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    master = read_master()
    pyBigWig = import_pybigwig()
    bw_paths = discover_bigwigs()
    raw_agg: dict = {}
    norm_agg: dict = {}
    handles = {sample: pyBigWig.open(str(path)) for sample, path in bw_paths.items()}
    try:
        for idx, row in master.iterrows():
            if idx and idx % 1000 == 0:
                print(f"Metaplot vectors processed for {idx}/{len(master)} ORF(s)")
            group = str(row["plot_group"])
            chrom, start, end, reverse = start_centered_window(row, flank=100)
            for sample, bw in handles.items():
                vector = bw_values(bw, chrom, start, end)
                if reverse:
                    vector = vector[::-1]
                status = "not_detected" if group == NEGATIVE_GROUP else ("detected" if (str(row["ORF_id"]), sample) in pairs else "not_detected")
                update_metaplot_aggregate(raw_agg, (group, status), vector)
                update_metaplot_aggregate(norm_agg, (group, status), vector_mean_normalize(vector))
    finally:
        for handle in handles.values():
            handle.close()
    return aggregate_to_long(raw_agg), aggregate_to_long(norm_agg)


def plot_overlay_metaplot(df: pd.DataFrame, ylabel: str, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for group in PLOT_GROUP_ORDER:
        for status in metaplot_status_order(group):
            sub = df[df["plot_group"].eq(group) & df["detection_status"].eq(status)].sort_values("position")
            if sub.empty:
                continue
            linestyle = "-" if status == "detected" else "--"
            ax.plot(
                sub["position"],
                pd.to_numeric(sub["mean_coverage"], errors="coerce"),
                color=METAPLOT_COLORS.get(group, "black"),
                linestyle=linestyle,
                linewidth=2.0,
                label=line_label(group, status),
            )
    ax.axvline(0, color="black", linestyle="--", linewidth=0.9)
    if "normalized" in title.lower():
        ax.axhline(1, color="gray", linestyle=":", linewidth=0.9)
    ax.set_xlabel("Position relative to start codon")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Group/status", fontsize=8, ncol=2)
    return fig


def generate_metaplots(pairs: set[tuple[str, str]]) -> None:
    try:
        raw_df, norm_df = build_metaplot_tables(pairs)
    except (Exception, SystemExit) as exc:  # keep boxplots/expression outputs usable.
        METAPLOT_LOG.write_text(f"Metaplot generation skipped/failed: {exc}\n")
        print(f"WARNING: metaplot generation skipped/failed; see {METAPLOT_LOG}")
        return
    raw_df.to_csv(METAPLOT_RAW_OUT, sep="\t", index=False, na_rep="NA")
    norm_df.to_csv(METAPLOT_NORMALIZED_OUT, sep="\t", index=False, na_rep="NA")
    METAPLOT_LOG.write_text("Metaplot generation completed.\n")
    print(f"Wrote {METAPLOT_RAW_OUT}")
    print(f"Wrote {METAPLOT_NORMALIZED_OUT}")
    save_pdf_png(
        plot_overlay_metaplot(raw_df, "Mean CPM coverage", "Start-centered raw CPM metaplot"),
        RAW_METAPLOT_PDF,
    )
    save_pdf_png(
        plot_overlay_metaplot(
            norm_df,
            "Coverage / vector mean",
            "Start-centered vector-mean normalized metaplot",
        ),
        NORM_METAPLOT_PDF,
    )


def expression_feature_table(pairs: set[tuple[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRANSCRIPT_EXPRESSION_TABLE.exists():
        EXPRESSION_LOG.write_text(f"Missing transcript expression table: {TRANSCRIPT_EXPRESSION_TABLE}\n")
        print(f"WARNING: transcript expression detected-vs-not_detected plots skipped; see {EXPRESSION_LOG}")
        return pd.DataFrame(), pd.DataFrame()
    expr = read_tsv(TRANSCRIPT_EXPRESSION_TABLE, required=["ORF_id", "sample", "plot_group", "transcript_TPM", "transcript_usage"])
    expr = add_detection_status(expr, pairs)
    expr["transcript_TPM"] = pd.to_numeric(expr["transcript_TPM"], errors="coerce")
    expr["transcript_usage"] = pd.to_numeric(expr["transcript_usage"], errors="coerce")
    variance = (
        expr.groupby(["ORF_id", "plot_group", "detection_status"], observed=True, as_index=False)
        .agg(transcript_TPM_variance=("transcript_TPM", lambda values: float(np.nanvar(pd.to_numeric(values, errors="coerce"), ddof=0))))
    )
    EXPRESSION_LOG.write_text("Transcript expression detected-vs-not_detected inputs loaded.\n")
    return expr, variance


EXPRESSION_SPECS = [
    {
        "feature_id": "transcript_TPM",
        "label": "Transcript TPM",
        "pdf": TPM_PDF,
        "source": "sample",
    },
    {
        "feature_id": "transcript_TPM_variance",
        "label": "Transcript TPM variance",
        "pdf": TPM_VARIANCE_PDF,
        "source": "variance",
    },
    {
        "feature_id": "transcript_usage",
        "label": "Transcript usage (TU)",
        "pdf": TU_PDF,
        "source": "sample",
    },
]


def positive_and_negative_order() -> list[str]:
    return PLOT_GROUP_ORDER


def plot_expression_feature(ax, df: pd.DataFrame, feature_id: str, label: str, stats_df: pd.DataFrame) -> None:
    if df.empty or feature_id not in df.columns:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{label}\nnot available", ha="center", va="center")
        return
    work = df.dropna(subset=[feature_id, "plot_group", "detection_status"]).copy()
    if work.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{label}\nnot available", ha="center", va="center")
        return
    sns.boxplot(
        data=work,
        x="plot_group",
        y=feature_id,
        hue="detection_status",
        order=positive_and_negative_order(),
        hue_order=STATUS_ORDER,
        palette=STATUS_PALETTE,
        showfliers=False,
        linewidth=1.1,
        ax=ax,
    )
    ax.set_title(label, pad=18)
    ax.set_xlabel("")
    ax.set_ylabel(label)
    ax.tick_params(axis="x", rotation=25)
    annotate_detected_vs_not(ax, work, feature_id, stats_df[stats_df["feature"].eq(feature_id)])
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:2], labels[:2], title="Detection status", fontsize=8)


def generate_expression_plots(pairs: set[tuple[str, str]]) -> None:
    sample_df, variance_df = expression_feature_table(pairs)
    if sample_df.empty and variance_df.empty:
        return
    stats_tables = []
    for spec in EXPRESSION_SPECS:
        source = sample_df if spec["source"] == "sample" else variance_df
        stats_tables.append(mannwhitney_by_group(source, str(spec["feature_id"])))
    stats_df = pd.concat(stats_tables, ignore_index=True)
    stats_df.to_csv(EXPRESSION_STATS_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {EXPRESSION_STATS_OUT}")

    for spec in EXPRESSION_SPECS:
        source = sample_df if spec["source"] == "sample" else variance_df
        fig, ax = plt.subplots(figsize=(9.5, 6.8))
        plot_expression_feature(ax, source, str(spec["feature_id"]), str(spec["label"]), stats_df)
        save_pdf_png(fig, Path(spec["pdf"]))

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.8), squeeze=False)
    for ax, spec in zip(axes.flat, EXPRESSION_SPECS):
        source = sample_df if spec["source"] == "sample" else variance_df
        plot_expression_feature(ax, source, str(spec["feature_id"]), str(spec["label"]), stats_df)
    save_pdf_png(fig, EXPRESSION_COMBINED_PDF)


def write_readme() -> None:
    definitions = "\n".join(f"- {spec['feature_id']}: {spec['definition']}" for spec in FEATURES)
    README_OUT.write_text(
        f"""Detected-vs-not-detected coverage comparison

Inputs:
- Detection table: {DETECTION_TABLE}
- Sample-level coverage table: {COVERAGE_TABLE}
- Optional ORF-level coverage table: {ORF_LEVEL_COVERAGE_TABLE}
- Optional master metadata and BigWigs for overlay metaplots: {MASTER_TABLE}, {INPUT_DIR / 'bigwig/*.CPM.bw'}
- Optional transcript-expression table: {TRANSCRIPT_EXPRESSION_TABLE}

Detection status definition:
- For positive ORFs, detected means the ORF_id/sample pair is present in pancreas.translated_orfs.sample_level.tsv.
- For positive ORFs, not_detected means the same ORF_id is absent in that sample.
- CPAT-negative noncoding ORFs are always labelled not_detected.

Feature definitions:
{definitions}

Statistics:
- Mann-Whitney U tests compare detected vs not_detected within each positive group.
- Cross-group p-values are intentionally not shown.
- CPAT-negative noncoding p-values are skipped because there is no detected group.

Excluded feature:
- Sample-specific variance was excluded because it is ORF-level across samples, not ORF-sample-level.

Overlay metaplots:
- Raw CPM and vector-mean normalized start-centered metaplots are plotted with group colors.
- Detected is solid; not_detected is dashed.
- CPAT-negative noncoding is plotted only as a red dashed not_detected line.

Transcript expression detected-vs-not_detected plots:
- Transcript TPM and transcript usage use ORF-sample-level values.
- Transcript TPM variance is computed per ORF within each detection_status subset across samples using population variance.
- Only detected vs not_detected within the same ORF group is annotated.
"""
    )
    print(f"Wrote {README_OUT}")


def main() -> int:
    ensure_dirs()
    archive_script(__file__)
    print("08_detected_vs_not_detected_coverage.py")
    print(f"INPUT_DIR={INPUT_DIR}")
    print(f"FIG_DIR={FIG_DIR}")
    print(f"OUT_DIR={OUT_DIR}")
    print(f"input={DETECTION_TABLE}")
    print(f"input={COVERAGE_TABLE}")
    print(f"output={COMBINED_PDF}")
    print(f"output={STATS_OUT}")

    pairs = detected_pairs()
    df = standardize_coverage_table()
    stats_tables = [mannwhitney_by_group(df, str(spec["feature_id"])) for spec in FEATURES]
    stats_df = pd.concat(stats_tables, ignore_index=True)
    stats_df.to_csv(STATS_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {STATS_OUT}")

    fig, axes = plt.subplots(4, 2, figsize=(18, 24), squeeze=False)
    for ax, spec in zip(axes.flat, FEATURES):
        plot_feature(ax, df, spec, stats_df)
    axes.flat[-1].axis("off")
    save_pdf_png(fig, COMBINED_PDF, COMBINED_PNG)

    for spec in FEATURES:
        fig, ax = plt.subplots(figsize=(9.5, 6.8))
        plot_feature(ax, df, spec, stats_df, title=f"{spec['label']}: detected vs not_detected")
        pdf_path = PDF_DIR / str(spec["individual_pdf"])
        save_pdf_png(fig, pdf_path)

    generate_metaplots(pairs)
    generate_expression_plots(pairs)
    write_readme()
    print("08_detected_vs_not_detected_coverage.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
