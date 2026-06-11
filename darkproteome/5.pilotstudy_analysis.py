#!/usr/bin/env python3
"""Exploratory analysis for pilot Ribo-seq ORF feature tables."""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_ANALYSIS_DIR = Path("/home/jiye/jiye/darkproteome/pilotstudy/analysis")
DEFAULT_DATA_DIR = Path("/home/jiye/jiye/darkproteome/pilotstudy/data")

FEATURE_LABELS = {
    "transcript_length": "Transcript length",
    "utr5_length": "5'UTR length",
    "utr3_length": "3'UTR length",
    "exon_count": "Exon count",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create exploratory figures and report from pilot ORF/transcript feature tables."
    )
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--orf-features", type=Path, default=None)
    parser.add_argument("--transcript-features", type=Path, default=None)
    parser.add_argument("--gtf-file", type=Path, default=None)
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=None,
        help="Directory for PNG figures. Defaults to analysis-dir.",
    )
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t", low_memory=False)


def choose_gtf(data_dir: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit:
        return explicit
    matches = sorted(data_dir.glob("*.gtf"))
    return matches[0] if matches else None


def load_prep_module():
    prep_path = Path(__file__).with_name("5.pilotstudy_prep.py")
    if not prep_path.exists():
        raise FileNotFoundError(
            f"Cannot find helper script for GTF parsing: {prep_path}"
        )
    spec = importlib.util.spec_from_file_location("pilotstudy_prep_helpers", prep_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load helper script: {prep_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def pvalue_text(pvalue: float) -> str:
    if pd.isna(pvalue):
        return "NA"
    if pvalue == 0:
        return "<1e-300"
    if pvalue < 1e-4:
        return f"{pvalue:.2e}"
    return f"{pvalue:.4f}"


def number_text(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        if math.isfinite(float(value)) and abs(float(value) - round(float(value))) < 1e-9:
            return f"{int(round(float(value))):,}"
        return f"{float(value):,.{digits}f}"
    return str(value)


def markdown_table(df: pd.DataFrame, columns: Sequence[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    shown = df.loc[:, [col for col in columns if col in df.columns]].head(max_rows).copy()
    if shown.empty:
        return "_No displayable columns._"

    def clean_cell(value: object) -> str:
        if value is None or pd.isna(value):
            return ""
        text = number_text(value) if isinstance(value, (int, float, np.integer, np.floating)) else str(value)
        return text.replace("|", "\\|")

    header = "| " + " | ".join(shown.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(shown.columns)) + " |"
    rows = [
        "| " + " | ".join(clean_cell(row[col]) for col in shown.columns) + " |"
        for _idx, row in shown.iterrows()
    ]
    return "\n".join([header, sep] + rows)


def sample_for_plot(values: pd.Series, max_n: int = 60000, seed: int = 13) -> np.ndarray:
    values = values.dropna()
    if len(values) > max_n:
        values = values.sample(max_n, random_state=seed)
    return values.to_numpy()


def safe_log1p(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr >= 0]
    return np.log1p(arr)


def set_common_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 180,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_transcript_universe(
    transcript_features: pd.DataFrame,
    orf_features: pd.DataFrame,
    gtf_file: Optional[Path],
) -> Tuple[pd.DataFrame, str]:
    orf_counts = Counter(
        orf_features["mother_transcript_id"].dropna().astype(str)
        if "mother_transcript_id" in orf_features.columns
        else []
    )
    positive_ids = set(orf_counts)

    has_negatives = (
        "unique_orf_count" in transcript_features.columns
        and pd.to_numeric(transcript_features["unique_orf_count"], errors="coerce").fillna(0).eq(0).any()
    )

    if has_negatives:
        universe = transcript_features.copy()
        source = "transcript_features.tsv"
    elif gtf_file is not None and gtf_file.exists():
        prep = load_prep_module()
        print(f"Parsing GTF transcript universe for ORF-negative controls: {gtf_file}")
        records = prep.parse_gtf_features(gtf_file, [])
        universe = prep.build_transcript_feature_table(records.keys(), records, {}, orf_counts)
        source = str(gtf_file)
    else:
        universe = transcript_features.copy()
        source = "transcript_features.tsv only; ORF-negative comparison unavailable"

    if "transcript_id" not in universe.columns:
        raise ValueError("Transcript feature table must contain transcript_id.")

    universe["transcript_id"] = universe["transcript_id"].astype(str)
    universe["unique_orf_count"] = universe["transcript_id"].map(orf_counts).fillna(0).astype(int)
    universe["orf_status"] = np.where(
        universe["transcript_id"].isin(positive_ids), "ORF-positive", "ORF-negative"
    )

    numeric_cols = [
        "transcript_length",
        "utr5_length",
        "utr3_length",
        "exon_count",
        "cds_length",
        "transcript_sequence_length",
        "gc_content",
        "num_aug_codons",
        "unique_orf_count",
    ]
    universe = coerce_numeric(universe, numeric_cols)
    return universe, source


def transcript_distribution_stats(universe: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for feature, label in FEATURE_LABELS.items():
        if feature not in universe.columns:
            continue
        pos = universe.loc[universe["orf_status"] == "ORF-positive", feature].dropna()
        neg = universe.loc[universe["orf_status"] == "ORF-negative", feature].dropna()
        row: Dict[str, object] = {
            "feature": label,
            "n_orf_positive": len(pos),
            "median_orf_positive": pos.median() if len(pos) else np.nan,
            "iqr_orf_positive": pos.quantile(0.75) - pos.quantile(0.25) if len(pos) else np.nan,
            "n_orf_negative": len(neg),
            "median_orf_negative": neg.median() if len(neg) else np.nan,
            "iqr_orf_negative": neg.quantile(0.75) - neg.quantile(0.25) if len(neg) else np.nan,
            "test": "Mann-Whitney U",
            "p_value": np.nan,
            "rank_biserial": np.nan,
        }
        if len(pos) > 0 and len(neg) > 0:
            result = stats.mannwhitneyu(pos, neg, alternative="two-sided", method="asymptotic")
            row["p_value"] = result.pvalue
            row["rank_biserial"] = (2 * result.statistic / (len(pos) * len(neg))) - 1
        rows.append(row)
    return pd.DataFrame(rows)


def plot_transcript_distributions(universe: pd.DataFrame, figure_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    colors = ["#277DA1", "#B8B8B8"]
    statuses = ["ORF-positive", "ORF-negative"]
    counts = universe["orf_status"].value_counts()
    labels = [f"{status}\nn={counts.get(status, 0):,}" for status in statuses]

    for ax, (feature, label) in zip(axes, FEATURE_LABELS.items()):
        if feature not in universe.columns:
            ax.axis("off")
            continue
        data = [
            safe_log1p(sample_for_plot(universe.loc[universe["orf_status"] == status, feature]))
            for status in statuses
        ]
        bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for median in bp["medians"]:
            median.set_color("#222222")
            median.set_linewidth(1.6)
        ax.set_title(label)
        ax.set_ylabel("log1p(value)")
        ax.grid(axis="y", alpha=0.25)

    path = figure_dir / "transcript_feature_distributions.png"
    save_figure(fig, path)
    return path


def plot_orf_recurrence(orf_df: pd.DataFrame, figure_dir: Path) -> Path:
    counts = (
        pd.to_numeric(orf_df["supporting_samples"], errors="coerce")
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(counts.index.astype(str), counts.values, color="#43AA8B")
    ax.set_xlabel("Supporting samples per ORF")
    ax.set_ylabel("ORF count")
    ax.set_title("ORF recurrence across samples")
    ax.grid(axis="y", alpha=0.25)
    path = figure_dir / "orf_supporting_samples_histogram.png"
    save_figure(fig, path)
    return path


def truncate_label(text: object, width: int = 70) -> str:
    text = "" if pd.isna(text) else str(text)
    if len(text) <= width:
        return text
    return text[: width - 3] + "..."


def orf_display_label(row: pd.Series) -> str:
    gene = "" if pd.isna(row.get("gene_name", "")) else str(row.get("gene_name", ""))
    orf_id = "" if pd.isna(row.get("ORF_id", "")) else str(row.get("ORF_id", ""))
    if gene:
        label = f"{gene} | {orf_id}"
    else:
        label = orf_id
    return truncate_label(label, 72)


def plot_top_recurrent_orfs(top_orfs: pd.DataFrame, figure_dir: Path) -> Optional[Path]:
    if top_orfs.empty:
        return None
    plot_df = top_orfs.copy().iloc[::-1]
    labels = [orf_display_label(row) for _idx, row in plot_df.iterrows()]
    values = pd.to_numeric(plot_df["supporting_samples"], errors="coerce")
    colors = pd.to_numeric(plot_df.get("supporting_callers", 0), errors="coerce").fillna(0)

    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(plot_df) + 1.5)))
    bars = ax.barh(range(len(plot_df)), values, color="#F8961E")
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Supporting samples")
    ax.set_title("Top recurrent ORFs")
    ax.grid(axis="x", alpha=0.25)
    for idx, (bar, caller_count) in enumerate(zip(bars, colors)):
        ax.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{int(caller_count)} callers",
            va="center",
            fontsize=8,
        )
    path = figure_dir / "top_recurrent_orfs.png"
    save_figure(fig, path)
    return path


def plot_caller_count(orf_df: pd.DataFrame, figure_dir: Path) -> Path:
    counts = (
        pd.to_numeric(orf_df["supporting_callers"], errors="coerce")
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(counts.index.astype(str), counts.values, color="#577590")
    ax.set_xlabel("Number of supporting callers")
    ax.set_ylabel("ORF count")
    ax.set_title("Caller agreement")
    ax.grid(axis="y", alpha=0.25)
    path = figure_dir / "caller_agreement_counts.png"
    save_figure(fig, path)
    return path


def caller_length_stats(orf_df: pd.DataFrame) -> Dict[str, object]:
    valid = orf_df[["supporting_callers", "orf_length"]].dropna().copy()
    valid["supporting_callers"] = valid["supporting_callers"].astype(int)
    groups = [
        group["orf_length"].to_numpy()
        for _caller_count, group in valid.groupby("supporting_callers")
        if len(group) > 0
    ]
    out: Dict[str, object] = {
        "kruskal_h": np.nan,
        "kruskal_p": np.nan,
        "spearman_r": np.nan,
        "spearman_p": np.nan,
    }
    if len(groups) >= 2:
        result = stats.kruskal(*groups)
        out["kruskal_h"] = result.statistic
        out["kruskal_p"] = result.pvalue
    if len(valid) >= 3 and valid["supporting_callers"].nunique() >= 2:
        result = stats.spearmanr(valid["supporting_callers"], valid["orf_length"])
        out["spearman_r"] = result.statistic
        out["spearman_p"] = result.pvalue
    return out


def plot_orf_length_by_callers(orf_df: pd.DataFrame, figure_dir: Path) -> Path:
    valid = orf_df[["supporting_callers", "orf_length"]].dropna().copy()
    valid["supporting_callers"] = valid["supporting_callers"].astype(int)
    caller_counts = sorted(valid["supporting_callers"].unique())
    data = [
        safe_log1p(valid.loc[valid["supporting_callers"] == caller_count, "orf_length"])
        for caller_count in caller_counts
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(
        data,
        tick_labels=[str(x) for x in caller_counts],
        showfliers=False,
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#90BE6D")
        patch.set_alpha(0.8)
    for median in bp["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.6)
    ax.set_xlabel("Number of supporting callers")
    ax.set_ylabel("log1p(ORF length)")
    ax.set_title("ORF length by caller agreement")
    ax.grid(axis="y", alpha=0.25)
    path = figure_dir / "orf_length_by_callers.png"
    save_figure(fig, path)
    return path


def build_burden_table(universe: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "transcript_id",
        "gene_id",
        "gene_name",
        "transcript_type",
        "unique_orf_count",
        "transcript_length",
        "exon_count",
        "utr5_length",
        "utr3_length",
    ]
    return universe[[col for col in keep_cols if col in universe.columns]].copy()


def plot_transcript_burden(burden: pd.DataFrame, figure_dir: Path) -> Path:
    positive = burden.loc[burden["unique_orf_count"] > 0, "unique_orf_count"].dropna().astype(int)
    counts = positive.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(counts.index.astype(str), counts.values, color="#4D908E")
    ax.set_xlabel("Unique ORFs per ORF-positive transcript")
    ax.set_ylabel("Transcript count")
    ax.set_title("Transcript ORF burden")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    path = figure_dir / "transcript_orf_burden_histogram.png"
    save_figure(fig, path)
    return path


def plot_top_multiorf_transcripts(top_transcripts: pd.DataFrame, figure_dir: Path) -> Optional[Path]:
    if top_transcripts.empty:
        return None
    plot_df = top_transcripts.copy().iloc[::-1]
    labels = []
    for _idx, row in plot_df.iterrows():
        gene = "" if pd.isna(row.get("gene_name", "")) else str(row.get("gene_name", ""))
        tid = "" if pd.isna(row.get("transcript_id", "")) else str(row.get("transcript_id", ""))
        labels.append(truncate_label(f"{gene} | {tid}" if gene else tid, 65))

    fig, ax = plt.subplots(figsize=(9, max(5, 0.35 * len(plot_df) + 1.5)))
    ax.barh(range(len(plot_df)), plot_df["unique_orf_count"], color="#F3722C")
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Unique ORFs")
    ax.set_title("Top transcripts with multiple ORFs")
    ax.grid(axis="x", alpha=0.25)
    path = figure_dir / "top_multiorf_transcripts.png"
    save_figure(fig, path)
    return path


def relative_path(path: Optional[Path], root: Path) -> Optional[str]:
    if path is None:
        return None
    return os.path.relpath(path, root)


def write_report(
    report_path: Path,
    figure_paths: Dict[str, Optional[Path]],
    orf_df: pd.DataFrame,
    universe: pd.DataFrame,
    transcript_source: str,
    dist_stats: pd.DataFrame,
    top_orfs: pd.DataFrame,
    caller_counts: pd.Series,
    caller_stats: Dict[str, object],
    burden: pd.DataFrame,
    top_transcripts: pd.DataFrame,
    top_orfs_path: Path,
    top_transcripts_path: Path,
) -> None:
    root = report_path.parent
    positive_n = int((universe["orf_status"] == "ORF-positive").sum())
    negative_n = int((universe["orf_status"] == "ORF-negative").sum())
    recurrent_n = int((orf_df["supporting_samples"] > 1).sum())
    multi_orf_n = int((burden["unique_orf_count"] > 1).sum())

    dist_display = dist_stats.copy()
    if not dist_display.empty:
        dist_display["p_value"] = dist_display["p_value"].map(pvalue_text)
        dist_display["rank_biserial"] = dist_display["rank_biserial"].map(
            lambda x: number_text(x, 3)
        )
        for col in [
            "median_orf_positive",
            "iqr_orf_positive",
            "median_orf_negative",
            "iqr_orf_negative",
        ]:
            dist_display[col] = dist_display[col].map(lambda x: number_text(x, 1))

    caller_table = caller_counts.rename_axis("supporting_callers").reset_index(name="orf_count")

    rel_figs = {key: relative_path(path, root) for key, path in figure_paths.items()}
    lines = [
        "# Pilot ORF Exploratory Analysis",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Analysis Set",
        f"- Unique ORFs: {len(orf_df):,}",
        f"- Transcript universe source: `{transcript_source}`",
        f"- ORF-positive transcripts: {positive_n:,}",
        f"- ORF-negative transcripts: {negative_n:,}",
        f"- Recurrent ORFs detected in more than one sample: {recurrent_n:,}",
        f"- Transcripts containing more than one unique ORF: {multi_orf_n:,}",
        "",
        "## Transcript Feature Distributions",
        "ORF-positive transcripts were compared with ORF-negative GENCODE transcripts using two-sided Mann-Whitney U tests. Rank-biserial values above zero mean the ORF-positive group tends to have larger values.",
        "",
        markdown_table(
            dist_display,
            [
                "feature",
                "n_orf_positive",
                "median_orf_positive",
                "iqr_orf_positive",
                "n_orf_negative",
                "median_orf_negative",
                "iqr_orf_negative",
                "test",
                "p_value",
                "rank_biserial",
            ],
            max_rows=10,
        ),
        "",
    ]
    if rel_figs.get("transcript_distributions"):
        lines.extend(
            [
                f"![Transcript feature distributions]({rel_figs['transcript_distributions']})",
                "",
            ]
        )

    lines.extend(
        [
            "## ORF Recurrence Analysis",
            f"- Median supporting samples per ORF: {number_text(orf_df['supporting_samples'].median(), 1)}",
            f"- Maximum supporting samples per ORF: {number_text(orf_df['supporting_samples'].max(), 0)}",
            f"- Top recurrent ORF table: `{relative_path(top_orfs_path, root)}`",
            "",
        ]
    )
    if rel_figs.get("orf_recurrence"):
        lines.extend([f"![ORF recurrence]({rel_figs['orf_recurrence']})", ""])
    if rel_figs.get("top_recurrent_orfs"):
        lines.extend([f"![Top recurrent ORFs]({rel_figs['top_recurrent_orfs']})", ""])
    lines.extend(
        [
            "Top recurrent ORFs:",
            "",
            markdown_table(
                top_orfs,
                [
                    "ORF_id",
                    "mother_transcript_id",
                    "gene_name",
                    "ORF_type",
                    "supporting_samples",
                    "supporting_callers",
                    "orf_length",
                    "caller_list",
                    "sample_list",
                ],
                max_rows=15,
            ),
            "",
            "## Caller Agreement Analysis",
            "ORF length was compared across caller-count groups using a Kruskal-Wallis test; Spearman correlation summarizes the monotonic association between ORF length and caller count.",
            "",
            markdown_table(caller_table, ["supporting_callers", "orf_count"], max_rows=20),
            "",
            f"- Kruskal-Wallis H: {number_text(caller_stats.get('kruskal_h'), 3)}",
            f"- Kruskal-Wallis p-value: {pvalue_text(caller_stats.get('kruskal_p'))}",
            f"- Spearman rho: {number_text(caller_stats.get('spearman_r'), 3)}",
            f"- Spearman p-value: {pvalue_text(caller_stats.get('spearman_p'))}",
            "",
        ]
    )
    if rel_figs.get("caller_counts"):
        lines.extend([f"![Caller agreement counts]({rel_figs['caller_counts']})", ""])
    if rel_figs.get("orf_length_by_callers"):
        lines.extend([f"![ORF length by callers]({rel_figs['orf_length_by_callers']})", ""])

    lines.extend(
        [
            "## Transcript Burden Analysis",
            f"- ORF-positive transcripts with exactly one ORF: {int((burden['unique_orf_count'] == 1).sum()):,}",
            f"- ORF-positive transcripts with multiple ORFs: {multi_orf_n:,}",
            f"- Maximum ORFs in one transcript: {int(burden['unique_orf_count'].max()):,}",
            f"- Top multi-ORF transcript table: `{relative_path(top_transcripts_path, root)}`",
            "",
        ]
    )
    if rel_figs.get("burden"):
        lines.extend([f"![Transcript ORF burden]({rel_figs['burden']})", ""])
    if rel_figs.get("top_multiorf_transcripts"):
        lines.extend(
            [f"![Top multi-ORF transcripts]({rel_figs['top_multiorf_transcripts']})", ""]
        )
    lines.extend(
        [
            "Top transcripts containing multiple ORFs:",
            "",
            markdown_table(
                top_transcripts,
                [
                    "transcript_id",
                    "gene_name",
                    "transcript_type",
                    "unique_orf_count",
                    "transcript_length",
                    "exon_count",
                    "utr5_length",
                    "utr3_length",
                ],
                max_rows=15,
            ),
            "",
            "## Notes",
            "- Length-like distribution plots use log1p values for readability; statistical tests use the original values.",
            "- ORF-negative transcripts are transcripts in the GENCODE transcript universe with no ORF in `orf_features.tsv`.",
            "- Very small p-values are expected with this transcript universe size, so medians and rank-biserial effect sizes should be interpreted alongside p-values.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir
    figure_dir = args.figure_dir or analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    set_common_style()

    orf_path = args.orf_features or analysis_dir / "orf_features.tsv"
    transcript_path = args.transcript_features or analysis_dir / "transcript_features.tsv"
    gtf_file = choose_gtf(args.data_dir, args.gtf_file)

    print(f"Reading ORF features: {orf_path}")
    orf_df = read_table(orf_path)
    print(f"Reading transcript features: {transcript_path}")
    transcript_df = read_table(transcript_path)

    orf_numeric_cols = [
        "orf_length",
        "supporting_samples",
        "supporting_callers",
        "transcript_length",
        "exon_count",
        "utr5_length",
        "utr3_length",
    ]
    orf_df = coerce_numeric(orf_df, orf_numeric_cols)

    required_orf_cols = {"ORF_id", "mother_transcript_id", "supporting_samples", "supporting_callers", "orf_length"}
    missing = sorted(required_orf_cols - set(orf_df.columns))
    if missing:
        raise ValueError(f"Missing required ORF feature columns: {', '.join(missing)}")

    universe, transcript_source = build_transcript_universe(transcript_df, orf_df, gtf_file)
    dist_stats = transcript_distribution_stats(universe)

    sort_cols = ["supporting_samples", "supporting_callers", "input_row_count", "orf_length"]
    sort_cols = [col for col in sort_cols if col in orf_df.columns]
    top_orfs = (
        orf_df.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        .head(args.top_n)
        .copy()
    )

    caller_counts = (
        orf_df["supporting_callers"].dropna().astype(int).value_counts().sort_index()
    )
    caller_stats = caller_length_stats(orf_df)

    burden = build_burden_table(universe)
    top_transcripts = (
        burden.loc[burden["unique_orf_count"] > 1]
        .sort_values(["unique_orf_count", "transcript_length"], ascending=[False, False])
        .head(args.top_n)
        .copy()
    )

    top_orfs_path = analysis_dir / "top_recurrent_orfs.tsv"
    top_transcripts_path = analysis_dir / "top_multiorf_transcripts.tsv"
    top_orfs.to_csv(top_orfs_path, sep="\t", index=False, na_rep="")
    top_transcripts.to_csv(top_transcripts_path, sep="\t", index=False, na_rep="")

    figure_paths: Dict[str, Optional[Path]] = {
        "transcript_distributions": plot_transcript_distributions(universe, figure_dir),
        "orf_recurrence": plot_orf_recurrence(orf_df, figure_dir),
        "top_recurrent_orfs": plot_top_recurrent_orfs(top_orfs, figure_dir),
        "caller_counts": plot_caller_count(orf_df, figure_dir),
        "orf_length_by_callers": plot_orf_length_by_callers(orf_df, figure_dir),
        "burden": plot_transcript_burden(burden, figure_dir),
        "top_multiorf_transcripts": plot_top_multiorf_transcripts(top_transcripts, figure_dir),
    }

    report_path = analysis_dir / "exploratory_report.md"
    write_report(
        report_path=report_path,
        figure_paths=figure_paths,
        orf_df=orf_df,
        universe=universe,
        transcript_source=transcript_source,
        dist_stats=dist_stats,
        top_orfs=top_orfs,
        caller_counts=caller_counts,
        caller_stats=caller_stats,
        burden=burden,
        top_transcripts=top_transcripts,
        top_orfs_path=top_orfs_path,
        top_transcripts_path=top_transcripts_path,
    )

    print(f"Wrote report: {report_path}")
    print(f"Wrote top recurrent ORF table: {top_orfs_path}")
    print(f"Wrote top multi-ORF transcript table: {top_transcripts_path}")
    print(f"Wrote figures to: {figure_dir}")


if __name__ == "__main__":
    main()
