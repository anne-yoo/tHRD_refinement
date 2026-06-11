#!/usr/bin/env python3
"""Cryptic/noncanonical ORF exploratory analysis for the pilot study."""

from __future__ import annotations

import argparse
import math
import os
import re
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
DEFAULT_FASTA = Path(
    "/home/jiye/jiye/darkproteome/pilotstudy/data/gencode.v48.annotation.gtf.transcripts.fa"
)
STOP_CODONS = {"TAA", "TAG", "TGA"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare cryptic/noncanonical ORFs with canonical ORFs."
    )
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--orf-features", type=Path, default=None)
    parser.add_argument("--transcript-features", type=Path, default=None)
    parser.add_argument("--transcript-fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--figure-dir", type=Path, default=None)
    parser.add_argument("--top-categories", type=int, default=10)
    return parser.parse_args()


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "na", "n/a", "."}:
        return ""
    return text


def to_float(value: object) -> float:
    text = clean_text(value)
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def to_int(value: object) -> Optional[int]:
    value = to_float(value)
    if pd.isna(value):
        return None
    return int(round(float(value)))


def numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def resolve_fasta_path(path: Path) -> Tuple[Path, str]:
    if path.exists():
        return path, ""
    candidates = []
    if path.suffix == ".fa":
        candidates.append(path.with_suffix(".fasta"))
    candidates.extend(sorted(path.parent.glob("*.transcripts.fasta")))
    candidates.extend(sorted(path.parent.glob("*.fasta")))
    candidates.extend(sorted(path.parent.glob("*.fa")))
    for candidate in candidates:
        if candidate.exists():
            return candidate, f"Requested FASTA `{path}` was not found; used `{candidate}`."
    raise FileNotFoundError(path)


def split_tokens(value: object) -> List[str]:
    text = clean_text(value)
    if not text:
        return []
    return [token.strip() for token in re.split(r"[|,;]+", text) if token.strip()]


def normalize_codon(value: object) -> str:
    token = clean_text(value).upper().replace("U", "T")
    token = re.sub(r"[^ACGTN]", "", token)
    return token if token else ""


def primary_codon(value: object) -> str:
    tokens = [normalize_codon(token) for token in split_tokens(value)]
    tokens = [token for token in tokens if token]
    if not tokens:
        return normalize_codon(value)
    return tokens[0]


def classify_orf(row: pd.Series) -> Tuple[str, str, str]:
    type2 = clean_text(row.get("ORF_type2", ""))
    if type2:
        tokens = {token.lower().replace("_", "-") for token in split_tokens(type2)}
        has_canonical = "canonical" in tokens
        has_noncanonical = any(
            token in {"non-canonical", "noncanonical"} for token in tokens
        )
        if has_canonical and not has_noncanonical:
            return "canonical", "ORF_type2", "ORF_type2 is canonical"
        if has_noncanonical and not has_canonical:
            return "noncanonical", "ORF_type2", "ORF_type2 is non-canonical"
        if has_canonical and has_noncanonical:
            return "ambiguous", "ORF_type2", "ORF_type2 contains both labels"

    orf_type = clean_text(row.get("ORF_type", ""))
    if orf_type:
        if orf_type.lower() == "canonical":
            return "canonical", "ORF_type", "fallback exact ORF_type canonical"
        return "noncanonical", "ORF_type", "fallback non-canonical ORF_type"
    return "unknown", "", "no usable ORF type label"


def gc_content(seq: str) -> float:
    seq = seq.upper().replace("U", "T")
    bases = [base for base in seq if base in {"A", "C", "G", "T"}]
    if not bases:
        return np.nan
    return (bases.count("G") + bases.count("C")) / len(bases)


def count_aug(seq: str) -> int:
    seq = seq.upper().replace("U", "T")
    return sum(1 for idx in range(max(0, len(seq) - 2)) if seq[idx : idx + 3] == "ATG")


def density_per_kb(count: object, length: object) -> float:
    count_val = to_float(count)
    length_val = to_float(length)
    if pd.isna(count_val) or pd.isna(length_val) or length_val <= 0:
        return np.nan
    return count_val / length_val * 1000.0


def subseq_with_padding(seq: str, start_1based: int, end_1based: int) -> str:
    out: List[str] = []
    for pos in range(start_1based, end_1based + 1):
        if 1 <= pos <= len(seq):
            out.append(seq[pos - 1])
        else:
            out.append("N")
    return "".join(out)


def parse_target_fasta(fasta_path: Path, target_ids: Iterable[str]) -> Dict[str, str]:
    targets = {tid for tid in target_ids if tid}
    stripped_lookup: Dict[str, str] = {
        tid.split(".", 1)[0]: tid for tid in targets if tid.split(".", 1)[0]
    }
    seqs: Dict[str, List[str]] = {}
    current_id: Optional[str] = None

    with fasta_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                observed = line[1:].split()[0]
                current_id = None
                if observed in targets:
                    current_id = observed
                else:
                    current_id = stripped_lookup.get(observed.split(".", 1)[0])
                if current_id and current_id not in seqs:
                    seqs[current_id] = []
                elif current_id in seqs:
                    current_id = None
                if len(seqs) == len(targets):
                    break
                continue
            if current_id:
                seqs[current_id].append(line.upper().replace("U", "T"))

    return {tid: "".join(chunks) for tid, chunks in seqs.items()}


def add_sequence_context(orf_df: pd.DataFrame, transcript_seqs: Dict[str, str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _idx, row in orf_df.iterrows():
        tid = clean_text(row.get("mother_transcript_id", ""))
        seq = transcript_seqs.get(tid, "")
        start_pos = to_int(row.get("orf_start_transcript_pos", ""))
        end_pos = to_int(row.get("orf_end_transcript_pos", ""))
        utr5_len = to_int(row.get("utr5_length", ""))

        feature: Dict[str, object] = {
            "sequence_found": bool(seq),
            "orf_sequence_length_from_fasta": np.nan,
            "sequence_start_codon": "",
            "inferred_stop_codon": "",
            "stop_codon_source": "",
            "stop_codon_is_standard": np.nan,
            "orf_gc_content": np.nan,
            "orf_start_context_gc_63nt": np.nan,
            "kozak_context_minus6_to_plus4": "",
            "kozak_minus3": "",
            "kozak_plus4": "",
            "kozak_score": np.nan,
            "mother_aug_density_per_kb": density_per_kb(
                row.get("num_aug_codons", np.nan), row.get("transcript_sequence_length", np.nan)
            ),
            "utr5_aug_count": np.nan,
            "utr5_aug_density_per_kb": np.nan,
        }

        if seq and utr5_len is not None and utr5_len > 0:
            utr5_seq = seq[: min(utr5_len, len(seq))]
            feature["utr5_aug_count"] = count_aug(utr5_seq)
            feature["utr5_aug_density_per_kb"] = density_per_kb(
                feature["utr5_aug_count"], len(utr5_seq)
            )
        elif seq and utr5_len == 0:
            feature["utr5_aug_count"] = 0

        if not seq or start_pos is None or end_pos is None:
            rows.append(feature)
            continue

        if start_pos > end_pos:
            start_pos, end_pos = end_pos, start_pos
        start_pos = max(1, start_pos)
        end_pos = min(len(seq), end_pos)
        if start_pos > len(seq) or end_pos < 1 or start_pos > end_pos:
            rows.append(feature)
            continue

        orf_seq = seq[start_pos - 1 : end_pos]
        feature["orf_sequence_length_from_fasta"] = len(orf_seq)
        feature["sequence_start_codon"] = orf_seq[:3] if len(orf_seq) >= 3 else ""
        feature["orf_gc_content"] = gc_content(orf_seq)

        downstream = seq[end_pos : end_pos + 3]
        last3 = orf_seq[-3:] if len(orf_seq) >= 3 else ""
        if downstream in STOP_CODONS:
            feature["inferred_stop_codon"] = downstream
            feature["stop_codon_source"] = "downstream_3nt_after_orf"
        elif last3 in STOP_CODONS:
            feature["inferred_stop_codon"] = last3
            feature["stop_codon_source"] = "last_3nt_of_orf"
        elif len(downstream) == 3:
            feature["inferred_stop_codon"] = downstream
            feature["stop_codon_source"] = "downstream_3nt_after_orf_nonstandard"
        feature["stop_codon_is_standard"] = feature["inferred_stop_codon"] in STOP_CODONS

        context_start = start_pos - 30
        context_end = start_pos + 2 + 30
        feature["orf_start_context_gc_63nt"] = gc_content(
            subseq_with_padding(seq, context_start, context_end).replace("N", "")
        )

        kozak = subseq_with_padding(seq, start_pos - 6, start_pos + 3)
        feature["kozak_context_minus6_to_plus4"] = kozak
        minus3 = subseq_with_padding(seq, start_pos - 3, start_pos - 3)
        plus4 = subseq_with_padding(seq, start_pos + 3, start_pos + 3)
        feature["kozak_minus3"] = minus3
        feature["kozak_plus4"] = plus4
        feature["kozak_score"] = int(minus3 in {"A", "G"}) + int(plus4 == "G")
        rows.append(feature)

    return pd.concat([orf_df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def pvalue_text(pvalue: object) -> str:
    value = to_float(pvalue)
    if pd.isna(value):
        return "NA"
    if value == 0:
        return "<1e-300"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def number_text(value: object, digits: int = 3) -> str:
    value = to_float(value)
    if pd.isna(value):
        return "NA"
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    return f"{value:,.{digits}f}"


def iqr(values: pd.Series) -> float:
    values = values.dropna()
    if values.empty:
        return np.nan
    return values.quantile(0.75) - values.quantile(0.25)


def rank_biserial_from_u(u_stat: float, n_a: int, n_b: int) -> float:
    if n_a == 0 or n_b == 0:
        return np.nan
    return (2.0 * u_stat / (n_a * n_b)) - 1.0


def continuous_test(
    df: pd.DataFrame,
    feature: str,
    group_col: str,
    group_a: str,
    group_b: str,
    label: Optional[str] = None,
) -> Dict[str, object]:
    a = pd.to_numeric(df.loc[df[group_col] == group_a, feature], errors="coerce").dropna()
    b = pd.to_numeric(df.loc[df[group_col] == group_b, feature], errors="coerce").dropna()
    out: Dict[str, object] = {
        "feature": label or feature,
        "feature_column": feature,
        "feature_type": "continuous",
        "group_a": group_a,
        "group_b": group_b,
        "n_a": len(a),
        "n_b": len(b),
        "median_a": a.median() if len(a) else np.nan,
        "median_b": b.median() if len(b) else np.nan,
        "iqr_a": iqr(a),
        "iqr_b": iqr(b),
        "test": "Mann-Whitney U",
        "statistic": np.nan,
        "p_value": np.nan,
        "effect_size": np.nan,
        "effect_size_name": "rank_biserial",
        "note": "positive effect means group_a tends higher",
    }
    if len(a) > 0 and len(b) > 0:
        result = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
        out["statistic"] = result.statistic
        out["p_value"] = result.pvalue
        out["effect_size"] = rank_biserial_from_u(result.statistic, len(a), len(b))
    return out


def cramer_v(table: pd.DataFrame) -> float:
    chi2, _p, _dof, _expected = stats.chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    if n == 0:
        return np.nan
    r, k = table.shape
    denom = n * (min(k - 1, r - 1))
    if denom <= 0:
        return np.nan
    return math.sqrt(chi2 / denom)


def summarize_categories(values: pd.Series, max_items: int = 8) -> str:
    counts = values.fillna("missing").replace("", "missing").value_counts().head(max_items)
    return "; ".join(f"{idx}:{count}" for idx, count in counts.items())


def categorical_test(
    df: pd.DataFrame,
    feature: str,
    group_col: str,
    group_a: str,
    group_b: str,
    label: Optional[str] = None,
) -> Dict[str, object]:
    sub = df.loc[df[group_col].isin([group_a, group_b]), [group_col, feature]].copy()
    sub[feature] = sub[feature].fillna("missing").replace("", "missing").astype(str)
    table = pd.crosstab(sub[group_col], sub[feature])
    table = table.reindex([group_a, group_b]).fillna(0)
    table = table.loc[:, table.sum(axis=0) > 0]

    out: Dict[str, object] = {
        "feature": label or feature,
        "feature_column": feature,
        "feature_type": "categorical",
        "group_a": group_a,
        "group_b": group_b,
        "n_a": int((sub[group_col] == group_a).sum()),
        "n_b": int((sub[group_col] == group_b).sum()),
        "median_a": np.nan,
        "median_b": np.nan,
        "iqr_a": np.nan,
        "iqr_b": np.nan,
        "test": "chi-square",
        "statistic": np.nan,
        "p_value": np.nan,
        "effect_size": np.nan,
        "effect_size_name": "Cramer's V",
        "note": f"{group_a}: {summarize_categories(sub.loc[sub[group_col] == group_a, feature])} | "
        f"{group_b}: {summarize_categories(sub.loc[sub[group_col] == group_b, feature])}",
    }
    if table.shape[0] != 2 or table.shape[1] < 2:
        return out

    if table.shape == (2, 2):
        result = stats.fisher_exact(table.to_numpy())
        out["test"] = "Fisher exact"
        out["statistic"] = result.statistic
        out["p_value"] = result.pvalue
        out["effect_size"] = cramer_v(table)
        out["effect_size_name"] = "Cramer's V; statistic is odds_ratio"
    else:
        result = stats.chi2_contingency(table, correction=False)
        out["statistic"] = result.statistic
        out["p_value"] = result.pvalue
        out["effect_size"] = cramer_v(table)
    return out


def compact_category(series: pd.Series, top_n: int) -> pd.Series:
    clean = series.fillna("missing").replace("", "missing").astype(str)
    top = clean.value_counts().head(top_n).index
    return clean.where(clean.isin(top), "Other")


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 180,
        }
    )


def boxplot_two_groups(
    df: pd.DataFrame,
    feature: str,
    group_col: str,
    groups: Sequence[str],
    title: str,
    ylabel: str,
    path: Path,
    log1p: bool = False,
) -> None:
    data = []
    labels = []
    for group in groups:
        values = pd.to_numeric(df.loc[df[group_col] == group, feature], errors="coerce").dropna()
        if log1p:
            values = np.log1p(values[values >= 0])
        data.append(values.to_numpy())
        labels.append(f"{group}\nn={len(values):,}")

    fig, ax = plt.subplots(figsize=(7, 4.8))
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    colors = ["#E76F51", "#2A9D8F", "#8D99AE"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
    for median in bp["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.6)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, path)


def grouped_category_plot(
    df: pd.DataFrame,
    feature: str,
    group_col: str,
    groups: Sequence[str],
    title: str,
    ylabel: str,
    path: Path,
    top_n: int = 10,
) -> None:
    plot_df = df.loc[df[group_col].isin(groups), [group_col, feature]].copy()
    plot_df[feature] = compact_category(plot_df[feature], top_n)
    counts = pd.crosstab(plot_df[feature], plot_df[group_col])
    counts = counts.reindex(columns=groups).fillna(0)
    proportions = counts.div(counts.sum(axis=0), axis=1).fillna(0)
    order = counts.sum(axis=1).sort_values(ascending=False).index
    proportions = proportions.loc[order]

    x = np.arange(len(proportions))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(7.5, 0.55 * len(proportions) + 3), 4.8))
    for idx, group in enumerate(groups):
        ax.bar(
            x + (idx - 0.5) * width,
            proportions[group],
            width=width,
            label=f"{group} (n={int(counts[group].sum()):,})",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(proportions.index, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, path)


def build_orf_stats(orf_df: pd.DataFrame) -> pd.DataFrame:
    compare = orf_df.loc[orf_df["orf_class"].isin(["noncanonical", "canonical"])].copy()
    continuous = [
        ("orf_length", "ORF length"),
        ("supporting_callers", "Number of supporting callers"),
        ("supporting_samples", "Number of supporting samples"),
        ("transcript_length", "Mother transcript length"),
        ("gc_content", "Mother transcript GC content"),
        ("num_aug_codons", "Mother transcript AUG count"),
        ("mother_aug_density_per_kb", "Mother transcript AUG density per kb"),
        ("utr5_length", "5'UTR length"),
        ("utr3_length", "3'UTR length"),
        ("exon_count", "Exon count"),
        ("orf_gc_content", "ORF GC content"),
        ("orf_start_context_gc_63nt", "GC around ORF start codon"),
        ("kozak_score", "Simple Kozak score"),
        ("utr5_aug_count", "5'UTR AUG count"),
        ("utr5_aug_density_per_kb", "5'UTR AUG density per kb"),
    ]
    categorical = [
        ("start_codon_primary", "Start codon composition"),
        ("inferred_stop_codon", "Stop codon composition"),
        ("orf_frame", "ORF frame"),
        ("stop_codon_is_standard", "Standard stop codon"),
    ]

    rows: List[Dict[str, object]] = []
    for col, label in continuous:
        if col in compare.columns:
            rows.append(continuous_test(compare, col, "orf_class", "noncanonical", "canonical", label))
    for col, label in categorical:
        if col in compare.columns:
            rows.append(categorical_test(compare, col, "orf_class", "noncanonical", "canonical", label))
    return pd.DataFrame(rows)


def build_transcript_groups(transcript_df: pd.DataFrame, orf_df: pd.DataFrame) -> pd.DataFrame:
    class_by_tx: Dict[str, set] = {}
    for tid, group in orf_df.groupby("mother_transcript_id"):
        class_by_tx[str(tid)] = set(group["orf_class"].dropna().astype(str))

    out = transcript_df.copy()
    out["transcript_id"] = out["transcript_id"].astype(str)
    groups = []
    noncanonical_counts = []
    canonical_counts = []
    ambiguous_counts = []
    for tid in out["transcript_id"]:
        tx_orfs = orf_df.loc[orf_df["mother_transcript_id"].astype(str) == tid, "orf_class"]
        counts = tx_orfs.value_counts()
        noncanonical = int(counts.get("noncanonical", 0))
        canonical = int(counts.get("canonical", 0))
        ambiguous = int(counts.get("ambiguous", 0) + counts.get("unknown", 0))
        noncanonical_counts.append(noncanonical)
        canonical_counts.append(canonical)
        ambiguous_counts.append(ambiguous)
        classes = class_by_tx.get(tid, set())
        if "noncanonical" in classes:
            groups.append("noncanonical_orf_transcript")
        elif classes and classes.issubset({"canonical"}):
            groups.append("canonical_only_orf_transcript")
        elif classes:
            groups.append("ambiguous_orf_transcript")
        else:
            groups.append("no_orf_call")

    out["transcript_cryptic_group"] = groups
    out["noncanonical_orf_count"] = noncanonical_counts
    out["canonical_orf_count"] = canonical_counts
    out["ambiguous_orf_count"] = ambiguous_counts
    out["aug_density_per_kb"] = [
        density_per_kb(count, length)
        for count, length in zip(out.get("num_aug_codons", np.nan), out.get("transcript_sequence_length", np.nan))
    ]
    return out


def add_transcript_sequence_context(
    transcript_df: pd.DataFrame, transcript_seqs: Dict[str, str]
) -> pd.DataFrame:
    out = transcript_df.copy()
    utr5_counts = []
    utr5_densities = []
    for _idx, row in out.iterrows():
        tid = clean_text(row.get("transcript_id", ""))
        seq = transcript_seqs.get(tid, "")
        utr5_len = to_int(row.get("utr5_length", ""))
        if seq and utr5_len is not None and utr5_len > 0:
            utr5_seq = seq[: min(utr5_len, len(seq))]
            count = count_aug(utr5_seq)
            utr5_counts.append(count)
            utr5_densities.append(density_per_kb(count, len(utr5_seq)))
        elif seq and utr5_len == 0:
            utr5_counts.append(0)
            utr5_densities.append(np.nan)
        else:
            utr5_counts.append(np.nan)
            utr5_densities.append(np.nan)
    out["utr5_aug_count"] = utr5_counts
    out["utr5_aug_density_per_kb"] = utr5_densities
    return out


def build_transcript_stats(transcript_df: pd.DataFrame) -> pd.DataFrame:
    compare = transcript_df.loc[
        transcript_df["transcript_cryptic_group"].isin(
            ["noncanonical_orf_transcript", "canonical_only_orf_transcript"]
        )
    ].copy()
    features = [
        ("transcript_length", "Transcript length"),
        ("utr5_length", "5'UTR length"),
        ("utr3_length", "3'UTR length"),
        ("exon_count", "Exon count"),
        ("gc_content", "Transcript GC content"),
        ("num_aug_codons", "Transcript AUG count"),
        ("aug_density_per_kb", "Transcript AUG density per kb"),
        ("utr5_aug_count", "5'UTR AUG count"),
        ("utr5_aug_density_per_kb", "5'UTR AUG density per kb"),
        ("unique_orf_count", "Unique ORF count"),
    ]
    rows = []
    for col, label in features:
        if col in compare.columns:
            rows.append(
                continuous_test(
                    compare,
                    col,
                    "transcript_cryptic_group",
                    "noncanonical_orf_transcript",
                    "canonical_only_orf_transcript",
                    label,
                )
            )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: Sequence[str], max_rows: int = 12) -> str:
    if df.empty:
        return "_No rows._"
    shown = df.loc[:, [col for col in columns if col in df.columns]].head(max_rows).copy()
    for col in shown.columns:
        if col in {"p_value"}:
            shown[col] = shown[col].map(pvalue_text)
        elif shown[col].dtype.kind in "if":
            shown[col] = shown[col].map(lambda value: number_text(value, 3))
        else:
            shown[col] = shown[col].astype(str).str.replace("|", "\\|", regex=False)
    header = "| " + " | ".join(shown.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(shown.columns)) + " |"
    body = ["| " + " | ".join(str(row[col]) for col in shown.columns) + " |" for _idx, row in shown.iterrows()]
    return "\n".join([header, sep] + body)


def write_report(
    report_path: Path,
    figure_dir: Path,
    fasta_path: Path,
    fasta_note: str,
    orf_df: pd.DataFrame,
    transcript_df: pd.DataFrame,
    orf_stats: pd.DataFrame,
    transcript_stats: pd.DataFrame,
    figure_paths: Dict[str, Path],
) -> None:
    root = report_path.parent
    class_counts = orf_df["orf_class"].value_counts()
    tx_group_counts = transcript_df["transcript_cryptic_group"].value_counts()
    sequence_found = int(orf_df["sequence_found"].sum())
    standard_stops = int(orf_df["stop_codon_is_standard"].eq(True).sum())

    def rel(path: Path) -> str:
        return os.path.relpath(path, root)

    display_orf_stats = orf_stats.copy()
    display_tx_stats = transcript_stats.copy()
    stat_cols = [
        "feature",
        "feature_type",
        "n_a",
        "median_a",
        "n_b",
        "median_b",
        "test",
        "p_value",
        "effect_size",
        "effect_size_name",
    ]

    lines = [
        "# Cryptic/Noncanonical ORF Exploratory Analysis",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Inputs And Assumptions",
        f"- ORF feature table: `{report_path.parent / 'orf_features.tsv'}`",
        f"- Transcript feature table: `{report_path.parent / 'transcript_features.tsv'}`",
        f"- Transcript FASTA used: `{fasta_path}`",
    ]
    if fasta_note:
        lines.append(f"- FASTA path note: {fasta_note}")
    lines.extend(
        [
            "- Primary canonical/noncanonical label: `ORF_type2`.",
            "- `ORF_type2 == canonical` is treated as canonical.",
            "- `ORF_type2 == non-canonical` is treated as noncanonical/cryptic.",
            "- Mixed labels such as `canonical|non-canonical` are marked `ambiguous` and excluded from canonical-vs-noncanonical tests.",
            "- Simple Kozak score is 0-2: +1 for purine at -3 and +1 for G at +4 relative to the ORF start codon.",
            "- Stop codon is inferred from the 3 nt immediately after the ORF interval in transcript coordinates when available; if that is a standard stop, it is used.",
            "",
            "## Analysis Set",
        ]
    )
    for label, count in class_counts.items():
        lines.append(f"- ORFs classified as {label}: {int(count):,}")
    lines.extend(
        [
            f"- ORFs with transcript sequence found: {sequence_found:,} / {len(orf_df):,}",
            f"- ORFs with inferred standard stop codon: {standard_stops:,} / {len(orf_df):,}",
            "",
            "Transcript groups:",
        ]
    )
    for label, count in tx_group_counts.items():
        lines.append(f"- {label}: {int(count):,}")
    if int(tx_group_counts.get("no_orf_call", 0)) == 0:
        lines.append(
            "- `no_orf_call` is not included here because the provided transcript feature table contains mother transcripts from ORF calls."
        )

    lines.extend(
        [
            "",
            "## ORF-Level Comparison",
            "Main comparison: noncanonical ORFs vs canonical ORFs. Positive rank-biserial values mean noncanonical ORFs tend to have larger values.",
            "",
            markdown_table(display_orf_stats, stat_cols, max_rows=30),
            "",
        ]
    )
    for key in [
        "orf_length",
        "start_codon",
        "caller_agreement",
        "sample_recurrence",
    ]:
        lines.extend([f"![{key}]({rel(figure_paths[key])})", ""])

    lines.extend(
        [
            "## Transcript-Level Comparison",
            "Main comparison: transcripts with at least one noncanonical ORF vs transcripts with canonical ORF calls only.",
            "",
            markdown_table(display_tx_stats, stat_cols, max_rows=20),
            "",
        ]
    )
    for key in [
        "transcript_length",
        "transcript_utr5",
        "transcript_aug_density",
        "transcript_gc",
    ]:
        lines.extend([f"![{key}]({rel(figure_paths[key])})", ""])

    lines.extend(
        [
            "## Notes",
            "- Distribution figures use log1p scaling for length/count features where helpful; statistical tests use original values.",
            "- 5'UTR AUG counts are estimated from the first `utr5_length` bases of the transcript sequence.",
            "- Sequence-context features are blank for transcripts not found in the FASTA.",
            f"- Figure directory: `{figure_dir}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir
    figure_dir = args.figure_dir or analysis_dir / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    orf_path = args.orf_features or analysis_dir / "orf_features.tsv"
    transcript_path = args.transcript_features or analysis_dir / "transcript_features.tsv"
    fasta_path, fasta_note = resolve_fasta_path(args.transcript_fasta)

    print(f"Reading ORF features: {orf_path}")
    orf_df = pd.read_csv(orf_path, sep="\t", low_memory=False)
    print(f"Reading transcript features: {transcript_path}")
    transcript_df = pd.read_csv(transcript_path, sep="\t", low_memory=False)
    print(f"Using transcript FASTA: {fasta_path}")
    if fasta_note:
        print(fasta_note)

    required = {"ORF_type2", "ORF_type", "mother_transcript_id"}
    missing = sorted(required - set(orf_df.columns))
    if missing:
        raise ValueError(f"Missing required ORF feature columns: {', '.join(missing)}")

    numeric_orf_cols = [
        "orf_length",
        "orf_frame",
        "orf_start_transcript_pos",
        "orf_end_transcript_pos",
        "supporting_callers",
        "supporting_samples",
        "transcript_length",
        "gc_content",
        "num_aug_codons",
        "utr5_length",
        "utr3_length",
        "exon_count",
        "transcript_sequence_length",
    ]
    numeric_tx_cols = [
        "transcript_length",
        "exon_count",
        "utr5_length",
        "utr3_length",
        "gc_content",
        "num_aug_codons",
        "transcript_sequence_length",
        "unique_orf_count",
    ]
    orf_df = numeric_columns(orf_df, numeric_orf_cols)
    transcript_df = numeric_columns(transcript_df, numeric_tx_cols)

    class_info = orf_df.apply(classify_orf, axis=1)
    orf_df["orf_class"] = [item[0] for item in class_info]
    orf_df["orf_class_source"] = [item[1] for item in class_info]
    orf_df["orf_class_note"] = [item[2] for item in class_info]
    orf_df["start_codon_primary"] = orf_df["start_codon"].map(primary_codon)

    target_transcripts = sorted(set(orf_df["mother_transcript_id"].dropna().astype(str)))
    print(f"Parsing FASTA for {len(target_transcripts):,} mother transcripts")
    transcript_seqs = parse_target_fasta(fasta_path, target_transcripts)
    print(f"Transcript sequences found: {len(transcript_seqs):,}")

    cryptic_orf = add_sequence_context(orf_df, transcript_seqs)
    transcript_df = build_transcript_groups(transcript_df, cryptic_orf)
    transcript_df = add_transcript_sequence_context(transcript_df, transcript_seqs)

    orf_stats = build_orf_stats(cryptic_orf)
    transcript_stats = build_transcript_stats(transcript_df)

    cryptic_orf_path = analysis_dir / "cryptic_orf_features.tsv"
    orf_stats_path = analysis_dir / "cryptic_orf_comparison_stats.tsv"
    transcript_stats_path = analysis_dir / "cryptic_transcript_comparison_stats.tsv"
    report_path = analysis_dir / "cryptic_exploratory_report.md"

    cryptic_orf.to_csv(cryptic_orf_path, sep="\t", index=False, na_rep="")
    orf_stats.to_csv(orf_stats_path, sep="\t", index=False, na_rep="")
    transcript_stats.to_csv(transcript_stats_path, sep="\t", index=False, na_rep="")

    orf_compare = cryptic_orf.loc[
        cryptic_orf["orf_class"].isin(["noncanonical", "canonical"])
    ].copy()
    tx_compare = transcript_df.loc[
        transcript_df["transcript_cryptic_group"].isin(
            ["noncanonical_orf_transcript", "canonical_only_orf_transcript"]
        )
    ].copy()

    figure_paths: Dict[str, Path] = {}
    figure_paths["orf_length"] = figure_dir / "cryptic_orf_length_distribution.png"
    boxplot_two_groups(
        orf_compare,
        "orf_length",
        "orf_class",
        ["noncanonical", "canonical"],
        "ORF length: noncanonical vs canonical",
        "log1p(ORF length)",
        figure_paths["orf_length"],
        log1p=True,
    )
    figure_paths["start_codon"] = figure_dir / "cryptic_start_codon_composition.png"
    grouped_category_plot(
        orf_compare,
        "start_codon_primary",
        "orf_class",
        ["noncanonical", "canonical"],
        "Start codon composition",
        "Proportion within ORF class",
        figure_paths["start_codon"],
        args.top_categories,
    )
    figure_paths["caller_agreement"] = figure_dir / "cryptic_caller_agreement.png"
    grouped_category_plot(
        orf_compare,
        "supporting_callers",
        "orf_class",
        ["noncanonical", "canonical"],
        "Caller agreement",
        "Proportion within ORF class",
        figure_paths["caller_agreement"],
        args.top_categories,
    )
    figure_paths["sample_recurrence"] = figure_dir / "cryptic_sample_recurrence.png"
    grouped_category_plot(
        orf_compare,
        "supporting_samples",
        "orf_class",
        ["noncanonical", "canonical"],
        "Sample recurrence",
        "Proportion within ORF class",
        figure_paths["sample_recurrence"],
        args.top_categories,
    )

    tx_groups = ["noncanonical_orf_transcript", "canonical_only_orf_transcript"]
    figure_paths["transcript_length"] = figure_dir / "cryptic_transcript_length_distribution.png"
    boxplot_two_groups(
        tx_compare,
        "transcript_length",
        "transcript_cryptic_group",
        tx_groups,
        "Transcript length by cryptic ORF status",
        "log1p(transcript length)",
        figure_paths["transcript_length"],
        log1p=True,
    )
    figure_paths["transcript_utr5"] = figure_dir / "cryptic_transcript_utr5_distribution.png"
    boxplot_two_groups(
        tx_compare,
        "utr5_length",
        "transcript_cryptic_group",
        tx_groups,
        "5'UTR length by cryptic ORF status",
        "log1p(5'UTR length)",
        figure_paths["transcript_utr5"],
        log1p=True,
    )
    figure_paths["transcript_aug_density"] = figure_dir / "cryptic_transcript_aug_density.png"
    boxplot_two_groups(
        tx_compare,
        "aug_density_per_kb",
        "transcript_cryptic_group",
        tx_groups,
        "Transcript AUG density by cryptic ORF status",
        "AUGs per kb",
        figure_paths["transcript_aug_density"],
        log1p=False,
    )
    figure_paths["transcript_gc"] = figure_dir / "cryptic_transcript_gc_content.png"
    boxplot_two_groups(
        tx_compare,
        "gc_content",
        "transcript_cryptic_group",
        tx_groups,
        "Transcript GC content by cryptic ORF status",
        "GC content",
        figure_paths["transcript_gc"],
        log1p=False,
    )

    write_report(
        report_path,
        figure_dir,
        fasta_path,
        fasta_note,
        cryptic_orf,
        transcript_df,
        orf_stats,
        transcript_stats,
        figure_paths,
    )

    print(f"Wrote {cryptic_orf_path}")
    print(f"Wrote {orf_stats_path}")
    print(f"Wrote {transcript_stats_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote figures to {figure_dir}")


if __name__ == "__main__":
    main()
