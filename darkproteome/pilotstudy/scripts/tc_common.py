#!/usr/bin/env python3
"""Shared helpers for Transcript Context Module pilot feature scripts."""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


INPUT_DIR = Path(
    os.environ.get("INPUT_DIR", "/home/jiye/jiye/darkproteome/ORFstudy/pilot/pancreas8samples")
)
FIG_DIR = Path(os.environ.get("FIG_DIR", "/home/jiye/jiye/darkproteome/ORFstudy/pilot/figures"))
GENOME_FA = Path(os.environ.get("GENOME_FA", "/home/jiye/jiye/darkproteome/ORFstudy/pilot/hg38.fa"))
PROGRESS_EVERY = int(os.environ.get("PROGRESS_EVERY", "10000"))

TABLE_DIR = FIG_DIR / "tables"
PDF_DIR = FIG_DIR / "pdf"
PNG_DIR = FIG_DIR / "png"
SCRIPT_ARCHIVE_DIR = FIG_DIR / "scripts"

GROUP_ORDER = [
    "group1_canonical_translated_ORF",
    "group2_translated_AUG_cryptic_ORF",
    "group3_translated_nonAUG_cryptic_ORF",
]
GROUP_LABELS = {
    "group1_canonical_translated_ORF": "Canonical",
    "group2_translated_AUG_cryptic_ORF": "AUG cryptic",
    "group3_translated_nonAUG_cryptic_ORF": "nonAUG cryptic",
}
PAIRWISE_GROUPS = [
    ("group1_canonical_translated_ORF", "group2_translated_AUG_cryptic_ORF"),
    ("group1_canonical_translated_ORF", "group3_translated_nonAUG_cryptic_ORF"),
    ("group2_translated_AUG_cryptic_ORF", "group3_translated_nonAUG_cryptic_ORF"),
]

UNSTRANDED_SAMPLES = [
    "GSM3395010",
    "GSM3395011",
    "GSM3395012",
    "GSM3395013",
    "GSM3395014",
    "GSM3395015",
]
FORWARD_SENSE_SAMPLES = ["GSM5099832", "GSM5099835"]
ALL_SAMPLES = UNSTRANDED_SAMPLES + FORWARD_SENSE_SAMPLES

NEAR_COGNATE_STARTS = {
    "CTG",
    "GTG",
    "TTG",
    "ACG",
    "ATA",
    "ATT",
    "ATC",
    "AAG",
    "AGG",
}

REVCOMP_TABLE = str.maketrans("ACGTUNacgtun", "TGCAANtgcaan")


def ensure_fig_dirs() -> None:
    for path in [TABLE_DIR, PDF_DIR, PNG_DIR, SCRIPT_ARCHIVE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def archive_script(script_path: str | Path) -> None:
    ensure_fig_dirs()
    script_path = Path(script_path)
    if script_path.exists():
        shutil.copy2(script_path, SCRIPT_ARCHIVE_DIR / script_path.name)
    helper = Path(__file__)
    if helper.exists():
        shutil.copy2(helper, SCRIPT_ARCHIVE_DIR / helper.name)


def print_paths(*, script_name: str, inputs: Sequence[Path], outputs: Sequence[Path]) -> None:
    print(script_name)
    print(f"INPUT_DIR={INPUT_DIR}")
    print(f"FIG_DIR={FIG_DIR}")
    for path in inputs:
        print(f"input={path}")
    for path in outputs:
        print(f"output={path}")


def read_tsv(path: Path, required: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Input table not found: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str)
    if required:
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise SystemExit(f"Missing required column(s) in {path}: {', '.join(missing)}")
    return df


def read_groups() -> pd.DataFrame:
    path = INPUT_DIR / "tables" / "orf_groups.combined_metadata.tsv"
    required = [
        "ORF_id",
        "group",
        "primary_noncanonical_category",
        "chr",
        "start0",
        "end0",
        "strand",
        "ORF_type",
        "start_codon",
        "detected_samples",
        "n_detected_samples",
    ]
    df = read_tsv(path, required=required).reset_index(drop=True)
    df["start0"] = pd.to_numeric(df["start0"], errors="coerce")
    df["end0"] = pd.to_numeric(df["end0"], errors="coerce")
    df["orf_length_nt"] = df["end0"] - df["start0"]
    df["start_codon"] = df["start_codon"].map(normalize_codon)
    return df


def read_sequence_context() -> pd.DataFrame:
    path = INPUT_DIR / "tables" / "orf_sequence_context_features.tsv"
    required = ["ORF_id", "group", "start_codon"]
    df = read_tsv(path, required=required).reset_index(drop=True)
    for col in [
        "kozak_minus3_AG",
        "kozak_plus4_G",
        "strong_kozak",
        "start_codon_matches",
    ]:
        if col in df.columns:
            df[col] = to_bool(df[col])
    for col in [
        "start0",
        "end0",
        "GC_start_window_20nt",
        "upstream_AUG_count_200nt",
        "upstream_AUG_density_200nt",
        "distance_to_nearest_upstream_AUG",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["start_codon"] = df["start_codon"].map(normalize_codon)
    df["start_codon_type"] = df["start_codon"].map(start_codon_type)
    return df


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "na", "n/a", "."}:
        return ""
    return text


def normalize_codon(value: object) -> str:
    text = clean_text(value).upper().replace("U", "T")
    return re.split(r"[\s,;|/]+", text)[0] if text else ""


def start_codon_type(codon: object) -> str:
    norm = normalize_codon(codon)
    if norm == "ATG":
        return "AUG"
    if norm in NEAR_COGNATE_STARTS:
        return "near-cognate"
    if norm:
        return "other"
    return "missing"


def to_bool(series: pd.Series) -> pd.Series:
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
    }
    return series.map(lambda x: mapping.get(clean_text(x).lower(), pd.NA)).astype("boolean")


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def group_order_present(df: pd.DataFrame, group_col: str = "group") -> List[str]:
    present = set(df[group_col].dropna().astype(str))
    ordered = [group for group in GROUP_ORDER if group in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def display_group(group: str) -> str:
    return GROUP_LABELS.get(group, group)


def bh_adjust(pvalues: Iterable[object]) -> List[float]:
    values = [np.nan if pd.isna(p) else float(p) for p in pvalues]
    valid = [(idx, p) for idx, p in enumerate(values) if not np.isnan(p)]
    if not valid:
        return [np.nan] * len(values)
    valid_sorted = sorted(valid, key=lambda item: item[1])
    m = len(valid_sorted)
    adjusted = [np.nan] * len(values)
    running = 1.0
    for rank_from_end, (idx, p) in enumerate(reversed(valid_sorted), start=1):
        rank = m - rank_from_end + 1
        running = min(running, p * m / rank)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def pvalue_label(p: object) -> str:
    if pd.isna(p):
        return "p=NA"
    p = float(p)
    if p < 1e-4:
        return f"p={p:.1e}"
    if p < 0.001:
        return f"p={p:.3f}"
    return f"p={p:.3g}"


def fisher_binary_stats(
    df: pd.DataFrame,
    feature_col: str,
    *,
    group_col: str = "group",
    pairs: Sequence[Tuple[str, str]] = PAIRWISE_GROUPS,
) -> pd.DataFrame:
    rows = []
    work = df[[group_col, feature_col]].copy()
    work[feature_col] = to_bool(work[feature_col])
    for group_a, group_b in pairs:
        a = work.loc[work[group_col].eq(group_a), feature_col].dropna()
        b = work.loc[work[group_col].eq(group_b), feature_col].dropna()
        a_true = int(a.eq(True).sum())
        a_false = int(a.eq(False).sum())
        b_true = int(b.eq(True).sum())
        b_false = int(b.eq(False).sum())
        if len(a) == 0 or len(b) == 0:
            odds_ratio = np.nan
            pvalue = np.nan
        else:
            table = [[a_true, a_false], [b_true, b_false]]
            odds_ratio, pvalue = stats.fisher_exact(table)
        log2_or = math.log2(((a_true + 0.5) * (b_false + 0.5)) / ((a_false + 0.5) * (b_true + 0.5)))
        rows.append(
            {
                "feature": feature_col,
                "group1": group_a,
                "group2": group_b,
                "group1_label": display_group(group_a),
                "group2_label": display_group(group_b),
                "group1_true": a_true,
                "group1_false": a_false,
                "group2_true": b_true,
                "group2_false": b_false,
                "odds_ratio": odds_ratio,
                "log2_odds_ratio": log2_or,
                "pvalue": pvalue,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["padj_bh"] = bh_adjust(out["pvalue"])
    return out


def mannwhitney_stats(
    df: pd.DataFrame,
    feature_col: str,
    *,
    group_col: str = "group",
    pairs: Sequence[Tuple[str, str]] = PAIRWISE_GROUPS,
) -> pd.DataFrame:
    rows = []
    work = df[[group_col, feature_col]].copy()
    work[feature_col] = pd.to_numeric(work[feature_col], errors="coerce")
    for group_a, group_b in pairs:
        a = work.loc[work[group_col].eq(group_a), feature_col].dropna().astype(float)
        b = work.loc[work[group_col].eq(group_b), feature_col].dropna().astype(float)
        if len(a) == 0 or len(b) == 0:
            u_stat = np.nan
            pvalue = np.nan
            rank_biserial = np.nan
        else:
            result = stats.mannwhitneyu(a, b, alternative="two-sided")
            u_stat = float(result.statistic)
            pvalue = float(result.pvalue)
            rank_biserial = (2.0 * u_stat / (len(a) * len(b))) - 1.0
        rows.append(
            {
                "feature": feature_col,
                "group1": group_a,
                "group2": group_b,
                "group1_label": display_group(group_a),
                "group2_label": display_group(group_b),
                "group1_n": int(len(a)),
                "group2_n": int(len(b)),
                "group1_median": float(a.median()) if len(a) else np.nan,
                "group2_median": float(b.median()) if len(b) else np.nan,
                "rank_biserial": rank_biserial,
                "mannwhitney_u": u_stat,
                "pvalue": pvalue,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["padj_bh"] = bh_adjust(out["pvalue"])
    return out


def add_pvalue_brackets(
    ax,
    order: Sequence[str],
    stats_df: pd.DataFrame,
    *,
    y_col: Optional[str] = None,
    p_col: str = "padj_bh",
    group1_col: str = "group1",
    group2_col: str = "group2",
) -> None:
    if stats_df.empty:
        return
    ymin, ymax = ax.get_ylim()
    if y_col is not None:
        ymax = max(ymax, pd.to_numeric(stats_df.get(y_col, pd.Series(dtype=float)), errors="coerce").max(skipna=True))
    if not np.isfinite(ymax):
        ymax = 1.0
    span = ymax - ymin
    if span <= 0:
        span = max(abs(ymax), 1.0)
    step = span * 0.08
    current_y = ymax + step
    x_lookup = {group: idx for idx, group in enumerate(order)}
    for _, row in stats_df.iterrows():
        group_a = row.get(group1_col)
        group_b = row.get(group2_col)
        if group_a not in x_lookup or group_b not in x_lookup:
            continue
        x1 = x_lookup[group_a]
        x2 = x_lookup[group_b]
        if x1 > x2:
            x1, x2 = x2, x1
        ax.plot([x1, x1, x2, x2], [current_y, current_y + step * 0.25, current_y + step * 0.25, current_y], color="black", lw=0.8)
        ax.text((x1 + x2) / 2, current_y + step * 0.28, pvalue_label(row.get(p_col)), ha="center", va="bottom", fontsize=8)
        current_y += step
    ax.set_ylim(ymin, current_y + step)


def boxplot_with_points(
    ax,
    df: pd.DataFrame,
    value_col: str,
    *,
    y_label: str,
    title: str,
    group_col: str = "group",
    pairs: Sequence[Tuple[str, str]] = PAIRWISE_GROUPS,
) -> pd.DataFrame:
    work = df[[group_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[group_col, value_col])
    order = group_order_present(work, group_col=group_col)
    sns.boxplot(
        data=work,
        x=group_col,
        y=value_col,
        order=order,
        ax=ax,
        showfliers=False,
        color="#d6e4f0",
        linewidth=1.2,
    )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([display_group(label) for label in order], rotation=25, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    stats_df = mannwhitney_stats(work, value_col, group_col=group_col, pairs=pairs)
    add_pvalue_brackets(ax, order, stats_df)
    return stats_df


def fraction_barplot(
    ax,
    df: pd.DataFrame,
    bool_col: str,
    *,
    y_label: str,
    title: str,
    group_col: str = "group",
    pairs: Optional[Sequence[Tuple[str, str]]] = None,
    annotate: bool = True,
) -> pd.DataFrame:
    work = df[[group_col, bool_col]].copy()
    work[bool_col] = to_bool(work[bool_col])
    work = work.dropna(subset=[group_col])
    rows = []
    for group, sub in work.groupby(group_col, dropna=False):
        valid = sub[bool_col].dropna()
        rows.append(
            {
                group_col: group,
                "n": int(len(valid)),
                "fraction": float(valid.eq(True).mean()) if len(valid) else np.nan,
            }
        )
    plot_df = pd.DataFrame(rows)
    order = group_order_present(plot_df, group_col=group_col)
    sns.barplot(data=plot_df, x=group_col, y="fraction", order=order, ax=ax, color="#7ca6c0")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([display_group(label) for label in order], rotation=25, ha="right")
    ax.set_ylim(0, min(1.0, max(1.0, plot_df["fraction"].max(skipna=True) * 1.15 if not plot_df.empty else 1.0)))
    ax.set_xlabel("")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if annotate and len(order) >= 2:
        annotation_pairs = pairs
        if annotation_pairs is None:
            annotation_pairs = [
                (order[i], order[j])
                for i in range(len(order))
                for j in range(i + 1, len(order))
            ]
        stats_df = fisher_binary_stats(
            work,
            bool_col,
            group_col=group_col,
            pairs=annotation_pairs,
        )
        add_pvalue_brackets(ax, order, stats_df)
    return plot_df


def save_figure(fig, pdf_name: str) -> None:
    ensure_fig_dirs()
    pdf_path = PDF_DIR / pdf_name
    png_path = PNG_DIR / re.sub(r"\.pdf$", ".png", pdf_name)
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


def reverse_complement(seq: Optional[str]) -> Optional[str]:
    if seq is None:
        return None
    return seq.translate(REVCOMP_TABLE)[::-1]


def resolve_chrom_key(fasta: Dict[str, object], chrom: str) -> Optional[str]:
    if chrom in fasta:
        return chrom
    alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
    if alt in fasta:
        return alt
    return None


def safe_subseq(chrom_seq: str, start: int, end: int) -> Optional[str]:
    if start < 0 or end < start or end > len(chrom_seq):
        return None
    return chrom_seq[start:end]


def oriented_start_window(
    chrom_seq: str,
    start0: int,
    end0: int,
    strand: str,
    upstream: int,
    downstream: int,
) -> Optional[str]:
    if strand == "+":
        return safe_subseq(chrom_seq, start0 - upstream, start0 + 3 + downstream)
    if strand == "-":
        seq = safe_subseq(chrom_seq, end0 - 3 - downstream, end0 + upstream)
        return reverse_complement(seq)
    return None


def gc_content(seq: Optional[str]) -> float:
    if not seq:
        return np.nan
    upper = seq.upper()
    valid = [base for base in upper if base in {"A", "C", "G", "T", "U"}]
    if not valid:
        return np.nan
    gc = sum(1 for base in valid if base in {"G", "C"})
    return gc / len(valid)


def parse_detected_samples(value: object) -> List[str]:
    text = clean_text(value)
    if not text:
        return []
    return [part for part in re.split(r"[,;|]+", text) if part]


def run_rnafold(seq: str) -> float:
    rna = seq.upper().replace("T", "U")
    proc = subprocess.run(
        ["RNAfold", "--noPS"],
        input=rna + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return np.nan
    matches = re.findall(r"\((\s*-?\d+(?:\.\d+)?)\)", proc.stdout)
    if matches:
        return float(matches[-1])
    matches = re.findall(r"(-?\d+(?:\.\d+)?)\s*$", proc.stdout.strip())
    return float(matches[-1]) if matches else np.nan


def command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def derive_position_flags(orf_type: object, primary_category: object = "") -> Dict[str, bool]:
    text = clean_text(orf_type)
    lower = text.lower()
    category = clean_text(primary_category).lower()
    return {
        "is_5UTR": ("5'utr" in lower) or ("five_prime" in category),
        "is_CDSFrameOverlap": "cdsframeoverlap" in lower,
        "is_3UTR": ("3'utr" in lower) or ("three_prime" in category),
        "is_lncRNA_or_ncRNA": any(token in lower for token in ["lncrna", "ncrna", "varrna-orf"]) or "lncrna" in category or "ncrna" in category,
        "is_internal": any(token in lower for token in ["internal", "iorf", "intorf", "out-of-frame", "out_of_frame"]) or "internal" in category,
        "is_uORF": any(token in lower for token in ["uorf", "uoorf", "overlap_uorf"]) or "uorf" in category,
        "is_dORF": any(token in lower for token in ["dorf", "doorf", "overlap_dorf"]) or "dorf" in category,
        "is_novel": "novel" in lower,
        "is_truncated": "truncated" in lower,
        "is_extended": "extended" in lower,
    }


def primary_position_label(row: pd.Series) -> str:
    category = clean_text(row.get("primary_noncanonical_category", ""))
    if category and category != "canonical_ORF":
        return category
    flags = derive_position_flags(row.get("ORF_type", ""), category)
    for key, label in [
        ("is_lncRNA_or_ncRNA", "lncRNA_or_ncRNA_ORF"),
        ("is_5UTR", "five_prime_uORF"),
        ("is_3UTR", "three_prime_dORF"),
        ("is_internal", "out_of_frame_or_internal_ORF"),
        ("is_uORF", "five_prime_uORF"),
        ("is_dORF", "three_prime_dORF"),
        ("is_novel", "novel_ORF"),
    ]:
        if flags.get(key):
            return label
    return "canonical_ORF" if row.get("group") == "group1_canonical_translated_ORF" else "other_noncanonical_ORF"
