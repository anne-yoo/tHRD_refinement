#!/usr/bin/env python3
"""Shared helpers for final four-group ORF pilot comparisons."""

from __future__ import annotations

import itertools
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/home/jiye/jiye/darkproteome/ORFstudy/pilot/pancreas8samples"))
NEG_DIR = Path(
    os.environ.get(
        "NEG_DIR",
        "/home/jiye/jiye/darkproteome/ORFstudy/pilot/pancreas8samples/cpat_negative_orfs",
    )
)
FIG_DIR = Path(
    os.environ.get("FIG_DIR", "/home/jiye/jiye/darkproteome/ORFstudy/pilot/figures/finalcomparison")
)
PILOT_DIR = Path(os.environ.get("PILOT_DIR", "/home/jiye/jiye/darkproteome/ORFstudy/pilot"))
GENOME_FA = Path(os.environ.get("GENOME_FA", "/home/jiye/jiye/darkproteome/ORFstudy/pilot/hg38.fa"))
RNAFOLD = Path(
    os.environ.get("RNAFOLD", "/home/jiye/jiye/darkproteome/tools/ViennaRNA-2.7.2/src/bin/RNAfold")
)
RNAPLFOLD = Path(
    os.environ.get("RNAPLFOLD", "/home/jiye/jiye/darkproteome/tools/ViennaRNA-2.7.2/src/bin/RNAplfold")
)

TABLE_DIR = FIG_DIR / "tables"
PDF_DIR = FIG_DIR / "pdf"
PNG_DIR = FIG_DIR / "png"
LOG_DIR = FIG_DIR / "logs"
SCRIPT_DIR = FIG_DIR / "scripts"

PLOT_GROUP_ORDER = [
    "Canonical",
    "AUG noncanonical",
    "nonAUG noncanonical",
    "CPAT-negative noncoding",
]
GROUP_PAIRS = list(itertools.combinations(PLOT_GROUP_ORDER, 2))
POSITIVE_GROUP_MAP = {
    "group1_canonical_translated_ORF": "Canonical",
    "group2_translated_AUG_cryptic_ORF": "AUG noncanonical",
    "group3_translated_nonAUG_cryptic_ORF": "nonAUG noncanonical",
}
NEGATIVE_GROUP = "CPAT-negative noncoding"
GROUP_COLORS = {
    "Canonical": "#4C78A8",
    "AUG noncanonical": "#F58518",
    "nonAUG noncanonical": "#54A24B",
    "CPAT-negative noncoding": "#B279A2",
}

UNSTRANDED_SAMPLES = [
    "GSM3395010",
    "GSM3395011",
    "GSM3395012",
    "GSM3395013",
    "GSM3395014",
    "GSM3395015",
]
FORWARD_SENSE_SAMPLES = ["GSM5099832", "GSM5099835"]
PILOT_SAMPLES = UNSTRANDED_SAMPLES + FORWARD_SENSE_SAMPLES

REVCOMP_TABLE = str.maketrans("ACGTUNacgtun", "TGCAANtgcaan")
POSITIONS = np.arange(-100, 101, dtype=int)
POS_COLS = [f"pos_{pos}" for pos in POSITIONS]
START_POSITION_TO_INDEX = {pos: idx for idx, pos in enumerate(list(range(-100, 0)) + list(range(1, 102)))}

NEAR_COGNATE_STARTS = {"CTG", "GTG", "TTG", "ACG", "ATA", "ATT", "ATC", "AAG", "AGG"}


def ensure_dirs() -> None:
    for path in [TABLE_DIR, PDF_DIR, PNG_DIR, LOG_DIR, SCRIPT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def archive_script(script_path: str | Path) -> None:
    ensure_dirs()
    src = Path(script_path)
    if src.exists():
        shutil.copy2(src, SCRIPT_DIR / src.name)
    helper = Path(__file__)
    if helper.exists() and helper.name != src.name:
        shutil.copy2(helper, SCRIPT_DIR / helper.name)


def print_paths(script_name: str, inputs: Sequence[Path], outputs: Sequence[Path]) -> None:
    print(script_name)
    print(f"INPUT_DIR={INPUT_DIR}")
    print(f"NEG_DIR={NEG_DIR}")
    print(f"FIG_DIR={FIG_DIR}")
    for path in inputs:
        print(f"input={path}")
    for path in outputs:
        print(f"output={path}")


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


def start_codon_type(value: object) -> str:
    codon = normalize_codon(value)
    if codon == "ATG":
        return "AUG"
    if codon in NEAR_COGNATE_STARTS:
        return "near-cognate"
    if codon:
        return "other"
    return "missing"


def read_tsv(path: Path, required: Optional[Sequence[str]] = None, allow_missing: bool = False) -> pd.DataFrame:
    if not path.exists():
        if allow_missing:
            return pd.DataFrame()
        raise SystemExit(f"Input file not found: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str)
    if required:
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise SystemExit(f"Missing required column(s) in {path}: {', '.join(missing)}")
    return df


def to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def to_bool_series(series: pd.Series) -> pd.Series:
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


def plot_group_from_group(value: object) -> str:
    text = clean_text(value)
    if text in POSITIVE_GROUP_MAP:
        return POSITIVE_GROUP_MAP[text]
    if text == "group4_CPAT_negative_noncoding_ORF":
        return NEGATIVE_GROUP
    if text in PLOT_GROUP_ORDER:
        return text
    return text


def order_present(df: pd.DataFrame, group_col: str = "plot_group") -> List[str]:
    present = set(df[group_col].dropna().astype(str))
    order = [group for group in PLOT_GROUP_ORDER if group in present]
    order.extend(sorted(present - set(order)))
    return order


def pairs_present(order: Sequence[str]) -> List[Tuple[str, str]]:
    return [(a, b) for a, b in GROUP_PAIRS if a in order and b in order]


def bh_adjust(pvalues: Iterable[object]) -> List[float]:
    values = [np.nan if pd.isna(p) else float(p) for p in pvalues]
    valid = [(idx, p) for idx, p in enumerate(values) if np.isfinite(p)]
    if not valid:
        return [np.nan] * len(values)
    ranked = sorted(valid, key=lambda item: item[1])
    m = len(ranked)
    adjusted = [np.nan] * len(values)
    running = 1.0
    for reverse_rank, (idx, p) in enumerate(reversed(ranked), start=1):
        rank = m - reverse_rank + 1
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


def mannwhitney_stats(
    df: pd.DataFrame,
    feature: str,
    *,
    group_col: str = "plot_group",
    pairs: Optional[Sequence[Tuple[str, str]]] = None,
) -> pd.DataFrame:
    order = order_present(df, group_col)
    pairs = list(pairs or pairs_present(order))
    rows = []
    work = df[[group_col, feature]].copy()
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    for a_group, b_group in pairs:
        a = work.loc[work[group_col].eq(a_group), feature].dropna().astype(float)
        b = work.loc[work[group_col].eq(b_group), feature].dropna().astype(float)
        if len(a) and len(b):
            result = stats.mannwhitneyu(a, b, alternative="two-sided")
            u_stat = float(result.statistic)
            pvalue = float(result.pvalue)
            rank_biserial = (2.0 * u_stat / (len(a) * len(b))) - 1.0
        else:
            u_stat = np.nan
            pvalue = np.nan
            rank_biserial = np.nan
        rows.append(
            {
                "feature": feature,
                "test": "mannwhitney_u",
                "group1": a_group,
                "group2": b_group,
                "group1_n": int(len(a)),
                "group2_n": int(len(b)),
                "group1_median": float(a.median()) if len(a) else np.nan,
                "group2_median": float(b.median()) if len(b) else np.nan,
                "effect_size": rank_biserial,
                "effect_size_type": "rank_biserial",
                "statistic": u_stat,
                "pvalue": pvalue,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["padj_bh"] = bh_adjust(out["pvalue"])
    return out


def fisher_stats(
    df: pd.DataFrame,
    feature: str,
    *,
    group_col: str = "plot_group",
    pairs: Optional[Sequence[Tuple[str, str]]] = None,
) -> pd.DataFrame:
    order = order_present(df, group_col)
    pairs = list(pairs or pairs_present(order))
    rows = []
    work = df[[group_col, feature]].copy()
    work[feature] = to_bool_series(work[feature])
    for a_group, b_group in pairs:
        a = work.loc[work[group_col].eq(a_group), feature].dropna()
        b = work.loc[work[group_col].eq(b_group), feature].dropna()
        a_true = int(a.eq(True).sum())
        a_false = int(a.eq(False).sum())
        b_true = int(b.eq(True).sum())
        b_false = int(b.eq(False).sum())
        if len(a) and len(b):
            odds_ratio, pvalue = stats.fisher_exact([[a_true, a_false], [b_true, b_false]])
        else:
            odds_ratio = np.nan
            pvalue = np.nan
        log2_or = math.log2(((a_true + 0.5) * (b_false + 0.5)) / ((a_false + 0.5) * (b_true + 0.5)))
        rows.append(
            {
                "feature": feature,
                "test": "fisher_exact",
                "group1": a_group,
                "group2": b_group,
                "group1_n": int(len(a)),
                "group2_n": int(len(b)),
                "group1_true": a_true,
                "group1_false": a_false,
                "group2_true": b_true,
                "group2_false": b_false,
                "effect_size": log2_or,
                "effect_size_type": "log2_odds_ratio",
                "statistic": odds_ratio,
                "pvalue": pvalue,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["padj_bh"] = bh_adjust(out["pvalue"])
    return out


def add_manual_pvalue_brackets(
    ax,
    order: Sequence[str],
    stats_df: pd.DataFrame,
    *,
    p_col: str = "pvalue",
) -> None:
    if stats_df.empty:
        return
    ymin, original_ymax = ax.get_ylim()
    if not np.isfinite(original_ymax):
        original_ymax = 1.0
    span = original_ymax - ymin
    if span <= 0:
        span = max(abs(original_ymax), 1.0)
    n_brackets = len(stats_df)
    step = span * 0.075
    top_padding = step * (n_brackets + 2.8)
    ax.set_ylim(ymin, original_ymax + top_padding)
    y = original_ymax + step * 0.7
    lookup = {group: idx for idx, group in enumerate(order)}
    for _, row in stats_df.iterrows():
        group1 = row.get("group1")
        group2 = row.get("group2")
        if group1 not in lookup or group2 not in lookup:
            continue
        x1, x2 = lookup[group1], lookup[group2]
        if x1 > x2:
            x1, x2 = x2, x1
        ax.plot([x1, x1, x2, x2], [y, y + step * 0.25, y + step * 0.25, y], color="black", lw=0.8)
        ax.text(
            (x1 + x2) / 2,
            y + step * 0.28,
            pvalue_label(row.get(p_col)),
            ha="center",
            va="bottom",
            fontsize=8,
            clip_on=True,
        )
        y += step


def annotate_with_statannotations_or_manual(
    ax,
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    order: Sequence[str],
    stats_df: pd.DataFrame,
    p_col: str = "pvalue",
) -> None:
    pairs = [(row["group1"], row["group2"]) for _, row in stats_df.iterrows()]
    pvalues = [row[p_col] for _, row in stats_df.iterrows()]
    if not pairs:
        return
    try:
        from statannotations.Annotator import Annotator

        annotator = Annotator(ax, pairs, data=data, x=x, y=y, order=order)
        annotator.configure(test=None, text_format="simple", loc="inside", verbose=0)
        annotator.set_pvalues_and_annotate(pvalues)
    except Exception:
        add_manual_pvalue_brackets(ax, order, stats_df, p_col=p_col)


def boxplot_four_group(
    ax,
    df: pd.DataFrame,
    feature: str,
    *,
    y_label: str,
    title: str,
    group_col: str = "plot_group",
    annotate: bool = True,
) -> pd.DataFrame:
    work = df[[group_col, feature]].copy()
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work = work.dropna(subset=[group_col, feature])
    if work.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{title}\nnot available", ha="center", va="center")
        return pd.DataFrame()
    order = order_present(work, group_col)
    sns.boxplot(
        data=work,
        x=group_col,
        y=feature,
        hue=group_col,
        order=order,
        hue_order=order,
        ax=ax,
        showfliers=False,
        palette=GROUP_COLORS,
        linewidth=1.1,
        legend=False,
    )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel(y_label)
    ax.set_title(title, pad=18)
    stats_df = mannwhitney_stats(work, feature, group_col=group_col, pairs=pairs_present(order))
    if annotate and len(order) >= 2 and not stats_df.empty:
        annotate_with_statannotations_or_manual(ax, work, x=group_col, y=feature, order=order, stats_df=stats_df)
    return stats_df


def fraction_barplot_four_group(
    ax,
    df: pd.DataFrame,
    feature: str,
    *,
    title: str,
    y_label: str = "Fraction",
    group_col: str = "plot_group",
    annotate: bool = True,
) -> pd.DataFrame:
    work = df[[group_col, feature]].copy()
    work[feature] = to_bool_series(work[feature])
    rows = []
    for group, sub in work.groupby(group_col, dropna=False):
        valid = sub[feature].dropna()
        rows.append({"plot_group": group, "n": int(len(valid)), "fraction": float(valid.eq(True).mean()) if len(valid) else np.nan})
    plot_df = pd.DataFrame(rows).dropna(subset=["plot_group", "fraction"])
    if plot_df.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{title}\nnot available", ha="center", va="center")
        return pd.DataFrame()
    order = order_present(plot_df, "plot_group")
    sns.barplot(
        data=plot_df,
        x="plot_group",
        y="fraction",
        hue="plot_group",
        order=order,
        hue_order=order,
        palette=GROUP_COLORS,
        legend=False,
        ax=ax,
    )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel(y_label)
    ax.set_title(title, pad=18)
    stats_df = fisher_stats(work.rename(columns={group_col: "plot_group"}), feature, group_col="plot_group", pairs=pairs_present(order))
    if annotate and len(order) >= 2 and not stats_df.empty:
        annotate_with_statannotations_or_manual(
            ax,
            plot_df,
            x="plot_group",
            y="fraction",
            order=order,
            stats_df=stats_df,
        )
    return stats_df


def save_figure(fig, pdf_name: str) -> None:
    ensure_dirs()
    pdf_path = PDF_DIR / pdf_name
    png_path = PNG_DIR / re.sub(r"\.pdf$", ".png", pdf_name)
    fig.tight_layout(pad=2.6, h_pad=4.0, w_pad=2.2)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


def read_master() -> pd.DataFrame:
    path = TABLE_DIR / "four_group_orf_metadata.tsv"
    return read_tsv(path, required=["plot_group", "ORF_id", "transcript_id", "chr", "start0", "end0", "strand"])


def reverse_complement(seq: Optional[str]) -> Optional[str]:
    if seq is None:
        return None
    return seq.translate(REVCOMP_TABLE)[::-1]


def safe_subseq(chrom_seq: str, start: int, end: int) -> Optional[str]:
    if start < 0 or end < start or end > len(chrom_seq):
        return None
    return chrom_seq[start:end]


def resolve_chrom_key(fasta: Dict[str, object], chrom: str) -> Optional[str]:
    if chrom in fasta:
        return chrom
    alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
    if alt in fasta:
        return alt
    return None


def start_centered_window(row: pd.Series, flank: int = 100) -> Tuple[str, int, int, bool]:
    chrom = str(row["chr"])
    start0 = int(float(row["start0"]))
    end0 = int(float(row["end0"]))
    strand = str(row["strand"])
    if strand == "+":
        return chrom, start0 - flank, start0 + flank + 1, False
    if strand == "-":
        return chrom, end0 - flank - 1, end0 + flank, True
    raise ValueError(f"Unsupported strand: {strand}")


def import_pybigwig():
    try:
        import pyBigWig
    except ImportError as exc:
        raise SystemExit("pyBigWig is required for coverage extraction. Activate the environment with pyBigWig.") from exc
    return pyBigWig


def discover_bigwigs(input_dir: Path = INPUT_DIR) -> Dict[str, Path]:
    bw_dir = input_dir / "bigwig"
    paths: Dict[str, Path] = {}
    for sample in PILOT_SAMPLES:
        candidates = [
            bw_dir / f"{sample}.unstranded.CPM.bw",
            bw_dir / f"{sample}.sense.CPM.bw",
            bw_dir / f"{sample}.CPM.bw",
        ]
        for candidate in candidates:
            if candidate.exists():
                paths[sample] = candidate
                break
    for path in sorted(bw_dir.glob("*.CPM.bw")):
        sample = path.name.split(".")[0]
        if "antisense" in path.name:
            continue
        paths.setdefault(sample, path)
    if not paths:
        raise SystemExit(f"No CPM BigWigs found under {bw_dir}")
    return paths


def resolve_bw_chrom(chroms: Dict[str, int], chrom: str) -> Optional[str]:
    if chrom in chroms:
        return chrom
    alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
    return alt if alt in chroms else None


def bw_values(bw, chrom: str, start: int, end: int) -> np.ndarray:
    length = max(0, end - start)
    out = np.full(length, np.nan, dtype=float)
    if length == 0:
        return out
    chroms = bw.chroms()
    resolved = resolve_bw_chrom(chroms, chrom)
    if resolved is None:
        return out
    chrom_len = int(chroms[resolved])
    clipped_start = max(0, start)
    clipped_end = min(end, chrom_len)
    if clipped_end <= clipped_start:
        return out
    values = np.asarray(bw.values(resolved, clipped_start, clipped_end, numpy=True), dtype=float)
    offset = clipped_start - start
    out[offset : offset + len(values)] = values
    return out


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan


def finite_sum(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nansum(values)) if np.isfinite(values).any() else np.nan


def finite_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    return float(stats.linregress(x[mask], y[mask]).slope)


def coverage_features_from_vector(vector: np.ndarray) -> Dict[str, float]:
    vector = np.asarray(vector, dtype=float)
    upstream_mask = (POSITIONS >= -100) & (POSITIONS < 0)
    downstream_mask = (POSITIONS > 0) & (POSITIONS <= 100)
    start_peak_mask = (POSITIONS >= -5) & (POSITIONS <= 5)
    upstream_slope_mask = (POSITIONS >= -100) & (POSITIONS <= 0)
    downstream_slope_mask = (POSITIONS >= 0) & (POSITIONS <= 100)
    mean_window = finite_mean(vector)
    start_peak = finite_mean(vector[start_peak_mask])
    auc_upstream = finite_sum(vector[upstream_slope_mask])
    auc_downstream = finite_sum(vector[downstream_slope_mask])
    return {
        "mean_window_coverage": mean_window,
        "mean_upstream_100": finite_mean(vector[upstream_mask]),
        "mean_downstream_100": finite_mean(vector[downstream_mask]),
        "start_peak_coverage": start_peak,
        "start_peak_ratio": start_peak / mean_window if mean_window and mean_window > 0 else np.nan,
        "upstream_slope": finite_slope(POSITIONS[upstream_slope_mask], vector[upstream_slope_mask]),
        "downstream_slope": finite_slope(POSITIONS[downstream_slope_mask], vector[downstream_slope_mask]),
        "AUC_upstream": auc_upstream,
        "AUC_downstream": auc_downstream,
        "asymmetry_ratio": auc_downstream / auc_upstream if auc_upstream and auc_upstream > 0 else np.nan,
    }


def vector_mean_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    mean = finite_mean(vector)
    if not np.isfinite(mean) or mean <= 0:
        return np.full_like(vector, np.nan, dtype=float)
    return vector / mean


def resolve_executable(path: Path, command_name: str) -> Optional[str]:
    if path.is_file() and os.access(path, os.X_OK):
        return str(path)
    return shutil.which(command_name)


def clean_rna_sequence(seq: str) -> str:
    return re.sub(r"[^ACGUN]", "N", seq.upper().replace("T", "U"))


def run_rnafold(seq: Optional[str], rnafold_bin: Optional[str]) -> float:
    if not seq or rnafold_bin is None:
        return np.nan
    proc = subprocess.run(
        [rnafold_bin, "--noPS"],
        input=clean_rna_sequence(seq) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        return np.nan
    matches = re.findall(r"\(\s*(-?\d+(?:\.\d+)?)\s*\)", proc.stdout)
    return float(matches[-1]) if matches else np.nan


def parse_lunp_file(path: Path) -> Dict[int, Dict[int, float]]:
    probabilities: Dict[int, Dict[int, float]] = {}
    header_lengths: Optional[List[int]] = None
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                nums = [int(x) for x in re.findall(r"\d+", line)]
                if nums:
                    header_lengths = nums
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                pos0 = int(float(parts[0])) - 1
            except ValueError:
                continue
            values = []
            for token in parts[1:]:
                try:
                    values.append(float(token))
                except ValueError:
                    values.append(np.nan)
            lengths = header_lengths[-len(values) :] if header_lengths and len(header_lengths) >= len(values) else list(range(1, len(values) + 1))
            probabilities[pos0] = {u: val for u, val in zip(lengths, values) if np.isfinite(val)}
    return probabilities


def run_rnaplfold(seq: Optional[str], rnaplfold_bin: Optional[str], *, window: int = 80, span: int = 40, max_unpaired: int = 20) -> Optional[Dict[int, Dict[int, float]]]:
    if not seq or rnaplfold_bin is None:
        return None
    with TemporaryDirectory(prefix="finalcomparison_rnaplfold_") as tmp:
        proc = subprocess.run(
            [rnaplfold_bin, "-W", str(window), "-L", str(span), "-u", str(max_unpaired)],
            input=f">orf\n{clean_rna_sequence(seq)}\n",
            cwd=tmp,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            return None
        files = sorted(Path(tmp).glob("*lunp*"))
        if not files:
            return None
        return parse_lunp_file(files[0])


def position_indices(start: int, end: int) -> List[int]:
    if start <= -1 and end <= -1:
        positions = range(start, end + 1)
    elif start >= 1 and end >= 1:
        positions = range(start, end + 1)
    else:
        positions = list(range(start, 0)) + list(range(1, end + 1))
    return [START_POSITION_TO_INDEX[pos] for pos in positions if pos in START_POSITION_TO_INDEX]


def mean_unpaired(probabilities: Optional[Dict[int, Dict[int, float]]], start: int, end: int, u: int = 1) -> float:
    if probabilities is None:
        return np.nan
    vals = []
    for idx in position_indices(start, end):
        val = probabilities.get(idx, {}).get(u, np.nan)
        if np.isfinite(val):
            vals.append(val)
    return float(np.mean(vals)) if vals else np.nan
