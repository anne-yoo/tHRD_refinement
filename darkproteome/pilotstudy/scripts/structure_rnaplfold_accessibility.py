#!/usr/bin/env python3
"""RNAplfold local accessibility analysis around translated ORF starts."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO
from scipy import stats


DEFAULT_INPUT_DIR = Path("/home/jiye/jiye/darkproteome/ORFstudy/pilot/pancreas8samples")
DEFAULT_FIG_DIR = Path("/home/jiye/jiye/darkproteome/ORFstudy/pilot/figures")
DEFAULT_GENOME_FA = Path("/home/jiye/jiye/darkproteome/ORFstudy/pilot/hg38.fa")
DEFAULT_RNAPLFOLD = Path("/home/jiye/jiye/darkproteome/tools/ViennaRNA-2.7.2/src/bin/RNAplfold")

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
GROUP_PAIRS = [
    ("group1_canonical_translated_ORF", "group2_translated_AUG_cryptic_ORF"),
    ("group1_canonical_translated_ORF", "group3_translated_nonAUG_cryptic_ORF"),
    ("group2_translated_AUG_cryptic_ORF", "group3_translated_nonAUG_cryptic_ORF"),
]

REVCOMP_TABLE = str.maketrans("ACGTUNacgtun", "TGCAANtgcaan")
POSITION_TO_INDEX = {pos: idx for idx, pos in enumerate(list(range(-100, 0)) + list(range(1, 102)))}

ACCESSIBILITY_COLUMNS = [
    "accessibility_minus3_plus4",
    "accessibility_start_codon",
    "accessibility_kozak_core",
    "accessibility_start_pm10",
    "accessibility_start_pm20",
    "accessibility_upstream_50",
    "accessibility_downstream_50",
    "accessibility_asymmetry",
    "min_accessibility_start_pm10",
    "accessibility_start_codon_u3",
    "accessibility_minus3_plus4_u7",
]

OUTPUT_COLUMNS = [
    "ORF_id",
    "group",
    "primary_noncanonical_category",
    "chr",
    "start0",
    "end0",
    "strand",
    "start_codon",
    "sequence_length",
    "rnaplfold_status",
    "accessibility_minus3_plus4",
    "accessibility_start_codon",
    "accessibility_kozak_core",
    "accessibility_start_pm10",
    "accessibility_start_pm20",
    "accessibility_upstream_50",
    "accessibility_downstream_50",
    "accessibility_asymmetry",
    "min_accessibility_start_pm10",
    "start_region_highly_accessible",
    "accessibility_start_codon_u3",
    "accessibility_minus3_plus4_u7",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract start_pm100 ORF sequences and compute RNAplfold local "
            "unpaired/accessibility features around translation starts."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(os.environ.get("INPUT_DIR", DEFAULT_INPUT_DIR)),
        help="Pilot input directory containing tables/orf_groups.combined_metadata.tsv.",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path(os.environ.get("FIG_DIR", DEFAULT_FIG_DIR)),
        help="Output figure directory.",
    )
    parser.add_argument(
        "--genome-fa",
        type=Path,
        default=Path(os.environ.get("GENOME_FA", DEFAULT_GENOME_FA)),
        help="Genome FASTA.",
    )
    parser.add_argument(
        "--rnaplfold",
        type=Path,
        default=Path(
            os.environ.get(
                "RNAPLFOLD",
                os.environ.get("RNAPLFOLD_BIN", os.environ.get("RNAplfold", DEFAULT_RNAPLFOLD)),
            )
        ),
        help="RNAplfold executable path.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Analyze only the first N ORFs for testing.")
    parser.add_argument("--threads", type=int, default=int(os.environ.get("THREADS", "1")), help="RNAplfold jobs to run in parallel.")
    parser.add_argument("--window", type=int, default=int(os.environ.get("RNAPLFOLD_W", "80")), help="RNAplfold -W value.")
    parser.add_argument("--span", type=int, default=int(os.environ.get("RNAPLFOLD_L", "40")), help="RNAplfold -L value.")
    parser.add_argument("--max-unpaired", type=int, default=int(os.environ.get("RNAPLFOLD_U", "20")), help="RNAplfold -u value.")
    parser.add_argument("--progress-every", type=int, default=1000, help="Print progress every N ORFs.")
    return parser.parse_args()


def ensure_output_dirs(fig_dir: Path) -> Dict[str, Path]:
    paths = {
        "tables": fig_dir / "tables",
        "pdf": fig_dir / "pdf",
        "png": fig_dir / "png",
        "logs": fig_dir / "logs",
        "scripts": fig_dir / "scripts",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "na", "n/a", "."} else text


def display_group(group: object) -> str:
    return GROUP_LABELS.get(str(group), str(group))


def group_order_present(df: pd.DataFrame, group_col: str = "group") -> List[str]:
    present = set(df[group_col].dropna().astype(str))
    order = [group for group in GROUP_ORDER if group in present]
    order.extend(sorted(present - set(order)))
    return order


def pvalue_label(pvalue: object) -> str:
    if pd.isna(pvalue):
        return "p=NA"
    pvalue = float(pvalue)
    if pvalue < 1e-4:
        return "p={:.1e}".format(pvalue)
    if pvalue < 0.001:
        return "p={:.3f}".format(pvalue)
    return "p={:.3g}".format(pvalue)


def bh_adjust(pvalues: Sequence[object]) -> List[float]:
    values = [np.nan if pd.isna(pvalue) else float(pvalue) for pvalue in pvalues]
    valid = [(idx, pvalue) for idx, pvalue in enumerate(values) if np.isfinite(pvalue)]
    if not valid:
        return [np.nan] * len(values)

    adjusted = [np.nan] * len(values)
    ranked = sorted(valid, key=lambda item: item[1])
    m = len(ranked)
    running = 1.0
    for reverse_rank, (idx, pvalue) in enumerate(reversed(ranked), start=1):
        rank = m - reverse_rank + 1
        running = min(running, pvalue * m / rank)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def read_orf_table(input_dir: Path, limit: Optional[int]) -> pd.DataFrame:
    path = input_dir / "tables" / "orf_groups.combined_metadata.tsv"
    required = ["ORF_id", "group", "primary_noncanonical_category", "chr", "start0", "end0", "strand"]
    if not path.exists():
        raise SystemExit("Input table not found: {}".format(path))
    df = pd.read_csv(path, sep="\t", dtype=str)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SystemExit("Missing required column(s) in {}: {}".format(path, ", ".join(missing)))
    if "start_codon" not in df.columns:
        df["start_codon"] = ""
    df["start0"] = pd.to_numeric(df["start0"], errors="coerce")
    df["end0"] = pd.to_numeric(df["end0"], errors="coerce")
    if limit is not None:
        df = df.head(limit).copy()
    return df.reset_index(drop=True)


def resolve_executable(path: Path, command_name: str) -> Tuple[Optional[str], str]:
    if path.is_file() and os.access(path, os.X_OK):
        return str(path), "path"
    found = shutil.which(command_name)
    if found:
        return found, "PATH"
    return None, "missing"


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
    alt = chrom[3:] if chrom.startswith("chr") else "chr{}".format(chrom)
    if alt in fasta:
        return alt
    return None


def extract_start_pm100(chrom_seq: str, start0: int, end0: int, strand: str) -> Optional[str]:
    """Extract -100..-1 and +1..+101 around the start codon first base.

    This produces 201 nt when not clipped by chromosome boundaries. Position 0
    is intentionally absent; +1..+3 are the start codon bases.
    """
    if strand == "+":
        return safe_subseq(chrom_seq, start0 - 100, start0 + 101)
    if strand == "-":
        seq = safe_subseq(chrom_seq, end0 - 101, end0 + 100)
        return reverse_complement(seq)
    return None


def normalize_dna(seq: Optional[str]) -> Optional[str]:
    if seq is None:
        return None
    return re.sub(r"[^ACGTN]", "N", seq.upper().replace("U", "T"))


def normalize_rna(seq: str) -> str:
    return re.sub(r"[^ACGUN]", "N", seq.upper().replace("T", "U"))


def infer_start_codon(row_start_codon: object, seq: Optional[str]) -> str:
    text = clean_text(row_start_codon).upper().replace("U", "T")
    if text:
        return re.split(r"[\s,;|/]+", text)[0]
    if seq and len(seq) >= 103:
        return seq[POSITION_TO_INDEX[1] : POSITION_TO_INDEX[3] + 1].upper()
    return ""


def prepare_sequences(groups: pd.DataFrame, genome_fa: Path) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    if not genome_fa.exists():
        raise SystemExit("Genome FASTA not found: {}".format(genome_fa))

    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []
    fasta = SeqIO.index(str(genome_fa), "fasta")
    try:
        chrom_keys = {
            chrom: resolve_chrom_key(fasta, chrom)
            for chrom in groups["chr"].dropna().astype(str).unique()
        }
        for chrom, sub in groups.groupby("chr", sort=False, dropna=False):
            chrom_text = str(chrom)
            chrom_key = chrom_keys.get(chrom_text)
            if chrom_key is None:
                chrom_seq = None
                print("Chromosome not found in FASTA: {}".format(chrom_text))
            else:
                print("Loading {} from FASTA as {} for {} ORF(s)".format(chrom_text, chrom_key, len(sub)))
                chrom_seq = str(fasta[chrom_key].seq)

            for _, row in sub.iterrows():
                seq = None
                status = "sequence_ok"
                message = ""
                start0 = row["start0"]
                end0 = row["end0"]
                if chrom_seq is None:
                    status = "missing_chromosome"
                    message = "chromosome not found in FASTA"
                elif pd.isna(start0) or pd.isna(end0) or int(end0) <= int(start0):
                    status = "invalid_coordinates"
                    message = "start0/end0 invalid"
                else:
                    seq = normalize_dna(
                        extract_start_pm100(chrom_seq, int(start0), int(end0), clean_text(row["strand"]))
                    )
                    if seq is None:
                        status = "boundary_or_strand_failure"
                        message = "start_pm100 extends beyond chromosome boundary or strand is invalid"

                start_codon = infer_start_codon(row.get("start_codon", ""), seq)
                rows.append(
                    {
                        "ORF_id": clean_text(row["ORF_id"]),
                        "group": clean_text(row["group"]),
                        "primary_noncanonical_category": clean_text(row["primary_noncanonical_category"]),
                        "chr": clean_text(row["chr"]),
                        "start0": int(start0) if pd.notna(start0) else np.nan,
                        "end0": int(end0) if pd.notna(end0) else np.nan,
                        "strand": clean_text(row["strand"]),
                        "start_codon": start_codon,
                        "sequence": seq,
                        "sequence_length": len(seq) if seq else 0,
                        "rnaplfold_status": status,
                    }
                )
                if status != "sequence_ok":
                    failures.append(
                        {
                            "ORF_id": clean_text(row["ORF_id"]),
                            "status": status,
                            "message": message,
                        }
                    )
    finally:
        fasta.close()
    return pd.DataFrame(rows), failures


def parse_lunp_lengths_from_header(line: str, n_values: int) -> Optional[List[int]]:
    tokens = re.findall(r"\d+", line)
    if not tokens:
        return None
    lengths = [int(token) for token in tokens]
    if len(lengths) >= n_values:
        return lengths[-n_values:]
    return None


def parse_lunp_file(path: Path) -> Tuple[Dict[int, Dict[int, float]], List[int]]:
    """Parse RNAplfold lunp output.

    Returns a mapping from 0-based sequence index to {u_length: probability}.
    If the file lacks a usable header, subsequent columns are assumed to be
    u=1,2,... in order, which is the standard RNAplfold lunp layout.
    """
    probabilities: Dict[int, Dict[int, float]] = {}
    header_lengths: Optional[List[int]] = None
    inferred_lengths: List[int] = []

    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                pos0 = int(float(parts[0])) - 1
            except ValueError:
                continue

            values: List[float] = []
            for token in parts[1:]:
                try:
                    values.append(float(token))
                except ValueError:
                    values.append(np.nan)
            if header_lengths is None:
                # Re-read comments cheaply to pick up headers that list u lengths.
                with path.open() as header_handle:
                    for header_line in header_handle:
                        header_line = header_line.strip()
                        if not header_line.startswith("#"):
                            break
                        candidate = parse_lunp_lengths_from_header(header_line, len(values))
                        if candidate is not None:
                            header_lengths = candidate
                if header_lengths is None:
                    header_lengths = list(range(1, len(values) + 1))
                inferred_lengths = header_lengths

            lengths = header_lengths[: len(values)] if header_lengths else list(range(1, len(values) + 1))
            probabilities[pos0] = {
                length: value
                for length, value in zip(lengths, values)
                if np.isfinite(value)
            }

    return probabilities, inferred_lengths


def find_lunp_file(tmp_dir: Path) -> Optional[Path]:
    candidates = sorted(tmp_dir.glob("*lunp*"))
    if candidates:
        return candidates[0]
    return None


def run_rnaplfold_one(
    orf_id: str,
    seq: Optional[str],
    rnaplfold_bin: str,
    window: int,
    span: int,
    max_unpaired: int,
) -> Tuple[str, str, Optional[Dict[int, Dict[int, float]]], Optional[List[int]], str]:
    if not seq:
        return orf_id, "sequence_unavailable", None, None, "sequence missing"

    rna = normalize_rna(seq)
    command = [rnaplfold_bin, "-W", str(window), "-L", str(span), "-u", str(max_unpaired)]
    with TemporaryDirectory(prefix="rnaplfold_") as tmp:
        tmp_dir = Path(tmp)
        fasta_input = ">{}\n{}\n".format(re.sub(r"[^A-Za-z0-9_.-]", "_", orf_id), rna)
        proc = subprocess.run(
            command,
            input=fasta_input,
            cwd=str(tmp_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            message = proc.stderr.strip() or proc.stdout.strip()
            return orf_id, "rnaplfold_failed", None, None, message[:1000]

        lunp_path = find_lunp_file(tmp_dir)
        if lunp_path is None:
            return orf_id, "lunp_missing", None, None, "RNAplfold produced no lunp output"

        try:
            probabilities, lengths = parse_lunp_file(lunp_path)
        except Exception as exc:  # noqa: BLE001 - report parser failures per ORF.
            return orf_id, "lunp_parse_failed", None, None, str(exc)[:1000]
    return orf_id, "rnaplfold_ok", probabilities, lengths, ""


def position_indices(start: int, end: int) -> List[int]:
    if start <= -1 and end <= -1:
        positions = list(range(start, end + 1))
    elif start >= 1 and end >= 1:
        positions = list(range(start, end + 1))
    else:
        positions = list(range(start, 0)) + list(range(1, end + 1))
    return [POSITION_TO_INDEX[pos] for pos in positions if pos in POSITION_TO_INDEX]


def mean_probability(probabilities: Dict[int, Dict[int, float]], indices: Sequence[int], u_length: int = 1) -> float:
    values = [
        probabilities[idx][u_length]
        for idx in indices
        if idx in probabilities and u_length in probabilities[idx] and np.isfinite(probabilities[idx][u_length])
    ]
    return float(np.mean(values)) if values else np.nan


def min_probability(probabilities: Dict[int, Dict[int, float]], indices: Sequence[int], u_length: int = 1) -> float:
    values = [
        probabilities[idx][u_length]
        for idx in indices
        if idx in probabilities and u_length in probabilities[idx] and np.isfinite(probabilities[idx][u_length])
    ]
    return float(np.min(values)) if values else np.nan


def segment_probability(probabilities: Dict[int, Dict[int, float]], first_position: int, u_length: int) -> float:
    idx = POSITION_TO_INDEX.get(first_position)
    if idx is None or idx not in probabilities:
        return np.nan
    return float(probabilities[idx].get(u_length, np.nan))


def compute_accessibility_features(
    probabilities: Optional[Dict[int, Dict[int, float]]],
) -> Dict[str, object]:
    features: Dict[str, object] = {col: np.nan for col in ACCESSIBILITY_COLUMNS}
    features["start_region_highly_accessible"] = pd.NA
    if probabilities is None:
        return features

    minus3_plus4 = position_indices(-3, 4)
    start_codon = position_indices(1, 3)
    start_pm10 = position_indices(-10, 10)
    start_pm20 = position_indices(-20, 20)
    upstream_50 = position_indices(-50, -1)
    downstream_50 = position_indices(4, 50)

    features["accessibility_minus3_plus4"] = mean_probability(probabilities, minus3_plus4, 1)
    features["accessibility_start_codon"] = mean_probability(probabilities, start_codon, 1)
    features["accessibility_kozak_core"] = features["accessibility_minus3_plus4"]
    features["accessibility_start_pm10"] = mean_probability(probabilities, start_pm10, 1)
    features["accessibility_start_pm20"] = mean_probability(probabilities, start_pm20, 1)
    features["accessibility_upstream_50"] = mean_probability(probabilities, upstream_50, 1)
    features["accessibility_downstream_50"] = mean_probability(probabilities, downstream_50, 1)
    upstream = features["accessibility_upstream_50"]
    downstream = features["accessibility_downstream_50"]
    if pd.notna(upstream) and float(upstream) > 0 and pd.notna(downstream):
        features["accessibility_asymmetry"] = float(downstream) / float(upstream)
    features["min_accessibility_start_pm10"] = min_probability(probabilities, start_pm10, 1)
    start_pm10_value = features["accessibility_start_pm10"]
    if pd.notna(start_pm10_value):
        features["start_region_highly_accessible"] = bool(float(start_pm10_value) >= 0.5)

    # RNAplfold u=3 at +1 spans the start codon; u=7 at -3 spans -3..+4.
    features["accessibility_start_codon_u3"] = segment_probability(probabilities, 1, 3)
    features["accessibility_minus3_plus4_u7"] = segment_probability(probabilities, -3, 7)
    return features


def run_accessibility_jobs(
    df: pd.DataFrame,
    rnaplfold_bin: Optional[str],
    threads: int,
    window: int,
    span: int,
    max_unpaired: int,
    progress_every: int,
) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
    out = df.copy()
    failures: List[Dict[str, str]] = []
    for col in ACCESSIBILITY_COLUMNS + ["start_region_highly_accessible"]:
        out[col] = pd.NA

    if rnaplfold_bin is None:
        out["rnaplfold_status"] = np.where(
            out["rnaplfold_status"].eq("sequence_ok"),
            "rnaplfold_unavailable",
            out["rnaplfold_status"],
        )
        for _, row in out.iterrows():
            if row["rnaplfold_status"] == "rnaplfold_unavailable":
                failures.append(
                    {
                        "ORF_id": row["ORF_id"],
                        "status": "rnaplfold_unavailable",
                        "message": "RNAplfold executable not found",
                    }
                )
        return out, failures

    runnable = out.index[out["rnaplfold_status"].eq("sequence_ok")].tolist()
    print(
        "Running RNAplfold for {} ORF(s) with -W {} -L {} -u {} using {} thread(s)".format(
            len(runnable), window, span, max_unpaired, max(1, threads)
        )
    )

    future_to_idx = {}
    with ThreadPoolExecutor(max_workers=max(1, threads)) as executor:
        for idx in runnable:
            row = out.loc[idx]
            future = executor.submit(
                run_rnaplfold_one,
                row["ORF_id"],
                row["sequence"],
                rnaplfold_bin,
                window,
                span,
                max_unpaired,
            )
            future_to_idx[future] = idx

        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            completed += 1
            try:
                orf_id, status, probabilities, _lengths, message = future.result()
            except Exception as exc:  # noqa: BLE001 - keep large runs going.
                orf_id = out.at[idx, "ORF_id"]
                status = "rnaplfold_exception"
                probabilities = None
                message = str(exc)[:1000]

            out.at[idx, "rnaplfold_status"] = status
            if status == "rnaplfold_ok":
                features = compute_accessibility_features(probabilities)
                for col, value in features.items():
                    out.at[idx, col] = value
            else:
                failures.append({"ORF_id": orf_id, "status": status, "message": message})

            if progress_every > 0 and (completed % progress_every == 0 or completed == len(runnable)):
                print("RNAplfold completed {}/{} ORF(s)".format(completed, len(runnable)))

    return out, failures


def mannwhitney_feature_stats(df: pd.DataFrame, feature: str, pairs: Sequence[Tuple[str, str]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    work = df[["group", feature]].copy()
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    for group1, group2 in pairs:
        a = work.loc[work["group"].eq(group1), feature].dropna().astype(float)
        b = work.loc[work["group"].eq(group2), feature].dropna().astype(float)
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
                "group1": group1,
                "group2": group2,
                "group1_label": display_group(group1),
                "group2_label": display_group(group2),
                "group1_n": int(len(a)),
                "group2_n": int(len(b)),
                "group1_median": float(a.median()) if len(a) else np.nan,
                "group2_median": float(b.median()) if len(b) else np.nan,
                "rank_biserial": rank_biserial,
                "statistic": u_stat,
                "pvalue": pvalue,
            }
        )
    return pd.DataFrame(rows)


def fisher_feature_stats(df: pd.DataFrame, feature: str, pairs: Sequence[Tuple[str, str]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    work = df[["group", feature]].copy()
    work[feature] = work[feature].map(
        lambda value: pd.NA
        if pd.isna(value)
        else str(value).strip().lower() in {"true", "1", "yes", "y"}
    )
    for group1, group2 in pairs:
        a = work.loc[work["group"].eq(group1), feature].dropna()
        b = work.loc[work["group"].eq(group2), feature].dropna()
        a_true = int(a.eq(True).sum())
        a_false = int(a.eq(False).sum())
        b_true = int(b.eq(True).sum())
        b_false = int(b.eq(False).sum())
        if len(a) and len(b):
            odds_ratio, pvalue = stats.fisher_exact([[a_true, a_false], [b_true, b_false]])
        else:
            odds_ratio = np.nan
            pvalue = np.nan
        rows.append(
            {
                "feature": feature,
                "test": "fisher_exact",
                "group1": group1,
                "group2": group2,
                "group1_label": display_group(group1),
                "group2_label": display_group(group2),
                "group1_n": int(len(a)),
                "group2_n": int(len(b)),
                "group1_true": a_true,
                "group1_false": a_false,
                "group2_true": b_true,
                "group2_false": b_false,
                "odds_ratio": odds_ratio,
                "statistic": odds_ratio,
                "pvalue": pvalue,
            }
        )
    return pd.DataFrame(rows)


def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    continuous_features = [
        "accessibility_minus3_plus4",
        "accessibility_start_codon",
        "accessibility_kozak_core",
        "accessibility_start_pm10",
        "accessibility_start_pm20",
        "accessibility_upstream_50",
        "accessibility_downstream_50",
        "accessibility_asymmetry",
        "min_accessibility_start_pm10",
        "accessibility_start_codon_u3",
        "accessibility_minus3_plus4_u7",
    ]
    tables = [mannwhitney_feature_stats(df, feature, GROUP_PAIRS) for feature in continuous_features]
    tables.append(fisher_feature_stats(df, "start_region_highly_accessible", GROUP_PAIRS))
    out = pd.concat(tables, ignore_index=True)
    out["padj_bh"] = bh_adjust(out["pvalue"])
    return out


def add_pvalue_brackets(ax, order: Sequence[str], stats_df: pd.DataFrame, p_col: str = "padj_bh") -> None:
    if stats_df.empty:
        return
    ymin, ymax = ax.get_ylim()
    if not np.isfinite(ymax):
        ymax = 1.0
    span = ymax - ymin
    if span <= 0:
        span = max(abs(ymax), 1.0)
    step = span * 0.08
    current_y = ymax + step
    lookup = {group: idx for idx, group in enumerate(order)}
    for _, row in stats_df.iterrows():
        group1 = row.get("group1")
        group2 = row.get("group2")
        if group1 not in lookup or group2 not in lookup:
            continue
        x1 = lookup[group1]
        x2 = lookup[group2]
        if x1 > x2:
            x1, x2 = x2, x1
        ax.plot([x1, x1, x2, x2], [current_y, current_y + step * 0.25, current_y + step * 0.25, current_y], color="black", lw=0.8)
        ax.text((x1 + x2) / 2, current_y + step * 0.28, pvalue_label(row.get(p_col)), ha="center", va="bottom", fontsize=8)
        current_y += step
    ax.set_ylim(ymin, current_y + step)


def boxplot_by_group(ax, df: pd.DataFrame, feature: str, title: str, y_label: str, stats_df: pd.DataFrame) -> None:
    work = df[["group", feature]].copy()
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work = work.dropna(subset=["group", feature])
    if work.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "{}\nnot available".format(title), ha="center", va="center")
        return
    order = group_order_present(work)
    sns.boxplot(
        data=work,
        x="group",
        y=feature,
        order=order,
        ax=ax,
        showfliers=False,
        color="#d6e4f0",
        linewidth=1.2,
    )
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([display_group(group) for group in order], rotation=25, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    add_pvalue_brackets(ax, order, stats_df[stats_df["feature"].eq(feature)])


def fraction_barplot_by_group(ax, df: pd.DataFrame, feature: str, title: str, stats_df: pd.DataFrame) -> None:
    work = df[["group", feature]].copy()
    work[feature] = work[feature].map(
        lambda value: pd.NA
        if pd.isna(value)
        else str(value).strip().lower() in {"true", "1", "yes", "y"}
    )
    rows = []
    for group, sub in work.groupby("group", dropna=False):
        valid = sub[feature].dropna()
        rows.append(
            {
                "group": group,
                "n": int(len(valid)),
                "fraction": float(valid.eq(True).mean()) if len(valid) else np.nan,
            }
        )
    plot_df = pd.DataFrame(rows)
    plot_df = plot_df.dropna(subset=["group", "fraction"])
    if plot_df.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "{}\nnot available".format(title), ha="center", va="center")
        return
    order = group_order_present(plot_df)
    sns.barplot(data=plot_df, x="group", y="fraction", order=order, ax=ax, color="#7ca6c0")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([display_group(group) for group in order], rotation=25, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    add_pvalue_brackets(ax, order, stats_df[stats_df["feature"].eq(feature)])


def save_figure(fig, pdf_path: Path, png_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote {}".format(pdf_path))
    print("Wrote {}".format(png_path))


def make_group_figure(df: pd.DataFrame, stats_df: pd.DataFrame, pdf_dir: Path, png_dir: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(13, 16), squeeze=False)
    box_specs = [
        ("accessibility_minus3_plus4", "Mean unpaired probability", "-3 to +4 accessibility"),
        ("accessibility_start_codon", "Mean unpaired probability", "Start codon accessibility"),
        ("accessibility_start_pm10", "Mean unpaired probability", "Start pm10 accessibility"),
        ("accessibility_start_pm20", "Mean unpaired probability", "Start pm20 accessibility"),
        ("accessibility_upstream_50", "Mean unpaired probability", "Upstream 50 accessibility"),
        ("accessibility_asymmetry", "Downstream / upstream", "Accessibility asymmetry"),
    ]
    for ax, (feature, ylabel, title) in zip(axes.flat, box_specs):
        boxplot_by_group(ax, df, feature, title, ylabel, stats_df)
    fraction_barplot_by_group(
        axes[3, 0],
        df,
        "start_region_highly_accessible",
        "Highly accessible start region fraction",
        stats_df,
    )
    axes[3, 1].axis("off")
    save_figure(
        fig,
        pdf_dir / "Fig_structure_RNAplfold_accessibility_by_group.pdf",
        png_dir / "Fig_structure_RNAplfold_accessibility_by_group.png",
    )


def noncanonical_subset(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[~df["group"].eq("group1_canonical_translated_ORF")].copy()
    sub["primary_noncanonical_category"] = sub["primary_noncanonical_category"].replace("", "unknown")
    return sub


def subtype_order(df: pd.DataFrame, category_col: str) -> List[str]:
    counts = df[category_col].value_counts(dropna=False)
    return counts.index.astype(str).tolist()


def subtype_fraction_table(df: pd.DataFrame, category_col: str, bool_col: str) -> pd.DataFrame:
    rows = []
    for category, sub in df.groupby(category_col, dropna=False):
        valid = sub[bool_col].dropna()
        rows.append(
            {
                category_col: category,
                "n": int(len(valid)),
                "fraction": float(valid.eq(True).mean()) if len(valid) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def make_subtype_figure(df: pd.DataFrame, pdf_dir: Path, png_dir: Path) -> None:
    sub = noncanonical_subset(df)
    category_col = "primary_noncanonical_category"
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), squeeze=False)
    ax = axes[0, 0]
    work = sub[[category_col, "accessibility_start_pm10"]].copy()
    work["accessibility_start_pm10"] = pd.to_numeric(work["accessibility_start_pm10"], errors="coerce")
    work = work.dropna(subset=[category_col, "accessibility_start_pm10"])
    if work.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "Subtype start pm10 accessibility\nnot available", ha="center", va="center")
    else:
        order = subtype_order(work, category_col)
        sns.boxplot(
            data=work,
            x=category_col,
            y="accessibility_start_pm10",
            order=order,
            ax=ax,
            showfliers=False,
            color="#d6e4f0",
            linewidth=1.2,
        )
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=35, ha="right")
        ax.set_xlabel("")
        ax.set_ylabel("Mean unpaired probability")
        ax.set_title("Start pm10 accessibility by noncanonical subtype")

    ax = axes[0, 1]
    bool_work = sub[[category_col, "start_region_highly_accessible"]].copy()
    bool_work["start_region_highly_accessible"] = bool_work["start_region_highly_accessible"].map(
        lambda value: pd.NA
        if pd.isna(value)
        else str(value).strip().lower() in {"true", "1", "yes", "y"}
    )
    fraction_df = subtype_fraction_table(bool_work, category_col, "start_region_highly_accessible")
    fraction_df = fraction_df.dropna(subset=[category_col, "fraction"])
    if fraction_df.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "Subtype accessible fraction\nnot available", ha="center", va="center")
    else:
        order = subtype_order(bool_work.dropna(subset=[category_col]), category_col)
        sns.barplot(data=fraction_df, x=category_col, y="fraction", order=order, ax=ax, color="#7ca6c0")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=35, ha="right")
        ax.set_xlabel("")
        ax.set_ylabel("Fraction")
        ax.set_ylim(0, 1)
        ax.set_title("Highly accessible fraction by noncanonical subtype")

    save_figure(
        fig,
        pdf_dir / "Fig_structure_RNAplfold_accessibility_by_noncanonical_subtype.pdf",
        png_dir / "Fig_structure_RNAplfold_accessibility_by_noncanonical_subtype.png",
    )


def main() -> int:
    args = parse_args()
    output_dirs = ensure_output_dirs(args.fig_dir)
    table_out = output_dirs["tables"] / "orf_rnaplfold_accessibility_features.tsv"
    stats_out = output_dirs["tables"] / "rnaplfold_accessibility_statistics.tsv"
    failed_out = output_dirs["logs"] / "rnaplfold_failed_orfs.tsv"

    print("structure_rnaplfold_accessibility.py")
    print("INPUT_DIR={}".format(args.input_dir))
    print("FIG_DIR={}".format(args.fig_dir))
    print("GENOME_FA={}".format(args.genome_fa))
    print("RNAPLFOLD requested={}".format(args.rnaplfold))
    print("RNAplfold parameters: -W {} -L {} -u {}".format(args.window, args.span, args.max_unpaired))
    if args.limit is not None:
        print("limit={}".format(args.limit))

    rnaplfold_bin, rnaplfold_source = resolve_executable(args.rnaplfold, "RNAplfold")
    print("RNAplfold resolved={} ({})".format(rnaplfold_bin if rnaplfold_bin else "unavailable", rnaplfold_source))

    groups = read_orf_table(args.input_dir, args.limit)
    print("Loaded {} ORF(s)".format(len(groups)))
    sequence_df, sequence_failures = prepare_sequences(groups, args.genome_fa)
    result_df, rnaplfold_failures = run_accessibility_jobs(
        sequence_df,
        rnaplfold_bin,
        args.threads,
        args.window,
        args.span,
        args.max_unpaired,
        args.progress_every,
    )

    for col in ACCESSIBILITY_COLUMNS:
        result_df[col] = pd.to_numeric(result_df[col], errors="coerce")
    result_df = result_df.drop(columns=["sequence"])
    result_df = result_df[OUTPUT_COLUMNS]
    result_df.to_csv(table_out, sep="\t", index=False, na_rep="NA")
    print("Wrote {}".format(table_out))

    failures = sequence_failures + rnaplfold_failures
    if failures:
        failed_df = pd.DataFrame(failures).drop_duplicates()
    else:
        failed_df = pd.DataFrame(columns=["ORF_id", "status", "message"])
    failed_df.to_csv(failed_out, sep="\t", index=False, na_rep="NA")
    print("Wrote {}".format(failed_out))

    stats_df = compute_statistics(result_df)
    stats_df.to_csv(stats_out, sep="\t", index=False, na_rep="NA")
    print("Wrote {}".format(stats_out))

    make_group_figure(result_df, stats_df, output_dirs["pdf"], output_dirs["png"])
    make_subtype_figure(result_df, output_dirs["pdf"], output_dirs["png"])

    ok_count = int(result_df["rnaplfold_status"].eq("rnaplfold_ok").sum())
    print("RNAplfold OK ORFs: {}/{}".format(ok_count, len(result_df)))
    print("Higher accessibility means higher local unpaired/open probability.")
    print("structure_rnaplfold_accessibility.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
