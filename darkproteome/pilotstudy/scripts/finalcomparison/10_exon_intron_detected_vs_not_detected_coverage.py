#!/usr/bin/env python3
"""Recompute detected-vs-not-detected RNA coverage with exon/intron separation."""

from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
    NEG_DIR,
    PILOT_DIR,
    PLOT_GROUP_ORDER,
    bw_values,
    discover_bigwigs,
    finite_mean,
    finite_sum,
    import_pybigwig,
    normalize_codon,
    plot_group_from_group,
    pvalue_label,
    read_tsv,
    vector_mean_normalize,
)


BASE_DIR = Path(os.environ.get("BASE_DIR", str(PILOT_DIR)))
GTF = Path(os.environ.get("GTF", str(BASE_DIR / "gencode.v48.annotation.gtf")))
GENOME_FA = Path(os.environ.get("GENOME_FA", str(BASE_DIR / "hg38.fa")))
OUT_DIR = Path(os.environ.get("OUT_DIR", str(FIG_DIR / "coverage_exon_intron_detected_vs_not_detected")))

POSITIVE_METADATA = INPUT_DIR / "tables" / "orf_groups.combined_metadata.tsv"
NEGATIVE_METADATA = NEG_DIR / "tables" / "cpat_negative_orfs.combined_metadata_compatible.tsv"
DETECTION_TABLE = INPUT_DIR / "tables" / "pancreas.translated_orfs.sample_level.tsv"
OLD_COVERAGE_TABLE = FIG_DIR / "tables" / "four_group_coverage_features.sample_level.tsv"

TABLE_DIR = OUT_DIR / "tables"
PDF_DIR = OUT_DIR / "pdf"
PNG_DIR = OUT_DIR / "png"
LOG_DIR = OUT_DIR / "logs"
SCRIPT_DIR = OUT_DIR / "scripts"

BLOCKS_OUT = TABLE_DIR / "orf_exon_intron_blocks.tsv"
SAMPLE_OUT = TABLE_DIR / "coverage_exon_intron_features.sample_level.tsv"
ORF_OUT = TABLE_DIR / "coverage_exon_intron_features.orf_level.tsv"
STATS_OUT = TABLE_DIR / "coverage_exon_intron_detected_vs_not_detected_statistics.tsv"
METAPLOT_OUT = TABLE_DIR / "metaplot_exon_only_detected_vs_not_detected.tsv"
QC_SUMMARY_OUT = TABLE_DIR / "orf_exon_intron_qc_summary.tsv"
WARNINGS_OUT = LOG_DIR / "exon_block_mapping_warnings.tsv"
VALIDATION_OUT = LOG_DIR / "start_codon_orientation_validation.tsv"
README_OUT = OUT_DIR / "README_exon_intron_coverage_analysis.txt"

QC_PDF = PDF_DIR / "Fig_QC_exon_intron_coverage.pdf"
QC_PNG = PNG_DIR / "Fig_QC_exon_intron_coverage.png"
EXON_PDF = PDF_DIR / "Fig_exon_only_detected_vs_not_detected_coverage.pdf"
EXON_PNG = PNG_DIR / "Fig_exon_only_detected_vs_not_detected_coverage.png"
INTRON_PDF = PDF_DIR / "Fig_intronic_span_detected_vs_not_detected_coverage.pdf"
INTRON_PNG = PNG_DIR / "Fig_intronic_span_detected_vs_not_detected_coverage.png"
METAPLOT_PDF = PDF_DIR / "Fig_exon_only_detected_vs_not_detected_metaplot.pdf"
METAPLOT_PNG = PNG_DIR / "Fig_exon_only_detected_vs_not_detected_metaplot.png"

STATUS_ORDER = ["detected", "not_detected"]
STATUS_PALETTE = {"detected": "#252525", "not_detected": "#BDBDBD"}
METAPLOT_COLORS = dict(GROUP_COLORS)
METAPLOT_COLORS[NEGATIVE_GROUP] = "#E45756"
LINESTYLES = {"detected": "-", "not_detected": "--"}
REVCOMP = str.maketrans("ACGTUNacgtun", "TGCAANtgcaan")

EXON_FEATURES = [
    ("exon_mean_ORF_coverage", "Exon-only mean ORF coverage"),
    ("exon_start_peak_coverage", "Exon-only start peak coverage"),
    ("exon_start_peak_ratio", "Exon-only start peak ratio"),
    ("exon_upstream_slope", "Exon-only upstream slope"),
    ("exon_AUC_upstream", "Exon-only AUC upstream"),
    ("exon_AUC_downstream", "Exon-only AUC downstream"),
]

INTRON_FEATURES = [
    ("intron_mean_coverage", "Intronic-span mean coverage"),
    ("intron_total_coverage", "Intronic-span total coverage"),
    ("intron_covered_fraction", "Intronic-span covered fraction"),
    ("intron_exon_coverage_ratio", "Intron/exon coverage ratio"),
    ("intronic_fraction_of_genomic_span", "Intronic fraction of ORF genomic span"),
    ("full_span_vs_exon_mean_ratio", "Full-span/exon mean ratio"),
]

WARNING_COLUMNS = ["ORF_id", "transcript_id", "warning_type", "message"]


Block = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Separate ORF RNA coverage into transcript exon-only and intronic-span "
            "signals, then redraw detected-vs-not-detected coverage plots."
        )
    )
    parser.add_argument("--flank", type=int, default=100, help="Start-centered metaplot flank in transcript bases.")
    parser.add_argument("--validation-per-strand", type=int, default=10, help="Random ORFs per strand for start codon validation.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for validation subset selection.")
    parser.add_argument(
        "--max-scatter-points",
        type=int,
        default=50000,
        help="Maximum points to draw in the old full-span vs exon-only QC scatter panel.",
    )
    return parser.parse_args()


def ensure_out_dirs() -> None:
    for path in [TABLE_DIR, PDF_DIR, PNG_DIR, LOG_DIR, SCRIPT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def archive_script() -> None:
    src = Path(__file__)
    if src.exists():
        shutil.copy2(src, SCRIPT_DIR / src.name)
    helper = Path(__file__).with_name("fc_common.py")
    if helper.exists():
        shutil.copy2(helper, SCRIPT_DIR / helper.name)


def strip_version(identifier: object) -> str:
    text = "" if pd.isna(identifier) else str(identifier).strip()
    return text.split(".", 1)[0] if text else ""


def parse_gtf_attributes(text: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for field in text.strip().rstrip(";").split(";"):
        field = field.strip()
        if not field:
            continue
        if " " not in field:
            continue
        key, value = field.split(" ", 1)
        attrs[key] = value.strip().strip('"')
    return attrs


def parse_blocks(text: object) -> List[Block]:
    if pd.isna(text):
        return []
    blocks: List[Block] = []
    for item in str(text).split(";"):
        item = item.strip()
        if not item or item == "NA":
            continue
        if ":" in item:
            item = item.split(":", 1)[1]
        match = re.match(r"^(-?\d+)-(-?\d+)$", item)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            if end > start:
                blocks.append((start, end))
    return blocks


def format_blocks(chrom: str, blocks: Sequence[Block]) -> str:
    if not blocks:
        return "NA"
    return ";".join(f"{chrom}:{start}-{end}" for start, end in blocks)


def merge_blocks(blocks: Sequence[Block]) -> List[Block]:
    merged: List[Block] = []
    for start, end in sorted(blocks):
        if end <= start:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def intersect_blocks(span_start: int, span_end: int, blocks: Sequence[Block]) -> List[Block]:
    out = []
    for start, end in blocks:
        clipped_start = max(span_start, int(start))
        clipped_end = min(span_end, int(end))
        if clipped_end > clipped_start:
            out.append((clipped_start, clipped_end))
    return merge_blocks(out)


def complement_blocks(span_start: int, span_end: int, covered_blocks: Sequence[Block]) -> List[Block]:
    introns: List[Block] = []
    cursor = span_start
    for start, end in merge_blocks(covered_blocks):
        if start > cursor:
            introns.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < span_end:
        introns.append((cursor, span_end))
    return introns


def block_length(blocks: Sequence[Block]) -> int:
    return int(sum(max(0, int(end) - int(start)) for start, end in blocks))


def orient_blocks(blocks: Sequence[Block], strand: str) -> List[Block]:
    blocks = list(blocks)
    return sorted(blocks) if strand == "+" else sorted(blocks, reverse=True)


def load_metadata() -> pd.DataFrame:
    positive = read_tsv(POSITIVE_METADATA, required=["ORF_id", "transcript_id", "chr", "start0", "end0", "strand"])
    negative = read_tsv(NEGATIVE_METADATA, required=["ORF_id", "transcript_id", "chr", "start0", "end0", "strand"])
    positive = positive.copy()
    negative = negative.copy()
    if "plot_group" not in positive.columns:
        if "group" not in positive.columns:
            raise SystemExit(f"Positive metadata needs either group or plot_group column: {POSITIVE_METADATA}")
        positive["plot_group"] = positive["group"].map(plot_group_from_group)
    else:
        positive["plot_group"] = positive["plot_group"].map(plot_group_from_group)
    negative["plot_group"] = NEGATIVE_GROUP
    for df in [positive, negative]:
        if "chrom" in df.columns and "chr" not in df.columns:
            df.rename(columns={"chrom": "chr"}, inplace=True)
        for col in ["start_codon", "ORF_type", "ORF_type2", "primary_noncanonical_category", "source"]:
            if col not in df.columns:
                df[col] = pd.NA
    columns = [
        "plot_group",
        "ORF_id",
        "transcript_id",
        "chr",
        "start0",
        "end0",
        "strand",
        "start_codon",
        "ORF_type",
        "ORF_type2",
        "primary_noncanonical_category",
        "source",
    ]
    master = pd.concat([positive[columns], negative[columns]], ignore_index=True)
    master["start0"] = pd.to_numeric(master["start0"], errors="coerce")
    master["end0"] = pd.to_numeric(master["end0"], errors="coerce")
    master = master.dropna(subset=["ORF_id", "transcript_id", "chr", "start0", "end0", "strand"])
    master["start0"] = master["start0"].astype(int)
    master["end0"] = master["end0"].astype(int)
    master = master[master["end0"].gt(master["start0"])].copy()
    master = master[master["plot_group"].isin(PLOT_GROUP_ORDER)].copy()
    master = master.drop_duplicates(subset=["ORF_id"], keep="first").reset_index(drop=True)
    return master


def load_detection_pairs() -> set[tuple[str, str]]:
    detected = read_tsv(DETECTION_TABLE, required=["ORF_id", "sample"])
    return set(zip(detected["ORF_id"].astype(str), detected["sample"].astype(str)))


def load_gtf_exons(transcript_ids: Iterable[str]) -> tuple[Dict[str, dict], List[dict]]:
    needed_full = {str(tid) for tid in transcript_ids if str(tid)}
    needed_base = {strip_version(tid) for tid in needed_full}
    transcripts: Dict[str, dict] = {}
    alias: Dict[str, str] = {}
    warnings: List[dict] = []
    if not GTF.exists():
        raise SystemExit(f"GENCODE GTF not found: {GTF}")
    opener = open
    if GTF.suffix == ".gz":
        import gzip

        opener = gzip.open
    with opener(GTF, "rt") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "exon":
                continue
            attrs = parse_gtf_attributes(fields[8])
            transcript_id = attrs.get("transcript_id", "")
            if transcript_id not in needed_full and strip_version(transcript_id) not in needed_base:
                continue
            chrom = fields[0]
            start = int(fields[3]) - 1
            end = int(fields[4])
            strand = fields[6]
            key = transcript_id
            if key not in transcripts:
                transcripts[key] = {"transcript_id": transcript_id, "chr": chrom, "strand": strand, "exons": []}
            transcripts[key]["exons"].append((start, end))
            base = strip_version(transcript_id)
            previous = alias.get(base)
            if previous is None:
                alias[base] = key
            elif previous != key:
                warnings.append(
                    {
                        "ORF_id": "NA",
                        "transcript_id": transcript_id,
                        "warning_type": "transcript_version_alias_collision",
                        "message": f"{base} maps to both {previous} and {key}; exact IDs are preferred.",
                    }
                )
    for key, record in transcripts.items():
        record["exons"] = merge_blocks(record["exons"])
    for base, key in alias.items():
        transcripts.setdefault(base, transcripts[key])
    return transcripts, warnings


def transcript_record(transcripts: Dict[str, dict], transcript_id: str) -> Optional[dict]:
    return transcripts.get(transcript_id) or transcripts.get(strip_version(transcript_id))


def transcript_oriented_coords(exons: Sequence[Block], strand: str) -> List[int]:
    coords: List[int] = []
    if strand == "+":
        for start, end in sorted(exons):
            coords.extend(range(start, end))
    else:
        for start, end in sorted(exons, reverse=True):
            coords.extend(range(end - 1, start - 1, -1))
    return coords


def start_coord(row: pd.Series) -> int:
    return int(row["start0"]) if str(row["strand"]) == "+" else int(row["end0"]) - 1


def build_block_table(master: pd.DataFrame, transcripts: Dict[str, dict], warnings: List[dict]) -> pd.DataFrame:
    rows = []
    for _, row in master.iterrows():
        chrom = str(row["chr"])
        strand = str(row["strand"])
        start0 = int(row["start0"])
        end0 = int(row["end0"])
        transcript_id = str(row["transcript_id"])
        record = transcript_record(transcripts, transcript_id)
        status = "ok"
        transcript_exons: List[Block] = []
        matched_transcript_id = "NA"
        if record is None:
            status = "missing_transcript_in_gtf"
            warnings.append(
                {
                    "ORF_id": row["ORF_id"],
                    "transcript_id": transcript_id,
                    "warning_type": status,
                    "message": "No transcript exon annotation found in GENCODE GTF.",
                }
            )
        else:
            matched_transcript_id = str(record["transcript_id"])
            transcript_exons = list(record["exons"])
            if str(record["chr"]) != chrom:
                status = "chrom_mismatch"
                warnings.append(
                    {
                        "ORF_id": row["ORF_id"],
                        "transcript_id": transcript_id,
                        "warning_type": status,
                        "message": f"ORF chrom={chrom}; GTF chrom={record['chr']}.",
                    }
                )
            if str(record["strand"]) != strand:
                status = "strand_mismatch" if status == "ok" else status + ";strand_mismatch"
                warnings.append(
                    {
                        "ORF_id": row["ORF_id"],
                        "transcript_id": transcript_id,
                        "warning_type": "strand_mismatch",
                        "message": f"ORF strand={strand}; GTF strand={record['strand']}.",
                    }
                )
        exons_on_chrom = transcript_exons if record is not None and str(record["chr"]) == chrom else []
        exon_blocks_genomic = intersect_blocks(start0, end0, exons_on_chrom)
        intron_blocks = complement_blocks(start0, end0, exon_blocks_genomic) if exons_on_chrom else []
        exon_blocks_tx = orient_blocks(exon_blocks_genomic, strand)
        genomic_span_length = end0 - start0
        exon_length = block_length(exon_blocks_genomic)
        intron_length = block_length(intron_blocks)
        intronic_fraction = intron_length / genomic_span_length if genomic_span_length else np.nan
        coords = transcript_oriented_coords(exons_on_chrom, strand) if exons_on_chrom else []
        coord_to_index = {coord: idx for idx, coord in enumerate(coords)}
        start_genomic_coord = start_coord(row)
        start_index = coord_to_index.get(start_genomic_coord)
        if exon_length == 0 and status == "ok":
            status = "orf_does_not_overlap_gtf_exons"
            warnings.append(
                {
                    "ORF_id": row["ORF_id"],
                    "transcript_id": transcript_id,
                    "warning_type": status,
                    "message": "ORF genomic span did not intersect transcript exons.",
                }
            )
        if start_index is None and exons_on_chrom:
            warnings.append(
                {
                    "ORF_id": row["ORF_id"],
                    "transcript_id": transcript_id,
                    "warning_type": "start_coord_not_in_exons",
                    "message": f"Translation start coordinate {start_genomic_coord} was not inside the transcript exon blocks.",
                }
            )
        rows.append(
            {
                "ORF_id": row["ORF_id"],
                "plot_group": row["plot_group"],
                "transcript_id": transcript_id,
                "gtf_transcript_id": matched_transcript_id,
                "chr": chrom,
                "start0": start0,
                "end0": end0,
                "strand": strand,
                "start_codon": row.get("start_codon", pd.NA),
                "ORF_type": row.get("ORF_type", pd.NA),
                "ORF_type2": row.get("ORF_type2", pd.NA),
                "primary_noncanonical_category": row.get("primary_noncanonical_category", pd.NA),
                "source": row.get("source", pd.NA),
                "genomic_span_length": genomic_span_length,
                "exon_only_length": exon_length,
                "intronic_span_length": intron_length,
                "intronic_fraction_of_genomic_span": intronic_fraction,
                "n_exon_blocks": len(exon_blocks_genomic),
                "n_intron_blocks": len(intron_blocks),
                "transcript_exon_blocks_genomic_order": format_blocks(chrom, sorted(exons_on_chrom)),
                "orf_exon_blocks_transcript_order": format_blocks(chrom, exon_blocks_tx),
                "orf_intron_blocks_genomic_order": format_blocks(chrom, intron_blocks),
                "start_genomic_coord": start_genomic_coord,
                "start_transcript_index": start_index if start_index is not None else pd.NA,
                "block_mapping_status": status,
            }
        )
    return pd.DataFrame(rows)


def values_from_blocks(bw, chrom: str, blocks: Sequence[Block], strand: str) -> np.ndarray:
    pieces = []
    for start, end in orient_blocks(blocks, strand):
        values = bw_values(bw, chrom, int(start), int(end))
        if strand == "-":
            values = values[::-1]
        pieces.append(values)
    return np.concatenate(pieces) if pieces else np.asarray([], dtype=float)


def values_for_ordered_coords(bw, chrom: str, coords: Sequence[Optional[int]]) -> np.ndarray:
    out = np.full(len(coords), np.nan, dtype=float)
    run_indices: List[int] = []
    run_coords: List[int] = []

    def flush() -> None:
        nonlocal run_indices, run_coords
        if not run_indices:
            return
        start = min(run_coords)
        end = max(run_coords) + 1
        values = bw_values(bw, chrom, start, end)
        lookup = {coord: values[coord - start] for coord in range(start, end)}
        for idx, coord in zip(run_indices, run_coords):
            out[idx] = lookup.get(coord, np.nan)
        run_indices = []
        run_coords = []

    for idx, coord in enumerate(coords):
        if coord is None:
            flush()
            continue
        coord = int(coord)
        if run_coords and abs(coord - run_coords[-1]) != 1:
            flush()
        run_indices.append(idx)
        run_coords.append(coord)
    flush()
    return out


def start_centered_transcript_vector(
    bw,
    chrom: str,
    transcript_exons: Sequence[Block],
    strand: str,
    start_genomic_coord: int,
    flank: int,
) -> tuple[np.ndarray, List[Optional[int]], bool, Optional[int]]:
    coords = transcript_oriented_coords(transcript_exons, strand)
    coord_to_index = {coord: idx for idx, coord in enumerate(coords)}
    start_index = coord_to_index.get(start_genomic_coord)
    target_coords: List[Optional[int]] = []
    if start_index is None:
        return np.full((2 * flank) + 1, np.nan, dtype=float), [None] * ((2 * flank) + 1), strand == "-", None
    for pos in range(-flank, flank + 1):
        target_index = start_index + pos
        target_coords.append(coords[target_index] if 0 <= target_index < len(coords) else None)
    values = values_for_ordered_coords(bw, chrom, target_coords)
    if strand == "-":
        finite = np.isfinite(values)
        values[finite] = values[finite]
    return values, target_coords, strand == "-", start_index


def finite_fraction_gt_zero(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    if not mask.any():
        return np.nan
    return float(np.mean(values[mask] > 0))


def finite_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    return float(stats.linregress(x[mask], y[mask]).slope)


def start_vector_features(vector: np.ndarray, positions: np.ndarray) -> Dict[str, float]:
    vector = np.asarray(vector, dtype=float)
    positions = np.asarray(positions, dtype=int)
    upstream_mask = (positions >= -100) & (positions <= 0)
    downstream_mask = (positions >= 0) & (positions <= 100)
    start_peak_mask = (positions >= -5) & (positions <= 5)
    mean_window = finite_mean(vector)
    start_peak = finite_mean(vector[start_peak_mask])
    auc_upstream = finite_sum(vector[upstream_mask])
    auc_downstream = finite_sum(vector[downstream_mask])
    return {
        "mean_window_coverage": mean_window,
        "start_peak_coverage": start_peak,
        "start_peak_ratio": safe_ratio(start_peak, mean_window),
        "upstream_slope": finite_slope(positions[upstream_mask], vector[upstream_mask]),
        "AUC_upstream": auc_upstream,
        "AUC_downstream": auc_downstream,
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return float(numerator / denominator)


def update_metaplot_agg(
    agg: dict,
    group: str,
    status: str,
    raw_vector: np.ndarray,
    norm_vector: np.ndarray,
    positions: np.ndarray,
) -> None:
    for pos, raw_value, norm_value in zip(positions, raw_vector, norm_vector):
        key = (group, status, int(pos))
        item = agg.setdefault(key, {"raw_sum": 0.0, "raw_n": 0, "norm_sum": 0.0, "norm_n": 0})
        if np.isfinite(raw_value):
            item["raw_sum"] += float(raw_value)
            item["raw_n"] += 1
        if np.isfinite(norm_value):
            item["norm_sum"] += float(norm_value)
            item["norm_n"] += 1


def metaplot_agg_to_df(agg: dict) -> pd.DataFrame:
    rows = []
    for (group, status, pos), item in sorted(agg.items()):
        rows.append(
            {
                "plot_group": group,
                "detection_status": status,
                "position": pos,
                "mean_CPM": item["raw_sum"] / item["raw_n"] if item["raw_n"] else np.nan,
                "n_raw_vectors": item["raw_n"],
                "mean_vector_normalized": item["norm_sum"] / item["norm_n"] if item["norm_n"] else np.nan,
                "n_normalized_vectors": item["norm_n"],
            }
        )
    return pd.DataFrame(rows)


def compute_coverage_tables(
    block_df: pd.DataFrame,
    bw_paths: Dict[str, Path],
    detected_pairs: set[tuple[str, str]],
    flank: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pyBigWig = import_pybigwig()
    handles = {sample: pyBigWig.open(str(path)) for sample, path in bw_paths.items()}
    positions = np.arange(-flank, flank + 1, dtype=int)
    sample_rows: List[dict] = []
    metaplot_agg: dict = {}
    try:
        for sample, bw in handles.items():
            print(f"Processing coverage for {sample}")
            for idx, row in block_df.iterrows():
                if idx and idx % 1000 == 0:
                    print(f"  {sample}: {idx}/{len(block_df)} ORFs")
                group = str(row["plot_group"])
                status = (
                    "not_detected"
                    if group == NEGATIVE_GROUP
                    else ("detected" if (str(row["ORF_id"]), sample) in detected_pairs else "not_detected")
                )
                chrom = str(row["chr"])
                strand = str(row["strand"])
                exon_blocks = parse_blocks(row["orf_exon_blocks_transcript_order"])
                intron_blocks = parse_blocks(row["orf_intron_blocks_genomic_order"])
                transcript_exons = parse_blocks(row["transcript_exon_blocks_genomic_order"])
                exon_values = values_from_blocks(bw, chrom, exon_blocks, strand)
                intron_values = values_from_blocks(bw, chrom, intron_blocks, "+")
                full_span_values = bw_values(bw, chrom, int(row["start0"]), int(row["end0"]))
                start_vector, _, _, _ = start_centered_transcript_vector(
                    bw,
                    chrom,
                    transcript_exons,
                    strand,
                    int(row["start_genomic_coord"]),
                    flank,
                )
                start_features = start_vector_features(start_vector, positions)
                exon_mean = finite_mean(exon_values)
                intron_mean = finite_mean(intron_values)
                full_span_mean = finite_mean(full_span_values)
                norm_vector = vector_mean_normalize(start_vector)
                update_metaplot_agg(metaplot_agg, group, status, start_vector, norm_vector, positions)
                sample_rows.append(
                    {
                        "ORF_id": row["ORF_id"],
                        "sample": sample,
                        "plot_group": group,
                        "detection_status": status,
                        "transcript_id": row["transcript_id"],
                        "chr": chrom,
                        "start0": int(row["start0"]),
                        "end0": int(row["end0"]),
                        "strand": strand,
                        "genomic_span_length": int(row["genomic_span_length"]),
                        "exon_only_length": int(row["exon_only_length"]),
                        "intronic_span_length": int(row["intronic_span_length"]),
                        "intronic_fraction_of_genomic_span": row["intronic_fraction_of_genomic_span"],
                        "n_exon_blocks": int(row["n_exon_blocks"]),
                        "n_intron_blocks": int(row["n_intron_blocks"]),
                        "finite_exon_base_count": int(np.isfinite(exon_values).sum()),
                        "finite_intron_base_count": int(np.isfinite(intron_values).sum()),
                        "finite_full_span_base_count": int(np.isfinite(full_span_values).sum()),
                        "exon_mean_ORF_coverage": exon_mean,
                        "exon_start_peak_coverage": start_features["start_peak_coverage"],
                        "exon_start_peak_ratio": start_features["start_peak_ratio"],
                        "exon_upstream_slope": start_features["upstream_slope"],
                        "exon_AUC_upstream": start_features["AUC_upstream"],
                        "exon_AUC_downstream": start_features["AUC_downstream"],
                        "exon_start_centered_window_mean": start_features["mean_window_coverage"],
                        "intron_mean_coverage": intron_mean,
                        "intron_total_coverage": finite_sum(intron_values),
                        "intron_covered_fraction": finite_fraction_gt_zero(intron_values),
                        "intron_exon_coverage_ratio": safe_ratio(intron_mean, exon_mean),
                        "full_span_mean_coverage": full_span_mean,
                        "full_span_vs_exon_mean_ratio": safe_ratio(full_span_mean, exon_mean),
                        "bigwig_path": str(bw_paths[sample]),
                    }
                )
    finally:
        for handle in handles.values():
            handle.close()
    return pd.DataFrame(sample_rows), metaplot_agg_to_df(metaplot_agg)


def aggregate_orf_level(sample_df: pd.DataFrame) -> pd.DataFrame:
    id_cols = [
        "ORF_id",
        "plot_group",
        "transcript_id",
        "chr",
        "start0",
        "end0",
        "strand",
        "genomic_span_length",
        "exon_only_length",
        "intronic_span_length",
        "intronic_fraction_of_genomic_span",
        "n_exon_blocks",
        "n_intron_blocks",
    ]
    numeric_cols = [
        col
        for col in sample_df.columns
        if col not in set(id_cols + ["sample", "detection_status", "bigwig_path"])
        and pd.api.types.is_numeric_dtype(pd.to_numeric(sample_df[col], errors="coerce"))
    ]
    agg = sample_df.copy()
    for col in numeric_cols:
        agg[col] = pd.to_numeric(agg[col], errors="coerce")
    out = agg.groupby(id_cols, dropna=False, as_index=False)[numeric_cols].mean(numeric_only=True)
    detected_n = (
        sample_df.assign(detected_flag=sample_df["detection_status"].eq("detected").astype(int))
        .groupby("ORF_id")["detected_flag"]
        .sum()
        .rename("n_detected_samples")
        .reset_index()
    )
    return out.merge(detected_n, on="ORF_id", how="left")


def mannwhitney_detected_stats(df: pd.DataFrame, feature: str, figure: str) -> pd.DataFrame:
    rows = []
    work = df[["plot_group", "detection_status", feature]].copy()
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work = work.dropna(subset=["plot_group", "detection_status", feature])
    for group in PLOT_GROUP_ORDER:
        sub = work[work["plot_group"].eq(group)]
        detected = sub.loc[sub["detection_status"].eq("detected"), feature].dropna().astype(float)
        not_detected = sub.loc[sub["detection_status"].eq("not_detected"), feature].dropna().astype(float)
        if group == NEGATIVE_GROUP or not len(detected) or not len(not_detected):
            continue
        result = stats.mannwhitneyu(detected, not_detected, alternative="two-sided")
        rows.append(
            {
                "figure": figure,
                "feature": feature,
                "plot_group": group,
                "test": "mannwhitney_u",
                "status1": "detected",
                "status2": "not_detected",
                "status1_n": int(len(detected)),
                "status2_n": int(len(not_detected)),
                "status1_median": float(detected.median()),
                "status2_median": float(not_detected.median()),
                "u_statistic": float(result.statistic),
                "pvalue": float(result.pvalue),
            }
        )
    return pd.DataFrame(rows)


def add_hue_pvalues(ax, data: pd.DataFrame, feature: str, stats_df: pd.DataFrame) -> None:
    if stats_df.empty:
        return
    pairs = [((row["plot_group"], "detected"), (row["plot_group"], "not_detected")) for _, row in stats_df.iterrows()]
    pvalues = [row["pvalue"] for _, row in stats_df.iterrows()]
    order = [group for group in PLOT_GROUP_ORDER if group in set(data["plot_group"])]
    try:
        from statannotations.Annotator import Annotator

        annotator = Annotator(
            ax,
            pairs,
            data=data,
            x="plot_group",
            y=feature,
            hue="detection_status",
            order=order,
            hue_order=STATUS_ORDER,
        )
        annotator.configure(test=None, text_format="simple", loc="inside", verbose=0)
        annotator.set_pvalues_and_annotate(pvalues)
    except Exception:
        ymax = pd.to_numeric(data[feature], errors="coerce").max()
        ymin = pd.to_numeric(data[feature], errors="coerce").min()
        if not np.isfinite(ymax):
            return
        span = ymax - ymin if np.isfinite(ymin) and ymax > ymin else max(abs(ymax), 1.0)
        y = ymax + span * 0.04
        step = span * 0.08
        lookup = {group: idx for idx, group in enumerate(order)}
        for _, row in stats_df.iterrows():
            group = row["plot_group"]
            if group not in lookup:
                continue
            x = lookup[group]
            x1, x2 = x - 0.2, x + 0.2
            ax.plot([x1, x1, x2, x2], [y, y + step * 0.25, y + step * 0.25, y], color="black", lw=0.8)
            ax.text((x1 + x2) / 2, y + step * 0.28, pvalue_label(row["pvalue"]), ha="center", va="bottom", fontsize=8)
            y += step
        ax.set_ylim(top=y + step)


def plot_detected_boxplot(ax, df: pd.DataFrame, feature: str, title: str, figure: str) -> pd.DataFrame:
    work = df[["plot_group", "detection_status", feature]].copy()
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work = work.dropna(subset=["plot_group", "detection_status", feature])
    if work.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, f"{title}\nnot available", ha="center", va="center")
        return pd.DataFrame()
    order = [group for group in PLOT_GROUP_ORDER if group in set(work["plot_group"])]
    sns.boxplot(
        data=work,
        x="plot_group",
        y=feature,
        hue="detection_status",
        order=order,
        hue_order=STATUS_ORDER,
        palette=STATUS_PALETTE,
        showfliers=False,
        linewidth=1.0,
        ax=ax,
    )
    ax.set_title(title, pad=18)
    ax.set_xlabel("")
    ax.set_ylabel(title)
    ax.tick_params(axis="x", rotation=25)
    stats_df = mannwhitney_detected_stats(work, feature, figure)
    add_hue_pvalues(ax, work, feature, stats_df)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, title="Detection", frameon=False, loc="best")
    return stats_df


def save_figure(fig, pdf_path: Path, png_path: Path) -> None:
    fig.tight_layout(pad=2.4, h_pad=3.8, w_pad=2.4)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


def plot_detected_feature_panels(sample_df: pd.DataFrame) -> pd.DataFrame:
    stats_tables = []
    fig, axes = plt.subplots(2, 3, figsize=(22, 12), squeeze=False)
    for ax, (feature, title) in zip(axes.flat, EXON_FEATURES):
        stats_tables.append(plot_detected_boxplot(ax, sample_df, feature, title, "exon_only"))
    save_figure(fig, EXON_PDF, EXON_PNG)

    fig, axes = plt.subplots(2, 3, figsize=(22, 12), squeeze=False)
    for ax, (feature, title) in zip(axes.flat, INTRON_FEATURES):
        stats_tables.append(plot_detected_boxplot(ax, sample_df, feature, title, "intronic_span"))
    save_figure(fig, INTRON_PDF, INTRON_PNG)
    stats_df = pd.concat([df for df in stats_tables if df is not None and not df.empty], ignore_index=True) if stats_tables else pd.DataFrame()
    if not stats_df.empty:
        stats_df["pvalue_bh"] = bh_adjust(stats_df["pvalue"])
    return stats_df


def bh_adjust(pvalues: Iterable[object]) -> List[float]:
    values = [np.nan if pd.isna(p) else float(p) for p in pvalues]
    valid = [(idx, p) for idx, p in enumerate(values) if np.isfinite(p)]
    adjusted = [np.nan] * len(values)
    if not valid:
        return adjusted
    ranked = sorted(valid, key=lambda item: item[1])
    m = len(ranked)
    running = 1.0
    for reverse_rank, (idx, p) in enumerate(reversed(ranked), start=1):
        rank = m - reverse_rank + 1
        running = min(running, p * m / rank)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def plot_qc(block_df: pd.DataFrame, sample_df: pd.DataFrame, max_scatter_points: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), squeeze=False)
    order = [group for group in PLOT_GROUP_ORDER if group in set(block_df["plot_group"])]
    sns.boxplot(
        data=block_df,
        x="plot_group",
        y="intronic_fraction_of_genomic_span",
        hue="plot_group",
        order=order,
        hue_order=order,
        palette=METAPLOT_COLORS,
        showfliers=False,
        legend=False,
        ax=axes[0, 0],
    )
    axes[0, 0].set_title("Intronic fraction of ORF genomic span")
    axes[0, 0].set_xlabel("")
    axes[0, 0].tick_params(axis="x", rotation=25)

    scatter = sample_df[["ORF_id", "sample", "plot_group", "exon_mean_ORF_coverage", "full_span_mean_coverage"]].copy()
    x_col = "full_span_mean_coverage"
    x_label = "Full-span mean coverage"
    if OLD_COVERAGE_TABLE.exists():
        old = read_tsv(OLD_COVERAGE_TABLE, required=["ORF_id", "sample"])
        old_col = "mean_coverage" if "mean_coverage" in old.columns else "mean_ORF_coverage" if "mean_ORF_coverage" in old.columns else None
        if old_col is not None:
            old = old[["ORF_id", "sample", old_col]].rename(columns={old_col: "old_full_span_mean_coverage"})
            scatter = scatter.merge(old, on=["ORF_id", "sample"], how="left")
            x_col = "old_full_span_mean_coverage"
            x_label = "Previous full-span mean coverage"
    scatter[x_col] = pd.to_numeric(scatter[x_col], errors="coerce")
    scatter["exon_mean_ORF_coverage"] = pd.to_numeric(scatter["exon_mean_ORF_coverage"], errors="coerce")
    scatter = scatter.dropna(subset=[x_col, "exon_mean_ORF_coverage"])
    if len(scatter) > max_scatter_points:
        scatter = scatter.sample(max_scatter_points, random_state=42)
    if scatter.empty:
        axes[0, 1].axis("off")
        axes[0, 1].text(0.5, 0.5, "Full-span vs exon-only\nnot available", ha="center", va="center")
    else:
        sns.scatterplot(
            data=scatter,
            x=x_col,
            y="exon_mean_ORF_coverage",
            hue="plot_group",
            hue_order=PLOT_GROUP_ORDER,
            palette=METAPLOT_COLORS,
            s=8,
            alpha=0.35,
            linewidth=0,
            ax=axes[0, 1],
        )
        axes[0, 1].set_title("Full-span vs exon-only mean coverage")
        axes[0, 1].set_xlabel(x_label)
        axes[0, 1].set_ylabel("Exon-only mean ORF coverage")
        axes[0, 1].legend(frameon=False, fontsize=8)

    for ax, feature, title in [
        (axes[1, 0], "intron_mean_coverage", "Intronic-span mean coverage"),
        (axes[1, 1], "intron_exon_coverage_ratio", "Intron/exon coverage ratio"),
    ]:
        work = sample_df.copy()
        work[feature] = pd.to_numeric(work[feature], errors="coerce")
        work = work.dropna(subset=[feature])
        if work.empty:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{title}\nnot available", ha="center", va="center")
            continue
        sns.boxplot(
            data=work,
            x="plot_group",
            y=feature,
            hue="plot_group",
            order=order,
            hue_order=order,
            palette=METAPLOT_COLORS,
            showfliers=False,
            legend=False,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
    save_figure(fig, QC_PDF, QC_PNG)


def plot_metaplot(metaplot_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), squeeze=False)
    specs = [
        ("mean_CPM", "Raw CPM exon-only transcript-oriented metaplot", "Mean CPM"),
        ("mean_vector_normalized", "Vector-mean normalized exon-only metaplot", "Mean normalized coverage"),
    ]
    for ax, (value_col, title, ylabel) in zip(axes.flat, specs):
        for group in PLOT_GROUP_ORDER:
            for status in STATUS_ORDER:
                sub = metaplot_df[
                    metaplot_df["plot_group"].eq(group) & metaplot_df["detection_status"].eq(status)
                ].copy()
                if group == NEGATIVE_GROUP and status == "detected":
                    continue
                if sub.empty:
                    continue
                sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
                label = f"{group} ({status})"
                ax.plot(
                    sub["position"],
                    sub[value_col],
                    color=METAPLOT_COLORS.get(group, "black"),
                    linestyle=LINESTYLES[status],
                    linewidth=2.0,
                    label=label,
                )
        ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
        if value_col == "mean_vector_normalized":
            ax.axhline(1, color="gray", linestyle=":", linewidth=0.9)
        ax.set_title(title)
        ax.set_xlabel("Position relative to start codon")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=True, fontsize=8)
    save_figure(fig, METAPLOT_PDF, METAPLOT_PNG)


def qc_summary(block_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in PLOT_GROUP_ORDER:
        sub = block_df[block_df["plot_group"].eq(group)].copy()
        if sub.empty:
            continue
        rows.append(
            {
                "plot_group": group,
                "n_orfs": len(sub),
                "n_missing_gtf_transcript": int(sub["block_mapping_status"].astype(str).str.contains("missing_transcript").sum()),
                "n_start_coord_not_in_exons": int(sub["start_transcript_index"].isna().sum()),
                "median_genomic_span_length": float(pd.to_numeric(sub["genomic_span_length"], errors="coerce").median()),
                "median_exon_only_length": float(pd.to_numeric(sub["exon_only_length"], errors="coerce").median()),
                "median_intronic_span_length": float(pd.to_numeric(sub["intronic_span_length"], errors="coerce").median()),
                "fraction_with_intronic_span": float((pd.to_numeric(sub["intronic_span_length"], errors="coerce") > 0).mean()),
                "median_intronic_fraction_of_genomic_span": float(
                    pd.to_numeric(sub["intronic_fraction_of_genomic_span"], errors="coerce").median()
                ),
            }
        )
    return pd.DataFrame(rows)


def open_fasta_index():
    if not GENOME_FA.exists():
        return None, f"GENOME_FA not found: {GENOME_FA}"
    try:
        from Bio import SeqIO
    except ImportError:
        return None, "Biopython is not available; start codon sequence validation skipped."
    return SeqIO.index(str(GENOME_FA), "fasta"), ""


def resolve_fasta_chrom(index, chrom: str) -> Optional[str]:
    if index is None:
        return None
    if chrom in index:
        return chrom
    alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
    return alt if alt in index else None


def complement_base(base: str) -> str:
    return base.translate(REVCOMP)


def transcript_base(index, chrom: str, coord: int, strand: str) -> str:
    resolved = resolve_fasta_chrom(index, chrom)
    if resolved is None:
        return "N"
    base = str(index[resolved].seq[int(coord) : int(coord) + 1]).upper()
    return complement_base(base) if strand == "-" else base


def validation_position_summary(positions: Sequence[int], coords: Sequence[Optional[int]], chrom: str, n: int = 10) -> tuple[str, str]:
    pairs = [f"{pos}:{chrom}:{coord if coord is not None else 'NA'}" for pos, coord in zip(positions, coords)]
    return ";".join(pairs[:n]), ";".join(pairs[-n:])


def write_validation(block_df: pd.DataFrame, per_strand: int, seed: int, flank: int) -> None:
    fasta, fasta_note = open_fasta_index()
    rng = random.Random(seed)
    rows = []
    candidates = block_df[block_df["start_transcript_index"].notna()].copy()
    for strand in ["+", "-"]:
        strand_df = candidates[candidates["strand"].eq(strand)].copy()
        records = strand_df.to_dict("records")
        rng.shuffle(records)
        for row in records[:per_strand]:
            transcript_exons = parse_blocks(row["transcript_exon_blocks_genomic_order"])
            coords = transcript_oriented_coords(transcript_exons, strand)
            start_idx = int(row["start_transcript_index"])
            target_coords = [
                coords[start_idx + pos] if 0 <= start_idx + pos < len(coords) else None
                for pos in range(-flank, flank + 1)
            ]
            codon_coords = [
                coords[start_idx + offset] if 0 <= start_idx + offset < len(coords) else None
                for offset in range(3)
            ]
            if fasta is None or any(coord is None for coord in codon_coords):
                extracted = "NA"
            else:
                extracted = "".join(transcript_base(fasta, row["chr"], int(coord), strand) for coord in codon_coords)
            expected = normalize_codon(row.get("start_codon", ""))
            first_10, last_10 = validation_position_summary(range(-flank, flank + 1), target_coords, row["chr"])
            rows.append(
                {
                    "ORF_id": row["ORF_id"],
                    "transcript_id": row["transcript_id"],
                    "strand": strand,
                    "start_codon_expected": expected if expected else "NA",
                    "start_codon_extracted": extracted,
                    "start_codon_match": bool(expected and extracted != "NA" and expected == extracted),
                    "exon_blocks_used": row["orf_exon_blocks_transcript_order"],
                    "was_vector_reversed": strand == "-",
                    "first_10_positions_relative_to_start": first_10,
                    "last_10_positions_relative_to_start": last_10,
                    "validation_note": fasta_note,
                }
            )
    pd.DataFrame(rows).to_csv(VALIDATION_OUT, sep="\t", index=False, na_rep="NA")
    if fasta is not None:
        fasta.close()


def write_readme() -> None:
    README_OUT.write_text(
        f"""Exon/intron separated RNA coverage analysis

Purpose
- The previous RNA coverage calculation used the full genomic ORF span.
- For transcript-based ORFs, exon-only coverage is the main ORF body feature.
- Intronic-span coverage is reported separately as QC and as possible intron-retention or nascent RNA signal.

Inputs
- GENCODE GTF: {GTF}
- Positive ORF metadata: {POSITIVE_METADATA}
- CPAT-negative metadata: {NEGATIVE_METADATA}
- Positive sample-level detection table: {DETECTION_TABLE}
- CPM BigWigs: {INPUT_DIR / "bigwig/*.CPM.bw"}

Detection status
- Positive ORF-sample pairs are detected when the ORF_id, sample pair is present in pancreas.translated_orfs.sample_level.tsv.
- Positive ORF-sample pairs absent from that table are not_detected.
- CPAT-negative noncoding ORFs are always not_detected.

Coordinate and orientation rules
- GTF exon coordinates are converted from 1-based closed to 0-based half-open.
- ORF exon blocks are the intersection of ORF genomic [start0,end0) with the transcript exon blocks.
- Intronic-span blocks are bases inside [start0,end0) not covered by transcript exons.
- Start-centered metaplots use transcript-oriented exon coordinates, skipping introns at exon junctions.
- Plus-strand position 0 is start0.
- Minus-strand position 0 is end0-1, with the vector oriented in translation direction.

Feature definitions
- exon_mean_ORF_coverage: mean CPM over concatenated ORF exon blocks.
- exon_start_peak_coverage: mean CPM from -5 to +5 in the transcript-oriented start-centered vector.
- exon_start_peak_ratio: exon_start_peak_coverage divided by the mean of the -100 to +100 start-centered vector.
- exon_upstream_slope: linear slope over -100 to 0 in the transcript-oriented start-centered vector.
- exon_AUC_upstream: sum CPM over -100 to 0.
- exon_AUC_downstream: sum CPM over 0 to +100.
- intron_mean_coverage: mean CPM over intronic-span blocks inside the ORF genomic span.
- intron_total_coverage: sum CPM over intronic-span blocks.
- intron_covered_fraction: fraction of finite intronic-span bases with CPM > 0.
- intron_exon_coverage_ratio: intron_mean_coverage / exon_mean_ORF_coverage.
- full_span_mean_coverage: mean CPM over the old full genomic [start0,end0) span, kept only as QC.
- full_span_vs_exon_mean_ratio: full_span_mean_coverage / exon_mean_ORF_coverage.

Warning
- Previous full-span coverage may be confounded by intronic bases, especially for multi-exon ORFs or long introns.
""",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    ensure_out_dirs()
    archive_script()
    print("10_exon_intron_detected_vs_not_detected_coverage.py")
    print(f"BASE_DIR={BASE_DIR}")
    print(f"INPUT_DIR={INPUT_DIR}")
    print(f"NEG_DIR={NEG_DIR}")
    print(f"FIG_DIR={FIG_DIR}")
    print(f"OUT_DIR={OUT_DIR}")
    print(f"GTF={GTF}")
    print(f"GENOME_FA={GENOME_FA}")

    master = load_metadata()
    detected_pairs = load_detection_pairs()
    transcripts, gtf_warnings = load_gtf_exons(master["transcript_id"])
    block_df = build_block_table(master, transcripts, gtf_warnings)
    block_df.to_csv(BLOCKS_OUT, sep="\t", index=False, na_rep="NA")
    pd.DataFrame(gtf_warnings, columns=WARNING_COLUMNS).to_csv(WARNINGS_OUT, sep="\t", index=False, na_rep="NA")
    qc_summary(block_df).to_csv(QC_SUMMARY_OUT, sep="\t", index=False, na_rep="NA")

    bw_paths = discover_bigwigs(INPUT_DIR)
    sample_df, metaplot_df = compute_coverage_tables(block_df, bw_paths, detected_pairs, flank=args.flank)
    sample_df.to_csv(SAMPLE_OUT, sep="\t", index=False, na_rep="NA")
    orf_df = aggregate_orf_level(sample_df)
    orf_df.to_csv(ORF_OUT, sep="\t", index=False, na_rep="NA")
    metaplot_df.to_csv(METAPLOT_OUT, sep="\t", index=False, na_rep="NA")

    stats_df = plot_detected_feature_panels(sample_df)
    stats_df.to_csv(STATS_OUT, sep="\t", index=False, na_rep="NA")
    plot_qc(block_df, sample_df, args.max_scatter_points)
    plot_metaplot(metaplot_df)
    write_validation(block_df, args.validation_per_strand, args.random_seed, args.flank)
    write_readme()

    print(f"Wrote {BLOCKS_OUT}")
    print(f"Wrote {SAMPLE_OUT}")
    print(f"Wrote {ORF_OUT}")
    print(f"Wrote {STATS_OUT}")
    print(f"Wrote {METAPLOT_OUT}")
    print(f"Wrote {README_OUT}")
    print("10_exon_intron_detected_vs_not_detected_coverage.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
