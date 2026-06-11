#!/usr/bin/env python3
"""Coverage helpers for sample-specific ORF RNA coverage analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from tc_common import (
    FORWARD_SENSE_SAMPLES,
    INPUT_DIR,
    UNSTRANDED_SAMPLES,
    clean_text,
    read_tsv,
)


POSITIONS = np.arange(-100, 101, dtype=int)
POS_COLS = [f"pos_{pos}" for pos in POSITIONS]


def import_pybigwig():
    try:
        import pyBigWig
    except ImportError as exc:
        raise SystemExit(
            "pyBigWig is required for RNA coverage analysis. "
            "Activate the intended environment or install pyBigWig."
        ) from exc
    return pyBigWig


def discover_bigwigs(input_dir: Path = INPUT_DIR) -> Dict[str, Path]:
    bw_dir = input_dir / "bigwig"
    paths: Dict[str, Path] = {}
    for sample in UNSTRANDED_SAMPLES:
        path = bw_dir / f"{sample}.unstranded.CPM.bw"
        if path.exists():
            paths[sample] = path
    for sample in FORWARD_SENSE_SAMPLES:
        path = bw_dir / f"{sample}.sense.CPM.bw"
        if path.exists():
            paths[sample] = path
    if not paths:
        raise SystemExit(f"No expected BigWig files found under {bw_dir}")
    return paths


def open_bigwigs(paths: Dict[str, Path]) -> dict:
    pyBigWig = import_pybigwig()
    return {sample: pyBigWig.open(str(path)) for sample, path in paths.items()}


def close_bigwigs(handles: dict) -> None:
    for handle in handles.values():
        handle.close()


def resolve_bw_chrom(chroms: dict, chrom: str) -> Optional[str]:
    if chrom in chroms:
        return chrom
    alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
    return alt if alt in chroms else None


def bw_values(bw, chrom: str, start: int, end: int) -> np.ndarray:
    length = max(0, end - start)
    if length == 0:
        return np.array([], dtype=float)
    chroms = bw.chroms()
    resolved = resolve_bw_chrom(chroms, chrom)
    out = np.full(length, np.nan, dtype=float)
    if resolved is None:
        return out
    chrom_len = int(chroms[resolved])
    clipped_start = max(0, start)
    clipped_end = min(end, chrom_len)
    if clipped_end <= clipped_start:
        return out
    values = bw.values(resolved, clipped_start, clipped_end, numpy=True)
    values = np.asarray(values, dtype=float)
    offset = clipped_start - start
    out[offset : offset + len(values)] = values
    return out


def start_centered_window(row: pd.Series) -> Tuple[str, int, int, bool]:
    chrom = str(row["chr"])
    start0 = int(row["start0"])
    end0 = int(row["end0"])
    strand = str(row["strand"])
    if strand == "+":
        return chrom, start0 - 100, start0 + 101, False
    if strand == "-":
        return chrom, end0 - 101, end0 + 100, True
    raise ValueError(f"Unsupported strand for {row.get('ORF_id', '')}: {strand}")


def start_centered_vector(bw, row: pd.Series) -> np.ndarray:
    chrom, start, end, reverse = start_centered_window(row)
    vector = bw_values(bw, chrom, start, end)
    if reverse:
        vector = vector[::-1]
    return vector


def vector_mean_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    finite = vector[np.isfinite(vector)]
    if len(finite) == 0:
        return np.full_like(vector, np.nan, dtype=float)
    scale = float(np.nanmean(finite))
    if not np.isfinite(scale) or scale <= 0:
        return np.full_like(vector, np.nan, dtype=float)
    return vector / scale


def nanmean_stack(vectors: Iterable[np.ndarray]) -> Optional[np.ndarray]:
    vectors = [np.asarray(vector, dtype=float) for vector in vectors]
    if not vectors:
        return None
    return np.nanmean(np.vstack(vectors), axis=0)


def vector_to_row(prefix: str, vector: np.ndarray) -> dict:
    return {f"{prefix}{pos}": value for pos, value in zip(POSITIONS, vector)}


def read_sample_level_detection() -> pd.DataFrame:
    path = INPUT_DIR / "tables" / "pancreas.translated_orfs.sample_level.tsv"
    required = ["ORF_id", "sample"]
    return read_tsv(path, required=required)


def detection_pairs(sample_level: pd.DataFrame) -> set[tuple[str, str]]:
    return set(
        zip(
            sample_level["ORF_id"].astype(str),
            sample_level["sample"].astype(str),
        )
    )


def detected_orfs_by_sample(sample_level: pd.DataFrame) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for sample, sub in sample_level.groupby("sample"):
        out[str(sample)] = set(sub["ORF_id"].astype(str))
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


def coverage_features_from_vector(vector: np.ndarray) -> dict:
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


def status_label(detected: int) -> str:
    return "detected" if int(detected) == 1 else "not_detected"

