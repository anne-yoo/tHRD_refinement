#!/usr/bin/env python3
"""Verify sample-wise mean ORF coverage calculations from CPM BigWigs."""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from fc_common import (
    FIG_DIR,
    INPUT_DIR,
    LOG_DIR,
    TABLE_DIR,
    archive_script,
    bw_values,
    discover_bigwigs,
    ensure_dirs,
    finite_mean,
    import_pybigwig,
    print_paths,
    read_master,
    read_tsv,
)


COVERAGE_TABLE = TABLE_DIR / "four_group_coverage_features.sample_level.tsv"
MASTER_TABLE = TABLE_DIR / "four_group_orf_metadata.tsv"
SAMPLE_SUMMARY_OUT = TABLE_DIR / "mean_orf_coverage_verification.sample_summary.tsv"
EXAMPLES_OUT = TABLE_DIR / "mean_orf_coverage_verification.example_orfs.tsv"
RECOMPUTED_OUT = TABLE_DIR / "mean_orf_coverage_verification.recomputed_orf_sample.tsv.gz"
LOG_OUT = LOG_DIR / "mean_orf_coverage_verification.txt"

CPM_METHOD = (
    "BigWigs were generated with deepTools bamCoverage --normalizeUsing CPM "
    "--binSize 1. CPM scales read/bin counts by mapped-read depth in millions; "
    "coverage values used here are the resulting per-base CPM BigWig values."
)
ORF_MEAN_FORMULA = (
    "mean_ORF_coverage = np.nanmean(pyBigWig.values(chrom, start0, end0)) "
    "over finite values from the strict 0-based half-open ORF interval [start0,end0)."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute mean ORF coverage from CPM BigWigs, summarize each sample, "
            "and write raw coverage vectors for example ORFs."
        )
    )
    parser.add_argument(
        "--example-samples",
        nargs="+",
        default=["GSM3395010", "GSM5099832"],
        help="Samples for which to write one raw ORF coverage vector example.",
    )
    parser.add_argument(
        "--example-max-length",
        type=int,
        default=300,
        help="Prefer example ORFs no longer than this many bases. Falls back to the shortest finite ORF if needed.",
    )
    parser.add_argument(
        "--skip-genome-mean",
        action="store_true",
        help="Skip weighted mean coverage across all BigWig intervals if that pass is too slow.",
    )
    parser.add_argument(
        "--write-recomputed-table",
        action="store_true",
        help="Also write every recomputed ORF-sample mean to a gzipped TSV.",
    )
    return parser.parse_args()


def numeric_metadata(master: pd.DataFrame) -> pd.DataFrame:
    required = ["ORF_id", "plot_group", "chr", "start0", "end0", "strand"]
    missing = [col for col in required if col not in master.columns]
    if missing:
        raise SystemExit(f"Missing required column(s) in {MASTER_TABLE}: {', '.join(missing)}")
    out = master.copy()
    out["start0"] = pd.to_numeric(out["start0"], errors="coerce")
    out["end0"] = pd.to_numeric(out["end0"], errors="coerce")
    out = out.dropna(subset=["ORF_id", "chr", "start0", "end0"])
    out["start0"] = out["start0"].astype(int)
    out["end0"] = out["end0"].astype(int)
    out["ORF_length"] = out["end0"] - out["start0"]
    out = out[out["ORF_length"].gt(0)].copy()
    return out


def finite_median(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.nanmedian(arr)) if np.isfinite(arr).any() else np.nan


def format_vector(values: np.ndarray) -> str:
    out = []
    for value in np.asarray(values, dtype=float):
        if np.isfinite(value):
            out.append(f"{float(value):.8g}")
        else:
            out.append("NA")
    return ",".join(out)


def bigwig_interval_weighted_mean(bw) -> tuple[float, int, int]:
    weighted_sum = 0.0
    represented_bp = 0
    interval_count = 0
    for chrom in sorted(bw.chroms()):
        intervals = bw.intervals(chrom)
        if not intervals:
            continue
        interval_count += len(intervals)
        for start, end, value in intervals:
            if value is None or not np.isfinite(float(value)):
                continue
            width = int(end) - int(start)
            if width <= 0:
                continue
            weighted_sum += width * float(value)
            represented_bp += width
    mean_value = weighted_sum / represented_bp if represented_bp else np.nan
    return float(mean_value), int(represented_bp), int(interval_count)


def existing_summary() -> tuple[pd.DataFrame, Optional[str]]:
    if not COVERAGE_TABLE.exists():
        return pd.DataFrame(), None
    df = read_tsv(COVERAGE_TABLE, required=["ORF_id", "sample"])
    coverage_col = None
    for candidate in ["mean_ORF_coverage", "mean_coverage"]:
        if candidate in df.columns:
            coverage_col = candidate
            break
    if coverage_col is None:
        return pd.DataFrame(), None
    df[coverage_col] = pd.to_numeric(df[coverage_col], errors="coerce")
    summary = (
        df.groupby("sample", dropna=False)[coverage_col]
        .agg(
            existing_n_orf_sample_pairs="count",
            existing_median_ORF_coverage="median",
            existing_mean_ORF_coverage="mean",
        )
        .reset_index()
    )
    return summary, coverage_col


def recompute_orf_means(master: pd.DataFrame, bw_paths: Dict[str, Path], pyBigWig) -> pd.DataFrame:
    rows: List[dict] = []
    handles = {sample: pyBigWig.open(str(path)) for sample, path in bw_paths.items()}
    try:
        for sample, bw in handles.items():
            for row in master.itertuples(index=False):
                values = bw_values(bw, str(row.chr), int(row.start0), int(row.end0))
                rows.append(
                    {
                        "sample": sample,
                        "ORF_id": str(row.ORF_id),
                        "plot_group": str(row.plot_group),
                        "chr": str(row.chr),
                        "start0": int(row.start0),
                        "end0": int(row.end0),
                        "strand": str(row.strand),
                        "ORF_length": int(row.ORF_length),
                        "finite_base_count": int(np.isfinite(values).sum()),
                        "recomputed_mean_ORF_coverage": finite_mean(values),
                    }
                )
    finally:
        for handle in handles.values():
            handle.close()
    return pd.DataFrame(rows)


def summarize_recomputed(recomputed: pd.DataFrame) -> pd.DataFrame:
    if recomputed.empty:
        return pd.DataFrame()
    work = recomputed.copy()
    work["recomputed_mean_ORF_coverage"] = pd.to_numeric(work["recomputed_mean_ORF_coverage"], errors="coerce")
    summary = (
        work.groupby("sample", dropna=False)
        .agg(
            recomputed_n_orf_sample_pairs=("recomputed_mean_ORF_coverage", "count"),
            recomputed_median_ORF_coverage=("recomputed_mean_ORF_coverage", "median"),
            recomputed_mean_ORF_coverage=("recomputed_mean_ORF_coverage", "mean"),
            median_finite_base_count=("finite_base_count", "median"),
            min_finite_base_count=("finite_base_count", "min"),
            max_finite_base_count=("finite_base_count", "max"),
        )
        .reset_index()
    )
    return summary


def add_existing_comparison(summary: pd.DataFrame, recomputed: pd.DataFrame, existing_col: Optional[str]) -> pd.DataFrame:
    if existing_col is None or not COVERAGE_TABLE.exists() or recomputed.empty:
        summary["existing_vs_recomputed_max_abs_diff"] = np.nan
        summary["existing_vs_recomputed_median_abs_diff"] = np.nan
        return summary
    existing = read_tsv(COVERAGE_TABLE, required=["ORF_id", "sample"])
    if existing_col not in existing.columns:
        return summary
    existing = existing[["ORF_id", "sample", existing_col]].copy()
    existing[existing_col] = pd.to_numeric(existing[existing_col], errors="coerce")
    merged = existing.merge(
        recomputed[["ORF_id", "sample", "recomputed_mean_ORF_coverage"]],
        on=["ORF_id", "sample"],
        how="inner",
    )
    merged["abs_diff"] = (merged[existing_col] - merged["recomputed_mean_ORF_coverage"]).abs()
    diffs = (
        merged.groupby("sample")["abs_diff"]
        .agg(existing_vs_recomputed_max_abs_diff="max", existing_vs_recomputed_median_abs_diff="median")
        .reset_index()
    )
    return summary.merge(diffs, on="sample", how="left")


def add_bigwig_genome_means(summary: pd.DataFrame, bw_paths: Dict[str, Path], pyBigWig, skip: bool) -> pd.DataFrame:
    rows = []
    if skip:
        for sample, path in bw_paths.items():
            rows.append(
                {
                    "sample": sample,
                    "bigwig_path": str(path),
                    "bigwig_represented_position_mean_CPM": np.nan,
                    "bigwig_represented_bp": np.nan,
                    "bigwig_interval_count": np.nan,
                    "bigwig_genome_mean_note": "skipped_by_--skip-genome-mean",
                }
            )
    else:
        for sample, path in bw_paths.items():
            bw = pyBigWig.open(str(path))
            try:
                mean_value, represented_bp, interval_count = bigwig_interval_weighted_mean(bw)
            finally:
                bw.close()
            rows.append(
                {
                    "sample": sample,
                    "bigwig_path": str(path),
                    "bigwig_represented_position_mean_CPM": mean_value,
                    "bigwig_represented_bp": represented_bp,
                    "bigwig_interval_count": interval_count,
                    "bigwig_genome_mean_note": "weighted_by_interval_width_over_positions_stored_in_BigWig",
                }
            )
    return summary.merge(pd.DataFrame(rows), on="sample", how="outer")


def choose_example_for_sample(
    master: pd.DataFrame,
    bw,
    sample: str,
    bw_path: Path,
    max_length: int,
) -> Optional[dict]:
    candidates = master.sort_values(["ORF_length", "ORF_id"]).copy()
    passes = [
        (candidates[candidates["ORF_length"].between(30, max_length)], True),
        (candidates[candidates["ORF_length"].le(max_length)], True),
        (candidates, True),
        (candidates[candidates["ORF_length"].between(30, max_length)], False),
        (candidates[candidates["ORF_length"].le(max_length)], False),
        (candidates, False),
    ]
    for sub, require_positive_mean in passes:
        for row in sub.itertuples(index=False):
            values = bw_values(bw, str(row.chr), int(row.start0), int(row.end0))
            mean_value = finite_mean(values)
            if not np.isfinite(mean_value):
                continue
            if require_positive_mean and mean_value <= 0:
                continue
            return {
                "sample": sample,
                "bigwig_path": str(bw_path),
                "ORF_id": str(row.ORF_id),
                "plot_group": str(row.plot_group),
                "chr": str(row.chr),
                "start0": int(row.start0),
                "end0": int(row.end0),
                "strand": str(row.strand),
                "ORF_length": int(row.ORF_length),
                "finite_base_count": int(np.isfinite(values).sum()),
                "raw_coverage_vector_CPM": format_vector(values),
                "computed_mean_ORF_coverage": mean_value,
                "formula": ORF_MEAN_FORMULA,
            }
    return None


def write_examples(master: pd.DataFrame, bw_paths: Dict[str, Path], pyBigWig, samples: List[str], max_length: int) -> pd.DataFrame:
    rows = []
    for sample in samples:
        path = bw_paths.get(sample)
        if path is None:
            rows.append({"sample": sample, "note": "No CPM BigWig discovered for this sample"})
            continue
        bw = pyBigWig.open(str(path))
        try:
            row = choose_example_for_sample(master, bw, sample, path, max_length)
        finally:
            bw.close()
        rows.append(row if row is not None else {"sample": sample, "note": "No finite ORF coverage vector found"})
    examples = pd.DataFrame(rows)
    examples.to_csv(EXAMPLES_OUT, sep="\t", index=False, na_rep="NA")
    return examples


def write_log(summary: pd.DataFrame, examples: pd.DataFrame, existing_col: Optional[str]) -> None:
    lines = [
        "Mean ORF coverage verification",
        f"INPUT_DIR={INPUT_DIR}",
        f"FIG_DIR={FIG_DIR}",
        f"master_table={MASTER_TABLE}",
        f"coverage_table={COVERAGE_TABLE if COVERAGE_TABLE.exists() else 'missing'}",
        f"existing_coverage_column={existing_col or 'NA'}",
        "",
        "CPM normalization method:",
        CPM_METHOD,
        "",
        "ORF mean coverage formula:",
        ORF_MEAN_FORMULA,
        "",
        "Formula consistency:",
        "The same coordinate extraction and np.nanmean formula is used for every sample.",
        "Only the input BigWig path differs by sample.",
        "",
        "Sample summary:",
        summary.to_string(index=False),
        "",
        "Example ORFs:",
        examples.drop(columns=["raw_coverage_vector_CPM"], errors="ignore").to_string(index=False),
        "",
        f"Wrote sample summary: {SAMPLE_SUMMARY_OUT}",
        f"Wrote example vectors: {EXAMPLES_OUT}",
    ]
    LOG_OUT.write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    ensure_dirs()
    archive_script(__file__)
    print_paths(
        "09_verify_mean_orf_coverage.py",
        inputs=[MASTER_TABLE, COVERAGE_TABLE, INPUT_DIR / "bigwig/*.CPM.bw"],
        outputs=[SAMPLE_SUMMARY_OUT, EXAMPLES_OUT, RECOMPUTED_OUT, LOG_OUT],
    )

    pyBigWig = import_pybigwig()
    bw_paths = discover_bigwigs()
    master = numeric_metadata(read_master())
    existing, existing_col = existing_summary()

    recomputed = recompute_orf_means(master, bw_paths, pyBigWig)
    if args.write_recomputed_table:
        with gzip.open(RECOMPUTED_OUT, "wt") as handle:
            recomputed.to_csv(handle, sep="\t", index=False, na_rep="NA")

    summary = summarize_recomputed(recomputed)
    if not existing.empty:
        summary = summary.merge(existing, on="sample", how="outer")
    summary = add_existing_comparison(summary, recomputed, existing_col)
    summary = add_bigwig_genome_means(summary, bw_paths, pyBigWig, args.skip_genome_mean)
    summary["CPM_normalization_method"] = CPM_METHOD
    summary["ORF_mean_formula"] = ORF_MEAN_FORMULA
    summary["same_formula_used_for_all_samples"] = True
    summary = summary.sort_values("sample").reset_index(drop=True)
    summary.to_csv(SAMPLE_SUMMARY_OUT, sep="\t", index=False, na_rep="NA")

    examples = write_examples(master, bw_paths, pyBigWig, args.example_samples, args.example_max_length)
    write_log(summary, examples, existing_col)

    print(summary.to_string(index=False))
    print()
    print(examples.drop(columns=["raw_coverage_vector_CPM"], errors="ignore").to_string(index=False))
    print(f"Wrote {SAMPLE_SUMMARY_OUT}")
    print(f"Wrote {EXAMPLES_OUT}")
    if args.write_recomputed_table:
        print(f"Wrote {RECOMPUTED_OUT}")
    print(f"Wrote {LOG_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
