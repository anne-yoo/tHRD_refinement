#!/usr/bin/env python3
"""Compute and compare sample-specific RNA coverage across four ORF groups."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fc_common import (
    INPUT_DIR,
    LOG_DIR,
    POSITIONS,
    POS_COLS,
    TABLE_DIR,
    archive_script,
    boxplot_four_group,
    bw_values,
    coverage_features_from_vector,
    discover_bigwigs,
    ensure_dirs,
    finite_mean,
    import_pybigwig,
    print_paths,
    read_master,
    read_tsv,
    save_figure,
    start_centered_window,
    vector_mean_normalize,
)


POS_SAMPLE_COV = INPUT_DIR / "tables" / "orf_rna_coverage_features.sample_level.tsv"
POS_ORF_COV = INPUT_DIR / "tables" / "orf_rna_coverage_features.orf_level.tsv"
SAMPLE_OUT = TABLE_DIR / "four_group_coverage_features.sample_level.tsv"
ORF_OUT = TABLE_DIR / "four_group_coverage_features.orf_level.tsv"
MATRIX_OUT = TABLE_DIR / "four_group_start_centered_coverage_matrix.tsv"
NORM_MATRIX_OUT = TABLE_DIR / "four_group_start_centered_coverage_matrix.vector_mean_normalized.tsv"
STATS_OUT = TABLE_DIR / "coverage_four_group_statistics.tsv"
LOG_OUT = LOG_DIR / "coverage_extraction_inputs.log"

MAIN_FEATURES = [
    ("mean_coverage", "Mean ORF coverage", "Mean ORF coverage"),
    ("orf_upstream_coverage_ratio", "ORF/upstream coverage ratio", "ORF/upstream coverage ratio"),
    ("start_peak_coverage", "Start peak coverage", "Start peak coverage"),
    ("start_peak_ratio", "Start peak / window mean", "Start peak ratio"),
    ("upstream_slope", "Slope", "Upstream slope"),
    ("AUC_upstream", "AUC", "AUC upstream"),
    ("AUC_downstream", "AUC", "AUC downstream"),
    ("sample_specific_variance_mean_coverage", "Variance", "Sample-specific variance"),
]


def detected_pairs() -> set[tuple[str, str]]:
    sample_level = INPUT_DIR / "tables" / "pancreas.translated_orfs.sample_level.tsv"
    if not sample_level.exists():
        return set()
    df = read_tsv(sample_level, required=["ORF_id", "sample"])
    return set(zip(df["ORF_id"].astype(str), df["sample"].astype(str)))


def row_windows(row: pd.Series) -> Dict[str, object]:
    chrom = str(row["chr"])
    start0 = int(float(row["start0"]))
    end0 = int(float(row["end0"]))
    strand = str(row["strand"])
    if strand == "+":
        upstream = (start0 - 200, start0)
        downstream = (end0, end0 + 200)
    else:
        upstream = (end0, end0 + 200)
        downstream = (start0 - 200, start0)
    centered_chrom, centered_start, centered_end, reverse = start_centered_window(row, flank=100)
    return {
        "chrom": chrom,
        "orf": (start0, end0),
        "upstream": upstream,
        "downstream": downstream,
        "centered": (centered_chrom, centered_start, centered_end, reverse),
    }


def build_coverage_tables(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pyBigWig = import_pybigwig()
    bw_paths = discover_bigwigs()
    with LOG_OUT.open("w") as log:
        for sample, path in bw_paths.items():
            log.write(f"{sample}\t{path}\n")
    handles = {sample: pyBigWig.open(str(path)) for sample, path in bw_paths.items()}
    detected = detected_pairs()
    sample_rows: List[dict] = []
    matrix_rows: List[dict] = []
    try:
        for idx, row in master.iterrows():
            if idx and idx % 1000 == 0:
                print(f"Processed {idx}/{len(master)} ORF(s)")
            windows = row_windows(row)
            sample_vectors = []
            for sample, bw in handles.items():
                orf_values = bw_values(bw, windows["chrom"], *windows["orf"])
                upstream_values = bw_values(bw, windows["chrom"], *windows["upstream"])
                downstream_values = bw_values(bw, windows["chrom"], *windows["downstream"])
                chrom, centered_start, centered_end, reverse = windows["centered"]
                vector = bw_values(bw, chrom, centered_start, centered_end)
                if reverse:
                    vector = vector[::-1]
                sample_vectors.append(vector)
                mean_coverage = finite_mean(orf_values)
                upstream_mean = finite_mean(upstream_values)
                downstream_mean = finite_mean(downstream_values)
                features = coverage_features_from_vector(vector)
                features.update(
                    {
                        "ORF_id": row["ORF_id"],
                        "sample": sample,
                        "plot_group": row["plot_group"],
                        "transcript_id": row["transcript_id"],
                        "mean_coverage": mean_coverage,
                        "upstream_200nt_mean_coverage": upstream_mean,
                        "downstream_200nt_mean_coverage": downstream_mean,
                        "orf_upstream_coverage_ratio": mean_coverage / upstream_mean if upstream_mean and upstream_mean > 0 else np.nan,
                        "orf_downstream_coverage_ratio": mean_coverage / downstream_mean if downstream_mean and downstream_mean > 0 else np.nan,
                        "detected_in_sample": int((row["ORF_id"], sample) in detected) if row["plot_group"] != "CPAT-negative noncoding" else 0,
                    }
                )
                sample_rows.append(features)
            mean_vector = np.nanmean(np.vstack(sample_vectors), axis=0) if sample_vectors else np.full(len(POSITIONS), np.nan)
            matrix_row = {
                "ORF_id": row["ORF_id"],
                "plot_group": row["plot_group"],
                "n_samples_used": len(sample_vectors),
            }
            matrix_row.update({col: value for col, value in zip(POS_COLS, mean_vector)})
            matrix_rows.append(matrix_row)
    finally:
        for handle in handles.values():
            handle.close()

    sample_df = pd.DataFrame(sample_rows)
    agg_cols = [
        "mean_coverage",
        "upstream_200nt_mean_coverage",
        "downstream_200nt_mean_coverage",
        "orf_upstream_coverage_ratio",
        "orf_downstream_coverage_ratio",
        "mean_window_coverage",
        "mean_upstream_100",
        "mean_downstream_100",
        "start_peak_coverage",
        "start_peak_ratio",
        "upstream_slope",
        "downstream_slope",
        "AUC_upstream",
        "AUC_downstream",
        "asymmetry_ratio",
    ]
    orf_df = sample_df.groupby(["ORF_id", "plot_group", "transcript_id"], as_index=False)[agg_cols].mean(numeric_only=True)
    variance = sample_df.groupby("ORF_id")["mean_coverage"].var().rename("sample_specific_variance_mean_coverage")
    detected_n = sample_df.groupby("ORF_id")["detected_in_sample"].sum().rename("n_detected_samples")
    orf_df = orf_df.merge(variance, on="ORF_id", how="left").merge(detected_n, on="ORF_id", how="left")

    matrix_df = pd.DataFrame(matrix_rows)
    norm_df = matrix_df.copy()
    if not matrix_df.empty:
        values = matrix_df[POS_COLS].apply(pd.to_numeric, errors="coerce")
        scale = values.mean(axis=1, skipna=True)
        for col in POS_COLS:
            norm_df[col] = np.where(scale.gt(0), values[col] / scale, np.nan)
        norm_df["start_centered_vector_mean_for_normalization"] = scale
    return sample_df, orf_df, matrix_df, norm_df


def plot_metaplot(matrix_df: pd.DataFrame, value_label: str, title: str, ax) -> None:
    if matrix_df.empty:
        ax.axis("off")
        return
    rows = []
    for group, sub in matrix_df.groupby("plot_group"):
        values = sub[POS_COLS].apply(pd.to_numeric, errors="coerce")
        means = values.mean(axis=0, skipna=True)
        for pos, value in zip(POSITIONS, means):
            rows.append({"plot_group": group, "position": pos, value_label: value})
    long_df = pd.DataFrame(rows)
    sns.lineplot(data=long_df, x="position", y=value_label, hue="plot_group", hue_order=["Canonical", "AUG noncanonical", "nonAUG noncanonical", "CPAT-negative noncoding"], ax=ax)
    ax.axvline(0, color="black", ls="--", lw=0.8)
    ax.set_title(title)
    ax.set_ylabel(value_label)


def main() -> int:
    ensure_dirs()
    archive_script(__file__)
    print_paths(
        "03_sample_specific_coverage_comparison.py",
        inputs=[TABLE_DIR / "four_group_orf_metadata.tsv", POS_SAMPLE_COV, POS_ORF_COV, INPUT_DIR / "bigwig/*.CPM.bw"],
        outputs=[SAMPLE_OUT, ORF_OUT, MATRIX_OUT, NORM_MATRIX_OUT, STATS_OUT],
    )
    master = read_master()
    sample_df, orf_df, matrix_df, norm_df = build_coverage_tables(master)
    sample_df.to_csv(SAMPLE_OUT, sep="\t", index=False, na_rep="NA")
    orf_df.to_csv(ORF_OUT, sep="\t", index=False, na_rep="NA")
    matrix_df.to_csv(MATRIX_OUT, sep="\t", index=False, na_rep="NA")
    norm_df.to_csv(NORM_MATRIX_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {SAMPLE_OUT}")
    print(f"Wrote {ORF_OUT}")

    stats_tables = []
    fig, axes = plt.subplots(4, 2, figsize=(17, 25), squeeze=False)
    for ax, (feature, ylabel, title) in zip(axes.flat, MAIN_FEATURES):
        stats_tables.append(boxplot_four_group(ax, orf_df, feature, y_label=ylabel, title=title, annotate=True))
    stats_df = pd.concat([table for table in stats_tables if table is not None and not table.empty], ignore_index=True, sort=False)
    stats_df.to_csv(STATS_OUT, sep="\t", index=False, na_rep="NA")
    save_figure(fig, "Fig2_coverage_four_group.pdf")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), squeeze=False)
    for ax, feature, ylabel, title in [
        (axes[0, 0], "mean_coverage", "Mean ORF coverage", "Sample-wise mean ORF coverage"),
        (axes[0, 1], "start_peak_coverage", "Start peak coverage", "Sample-wise start peak coverage"),
        (axes[1, 0], "orf_upstream_coverage_ratio", "ORF/upstream ratio", "Sample-wise ORF/upstream ratio"),
        (axes[1, 1], "start_peak_ratio", "Start peak ratio", "Sample-wise start peak ratio"),
    ]:
        work = sample_df.copy()
        work[feature] = pd.to_numeric(work[feature], errors="coerce")
        sns.boxplot(data=work, x="sample", y=feature, hue="plot_group", showfliers=False, ax=ax)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=35)
    save_figure(fig, "Fig3_coverage_samplewise_four_group.pdf")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_metaplot(matrix_df, "mean_CPM", "Start-centered raw CPM metaplot", axes[0])
    plot_metaplot(norm_df, "vector_mean_normalized_CPM", "Start-centered vector-mean normalized metaplot", axes[1])
    save_figure(fig, "Fig3_coverage_metaplot_four_group.pdf")

    print("03_sample_specific_coverage_comparison.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
