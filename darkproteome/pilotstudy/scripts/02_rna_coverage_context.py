#!/usr/bin/env python3
"""Extract RNA coverage features from available CPM BigWigs."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tc_common import (
    ALL_SAMPLES,
    FIG_DIR,
    FORWARD_SENSE_SAMPLES,
    GROUP_ORDER,
    INPUT_DIR,
    PAIRWISE_GROUPS,
    TABLE_DIR,
    UNSTRANDED_SAMPLES,
    archive_script,
    boxplot_with_points,
    display_group,
    ensure_fig_dirs,
    mannwhitney_stats,
    parse_detected_samples,
    print_paths,
    read_groups,
    save_figure,
)


SAMPLE_OUT = TABLE_DIR / "orf_rna_coverage_features.sample_level.tsv"
ORF_OUT = TABLE_DIR / "orf_rna_coverage_features.orf_level.tsv"
MATRIX_OUT = TABLE_DIR / "start_centered_coverage_matrix.tsv"
NORMALIZED_MATRIX_OUT = TABLE_DIR / "start_centered_coverage_matrix.ORFmean_normalized.tsv"
ORF_BODY_MEAN_NORMALIZED_MATRIX_OUT = TABLE_DIR / "start_centered_coverage_matrix.ORFbody_mean_normalized.tsv"
NORMALIZATION_EXAMPLES_OUT = TABLE_DIR / "start_centered_coverage_normalization_examples.tsv"
STATS_OUT = TABLE_DIR / "rna_coverage_statistics.tsv"


def import_pybigwig():
    try:
        import pyBigWig
    except ImportError as exc:
        raise SystemExit(
            "pyBigWig is required for 02_rna_coverage_context.py. "
            "Activate the intended environment or install pyBigWig."
        ) from exc
    return pyBigWig


def discover_bigwigs() -> dict:
    bw_dir = INPUT_DIR / "bigwig"
    paths = {}
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


def resolve_bw_chrom(chroms: dict, chrom: str) -> str | None:
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


def coverage_metrics(values: np.ndarray) -> dict:
    clean = values[np.isfinite(values)]
    if len(clean) == 0:
        return {
            "mean_coverage": np.nan,
            "median_coverage": np.nan,
            "covered_fraction": np.nan,
            "coverage_cv": np.nan,
            "zero_fraction": np.nan,
        }
    mean = float(np.nanmean(clean))
    std = float(np.nanstd(clean))
    return {
        "mean_coverage": mean,
        "median_coverage": float(np.nanmedian(clean)),
        "covered_fraction": float(np.mean(clean > 0)),
        "coverage_cv": std / mean if mean > 0 else np.nan,
        "zero_fraction": float(np.mean(clean <= 0)),
    }


def normalize_position_columns_by_scale(
    df: pd.DataFrame,
    pos_cols: list[str],
    scale: pd.Series,
) -> pd.DataFrame:
    out = df.copy()
    valid_scale = pd.to_numeric(scale, errors="coerce").gt(0)
    scale = pd.to_numeric(scale, errors="coerce")
    for col in pos_cols:
        values = pd.to_numeric(out[col], errors="coerce")
        out[col] = np.where(valid_scale, values / scale, np.nan)
    return out


def write_normalization_examples(
    matrix_df: pd.DataFrame,
    vector_norm_df: pd.DataFrame,
    body_norm_df: pd.DataFrame,
    pos_cols: list[str],
) -> None:
    example_positions = [pos for pos in ["pos_-100", "pos_-50", "pos_0", "pos_50", "pos_100"] if pos in pos_cols]
    rows = []
    groups_seen = set()
    for _, raw_row in matrix_df.iterrows():
        group = raw_row["group"]
        if group in groups_seen:
            continue
        groups_seen.add(group)
        orf_id = raw_row["ORF_id"]
        vector_row = vector_norm_df[vector_norm_df["ORF_id"].eq(orf_id)].iloc[0]
        body_row = body_norm_df[body_norm_df["ORF_id"].eq(orf_id)].iloc[0]
        example = {
            "ORF_id": orf_id,
            "group": group,
            "vector_mean_denominator": vector_row.get("start_centered_vector_mean_for_normalization", np.nan),
            "ORF_body_mean_denominator": body_row.get("orf_mean_coverage_for_normalization", np.nan),
        }
        for pos in example_positions:
            example[f"{pos}_raw_CPM"] = raw_row[pos]
            example[f"{pos}_vector_mean_normalized"] = vector_row[pos]
            example[f"{pos}_ORF_body_mean_normalized"] = body_row[pos]
        rows.append(example)
        if len(groups_seen) >= 3:
            break
    pd.DataFrame(rows).to_csv(NORMALIZATION_EXAMPLES_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {NORMALIZATION_EXAMPLES_OUT}")


def row_windows(row: pd.Series) -> dict:
    chrom = str(row["chr"])
    start0 = int(row["start0"])
    end0 = int(row["end0"])
    strand = str(row["strand"])
    if strand == "+":
        upstream = (start0 - 200, start0)
        downstream = (end0, end0 + 200)
        centered = (start0 - 100, start0 + 101)
        reverse_center = False
    else:
        upstream = (end0, end0 + 200)
        downstream = (start0 - 200, start0)
        centered = (end0 - 101, end0 + 100)
        reverse_center = True
    return {
        "chrom": chrom,
        "orf": (start0, end0),
        "upstream": upstream,
        "downstream": downstream,
        "centered": centered,
        "reverse_center": reverse_center,
    }


def main() -> int:
    ensure_fig_dirs()
    archive_script(__file__)
    print_paths(
        script_name=Path(__file__).name,
        inputs=[INPUT_DIR / "tables" / "orf_groups.combined_metadata.tsv", INPUT_DIR / "bigwig"],
        outputs=[
            SAMPLE_OUT,
            ORF_OUT,
            MATRIX_OUT,
            NORMALIZED_MATRIX_OUT,
            ORF_BODY_MEAN_NORMALIZED_MATRIX_OUT,
            NORMALIZATION_EXAMPLES_OUT,
            STATS_OUT,
        ],
    )

    pyBigWig = import_pybigwig()
    groups = read_groups()
    bigwig_paths = discover_bigwigs()
    handles = {sample: pyBigWig.open(str(path)) for sample, path in bigwig_paths.items()}

    sample_rows = []
    matrix_rows = []
    try:
        for idx, row in groups.iterrows():
            if idx and idx % 1000 == 0:
                print(f"Processed {idx}/{len(groups)} ORF(s)")
            windows = row_windows(row)
            detected = parse_detected_samples(row.get("detected_samples", ""))
            aggregate_samples = [sample for sample in detected if sample in handles] if detected else list(handles)
            if not aggregate_samples:
                aggregate_samples = list(handles)
            centered_vectors = []
            for sample, bw in handles.items():
                orf_values = bw_values(bw, windows["chrom"], *windows["orf"])
                upstream_values = bw_values(bw, windows["chrom"], *windows["upstream"])
                downstream_values = bw_values(bw, windows["chrom"], *windows["downstream"])
                centered_values = bw_values(bw, windows["chrom"], *windows["centered"])
                if windows["reverse_center"]:
                    centered_values = centered_values[::-1]
                metrics = coverage_metrics(orf_values)
                upstream_mean = float(np.nanmean(upstream_values)) if np.isfinite(upstream_values).any() else np.nan
                downstream_mean = float(np.nanmean(downstream_values)) if np.isfinite(downstream_values).any() else np.nan
                metrics.update(
                    {
                        "ORF_id": row["ORF_id"],
                        "sample": sample,
                        "group": row["group"],
                        "primary_noncanonical_category": row["primary_noncanonical_category"],
                        "upstream_200nt_mean_coverage": upstream_mean,
                        "downstream_200nt_mean_coverage": downstream_mean,
                        "orf_upstream_coverage_ratio": metrics["mean_coverage"] / upstream_mean if upstream_mean and upstream_mean > 0 else np.nan,
                        "orf_downstream_coverage_ratio": metrics["mean_coverage"] / downstream_mean if downstream_mean and downstream_mean > 0 else np.nan,
                        "used_for_orf_aggregate": sample in aggregate_samples,
                    }
                )
                sample_rows.append(metrics)
                if sample in aggregate_samples:
                    centered_vectors.append(centered_values)
            if centered_vectors:
                matrix_row = {
                    "ORF_id": row["ORF_id"],
                    "group": row["group"],
                    "primary_noncanonical_category": row["primary_noncanonical_category"],
                    "n_samples_used": len(centered_vectors),
                }
                mean_vector = np.nanmean(np.vstack(centered_vectors), axis=0)
                for pos, value in zip(range(-100, 101), mean_vector):
                    matrix_row[f"pos_{pos}"] = value
                matrix_rows.append(matrix_row)
    finally:
        for bw in handles.values():
            bw.close()

    sample_df = pd.DataFrame(sample_rows)
    sample_df.to_csv(SAMPLE_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {SAMPLE_OUT}")

    agg_features = [
        "mean_coverage",
        "median_coverage",
        "covered_fraction",
        "coverage_cv",
        "zero_fraction",
        "upstream_200nt_mean_coverage",
        "downstream_200nt_mean_coverage",
        "orf_upstream_coverage_ratio",
        "orf_downstream_coverage_ratio",
    ]
    orf_df = (
        sample_df[sample_df["used_for_orf_aggregate"]]
        .groupby(["ORF_id", "group", "primary_noncanonical_category"], as_index=False)[agg_features]
        .mean(numeric_only=True)
    )
    orf_df = orf_df.merge(groups[["ORF_id", "n_detected_samples", "detected_samples"]], on="ORF_id", how="left")
    orf_df.to_csv(ORF_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {ORF_OUT}")

    matrix_df = pd.DataFrame(matrix_rows)
    matrix_df.to_csv(MATRIX_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {MATRIX_OUT}")

    normalized_matrix_df = pd.DataFrame()
    body_mean_normalized_matrix_df = pd.DataFrame()
    if not matrix_df.empty:
        pos_cols = [col for col in matrix_df.columns if col.startswith("pos_")]

        # Intended normalization: each ORF start-centered vector is divided by
        # that same vector's mean across positions -100..+100.
        normalized_matrix_df = matrix_df.copy()
        normalized_values = normalized_matrix_df[pos_cols].apply(pd.to_numeric, errors="coerce")
        vector_scale = normalized_values.mean(axis=1, skipna=True)
        normalized_matrix_df["start_centered_vector_mean_for_normalization"] = vector_scale
        normalized_matrix_df = normalize_position_columns_by_scale(
            normalized_matrix_df,
            pos_cols,
            vector_scale,
        )
        normalized_matrix_df.to_csv(NORMALIZED_MATRIX_OUT, sep="\t", index=False, na_rep="NA")
        print(f"Wrote {NORMALIZED_MATRIX_OUT}")

        # Diagnostic legacy normalization: start-centered vector divided by the
        # mean coverage across the full ORF body. This can be much larger than 1
        # near the start if start-proximal coverage exceeds full-ORF coverage.
        body_scale_df = orf_df[["ORF_id", "mean_coverage"]].rename(
            columns={"mean_coverage": "orf_mean_coverage_for_normalization"}
        )
        body_mean_normalized_matrix_df = matrix_df.merge(body_scale_df, on="ORF_id", how="left")
        body_scale = pd.to_numeric(
            body_mean_normalized_matrix_df["orf_mean_coverage_for_normalization"],
            errors="coerce",
        )
        body_mean_normalized_matrix_df = normalize_position_columns_by_scale(
            body_mean_normalized_matrix_df,
            pos_cols,
            body_scale,
        )
        body_mean_normalized_matrix_df.to_csv(
            ORF_BODY_MEAN_NORMALIZED_MATRIX_OUT,
            sep="\t",
            index=False,
            na_rep="NA",
        )
        print(f"Wrote {ORF_BODY_MEAN_NORMALIZED_MATRIX_OUT}")
        write_normalization_examples(
            matrix_df,
            normalized_matrix_df,
            body_mean_normalized_matrix_df,
            pos_cols,
        )

    stats_tables = []
    plot_features = [
        ("mean_coverage", "Mean ORF coverage", "Mean ORF coverage"),
        ("covered_fraction", "Covered fraction", "Covered fraction"),
        ("coverage_cv", "Coverage CV", "Coverage CV"),
        ("orf_upstream_coverage_ratio", "ORF / upstream coverage", "ORF/upstream coverage ratio"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (feature, ylabel, title) in zip(axes.flat, plot_features):
        stats_tables.append(boxplot_with_points(ax, orf_df, feature, y_label=ylabel, title=title))
    stats_df = pd.concat(stats_tables, ignore_index=True)
    stats_df.to_csv(STATS_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {STATS_OUT}")
    save_figure(fig, "Fig2_rna_coverage_context_boxplots.pdf")

    if not matrix_df.empty:
        long_rows = []
        pos_cols = [col for col in matrix_df.columns if col.startswith("pos_")]
        for group in GROUP_ORDER:
            sub = matrix_df[matrix_df["group"].eq(group)]
            if sub.empty:
                continue
            for col in pos_cols:
                long_rows.append({"group": group, "position": int(col.replace("pos_", "")), "mean_coverage": pd.to_numeric(sub[col], errors="coerce").mean()})
        long_df = pd.DataFrame(long_rows)
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.lineplot(data=long_df, x="position", y="mean_coverage", hue="group", hue_order=GROUP_ORDER, ax=ax)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_ylabel("Mean CPM coverage")
        ax.set_title("Start-centered RNA coverage metaplot")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [display_group(label) for label in labels], title="Group")
        save_figure(fig, "Fig2_start_centered_coverage_metaplot.pdf")

    if not normalized_matrix_df.empty:
        long_rows = []
        pos_cols = [col for col in normalized_matrix_df.columns if col.startswith("pos_")]
        for group in GROUP_ORDER:
            sub = normalized_matrix_df[normalized_matrix_df["group"].eq(group)]
            if sub.empty:
                continue
            for col in pos_cols:
                long_rows.append(
                    {
                        "group": group,
                        "position": int(col.replace("pos_", "")),
                        "mean_normalized_coverage": pd.to_numeric(sub[col], errors="coerce").mean(),
                    }
                )
        long_df = pd.DataFrame(long_rows)
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.lineplot(
            data=long_df,
            x="position",
            y="mean_normalized_coverage",
            hue="group",
            hue_order=GROUP_ORDER,
            ax=ax,
        )
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.axhline(1, color="gray", lw=0.8, ls=":")
        ax.set_ylabel("Coverage / ORF vector mean coverage")
        ax.set_title("Start-centered RNA coverage metaplot, vector-mean normalized")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [display_group(label) for label in labels], title="Group")
        save_figure(fig, "Fig2_start_centered_coverage_metaplot_ORFmean_normalized.pdf")

    print("02_rna_coverage_context.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
