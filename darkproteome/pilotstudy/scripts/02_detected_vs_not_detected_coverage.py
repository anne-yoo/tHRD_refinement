#!/usr/bin/env python3
"""Compare start-centered coverage features for detected vs not-detected ORF-sample pairs."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tc_common import (
    GROUP_ORDER,
    TABLE_DIR,
    add_pvalue_brackets,
    archive_script,
    bh_adjust,
    display_group,
    ensure_fig_dirs,
    mannwhitney_stats,
    print_paths,
    read_groups,
    save_figure,
)
from tc_coverage_common import (
    close_bigwigs,
    coverage_features_from_vector,
    detection_pairs,
    discover_bigwigs,
    open_bigwigs,
    read_sample_level_detection,
    start_centered_vector,
    status_label,
)


FEATURE_OUT = TABLE_DIR / "detected_vs_not_detected_coverage_features.tsv"
STATS_OUT = TABLE_DIR / "detected_vs_not_detected_statistics.tsv"

FEATURES = [
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
STATUS_ORDER = ["not_detected", "detected"]


def build_feature_table(groups: pd.DataFrame, handles: dict, detected_pairs: set[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    total = len(groups) * len(handles)
    processed = 0
    for _, row in groups.iterrows():
        for sample, bw in handles.items():
            processed += 1
            if processed % 10000 == 0:
                print(f"Processed {processed}/{total} ORF-sample pair(s)")
            detected = int((row["ORF_id"], sample) in detected_pairs)
            vector = start_centered_vector(bw, row)
            feature_row = coverage_features_from_vector(vector)
            feature_row.update(
                {
                    "ORF_id": row["ORF_id"],
                    "sample": sample,
                    "group": row["group"],
                    "primary_noncanonical_category": row["primary_noncanonical_category"],
                    "detected": detected,
                    "detection_status": status_label(detected),
                }
            )
            rows.append(feature_row)
    return pd.DataFrame(rows)


def plot_feature_grid(feature_df: pd.DataFrame) -> pd.DataFrame:
    stats_rows = []
    fig, axes = plt.subplots(len(FEATURES), len(GROUP_ORDER), figsize=(15, 3.0 * len(FEATURES)), squeeze=False)
    for row_idx, feature in enumerate(FEATURES):
        for col_idx, group in enumerate(GROUP_ORDER):
            ax = axes[row_idx, col_idx]
            sub = feature_df[feature_df["group"].eq(group)].copy()
            sub[feature] = pd.to_numeric(sub[feature], errors="coerce")
            sub = sub.dropna(subset=[feature, "detection_status"])
            if sub.empty or sub["detection_status"].nunique() < 2:
                ax.axis("off")
                ax.set_title(f"{display_group(group)}\n{feature}")
                continue
            sns.boxplot(
                data=sub,
                x="detection_status",
                y=feature,
                order=STATUS_ORDER,
                showfliers=False,
                color="#d6e4f0",
                linewidth=1.2,
                ax=ax,
            )
            ax.set_xlabel("")
            ax.set_ylabel(feature)
            ax.set_title(display_group(group) if row_idx == 0 else "")
            stat = mannwhitney_stats(
                sub,
                feature,
                group_col="detection_status",
                pairs=[("not_detected", "detected")],
            )
            if not stat.empty:
                stat["orf_group"] = group
                stat["orf_group_label"] = display_group(group)
                stats_rows.append(stat)
                add_pvalue_brackets(ax, STATUS_ORDER, stat)
    stats_df = pd.concat(stats_rows, ignore_index=True, sort=False) if stats_rows else pd.DataFrame()
    if not stats_df.empty:
        stats_df["padj_bh_all_tests"] = bh_adjust(stats_df["pvalue"])
    save_figure(fig, "detected_vs_not_detected_coverage_features.pdf")
    return stats_df


def main() -> int:
    ensure_fig_dirs()
    archive_script(__file__)
    print_paths(
        script_name=Path(__file__).name,
        inputs=[
            Path("pancreas.translated_orfs.sample_level.tsv"),
            Path("orf_groups.combined_metadata.tsv"),
            Path("bigwig/*.CPM.bw"),
        ],
        outputs=[FEATURE_OUT, STATS_OUT],
    )

    groups = read_groups()
    sample_level = read_sample_level_detection()
    detected = detection_pairs(sample_level)
    handles = open_bigwigs(discover_bigwigs())
    try:
        feature_df = build_feature_table(groups, handles, detected)
    finally:
        close_bigwigs(handles)
    feature_df.to_csv(FEATURE_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {FEATURE_OUT}")

    stats_df = plot_feature_grid(feature_df)
    stats_df.to_csv(STATS_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {STATS_OUT}")
    print("02_detected_vs_not_detected_coverage.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

