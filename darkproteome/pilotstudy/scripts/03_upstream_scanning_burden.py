#!/usr/bin/env python3
"""Analyze upstream AUG burden around translated ORFs."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tc_common import (
    TABLE_DIR,
    archive_script,
    boxplot_with_points,
    ensure_fig_dirs,
    fisher_binary_stats,
    fraction_barplot,
    mannwhitney_stats,
    print_paths,
    read_sequence_context,
    save_figure,
)


FEATURE_OUT = TABLE_DIR / "upstream_scanning_features.tsv"
STATS_OUT = TABLE_DIR / "upstream_scanning_statistics.tsv"


def main() -> int:
    ensure_fig_dirs()
    archive_script(__file__)
    print_paths(
        script_name=Path(__file__).name,
        inputs=[Path("orf_sequence_context_features.tsv")],
        outputs=[FEATURE_OUT, STATS_OUT],
    )

    df = read_sequence_context()
    df["has_upstream_AUG_200nt"] = pd.to_numeric(df["upstream_AUG_count_200nt"], errors="coerce").gt(0)
    df["upstream_ORF_count_200nt"] = pd.NA
    df["upstream_ORF_count_500nt"] = pd.NA
    df["upstream_ORF_explanation"] = "Not computed: upstream ORF inference requires transcript exon/CDS structure and reading-frame-aware scanning."
    df.to_csv(FEATURE_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {FEATURE_OUT}")

    stats_tables = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, feature, ylabel, title in [
        (axes[0, 0], "upstream_AUG_count_200nt", "Count", "Upstream AUG count"),
        (axes[0, 1], "upstream_AUG_density_200nt", "Density", "Upstream AUG density"),
        (axes[1, 0], "distance_to_nearest_upstream_AUG", "nt", "Distance to nearest upstream AUG"),
    ]:
        stats_tables.append(boxplot_with_points(ax, df, feature, y_label=ylabel, title=title))
    fraction_barplot(axes[1, 1], df, "has_upstream_AUG_200nt", y_label="Fraction", title="Fraction with at least one upstream AUG")
    binary_stats = fisher_binary_stats(df, "has_upstream_AUG_200nt")
    binary_stats["test"] = "Fisher exact"
    continuous_stats = pd.concat(stats_tables, ignore_index=True)
    continuous_stats["test"] = "Mann-Whitney U"
    stats_df = pd.concat([continuous_stats, binary_stats], ignore_index=True, sort=False)
    stats_df.to_csv(STATS_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {STATS_OUT}")
    save_figure(fig, "Fig3_upstream_scanning_burden.pdf")

    print("03_upstream_scanning_burden.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

