#!/usr/bin/env python3
"""Derive ORF positional context flags and visualize noncanonical categories."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tc_common import (
    TABLE_DIR,
    archive_script,
    boxplot_with_points,
    derive_position_flags,
    display_group,
    ensure_fig_dirs,
    fisher_binary_stats,
    fraction_barplot,
    mannwhitney_stats,
    primary_position_label,
    print_paths,
    read_groups,
    read_sequence_context,
    save_figure,
)


FEATURE_OUT = TABLE_DIR / "orf_positional_context_features.tsv"
STATS_OUT = TABLE_DIR / "positional_context_statistics.tsv"


def main() -> int:
    ensure_fig_dirs()
    archive_script(__file__)
    print_paths(
        script_name=Path(__file__).name,
        inputs=[Path("orf_groups.combined_metadata.tsv"), Path("orf_sequence_context_features.tsv")],
        outputs=[FEATURE_OUT, STATS_OUT],
    )

    groups = read_groups()
    seq = read_sequence_context()[["ORF_id", "strong_kozak", "start_codon_type"]]
    df = groups.merge(seq, on="ORF_id", how="left")
    flag_rows = []
    for _, row in df.iterrows():
        flags = derive_position_flags(row.get("ORF_type", ""), row.get("primary_noncanonical_category", ""))
        flag_rows.append(flags)
    flags_df = pd.DataFrame(flag_rows)
    out_df = pd.concat([df.reset_index(drop=True), flags_df], axis=1)
    out_df["primary_position_label"] = out_df.apply(primary_position_label, axis=1)
    out_df["distance_from_transcript_start"] = pd.NA
    out_df["distance_from_CDS_start"] = pd.NA
    out_df["distance_explanation"] = "Not computed: GTF-based transcript coordinate mapping is required."
    out_df.to_csv(FEATURE_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {FEATURE_OUT}")

    noncanon = out_df[out_df["group"].ne("group1_canonical_translated_ORF")].copy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    category_counts = (
        noncanon.groupby(["group", "primary_position_label"])
        .size()
        .reset_index(name="n")
    )
    if not category_counts.empty:
        category_counts["fraction"] = category_counts["n"] / category_counts.groupby("group")["n"].transform("sum")
        pivot = category_counts.pivot_table(index="group", columns="primary_position_label", values="fraction", fill_value=0)
        pivot.plot(kind="bar", stacked=True, ax=axes[0, 0], colormap="tab20")
        axes[0, 0].set_xticks(range(len(pivot.index)))
        axes[0, 0].set_xticklabels([display_group(label) for label in pivot.index], rotation=25, ha="right")
        axes[0, 0].set_ylabel("Fraction")
        axes[0, 0].set_title("Positional categories among noncanonical ORFs")
        axes[0, 0].legend(title="Category", fontsize=7, bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        axes[0, 0].axis("off")

    codon_counts = (
        noncanon.groupby(["primary_position_label", "start_codon_type"])
        .size()
        .reset_index(name="n")
    )
    if not codon_counts.empty:
        codon_counts["fraction"] = codon_counts["n"] / codon_counts.groupby("primary_position_label")["n"].transform("sum")
        sns.barplot(data=codon_counts, x="primary_position_label", y="fraction", hue="start_codon_type", ax=axes[0, 1])
        for label in axes[0, 1].get_xticklabels():
            label.set_rotation(35)
            label.set_ha("right")
        axes[0, 1].set_xlabel("")
        axes[0, 1].set_ylabel("Fraction")
        axes[0, 1].set_title("Start codon type by positional category")
    else:
        axes[0, 1].axis("off")

    fraction_barplot(
        axes[1, 0],
        noncanon.rename(columns={"primary_position_label": "category"}),
        "strong_kozak",
        y_label="Fraction",
        title="Strong Kozak fraction by positional category",
        group_col="category",
    )

    top_categories = noncanon["primary_position_label"].value_counts()
    selected = top_categories[top_categories >= 3].head(3).index.tolist()
    stats_tables = []
    if len(selected) >= 2:
        pairs = list(itertools.combinations(selected, 2))
        length_df = noncanon[noncanon["primary_position_label"].isin(selected)].rename(
            columns={"primary_position_label": "category"}
        )
        stats_tables.append(
            boxplot_with_points(
                axes[1, 1],
                length_df,
                "orf_length_nt",
                y_label="nt",
                title="ORF length by selected positional category",
                group_col="category",
                pairs=pairs,
            )
        )
    else:
        axes[1, 1].axis("off")
        axes[1, 1].set_title("ORF length: not enough categories with n >= 3")

    binary_stats = []
    for flag in [
        "is_5UTR",
        "is_CDSFrameOverlap",
        "is_3UTR",
        "is_lncRNA_or_ncRNA",
        "is_internal",
        "is_uORF",
        "is_dORF",
        "is_novel",
        "is_truncated",
        "is_extended",
    ]:
        binary_stats.append(fisher_binary_stats(out_df, flag))
    stats_df = pd.concat(binary_stats + stats_tables, ignore_index=True, sort=False) if binary_stats or stats_tables else pd.DataFrame()
    stats_df.to_csv(STATS_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {STATS_OUT}")
    save_figure(fig, "Fig5_orf_positional_context.pdf")

    print("05_orf_positional_context.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
