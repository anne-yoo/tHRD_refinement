#!/usr/bin/env python3
"""Compare sequence-derived ORF features across four groups."""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from fc_common import (
    INPUT_DIR,
    NEG_DIR,
    TABLE_DIR,
    archive_script,
    boxplot_four_group,
    clean_text,
    ensure_dirs,
    fisher_stats,
    fraction_barplot_four_group,
    normalize_codon,
    print_paths,
    read_master,
    read_tsv,
    save_figure,
    start_codon_type,
)


POS_SEQ = INPUT_DIR / "tables" / "orf_sequence_context_features.tsv"
NEG_SEQ = NEG_DIR / "tables" / "cpat_negative_orfs.sequence_context_features.tsv"
STATS_OUT = TABLE_DIR / "sequence_context_four_group_statistics.tsv"
FEATURE_OUT = TABLE_DIR / "sequence_context_four_group_features.tsv"

BINARY_FEATURES = ["kozak_minus3_AG", "kozak_plus4_G", "strong_kozak"]
CONTINUOUS_FEATURES = [
    "GC_start_window_20nt",
    "upstream_AUG_count_200nt",
    "upstream_AUG_density_200nt",
    "distance_to_nearest_upstream_AUG",
    "orf_length_nt",
]


def load_features() -> pd.DataFrame:
    master = read_master()
    pos = read_tsv(POS_SEQ, required=["ORF_id"])
    neg = read_tsv(NEG_SEQ, required=["ORF_id"])
    seq = pd.concat([pos, neg], ignore_index=True, sort=False)
    seq = seq.drop_duplicates("ORF_id", keep="first")
    keep = [
        "ORF_id",
        "plot_group",
        "orf_length_nt",
        "start_codon",
        "minus3_base",
        "plus4_base",
        "kozak_minus3_AG",
        "kozak_plus4_G",
        "strong_kozak",
        "GC_start_window_20nt",
        "upstream_AUG_count_200nt",
        "upstream_AUG_density_200nt",
        "distance_to_nearest_upstream_AUG",
    ]
    merged = master[["ORF_id", "plot_group", "orf_length_nt", "start_codon"]].merge(seq, on="ORF_id", how="left", suffixes=("", "_seq"))
    if "start_codon_seq" in merged.columns:
        merged["start_codon"] = merged["start_codon_seq"].combine_first(merged["start_codon"])
    merged["start_codon"] = merged["start_codon"].map(normalize_codon)
    merged["start_codon_type"] = merged["start_codon"].map(start_codon_type)
    for col in keep:
        if col not in merged.columns:
            merged[col] = pd.NA
    return merged


def main() -> int:
    ensure_dirs()
    archive_script(__file__)
    print_paths("02_sequence_context_comparison.py", inputs=[POS_SEQ, NEG_SEQ, TABLE_DIR / "four_group_orf_metadata.tsv"], outputs=[FEATURE_OUT, STATS_OUT])

    df = load_features()
    df.to_csv(FEATURE_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote feature table: {FEATURE_OUT}")

    stats_tables = []
    fig, axes = plt.subplots(3, 3, figsize=(18, 21), squeeze=False)
    for ax, feature, title in zip(
        axes[0],
        BINARY_FEATURES,
        ["-3 A/G fraction", "+4 G fraction", "Strong Kozak fraction"],
    ):
        stats_tables.append(fraction_barplot_four_group(ax, df, feature, title=title))

    ax = axes[1, 0]
    codon = df.groupby(["plot_group", "start_codon_type"], dropna=False).size().reset_index(name="n")
    codon["fraction"] = codon["n"] / codon.groupby("plot_group")["n"].transform("sum")
    pivot = codon.pivot_table(index="plot_group", columns="start_codon_type", values="fraction", fill_value=0)
    order = [g for g in ["Canonical", "AUG noncanonical", "nonAUG noncanonical", "CPAT-negative noncoding"] if g in pivot.index]
    pivot.reindex(order).plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_xlabel("")
    ax.set_ylabel("Fraction")
    ax.set_title("Start codon type")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Start type", fontsize=8)

    for ax, feature, ylabel, title in [
        (axes[1, 1], "upstream_AUG_count_200nt", "Count", "Upstream AUG count"),
        (axes[1, 2], "upstream_AUG_density_200nt", "Density", "Upstream AUG density"),
        (axes[2, 0], "distance_to_nearest_upstream_AUG", "nt", "Nearest upstream AUG distance"),
        (axes[2, 1], "GC_start_window_20nt", "GC fraction", "GC around start"),
        (axes[2, 2], "orf_length_nt", "nt", "ORF length"),
    ]:
        stats_tables.append(boxplot_four_group(ax, df, feature, y_label=ylabel, title=title))

    stats_df = pd.concat([table for table in stats_tables if table is not None and not table.empty], ignore_index=True, sort=False)
    stats_df.to_csv(STATS_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote statistics: {STATS_OUT}")
    save_figure(fig, "Fig1_sequence_context_four_group.pdf")
    print("02_sequence_context_comparison.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
