#!/usr/bin/env python3
"""Analyze start codon and Kozak context features by ORF group."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tc_common import (
    GROUP_ORDER,
    TABLE_DIR,
    archive_script,
    bh_adjust,
    display_group,
    ensure_fig_dirs,
    fisher_binary_stats,
    fraction_barplot,
    print_paths,
    read_sequence_context,
    save_figure,
)


STATS_OUT = TABLE_DIR / "start_context_statistics.tsv"


def base_frequency_matrix(sequences: pd.Series) -> pd.DataFrame:
    seqs = [str(seq).upper().replace("U", "T") for seq in sequences.dropna() if len(str(seq)) == 23]
    positions = list(range(-10, 0)) + [1, 2, 3] + list(range(4, 14))
    if not seqs:
        return pd.DataFrame(0.0, index=positions, columns=list("ACGT"))
    rows = []
    for pos_idx, pos in enumerate(positions):
        counts = {base: 0 for base in "ACGT"}
        for seq in seqs:
            base = seq[pos_idx]
            if base in counts:
                counts[base] += 1
        total = sum(counts.values())
        rows.append({base: counts[base] / total if total else 0.0 for base in "ACGT"})
    return pd.DataFrame(rows, index=positions)


def plot_logo_panel(ax, matrix: pd.DataFrame, title: str) -> None:
    try:
        import logomaker

        logo_df = matrix.copy()
        logo_df.index = logo_df.index.astype(str)
        logomaker.Logo(logo_df, ax=ax)
        ax.set_ylabel("Frequency")
    except Exception:
        sns.heatmap(matrix.T, cmap="viridis", vmin=0, vmax=1, ax=ax, cbar=False)
        ax.set_ylabel("Base")
    ax.set_title(title)
    ax.set_xlabel("Position relative to start codon")


def main() -> int:
    ensure_fig_dirs()
    archive_script(__file__)
    print_paths(
        script_name=Path(__file__).name,
        inputs=[Path("orf_sequence_context_features.tsv")],
        outputs=[STATS_OUT],
    )

    df = read_sequence_context()
    for col in ["kozak_minus3_AG", "kozak_plus4_G", "strong_kozak"]:
        if col not in df.columns:
            raise SystemExit(f"Missing required sequence-context column: {col}")

    stats_tables = []
    for feature in ["kozak_minus3_AG", "kozak_plus4_G", "strong_kozak"]:
        stats_tables.append(fisher_binary_stats(df, feature))
    stats_df = pd.concat(stats_tables, ignore_index=True)
    stats_df["padj_bh_all_features"] = bh_adjust(stats_df["pvalue"])
    stats_df.to_csv(STATS_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {STATS_OUT}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fraction_barplot(axes[0, 0], df, "kozak_minus3_AG", y_label="Fraction", title="-3 A/G fraction by group")
    fraction_barplot(axes[0, 1], df, "kozak_plus4_G", y_label="Fraction", title="+4 G fraction by group")
    fraction_barplot(axes[1, 0], df, "strong_kozak", y_label="Fraction", title="Strong Kozak fraction by group")

    codon_counts = (
        df.groupby(["group", "start_codon_type"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    codon_counts["fraction"] = codon_counts["n"] / codon_counts.groupby("group")["n"].transform("sum")
    pivot = codon_counts.pivot_table(index="group", columns="start_codon_type", values="fraction", fill_value=0)
    order = [group for group in GROUP_ORDER if group in pivot.index]
    pivot = pivot.reindex(order)
    pivot.plot(kind="bar", stacked=True, ax=axes[1, 1], colormap="Set2")
    axes[1, 1].set_xticks(range(len(order)))
    axes[1, 1].set_xticklabels([display_group(group) for group in order], rotation=25, ha="right")
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("Fraction")
    axes[1, 1].set_title("Start codon type distribution")
    axes[1, 1].legend(title="Start codon type", fontsize=8)
    save_figure(fig, "Fig1_start_context_kozak.pdf")

    fig, axes = plt.subplots(len(GROUP_ORDER), 1, figsize=(12, 3.2 * len(GROUP_ORDER)), squeeze=False)
    for ax, group in zip(axes[:, 0], GROUP_ORDER):
        sub = df[df["group"].eq(group)]
        matrix = base_frequency_matrix(sub.get("start_context_10nt_up_10nt_down", pd.Series(dtype=str)))
        plot_logo_panel(ax, matrix, f"{display_group(group)} start context")
    save_figure(fig, "Fig1_start_context_logo.pdf")

    print("01_start_context_analysis.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
