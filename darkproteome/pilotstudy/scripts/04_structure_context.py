#!/usr/bin/env python3
"""Compute sequence-based structure proxy features around ORF starts."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from Bio import SeqIO

from tc_common import (
    GENOME_FA,
    TABLE_DIR,
    archive_script,
    boxplot_with_points,
    command_exists,
    ensure_fig_dirs,
    gc_content,
    oriented_start_window,
    print_paths,
    read_groups,
    resolve_chrom_key,
    run_rnafold,
    save_figure,
)


FEATURE_OUT = TABLE_DIR / "orf_structure_context_features.tsv"
STATS_OUT = TABLE_DIR / "structure_context_statistics.tsv"
WARNING_OUT = TABLE_DIR / "structure_context_warnings.txt"


def main() -> int:
    ensure_fig_dirs()
    archive_script(__file__)
    print_paths(
        script_name=Path(__file__).name,
        inputs=[GENOME_FA, Path("orf_groups.combined_metadata.tsv")],
        outputs=[FEATURE_OUT, STATS_OUT, WARNING_OUT],
    )

    groups = read_groups()
    if not GENOME_FA.exists():
        raise SystemExit(f"Genome FASTA not found: {GENOME_FA}")

    rnafold_available = command_exists("RNAfold")
    warnings = []
    if not rnafold_available:
        warnings.append("RNAfold unavailable: MFE features were written as NA.")

    rows = []
    fasta = SeqIO.index(str(GENOME_FA), "fasta")
    try:
        chrom_keys = {chrom: resolve_chrom_key(fasta, chrom) for chrom in groups["chr"].dropna().astype(str).unique()}
        for chrom, sub in groups.groupby("chr", sort=False, dropna=False):
            chrom_text = str(chrom)
            chrom_key = chrom_keys.get(chrom_text)
            if chrom_key is None:
                warnings.append(f"Chromosome not found in FASTA: {chrom_text}")
                chrom_seq = None
            else:
                print(f"Loading {chrom_text} from FASTA as {chrom_key} for {len(sub)} ORF(s)")
                chrom_seq = str(fasta[chrom_key].seq)
            for _, row in sub.iterrows():
                start0 = int(row["start0"]) if pd.notna(row["start0"]) else -1
                end0 = int(row["end0"]) if pd.notna(row["end0"]) else -1
                seq20 = oriented_start_window(chrom_seq, start0, end0, row["strand"], 20, 20) if chrom_seq else None
                seq50 = oriented_start_window(chrom_seq, start0, end0, row["strand"], 50, 50) if chrom_seq else None
                seq100 = oriented_start_window(chrom_seq, start0, end0, row["strand"], 100, 100) if chrom_seq else None
                rows.append(
                    {
                        "ORF_id": row["ORF_id"],
                        "group": row["group"],
                        "primary_noncanonical_category": row["primary_noncanonical_category"],
                        "GC_start_pm20nt": gc_content(seq20),
                        "GC_start_pm50nt": gc_content(seq50),
                        "MFE_start_pm50nt": run_rnafold(seq50) if rnafold_available and seq50 else pd.NA,
                        "MFE_start_pm100nt": run_rnafold(seq100) if rnafold_available and seq100 else pd.NA,
                        "RNAfold_available": rnafold_available,
                    }
                )
    finally:
        fasta.close()

    out_df = pd.DataFrame(rows)
    out_df.to_csv(FEATURE_OUT, sep="\t", index=False, na_rep="NA")
    WARNING_OUT.write_text("\n".join(warnings) + ("\n" if warnings else "No warnings.\n"))
    print(f"Wrote {FEATURE_OUT}")
    print(f"Wrote {WARNING_OUT}")

    plot_features = [
        ("GC_start_pm20nt", "GC fraction", "GC around start pm20 nt"),
        ("GC_start_pm50nt", "GC fraction", "GC around start pm50 nt"),
    ]
    if rnafold_available:
        plot_features.extend(
            [
                ("MFE_start_pm50nt", "MFE", "RNAfold MFE pm50 nt"),
                ("MFE_start_pm100nt", "MFE", "RNAfold MFE pm100 nt"),
            ]
        )
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    stats_tables = []
    for ax, (feature, ylabel, title) in zip(axes.flat, plot_features):
        stats_tables.append(boxplot_with_points(ax, out_df, feature, y_label=ylabel, title=title))
    for ax in axes.flat[len(plot_features) :]:
        ax.axis("off")
    stats_df = pd.concat(stats_tables, ignore_index=True) if stats_tables else pd.DataFrame()
    stats_df.to_csv(STATS_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {STATS_OUT}")
    save_figure(fig, "Fig4_structure_context.pdf")

    print("04_structure_context.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

