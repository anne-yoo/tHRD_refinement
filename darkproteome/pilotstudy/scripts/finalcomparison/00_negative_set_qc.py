#!/usr/bin/env python3
"""Summarize CPAT-negative set selection and QC before final comparisons."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from fc_common import NEG_DIR, TABLE_DIR, archive_script, clean_text, ensure_dirs, print_paths, read_tsv, save_figure


QC_SUMMARY = NEG_DIR / "tables" / "cpat_negative_orf_qc_summary.tsv"
BEST_ORFS = NEG_DIR / "tables" / "cpat_best_orf_per_negative_transcript.tsv"
FILTERED_ORFS = NEG_DIR / "tables" / "cpat_best_orf_per_negative_transcript.prob_lt_0.364.tsv"
SAMPLED_GENOMIC = NEG_DIR / "tables" / "cpat_negative_orfs.sampled_25863.genomic.tsv"
PROB_SUMMARY = NEG_DIR / "tables" / "cpat_negative_probability_summary.tsv"
START_CODON_DIST = NEG_DIR / "tables" / "cpat_negative_start_codon_distribution.tsv"
LENGTH_SUMMARY = NEG_DIR / "tables" / "cpat_negative_length_summary.tsv"

SELECTION_OUT = TABLE_DIR / "negative_set_selection_summary.tsv"


def metric_lookup(summary: pd.DataFrame) -> dict:
    if summary.empty or "metric" not in summary.columns or "value" not in summary.columns:
        return {}
    return {clean_text(row["metric"]): row["value"] for _, row in summary.iterrows()}


def count_or_na(path: Path) -> object:
    if not path.exists():
        return pd.NA
    return len(pd.read_csv(path, sep="\t", dtype=str))


def probability_column(df: pd.DataFrame) -> str | None:
    for col in ["CPAT_coding_probability", "Coding_prob"]:
        if col in df.columns:
            return col
    return None


def main() -> int:
    ensure_dirs()
    archive_script(__file__)
    print_paths(
        "00_negative_set_qc.py",
        inputs=[QC_SUMMARY, BEST_ORFS, FILTERED_ORFS, SAMPLED_GENOMIC, PROB_SUMMARY, START_CODON_DIST, LENGTH_SUMMARY],
        outputs=[SELECTION_OUT],
    )

    qc = read_tsv(QC_SUMMARY, allow_missing=True)
    lookup = metric_lookup(qc)
    rows = [
        {
            "step": "negative_noncoding_transcripts",
            "n": lookup.get("number_of_negative_noncoding_transcripts", pd.NA),
            "description": "GENCODE noncoding transcripts with zero positive ORFs",
        },
        {
            "step": "CPAT_ORFs_from_negative_transcripts",
            "n": lookup.get("number_of_CPAT_ORFs_from_negative_noncoding_transcripts", pd.NA),
            "description": "Existing CPAT-enumerated ORFs on negative transcripts",
        },
        {
            "step": "transcript_level_best_ORFs",
            "n": lookup.get("number_of_CPAT_best_ORFs", count_or_na(BEST_ORFS)),
            "description": "One highest CPAT probability ORF per transcript",
        },
        {
            "step": "CPAT_probability_lt_0.364",
            "n": lookup.get("number_with_Coding_prob_lt_0.364", count_or_na(FILTERED_ORFS)),
            "description": "Best ORFs below human CPAT coding cutoff",
        },
        {
            "step": "sampled_ORFs",
            "n": lookup.get("number_sampled", count_or_na(SAMPLED_GENOMIC)),
            "description": "Final sampled CPAT-negative noncoding ORFs",
        },
    ]
    flow = pd.DataFrame(rows)
    flow.to_csv(SELECTION_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {SELECTION_OUT}")

    best = read_tsv(BEST_ORFS, allow_missing=True)
    filtered = read_tsv(FILTERED_ORFS, allow_missing=True)
    sampled = read_tsv(SAMPLED_GENOMIC, allow_missing=True)
    codon_dist = read_tsv(START_CODON_DIST, allow_missing=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))
    ax = axes[0, 0]
    plot_flow = flow.copy()
    plot_flow["n_numeric"] = pd.to_numeric(plot_flow["n"], errors="coerce")
    sns.barplot(data=plot_flow, x="step", y="n_numeric", ax=ax, color="#7ca6c0")
    ax.tick_params(axis="x", rotation=25)
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.set_title("CPAT-negative selection flow")

    ax = axes[0, 1]
    hist_rows = []
    for label, df in [("Best ORFs", best), ("CPAT < 0.364", filtered), ("Sampled", sampled)]:
        col = probability_column(df)
        if col:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            hist_rows.extend({"set": label, "CPAT probability": value} for value in vals)
    hist_df = pd.DataFrame(hist_rows)
    if hist_df.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "CPAT probability\nnot available", ha="center", va="center")
    else:
        sns.histplot(data=hist_df, x="CPAT probability", hue="set", element="step", stat="density", common_norm=False, ax=ax)
        ax.axvline(0.364, color="black", ls="--", lw=0.9)
        ax.set_title("CPAT probability distribution")

    ax = axes[1, 0]
    if codon_dist.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "Start codon distribution\nnot available", ha="center", va="center")
    else:
        codon_col = "start_codon" if "start_codon" in codon_dist.columns else codon_dist.columns[0]
        count_col = "n" if "n" in codon_dist.columns else codon_dist.columns[1]
        codon_dist[count_col] = pd.to_numeric(codon_dist[count_col], errors="coerce")
        sns.barplot(data=codon_dist, x=codon_col, y=count_col, ax=ax, color="#b8a1d9")
        ax.tick_params(axis="x", rotation=25)
        ax.set_xlabel("Start codon")
        ax.set_ylabel("Count")
        ax.set_title("CPAT-negative start codons")

    ax = axes[1, 1]
    if sampled.empty or "orf_length_nt" not in sampled.columns:
        ax.axis("off")
        ax.text(0.5, 0.5, "ORF length distribution\nnot available", ha="center", va="center")
    else:
        lengths = pd.to_numeric(sampled["orf_length_nt"], errors="coerce").dropna()
        sns.histplot(lengths, bins=60, ax=ax, color="#9bbf7a")
        ax.set_xlabel("ORF length (nt)")
        ax.set_title("CPAT-negative ORF length")

    save_figure(fig, "Fig0_CPAT_negative_set_QC.pdf")
    print("00_negative_set_qc.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
