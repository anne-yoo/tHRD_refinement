#!/usr/bin/env python3
"""Summarize QC metrics for the CPAT negative ORF set."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from cpat_negative_common import CPAT_CUTOFF, SAMPLE_SIZE, TABLE_DIR, LOG_DIR, archive_script, ensure_dirs, print_header


NEGATIVE_TX = TABLE_DIR / "negative_noncoding_transcripts_no_positive_orf.tsv"
FROM_NEGATIVE = TABLE_DIR / "cpat_orfs.from_negative_noncoding_transcripts.tsv"
BEST = TABLE_DIR / "cpat_best_orf_per_negative_transcript.tsv"
NONCODING = TABLE_DIR / f"cpat_best_orf_per_negative_transcript.prob_lt_{CPAT_CUTOFF}.tsv"
SAMPLED = TABLE_DIR / f"cpat_negative_orfs.sampled_{SAMPLE_SIZE}.tsv"
GENOMIC = TABLE_DIR / f"cpat_negative_orfs.sampled_{SAMPLE_SIZE}.genomic.tsv"
COMPATIBLE = TABLE_DIR / "cpat_negative_orfs.combined_metadata_compatible.tsv"
SEQUENCE_CONTEXT = TABLE_DIR / "cpat_negative_orfs.sequence_context_features.tsv"
MAPPING_MISMATCH = LOG_DIR / "cpat_orf_genomic_mapping_mismatches.tsv"
START_MISMATCH = LOG_DIR / "cpat_negative_start_codon_mismatches.tsv"

SUMMARY_OUT = TABLE_DIR / "cpat_negative_orf_qc_summary.tsv"
START_CODON_DIST_OUT = TABLE_DIR / "cpat_negative_start_codon_distribution.tsv"
LENGTH_SUMMARY_OUT = TABLE_DIR / "cpat_negative_length_summary.tsv"
PROBABILITY_SUMMARY_OUT = TABLE_DIR / "cpat_negative_probability_summary.tsv"


def read_optional(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"WARNING: missing optional input: {path}")
        return None
    return pd.read_csv(path, sep="\t", dtype=str)


def n_rows(path: Path) -> object:
    df = read_optional(path)
    return len(df) if df is not None else pd.NA


def numeric_summary(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        rows.append(
            {
                "feature": column,
                "count": int(len(values)),
                "mean": float(values.mean()) if len(values) else np.nan,
                "sd": float(values.std()) if len(values) > 1 else np.nan,
                "min": float(values.min()) if len(values) else np.nan,
                "q25": float(values.quantile(0.25)) if len(values) else np.nan,
                "median": float(values.median()) if len(values) else np.nan,
                "q75": float(values.quantile(0.75)) if len(values) else np.nan,
                "max": float(values.max()) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def start_codon_distribution(context_df: Optional[pd.DataFrame], compatible_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    source = context_df if context_df is not None and "extracted_start_codon" in context_df.columns else compatible_df
    if source is None:
        return pd.DataFrame(columns=["start_codon", "n", "fraction"])
    column = "extracted_start_codon" if "extracted_start_codon" in source.columns else "start_codon"
    values = source[column].fillna("NA").astype(str)
    counts = values.value_counts(dropna=False).reset_index()
    counts.columns = ["start_codon", "n"]
    total = counts["n"].sum()
    counts["fraction"] = counts["n"] / total if total else np.nan
    return counts


def main() -> int:
    ensure_dirs()
    archive_script(__file__)
    print_header(
        "06_qc_summary.py",
        inputs=[
            NEGATIVE_TX,
            FROM_NEGATIVE,
            BEST,
            NONCODING,
            SAMPLED,
            GENOMIC,
            COMPATIBLE,
            SEQUENCE_CONTEXT,
            MAPPING_MISMATCH,
            START_MISMATCH,
        ],
        outputs=[SUMMARY_OUT, START_CODON_DIST_OUT, LENGTH_SUMMARY_OUT, PROBABILITY_SUMMARY_OUT],
    )

    negative_tx = read_optional(NEGATIVE_TX)
    from_negative = read_optional(FROM_NEGATIVE)
    best = read_optional(BEST)
    noncoding = read_optional(NONCODING)
    sampled = read_optional(SAMPLED)
    genomic = read_optional(GENOMIC)
    compatible = read_optional(COMPATIBLE)
    context = read_optional(SEQUENCE_CONTEXT)
    mapping_mismatch = read_optional(MAPPING_MISMATCH)
    start_mismatch = read_optional(START_MISMATCH)

    rows = [
        {
            "metric": "number_of_negative_noncoding_transcripts",
            "value": len(negative_tx) if negative_tx is not None else pd.NA,
        },
        {
            "metric": "number_of_CPAT_ORFs_from_negative_noncoding_transcripts",
            "value": len(from_negative) if from_negative is not None else pd.NA,
        },
        {
            "metric": "number_of_transcripts_with_at_least_one_CPAT_ORF",
            "value": from_negative["transcript_id"].nunique() if from_negative is not None and "transcript_id" in from_negative.columns else pd.NA,
        },
        {
            "metric": "number_of_CPAT_best_ORFs",
            "value": len(best) if best is not None else pd.NA,
        },
        {
            "metric": f"number_with_Coding_prob_lt_{CPAT_CUTOFF}",
            "value": len(noncoding) if noncoding is not None else pd.NA,
        },
        {
            "metric": "number_sampled",
            "value": len(sampled) if sampled is not None else pd.NA,
        },
        {
            "metric": "number_genomically_mapped",
            "value": len(genomic) if genomic is not None else pd.NA,
        },
        {
            "metric": "number_pipeline_compatible_metadata_rows",
            "value": len(compatible) if compatible is not None else pd.NA,
        },
        {
            "metric": "number_sequence_context_rows",
            "value": len(context) if context is not None else pd.NA,
        },
        {
            "metric": "number_of_genomic_mapping_mismatches",
            "value": len(mapping_mismatch) if mapping_mismatch is not None else pd.NA,
        },
        {
            "metric": "number_of_start_codon_mismatches",
            "value": len(start_mismatch) if start_mismatch is not None else pd.NA,
        },
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_OUT, sep="\t", index=False, na_rep="NA")

    codon_dist = start_codon_distribution(context, compatible)
    codon_dist.to_csv(START_CODON_DIST_OUT, sep="\t", index=False, na_rep="NA")

    length_source = genomic if genomic is not None else sampled
    if length_source is not None:
        length_summary = numeric_summary(length_source, ["orf_length_nt", "orf_length_aa"])
    else:
        length_summary = pd.DataFrame(columns=["feature", "count", "mean", "sd", "min", "q25", "median", "q75", "max"])
    length_summary.to_csv(LENGTH_SUMMARY_OUT, sep="\t", index=False, na_rep="NA")

    prob_source = genomic if genomic is not None else sampled
    if prob_source is not None:
        probability_summary = numeric_summary(prob_source, ["CPAT_coding_probability", "Fickett", "Hexamer"])
    else:
        probability_summary = pd.DataFrame(columns=["feature", "count", "mean", "sd", "min", "q25", "median", "q75", "max"])
    probability_summary.to_csv(PROBABILITY_SUMMARY_OUT, sep="\t", index=False, na_rep="NA")

    print(summary.to_string(index=False))
    print(f"Wrote QC summary: {SUMMARY_OUT}")
    print(f"Wrote start codon distribution: {START_CODON_DIST_OUT}")
    print(f"Wrote length summary: {LENGTH_SUMMARY_OUT}")
    print(f"Wrote probability summary: {PROBABILITY_SUMMARY_OUT}")
    print("06_qc_summary.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
