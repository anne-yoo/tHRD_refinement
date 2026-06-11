#!/usr/bin/env python3
"""Create positive-pipeline-compatible metadata for CPAT negative ORFs."""

from __future__ import annotations

import sys

import pandas as pd

from cpat_negative_common import SAMPLE_SIZE, TABLE_DIR, archive_script, ensure_dirs, print_header, read_tsv


GENOMIC_IN = TABLE_DIR / f"cpat_negative_orfs.sampled_{SAMPLE_SIZE}.genomic.tsv"
COMPATIBLE_OUT = TABLE_DIR / "cpat_negative_orfs.combined_metadata_compatible.tsv"

REQUIRED = [
    "ORF_id",
    "transcript_id",
    "gene_id",
    "gene_name",
    "transcript_type",
    "chr",
    "start0",
    "end0",
    "strand",
    "start_codon",
    "orf_length_nt",
    "orf_length_aa",
    "Fickett",
    "Hexamer",
    "CPAT_coding_probability",
    "CPAT_prediction",
]

OUTPUT_COLUMNS = [
    "group",
    "primary_noncanonical_category",
    "ORF_id",
    "transcript_id",
    "chr",
    "start0",
    "end0",
    "strand",
    "ORF_type",
    "start_codon",
    "ORF_type2",
    "detected_samples",
    "n_detected_samples",
    "caller_union",
    "source",
    "gene_id",
    "gene_name",
    "transcript_type",
    "orf_length_nt",
    "orf_length_aa",
    "Fickett",
    "Hexamer",
    "CPAT_coding_probability",
    "CPAT_prediction",
]


def main() -> int:
    ensure_dirs()
    archive_script(__file__)
    print_header(
        "04_make_pipeline_compatible_metadata.py",
        inputs=[GENOMIC_IN],
        outputs=[COMPATIBLE_OUT],
    )

    df = read_tsv(GENOMIC_IN, required=REQUIRED)
    out = pd.DataFrame(
        {
            "group": "group4_CPAT_negative_noncoding_ORF",
            "primary_noncanonical_category": "CPAT_negative_noncoding_ORF",
            "ORF_id": df["ORF_id"],
            "transcript_id": df["transcript_id"],
            "chr": df["chr"],
            "start0": df["start0"],
            "end0": df["end0"],
            "strand": df["strand"],
            "ORF_type": "CPAT_negative_noncoding_ORF",
            "start_codon": df["start_codon"],
            "ORF_type2": "non-canonical",
            "detected_samples": "NA",
            "n_detected_samples": 0,
            "caller_union": "CPAT_negative",
            "source": "CPAT_negative_from_positive_ORF_zero_noncoding_transcripts",
            "gene_id": df["gene_id"],
            "gene_name": df["gene_name"],
            "transcript_type": df["transcript_type"],
            "orf_length_nt": df["orf_length_nt"],
            "orf_length_aa": df["orf_length_aa"],
            "Fickett": df["Fickett"],
            "Hexamer": df["Hexamer"],
            "CPAT_coding_probability": df["CPAT_coding_probability"],
            "CPAT_prediction": df["CPAT_prediction"],
        }
    )
    out[OUTPUT_COLUMNS].to_csv(COMPATIBLE_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {len(out)} compatible metadata row(s): {COMPATIBLE_OUT}")
    print("04_make_pipeline_compatible_metadata.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
