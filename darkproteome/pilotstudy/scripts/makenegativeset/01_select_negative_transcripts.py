#!/usr/bin/env python3
"""Select noncoding GENCODE transcripts with zero positive translated ORFs."""

from __future__ import annotations

import sys

import pandas as pd

from cpat_negative_common import (
    GENCODE_GTF,
    INPUT_DIR,
    LOG_DIR,
    TABLE_DIR,
    archive_script,
    clean_text,
    ensure_dirs,
    infer_transcript_ids_from_table,
    parse_gtf_transcripts,
    print_header,
)


POSITIVE_ORFS = INPUT_DIR / "tables" / "pancreas.translated_orfs.unique.tsv"
TRANSCRIPTS_OUT = TABLE_DIR / "gencode_transcripts.tsv"
NONCODING_OUT = TABLE_DIR / "gencode_noncoding_transcripts.tsv"
POSITIVE_TX_OUT = TABLE_DIR / "positive_orf_transcripts.tsv"
NEGATIVE_OUT = TABLE_DIR / "negative_noncoding_transcripts_no_positive_orf.tsv"
SUMMARY_OUT = LOG_DIR / "negative_transcript_selection_summary.tsv"


def main() -> int:
    ensure_dirs()
    archive_script(__file__)
    print_header(
        "01_select_negative_transcripts.py",
        inputs=[GENCODE_GTF, POSITIVE_ORFS],
        outputs=[TRANSCRIPTS_OUT, NONCODING_OUT, POSITIVE_TX_OUT, NEGATIVE_OUT, SUMMARY_OUT],
    )

    transcripts = parse_gtf_transcripts(GENCODE_GTF)
    transcripts.to_csv(TRANSCRIPTS_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {len(transcripts)} GENCODE transcript row(s): {TRANSCRIPTS_OUT}")

    noncoding = transcripts[transcripts["is_noncoding"].eq(True)].copy()
    noncoding.to_csv(NONCODING_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {len(noncoding)} noncoding transcript row(s): {NONCODING_OUT}")

    if not POSITIVE_ORFS.exists():
        raise SystemExit(f"Positive ORF table not found: {POSITIVE_ORFS}")
    positive_df = pd.read_csv(POSITIVE_ORFS, sep="\t", dtype=str)
    positive_ids = infer_transcript_ids_from_table(positive_df).map(clean_text)
    positive_tx = (
        pd.DataFrame({"transcript_id": sorted({tx for tx in positive_ids if tx})})
        .merge(
            transcripts[
                ["transcript_id", "gene_id", "gene_name", "transcript_type", "chr", "strand"]
            ],
            on="transcript_id",
            how="left",
        )
    )
    positive_tx.to_csv(POSITIVE_TX_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {len(positive_tx)} positive-ORF transcript row(s): {POSITIVE_TX_OUT}")

    positive_set = set(positive_tx["transcript_id"].dropna().astype(str))
    noncoding_positive = noncoding[noncoding["transcript_id"].isin(positive_set)].copy()
    negative = noncoding[~noncoding["transcript_id"].isin(positive_set)].copy()
    negative.to_csv(NEGATIVE_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {len(negative)} negative noncoding transcript row(s): {NEGATIVE_OUT}")

    summary = pd.DataFrame(
        [
            {"metric": "total_GENCODE_transcripts", "value": len(transcripts)},
            {"metric": "total_noncoding_transcripts", "value": len(noncoding)},
            {"metric": "positive_ORF_transcripts", "value": len(positive_tx)},
            {
                "metric": "noncoding_transcripts_with_ge1_positive_ORF",
                "value": len(noncoding_positive),
            },
            {
                "metric": "noncoding_transcripts_with_0_positive_ORF",
                "value": len(negative),
            },
        ]
    )
    summary.to_csv(SUMMARY_OUT, sep="\t", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote summary: {SUMMARY_OUT}")
    print("01_select_negative_transcripts.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
