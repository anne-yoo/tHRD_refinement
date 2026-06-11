#!/usr/bin/env python3
"""Filter existing CPAT ORFs to a sampled noncoding negative ORF set."""

from __future__ import annotations

import sys

import pandas as pd

from cpat_negative_common import (
    CPAT_CUTOFF,
    CPAT_DIR,
    FASTA_DIR,
    RANDOM_SEED,
    SAMPLE_SIZE,
    TABLE_DIR,
    archive_script,
    clean_text,
    ensure_dirs,
    extract_transcript_id_from_cpat_orf_id,
    print_header,
    read_fasta_dict,
    read_tsv,
    write_fasta,
)


CPAT_PROB = CPAT_DIR / "CPAT2.ORF_prob.tsv"
CPAT_FASTA = CPAT_DIR / "CPAT2.ORF_seqs.fa"
NEGATIVE_TX = TABLE_DIR / "negative_noncoding_transcripts_no_positive_orf.tsv"

FROM_NEGATIVE_OUT = TABLE_DIR / "cpat_orfs.from_negative_noncoding_transcripts.tsv"
BEST_OUT = TABLE_DIR / "cpat_best_orf_per_negative_transcript.tsv"
NONCODING_OUT = TABLE_DIR / f"cpat_best_orf_per_negative_transcript.prob_lt_{CPAT_CUTOFF}.tsv"
SAMPLED_OUT = TABLE_DIR / f"cpat_negative_orfs.sampled_{SAMPLE_SIZE}.tsv"
SAMPLED_FASTA_OUT = FASTA_DIR / f"cpat_negative_orfs.sampled_{SAMPLE_SIZE}.fa"

CPAT_REQUIRED = [
    "ID",
    "mRNA",
    "ORF_strand",
    "ORF_frame",
    "ORF_start",
    "ORF_end",
    "ORF",
    "Fickett",
    "Hexamer",
    "Coding_prob",
]

OUTPUT_COLUMNS = [
    "ORF_id",
    "transcript_id",
    "mRNA_length",
    "ORF_strand",
    "ORF_frame",
    "ORF_start_transcript_1based",
    "ORF_end_transcript_1based",
    "orf_length_nt",
    "orf_length_aa",
    "Fickett",
    "Hexamer",
    "CPAT_coding_probability",
    "CPAT_prediction",
]


def cpat_prediction(probability: object) -> str:
    value = pd.to_numeric(pd.Series([probability]), errors="coerce").iloc[0]
    if pd.isna(value):
        return "unknown"
    return "noncoding" if float(value) < CPAT_CUTOFF else "coding"


def standardize_cpat(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "ORF_id": df["ID"].map(clean_text),
            "transcript_id": df["ID"].map(extract_transcript_id_from_cpat_orf_id),
            "mRNA_length": pd.to_numeric(df["mRNA"], errors="coerce").astype("Int64"),
            "ORF_strand": df["ORF_strand"].map(clean_text),
            "ORF_frame": df["ORF_frame"].map(clean_text),
            "ORF_start_transcript_1based": pd.to_numeric(df["ORF_start"], errors="coerce").astype("Int64"),
            "ORF_end_transcript_1based": pd.to_numeric(df["ORF_end"], errors="coerce").astype("Int64"),
            "Fickett": pd.to_numeric(df["Fickett"], errors="coerce"),
            "Hexamer": pd.to_numeric(df["Hexamer"], errors="coerce"),
            "CPAT_coding_probability": pd.to_numeric(df["Coding_prob"], errors="coerce"),
        }
    )
    out["orf_length_nt"] = (
        out["ORF_end_transcript_1based"] - out["ORF_start_transcript_1based"] + 1
    ).astype("Int64")
    out["orf_length_aa"] = (out["orf_length_nt"] // 3).astype("Int64")
    out["CPAT_prediction"] = out["CPAT_coding_probability"].map(cpat_prediction)
    return out[OUTPUT_COLUMNS]


def main() -> int:
    ensure_dirs()
    archive_script(__file__)
    print_header(
        "02_filter_cpat_orfs.py",
        inputs=[CPAT_PROB, CPAT_FASTA, NEGATIVE_TX],
        outputs=[FROM_NEGATIVE_OUT, BEST_OUT, NONCODING_OUT, SAMPLED_OUT, SAMPLED_FASTA_OUT],
    )
    print(f"CPAT cutoff={CPAT_CUTOFF}")
    print(f"sample_size={SAMPLE_SIZE}; random_seed={RANDOM_SEED}")

    cpat = read_tsv(CPAT_PROB, required=CPAT_REQUIRED)
    negative_tx = read_tsv(NEGATIVE_TX, required=["transcript_id"])
    negative_set = set(negative_tx["transcript_id"].dropna().astype(str))
    cpat_std = standardize_cpat(cpat)
    from_negative = cpat_std[cpat_std["transcript_id"].isin(negative_set)].copy()
    from_negative.to_csv(FROM_NEGATIVE_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {len(from_negative)} CPAT ORF row(s) from negative transcripts: {FROM_NEGATIVE_OUT}")

    sortable = from_negative.copy()
    sortable["CPAT_coding_probability_sort"] = sortable["CPAT_coding_probability"].fillna(-1)
    sortable["orf_length_nt_sort"] = sortable["orf_length_nt"].fillna(-1).astype(int)
    best = (
        sortable.sort_values(
            ["transcript_id", "CPAT_coding_probability_sort", "orf_length_nt_sort", "ORF_id"],
            ascending=[True, False, False, True],
        )
        .drop_duplicates("transcript_id", keep="first")
        .drop(columns=["CPAT_coding_probability_sort", "orf_length_nt_sort"])
        .reset_index(drop=True)
    )
    best.to_csv(BEST_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {len(best)} best CPAT ORF row(s): {BEST_OUT}")

    noncoding = best[best["CPAT_coding_probability"].lt(CPAT_CUTOFF)].copy()
    noncoding.to_csv(NONCODING_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {len(noncoding)} best CPAT ORF row(s) below cutoff: {NONCODING_OUT}")

    if len(noncoding) > SAMPLE_SIZE:
        sampled = noncoding.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).sort_values("ORF_id").reset_index(drop=True)
    else:
        sampled = noncoding.sort_values("ORF_id").reset_index(drop=True)
        print(
            f"WARNING: only {len(sampled)} ORFs remain below cutoff; requested {SAMPLE_SIZE}. Keeping all."
        )
    sampled.to_csv(SAMPLED_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {len(sampled)} sampled CPAT-negative ORF row(s): {SAMPLED_OUT}")

    seqs = read_fasta_dict(CPAT_FASTA)
    write_fasta(seqs, sampled["ORF_id"], SAMPLED_FASTA_OUT)
    print(f"Wrote sampled FASTA: {SAMPLED_FASTA_OUT}")
    print("02_filter_cpat_orfs.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
