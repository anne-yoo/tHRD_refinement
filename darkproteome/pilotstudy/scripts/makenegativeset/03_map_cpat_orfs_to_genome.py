#!/usr/bin/env python3
"""Map sampled CPAT negative ORFs from transcript to genomic coordinates."""

from __future__ import annotations

import sys
from typing import Dict, List

import pandas as pd
from Bio import SeqIO

from cpat_negative_common import (
    BASE_DIR,
    BED_DIR,
    CPAT_DIR,
    GENCODE_GTF,
    GENOME_FA,
    LOG_DIR,
    RANDOM_SEED,
    SAMPLE_SIZE,
    TABLE_DIR,
    archive_script,
    clean_text,
    ensure_dirs,
    extract_transcript_orf_sequence,
    genomic_interval_from_transcript_orf,
    parse_gtf_exon_models,
    print_header,
    read_fasta_dict,
    read_tsv,
    resolve_chrom_key,
    write_bed,
)


SAMPLED_IN = TABLE_DIR / f"cpat_negative_orfs.sampled_{SAMPLE_SIZE}.tsv"
CPAT_FASTA = CPAT_DIR / "CPAT2.ORF_seqs.fa"
GENOMIC_OUT = TABLE_DIR / f"cpat_negative_orfs.sampled_{SAMPLE_SIZE}.genomic.tsv"
BED_OUT = BED_DIR / "group4_CPAT_negative_noncoding_ORF.bed"
VALIDATION_OUT = LOG_DIR / "cpat_orf_genomic_mapping_validation.tsv"
MISMATCH_OUT = LOG_DIR / "cpat_orf_genomic_mapping_mismatches.tsv"
FAILURE_OUT = LOG_DIR / "cpat_orf_genomic_mapping_failures.tsv"

REQUIRED = [
    "ORF_id",
    "transcript_id",
    "ORF_start_transcript_1based",
    "ORF_end_transcript_1based",
    "orf_length_nt",
    "orf_length_aa",
    "Fickett",
    "Hexamer",
    "CPAT_coding_probability",
    "CPAT_prediction",
]

OUTPUT_COLUMNS = [
    "ORF_id",
    "transcript_id",
    "gene_id",
    "gene_name",
    "transcript_type",
    "chr",
    "start0",
    "end0",
    "strand",
    "ORF_start_transcript_1based",
    "ORF_end_transcript_1based",
    "start_codon",
    "orf_length_nt",
    "orf_length_aa",
    "Fickett",
    "Hexamer",
    "CPAT_coding_probability",
    "CPAT_prediction",
]


def normalize_seq(value: str) -> str:
    return clean_text(value).upper().replace("U", "T")


def validation_subset(df: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for strand in ["+", "-"]:
        sub = df[df["strand"].eq(strand)]
        if sub.empty:
            continue
        n = min(10, len(sub))
        pieces.append(sub.sample(n=n, random_state=RANDOM_SEED))
    if not pieces:
        return df.head(0)
    return pd.concat(pieces, ignore_index=True)


def main() -> int:
    ensure_dirs()
    archive_script(__file__)
    print_header(
        "03_map_cpat_orfs_to_genome.py",
        inputs=[GENCODE_GTF, SAMPLED_IN, GENOME_FA, CPAT_FASTA],
        outputs=[GENOMIC_OUT, BED_OUT, VALIDATION_OUT, MISMATCH_OUT, FAILURE_OUT],
    )

    cpat = read_tsv(SAMPLED_IN, required=REQUIRED)
    models = parse_gtf_exon_models(GENCODE_GTF)
    cpat_seqs = read_fasta_dict(CPAT_FASTA)
    print(f"Loaded {len(models)} transcript exon model(s)")
    print(f"Loaded {len(cpat_seqs)} CPAT ORF FASTA sequence(s)")

    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    for idx, row in cpat.iterrows():
        if idx and idx % 10000 == 0:
            print(f"Mapped {idx}/{len(cpat)} CPAT ORF(s)")
        orf_id = clean_text(row["ORF_id"])
        transcript_id = clean_text(row["transcript_id"])
        model = models.get(transcript_id)
        if model is None:
            failures.append(
                {
                    "ORF_id": orf_id,
                    "transcript_id": transcript_id,
                    "status": "missing_transcript_model",
                    "message": "transcript_id not found in GTF exon records",
                }
            )
            continue
        try:
            start_tx = int(float(row["ORF_start_transcript_1based"]))
            end_tx = int(float(row["ORF_end_transcript_1based"]))
            start0, end0 = genomic_interval_from_transcript_orf(model, start_tx, end_tx)
        except Exception as exc:  # noqa: BLE001 - report and continue.
            failures.append(
                {
                    "ORF_id": orf_id,
                    "transcript_id": transcript_id,
                    "status": "coordinate_mapping_failed",
                    "message": str(exc),
                }
            )
            continue

        cpat_seq = normalize_seq(cpat_seqs.get(orf_id, ""))
        start_codon = cpat_seq[:3] if len(cpat_seq) >= 3 else ""
        rows.append(
            {
                "ORF_id": orf_id,
                "transcript_id": transcript_id,
                "gene_id": model.gene_id,
                "gene_name": model.gene_name,
                "transcript_type": model.transcript_type,
                "chr": model.chrom,
                "start0": start0,
                "end0": end0,
                "strand": model.strand,
                "ORF_start_transcript_1based": start_tx,
                "ORF_end_transcript_1based": end_tx,
                "start_codon": start_codon,
                "orf_length_nt": row["orf_length_nt"],
                "orf_length_aa": row["orf_length_aa"],
                "Fickett": row["Fickett"],
                "Hexamer": row["Hexamer"],
                "CPAT_coding_probability": row["CPAT_coding_probability"],
                "CPAT_prediction": row["CPAT_prediction"],
            }
        )

    out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out_df.to_csv(GENOMIC_OUT, sep="\t", index=False, na_rep="NA")
    write_bed(out_df, BED_OUT)
    pd.DataFrame(failures, columns=["ORF_id", "transcript_id", "status", "message"]).to_csv(
        FAILURE_OUT, sep="\t", index=False, na_rep="NA"
    )
    print(f"Wrote {len(out_df)} mapped genomic ORF row(s): {GENOMIC_OUT}")
    print(f"Wrote BED: {BED_OUT}")
    print(f"Wrote {len(failures)} mapping failure row(s): {FAILURE_OUT}")

    validation_rows: List[Dict[str, object]] = []
    fasta = SeqIO.index(str(GENOME_FA), "fasta")
    try:
        for _, row in validation_subset(out_df).iterrows():
            orf_id = clean_text(row["ORF_id"])
            transcript_id = clean_text(row["transcript_id"])
            model = models[transcript_id]
            chrom_key = resolve_chrom_key(fasta, model.chrom)
            status = "validated"
            message = ""
            extracted = ""
            if chrom_key is None:
                status = "missing_chromosome"
                message = f"{model.chrom} not found in genome FASTA"
            else:
                chrom_seq = str(fasta[chrom_key].seq)
                try:
                    extracted = extract_transcript_orf_sequence(
                        model,
                        int(row["ORF_start_transcript_1based"]),
                        int(row["ORF_end_transcript_1based"]),
                        chrom_seq,
                    )
                except Exception as exc:  # noqa: BLE001
                    status = "validation_extraction_failed"
                    message = str(exc)

            cpat_seq = normalize_seq(cpat_seqs.get(orf_id, ""))
            match = bool(extracted and cpat_seq and extracted == cpat_seq)
            if not cpat_seq:
                status = "missing_cpat_fasta_sequence"
                match = False
            validation_rows.append(
                {
                    "ORF_id": orf_id,
                    "transcript_id": transcript_id,
                    "chr": row["chr"],
                    "start0": row["start0"],
                    "end0": row["end0"],
                    "strand": row["strand"],
                    "ORF_start_transcript_1based": row["ORF_start_transcript_1based"],
                    "ORF_end_transcript_1based": row["ORF_end_transcript_1based"],
                    "genome_oriented_sequence_length": len(extracted),
                    "cpat_sequence_length": len(cpat_seq),
                    "sequence_matches_cpat": match,
                    "status": status,
                    "message": message,
                    "genome_seq_first20": extracted[:20],
                    "cpat_seq_first20": cpat_seq[:20],
                }
            )
    finally:
        fasta.close()

    validation_df = pd.DataFrame(validation_rows)
    validation_df.to_csv(VALIDATION_OUT, sep="\t", index=False, na_rep="NA")
    mismatches = validation_df[~validation_df["sequence_matches_cpat"].eq(True)].copy()
    mismatches.to_csv(MISMATCH_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote validation rows: {VALIDATION_OUT}")
    print(f"Wrote {len(mismatches)} validation mismatch row(s): {MISMATCH_OUT}")
    print("03_map_cpat_orfs_to_genome.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
