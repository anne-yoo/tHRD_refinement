#!/usr/bin/env python3
"""Extract start-codon sequence context for CPAT negative ORFs."""

from __future__ import annotations

import sys
from typing import Dict, Optional

import pandas as pd
from Bio import SeqIO

from cpat_negative_common import (
    GENOME_FA,
    LOG_DIR,
    PROGRESS_EVERY,
    TABLE_DIR,
    archive_script,
    clean_text,
    ensure_dirs,
    normalize_codon,
    print_header,
    read_tsv,
    resolve_chrom_key,
    reverse_complement,
)


INPUT_METADATA = TABLE_DIR / "cpat_negative_orfs.combined_metadata_compatible.tsv"
OUTPUT = TABLE_DIR / "cpat_negative_orfs.sequence_context_features.tsv"
MISMATCH_OUT = LOG_DIR / "cpat_negative_start_codon_mismatches.tsv"

REQUIRED = ["ORF_id", "group", "chr", "start0", "end0", "strand", "start_codon"]

OUTPUT_COLUMNS = [
    "ORF_id",
    "group",
    "chr",
    "start0",
    "end0",
    "strand",
    "start_codon",
    "extracted_start_codon",
    "start_codon_matches",
    "minus3_base",
    "plus4_base",
    "kozak_minus3_AG",
    "kozak_plus4_G",
    "strong_kozak",
    "start_context_10nt_up_10nt_down",
    "GC_start_window_20nt",
    "upstream_AUG_count_200nt",
    "upstream_AUG_density_200nt",
    "distance_to_nearest_upstream_AUG",
]


def get_seq(chrom_seq: str, start: int, end: int) -> Optional[str]:
    if start < 0 or end < start or end > len(chrom_seq):
        return None
    return chrom_seq[start:end]


def gc_content(seq: Optional[str]) -> object:
    if seq is None:
        return pd.NA
    upper = seq.upper()
    valid = [base for base in upper if base in {"A", "C", "G", "T"}]
    if not valid:
        return pd.NA
    return round(sum(1 for base in valid if base in {"G", "C"}) / len(valid), 6)


def count_atg(seq: Optional[str]) -> object:
    if seq is None:
        return pd.NA
    upper = seq.upper()
    return sum(1 for idx in range(0, max(0, len(upper) - 2)) if upper[idx : idx + 3] == "ATG")


def nearest_upstream_atg_distance(seq: Optional[str]) -> object:
    if seq is None:
        return pd.NA
    upper = seq.upper()
    positions = [idx for idx in range(0, max(0, len(upper) - 2)) if upper[idx : idx + 3] == "ATG"]
    if not positions:
        return pd.NA
    return len(upper) - max(positions)


def extract_features(row: pd.Series, chrom_seq: Optional[str]) -> Dict[str, object]:
    out: Dict[str, object] = {
        "ORF_id": clean_text(row.get("ORF_id", "")),
        "group": clean_text(row.get("group", "")),
        "chr": clean_text(row.get("chr", "")),
        "start0": pd.NA,
        "end0": pd.NA,
        "strand": clean_text(row.get("strand", "")),
        "start_codon": normalize_codon(row.get("start_codon", "")),
        "extracted_start_codon": pd.NA,
        "start_codon_matches": pd.NA,
        "minus3_base": pd.NA,
        "plus4_base": pd.NA,
        "kozak_minus3_AG": pd.NA,
        "kozak_plus4_G": pd.NA,
        "strong_kozak": pd.NA,
        "start_context_10nt_up_10nt_down": pd.NA,
        "GC_start_window_20nt": pd.NA,
        "upstream_AUG_count_200nt": pd.NA,
        "upstream_AUG_density_200nt": pd.NA,
        "distance_to_nearest_upstream_AUG": pd.NA,
    }

    try:
        start0 = int(float(row["start0"]))
        end0 = int(float(row["end0"]))
    except (TypeError, ValueError):
        return out
    strand = out["strand"]
    out["start0"] = start0
    out["end0"] = end0
    if chrom_seq is None or end0 <= start0 or strand not in {"+", "-"}:
        return out

    if strand == "+":
        codon_seq = get_seq(chrom_seq, start0, start0 + 3)
        minus3_seq = get_seq(chrom_seq, start0 - 3, start0 - 2)
        plus4_seq = get_seq(chrom_seq, start0 + 3, start0 + 4)
        context_seq = get_seq(chrom_seq, start0 - 10, start0 + 13)
        gc_window_seq = get_seq(chrom_seq, start0 - 10, start0 + 10)
        upstream_seq = get_seq(chrom_seq, start0 - 200, start0)
    else:
        codon_seq = reverse_complement(get_seq(chrom_seq, end0 - 3, end0))
        minus3_seq = reverse_complement(get_seq(chrom_seq, end0 + 2, end0 + 3))
        plus4_seq = reverse_complement(get_seq(chrom_seq, end0 - 4, end0 - 3))
        context_seq = reverse_complement(get_seq(chrom_seq, end0 - 13, end0 + 10))
        gc_window_seq = reverse_complement(get_seq(chrom_seq, end0 - 10, end0 + 10))
        upstream_seq = reverse_complement(get_seq(chrom_seq, end0, end0 + 200))

    if codon_seq is not None and len(codon_seq) == 3:
        extracted = codon_seq.upper()
        out["extracted_start_codon"] = extracted
        if out["start_codon"]:
            out["start_codon_matches"] = extracted == out["start_codon"]

    if minus3_seq is not None and len(minus3_seq) == 1:
        out["minus3_base"] = minus3_seq.upper()
        out["kozak_minus3_AG"] = minus3_seq.upper() in {"A", "G"}
    if plus4_seq is not None and len(plus4_seq) == 1:
        out["plus4_base"] = plus4_seq.upper()
        out["kozak_plus4_G"] = plus4_seq.upper() == "G"
    if pd.notna(out["kozak_minus3_AG"]) and pd.notna(out["kozak_plus4_G"]):
        out["strong_kozak"] = bool(out["kozak_minus3_AG"]) and bool(out["kozak_plus4_G"])

    if context_seq is not None and len(context_seq) == 23:
        out["start_context_10nt_up_10nt_down"] = context_seq.upper()
    if gc_window_seq is not None and len(gc_window_seq) == 20:
        out["GC_start_window_20nt"] = gc_content(gc_window_seq)
    if upstream_seq is not None and len(upstream_seq) == 200:
        atg_count = count_atg(upstream_seq)
        out["upstream_AUG_count_200nt"] = atg_count
        out["upstream_AUG_density_200nt"] = round(float(atg_count) / 200.0, 6)
        out["distance_to_nearest_upstream_AUG"] = nearest_upstream_atg_distance(upstream_seq)
    return out


def main() -> int:
    ensure_dirs()
    archive_script(__file__)
    print_header(
        "05_extract_sequence_context_for_cpat_negative.py",
        inputs=[INPUT_METADATA, GENOME_FA],
        outputs=[OUTPUT, MISMATCH_OUT],
    )

    df = read_tsv(INPUT_METADATA, required=REQUIRED).reset_index(drop=True)
    if not GENOME_FA.exists():
        raise SystemExit(f"Genome FASTA not found: {GENOME_FA}")

    fasta = SeqIO.index(str(GENOME_FA), "fasta")
    try:
        rows = [None] * len(df)
        processed = 0
        chrom_keys = {
            chrom: resolve_chrom_key(fasta, chrom)
            for chrom in sorted(df["chr"].dropna().astype(str).unique())
        }
        for chrom, group in df.groupby("chr", sort=False, dropna=False):
            chrom_text = clean_text(chrom)
            chrom_key = chrom_keys.get(chrom_text)
            if chrom_key is None:
                print(f"WARNING: chromosome not found in FASTA: {chrom_text}")
                chrom_seq = None
            else:
                print(f"Loading {chrom_text} from FASTA as {chrom_key} for {len(group)} ORF(s)")
                chrom_seq = str(fasta[chrom_key].seq)
            for idx, row in group.iterrows():
                rows[idx] = extract_features(row, chrom_seq)
                processed += 1
                if PROGRESS_EVERY > 0 and processed % PROGRESS_EVERY == 0:
                    print(f"Processed {processed}/{len(df)} ORF(s)")
    finally:
        fasta.close()

    out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out_df.to_csv(OUTPUT, sep="\t", index=False, na_rep="NA")
    mismatches = out_df[out_df["start_codon_matches"].eq(False)].copy()
    mismatches.to_csv(MISMATCH_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {len(out_df)} sequence-context row(s): {OUTPUT}")
    print(f"Wrote {len(mismatches)} start-codon mismatch row(s): {MISMATCH_OUT}")
    print("05_extract_sequence_context_for_cpat_negative.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
