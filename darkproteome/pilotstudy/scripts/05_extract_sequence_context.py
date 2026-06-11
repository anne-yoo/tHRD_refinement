#!/usr/bin/env python3
"""Extract strand-aware genomic sequence context around ORF start codons."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from Bio import SeqIO


OUT_DIR = Path(
    os.environ.get("OUT_DIR", "/home/jiye/jiye/darkproteome/ORFstudy/pilot/pancreas8samples")
)
GENOME_FA = Path(
    os.environ.get("GENOME_FA", "/home/jiye/jiye/darkproteome/ORFstudy/pilot/hg38.fa")
)
PROGRESS_EVERY = int(os.environ.get("PROGRESS_EVERY", "10000"))
INPUT_GROUPS = OUT_DIR / "tables" / "orf_groups.combined_metadata.tsv"
OUTPUT = OUT_DIR / "tables" / "orf_sequence_context_features.tsv"

REQUIRED_COLUMNS = [
    "ORF_id",
    "group",
    "chr",
    "start0",
    "end0",
    "strand",
    "start_codon",
]

REVCOMP_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "na", "n/a", "."}:
        return ""
    return text


def normalize_codon(value: object) -> str:
    text = clean_text(value).upper().replace("U", "T")
    return re.split(r"[\s,;|/]+", text)[0] if text else ""


def reverse_complement(seq: Optional[str]) -> Optional[str]:
    if seq is None:
        return None
    return seq.translate(REVCOMP_TABLE)[::-1]


def gc_content(seq: Optional[str]) -> object:
    if seq is None:
        return pd.NA
    upper = seq.upper()
    valid = [base for base in upper if base in {"A", "C", "G", "T"}]
    if not valid:
        return pd.NA
    gc = sum(1 for base in valid if base in {"G", "C"})
    return round(gc / len(valid), 6)


def resolve_chrom_key(fasta: Dict[str, object], chrom: str) -> Optional[str]:
    if chrom in fasta:
        return chrom
    alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
    if alt in fasta:
        return alt
    return None


def get_seq(chrom_seq: str, start: int, end: int) -> Optional[str]:
    if start < 0 or end < start:
        return None
    if end > len(chrom_seq):
        return None
    if start == end:
        return ""
    return chrom_seq[start:end]


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


def require_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing required input column(s): {', '.join(missing)}")


def extract_features(row: pd.Series, chrom_seq: Optional[str]) -> dict:
    orf_id = clean_text(row.get("ORF_id", ""))
    group = clean_text(row.get("group", ""))
    chrom = clean_text(row.get("chr", ""))
    strand = clean_text(row.get("strand", ""))
    start_codon = normalize_codon(row.get("start_codon", ""))

    out = {
        "ORF_id": orf_id,
        "group": group,
        "chr": chrom,
        "start0": pd.NA,
        "end0": pd.NA,
        "strand": strand,
        "start_codon": start_codon,
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
        start0 = int(row["start0"])
        end0 = int(row["end0"])
    except (TypeError, ValueError):
        return out

    out["start0"] = start0
    out["end0"] = end0
    if end0 <= start0 or strand not in {"+", "-"}:
        return out

    if chrom_seq is None:
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
        if start_codon:
            out["start_codon_matches"] = extracted == start_codon

    if minus3_seq is not None and len(minus3_seq) == 1:
        out["minus3_base"] = minus3_seq
        out["kozak_minus3_AG"] = minus3_seq.upper() in {"A", "G"}
    if plus4_seq is not None and len(plus4_seq) == 1:
        out["plus4_base"] = plus4_seq
        out["kozak_plus4_G"] = plus4_seq.upper() == "G"

    if out["kozak_minus3_AG"] is not pd.NA and out["kozak_plus4_G"] is not pd.NA:
        out["strong_kozak"] = bool(out["kozak_minus3_AG"]) and bool(out["kozak_plus4_G"])

    if context_seq is not None and len(context_seq) == 23:
        out["start_context_10nt_up_10nt_down"] = context_seq

    out["GC_start_window_20nt"] = gc_content(gc_window_seq if gc_window_seq and len(gc_window_seq) == 20 else None)

    if upstream_seq is not None and len(upstream_seq) == 200:
        atg_count = count_atg(upstream_seq)
        out["upstream_AUG_count_200nt"] = atg_count
        out["upstream_AUG_density_200nt"] = round(float(atg_count) / 200.0, 6)
        out["distance_to_nearest_upstream_AUG"] = nearest_upstream_atg_distance(upstream_seq)

    return out


def main() -> int:
    print("05_extract_sequence_context.py")
    print(f"input={INPUT_GROUPS}")
    print(f"genome_fa={GENOME_FA}")
    print(f"output={OUTPUT}")

    if not INPUT_GROUPS.exists():
        raise SystemExit(f"Input group metadata not found: {INPUT_GROUPS}")
    if not GENOME_FA.exists():
        raise SystemExit(f"Genome FASTA not found: {GENOME_FA}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_GROUPS, sep="\t", dtype=str).reset_index(drop=True)
    require_columns(df)

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

        if processed != len(df):
            raise RuntimeError(f"Processed {processed} row(s), expected {len(df)}")
    finally:
        fasta.close()

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT, sep="\t", index=False, na_rep="NA")

    valid = out_df["extracted_start_codon"].notna().sum()
    print(f"Wrote {len(out_df)} row(s); valid extracted start codons: {valid}")
    print("05_extract_sequence_context.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
