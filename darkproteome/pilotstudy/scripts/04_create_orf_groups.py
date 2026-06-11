#!/usr/bin/env python3
"""Create canonical, cryptic, and noncanonical subtype BED files."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Dict

import pandas as pd


OUT_DIR = Path(
    os.environ.get("OUT_DIR", "/home/jiye/jiye/darkproteome/ORFstudy/pilot/pancreas8samples")
)
INPUT_UNIQUE = OUT_DIR / "tables" / "pancreas.translated_orfs.unique.tsv"
BED_DIR = OUT_DIR / "bed"
COMBINED_OUT = OUT_DIR / "tables" / "orf_groups.combined_metadata.tsv"
SOURCE_NAME = "Pancreas.4caller.merged.2caller.tsv"

GROUP_BEDS: Dict[str, str] = {
    "group1_canonical_translated_ORF": "group1_canonical_translated_ORF.bed",
    "group2_translated_AUG_cryptic_ORF": "group2_translated_AUG_cryptic_ORF.bed",
    "group3_translated_nonAUG_cryptic_ORF": "group3_translated_nonAUG_cryptic_ORF.bed",
}

CATEGORY_BEDS: Dict[str, str] = {
    "lncRNA_or_ncRNA_ORF": "noncanonical_lncRNA_or_ncRNA_ORF.bed",
    "five_prime_overlap_uORF": "noncanonical_five_prime_overlap_uORF.bed",
    "five_prime_uORF": "noncanonical_five_prime_uORF.bed",
    "three_prime_overlap_dORF": "noncanonical_three_prime_overlap_dORF.bed",
    "three_prime_dORF": "noncanonical_three_prime_dORF.bed",
    "out_of_frame_or_internal_ORF": "noncanonical_out_of_frame_or_internal_ORF.bed",
    "pseudogene_ORF": "noncanonical_pseudogene_ORF.bed",
    "other_noncanonical_ORF": "noncanonical_other_noncanonical_ORF.bed",
}

REQUIRED_COLUMNS = [
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
]


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


def has_any(text: str, needles) -> bool:
    return any(needle in text for needle in needles)


def primary_noncanonical_category(orf_type: object) -> str:
    raw = clean_text(orf_type)
    lower = raw.lower()

    if has_any(lower, ["lncrna", "ncrna", "varrna-orf", "novel"]):
        return "lncRNA_or_ncRNA_ORF"
    if "overlap_uorf" in lower or ("5'utr" in lower and "cdsframeoverlap" in lower):
        return "five_prime_overlap_uORF"
    if has_any(lower, ["uorf", "uoorf", "5'utr"]):
        return "five_prime_uORF"
    if "overlap_dorf" in lower or ("3'utr" in lower and "cdsframeoverlap" in lower):
        return "three_prime_overlap_dORF"
    if has_any(lower, ["dorf", "doorf", "3'utr"]):
        return "three_prime_dORF"
    if has_any(lower, ["internal", "iorf", "intorf", "out-of-frame", "out_of_frame"]):
        return "out_of_frame_or_internal_ORF"
    if "pseudogene" in lower:
        return "pseudogene_ORF"
    return "other_noncanonical_ORF"


def require_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing required input column(s): {', '.join(missing)}")


def write_bed(df: pd.DataFrame, path: Path) -> None:
    bed = df[["chr", "start0", "end0", "ORF_id", "strand"]].copy()
    bed.insert(4, "score", 0)
    bed[["chr", "start0", "end0", "ORF_id", "score", "strand"]].to_csv(
        path, sep="\t", index=False, header=False
    )


def main() -> int:
    print("04_create_orf_groups.py")
    print(f"input={INPUT_UNIQUE}")
    print(f"bed_dir={BED_DIR}")
    print(f"combined_out={COMBINED_OUT}")

    if not INPUT_UNIQUE.exists():
        raise SystemExit(f"Input unique ORF table not found: {INPUT_UNIQUE}")

    BED_DIR.mkdir(parents=True, exist_ok=True)
    COMBINED_OUT.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_UNIQUE, sep="\t", dtype=str)
    require_columns(df)
    df["start0"] = pd.to_numeric(df["start0"], errors="raise").astype(int)
    df["end0"] = pd.to_numeric(df["end0"], errors="raise").astype(int)
    df["n_detected_samples"] = pd.to_numeric(df["n_detected_samples"], errors="coerce").astype("Int64")
    df["start_codon"] = df["start_codon"].map(normalize_codon)

    label = df["ORF_type2"].map(lambda x: clean_text(x).lower())
    is_canonical = label.eq("canonical")
    is_noncanonical = label.eq("non-canonical")
    is_aug = df["start_codon"].eq("ATG")

    df["group"] = ""
    df.loc[is_canonical, "group"] = "group1_canonical_translated_ORF"
    df.loc[is_noncanonical & is_aug, "group"] = "group2_translated_AUG_cryptic_ORF"
    df.loc[is_noncanonical & ~is_aug, "group"] = "group3_translated_nonAUG_cryptic_ORF"
    df["primary_noncanonical_category"] = "canonical_ORF"
    df.loc[is_noncanonical, "primary_noncanonical_category"] = df.loc[
        is_noncanonical, "ORF_type"
    ].map(primary_noncanonical_category)

    grouped_df = df[df["group"].ne("")].copy()
    grouped_df["source"] = SOURCE_NAME

    combined_cols = [
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
    ]
    grouped_df[combined_cols].to_csv(COMBINED_OUT, sep="\t", index=False, na_rep="NA")

    for group_name, filename in GROUP_BEDS.items():
        subset = grouped_df[grouped_df["group"].eq(group_name)]
        write_bed(subset, BED_DIR / filename)
        print(f"Wrote {len(subset)} row(s): {BED_DIR / filename}")

    noncanonical = grouped_df[grouped_df["ORF_type2"].map(lambda x: clean_text(x).lower()).eq("non-canonical")]
    for category, filename in CATEGORY_BEDS.items():
        subset = noncanonical[noncanonical["primary_noncanonical_category"].eq(category)]
        write_bed(subset, BED_DIR / filename)
        print(f"Wrote {len(subset)} row(s): {BED_DIR / filename}")

    skipped = len(df) - len(grouped_df)
    if skipped:
        print(f"Skipped {skipped} ORF(s) with ORF_type2 outside canonical/non-canonical")
    print(f"Wrote combined metadata: {COMBINED_OUT}")
    print("04_create_orf_groups.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

