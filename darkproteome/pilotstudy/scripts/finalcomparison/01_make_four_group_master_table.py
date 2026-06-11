#!/usr/bin/env python3
"""Merge positive and CPAT-negative ORFs into one four-group metadata table."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from fc_common import (
    INPUT_DIR,
    NEGATIVE_GROUP,
    NEG_DIR,
    POSITIVE_GROUP_MAP,
    TABLE_DIR,
    archive_script,
    clean_text,
    ensure_dirs,
    normalize_codon,
    plot_group_from_group,
    print_paths,
    read_tsv,
)


POSITIVE_METADATA = INPUT_DIR / "tables" / "orf_groups.combined_metadata.tsv"
NEGATIVE_METADATA = NEG_DIR / "tables" / "cpat_negative_orfs.combined_metadata_compatible.tsv"
MASTER_OUT = TABLE_DIR / "four_group_orf_metadata.tsv"

OUTPUT_COLUMNS = [
    "plot_group",
    "ORF_id",
    "transcript_id",
    "chr",
    "start0",
    "end0",
    "strand",
    "start_codon",
    "ORF_type",
    "ORF_type2",
    "primary_noncanonical_category",
    "orf_length_nt",
    "orf_length_aa",
    "source",
    "CPAT_coding_probability",
    "CPAT_prediction",
]


def standardize(df: pd.DataFrame, negative: bool = False) -> pd.DataFrame:
    out = df.copy()
    if negative:
        out["plot_group"] = NEGATIVE_GROUP
    else:
        out["plot_group"] = out["group"].map(plot_group_from_group)
    for col in ["CPAT_coding_probability", "CPAT_prediction", "source", "orf_length_nt", "orf_length_aa"]:
        if col not in out.columns:
            out[col] = pd.NA
    if "start_codon" in out.columns:
        out["start_codon"] = out["start_codon"].map(normalize_codon)
    out["start0"] = pd.to_numeric(out["start0"], errors="coerce")
    out["end0"] = pd.to_numeric(out["end0"], errors="coerce")
    if out["orf_length_nt"].isna().all() or pd.to_numeric(out["orf_length_nt"], errors="coerce").isna().all():
        out["orf_length_nt"] = out["end0"] - out["start0"]
    out["orf_length_nt"] = pd.to_numeric(out["orf_length_nt"], errors="coerce")
    if out["orf_length_aa"].isna().all() or pd.to_numeric(out["orf_length_aa"], errors="coerce").isna().all():
        out["orf_length_aa"] = np.floor(out["orf_length_nt"] / 3)
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[OUTPUT_COLUMNS].copy()


def main() -> int:
    ensure_dirs()
    archive_script(__file__)
    print_paths("01_make_four_group_master_table.py", inputs=[POSITIVE_METADATA, NEGATIVE_METADATA], outputs=[MASTER_OUT])

    positive = read_tsv(POSITIVE_METADATA, required=["group", "ORF_id", "transcript_id", "chr", "start0", "end0", "strand"])
    negative = read_tsv(NEGATIVE_METADATA, required=["ORF_id", "transcript_id", "chr", "start0", "end0", "strand"])
    master = pd.concat([standardize(positive), standardize(negative, negative=True)], ignore_index=True)
    master["ORF_id"] = master["ORF_id"].map(clean_text)
    master = master[master["ORF_id"].ne("")]
    master.to_csv(MASTER_OUT, sep="\t", index=False, na_rep="NA")
    print(master["plot_group"].value_counts().reindex(list(POSITIVE_GROUP_MAP.values()) + [NEGATIVE_GROUP]).fillna(0).astype(int).to_string())
    print(f"Wrote {len(master)} ORF row(s): {MASTER_OUT}")
    print("01_make_four_group_master_table.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
