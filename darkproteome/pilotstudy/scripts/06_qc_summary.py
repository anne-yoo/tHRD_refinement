#!/usr/bin/env python3
"""Summarize preprocessing outputs for the pancreas ORF pilot study."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import pandas as pd


OUT_DIR = Path(
    os.environ.get("OUT_DIR", "/home/jiye/jiye/darkproteome/ORFstudy/pilot/pancreas8samples")
)

METADATA = OUT_DIR / "tables" / "pancreas8samples.metadata.tsv"
SETUP_LOG = OUT_DIR / "logs" / "setup_metadata.log"
GROUPS = OUT_DIR / "tables" / "orf_groups.combined_metadata.tsv"
SEQUENCE_CONTEXT = OUT_DIR / "tables" / "orf_sequence_context_features.tsv"
INCONSISTENCY_WARNINGS = OUT_DIR / "logs" / "orf_inconsistency_warnings.tsv"

QC_SUMMARY_OUT = OUT_DIR / "tables" / "preprocessing_qc_summary.tsv"
START_CODON_OUT = OUT_DIR / "tables" / "start_codon_distribution_by_group.tsv"
KOZAK_OUT = OUT_DIR / "tables" / "kozak_summary_by_group.tsv"
LENGTH_OUT = OUT_DIR / "tables" / "orf_length_summary_by_group.tsv"


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t", dtype=str)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().map({"true": True, "false": False})


def add_summary(rows: List[dict], section: str, metric: str, value: object) -> None:
    rows.append({"section": section, "metric": metric, "value": value})


def count_setup_warnings() -> tuple[int, int]:
    if not SETUP_LOG.exists():
        return 0, 0
    warning_count = 0
    bam_bai_warning_count = 0
    with SETUP_LOG.open() as handle:
        for line in handle:
            if "WARNING" not in line:
                continue
            warning_count += 1
            upper = line.upper()
            if "BAM" in upper or "BAI" in upper:
                bam_bai_warning_count += 1
    return warning_count, bam_bai_warning_count


def write_start_codon_distribution(groups: pd.DataFrame) -> None:
    columns = ["group", "start_codon", "n_orfs", "fraction_within_group"]
    if groups.empty or not {"group", "start_codon"}.issubset(groups.columns):
        pd.DataFrame(columns=columns).to_csv(START_CODON_OUT, sep="\t", index=False)
        return
    counts = groups.groupby(["group", "start_codon"], dropna=False).size().reset_index(name="n_orfs")
    totals = counts.groupby("group")["n_orfs"].transform("sum")
    counts["fraction_within_group"] = (counts["n_orfs"] / totals).round(6)
    counts[columns].to_csv(START_CODON_OUT, sep="\t", index=False)


def write_kozak_summary(seq: pd.DataFrame) -> None:
    columns = [
        "group",
        "n_orfs",
        "n_valid_extracted_start_codon",
        "n_start_codon_matches_true",
        "n_start_codon_matches_false",
        "n_kozak_minus3_AG_true",
        "n_kozak_plus4_G_true",
        "n_strong_kozak_true",
        "fraction_strong_kozak",
    ]
    if seq.empty or "group" not in seq.columns:
        pd.DataFrame(columns=columns).to_csv(KOZAK_OUT, sep="\t", index=False)
        return

    rows = []
    for group, sub in seq.groupby("group", dropna=False):
        match = bool_series(sub.get("start_codon_matches", pd.Series(index=sub.index, dtype=str)))
        minus3 = bool_series(sub.get("kozak_minus3_AG", pd.Series(index=sub.index, dtype=str)))
        plus4 = bool_series(sub.get("kozak_plus4_G", pd.Series(index=sub.index, dtype=str)))
        strong = bool_series(sub.get("strong_kozak", pd.Series(index=sub.index, dtype=str)))
        n_orfs = len(sub)
        rows.append(
            {
                "group": group,
                "n_orfs": n_orfs,
                "n_valid_extracted_start_codon": sub.get(
                    "extracted_start_codon", pd.Series(index=sub.index, dtype=str)
                ).replace("NA", pd.NA).notna().sum(),
                "n_start_codon_matches_true": int(match.eq(True).sum()),
                "n_start_codon_matches_false": int(match.eq(False).sum()),
                "n_kozak_minus3_AG_true": int(minus3.eq(True).sum()),
                "n_kozak_plus4_G_true": int(plus4.eq(True).sum()),
                "n_strong_kozak_true": int(strong.eq(True).sum()),
                "fraction_strong_kozak": round(float(strong.eq(True).sum()) / n_orfs, 6) if n_orfs else 0,
            }
        )
    pd.DataFrame(rows, columns=columns).to_csv(KOZAK_OUT, sep="\t", index=False)


def write_length_summary(groups: pd.DataFrame) -> None:
    columns = ["group", "n_orfs", "min", "q1", "median", "mean", "q3", "max"]
    if groups.empty or not {"group", "start0", "end0"}.issubset(groups.columns):
        pd.DataFrame(columns=columns).to_csv(LENGTH_OUT, sep="\t", index=False)
        return

    work = groups.copy()
    work["start0"] = pd.to_numeric(work["start0"], errors="coerce")
    work["end0"] = pd.to_numeric(work["end0"], errors="coerce")
    work["orf_length_nt"] = work["end0"] - work["start0"]
    work = work[work["orf_length_nt"].notna()]

    rows = []
    for group, sub in work.groupby("group", dropna=False):
        lengths = sub["orf_length_nt"]
        rows.append(
            {
                "group": group,
                "n_orfs": int(lengths.count()),
                "min": int(lengths.min()),
                "q1": round(float(lengths.quantile(0.25)), 3),
                "median": round(float(lengths.median()), 3),
                "mean": round(float(lengths.mean()), 3),
                "q3": round(float(lengths.quantile(0.75)), 3),
                "max": int(lengths.max()),
            }
        )
    pd.DataFrame(rows, columns=columns).to_csv(LENGTH_OUT, sep="\t", index=False)


def main() -> int:
    print("06_qc_summary.py")
    print(f"OUT_DIR={OUT_DIR}")
    print(f"summary_out={QC_SUMMARY_OUT}")
    print(f"start_codon_out={START_CODON_OUT}")
    print(f"kozak_out={KOZAK_OUT}")
    print(f"length_out={LENGTH_OUT}")

    QC_SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)

    metadata = read_tsv(METADATA)
    groups = read_tsv(GROUPS)
    seq = read_tsv(SEQUENCE_CONTEXT)
    inconsistency = read_tsv(INCONSISTENCY_WARNINGS)
    warning_count, bam_bai_warning_count = count_setup_warnings()

    rows: List[dict] = []
    add_summary(rows, "inputs", "number_of_samples", len(metadata) if not metadata.empty else 0)
    add_summary(rows, "bigwig", "number_of_bigwig_files_generated", len(list((OUT_DIR / "bigwig").glob("*.bw"))))
    add_summary(rows, "warnings", "setup_metadata_warning_count", warning_count)
    add_summary(rows, "warnings", "missing_BAM_BAI_warning_count", bam_bai_warning_count)
    add_summary(rows, "warnings", "inconsistent_ORF_id_warning_count", len(inconsistency))

    if not groups.empty and "group" in groups.columns:
        add_summary(rows, "orfs", "number_of_grouped_ORFs", len(groups))
        for group, count in groups["group"].value_counts().sort_index().items():
            add_summary(rows, "orfs_per_group", group, int(count))
    else:
        add_summary(rows, "orfs", "number_of_grouped_ORFs", 0)

    if not groups.empty and "primary_noncanonical_category" in groups.columns:
        noncanonical = groups[groups["primary_noncanonical_category"].ne("canonical_ORF")]
        for category, count in noncanonical["primary_noncanonical_category"].value_counts().sort_index().items():
            add_summary(rows, "orfs_per_noncanonical_subtype", category, int(count))

    if not seq.empty:
        valid_context = seq.get("extracted_start_codon", pd.Series(dtype=str)).replace("NA", pd.NA).notna()
        add_summary(rows, "sequence_context", "valid_extracted_start_codon_count", int(valid_context.sum()))
        add_summary(rows, "sequence_context", "missing_extracted_start_codon_count", int((~valid_context).sum()))
        matches = bool_series(seq.get("start_codon_matches", pd.Series(dtype=str)))
        add_summary(rows, "sequence_context", "start_codon_matches_true_count", int(matches.eq(True).sum()))
        add_summary(rows, "sequence_context", "start_codon_matches_false_count", int(matches.eq(False).sum()))
    else:
        add_summary(rows, "sequence_context", "valid_extracted_start_codon_count", 0)
        add_summary(rows, "sequence_context", "missing_extracted_start_codon_count", 0)
        add_summary(rows, "sequence_context", "start_codon_matches_true_count", 0)
        add_summary(rows, "sequence_context", "start_codon_matches_false_count", 0)

    pd.DataFrame(rows, columns=["section", "metric", "value"]).to_csv(
        QC_SUMMARY_OUT, sep="\t", index=False
    )
    write_start_codon_distribution(groups)
    write_kozak_summary(seq)
    write_length_summary(groups)

    print(f"Wrote {QC_SUMMARY_OUT}")
    print(f"Wrote {START_CODON_OUT}")
    print(f"Wrote {KOZAK_OUT}")
    print(f"Wrote {LENGTH_OUT}")
    print("06_qc_summary.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

