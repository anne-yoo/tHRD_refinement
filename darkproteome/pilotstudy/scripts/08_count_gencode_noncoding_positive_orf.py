#!/usr/bin/env python3
"""Count GENCODE noncoding transcripts with and without pancreas positive ORFs."""

from __future__ import annotations

import gzip
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, TextIO

import pandas as pd


GENCODE_GTF = Path(
    os.environ.get(
        "GENCODE_GTF",
        "/home/jiye/jiye/darkproteome/pilotstudy/data/gencode.v48.annotation.gtf",
    )
)
ORF_TSV = Path(
    os.environ.get(
        "ORF_TSV",
        "/home/jiye/jiye/darkproteome/pilotstudy/data/Pancreas.4caller.merged.2caller.tsv",
    )
)
FIG_DIR = Path(os.environ.get("FIG_DIR", "/home/jiye/jiye/darkproteome/ORFstudy/pilot/figures"))
OUT_DIR = FIG_DIR / "tables"
SUMMARY_OUT = OUT_DIR / "gencode_noncoding_positive_orf_summary.tsv"
TRANSCRIPT_OUT = OUT_DIR / "gencode_noncoding_positive_orf_by_transcript.tsv"
BIOTYPE_OUT = OUT_DIR / "gencode_noncoding_positive_orf_by_transcript_type.tsv"


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open()


def parse_attributes(attribute_text: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for key, value in re.findall(r'([A-Za-z0-9_]+) "([^"]*)"', attribute_text):
        attrs[key] = value
    return attrs


def strip_version(identifier: object) -> str:
    text = "" if pd.isna(identifier) else str(identifier).strip()
    return text.split(".", 1)[0] if text else ""


def parse_gencode_transcripts(gtf_path: Path) -> pd.DataFrame:
    rows = []
    with open_text(gtf_path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 9 or parts[2] != "transcript":
                continue
            attrs = parse_attributes(parts[8])
            transcript_id = attrs.get("transcript_id", "")
            if not transcript_id:
                continue
            rows.append(
                {
                    "transcript_id": transcript_id,
                    "transcript_id_no_version": strip_version(transcript_id),
                    "gene_id": attrs.get("gene_id", ""),
                    "gene_name": attrs.get("gene_name", ""),
                    "gene_type": attrs.get("gene_type", ""),
                    "transcript_type": attrs.get("transcript_type", ""),
                    "transcript_name": attrs.get("transcript_name", ""),
                    "source": parts[1],
                    "chr": parts[0],
                    "start_1based": int(parts[3]),
                    "end_1based": int(parts[4]),
                    "strand": parts[6],
                }
            )
    if not rows:
        raise SystemExit(f"No transcript features parsed from GTF: {gtf_path}")
    return pd.DataFrame(rows).drop_duplicates(subset=["transcript_id"])


def parse_positive_orf_transcripts(orf_tsv: Path) -> pd.DataFrame:
    df = pd.read_csv(orf_tsv, sep="\t", dtype=str, usecols=["ORF_id"])
    df["transcript_id"] = df["ORF_id"].astype(str).str.split(":", n=1).str[0]
    out = (
        df.groupby("transcript_id", as_index=False)
        .agg(n_positive_orf_rows=("ORF_id", "size"), n_unique_positive_orfs=("ORF_id", "nunique"))
    )
    out["transcript_id_no_version"] = out["transcript_id"].map(strip_version)
    return out


def summarize(df: pd.DataFrame, noncoding_col: str, label: str) -> pd.DataFrame:
    sub = df[df[noncoding_col]].copy()
    with_orf = int(sub["has_positive_orf"].sum())
    without_orf = int((~sub["has_positive_orf"]).sum())
    return pd.DataFrame(
        [
            {
                "definition": label,
                "noncoding_transcripts_total": int(len(sub)),
                "noncoding_transcripts_with_at_least_one_positive_ORF": with_orf,
                "noncoding_transcripts_with_zero_positive_ORF": without_orf,
                "fraction_with_positive_ORF": with_orf / len(sub) if len(sub) else pd.NA,
            }
        ]
    )


def main() -> int:
    print("08_count_gencode_noncoding_positive_orf.py")
    print(f"GENCODE_GTF={GENCODE_GTF}")
    print(f"ORF_TSV={ORF_TSV}")
    print(f"SUMMARY_OUT={SUMMARY_OUT}")
    print(f"TRANSCRIPT_OUT={TRANSCRIPT_OUT}")
    print(f"BIOTYPE_OUT={BIOTYPE_OUT}")

    if not GENCODE_GTF.exists():
        raise SystemExit(f"GENCODE GTF not found: {GENCODE_GTF}")
    if not ORF_TSV.exists():
        raise SystemExit(f"ORF TSV not found: {ORF_TSV}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gencode = parse_gencode_transcripts(GENCODE_GTF)
    positives = parse_positive_orf_transcripts(ORF_TSV)

    by_unversioned = (
        positives.groupby("transcript_id_no_version", as_index=False)
        .agg(
            n_positive_orf_rows=("n_positive_orf_rows", "sum"),
            n_unique_positive_orfs=("n_unique_positive_orfs", "sum"),
            positive_orf_transcript_ids=("transcript_id", lambda x: "|".join(sorted(set(x)))),
        )
    )

    merged = gencode.merge(by_unversioned, on="transcript_id_no_version", how="left")
    merged["n_positive_orf_rows"] = merged["n_positive_orf_rows"].fillna(0).astype(int)
    merged["n_unique_positive_orfs"] = merged["n_unique_positive_orfs"].fillna(0).astype(int)
    merged["positive_orf_transcript_ids"] = merged["positive_orf_transcript_ids"].fillna("")
    merged["has_positive_orf"] = merged["n_unique_positive_orfs"].gt(0)

    # Primary definition used in the requested count: GENCODE transcript_type is not protein_coding.
    merged["is_noncoding_transcript_type_not_protein_coding"] = merged["transcript_type"].ne("protein_coding")
    # More conservative alternate definition: transcript_type does not contain "protein_coding".
    merged["is_noncoding_transcript_type_no_protein_coding_substring"] = ~merged[
        "transcript_type"
    ].str.contains("protein_coding", case=False, na=False)

    summary = pd.concat(
        [
            summarize(
                merged,
                "is_noncoding_transcript_type_not_protein_coding",
                "transcript_type != protein_coding",
            ),
            summarize(
                merged,
                "is_noncoding_transcript_type_no_protein_coding_substring",
                "transcript_type does not contain protein_coding",
            ),
        ],
        ignore_index=True,
    )
    summary.insert(0, "gencode_transcripts_total", len(merged))
    summary.insert(1, "pancreas_positive_orf_transcripts_in_gencode", int(merged["has_positive_orf"].sum()))
    summary.to_csv(SUMMARY_OUT, sep="\t", index=False, na_rep="NA")

    transcript_cols = [
        "transcript_id",
        "transcript_id_no_version",
        "gene_id",
        "gene_name",
        "gene_type",
        "transcript_type",
        "transcript_name",
        "chr",
        "start_1based",
        "end_1based",
        "strand",
        "is_noncoding_transcript_type_not_protein_coding",
        "is_noncoding_transcript_type_no_protein_coding_substring",
        "has_positive_orf",
        "n_unique_positive_orfs",
        "n_positive_orf_rows",
        "positive_orf_transcript_ids",
    ]
    merged[transcript_cols].to_csv(TRANSCRIPT_OUT, sep="\t", index=False, na_rep="NA")

    biotype = (
        merged[merged["is_noncoding_transcript_type_not_protein_coding"]]
        .groupby("transcript_type", dropna=False)
        .agg(
            n_gencode_noncoding_transcripts=("transcript_id", "size"),
            n_with_positive_orf=("has_positive_orf", "sum"),
        )
        .reset_index()
    )
    biotype["n_without_positive_orf"] = (
        biotype["n_gencode_noncoding_transcripts"] - biotype["n_with_positive_orf"]
    )
    biotype["fraction_with_positive_orf"] = (
        biotype["n_with_positive_orf"] / biotype["n_gencode_noncoding_transcripts"]
    )
    biotype.sort_values(
        ["n_with_positive_orf", "n_gencode_noncoding_transcripts"],
        ascending=[False, False],
    ).to_csv(BIOTYPE_OUT, sep="\t", index=False, na_rep="NA")

    print(summary.to_string(index=False))
    print(f"Wrote {SUMMARY_OUT}")
    print(f"Wrote {TRANSCRIPT_OUT}")
    print(f"Wrote {BIOTYPE_OUT}")
    print("08_count_gencode_noncoding_positive_orf.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

