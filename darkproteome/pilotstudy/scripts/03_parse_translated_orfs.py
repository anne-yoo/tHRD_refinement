#!/usr/bin/env python3
"""Parse translated ORF calls into sample-level, unique-level, and BED files."""

from __future__ import annotations

import os
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List

import pandas as pd


BASE_DIR = Path(os.environ.get("BASE_DIR", "/home/jiye/jiye/darkproteome/ORFstudy/pilot"))
OUT_DIR = Path(
    os.environ.get("OUT_DIR", "/home/jiye/jiye/darkproteome/ORFstudy/pilot/pancreas8samples")
)
INPUT_ORF = BASE_DIR / "Pancreas.4caller.merged.2caller.tsv"

SAMPLE_LEVEL_OUT = OUT_DIR / "tables" / "pancreas.translated_orfs.sample_level.tsv"
UNIQUE_OUT = OUT_DIR / "tables" / "pancreas.translated_orfs.unique.tsv"
BED_OUT = OUT_DIR / "bed" / "pancreas.translated_orfs.all.bed"
WARNING_OUT = OUT_DIR / "logs" / "orf_inconsistency_warnings.tsv"

REQUIRED_COLUMNS = [
    "ORF_id",
    "sample",
    "chr",
    "start(0-based)",
    "end(0-based)",
    "strand",
    "ORF_type",
    "start_codon",
    "caller",
    "num_of_callers",
    "ORF_type2",
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
    if not text:
        return ""
    token = re.split(r"[\s,;|/]+", text)[0]
    return token


def stable_unique(values: Iterable[object]) -> List[str]:
    seen: OrderedDict[str, None] = OrderedDict()
    for value in values:
        text = clean_text(value)
        if text and text not in seen:
            seen[text] = None
    return list(seen)


def split_support_values(values: Iterable[object]) -> List[str]:
    seen: OrderedDict[str, None] = OrderedDict()
    for value in values:
        text = clean_text(value)
        if not text:
            continue
        for part in re.split(r"[,;|]+", text):
            part = part.strip()
            if part and part not in seen:
                seen[part] = None
    return list(seen)


def first_value(values: Iterable[object]) -> str:
    values = stable_unique(values)
    return values[0] if values else ""


def transcript_from_orf_id(orf_id: str) -> str:
    return clean_text(orf_id).split(":", 1)[0]


def require_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing required input column(s): {', '.join(missing)}")


def to_int_series(series: pd.Series, column_name: str) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    if converted.isna().any():
        bad = series[converted.isna()].head(5).tolist()
        raise SystemExit(f"Column {column_name} contains non-numeric values, examples: {bad}")
    return converted.astype(int)


def build_sample_level(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "ORF_id": df["ORF_id"].map(clean_text),
            "transcript_id": df["ORF_id"].map(transcript_from_orf_id),
            "sample": df["sample"].map(clean_text),
            "chr": df["chr"].map(clean_text),
            "start0": to_int_series(df["start(0-based)"], "start(0-based)"),
            "end0": to_int_series(df["end(0-based)"], "end(0-based)"),
            "strand": df["strand"].map(clean_text),
            "ORF_type": df["ORF_type"].map(clean_text),
            "start_codon": df["start_codon"].map(normalize_codon),
            "caller": df["caller"].map(clean_text),
            "num_of_callers": pd.to_numeric(df["num_of_callers"], errors="coerce").astype("Int64"),
            "ORF_type2": df["ORF_type2"].map(clean_text),
        }
    )

    missing_orf_id = out["ORF_id"].eq("").sum()
    if missing_orf_id:
        raise SystemExit(f"Found {missing_orf_id} row(s) with missing ORF_id")

    bad_strand = sorted(set(out.loc[~out["strand"].isin(["+", "-"]), "strand"]))
    if bad_strand:
        raise SystemExit(f"Found unsupported strand value(s): {bad_strand}")

    invalid_coords = out["end0"] <= out["start0"]
    if invalid_coords.any():
        examples = out.loc[invalid_coords, ["ORF_id", "start0", "end0"]].head(5).to_dict("records")
        raise SystemExit(f"Found end0 <= start0 for ORF(s): {examples}")

    return out


def build_inconsistency_warnings(sample_df: pd.DataFrame) -> pd.DataFrame:
    check_fields = ["chr", "start0", "end0", "strand", "ORF_type", "start_codon", "ORF_type2"]
    rows = []
    for orf_id, group in sample_df.groupby("ORF_id", sort=False):
        for field in check_fields:
            values = stable_unique(group[field])
            if len(values) > 1:
                rows.append(
                    {
                        "ORF_id": orf_id,
                        "field": field,
                        "values": "|".join(map(str, values)),
                        "n_values": len(values),
                        "n_rows": len(group),
                    }
                )
    return pd.DataFrame(rows, columns=["ORF_id", "field", "values", "n_values", "n_rows"])


def build_unique(sample_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for orf_id, group in sample_df.groupby("ORF_id", sort=False):
        callers = split_support_values(group["caller"])
        samples = sorted(stable_unique(group["sample"]))
        max_callers = group["num_of_callers"].dropna()
        rows.append(
            {
                "ORF_id": orf_id,
                "transcript_id": transcript_from_orf_id(orf_id),
                "chr": first_value(group["chr"]),
                "start0": int(first_value(group["start0"])),
                "end0": int(first_value(group["end0"])),
                "strand": first_value(group["strand"]),
                "ORF_type": first_value(group["ORF_type"]),
                "start_codon": first_value(group["start_codon"]),
                "ORF_type2": first_value(group["ORF_type2"]),
                "detected_samples": "|".join(samples),
                "n_detected_samples": len(samples),
                "caller_union": "|".join(callers),
                "max_num_callers": int(max_callers.max()) if len(max_callers) else pd.NA,
                "n_rows": len(group),
            }
        )
    return pd.DataFrame(rows)


def write_bed(unique_df: pd.DataFrame) -> None:
    bed = unique_df[["chr", "start0", "end0", "ORF_id", "strand"]].copy()
    bed.insert(4, "score", 0)
    bed[["chr", "start0", "end0", "ORF_id", "score", "strand"]].to_csv(
        BED_OUT, sep="\t", index=False, header=False
    )


def main() -> int:
    print("03_parse_translated_orfs.py")
    print(f"input={INPUT_ORF}")
    print(f"sample_level_out={SAMPLE_LEVEL_OUT}")
    print(f"unique_out={UNIQUE_OUT}")
    print(f"bed_out={BED_OUT}")
    print(f"warning_out={WARNING_OUT}")

    if not INPUT_ORF.exists():
        raise SystemExit(f"Input ORF file not found: {INPUT_ORF}")

    (OUT_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "bed").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "logs").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_ORF, sep="\t", dtype=str, encoding="utf-8-sig")
    df.columns = [col.strip() for col in df.columns]
    require_columns(df)

    sample_df = build_sample_level(df)
    warnings_df = build_inconsistency_warnings(sample_df)
    unique_df = build_unique(sample_df)

    sample_df.to_csv(SAMPLE_LEVEL_OUT, sep="\t", index=False, na_rep="NA")
    unique_df.to_csv(UNIQUE_OUT, sep="\t", index=False, na_rep="NA")
    warnings_df.to_csv(WARNING_OUT, sep="\t", index=False, na_rep="NA")
    write_bed(unique_df)

    print(f"Wrote {len(sample_df)} sample-level row(s)")
    print(f"Wrote {len(unique_df)} unique ORF row(s)")
    print(f"Wrote {len(warnings_df)} inconsistency warning row(s)")
    print("03_parse_translated_orfs.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

