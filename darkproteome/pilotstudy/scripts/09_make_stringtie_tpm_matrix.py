#!/usr/bin/env python3
"""Build transcript x sample TPM matrices from StringTie GTF outputs."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


DEFAULT_RNA_BASE = Path("/home/jiye/jiye/darkproteome/data/RPFdb/pancreas/finaldata/RNAseq")
DEFAULT_REF_GTF = Path("/home/jiye/jiye/darkproteome/data/refdata/gencode.v48.annotation.gtf")
DEFAULT_OUT_DIR = Path("/home/jiye/jiye/darkproteome/ORFstudy/pilot")
DEFAULT_PILOT_SAMPLES = [
    "GSM3395010",
    "GSM3395011",
    "GSM3395012",
    "GSM3395013",
    "GSM3395014",
    "GSM3395015",
    "GSM5099832",
    "GSM5099835",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse transcript TPM values from sample/stringtie/sample.stringtie.gtf "
            "files and write all-sample and pilot-8 transcript TPM matrices."
        )
    )
    parser.add_argument(
        "--rna-base",
        type=Path,
        default=Path(os.environ.get("RNA_BASE", DEFAULT_RNA_BASE)),
        help="Directory containing per-sample folders with stringtie/*.stringtie.gtf files.",
    )
    parser.add_argument(
        "--ref-gtf",
        type=Path,
        default=Path(os.environ.get("REF_GTF", os.environ.get("GENCODE_GTF", DEFAULT_REF_GTF))),
        help="Reference GENCODE GTF used for quantification.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("OUT_DIR", DEFAULT_OUT_DIR)),
        help="Directory where output matrices will be written.",
    )
    parser.add_argument(
        "--pilot-samples",
        default=os.environ.get("PILOT_SAMPLES", ",".join(DEFAULT_PILOT_SAMPLES)),
        help="Comma-separated sample names to keep in the pilot-8 matrix.",
    )
    parser.add_argument(
        "--fill-missing",
        type=float,
        default=float(os.environ.get("FILL_MISSING_TPM", "0")),
        help="TPM value to use when a reference transcript is missing from a StringTie GTF.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=int(os.environ.get("PROGRESS_EVERY", "10")),
        help="Print progress every N samples.",
    )
    return parser.parse_args()


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "na", "n/a", "."}:
        return ""
    return text


def parse_gtf_attributes(attr_text: str) -> Dict[str, str]:
    return {match.group(1): match.group(2) for match in re.finditer(r'(\S+)\s+"([^"]*)"', attr_text)}


def first_present(attrs: Dict[str, str], keys: Iterable[str]) -> str:
    for key in keys:
        value = clean_text(attrs.get(key, ""))
        if value:
            return value
    return ""


def parse_reference_transcripts(ref_gtf: Path) -> pd.DataFrame:
    if not ref_gtf.exists():
        raise SystemExit(f"Reference GTF not found: {ref_gtf}")

    rows: OrderedDict[str, Dict[str, object]] = OrderedDict()
    fallback_from_exons: Dict[str, Dict[str, object]] = {}
    with ref_gtf.open() as handle:
        for raw_line in handle:
            if not raw_line or raw_line.startswith("#"):
                continue
            parts = raw_line.rstrip("\n").split("\t")
            if len(parts) != 9:
                continue
            chrom, source, feature, start, end, _score, strand, _frame, attrs_text = parts
            if feature not in {"transcript", "exon"}:
                continue
            attrs = parse_gtf_attributes(attrs_text)
            transcript_id = clean_text(attrs.get("transcript_id", ""))
            if not transcript_id:
                continue
            row = {
                "transcript_id": transcript_id,
                "gene_id": clean_text(attrs.get("gene_id", "")),
                "gene_name": first_present(attrs, ["gene_name", "ref_gene_name", "gene"]),
                "gene_type": first_present(attrs, ["gene_type", "gene_biotype"]),
                "transcript_type": first_present(attrs, ["transcript_type", "transcript_biotype"]),
                "chr": chrom,
                "start_1based": int(start),
                "end_1based": int(end),
                "strand": strand,
                "source": source,
            }
            if feature == "transcript":
                rows.setdefault(transcript_id, row)
            elif transcript_id not in fallback_from_exons:
                fallback_from_exons[transcript_id] = row

    for transcript_id, row in fallback_from_exons.items():
        rows.setdefault(transcript_id, row)

    if not rows:
        raise SystemExit(f"No transcript IDs found in reference GTF: {ref_gtf}")
    return pd.DataFrame(rows.values())


def sample_name_from_gtf(path: Path, rna_base: Path) -> str:
    suffix = ".stringtie.gtf"
    if path.name.endswith(suffix):
        return path.name[: -len(suffix)]
    try:
        return path.relative_to(rna_base).parts[0]
    except ValueError:
        return path.stem


def discover_stringtie_gtfs(rna_base: Path) -> pd.DataFrame:
    if not rna_base.exists():
        raise SystemExit(f"RNA base directory not found: {rna_base}")

    rows = []
    for sample_dir in sorted([path for path in rna_base.iterdir() if path.is_dir()]):
        sample = sample_dir.name
        expected = sample_dir / "stringtie" / f"{sample}.stringtie.gtf"
        if expected.exists():
            rows.append({"sample": sample, "gtf_path": str(expected), "discovery": "expected_path"})
        else:
            matches = sorted((sample_dir / "stringtie").glob("*.stringtie.gtf")) if (sample_dir / "stringtie").exists() else []
            for match in matches:
                rows.append(
                    {
                        "sample": sample_name_from_gtf(match, rna_base),
                        "gtf_path": str(match),
                        "discovery": "glob_fallback",
                    }
                )

    manifest = pd.DataFrame(rows).drop_duplicates(subset=["sample"], keep="first")
    if manifest.empty:
        raise SystemExit(f"No *.stringtie.gtf files found under {rna_base}")
    return manifest.sort_values("sample").reset_index(drop=True)


def parse_stringtie_tpm(gtf_path: Path) -> Dict[str, float]:
    transcript_tpm: Dict[str, float] = {}
    with gtf_path.open() as handle:
        for raw_line in handle:
            if not raw_line or raw_line.startswith("#"):
                continue
            parts = raw_line.rstrip("\n").split("\t")
            if len(parts) != 9 or parts[2] != "transcript":
                continue
            attrs = parse_gtf_attributes(parts[8])
            transcript_id = clean_text(attrs.get("transcript_id", ""))
            if not transcript_id:
                continue
            tpm_text = clean_text(attrs.get("TPM", ""))
            if not tpm_text:
                tpm = 0.0
            else:
                try:
                    tpm = float(tpm_text)
                except ValueError:
                    tpm = float("nan")
            transcript_tpm[transcript_id] = tpm
    return transcript_tpm


def write_matrix(
    matrix: pd.DataFrame,
    path: Path,
    sample_columns: List[str],
) -> None:
    missing = [sample for sample in sample_columns if sample not in matrix.columns]
    if missing:
        print(f"WARNING: {len(missing)} requested sample(s) missing from all-sample matrix: {', '.join(missing)}")
    present = [sample for sample in sample_columns if sample in matrix.columns]
    matrix[["transcript_id"] + present].to_csv(path, sep="\t", index=False, na_rep="NA")


def copy_script_to_out_dir(out_dir: Path) -> None:
    script_dir = out_dir / "scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    src = Path(__file__)
    if src.exists():
        shutil.copy2(src, script_dir / src.name)


def main() -> int:
    args = parse_args()
    pilot_samples = [sample.strip() for sample in args.pilot_samples.split(",") if sample.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    copy_script_to_out_dir(args.out_dir)

    annotation_out = args.out_dir / "stringtie_tpm_matrix.transcript_annotation.tsv"
    manifest_out = args.out_dir / "stringtie_tpm_matrix.sample_manifest.tsv"
    all_matrix_out = args.out_dir / "stringtie_tpm_matrix.all_samples.tsv"
    pilot_matrix_out = args.out_dir / "stringtie_tpm_matrix.pilot8.tsv"
    summary_out = args.out_dir / "stringtie_tpm_matrix.summary.tsv"

    print("09_make_stringtie_tpm_matrix.py")
    print(f"RNA_BASE={args.rna_base}")
    print(f"REF_GTF={args.ref_gtf}")
    print(f"OUT_DIR={args.out_dir}")
    print(f"fill_missing_TPM={args.fill_missing}")
    print(f"pilot_samples={','.join(pilot_samples)}")
    print(f"output={annotation_out}")
    print(f"output={manifest_out}")
    print(f"output={all_matrix_out}")
    print(f"output={pilot_matrix_out}")
    print(f"output={summary_out}")

    annotation = parse_reference_transcripts(args.ref_gtf)
    annotation.to_csv(annotation_out, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {len(annotation)} reference transcript annotation row(s): {annotation_out}")

    manifest = discover_stringtie_gtfs(args.rna_base)
    manifest.to_csv(manifest_out, sep="\t", index=False)
    print(f"Discovered {len(manifest)} StringTie GTF file(s): {manifest_out}")

    transcript_ids = annotation["transcript_id"].astype(str).tolist()
    matrix = pd.DataFrame({"transcript_id": transcript_ids})
    summary_rows = []

    for idx, row in manifest.iterrows():
        sample = row["sample"]
        gtf_path = Path(row["gtf_path"])
        tpm = parse_stringtie_tpm(gtf_path)
        matrix[sample] = matrix["transcript_id"].map(tpm).fillna(args.fill_missing).astype(float)
        n_present = int(matrix["transcript_id"].isin(tpm.keys()).sum())
        summary_rows.append(
            {
                "sample": sample,
                "gtf_path": str(gtf_path),
                "n_transcripts_with_stringtie_tpm": n_present,
                "n_reference_transcripts": len(transcript_ids),
                "n_reference_transcripts_missing_from_stringtie_gtf": len(transcript_ids) - n_present,
                "sum_TPM_over_reference_transcripts": float(matrix[sample].sum(skipna=True)),
                "mean_TPM_over_reference_transcripts": float(matrix[sample].mean(skipna=True)),
            }
        )
        if args.progress_every > 0 and ((idx + 1) % args.progress_every == 0 or (idx + 1) == len(manifest)):
            print(f"Parsed {idx + 1}/{len(manifest)} sample GTF(s)")

    all_sample_columns = manifest["sample"].astype(str).tolist()
    matrix[["transcript_id"] + all_sample_columns].to_csv(all_matrix_out, sep="\t", index=False, na_rep="NA")
    print(f"Wrote all-sample TPM matrix: {all_matrix_out}")

    write_matrix(matrix, pilot_matrix_out, pilot_samples)
    print(f"Wrote pilot-sample TPM matrix: {pilot_matrix_out}")

    pilot_missing = [sample for sample in pilot_samples if sample not in all_sample_columns]
    summary = pd.DataFrame(summary_rows)
    summary_meta = pd.DataFrame(
        [
            {"sample": "__META__", "gtf_path": "n_all_samples", "n_transcripts_with_stringtie_tpm": len(all_sample_columns)},
            {"sample": "__META__", "gtf_path": "n_reference_transcripts", "n_transcripts_with_stringtie_tpm": len(transcript_ids)},
            {"sample": "__META__", "gtf_path": "n_pilot_samples_requested", "n_transcripts_with_stringtie_tpm": len(pilot_samples)},
            {"sample": "__META__", "gtf_path": "n_pilot_samples_found", "n_transcripts_with_stringtie_tpm": len(pilot_samples) - len(pilot_missing)},
            {"sample": "__META__", "gtf_path": "missing_pilot_samples", "n_transcripts_with_stringtie_tpm": ",".join(pilot_missing) if pilot_missing else "none"},
        ]
    )
    pd.concat([summary, summary_meta], ignore_index=True).to_csv(summary_out, sep="\t", index=False, na_rep="NA")
    print(f"Wrote summary: {summary_out}")

    print("09_make_stringtie_tpm_matrix.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
