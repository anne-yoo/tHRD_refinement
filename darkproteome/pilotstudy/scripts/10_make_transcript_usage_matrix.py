#!/usr/bin/env python3
"""Convert transcript TPM matrices to within-gene transcript usage matrices."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import List

import pandas as pd


DEFAULT_OUT_DIR = Path("/home/jiye/jiye/darkproteome/ORFstudy/pilot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute transcript usage = transcript TPM / sum(TPM of transcripts "
            "from the same gene) for StringTie TPM matrices."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(os.environ.get("OUT_DIR", DEFAULT_OUT_DIR)),
        help="Directory containing stringtie_tpm_matrix.*.tsv outputs.",
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=None,
        help="Transcript annotation TSV. Default: OUT_DIR/stringtie_tpm_matrix.transcript_annotation.tsv",
    )
    parser.add_argument(
        "--all-matrix",
        type=Path,
        default=None,
        help="All-sample TPM matrix. Default: OUT_DIR/stringtie_tpm_matrix.all_samples.tsv",
    )
    parser.add_argument(
        "--pilot-matrix",
        type=Path,
        default=None,
        help="Pilot TPM matrix. Default: OUT_DIR/stringtie_tpm_matrix.pilot8.tsv",
    )
    parser.add_argument(
        "--gene-column",
        default=os.environ.get("GENE_COLUMN", "gene_id"),
        help="Annotation column defining genes. Default: gene_id.",
    )
    return parser.parse_args()


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "na", "n/a", "."}:
        return ""
    return text


def copy_script_to_out_dir(out_dir: Path) -> None:
    script_dir = out_dir / "scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    src = Path(__file__)
    if src.exists():
        shutil.copy2(src, script_dir / src.name)


def read_annotation(path: Path, gene_column: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Transcript annotation file not found: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str)
    if "transcript_id" not in df.columns:
        raise SystemExit(f"Missing transcript_id column in annotation file: {path}")
    if gene_column not in df.columns:
        if "gene_name" in df.columns:
            print(f"WARNING: {gene_column} not found; using gene_name instead.")
            gene_column = "gene_name"
        else:
            raise SystemExit(f"Missing {gene_column} column in annotation file: {path}")
    out = df[["transcript_id", gene_column]].copy()
    out = out.rename(columns={gene_column: "gene_id_for_usage"})
    out = out.drop_duplicates(subset=["transcript_id"], keep="first")
    return out


def sample_columns(matrix: pd.DataFrame) -> List[str]:
    return [col for col in matrix.columns if col != "transcript_id"]


def compute_usage(matrix_path: Path, annotation: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    if not matrix_path.exists():
        raise SystemExit(f"TPM matrix file not found: {matrix_path}")
    matrix = pd.read_csv(matrix_path, sep="\t", dtype=str)
    if "transcript_id" not in matrix.columns:
        raise SystemExit(f"Missing transcript_id column in TPM matrix: {matrix_path}")

    samples = sample_columns(matrix)
    if not samples:
        raise SystemExit(f"No sample columns found in TPM matrix: {matrix_path}")

    meta = matrix[["transcript_id"]].merge(annotation, on="transcript_id", how="left")
    gene_ids = meta["gene_id_for_usage"].map(clean_text)
    missing_gene = gene_ids.eq("")
    gene_ids.loc[missing_gene] = "missing_gene_id|" + meta.loc[missing_gene, "transcript_id"].astype(str)

    tpm = matrix[samples].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    denominator = tpm.groupby(gene_ids, sort=False).transform("sum")
    usage_values = tpm.divide(denominator.where(denominator.ne(0)))
    usage_values = usage_values.where(denominator.ne(0), 0.0).fillna(0.0)

    usage = pd.concat([matrix[["transcript_id"]], usage_values], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    usage.to_csv(output_path, sep="\t", index=False, na_rep="NA")
    print(f"Wrote transcript usage matrix: {output_path}")

    gene_tpm = tpm.groupby(gene_ids, sort=False).sum()
    gene_usage = usage_values.groupby(gene_ids, sort=False).sum()
    positive_gene_tpm = gene_tpm.gt(0)
    max_deviation = (gene_usage.where(positive_gene_tpm).sub(1.0).abs()).max(axis=0, skipna=True)
    zero_denominator_gene_count = gene_tpm.eq(0).sum(axis=0)
    summary = pd.DataFrame(
        {
            "matrix": matrix_path.name,
            "sample": samples,
            "n_transcripts": len(matrix),
            "n_genes_for_usage": gene_ids.nunique(),
            "n_transcripts_missing_gene_id": int(missing_gene.sum()),
            "n_zero_denominator_genes": [int(zero_denominator_gene_count[sample]) for sample in samples],
            "max_abs_gene_usage_sum_minus_1_for_positive_genes": [
                float(max_deviation[sample]) if pd.notna(max_deviation[sample]) else 0.0 for sample in samples
            ],
        }
    )
    return summary


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    copy_script_to_out_dir(args.out_dir)

    annotation_path = args.annotation or args.out_dir / "stringtie_tpm_matrix.transcript_annotation.tsv"
    all_matrix_path = args.all_matrix or args.out_dir / "stringtie_tpm_matrix.all_samples.tsv"
    pilot_matrix_path = args.pilot_matrix or args.out_dir / "stringtie_tpm_matrix.pilot8.tsv"
    all_usage_out = args.out_dir / "stringtie_transcript_usage.all_samples.tsv"
    pilot_usage_out = args.out_dir / "stringtie_transcript_usage.pilot8.tsv"
    summary_out = args.out_dir / "stringtie_transcript_usage.summary.tsv"

    print("10_make_transcript_usage_matrix.py")
    print(f"OUT_DIR={args.out_dir}")
    print(f"annotation={annotation_path}")
    print(f"all_matrix={all_matrix_path}")
    print(f"pilot_matrix={pilot_matrix_path}")
    print(f"gene_column={args.gene_column}")
    print(f"output={all_usage_out}")
    print(f"output={pilot_usage_out}")
    print(f"output={summary_out}")

    annotation = read_annotation(annotation_path, args.gene_column)
    summaries = [
        compute_usage(all_matrix_path, annotation, all_usage_out),
        compute_usage(pilot_matrix_path, annotation, pilot_usage_out),
    ]
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(summary_out, sep="\t", index=False, na_rep="NA")
    print(f"Wrote usage summary: {summary_out}")
    print("Formula: transcript_usage = transcript_TPM / sum_TPM_of_same_gene; denominator 0 -> usage 0.")
    print("10_make_transcript_usage_matrix.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
