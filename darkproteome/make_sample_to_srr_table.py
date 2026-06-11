#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip()


def load_tsv_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return fieldnames, rows


def load_sample_srr_rows(sample_srr_map_path):
    if not sample_srr_map_path.is_file():
        raise FileNotFoundError(f"Missing sample_srr_map.tsv: {sample_srr_map_path}")

    fieldnames, rows = load_tsv_rows(sample_srr_map_path)
    required = {"sample_id", "final_srrs", "status"}
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(
            f"sample_srr_map.tsv is missing required columns: {', '.join(sorted(missing))}"
        )

    output = []
    for row in rows:
        if normalize_text(row.get("status")) != "ok":
            continue
        output.append(row)
    return output


def load_gsm_manifest_map(gsm_manifest_path):
    if not gsm_manifest_path.is_file():
        raise FileNotFoundError(f"Missing finalfastq_forbam_GSM manifest: {gsm_manifest_path}")

    _, rows = load_tsv_rows(gsm_manifest_path)
    mapping = {}
    for row in rows:
        status = normalize_text(row.get("status"))
        if status not in {"LINKED", "LINKED_SRX_FALLBACK"}:
            continue
        sample_id = normalize_text(row.get("sample_id"))
        target_sample_name = normalize_text(row.get("target_sample_name"))
        if sample_id and target_sample_name:
            mapping[sample_id] = target_sample_name
    return mapping


def first_gsm_from_input_accessions(text):
    for token in [part.strip() for part in normalize_text(text).split(",") if part.strip()]:
        if token.startswith("GSM"):
            return token
    return ""


def build_table(args):
    riboseq_dir = args.riboseq_dir.expanduser().resolve()
    sample_srr_map_path = riboseq_dir / "flattened_metadata" / "sample_srr_map.tsv"
    gsm_manifest_path = riboseq_dir / "fastq" / "finalfastq_forbam_GSM_manifest.tsv"
    output_path = (
        args.output_path.expanduser().resolve()
        if args.output_path
        else riboseq_dir / "fastq" / "sample_to_srr_table.tsv"
    )

    sample_rows = load_sample_srr_rows(sample_srr_map_path)
    gsm_manifest_map = load_gsm_manifest_map(gsm_manifest_path)

    written = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        if not args.no_header:
            writer.writerow(["sample", "srr"])

        for row in sample_rows:
            sample_id = normalize_text(row.get("sample_id"))
            sample_name = gsm_manifest_map.get(sample_id, "")
            if not sample_name:
                sample_name = first_gsm_from_input_accessions(row.get("input_accessions"))
            if not sample_name:
                sample_name = sample_id

            srrs = normalize_text(row.get("final_srrs"))
            writer.writerow([sample_name, srrs])
            written += 1

    print(f"Riboseq dir: {riboseq_dir}")
    print(f"Rows written: {written}")
    print(f"Output: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a 2-column tab-delimited table: sample(GSM or SRX fallback) and SRR list. "
            "Multiple SRRs remain comma-joined in the second column."
        )
    )
    parser.add_argument(
        "riboseq_dir",
        type=Path,
        help="Riboseq dataset directory such as /home/.../data/RPFdb/kidney/Riboseq",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output TSV path. Default: <riboseq_dir>/fastq/sample_to_srr_table.tsv",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Write data rows only, without the header line.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    build_table(args)


if __name__ == "__main__":
    main()
