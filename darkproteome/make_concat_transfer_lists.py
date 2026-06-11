#!/usr/bin/env python3

import argparse
import csv
from collections import Counter
from pathlib import Path


MANIFEST_REQUIRED_COLUMNS = {
    "sample_id",
    "target_sample_name",
    "target_fastq_files",
    "status",
}


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip()


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def safe_symlink(source, target):
    ensure_dir(target.parent)
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(source)


def load_tsv_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return fieldnames, rows


def load_sample_srr_map(sample_srr_map_path):
    if not sample_srr_map_path.is_file():
        raise FileNotFoundError(f"Missing sample_srr_map.tsv: {sample_srr_map_path}")

    fieldnames, rows = load_tsv_rows(sample_srr_map_path)
    required = {"sample_id", "final_srrs", "num_srrs", "status"}
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(
            f"sample_srr_map.tsv is missing required columns: {', '.join(sorted(missing))}"
        )

    sample_map = {}
    for row in rows:
        if normalize_text(row.get("status")) != "ok":
            continue
        sample_id = normalize_text(row.get("sample_id"))
        if sample_id:
            sample_map[sample_id] = row
    return sample_map


def load_gsm_manifest(gsm_manifest_path):
    if not gsm_manifest_path.is_file():
        raise FileNotFoundError(f"Missing finalfastq_forbam_GSM manifest: {gsm_manifest_path}")

    fieldnames, rows = load_tsv_rows(gsm_manifest_path)
    missing = MANIFEST_REQUIRED_COLUMNS.difference(fieldnames)
    if missing:
        raise ValueError(
            f"GSM manifest is missing required columns: {', '.join(sorted(missing))}"
        )

    manifest_map = {}
    for row in rows:
        status = normalize_text(row.get("status"))
        if status not in {"LINKED", "LINKED_SRX_FALLBACK"}:
            continue
        sample_id = normalize_text(row.get("sample_id"))
        if sample_id:
            manifest_map[sample_id] = row
    return manifest_map


def find_raw_fastq_files(fastq_dir, srr_accession):
    direct = fastq_dir / f"{srr_accession}.fastq.gz"
    read1 = fastq_dir / f"{srr_accession}_1.fastq.gz"
    read2 = fastq_dir / f"{srr_accession}_2.fastq.gz"

    if direct.is_file():
        return [direct]
    if read1.is_file() and read2.is_file():
        return [read1, read2]

    candidates = sorted(
        path for path in fastq_dir.glob(f"{srr_accession}*.fastq.gz") if path.is_file()
    )
    if candidates:
        return candidates

    return [direct]


def split_csv_cell(value):
    return [part.strip() for part in normalize_text(value).split(",") if part.strip()]


def write_text_list(path, values):
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(f"{value}\n")


def write_tsv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_lists(args):
    riboseq_dir = args.riboseq_dir.expanduser().resolve()
    fastq_dir = riboseq_dir / "fastq"
    final_gsm_dir = fastq_dir / "finalfastq_forbam_GSM"
    concat_only_dir = (
        args.concat_only_dir.expanduser().resolve()
        if args.concat_only_dir
        else fastq_dir / "onlyconcat_finalfastq_forbam_GSM"
    )
    sample_srr_map_path = riboseq_dir / "flattened_metadata" / "sample_srr_map.tsv"
    gsm_manifest_path = fastq_dir / "finalfastq_forbam_GSM_manifest.tsv"
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else fastq_dir / "concat_transfer_lists"
    )

    ensure_dir(output_dir)
    ensure_dir(concat_only_dir)
    sample_map = load_sample_srr_map(sample_srr_map_path)
    gsm_manifest_map = load_gsm_manifest(gsm_manifest_path)

    summary_rows = []
    raw_paths_to_remove = []
    raw_srrs_to_remove = []
    final_paths_to_fetch = []
    concat_only_paths = []

    seen_raw_paths = set()
    seen_raw_srrs = set()
    seen_final_paths = set()
    seen_concat_only_paths = set()

    for sample_id, row in sample_map.items():
        srrs = split_csv_cell(row.get("final_srrs"))
        if len(srrs) <= 1:
            continue

        manifest_row = gsm_manifest_map.get(sample_id)
        if not manifest_row:
            summary_rows.append(
                {
                    "sample_id": sample_id,
                    "target_sample_name": "",
                    "num_srrs": len(srrs),
                    "final_srrs": ",".join(srrs),
                    "raw_fastq_files": "",
                    "final_gsm_fastq_files": "",
                    "concat_only_fastq_files": "",
                    "status": "MISSING_GSM_MANIFEST",
                    "note": "No linked row found in finalfastq_forbam_GSM_manifest.tsv",
                }
            )
            continue

        target_sample_name = normalize_text(manifest_row.get("target_sample_name"))
        target_fastq_files = split_csv_cell(manifest_row.get("target_fastq_files"))
        if not target_fastq_files:
            target_fastq_files = []
            for suffix in (".fastq.gz", "_1.fastq.gz", "_2.fastq.gz"):
                candidate = final_gsm_dir / f"{target_sample_name}{suffix}"
                if candidate.exists():
                    target_fastq_files.append(str(candidate))

        raw_fastq_files = []
        for srr in srrs:
            if srr not in seen_raw_srrs:
                seen_raw_srrs.add(srr)
                raw_srrs_to_remove.append(srr)

            for raw_path in find_raw_fastq_files(fastq_dir, srr):
                raw_path_str = str(raw_path)
                raw_fastq_files.append(raw_path_str)
                if raw_path_str not in seen_raw_paths:
                    seen_raw_paths.add(raw_path_str)
                    raw_paths_to_remove.append(raw_path_str)

        for final_path in target_fastq_files:
            if final_path not in seen_final_paths:
                seen_final_paths.add(final_path)
                final_paths_to_fetch.append(final_path)

        concat_only_target_paths = []
        for final_path in target_fastq_files:
            source_path = Path(final_path)
            concat_only_target = concat_only_dir / source_path.name
            safe_symlink(source_path.resolve(strict=False), concat_only_target)
            concat_only_target_str = str(concat_only_target)
            concat_only_target_paths.append(concat_only_target_str)
            if concat_only_target_str not in seen_concat_only_paths:
                seen_concat_only_paths.add(concat_only_target_str)
                concat_only_paths.append(concat_only_target_str)

        summary_rows.append(
            {
                "sample_id": sample_id,
                "target_sample_name": target_sample_name,
                "num_srrs": len(srrs),
                "final_srrs": ",".join(srrs),
                "raw_fastq_files": ",".join(raw_fastq_files),
                "final_gsm_fastq_files": ",".join(target_fastq_files),
                "concat_only_fastq_files": ",".join(concat_only_target_paths),
                "status": "OK",
                "note": "",
            }
        )

    summary_rows.sort(key=lambda row: (row["status"] != "OK", row["sample_id"]))

    write_text_list(output_dir / "concat_component_srrs.txt", raw_srrs_to_remove)
    write_text_list(output_dir / "concat_component_raw_fastq_paths.txt", raw_paths_to_remove)
    write_text_list(output_dir / "concat_finalfastq_forbam_GSM_paths.txt", final_paths_to_fetch)
    write_text_list(output_dir / "concat_only_finalfastq_forbam_GSM_paths.txt", concat_only_paths)
    write_tsv(
        output_dir / "concat_transfer_summary.tsv",
        [
            "sample_id",
            "target_sample_name",
            "num_srrs",
            "final_srrs",
            "raw_fastq_files",
            "final_gsm_fastq_files",
            "concat_only_fastq_files",
            "status",
            "note",
        ],
        summary_rows,
    )

    status_counts = Counter(row["status"] for row in summary_rows)
    print(f"Riboseq dir: {riboseq_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Concat-only symlink dir: {concat_only_dir}")
    print(f"Multi-SRR samples: {len(summary_rows)}")
    print(f"Status counts: {dict(status_counts)}")
    print(f"Raw SRR accessions to remove: {len(raw_srrs_to_remove)}")
    print(f"Raw FASTQ paths to remove: {len(raw_paths_to_remove)}")
    print(f"Final GSM FASTQ paths to fetch: {len(final_paths_to_fetch)}")
    print(f"Concat-only final GSM FASTQ paths: {len(concat_only_paths)}")
    print(output_dir / "concat_component_srrs.txt")
    print(output_dir / "concat_component_raw_fastq_paths.txt")
    print(output_dir / "concat_finalfastq_forbam_GSM_paths.txt")
    print(output_dir / "concat_only_finalfastq_forbam_GSM_paths.txt")
    print(output_dir / "concat_transfer_summary.tsv")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build transfer/removal lists for multi-SRR Riboseq samples. "
            "The script writes the raw SRR FASTQ files that were concatenated and "
            "the corresponding finalfastq_forbam_GSM FASTQ files that should be fetched instead."
        )
    )
    parser.add_argument(
        "riboseq_dir",
        type=Path,
        help="Riboseq dataset directory such as /home/.../data/RPFdb/kidney/Riboseq",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for generated list files. Default: <riboseq_dir>/fastq/concat_transfer_lists",
    )
    parser.add_argument(
        "--concat-only-dir",
        type=Path,
        default=None,
        help=(
            "Directory where symlinks for multi-SRR final GSM FASTQs will be created. "
            "Default: <riboseq_dir>/fastq/onlyconcat_finalfastq_forbam_GSM"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    build_lists(args)


if __name__ == "__main__":
    main()
