#!/usr/bin/env python3

import argparse
import csv
import io
import os
import re
import shutil
import subprocess
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path


DEFAULT_SRA_TOOLKIT_BIN = (
    "/home/jiye/jiye/darkproteome/data/nuORFdb/riboseq/"
    "sratoolkit.3.2.0-centos_linux64/bin"
)

GEO_TEXT_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}"
    "&targ=self&view=full&form=text"
)
RUNINFO_URL = "https://trace.ncbi.nlm.nih.gov/Traces/sra-db-be/runinfo?acc={accession}"

ACCESSION_RE = re.compile(r"\b(GSM\d+|SRX\d+|SRR\d+)\b")
GSM_SRA_RELATION_RE = re.compile(
    r"^!Sample_relation = SRA: .*?\b(SRX\d+)\b", re.MULTILINE
)

XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
PKG_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"

OUTPUT_COLUMNS = [
    "sample_id",
    "legacy_sample_id",
    "source_id",
    "primary_tissue",
    "excel_row",
    "excel_col",
    "original_cell_value",
    "input_accessions",
    "resolved_srxs",
    "final_srrs",
    "num_srrs",
    "status",
    "error",
]


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip()


def dedupe_keep_order(values):
    seen = set()
    deduped = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def detect_accession_type(accession):
    if accession.startswith("GSM"):
        return "GSM"
    if accession.startswith("SRX"):
        return "SRX"
    if accession.startswith("SRR"):
        return "SRR"
    return "UNKNOWN"


def extract_accessions(cell_value):
    text = normalize_text(cell_value)
    if not text:
        return []
    return ACCESSION_RE.findall(text)


def col_to_num(col_ref):
    number = 0
    for char in col_ref:
        if char.isalpha():
            number = (number * 26) + (ord(char.upper()) - 64)
    return number


def parse_shared_strings(workbook_zip):
    if "xl/sharedStrings.xml" not in workbook_zip.namelist():
        return []

    root = ET.fromstring(workbook_zip.read("xl/sharedStrings.xml"))
    shared_strings = []
    for item in root.findall(f"{XLSX_NS}si"):
        parts = []
        for text_node in item.iter(f"{XLSX_NS}t"):
            parts.append(text_node.text or "")
        shared_strings.append("".join(parts))
    return shared_strings


def get_first_sheet_path(workbook_zip):
    workbook = ET.fromstring(workbook_zip.read("xl/workbook.xml"))
    rels = ET.fromstring(workbook_zip.read("xl/_rels/workbook.xml.rels"))
    rel_map = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall(f"{PKG_REL_NS}Relationship")
    }

    sheets = workbook.find(f"{XLSX_NS}sheets")
    if sheets is None or not list(sheets):
        raise ValueError("No worksheets found in workbook")

    first_sheet = list(sheets)[0]
    rel_id = first_sheet.attrib.get(
        "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    )
    if not rel_id or rel_id not in rel_map:
        raise ValueError("Could not resolve first worksheet relationship")

    target = rel_map[rel_id]
    if target.startswith("/"):
        return target.lstrip("/")
    if target.startswith("xl/"):
        return target
    return f"xl/{target}"


def get_cell_text(cell_elem, shared_strings):
    cell_type = cell_elem.attrib.get("t")
    value_node = cell_elem.find(f"{XLSX_NS}v")

    if cell_type == "s" and value_node is not None and value_node.text is not None:
        return shared_strings[int(value_node.text)]

    if cell_type == "inlineStr":
        inline = cell_elem.find(f"{XLSX_NS}is")
        if inline is None:
            return ""
        return "".join(text_node.text or "" for text_node in inline.iter(f"{XLSX_NS}t"))

    if value_node is not None and value_node.text is not None:
        return value_node.text

    return ""


def iter_sheet_rows(xlsx_file):
    with zipfile.ZipFile(xlsx_file) as workbook_zip:
        shared_strings = parse_shared_strings(workbook_zip)
        sheet_path = get_first_sheet_path(workbook_zip)
        sheet_root = ET.fromstring(workbook_zip.read(sheet_path))

    sheet_data = sheet_root.find(f"{XLSX_NS}sheetData")
    if sheet_data is None:
        return

    for row_elem in sheet_data.findall(f"{XLSX_NS}row"):
        row_num = int(row_elem.attrib["r"])
        values_by_col = {}
        for cell_elem in row_elem.findall(f"{XLSX_NS}c"):
            cell_ref = cell_elem.attrib["r"]
            col_num = col_to_num(re.match(r"[A-Z]+", cell_ref).group(0))
            values_by_col[col_num] = get_cell_text(cell_elem, shared_strings)
        yield row_num, values_by_col


def fetch_text(url, cache_path, timeout, retries, backoff_seconds):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.is_file():
        return cache_path.read_text(encoding="utf-8")

    last_error = None
    headers = {"User-Agent": "Mozilla/5.0 (compatible; rpfdb_pipeline/1.0)"}

    for attempt in range(1, retries + 1):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                text = response.read().decode("utf-8", "replace")
            cache_path.write_text(text, encoding="utf-8")
            return text
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(backoff_seconds * attempt)

    raise RuntimeError(f"Request failed after {retries} attempts: {url}") from last_error


def resolve_gsm_to_srx(gsm_accession, gsm_cache, http_cache_dir, timeout, retries, backoff_seconds):
    if gsm_accession in gsm_cache:
        return gsm_cache[gsm_accession]

    cache_path = http_cache_dir / "geo" / f"{gsm_accession}.txt"
    geo_text = fetch_text(
        GEO_TEXT_URL.format(accession=gsm_accession),
        cache_path,
        timeout,
        retries,
        backoff_seconds,
    )
    srxs = dedupe_keep_order(GSM_SRA_RELATION_RE.findall(geo_text))

    if len(srxs) != 1:
        raise ValueError(
            f"{gsm_accession}: expected exactly 1 SRX from GEO relation, found {len(srxs)}"
        )

    gsm_cache[gsm_accession] = srxs[0]
    return srxs[0]


def resolve_srx_to_srrs(srx_accession, srx_cache, http_cache_dir, timeout, retries, backoff_seconds):
    if srx_accession in srx_cache:
        return srx_cache[srx_accession]

    cache_path = http_cache_dir / "runinfo" / f"{srx_accession}.csv"
    runinfo_text = fetch_text(
        RUNINFO_URL.format(accession=srx_accession),
        cache_path,
        timeout,
        retries,
        backoff_seconds,
    )
    reader = csv.DictReader(io.StringIO(runinfo_text))
    runs = []
    for row in reader:
        run = normalize_text(row.get("Run"))
        if run:
            runs.append(run)

    runs = dedupe_keep_order(runs)
    if not runs:
        raise ValueError(f"{srx_accession}: no SRR runs found in SRA runinfo")

    srx_cache[srx_accession] = runs
    return runs


def resolve_srr_to_srx(srr_accession, srr_cache, http_cache_dir, timeout, retries, backoff_seconds):
    if srr_accession in srr_cache:
        return srr_cache[srr_accession]

    cache_path = http_cache_dir / "runinfo" / f"{srr_accession}.csv"
    runinfo_text = fetch_text(
        RUNINFO_URL.format(accession=srr_accession),
        cache_path,
        timeout,
        retries,
        backoff_seconds,
    )
    reader = csv.DictReader(io.StringIO(runinfo_text))
    experiments = []
    for row in reader:
        experiment = normalize_text(row.get("Experiment"))
        if experiment:
            experiments.append(experiment)

    experiments = dedupe_keep_order(experiments)
    if len(experiments) != 1:
        raise ValueError(
            f"{srr_accession}: expected exactly 1 SRX from SRA runinfo, found {len(experiments)}"
        )

    srr_cache[srr_accession] = experiments[0]
    return experiments[0]


def collect_sample_cells(xlsx_file):
    sample_cells = []
    sample_type_counts = Counter()
    source_row_count = 0

    for row_num, values_by_col in iter_sheet_rows(xlsx_file):
        if row_num == 1:
            continue

        source_id = normalize_text(values_by_col.get(1))
        primary_tissue = normalize_text(values_by_col.get(2))
        if source_id:
            source_row_count += 1

        for col_num in sorted(col for col in values_by_col if col >= 4):
            original_cell_value = normalize_text(values_by_col.get(col_num))
            if not original_cell_value:
                continue

            input_accessions = extract_accessions(original_cell_value)
            if not input_accessions:
                continue

            sample_type_counts.update(detect_accession_type(acc) for acc in input_accessions)
            sample_cells.append(
                {
                    "legacy_sample_id": f"{source_id}_r{row_num}_c{col_num}",
                    "source_id": source_id,
                    "primary_tissue": primary_tissue,
                    "excel_row": row_num,
                    "excel_col": col_num,
                    "original_cell_value": original_cell_value,
                    "input_accessions": input_accessions,
                }
            )

    return sample_cells, sample_type_counts, source_row_count


def resolve_sample_cell(
    sample_cell,
    gsm_cache,
    srx_cache,
    srr_cache,
    http_cache_dir,
    timeout,
    retries,
    backoff_seconds,
):
    resolved_srxs = []
    final_srrs = []
    errors = []

    for accession in sample_cell["input_accessions"]:
        accession_type = detect_accession_type(accession)

        try:
            if accession_type == "GSM":
                srx = resolve_gsm_to_srx(
                    accession, gsm_cache, http_cache_dir, timeout, retries, backoff_seconds
                )
                resolved_srxs.append(srx)
                final_srrs.extend(
                    resolve_srx_to_srrs(
                        srx, srx_cache, http_cache_dir, timeout, retries, backoff_seconds
                    )
                )
            elif accession_type == "SRX":
                resolved_srxs.append(accession)
                final_srrs.extend(
                    resolve_srx_to_srrs(
                        accession, srx_cache, http_cache_dir, timeout, retries, backoff_seconds
                    )
                )
            elif accession_type == "SRR":
                resolved_srxs.append(
                    resolve_srr_to_srx(
                        accession, srr_cache, http_cache_dir, timeout, retries, backoff_seconds
                    )
                )
                final_srrs.append(accession)
            else:
                raise ValueError(f"{accession}: unsupported accession type")
        except Exception as exc:
            errors.append(str(exc))

    resolved_srxs = dedupe_keep_order(resolved_srxs)
    final_srrs = dedupe_keep_order(final_srrs)

    if len(resolved_srxs) > 1:
        errors.append(f"Multiple SRXs resolved for sample: {','.join(resolved_srxs)}")

    if not final_srrs and not errors:
        errors.append("No final SRRs resolved for sample")

    status = "error" if errors else "ok"
    sample_id = resolved_srxs[0] if len(resolved_srxs) == 1 else sample_cell["legacy_sample_id"]
    return {
        "sample_id": sample_id,
        "legacy_sample_id": sample_cell["legacy_sample_id"],
        "source_id": sample_cell["source_id"],
        "primary_tissue": sample_cell["primary_tissue"],
        "excel_row": sample_cell["excel_row"],
        "excel_col": sample_cell["excel_col"],
        "original_cell_value": sample_cell["original_cell_value"],
        "input_accessions": ",".join(sample_cell["input_accessions"]),
        "resolved_srxs": ",".join(resolved_srxs),
        "final_srrs": ",".join(final_srrs),
        "num_srrs": str(len(final_srrs)),
        "status": status,
        "error": " | ".join(errors),
    }


def assign_unique_sample_ids(resolved_rows):
    counts = Counter(row["sample_id"] for row in resolved_rows)
    seen = Counter()

    for row in resolved_rows:
        base_sample_id = row["sample_id"]
        seen[base_sample_id] += 1
        if counts[base_sample_id] > 1:
            row["sample_id"] = f"{base_sample_id}__dup{seen[base_sample_id]}"


def write_tsv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_prefetch_list(path, resolved_rows):
    unique_srrs = []
    seen = set()

    for row in resolved_rows:
        if row["status"] != "ok":
            continue
        for srr in [value for value in row["final_srrs"].split(",") if value]:
            if srr not in seen:
                seen.add(srr)
                unique_srrs.append(srr)

    with path.open("w", encoding="utf-8") as handle:
        for srr in unique_srrs:
            handle.write(f"{srr}\n")

    return unique_srrs


def validate_outputs(resolved_rows, prefetch_srrs):
    for row in resolved_rows:
        if row["status"] != "ok":
            continue
        srrs = [value for value in row["final_srrs"].split(",") if value]
        if int(row["num_srrs"]) != len(srrs):
            raise AssertionError(
                f"{row['sample_id']}: num_srrs={row['num_srrs']} but final_srrs has {len(srrs)} items"
            )

    unfolded_srrs = []
    seen = set()
    for row in resolved_rows:
        if row["status"] != "ok":
            continue
        for srr in [value for value in row["final_srrs"].split(",") if value]:
            if srr not in seen:
                seen.add(srr)
                unfolded_srrs.append(srr)

    if unfolded_srrs != prefetch_srrs:
        raise AssertionError("prefetch_srr_list.txt does not match unique SRRs from sample_srr_map.tsv")


def build_output_paths(output_dir):
    root = output_dir.resolve()
    return {
        "root": root,
        "metadata_dir": root / "flattened_metadata",
        "fastq_dir": root / "fastq",
        "final_fastq_dir": root / "fastq" / "finalfastq_forbam",
        "sra_dir": root / "sra_cache",
        "tmp_dir": root / "fasterq_tmp",
        "cache_dir": root / "cache",
        "http_cache_dir": root / "cache" / "http",
        "validation_dir": root / "validation_reports",
    }


def ensure_directories(paths):
    for key in (
        "root",
        "metadata_dir",
        "fastq_dir",
        "final_fastq_dir",
        "sra_dir",
        "tmp_dir",
        "cache_dir",
        "http_cache_dir",
        "validation_dir",
    ):
        paths[key].mkdir(parents=True, exist_ok=True)


def write_run_manifest(paths, xlsx_path, args):
    manifest_path = paths["metadata_dir"] / "pipeline_manifest.tsv"
    compress_threads = args.compress_threads if args.compress_threads is not None else args.threads
    rows = [
        ("xlsx_path", str(xlsx_path.resolve())),
        ("output_dir", str(paths["root"])),
        ("metadata_dir", str(paths["metadata_dir"])),
        ("fastq_dir", str(paths["fastq_dir"])),
        ("final_fastq_dir", str(paths["final_fastq_dir"])),
        ("sra_dir", str(paths["sra_dir"])),
        ("tmp_dir", str(paths["tmp_dir"])),
        ("cache_dir", str(paths["cache_dir"])),
        ("validation_dir", str(paths["validation_dir"])),
        ("toolkit_bin", args.toolkit_bin),
        ("threads", str(args.threads)),
        ("compress_threads", str(compress_threads)),
        ("request_timeout", str(args.request_timeout)),
        ("request_retries", str(args.request_retries)),
        ("request_backoff_seconds", str(args.request_backoff_seconds)),
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["key", "value"])
        writer.writerows(rows)


def generate_flattened_metadata(xlsx_path, metadata_dir, http_cache_dir, args):
    print(f"Input xlsx: {xlsx_path}")
    print(f"Flattened metadata dir: {metadata_dir}")

    sample_cells, sample_type_counts, source_row_count = collect_sample_cells(xlsx_path)
    print(f"Rows with source_id: {source_row_count}")
    print(f"Sample cells with accessions: {len(sample_cells)}")
    print(f"Accession token counts: {dict(sample_type_counts)}")

    gsm_cache = {}
    srx_cache = {}
    srr_cache = {}
    resolved_rows = []

    for index, sample_cell in enumerate(sample_cells, start=1):
        if index % 25 == 0 or index == len(sample_cells):
            print(f"Resolving sample {index}/{len(sample_cells)}")
        resolved_rows.append(
            resolve_sample_cell(
                sample_cell,
                gsm_cache,
                srx_cache,
                srr_cache,
                http_cache_dir,
                args.request_timeout,
                args.request_retries,
                args.request_backoff_seconds,
            )
        )

    assign_unique_sample_ids(resolved_rows)

    sample_srr_map_path = metadata_dir / "sample_srr_map.tsv"
    unresolved_samples_path = metadata_dir / "unresolved_samples.tsv"
    prefetch_srr_list_path = metadata_dir / "prefetch_srr_list.txt"

    write_tsv(sample_srr_map_path, resolved_rows, OUTPUT_COLUMNS)

    unresolved_rows = [row for row in resolved_rows if row["status"] == "error"]
    write_tsv(unresolved_samples_path, unresolved_rows, OUTPUT_COLUMNS)

    prefetch_srrs = write_prefetch_list(prefetch_srr_list_path, resolved_rows)
    validate_outputs(resolved_rows, prefetch_srrs)

    print(sample_srr_map_path)
    print(unresolved_samples_path)
    print(prefetch_srr_list_path)
    print(f"Resolved samples: {sum(row['status'] == 'ok' for row in resolved_rows)}")
    print(f"Unresolved samples: {len(unresolved_rows)}")
    print(f"Unique SRRs for fetch: {len(prefetch_srrs)}")
    print(f"GSM cache size: {len(gsm_cache)}")
    print(f"SRX cache size: {len(srx_cache)}")
    print(f"SRR cache size: {len(srr_cache)}")

    if unresolved_rows:
        raise SystemExit(1)


def copy_helper_scripts(output_dir):
    script_dir = Path(__file__).resolve().parent
    helper_names = [
        "download_and_merge_fastq.sh",
        "validate_rpfdb_downloads.sh",
    ]

    copied_paths = []
    for helper_name in helper_names:
        source = script_dir / helper_name
        if not source.is_file():
            raise FileNotFoundError(f"Missing helper script template: {source}")
        target = output_dir / helper_name
        shutil.copyfile(source, target)
        target.chmod(0o755)
        copied_paths.append(target)

    return copied_paths


def run_helper_script(
    script_path,
    dataset_dir,
    toolkit_bin,
    threads=None,
    compress_threads=None,
    extra_env=None,
):
    env = os.environ.copy()
    env["DATASET_DIR"] = str(dataset_dir)
    env["SRA_TOOLKIT_BIN"] = toolkit_bin
    if threads is not None:
        env["THREADS"] = str(threads)
    if compress_threads is not None:
        env["COMPRESS_THREADS"] = str(compress_threads)
    if extra_env:
        env.update(extra_env)

    subprocess.run([str(script_path), str(dataset_dir)], check=True, env=env)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Flatten the first sheet of an RPFdb-style xlsx, then optionally "
            "download/merge FASTQ files and run vdb-validate."
        )
    )
    parser.add_argument("xlsx_path", help="Input xlsx file. The first worksheet is used.")
    parser.add_argument("output_dir", help="Dataset output directory to create or reuse.")
    parser.add_argument(
        "--toolkit-bin",
        default=DEFAULT_SRA_TOOLKIT_BIN,
        help="Directory containing prefetch, fasterq-dump, and vdb-validate.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Thread count for fasterq-dump.",
    )
    parser.add_argument(
        "--compress-threads",
        type=int,
        default=None,
        help="Thread count for pigz compression. Defaults to --threads.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip prefetch and fasterq-dump. Merge/final FASTQ creation can still run.",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip creation of final sample-level FASTQ files.",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip the vdb-validate phase.",
    )
    parser.add_argument(
        "--write-md5",
        action="store_true",
        help="Also record local md5sum values during validation.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds for GEO/SRA metadata requests.",
    )
    parser.add_argument(
        "--request-retries",
        type=int,
        default=3,
        help="Number of retries for GEO/SRA metadata requests.",
    )
    parser.add_argument(
        "--request-backoff-seconds",
        type=float,
        default=1.5,
        help="Base backoff multiplier in seconds for metadata request retries.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    compress_threads = args.compress_threads if args.compress_threads is not None else args.threads
    xlsx_path = Path(args.xlsx_path).expanduser().resolve()
    if not xlsx_path.is_file():
        raise SystemExit(f"Input xlsx not found: {xlsx_path}")

    output_dir = Path(args.output_dir).expanduser()
    paths = build_output_paths(output_dir)
    ensure_directories(paths)

    generate_flattened_metadata(xlsx_path, paths["metadata_dir"], paths["http_cache_dir"], args)
    write_run_manifest(paths, xlsx_path, args)
    copied_scripts = copy_helper_scripts(paths["root"])

    print("Helper scripts:")
    for copied_script in copied_scripts:
        print(copied_script)

    download_script = paths["root"] / "download_and_merge_fastq.sh"
    validate_script = paths["root"] / "validate_rpfdb_downloads.sh"

    download_env = {}
    if args.skip_download:
        download_env["SKIP_DOWNLOAD"] = "1"
    if args.skip_merge:
        download_env["SKIP_MERGE"] = "1"

    run_helper_script(
        download_script,
        paths["root"],
        args.toolkit_bin,
        threads=args.threads,
        compress_threads=compress_threads,
        extra_env=download_env,
    )

    if not args.skip_validate:
        validate_env = {"WRITE_MD5": "1" if args.write_md5 else "0"}
        run_helper_script(
            validate_script,
            paths["root"],
            args.toolkit_bin,
            compress_threads=compress_threads,
            extra_env=validate_env,
        )
    else:
        print("Skipping validation phase")


if __name__ == "__main__":
    main()
