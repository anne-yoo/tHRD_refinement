#!/usr/bin/env python3

import argparse
import csv
import time
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


GSM_RE = re.compile(r"\bGSM\d+\b")
XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
PKG_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
SRX_EFETCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=sra&id={accession}&rettype=full&retmode=xml"
)
REQUEST_TIMEOUT = 30
REQUEST_RETRIES = 3
REQUEST_BACKOFF_SECONDS = 1.5

MANIFEST_COLUMNS = [
    "sample_id",
    "legacy_sample_id",
    "source_id",
    "excel_row",
    "excel_col",
    "input_accessions",
    "resolved_srxs",
    "base_gsm",
    "target_sample_name",
    "gsm_source",
    "naming_mode",
    "source_fastq_layout",
    "source_fastq_files",
    "target_fastq_files",
    "status",
    "note",
]


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


def dedupe_keep_order(values):
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def extract_gsms(text):
    return dedupe_keep_order(GSM_RE.findall(normalize_text(text)))


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


def get_sheet_paths(workbook_zip):
    workbook = ET.fromstring(workbook_zip.read("xl/workbook.xml"))
    rels = ET.fromstring(workbook_zip.read("xl/_rels/workbook.xml.rels"))
    rel_map = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall(f"{PKG_REL_NS}Relationship")
    }

    sheets = workbook.find(f"{XLSX_NS}sheets")
    if sheets is None or not list(sheets):
        raise ValueError("No worksheets found in workbook")

    output = []
    for sheet in list(sheets):
        rel_id = sheet.attrib.get(
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        )
        target = rel_map[rel_id]
        if target.startswith("/"):
            target = target.lstrip("/")
        elif not target.startswith("xl/"):
            target = f"xl/{target}"
        output.append((sheet.attrib.get("name", ""), target))
    return output


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


def iter_sheet_rows(xlsx_path, sheet_index):
    with zipfile.ZipFile(xlsx_path) as workbook_zip:
        shared_strings = parse_shared_strings(workbook_zip)
        sheet_paths = get_sheet_paths(workbook_zip)
        if sheet_index >= len(sheet_paths):
            raise IndexError(
                f"Workbook has {len(sheet_paths)} sheet(s); sheet index {sheet_index} is out of range"
            )
        sheet_name, sheet_path = sheet_paths[sheet_index]
        sheet_root = ET.fromstring(workbook_zip.read(sheet_path))

    sheet_data = sheet_root.find(f"{XLSX_NS}sheetData")
    if sheet_data is None:
        return sheet_name, []

    rows = []
    for row_elem in sheet_data.findall(f"{XLSX_NS}row"):
        row_num = int(row_elem.attrib["r"])
        values_by_col = {}
        for cell_elem in row_elem.findall(f"{XLSX_NS}c"):
            cell_ref = cell_elem.attrib["r"]
            col_match = re.match(r"[A-Z]+", cell_ref)
            if not col_match:
                continue
            col_num = col_to_num(col_match.group(0))
            values_by_col[col_num] = get_cell_text(cell_elem, shared_strings)
        rows.append((row_num, values_by_col))
    return sheet_name, rows


def load_second_sheet_riboseq_gsms(xlsx_path):
    _, rows = iter_sheet_rows(xlsx_path, sheet_index=1)
    rows_by_num = {row_num: values_by_col for row_num, values_by_col in rows}
    if not rows_by_num:
        return {}

    mapping = defaultdict(list)
    max_row = max(rows_by_num)
    for ribo_row_num in range(2, max_row + 1, 2):
        ribo_row = rows_by_num.get(ribo_row_num, {})
        if not ribo_row:
            continue

        source_id = normalize_text(ribo_row.get(1))
        if not source_id:
            continue

        for col_num, value in sorted(ribo_row.items()):
            if col_num < 4:
                continue
            gsms = extract_gsms(value)
            if not gsms:
                continue
            key = (source_id, str(col_num))
            mapping[key].extend(gsms)

    return {key: dedupe_keep_order(values) for key, values in mapping.items()}


def load_manifest_riboseq_gsms(manifest_path):
    if not manifest_path or not manifest_path.is_file():
        return {}

    mapping = defaultdict(list)
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            source_id = normalize_text(row.get("source_id"))
            excel_col = normalize_text(row.get("excel_col"))
            riboseq_gsm = normalize_text(row.get("riboseq_gsm"))
            if not source_id or not excel_col or not riboseq_gsm:
                continue
            mapping[(source_id, excel_col)].append(riboseq_gsm)

    return {key: dedupe_keep_order(values) for key, values in mapping.items()}


def load_sample_rows(sample_srr_map_path):
    if not sample_srr_map_path.is_file():
        raise FileNotFoundError(f"Missing sample_srr_map.tsv: {sample_srr_map_path}")

    rows = []
    with sample_srr_map_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if normalize_text(row.get("status")) != "ok":
                continue
            rows.append(row)
    return rows


def fetch_text_with_cache(url, cache_path):
    if cache_path is not None and cache_path.is_file():
        return cache_path.read_text(encoding="utf-8")

    headers = {"User-Agent": "Mozilla/5.0 (compatible; link_riboseq_fastq_by_gsm/1.0)"}
    last_error = None
    for attempt in range(1, REQUEST_RETRIES + 1):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                text = response.read().decode("utf-8", "replace")
            if cache_path is not None:
                ensure_dir(cache_path.parent)
                cache_path.write_text(text, encoding="utf-8")
            return text
        except Exception as exc:
            last_error = exc
            if attempt == REQUEST_RETRIES:
                break
            time.sleep(REQUEST_BACKOFF_SECONDS * attempt)

    raise RuntimeError(f"Request failed after {REQUEST_RETRIES} attempts: {url}") from last_error


def resolve_srx_to_gsm_via_sra_xml(srx_accession, cache_dir, srx_gsm_cache):
    if srx_accession in srx_gsm_cache:
        return srx_gsm_cache[srx_accession]

    cache_path = None
    if cache_dir is not None:
        cache_path = cache_dir / f"{srx_accession}.xml"

    xml_text = fetch_text_with_cache(SRX_EFETCH_URL.format(accession=srx_accession), cache_path)
    root = ET.fromstring(xml_text)

    candidates = []
    for experiment in root.findall(".//EXPERIMENT"):
        alias = normalize_text(experiment.attrib.get("alias"))
        if GSM_RE.fullmatch(alias):
            candidates.append(alias)

    for external_id in root.findall(".//EXTERNAL_ID"):
        namespace = normalize_text(external_id.attrib.get("namespace")).upper()
        text = normalize_text(external_id.text)
        if namespace == "GEO" and GSM_RE.fullmatch(text):
            candidates.append(text)

    title_text = " ".join(normalize_text(title.text) for title in root.findall(".//TITLE"))
    candidates.extend(extract_gsms(title_text))

    candidates = dedupe_keep_order(candidates)
    if len(candidates) == 1:
        srx_gsm_cache[srx_accession] = candidates[0]
        return candidates[0]

    srx_gsm_cache[srx_accession] = ""
    return ""


def pick_single_gsm(candidate_values, source_name):
    deduped = dedupe_keep_order([value for value in candidate_values if value])
    if not deduped:
        return "", "", ""
    if len(deduped) == 1:
        return deduped[0], source_name, ""
    return "", source_name, f"Conflicting GSM candidates in {source_name}: {','.join(deduped)}"


def resolve_base_gsm(sample_row, manifest_map, second_sheet_map, srx_cache_dir, srx_gsm_cache):
    source_id = normalize_text(sample_row.get("source_id"))
    excel_col = normalize_text(sample_row.get("excel_col"))
    lookup_key = (source_id, excel_col)

    metadata_gsms = extract_gsms(sample_row.get("input_accessions"))
    manifest_gsms = manifest_map.get(lookup_key, [])
    second_sheet_gsms = second_sheet_map.get(lookup_key, [])

    sample_id = normalize_text(sample_row.get("sample_id"))
    resolved_srxs = dedupe_keep_order(
        [value.strip() for value in normalize_text(sample_row.get("resolved_srxs")).split(",") if value.strip()]
    )
    srx_lookup_gsm = ""
    if source_id.startswith("GSE") and not metadata_gsms:
        srx_candidates = resolved_srxs or ([sample_id] if sample_id.startswith("SRX") else [])
        if len(srx_candidates) == 1:
            srx_lookup_gsm = resolve_srx_to_gsm_via_sra_xml(
                srx_candidates[0], srx_cache_dir, srx_gsm_cache
            )

    for source_name, values in (
        ("sample_srr_map", metadata_gsms),
        ("sra_xml", [srx_lookup_gsm] if srx_lookup_gsm else []),
        ("finaldata_manifest", manifest_gsms),
        ("xlsx_second_sheet", second_sheet_gsms),
    ):
        gsm, gsm_source, note = pick_single_gsm(values, source_name)
        if gsm:
            return gsm, gsm_source, ""
        if note:
            return "", gsm_source, note

    source_parts = []
    if metadata_gsms:
        source_parts.append("sample_srr_map")
    if srx_lookup_gsm:
        source_parts.append("sra_xml")
    if manifest_gsms:
        source_parts.append("finaldata_manifest")
    if second_sheet_gsms:
        source_parts.append("xlsx_second_sheet")
    gsm_source = "+".join(source_parts)
    return "", gsm_source, "No GSM found in sample_srr_map, SRA XML, finaldata manifest, or xlsx second sheet"


def discover_source_fastqs(source_dir, sample_id):
    main_fastq = source_dir / f"{sample_id}.fastq.gz"
    read1_fastq = source_dir / f"{sample_id}_1.fastq.gz"
    read2_fastq = source_dir / f"{sample_id}_2.fastq.gz"

    if main_fastq.is_file() and not read1_fastq.exists() and not read2_fastq.exists():
        return "single", [main_fastq], ""

    if read1_fastq.is_file() and read2_fastq.is_file() and not main_fastq.exists():
        return "paired", [read1_fastq, read2_fastq], ""

    candidates = sorted(source_dir.glob(f"{sample_id}*.fastq.gz"))
    if not candidates:
        return "", [], f"No FASTQ found under {source_dir} for sample_id {sample_id}"

    if len(candidates) == 1:
        return "single", candidates, ""

    exact_pair_names = {f"{sample_id}_1.fastq.gz", f"{sample_id}_2.fastq.gz"}
    if len(candidates) == 2 and {path.name for path in candidates} == exact_pair_names:
        return "paired", sorted(candidates), ""

    return (
        "",
        candidates,
        f"Ambiguous FASTQ layout for {sample_id}: {','.join(path.name for path in candidates)}",
    )


def assign_unique_target_names(records):
    counts = Counter(record["base_gsm"] for record in records if record["status"] == "READY")
    seen = Counter()

    for record in records:
        if record["status"] != "READY":
            continue
        base_gsm = record["base_gsm"]
        seen[base_gsm] += 1
        if counts[base_gsm] == 1:
            record["target_sample_name"] = base_gsm
            continue
        if seen[base_gsm] == 1:
            record["target_sample_name"] = base_gsm
            record["note"] = (
                f"{record['note']} | Duplicate GSM target; first sample kept as plain GSM"
                if record["note"]
                else "Duplicate GSM target; first sample kept as plain GSM"
            )
        else:
            record["target_sample_name"] = f"{base_gsm}__dup{seen[base_gsm]}"
            record["note"] = (
                f"{record['note']} | Duplicate GSM target; suffixed target name"
                if record["note"]
                else "Duplicate GSM target; suffixed target name"
            )


def assign_fallback_target_names(records):
    for record in records:
        if record["status"] != "NO_GSM_MAPPING":
            continue

        sample_id = record["sample_id"]
        if not sample_id:
            continue

        record["target_sample_name"] = sample_id
        record["naming_mode"] = "srx_fallback"
        record["status"] = "READY_SRX_FALLBACK"
        record["note"] = (
            f"{record['note']} | Using sample_id as fallback target name"
            if record["note"]
            else "Using sample_id as fallback target name"
        )


def build_target_paths(output_dir, target_sample_name, layout):
    if layout == "single":
        return [output_dir / f"{target_sample_name}.fastq.gz"]
    if layout == "paired":
        return [
            output_dir / f"{target_sample_name}_1.fastq.gz",
            output_dir / f"{target_sample_name}_2.fastq.gz",
        ]
    raise ValueError(f"Unsupported layout: {layout}")


def write_manifest(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_links(args):
    riboseq_dir = args.riboseq_dir.expanduser().resolve()
    fastq_dir = riboseq_dir / "fastq" / "finalfastq_forbam"
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else riboseq_dir / "fastq" / "finalfastq_forbam_GSM"
    )
    manifest_path = (
        args.manifest_out.expanduser().resolve()
        if args.manifest_out
        else riboseq_dir / "fastq" / "finalfastq_forbam_GSM_manifest.tsv"
    )
    sample_srr_map_path = riboseq_dir / "flattened_metadata" / "sample_srr_map.tsv"
    finaldata_manifest_path = (
        args.match_manifest.expanduser().resolve()
        if args.match_manifest
        else riboseq_dir.parent / "finaldata" / "manifests" / "matched_rnaseq_link_manifest.tsv"
    )
    srx_cache_dir = (
        args.srx_cache_dir.expanduser().resolve()
        if args.srx_cache_dir
        else riboseq_dir / "cache" / "srx_to_gsm"
    )

    if not fastq_dir.is_dir():
        raise SystemExit(f"Source FASTQ directory not found: {fastq_dir}")

    ensure_dir(output_dir)
    sample_rows = load_sample_rows(sample_srr_map_path)
    manifest_map = load_manifest_riboseq_gsms(finaldata_manifest_path)
    second_sheet_map = {}
    if args.xlsx_path:
        second_sheet_map = load_second_sheet_riboseq_gsms(args.xlsx_path.expanduser().resolve())
    srx_gsm_cache = {}

    records = []
    for row in sample_rows:
        record = {
            "sample_id": normalize_text(row.get("sample_id")),
            "legacy_sample_id": normalize_text(row.get("legacy_sample_id")),
            "source_id": normalize_text(row.get("source_id")),
            "excel_row": normalize_text(row.get("excel_row")),
            "excel_col": normalize_text(row.get("excel_col")),
            "input_accessions": normalize_text(row.get("input_accessions")),
            "resolved_srxs": normalize_text(row.get("resolved_srxs")),
            "base_gsm": "",
            "target_sample_name": "",
            "gsm_source": "",
            "naming_mode": "",
            "source_fastq_layout": "",
            "source_fastq_files": "",
            "target_fastq_files": "",
            "status": "",
            "note": "",
        }

        base_gsm, gsm_source, gsm_note = resolve_base_gsm(
            row, manifest_map, second_sheet_map, srx_cache_dir, srx_gsm_cache
        )
        record["base_gsm"] = base_gsm
        record["gsm_source"] = gsm_source
        if gsm_note:
            record["status"] = "NO_GSM_MAPPING"
            record["note"] = gsm_note
            records.append(record)
            continue

        layout, source_fastqs, fastq_note = discover_source_fastqs(fastq_dir, record["sample_id"])
        record["source_fastq_layout"] = layout
        record["source_fastq_files"] = ",".join(str(path) for path in source_fastqs)
        if fastq_note:
            record["status"] = "FASTQ_NOT_RESOLVED"
            record["note"] = fastq_note
            records.append(record)
            continue

        record["status"] = "READY"
        record["naming_mode"] = "gsm"
        records.append(record)

    assign_unique_target_names(records)
    assign_fallback_target_names(records)

    for record in records:
        if record["status"] not in {"READY", "READY_SRX_FALLBACK"}:
            continue

        sample_id = record["sample_id"]
        layout, source_fastqs, _ = discover_source_fastqs(fastq_dir, sample_id)
        target_paths = build_target_paths(output_dir, record["target_sample_name"], layout)
        if len(source_fastqs) != len(target_paths):
            record["status"] = "TARGET_BUILD_ERROR"
            record["note"] = (
                f"Source/target file count mismatch: {len(source_fastqs)} vs {len(target_paths)}"
            )
            continue

        for source_path, target_path in zip(source_fastqs, target_paths):
            safe_symlink(source_path, target_path)

        record["source_fastq_layout"] = layout
        record["source_fastq_files"] = ",".join(str(path) for path in source_fastqs)
        record["target_fastq_files"] = ",".join(str(path) for path in target_paths)
        if record["naming_mode"] == "srx_fallback":
            record["status"] = "LINKED_SRX_FALLBACK"
        else:
            record["status"] = "LINKED"

    if args.clean_unmanaged_links:
        expected_targets = {
            target_path
            for record in records
            if record["status"] in {"LINKED", "LINKED_SRX_FALLBACK"}
            for target_path in [
                Path(path_text)
                for path_text in record["target_fastq_files"].split(",")
                if path_text
            ]
        }
        for child in output_dir.iterdir():
            if child not in expected_targets and (child.is_symlink() or child.is_file()):
                child.unlink()

    write_manifest(manifest_path, records)

    summary = Counter(record["status"] for record in records)
    print(f"Riboseq dir: {riboseq_dir}")
    print(f"Source FASTQ dir: {fastq_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Sample map: {sample_srr_map_path}")
    print(f"Finaldata manifest: {finaldata_manifest_path}")
    print(f"SRX->GSM cache dir: {srx_cache_dir}")
    print(f"Second-sheet xlsx: {args.xlsx_path if args.xlsx_path else 'not provided'}")
    print(f"Manifest written: {manifest_path}")
    print(f"Status counts: {dict(summary)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create GSM-named symlinks for Riboseq final FASTQ files. "
            "The script resolves sample_id(SRX) -> Riboseq GSM using sample_srr_map.tsv, "
            "finaldata matched_rnaseq_link_manifest.tsv, and optionally the second sheet of an xlsx workbook."
        )
    )
    parser.add_argument(
        "riboseq_dir",
        type=Path,
        help="Riboseq dataset directory such as /home/.../data/RPFdb/kidney/Riboseq",
    )
    parser.add_argument(
        "--match-manifest",
        type=Path,
        default=None,
        help=(
            "Optional finaldata matched manifest TSV. Defaults to "
            "<riboseq_dir>/../finaldata/manifests/matched_rnaseq_link_manifest.tsv"
        ),
    )
    parser.add_argument(
        "--xlsx-path",
        type=Path,
        default=None,
        help="Optional workbook path. If provided, the second sheet is used as an extra GSM fallback source.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for GSM-named symlinks. Default: <riboseq_dir>/fastq/finalfastq_forbam_GSM",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Output TSV manifest path. Default: <riboseq_dir>/fastq/finalfastq_forbam_GSM_manifest.tsv",
    )
    parser.add_argument(
        "--clean-unmanaged-links",
        action="store_true",
        help="Remove existing files/symlinks in the output directory that are not produced by the current run.",
    )
    parser.add_argument(
        "--srx-cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory for SRX->GSM SRA XML lookups. Default: <riboseq_dir>/cache/srx_to_gsm",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    build_links(args)


if __name__ == "__main__":
    main()
