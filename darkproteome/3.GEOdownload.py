import csv
import io
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path


xlsx_path = "/home/jiye/jiye/darkproteome/RPFdb/riboseq_kidney.xlsx"
out_dir = Path("/home/jiye/jiye/darkproteome/RPFdb/flattened_metadata")
out_dir.mkdir(parents=True, exist_ok=True)

GEO_TEXT_URL = (
    "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}"
    "&targ=self&view=full&form=text"
)
RUNINFO_URL = "https://trace.ncbi.nlm.nih.gov/Traces/sra-db-be/runinfo?acc={accession}"

REQUEST_TIMEOUT = 30
REQUEST_RETRIES = 3
REQUEST_BACKOFF_SECONDS = 1.5

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


def extract_accessions(cell_value):
    text = normalize_text(cell_value)
    if not text:
        return []
    return ACCESSION_RE.findall(text)


def detect_accession_type(accession):
    if accession.startswith("GSM"):
        return "GSM"
    if accession.startswith("SRX"):
        return "SRX"
    if accession.startswith("SRR"):
        return "SRR"
    return "UNKNOWN"


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


def fetch_text(url):
    last_error = None
    headers = {"User-Agent": "Mozilla/5.0 (compatible; GEOdownload/1.0)"}

    for attempt in range(1, REQUEST_RETRIES + 1):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                return response.read().decode("utf-8", "replace")
        except Exception as exc:
            last_error = exc
            if attempt == REQUEST_RETRIES:
                break
            time.sleep(REQUEST_BACKOFF_SECONDS * attempt)

    raise RuntimeError(f"Request failed after {REQUEST_RETRIES} attempts: {url}") from last_error


def resolve_gsm_to_srx(gsm_accession, gsm_cache):
    if gsm_accession in gsm_cache:
        return gsm_cache[gsm_accession]

    geo_text = fetch_text(GEO_TEXT_URL.format(accession=gsm_accession))
    srxs = dedupe_keep_order(GSM_SRA_RELATION_RE.findall(geo_text))

    if len(srxs) != 1:
        raise ValueError(
            f"{gsm_accession}: expected exactly 1 SRX from GEO relation, found {len(srxs)}"
        )

    gsm_cache[gsm_accession] = srxs[0]
    return srxs[0]


def resolve_srx_to_srrs(srx_accession, srx_cache):
    if srx_accession in srx_cache:
        return srx_cache[srx_accession]

    runinfo_text = fetch_text(RUNINFO_URL.format(accession=srx_accession))
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


def resolve_srr_to_srx(srr_accession, srr_cache):
    if srr_accession in srr_cache:
        return srr_cache[srr_accession]

    runinfo_text = fetch_text(RUNINFO_URL.format(accession=srr_accession))
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


def resolve_sample_cell(sample_cell, gsm_cache, srx_cache, srr_cache):
    resolved_srxs = []
    final_srrs = []
    errors = []

    for accession in sample_cell["input_accessions"]:
        accession_type = detect_accession_type(accession)

        try:
            if accession_type == "GSM":
                srx = resolve_gsm_to_srx(accession, gsm_cache)
                resolved_srxs.append(srx)
                final_srrs.extend(resolve_srx_to_srrs(srx, srx_cache))
            elif accession_type == "SRX":
                resolved_srxs.append(accession)
                final_srrs.extend(resolve_srx_to_srrs(accession, srx_cache))
            elif accession_type == "SRR":
                resolved_srxs.append(resolve_srr_to_srx(accession, srr_cache))
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


def main():
    print(f"Input: {xlsx_path}")
    print(f"Output dir: {out_dir}")

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
        resolved_rows.append(resolve_sample_cell(sample_cell, gsm_cache, srx_cache, srr_cache))

    assign_unique_sample_ids(resolved_rows)

    sample_srr_map_path = out_dir / "sample_srr_map.tsv"
    unresolved_samples_path = out_dir / "unresolved_samples.tsv"
    prefetch_srr_list_path = out_dir / "prefetch_srr_list.txt"

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


if __name__ == "__main__":
    main()
