#!/usr/bin/env python3

import argparse
import csv
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path


XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
PKG_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
GSM_RE = re.compile(r"\bGSM\d+\b")


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip()


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


def extract_first_gsm(value):
    match = GSM_RE.search(normalize_text(value))
    return match.group(0) if match else ""


def load_match_pairs_from_second_sheet(xlsx_path):
    sheet_name, rows = iter_sheet_rows(xlsx_path, sheet_index=1)
    if not rows:
        raise ValueError(f"Second sheet is empty in {xlsx_path}")

    rows_by_num = {row_num: values_by_col for row_num, values_by_col in rows}
    max_row = max(rows_by_num)
    pairs = []

    for ribo_row_num in range(2, max_row + 1, 2):
        ribo_row = rows_by_num.get(ribo_row_num, {})
        rna_row = rows_by_num.get(ribo_row_num + 1, {})
        if not ribo_row and not rna_row:
            continue

        source_id = normalize_text(ribo_row.get(1))
        primary_tissue = normalize_text(ribo_row.get(2))
        max_col = max(
            [col for col in ribo_row.keys() if col >= 4] + [col for col in rna_row.keys() if col >= 4],
            default=3,
        )

        for col_num in range(4, max_col + 1):
            ribo_gsm = extract_first_gsm(ribo_row.get(col_num, ""))
            rna_gsm = extract_first_gsm(rna_row.get(col_num, ""))
            if not ribo_gsm and not rna_gsm:
                continue
            if not ribo_gsm:
                continue

            pairs.append(
                {
                    "sheet_name": sheet_name,
                    "source_id": source_id,
                    "primary_tissue": primary_tissue,
                    "riboseq_row": ribo_row_num,
                    "rnaseq_row": ribo_row_num + 1,
                    "excel_col": col_num,
                    "riboseq_gsm": ribo_gsm,
                    "rnaseq_gsm": rna_gsm,
                }
            )

    return pairs


def load_rnaseq_metadata_map(sample_srr_map_path):
    if not sample_srr_map_path.is_file():
        raise FileNotFoundError(f"Missing RNAseq sample map: {sample_srr_map_path}")

    gsm_to_rows = {}
    with sample_srr_map_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("status") != "ok":
                continue
            input_accessions = [part.strip() for part in normalize_text(row.get("input_accessions")).split(",") if part.strip()]
            gsms = [acc for acc in input_accessions if GSM_RE.fullmatch(acc)]
            for gsm in gsms:
                gsm_to_rows.setdefault(gsm, []).append(row)

    return gsm_to_rows


def find_alignment_files(alignments_root, sample_id):
    sample_dir_star = alignments_root / "star" / sample_id
    sample_dir_minimap2 = alignments_root / "minimap2" / sample_id

    bam_candidates = []
    bai_candidates = []

    if sample_dir_star.is_dir():
        bam_candidates.extend(sample_dir_star.glob("*.bam"))
        bai_candidates.extend(sample_dir_star.glob("*.bam.bai"))
        bai_candidates.extend(sample_dir_star.glob("*.bai"))

    if sample_dir_minimap2.is_dir():
        bam_candidates.extend(sample_dir_minimap2.glob("*.bam"))
        bai_candidates.extend(sample_dir_minimap2.glob("*.bam.bai"))
        bai_candidates.extend(sample_dir_minimap2.glob("*.bai"))

    bam_candidates = sorted({path.resolve() for path in bam_candidates})
    bai_candidates = sorted({path.resolve() for path in bai_candidates})
    return bam_candidates, bai_candidates


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def safe_symlink(source, target):
    ensure_dir(target.parent)
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(source)


def write_manifest(path, rows):
    fieldnames = [
        "sheet_name",
        "source_id",
        "primary_tissue",
        "riboseq_row",
        "rnaseq_row",
        "excel_col",
        "riboseq_gsm",
        "rnaseq_gsm",
        "rnaseq_sample_id",
        "bam_source",
        "bai_source",
        "rnaseq_target_dir",
        "riboseq_target_dir",
        "status",
        "note",
    ]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def prune_unmatched_directories(finaldata_rnaseq_dir, finaldata_riboseq_dir, manifest_rows):
    linked_ribo_gsms = {
        normalize_text(row.get("riboseq_gsm"))
        for row in manifest_rows
        if normalize_text(row.get("status")) == "LINKED" and normalize_text(row.get("riboseq_gsm"))
    }

    removed = {"RNAseq": 0, "Riboseq": 0}
    for label, root_dir in (("RNAseq", finaldata_rnaseq_dir), ("Riboseq", finaldata_riboseq_dir)):
        if not root_dir.is_dir():
            continue
        for child in root_dir.iterdir():
            if not child.is_dir():
                continue
            if child.name in linked_ribo_gsms:
                continue
            shutil.rmtree(child)
            removed[label] += 1
    return removed


def build_finaldata(args):
    organ_dir = args.organ_dir.expanduser().resolve()
    rnaseq_dir = organ_dir / "RNAseq"
    finaldata_dir = organ_dir / "finaldata"
    finaldata_rnaseq_dir = finaldata_dir / "RNAseq"
    finaldata_riboseq_dir = finaldata_dir / "Riboseq"
    manifests_dir = finaldata_dir / "manifests"

    sample_srr_map_path = rnaseq_dir / "flattened_metadata" / "sample_srr_map.tsv"
    alignments_root = rnaseq_dir / "processing" / "alignments"

    ensure_dir(finaldata_rnaseq_dir)
    ensure_dir(finaldata_riboseq_dir)
    ensure_dir(manifests_dir)

    match_pairs = load_match_pairs_from_second_sheet(args.xlsx_path)
    gsm_to_rows = load_rnaseq_metadata_map(sample_srr_map_path)

    manifest_rows = []

    for pair in match_pairs:
        ribo_gsm = pair["riboseq_gsm"]
        rna_gsm = pair["rnaseq_gsm"]
        rnaseq_target_dir = finaldata_rnaseq_dir / ribo_gsm
        riboseq_target_dir = finaldata_riboseq_dir / ribo_gsm

        manifest = {
            **pair,
            "rnaseq_sample_id": "",
            "bam_source": "",
            "bai_source": "",
            "rnaseq_target_dir": str(rnaseq_target_dir),
            "riboseq_target_dir": str(riboseq_target_dir),
            "status": "",
            "note": "",
        }

        if not rna_gsm:
            manifest["status"] = "NO_MATCHED_RNASEQ_GSM"
            manifest["note"] = "RNAseq matched cell is blank"
            manifest_rows.append(manifest)
            continue

        matching_rows = gsm_to_rows.get(rna_gsm, [])
        if not matching_rows:
            manifest["status"] = "RNASEQ_GSM_NOT_IN_METADATA"
            manifest["note"] = "Matched RNAseq GSM was not found in RNAseq sample_srr_map.tsv"
            manifest_rows.append(manifest)
            continue

        if len(matching_rows) > 1:
            manifest["status"] = "MULTIPLE_METADATA_ROWS"
            manifest["note"] = f"Matched RNAseq GSM maps to {len(matching_rows)} metadata rows"
            manifest_rows.append(manifest)
            continue

        sample_row = matching_rows[0]
        sample_id = normalize_text(sample_row.get("sample_id"))
        manifest["rnaseq_sample_id"] = sample_id

        bam_candidates, bai_candidates = find_alignment_files(alignments_root, sample_id)
        if len(bam_candidates) != 1:
            manifest["status"] = "BAM_NOT_RESOLVED"
            manifest["note"] = f"Expected exactly 1 BAM for {sample_id}, found {len(bam_candidates)}"
            if bam_candidates:
                manifest["bam_source"] = ",".join(str(path) for path in bam_candidates)
            manifest_rows.append(manifest)
            continue

        if len(bai_candidates) == 0:
            manifest["status"] = "BAI_NOT_FOUND"
            manifest["bam_source"] = str(bam_candidates[0])
            manifest["note"] = f"No BAI found for {sample_id}"
            manifest_rows.append(manifest)
            continue

        if len(bai_candidates) > 1:
            matching_bai = [path for path in bai_candidates if path.name.startswith(bam_candidates[0].name)]
            if len(matching_bai) == 1:
                bai_candidates = matching_bai
            else:
                manifest["status"] = "BAI_NOT_RESOLVED"
                manifest["bam_source"] = str(bam_candidates[0])
                manifest["bai_source"] = ",".join(str(path) for path in bai_candidates)
                manifest["note"] = f"Expected exactly 1 BAI for {sample_id}, found {len(bai_candidates)}"
                manifest_rows.append(manifest)
                continue

        bam_source = bam_candidates[0]
        bai_source = bai_candidates[0]
        ensure_dir(rnaseq_target_dir)
        ensure_dir(riboseq_target_dir)
        target_bam = rnaseq_target_dir / f"{ribo_gsm}.bam"
        target_bai = rnaseq_target_dir / f"{ribo_gsm}.bam.bai"

        safe_symlink(bam_source, target_bam)
        safe_symlink(bai_source, target_bai)

        manifest["bam_source"] = str(bam_source)
        manifest["bai_source"] = str(bai_source)
        manifest["status"] = "LINKED"
        manifest["note"] = ""
        manifest_rows.append(manifest)

    manifest_path = manifests_dir / "matched_rnaseq_link_manifest.tsv"
    write_manifest(manifest_path, manifest_rows)
    removed_counts = {"RNAseq": 0, "Riboseq": 0}
    if args.prune_unmatched_existing:
        removed_counts = prune_unmatched_directories(
            finaldata_rnaseq_dir, finaldata_riboseq_dir, manifest_rows
        )

    summary = Counter(row["status"] for row in manifest_rows)
    print(f"Input xlsx: {args.xlsx_path}")
    print(f"Organ dir: {organ_dir}")
    print(f"Finaldata dir: {finaldata_dir}")
    print(f"Pairs parsed: {len(match_pairs)}")
    print(f"Manifest: {manifest_path}")
    print(f"Status counts: {dict(summary)}")
    if args.prune_unmatched_existing:
        print(f"Pruned RNAseq dirs: {removed_counts['RNAseq']}")
        print(f"Pruned Riboseq dirs: {removed_counts['Riboseq']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build finaldata/RNAseq and finaldata/Riboseq directories from the "
            "second sheet of an RNAseq xlsx workbook, and symlink matched RNAseq BAM/BAI "
            "files using matched Riboseq GSM names."
        )
    )
    parser.add_argument("xlsx_path", type=Path, help="RNAseq workbook path")
    parser.add_argument(
        "organ_dir",
        type=Path,
        help="Organ root directory such as /home/.../data/RPFdb/pancreas or kidney",
    )
    parser.add_argument(
        "--prune-unmatched-existing",
        action="store_true",
        help=(
            "After rebuilding the manifest, remove existing finaldata/RNAseq and "
            "finaldata/Riboseq sample directories whose Riboseq GSM is not LINKED."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.xlsx_path = args.xlsx_path.expanduser().resolve()
    if not args.xlsx_path.is_file():
        raise SystemExit(f"Input xlsx not found: {args.xlsx_path}")
    build_finaldata(args)


if __name__ == "__main__":
    main()
