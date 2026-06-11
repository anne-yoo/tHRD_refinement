#!/usr/bin/env python3
"""Create simple pilot feature tables for Ribo-seq ORF calls.

The script intentionally inspects and infers ORF caller columns instead of
assuming a fixed schema. Defaults point to the current pilot-study inputs.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd


DEFAULT_INPUT_DIR = Path("/home/jiye/jiye/darkproteome/pilotstudy/data")
DEFAULT_OUTPUT_DIR = Path("/home/jiye/jiye/darkproteome/pilotstudy/analysis")
DEFAULT_ORF_FILE = "Pancreas.4caller.merged.2caller.tsv"


@dataclass
class TranscriptRecord:
    requested_id: str
    gtf_transcript_id: str = ""
    gene_id: str = ""
    gene_name: str = ""
    transcript_type: str = ""
    chrom: str = ""
    strand: str = ""
    transcript_start: Optional[int] = None
    transcript_end: Optional[int] = None
    exons: List[Tuple[int, int]] = field(default_factory=list)
    cds: List[Tuple[int, int]] = field(default_factory=list)
    generic_utr: List[Tuple[int, int]] = field(default_factory=list)
    five_utr: List[Tuple[int, int]] = field(default_factory=list)
    three_utr: List[Tuple[int, int]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ORF and mother-transcript feature tables for the pilot Ribo-seq analysis."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--orf-file", type=Path, default=None)
    parser.add_argument("--gtf-file", type=Path, default=None)
    parser.add_argument("--fasta-file", type=Path, default=None)
    parser.add_argument(
        "--coordinate-system",
        choices=["zero-based-half-open", "one-based-closed"],
        default="zero-based-half-open",
        help="Coordinate system for genomic ORF start/end columns.",
    )
    return parser.parse_args()


def normalize_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def clean_scalar(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "na", "n/a", "."}:
        return ""
    return text


def first_nonmissing(values: Iterable[object]) -> str:
    for value in values:
        text = clean_scalar(value)
        if text:
            return text
    return ""


def stable_unique(values: Iterable[object]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for value in values:
        text = clean_scalar(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def to_int(value: object) -> Optional[int]:
    text = clean_scalar(value)
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def find_column(
    columns: Sequence[str],
    exact: Sequence[str] = (),
    contains_all: Sequence[str] = (),
    exclude_any: Sequence[str] = (),
) -> Optional[str]:
    normalized = {normalize_column(col): col for col in columns}
    for candidate in exact:
        hit = normalized.get(normalize_column(candidate))
        if hit is not None:
            return hit

    contains_norm = [normalize_column(token) for token in contains_all]
    exclude_norm = [normalize_column(token) for token in exclude_any]
    for col in columns:
        norm = normalize_column(col)
        if all(token in norm for token in contains_norm) and not any(
            token in norm for token in exclude_norm
        ):
            return col
    return None


def infer_orf_columns(df: pd.DataFrame) -> Dict[str, object]:
    columns = list(df.columns)
    colmap: Dict[str, object] = {
        "orf_id": find_column(columns, exact=["ORF_id", "orf_id"], contains_all=["orf", "id"]),
        "transcript_id": find_column(
            columns,
            exact=["transcript_id", "transcript", "tx_id", "mother_transcript_id"],
            contains_all=["transcript", "id"],
        ),
        "sample": find_column(columns, exact=["sample", "sample_id", "biosample"]),
        "caller": find_column(columns, exact=["caller", "callers", "orf_caller"]),
        "num_callers": find_column(
            columns,
            exact=["num_of_callers", "n_callers", "number_of_callers", "supporting_callers"],
            contains_all=["caller"],
        ),
        "chrom": find_column(columns, exact=["chr", "chrom", "chromosome", "seqname"]),
        "start": find_column(
            columns,
            exact=["start(0-based)", "start_0based", "start0", "orf_start", "start"],
            contains_all=["start"],
            exclude_any=["codon"],
        ),
        "end": find_column(
            columns,
            exact=["end(0-based)", "end_0based", "end0", "orf_end", "end"],
            contains_all=["end"],
        ),
        "strand": find_column(columns, exact=["strand"]),
        "start_codon": find_column(columns, exact=["start_codon", "startcodon"]),
    }

    type_cols: List[str] = []
    for col in columns:
        norm = normalize_column(col)
        if "orf" in norm and ("type" in norm or "category" in norm):
            type_cols.append(col)
        elif norm in {"type", "category", "orfcategory"}:
            type_cols.append(col)
    colmap["type_cols"] = type_cols
    return colmap


def infer_transcript_id(value: object) -> str:
    text = clean_scalar(value)
    if not text:
        return ""

    match = re.search(r"ENST\d+(?:\.\d+)?", text)
    if match:
        return match.group(0)

    if ":" in text:
        return text.split(":", 1)[0]
    return ""


def split_support_values(values: Iterable[object]) -> List[str]:
    found: Set[str] = set()
    for value in values:
        text = clean_scalar(value)
        if not text:
            continue
        for part in re.split(r"[|,;]+", text):
            part = part.strip()
            if part:
                found.add(part)
    return sorted(found)


def join_values(values: Iterable[str]) -> str:
    return "|".join([value for value in values if value])


def build_orf_key(df: pd.DataFrame, colmap: Dict[str, object]) -> pd.Series:
    orf_col = colmap.get("orf_id")
    if orf_col:
        return df[str(orf_col)].astype(str)

    key_cols = [
        str(colmap[col])
        for col in ["transcript_id", "chrom", "start", "end", "strand"]
        if colmap.get(col)
    ]
    if len(key_cols) < 3:
        raise ValueError(
            "Could not infer a unique ORF key. Need ORF_id or at least coordinate columns."
        )
    return df[key_cols].astype(str).agg(":".join, axis=1)


def collapse_orf_calls(df: pd.DataFrame, colmap: Dict[str, object]) -> pd.DataFrame:
    work = df.copy()
    work["_unique_orf_key"] = build_orf_key(work, colmap)

    transcript_col = colmap.get("transcript_id")
    orf_col = colmap.get("orf_id")
    if transcript_col:
        work["_mother_transcript_id"] = work[str(transcript_col)].map(clean_scalar)
    elif orf_col:
        work["_mother_transcript_id"] = work[str(orf_col)].map(infer_transcript_id)
    else:
        work["_mother_transcript_id"] = ""

    rows: List[Dict[str, object]] = []
    for key, group in work.groupby("_unique_orf_key", sort=False):
        row: Dict[str, object] = {
            "ORF_id": first_nonmissing(group[str(orf_col)]) if orf_col else key,
            "mother_transcript_id": first_nonmissing(group["_mother_transcript_id"]),
            "input_row_count": len(group),
        }

        for label, out_col in [
            ("chrom", "chrom"),
            ("start", "start_0based"),
            ("end", "end_0based"),
            ("strand", "strand"),
        ]:
            source_col = colmap.get(label)
            if source_col:
                row[out_col] = first_nonmissing(group[str(source_col)])

        start_codon_col = colmap.get("start_codon")
        row["start_codon"] = (
            join_values(stable_unique(group[str(start_codon_col)])) if start_codon_col else ""
        )

        for type_col in colmap.get("type_cols", []):
            row[str(type_col)] = join_values(stable_unique(group[str(type_col)]))

        caller_col = colmap.get("caller")
        callers = split_support_values(group[str(caller_col)]) if caller_col else []
        row["caller_list"] = join_values(callers)

        num_callers_col = colmap.get("num_callers")
        reported_counts: List[int] = []
        if num_callers_col:
            reported_counts = [
                parsed
                for parsed in (to_int(value) for value in group[str(num_callers_col)])
                if parsed is not None
            ]
        row["max_reported_num_callers"] = max(reported_counts) if reported_counts else ""
        row["supporting_callers"] = (
            len(callers)
            if callers
            else (max(reported_counts) if reported_counts else len(group))
        )

        sample_col = colmap.get("sample")
        samples = sorted(stable_unique(group[str(sample_col)])) if sample_col else []
        row["sample_list"] = join_values(samples)
        row["supporting_samples"] = len(samples) if samples else len(group)

        rows.append(row)

    return pd.DataFrame(rows)


def strip_version(transcript_id: str) -> str:
    return transcript_id.split(".", 1)[0] if transcript_id else ""


def build_target_lookup(target_ids: Iterable[str]) -> Tuple[Set[str], Dict[str, List[str]]]:
    exact = {tid for tid in target_ids if tid}
    stripped: Dict[str, List[str]] = defaultdict(list)
    for tid in exact:
        stripped[strip_version(tid)].append(tid)
    return exact, stripped


def resolve_requested_id(
    observed_id: str, exact_targets: Set[str], stripped_targets: Dict[str, List[str]]
) -> Optional[str]:
    if not exact_targets:
        return observed_id
    if observed_id in exact_targets:
        return observed_id
    stripped = strip_version(observed_id)
    if stripped in stripped_targets:
        return sorted(stripped_targets[stripped])[0]
    return None


def parse_gtf_attributes(attribute_text: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for item in attribute_text.rstrip(";").split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(None, 1)
        if len(parts) != 2:
            continue
        key, value = parts
        attrs[key] = value.strip().strip('"')
    return attrs


def utr_feature_class(feature: str) -> str:
    norm = normalize_column(feature)
    if norm == "utr":
        return "generic"
    if "utr" in norm and ("five" in norm or norm.startswith("5")):
        return "five"
    if "utr" in norm and ("three" in norm or norm.startswith("3")):
        return "three"
    return ""


def parse_gtf_features(gtf_path: Path, target_ids: Iterable[str]) -> Dict[str, TranscriptRecord]:
    exact_targets, stripped_targets = build_target_lookup(target_ids)
    records: Dict[str, TranscriptRecord] = {}
    useful_features = {"transcript", "exon", "CDS", "UTR"}

    with gtf_path.open() as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 9:
                continue
            chrom, _source, feature, start_text, end_text, _score, strand, _frame, attrs_text = parts
            if feature not in useful_features and not utr_feature_class(feature):
                continue

            attrs = parse_gtf_attributes(attrs_text)
            observed_tid = attrs.get("transcript_id")
            if not observed_tid:
                continue
            requested_id = resolve_requested_id(observed_tid, exact_targets, stripped_targets)
            if requested_id is None:
                continue

            rec = records.setdefault(requested_id, TranscriptRecord(requested_id=requested_id))
            rec.gtf_transcript_id = observed_tid
            rec.chrom = rec.chrom or chrom
            rec.strand = rec.strand or strand
            rec.gene_id = rec.gene_id or attrs.get("gene_id", "")
            rec.gene_name = rec.gene_name or attrs.get("gene_name", "")
            rec.transcript_type = rec.transcript_type or attrs.get(
                "transcript_type", attrs.get("gene_type", "")
            )

            start = int(start_text)
            end = int(end_text)
            if feature == "transcript":
                rec.transcript_start = start
                rec.transcript_end = end
            elif feature == "exon":
                rec.exons.append((start, end))
            elif feature == "CDS":
                rec.cds.append((start, end))
            else:
                utr_class = utr_feature_class(feature)
                if utr_class == "five":
                    rec.five_utr.append((start, end))
                elif utr_class == "three":
                    rec.three_utr.append((start, end))
                elif utr_class == "generic":
                    rec.generic_utr.append((start, end))

    return records


def merge_intervals(intervals: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    merged: List[Tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def interval_length(intervals: Sequence[Tuple[int, int]]) -> int:
    return sum(end - start + 1 for start, end in merge_intervals(intervals))


def bounded_overlap_length(start: int, end: int, bound_start: int, bound_end: int) -> int:
    overlap_start = max(start, bound_start)
    overlap_end = min(end, bound_end)
    if overlap_start > overlap_end:
        return 0
    return overlap_end - overlap_start + 1


def classify_utr_lengths(record: TranscriptRecord) -> Tuple[int, int, int]:
    five_len = interval_length(record.five_utr)
    three_len = interval_length(record.three_utr)
    unclassified_len = 0

    generic = merge_intervals(record.generic_utr)
    cds = merge_intervals(record.cds)
    if not generic:
        return five_len, three_len, unclassified_len
    if not cds:
        return five_len, three_len, interval_length(generic)

    cds_min = min(start for start, _end in cds)
    cds_max = max(end for _start, end in cds)
    for start, end in generic:
        lower_len = bounded_overlap_length(start, end, start, min(end, cds_min - 1))
        upper_len = bounded_overlap_length(start, end, max(start, cds_max + 1), end)
        inside_len = (end - start + 1) - lower_len - upper_len

        if record.strand == "-":
            three_len += lower_len
            five_len += upper_len
        else:
            five_len += lower_len
            three_len += upper_len
        unclassified_len += max(0, inside_len)

    return five_len, three_len, unclassified_len


def parse_fasta_features(
    fasta_path: Path, target_ids: Iterable[str]
) -> Dict[str, Dict[str, object]]:
    exact_targets, stripped_targets = build_target_lookup(target_ids)
    features: Dict[str, Dict[str, object]] = {}

    current_requested: Optional[str] = None
    seq_len = 0
    valid_bases = 0
    gc_bases = 0
    aug_count = 0
    carry = ""

    def flush_current() -> None:
        nonlocal current_requested, seq_len, valid_bases, gc_bases, aug_count, carry
        if current_requested is None:
            return
        features[current_requested] = {
            "transcript_sequence_length": seq_len,
            "gc_content": round(gc_bases / valid_bases, 6) if valid_bases else "",
            "num_aug_codons": aug_count,
        }
        current_requested = None
        seq_len = 0
        valid_bases = 0
        gc_bases = 0
        aug_count = 0
        carry = ""

    with fasta_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                flush_current()
                observed_id = line[1:].split()[0]
                current_requested = resolve_requested_id(
                    observed_id, exact_targets, stripped_targets
                )
                if current_requested in features:
                    current_requested = None
                if len(features) == len(exact_targets):
                    break
                continue

            if current_requested is None:
                continue

            chunk = line.upper().replace("T", "U")
            seq_len += len(chunk)
            valid_bases += sum(1 for base in chunk if base in {"A", "C", "G", "U"})
            gc_bases += sum(1 for base in chunk if base in {"G", "C"})

            scan = carry + chunk
            aug_count += sum(1 for idx in range(max(0, len(scan) - 2)) if scan[idx : idx + 3] == "AUG")
            carry = scan[-2:]

    flush_current()
    return features


def exonic_overlap_length(
    start_1based: int, end_1based: int, exons: Sequence[Tuple[int, int]]
) -> int:
    if start_1based > end_1based:
        start_1based, end_1based = end_1based, start_1based
    return sum(
        bounded_overlap_length(start_1based, end_1based, exon_start, exon_end)
        for exon_start, exon_end in merge_intervals(exons)
    )


def genomic_to_transcript_pos(
    coord_1based: int, exons: Sequence[Tuple[int, int]], strand: str
) -> Optional[int]:
    ordered_exons = sorted(exons, reverse=(strand == "-"))
    offset = 0
    for start, end in ordered_exons:
        length = end - start + 1
        if start <= coord_1based <= end:
            if strand == "-":
                return offset + (end - coord_1based + 1)
            return offset + (coord_1based - start + 1)
        offset += length
    return None


def genomic_interval_from_orf_row(
    row: pd.Series, coordinate_system: str
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    start = to_int(row.get("start_0based", ""))
    end = to_int(row.get("end_0based", ""))
    if start is None or end is None:
        return None, None, None

    if coordinate_system == "zero-based-half-open":
        start_1based = start + 1
        end_1based = end
        genomic_span = max(0, end - start)
    else:
        start_1based = start
        end_1based = end
        genomic_span = max(0, end - start + 1)
    return start_1based, end_1based, genomic_span


def add_orf_coordinate_features(
    orf_df: pd.DataFrame,
    transcript_records: Dict[str, TranscriptRecord],
    coordinate_system: str,
) -> pd.DataFrame:
    out = orf_df.copy()
    lengths: List[object] = []
    spans: List[object] = []
    frames: List[object] = []
    tx_starts: List[object] = []
    tx_ends: List[object] = []
    length_sources: List[str] = []

    for _idx, row in out.iterrows():
        start_1based, end_1based, genomic_span = genomic_interval_from_orf_row(
            row, coordinate_system
        )
        spans.append(genomic_span if genomic_span is not None else "")

        tid = clean_scalar(row.get("mother_transcript_id", ""))
        record = transcript_records.get(tid)
        strand = clean_scalar(row.get("strand", "")) or (record.strand if record else "")
        exons = record.exons if record else []

        if start_1based is None or end_1based is None:
            lengths.append("")
            frames.append("")
            tx_starts.append("")
            tx_ends.append("")
            length_sources.append("")
            continue

        orf_length = exonic_overlap_length(start_1based, end_1based, exons) if exons else 0
        if orf_length > 0:
            length_sources.append("gtf_exonic_overlap")
            lengths.append(orf_length)
        else:
            length_sources.append("genomic_span_fallback")
            lengths.append(genomic_span if genomic_span is not None else "")

        if exons and strand in {"+", "-"}:
            if strand == "-":
                translation_start_coord = end_1based
                translation_end_coord = start_1based
            else:
                translation_start_coord = start_1based
                translation_end_coord = end_1based
            tx_start = genomic_to_transcript_pos(translation_start_coord, exons, strand)
            tx_end = genomic_to_transcript_pos(translation_end_coord, exons, strand)
        else:
            tx_start = None
            tx_end = None

        tx_starts.append(tx_start if tx_start is not None else "")
        tx_ends.append(tx_end if tx_end is not None else "")
        frames.append((tx_start - 1) % 3 if tx_start is not None else "")

    out["orf_length"] = lengths
    out["orf_genomic_span"] = spans
    out["orf_frame"] = frames
    out["orf_start_transcript_pos"] = tx_starts
    out["orf_end_transcript_pos"] = tx_ends
    out["orf_length_source"] = length_sources
    return out


def build_transcript_feature_table(
    target_ids: Iterable[str],
    transcript_records: Dict[str, TranscriptRecord],
    fasta_features: Dict[str, Dict[str, object]],
    orf_counts: Counter,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for tid in sorted(set(target_ids)):
        record = transcript_records.get(tid, TranscriptRecord(requested_id=tid))
        five_len, three_len, unclassified_len = classify_utr_lengths(record)
        transcript_len = interval_length(record.exons)
        if not transcript_len and record.transcript_start and record.transcript_end:
            transcript_len = record.transcript_end - record.transcript_start + 1

        seq = fasta_features.get(tid, {})
        rows.append(
            {
                "transcript_id": tid,
                "gtf_transcript_id": record.gtf_transcript_id,
                "gene_id": record.gene_id,
                "gene_name": record.gene_name,
                "transcript_type": record.transcript_type,
                "chrom": record.chrom,
                "strand": record.strand,
                "transcript_length": transcript_len if transcript_len else "",
                "exon_count": len(record.exons) if record.exons else "",
                "cds_length": interval_length(record.cds),
                "utr5_length": five_len,
                "utr3_length": three_len,
                "utr_unclassified_length": unclassified_len,
                "transcript_sequence_length": seq.get("transcript_sequence_length", ""),
                "gc_content": seq.get("gc_content", ""),
                "num_aug_codons": seq.get("num_aug_codons", ""),
                "unique_orf_count": orf_counts.get(tid, 0),
            }
        )
    return pd.DataFrame(rows)


def attach_transcript_context(
    orf_df: pd.DataFrame, transcript_df: pd.DataFrame
) -> pd.DataFrame:
    context_cols = [
        "transcript_id",
        "gene_id",
        "gene_name",
        "transcript_type",
        "transcript_length",
        "exon_count",
        "cds_length",
        "utr5_length",
        "utr3_length",
        "transcript_sequence_length",
        "gc_content",
        "num_aug_codons",
    ]
    context = transcript_df[[col for col in context_cols if col in transcript_df.columns]].rename(
        columns={"transcript_id": "mother_transcript_id"}
    )
    return orf_df.merge(context, on="mother_transcript_id", how="left")


def choose_input_file(input_dir: Path, explicit: Optional[Path], patterns: Sequence[str]) -> Path:
    if explicit:
        return explicit
    for pattern in patterns:
        matches = sorted(input_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No input file found for patterns: {', '.join(patterns)}")


def write_summary_report(
    report_path: Path,
    orf_path: Path,
    transcript_path: Path,
    orf_file: Path,
    gtf_file: Path,
    fasta_file: Path,
    input_rows: int,
    orf_df: pd.DataFrame,
    transcript_df: pd.DataFrame,
    colmap: Dict[str, object],
    coordinate_system: str,
) -> None:
    type_lines: List[str] = []
    for col in ["ORF_type", "ORF_type2"]:
        if col in orf_df.columns:
            counts = orf_df[col].replace("", pd.NA).dropna().value_counts().head(10)
            if not counts.empty:
                type_lines.append(f"### {col}")
                type_lines.extend([f"- {idx}: {count}" for idx, count in counts.items()])

    caller_dist = orf_df["supporting_callers"].value_counts().sort_index()
    sample_dist = orf_df["supporting_samples"].value_counts().sort_index()
    gtf_found = transcript_df["gtf_transcript_id"].replace("", pd.NA).notna().sum()
    fasta_found = (
        transcript_df["transcript_sequence_length"].replace("", pd.NA).notna().sum()
        if "transcript_sequence_length" in transcript_df.columns
        else 0
    )

    lines = [
        "# Pilot ORF Feature Summary",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Inputs",
        f"- ORF caller table: `{orf_file}`",
        f"- GENCODE GTF: `{gtf_file}`",
        f"- Transcript FASTA: `{fasta_file}`",
        f"- ORF coordinate system: `{coordinate_system}`",
        "",
        "## Inferred ORF Columns",
    ]
    for key in [
        "orf_id",
        "transcript_id",
        "sample",
        "caller",
        "num_callers",
        "chrom",
        "start",
        "end",
        "strand",
        "start_codon",
    ]:
        lines.append(f"- {key}: `{colmap.get(key) or ''}`")
    type_cols = ", ".join(f"`{col}`" for col in colmap.get("type_cols", []))
    lines.append(f"- type/category columns: {type_cols or '`none`'}")

    lines.extend(
        [
            "",
            "## Output Counts",
            f"- Input ORF rows: {input_rows:,}",
            f"- Unique ORFs: {len(orf_df):,}",
            f"- Mother transcripts: {len(transcript_df):,}",
            f"- Mother transcripts found in GTF: {gtf_found:,}",
            f"- Mother transcripts found in FASTA: {fasta_found:,}",
            "",
            "## Supporting Caller Distribution",
        ]
    )
    lines.extend([f"- {idx} caller(s): {count}" for idx, count in caller_dist.items()])
    lines.append("")
    lines.append("## Supporting Sample Distribution")
    lines.extend([f"- {idx} sample(s): {count}" for idx, count in sample_dist.items()])

    if type_lines:
        lines.append("")
        lines.append("## ORF Type Counts")
        lines.extend(type_lines)

    lines.extend(
        [
            "",
            "## Notes",
            "- Duplicate ORF calls were collapsed by inferred unique ORF key, using `ORF_id` when available.",
            "- `orf_length` is the GTF exon-overlap length of the genomic ORF interval; `orf_genomic_span` keeps the raw genomic span.",
            "- `orf_frame` is the zero-based mother-transcript start position modulo 3.",
            "- `num_aug_codons` counts all ATG/AUG occurrences in the transcript sequence, regardless of frame.",
            "",
            "## Outputs",
            f"- ORF features: `{orf_path}`",
            f"- Transcript features: `{transcript_path}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    orf_file = choose_input_file(input_dir, args.orf_file, [DEFAULT_ORF_FILE, "*.merged*.tsv", "*.tsv"])
    gtf_file = choose_input_file(input_dir, args.gtf_file, ["*.gtf"])
    fasta_file = choose_input_file(
        input_dir, args.fasta_file, ["*.transcripts.fasta", "*.transcripts.fa", "*.fasta", "*.fa"]
    )

    print(f"Reading ORF calls: {orf_file}")
    orf_raw = pd.read_csv(orf_file, sep="\t", low_memory=False)
    print(f"Input ORF rows: {len(orf_raw):,}")
    print("Input ORF columns: " + ", ".join(map(str, orf_raw.columns)))

    colmap = infer_orf_columns(orf_raw)
    if not colmap.get("orf_id") and not (
        colmap.get("start") and colmap.get("end") and colmap.get("chrom")
    ):
        raise ValueError(
            "The ORF table needs either ORF_id or enough coordinate columns to define unique ORFs."
        )

    orf_collapsed = collapse_orf_calls(orf_raw, colmap)
    target_transcripts = sorted(
        {tid for tid in orf_collapsed["mother_transcript_id"].map(clean_scalar) if tid}
    )
    print(f"Collapsed unique ORFs: {len(orf_collapsed):,}")
    print(f"Mother transcripts inferred from ORFs: {len(target_transcripts):,}")

    print(f"Parsing GTF for mother transcripts: {gtf_file}")
    transcript_records = parse_gtf_features(gtf_file, target_transcripts)
    print(f"Mother transcripts found in GTF: {len(transcript_records):,}")

    print(f"Parsing transcript FASTA: {fasta_file}")
    fasta_features = parse_fasta_features(fasta_file, target_transcripts)
    print(f"Mother transcripts found in FASTA: {len(fasta_features):,}")

    orf_counts = Counter(orf_collapsed["mother_transcript_id"])
    transcript_features = build_transcript_feature_table(
        target_transcripts, transcript_records, fasta_features, orf_counts
    )
    orf_features = add_orf_coordinate_features(
        orf_collapsed, transcript_records, args.coordinate_system
    )
    orf_features = attach_transcript_context(orf_features, transcript_features)

    preferred_orf_cols = [
        "ORF_id",
        "mother_transcript_id",
        "gene_id",
        "gene_name",
        "transcript_type",
        "chrom",
        "start_0based",
        "end_0based",
        "strand",
        "orf_length",
        "orf_genomic_span",
        "orf_frame",
        "orf_start_transcript_pos",
        "orf_end_transcript_pos",
        "orf_length_source",
        "start_codon",
        "ORF_type",
        "ORF_type2",
        "supporting_callers",
        "supporting_samples",
        "caller_list",
        "sample_list",
        "input_row_count",
        "max_reported_num_callers",
        "transcript_length",
        "exon_count",
        "cds_length",
        "utr5_length",
        "utr3_length",
        "transcript_sequence_length",
        "gc_content",
        "num_aug_codons",
    ]
    orf_features = orf_features[
        [col for col in preferred_orf_cols if col in orf_features.columns]
        + [col for col in orf_features.columns if col not in preferred_orf_cols]
    ]

    transcript_path = output_dir / "transcript_features.tsv"
    orf_path = output_dir / "orf_features.tsv"
    report_path = output_dir / "summary_report.md"

    transcript_features.to_csv(transcript_path, sep="\t", index=False, na_rep="")
    orf_features.to_csv(orf_path, sep="\t", index=False, na_rep="")
    write_summary_report(
        report_path=report_path,
        orf_path=orf_path,
        transcript_path=transcript_path,
        orf_file=orf_file,
        gtf_file=gtf_file,
        fasta_file=fasta_file,
        input_rows=len(orf_raw),
        orf_df=orf_features,
        transcript_df=transcript_features,
        colmap=colmap,
        coordinate_system=args.coordinate_system,
    )

    print(f"Wrote {orf_path}")
    print(f"Wrote {transcript_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
