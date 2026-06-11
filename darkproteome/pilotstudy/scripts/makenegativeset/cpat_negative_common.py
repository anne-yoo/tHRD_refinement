#!/usr/bin/env python3
"""Shared utilities for creating CPAT-derived negative ORF sets."""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


BASE_DIR = Path(os.environ.get("BASE_DIR", "/home/jiye/jiye/darkproteome/ORFstudy/pilot"))
INPUT_DIR = Path(
    os.environ.get("INPUT_DIR", "/home/jiye/jiye/darkproteome/ORFstudy/pilot/pancreas8samples")
)
CPAT_DIR = Path(os.environ.get("CPAT_DIR", str(INPUT_DIR / "CPAT2")))
OUT_DIR = Path(os.environ.get("OUT_DIR", str(INPUT_DIR / "cpat_negative_orfs")))
GENOME_FA = Path(os.environ.get("GENOME_FA", str(BASE_DIR / "hg38.fa")))
GENCODE_GTF = Path(os.environ.get("GENCODE_GTF", str(BASE_DIR / "gencode.v48.annotation.gtf")))
CPAT_CUTOFF = float(os.environ.get("CPAT_CUTOFF", "0.364"))
SAMPLE_SIZE = int(os.environ.get("NEGATIVE_ORF_SAMPLE_SIZE", "25863"))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))
PROGRESS_EVERY = int(os.environ.get("PROGRESS_EVERY", "100000"))

TABLE_DIR = OUT_DIR / "tables"
BED_DIR = OUT_DIR / "bed"
FASTA_DIR = OUT_DIR / "fasta"
LOG_DIR = OUT_DIR / "logs"
SCRIPT_DIR = OUT_DIR / "scripts"

REVCOMP_TABLE = str.maketrans("ACGTUNacgtun", "TGCAANtgcaan")
ENST_RE = re.compile(r"(ENST\d+(?:\.\d+)?)")


@dataclass
class TranscriptModel:
    transcript_id: str
    gene_id: str
    gene_name: str
    transcript_type: str
    chrom: str
    strand: str
    exons: List[Tuple[int, int]]
    exon_tx_ranges: List[Tuple[int, int, int, int]]
    transcript_length: int


def ensure_dirs() -> None:
    for path in [TABLE_DIR, BED_DIR, FASTA_DIR, LOG_DIR, SCRIPT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def archive_script(script_path: str) -> None:
    ensure_dirs()
    src = Path(script_path)
    if src.exists():
        shutil.copy2(src, SCRIPT_DIR / src.name)
    helper = Path(__file__)
    if helper.exists() and helper.name != src.name:
        shutil.copy2(helper, SCRIPT_DIR / helper.name)


def print_header(script_name: str, inputs: Sequence[Path], outputs: Sequence[Path]) -> None:
    print(script_name)
    print(f"BASE_DIR={BASE_DIR}")
    print(f"INPUT_DIR={INPUT_DIR}")
    print(f"CPAT_DIR={CPAT_DIR}")
    print(f"OUT_DIR={OUT_DIR}")
    for path in inputs:
        print(f"input={path}")
    for path in outputs:
        print(f"output={path}")


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "na", "n/a", "."}:
        return ""
    return text


def normalize_codon(value: object) -> str:
    text = clean_text(value).upper().replace("U", "T")
    return re.split(r"[\s,;|/]+", text)[0] if text else ""


def reverse_complement(seq: Optional[str]) -> Optional[str]:
    if seq is None:
        return None
    return seq.translate(REVCOMP_TABLE)[::-1]


def parse_gtf_attributes(attr_text: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for match in re.finditer(r'(\S+)\s+"([^"]*)"', attr_text):
        attrs[match.group(1)] = match.group(2)
    return attrs


def transcript_type_from_attrs(attrs: Dict[str, str]) -> str:
    for key in ["transcript_type", "transcript_biotype", "gene_type", "gene_biotype"]:
        value = clean_text(attrs.get(key, ""))
        if value:
            return value
    return ""


def is_noncoding_transcript_type(transcript_type: object) -> bool:
    lower = clean_text(transcript_type).lower()
    return lower not in {"protein_coding", "coding"}


def strip_gene_version(gene_id: object) -> str:
    return re.sub(r"\.\d+$", "", clean_text(gene_id))


def extract_transcript_id_from_cpat_orf_id(orf_id: object) -> str:
    text = clean_text(orf_id)
    match = re.match(r"(.+)_ORF_\d+$", text)
    if match:
        return match.group(1)
    match = ENST_RE.search(text)
    return match.group(1) if match else text


def infer_transcript_ids_from_table(df: pd.DataFrame) -> pd.Series:
    for column in ["transcript_id", "Transcript_ID", "transcript", "transcriptId"]:
        if column in df.columns:
            return df[column].map(clean_text)
    if "ORF_id" in df.columns:
        return df["ORF_id"].map(lambda value: clean_text(value).split(":", 1)[0])

    ids: List[str] = []
    for _, row in df.iterrows():
        found = ""
        for value in row.astype(str):
            match = ENST_RE.search(value)
            if match:
                found = match.group(1)
                break
        ids.append(found)
    return pd.Series(ids, index=df.index)


def require_columns(df: pd.DataFrame, required: Sequence[str], path: Path) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise SystemExit(f"Missing required column(s) in {path}: {', '.join(missing)}")


def read_tsv(path: Path, required: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Input table not found: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str)
    if required:
        require_columns(df, required, path)
    return df


def read_fasta_dict(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise SystemExit(f"FASTA file not found: {path}")
    records: Dict[str, str] = {}
    header = None
    chunks: List[str] = []
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records[header] = "".join(chunks).upper().replace("U", "T")
                header = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
        if header is not None:
            records[header] = "".join(chunks).upper().replace("U", "T")
    return records


def write_fasta(records: Dict[str, str], ids: Iterable[str], path: Path, width: int = 60) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    missing = []
    with path.open("w") as handle:
        for record_id in ids:
            seq = records.get(record_id)
            if seq is None:
                missing.append(record_id)
                continue
            handle.write(f">{record_id}\n")
            for start in range(0, len(seq), width):
                handle.write(seq[start : start + width] + "\n")
    if missing:
        print(f"WARNING: {len(missing)} sampled ORF sequence(s) missing from FASTA; first={missing[0]}")


def parse_gtf_transcripts(gtf_path: Path) -> pd.DataFrame:
    if not gtf_path.exists():
        raise SystemExit(f"GENCODE GTF not found: {gtf_path}")
    rows: List[Dict[str, object]] = []
    fallback: Dict[str, Dict[str, object]] = {}
    with gtf_path.open() as handle:
        for raw_line in handle:
            if not raw_line or raw_line.startswith("#"):
                continue
            parts = raw_line.rstrip("\n").split("\t")
            if len(parts) != 9:
                continue
            chrom, source, feature, start, end, score, strand, frame, attrs_text = parts
            attrs = parse_gtf_attributes(attrs_text)
            transcript_id = clean_text(attrs.get("transcript_id", ""))
            if not transcript_id:
                continue
            if feature == "transcript":
                rows.append(
                    {
                        "transcript_id": transcript_id,
                        "gene_id": clean_text(attrs.get("gene_id", "")),
                        "gene_name": clean_text(attrs.get("gene_name", "")),
                        "gene_type": clean_text(attrs.get("gene_type", attrs.get("gene_biotype", ""))),
                        "transcript_type": transcript_type_from_attrs(attrs),
                        "chr": chrom,
                        "start_1based": int(start),
                        "end_1based": int(end),
                        "strand": strand,
                        "source": source,
                    }
                )
            elif feature == "exon" and transcript_id not in fallback:
                fallback[transcript_id] = {
                    "transcript_id": transcript_id,
                    "gene_id": clean_text(attrs.get("gene_id", "")),
                    "gene_name": clean_text(attrs.get("gene_name", "")),
                    "gene_type": clean_text(attrs.get("gene_type", attrs.get("gene_biotype", ""))),
                    "transcript_type": transcript_type_from_attrs(attrs),
                    "chr": chrom,
                    "start_1based": int(start),
                    "end_1based": int(end),
                    "strand": strand,
                    "source": source,
                }

    transcript_df = pd.DataFrame(rows)
    if transcript_df.empty and fallback:
        transcript_df = pd.DataFrame(fallback.values())
    elif fallback:
        missing_ids = set(fallback) - set(transcript_df["transcript_id"])
        if missing_ids:
            transcript_df = pd.concat(
                [transcript_df, pd.DataFrame([fallback[tx] for tx in sorted(missing_ids)])],
                ignore_index=True,
            )
    if transcript_df.empty:
        raise SystemExit(f"No transcript records found in {gtf_path}")
    transcript_df["is_noncoding"] = transcript_df["transcript_type"].map(is_noncoding_transcript_type)
    return transcript_df.drop_duplicates(subset=["transcript_id"], keep="first").reset_index(drop=True)


def parse_gtf_exon_models(gtf_path: Path) -> Dict[str, TranscriptModel]:
    transcript_meta: Dict[str, Dict[str, str]] = {}
    exons: Dict[str, List[Tuple[int, int]]] = {}
    if not gtf_path.exists():
        raise SystemExit(f"GENCODE GTF not found: {gtf_path}")

    with gtf_path.open() as handle:
        for raw_line in handle:
            if not raw_line or raw_line.startswith("#"):
                continue
            parts = raw_line.rstrip("\n").split("\t")
            if len(parts) != 9:
                continue
            chrom, _source, feature, start, end, _score, strand, _frame, attrs_text = parts
            attrs = parse_gtf_attributes(attrs_text)
            transcript_id = clean_text(attrs.get("transcript_id", ""))
            if not transcript_id:
                continue
            if feature in {"transcript", "exon"} and transcript_id not in transcript_meta:
                transcript_meta[transcript_id] = {
                    "transcript_id": transcript_id,
                    "gene_id": clean_text(attrs.get("gene_id", "")),
                    "gene_name": clean_text(attrs.get("gene_name", "")),
                    "transcript_type": transcript_type_from_attrs(attrs),
                    "chr": chrom,
                    "strand": strand,
                }
            if feature == "exon":
                exons.setdefault(transcript_id, []).append((int(start), int(end)))

    models: Dict[str, TranscriptModel] = {}
    for transcript_id, exon_list in exons.items():
        meta = transcript_meta.get(transcript_id, {})
        strand = meta.get("strand", "")
        if strand == "+":
            ordered = sorted(exon_list, key=lambda item: (item[0], item[1]))
        elif strand == "-":
            ordered = sorted(exon_list, key=lambda item: (item[0], item[1]), reverse=True)
        else:
            continue

        tx_ranges: List[Tuple[int, int, int, int]] = []
        cursor = 1
        for exon_start, exon_end in ordered:
            exon_len = exon_end - exon_start + 1
            tx_start = cursor
            tx_end = cursor + exon_len - 1
            tx_ranges.append((tx_start, tx_end, exon_start, exon_end))
            cursor = tx_end + 1

        models[transcript_id] = TranscriptModel(
            transcript_id=transcript_id,
            gene_id=meta.get("gene_id", ""),
            gene_name=meta.get("gene_name", ""),
            transcript_type=meta.get("transcript_type", ""),
            chrom=meta.get("chr", ""),
            strand=strand,
            exons=ordered,
            exon_tx_ranges=tx_ranges,
            transcript_length=cursor - 1,
        )
    return models


def map_transcript_pos_to_genome(model: TranscriptModel, tx_pos: int) -> int:
    if tx_pos < 1 or tx_pos > model.transcript_length:
        raise ValueError(f"transcript position {tx_pos} outside transcript length {model.transcript_length}")
    for tx_start, tx_end, exon_start, exon_end in model.exon_tx_ranges:
        if tx_start <= tx_pos <= tx_end:
            offset = tx_pos - tx_start
            if model.strand == "+":
                return exon_start + offset
            return exon_end - offset
    raise ValueError(f"transcript position {tx_pos} could not be mapped")


def extract_transcript_orf_sequence(model: TranscriptModel, start_tx: int, end_tx: int, chrom_seq: str) -> str:
    chunks: List[str] = []
    for tx_start, tx_end, exon_start, exon_end in model.exon_tx_ranges:
        overlap_start = max(start_tx, tx_start)
        overlap_end = min(end_tx, tx_end)
        if overlap_start > overlap_end:
            continue
        if model.strand == "+":
            g_start = exon_start + (overlap_start - tx_start)
            g_end = exon_start + (overlap_end - tx_start)
            chunks.append(chrom_seq[g_start - 1 : g_end])
        else:
            g_high = exon_end - (overlap_start - tx_start)
            g_low = exon_end - (overlap_end - tx_start)
            segment = chrom_seq[g_low - 1 : g_high]
            chunks.append(reverse_complement(segment) or "")
    return "".join(chunks).upper()


def resolve_chrom_key(fasta: Dict[str, object], chrom: str) -> Optional[str]:
    if chrom in fasta:
        return chrom
    alt = chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"
    if alt in fasta:
        return alt
    return None


def genomic_interval_from_transcript_orf(model: TranscriptModel, start_tx: int, end_tx: int) -> Tuple[int, int]:
    g1 = map_transcript_pos_to_genome(model, start_tx)
    g2 = map_transcript_pos_to_genome(model, end_tx)
    low = min(g1, g2)
    high = max(g1, g2)
    return low - 1, high


def write_bed(df: pd.DataFrame, path: Path) -> None:
    bed = df[["chr", "start0", "end0", "ORF_id", "strand"]].copy()
    bed.insert(4, "score", 0)
    bed[["chr", "start0", "end0", "ORF_id", "score", "strand"]].to_csv(
        path, sep="\t", index=False, header=False
    )

