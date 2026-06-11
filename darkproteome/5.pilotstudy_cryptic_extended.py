#!/usr/bin/env python3
"""Extended ORF-level sequence and positional features for cryptic ORFs."""

from __future__ import annotations

import argparse
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_ANALYSIS_DIR = Path("/home/jiye/jiye/darkproteome/pilotstudy/analysis")
DEFAULT_DATA_DIR = Path("/home/jiye/jiye/darkproteome/pilotstudy/data")
NEAR_COGNATE_STARTS = {"CTG", "GTG", "TTG", "ACG", "ATT", "ATC", "ATA", "AGG", "AAG"}


@dataclass
class TranscriptModel:
    transcript_id: str
    chrom: str = ""
    strand: str = ""
    exons: List[Tuple[int, int]] = field(default_factory=list)
    cds: List[Tuple[int, int]] = field(default_factory=list)

    def ordered_exons(self) -> List[Tuple[int, int]]:
        return sorted(self.exons, reverse=(self.strand == "-"))

    def genome_to_transcript_pos(self, coord_1based: int) -> Optional[int]:
        offset = 0
        for start, end in self.ordered_exons():
            length = end - start + 1
            if start <= coord_1based <= end:
                if self.strand == "-":
                    return offset + (end - coord_1based + 1)
                return offset + (coord_1based - start + 1)
            offset += length
        return None

    def cds_start_stop_transcript_pos(self) -> Tuple[Optional[int], Optional[int]]:
        if not self.cds:
            return None, None
        if self.strand == "-":
            cds_start_genomic = max(end for _start, end in self.cds)
            cds_stop_genomic = min(start for start, _end in self.cds)
        else:
            cds_start_genomic = min(start for start, _end in self.cds)
            cds_stop_genomic = max(end for _start, end in self.cds)
        return (
            self.genome_to_transcript_pos(cds_start_genomic),
            self.genome_to_transcript_pos(cds_stop_genomic),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extend cryptic ORF features with sequence-context and positional features."
    )
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--cryptic-orf-features", type=Path, default=None)
    parser.add_argument("--orf-features", type=Path, default=None)
    parser.add_argument("--transcript-features", type=Path, default=None)
    parser.add_argument("--gtf-file", type=Path, default=DEFAULT_DATA_DIR / "gencode.v48.annotation.gtf")
    parser.add_argument(
        "--transcript-fasta",
        type=Path,
        default=DEFAULT_DATA_DIR / "gencode.v48.annotation.gtf.transcripts.fasta",
    )
    parser.add_argument("--figure-dir", type=Path, default=None)
    return parser.parse_args()


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "na", "n/a", "."}:
        return ""
    return text


def to_float(value: object) -> float:
    text = clean_text(value)
    if not text:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def to_int(value: object) -> Optional[int]:
    value = to_float(value)
    if pd.isna(value):
        return None
    return int(round(float(value)))


def numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def normalize_codon(value: object) -> str:
    text = clean_text(value).upper().replace("U", "T")
    text = re.sub(r"[^ACGTN]", "", text)
    return text


def primary_codon(value: object) -> str:
    text = clean_text(value)
    if not text:
        return ""
    for token in re.split(r"[|,;]+", text):
        codon = normalize_codon(token)
        if codon:
            return codon
    return normalize_codon(text)


def start_type(codon: str) -> str:
    codon = normalize_codon(codon)
    if codon == "ATG":
        return "ATG"
    if codon in NEAR_COGNATE_STARTS:
        return "near_cognate"
    if codon:
        return "other"
    return "missing"


def gc_content(seq: str) -> float:
    seq = seq.upper().replace("U", "T")
    bases = [base for base in seq if base in {"A", "C", "G", "T"}]
    if not bases:
        return np.nan
    return (bases.count("G") + bases.count("C")) / len(bases)


def count_motifs(seq: str, motifs: Iterable[str]) -> int:
    motifs = {motif.upper().replace("U", "T") for motif in motifs}
    seq = seq.upper().replace("U", "T")
    return sum(1 for idx in range(max(0, len(seq) - 2)) if seq[idx : idx + 3] in motifs)


def density_per_kb(count: object, length: object) -> float:
    count_val = to_float(count)
    length_val = to_float(length)
    if pd.isna(count_val) or pd.isna(length_val) or length_val <= 0:
        return np.nan
    return count_val / length_val * 1000.0


def subseq_with_padding(seq: str, start_1based: int, end_1based: int) -> str:
    out: List[str] = []
    for pos in range(start_1based, end_1based + 1):
        if 1 <= pos <= len(seq):
            out.append(seq[pos - 1])
        else:
            out.append("N")
    return "".join(out)


def parse_gtf_attributes(attribute_text: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for item in attribute_text.rstrip(";").split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(None, 1)
        if len(parts) != 2:
            continue
        attrs[parts[0]] = parts[1].strip().strip('"')
    return attrs


def build_target_lookup(target_ids: Iterable[str]) -> Tuple[set, Dict[str, List[str]]]:
    exact = {clean_text(tid) for tid in target_ids if clean_text(tid)}
    stripped: Dict[str, List[str]] = defaultdict(list)
    for tid in exact:
        stripped[tid.split(".", 1)[0]].append(tid)
    return exact, stripped


def resolve_transcript_id(
    observed_id: str, exact_targets: set, stripped_targets: Dict[str, List[str]]
) -> Optional[str]:
    if observed_id in exact_targets:
        return observed_id
    stripped = observed_id.split(".", 1)[0]
    if stripped in stripped_targets:
        return sorted(stripped_targets[stripped])[0]
    return None


def parse_gtf_models(gtf_path: Path, target_ids: Iterable[str]) -> Dict[str, TranscriptModel]:
    exact_targets, stripped_targets = build_target_lookup(target_ids)
    models: Dict[str, TranscriptModel] = {}
    with gtf_path.open() as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 9:
                continue
            chrom, _source, feature, start_text, end_text, _score, strand, _frame, attrs_text = parts
            if feature not in {"exon", "CDS"}:
                continue
            attrs = parse_gtf_attributes(attrs_text)
            observed_tid = attrs.get("transcript_id", "")
            tid = resolve_transcript_id(observed_tid, exact_targets, stripped_targets)
            if tid is None:
                continue
            model = models.setdefault(tid, TranscriptModel(transcript_id=tid))
            model.chrom = model.chrom or chrom
            model.strand = model.strand or strand
            interval = (int(start_text), int(end_text))
            if feature == "exon":
                model.exons.append(interval)
            elif feature == "CDS":
                model.cds.append(interval)
    return models


def parse_target_fasta(fasta_path: Path, target_ids: Iterable[str]) -> Dict[str, str]:
    exact_targets, stripped_targets = build_target_lookup(target_ids)
    seqs: Dict[str, List[str]] = {}
    current_id: Optional[str] = None
    with fasta_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                observed_id = line[1:].split()[0]
                current_id = resolve_transcript_id(observed_id, exact_targets, stripped_targets)
                if current_id in seqs:
                    current_id = None
                elif current_id:
                    seqs[current_id] = []
                if len(seqs) == len(exact_targets):
                    break
                continue
            if current_id:
                seqs[current_id].append(line.upper().replace("U", "T"))
    return {tid: "".join(chunks) for tid, chunks in seqs.items()}


def map_orf_to_transcript(row: pd.Series, model: Optional[TranscriptModel]) -> Tuple[object, object]:
    if model is None or not model.exons:
        return np.nan, np.nan
    start0 = to_int(row.get("start_0based", ""))
    end0 = to_int(row.get("end_0based", ""))
    if start0 is None or end0 is None:
        return np.nan, np.nan
    genomic_start_1based = start0 + 1
    genomic_end_1based = end0
    strand = clean_text(row.get("strand", "")) or model.strand
    if strand == "-":
        orf_start_genomic = genomic_end_1based
        orf_end_genomic = genomic_start_1based
    else:
        orf_start_genomic = genomic_start_1based
        orf_end_genomic = genomic_end_1based
    return (
        model.genome_to_transcript_pos(orf_start_genomic),
        model.genome_to_transcript_pos(orf_end_genomic),
    )


def kozak_strength(score: object) -> str:
    score_val = to_int(score)
    if score_val is None:
        return "missing"
    if score_val >= 2:
        return "strong"
    if score_val == 1:
        return "moderate"
    return "weak"


def choose_coordinate(mapped: object, existing: object) -> float:
    mapped_val = to_float(mapped)
    if not pd.isna(mapped_val):
        return mapped_val
    return to_float(existing)


def add_mapped_and_sequence_features(
    orf_df: pd.DataFrame,
    models: Dict[str, TranscriptModel],
    transcript_seqs: Dict[str, str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _idx, row in orf_df.iterrows():
        tid = clean_text(row.get("mother_transcript_id", ""))
        model = models.get(tid)
        seq = transcript_seqs.get(tid, "")
        mapped_start, mapped_end = map_orf_to_transcript(row, model)
        final_start = choose_coordinate(mapped_start, row.get("orf_start_transcript_pos", np.nan))
        final_end = choose_coordinate(mapped_end, row.get("orf_end_transcript_pos", np.nan))
        existing_start = to_float(row.get("orf_start_transcript_pos", np.nan))
        existing_end = to_float(row.get("orf_end_transcript_pos", np.nan))

        cds_start_tx, cds_stop_tx = (None, None)
        if model is not None:
            cds_start_tx, cds_stop_tx = model.cds_start_stop_transcript_pos()

        start_pos = to_int(final_start)
        mapped_start_val = to_float(mapped_start)
        mapped_end_val = to_float(mapped_end)
        start_match = np.nan
        end_match = np.nan
        if not pd.isna(mapped_start_val) and not pd.isna(existing_start):
            start_match = int(round(mapped_start_val)) == int(round(existing_start))
        if not pd.isna(mapped_end_val) and not pd.isna(existing_end):
            end_match = int(round(mapped_end_val)) == int(round(existing_end))

        feature: Dict[str, object] = {
            "orf_start_transcript_pos_mapped_from_gtf": mapped_start,
            "orf_end_transcript_pos_mapped_from_gtf": mapped_end,
            "transcript_coordinate_source": "gtf_mapped"
            if not pd.isna(mapped_start_val)
            else "existing_table",
            "transcript_start_coordinate_matches_existing": start_match,
            "transcript_end_coordinate_matches_existing": end_match,
            "orf_start_transcript_pos_final": final_start,
            "orf_end_transcript_pos_final": final_end,
            "cds_start_transcript_pos": cds_start_tx if cds_start_tx is not None else np.nan,
            "cds_stop_transcript_pos": cds_stop_tx if cds_stop_tx is not None else np.nan,
            "distance_from_transcript_start": np.nan,
            "distance_from_CDS_start": np.nan,
            "distance_from_CDS_stop": np.nan,
            "start_codon_normalized": primary_codon(row.get("start_codon", "")),
            "start_codon_from_sequence_extended": "",
            "start_codon_match_fasta": np.nan,
            "is_ATG_start": np.nan,
            "is_near_cognate_start": np.nan,
            "start_codon_type": "missing",
            "Kozak_context_sequence": "",
            "Kozak_minus3_base": "",
            "Kozak_plus4_base": "",
            "simple_Kozak_score": np.nan,
            "Kozak_strength": "missing",
            "GC_content_around_start_21nt": np.nan,
            "GC_content_around_start_51nt": np.nan,
        }

        codon_for_flags = feature["start_codon_normalized"]
        if seq and start_pos is not None and 1 <= start_pos <= len(seq):
            seq_codon = seq[start_pos - 1 : start_pos + 2]
            feature["start_codon_from_sequence_extended"] = seq_codon if len(seq_codon) == 3 else ""
            if feature["start_codon_from_sequence_extended"]:
                codon_for_flags = feature["start_codon_from_sequence_extended"]
                feature["start_codon_match_fasta"] = (
                    normalize_codon(feature["start_codon_normalized"]) == codon_for_flags
                )

            kozak = subseq_with_padding(seq, start_pos - 6, start_pos + 3)
            minus3 = subseq_with_padding(seq, start_pos - 3, start_pos - 3)
            plus4 = subseq_with_padding(seq, start_pos + 3, start_pos + 3)
            score = int(minus3 in {"A", "G"}) + int(plus4 == "G")
            feature["Kozak_context_sequence"] = kozak
            feature["Kozak_minus3_base"] = minus3
            feature["Kozak_plus4_base"] = plus4
            feature["simple_Kozak_score"] = score
            feature["Kozak_strength"] = kozak_strength(score)
            feature["GC_content_around_start_21nt"] = gc_content(
                subseq_with_padding(seq, start_pos - 10, start_pos + 10).replace("N", "")
            )
            feature["GC_content_around_start_51nt"] = gc_content(
                subseq_with_padding(seq, start_pos - 25, start_pos + 25).replace("N", "")
            )

        feature["start_codon_type"] = start_type(codon_for_flags)
        feature["is_ATG_start"] = feature["start_codon_type"] == "ATG"
        feature["is_near_cognate_start"] = feature["start_codon_type"] == "near_cognate"

        if start_pos is not None:
            feature["distance_from_transcript_start"] = start_pos - 1
            if cds_start_tx is not None:
                feature["distance_from_CDS_start"] = start_pos - cds_start_tx
            if cds_stop_tx is not None:
                feature["distance_from_CDS_stop"] = start_pos - cds_stop_tx
        rows.append(feature)
    return pd.concat([orf_df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def build_transcript_groups(transcript_df: pd.DataFrame, orf_df: pd.DataFrame) -> pd.DataFrame:
    class_sets: Dict[str, set] = defaultdict(set)
    for tid, klass in zip(orf_df["mother_transcript_id"].astype(str), orf_df["orf_class"].astype(str)):
        class_sets[tid].add(klass)

    groups = []
    canonical_counts = []
    noncanonical_counts = []
    ambiguous_counts = []
    for tid in transcript_df["transcript_id"].astype(str):
        classes = class_sets.get(tid, set())
        tx_orfs = orf_df.loc[orf_df["mother_transcript_id"].astype(str) == tid, "orf_class"]
        counts = tx_orfs.value_counts()
        canonical_counts.append(int(counts.get("canonical", 0)))
        noncanonical_counts.append(int(counts.get("noncanonical", 0)))
        ambiguous_counts.append(int(counts.get("ambiguous", 0) + counts.get("unknown", 0)))
        if "noncanonical" in classes:
            groups.append("noncanonical_orf_transcript")
        elif classes and classes.issubset({"canonical"}):
            groups.append("canonical_only_orf_transcript")
        elif classes:
            groups.append("ambiguous_orf_transcript")
        else:
            groups.append("no_orf_call")
    out = transcript_df.copy()
    out["transcript_cryptic_group"] = groups
    out["canonical_orf_count"] = canonical_counts
    out["noncanonical_orf_count"] = noncanonical_counts
    out["ambiguous_orf_count"] = ambiguous_counts
    return out


def add_transcript_sequence_features(
    transcript_df: pd.DataFrame, transcript_seqs: Dict[str, str]
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _idx, row in transcript_df.iterrows():
        tid = clean_text(row.get("transcript_id", ""))
        seq = transcript_seqs.get(tid, "")
        utr5_len = to_int(row.get("utr5_length", ""))
        feature: Dict[str, object] = {
            "5UTR_AUG_count": np.nan,
            "5UTR_AUG_density_per_kb": np.nan,
            "5UTR_near_cognate_start_count": np.nan,
            "5UTR_near_cognate_start_density_per_kb": np.nan,
            "transcript_AUG_density_per_kb": np.nan,
            "transcript_near_cognate_start_density_per_kb": np.nan,
        }
        if seq:
            feature["transcript_AUG_density_per_kb"] = density_per_kb(
                count_motifs(seq, {"ATG"}), len(seq)
            )
            feature["transcript_near_cognate_start_density_per_kb"] = density_per_kb(
                count_motifs(seq, NEAR_COGNATE_STARTS), len(seq)
            )
            if utr5_len is not None and utr5_len >= 0:
                utr5_seq = seq[: min(utr5_len, len(seq))]
                feature["5UTR_AUG_count"] = count_motifs(utr5_seq, {"ATG"})
                feature["5UTR_near_cognate_start_count"] = count_motifs(
                    utr5_seq, NEAR_COGNATE_STARTS
                )
                feature["5UTR_AUG_density_per_kb"] = density_per_kb(
                    feature["5UTR_AUG_count"], len(utr5_seq)
                )
                feature["5UTR_near_cognate_start_density_per_kb"] = density_per_kb(
                    feature["5UTR_near_cognate_start_count"], len(utr5_seq)
                )
        rows.append(feature)
    return pd.concat([transcript_df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def attach_mother_transcript_features(orf_df: pd.DataFrame, transcript_df: pd.DataFrame) -> pd.DataFrame:
    attach_cols = [
        "transcript_id",
        "transcript_cryptic_group",
        "5UTR_AUG_count",
        "5UTR_AUG_density_per_kb",
        "5UTR_near_cognate_start_count",
        "5UTR_near_cognate_start_density_per_kb",
        "transcript_AUG_density_per_kb",
        "transcript_near_cognate_start_density_per_kb",
    ]
    context = transcript_df[[col for col in attach_cols if col in transcript_df.columns]].rename(
        columns={
            "transcript_id": "mother_transcript_id",
            "5UTR_AUG_count": "mother_5UTR_AUG_count",
            "5UTR_AUG_density_per_kb": "mother_5UTR_AUG_density_per_kb",
            "5UTR_near_cognate_start_count": "mother_5UTR_near_cognate_start_count",
            "5UTR_near_cognate_start_density_per_kb": "mother_5UTR_near_cognate_start_density_per_kb",
            "transcript_AUG_density_per_kb": "mother_transcript_AUG_density_per_kb",
            "transcript_near_cognate_start_density_per_kb": "mother_transcript_near_cognate_start_density_per_kb",
        }
    )
    return orf_df.merge(context, on="mother_transcript_id", how="left")


def iqr(values: pd.Series) -> float:
    values = values.dropna()
    if values.empty:
        return np.nan
    return values.quantile(0.75) - values.quantile(0.25)


def rank_biserial_from_u(u_stat: float, n_a: int, n_b: int) -> float:
    if n_a == 0 or n_b == 0:
        return np.nan
    return (2.0 * u_stat / (n_a * n_b)) - 1.0


def continuous_test(
    df: pd.DataFrame,
    feature: str,
    group_col: str,
    group_a: str,
    group_b: str,
    comparison: str,
    analysis_level: str,
    label: Optional[str] = None,
) -> Dict[str, object]:
    a = pd.to_numeric(df.loc[df[group_col] == group_a, feature], errors="coerce").dropna()
    b = pd.to_numeric(df.loc[df[group_col] == group_b, feature], errors="coerce").dropna()
    out: Dict[str, object] = {
        "analysis_level": analysis_level,
        "comparison": comparison,
        "feature": label or feature,
        "feature_column": feature,
        "feature_type": "continuous",
        "group_a": group_a,
        "group_b": group_b,
        "n_a": len(a),
        "n_b": len(b),
        "median_a": a.median() if len(a) else np.nan,
        "median_b": b.median() if len(b) else np.nan,
        "iqr_a": iqr(a),
        "iqr_b": iqr(b),
        "test": "Mann-Whitney U",
        "statistic": np.nan,
        "p_value": np.nan,
        "effect_size": np.nan,
        "effect_size_name": "rank_biserial",
        "note": "positive effect means group_a tends higher",
    }
    if len(a) > 0 and len(b) > 0:
        result = stats.mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
        out["statistic"] = result.statistic
        out["p_value"] = result.pvalue
        out["effect_size"] = rank_biserial_from_u(result.statistic, len(a), len(b))
    return out


def cramer_v(table: pd.DataFrame) -> float:
    chi2, _p, _dof, _expected = stats.chi2_contingency(table, correction=False)
    n = table.to_numpy().sum()
    denom = n * min(table.shape[0] - 1, table.shape[1] - 1)
    if denom <= 0:
        return np.nan
    return math.sqrt(chi2 / denom)


def summarize_categories(values: pd.Series, max_items: int = 8) -> str:
    counts = values.fillna("missing").replace("", "missing").value_counts().head(max_items)
    return "; ".join(f"{idx}:{count}" for idx, count in counts.items())


def categorical_test(
    df: pd.DataFrame,
    feature: str,
    group_col: str,
    group_a: str,
    group_b: str,
    comparison: str,
    analysis_level: str,
    label: Optional[str] = None,
) -> Dict[str, object]:
    sub = df.loc[df[group_col].isin([group_a, group_b]), [group_col, feature]].copy()
    sub[feature] = sub[feature].fillna("missing").replace("", "missing").astype(str)
    table = pd.crosstab(sub[group_col], sub[feature]).reindex([group_a, group_b]).fillna(0)
    table = table.loc[:, table.sum(axis=0) > 0]
    out: Dict[str, object] = {
        "analysis_level": analysis_level,
        "comparison": comparison,
        "feature": label or feature,
        "feature_column": feature,
        "feature_type": "categorical",
        "group_a": group_a,
        "group_b": group_b,
        "n_a": int((sub[group_col] == group_a).sum()),
        "n_b": int((sub[group_col] == group_b).sum()),
        "median_a": np.nan,
        "median_b": np.nan,
        "iqr_a": np.nan,
        "iqr_b": np.nan,
        "test": "chi-square",
        "statistic": np.nan,
        "p_value": np.nan,
        "effect_size": np.nan,
        "effect_size_name": "Cramer's V",
        "note": f"{group_a}: {summarize_categories(sub.loc[sub[group_col] == group_a, feature])} | "
        f"{group_b}: {summarize_categories(sub.loc[sub[group_col] == group_b, feature])}",
    }
    if table.shape[0] != 2 or table.shape[1] < 2:
        return out
    if table.shape == (2, 2):
        result = stats.fisher_exact(table.to_numpy())
        out["test"] = "Fisher exact"
        out["statistic"] = result.statistic
        out["p_value"] = result.pvalue
        out["effect_size"] = cramer_v(table)
        out["effect_size_name"] = "Cramer's V; statistic is odds_ratio"
    else:
        result = stats.chi2_contingency(table, correction=False)
        out["statistic"] = result.statistic
        out["p_value"] = result.pvalue
        out["effect_size"] = cramer_v(table)
    return out


def build_stats(orf_df: pd.DataFrame, transcript_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    orf_features = [
        ("simple_Kozak_score", "Simple Kozak score"),
        ("GC_content_around_start_21nt", "GC around ORF start 21nt"),
        ("GC_content_around_start_51nt", "GC around ORF start 51nt"),
        ("distance_from_transcript_start", "Distance from transcript start"),
        ("distance_from_CDS_start", "Distance from CDS start"),
        ("distance_from_CDS_stop", "Distance from CDS stop"),
        ("supporting_samples", "Supporting samples"),
        ("supporting_callers", "Supporting callers"),
    ]
    categorical = [
        ("start_codon_type", "Start codon type"),
        ("Kozak_strength", "Kozak strength"),
        ("is_ATG_start", "ATG start"),
        ("is_near_cognate_start", "Near-cognate start"),
    ]

    cn = orf_df.loc[orf_df["orf_class"].isin(["noncanonical", "canonical"])].copy()
    for col, label in orf_features:
        rows.append(
            continuous_test(
                cn, col, "orf_class", "noncanonical", "canonical",
                "noncanonical_orf_vs_canonical_orf", "orf", label
            )
        )
    for col, label in categorical:
        rows.append(
            categorical_test(
                cn, col, "orf_class", "noncanonical", "canonical",
                "noncanonical_orf_vs_canonical_orf", "orf", label
            )
        )

    noncanon = orf_df.loc[orf_df["orf_class"] == "noncanonical"].copy()
    noncanon["noncanonical_start_group"] = np.where(
        noncanon["start_codon_type"] == "ATG",
        "ATG_noncanonical",
        np.where(
            noncanon["start_codon_type"] == "near_cognate",
            "near_cognate_noncanonical",
            "other_noncanonical",
        ),
    )
    atg_nc = noncanon.loc[
        noncanon["noncanonical_start_group"].isin(
            ["ATG_noncanonical", "near_cognate_noncanonical"]
        )
    ].copy()
    for col, label in orf_features:
        rows.append(
            continuous_test(
                atg_nc, col, "noncanonical_start_group",
                "ATG_noncanonical", "near_cognate_noncanonical",
                "ATG_noncanonical_vs_near_cognate_noncanonical", "orf", label
            )
        )
    for col, label in [("Kozak_strength", "Kozak strength")]:
        rows.append(
            categorical_test(
                atg_nc, col, "noncanonical_start_group",
                "ATG_noncanonical", "near_cognate_noncanonical",
                "ATG_noncanonical_vs_near_cognate_noncanonical", "orf", label
            )
        )

    tx = transcript_df.loc[
        transcript_df["transcript_cryptic_group"].isin(
            ["noncanonical_orf_transcript", "canonical_only_orf_transcript"]
        )
    ].copy()
    tx_features = [
        ("5UTR_AUG_count", "5UTR AUG count"),
        ("5UTR_AUG_density_per_kb", "5UTR AUG density per kb"),
        ("5UTR_near_cognate_start_count", "5UTR near-cognate start count"),
        ("5UTR_near_cognate_start_density_per_kb", "5UTR near-cognate density per kb"),
        ("transcript_AUG_density_per_kb", "Transcript AUG density per kb"),
        ("transcript_near_cognate_start_density_per_kb", "Transcript near-cognate density per kb"),
        ("transcript_length", "Transcript length"),
        ("gc_content", "Transcript GC content"),
    ]
    for col, label in tx_features:
        rows.append(
            continuous_test(
                tx, col, "transcript_cryptic_group",
                "noncanonical_orf_transcript", "canonical_only_orf_transcript",
                "noncanonical_orf_transcript_vs_canonical_only_orf_transcript",
                "transcript", label
            )
        )
    return pd.DataFrame(rows)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 180,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=180)
    plt.close(fig)


def grouped_bar_proportions(
    df: pd.DataFrame,
    category_col: str,
    group_col: str,
    groups: Sequence[str],
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    sub = df.loc[df[group_col].isin(groups), [category_col, group_col]].copy()
    sub[category_col] = sub[category_col].fillna("missing").replace("", "missing").astype(str)
    counts = pd.crosstab(sub[category_col], sub[group_col]).reindex(columns=groups).fillna(0)
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]
    props = counts.div(counts.sum(axis=0), axis=1).fillna(0)

    x = np.arange(len(props))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(7, len(props) * 0.7 + 3), 4.8))
    for idx, group in enumerate(groups):
        ax.bar(x + (idx - 0.5) * width, props[group], width=width, label=f"{group} n={int(counts[group].sum()):,}")
    ax.set_xticks(x)
    ax.set_xticklabels(props.index, rotation=35, ha="right")
    ax.set_ylim(0, max(0.05, min(1.0, props.to_numpy().max() * 1.2)))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, path)


def boxplot_groups(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    groups: Sequence[str],
    title: str,
    ylabel: str,
    path: Path,
    log1p: bool = False,
    symlog: bool = False,
) -> None:
    data = []
    labels = []
    for group in groups:
        values = pd.to_numeric(df.loc[df[group_col] == group, value_col], errors="coerce").dropna()
        if log1p:
            values = np.log1p(values[values >= 0])
        data.append(values.to_numpy())
        labels.append(f"{group}\nn={len(values):,}")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    colors = ["#E76F51", "#2A9D8F", "#8D99AE", "#F4A261"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
    for median in bp["medians"]:
        median.set_color("#222222")
        median.set_linewidth(1.6)
    if symlog:
        ax.set_yscale("symlog", linthresh=100)
        ax.axhline(0, color="#555555", linewidth=0.8, alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, path)


def two_panel_boxplot(
    df: pd.DataFrame,
    value_cols: Sequence[Tuple[str, str]],
    group_col: str,
    groups: Sequence[str],
    title: str,
    path: Path,
    symlog: bool = False,
) -> None:
    fig, axes = plt.subplots(1, len(value_cols), figsize=(6.2 * len(value_cols), 4.8))
    if len(value_cols) == 1:
        axes = [axes]
    colors = ["#E76F51", "#2A9D8F", "#8D99AE"]
    for ax, (value_col, label) in zip(axes, value_cols):
        data = []
        tick_labels = []
        for group in groups:
            values = pd.to_numeric(df.loc[df[group_col] == group, value_col], errors="coerce").dropna()
            data.append(values.to_numpy())
            tick_labels.append(f"{group}\nn={len(values):,}")
        bp = ax.boxplot(data, tick_labels=tick_labels, showfliers=False, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.78)
        for median in bp["medians"]:
            median.set_color("#222222")
            median.set_linewidth(1.6)
        if symlog:
            ax.set_yscale("symlog", linthresh=100)
            ax.axhline(0, color="#555555", linewidth=0.8, alpha=0.7)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(title)
    save_figure(fig, path)


def number_text(value: object, digits: int = 3) -> str:
    value = to_float(value)
    if pd.isna(value):
        return "NA"
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    return f"{value:,.{digits}f}"


def pvalue_text(value: object) -> str:
    value = to_float(value)
    if pd.isna(value):
        return "NA"
    if value == 0:
        return "<1e-300"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def markdown_table(df: pd.DataFrame, columns: Sequence[str], max_rows: int = 16) -> str:
    if df.empty:
        return "_No rows._"
    shown = df[[col for col in columns if col in df.columns]].head(max_rows).copy()
    for col in shown.columns:
        if col == "p_value":
            shown[col] = shown[col].map(pvalue_text)
        elif shown[col].dtype.kind in "if":
            shown[col] = shown[col].map(lambda x: number_text(x, 3))
        else:
            shown[col] = shown[col].astype(str).str.replace("|", "\\|", regex=False)
    header = "| " + " | ".join(shown.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(shown.columns)) + " |"
    rows = ["| " + " | ".join(str(row[col]) for col in shown.columns) + " |" for _idx, row in shown.iterrows()]
    return "\n".join([header, sep] + rows)


def write_report(
    report_path: Path,
    stats_df: pd.DataFrame,
    extended_orf: pd.DataFrame,
    transcript_df: pd.DataFrame,
    figure_paths: Dict[str, Path],
    gtf_path: Path,
    fasta_path: Path,
) -> None:
    root = report_path.parent
    class_counts = extended_orf["orf_class"].value_counts()
    start_counts = extended_orf["start_codon_type"].value_counts()
    tx_counts = transcript_df["transcript_cryptic_group"].value_counts()
    start_mapped = extended_orf["orf_start_transcript_pos_mapped_from_gtf"].notna().sum()
    end_mapped = extended_orf["orf_end_transcript_pos_mapped_from_gtf"].notna().sum()
    start_checked = extended_orf["transcript_start_coordinate_matches_existing"].notna().sum()
    end_checked = extended_orf["transcript_end_coordinate_matches_existing"].notna().sum()
    start_match = extended_orf["transcript_start_coordinate_matches_existing"].eq(True).sum()
    end_match = extended_orf["transcript_end_coordinate_matches_existing"].eq(True).sum()
    fasta_start_match = extended_orf["start_codon_match_fasta"].eq(True).sum()
    fasta_start_checked = extended_orf["start_codon_match_fasta"].notna().sum()

    def rel(path: Path) -> str:
        return os.path.relpath(path, root)

    stat_cols = [
        "analysis_level",
        "comparison",
        "feature",
        "n_a",
        "median_a",
        "n_b",
        "median_b",
        "test",
        "p_value",
        "effect_size",
        "effect_size_name",
    ]
    lines = [
        "# Extended Cryptic ORF Feature Analysis",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Inputs",
        f"- Extended base ORF table: `{root / 'cryptic_orf_features.tsv'}`",
        f"- ORF feature table: `{root / 'orf_features.tsv'}`",
        f"- Transcript feature table: `{root / 'transcript_features.tsv'}`",
        f"- GTF: `{gtf_path}`",
        f"- Transcript FASTA: `{fasta_path}`",
        "",
        "## Coordinate Assumptions",
        "- `start_0based` and `end_0based` are treated as genomic 0-based half-open ORF intervals.",
        "- For plus-strand ORFs, genomic ORF start is `start_0based + 1`; for minus-strand ORFs, translation start is `end_0based` in 1-based genomic coordinates.",
        "- GTF exon structure is used to map genomic ORF start/end to 1-based transcript coordinates along transcript sequence order.",
        "- `orf_start_transcript_pos_final` prefers the GTF-mapped coordinate and falls back to the existing `orf_start_transcript_pos` column only if mapping is unavailable.",
        "- `distance_from_transcript_start` is `orf_start_transcript_pos_final - 1`.",
        "- `distance_from_CDS_start` and `distance_from_CDS_stop` are signed distances from ORF start to the first and last annotated CDS base in transcript orientation; negative values are upstream of that CDS landmark.",
        "- CDS positions are based on GTF `CDS` features, not separate stop-codon features.",
        "",
        "## Verification",
        f"- ORF starts mapped from GTF exon structure: {int(start_mapped):,} / {len(extended_orf):,}",
        f"- ORF ends mapped from GTF exon structure: {int(end_mapped):,} / {len(extended_orf):,}",
        f"- Mapped start coordinate matched existing table: {int(start_match):,} / {int(start_checked):,}",
        f"- Mapped end coordinate matched existing table: {int(end_match):,} / {int(end_checked):,}",
        f"- FASTA-derived start codon matched reported normalized start codon: {int(fasta_start_match):,} / {int(fasta_start_checked):,}",
        "",
        "## ORF Classes",
    ]
    for label, count in class_counts.items():
        lines.append(f"- {label}: {int(count):,}")
    lines.append("")
    lines.append("Start codon type:")
    for label, count in start_counts.items():
        lines.append(f"- {label}: {int(count):,}")
    lines.append("")
    lines.append("Transcript cryptic ORF status:")
    for label, count in tx_counts.items():
        lines.append(f"- {label}: {int(count):,}")

    lines.extend(
        [
            "",
            "## Statistical Comparisons",
            "Mann-Whitney U tests are used for continuous features; Fisher exact or chi-square tests are used for categorical features. Rank-biserial effects above zero mean group A tends to have larger values.",
            "",
            markdown_table(stats_df, stat_cols, max_rows=32),
            "",
            "## Figures",
        ]
    )
    for title, key in [
        ("Kozak score distribution by ORF class", "kozak_score"),
        ("GC around start by ORF class", "gc_start"),
        ("Distance from transcript start by ORF class", "distance_tx_start"),
        ("Distance from CDS start/stop by ORF class", "distance_cds"),
        ("5UTR AUG density by transcript status", "utr5_aug_density"),
        ("5UTR near-cognate density by transcript status", "utr5_near_density"),
        ("Start codon type vs sample recurrence", "start_type_samples"),
        ("Start codon type vs caller agreement", "start_type_callers"),
    ]:
        lines.extend([f"### {title}", f"![{title}]({rel(figure_paths[key])})", ""])

    lines.extend(
        [
            "## Feature Definitions",
            "- Near-cognate start codons are CTG, GTG, TTG, ACG, ATT, ATC, ATA, AGG, and AAG.",
            "- `Kozak_context_sequence` spans positions -6 to +4 relative to the first base of the ORF start codon.",
            "- `simple_Kozak_score` is +1 for A/G at -3 and +1 for G at +4.",
            "- `Kozak_strength` is weak for score 0, moderate for score 1, and strong for score 2.",
            "- GC windows are centered on the first base of the ORF start codon: -10..+10 for 21 nt and -25..+25 for 51 nt.",
            "- 5UTR motif counts scan all overlapping 3-mers in the first `utr5_length` transcript bases.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    analysis_dir = args.analysis_dir
    figure_dir = args.figure_dir or analysis_dir / "figures"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    set_style()

    cryptic_path = args.cryptic_orf_features or analysis_dir / "cryptic_orf_features.tsv"
    orf_path = args.orf_features or analysis_dir / "orf_features.tsv"
    transcript_path = args.transcript_features or analysis_dir / "transcript_features.tsv"
    gtf_path = args.gtf_file
    fasta_path = args.transcript_fasta

    print(f"Reading cryptic ORF features: {cryptic_path}")
    cryptic_orf = pd.read_csv(cryptic_path, sep="\t", low_memory=False)
    print(f"Reading ORF features for verification: {orf_path}")
    _orf_features = pd.read_csv(orf_path, sep="\t", low_memory=False)
    print(f"Reading transcript features: {transcript_path}")
    transcript_df = pd.read_csv(transcript_path, sep="\t", low_memory=False)

    numeric_orf = [
        "start_0based", "end_0based", "orf_start_transcript_pos", "orf_end_transcript_pos",
        "supporting_samples", "supporting_callers", "orf_length", "transcript_length",
        "utr5_length", "utr3_length", "exon_count", "gc_content", "num_aug_codons",
        "transcript_sequence_length",
    ]
    numeric_tx = [
        "transcript_length", "utr5_length", "utr3_length", "exon_count", "gc_content",
        "num_aug_codons", "transcript_sequence_length", "unique_orf_count",
    ]
    cryptic_orf = numeric_columns(cryptic_orf, numeric_orf)
    transcript_df = numeric_columns(transcript_df, numeric_tx)

    if "orf_class" not in cryptic_orf.columns:
        raise ValueError("cryptic_orf_features.tsv must contain `orf_class`.")
    if "transcript_id" not in transcript_df.columns:
        raise ValueError("transcript_features.tsv must contain `transcript_id`.")

    target_transcripts = sorted(set(cryptic_orf["mother_transcript_id"].dropna().astype(str)))
    print(f"Parsing GTF exon/CDS models for {len(target_transcripts):,} mother transcripts")
    models = parse_gtf_models(gtf_path, target_transcripts)
    print(f"Transcript models found in GTF: {len(models):,}")
    print(f"Parsing transcript FASTA: {fasta_path}")
    transcript_seqs = parse_target_fasta(fasta_path, target_transcripts)
    print(f"Transcript sequences found: {len(transcript_seqs):,}")

    extended_orf = add_mapped_and_sequence_features(cryptic_orf, models, transcript_seqs)
    transcript_ext = build_transcript_groups(transcript_df, extended_orf)
    transcript_ext = add_transcript_sequence_features(transcript_ext, transcript_seqs)
    extended_orf = attach_mother_transcript_features(extended_orf, transcript_ext)
    stats_df = build_stats(extended_orf, transcript_ext)

    extended_path = analysis_dir / "cryptic_orf_features.extended.tsv"
    stats_path = analysis_dir / "cryptic_orf_extended_stats.tsv"
    report_path = analysis_dir / "cryptic_orf_extended_report.md"
    extended_orf.to_csv(extended_path, sep="\t", index=False, na_rep="")
    stats_df.to_csv(stats_path, sep="\t", index=False, na_rep="")

    cn = extended_orf.loc[extended_orf["orf_class"].isin(["canonical", "noncanonical"])].copy()
    tx = transcript_ext.loc[
        transcript_ext["transcript_cryptic_group"].isin(
            ["noncanonical_orf_transcript", "canonical_only_orf_transcript"]
        )
    ].copy()
    groups_orf = ["canonical", "noncanonical"]
    groups_tx = ["canonical_only_orf_transcript", "noncanonical_orf_transcript"]
    figure_paths = {
        "kozak_score": figure_dir / "extended_kozak_score_by_orf_class.png",
        "gc_start": figure_dir / "extended_gc_around_start_by_orf_class.png",
        "distance_tx_start": figure_dir / "extended_distance_from_transcript_start_by_orf_class.png",
        "distance_cds": figure_dir / "extended_distance_from_cds_landmarks_by_orf_class.png",
        "utr5_aug_density": figure_dir / "extended_5utr_aug_density_by_transcript_status.png",
        "utr5_near_density": figure_dir / "extended_5utr_near_cognate_density_by_transcript_status.png",
        "start_type_samples": figure_dir / "extended_start_codon_type_vs_sample_recurrence.png",
        "start_type_callers": figure_dir / "extended_start_codon_type_vs_caller_agreement.png",
    }
    grouped_bar_proportions(
        cn, "simple_Kozak_score", "orf_class", groups_orf,
        "Kozak score distribution by ORF class", "Proportion within ORF class",
        figure_paths["kozak_score"]
    )
    two_panel_boxplot(
        cn,
        [("GC_content_around_start_21nt", "21 nt window"), ("GC_content_around_start_51nt", "51 nt window")],
        "orf_class", groups_orf, "GC content around ORF start by ORF class", figure_paths["gc_start"]
    )
    boxplot_groups(
        cn, "distance_from_transcript_start", "orf_class", groups_orf,
        "Distance from transcript start by ORF class", "log1p(distance from transcript start)",
        figure_paths["distance_tx_start"], log1p=True
    )
    two_panel_boxplot(
        cn,
        [("distance_from_CDS_start", "Signed distance from CDS start"), ("distance_from_CDS_stop", "Signed distance from CDS stop")],
        "orf_class", groups_orf, "Signed distance from CDS landmarks by ORF class",
        figure_paths["distance_cds"], symlog=True
    )
    boxplot_groups(
        tx, "5UTR_AUG_density_per_kb", "transcript_cryptic_group", groups_tx,
        "5UTR AUG density by transcript cryptic ORF status", "5UTR AUGs per kb",
        figure_paths["utr5_aug_density"]
    )
    boxplot_groups(
        tx, "5UTR_near_cognate_start_density_per_kb", "transcript_cryptic_group", groups_tx,
        "5UTR near-cognate start density by transcript cryptic ORF status",
        "5UTR near-cognate 3-mers per kb", figure_paths["utr5_near_density"]
    )
    boxplot_groups(
        cn.loc[cn["start_codon_type"].isin(["ATG", "near_cognate", "other"])],
        "supporting_samples", "start_codon_type", ["ATG", "near_cognate", "other"],
        "Start codon type vs sample recurrence", "Supporting samples",
        figure_paths["start_type_samples"]
    )
    boxplot_groups(
        cn.loc[cn["start_codon_type"].isin(["ATG", "near_cognate", "other"])],
        "supporting_callers", "start_codon_type", ["ATG", "near_cognate", "other"],
        "Start codon type vs caller agreement", "Supporting callers",
        figure_paths["start_type_callers"]
    )

    write_report(report_path, stats_df, extended_orf, transcript_ext, figure_paths, gtf_path, fasta_path)
    print(f"Wrote {extended_path}")
    print(f"Wrote {stats_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote figures to {figure_dir}")


if __name__ == "__main__":
    main()
