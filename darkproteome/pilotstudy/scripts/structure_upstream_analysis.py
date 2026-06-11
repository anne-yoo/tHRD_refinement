#!/usr/bin/env python3
"""Compute upstream/start RNA structure features for pilot ORFs."""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO

from tc_common import (
    GENOME_FA,
    INPUT_DIR,
    PAIRWISE_GROUPS,
    TABLE_DIR,
    archive_script,
    boxplot_with_points,
    ensure_fig_dirs,
    mannwhitney_stats,
    print_paths,
    resolve_chrom_key,
    reverse_complement,
    safe_subseq,
    save_figure,
    read_tsv,
)


GROUP_IN = INPUT_DIR / "tables" / "orf_groups.combined_metadata.tsv"
FEATURE_OUT = TABLE_DIR / "orf_upstream_structure_features.tsv"
STATS_OUT = TABLE_DIR / "upstream_structure_statistics.tsv"
WARNING_OUT = TABLE_DIR / "upstream_structure_warnings.txt"

FIGURE_NAME = "Fig_structure_upstream_MFE.pdf"

DEFAULT_RNAFOLD = Path("/home/jiye/jiye/darkproteome/tools/ViennaRNA-2.7.2/src/bin/RNAfold")
DEFAULT_RNAPLFOLD = Path("/home/jiye/jiye/darkproteome/tools/ViennaRNA-2.7.2/src/bin/RNAplfold")

WINDOWS = {
    "upstream_50": ("upstream", 50),
    "upstream_100": ("upstream", 100),
    "start_pm50": ("start_pm", 50),
    "start_pm100": ("start_pm", 100),
}

OUTPUT_COLUMNS = [
    "ORF_id",
    "group",
    "primary_noncanonical_category",
    "upstream_50_MFE",
    "upstream_50_MFE_norm",
    "upstream_100_MFE",
    "upstream_100_MFE_norm",
    "start_pm50_MFE",
    "start_pm50_MFE_norm",
    "start_pm100_MFE",
    "start_pm100_MFE_norm",
    "accessibility_minus3_plus4",
    "accessibility_start_pm10",
]

MFE_RE = re.compile(r"\(\s*(-?\d+(?:\.\d+)?)\s*\)")


def resolve_executable(env_name: str, default_path: Path, command_name: str) -> tuple[str | None, list[str]]:
    """Resolve a requested executable path, then PATH fallback."""
    warnings: list[str] = []
    requested = os.environ.get(env_name)
    candidates: list[tuple[str, Path]] = []
    if requested:
        candidates.append((env_name, Path(requested)))
    candidates.append(("default", default_path))

    for label, candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate), warnings
        if label != "default" or candidate.exists():
            warnings.append(f"{command_name} candidate is not executable ({label}): {candidate}")

    found = shutil.which(command_name)
    if found:
        return found, warnings

    warnings.append(
        f"{command_name} unavailable: tried {env_name or 'env'} override, {default_path}, and PATH."
    )
    return None, warnings


def read_group_table() -> pd.DataFrame:
    required = [
        "ORF_id",
        "group",
        "primary_noncanonical_category",
        "chr",
        "start0",
        "end0",
        "strand",
    ]
    df = read_tsv(GROUP_IN, required=required).reset_index(drop=True)
    df["start0"] = pd.to_numeric(df["start0"], errors="coerce")
    df["end0"] = pd.to_numeric(df["end0"], errors="coerce")
    return df


def valid_coordinate_pair(start0: object, end0: object) -> bool:
    return pd.notna(start0) and pd.notna(end0) and int(start0) >= 0 and int(end0) > int(start0)


def clean_rna_sequence(seq: str) -> str:
    rna = seq.upper().replace("T", "U")
    return re.sub(r"[^ACGUN]", "N", rna)


def extract_upstream(chrom_seq: str, start0: int, end0: int, strand: str, length: int) -> str | None:
    """Return transcript-oriented upstream sequence, excluding the start codon."""
    if strand == "+":
        return safe_subseq(chrom_seq, start0 - length, start0)
    if strand == "-":
        seq = safe_subseq(chrom_seq, end0, end0 + length)
        return reverse_complement(seq)
    return None


def extract_start_pm(chrom_seq: str, start0: int, end0: int, strand: str, flank: int) -> str | None:
    """Return transcript-oriented positions -flank..-1 and +1..+flank.

    The first base of the start codon is treated as +1, matching the pilot
    coordinate convention where plus-strand +4 is start0 + 3. There is no
    position 0 in the biological coordinate labels.
    """
    if strand == "+":
        return safe_subseq(chrom_seq, start0 - flank, start0 + flank)
    if strand == "-":
        seq = safe_subseq(chrom_seq, end0 - flank, end0 + flank)
        return reverse_complement(seq)
    return None


def extract_windows(chrom_seq: str, start0: int, end0: int, strand: str) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name, (window_type, size) in WINDOWS.items():
        if window_type == "upstream":
            out[name] = extract_upstream(chrom_seq, start0, end0, strand, size)
        else:
            out[name] = extract_start_pm(chrom_seq, start0, end0, strand, size)
    return out


def iter_chunks(items: Sequence[tuple[str, str]], size: int) -> Iterable[list[tuple[str, str]]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def parse_rnafold_stdout(stdout: str) -> dict[str, float]:
    results: dict[str, float] = {}
    current_id: str | None = None
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            current_id = line[1:].split()[0]
            continue
        if current_id is None:
            continue
        matches = MFE_RE.findall(line)
        if matches:
            results[current_id] = float(matches[-1])
            current_id = None
    return results


def run_rnafold_batch(
    records: Sequence[tuple[str, str | None]],
    rnafold_bin: str | None,
    warnings: list[str],
) -> dict[str, float]:
    results = {record_id: np.nan for record_id, _ in records}
    if rnafold_bin is None:
        return results

    valid_records = [(record_id, clean_rna_sequence(seq)) for record_id, seq in records if seq]
    if not valid_records:
        return results

    batch_size = max(1, int(os.environ.get("RNAFOLD_BATCH_SIZE", "2000")))
    for batch_number, batch in enumerate(iter_chunks(valid_records, batch_size), start=1):
        fasta_input = "".join(f">{record_id}\n{seq}\n" for record_id, seq in batch)
        proc = subprocess.run(
            [rnafold_bin, "--noPS"],
            input=fasta_input,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            warnings.append(
                f"RNAfold failed for batch {batch_number} with exit code {proc.returncode}: "
                f"{proc.stderr.strip()[:500]}"
            )
            continue
        parsed = parse_rnafold_stdout(proc.stdout)
        missing = [record_id for record_id, _ in batch if record_id not in parsed]
        if missing:
            warnings.append(
                f"RNAfold output missing {len(missing)} record(s) in batch {batch_number}; "
                f"first missing={missing[0]}"
            )
        for record_id, value in parsed.items():
            results[record_id] = value
    return results


def parse_lunp_file(path: Path) -> dict[int, float]:
    """Parse RNAplfold *_lunp output as 0-based position -> single-base unpaired probability."""
    probabilities: dict[int, float] = {}
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                pos0 = int(float(parts[0])) - 1
            except (IndexError, ValueError):
                continue
            value = np.nan
            for token in parts[1:]:
                if token.upper() == "NA":
                    continue
                try:
                    value = float(token)
                    break
                except ValueError:
                    continue
            if np.isfinite(value):
                probabilities[pos0] = value
    return probabilities


def infer_lunp_record_id(path: Path, valid_ids: set[str]) -> str | None:
    name = path.name
    candidates = []
    if name.endswith("_lunp"):
        candidates.append(name[: -len("_lunp")])
    if ".lunp" in name:
        candidates.append(name.split(".lunp", 1)[0])
    candidates.append(path.stem.replace("_lunp", ""))
    for candidate in candidates:
        if candidate in valid_ids:
            return candidate
    return None


def mean_unpaired(probabilities: dict[int, float], indices: Iterable[int], seq_len: int) -> float:
    values = [
        probabilities[idx]
        for idx in indices
        if 0 <= idx < seq_len and idx in probabilities and np.isfinite(probabilities[idx])
    ]
    return float(np.mean(values)) if values else np.nan


def accessibility_from_lunp(probabilities: dict[int, float], seq_len: int, flank: int = 100) -> tuple[float, float]:
    start_index = flank
    minus3_plus4 = range(start_index - 3, start_index + 4)
    start_pm10 = range(start_index - 10, start_index + 10)
    return (
        mean_unpaired(probabilities, minus3_plus4, seq_len),
        mean_unpaired(probabilities, start_pm10, seq_len),
    )


def run_rnaplfold_batch(
    records: Sequence[tuple[str, str | None]],
    rnaplfold_bin: str | None,
    warnings: list[str],
) -> dict[str, tuple[float, float]]:
    results = {record_id: (np.nan, np.nan) for record_id, _ in records}
    if rnaplfold_bin is None:
        return results

    valid_records = [(record_id, clean_rna_sequence(seq)) for record_id, seq in records if seq]
    if not valid_records:
        return results

    batch_size = max(1, int(os.environ.get("RNAPLFOLD_BATCH_SIZE", "100")))
    window_size = os.environ.get("RNAPLFOLD_W", "80")
    span = os.environ.get("RNAPLFOLD_L", "40")
    max_unpaired = os.environ.get("RNAPLFOLD_U", "20")
    command = [rnaplfold_bin, "--noPS", "-W", window_size, "-L", span, "-u", max_unpaired]

    for batch_number, batch in enumerate(iter_chunks(valid_records, batch_size), start=1):
        fasta_input = "".join(f">{record_id}\n{seq}\n" for record_id, seq in batch)
        valid_ids = {record_id for record_id, _ in batch}
        with TemporaryDirectory(prefix="rnaplfold_") as tmp:
            seq_lookup = dict(batch)
            proc = subprocess.run(
                command,
                input=fasta_input,
                cwd=tmp,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if proc.returncode != 0:
                warnings.append(
                    f"RNAplfold failed for batch {batch_number} with exit code {proc.returncode}: "
                    f"{proc.stderr.strip()[:500]}"
                )
                continue
            lunp_files = sorted(Path(tmp).glob("*lunp*"))
            if not lunp_files:
                warnings.append(f"RNAplfold produced no lunp output for batch {batch_number}.")
                continue
            parsed_ids = set()
            for lunp_path in lunp_files:
                record_id = infer_lunp_record_id(lunp_path, valid_ids)
                if record_id is None and len(batch) == 1:
                    record_id = batch[0][0]
                if record_id is None:
                    continue
                probabilities = parse_lunp_file(lunp_path)
                seq_len = len(seq_lookup[record_id])
                results[record_id] = accessibility_from_lunp(probabilities, seq_len)
                parsed_ids.add(record_id)
            missing = valid_ids - parsed_ids
            if missing:
                warnings.append(
                    f"RNAplfold output missing {len(missing)} record(s) in batch {batch_number}; "
                    f"first missing={sorted(missing)[0]}"
                )
    return results


def plot_feature_or_empty(ax, df: pd.DataFrame, feature: str, y_label: str, title: str) -> pd.DataFrame:
    numeric = pd.to_numeric(df[feature], errors="coerce")
    if numeric.notna().any():
        return boxplot_with_points(ax, df, feature, y_label=y_label, title=title)
    ax.axis("off")
    ax.text(0.5, 0.5, f"{title}\nnot available", ha="center", va="center")
    stats_df = mannwhitney_stats(df.assign(**{feature: numeric}), feature, pairs=PAIRWISE_GROUPS)
    return stats_df


def main() -> int:
    ensure_fig_dirs()
    archive_script(__file__)
    print_paths(
        script_name=Path(__file__).name,
        inputs=[GROUP_IN, GENOME_FA],
        outputs=[FEATURE_OUT, STATS_OUT, WARNING_OUT],
    )

    warnings: list[str] = []
    rnafold_bin, tool_warnings = resolve_executable("RNAFOLD_BIN", DEFAULT_RNAFOLD, "RNAfold")
    warnings.extend(tool_warnings)
    rnaplfold_bin, tool_warnings = resolve_executable("RNAPLFOLD_BIN", DEFAULT_RNAPLFOLD, "RNAplfold")
    warnings.extend(tool_warnings)

    print(f"RNAfold={rnafold_bin if rnafold_bin else 'unavailable'}")
    print(f"RNAplfold={rnaplfold_bin if rnaplfold_bin else 'unavailable'}")

    groups = read_group_table()
    if not GENOME_FA.exists():
        raise SystemExit(f"Genome FASTA not found: {GENOME_FA}")

    rows = []
    rnafold_records: list[tuple[str, str | None]] = []
    record_to_row_window: dict[str, tuple[int, str]] = {}
    record_to_seq: dict[str, str | None] = {}
    rnaplfold_records: list[tuple[str, str | None]] = []
    record_to_row: dict[str, int] = {}

    fasta = SeqIO.index(str(GENOME_FA), "fasta")
    try:
        chrom_keys = {chrom: resolve_chrom_key(fasta, chrom) for chrom in groups["chr"].dropna().astype(str).unique()}
        for chrom, sub in groups.groupby("chr", sort=False, dropna=False):
            chrom_text = str(chrom)
            chrom_key = chrom_keys.get(chrom_text)
            if chrom_key is None:
                warnings.append(f"Chromosome not found in FASTA: {chrom_text}")
                chrom_seq = None
            else:
                print(f"Loading {chrom_text} from FASTA as {chrom_key} for {len(sub)} ORF(s)")
                chrom_seq = str(fasta[chrom_key].seq)

            for _, row in sub.iterrows():
                out_row = {
                    "ORF_id": row["ORF_id"],
                    "group": row["group"],
                    "primary_noncanonical_category": row["primary_noncanonical_category"],
                    "upstream_50_MFE": np.nan,
                    "upstream_50_MFE_norm": np.nan,
                    "upstream_100_MFE": np.nan,
                    "upstream_100_MFE_norm": np.nan,
                    "start_pm50_MFE": np.nan,
                    "start_pm50_MFE_norm": np.nan,
                    "start_pm100_MFE": np.nan,
                    "start_pm100_MFE_norm": np.nan,
                    "accessibility_minus3_plus4": np.nan,
                    "accessibility_start_pm10": np.nan,
                }
                row_idx = len(rows)
                rows.append(out_row)

                if chrom_seq is None or not valid_coordinate_pair(row["start0"], row["end0"]):
                    warnings.append(f"Skipping sequence extraction for invalid ORF coordinates: {row['ORF_id']}")
                    continue

                start0 = int(row["start0"])
                end0 = int(row["end0"])
                strand = str(row["strand"])
                sequences = extract_windows(chrom_seq, start0, end0, strand)
                for window_name, seq in sequences.items():
                    record_id = f"r{row_idx}_{window_name}"
                    rnafold_records.append((record_id, seq))
                    record_to_seq[record_id] = seq
                    record_to_row_window[record_id] = (row_idx, window_name)
                    if seq is not None:
                        rows[row_idx][f"{window_name}_MFE_norm"] = math.nan
                start_seq = sequences.get("start_pm100")
                plfold_record_id = f"r{row_idx}"
                rnaplfold_records.append((plfold_record_id, start_seq))
                record_to_row[plfold_record_id] = row_idx
    finally:
        fasta.close()

    print(f"Running RNAfold on {sum(1 for _, seq in rnafold_records if seq)} sequence window(s)")
    mfe_by_record = run_rnafold_batch(rnafold_records, rnafold_bin, warnings)
    for record_id, mfe in mfe_by_record.items():
        row_idx, window_name = record_to_row_window[record_id]
        seq = record_to_seq.get(record_id)
        rows[row_idx][f"{window_name}_MFE"] = mfe
        if seq and pd.notna(mfe) and np.isfinite(float(mfe)):
            rows[row_idx][f"{window_name}_MFE_norm"] = float(mfe) / len(seq)

    print(f"Running RNAplfold on {sum(1 for _, seq in rnaplfold_records if seq)} start_pm100 sequence(s)")
    accessibility_by_record = run_rnaplfold_batch(rnaplfold_records, rnaplfold_bin, warnings)
    for record_id, (minus3_plus4, start_pm10) in accessibility_by_record.items():
        row_idx = record_to_row[record_id]
        rows[row_idx]["accessibility_minus3_plus4"] = minus3_plus4
        rows[row_idx]["accessibility_start_pm10"] = start_pm10

    out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out_df.to_csv(FEATURE_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {FEATURE_OUT}")

    plot_features = [
        ("upstream_50_MFE_norm", "MFE / nt", "Upstream 50 nt normalized MFE"),
        ("upstream_100_MFE_norm", "MFE / nt", "Upstream 100 nt normalized MFE"),
        ("start_pm50_MFE_norm", "MFE / nt", "Start pm50 normalized MFE"),
    ]
    if pd.to_numeric(out_df["accessibility_start_pm10"], errors="coerce").notna().any():
        plot_features.append(
            ("accessibility_start_pm10", "Mean unpaired probability", "Start pm10 accessibility")
        )

    ncols = 2
    nrows = math.ceil(len(plot_features) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.4 * nrows), squeeze=False)
    stats_tables = []
    for ax, (feature, ylabel, title) in zip(axes.flat, plot_features):
        stats_tables.append(plot_feature_or_empty(ax, out_df, feature, ylabel, title))
    for ax in axes.flat[len(plot_features) :]:
        ax.axis("off")
    stats_df = pd.concat(stats_tables, ignore_index=True) if stats_tables else pd.DataFrame()
    stats_df.to_csv(STATS_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {STATS_OUT}")
    save_figure(fig, FIGURE_NAME)

    if not warnings:
        warnings.append("No warnings.")
    WARNING_OUT.write_text("\n".join(warnings) + "\n")
    print(f"Wrote {WARNING_OUT}")

    print("structure_upstream_analysis.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
