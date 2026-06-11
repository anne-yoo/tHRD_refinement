#!/usr/bin/env python3
"""Compute local RNA structure and accessibility features for four ORF groups."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO

from fc_common import (
    GENOME_FA,
    LOG_DIR,
    RNAFOLD,
    RNAPLFOLD,
    START_POSITION_TO_INDEX,
    TABLE_DIR,
    archive_script,
    boxplot_four_group,
    ensure_dirs,
    mean_unpaired,
    position_indices,
    print_paths,
    read_master,
    resolve_chrom_key,
    resolve_executable,
    reverse_complement,
    run_rnafold,
    run_rnaplfold,
    safe_subseq,
    save_figure,
)


FEATURE_OUT = TABLE_DIR / "four_group_local_RNA_structure_features.tsv"
STATS_OUT = TABLE_DIR / "local_RNA_structure_four_group_statistics.tsv"
LOG_OUT = LOG_DIR / "local_RNA_structure_tools.log"

MFE_FEATURES = [
    ("upstream_50_MFE_norm", "MFE / nt", "Upstream 50 MFE norm"),
    ("upstream_100_MFE_norm", "MFE / nt", "Upstream 100 MFE norm"),
    ("downstream_50_MFE_norm", "MFE / nt", "Downstream 50 MFE norm"),
    ("start_pm50_MFE_norm", "MFE / nt", "Start pm50 MFE norm"),
    ("start_pm100_MFE_norm", "MFE / nt", "Start pm100 MFE norm"),
]
ACCESS_FEATURES = [
    ("accessibility_minus3_plus4", "Mean unpaired prob.", "-3 to +4 accessibility"),
    ("accessibility_start_codon", "Mean unpaired prob.", "Start codon accessibility"),
    ("accessibility_start_pm10", "Mean unpaired prob.", "Start pm10 accessibility"),
    ("accessibility_start_pm20", "Mean unpaired prob.", "Start pm20 accessibility"),
    ("accessibility_upstream_50", "Mean unpaired prob.", "Upstream 50 accessibility"),
    ("accessibility_downstream_50", "Mean unpaired prob.", "Downstream 50 accessibility"),
    ("accessibility_asymmetry", "Downstream / upstream", "Accessibility asymmetry"),
]
SCAN_FEATURES = [
    ("accessibility_scan_max", "Max unpaired prob.", "Scan max"),
    ("accessibility_scan_peak_position", "Position", "Scan peak position"),
    ("accessibility_scan_peak_width", "nt", "Peak width >=80% max"),
    ("accessibility_scan_width_ge_0.5", "nt", "Width accessibility >=0.5"),
    ("accessibility_scan_auc", "AUC", "Scan AUC"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final four-group local RNA structure analysis.")
    parser.add_argument("--limit", type=int, default=None, help="Analyze only first N ORFs for testing.")
    parser.add_argument("--threads", type=int, default=1, help="RNAplfold worker threads.")
    return parser.parse_args()


def extract_windows(chrom_seq: str, start0: int, end0: int, strand: str) -> Dict[str, Optional[str]]:
    if strand == "+":
        return {
            "upstream_50": safe_subseq(chrom_seq, start0 - 50, start0),
            "upstream_100": safe_subseq(chrom_seq, start0 - 100, start0),
            "downstream_50": safe_subseq(chrom_seq, start0, start0 + 50),
            "start_pm50": safe_subseq(chrom_seq, start0 - 50, start0 + 51),
            "start_pm100": safe_subseq(chrom_seq, start0 - 100, start0 + 101),
        }
    if strand == "-":
        return {
            "upstream_50": reverse_complement(safe_subseq(chrom_seq, end0, end0 + 50)),
            "upstream_100": reverse_complement(safe_subseq(chrom_seq, end0, end0 + 100)),
            "downstream_50": reverse_complement(safe_subseq(chrom_seq, end0 - 50, end0)),
            "start_pm50": reverse_complement(safe_subseq(chrom_seq, end0 - 51, end0 + 50)),
            "start_pm100": reverse_complement(safe_subseq(chrom_seq, end0 - 101, end0 + 100)),
        }
    return {name: None for name in ["upstream_50", "upstream_100", "downstream_50", "start_pm50", "start_pm100"]}


def contiguous_width(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def scan_features(probabilities: Optional[Dict[int, Dict[int, float]]]) -> Dict[str, float]:
    scan_positions = list(range(-30, 0)) + list(range(1, 11))
    vals = []
    for pos in scan_positions:
        idx = START_POSITION_TO_INDEX.get(pos)
        val = probabilities.get(idx, {}).get(1, np.nan) if probabilities is not None and idx is not None else np.nan
        vals.append(val)
    arr = np.asarray(vals, dtype=float)
    if not np.isfinite(arr).any():
        return {
            "accessibility_scan_max": np.nan,
            "accessibility_scan_peak_position": np.nan,
            "accessibility_scan_peak_width": np.nan,
            "accessibility_scan_width_ge_0.5": np.nan,
            "accessibility_scan_auc": np.nan,
        }
    max_val = float(np.nanmax(arr))
    peak_idx = int(np.nanargmax(arr))
    threshold = 0.8 * max_val
    return {
        "accessibility_scan_max": max_val,
        "accessibility_scan_peak_position": scan_positions[peak_idx],
        "accessibility_scan_peak_width": contiguous_width(np.isfinite(arr) & (arr >= threshold)),
        "accessibility_scan_width_ge_0.5": contiguous_width(np.isfinite(arr) & (arr >= 0.5)),
        "accessibility_scan_auc": float(np.nansum(arr)),
    }


def structure_features_for_row(row: pd.Series, chrom_seq: Optional[str], rnafold_bin: Optional[str], rnaplfold_bin: Optional[str]) -> Dict[str, object]:
    out: Dict[str, object] = {
        "ORF_id": row["ORF_id"],
        "plot_group": row["plot_group"],
        "chr": row["chr"],
        "start0": row["start0"],
        "end0": row["end0"],
        "strand": row["strand"],
    }
    for feature, _, _ in MFE_FEATURES + ACCESS_FEATURES + SCAN_FEATURES:
        out[feature] = np.nan
    if chrom_seq is None:
        out["structure_status"] = "missing_chromosome"
        return out
    try:
        start0 = int(float(row["start0"]))
        end0 = int(float(row["end0"]))
    except (TypeError, ValueError):
        out["structure_status"] = "invalid_coordinates"
        return out
    windows = extract_windows(chrom_seq, start0, end0, str(row["strand"]))
    for name in ["upstream_50", "upstream_100", "downstream_50", "start_pm50", "start_pm100"]:
        seq = windows.get(name)
        mfe = run_rnafold(seq, rnafold_bin)
        out[f"{name}_MFE"] = mfe
        if seq and np.isfinite(mfe):
            out[f"{name}_MFE_norm"] = mfe / len(seq)
    probabilities = run_rnaplfold(windows.get("start_pm100"), rnaplfold_bin)
    out["accessibility_minus3_plus4"] = mean_unpaired(probabilities, -3, 4, 1)
    out["accessibility_start_codon"] = mean_unpaired(probabilities, 1, 3, 1)
    out["accessibility_start_pm10"] = mean_unpaired(probabilities, -10, 10, 1)
    out["accessibility_start_pm20"] = mean_unpaired(probabilities, -20, 20, 1)
    out["accessibility_upstream_50"] = mean_unpaired(probabilities, -50, -1, 1)
    out["accessibility_downstream_50"] = mean_unpaired(probabilities, 4, 50, 1)
    upstream = out["accessibility_upstream_50"]
    downstream = out["accessibility_downstream_50"]
    if pd.notna(upstream) and float(upstream) > 0 and pd.notna(downstream):
        out["accessibility_asymmetry"] = float(downstream) / float(upstream)
    out.update(scan_features(probabilities))
    out["structure_status"] = "ok"
    return out


def main() -> int:
    args = parse_args()
    ensure_dirs()
    archive_script(__file__)
    print_paths(
        "05_local_RNA_structure_four_group.py",
        inputs=[TABLE_DIR / "four_group_orf_metadata.tsv", GENOME_FA, RNAFOLD, RNAPLFOLD],
        outputs=[FEATURE_OUT, STATS_OUT, LOG_OUT],
    )
    rnafold_bin = resolve_executable(RNAFOLD, "RNAfold")
    rnaplfold_bin = resolve_executable(RNAPLFOLD, "RNAplfold")
    LOG_OUT.write_text(f"RNAfold={rnafold_bin or 'unavailable'}\nRNAplfold={rnaplfold_bin or 'unavailable'}\n")
    print(LOG_OUT.read_text().rstrip())

    master = read_master()
    if args.limit is not None:
        master = master.head(args.limit).copy()
        print(f"limit={args.limit}")
    if not GENOME_FA.exists():
        raise SystemExit(f"Genome FASTA not found: {GENOME_FA}")

    rows: List[Dict[str, object]] = []
    fasta = SeqIO.index(str(GENOME_FA), "fasta")
    try:
        chrom_keys = {chrom: resolve_chrom_key(fasta, str(chrom)) for chrom in master["chr"].dropna().astype(str).unique()}
        for chrom, sub in master.groupby("chr", sort=False, dropna=False):
            chrom_text = str(chrom)
            chrom_key = chrom_keys.get(chrom_text)
            chrom_seq = str(fasta[chrom_key].seq) if chrom_key is not None else None
            print(f"Processing {chrom_text}: {len(sub)} ORF(s)")
            if args.threads > 1:
                with ThreadPoolExecutor(max_workers=args.threads) as executor:
                    futures = [executor.submit(structure_features_for_row, row, chrom_seq, rnafold_bin, rnaplfold_bin) for _, row in sub.iterrows()]
                    for idx, future in enumerate(as_completed(futures), start=1):
                        rows.append(future.result())
                        if idx % 1000 == 0:
                            print(f"  completed {idx}/{len(futures)} on {chrom_text}")
            else:
                for idx, (_, row) in enumerate(sub.iterrows(), start=1):
                    rows.append(structure_features_for_row(row, chrom_seq, rnafold_bin, rnaplfold_bin))
                    if idx % 1000 == 0:
                        print(f"  completed {idx}/{len(sub)} on {chrom_text}")
    finally:
        fasta.close()

    out_df = pd.DataFrame(rows)
    out_df.to_csv(FEATURE_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {FEATURE_OUT}")

    stats_tables = []
    plot_specs = MFE_FEATURES + ACCESS_FEATURES + SCAN_FEATURES
    ncols = 4
    nrows = int(np.ceil(len(plot_specs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 5.8 * nrows), squeeze=False)
    for ax, (feature, ylabel, title) in zip(axes.flat, plot_specs):
        stats_tables.append(boxplot_four_group(ax, out_df, feature, y_label=ylabel, title=title, annotate=True))
    for ax in axes.flat[len(plot_specs) :]:
        ax.axis("off")
    stats_df = pd.concat([t for t in stats_tables if t is not None and not t.empty], ignore_index=True, sort=False)
    stats_df.to_csv(STATS_OUT, sep="\t", index=False, na_rep="NA")
    save_figure(fig, "Fig5_local_RNA_structure_four_group.pdf")
    print("05_local_RNA_structure_four_group.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
