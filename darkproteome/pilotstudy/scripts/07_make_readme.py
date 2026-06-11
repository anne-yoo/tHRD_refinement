#!/usr/bin/env python3
"""Create a text README summarizing generated Transcript Context figures."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from tc_common import FIG_DIR, INPUT_DIR, TABLE_DIR, archive_script, display_group, ensure_fig_dirs, print_paths


README_OUT = FIG_DIR / "README_figures.txt"


def safe_read(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, sep="\t", dtype=str)
    return pd.DataFrame()


def main() -> int:
    ensure_fig_dirs()
    archive_script(__file__)
    print_paths(script_name=Path(__file__).name, inputs=[INPUT_DIR], outputs=[README_OUT])

    groups = safe_read(INPUT_DIR / "tables" / "orf_groups.combined_metadata.tsv")
    lines = []
    lines.append("Transcript Context Module pilot figures")
    lines.append("=" * 48)
    lines.append("")
    lines.append("Input files used:")
    lines.append(f"- {INPUT_DIR / 'tables' / 'orf_groups.combined_metadata.tsv'}")
    lines.append(f"- {INPUT_DIR / 'tables' / 'orf_sequence_context_features.tsv'}")
    lines.append(f"- {INPUT_DIR / 'bigwig' / '*.CPM.bw'}")
    lines.append(f"- {INPUT_DIR / 'bed' / '*.bed'}")
    lines.append("")
    lines.append("ORF counts by group:")
    if groups.empty or "group" not in groups.columns:
        lines.append("- unavailable")
    else:
        for group, count in groups["group"].value_counts().sort_index().items():
            lines.append(f"- {display_group(group)}: {count}")
    lines.append("")
    lines.append("Figures:")
    lines.append("- Fig1_start_context_kozak: Kozak -3/+4/strong fractions and start codon type distribution.")
    lines.append("- Fig1_start_context_logo: sequence context logo or fallback base-frequency heatmap for each group.")
    lines.append("- Fig2_rna_coverage_context_boxplots: RNA coverage, covered fraction, CV, and ORF/upstream ratio.")
    lines.append("- Fig2_start_centered_coverage_metaplot: absolute mean start-centered RNA CPM coverage from -100 to +100.")
    lines.append("- Fig2_start_centered_coverage_metaplot_ORFmean_normalized: per-ORF start-centered coverage divided by that ORF's own start-centered vector mean, then averaged by group.")
    lines.append("- start_centered_coverage_matrix.ORFbody_mean_normalized.tsv: diagnostic legacy-style normalization using full-ORF body mean coverage as denominator.")
    lines.append("- Fig3_upstream_scanning_burden: upstream AUG burden features.")
    lines.append("- Fig4_structure_context: GC and optional RNAfold MFE proxy features around ORF starts.")
    lines.append("- Fig5_orf_positional_context: ORF positional category summaries.")
    lines.append("- Fig6_integrated_feature_summary: merged feature matrix heatmap and effect-size overview.")
    lines.append("")
    lines.append("Warnings and limitations:")
    warning_path = TABLE_DIR / "structure_context_warnings.txt"
    if warning_path.exists():
        lines.append(warning_path.read_text().strip())
    else:
        lines.append("- structure_context_warnings.txt not found; run 04_structure_context.py to check RNAfold availability.")
    if not (TABLE_DIR / "orf_rna_coverage_features.orf_level.tsv").exists():
        lines.append("- RNA coverage feature table not found; run 02_rna_coverage_context.py after BigWigs are available.")
    lines.append("- Upstream ORF counts are currently NA because confident uORF inference needs transcript exon/CDS mapping.")
    lines.append("- Distance from transcript start/CDS is currently NA because GTF-based transcript coordinate mapping is required.")
    lines.append("- Current analysis only uses translated ORFs and does not yet include true non-translated ORF negatives.")
    lines.append("")
    lines.append("Output folders:")
    lines.append(f"- Tables: {TABLE_DIR}")
    lines.append(f"- PDF: {FIG_DIR / 'pdf'}")
    lines.append(f"- PNG: {FIG_DIR / 'png'}")
    lines.append(f"- Archived scripts: {FIG_DIR / 'scripts'}")

    README_OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {README_OUT}")
    print("07_make_readme.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
