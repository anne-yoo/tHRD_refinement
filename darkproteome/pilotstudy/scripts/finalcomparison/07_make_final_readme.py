#!/usr/bin/env python3
"""Write README for final four-group comparison outputs."""

from __future__ import annotations

import sys
from pathlib import Path

from fc_common import FIG_DIR, INPUT_DIR, NEG_DIR, PILOT_DIR, archive_script, ensure_dirs


README_OUT = FIG_DIR / "README_finalcomparison.txt"


def exists_line(path: Path) -> str:
    return f"{path}\t{'FOUND' if path.exists() else 'MISSING'}"


def main() -> int:
    ensure_dirs()
    archive_script(__file__)
    inputs = [
        INPUT_DIR / "tables" / "orf_groups.combined_metadata.tsv",
        INPUT_DIR / "tables" / "orf_sequence_context_features.tsv",
        INPUT_DIR / "tables" / "orf_rna_coverage_features.sample_level.tsv",
        INPUT_DIR / "tables" / "orf_rna_coverage_features.orf_level.tsv",
        NEG_DIR / "tables" / "cpat_negative_orfs.combined_metadata_compatible.tsv",
        NEG_DIR / "tables" / "cpat_negative_orfs.sequence_context_features.tsv",
        NEG_DIR / "tables" / "cpat_negative_orfs.sampled_25863.genomic.tsv",
        NEG_DIR / "tables" / "cpat_negative_orf_qc_summary.tsv",
        PILOT_DIR / "stringtie_tpm_matrix.pilot8.tsv",
        PILOT_DIR / "stringtie_transcript_usage.pilot8.tsv",
    ]
    missing = [path for path in inputs if not path.exists()]
    text = f"""Final ORF Pilot Four-Group Comparison

Input files:
{chr(10).join(exists_line(path) for path in inputs)}

Group definitions:
1. Canonical: group1_canonical_translated_ORF from positive translated ORFs.
2. AUG noncanonical: group2_translated_AUG_cryptic_ORF from positive translated ORFs.
3. nonAUG noncanonical: group3_translated_nonAUG_cryptic_ORF from positive translated ORFs.
4. CPAT-negative noncoding: CPAT best ORF per noncoding transcript with zero positive ORFs, Coding_prob < 0.364, sampled to the requested negative set size.

CPAT-negative selection rule:
Noncoding GENCODE transcripts with no positive translated ORF were selected. Existing CPAT ORF output was filtered to those transcripts, one highest Coding_prob ORF was selected per transcript, ORFs below the human CPAT cutoff 0.364 were retained, then the final sampled set was mapped back to genomic coordinates.

Sample-specific features:
RNA coverage features, transcript TPM, gene TPM, transcript/gene TPM ratio, transcript usage, expression percentile, and sample-specific variance.

Sequence-specific features:
Kozak context, start codon class, upstream AUG burden, GC around the start, ORF length, and local RNA structure/accessibility.

Figure descriptions:
Fig0_CPAT_negative_set_QC.pdf: CPAT-negative selection flow, CPAT probability, start codons, and ORF length.
Fig1_sequence_context_four_group.pdf: Kozak, start-codon, upstream AUG, GC, and ORF length comparisons.
Fig2_coverage_four_group.pdf: ORF-level RNA coverage comparisons with all pairwise p-value annotations.
Fig3_coverage_samplewise_four_group.pdf: sample-wise RNA coverage trend plots without p-value annotations.
Fig3_coverage_metaplot_four_group.pdf: raw and vector-normalized start-centered metaplots.
Fig4_transcript_expression_four_group.pdf: ORF-level transcript expression/usage comparisons.
Fig4_transcript_expression_samplewise_four_group.pdf: sample-wise expression/usage trends without p-value annotations.
Fig5_local_RNA_structure_four_group.pdf: RNAfold MFE and RNAplfold accessibility comparisons.
Fig6_integrated_feature_summary.pdf: standardized median feature heatmap and effect-size summary.

Plotting/statistics notes:
Main four-group boxplots and binary fraction barplots use all six pairwise group comparisons. The scripts prefer statannotations.Annotator when available and fall back to manual p-value brackets. Sample-wise trend figures intentionally omit p-value annotations because ORF-sample counts are very large.

Missing data warnings:
{chr(10).join(str(path) for path in missing) if missing else 'No missing required input files detected at README creation time.'}
"""
    README_OUT.write_text(text)
    print(f"Wrote {README_OUT}")
    print("07_make_final_readme.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
