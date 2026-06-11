#!/usr/bin/env python3
"""Compare transcript expression and usage across four ORF groups."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fc_common import (
    LOG_DIR,
    PILOT_DIR,
    TABLE_DIR,
    archive_script,
    boxplot_four_group,
    clean_text,
    ensure_dirs,
    print_paths,
    read_master,
    save_figure,
)


ANNOTATION = PILOT_DIR / "stringtie_tpm_matrix.transcript_annotation.tsv"
TPM_MATRIX = PILOT_DIR / "stringtie_tpm_matrix.pilot8.tsv"
USAGE_MATRIX = PILOT_DIR / "stringtie_transcript_usage.pilot8.tsv"
FEATURE_OUT = TABLE_DIR / "four_group_transcript_expression_features.tsv"
STATS_OUT = TABLE_DIR / "transcript_expression_four_group_statistics.tsv"
MISSING_LOG = LOG_DIR / "missing_transcript_expression_files.txt"

SAMPLE_FEATURES = ["transcript_TPM", "gene_TPM", "transcript_gene_TPM_ratio", "transcript_usage"]
ORF_FEATURES = [
    ("mean_transcript_TPM", "TPM", "Mean transcript TPM"),
    ("var_transcript_TPM", "Variance", "Transcript TPM sample variance"),
    ("mean_gene_TPM", "TPM", "Mean gene TPM"),
    ("mean_transcript_usage", "Usage", "Mean transcript usage"),
    ("mean_transcript_gene_TPM_ratio", "Ratio", "Mean transcript/gene TPM ratio"),
]


def missing_inputs() -> List[Path]:
    return [path for path in [ANNOTATION, TPM_MATRIX, USAGE_MATRIX] if not path.exists()]


def read_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    if "transcript_id" not in df.columns:
        raise SystemExit(f"Missing transcript_id column in {path}")
    return df


def long_matrix(matrix: pd.DataFrame, value_name: str) -> pd.DataFrame:
    samples = [col for col in matrix.columns if col != "transcript_id"]
    long = matrix.melt(id_vars="transcript_id", value_vars=samples, var_name="sample", value_name=value_name)
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce").fillna(0.0)
    return long


def build_expression_features() -> pd.DataFrame:
    master = read_master()
    annotation = pd.read_csv(ANNOTATION, sep="\t", dtype=str)
    if "gene_id" not in annotation.columns:
        if "gene_name" in annotation.columns:
            annotation["gene_id"] = annotation["gene_name"]
        else:
            raise SystemExit(f"Annotation lacks gene_id/gene_name: {ANNOTATION}")
    annotation = annotation[["transcript_id", "gene_id"]].drop_duplicates("transcript_id")
    tpm = read_matrix(TPM_MATRIX)
    usage = read_matrix(USAGE_MATRIX)
    samples = [col for col in tpm.columns if col != "transcript_id"]

    tpm_annotated = tpm.merge(annotation, on="transcript_id", how="left")
    tpm_values = tpm_annotated[samples].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    gene_ids = tpm_annotated["gene_id"].map(clean_text)
    missing_gene = gene_ids.eq("")
    gene_ids.loc[missing_gene] = "missing_gene_id|" + tpm_annotated.loc[missing_gene, "transcript_id"].astype(str)
    gene_tpm_values = tpm_values.groupby(gene_ids, sort=False).transform("sum")
    ratio_values = tpm_values.divide(gene_tpm_values.where(gene_tpm_values.ne(0))).fillna(0.0)
    percentile_values = tpm_values.rank(axis=0, pct=True)
    max_usage_values = ratio_values.groupby(gene_ids, sort=False).transform("max")
    major_values = ratio_values.eq(max_usage_values) & gene_tpm_values.gt(0)

    gene_tpm = pd.concat([tpm[["transcript_id"]], gene_tpm_values], axis=1)
    ratio = pd.concat([tpm[["transcript_id"]], ratio_values], axis=1)
    percentile = pd.concat([tpm[["transcript_id"]], percentile_values], axis=1)
    major = pd.concat([tpm[["transcript_id"]], major_values.astype(int)], axis=1)

    long = long_matrix(tpm, "transcript_TPM")
    long = long.merge(long_matrix(gene_tpm, "gene_TPM"), on=["transcript_id", "sample"], how="left")
    long = long.merge(long_matrix(ratio, "transcript_gene_TPM_ratio"), on=["transcript_id", "sample"], how="left")
    long = long.merge(long_matrix(usage, "transcript_usage"), on=["transcript_id", "sample"], how="left")
    long = long.merge(long_matrix(percentile, "expression_percentile"), on=["transcript_id", "sample"], how="left")
    long = long.merge(long_matrix(major, "major_isoform_flag"), on=["transcript_id", "sample"], how="left")

    expanded = master[["ORF_id", "transcript_id", "plot_group"]].merge(long, on="transcript_id", how="left")
    expanded["detected_or_positive_ORF"] = np.where(expanded["plot_group"].eq("CPAT-negative noncoding"), 0, 1)
    return expanded


def main() -> int:
    ensure_dirs()
    archive_script(__file__)
    print_paths("04_transcript_expression_comparison.py", inputs=[ANNOTATION, TPM_MATRIX, USAGE_MATRIX, TABLE_DIR / "four_group_orf_metadata.tsv"], outputs=[FEATURE_OUT, STATS_OUT, MISSING_LOG])

    missing = missing_inputs()
    if missing:
        MISSING_LOG.write_text("\n".join(str(path) for path in missing) + "\n")
        print(f"Missing transcript expression input(s); wrote {MISSING_LOG}")
        pd.DataFrame().to_csv(FEATURE_OUT, sep="\t", index=False)
        pd.DataFrame().to_csv(STATS_OUT, sep="\t", index=False)
        return 0
    MISSING_LOG.write_text("No missing transcript expression files.\n")

    sample_df = build_expression_features()
    sample_df.to_csv(FEATURE_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {FEATURE_OUT}")

    numeric_cols = ["transcript_TPM", "gene_TPM", "transcript_gene_TPM_ratio", "transcript_usage", "expression_percentile", "major_isoform_flag"]
    for col in numeric_cols:
        sample_df[col] = pd.to_numeric(sample_df[col], errors="coerce")
    agg = sample_df.groupby(["ORF_id", "transcript_id", "plot_group"], as_index=False).agg(
        mean_transcript_TPM=("transcript_TPM", "mean"),
        var_transcript_TPM=("transcript_TPM", "var"),
        mean_gene_TPM=("gene_TPM", "mean"),
        mean_transcript_gene_TPM_ratio=("transcript_gene_TPM_ratio", "mean"),
        mean_transcript_usage=("transcript_usage", "mean"),
        mean_expression_percentile=("expression_percentile", "mean"),
        major_isoform_fraction=("major_isoform_flag", "mean"),
    )

    stats_tables = []
    fig, axes = plt.subplots(3, 2, figsize=(16, 18), squeeze=False)
    for ax, (feature, ylabel, title) in zip(axes.flat, ORF_FEATURES):
        stats_tables.append(boxplot_four_group(ax, agg, feature, y_label=ylabel, title=title, annotate=True))
    axes.flat[-1].axis("off")
    stats_df = pd.concat([t for t in stats_tables if t is not None and not t.empty], ignore_index=True, sort=False)
    stats_df.to_csv(STATS_OUT, sep="\t", index=False, na_rep="NA")
    save_figure(fig, "Fig4_transcript_expression_four_group.pdf")

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), squeeze=False)
    for ax, feature, ylabel, title in [
        (axes[0, 0], "transcript_TPM", "TPM", "Transcript TPM by sample"),
        (axes[0, 1], "gene_TPM", "TPM", "Gene TPM by sample"),
        (axes[1, 0], "transcript_gene_TPM_ratio", "Ratio", "Transcript/gene TPM ratio by sample"),
        (axes[1, 1], "transcript_usage", "Usage", "Transcript usage by sample"),
    ]:
        sns.boxplot(data=sample_df, x="sample", y=feature, hue="plot_group", showfliers=False, ax=ax)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=35)
    save_figure(fig, "Fig4_transcript_expression_samplewise_four_group.pdf")

    print("04_transcript_expression_comparison.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
