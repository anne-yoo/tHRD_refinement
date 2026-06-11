#!/usr/bin/env python3
"""Merge Transcript Context Module features and summarize group-level signals."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from tc_common import (
    GROUP_ORDER,
    TABLE_DIR,
    archive_script,
    bh_adjust,
    display_group,
    ensure_fig_dirs,
    fisher_binary_stats,
    mannwhitney_stats,
    print_paths,
    read_groups,
    read_sequence_context,
    save_figure,
)


MATRIX_OUT = TABLE_DIR / "orf_transcript_context_feature_matrix.tsv"
STATS_OUT = TABLE_DIR / "integrated_feature_statistics.tsv"


def optional_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"WARNING: optional table missing, skipping: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t", dtype=str)


def merge_optional(base: pd.DataFrame, path: Path, suffix: str) -> pd.DataFrame:
    other = optional_table(path)
    if other.empty or "ORF_id" not in other.columns:
        return base
    drop_cols = [col for col in ["group", "primary_noncanonical_category"] if col in other.columns]
    other = other.drop(columns=drop_cols)
    overlap = [col for col in other.columns if col in base.columns and col != "ORF_id"]
    other = other.rename(columns={col: f"{col}_{suffix}" for col in overlap})
    return base.merge(other, on="ORF_id", how="left")


def is_binary_feature(series: pd.Series) -> bool:
    vals = series.dropna().astype(str).str.lower().unique()
    return len(vals) > 0 and set(vals).issubset({"true", "false", "1", "0", "yes", "no"})


def continuous_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"start0", "end0"}
    cols = []
    for col in df.columns:
        if col in exclude or col in {"ORF_id", "group", "primary_noncanonical_category", "chr", "strand", "ORF_type", "start_codon"}:
            continue
        if is_binary_feature(df[col]):
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() >= 3:
            cols.append(col)
    return cols


def binary_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if is_binary_feature(df[col])]


def all_cryptic_effects(df: pd.DataFrame, feature: str, binary: bool) -> dict:
    canonical = df[df["group"].eq("group1_canonical_translated_ORF")]
    cryptic = df[df["group"].isin(["group2_translated_AUG_cryptic_ORF", "group3_translated_nonAUG_cryptic_ORF"])]
    if binary:
        work = pd.concat(
            [
                canonical.assign(_comparison_group="canonical"),
                cryptic.assign(_comparison_group="all_cryptic"),
            ],
            ignore_index=True,
        )
        stat = fisher_binary_stats(
            work,
            feature,
            group_col="_comparison_group",
            pairs=[("canonical", "all_cryptic")],
        )
        return stat.iloc[0].to_dict() if not stat.empty else {}
    work = pd.concat(
        [
            canonical.assign(_comparison_group="canonical"),
            cryptic.assign(_comparison_group="all_cryptic"),
        ],
        ignore_index=True,
    )
    stat = mannwhitney_stats(
        work,
        feature,
        group_col="_comparison_group",
        pairs=[("canonical", "all_cryptic")],
    )
    return stat.iloc[0].to_dict() if not stat.empty else {}


def main() -> int:
    ensure_fig_dirs()
    archive_script(__file__)
    print_paths(
        script_name=Path(__file__).name,
        inputs=[Path("feature tables under FIG_DIR/tables")],
        outputs=[MATRIX_OUT, STATS_OUT],
    )

    groups = read_groups()
    seq_context = read_sequence_context()
    seq = seq_context.drop(columns=[col for col in ["group"] if col in seq_context.columns])
    base = groups.merge(seq, on="ORF_id", how="left", suffixes=("", "_seq"))
    base = merge_optional(base, TABLE_DIR / "orf_rna_coverage_features.orf_level.tsv", "rna")
    base = merge_optional(base, TABLE_DIR / "upstream_scanning_features.tsv", "upstream")
    base = merge_optional(base, TABLE_DIR / "orf_structure_context_features.tsv", "structure")
    base = merge_optional(base, TABLE_DIR / "orf_positional_context_features.tsv", "position")
    base.to_csv(MATRIX_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {MATRIX_OUT}")

    continuous_cols = continuous_feature_columns(base)
    binary_cols = binary_feature_columns(base)
    stats_tables = []
    for feature in continuous_cols:
        stat_df = mannwhitney_stats(base, feature)
        stat_df["feature_type"] = "continuous"
        stats_tables.append(stat_df)
        all_row = all_cryptic_effects(base, feature, binary=False)
        if all_row:
            all_row["feature"] = feature
            all_row["feature_type"] = "continuous"
            stats_tables.append(pd.DataFrame([all_row]))
    for feature in binary_cols:
        stat_df = fisher_binary_stats(base, feature)
        stat_df["feature_type"] = "binary"
        stats_tables.append(stat_df)
        all_row = all_cryptic_effects(base, feature, binary=True)
        if all_row:
            all_row["feature"] = feature
            all_row["feature_type"] = "binary"
            stats_tables.append(pd.DataFrame([all_row]))
    stats_df = pd.concat(stats_tables, ignore_index=True, sort=False) if stats_tables else pd.DataFrame()
    if not stats_df.empty and "pvalue" in stats_df.columns:
        stats_df["padj_bh_all_features"] = bh_adjust(stats_df["pvalue"])
    stats_df.to_csv(STATS_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {STATS_OUT}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    heatmap_cols = continuous_cols[:30]
    if heatmap_cols:
        med = base.groupby("group")[heatmap_cols].median(numeric_only=True).reindex(GROUP_ORDER)
        z = med.apply(lambda col: (col - col.mean()) / col.std(ddof=0) if col.std(ddof=0) else col * np.nan, axis=0)
        sns.heatmap(z, cmap="vlag", center=0, ax=axes[0], cbar_kws={"label": "z-scored median"})
        axes[0].set_yticklabels([display_group(label.get_text()) for label in axes[0].get_yticklabels()], rotation=0)
        axes[0].set_title("Standardized median features by group")
    else:
        axes[0].axis("off")
        axes[0].set_title("No continuous features available")

    effect_df = stats_df.copy()
    if not effect_df.empty:
        effect_df["comparison"] = effect_df["group1_label"].fillna(effect_df["group1"].astype(str)) + " vs " + effect_df["group2_label"].fillna(effect_df["group2"].astype(str))
        effect_df["effect_size"] = pd.to_numeric(effect_df.get("rank_biserial", pd.Series(dtype=float)), errors="coerce")
        if "log2_odds_ratio" in effect_df.columns:
            effect_df["effect_size"] = effect_df["effect_size"].fillna(pd.to_numeric(effect_df["log2_odds_ratio"], errors="coerce"))
        effect_df = effect_df.dropna(subset=["effect_size"])
        top = effect_df.reindex(effect_df["effect_size"].abs().sort_values(ascending=False).index).head(30)
        sns.scatterplot(data=top, x="effect_size", y="feature", hue="comparison", ax=axes[1])
        axes[1].axvline(0, color="black", lw=0.8)
        axes[1].set_title("Largest feature effect sizes")
        axes[1].set_xlabel("Rank-biserial correlation or log2 odds ratio")
        axes[1].set_ylabel("")
    else:
        axes[1].axis("off")
        axes[1].set_title("No statistics available")

    save_figure(fig, "Fig6_integrated_feature_summary.pdf")
    print("06_integrated_feature_summary.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
