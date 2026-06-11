#!/usr/bin/env python3
"""Create integrated four-group feature matrix and summary figures."""

from __future__ import annotations

import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fc_common import (
    NEGATIVE_GROUP,
    PLOT_GROUP_ORDER,
    TABLE_DIR,
    archive_script,
    bh_adjust,
    clean_text,
    ensure_dirs,
    fisher_stats,
    mannwhitney_stats,
    read_master,
    read_tsv,
    save_figure,
)


INTEGRATED_OUT = TABLE_DIR / "four_group_integrated_feature_matrix.tsv"
STATS_OUT = TABLE_DIR / "four_group_integrated_statistics.tsv"

FEATURE_INPUTS = [
    ("seq", TABLE_DIR / "sequence_context_four_group_features.tsv"),
    ("cov", TABLE_DIR / "four_group_coverage_features.orf_level.tsv"),
    ("expr", TABLE_DIR / "four_group_transcript_expression_features.tsv"),
    ("struct", TABLE_DIR / "four_group_local_RNA_structure_features.tsv"),
]


def add_prefixed_features(base: pd.DataFrame, prefix: str, path) -> pd.DataFrame:
    df = read_tsv(path, allow_missing=True)
    if df.empty or "ORF_id" not in df.columns:
        print(f"Skipping missing/empty feature file: {path}")
        return base
    if prefix == "expr" and "sample" in df.columns:
        numeric_cols = [col for col in df.columns if col not in {"ORF_id", "transcript_id", "plot_group", "sample"}]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.groupby("ORF_id", as_index=False)[numeric_cols].mean(numeric_only=True)
    keep_cols = [col for col in df.columns if col != "ORF_id" and col not in {"plot_group", "transcript_id"}]
    rename = {col: f"{prefix}_{col}" for col in keep_cols}
    return base.merge(df[["ORF_id"] + keep_cols].rename(columns=rename), on="ORF_id", how="left")


def boolean_like(series: pd.Series) -> bool:
    vals = {clean_text(v).lower() for v in series.dropna().unique()}
    vals.discard("")
    return bool(vals) and vals.issubset({"true", "false", "1", "0", "yes", "no"})


def numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    skip = {"ORF_id", "transcript_id", "plot_group", "chr", "strand", "start_codon", "ORF_type", "ORF_type2", "primary_noncanonical_category", "source", "CPAT_prediction"}
    cols = []
    for col in df.columns:
        if col in skip:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() >= 3:
            cols.append(col)
    return cols


def binary_feature_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if col not in {"ORF_id", "plot_group"} and boolean_like(df[col])]


def translated_noncanonical_vs_negative(df: pd.DataFrame, feature: str, binary: bool = False) -> pd.DataFrame:
    work = df[df["plot_group"].isin(["AUG noncanonical", "nonAUG noncanonical", NEGATIVE_GROUP])].copy()
    work["plot_group"] = np.where(work["plot_group"].eq(NEGATIVE_GROUP), NEGATIVE_GROUP, "Translated noncanonical")
    if binary:
        out = fisher_stats(work, feature, pairs=[("Translated noncanonical", NEGATIVE_GROUP)])
    else:
        out = mannwhitney_stats(work, feature, pairs=[("Translated noncanonical", NEGATIVE_GROUP)])
    out["comparison"] = "translated_noncanonical_vs_CPAT_negative"
    return out


def make_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    target_pairs = [
        ("Canonical", NEGATIVE_GROUP),
        ("AUG noncanonical", NEGATIVE_GROUP),
        ("nonAUG noncanonical", NEGATIVE_GROUP),
    ]
    for feature in numeric_feature_columns(df):
        stats_df = mannwhitney_stats(df, feature, pairs=target_pairs)
        stats_df["comparison"] = stats_df["group1"] + "_vs_" + stats_df["group2"]
        rows.append(stats_df)
        rows.append(translated_noncanonical_vs_negative(df, feature, binary=False))
    for feature in binary_feature_columns(df):
        stats_df = fisher_stats(df, feature, pairs=target_pairs)
        stats_df["comparison"] = stats_df["group1"] + "_vs_" + stats_df["group2"]
        rows.append(stats_df)
        rows.append(translated_noncanonical_vs_negative(df, feature, binary=True))
    out = pd.concat([r for r in rows if r is not None and not r.empty], ignore_index=True, sort=False) if rows else pd.DataFrame()
    if not out.empty:
        out["padj_bh_all_integrated_tests"] = bh_adjust(out["pvalue"])
    return out


def plot_heatmap(df: pd.DataFrame, ax) -> None:
    numeric_cols = numeric_feature_columns(df)
    if not numeric_cols:
        ax.axis("off")
        ax.text(0.5, 0.5, "No numeric integrated features", ha="center", va="center")
        return
    med = df.groupby("plot_group")[numeric_cols].median(numeric_only=True).reindex(PLOT_GROUP_ORDER)
    med = med.dropna(axis=1, how="all")
    if med.empty:
        ax.axis("off")
        return
    z = (med - med.mean(axis=0)) / med.std(axis=0).replace(0, np.nan)
    z = z.fillna(0)
    if z.shape[1] > 40:
        variance = z.var(axis=0).sort_values(ascending=False)
        z = z[variance.head(40).index]
    sns.heatmap(z, cmap="vlag", center=0, ax=ax, cbar_kws={"label": "z-scored group median"})
    ax.set_title("Standardized median features by group")
    ax.set_xlabel("Feature")
    ax.set_ylabel("")


def plot_effect_sizes(stats_df: pd.DataFrame, ax) -> None:
    if stats_df.empty or "effect_size" not in stats_df.columns:
        ax.axis("off")
        return
    work = stats_df[stats_df["group2"].eq(NEGATIVE_GROUP)].copy()
    work["effect_size"] = pd.to_numeric(work["effect_size"], errors="coerce")
    work = work.dropna(subset=["effect_size"])
    if work.empty:
        ax.axis("off")
        return
    work["abs_effect"] = work["effect_size"].abs()
    work = work.sort_values("abs_effect", ascending=False).head(30)
    sns.barplot(data=work, y="feature", x="effect_size", hue="group1", dodge=False, ax=ax)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Largest effects vs CPAT-negative")
    ax.set_xlabel("Effect size")
    ax.set_ylabel("")


def main() -> int:
    ensure_dirs()
    archive_script(__file__)
    print("06_integrated_feature_summary.py")
    master = read_master()
    integrated = master.copy()
    for prefix, path in FEATURE_INPUTS:
        integrated = add_prefixed_features(integrated, prefix, path)
    integrated.to_csv(INTEGRATED_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {INTEGRATED_OUT}")

    stats_df = make_statistics(integrated)
    stats_df.to_csv(STATS_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {STATS_OUT}")

    fig, axes = plt.subplots(2, 1, figsize=(17, 15.5))
    plot_heatmap(integrated, axes[0])
    plot_effect_sizes(stats_df, axes[1])
    save_figure(fig, "Fig6_integrated_feature_summary.pdf")
    print("06_integrated_feature_summary.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
