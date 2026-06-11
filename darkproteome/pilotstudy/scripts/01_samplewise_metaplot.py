#!/usr/bin/env python3
"""Draw sample-wise start-centered RNA coverage metaplots by ORF group."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tc_common import (
    GROUP_ORDER,
    PDF_DIR,
    archive_script,
    display_group,
    ensure_fig_dirs,
    print_paths,
    read_groups,
    save_figure,
)
from tc_coverage_common import (
    POSITIONS,
    close_bigwigs,
    detected_orfs_by_sample,
    discover_bigwigs,
    nanmean_stack,
    open_bigwigs,
    read_sample_level_detection,
    start_centered_vector,
    vector_mean_normalize,
)


RAW_OUT = PDF_DIR / "samplewise_start_centered_metaplot_raw.pdf"
NORM_OUT = PDF_DIR / "samplewise_start_centered_metaplot_vector_normalized.pdf"


def build_metaplot_rows(groups: pd.DataFrame, handles: dict, detected_by_sample: dict[str, set[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows = []
    norm_rows = []
    group_lookup = {group: sub.reset_index(drop=True) for group, sub in groups.groupby("group", sort=False)}
    for sample, bw in handles.items():
        detected_ids = detected_by_sample.get(sample, set())
        for group in GROUP_ORDER:
            sub = group_lookup.get(group, pd.DataFrame())
            if sub.empty:
                continue
            sub = sub[sub["ORF_id"].isin(detected_ids)]
            if sub.empty:
                continue
            raw_vectors = []
            norm_vectors = []
            for _, row in sub.iterrows():
                vector = start_centered_vector(bw, row)
                raw_vectors.append(vector)
                norm_vectors.append(vector_mean_normalize(vector))
            raw_mean = nanmean_stack(raw_vectors)
            norm_mean = nanmean_stack(norm_vectors)
            if raw_mean is None or norm_mean is None:
                continue
            for pos, raw_value, norm_value in zip(POSITIONS, raw_mean, norm_mean):
                raw_rows.append(
                    {
                        "sample": sample,
                        "group": group,
                        "position": int(pos),
                        "mean_coverage": raw_value,
                        "n_orfs": len(sub),
                    }
                )
                norm_rows.append(
                    {
                        "sample": sample,
                        "group": group,
                        "position": int(pos),
                        "mean_normalized_coverage": norm_value,
                        "n_orfs": len(sub),
                    }
                )
    return pd.DataFrame(raw_rows), pd.DataFrame(norm_rows)


def plot_samplewise(df: pd.DataFrame, y_col: str, y_label: str, title: str, pdf_name: str) -> None:
    fig, axes = plt.subplots(1, len(GROUP_ORDER), figsize=(18, 5), sharex=True, sharey=False)
    for ax, group in zip(axes, GROUP_ORDER):
        sub = df[df["group"].eq(group)]
        if sub.empty:
            ax.axis("off")
            ax.set_title(display_group(group))
            continue
        sns.lineplot(data=sub, x="position", y=y_col, hue="sample", units="sample", estimator=None, ax=ax)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        if "normalized" in y_col:
            ax.axhline(1, color="gray", lw=0.8, ls=":")
        ax.set_title(display_group(group))
        ax.set_xlabel("Position")
        ax.set_ylabel(y_label)
        ax.legend(title="Sample", fontsize=7)
    fig.suptitle(title)
    save_figure(fig, pdf_name)


def main() -> int:
    ensure_fig_dirs()
    archive_script(__file__)
    print_paths(
        script_name=Path(__file__).name,
        inputs=[
            Path("pancreas.translated_orfs.sample_level.tsv"),
            Path("orf_groups.combined_metadata.tsv"),
            Path("bigwig/*.CPM.bw"),
        ],
        outputs=[RAW_OUT, NORM_OUT],
    )

    groups = read_groups()
    sample_level = read_sample_level_detection()
    detected_by_sample = detected_orfs_by_sample(sample_level)
    handles = open_bigwigs(discover_bigwigs())
    try:
        raw_df, norm_df = build_metaplot_rows(groups, handles, detected_by_sample)
    finally:
        close_bigwigs(handles)

    plot_samplewise(
        raw_df,
        "mean_coverage",
        "Mean CPM coverage",
        "Sample-wise start-centered RNA coverage",
        "samplewise_start_centered_metaplot_raw.pdf",
    )
    plot_samplewise(
        norm_df,
        "mean_normalized_coverage",
        "Coverage / vector mean",
        "Sample-wise start-centered RNA coverage, vector-mean normalized",
        "samplewise_start_centered_metaplot_vector_normalized.pdf",
    )
    print("01_samplewise_metaplot.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

