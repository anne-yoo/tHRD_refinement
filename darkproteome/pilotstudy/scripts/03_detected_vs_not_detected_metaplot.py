#!/usr/bin/env python3
"""Plot detected vs not-detected start-centered metaplots by ORF group."""

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
    detection_pairs,
    discover_bigwigs,
    nanmean_stack,
    open_bigwigs,
    read_sample_level_detection,
    start_centered_vector,
    status_label,
    vector_mean_normalize,
)


RAW_OUT = PDF_DIR / "detected_vs_not_detected_metaplot_raw.pdf"
NORM_OUT = PDF_DIR / "detected_vs_not_detected_metaplot_vector_normalized.pdf"


def build_rows(groups: pd.DataFrame, handles: dict, detected_pairs: set[tuple[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_vectors: dict[tuple[str, str], list] = {}
    norm_vectors: dict[tuple[str, str], list] = {}
    total = len(groups) * len(handles)
    processed = 0
    for _, row in groups.iterrows():
        group = row["group"]
        for sample, bw in handles.items():
            processed += 1
            if processed % 10000 == 0:
                print(f"Processed {processed}/{total} ORF-sample pair(s)")
            status = status_label(int((row["ORF_id"], sample) in detected_pairs))
            vector = start_centered_vector(bw, row)
            raw_vectors.setdefault((group, status), []).append(vector)
            norm_vectors.setdefault((group, status), []).append(vector_mean_normalize(vector))

    raw_rows = []
    norm_rows = []
    for (group, status), vectors in raw_vectors.items():
        mean_vector = nanmean_stack(vectors)
        if mean_vector is None:
            continue
        for pos, value in zip(POSITIONS, mean_vector):
            raw_rows.append({"group": group, "detection_status": status, "position": int(pos), "mean_coverage": value, "n_vectors": len(vectors)})
    for (group, status), vectors in norm_vectors.items():
        mean_vector = nanmean_stack(vectors)
        if mean_vector is None:
            continue
        for pos, value in zip(POSITIONS, mean_vector):
            norm_rows.append({"group": group, "detection_status": status, "position": int(pos), "mean_normalized_coverage": value, "n_vectors": len(vectors)})
    return pd.DataFrame(raw_rows), pd.DataFrame(norm_rows)


def plot_rows(df: pd.DataFrame, y_col: str, y_label: str, title: str, pdf_name: str) -> None:
    fig, axes = plt.subplots(1, len(GROUP_ORDER), figsize=(18, 5), sharex=True, sharey=False)
    for ax, group in zip(axes, GROUP_ORDER):
        sub = df[df["group"].eq(group)]
        if sub.empty:
            ax.axis("off")
            ax.set_title(display_group(group))
            continue
        sns.lineplot(data=sub, x="position", y=y_col, hue="detection_status", ax=ax)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        if "normalized" in y_col:
            ax.axhline(1, color="gray", lw=0.8, ls=":")
        ax.set_title(display_group(group))
        ax.set_xlabel("Position")
        ax.set_ylabel(y_label)
        ax.legend(title="ORF-sample status")
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
    detected = detection_pairs(sample_level)
    handles = open_bigwigs(discover_bigwigs())
    try:
        raw_df, norm_df = build_rows(groups, handles, detected)
    finally:
        close_bigwigs(handles)

    plot_rows(
        raw_df,
        "mean_coverage",
        "Mean CPM coverage",
        "Detected vs not-detected start-centered RNA coverage",
        "detected_vs_not_detected_metaplot_raw.pdf",
    )
    plot_rows(
        norm_df,
        "mean_normalized_coverage",
        "Coverage / vector mean",
        "Detected vs not-detected start-centered RNA coverage, vector-mean normalized",
        "detected_vs_not_detected_metaplot_vector_normalized.pdf",
    )
    print("03_detected_vs_not_detected_metaplot.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

