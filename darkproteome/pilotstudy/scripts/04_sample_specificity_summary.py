#!/usr/bin/env python3
"""Summarize ORF sample recurrence and accumulation across samples."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tc_common import (
    ALL_SAMPLES,
    GROUP_ORDER,
    TABLE_DIR,
    archive_script,
    display_group,
    ensure_fig_dirs,
    print_paths,
    read_groups,
    save_figure,
)
from tc_coverage_common import read_sample_level_detection


RECURRENCE_OUT = TABLE_DIR / "orf_recurrence_summary.tsv"
ACCUMULATION_OUT = TABLE_DIR / "orf_accumulation_curve.tsv"
N_PERMUTATIONS = int(os.environ.get("N_ACCUMULATION_PERMUTATIONS", "100"))
RANDOM_SEED = int(os.environ.get("ACCUMULATION_RANDOM_SEED", "1"))


def recurrence_summary(groups: pd.DataFrame) -> pd.DataFrame:
    work = groups.copy()
    work["n_detected_samples"] = pd.to_numeric(work["n_detected_samples"], errors="coerce").astype("Int64")
    return (
        work.groupby(["group", "n_detected_samples"], dropna=False)
        .size()
        .reset_index(name="n_orfs")
        .sort_values(["group", "n_detected_samples"])
    )


def plot_recurrence(groups: pd.DataFrame) -> None:
    work = groups.copy()
    work["n_detected_samples"] = pd.to_numeric(work["n_detected_samples"], errors="coerce")
    counts = (
        work.groupby(["group", "n_detected_samples"], dropna=False)
        .size()
        .reset_index(name="n_orfs")
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=counts, x="n_detected_samples", y="n_orfs", hue="group", hue_order=GROUP_ORDER, ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [display_group(label) for label in labels], title="Group")
    ax.set_xlabel("Number of detected samples")
    ax.set_ylabel("Unique ORFs")
    ax.set_title("ORF recurrence distribution by group")
    save_figure(fig, "orf_recurrence_distribution_by_group.pdf")


def accumulation_for_order(order: list[str], sample_to_orfs: dict[str, set[str]]) -> list[int]:
    seen: set[str] = set()
    counts = []
    for sample in order:
        seen.update(sample_to_orfs.get(sample, set()))
        counts.append(len(seen))
    return counts


def accumulation_table(sample_level: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    available_samples = [sample for sample in ALL_SAMPLES if sample in set(sample_level["sample"].astype(str))]
    extra_samples = sorted(set(sample_level["sample"].astype(str)) - set(available_samples))
    available_samples.extend(extra_samples)

    scopes: dict[str, set[str]] = {"all": set(groups["ORF_id"].astype(str))}
    for group in GROUP_ORDER:
        scopes[group] = set(groups.loc[groups["group"].eq(group), "ORF_id"].astype(str))

    rows = []
    for scope, allowed_orfs in scopes.items():
        sample_to_orfs = {
            sample: set(sub["ORF_id"].astype(str)) & allowed_orfs
            for sample, sub in sample_level.groupby("sample")
        }
        curves = []
        for _ in range(N_PERMUTATIONS):
            order = list(rng.permutation(available_samples))
            curves.append(accumulation_for_order(order, sample_to_orfs))
        arr = np.asarray(curves, dtype=float)
        for idx in range(len(available_samples)):
            values = arr[:, idx]
            rows.append(
                {
                    "scope": scope,
                    "scope_label": "All ORFs" if scope == "all" else display_group(scope),
                    "n_samples_added": idx + 1,
                    "mean_unique_orfs": float(np.mean(values)),
                    "ci_lower": float(np.percentile(values, 2.5)),
                    "ci_upper": float(np.percentile(values, 97.5)),
                    "n_permutations": N_PERMUTATIONS,
                }
            )
    return pd.DataFrame(rows)


def plot_accumulation(accum: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for scope, sub in accum.groupby("scope", sort=False):
        sub = sub.sort_values("n_samples_added")
        label = sub["scope_label"].iloc[0]
        x = sub["n_samples_added"].to_numpy(dtype=float)
        y = sub["mean_unique_orfs"].to_numpy(dtype=float)
        lower = sub["ci_lower"].to_numpy(dtype=float)
        upper = sub["ci_upper"].to_numpy(dtype=float)
        ax.plot(x, y, label=label)
        ax.fill_between(x, lower, upper, alpha=0.18)
    ax.set_xlabel("Number of samples added")
    ax.set_ylabel("Accumulated unique ORFs")
    ax.set_title("Unique ORF accumulation curve")
    ax.legend(title="Scope")
    save_figure(fig, "orf_accumulation_curve.pdf")


def main() -> int:
    ensure_fig_dirs()
    archive_script(__file__)
    print_paths(
        script_name=Path(__file__).name,
        inputs=[Path("pancreas.translated_orfs.sample_level.tsv"), Path("orf_groups.combined_metadata.tsv")],
        outputs=[RECURRENCE_OUT, ACCUMULATION_OUT],
    )

    groups = read_groups()
    sample_level = read_sample_level_detection()
    recurrence = recurrence_summary(groups)
    recurrence.to_csv(RECURRENCE_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {RECURRENCE_OUT}")
    plot_recurrence(groups)

    accum = accumulation_table(sample_level, groups)
    accum.to_csv(ACCUMULATION_OUT, sep="\t", index=False, na_rep="NA")
    print(f"Wrote {ACCUMULATION_OUT}")
    plot_accumulation(accum)

    print("04_sample_specificity_summary.py completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
