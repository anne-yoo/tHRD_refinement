#%%
#!/usr/bin/env python3
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager, patches, rcParams
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns


BASE_DIR = Path("/home/jiye/jiye/copycomparison")
SEV_DIR = BASE_DIR / "GENCODEquant/SEV_prepost"
OUT_DIR = SEV_DIR / "2605figs/target_DUT_threepanel"

SQANTI_PATH = SEV_DIR / "sqantioutput/sqanti_hg19_classification.txt"
AR_DUT_PATH = SEV_DIR / "merged_cov5_analysis/whole_AR_stable_DUT_Wilcoxon_delta_withna.txt"
IR_DUT_PATH = SEV_DIR / "merged_cov5_analysis/whole_IR_stable_DUT_Wilcoxon_delta_withna.txt"
BASELINE_DUT_PATH = SEV_DIR / "merged_cov5_analysis/whole_baseline_ARpre_vs_IRpre_stable_DUT_MannWhitney_delta_withna.txt"
AR_DEG_PATH = SEV_DIR / "merged_cov5_analysis/whole_AR_Wilcoxon_DEGresult_FC.txt"
IR_DEG_PATH = SEV_DIR / "merged_cov5_analysis/whole_IR_Wilcoxon_DEGresult_FC.txt"
SAMPLEINFO_PATH = BASE_DIR / "gDUTresearch/GEN_FINALDATA/SEV_prepost_80_clinicalinfo.txt"
TRANSCRIPT_TPM_PATH = SEV_DIR / "merged_cov5_80_transcript_TPM.txt"
GENE_TPM_PATH = SEV_DIR / "merged_cov5_80_gene_TPM.txt"
MAJORMINOR_PATH = SEV_DIR / "merged_cov5_80_majorminorlist.txt"
GTF_PATH = SEV_DIR / "merged_80_cov5.gtf"

TARGETS = [
    {"gene": "CCNE1", "transcript_id": "ENST00000262643.3", "target_class": "Class1"},
    {"gene": "AURKA", "transcript_id": "ENST00000347343.2", "target_class": "Class1"},
    {"gene": "TOPBP1", "transcript_id": "ENST00000260810.5", "target_class": "Class1"},
    {"gene": "RAD51AP1", "transcript_id": "ENST00000352618.4", "target_class": "Class1"},
    {"gene": "FANCA", "transcript_id": "MSTRG.203888.31", "target_class": "Class3"},
    {"gene": "RECQL", "transcript_id": "MSTRG.107505.6", "target_class": "Class3"},
]

CONDITION_ORDER = ["AR pre", "IR pre", "IR post", "AR post"]
CONDITION_COLORS = {
    "AR pre": "#FDD49E",
    "IR pre": "#C7E9C0",
    "IR post": "#5AAE61",
    "AR post": "#F28E2B",
}
CLASS_ORDER = {"Class1": 0, "Class2": 1, "Class3": 2, "Unclassified": 3}
CLASS_COLORS = {
    "Class1": "#9CC7F2",
    "Class2": "#9EDC9A",
    "Class3": "#C7A3E6",
    "Unclassified": "#9E9E9E",
}


def configure_plot_style():
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    for arial_font_path in [
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/arialbd.ttf",
    ]:
        if Path(arial_font_path).exists():
            font_manager.fontManager.addfont(arial_font_path)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams.update(
        {
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 11,
            "hatch.linewidth": 0.9,
        }
    )
    sns.set_style("white")


def split_transcript_gene(transcript_gene):
    transcript_id, gene = str(transcript_gene).split("-", 1)
    return transcript_id, gene


def mean_and_ci(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return np.nan, 0.0, 0
    mean_value = float(np.mean(values))
    if values.size == 1:
        return mean_value, 0.0, 1
    sem = stats.sem(values, nan_policy="omit")
    ci = float(stats.t.ppf(0.975, values.size - 1) * sem) if np.isfinite(sem) else 0.0
    return mean_value, ci, int(values.size)


def style_panel_box(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.8)
    ax.tick_params(color="#333333", width=0.8)


def load_inputs():
    sqanti = pd.read_csv(SQANTI_PATH, sep="\t", low_memory=False)
    sqanti.dropna(axis=1, how="all", inplace=True)

    ar_dut = pd.read_csv(AR_DUT_PATH, sep="\t", index_col=0)
    ir_dut = pd.read_csv(IR_DUT_PATH, sep="\t", index_col=0)
    baseline_dut = pd.read_csv(BASELINE_DUT_PATH, sep="\t", index_col=0)
    ar_deg = pd.read_csv(AR_DEG_PATH, sep="\t")
    ir_deg = pd.read_csv(IR_DEG_PATH, sep="\t")

    ar_dutlist = ar_dut.loc[
        (ar_dut["p_value"] < 0.05) & (np.abs(ar_dut["delta_TU"]) > 0.05)
    ].index.to_list()
    ir_dutlist = ir_dut.loc[
        (ir_dut["p_value"] < 0.05) & (np.abs(ir_dut["delta_TU"]) > 0.05)
    ].index.to_list()
    baseline_dutlist = baseline_dut.loc[
        (baseline_dut["p_value"] < 0.05) & (np.abs(baseline_dut["delta_TU"]) > 0.05)
    ].index.to_list()

    sampleinfo = pd.read_csv(SAMPLEINFO_PATH, sep="\t", index_col=0)
    geneexp = pd.read_csv(GENE_TPM_PATH, sep="\t", index_col=0)
    majorminor = pd.read_csv(MAJORMINOR_PATH, sep="\t")

    transexp = pd.read_csv(TRANSCRIPT_TPM_PATH, sep="\t", index_col=0)
    transexp = transexp.iloc[:, :-1]
    transexp = transexp.loc[(transexp > 0).sum(axis=1) >= 8].copy()
    transcript_genes = transexp.index.to_series().str.split("-", n=1).str[-1]
    gene_sum = transexp.groupby(transcript_genes).transform("sum")
    filtered_trans = transexp.div(gene_sum.replace(0, np.nan)).fillna(0)

    return {
        "sqanti": sqanti,
        "sampleinfo": sampleinfo,
        "geneexp": geneexp,
        "filtered_trans": filtered_trans,
        "majorminor": majorminor,
        "ar_deg": ar_deg,
        "ir_deg": ir_deg,
        "ar_dutlist": ar_dutlist,
        "ir_dutlist": ir_dutlist,
        "baseline_dutlist": baseline_dutlist,
    }


def build_class_map(sqanti, majorminor):
    majorlist = majorminor.loc[majorminor["type"] == "major", "Transcript-Gene"].to_list()

    df_cat = sqanti[
        ["isoform", "structural_category", "subcategory", "within_CAGE_peak", "coding"]
    ].copy()
    df_cat["isoform_clean"] = df_cat["isoform"].astype(str).str.split("-", n=1).str[0]

    majorlist_set = {x.split("-", 1)[0] for x in majorlist}
    coding_set = set(sqanti.loc[sqanti["coding"] == "coding", "isoform"])
    majorlist_set = majorlist_set.intersection(coding_set)

    df_cat["major"] = df_cat["isoform"].isin(majorlist_set)
    valid_cat = {"full-splice_match", "novel_in_catalog"}
    df_cat["major"] = df_cat["major"] & df_cat["structural_category"].isin(valid_cat)
    df_cat.set_index("isoform", inplace=True)

    group1 = set(df_cat.loc[df_cat["major"] == True, "isoform_clean"])
    group2 = set(
        df_cat.loc[(df_cat["major"] == False) & (df_cat["coding"] == "coding"), "isoform_clean"]
    )
    group3 = set(
        df_cat.loc[
            (df_cat["major"] == False) & (df_cat["coding"] == "non_coding"),
            "isoform_clean",
        ]
    )

    class1 = set(majorminor.loc[majorminor["transcriptid"].isin(group1), "Transcript-Gene"])
    class2 = set(majorminor.loc[majorminor["transcriptid"].isin(group2), "Transcript-Gene"])
    class3 = set(majorminor.loc[majorminor["transcriptid"].isin(group3), "Transcript-Gene"])

    class_map = {}
    for transcript_gene in majorminor["Transcript-Gene"]:
        if transcript_gene in class1:
            class_map[transcript_gene] = "Class1"
        elif transcript_gene in class2:
            class_map[transcript_gene] = "Class2"
        elif transcript_gene in class3:
            class_map[transcript_gene] = "Class3"
        else:
            class_map[transcript_gene] = "Unclassified"
    return class_map


def build_condition_samples(sampleinfo, available_columns):
    available_columns = set(available_columns)
    condition_specs = {
        "AR pre": (1, "pre"),
        "IR pre": (0, "pre"),
        "IR post": (0, "post"),
        "AR post": (1, "post"),
    }
    condition_samples = {}
    for condition, (response_value, treatment_value) in condition_specs.items():
        samples = sampleinfo.loc[
            (sampleinfo["response"] == response_value)
            & (sampleinfo["treatment"] == treatment_value),
            "sample_full",
        ].to_list()
        condition_samples[condition] = [sample for sample in samples if sample in available_columns]
    return condition_samples


def summarize_values_by_condition(matrix, row_name, condition_samples, transform=None):
    if row_name not in matrix.index:
        raise ValueError(f"{row_name} was not found in {matrix.index.name or 'matrix index'}.")

    records = []
    row = matrix.loc[row_name]
    for condition in CONDITION_ORDER:
        values = row.reindex(condition_samples[condition])
        if transform is not None:
            values = transform(values)
        values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
        mean_value, ci, sample_count = mean_and_ci(values)
        records.append(
            {
                "Condition": condition,
                "Mean": mean_value,
                "CI95": ci,
                "Sample_count": sample_count,
                "Values": values,
            }
        )
    return pd.DataFrame(records)


def select_transcripts_for_target(target, filtered_trans, class_map):
    gene = target["gene"]
    target_tg = f"{target['transcript_id']}-{gene}"
    if target_tg not in filtered_trans.index:
        raise ValueError(f"Target transcript is missing from TU table: {target_tg}")

    gene_rows = [
        transcript_gene
        for transcript_gene in filtered_trans.index
        if split_transcript_gene(transcript_gene)[1] == gene
    ]
    if len(gene_rows) < 4:
        raise ValueError(f"{gene} has fewer than 4 detected transcripts in the TU table.")

    gene_means = filtered_trans.loc[gene_rows].mean(axis=1).sort_values(ascending=False)
    mean_rank = {transcript_gene: rank + 1 for rank, transcript_gene in enumerate(gene_means.index)}

    top4 = list(gene_means.head(4).index)
    if target_tg in top4:
        selected = top4
        inclusion_reason = {transcript_gene: "top4" for transcript_gene in selected}
    else:
        selected = top4[:3] + [target_tg]
        inclusion_reason = {transcript_gene: "top4" for transcript_gene in top4[:3]}
        inclusion_reason[target_tg] = "forced_target"

    selected = sorted(
        selected,
        key=lambda transcript_gene: (
            CLASS_ORDER.get(class_map.get(transcript_gene, "Unclassified"), 3),
            -float(gene_means.loc[transcript_gene]),
            split_transcript_gene(transcript_gene)[0],
        ),
    )

    records = []
    for display_order, transcript_gene in enumerate(selected, start=1):
        transcript_id, _ = split_transcript_gene(transcript_gene)
        records.append(
            {
                "gene": gene,
                "target_transcript_id": target["transcript_id"],
                "selected_transcript_id": transcript_id,
                "selected_transcript_gene": transcript_gene,
                "transcript_class": class_map.get(transcript_gene, "Unclassified"),
                "mean_TU_all_samples": float(gene_means.loc[transcript_gene]),
                "mean_TU_rank_in_gene": int(mean_rank[transcript_gene]),
                "display_order": display_order,
                "is_target": transcript_gene == target_tg,
                "inclusion_reason": inclusion_reason.get(transcript_gene, "top4"),
            }
        )
    return selected, pd.DataFrame(records)


def parse_gtf_exons(gtf_path, wanted_transcript_ids):
    wanted_transcript_ids = set(wanted_transcript_ids)
    transcript_id_pattern = re.compile(r'transcript_id "([^"]+)"')
    transcript_name_pattern = re.compile(r'transcript_name "([^"]+)"')
    exons_by_transcript = {transcript_id: [] for transcript_id in wanted_transcript_ids}

    with open(gtf_path) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "exon":
                continue

            match = transcript_id_pattern.search(fields[8])
            if match is None:
                match = transcript_name_pattern.search(fields[8])
            if match is None:
                continue

            transcript_id = match.group(1)
            if transcript_id not in wanted_transcript_ids:
                continue

            exons_by_transcript[transcript_id].append(
                {
                    "chrom": fields[0],
                    "start": int(fields[3]),
                    "end": int(fields[4]),
                    "strand": fields[6],
                }
            )

    for transcript_id, exons in exons_by_transcript.items():
        exons.sort(key=lambda exon: (exon["start"], exon["end"]))
    return exons_by_transcript


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def build_gap_transform(exons_by_transcript, selected_transcript_ids, max_gap=1200, display_gap=220):
    intervals = []
    for transcript_id in selected_transcript_ids:
        intervals.extend((exon["start"], exon["end"]) for exon in exons_by_transcript[transcript_id])
    merged = merge_intervals(intervals)
    if not merged:
        raise ValueError("No exons available for gap compression.")

    compressed_gaps = []
    for previous_interval, next_interval in zip(merged[:-1], merged[1:]):
        gap_start = previous_interval[1]
        gap_end = next_interval[0]
        gap_size = gap_end - gap_start
        if gap_size > max_gap:
            compressed_gaps.append((gap_start, gap_end, gap_size - display_gap))

    min_start = min(start for start, _ in intervals)

    def transform(position):
        reduction = 0
        for gap_start, gap_end, gap_reduction in compressed_gaps:
            if position >= gap_end:
                reduction += gap_reduction
            elif position > gap_start:
                reduction += max(0, position - gap_start - display_gap)
        return position - reduction - min_start

    return transform


def color_boxplot(boxplot_result, colors):
    for patch, color in zip(boxplot_result["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#4A4A4A")
        patch.set_linewidth(0.8)
    for key in ["whiskers", "caps", "medians"]:
        for artist in boxplot_result[key]:
            artist.set_color("#303030")
            artist.set_linewidth(0.8)


def plot_gene_expression(ax, summary_df, plot_kind="bar"):
    x_positions = np.arange(len(CONDITION_ORDER))
    plot_df = summary_df.set_index("Condition").loc[CONDITION_ORDER]
    colors = [CONDITION_COLORS[condition] for condition in CONDITION_ORDER]

    if plot_kind == "box":
        values_by_condition = plot_df["Values"].to_list()
        boxplot_result = ax.boxplot(
            values_by_condition,
            positions=x_positions,
            widths=0.58,
            patch_artist=True,
            showfliers=False,
        )
        color_boxplot(boxplot_result, colors)
        rng = np.random.default_rng(7)
        for x_position, values in zip(x_positions, values_by_condition):
            if len(values) == 0:
                continue
            jitter = rng.normal(0, 0.035, size=len(values))
            ax.scatter(
                np.full(len(values), x_position) + jitter,
                values,
                s=7,
                color="#303030",
                alpha=0.35,
                linewidths=0,
                zorder=3,
            )
        ax.set_ylabel("log2(TPM+1)")
    else:
        means = plot_df["Mean"].to_numpy()
        ci_values = plot_df["CI95"].to_numpy()
        ax.bar(
            x_positions,
            means,
            yerr=ci_values,
            color=colors,
            edgecolor="#4A4A4A",
            linewidth=0.6,
            capsize=2.5,
            error_kw={"elinewidth": 0.8, "capthick": 0.8, "ecolor": "#303030"},
            width=0.72,
        )
        ax.set_ylabel("Avg log2(TPM+1)")
    ax.set_title("gene\nexpression", fontweight="bold", pad=4)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["AR\npre", "IR\npre", "IR\npost", "AR\npost"])
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)
    style_panel_box(ax)


def plot_transcript_structure(ax, selected_tg, target_tg, class_map, exons_by_transcript):
    selected_transcript_ids = [split_transcript_gene(transcript_gene)[0] for transcript_gene in selected_tg]
    missing_exons = [
        transcript_id
        for transcript_id in selected_transcript_ids
        if not exons_by_transcript.get(transcript_id)
    ]
    if missing_exons:
        raise ValueError(f"GTF exon records were not found for: {', '.join(missing_exons)}")

    transform = build_gap_transform(exons_by_transcript, selected_transcript_ids)
    y_positions = np.arange(len(selected_tg))[::-1]
    exon_height = 0.46
    all_x = []
    all_coords = []
    all_chroms = []
    strands = []

    for y_position, transcript_gene in zip(y_positions, selected_tg):
        transcript_id, _ = split_transcript_gene(transcript_gene)
        transcript_class = class_map.get(transcript_gene, "Unclassified")
        exon_color = CLASS_COLORS.get(transcript_class, CLASS_COLORS["Unclassified"])
        exons = exons_by_transcript[transcript_id]
        strands.extend(exon["strand"] for exon in exons)
        all_chroms.extend(exon["chrom"] for exon in exons)

        exon_starts = [transform(exon["start"]) for exon in exons]
        exon_ends = [transform(exon["end"]) for exon in exons]
        all_coords.extend([coord for exon in exons for coord in (exon["start"], exon["end"])])
        line_start = min(exon_starts)
        line_end = max(exon_ends)
        all_x.extend([line_start, line_end])

        ax.hlines(
            y_position,
            line_start,
            line_end,
            color="#555555",
            linewidth=0.9,
            zorder=1,
        )

        for exon in exons:
            x_start = transform(exon["start"])
            x_end = transform(exon["end"])
            width = max(x_end - x_start, 12)
            is_target = transcript_gene == target_tg
            exon_patch = patches.Rectangle(
                (x_start, y_position - exon_height / 2),
                width,
                exon_height,
                facecolor=exon_color,
                edgecolor="#262626" if is_target else "#333333",
                linewidth=0.75 if is_target else 0.55,
                hatch="///" if is_target else None,
                zorder=2,
            )
            ax.add_patch(exon_patch)
            all_x.extend([x_start, x_start + width])

    if all_x:
        x_min, x_max = min(all_x), max(all_x)
        pad = max((x_max - x_min) * 0.035, 30)
        ax.set_xlim(x_min - pad, x_max + pad)

    if strands and strands.count("-") > strands.count("+"):
        ax.invert_xaxis()

    ax.set_ylim(-0.75, len(selected_tg) - 0.25)
    ax.set_yticks([])
    if all_coords:
        coord_min, coord_max = min(all_coords), max(all_coords)
        tick_fracs = np.array([0.18, 0.50, 0.82]) if coord_max > coord_min else np.array([0.5])
        coord_ticks = (coord_min + (coord_max - coord_min) * tick_fracs) / 1e6
        display_ticks = min(all_x) + (max(all_x) - min(all_x)) * tick_fracs
        ax.set_xticks(display_ticks)
        ax.set_xticklabels([f"{coord:.2f}" for coord in coord_ticks], fontsize=6)
        ax.tick_params(axis="x", length=2, pad=2)
        chrom_label = ",".join(sorted(set(all_chroms)))
        ax.set_xlabel(f"Genomic position on {chrom_label} (Mb, approx.)", fontsize=7, labelpad=2)
    ax.set_title("transcript structure", fontweight="bold", pad=4)
    legend_handles = [
        patches.Patch(facecolor=CLASS_COLORS["Class1"], edgecolor="#333333", label="Class1"),
        patches.Patch(facecolor=CLASS_COLORS["Class2"], edgecolor="#333333", label="Class2"),
        patches.Patch(facecolor=CLASS_COLORS["Class3"], edgecolor="#333333", label="Class3"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.9,
        borderaxespad=0.0,
        fontsize=7,
    )
    style_panel_box(ax)


def plot_isoform_usage(ax, summary_df, gene, transcript_id, plot_kind="bar"):
    plot_df = summary_df.set_index("Condition").loc[CONDITION_ORDER].reset_index()
    y_positions = np.arange(len(CONDITION_ORDER))
    colors = [CONDITION_COLORS[condition] for condition in CONDITION_ORDER]

    if plot_kind == "box":
        values_by_condition = plot_df["Values"].to_list()
        boxplot_result = ax.boxplot(
            values_by_condition,
            positions=y_positions,
            vert=False,
            widths=0.58,
            patch_artist=True,
            showfliers=False,
        )
        color_boxplot(boxplot_result, colors)
        rng = np.random.default_rng(13)
        for y_position, values in zip(y_positions, values_by_condition):
            if len(values) == 0:
                continue
            jitter = rng.normal(0, 0.035, size=len(values))
            ax.scatter(
                values,
                np.full(len(values), y_position) + jitter,
                s=7,
                color="#303030",
                alpha=0.35,
                linewidths=0,
                zorder=3,
            )
        non_empty_values = [values for values in values_by_condition if len(values)]
        all_values = np.concatenate(non_empty_values) if non_empty_values else np.array([])
        max_value = float(np.nanmax(all_values)) if all_values.size else 0.0
    else:
        means = plot_df["Mean"].to_numpy()
        ci_values = plot_df["CI95"].to_numpy()
        ax.barh(
            y_positions,
            means,
            xerr=ci_values,
            color=colors,
            edgecolor="#4A4A4A",
            linewidth=0.6,
            capsize=2.5,
            error_kw={"elinewidth": 0.8, "capthick": 0.8, "ecolor": "#303030"},
            height=0.62,
        )
        max_value = float(np.nanmax(means + ci_values)) if means.size else 0.0
    ax.set_yticks(y_positions)
    ax.set_yticklabels(CONDITION_ORDER)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.invert_yaxis()
    ax.set_title(f"{gene}\n{transcript_id}", fontweight="bold", pad=4)
    ax.set_xlabel("TU" if plot_kind == "box" else "Average TU")
    ax.grid(axis="x", alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)

    ax.set_xlim(0, max(max_value * 1.18, 0.05))
    style_panel_box(ax)


def make_three_panel_figure(
    target,
    selected_tg,
    class_map,
    exons_by_transcript,
    gene_expression_summary,
    isoform_usage_summary,
    plot_kind="bar",
):
    gene = target["gene"]
    transcript_id = target["transcript_id"]
    target_tg = f"{transcript_id}-{gene}"

    fig = plt.figure(figsize=(9, 3.5))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[0.95, 2.6, 1.25],
        left=0.07,
        right=0.88,
        bottom=0.23,
        top=0.86,
        wspace=0.08,
    )
    ax_gene = fig.add_subplot(grid[0, 0])
    ax_structure = fig.add_subplot(grid[0, 1])
    ax_usage = fig.add_subplot(grid[0, 2])

    plot_gene_expression(ax_gene, gene_expression_summary, plot_kind=plot_kind)
    plot_transcript_structure(ax_structure, selected_tg, target_tg, class_map, exons_by_transcript)
    plot_isoform_usage(ax_usage, isoform_usage_summary, gene, transcript_id, plot_kind=plot_kind)

    suffix = "_boxplot_threepanel" if plot_kind == "box" else "_threepanel"
    pdf_path = OUT_DIR / f"{gene}_{transcript_id}{suffix}.pdf"
    png_path = OUT_DIR / f"{gene}_{transcript_id}{suffix}.png"
    fig.savefig(pdf_path, dpi=300)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def main():
    configure_plot_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_inputs()
    class_map = build_class_map(data["sqanti"], data["majorminor"])
    condition_samples = build_condition_samples(data["sampleinfo"], data["filtered_trans"].columns)

    selected_by_target = {}
    summary_frames = []
    for target in TARGETS:
        selected_tg, summary_df = select_transcripts_for_target(
            target, data["filtered_trans"], class_map
        )
        selected_by_target[target["transcript_id"]] = selected_tg
        summary_frames.append(summary_df)

    selected_summary = pd.concat(summary_frames, ignore_index=True)
    selected_summary.to_csv(OUT_DIR / "selected_transcripts_summary.tsv", sep="\t", index=False)

    selected_transcript_ids = set(selected_summary["selected_transcript_id"])
    exons_by_transcript = parse_gtf_exons(GTF_PATH, selected_transcript_ids)
    missing_exon_transcripts = [
        transcript_id
        for transcript_id in sorted(selected_transcript_ids)
        if not exons_by_transcript.get(transcript_id)
    ]
    if missing_exon_transcripts:
        raise ValueError(
            "GTF exon records were not found for selected transcripts: "
            + ", ".join(missing_exon_transcripts)
        )

    saved_paths = []
    for target in TARGETS:
        gene = target["gene"]
        transcript_id = target["transcript_id"]
        transcript_gene = f"{transcript_id}-{gene}"

        gene_expression_summary = summarize_values_by_condition(
            data["geneexp"],
            gene,
            build_condition_samples(data["sampleinfo"], data["geneexp"].columns),
            transform=lambda values: np.log2(values.astype(float) + 1),
        )
        isoform_usage_summary = summarize_values_by_condition(
            data["filtered_trans"],
            transcript_gene,
            condition_samples,
            transform=None,
        )

        for plot_kind in ["bar", "box"]:
            pdf_path, png_path = make_three_panel_figure(
                target=target,
                selected_tg=selected_by_target[transcript_id],
                class_map=class_map,
                exons_by_transcript=exons_by_transcript,
                gene_expression_summary=gene_expression_summary,
                isoform_usage_summary=isoform_usage_summary,
                plot_kind=plot_kind,
            )
            saved_paths.append((pdf_path, png_path))
            print(f"saved {pdf_path}")
            print(f"saved {png_path}")

    print(f"saved {OUT_DIR / 'selected_transcripts_summary.tsv'}")
    print(f"generated {len(saved_paths)} target DUT three-panel figure variants")


if __name__ == "__main__":
    main()
