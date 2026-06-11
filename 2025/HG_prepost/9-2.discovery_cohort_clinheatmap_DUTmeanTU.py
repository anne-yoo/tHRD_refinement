import importlib.util
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import seaborn as sns


BASE_SCRIPT = Path(__file__).with_name("9.discovery_cohort_clinheatmap.py")
OUTPUT_PATH = "/home/jiye/jiye/copycomparison/GENCODEquant/figures/discovery_cohort_clinheatmap_with_AR_DUT_meanTU.pdf"
OUTPUT_PNG_PATH = "/home/jiye/jiye/copycomparison/GENCODEquant/figures/discovery_cohort_clinheatmap_with_AR_DUT_meanTU.png"
OUTPUT_TABLE_PATH = "/home/jiye/jiye/copycomparison/GENCODEquant/figures/discovery_cohort_clinheatmap_with_AR_DUT_meanTU_values.tsv"

SQANTI_PATH = "/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/sqantioutput/sqanti_hg19_classification.txt"
AR_DUT_PATH = "/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/whole_AR_stable_DUT_Wilcoxon_delta_withna.txt"
TRANSCRIPT_TPM_PATH = "/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_80_transcript_TPM.txt"
MAJORMINOR_PATH = "/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_80_majorminorlist.txt"


def load_base_module():
    spec = importlib.util.spec_from_file_location("discovery_clinheatmap_base", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


def configure_extra_fonts():
    for arial_font_path in [
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/arialbd.ttf",
    ]:
        if Path(arial_font_path).exists():
            font_manager.fontManager.addfont(arial_font_path)
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def draw_group_header(ax, x_positions, ar_count, ir_count):
    segments = [
        ("AR", 0, ar_count, base.AR_COLOR),
        ("IR", ar_count, ir_count, base.IR_COLOR),
    ]
    for label, start_idx, count, color in segments:
        segment_x = x_positions[start_idx : start_idx + count]
        left = min(segment_x) - 0.42
        right = max(segment_x) + 0.42
        center = (left + right) / 2

        ax.add_patch(
            Rectangle(
                (left, 4.62),
                right - left,
                0.22,
                facecolor=color,
                edgecolor="none",
                linewidth=0,
            )
        )
        ax.text(center, 5.02, label, ha="center", va="center", fontsize=9, fontweight="bold")


def get_class1_class3_dut_sets():
    sqanti = pd.read_csv(SQANTI_PATH, sep="\t")
    sqanti.dropna(axis=1, how="all", inplace=True)
    majorminor = pd.read_csv(MAJORMINOR_PATH, sep="\t")
    ar_dut = pd.read_csv(AR_DUT_PATH, sep="\t", index_col=0)

    df_cat = sqanti[["isoform", "structural_category", "subcategory", "within_CAGE_peak", "coding"]].copy()
    df_cat["isoform_clean"] = df_cat["isoform"].astype(str).str.split("-", n=1).str[0]

    majorlist = majorminor.loc[majorminor["type"] == "major", "Transcript-Gene"].tolist()
    majorlist_set = {x.split("-", 1)[0] for x in majorlist}
    coding_set = set(sqanti.loc[sqanti["coding"] == "coding", "isoform"])
    majorlist_set = majorlist_set.intersection(coding_set)

    valid_cat = {"full-splice_match", "novel_in_catalog"}
    df_cat["major"] = df_cat["isoform"].isin(majorlist_set)
    df_cat["major"] = df_cat["major"] & df_cat["structural_category"].isin(valid_cat)

    group1 = df_cat.loc[df_cat["major"] == True, "isoform_clean"].tolist()
    group3 = df_cat.loc[
        (df_cat["major"] == False) & (df_cat["coding"] == "non_coding"),
        "isoform_clean",
    ].tolist()

    class1 = majorminor.loc[majorminor["transcriptid"].isin(group1), "Transcript-Gene"].tolist()
    class3 = majorminor.loc[majorminor["transcriptid"].isin(group3), "Transcript-Gene"].tolist()

    ar_up_class1 = sorted(
        set(ar_dut.loc[(ar_dut["p_value"] < 0.05) & (ar_dut["delta_TU"] > 0.05)].index).intersection(class1)
    )
    ar_down_class3 = sorted(
        set(ar_dut.loc[(ar_dut["p_value"] < 0.05) & (ar_dut["delta_TU"] < -0.05)].index).intersection(class3)
    )
    return ar_up_class1, ar_down_class3


def load_transcript_usage():
    transexp = pd.read_csv(TRANSCRIPT_TPM_PATH, sep="\t", index_col=0)
    transexp = transexp.iloc[:, :-1]
    transexp = transexp.loc[(transexp > 0).sum(axis=1) >= 8]
    transexp["gene"] = transexp.index.str.split("-", n=1).str[-1]
    gene_sum = transexp.groupby("gene").transform("sum")
    return transexp.iloc[:, :-1].div(gene_sum)


def count_unique_genes(transcript_list):
    return (
        pd.Series(transcript_list, dtype="object")
        .dropna()
        .astype(str)
        .str.split("-", n=1)
        .str[-1]
        .nunique()
    )


def compute_pre_mean_tu_by_sample(filtered_trans, transcript_list, records):
    valid_tx = sorted(set(transcript_list).intersection(filtered_trans.index))
    mean_by_sample = {}
    for record in records:
        pre_col = f"{record['sample_id']}-bfD"
        if pre_col not in filtered_trans.columns or len(valid_tx) == 0:
            mean_by_sample[record["sample_id"]] = np.nan
            continue
        mean_by_sample[record["sample_id"]] = filtered_trans.loc[valid_tx, pre_col].mean(skipna=True)
    return mean_by_sample


def make_vlag_colors(values, cmap, norm):
    colors = []
    for value in values:
        if pd.isna(value):
            colors.append("#BDBDBD")
        else:
            colors.append(cmap(norm(value)))
    return colors


def plot_clinical_heatmap_with_dut_rows(records, output_path, output_png_path):
    sorted_records, ar_records, ir_records = base.sort_records(records)
    ar_count = len(ar_records)
    ir_count = len(ir_records)
    x_positions = [float(idx) for idx in range(len(sorted_records))]

    ar_up_class1, ar_down_class3 = get_class1_class3_dut_sets()
    filtered_trans = load_transcript_usage()
    class1_mean_by_sample = compute_pre_mean_tu_by_sample(filtered_trans, ar_up_class1, sorted_records)
    class3_mean_by_sample = compute_pre_mean_tu_by_sample(filtered_trans, ar_down_class3, sorted_records)

    interval_values = [record["interval"] for record in sorted_records]
    interval_cmap = base.LinearSegmentedColormap.from_list("interval_greys", ["#F4F4F4", "#111111"])
    interval_norm = Normalize(vmin=min(interval_values), vmax=max(interval_values))

    class1_values = [class1_mean_by_sample[record["sample_id"]] for record in sorted_records]
    class3_values = [class3_mean_by_sample[record["sample_id"]] for record in sorted_records]
    dut_values = pd.Series(class1_values + class3_values, dtype="float64").dropna()
    dut_norm = Normalize(vmin=float(dut_values.min()), vmax=float(dut_values.max()))
    dut_cmap = base.LinearSegmentedColormap.from_list("dut_mean_tu_reds", ["#FFFFFF", "#7F0000"])

    brca_colors = [base.BRCA_COLORS.get(record["brca"], "#BDBDBD") for record in sorted_records]
    line_colors = [base.LINE_COLORS[record["line_group"]] for record in sorted_records]
    drug_colors = [base.DRUG_COLORS.get(record["drug"], "#999999") for record in sorted_records]
    purpose_colors = [base.PURPOSE_COLORS.get(record["purpose"], "#999999") for record in sorted_records]
    interval_colors = [interval_cmap(interval_norm(record["interval"])) for record in sorted_records]
    class1_colors = make_vlag_colors(class1_values, dut_cmap, dut_norm)
    class3_colors = make_vlag_colors(class3_values, dut_cmap, dut_norm)

    max_x = max(x_positions)
    legend_x = max_x + 1.5
    fig, ax = plt.subplots(figsize=(9.8, 3.85))
    ax.set_xlim(-5.55, max_x + 8.35)
    ax.set_ylim(-0.05, 5.30)
    ax.axis("off")

    draw_group_header(ax, x_positions, ar_count, ir_count)

    row_y = {
        "BRCAmt": 4.05,
        "Line": 3.45,
        "Drug": 2.85,
        "Purpose": 2.25,
        "Interval": 1.65,
        "Class1": 1.05,
        "Class3": 0.45,
    }
    label_x = -0.95
    for label, y_pos in row_y.items():
        ax.text(label_x, y_pos, label, ha="right", va="center", fontsize=8.2, linespacing=0.88)

    base.draw_cell_row(ax, x_positions, row_y["BRCAmt"], brca_colors)
    base.draw_cell_row(ax, x_positions, row_y["Line"], line_colors)
    base.draw_cell_row(ax, x_positions, row_y["Drug"], drug_colors)
    base.draw_cell_row(ax, x_positions, row_y["Purpose"], purpose_colors)
    base.draw_cell_row(ax, x_positions, row_y["Interval"], interval_colors)
    base.draw_cell_row(ax, x_positions, row_y["Class1"], class1_colors)
    base.draw_cell_row(ax, x_positions, row_y["Class3"], class3_colors)

    legend_y = base.add_discrete_legend(
        ax,
        "Line",
        [("1L", base.LINE_COLORS["1L"]), (">=2L", base.LINE_COLORS[">=2L"])],
        legend_x,
        4.88,
    )
    present_drugs = [
        drug for drug in ["Olaparib", "Niraparib", "Rucaparib"]
        if any(record["drug"] == drug for record in sorted_records)
    ]
    legend_y = base.add_discrete_legend(
        ax,
        "Drug",
        [(drug, base.DRUG_COLORS[drug]) for drug in present_drugs],
        legend_x,
        legend_y - 0.10,
    )
    legend_y = base.add_discrete_legend(
        ax,
        "Purpose",
        [("Maintenance", base.PURPOSE_COLORS["maintenance"]), ("Salvage", base.PURPOSE_COLORS["salvage"])],
        legend_x,
        legend_y - 0.10,
    )
    base.add_discrete_legend(
        ax,
        "BRCAmt",
        [("Mutated", base.BRCA_COLORS["1"]), ("Wild-type", base.BRCA_COLORS["0"])],
        legend_x,
        legend_y - 0.10,
    )

    interval_cax = fig.add_axes([0.875, 0.29, 0.015, 0.18])
    interval_sm = plt.cm.ScalarMappable(cmap=interval_cmap, norm=interval_norm)
    interval_sm.set_array([])
    interval_cbar = fig.colorbar(interval_sm, cax=interval_cax)
    interval_cbar.set_label("Interval (days)", fontsize=8)
    interval_cbar.ax.tick_params(labelsize=7, width=0.5, length=2.5)
    interval_cbar.outline.set_linewidth(0.5)

    dut_cax = fig.add_axes([0.875, 0.09, 0.015, 0.16])
    dut_sm = plt.cm.ScalarMappable(cmap=dut_cmap, norm=dut_norm)
    dut_sm.set_array([])
    dut_cbar = fig.colorbar(dut_sm, cax=dut_cax)
    dut_cbar.set_label("Mean TU", fontsize=8)
    dut_cbar.ax.tick_params(labelsize=7, width=0.5, length=2.5)
    dut_cbar.outline.set_linewidth(0.5)

    value_df = pd.DataFrame(
        {
            "sample_id": [record["sample_id"] for record in sorted_records],
            "group": [record["group"] for record in sorted_records],
            "BRCAmt": [record["brca"] for record in sorted_records],
            "line_group": [record["line_group"] for record in sorted_records],
            "AR_up_Class1_mean_TU": class1_values,
            "AR_down_Class3_mean_TU": class3_values,
        }
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    value_df.to_csv(OUTPUT_TABLE_PATH, sep="\t", index=False)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "ar_count": ar_count,
        "ir_count": ir_count,
        "ar_up_class1_tx_count": len(ar_up_class1),
        "ar_up_class1_gene_count": count_unique_genes(ar_up_class1),
        "ar_down_class3_tx_count": len(ar_down_class3),
        "ar_down_class3_gene_count": count_unique_genes(ar_down_class3),
        "mean_tu_vmin": float(dut_values.min()),
        "mean_tu_vmax": float(dut_values.max()),
        "output_path": output_path,
        "output_png_path": output_png_path,
        "value_table_path": OUTPUT_TABLE_PATH,
    }


def main():
    font_name = base.configure_fonts()
    configure_extra_fonts()
    records = base.load_clinical_rows(base.INPUT_PATH)
    summary = plot_clinical_heatmap_with_dut_rows(records, OUTPUT_PATH, OUTPUT_PNG_PATH)
    print(f"font={font_name}")
    print(f"AR={summary['ar_count']}, IR={summary['ir_count']}")
    print(
        "AR upregulated Class1 DUT:",
        f"{summary['ar_up_class1_tx_count']} transcripts,",
        f"{summary['ar_up_class1_gene_count']} genes",
    )
    print(
        "AR downregulated Class3 DUT:",
        f"{summary['ar_down_class3_tx_count']} transcripts,",
        f"{summary['ar_down_class3_gene_count']} genes",
    )
    print(f"mean TU color range={summary['mean_tu_vmin']:.4f} to {summary['mean_tu_vmax']:.4f}")
    print(f"output={summary['output_path']}")
    print(f"png={summary['output_png_path']}")
    print(f"values={summary['value_table_path']}")


if __name__ == "__main__":
    main()
