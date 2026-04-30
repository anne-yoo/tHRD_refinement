import csv
import os
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle


INPUT_PATH = "/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/112_PARPi_clinicalinfo.txt"
EXPRESSION_HEADER_PATH = "/home/jiye/jiye/copycomparison/GENCODEquant/SEV_pre/111_pre/forval_111_gene_TPM.txt"
OUTPUT_PATH = "/home/jiye/jiye/copycomparison/GENCODEquant/figures/validation_cohort_clinheatmap.pdf"
GHRD_THRESHOLD = 42.0
ARIAL_FONT_FILES = [
    "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold_Italic.ttf",
]

AR_COLOR = "#FEB24C"
IR_COLOR = "#5AAE61"
CR_COLOR = "#58C1EE"
GROUP_COLORS = {
    "AR": AR_COLOR,
    "IR": IR_COLOR,
    "CR": CR_COLOR,
}
BRCA_COLORS = {
    "1": "#D73027",
    "0": "#E6E6E6",
}
GHRD_COLORS = {
    "high": "#6A51A3",
    "low": "#DCD6F7",
    "na": "#F2F2F2",
}
DRUG_COLORS = {
    "Olaparib": "#F26786",
    "Niraparib": "#38B7AD",
    "Rucaparib": "#8E3A8C",
}


def configure_fonts():
    registered_arial = False
    for font_path in ARIAL_FONT_FILES:
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)
            registered_arial = True

    if registered_arial:
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["font.sans-serif"] = ["Arial"]
        font_name = "Arial"
    else:
        plt.rcParams["font.family"] = "sans-serif"
        font_name = "matplotlib default sans-serif"

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    return font_name


def parse_line_num(line_value):
    match = re.search(r"(\d+)", str(line_value))
    if not match:
        raise ValueError(f"Could not parse treatment line from value: {line_value!r}")
    return int(match.group(1))


def parse_float(value, field):
    value = str(value).strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Could not parse {field} from value: {value!r}") from exc


def ghrd_group(score):
    if score is None:
        return "na"
    if score >= GHRD_THRESHOLD:
        return "high"
    return "low"


def response_group(response, recur):
    if response == "1" and recur == "1.0":
        return "AR"
    if response == "0":
        return "IR"
    if response == "1" and recur == "0.0":
        return "CR"
    return "i"


def load_expression_samples(path):
    with open(path) as handle:
        columns = handle.readline().rstrip("\n").split("\t")[1:]
    if columns and columns[-1] == "gene_name":
        columns = columns[:-1]
    return set(columns)


def load_clinical_rows(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    expression_samples = load_expression_samples(EXPRESSION_HEADER_PATH)
    rows = [row for row in rows if row["sample_id"] in expression_samples]

    records = []
    for order, row in enumerate(rows):
        response = str(row["response"]).strip()
        if response not in {"0", "1"}:
            raise ValueError(f"Unexpected response value: {response!r}")
        recur = str(row["recur"]).strip()
        group = response_group(response, recur)
        if group == "i":
            raise ValueError(
                f"Unexpected response/recur combination for {row['sample_id']!r}: "
                f"response={response!r}, recur={recur!r}"
            )

        ghrd_score = parse_float(row["gHRDscore"], "gHRDscore")
        pfs = parse_float(row["PFS"], "PFS")
        if pfs is None:
            raise ValueError(f"Missing PFS value for sample: {row['sample_id']!r}")

        records.append(
            {
                "sample_id": row["sample_id"],
                "response": response,
                "recur": recur,
                "group": group,
                "line_num": parse_line_num(row["line"]),
                "drug": row["drug"],
                "brca": str(row["BRCAmt"]).strip(),
                "ghrd_score": ghrd_score,
                "ghrd_group": ghrd_group(ghrd_score),
                "pfs": pfs,
                "order": order,
            }
        )

    return records


def sort_records(records):
    group_order = ["AR", "IR", "CR"]
    grouped_records = {group: [record for record in records if record["group"] == group] for group in group_order}
    sorted_records = []
    for group in group_order:
        sorted_records.extend(grouped_records[group])
    return sorted_records, grouped_records


def draw_cell_row(ax, x_positions, y_center, colors, cell_width=0.78, cell_height=0.38):
    for x_pos, color in zip(x_positions, colors):
        ax.add_patch(
            Rectangle(
                (x_pos - cell_width / 2, y_center - cell_height / 2),
                cell_width,
                cell_height,
                facecolor=color,
                edgecolor="white",
                linewidth=0.30,
            )
        )


def draw_group_header(ax, x_positions, group_counts):
    start_idx = 0
    for label in ["AR", "IR", "CR"]:
        count = group_counts[label]
        color = GROUP_COLORS[label]
        if count == 0:
            continue
        segment_x = x_positions[start_idx : start_idx + count]
        left = min(segment_x) - 0.39
        right = max(segment_x) + 0.39
        center = (left + right) / 2

        ax.add_patch(
            Rectangle(
                (left, 4.34),
                right - left,
                0.22,
                facecolor=color,
                edgecolor="none",
                linewidth=0,
            )
        )
        ax.text(center, 4.80, label, ha="center", va="center", fontsize=9, fontweight="bold")
        start_idx += count


def add_discrete_legend(ax, title, labels_and_colors, x, y, box_size=0.22, line_height=0.32):
    ax.text(x, y, title, ha="left", va="center", fontsize=8.5)
    y -= line_height
    for label, color in labels_and_colors:
        ax.add_patch(
            Rectangle(
                (x, y - box_size / 2),
                box_size,
                box_size,
                facecolor=color,
                edgecolor="#666666",
                linewidth=0.30,
            )
        )
        ax.text(x + 0.42, y, label, ha="left", va="center", fontsize=8)
        y -= line_height
    return y


def add_line_legend(ax, cmap, norm, min_line, max_line, x, y):
    ax.text(x, y, "Line", ha="left", va="center", fontsize=8.5)
    y -= 0.32
    box_gap = 0.29
    for idx, line_num in enumerate(range(min_line, max_line + 1)):
        color = cmap(norm(line_num))
        ax.add_patch(
            Rectangle(
                (x + idx * box_gap, y - 0.10),
                0.21,
                0.21,
                facecolor=color,
                edgecolor="#666666",
                linewidth=0.25,
            )
        )
    left_label_x = x - 0.14
    right_label_x = x + (max_line - min_line) * box_gap + 0.36
    ax.text(left_label_x, y - 0.32, f"{min_line}L", ha="left", va="center", fontsize=7.5)
    ax.text(
        right_label_x,
        y - 0.32,
        f"{max_line}L",
        ha="right",
        va="center",
        fontsize=7.5,
    )
    return y - 0.68


def plot_clinical_heatmap(records, output_path):
    sorted_records, grouped_records = sort_records(records)
    group_counts = {group: len(grouped_records[group]) for group in ["AR", "IR", "CR"]}

    x_positions = [float(idx) for idx in range(len(sorted_records))]

    line_values = [record["line_num"] for record in sorted_records]
    pfs_values = [record["pfs"] for record in sorted_records]
    min_line = min(line_values)
    max_line = max(line_values)
    min_pfs = min(pfs_values)
    max_pfs = max(pfs_values)

    line_cmap = LinearSegmentedColormap.from_list("line_navy", ["#D9E8F5", "#08306B"])
    pfs_cmap = LinearSegmentedColormap.from_list("pfs_greys", ["#F4F4F4", "#111111"])
    line_norm = Normalize(vmin=min_line, vmax=max_line)
    pfs_norm = Normalize(vmin=min_pfs, vmax=max_pfs)

    line_colors = [line_cmap(line_norm(record["line_num"])) for record in sorted_records]
    brca_colors = [BRCA_COLORS.get(record["brca"], "#BDBDBD") for record in sorted_records]
    ghrd_colors = [GHRD_COLORS[record["ghrd_group"]] for record in sorted_records]
    drug_colors = [DRUG_COLORS.get(record["drug"], "#999999") for record in sorted_records]
    pfs_colors = [pfs_cmap(pfs_norm(record["pfs"])) for record in sorted_records]

    max_x = max(x_positions)
    legend_x = max_x + 2.0
    fig_width = max(12.0, len(sorted_records) * 0.105 + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, 3.05))
    ax.set_xlim(-5.00, max_x + 9.4)
    ax.set_ylim(-0.05, 5.10)
    ax.axis("off")

    draw_group_header(ax, x_positions, group_counts)

    row_y = {
        "Line": 3.72,
        "BRCAmt": 3.08,
        "gHRD": 2.44,
        "Drug": 1.80,
        "PFS": 1.16,
    }
    label_x = -0.95
    for label, y_pos in row_y.items():
        ax.text(label_x, y_pos, label, ha="right", va="center", fontsize=9)

    draw_cell_row(ax, x_positions, row_y["Line"], line_colors)
    draw_cell_row(ax, x_positions, row_y["BRCAmt"], brca_colors)
    draw_cell_row(ax, x_positions, row_y["gHRD"], ghrd_colors)
    draw_cell_row(ax, x_positions, row_y["Drug"], drug_colors)
    draw_cell_row(ax, x_positions, row_y["PFS"], pfs_colors)

    y_after_line = add_line_legend(ax, line_cmap, line_norm, min_line, max_line, legend_x, 4.62)
    present_drugs = [drug for drug in ["Olaparib", "Niraparib", "Rucaparib"] if any(r["drug"] == drug for r in sorted_records)]
    add_discrete_legend(
        ax,
        "Drug",
        [(drug, DRUG_COLORS[drug]) for drug in present_drugs],
        legend_x,
        y_after_line,
    )
    add_discrete_legend(
        ax,
        "BRCAmt",
        [("Mutated", BRCA_COLORS["1"]), ("Wild-type", BRCA_COLORS["0"])],
        legend_x,
        2.70,
    )
    add_discrete_legend(
        ax,
        "gHRD",
        [
            (">=42", GHRD_COLORS["high"]),
            ("<42", GHRD_COLORS["low"]),
            ("NA", GHRD_COLORS["na"]),
        ],
        legend_x,
        1.76,
    )

    cax = fig.add_axes([0.895, 0.22, 0.010, 0.26])
    sm = plt.cm.ScalarMappable(cmap=pfs_cmap, norm=pfs_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("PFS (days)", fontsize=8)
    cbar.ax.tick_params(labelsize=7, width=0.5, length=2.5)
    cbar.outline.set_linewidth(0.5)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    ghrd_counts = {
        "high": sum(record["ghrd_group"] == "high" for record in sorted_records),
        "low": sum(record["ghrd_group"] == "low" for record in sorted_records),
        "na": sum(record["ghrd_group"] == "na" for record in sorted_records),
    }
    return {
        "group_counts": group_counts,
        "min_line": min_line,
        "max_line": max_line,
        "ghrd_counts": ghrd_counts,
        "output_path": output_path,
    }


def main():
    font_name = configure_fonts()
    records = load_clinical_rows(INPUT_PATH)
    summary = plot_clinical_heatmap(records, OUTPUT_PATH)
    print(f"font={font_name}")
    print(
        "groups: "
        f"AR={summary['group_counts']['AR']}, "
        f"IR={summary['group_counts']['IR']}, "
        f"CR={summary['group_counts']['CR']}"
    )
    print(f"line range={summary['min_line']}-{summary['max_line']}")
    print(
        "gHRD threshold=42: "
        f">=42={summary['ghrd_counts']['high']}, "
        f"<42={summary['ghrd_counts']['low']}, "
        f"NA={summary['ghrd_counts']['na']}"
    )
    print(f"output={summary['output_path']}")


if __name__ == "__main__":
    main()
