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
OUTPUT_PATH_AR_IR = "/home/jiye/jiye/copycomparison/GENCODEquant/figures/validation_cohort_clinheatmap_AR_IR.pdf"
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
LINE_COLORS = {
    "1L": "#2B6CB0",
    ">=2L": "#A6CEE3",
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
PURPOSE_COLORS = {
    "maintenance": "#6BAED6",
    "salvage": "#E6AB02",
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


def line_group(line_num):
    if line_num == 1:
        return "1L"
    return ">=2L"


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

        parsed_line = parse_line_num(row["line"])
        records.append(
            {
                "sample_id": row["sample_id"],
                "response": response,
                "recur": recur,
                "group": group,
                "line_num": parsed_line,
                "line_group": line_group(parsed_line),
                "drug": row["drug"],
                "purpose": row["setting"],
                "brca": str(row["BRCAmt"]).strip(),
                "ghrd_score": ghrd_score,
                "ghrd_group": ghrd_group(ghrd_score),
                "pfs": pfs,
                "order": order,
            }
        )

    return records


def sort_records(records, group_order):
    grouped_records = {}
    for group in group_order:
        group_records = [record for record in records if record["group"] == group]
        grouped_records[group] = sorted(
            group_records,
            key=lambda record: (
                0 if record["brca"] == "1" else 1,
                record["order"],
            ),
        )
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


def draw_group_header(ax, x_positions, group_counts, group_order):
    start_idx = 0
    for label in group_order:
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
                (left, 4.90),
                right - left,
                0.22,
                facecolor=color,
                edgecolor="none",
                linewidth=0,
            )
        )
        ax.text(center, 5.36, label, ha="center", va="center", fontsize=9, fontweight="bold")
        start_idx += count


def add_discrete_legend(ax, title, labels_and_colors, x, y, box_size=0.19, line_height=0.28):
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


def plot_clinical_heatmap(records, output_path, group_order):
    sorted_records, grouped_records = sort_records(records, group_order)
    group_counts = {group: len(grouped_records[group]) for group in group_order}

    x_positions = [float(idx) for idx in range(len(sorted_records))]

    pfs_values = [record["pfs"] for record in sorted_records]
    min_pfs = min(pfs_values)
    max_pfs = max(pfs_values)

    pfs_cmap = LinearSegmentedColormap.from_list("pfs_greys", ["#F4F4F4", "#111111"])
    pfs_norm = Normalize(vmin=min_pfs, vmax=max_pfs)

    brca_colors = [BRCA_COLORS.get(record["brca"], "#BDBDBD") for record in sorted_records]
    line_colors = [LINE_COLORS[record["line_group"]] for record in sorted_records]
    ghrd_colors = [GHRD_COLORS[record["ghrd_group"]] for record in sorted_records]
    drug_colors = [DRUG_COLORS.get(record["drug"], "#999999") for record in sorted_records]
    purpose_colors = [PURPOSE_COLORS.get(record["purpose"], "#999999") for record in sorted_records]
    pfs_colors = [pfs_cmap(pfs_norm(record["pfs"])) for record in sorted_records]

    max_x = max(x_positions)
    legend_x = max_x + 2.0
    fig_width = max(12.0, len(sorted_records) * 0.105 + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, 3.35))
    ax.set_xlim(-5.00, max_x + 9.4)
    ax.set_ylim(-0.05, 5.70)
    ax.axis("off")

    draw_group_header(ax, x_positions, group_counts, group_order)

    row_y = {
        "BRCAmt": 4.30,
        "Line": 3.70,
        "gHRD": 3.10,
        "Drug": 2.50,
        "Purpose": 1.90,
        "PFS": 1.30,
    }
    label_x = -0.95
    for label, y_pos in row_y.items():
        ax.text(label_x, y_pos, label, ha="right", va="center", fontsize=9)

    draw_cell_row(ax, x_positions, row_y["BRCAmt"], brca_colors)
    draw_cell_row(ax, x_positions, row_y["Line"], line_colors)
    draw_cell_row(ax, x_positions, row_y["gHRD"], ghrd_colors)
    draw_cell_row(ax, x_positions, row_y["Drug"], drug_colors)
    draw_cell_row(ax, x_positions, row_y["Purpose"], purpose_colors)
    draw_cell_row(ax, x_positions, row_y["PFS"], pfs_colors)

    legend_y = add_discrete_legend(
        ax,
        "Line",
        [("1L", LINE_COLORS["1L"]), (">=2L", LINE_COLORS[">=2L"])],
        legend_x,
        5.20,
    )
    present_drugs = [drug for drug in ["Olaparib", "Niraparib", "Rucaparib"] if any(r["drug"] == drug for r in sorted_records)]
    legend_y = add_discrete_legend(
        ax,
        "Drug",
        [(drug, DRUG_COLORS[drug]) for drug in present_drugs],
        legend_x,
        legend_y - 0.10,
    )
    legend_y = add_discrete_legend(
        ax,
        "Purpose",
        [("Maintenance", PURPOSE_COLORS["maintenance"]), ("Salvage", PURPOSE_COLORS["salvage"])],
        legend_x,
        legend_y - 0.10,
    )
    legend_y = add_discrete_legend(
        ax,
        "BRCAmt",
        [("Mutated", BRCA_COLORS["1"]), ("Wild-type", BRCA_COLORS["0"])],
        legend_x,
        legend_y - 0.10,
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
        legend_y - 0.10,
    )

    cax = fig.add_axes([0.895, 0.18, 0.010, 0.24])
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
        "line_counts": {
            "1L": sum(record["line_group"] == "1L" for record in sorted_records),
            ">=2L": sum(record["line_group"] == ">=2L" for record in sorted_records),
        },
        "ghrd_counts": ghrd_counts,
        "output_path": output_path,
    }


def main():
    font_name = configure_fonts()
    records = load_clinical_rows(INPUT_PATH)
    summary = plot_clinical_heatmap(records, OUTPUT_PATH, ["CR", "AR", "IR"])
    ar_ir_records = [record for record in records if record["group"] in {"AR", "IR"}]
    ar_ir_summary = plot_clinical_heatmap(ar_ir_records, OUTPUT_PATH_AR_IR, ["AR", "IR"])
    print(f"font={font_name}")
    print(
        "groups: "
        f"CR={summary['group_counts']['CR']}, "
        f"AR={summary['group_counts']['AR']}, "
        f"IR={summary['group_counts']['IR']}"
    )
    print(f"line groups: 1L={summary['line_counts']['1L']}, >=2L={summary['line_counts']['>=2L']}")
    print(
        "gHRD threshold=42: "
        f">=42={summary['ghrd_counts']['high']}, "
        f"<42={summary['ghrd_counts']['low']}, "
        f"NA={summary['ghrd_counts']['na']}"
    )
    print(f"output={summary['output_path']}")
    print(
        "AR/IR only groups: "
        f"AR={ar_ir_summary['group_counts']['AR']}, "
        f"IR={ar_ir_summary['group_counts']['IR']}"
    )
    print(
        "AR/IR only line groups: "
        f"1L={ar_ir_summary['line_counts']['1L']}, "
        f">=2L={ar_ir_summary['line_counts']['>=2L']}"
    )
    print(
        "AR/IR only gHRD threshold=42: "
        f">=42={ar_ir_summary['ghrd_counts']['high']}, "
        f"<42={ar_ir_summary['ghrd_counts']['low']}, "
        f"NA={ar_ir_summary['ghrd_counts']['na']}"
    )
    print(f"AR/IR only output={ar_ir_summary['output_path']}")


if __name__ == "__main__":
    main()
