import csv
import os
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle


INPUT_PATH = "/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost_80_clinicalinfo.txt"
OUTPUT_PATH = "/home/jiye/jiye/copycomparison/GENCODEquant/figures/discovery_cohort_clinheatmap.pdf"
ARIAL_FONT_FILES = [
    "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Arial_Italic.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold_Italic.ttf",
]

AR_COLOR = "#FEB24C"
IR_COLOR = "#5AAE61"
BRCA_COLORS = {
    "1": "#D73027",
    "0": "#E6E6E6",
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
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Could not parse {field} from value: {value!r}") from exc


def load_clinical_rows(path):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))

    # Odd-numbered data rows after the header: Python indices 0, 2, 4, ...
    odd_rows = rows[0::2]
    records = []
    for order, row in enumerate(odd_rows):
        response = str(row["response"]).strip()
        if response not in {"0", "1"}:
            raise ValueError(f"Unexpected response value: {response!r}")

        records.append(
            {
                "sample_id": row["sample_id"],
                "sample_full": row["sample_full"],
                "response": response,
                "group": "AR" if response == "1" else "IR",
                "line_num": parse_line_num(row["line"]),
                "drug": row["drug"],
                "brca": str(row["BRCAmut"]).strip(),
                "interval": parse_float(row["interval"], "interval"),
                "order": order,
            }
        )

    return records


def sort_records(records):
    ar_records = [record for record in records if record["group"] == "AR"]
    ir_records = [record for record in records if record["group"] == "IR"]
    return ar_records + ir_records, ar_records, ir_records


def draw_cell_row(ax, x_positions, y_center, colors, cell_width=0.84, cell_height=0.42):
    for x_pos, color in zip(x_positions, colors):
        ax.add_patch(
            Rectangle(
                (x_pos - cell_width / 2, y_center - cell_height / 2),
                cell_width,
                cell_height,
                facecolor=color,
                edgecolor="white",
                linewidth=0.35,
            )
        )


def draw_group_header(ax, x_positions, ar_count, ir_count, gap):
    segments = [
        ("AR", 0, ar_count, AR_COLOR),
        ("IR", ar_count, ir_count, IR_COLOR),
    ]
    for label, start_idx, count, color in segments:
        segment_x = x_positions[start_idx : start_idx + count]
        left = min(segment_x) - 0.42
        right = max(segment_x) + 0.42
        center = (left + right) / 2

        ax.add_patch(
            Rectangle(
                (left, 3.74),
                right - left,
                0.24,
                facecolor=color,
                edgecolor="none",
                linewidth=0,
            )
        )
        ax.text(center, 4.22, label, ha="center", va="center", fontsize=10)


def add_discrete_legend(ax, title, labels_and_colors, x, y, box_size=0.24, line_height=0.36):
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
                linewidth=0.35,
            )
        )
        ax.text(x + 0.46, y, label, ha="left", va="center", fontsize=8)
        y -= line_height
    return y


def add_line_legend(ax, cmap, norm, min_line, max_line, x, y):
    ax.text(x, y, "Line", ha="left", va="center", fontsize=8.5)
    y -= 0.35
    box_gap = 0.31
    for idx, line_num in enumerate(range(min_line, max_line + 1)):
        color = cmap(norm(line_num))
        ax.add_patch(
            Rectangle(
                (x + idx * box_gap, y - 0.11),
                0.23,
                0.23,
                facecolor=color,
                edgecolor="#666666",
                linewidth=0.25,
            )
        )
    left_label_x = x - 0.14
    right_label_x = x + (max_line - min_line) * box_gap + 0.38
    ax.text(left_label_x, y - 0.34, f"{min_line}L", ha="left", va="center", fontsize=7.5)
    ax.text(
        right_label_x,
        y - 0.34,
        f"{max_line}L",
        ha="right",
        va="center",
        fontsize=7.5,
    )
    return y - 0.78


def plot_clinical_heatmap(records, output_path):
    sorted_records, ar_records, ir_records = sort_records(records)
    ar_count = len(ar_records)
    ir_count = len(ir_records)
    gap = 0.0

    x_positions = []
    for idx in range(ar_count):
        x_positions.append(float(idx))
    for idx in range(ir_count):
        x_positions.append(float(ar_count + gap + idx))

    line_values = [record["line_num"] for record in sorted_records]
    interval_values = [record["interval"] for record in sorted_records]
    min_line = min(line_values)
    max_line = max(line_values)
    min_interval = min(interval_values)
    max_interval = max(interval_values)

    line_cmap = LinearSegmentedColormap.from_list("line_navy", ["#D9E8F5", "#08306B"])
    interval_cmap = LinearSegmentedColormap.from_list("interval_greys", ["#F4F4F4", "#111111"])
    line_norm = Normalize(vmin=min_line, vmax=max_line)
    interval_norm = Normalize(vmin=min_interval, vmax=max_interval)

    line_colors = [line_cmap(line_norm(record["line_num"])) for record in sorted_records]
    drug_colors = [DRUG_COLORS.get(record["drug"], "#999999") for record in sorted_records]
    brca_colors = [BRCA_COLORS.get(record["brca"], "#BDBDBD") for record in sorted_records]
    interval_colors = [interval_cmap(interval_norm(record["interval"])) for record in sorted_records]

    max_x = max(x_positions)
    legend_x = max_x + 1.5
    fig, ax = plt.subplots(figsize=(9.4, 2.7))
    ax.set_xlim(-5.00, max_x + 8.15)
    ax.set_ylim(-0.10, 4.55)
    ax.axis("off")

    draw_group_header(ax, x_positions, ar_count, ir_count, gap)

    row_y = {
        "Line": 3.08,
        "BRCAmt": 2.40,
        "Drug": 1.72,
        "Interval": 1.04,
    }
    label_x = -0.95
    for label, y_pos in row_y.items():
        ax.text(label_x, y_pos, label, ha="right", va="center", fontsize=9)

    draw_cell_row(ax, x_positions, row_y["Line"], line_colors)
    draw_cell_row(ax, x_positions, row_y["BRCAmt"], brca_colors)
    draw_cell_row(ax, x_positions, row_y["Drug"], drug_colors)
    draw_cell_row(ax, x_positions, row_y["Interval"], interval_colors)

    y_after_line = add_line_legend(ax, line_cmap, line_norm, min_line, max_line, legend_x, 4.08)
    add_discrete_legend(
        ax,
        "Drug",
        [(drug, DRUG_COLORS[drug]) for drug in ["Olaparib", "Niraparib", "Rucaparib"]],
        legend_x,
        y_after_line,
    )
    add_discrete_legend(
        ax,
        "BRCAmt",
        [("Mutated", BRCA_COLORS["1"]), ("Wild-type", BRCA_COLORS["0"])],
        legend_x,
        1.30,
    )

    cax = fig.add_axes([0.875, 0.43, 0.015, 0.28])
    sm = plt.cm.ScalarMappable(cmap=interval_cmap, norm=interval_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Interval (days)", fontsize=8)
    cbar.ax.tick_params(labelsize=7, width=0.5, length=2.5)
    cbar.outline.set_linewidth(0.5)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "ar_count": ar_count,
        "ir_count": ir_count,
        "min_line": min_line,
        "max_line": max_line,
        "output_path": output_path,
    }


def main():
    font_name = configure_fonts()
    records = load_clinical_rows(INPUT_PATH)
    summary = plot_clinical_heatmap(records, OUTPUT_PATH)
    print(f"font={font_name}")
    print(f"AR={summary['ar_count']}, IR={summary['ir_count']}")
    print(f"line range={summary['min_line']}-{summary['max_line']}")
    print(f"output={summary['output_path']}")


if __name__ == "__main__":
    main()
