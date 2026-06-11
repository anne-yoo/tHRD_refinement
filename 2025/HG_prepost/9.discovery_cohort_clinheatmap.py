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
LINE_COLORS = {
    "1L": "#2B6CB0",
    ">=2L": "#A6CEE3",
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
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Could not parse {field} from value: {value!r}") from exc


def line_group(line_num):
    if line_num == 1:
        return "1L"
    return ">=2L"


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

        parsed_line = parse_line_num(row["line"])
        records.append(
            {
                "sample_id": row["sample_id"],
                "sample_full": row["sample_full"],
                "response": response,
                "group": "AR" if response == "1" else "IR",
                "line_num": parsed_line,
                "line_group": line_group(parsed_line),
                "drug": row["drug"],
                "purpose": row["purpose"],
                "brca": str(row["BRCAmut"]).strip(),
                "interval": parse_float(row["interval"], "interval"),
                "order": order,
            }
        )

    return records


def sort_records(records):
    ar_records = sorted(
        [record for record in records if record["group"] == "AR"],
        key=lambda record: (0 if record["brca"] == "1" else 1, record["order"]),
    )
    ir_records = sorted(
        [record for record in records if record["group"] == "IR"],
        key=lambda record: (0 if record["brca"] == "1" else 1, record["order"]),
    )
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


def draw_group_header(ax, x_positions, ar_count, ir_count):
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
                (left, 4.34),
                right - left,
                0.22,
                facecolor=color,
                edgecolor="none",
                linewidth=0,
            )
        )
        ax.text(center, 4.80, label, ha="center", va="center", fontsize=9, fontweight="bold")


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
                linewidth=0.35,
            )
        )
        ax.text(x + 0.42, y, label, ha="left", va="center", fontsize=8)
        y -= line_height
    return y


def plot_clinical_heatmap(records, output_path):
    sorted_records, ar_records, ir_records = sort_records(records)
    ar_count = len(ar_records)
    ir_count = len(ir_records)

    x_positions = [float(idx) for idx in range(len(sorted_records))]

    interval_values = [record["interval"] for record in sorted_records]
    min_interval = min(interval_values)
    max_interval = max(interval_values)

    interval_cmap = LinearSegmentedColormap.from_list("interval_greys", ["#F4F4F4", "#111111"])
    interval_norm = Normalize(vmin=min_interval, vmax=max_interval)

    brca_colors = [BRCA_COLORS.get(record["brca"], "#BDBDBD") for record in sorted_records]
    line_colors = [LINE_COLORS[record["line_group"]] for record in sorted_records]
    drug_colors = [DRUG_COLORS.get(record["drug"], "#999999") for record in sorted_records]
    purpose_colors = [PURPOSE_COLORS.get(record["purpose"], "#999999") for record in sorted_records]
    interval_colors = [interval_cmap(interval_norm(record["interval"])) for record in sorted_records]

    max_x = max(x_positions)
    legend_x = max_x + 1.5
    fig, ax = plt.subplots(figsize=(9.4, 3.05))
    ax.set_xlim(-5.00, max_x + 8.15)
    ax.set_ylim(-0.05, 5.10)
    ax.axis("off")

    draw_group_header(ax, x_positions, ar_count, ir_count)

    row_y = {
        "BRCAmt": 3.72,
        "Line": 3.10,
        "Drug": 2.48,
        "Purpose": 1.86,
        "Interval": 1.24,
    }
    label_x = -0.95
    for label, y_pos in row_y.items():
        ax.text(label_x, y_pos, label, ha="right", va="center", fontsize=9)

    draw_cell_row(ax, x_positions, row_y["BRCAmt"], brca_colors)
    draw_cell_row(ax, x_positions, row_y["Line"], line_colors)
    draw_cell_row(ax, x_positions, row_y["Drug"], drug_colors)
    draw_cell_row(ax, x_positions, row_y["Purpose"], purpose_colors)
    draw_cell_row(ax, x_positions, row_y["Interval"], interval_colors)

    legend_y = add_discrete_legend(
        ax,
        "Line",
        [("1L", LINE_COLORS["1L"]), (">=2L", LINE_COLORS[">=2L"])],
        legend_x,
        4.62,
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
    add_discrete_legend(
        ax,
        "BRCAmt",
        [("Mutated", BRCA_COLORS["1"]), ("Wild-type", BRCA_COLORS["0"])],
        legend_x,
        legend_y - 0.10,
    )

    cax = fig.add_axes([0.875, 0.18, 0.015, 0.24])
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
        "line_counts": {
            "1L": sum(record["line_group"] == "1L" for record in sorted_records),
            ">=2L": sum(record["line_group"] == ">=2L" for record in sorted_records),
        },
        "output_path": output_path,
    }


def main():
    font_name = configure_fonts()
    records = load_clinical_rows(INPUT_PATH)
    summary = plot_clinical_heatmap(records, OUTPUT_PATH)
    print(f"font={font_name}")
    print(f"AR={summary['ar_count']}, IR={summary['ir_count']}")
    print(f"line groups: 1L={summary['line_counts']['1L']}, >=2L={summary['line_counts']['>=2L']}")
    print(f"output={summary['output_path']}")


if __name__ == "__main__":
    main()
