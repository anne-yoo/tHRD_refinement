import csv
from collections import Counter
from pathlib import Path


KIDNEY_DIR = Path("/home/jiye/jiye/darkproteome/data/RPFdb/kidney")
sample_map_tsv = KIDNEY_DIR / "flattened_metadata" / "sample_srr_map.tsv"
final_fastq_dir = KIDNEY_DIR / "fastq" / "finalfastq_forbam"
out_csv_path = KIDNEY_DIR / "flattened_metadata" / "Kidney_sample.simple.csv"


def dedupe_keep_order(values):
    seen = set()
    deduped = []
    for value in values:
        if value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def choose_sample_label(row):
    input_accessions = [value for value in row["input_accessions"].split(",") if value]
    gsm_accessions = dedupe_keep_order(
        accession for accession in input_accessions if accession.startswith("GSM")
    )
    if len(gsm_accessions) == 1:
        return gsm_accessions[0], "GSM"
    return row["sample_id"], "SRX_OR_FALLBACK"


def choose_fastq_path(sample_id):
    paired_fastq_1 = final_fastq_dir / f"{sample_id}_1.fastq.gz"
    single_fastq = final_fastq_dir / f"{sample_id}.fastq.gz"

    if paired_fastq_1.exists():
        return paired_fastq_1, "paired"
    if single_fastq.exists():
        return single_fastq, "single"

    # Kidney set is expected to be single-end. If the file is not present yet,
    # emit the predicted single-end path so downstream code can still use the CSV.
    return single_fastq, "missing_predicted_single"


def load_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [row for row in reader if row["status"] == "ok"]


def main():
    if not sample_map_tsv.exists():
        raise FileNotFoundError(f"Missing sample_srr_map.tsv: {sample_map_tsv}")

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(sample_map_tsv)
    output_rows = []
    label_source_counter = Counter()
    fastq_type_counter = Counter()

    for row in rows:
        sample_label, label_source = choose_sample_label(row)
        fastq_path, fastq_type = choose_fastq_path(row["sample_id"])
        output_rows.append(
            {
                "sample": sample_label,
                "fastq_1": str(fastq_path),
            }
        )
        label_source_counter[label_source] += 1
        fastq_type_counter[fastq_type] += 1

    duplicate_samples = [
        sample for sample, count in Counter(row["sample"] for row in output_rows).items() if count > 1
    ]
    if duplicate_samples:
        preview = ", ".join(sorted(duplicate_samples)[:10])
        raise ValueError(f"Duplicate sample labels detected: {preview}")

    with out_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample", "fastq_1"])
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Input TSV: {sample_map_tsv}")
    print(f"Final FASTQ dir: {final_fastq_dir}")
    print(f"Output CSV: {out_csv_path}")
    print(f"Rows written: {len(output_rows)}")
    print(f"Label source counts: {dict(label_source_counter)}")
    print(f"FASTQ path type counts: {dict(fastq_type_counter)}")


if __name__ == "__main__":
    main()
