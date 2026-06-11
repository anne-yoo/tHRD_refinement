#!/usr/bin/env python3

import argparse
import csv
import io
import json
import os
import re
import shlex
import subprocess
import textwrap
import time
import urllib.request
from collections import Counter
from pathlib import Path


RUNINFO_URL = "https://trace.ncbi.nlm.nih.gov/Traces/sra-db-be/runinfo?acc={accession}"
DEFAULT_STAR_GENOME_DIR = "/home/jiye/jiye/darkproteome/data/refdata/STAR_hg38_v48"
DEFAULT_MINIMAP2_BIN = "/home/jiye/jiye/darkproteome/tools/minimap2-2.30_x64-linux/minimap2"
DEFAULT_MINIMAP2_INDEX = "/home/omics/DATA3/jiye/darkproteome/data/refdata/hg38_v48.mmi"
DEFAULT_STAR_BIN = "STAR"
DEFAULT_SAMTOOLS_BIN = "samtools"
DEFAULT_FASTQC_BIN = "fastqc"
DEFAULT_MULTIQC_BIN = "multiqc"
DEFAULT_TRIM_GALORE_BIN = "trim_galore"

KNOWN_PANCREAS_NANOPORE_GSMS = {
    "GSM5099840",
    "GSM5099836",
    "GSM5099837",
    "GSM5099841",
    "GSM5099838",
    "GSM5099839",
}

GSM_RE = re.compile(r"\bGSM\d+\b")
SRR_RE = re.compile(r"\bSRR\d+\b")
NANOPORE_KEYWORDS = (
    "nanopore",
    "oxford nanopore",
    "oxford_nanopore",
    "minion",
    "gridion",
    "promethion",
    "ont",
)
ILLUMINA_KEYWORDS = (
    "illumina",
    "nextseq",
    "novaseq",
    "miseq",
    "hiseq",
)


def normalize_text(value):
    if value is None:
        return ""
    return str(value).strip()


def split_csv_field(value):
    return [part.strip() for part in normalize_text(value).split(",") if part.strip()]


def dedupe_keep_order(values):
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def quote_command(parts):
    return " ".join(shlex.quote(str(part)) for part in parts)


def sanitize_name(value):
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return sanitized.strip("._-") or "sample"


def normalize_script_text(text):
    lines = textwrap.dedent(text).splitlines()
    cleaned = []
    for line in lines:
        if line.startswith("        "):
            cleaned.append(line[8:])
        else:
            cleaned.append(line)
    return "\n".join(cleaned).strip() + "\n"


def fetch_text(url, cache_path, timeout, retries, backoff_seconds):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.is_file():
        return cache_path.read_text(encoding="utf-8")

    headers = {"User-Agent": "Mozilla/5.0 (compatible; rnaseq_preprocess_align/1.0)"}
    last_error = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                text = response.read().decode("utf-8", "replace")
            cache_path.write_text(text, encoding="utf-8")
            return text
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(backoff_seconds * attempt)

    raise RuntimeError(f"Request failed after {retries} attempts: {url}") from last_error


def load_sample_rows(map_tsv_path):
    rows = []
    with map_tsv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("status") != "ok":
                continue
            rows.append(row)
    return rows


def parse_fastq_layout(sample_id, final_fastq_dir):
    single = final_fastq_dir / f"{sample_id}.fastq.gz"
    r1 = final_fastq_dir / f"{sample_id}_1.fastq.gz"
    r2 = final_fastq_dir / f"{sample_id}_2.fastq.gz"

    if single.is_file() and not r1.exists() and not r2.exists():
        return {
            "file_layout": "SINGLE",
            "fastq_paths": [single],
            "notes": [],
        }

    if r1.is_file() and r2.is_file() and not single.exists():
        return {
            "file_layout": "PAIRED",
            "fastq_paths": [r1, r2],
            "notes": [],
        }

    notes = []
    fastq_paths = []
    if single.exists():
        fastq_paths.append(single)
    if r1.exists():
        fastq_paths.append(r1)
    if r2.exists():
        fastq_paths.append(r2)

    if not fastq_paths:
        notes.append("No final FASTQ file found for sample")
        file_layout = "MISSING"
    else:
        notes.append("Ambiguous FASTQ naming pattern")
        file_layout = "AMBIGUOUS"

    return {
        "file_layout": file_layout,
        "fastq_paths": fastq_paths,
        "notes": notes,
    }


def fetch_runinfo(accession, runinfo_cache_dir, timeout, retries, backoff_seconds):
    cache_path = runinfo_cache_dir / f"{accession}.csv"
    text = fetch_text(
        RUNINFO_URL.format(accession=accession),
        cache_path,
        timeout,
        retries,
        backoff_seconds,
    )
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ValueError(f"No runinfo rows found for accession: {accession}")
    return rows


def summarize_runinfo(srrs, runinfo_cache_dir, timeout, retries, backoff_seconds):
    layouts = []
    platforms = []
    models = []
    instruments = []
    experiments = []
    runs = []

    for srr in srrs:
        rows = fetch_runinfo(srr, runinfo_cache_dir, timeout, retries, backoff_seconds)
        for row in rows:
            layout = normalize_text(row.get("LibraryLayout"))
            platform = normalize_text(row.get("Platform"))
            model = normalize_text(row.get("Model"))
            instrument = normalize_text(row.get("Instrument"))
            experiment = normalize_text(row.get("Experiment"))
            run = normalize_text(row.get("Run"))
            if layout:
                layouts.append(layout.upper())
            if platform:
                platforms.append(platform)
            if model:
                models.append(model)
            if instrument:
                instruments.append(instrument)
            if experiment:
                experiments.append(experiment)
            if run:
                runs.append(run)

    layouts = dedupe_keep_order(layouts)
    platforms = dedupe_keep_order(platforms)
    models = dedupe_keep_order(models)
    instruments = dedupe_keep_order(instruments)
    experiments = dedupe_keep_order(experiments)
    runs = dedupe_keep_order(runs)

    if len(layouts) == 1:
        metadata_layout = layouts[0]
    elif len(layouts) == 0:
        metadata_layout = "UNKNOWN"
    else:
        metadata_layout = "MIXED"

    return {
        "metadata_layout": metadata_layout,
        "layout_values": layouts,
        "platform_values": platforms,
        "model_values": models,
        "instrument_values": instruments,
        "experiment_values": experiments,
        "run_values": runs,
    }


def contains_keyword(values, keywords):
    lowered = " | ".join(values).lower()
    return any(keyword in lowered for keyword in keywords)


def detect_mode(sample, known_pancreas_nanopore):
    warnings = list(sample["warnings"])
    metadata_platform_values = sample["platform_values"]
    metadata_model_values = sample["model_values"]
    metadata_instrument_values = sample["instrument_values"]

    known_nanopore = any(gsm in known_pancreas_nanopore for gsm in sample["gsm_accessions"])
    metadata_nanopore = (
        contains_keyword(metadata_platform_values, NANOPORE_KEYWORDS)
        or contains_keyword(metadata_model_values, NANOPORE_KEYWORDS)
        or contains_keyword(metadata_instrument_values, NANOPORE_KEYWORDS)
    )
    metadata_illumina = (
        contains_keyword(metadata_platform_values, ILLUMINA_KEYWORDS)
        or contains_keyword(metadata_model_values, ILLUMINA_KEYWORDS)
        or contains_keyword(metadata_instrument_values, ILLUMINA_KEYWORDS)
    )

    if known_nanopore or metadata_nanopore:
        mode = "NANOPORE_LONG_READ"
        aligner = "minimap2"
        trim_mode = "skip"
        if sample["resolved_layout"] == "PAIRED":
            warnings.append("Nanopore sample resolved as paired-end; review required")
    else:
        mode = "ILLUMINA_SHORT_READ"
        aligner = "STAR"
        trim_mode = "trim_galore_illumina"
        if not metadata_illumina and metadata_platform_values:
            warnings.append(
                "Metadata platform does not clearly say Illumina; defaulting to short-read STAR"
            )

    return mode, aligner, trim_mode, warnings, known_nanopore, metadata_nanopore


def choose_layout(file_layout, metadata_layout):
    warnings = []
    if file_layout in {"SINGLE", "PAIRED"} and metadata_layout in {"SINGLE", "PAIRED"}:
        if file_layout != metadata_layout:
            warnings.append(
                f"FASTQ naming suggests {file_layout} but metadata says {metadata_layout}; using FASTQ layout"
            )
        return file_layout, warnings

    if file_layout in {"SINGLE", "PAIRED"}:
        if metadata_layout == "MIXED":
            warnings.append("Metadata layout is mixed across runs; using FASTQ layout")
        return file_layout, warnings

    if metadata_layout in {"SINGLE", "PAIRED"}:
        warnings.append("FASTQ naming is ambiguous; using metadata layout")
        return metadata_layout, warnings

    warnings.append("Could not determine sample layout from FASTQ names or metadata")
    return "UNKNOWN", warnings


def collect_samples(args, processing_dirs):
    map_tsv = args.dataset_dir / "flattened_metadata" / "sample_srr_map.tsv"
    final_fastq_dir = args.dataset_dir / "fastq" / "finalfastq_forbam"
    runinfo_cache_dir = processing_dirs["cache"] / "runinfo"
    sample_rows = load_sample_rows(map_tsv)

    samples = []
    for row in sample_rows:
        sample_id = normalize_text(row["sample_id"])
        input_accessions = split_csv_field(row.get("input_accessions", ""))
        gsm_accessions = [acc for acc in input_accessions if GSM_RE.fullmatch(acc)]
        final_srrs = split_csv_field(row.get("final_srrs", ""))

        fastq_info = parse_fastq_layout(sample_id, final_fastq_dir)
        runinfo_summary = summarize_runinfo(
            final_srrs,
            runinfo_cache_dir,
            args.request_timeout,
            args.request_retries,
            args.request_backoff_seconds,
        )
        resolved_layout, layout_warnings = choose_layout(
            fastq_info["file_layout"], runinfo_summary["metadata_layout"]
        )

        warnings = list(fastq_info["notes"]) + layout_warnings
        sample_label = gsm_accessions[0] if len(gsm_accessions) == 1 else sample_id
        sample_prefix = sanitize_name(sample_id)

        sample = {
            "sample_id": sample_id,
            "sample_label": sample_label,
            "sample_prefix": sample_prefix,
            "gsm_accessions": gsm_accessions,
            "source_id": normalize_text(row.get("source_id")),
            "primary_tissue": normalize_text(row.get("primary_tissue")),
            "legacy_sample_id": normalize_text(row.get("legacy_sample_id")),
            "input_accessions": input_accessions,
            "resolved_srxs": split_csv_field(row.get("resolved_srxs", "")),
            "final_srrs": final_srrs,
            "file_layout": fastq_info["file_layout"],
            "metadata_layout": runinfo_summary["metadata_layout"],
            "resolved_layout": resolved_layout,
            "fastq_paths": fastq_info["fastq_paths"],
            "warnings": warnings,
            "platform_values": runinfo_summary["platform_values"],
            "model_values": runinfo_summary["model_values"],
            "instrument_values": runinfo_summary["instrument_values"],
            "experiment_values": runinfo_summary["experiment_values"],
            "run_values": runinfo_summary["run_values"],
        }

        mode, aligner, trim_mode, warnings, known_nanopore, metadata_nanopore = detect_mode(
            sample, KNOWN_PANCREAS_NANOPORE_GSMS
        )
        sample["mode"] = mode
        sample["aligner"] = aligner
        sample["trim_mode"] = trim_mode
        sample["warnings"] = dedupe_keep_order(warnings)
        sample["known_nanopore"] = known_nanopore
        sample["metadata_nanopore"] = metadata_nanopore

        status = "READY"
        if not sample["fastq_paths"]:
            status = "REVIEW_REQUIRED"
        if sample["resolved_layout"] not in {"SINGLE", "PAIRED"}:
            status = "REVIEW_REQUIRED"
        if sample["mode"] == "NANOPORE_LONG_READ" and sample["resolved_layout"] != "SINGLE":
            status = "REVIEW_REQUIRED"

        sample["status"] = status
        samples.append(sample)

    return samples


def build_processing_dirs(dataset_dir):
    processing_dir = dataset_dir / "processing"
    dirs = {
        "processing": processing_dir,
        "raw_fastqc": processing_dir / "raw_fastqc",
        "trimmed_fastq": processing_dir / "trimmed_fastq",
        "trimmed_fastqc": processing_dir / "trimmed_fastqc",
        "multiqc_raw": processing_dir / "multiqc" / "raw_fastqc",
        "multiqc_trimmed": processing_dir / "multiqc" / "trimmed_fastqc",
        "alignments_star": processing_dir / "alignments" / "star",
        "alignments_minimap2": processing_dir / "alignments" / "minimap2",
        "logs": processing_dir / "logs",
        "commands": processing_dir / "commands",
        "commands_samples": processing_dir / "commands" / "samples",
        "manifests": processing_dir / "manifests",
        "cache": processing_dir / "cache",
    }
    return dirs


def ensure_dirs(dirs):
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)


def write_json(path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)


def write_sample_plan_tsv(samples, output_path):
    fieldnames = [
        "sample_id",
        "sample_label",
        "sample_prefix",
        "status",
        "mode",
        "aligner",
        "trim_mode",
        "source_id",
        "primary_tissue",
        "gsm_accessions",
        "resolved_layout",
        "file_layout",
        "metadata_layout",
        "fastq_paths",
        "final_srrs",
        "platform_values",
        "model_values",
        "instrument_values",
        "known_nanopore",
        "metadata_nanopore",
        "warnings",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sample_id": sample["sample_id"],
                    "sample_label": sample["sample_label"],
                    "sample_prefix": sample["sample_prefix"],
                    "status": sample["status"],
                    "mode": sample["mode"],
                    "aligner": sample["aligner"],
                    "trim_mode": sample["trim_mode"],
                    "source_id": sample["source_id"],
                    "primary_tissue": sample["primary_tissue"],
                    "gsm_accessions": ",".join(sample["gsm_accessions"]),
                    "resolved_layout": sample["resolved_layout"],
                    "file_layout": sample["file_layout"],
                    "metadata_layout": sample["metadata_layout"],
                    "fastq_paths": ",".join(str(path) for path in sample["fastq_paths"]),
                    "final_srrs": ",".join(sample["final_srrs"]),
                    "platform_values": ",".join(sample["platform_values"]),
                    "model_values": ",".join(sample["model_values"]),
                    "instrument_values": ",".join(sample["instrument_values"]),
                    "known_nanopore": str(sample["known_nanopore"]).lower(),
                    "metadata_nanopore": str(sample["metadata_nanopore"]).lower(),
                    "warnings": " | ".join(sample["warnings"]),
                }
            )


def build_trimmed_glob(basename_without_ext, layout):
    if layout == "SINGLE":
        return [
            f"{basename_without_ext}_trimmed.fq.gz",
            f"{basename_without_ext}_trimmed.fastq.gz",
        ]
    return [
        f"{basename_without_ext}_val_1.fq.gz",
        f"{basename_without_ext}_val_1.fastq.gz",
        f"{basename_without_ext}_val_2.fq.gz",
        f"{basename_without_ext}_val_2.fastq.gz",
    ]


def strip_fastq_suffix(filename):
    for suffix in (".fastq.gz", ".fq.gz", ".fastq", ".fq"):
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return filename


def generate_sample_script(sample, args, dirs):
    script_path = dirs["commands_samples"] / f"{sample['sample_prefix']}.sh"
    log_path = dirs["logs"] / f"{sample['sample_prefix']}.log"

    fastq_paths = [str(path) for path in sample["fastq_paths"]]
    fastq_paths_shell = " ".join(shlex.quote(path) for path in fastq_paths)
    raw_fastqc_cmd = (
        f'{shlex.quote(args.fastqc_bin)} --threads {args.fastqc_threads} '
        f'--outdir {shlex.quote(str(dirs["raw_fastqc"]))} {fastq_paths_shell}'
    )

    sample_align_dir = (
        dirs["alignments_star"] / sample["sample_prefix"]
        if sample["aligner"] == "STAR"
        else dirs["alignments_minimap2"] / sample["sample_prefix"]
    )
    bam_index_cmd = ""

    warning_block = ""
    if sample["warnings"]:
        warning_lines = "\n".join(
            f'echo "[WARN] {warning.replace(chr(34), chr(39))}"'
            for warning in sample["warnings"]
        )
        warning_block = warning_lines + "\n"

    if sample["status"] != "READY":
        content = textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail

            mkdir -p {shlex.quote(str(dirs["logs"]))}
            exec > >(tee -a {shlex.quote(str(log_path))}) 2>&1

            echo "[INFO] Sample {sample['sample_id']} is marked as {sample['status']}"
            {warning_block}echo "[ERROR] Review required before execution"
            exit 1
            """
        )
        content = normalize_script_text(content)
        script_path.write_text(content, encoding="utf-8")
        script_path.chmod(0o755)
        return script_path

    trim_block = ""
    align_block = ""

    if sample["mode"] == "ILLUMINA_SHORT_READ":
        if sample["resolved_layout"] == "SINGLE":
            input_fastq = fastq_paths[0]
            base = strip_fastq_suffix(Path(input_fastq).name)
            trimmed_candidates = build_trimmed_glob(base, "SINGLE")
            trim_cmd = "\n".join(
                [
                    f'INPUT_FASTQ={shlex.quote(input_fastq)}',
                    f'TRIMMED_DIR={shlex.quote(str(dirs["trimmed_fastq"]))}',
                    f'TRIMMED_FASTQC_DIR={shlex.quote(str(dirs["trimmed_fastqc"]))}',
                    f'{shlex.quote(args.trim_galore_bin)} \\',
                    f'  --cores {args.trim_galore_cores} \\',
                    f'  --illumina \\',
                    f'  --fastqc \\',
                    f'  --fastqc_args "--outdir {dirs["trimmed_fastqc"]}" \\',
                    f'  --output_dir "$TRIMMED_DIR" \\',
                    f'  "$INPUT_FASTQ"',
                    'TRIMMED_FASTQ=""',
                ]
            )
            for candidate in trimmed_candidates:
                trim_cmd += f'\nif [[ -z "$TRIMMED_FASTQ" && -f "$TRIMMED_DIR/{candidate}" ]]; then TRIMMED_FASTQ="$TRIMMED_DIR/{candidate}"; fi'
            trim_cmd += '\nif [[ -z "$TRIMMED_FASTQ" ]]; then echo "[ERROR] Could not find trimmed FASTQ"; exit 1; fi'
            trim_block = trim_cmd

            align_block = textwrap.dedent(
                f"""\
                mkdir -p {shlex.quote(str(sample_align_dir))}
                {shlex.quote(args.star_bin)} \\
                  --genomeDir {shlex.quote(args.star_genome_dir)} \\
                  --readFilesIn "$TRIMMED_FASTQ" \\
                  --readFilesCommand zcat \\
                  --runThreadN {args.star_threads} \\
                  --twopassMode Basic \\
                  --outSAMtype BAM SortedByCoordinate \\
                  --quantMode GeneCounts \\
                  --outSAMstrandField intronMotif \\
                  --outSAMattributes NH HI NM MD AS XS \\
                  --limitBAMsortRAM {args.star_bam_sort_ram} \\
                  --outFileNamePrefix {shlex.quote(str(sample_align_dir / (sample["sample_prefix"] + ".")))}
                {shlex.quote(args.samtools_bin)} index {shlex.quote(str(sample_align_dir / f"{sample['sample_prefix']}.Aligned.sortedByCoord.out.bam"))}
                """
            )
        else:
            fq1, fq2 = fastq_paths
            base1 = strip_fastq_suffix(Path(fq1).name)
            base2 = strip_fastq_suffix(Path(fq2).name)
            trimmed_candidates_1 = build_trimmed_glob(base1, "PAIRED")
            trimmed_candidates_2 = build_trimmed_glob(base2, "PAIRED")
            trim_cmd = "\n".join(
                [
                    f'INPUT_FASTQ_1={shlex.quote(fq1)}',
                    f'INPUT_FASTQ_2={shlex.quote(fq2)}',
                    f'TRIMMED_DIR={shlex.quote(str(dirs["trimmed_fastq"]))}',
                    f'TRIMMED_FASTQC_DIR={shlex.quote(str(dirs["trimmed_fastqc"]))}',
                    f'{shlex.quote(args.trim_galore_bin)} \\',
                    f'  --cores {args.trim_galore_cores} \\',
                    f'  --illumina \\',
                    f'  --paired \\',
                    f'  --fastqc \\',
                    f'  --fastqc_args "--outdir {dirs["trimmed_fastqc"]}" \\',
                    f'  --output_dir "$TRIMMED_DIR" \\',
                    f'  "$INPUT_FASTQ_1" "$INPUT_FASTQ_2"',
                    'TRIMMED_FASTQ_1=""',
                    'TRIMMED_FASTQ_2=""',
                ]
            )
            for candidate in trimmed_candidates_1:
                if "_val_1" in candidate:
                    trim_cmd += f'\nif [[ -z "$TRIMMED_FASTQ_1" && -f "$TRIMMED_DIR/{candidate}" ]]; then TRIMMED_FASTQ_1="$TRIMMED_DIR/{candidate}"; fi'
            for candidate in trimmed_candidates_2:
                if "_val_2" in candidate:
                    trim_cmd += f'\nif [[ -z "$TRIMMED_FASTQ_2" && -f "$TRIMMED_DIR/{candidate}" ]]; then TRIMMED_FASTQ_2="$TRIMMED_DIR/{candidate}"; fi'
            trim_cmd += '\nif [[ -z "$TRIMMED_FASTQ_1" || -z "$TRIMMED_FASTQ_2" ]]; then echo "[ERROR] Could not find paired trimmed FASTQ files"; exit 1; fi'
            trim_block = trim_cmd

            align_block = textwrap.dedent(
                f"""\
                mkdir -p {shlex.quote(str(sample_align_dir))}
                {shlex.quote(args.star_bin)} \\
                  --genomeDir {shlex.quote(args.star_genome_dir)} \\
                  --readFilesIn "$TRIMMED_FASTQ_1" "$TRIMMED_FASTQ_2" \\
                  --readFilesCommand zcat \\
                  --runThreadN {args.star_threads} \\
                  --twopassMode Basic \\
                  --outSAMtype BAM SortedByCoordinate \\
                  --quantMode GeneCounts \\
                  --outSAMstrandField intronMotif \\
                  --outSAMattributes NH HI NM MD AS XS \\
                  --limitBAMsortRAM {args.star_bam_sort_ram} \\
                  --outFileNamePrefix {shlex.quote(str(sample_align_dir / (sample["sample_prefix"] + ".")))}
                {shlex.quote(args.samtools_bin)} index {shlex.quote(str(sample_align_dir / f"{sample['sample_prefix']}.Aligned.sortedByCoord.out.bam"))}
                """
            )
    else:
        input_fastq = fastq_paths[0]
        trim_block = 'echo "[INFO] Skipping Trim Galore for Nanopore long-read sample"'
        bam_path = sample_align_dir / f"{sample['sample_prefix']}.sorted.bam"
        align_block = textwrap.dedent(
            f"""\
            mkdir -p {shlex.quote(str(sample_align_dir))}
            {shlex.quote(args.minimap2_bin)} \\
              -ax splice \\
              -uf \\
              -k14 \\
              -t {args.minimap2_threads} \\
              {shlex.quote(args.minimap2_index)} \\
              {shlex.quote(input_fastq)} \\
              | {shlex.quote(args.samtools_bin)} sort -@ {args.samtools_sort_threads} -o {shlex.quote(str(bam_path))}
            {shlex.quote(args.samtools_bin)} index {shlex.quote(str(bam_path))}
            """
        )

    content = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        mkdir -p {shlex.quote(str(dirs["logs"]))}
        exec > >(tee -a {shlex.quote(str(log_path))}) 2>&1

        echo "[INFO] sample_id={sample['sample_id']}"
        echo "[INFO] sample_label={sample['sample_label']}"
        echo "[INFO] mode={sample['mode']}"
        echo "[INFO] aligner={sample['aligner']}"
        echo "[INFO] layout={sample['resolved_layout']}"
        {warning_block}mkdir -p {shlex.quote(str(dirs["raw_fastqc"]))} {shlex.quote(str(dirs["trimmed_fastq"]))} {shlex.quote(str(dirs["trimmed_fastqc"]))}

        echo "[INFO] Running raw FastQC"
        {raw_fastqc_cmd}

        echo "[INFO] Running trimming / trimmed FastQC step"
        {trim_block}

        echo "[INFO] Running alignment"
        {align_block}

        echo "[INFO] Sample completed successfully"
        """
    )
    content = normalize_script_text(content)
    script_path.write_text(content, encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def generate_multiqc_script(name, input_dir, output_dir, multiqc_bin):
    output_dir.mkdir(parents=True, exist_ok=True)
    script_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f'INPUT_DIR={shlex.quote(str(input_dir))}',
        f'OUTPUT_DIR={shlex.quote(str(output_dir))}',
        'mkdir -p "$OUTPUT_DIR"',
        'shopt -s nullglob',
        'FASTQC_FILES=("$INPUT_DIR"/*_fastqc.zip "$INPUT_DIR"/*_fastqc.html)',
        'shopt -u nullglob',
        'if (( ${#FASTQC_FILES[@]} > 0 )); then',
        f'  {shlex.quote(multiqc_bin)} "$INPUT_DIR" -o "$OUTPUT_DIR"',
        "else",
        f'  echo "No FastQC outputs found under $INPUT_DIR; skipping MultiQC"',
        "fi",
    ]
    return "\n".join(script_lines) + "\n"


def generate_command_files(samples, args, dirs):
    ready_commands = []
    review_commands = []
    ready_script_paths = []
    review_script_paths = []
    sample_script_map = {}

    for sample in samples:
        script_path = generate_sample_script(sample, args, dirs)
        sample_script_map[sample["sample_id"]] = script_path
        command_line = f"bash {shlex.quote(str(script_path))}"
        if sample["status"] == "READY":
            ready_commands.append(command_line)
            ready_script_paths.append(str(script_path))
        else:
            review_commands.append(command_line)
            review_script_paths.append(str(script_path))

    ready_list = dirs["commands"] / "sample_commands.ready.txt"
    ready_list.write_text("\n".join(ready_commands) + ("\n" if ready_commands else ""), encoding="utf-8")

    review_list = dirs["commands"] / "sample_commands.review_required.txt"
    review_list.write_text("\n".join(review_commands) + ("\n" if review_commands else ""), encoding="utf-8")

    ready_script_list = dirs["commands"] / "sample_scripts.ready.txt"
    ready_script_list.write_text(
        "\n".join(ready_script_paths) + ("\n" if ready_script_paths else ""),
        encoding="utf-8",
    )

    review_script_list = dirs["commands"] / "sample_scripts.review_required.txt"
    review_script_list.write_text(
        "\n".join(review_script_paths) + ("\n" if review_script_paths else ""),
        encoding="utf-8",
    )

    raw_multiqc_script = dirs["commands"] / "run_multiqc_raw.sh"
    raw_multiqc_script.write_text(
        generate_multiqc_script("raw", dirs["raw_fastqc"], dirs["multiqc_raw"], args.multiqc_bin),
        encoding="utf-8",
    )
    raw_multiqc_script.chmod(0o755)

    trimmed_multiqc_script = dirs["commands"] / "run_multiqc_trimmed.sh"
    trimmed_multiqc_script.write_text(
        generate_multiqc_script(
            "trimmed",
            dirs["trimmed_fastqc"],
            dirs["multiqc_trimmed"],
            args.multiqc_bin,
        ),
        encoding="utf-8",
    )
    trimmed_multiqc_script.chmod(0o755)

    run_all_script = dirs["commands"] / "run_all_samples_parallel.sh"
    master_log = dirs["logs"] / "run_all_samples_parallel.log"
    run_all_script.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail

            COMMAND_LIST={shlex.quote(str(ready_list))}
            SCRIPT_LIST={shlex.quote(str(ready_script_list))}
            PARALLEL_JOBS="${{PARALLEL_JOBS:-{args.parallel_jobs}}}"
            MASTER_LOG={shlex.quote(str(master_log))}

            if [[ ! -s "$SCRIPT_LIST" ]]; then
                echo "No ready sample scripts found in $SCRIPT_LIST"
                exit 0
            fi

            mkdir -p "$(dirname "$MASTER_LOG")"
            exec > >(tee -a "$MASTER_LOG") 2>&1

            echo "[INFO] Ready command file: $COMMAND_LIST"
            echo "[INFO] Ready script list: $SCRIPT_LIST"
            echo "[INFO] Running up to $PARALLEL_JOBS samples concurrently"

            if ! tr '\\n' '\\0' < "$SCRIPT_LIST" | xargs -0 -r -P "$PARALLEL_JOBS" -I {{}} bash "{{}}"; then
                echo "[ERROR] One or more sample jobs failed. Skipping MultiQC."
                exit 1
            fi

            bash {shlex.quote(str(raw_multiqc_script))}
            bash {shlex.quote(str(trimmed_multiqc_script))}
            """
        ),
        encoding="utf-8",
    )
    run_all_script.chmod(0o755)

    return {
        "ready_list": ready_list,
        "review_list": review_list,
        "ready_script_list": ready_script_list,
        "review_script_list": review_script_list,
        "raw_multiqc_script": raw_multiqc_script,
        "trimmed_multiqc_script": trimmed_multiqc_script,
        "run_all_script": run_all_script,
        "sample_script_map": sample_script_map,
    }


def write_run_manifest(args, dirs, output_path):
    rows = [
        ("dataset_dir", str(args.dataset_dir)),
        ("processing_dir", str(dirs["processing"])),
        ("star_bin", args.star_bin),
        ("star_genome_dir", args.star_genome_dir),
        ("minimap2_bin", args.minimap2_bin),
        ("minimap2_index", args.minimap2_index),
        ("samtools_bin", args.samtools_bin),
        ("fastqc_bin", args.fastqc_bin),
        ("trim_galore_bin", args.trim_galore_bin),
        ("multiqc_bin", args.multiqc_bin),
        ("fastqc_threads", str(args.fastqc_threads)),
        ("trim_galore_cores", str(args.trim_galore_cores)),
        ("star_threads", str(args.star_threads)),
        ("minimap2_threads", str(args.minimap2_threads)),
        ("samtools_sort_threads", str(args.samtools_sort_threads)),
        ("parallel_jobs", str(args.parallel_jobs)),
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["key", "value"])
        writer.writerows(rows)


def print_summary(samples):
    status_counts = Counter(sample["status"] for sample in samples)
    mode_counts = Counter((sample["mode"], sample["resolved_layout"]) for sample in samples)
    known_nanopore = [sample["sample_id"] for sample in samples if sample["known_nanopore"]]
    metadata_nanopore = [
        sample["sample_id"]
        for sample in samples
        if sample["metadata_nanopore"] and sample["sample_id"] not in known_nanopore
    ]

    print(f"Total samples: {len(samples)}")
    print(f"Status counts: {dict(status_counts)}")
    print(
        "Mode/layout counts: "
        + ", ".join(f"{mode}/{layout}={count}" for (mode, layout), count in sorted(mode_counts.items()))
    )
    print(f"Known pancreas Nanopore samples: {len(known_nanopore)}")
    if known_nanopore:
        print("Known pancreas Nanopore sample_ids:")
        for sample_id in known_nanopore:
            print(sample_id)
    print(f"Metadata-flagged Nanopore samples outside known list: {len(metadata_nanopore)}")
    if metadata_nanopore:
        print("Possible metadata-flagged Nanopore sample_ids:")
        for sample_id in metadata_nanopore:
            print(sample_id)


def run_parallel(run_all_script):
    subprocess.run(["bash", str(run_all_script)], check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a two-stage RNA-seq preprocessing and alignment workflow. "
            "Stage 1 writes per-sample shell scripts and review manifests. "
            "Stage 2 optionally runs them in parallel without requiring GNU parallel."
        )
    )
    parser.add_argument("dataset_dir", type=Path, help="RNA-seq dataset directory")
    parser.add_argument(
        "--run",
        action="store_true",
        help="After generating command files, run ready sample scripts with the generated bash runner.",
    )
    parser.add_argument(
        "--parallel-jobs",
        type=int,
        default=4,
        help="Number of samples to run in parallel during stage 2.",
    )
    parser.add_argument("--star-bin", default=DEFAULT_STAR_BIN)
    parser.add_argument("--star-genome-dir", default=DEFAULT_STAR_GENOME_DIR)
    parser.add_argument("--minimap2-bin", default=DEFAULT_MINIMAP2_BIN)
    parser.add_argument("--minimap2-index", default=DEFAULT_MINIMAP2_INDEX)
    parser.add_argument("--samtools-bin", default=DEFAULT_SAMTOOLS_BIN)
    parser.add_argument("--fastqc-bin", default=DEFAULT_FASTQC_BIN)
    parser.add_argument("--multiqc-bin", default=DEFAULT_MULTIQC_BIN)
    parser.add_argument("--trim-galore-bin", default=DEFAULT_TRIM_GALORE_BIN)
    parser.add_argument("--fastqc-threads", type=int, default=8)
    parser.add_argument("--trim-galore-cores", type=int, default=8)
    parser.add_argument("--star-threads", type=int, default=12)
    parser.add_argument("--minimap2-threads", type=int, default=12)
    parser.add_argument("--samtools-sort-threads", type=int, default=8)
    parser.add_argument("--star-bam-sort-ram", type=int, default=70000000000)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--request-backoff-seconds", type=float, default=1.5)
    return parser.parse_args()


def validate_inputs(dataset_dir):
    required_paths = [
        dataset_dir / "flattened_metadata" / "sample_srr_map.tsv",
        dataset_dir / "fastq" / "finalfastq_forbam",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise SystemExit("Missing required dataset inputs:\n" + "\n".join(missing))


def main():
    args = parse_args()
    args.dataset_dir = args.dataset_dir.expanduser().resolve()
    validate_inputs(args.dataset_dir)

    dirs = build_processing_dirs(args.dataset_dir)
    ensure_dirs(dirs)

    samples = collect_samples(args, dirs)
    command_files = generate_command_files(samples, args, dirs)

    sample_plan_tsv = dirs["manifests"] / "sample_plan.tsv"
    sample_plan_json = dirs["manifests"] / "sample_plan.json"
    run_manifest_tsv = dirs["manifests"] / "run_manifest.tsv"

    write_sample_plan_tsv(samples, sample_plan_tsv)
    write_json(sample_plan_json, samples)
    write_run_manifest(args, dirs, run_manifest_tsv)

    print_summary(samples)
    print(sample_plan_tsv)
    print(sample_plan_json)
    print(command_files["ready_list"])
    print(command_files["review_list"])
    print(command_files["ready_script_list"])
    print(command_files["review_script_list"])
    print(command_files["run_all_script"])
    print(command_files["raw_multiqc_script"])
    print(command_files["trimmed_multiqc_script"])

    if args.run:
        run_parallel(command_files["run_all_script"])


if __name__ == "__main__":
    main()
