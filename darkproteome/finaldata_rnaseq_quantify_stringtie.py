#!/usr/bin/env python3

import argparse
import csv
import io
import json
import re
import shlex
import subprocess
import textwrap
import time
import urllib.request
from collections import Counter
from pathlib import Path


RUNINFO_URL = "https://trace.ncbi.nlm.nih.gov/Traces/sra-db-be/runinfo?acc={accession}"
DEFAULT_RSEQC_BED = "/home/jiye/jiye/darkproteome/data/refdata/gencode.v48.annotation.bed"
DEFAULT_ANNOTATION_GTF = "/home/jiye/jiye/darkproteome/data/refdata/gencode.v48.annotation.gtf"
DEFAULT_INFER_EXPERIMENT_BIN = "infer_experiment.py"
DEFAULT_FEATURECOUNTS_BIN = "featureCounts"
DEFAULT_STRINGTIE_BIN = "/home/jiye/jiye/darkproteome/tools/stringtie/stringtie"
DEFAULT_SAMTOOLS_BIN = "samtools"

KNOWN_PANCREAS_NANOPORE_GSMS = {
    "GSM5099840",
    "GSM5099836",
    "GSM5099837",
    "GSM5099841",
    "GSM5099838",
    "GSM5099839",
}

GSM_RE = re.compile(r"\bGSM\d+\b")
NANOPORE_KEYWORDS = (
    "nanopore",
    "oxford nanopore",
    "oxford_nanopore",
    "minion",
    "gridion",
    "promethion",
    "ont",
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


def sanitize_name(value):
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return sanitized.strip("._-") or "sample"


def normalize_script_text(text):
    lines = textwrap.dedent(text).splitlines()
    cleaned = []
    in_heredoc = False
    for line in lines:
        stripped = line.strip()
        if not in_heredoc and line.startswith("        "):
            line = line[8:]
        cleaned.append(line)
        if "<<'PY'" in line or "<<PY" in line:
            in_heredoc = True
        elif in_heredoc and stripped == "PY":
            in_heredoc = False
    return "\n".join(cleaned).strip() + "\n"


def fetch_text(url, cache_path, timeout, retries, backoff_seconds):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.is_file():
        return cache_path.read_text(encoding="utf-8")

    headers = {"User-Agent": "Mozilla/5.0 (compatible; finaldata_rnaseq_quantify/1.0)"}
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


def fetch_runinfo(accession, cache_dir, timeout, retries, backoff_seconds):
    cache_path = cache_dir / f"{accession}.csv"
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


def summarize_runinfo(srrs, cache_dir, timeout, retries, backoff_seconds):
    layouts = []
    platforms = []
    models = []
    instruments = []
    runs = []
    for srr in srrs:
        rows = fetch_runinfo(srr, cache_dir, timeout, retries, backoff_seconds)
        for row in rows:
            layout = normalize_text(row.get("LibraryLayout"))
            platform = normalize_text(row.get("Platform"))
            model = normalize_text(row.get("Model"))
            instrument = normalize_text(row.get("Instrument"))
            run = normalize_text(row.get("Run"))
            if layout:
                layouts.append(layout.upper())
            if platform:
                platforms.append(platform)
            if model:
                models.append(model)
            if instrument:
                instruments.append(instrument)
            if run:
                runs.append(run)

    layouts = dedupe_keep_order(layouts)
    platforms = dedupe_keep_order(platforms)
    models = dedupe_keep_order(models)
    instruments = dedupe_keep_order(instruments)
    runs = dedupe_keep_order(runs)

    if len(layouts) == 1:
        resolved_layout = layouts[0]
    elif len(layouts) == 0:
        resolved_layout = "UNKNOWN"
    else:
        resolved_layout = "MIXED"

    metadata_nanopore = any(
        keyword in " | ".join(platforms + models + instruments).lower()
        for keyword in NANOPORE_KEYWORDS
    )

    return {
        "resolved_layout": resolved_layout,
        "platform_values": platforms,
        "model_values": models,
        "instrument_values": instruments,
        "run_values": runs,
        "metadata_nanopore": metadata_nanopore,
    }


def load_tsv_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def load_sample_plan_map(sample_plan_path):
    if not sample_plan_path.is_file():
        return {}

    rows = load_tsv_rows(sample_plan_path)
    return {normalize_text(row.get("sample_id")): row for row in rows}


def load_sample_srr_map(sample_srr_map_path):
    rows = load_tsv_rows(sample_srr_map_path)
    return {normalize_text(row.get("sample_id")): row for row in rows if row.get("status") == "ok"}


def find_single_bam(sample_dir):
    bam_candidates = sorted(path for path in sample_dir.glob("*.bam") if path.is_file() or path.is_symlink())
    bai_candidates = sorted(path for path in sample_dir.glob("*.bam.bai") if path.is_file() or path.is_symlink())
    bai_candidates.extend(
        sorted(path for path in sample_dir.glob("*.bai") if path.is_file() or path.is_symlink() and not str(path).endswith(".bam.bai"))
    )
    return bam_candidates, dedupe_keep_order(bai_candidates)


def build_processing_dirs(sample_dir):
    strandedness_dir = sample_dir / "strandedness"
    featurecounts_dir = sample_dir / "featurecounts"
    stringtie_dir = sample_dir / "stringtie"
    ensure_dir(strandedness_dir)
    ensure_dir(featurecounts_dir)
    ensure_dir(stringtie_dir)
    return strandedness_dir, featurecounts_dir, stringtie_dir


def determine_mode_and_layout(match_row, sample_plan_row, sample_srr_row, args, cache_dir):
    sample_id = normalize_text(match_row.get("rnaseq_sample_id"))
    rnaseq_gsm = normalize_text(match_row.get("rnaseq_gsm"))

    if sample_plan_row:
        mode = normalize_text(sample_plan_row.get("mode")) or "UNKNOWN"
        resolved_layout = normalize_text(sample_plan_row.get("resolved_layout")) or "UNKNOWN"
        platform_values = split_csv_field(sample_plan_row.get("platform_values", ""))
        model_values = split_csv_field(sample_plan_row.get("model_values", ""))
        instrument_values = split_csv_field(sample_plan_row.get("instrument_values", ""))
        metadata_nanopore = normalize_text(sample_plan_row.get("metadata_nanopore")).lower() == "true"
        return {
            "mode": mode,
            "resolved_layout": resolved_layout,
            "platform_values": platform_values,
            "model_values": model_values,
            "instrument_values": instrument_values,
            "metadata_nanopore": metadata_nanopore,
            "source": "sample_plan",
        }

    final_srrs = split_csv_field(sample_srr_row.get("final_srrs", "")) if sample_srr_row else []
    runinfo_summary = summarize_runinfo(
        final_srrs,
        cache_dir,
        args.request_timeout,
        args.request_retries,
        args.request_backoff_seconds,
    ) if final_srrs else {
        "resolved_layout": "UNKNOWN",
        "platform_values": [],
        "model_values": [],
        "instrument_values": [],
        "metadata_nanopore": False,
    }

    known_nanopore = rnaseq_gsm in KNOWN_PANCREAS_NANOPORE_GSMS
    metadata_nanopore = runinfo_summary["metadata_nanopore"]
    if known_nanopore or metadata_nanopore:
        mode = "NANOPORE_LONG_READ"
    else:
        mode = "ILLUMINA_SHORT_READ"

    return {
        "mode": mode,
        "resolved_layout": runinfo_summary["resolved_layout"],
        "platform_values": runinfo_summary["platform_values"],
        "model_values": runinfo_summary["model_values"],
        "instrument_values": runinfo_summary["instrument_values"],
        "metadata_nanopore": metadata_nanopore,
        "source": "runinfo",
    }


def generate_sample_script(sample, args):
    sample_dir = sample["sample_dir"]
    strandedness_dir = sample["strandedness_dir"]
    featurecounts_dir = sample["featurecounts_dir"]
    stringtie_dir = sample["stringtie_dir"]
    sample_name = sample["riboseq_gsm"]
    script_path = strandedness_dir / "run_quantification.sh"
    log_path = strandedness_dir / f"{sample_name}.quantification.log"
    infer_txt = strandedness_dir / f"{sample_name}.infer_experiment.txt"
    infer_json = strandedness_dir / f"{sample_name}.strandedness.json"
    fc_command_txt = strandedness_dir / f"{sample_name}.featureCounts.command.txt"
    counts_txt = featurecounts_dir / f"{sample_name}.featureCounts.txt"
    counts_summary = featurecounts_dir / f"{sample_name}.featureCounts.txt.summary"
    stringtie_command_txt = stringtie_dir / f"{sample_name}.stringtie.command.txt"
    stringtie_gtf = stringtie_dir / f"{sample_name}.stringtie.gtf"
    stringtie_gene_abund = stringtie_dir / f"{sample_name}.stringtie.gene_abund.tab"

    if sample["status"] != "READY":
        content = normalize_script_text(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail

            mkdir -p {shlex.quote(str(strandedness_dir))}
            exec > >(tee -a {shlex.quote(str(log_path))}) 2>&1

            echo "[INFO] Sample {sample_name} is marked as {sample['status']}"
            echo "[INFO] Note: {sample['note']}"
            exit 1
            """
        )
        script_path.write_text(content, encoding="utf-8")
        script_path.chmod(0o755)
        return script_path

    fc_extra_args = []
    if sample["mode"] == "ILLUMINA_SHORT_READ":
        if sample["resolved_layout"] == "PAIRED":
            fc_extra_args.append("-p")
    elif sample["mode"] == "NANOPORE_LONG_READ":
        fc_extra_args.extend(["-L", "--primary"])

    fc_extra_args_shell = " ".join(shlex.quote(arg) for arg in fc_extra_args)
    fc_extra_args_line = f"FC_EXTRA_ARGS=({fc_extra_args_shell})" if fc_extra_args else "FC_EXTRA_ARGS=()"

    infer_stage = textwrap.dedent(
        f"""\
        STRAND_OPTION={args.default_strand}
        FRACTION_1="NA"
        FRACTION_2="NA"
        STRAND_REASON="default_without_inference"
        REUSE_EXISTING_STRANDEDNESS={1 if args.reuse_existing_strandedness else 0}

        parse_infer_text() {{
            python3 - "$1" "$2" {args.strandedness_threshold} <<'PY' > "$3"
import json
import pathlib
import re
import sys

infer_path = pathlib.Path(sys.argv[1])
json_path = pathlib.Path(sys.argv[2])
threshold = float(sys.argv[3])
text = infer_path.read_text(encoding='utf-8', errors='replace')
m1 = re.search(r'Fraction of reads explained by "1\\+\\+,1--,2\\+-,2-\\+":\\s*([0-9.]+)', text)
m2 = re.search(r'Fraction of reads explained by "1\\+-,1-\\+,2\\+\\+,2--":\\s*([0-9.]+)', text)
fraction_1 = float(m1.group(1)) if m1 else None
fraction_2 = float(m2.group(1)) if m2 else None
strand_option = 0
reason = 'ambiguous_or_missing'
if fraction_1 is not None and fraction_2 is not None:
    if fraction_1 >= threshold and fraction_1 > fraction_2:
        strand_option = 1
        reason = 'first_pattern_dominant'
    elif fraction_2 >= threshold and fraction_2 > fraction_1:
        strand_option = 2
        reason = 'second_pattern_dominant'
payload = {{
    'fraction_1': fraction_1,
    'fraction_2': fraction_2,
    'strand_option': strand_option,
    'reason': reason,
    'threshold': threshold,
}}
json_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
print(payload['strand_option'])
print(payload['fraction_1'])
print(payload['fraction_2'])
print(payload['reason'])
PY
        }}

        parse_infer_json() {{
            python3 - "$1" <<'PY' > "$2"
import json
import pathlib
import sys

json_path = pathlib.Path(sys.argv[1])
payload = json.loads(json_path.read_text(encoding='utf-8'))
print(payload.get('strand_option', 0))
print(payload.get('fraction_1', 'NA'))
print(payload.get('fraction_2', 'NA'))
print(payload.get('reason', 'loaded_from_existing_json'))
PY
        }}

        if [[ "{1 if args.run_strandedness_inference else 0}" == "1" ]]; then
            if [[ "$REUSE_EXISTING_STRANDEDNESS" == "1" && -s {shlex.quote(str(infer_json))} ]]; then
                echo "[INFO] Reusing existing strandedness JSON: {infer_json}"
                parse_infer_json {shlex.quote(str(infer_json))} {shlex.quote(str(strandedness_dir / f"{sample_name}.strand_parse.tmp"))}
            elif [[ "$REUSE_EXISTING_STRANDEDNESS" == "1" && -s {shlex.quote(str(infer_txt))} ]]; then
                echo "[INFO] Reusing existing infer_experiment output: {infer_txt}"
                parse_infer_text {shlex.quote(str(infer_txt))} {shlex.quote(str(infer_json))} {shlex.quote(str(strandedness_dir / f"{sample_name}.strand_parse.tmp"))}
            else
                {shlex.quote(args.infer_experiment_bin)} \\
                  -r {shlex.quote(args.rseqc_bed)} \\
                  -i {shlex.quote(str(sample["bam_path"]))} \\
                  > {shlex.quote(str(infer_txt))}

                parse_infer_text {shlex.quote(str(infer_txt))} {shlex.quote(str(infer_json))} {shlex.quote(str(strandedness_dir / f"{sample_name}.strand_parse.tmp"))}
            fi

            mapfile -t STRAND_PARSE_LINES < {shlex.quote(str(strandedness_dir / f"{sample_name}.strand_parse.tmp"))}
            STRAND_OPTION="${{STRAND_PARSE_LINES[0]}}"
            FRACTION_1="${{STRAND_PARSE_LINES[1]}}"
            FRACTION_2="${{STRAND_PARSE_LINES[2]}}"
            STRAND_REASON="${{STRAND_PARSE_LINES[3]}}"
            rm -f {shlex.quote(str(strandedness_dir / f"{sample_name}.strand_parse.tmp"))}
        fi
        """
    ).strip()

    stringtie_extra_args = []
    if sample["mode"] == "NANOPORE_LONG_READ":
        stringtie_extra_args.append("-L")
    stringtie_extra_args_shell = " ".join(shlex.quote(arg) for arg in stringtie_extra_args)
    stringtie_extra_args_line = (
        f"STRINGTIE_EXTRA_ARGS=({stringtie_extra_args_shell})"
        if stringtie_extra_args
        else "STRINGTIE_EXTRA_ARGS=()"
    )

    content = normalize_script_text(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        mkdir -p {shlex.quote(str(strandedness_dir))} {shlex.quote(str(featurecounts_dir))} {shlex.quote(str(stringtie_dir))}
        exec > >(tee -a {shlex.quote(str(log_path))}) 2>&1

        echo "[INFO] sample={sample_name}"
        echo "[INFO] bam={sample['bam_path']}"
        echo "[INFO] rnaseq_sample_id={sample['rnaseq_sample_id']}"
        echo "[INFO] rnaseq_gsm={sample['rnaseq_gsm']}"
        echo "[INFO] mode={sample['mode']}"
        echo "[INFO] layout={sample['resolved_layout']}"

        {infer_stage}

        echo "[INFO] inferred_strand_option=$STRAND_OPTION"
        echo "[INFO] infer_fraction_1=$FRACTION_1"
        echo "[INFO] infer_fraction_2=$FRACTION_2"
        echo "[INFO] infer_reason=$STRAND_REASON"

        BAM_PATH={shlex.quote(str(sample["bam_path"]))}
        RUN_FEATURECOUNTS={1 if args.run_featurecounts else 0}
        RUN_STRINGTIE={1 if args.run_stringtie else 0}
        COUNTS_OUT={shlex.quote(str(counts_txt))}
        {fc_extra_args_line}
        STRINGTIE_OUT={shlex.quote(str(stringtie_gtf))}
        STRINGTIE_GENE_ABUND={shlex.quote(str(stringtie_gene_abund))}
        {stringtie_extra_args_line}

        STRINGTIE_STRAND_ARGS=()
        STRINGTIE_STRAND_LABEL="NONE"
        case "$STRAND_OPTION" in
            1)
                STRINGTIE_STRAND_ARGS=(--fr)
                STRINGTIE_STRAND_LABEL="--fr"
                ;;
            2)
                STRINGTIE_STRAND_ARGS=(--rf)
                STRINGTIE_STRAND_LABEL="--rf"
                ;;
            *)
                STRINGTIE_STRAND_ARGS=()
                STRINGTIE_STRAND_LABEL="NONE"
                ;;
        esac

        echo "[INFO] stringtie_strand_args=$STRINGTIE_STRAND_LABEL"

        if [[ "$RUN_FEATURECOUNTS" == "1" ]]; then
            FC_CMD=(
              {shlex.quote(args.featurecounts_bin)}
              -T {args.featurecounts_threads}
              -a {shlex.quote(args.annotation_gtf)}
              -o "$COUNTS_OUT"
              -t exon
              -g gene_id
              -s "$STRAND_OPTION"
            )

            if (( ${{#FC_EXTRA_ARGS[@]}} > 0 )); then
                FC_CMD+=("${{FC_EXTRA_ARGS[@]}}")
            fi
            FC_CMD+=("$BAM_PATH")

            printf '%q ' "${{FC_CMD[@]}}" > {shlex.quote(str(fc_command_txt))}
            printf '\\n' >> {shlex.quote(str(fc_command_txt))}

            "${{FC_CMD[@]}}"

            if [[ -f {shlex.quote(str(counts_summary))} ]]; then
                echo "[INFO] featureCounts summary: {counts_summary}"
            fi
        else
            echo "[INFO] Skipping featureCounts"
        fi

        if [[ "$RUN_STRINGTIE" == "1" ]]; then
            STRINGTIE_CMD=(
              {shlex.quote(args.stringtie_bin)}
              -p {args.stringtie_threads}
              -e
              -B
              -G {shlex.quote(args.annotation_gtf)}
              -o "$STRINGTIE_OUT"
              -A "$STRINGTIE_GENE_ABUND"
            )

            if (( ${{#STRINGTIE_STRAND_ARGS[@]}} > 0 )); then
                STRINGTIE_CMD+=("${{STRINGTIE_STRAND_ARGS[@]}}")
            fi
            if (( ${{#STRINGTIE_EXTRA_ARGS[@]}} > 0 )); then
                STRINGTIE_CMD+=("${{STRINGTIE_EXTRA_ARGS[@]}}")
            fi
            STRINGTIE_CMD+=("$BAM_PATH")

            printf '%q ' "${{STRINGTIE_CMD[@]}}" > {shlex.quote(str(stringtie_command_txt))}
            printf '\\n' >> {shlex.quote(str(stringtie_command_txt))}

            "${{STRINGTIE_CMD[@]}}"
            echo "[INFO] StringTie GTF: {stringtie_gtf}"
            echo "[INFO] StringTie Ballgown tables written alongside GTF in: {stringtie_dir}"
        else
            echo "[INFO] Skipping StringTie"
        fi

        echo "[INFO] Quantification completed successfully"
        """
    )
    script_path.write_text(content, encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def write_manifest(path, rows):
    fieldnames = [
        "riboseq_gsm",
        "rnaseq_gsm",
        "rnaseq_sample_id",
        "bam_path",
        "bai_path",
        "mode",
        "resolved_layout",
        "status",
        "note",
        "sample_dir",
        "featurecounts_dir",
        "stringtie_dir",
        "script_path",
        "mode_source",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_samples(args):
    organ_dir = args.organ_dir.expanduser().resolve()
    finaldata_dir = organ_dir / "finaldata"
    finaldata_rnaseq_dir = finaldata_dir / "RNAseq"
    manifests_dir = finaldata_dir / "manifests"
    processing_manifest_dir = organ_dir / "RNAseq" / "processing" / "manifests"
    sample_plan_path = processing_manifest_dir / "sample_plan.tsv"
    sample_srr_map_path = organ_dir / "RNAseq" / "flattened_metadata" / "sample_srr_map.tsv"
    matched_manifest_path = manifests_dir / "matched_rnaseq_link_manifest.tsv"
    runinfo_cache_dir = manifests_dir / "cache" / "runinfo"

    ensure_dir(manifests_dir)
    ensure_dir(runinfo_cache_dir)

    if not matched_manifest_path.is_file():
        raise FileNotFoundError(f"Missing finaldata match manifest: {matched_manifest_path}")
    if not sample_srr_map_path.is_file():
        raise FileNotFoundError(f"Missing RNAseq sample map: {sample_srr_map_path}")

    match_rows = load_tsv_rows(matched_manifest_path)
    sample_plan_map = load_sample_plan_map(sample_plan_path)
    sample_srr_map = load_sample_srr_map(sample_srr_map_path)

    samples = []
    for row in match_rows:
        riboseq_gsm = normalize_text(row.get("riboseq_gsm"))
        sample_dir = finaldata_rnaseq_dir / riboseq_gsm
        strandedness_dir, featurecounts_dir, stringtie_dir = build_processing_dirs(sample_dir)
        bam_candidates, bai_candidates = find_single_bam(sample_dir)
        rnaseq_sample_id = normalize_text(row.get("rnaseq_sample_id"))
        rnaseq_gsm = normalize_text(row.get("rnaseq_gsm"))

        sample = {
            "riboseq_gsm": riboseq_gsm,
            "rnaseq_gsm": rnaseq_gsm,
            "rnaseq_sample_id": rnaseq_sample_id,
            "sample_dir": sample_dir,
            "strandedness_dir": strandedness_dir,
            "featurecounts_dir": featurecounts_dir,
            "stringtie_dir": stringtie_dir,
            "bam_path": None,
            "bai_path": None,
            "mode": "UNKNOWN",
            "resolved_layout": "UNKNOWN",
            "status": "",
            "note": "",
            "mode_source": "",
            "script_path": "",
        }

        if normalize_text(row.get("status")) != "LINKED":
            sample["status"] = "SKIP_NOT_LINKED"
            sample["note"] = normalize_text(row.get("status")) or "Matched RNAseq link was not created"
            samples.append(sample)
            continue

        if len(bam_candidates) != 1:
            sample["status"] = "BAM_NOT_RESOLVED"
            sample["note"] = f"Expected exactly 1 BAM in {sample_dir}, found {len(bam_candidates)}"
            samples.append(sample)
            continue

        if len(bai_candidates) == 0:
            sample["status"] = "BAI_NOT_FOUND"
            sample["bam_path"] = bam_candidates[0]
            sample["note"] = "No BAI file found in sample folder"
            samples.append(sample)
            continue

        sample["bam_path"] = bam_candidates[0]
        sample["bai_path"] = bai_candidates[0]

        sample_plan_row = sample_plan_map.get(rnaseq_sample_id)
        sample_srr_row = sample_srr_map.get(rnaseq_sample_id)
        mode_info = determine_mode_and_layout(
            row, sample_plan_row, sample_srr_row, args, runinfo_cache_dir
        )
        sample["mode"] = mode_info["mode"]
        sample["resolved_layout"] = mode_info["resolved_layout"]
        sample["mode_source"] = mode_info["source"]

        if sample["mode"] == "ILLUMINA_SHORT_READ" and sample["resolved_layout"] not in {"SINGLE", "PAIRED"}:
            sample["status"] = "REVIEW_REQUIRED"
            sample["note"] = f"Could not resolve short-read layout for sample {rnaseq_sample_id}"
        elif sample["mode"] == "NANOPORE_LONG_READ" and sample["resolved_layout"] not in {"SINGLE", "UNKNOWN"}:
            sample["status"] = "REVIEW_REQUIRED"
            sample["note"] = f"Unexpected long-read layout for sample {rnaseq_sample_id}: {sample['resolved_layout']}"
        else:
            sample["status"] = "READY"
            sample["note"] = ""

        samples.append(sample)

    return organ_dir, finaldata_dir, manifests_dir, samples


def generate_command_files(samples, manifests_dir, args):
    ready_script_paths = []
    review_script_paths = []
    ready_commands = []
    review_commands = []
    for sample in samples:
        script_path = generate_sample_script(sample, args)
        sample["script_path"] = str(script_path)
        command_line = f"bash {shlex.quote(str(script_path))}"
        if sample["status"] == "READY":
            ready_script_paths.append(str(script_path))
            ready_commands.append(command_line)
        else:
            review_script_paths.append(str(script_path))
            review_commands.append(command_line)

    ready_scripts_path = manifests_dir / "quantification_scripts.ready.txt"
    ready_scripts_path.write_text(
        "\n".join(ready_script_paths) + ("\n" if ready_script_paths else ""),
        encoding="utf-8",
    )

    review_scripts_path = manifests_dir / "quantification_scripts.review_required.txt"
    review_scripts_path.write_text(
        "\n".join(review_script_paths) + ("\n" if review_script_paths else ""),
        encoding="utf-8",
    )

    ready_commands_path = manifests_dir / "quantification_commands.ready.txt"
    ready_commands_path.write_text(
        "\n".join(ready_commands) + ("\n" if ready_commands else ""),
        encoding="utf-8",
    )

    review_commands_path = manifests_dir / "quantification_commands.review_required.txt"
    review_commands_path.write_text(
        "\n".join(review_commands) + ("\n" if review_commands else ""),
        encoding="utf-8",
    )

    runner_path = manifests_dir / "run_quantification_parallel.sh"
    master_log = manifests_dir / "run_quantification_parallel.log"
    runner_path.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env bash
            set -euo pipefail

            SCRIPT_LIST={shlex.quote(str(ready_scripts_path))}
            PARALLEL_JOBS="${{PARALLEL_JOBS:-{args.parallel_jobs}}}"
            MASTER_LOG={shlex.quote(str(master_log))}

            if [[ ! -s "$SCRIPT_LIST" ]]; then
                echo "No ready quantification scripts found in $SCRIPT_LIST"
                exit 0
            fi

            mkdir -p "$(dirname "$MASTER_LOG")"
            exec > >(tee -a "$MASTER_LOG") 2>&1

            echo "[INFO] Ready script list: $SCRIPT_LIST"
            echo "[INFO] Running up to $PARALLEL_JOBS samples concurrently"

            tr '\\n' '\\0' < "$SCRIPT_LIST" | xargs -0 -r -P "$PARALLEL_JOBS" -I {{}} bash "{{}}"
            """
        ),
        encoding="utf-8",
    )
    runner_path.chmod(0o755)

    return {
        "ready_scripts": ready_scripts_path,
        "review_scripts": review_scripts_path,
        "ready_commands": ready_commands_path,
        "review_commands": review_commands_path,
        "runner": runner_path,
    }


def write_run_manifest(manifests_dir, args, organ_dir, finaldata_dir):
    run_manifest_path = manifests_dir / "quantification_run_manifest.tsv"
    rows = [
        ("organ_dir", str(organ_dir)),
        ("finaldata_dir", str(finaldata_dir)),
        ("rseqc_bed", args.rseqc_bed),
        ("annotation_gtf", args.annotation_gtf),
        ("infer_experiment_bin", args.infer_experiment_bin),
        ("featurecounts_bin", args.featurecounts_bin),
        ("stringtie_bin", args.stringtie_bin),
        ("samtools_bin", args.samtools_bin),
        ("featurecounts_threads", str(args.featurecounts_threads)),
        ("stringtie_threads", str(args.stringtie_threads)),
        ("parallel_jobs", str(args.parallel_jobs)),
        ("strandedness_threshold", str(args.strandedness_threshold)),
        ("default_strand", str(args.default_strand)),
        ("run_strandedness_inference", str(args.run_strandedness_inference)),
        ("reuse_existing_strandedness", str(args.reuse_existing_strandedness)),
        ("run_featurecounts", str(args.run_featurecounts)),
        ("run_stringtie", str(args.run_stringtie)),
    ]
    with run_manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["key", "value"])
        writer.writerows(rows)
    return run_manifest_path


def print_summary(samples):
    status_counts = Counter(sample["status"] for sample in samples)
    mode_counts = Counter((sample["mode"], sample["resolved_layout"]) for sample in samples if sample["status"] == "READY")
    print(f"Total finaldata RNAseq folders inspected: {len(samples)}")
    print(f"Status counts: {dict(status_counts)}")
    if mode_counts:
        print(
            "Ready mode/layout counts: "
            + ", ".join(
                f"{mode}/{layout}={count}" for (mode, layout), count in sorted(mode_counts.items())
            )
        )


def run_parallel(runner_path):
    subprocess.run(["bash", str(runner_path)], check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate strandedness inference, featureCounts quantification scripts, and "
            "StringTie quantification scripts "
            "for finaldata/RNAseq sample folders."
        )
    )
    parser.add_argument("organ_dir", type=Path, help="Organ directory such as /home/.../data/RPFdb/pancreas")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run ready quantification scripts after generation.",
    )
    parser.add_argument(
        "--skip-strandedness-inference",
        action="store_true",
        help="Skip infer_experiment.py and use --default-strand directly.",
    )
    parser.add_argument(
        "--reuse-existing-strandedness",
        action="store_true",
        help=(
            "If strandedness outputs already exist in each sample's strandedness directory, "
            "reuse them instead of re-running infer_experiment.py."
        ),
    )
    parser.add_argument("--rseqc-bed", default=DEFAULT_RSEQC_BED)
    parser.add_argument("--annotation-gtf", default=DEFAULT_ANNOTATION_GTF)
    parser.add_argument("--infer-experiment-bin", default=DEFAULT_INFER_EXPERIMENT_BIN)
    parser.add_argument("--featurecounts-bin", default=DEFAULT_FEATURECOUNTS_BIN)
    parser.add_argument("--stringtie-bin", default=DEFAULT_STRINGTIE_BIN)
    parser.add_argument("--samtools-bin", default=DEFAULT_SAMTOOLS_BIN)
    parser.add_argument("--featurecounts-threads", type=int, default=8)
    parser.add_argument("--stringtie-threads", type=int, default=8)
    parser.add_argument("--parallel-jobs", type=int, default=4)
    parser.add_argument("--strandedness-threshold", type=float, default=0.8)
    parser.add_argument("--default-strand", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument(
        "--skip-featurecounts",
        action="store_true",
        help="Generate and run strandedness + StringTie only, without featureCounts.",
    )
    parser.add_argument(
        "--skip-stringtie",
        action="store_true",
        help="Generate and run strandedness + featureCounts only, without StringTie.",
    )
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--request-backoff-seconds", type=float, default=1.5)
    return parser.parse_args()


def main():
    args = parse_args()
    args.organ_dir = args.organ_dir.expanduser().resolve()
    args.run_strandedness_inference = not args.skip_strandedness_inference
    args.reuse_existing_strandedness = args.reuse_existing_strandedness
    args.run_featurecounts = not args.skip_featurecounts
    args.run_stringtie = not args.skip_stringtie

    organ_dir, finaldata_dir, manifests_dir, samples = build_samples(args)
    command_files = generate_command_files(samples, manifests_dir, args)

    plan_manifest_path = manifests_dir / "quantification_plan.tsv"
    write_manifest(
        plan_manifest_path,
        [
            {
                "riboseq_gsm": sample["riboseq_gsm"],
                "rnaseq_gsm": sample["rnaseq_gsm"],
                "rnaseq_sample_id": sample["rnaseq_sample_id"],
                "bam_path": str(sample["bam_path"]) if sample["bam_path"] else "",
                "bai_path": str(sample["bai_path"]) if sample["bai_path"] else "",
                "mode": sample["mode"],
                "resolved_layout": sample["resolved_layout"],
                "status": sample["status"],
                "note": sample["note"],
                "sample_dir": str(sample["sample_dir"]),
                "featurecounts_dir": str(sample["featurecounts_dir"]),
                "stringtie_dir": str(sample["stringtie_dir"]),
                "script_path": sample["script_path"],
                "mode_source": sample["mode_source"],
            }
            for sample in samples
        ],
    )
    run_manifest_path = write_run_manifest(manifests_dir, args, organ_dir, finaldata_dir)

    print_summary(samples)
    print(plan_manifest_path)
    print(run_manifest_path)
    print(command_files["ready_scripts"])
    print(command_files["review_scripts"])
    print(command_files["ready_commands"])
    print(command_files["review_commands"])
    print(command_files["runner"])

    if args.run:
        run_parallel(command_files["runner"])


if __name__ == "__main__":
    main()
