#!/usr/bin/env bash

set -euo pipefail

DATASET_DIR="${DATASET_DIR:-${1:-}}"
if [[ -z "$DATASET_DIR" ]]; then
    printf 'Usage: %s <dataset_output_dir>\n' "$0" >&2
    exit 1
fi

METADATA_DIR="${METADATA_DIR:-$DATASET_DIR/flattened_metadata}"
MAP_TSV="${MAP_TSV:-$METADATA_DIR/sample_srr_map.tsv}"
PREFETCH_LIST="${PREFETCH_LIST:-$METADATA_DIR/prefetch_srr_list.txt}"

SRA_TOOLKIT_BIN="${SRA_TOOLKIT_BIN:-/home/jiye/jiye/darkproteome/data/nuORFdb/riboseq/sratoolkit.3.2.0-centos_linux64/bin}"
PREFETCH_BIN="${PREFETCH_BIN:-$SRA_TOOLKIT_BIN/prefetch}"
FASTERQ_DUMP_BIN="${FASTERQ_DUMP_BIN:-$SRA_TOOLKIT_BIN/fasterq-dump}"

SRA_DIR="${SRA_DIR:-$DATASET_DIR/sra_cache}"
FASTQ_DIR="${FASTQ_DIR:-$DATASET_DIR/fastq}"
FINAL_FASTQ_DIR="${FINAL_FASTQ_DIR:-$FASTQ_DIR/finalfastq_forbam}"
TMP_DIR="${TMP_DIR:-$DATASET_DIR/fasterq_tmp}"
MISSING_FINALFASTQ_REPORT="${MISSING_FINALFASTQ_REPORT:-$METADATA_DIR/finalfastq_missing_samples.tsv}"

THREADS="${THREADS:-8}"
COMPRESS_THREADS="${COMPRESS_THREADS:-$THREADS}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_MERGE="${SKIP_MERGE:-0}"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        printf 'Missing required command: %s\n' "$cmd" >&2
        exit 1
    fi
}

require_exec() {
    local path="$1"
    if [[ ! -x "$path" ]]; then
        printf 'Missing executable: %s\n' "$path" >&2
        exit 1
    fi
}

require_file() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        printf 'Missing required file: %s\n' "$path" >&2
        exit 1
    fi
}

has_pigz() {
    command -v pigz >/dev/null 2>&1
}

compress_files_in_place() {
    if has_pigz; then
        pigz -p "$COMPRESS_THREADS" -f "$@"
    else
        gzip -f "$@"
    fi
}

compress_stream_to_file() {
    local output_path="$1"
    if has_pigz; then
        pigz -p "$COMPRESS_THREADS" -c > "$output_path"
    else
        gzip -c > "$output_path"
    fi
}

normalize_single_end_fastq_artifacts() {
    local srr="$1"
    local plain_fastq="$FASTQ_DIR/${srr}.fastq"
    local gz_fastq="$FASTQ_DIR/${srr}.fastq.gz"
    local candidate

    if [[ ! -f "$plain_fastq" ]]; then
        shopt -s nullglob
        local plain_candidates=("$FASTQ_DIR/${srr}"*.fastq)
        shopt -u nullglob

        if (( ${#plain_candidates[@]} == 1 )); then
            candidate="${plain_candidates[0]}"
            if [[ "$candidate" != "$plain_fastq" ]]; then
                log "Normalizing single-end output for $srr from $(basename "$candidate")"
                mv "$candidate" "$plain_fastq"
            fi
        fi
    fi

    if [[ ! -f "$gz_fastq" ]]; then
        shopt -s nullglob
        local gz_candidates=("$FASTQ_DIR/${srr}"*.fastq.gz)
        shopt -u nullglob

        if (( ${#gz_candidates[@]} == 1 )); then
            candidate="${gz_candidates[0]}"
            if [[ "$candidate" != "$gz_fastq" ]]; then
                log "Normalizing single-end output for $srr from $(basename "$candidate")"
                mv "$candidate" "$gz_fastq"
            fi
        fi
    fi
}

has_gz_fastq_for_srr() {
    local srr="$1"
    normalize_single_end_fastq_artifacts "$srr"
    [[ -f "$FASTQ_DIR/${srr}.fastq.gz" ]] || [[ -f "$FASTQ_DIR/${srr}_1.fastq.gz" ]]
}

resolve_prefetch_target() {
    local srr="$1"
    local srr_dir="$SRA_DIR/$srr"
    local candidate

    if [[ -d "$srr_dir" ]]; then
        shopt -s nullglob
        local candidates=(
            "$srr_dir/$srr.sra"
            "$srr_dir/$srr.sralite"
            "$srr_dir/$srr.sralite.1"
            "$srr_dir/$srr".*
            "$srr_dir"/*
        )
        shopt -u nullglob

        for candidate in "${candidates[@]}"; do
            if [[ -f "$candidate" ]]; then
                printf '%s\n' "$candidate"
                return 0
            fi
        done
    fi

    shopt -s nullglob
    local root_candidates=(
        "$SRA_DIR/$srr.sra"
        "$SRA_DIR/$srr.sralite"
        "$SRA_DIR/$srr.sralite.1"
        "$SRA_DIR/$srr".*
    )
    shopt -u nullglob

    for candidate in "${root_candidates[@]}"; do
        if [[ -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    printf 'Could not locate downloaded SRA file for %s under %s\n' "$srr" "$SRA_DIR" >&2
    return 1
}

link_single_sample_fastq() {
    local sample_id="$1"
    local srr="$2"
    local source_fastq="$FASTQ_DIR/${srr}.fastq.gz"
    local final_fastq="$FINAL_FASTQ_DIR/${sample_id}.fastq.gz"

    [[ -f "$source_fastq" ]] || {
        printf 'Missing single-end FASTQ for %s\n' "$srr" >&2
        exit 1
    }

    rm -f "$final_fastq"
    ln -s "$source_fastq" "$final_fastq"
}

link_paired_sample_fastqs() {
    local sample_id="$1"
    local srr="$2"
    local source_r1="$FASTQ_DIR/${srr}_1.fastq.gz"
    local source_r2="$FASTQ_DIR/${srr}_2.fastq.gz"
    local final_r1="$FINAL_FASTQ_DIR/${sample_id}_1.fastq.gz"
    local final_r2="$FINAL_FASTQ_DIR/${sample_id}_2.fastq.gz"

    [[ -f "$source_r1" && -f "$source_r2" ]] || {
        printf 'Missing paired-end FASTQ for %s\n' "$srr" >&2
        exit 1
    }

    rm -f "$final_r1" "$final_r2"
    ln -s "$source_r1" "$final_r1"
    ln -s "$source_r2" "$final_r2"
}

merge_single_sample_fastqs() {
    local sample_id="$1"
    shift

    local inputs=()
    local srr
    for srr in "$@"; do
        [[ -f "$FASTQ_DIR/${srr}.fastq.gz" ]] || {
            printf 'Missing single-end FASTQ for %s\n' "$srr" >&2
            exit 1
        }
        inputs+=("$FASTQ_DIR/${srr}.fastq.gz")
    done

    log "Merging single-end sample $sample_id"
    rm -f "$FINAL_FASTQ_DIR/${sample_id}.fastq.gz"
    zcat "${inputs[@]}" | compress_stream_to_file "$FINAL_FASTQ_DIR/${sample_id}.fastq.gz"
}

merge_paired_sample_fastqs() {
    local sample_id="$1"
    shift

    local inputs_r1=()
    local inputs_r2=()
    local srr
    for srr in "$@"; do
        [[ -f "$FASTQ_DIR/${srr}_1.fastq.gz" && -f "$FASTQ_DIR/${srr}_2.fastq.gz" ]] || {
            printf 'Missing paired-end FASTQ for %s\n' "$srr" >&2
            exit 1
        }
        inputs_r1+=("$FASTQ_DIR/${srr}_1.fastq.gz")
        inputs_r2+=("$FASTQ_DIR/${srr}_2.fastq.gz")
    done

    log "Merging paired sample $sample_id"
    rm -f "$FINAL_FASTQ_DIR/${sample_id}_1.fastq.gz" "$FINAL_FASTQ_DIR/${sample_id}_2.fastq.gz"
    zcat "${inputs_r1[@]}" | compress_stream_to_file "$FINAL_FASTQ_DIR/${sample_id}_1.fastq.gz"
    zcat "${inputs_r2[@]}" | compress_stream_to_file "$FINAL_FASTQ_DIR/${sample_id}_2.fastq.gz"
}

gzip_srr_fastqs() {
    local srr="$1"
    local files=()

    normalize_single_end_fastq_artifacts "$srr"

    if [[ -f "$FASTQ_DIR/${srr}.fastq" ]]; then
        files+=("$FASTQ_DIR/${srr}.fastq")
    fi

    shopt -s nullglob
    local split_files=("$FASTQ_DIR/${srr}_"*.fastq)
    shopt -u nullglob
    if (( ${#split_files[@]} > 0 )); then
        files+=("${split_files[@]}")
    fi

    if (( ${#files[@]} == 0 )); then
        printf 'No FASTQ files found for %s after fasterq-dump\n' "$srr" >&2
        exit 1
    fi

    compress_files_in_place "${files[@]}"
}

download_all_srrs() {
    require_exec "$PREFETCH_BIN"
    require_exec "$FASTERQ_DUMP_BIN"
    if has_pigz; then
        require_cmd pigz
    else
        require_cmd gzip
    fi
    require_file "$PREFETCH_LIST"

    mkdir -p "$SRA_DIR" "$FASTQ_DIR" "$TMP_DIR"

    while IFS= read -r srr || [[ -n "$srr" ]]; do
        [[ -n "$srr" ]] || continue

        if has_gz_fastq_for_srr "$srr"; then
            log "Skipping existing FASTQ for $srr"
            continue
        fi

        log "Prefetch $srr"
        "$PREFETCH_BIN" --output-directory "$SRA_DIR" "$srr"

        local local_sra_path
        local_sra_path="$(resolve_prefetch_target "$srr")"

        log "fasterq-dump $srr"
        mkdir -p "$TMP_DIR/$srr"
        "$FASTERQ_DUMP_BIN" \
            --split-files \
            --threads "$THREADS" \
            --outdir "$FASTQ_DIR" \
            --temp "$TMP_DIR/$srr" \
            "$local_sra_path"

        log "gzip $srr"
        gzip_srr_fastqs "$srr"
    done < "$PREFETCH_LIST"
}

prepare_final_fastqs_for_bam() {
    require_cmd python3
    require_cmd zcat
    if has_pigz; then
        require_cmd pigz
    else
        require_cmd gzip
    fi
    require_file "$MAP_TSV"

    mkdir -p "$FINAL_FASTQ_DIR"
    printf 'sample_id\tfinal_srrs\treason\n' > "$MISSING_FINALFASTQ_REPORT"
    local missing_count=0

    while IFS=$'\t' read -r sample_id final_srrs; do
        [[ -n "$sample_id" ]] || continue

        IFS=',' read -r -a srrs <<< "$final_srrs"
        local paired_count=0
        local single_count=0
        local srr

        for srr in "${srrs[@]}"; do
            normalize_single_end_fastq_artifacts "$srr"
            [[ -f "$FASTQ_DIR/${srr}_1.fastq.gz" ]] && paired_count=$((paired_count + 1))
            [[ -f "$FASTQ_DIR/${srr}.fastq.gz" ]] && single_count=$((single_count + 1))
        done

        if (( paired_count > 0 && single_count > 0 )); then
            printf '%s\t%s\tmixed_single_and_paired_layout\n' "$sample_id" "$final_srrs" >> "$MISSING_FINALFASTQ_REPORT"
            log "Skipping sample $sample_id due to mixed single/paired layout"
            missing_count=$((missing_count + 1))
            continue
        fi

        if (( paired_count == ${#srrs[@]} )); then
            if (( ${#srrs[@]} == 1 )); then
                log "Linking paired sample $sample_id"
                link_paired_sample_fastqs "$sample_id" "${srrs[0]}"
            else
                merge_paired_sample_fastqs "$sample_id" "${srrs[@]}"
            fi
            continue
        fi

        if (( single_count == ${#srrs[@]} )); then
            if (( ${#srrs[@]} == 1 )); then
                log "Linking single-end sample $sample_id"
                link_single_sample_fastq "$sample_id" "${srrs[0]}"
            else
                merge_single_sample_fastqs "$sample_id" "${srrs[@]}"
            fi
            continue
        fi

        printf '%s\t%s\tmissing_raw_fastq\n' "$sample_id" "$final_srrs" >> "$MISSING_FINALFASTQ_REPORT"
        log "Skipping sample $sample_id because raw FASTQ files are missing"
        missing_count=$((missing_count + 1))
    done < <(
        python3 - "$MAP_TSV" <<'PY'
import csv
import sys

map_tsv = sys.argv[1]
with open(map_tsv, encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    for row in reader:
        if row["status"] != "ok":
            continue
        print(f"{row['sample_id']}\t{row['final_srrs']}")
PY
    )

    if (( missing_count > 0 )); then
        log "Missing final FASTQ inputs for $missing_count samples. See $MISSING_FINALFASTQ_REPORT"
        return 1
    fi
}

main() {
    require_file "$MAP_TSV"
    require_file "$PREFETCH_LIST"

    log "DATASET_DIR=$DATASET_DIR"
    log "METADATA_DIR=$METADATA_DIR"
    log "FASTQ_DIR=$FASTQ_DIR"
    log "FINAL_FASTQ_DIR=$FINAL_FASTQ_DIR"
    log "MISSING_FINALFASTQ_REPORT=$MISSING_FINALFASTQ_REPORT"
    log "PREFETCH_BIN=$PREFETCH_BIN"
    log "FASTERQ_DUMP_BIN=$FASTERQ_DUMP_BIN"
    log "THREADS=$THREADS"
    log "COMPRESS_THREADS=$COMPRESS_THREADS"
    if has_pigz; then
        log "Compression tool=pigz"
    else
        log "Compression tool=gzip"
    fi

    if [[ "$SKIP_DOWNLOAD" != "1" ]]; then
        download_all_srrs
    else
        log "Skipping download phase"
    fi

    if [[ "$SKIP_MERGE" != "1" ]]; then
        prepare_final_fastqs_for_bam
    else
        log "Skipping final FASTQ preparation phase"
    fi

    log "Done"
}

main "$@"
