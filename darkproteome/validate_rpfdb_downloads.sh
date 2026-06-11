#!/usr/bin/env bash

set -euo pipefail

DATASET_DIR="${DATASET_DIR:-${1:-}}"
if [[ -z "$DATASET_DIR" ]]; then
    printf 'Usage: %s <dataset_output_dir>\n' "$0" >&2
    exit 1
fi

METADATA_DIR="${METADATA_DIR:-$DATASET_DIR/flattened_metadata}"
PREFETCH_LIST="${PREFETCH_LIST:-$METADATA_DIR/prefetch_srr_list.txt}"

SRA_TOOLKIT_BIN="${SRA_TOOLKIT_BIN:-/home/jiye/jiye/darkproteome/data/nuORFdb/riboseq/sratoolkit.3.2.0-centos_linux64/bin}"
VDB_VALIDATE_BIN="${VDB_VALIDATE_BIN:-$SRA_TOOLKIT_BIN/vdb-validate}"

SRA_DIR="${SRA_DIR:-$DATASET_DIR/sra_cache}"
VALIDATION_DIR="${VALIDATION_DIR:-$DATASET_DIR/validation_reports}"

WRITE_MD5="${WRITE_MD5:-0}"
RUN_VDB_VALIDATE="${RUN_VDB_VALIDATE:-1}"

SRA_REPORT="${SRA_REPORT:-$VALIDATION_DIR/sra_validation.tsv}"
SUMMARY_REPORT="${SUMMARY_REPORT:-$VALIDATION_DIR/validation_summary.txt}"

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

maybe_md5() {
    local path="$1"
    if [[ "$WRITE_MD5" != "1" ]]; then
        printf 'skipped'
        return 0
    fi

    md5sum "$path" | awk '{print $1}'
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

    return 1
}

configure_vdb_validate() {
    case "$RUN_VDB_VALIDATE" in
        1)
            require_exec "$VDB_VALIDATE_BIN"
            printf '1'
            ;;
        0)
            printf '0'
            ;;
        auto)
            if [[ -x "$VDB_VALIDATE_BIN" ]]; then
                printf '1'
            else
                printf '0'
            fi
            ;;
        *)
            printf 'RUN_VDB_VALIDATE must be one of: auto, 0, 1\n' >&2
            exit 1
            ;;
    esac
}

write_sra_report() {
    local do_vdb="$1"
    local total=0
    local ok=0
    local missing=0
    local error=0
    local skipped=0

    printf 'accession\tsra_path\tsize_bytes\tvdb_validate_status\tmd5\tnote\n' > "$SRA_REPORT"

    while IFS= read -r srr || [[ -n "$srr" ]]; do
        [[ -n "$srr" ]] || continue
        total=$((total + 1))

        local sra_path=''
        if ! sra_path="$(resolve_prefetch_target "$srr")"; then
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$srr" "missing" "0" "missing" "skipped" "sra_file_not_found" >> "$SRA_REPORT"
            missing=$((missing + 1))
            continue
        fi

        local size_bytes
        size_bytes="$(stat -c '%s' "$sra_path")"
        local md5_value
        md5_value="$(maybe_md5 "$sra_path")"

        if [[ "$do_vdb" == "1" ]]; then
            local validate_log
            validate_log="$(mktemp)"
            if "$VDB_VALIDATE_BIN" "$sra_path" >"$validate_log" 2>&1; then
                printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "$srr" "$sra_path" "$size_bytes" "ok" "$md5_value" "-" >> "$SRA_REPORT"
                ok=$((ok + 1))
            else
                local note
                note="$(tail -n 3 "$validate_log" | tr '\n' ' ' | sed 's/[[:space:]]\\+/ /g; s/[[:space:]]$//')"
                printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                    "$srr" "$sra_path" "$size_bytes" "error" "$md5_value" "$note" >> "$SRA_REPORT"
                error=$((error + 1))
            fi
            rm -f "$validate_log"
        else
            printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
                "$srr" "$sra_path" "$size_bytes" "skipped" "$md5_value" "vdb_validate_disabled_or_unavailable" >> "$SRA_REPORT"
            skipped=$((skipped + 1))
        fi
    done < "$PREFETCH_LIST"

    printf '%s\t%s\t%s\t%s\t%s\n' "$total" "$ok" "$missing" "$error" "$skipped"
}

write_summary() {
    local do_vdb="$1"
    local sra_counts="$2"

    local sra_total sra_ok sra_missing sra_error sra_skipped
    IFS=$'\t' read -r sra_total sra_ok sra_missing sra_error sra_skipped <<< "$sra_counts"

    {
        printf 'validation_time\t%s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
        printf 'dataset_dir\t%s\n' "$DATASET_DIR"
        printf 'metadata_dir\t%s\n' "$METADATA_DIR"
        printf 'sra_dir\t%s\n' "$SRA_DIR"
        printf 'run_vdb_validate\t%s\n' "$do_vdb"
        printf 'write_md5\t%s\n' "$WRITE_MD5"
        printf 'note\tmd5sum values are local manifests only; no remote checksum comparison is performed\n'
        printf '\n'
        printf '[sra]\n'
        printf 'total\t%s\n' "$sra_total"
        printf 'ok\t%s\n' "$sra_ok"
        printf 'missing\t%s\n' "$sra_missing"
        printf 'error\t%s\n' "$sra_error"
        printf 'skipped\t%s\n' "$sra_skipped"
    } > "$SUMMARY_REPORT"
}

main() {
    require_file "$PREFETCH_LIST"

    if [[ "$WRITE_MD5" == "1" ]]; then
        require_cmd md5sum
    fi

    mkdir -p "$VALIDATION_DIR"

    local do_vdb
    do_vdb="$(configure_vdb_validate)"

    log "DATASET_DIR=$DATASET_DIR"
    log "METADATA_DIR=$METADATA_DIR"
    log "SRA_DIR=$SRA_DIR"
    log "VALIDATION_DIR=$VALIDATION_DIR"
    log "RUN_VDB_VALIDATE=$RUN_VDB_VALIDATE -> $do_vdb"
    log "WRITE_MD5=$WRITE_MD5"

    local sra_counts
    sra_counts="$(write_sra_report "$do_vdb")"
    write_summary "$do_vdb" "$sra_counts"

    log "SRA report: $SRA_REPORT"
    log "Summary: $SUMMARY_REPORT"
}

main "$@"
