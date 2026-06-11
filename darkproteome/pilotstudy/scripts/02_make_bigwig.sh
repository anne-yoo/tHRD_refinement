#!/usr/bin/env bash
set -euo pipefail

: "${OUT_DIR:=/home/jiye/jiye/darkproteome/ORFstudy/pilot/pancreas8samples}"
: "${MAX_SAMPLE_JOBS:=8}"
: "${BAMCOVERAGE_THREADS:=8}"

METADATA="${OUT_DIR}/tables/pancreas8samples.metadata.tsv"
BIGWIG_DIR="${OUT_DIR}/bigwig"
LOG_DIR="${OUT_DIR}/logs"
FORCE=0

usage() {
    echo "Usage: bash 02_make_bigwig.sh [--force]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            FORCE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

mkdir -p "${BIGWIG_DIR}" "${LOG_DIR}"

echo "02_make_bigwig.sh"
echo "OUT_DIR=${OUT_DIR}"
echo "metadata=${METADATA}"
echo "bigwig_dir=${BIGWIG_DIR}"
echo "force=${FORCE}"
echo "max_sample_jobs=${MAX_SAMPLE_JOBS}"
echo "bamCoverage_threads=${BAMCOVERAGE_THREADS}"

if [[ ! -f "${METADATA}" ]]; then
    echo "ERROR: metadata file not found: ${METADATA}" >&2
    exit 1
fi

if ! [[ "${MAX_SAMPLE_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: MAX_SAMPLE_JOBS must be a positive integer: ${MAX_SAMPLE_JOBS}" >&2
    exit 1
fi

if ! [[ "${BAMCOVERAGE_THREADS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: BAMCOVERAGE_THREADS must be a positive integer: ${BAMCOVERAGE_THREADS}" >&2
    exit 1
fi

if ! command -v bamCoverage >/dev/null 2>&1; then
    echo "ERROR: bamCoverage was not found in PATH" >&2
    exit 1
fi

run_bamcoverage() {
    local sample="$1"
    local bam="$2"
    local filter_strand="$3"
    local out_bw="$4"
    local log="$5"

    if [[ -e "${out_bw}" && "${FORCE}" -ne 1 ]]; then
        printf "SKIP\texists\t%s\n" "${out_bw}" >> "${log}"
        echo "Skipping existing BigWig: ${out_bw}"
        return 0
    fi

    local tmp_bw
    tmp_bw="${out_bw%.bw}.tmp.$$.bw"
    rm -f "${tmp_bw}"

    local filter_label
    local filter_args=()
    if [[ -n "${filter_strand}" ]]; then
        filter_label="${filter_strand}"
        filter_args=(--filterRNAstrand "${filter_strand}")
    else
        filter_label="none"
    fi

    {
        printf "RUN\tsample=%s\tfilterRNAstrand=%s\toutput=%s\n" "${sample}" "${filter_label}" "${out_bw}"
        if bamCoverage \
            -b "${bam}" \
            -o "${tmp_bw}" \
            --normalizeUsing CPM \
            --binSize 1 \
            -p "${BAMCOVERAGE_THREADS}" \
            "${filter_args[@]}"; then
            mv -f "${tmp_bw}" "${out_bw}"
            printf "DONE\t%s\n" "${out_bw}"
        else
            local rc=$?
            rm -f "${tmp_bw}"
            printf "ERROR\tbamCoverage failed\tstatus=%s\toutput=%s\n" "${rc}" "${out_bw}"
            return "${rc}"
        fi
    } >> "${log}" 2>&1
}

process_sample() {
    local sample="$1"
    local bam="$2"
    local bai="$3"
    local read_layout="$4"
    local library_strand="$5"
    local log
    local sense_filter
    local antisense_filter

    log="${LOG_DIR}/bamCoverage.${sample}.log"
    : > "${log}"
    printf "sample\t%s\nbam\t%s\nbai\t%s\nread_layout\t%s\nlibrary_strand\t%s\n" \
        "${sample}" "${bam}" "${bai}" "${read_layout}" "${library_strand}" >> "${log}"

    if [[ ! -f "${bam}" ]]; then
        printf "WARNING\tmissing BAM\t%s\n" "${bam}" >> "${log}"
        echo "Skipping ${sample}: missing BAM ${bam}"
        return 0
    fi
    if [[ ! -f "${bai}" ]]; then
        printf "WARNING\tmissing BAI\t%s\n" "${bai}" >> "${log}"
        echo "Skipping ${sample}: missing BAI ${bai}"
        return 0
    fi

    case "${library_strand}" in
        forward)
            sense_filter="forward"
            antisense_filter="reverse"
            run_bamcoverage "${sample}" "${bam}" "${sense_filter}" "${BIGWIG_DIR}/${sample}.sense.CPM.bw" "${log}"
            run_bamcoverage "${sample}" "${bam}" "${antisense_filter}" "${BIGWIG_DIR}/${sample}.antisense.CPM.bw" "${log}"
            ;;
        reverse)
            sense_filter="reverse"
            antisense_filter="forward"
            run_bamcoverage "${sample}" "${bam}" "${sense_filter}" "${BIGWIG_DIR}/${sample}.sense.CPM.bw" "${log}"
            run_bamcoverage "${sample}" "${bam}" "${antisense_filter}" "${BIGWIG_DIR}/${sample}.antisense.CPM.bw" "${log}"
            ;;
        unstranded|paired_end|paired-end|paired)
            run_bamcoverage "${sample}" "${bam}" "" "${BIGWIG_DIR}/${sample}.unstranded.CPM.bw" "${log}"
            ;;
        *)
            printf "ERROR\tunknown library_strand\t%s\n" "${library_strand}" >> "${log}"
            echo "ERROR: unknown library_strand for ${sample}: ${library_strand}" >&2
            return 1
            ;;
    esac

    echo "Finished ${sample}; log=${log}"
}

declare -a pids=()
declare -a labels=()
job_failure=0

wait_for_oldest_sample_job() {
    local pid="${pids[0]}"
    local label="${labels[0]}"

    if wait "${pid}"; then
        echo "Completed sample job: ${label}"
    else
        echo "ERROR: sample job failed: ${label}" >&2
        job_failure=1
    fi

    pids=("${pids[@]:1}")
    labels=("${labels[@]:1}")
}

queue_sample_job() {
    local sample="$1"
    local bam="$2"
    local bai="$3"
    local read_layout="$4"
    local library_strand="$5"

    while [[ "${#pids[@]}" -ge "${MAX_SAMPLE_JOBS}" ]]; do
        wait_for_oldest_sample_job
    done

    process_sample "${sample}" "${bam}" "${bai}" "${read_layout}" "${library_strand}" &
    pids+=("$!")
    labels+=("${sample}")
    echo "Queued sample job: ${sample}"
}

{
    read -r header
    IFS=$'\t' read -r -a header_cols <<< "${header}"
    has_read_layout=0
    if [[ "${#header_cols[@]}" -ge 5 && "${header_cols[3]}" == "read_layout" ]]; then
        has_read_layout=1
    fi

    while IFS=$'\t' read -r sample bam bai field4 field5 _extra; do
        [[ -n "${sample}" ]] || continue

        if [[ "${has_read_layout}" -eq 1 ]]; then
            read_layout="${field4}"
            library_strand="${field5}"
        else
            read_layout="unknown"
            library_strand="${field4}"
        fi

        queue_sample_job "${sample}" "${bam}" "${bai}" "${read_layout}" "${library_strand}"
    done
} < "${METADATA}"

while [[ "${#pids[@]}" -gt 0 ]]; do
    wait_for_oldest_sample_job
done

if [[ "${job_failure}" -ne 0 ]]; then
    echo "02_make_bigwig.sh failed; inspect ${LOG_DIR}/bamCoverage.*.log" >&2
    exit 1
fi

echo "02_make_bigwig.sh completed"
