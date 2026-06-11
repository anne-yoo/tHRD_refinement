#!/usr/bin/env bash
set -euo pipefail

: "${BASE_DIR:=/home/jiye/jiye/darkproteome/ORFstudy/pilot}"
: "${OUT_DIR:=/home/jiye/jiye/darkproteome/ORFstudy/pilot/pancreas8samples}"
: "${RNA_BASE:=/home/jiye/jiye/darkproteome/data/RPFdb/pancreas/finaldata/RNAseq}"
: "${GENOME_FA:=/home/jiye/jiye/darkproteome/ORFstudy/pilot/hg38.fa}"
: "${GENOME_FAI:=/home/jiye/jiye/darkproteome/ORFstudy/pilot/hg38.fa.fai}"
: "${BEDTOOLS:=/home/jiye/jiye/darkproteome/tools/bedtools2/bin/bedtools}"

METADATA="${OUT_DIR}/tables/pancreas8samples.metadata.tsv"
LOG="${OUT_DIR}/logs/setup_metadata.log"

mkdir -p "${OUT_DIR}/bigwig" "${OUT_DIR}/bed" "${OUT_DIR}/tables" "${OUT_DIR}/logs" "${OUT_DIR}/scripts"

echo "01_setup_metadata.sh"
echo "BASE_DIR=${BASE_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "RNA_BASE=${RNA_BASE}"
echo "GENOME_FA=${GENOME_FA}"
echo "GENOME_FAI=${GENOME_FAI}"
echo "BEDTOOLS=${BEDTOOLS}"
echo "metadata=${METADATA}"
echo "log=${LOG}"

{
    printf "sample\tbam\tbai\tread_layout\tlibrary_strand\n"
    for sample in GSM3395010 GSM3395011 GSM3395012 GSM3395013 GSM3395014 GSM3395015; do
        printf "%s\t%s/%s/%s.bam\t%s/%s/%s.bam.bai\tpaired_end\tunstranded\n" \
            "${sample}" "${RNA_BASE}" "${sample}" "${sample}" "${RNA_BASE}" "${sample}" "${sample}"
    done
    for sample in GSM5099832 GSM5099835; do
        printf "%s\t%s/%s/%s.bam\t%s/%s/%s.bam.bai\tunknown\tforward\n" \
            "${sample}" "${RNA_BASE}" "${sample}" "${sample}" "${RNA_BASE}" "${sample}" "${sample}"
    done
} > "${METADATA}"

: > "${LOG}"
warn_count=0

warn_missing() {
    local label="$1"
    local path="$2"
    if [[ ! -e "${path}" ]]; then
        printf "WARNING\tmissing %s\t%s\n" "${label}" "${path}" >> "${LOG}"
        warn_count=$((warn_count + 1))
    else
        printf "OK\t%s\t%s\n" "${label}" "${path}" >> "${LOG}"
    fi
}

warn_executable() {
    local label="$1"
    local path="$2"
    if [[ ! -x "${path}" ]]; then
        printf "WARNING\tmissing or non-executable %s\t%s\n" "${label}" "${path}" >> "${LOG}"
        warn_count=$((warn_count + 1))
    else
        printf "OK\t%s\t%s\n" "${label}" "${path}" >> "${LOG}"
    fi
}

warn_missing "input ORF table" "${BASE_DIR}/Pancreas.4caller.merged.2caller.tsv"
warn_missing "genome FASTA" "${GENOME_FA}"
warn_missing "genome FAI" "${GENOME_FAI}"
warn_executable "bedtools" "${BEDTOOLS}"

{
    read -r _header
    while IFS=$'\t' read -r sample bam bai read_layout library_strand; do
        [[ -n "${sample}" ]] || continue
        warn_missing "BAM for ${sample}" "${bam}"
        warn_missing "BAI for ${sample}" "${bai}"
        printf "OK\tread_layout for %s\t%s\n" "${sample}" "${read_layout}" >> "${LOG}"
        printf "OK\tlibrary_strand for %s\t%s\n" "${sample}" "${library_strand}" >> "${LOG}"
    done
} < "${METADATA}"

echo "Wrote ${METADATA}"
echo "Wrote ${LOG} with ${warn_count} warning(s)"
echo "01_setup_metadata.sh completed"
