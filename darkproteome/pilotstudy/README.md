# Pancreas 8-Sample ORF Pilot Preprocessing

This folder contains step-by-step preprocessing scripts for an ORF-level pilot
study using 8 pancreas RNA-seq samples. The scripts are intentionally separate
so each output can be inspected before running the next step.

## Defaults

The scripts use these default paths:

```bash
BASE_DIR=/home/jiye/jiye/darkproteome/ORFstudy/pilot
OUT_DIR=/home/jiye/jiye/darkproteome/ORFstudy/pilot/pancreas8samples
RNA_BASE=/home/jiye/jiye/darkproteome/data/RPFdb/pancreas/finaldata/RNAseq
GENOME_FA=/home/jiye/jiye/darkproteome/ORFstudy/pilot/hg38.fa
GENOME_FAI=/home/jiye/jiye/darkproteome/ORFstudy/pilot/hg38.fa.fai
BEDTOOLS=/home/jiye/jiye/darkproteome/tools/bedtools2/bin/bedtools
```

Each value can be overridden with an environment variable before running a
script, for example:

```bash
export BASE_DIR=/path/to/pilot
export OUT_DIR=/path/to/pilot/pancreas8samples
```

Activate an environment that already has `samtools`, `bamCoverage`, `python3`,
`pandas`, and `biopython`.

`01_setup_metadata.sh` writes the first six samples as paired-end/unstranded
RNA-seq and the last two samples as forward-stranded RNA-seq.

## Run Order

Run scripts one by one:

```bash
cd darkproteome/pilotstudy/scripts

bash 01_setup_metadata.sh
bash 02_make_bigwig.sh
python3 03_parse_translated_orfs.py
python3 04_create_orf_groups.py
python3 05_extract_sequence_context.py
python3 06_qc_summary.py
```

Use `bash 02_make_bigwig.sh --force` only when existing BigWig files should be
regenerated.

By default, `02_make_bigwig.sh` runs up to 8 samples in parallel
(`MAX_SAMPLE_JOBS=8`). Each `bamCoverage` call still uses 8 threads by default
(`BAMCOVERAGE_THREADS=8`). Override these if the server needs fewer concurrent
jobs:

```bash
MAX_SAMPLE_JOBS=4 BAMCOVERAGE_THREADS=4 bash 02_make_bigwig.sh
```

`02_make_bigwig.sh` creates unstranded CPM BigWigs for the paired-end samples:

```text
${OUT_DIR}/bigwig/${sample}.unstranded.CPM.bw
```

For the two forward-stranded samples, it creates strand-aware BigWigs:

```text
${OUT_DIR}/bigwig/${sample}.sense.CPM.bw
${OUT_DIR}/bigwig/${sample}.antisense.CPM.bw
```

The expected BigWig count after a successful run is 10: six unstranded files
plus sense/antisense files for two forward-stranded samples.

## Transcript Context Module Feature Experiments

After preprocessing scripts `01_setup_metadata.sh` through `06_qc_summary.py`
finish, run these exploratory feature scripts one by one:

```bash
python3 01_start_context_analysis.py
python3 02_rna_coverage_context.py
python3 03_upstream_scanning_burden.py
python3 04_structure_context.py
python3 05_orf_positional_context.py
python3 06_integrated_feature_summary.py
python3 07_make_readme.py
```

The feature scripts use these defaults:

```bash
INPUT_DIR=/home/jiye/jiye/darkproteome/ORFstudy/pilot/pancreas8samples
FIG_DIR=/home/jiye/jiye/darkproteome/ORFstudy/pilot/figures
GENOME_FA=/home/jiye/jiye/darkproteome/ORFstudy/pilot/hg38.fa
```

Outputs are written under `${FIG_DIR}/tables`, `${FIG_DIR}/pdf`,
`${FIG_DIR}/png`, and `${FIG_DIR}/scripts`.

Additional sample-specific RNA coverage analyses:

```bash
python3 01_samplewise_metaplot.py
python3 02_detected_vs_not_detected_coverage.py
python3 03_detected_vs_not_detected_metaplot.py
python3 04_sample_specificity_summary.py
```

These scripts use the sample-level translated ORF table to distinguish
sample-specific detected and not-detected ORF/sample pairs.

## Coordinate Rule

ORF coordinates are genomic 0-based half-open coordinates and are used directly.
No transcript-to-genome conversion is performed.

For plus-strand ORFs, the start codon is `genome[start0:start0+3]`.
For minus-strand ORFs, the translation start codon is the reverse complement of
`genome[end0-3:end0]`.

Kozak positions follow:

```text
... -4 -3 -2 -1 CODON +4 +5 ...
```

For a plus-strand sequence context like `cgccATGGGC`, `minus3_base` is `g` and
`plus4_base` is `G`.
