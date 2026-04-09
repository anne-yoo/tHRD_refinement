### DTU with DRIMseq #####
library(DRIMSeq)
library(dplyr)
library(stageR)
library(tidyr)
library(GenomicRanges)

### ---------------------------------------------------------
### 1. Load data
### ---------------------------------------------------------

# counts: transcript x sample matrix
# clin: clinical info with columns: sample_full, sample_id, treatment

counts <- read.table("/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/transcript_count_matrix.csv", sep=",",header=TRUE, row.names=1,check.names = F) 
clin <- read.table("/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost_80_clinicalinfo.txt", sep="\t", header=TRUE,check.names = F) 
iso_info <- read.table("/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/gencode_majorminorlist.txt", sep="\t", header=TRUE,check.names = F)
clin <- clin %>%
  filter(sample_full %in% colnames(counts)) %>%
  arrange(match(sample_full, colnames(counts)))

stopifnot(all(clin$sample_full == colnames(counts)))

tx2gene <- iso_info %>%
  dplyr::select(transcriptid, genename) %>%
  dplyr::rename(feature_id = transcriptid, gene_id = genename)

### ---------------------------------------------------------
### Function to run DRIMSeq DTU for a given group (AR or IR)
### ---------------------------------------------------------

run_dtu <- function(group_label) {
  
  message("Running DTU for group: ", group_label)
  
  # Subset clinical data
  clin_sub <- clin %>%
    filter(response == group_label)
  
  # Subset count matrix
  counts_sub <- counts[, clin_sub$sample_full, drop = FALSE]
  
  # Ensure alignment
  clin_sub <- clin_sub %>%
    arrange(match(sample_full, colnames(counts_sub)))
  
  stopifnot(all(clin_sub$sample_full == colnames(counts_sub)))
  
  # Melt count matrix
  count_df <- as.data.frame(counts_sub)
  count_df$feature_id <- rownames(counts_sub)
  
  count_long <- count_df %>%
    pivot_longer(
      cols = -feature_id,
      names_to = "sample_full",
      values_to = "count"
    ) %>%
    left_join(clin_sub, by = "sample_full") %>%
    left_join(tx2gene, by = "feature_id") %>%
    select(sample_full, sample_id, gene_id, feature_id, count, treatment)
  
  # DRIMSeq samples table
  samples_d <- clin_sub %>%
    mutate(
      sample_id = factor(sample_id),
      treatment = factor(treatment, levels = c("pre", "post"))
    )
  
  # Create dmDSdata object
  d <- dmDSdata(
    counts = count_long[, c("sample_full", "gene_id", "feature_id", "count")],
    samples = samples_d
  )
  
  # Filter
  d <- dmFilter(
    d,
    min_samps_feature_proportion = 0.5,
    min_feature_proportion = 0.05,
    min_samps_gene_proportion = 0.5,
    min_gene_proportion = 0.05
  )
  
  # Fit model (paired design)
  d <- dmPrecision(d, design = ~ sample_id + treatment)
  d <- dmFit(d, design = ~ sample_id + treatment)
  
  # Test post vs pre
  d <- dmTest(d, coef = "treatmentpost")
  
  # Extract results
  res_gene <- results(d, level = "gene")
  res_tx   <- results(d, level = "feature")
  
  # stageR correction
  gene_p <- res_gene$pvalue
  names(gene_p) <- res_gene$gene_id
  
  tx_p <- res_tx$pvalue
  names(tx_p) <- res_tx$feature_id
  
  tx2gene_map <- data.frame(
    transcript = res_tx$feature_id,
    gene = res_tx$gene_id
  )
  
  stage_obj <- stageRTx(
    geneLevel = gene_p,
    transcriptLevel = tx_p,
    tx2gene = tx2gene_map
  )
  
  stage_res <- stageWiseAdjustment(stage_obj)
  
  res_gene$padj_stageR <- getAdjustedPValues(stage_res, "gene")
  res_tx$padj_stageR   <- getAdjustedPValues(stage_res, "transcript")
  
  outdir <- "/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/DEGDUT"
  prefix <- "DRIMseq"
  
  # Save results
  write.csv(
    res_gene,
    file = file.path(outdir, paste0(prefix, "_DTU_gene_", group_label, ".csv")),
    row.names = FALSE
  )
  
  write.csv(
    res_tx,
    file = file.path(outdir, paste0(prefix, "_DTU_transcript_", group_label, ".csv")),
    row.names = FALSE
  )
  
  
  return(list(gene = res_gene, tx = res_tx))
}

### ---------------------------------------------------------
### Run for AR (response = 1) and IR (response = 0)
### ---------------------------------------------------------

res_AR <- run_dtu(group_label = 1)
res_IR <- run_dtu(group_label = 0)
