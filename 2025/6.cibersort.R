library(CIBERSORT)
library(tibble)
library(dplyr)
library(ImmuCellAI)

tpm <- read.csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/83_POLO_transcript_exp.txt', header = TRUE, sep="\t", row.names = 1)
sample_info <-read.csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/83_POLO_clinicalinfo.txt', sep = '\t', header = TRUE)

tpm_unique <- tpm %>% distinct(target_gene, .keep_all = TRUE)
rownames(tpm_unique) <- tpm_unique$target_gene
tpm_unique$target_gene <- NULL
tpm_unique <- as.matrix(tpm_unique)
#response_1_samples <- sample_info$sample_full[sample_info$response == 1]
#response_0_samples <- sample_info$sample_full[sample_info$response == 0]

#colnames(tpm_unique) <- gsub("\\.", "-", colnames(tpm_unique))

# Filter columns based on response
#tpm <- as.matrix(tpm_unique)
#AR <- tpm[, colnames(tpm) %in% response_1_samples]
#IR <- tpm[, colnames(tpm) %in% response_0_samples]

#ARpost <- AR[, seq(1, ncol(AR), by = 2)]
#ARpre <- AR[, seq(2, ncol(AR), by = 2)]
#IRpost <- IR[, seq(1, ncol(IR), by = 2)]
#IRpre <- IR[, seq(2, ncol(IR), by = 2)]

sig_matrix <- system.file("extdata", "LM22.txt", package = "CIBERSORT")
results <- cibersort(sig_matrix, tpm_unique)
write.table(results, file = "/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/POLO_cybersort.txt", sep = "\t", quote = FALSE, )


res <- ImmuCellAI_new(sample = tpm_unique, ###ImmuCellAI
                      data_type = "rnaseq",
                      group_tag = 0,
                      response_tag = 0)

write.table(res$Sample_abundance, file = "/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/POLO_immucellai.txt", sep = "\t", quote = FALSE, )


####### Cibersort on validation cohort #################

tpm <- read.csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_gene_TPM.txt', header = TRUE, sep="\t", row.names = 1)
sample_info <-read.csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_validation_clinicalinfo_new.txt', sep = '\t', header = TRUE)
geneinfo <- read.csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM_symbol.txt', header = TRUE, sep="\t", row.names = 1)
geneinfo <- geneinfo[,ncol(geneinfo), drop = FALSE]

tpm_unique <- cbind(tpm,geneinfo)
tpm_unique <- tpm_unique %>% distinct(Gene.Symbol, .keep_all = TRUE)
rownames(tpm_unique) <- tpm_unique$Gene.Symbol
tpm_unique$Gene.Symbol <- NULL
colnames(tpm_unique) <- gsub("\\.", "-", colnames(tpm_unique))
tpm_unique <- as.matrix(tpm_unique)

sig_matrix <- system.file("extdata", "LM22.txt", package = "CIBERSORT")
results <- cibersort(sig_matrix, tpm_unique)

write.table(results, file = "/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/cibersort/valcohort_cibersort.txt", sep = "\t", quote = FALSE, )



#######ImmuCellAI ############################
library(matrixStats) 
library(ImmuCellAI)
res <- ImmuCellAI_new(sample = tpm_unique,
                      data_type = "rnaseq",
                      group_tag = 0,
                      response_tag = 0)

write.table(res$Sample_abundance, file = "/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/cibersort/valcohort_immucellai.txt", sep = "\t", quote = FALSE, )



####### Stemness Index ###########
library(org.Hs.eg.db)
library(biomaRt)
library(dplyr)
tpm <- read.csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM.txt', header = TRUE, sep="\t", row.names = 1)

ensembl <- useEnsembl("ensembl", dataset = "hsapiens_gene_ensembl")
ensembl_ids <- rownames(tpm)
ensembl_ids_clean <- sub("\\..*", "", ensembl_ids)

gene_map <- getBM(
  filters = "ensembl_gene_id",
  attributes = c("ensembl_gene_id", "entrezgene_id"),
  values = ensembl_ids_clean,
  mart = ensembl
)

gene_map <- gene_map[!duplicated(gene_map$ensembl_gene_id), ]
gene_map_clean <- gene_map[!is.na(gene_map$entrezgene_id), ]
gene_map_unique <- gene_map_clean[!duplicated(gene_map_clean$entrezgene_id), ]
df_entrez <- tpm[gene_map_unique$ensembl_gene_id, ]
rownames(df_entrez) <- gene_map_unique$entrezgene_id
colnames(df_entrez) <- gsub("\\.", "-", colnames(df_entrez))

AR <- df_entrez[, colnames(df_entrez) %in% response_1_samples]
IR <- df_entrez[, colnames(df_entrez) %in% response_0_samples]

ARpost <- AR[, seq(1, ncol(AR), by = 2)]
ARpre <- AR[, seq(2, ncol(AR), by = 2)]
IRpost <- IR[, seq(1, ncol(IR), by = 2)]
IRpre <- IR[, seq(2, ncol(IR), by = 2)]

library(StemnessIndex)
label <- StemnessIndex(AR)
write.table(label, file = "/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/stemness/AR_stemnessindex.txt", sep = "\t", quote = FALSE, )
