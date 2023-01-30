#saturn example
library(satuRn)
library(AnnotationHub)
library(ensembldb)
library(edgeR)
library(SummarizedExperiment)
library(ggplot2)
library(DEXSeq)
library(stageR)

data(Tasic_counts_vignette)
data(Tasic_metadata_vignette)

ah <- AnnotationHub() # load the annotation resource.
all <- query(ah, "EnsDb") # query for all available EnsDb databases
ahEdb <- all[["AH75036"]] # for Mus musculus (choose correct release date)
txs <- transcripts(ahEdb)

# Get the transcript information in correct format
txInfo <- as.data.frame(matrix(data = NA, nrow = length(txs), ncol = 2))
colnames(txInfo) <- c("isoform_id", "gene_id")
txInfo$isoform_id <- txs$tx_id
txInfo$gene_id <- txs$gene_id
rownames(txInfo) <- txInfo$isoform_id

# Remove transcripts that are the only isoform expressed of a certain gene
rownames(Tasic_counts_vignette) <- sub("\\..*", "", 
                                       rownames(Tasic_counts_vignette)) 

# remove transcript version identifiers
txInfo <- txInfo[txInfo$isoform_id %in% rownames(Tasic_counts_vignette), ]
txInfo <- subset(txInfo, 
                 duplicated(gene_id) | duplicated(gene_id, fromLast = TRUE))

Tasic_counts_vignette <- Tasic_counts_vignette[which(
  rownames(Tasic_counts_vignette) %in% txInfo$isoform_id), ]

#feature level filtering using edgeR
filter_edgeR <- filterByExpr(Tasic_counts_vignette,
                             design = NULL,
                             group = Tasic_metadata_vignette$brain_region,
                             lib.size = NULL,
                             min.count = 10,
                             min.total.count = 30,
                             large.n = 20,
                             min.prop = 0.7
)
table(filter_edgeR)

Tasic_counts_vignette <- Tasic_counts_vignette[filter_edgeR, ]

# Update txInfo according to the filtering procedure
txInfo <- txInfo[which(
  txInfo$isoform_id %in% rownames(Tasic_counts_vignette)), ]

# remove txs that are the only isoform expressed within a gene (after filtering)
txInfo <- subset(txInfo, 
                 duplicated(gene_id) | duplicated(gene_id, fromLast = TRUE))
Tasic_counts_vignette <- Tasic_counts_vignette[which(rownames(
  Tasic_counts_vignette) %in% txInfo$isoform_id), ]

# satuRn requires the transcripts in the rowData and 
# the transcripts in the count matrix to be in the same order.
txInfo <- txInfo[match(rownames(Tasic_counts_vignette), txInfo$isoform_id), ]

# Create a design method
Tasic_metadata_vignette$group <- paste(Tasic_metadata_vignette$brain_region, 
                                       Tasic_metadata_vignette$cluster, 
                                       sep = ".")

# Generate SummarizedExperiment
sumExp <- SummarizedExperiment::SummarizedExperiment(
  assays = list(counts = Tasic_counts_vignette),
  colData = Tasic_metadata_vignette,
  rowData = txInfo
)

metadata(sumExp)$formula <- ~ 0 + as.factor(colData(sumExp)$group)
sumExp


# Fit quasi-binomial generalized linear models models with fitDTU of satuRn
system.time({
  sumExp <- satuRn::fitDTU(
    object = sumExp,
    formula = ~ 0 + group,
    parallel = FALSE,
    BPPARAM = BiocParallel::bpparam(),
    verbose = TRUE
  )
})

rowData(sumExp)[["fitDTUModels"]]$"ENSMUST00000037739"

# Test for DTU: Set up contrast matrix
group <- as.factor(Tasic_metadata_vignette$group)
design <- model.matrix(~ 0 + group) # construct design matrix
colnames(design) <- levels(group)

L <- matrix(0, ncol = 2, nrow = ncol(design)) # initialize contrast matrix
rownames(L) <- colnames(design)
colnames(L) <- c("Contrast1", "Contrast2")

L[c("VISp.L5_IT_VISp_Hsd11b1_Endou","ALM.L5_IT_ALM_Tnc"),1] <-c(1,-1)
L[c("VISp.L5_IT_VISp_Hsd11b1_Endou","ALM.L5_IT_ALM_Tmem163_Dmrtb1"),2] <-c(1,-1)
L # contrast matrix

#perform the DTU test
sumExp <- satuRn::testDTU(
  object = sumExp,
  contrasts = L,
  sort = FALSE
)

#visualize
group1 <- colnames(sumExp)[colData(sumExp)$group == 
                             "VISp.L5_IT_VISp_Hsd11b1_Endou"]
group2 <- colnames(sumExp)[colData(sumExp)$group == 
                             "ALM.L5_IT_ALM_Tnc"]

plots <- satuRn::plotDTU(
  object = sumExp,
  contrast = "Contrast1",
  groups = list(group1, group2),
  coefficients = list(c(0, 0, 1), c(0, 1, 0)),
  summaryStat = "model",
  transcripts = c("ENSMUST00000081554", 
                  "ENSMUST00000195963", 
                  "ENSMUST00000132062"),
  genes = NULL,
  top.n = 6
)

# to have same layout as in our paper
for (i in seq_along(plots)) {
  current_plot <- plots[[i]] +
    scale_fill_manual(labels = c("VISp", "ALM"), values = c("royalblue4", 
                                                            "firebrick")) +
    scale_x_discrete(labels = c("Hsd11b1_Endou", "Tnc"))
  
  print(current_plot)
}

# FDR post-processing

# transcript level p-values from satuRn
pvals <- rowData(sumExp)[["fitDTUResult_Contrast1"]]$empirical_pval

# compute gene level q-values
geneID <- factor(rowData(sumExp)$gene_id)
geneSplit <- split(seq(along = geneID), geneID)
pGene <- sapply(geneSplit, function(i) min(pvals[i]))
pGene[is.na(pGene)] <- 1
theta <- unique(sort(pGene))


# gene-level significance testing
q <- DEXSeq:::perGeneQValueExact(pGene, theta, geneSplit) 
qScreen <- rep(NA_real_, length(pGene))
qScreen <- q[match(pGene, theta)]
qScreen <- pmin(1, qScreen)
names(qScreen) <- names(geneSplit)

# prepare stageR input
tx2gene <- as.data.frame(rowData(sumExp)[c("isoform_id", "gene_id")])
colnames(tx2gene) <- c("transcript", "gene")

pConfirmation <- matrix(matrix(pvals),
                        ncol = 1,
                        dimnames = list(rownames(tx2gene), "transcript")
)


#remove genes with only one transcript
tx2gene <- subset(tx2gene, 
                  duplicated(gene) | duplicated(gene, fromLast = TRUE))

pConfirmation <- matrix(pConfirmation[row.names(pConfirmation) %in% tx2gene$transcript,],dimnames = list(rownames(tx2gene), "transcript") )


# create a stageRTx object
stageRObj <- stageR::stageRTx(
  pScreen = qScreen,
  pConfirmation = pConfirmation,
  pScreenAdjusted = TRUE,
  tx2gene = tx2gene
)



# perform the two-stage testing procedure
stageRObj <- stageR::stageWiseAdjustment(
  object = stageRObj,
  method = "dtu",
  alpha = 0.05,
  allowNA = TRUE
)

# retrieves the adjusted p-values from the stageRTx object
padj <- stageR::getAdjustedPValues(stageRObj,
                                   order = TRUE,
                                   onlySignificantGenes = FALSE
)

head(padj)


