library(satuRn)
library(AnnotationHub)
library(ensembldb)
library(edgeR)
library(SummarizedExperiment)
library(ggplot2)
library(DEXSeq)
library(stageR)


tu.data <- read.table("/home/jiye/jiye/copycomparison/OC_transcriptome/satuRn_inputs/new/stable_NR_TUdata.csv", sep=",",header=TRUE, row.names=1,check.names = F) 
tu.metadata <- read.table("/home/jiye/jiye/copycomparison/OC_transcriptome/satuRn_inputs/new/stable_NR_metadata.csv", sep=",",header = TRUE,check.names = F)
tu.genedata <- read.table("/home/jiye/jiye/copycomparison/OC_transcriptome/satuRn_inputs/new/stable_NR_geneinfo.csv", sep=",",header = TRUE, row.names=1,check.names = F)

colnames(tu.genedata) <- c("isoform_id","gene_id")

#Remove transcripts that are the only isoform expressed of a certain gene

tu.genedata <- tu.genedata[tu.genedata$isoform_id %in% rownames(tu.data), ]
tu.genedata <- subset(tu.genedata, 
                 duplicated(gene_id) | duplicated(gene_id, fromLast = TRUE))

tu.data <- tu.data[which(
  rownames(tu.data) %in% tu.genedata$isoform_id), ]


# 
# #feature level filtering using edgeR
# filter_edgeR <- filterByExpr(tu.data,
#                              design = NULL,
#                              group = tu.metadata$group,
#                              lib.size = colSums(tu.data),
#                              min.count = 0.01,
#                              min.total.count = 0.00001,
#                              large.n = 20,
#                              min.prop = 0.2
# )
# 
# table(filter_edgeR)
# 
# tu.data <- tu.data[filter_edgeR, ]
# 
# # Update tu.genedata according to the filtering procedure
# tu.genedata <- tu.genedata[which(
#   tu.genedata$isoform_id %in% rownames(tu.data)), ]
# 
# # remove txs that are the only isoform expressed within a gene (after filtering)
# tu.genedata <- subset(tu.genedata, 
#                  duplicated(gene_id) | duplicated(gene_id, fromLast = TRUE))
# tu.data <- tu.data[which(rownames(
#   tu.data) %in% tu.genedata$isoform_id), ]
# 
# # satuRn requires the transcripts in the rowData and 
# # the transcripts in the count matrix to be in the same order.
# tu.genedata <- tu.genedata[match(rownames(tu.data), tu.genedata$isoform_id), ]


# Generate SummarizedExperiment

sum.Exp <- SummarizedExperiment::SummarizedExperiment(
  assays = list(counts = tu.data),
  colData = tu.metadata,
  rowData = tu.genedata
)

metadata(sum.Exp)$formula <- ~ 0 + as.factor(colData(sum.Exp)$group)
sum.Exp

# Fit quasi-binomial generalized linear models models with fitDTU of satuRn
system.time({
  sum.Exp <- satuRn::fitDTU(
    object = sum.Exp,
    formula = ~ 0 + group,
    parallel = FALSE,
    BPPARAM = BiocParallel::bpparam(),
    verbose = TRUE
  )
})

rowData(sum.Exp)[["fitDTUModels"]]

# Test for DTU: Set up contrast matrix
group <- as.factor(tu.metadata$group)
design <- model.matrix(~ 0 + group) # construct design matrix
colnames(design) <- levels(group)
L <- matrix(0, ncol = 1, nrow = ncol(design)) # initialize contrast matrix
rownames(L) <- colnames(design)
colnames(L) <- c("Contrast1")
L[c("post","pre"),1] <-c(1,-1)


#perform the DTU test
sum.Exp <- satuRn::testDTU(
  object = sum.Exp,
  contrasts = L,
  sort = FALSE
)

head(rowData(sum.Exp)[["fitDTUResult_Contrast1"]]) 

write.csv(rowData(sum.Exp)[["fitDTUResult_Contrast1"]], "/home/jiye/jiye/copycomparison/OC_transcriptome/satuRn_results/new/stable_NR_DTUresult.csv", row.names = TRUE)



