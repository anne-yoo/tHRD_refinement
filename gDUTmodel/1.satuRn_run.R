library(satuRn)
library(AnnotationHub)
library(ensembldb)
library(edgeR)
library(SummarizedExperiment)
library(ggplot2)
library(DEXSeq)
library(stageR)


# tu.data <- read.table("/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/input/discovery/resistance/gDUT/comp_innate_variable_inputdata.csv", sep=",",header=TRUE, row.names=1,check.names = F) 
# tu.metadata <- read.table("/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/input/discovery/resistance/gDUT/comp_innate_variable_metadata.csv", sep=",",header = TRUE,check.names = F)
# tu.genedata <- read.table("/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/input/discovery/resistance/gDUT/comp_innate_variable_geneinfo.csv", sep=",",header = TRUE, check.names = F)

tu.data <- read.table("/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/input_DUTDEG/innate/DUT_innate_stable_countdata.txt", sep="\t",header=TRUE, row.names=1,check.names = F) 
tu.metadata <- read.table("/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/input_DUTDEG/innate/DUT_innate_metadata.txt", sep="\t", header = TRUE,check.names = F)
tu.genedata <- read.table("/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/input_DUTDEG/innate/DUT_innate_stable_geneinfo.txt", sep="\t",header = TRUE,check.names = F)


colnames(tu.genedata) <- c("isoform_id","gene_id")


#Remove transcripts that are the only isoform expressed of a certain gene

tu.genedata <- tu.genedata[tu.genedata$isoform_id %in% rownames(tu.data), ]
tu.genedata <- subset(tu.genedata, 
                      duplicated(gene_id) | duplicated(gene_id, fromLast = TRUE))

tu.data <- tu.data[which(
  rownames(tu.data) %in% tu.genedata$isoform_id), ]


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

L[c("pre_R","pre_NR"),1] <-c(1,-1)


#perform the DTU test
sum.Exp <- satuRn::testDTU(
  object = sum.Exp,
  contrasts = L,
  sort = FALSE
)

head(rowData(sum.Exp)[["fitDTUResult_Contrast1"]]) 

############ Add log2FC ##############
Exp.t = as.data.frame(rowData(sum.Exp)[["fitDTUResult_Contrast1"]][, c("pval","empirical_pval","regular_FDR","empirical_FDR")])
Exp.t$isoform_id <- rownames(Exp.t)
Exp.t$gene_id <- tu.genedata$gene_id

Exp.count = cbind(tu.genedata, tu.data)
row.names(Exp.count) = NULL

# Per-group  mean
tu.metadata$group <- as.factor(tu.metadata$group)

Exp.mean = as.data.frame(sapply( levels(tu.metadata$group), 
                                 function(lvl) {rowMeans(Exp.count[, 3:ncol(Exp.count)][, tu.metadata$group == lvl, drop = FALSE])} ))

# log2 fold change in expression
Exp.log2fc = log2(Exp.mean[2]/Exp.mean[1])
colnames(Exp.log2fc) = "log2fc"
rownames(Exp.log2fc) = Exp.count$isoform_id


Exp.log2fc$isoform_id <- rownames(Exp.log2fc)
final.data <- merge(Exp.t, Exp.log2fc, by = "isoform_id")




####### SAVE #########
###write.csv(rowData(sum.Exp)[["fitDTUResult_Contrast1"]], "/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/output/discovery/resistance/comp_innate_variable_satuRnresult.csv", row.names = TRUE)

write.csv(final.data, "/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/result_DUTDEG/innate/DUT_innate_stable_SatuRnResult.txt", row.names = FALSE)


