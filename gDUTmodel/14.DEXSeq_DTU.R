### DTU with DEXSeq !!! ###

library(EnhancedVolcano)
library(DEXSeq)
library(stageR)
library(ggplot2)
library(DRIMSeq)
library(DESeq2)

tu.data <- read.table("/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/input/discovery/response/inputdata.csv", sep=",",header=TRUE, row.names=1,check.names = F) 
names(tu.data) <- gsub("-", ".", names(tu.data))

tu.metadata <- read.table("/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/input/discovery/response/metadata.csv", sep=",",header = TRUE,check.names = F)
colnames(tu.metadata) <- c("sample_id","group")
tu.metadata$sample_id <- gsub("-", ".", tu.metadata$sample_id)

tu.genedata <- read.table("/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/input/discovery/response/geneinfo.csv", sep=",",header = TRUE, check.names = F, row.names=1)
colnames(tu.genedata) <- c("transcript","gene")

tu.data = tu.data[rowSums(tu.data) > 0,]

genedata.sub = tu.genedata[match(rownames(tu.data), tu.genedata$transcript),]
counts = data.frame(gene_id = genedata.sub$gene, feature_id = genedata.sub$transcript, tu.data)

d = DRIMSeq::dmDSdata(counts = counts, samples = tu.metadata)

## We will filter the object using dmFilter before running DEXSeq workflow -> 일단 필터 안 함

n = nrow(tu.metadata)
n.small = min(table(tu.metadata$group))
d = DRIMSeq::dmFilter(d,
                     min_samps_feature_expr = n.small, min_feature_expr = 3,
                     min_samps_feature_prop = n.small, min_feature_prop = 0.001,
                     min_samps_gene_expr = n, min_gene_expr = 3)

#### DEXSeq DTU ####

countData = round(as.matrix(counts(d)[,-c(1:2)]))

dxd = DEXSeq::DEXSeqDataSet(countData = countData, sampleData = tu.metadata, design = ~sample + exon + group:exon, featureID = counts(d)$feature_id, groupID = counts(d)$gene_id)

system.time({
  dxd = estimateSizeFactors(dxd)
  dxd = estimateDispersions(dxd)
  dxd = testForDEU(dxd, reducedModel = ~sample + exon)
})

dxr = DEXSeqResults(dxd, independentFiltering = FALSE)

qval = perGeneQValue(dxr)
dxr.g = data.frame(gene = names(qval), qval)
dxr.t = as.data.frame(dxr[, c("featureID","groupID","pvalue")])

dex.norm = cbind(as.data.frame(geneIDs(dxd)),as.data.frame(featureIDs(dxd)), as.data.frame(counts(dxd, normalized = TRUE)))
colnames(dex.norm) = c("groupID", "featureID", as.character(colData(dxd)$sample_id))
row.names(dex.norm) = NULL

# Per-group normalised mean
tu.metadata$group <- as.factor(tu.metadata$group)

dex.mean = as.data.frame(sapply( levels(tu.metadata$group), 
                                 function(lvl) {rowMeans(dex.norm[, 3:ncol(dex.norm)][, tu.metadata$group == lvl, drop = FALSE])} ))

# log2 fold change in expression
dex.log2fc = log2(dex.mean[2]/dex.mean[1])
colnames(dex.log2fc) = "log2fc"
rownames(dex.log2fc) = dex.norm$featureID

# Merge to create result data
dexData = cbind(dex.norm[,1:2], dex.mean, dex.norm[, 3:ncol(dex.norm)])
##dexData = merge(annoData, dexData, by.x = c("GeneID","TranscriptID"), by.y = c("groupID","featureID"))
dexData = dexData[order(dexData$groupID, dexData$featureID),]

dex.log2fc$featureID <- rownames(dex.log2fc)
final.data <- merge(dxr.t, dex.log2fc, by = "featureID")

#write.csv(final.data, "/home/jiye/jiye/copycomparison/gDUTresearch/DTUDEGcomp/DEXSeq_DTU_result.txt", row.names = FALSE)




