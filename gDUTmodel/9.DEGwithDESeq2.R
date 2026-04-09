## DEG with DESeq2 ##
## 1. acquired resistance: R pre vs. NR pre
## 2. innate resistance: R pre vs. R post
#################################################

library("DESeq2")
library(ggplot2)

countData <- read.csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_readcount.txt', header = TRUE, sep = "\t",check.names = F)
countData <- as.data.frame(countData[rowSums(countData > 0) >= ncol(countData)*0.6, ])

metaData <- read.csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.csv', header = TRUE, sep = ",", check.names = F, row.names=2)
metaData <- metaData[,c("sample_id","treatment")]

dds <- DESeqDataSetFromMatrix(countData=countData, 
                              colData=metaData, 
                              design= ~ sample_id + treatment, tidy = TRUE)

dds <- DESeq(dds)
resultnames = resultsNames(dds)
res <- DESeq2::results(dds)

res <- res[order(res$padj),]

DGE.results <- DESeq2::results(dds,
                       independentFiltering = TRUE,
                       alpha = 0.05)


######## IND ############
ddsind <- DESeqDataSetFromMatrix(countData=countData, 
                              colData=metaData, 
                              design= ~ treatment, tidy = TRUE)

ddsind <- DESeq(ddsind)
resultnames2 = resultsNames(ddsind)
res_ind <- DESeq2::results(ddsind)

res_ind <- res_ind[order(res_ind$padj),]

DGE.results_ind <- DESeq2::results(ddsind,
                               independentFiltering = TRUE,
                               alpha = 0.05)

#reset par
par(mfrow=c(1,1))
# Make a basic volcano plot
with(res_ind, plot(log2FoldChange, -log10(pvalue), pch=20, main="DEG pre vs. post",ylim=c(0,13), xlim=c(-3,3)))

# Add colored points: blue if padj<0.01, red if log2FC>1 and padj<0.01)
with(subset(res_ind, padj<.05 ), points(log2FoldChange, -log10(pvalue), pch=20, col="blue"))
with(subset(res_ind, padj<.05 & abs(log2FoldChange)>2), points(log2FoldChange, -log10(pvalue), pch=20, col="red"))

#MA plot
DESeq2::plotMA(DGE.results_ind, alpha = 0.05, main = "DEG pre vs. post")

#save
write.csv(res,'/home/jiye/jiye/copycomparison/gDUTresearch/202309_analysis/DEG/DESeq2result_paired.csv',row.names = TRUE)
write.csv(res_ind,'/home/jiye/jiye/copycomparison/gDUTresearch/202309_analysis/DEG/DESeq2result_ind.csv',row.names = TRUE)

