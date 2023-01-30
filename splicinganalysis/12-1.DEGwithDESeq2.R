###DEG pre vs. post
library("DESeq2")
library(ggplot2)

countData <- read.csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/DESeq2/countdata.csv', header = TRUE, sep = ",")
metaData <- read.csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/DESeq2/metadata.csv', header = TRUE, sep = ",")

dds <- DESeqDataSetFromMatrix(countData=countData, 
                              colData=metaData, 
                              design=~group, tidy = TRUE)

dds <- DESeq(dds)
res <- results(dds)
res <- res[order(res$padj),]

#reset par
par(mfrow=c(1,1))
# Make a basic volcano plot
with(res, plot(log2FoldChange, -log10(pvalue), pch=20, main="Volcano plot", xlim=c(-3,3)))

# Add colored points: blue if padj<0.01, red if log2FC>1 and padj<0.05)
with(subset(res, padj<.01 ), points(log2FoldChange, -log10(pvalue), pch=20, col="blue"))
with(subset(res, padj<.01 & abs(log2FoldChange)>2), points(log2FoldChange, -log10(pvalue), pch=20, col="red"))

#save
write.csv(res, '/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/DESeq2/DEGresult.csv',row.names = TRUE)
