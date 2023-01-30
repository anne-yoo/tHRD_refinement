library(pheatmap) ## for heatmap generation
library(tidyverse) ## for data wrangling
library(ggplotify) ## to convert pheatmap to ggplot2
library(heatmaply) ## for constructing interactive heatmap

mat <- read.csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/forpsiheatmap_AF.txt', header=TRUE,row.names=1,check.names=FALSE, sep='\t')

#change nan to zero
mat <- replace(mat, is.na(mat), 0)

#create data frame for annotations
sample_info <- read.csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/sample_info.txt', header=TRUE, row.names = 1, check.names=FALSE, sep='\t')

hm <- pheatmap(mat,scale="row", annotation_col = sample_info,
         annotation_colors=list(group=c(pre="orange", post="black")),
         color=colorRampPalette(c("navy", "white", "red"))(50),cutree_cols=2, cutree_rows=2,
         main="psi values of pre / post samples",
         fontsize=9, cellwidth=35, cellheight=10.25)
