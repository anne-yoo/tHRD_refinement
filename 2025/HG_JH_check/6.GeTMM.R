###GeTMM ######

library(edgeR)

x <- read.delim("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/quantification/SR_GeTMM_input.txt", header = TRUE, row.names = 1,check.names = FALSE)
rpk <- (x[,2:ncol(x)]/x[,1])
x <- x[, -1]
group <- c(rep("A",ncol(x)))

rpk.norm <- DGEList(counts=rpk,group=group)
rpk.norm <- calcNormFactors(rpk.norm)
norm.counts.rpk_edger <- cpm(rpk.norm)

write.table(norm.counts.rpk_edger, file = "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/quantification/SR_transcript_GeTMM.txt", sep = "\t", quote = FALSE, row.names = TRUE)




########## readcount to TMM ###############
x <- read.delim("/home/jiye/jiye/nanopore/2025finaldata/quantification/gene_count_matrix_final.csv", sep=',',header=TRUE, row.names = 1,check.names = FALSE)
x[is.na(x)] <- 0
#x <- x[ , -1]
group <- c(rep("A",ncol(x)))
x.norm.edger <- DGEList(counts=x,group=group)
x.norm.edger <- calcNormFactors(x.norm.edger)
norm.counts.edger <- cpm(x.norm.edger)
write.table(norm.counts.edger, file = "/home/jiye/jiye/nanopore/2025finaldata/quantification/LR_gene_TMM.txt", sep = "\t", quote = FALSE, row.names = TRUE)

################ CPM #######################
x <- read.delim("/home/jiye/jiye/nanopore/2025finaldata/SRLR_comparison/LR_gene_count_matrix.csv", sep=',',header=TRUE, row.names = 1,check.names = FALSE)
x[is.na(x)] <- 0
cpmresult <- cpm(x, normalized.lib.sizes=FALSE)
write.table(cpmresult, file = "/home/jiye/jiye/nanopore/2025finaldata/SRLR_comparison/LR_gene_CPM.txt", sep = "\t", quote = FALSE, row.names = TRUE)


