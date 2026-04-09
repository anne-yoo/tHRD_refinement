###### PCA !!! DEG vs. DTU data frame with sample info######
#### sample info: R vs. NR 

library(ggplot2)
library(ggfortify)
library(dplyr)
library(tidyverse)
library(broom)
library(FactoMineR)
library(factoextra)
library(psych)

library(extrafont)

font_import()
loadfonts(device = "pdf")


deg <- read.csv('//home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM_symbol.txt', header = TRUE, sep = "\t", row.names = 1)
#deg <- read.csv('/home/jiye/jiye/copycomparison/gDUTresearch/DESeq2/response/onlydeg_countdata.csv', header = TRUE, sep = ",")
#degsample <- read.csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/input_DUTDEG/innate/DEG_innate_metadata.txt', header = TRUE, sep = "\t")
degsample <- read.csv('//home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.csv', header = TRUE, sep = ",", row.names=2)

degsample <- degsample %>%
  mutate(interval_binary = ifelse(interval > 180, "long", "short"))

############ MAKE TMM from EdgeR ################
library(edgeR)

y=DGEList(counts=deg)
y=calcNormFactors(y)
tmm_mat=cpm(y,normalized.lib.sizes=TRUE)

#deg_mat <- tmm %>% 
#  column_to_rownames("X") %>% 
#  as.matrix() %>% 
#  t() %>% 
#  as.data.frame()
 
deg_mat <- tmm_mat %>% 
  t() %>% 
  as.data.frame()

tmm_df <- as.data.frame(tmm_mat)
#write.csv(tmm_df, '/home/jiye/jiye/copycomparison/gDUTresearch/data/202306_newRNAseq/processed/final_202308_discovery_gene_TMM.txt',row.names = TRUE) 
##################################################

deg_n <- as.data.frame(deg[,1:(ncol(deg)-1)])
filtered_deg <- data.frame(deg_n[rowSums(deg_n > 0) >= ncol(deg_n)*0.6, ])

#deg_mat <- as.data.frame(t(filtered_deg))
deg_mat <- as.data.frame(t(deg_n))

group <- degsample$group
treatment <- degsample$treatment
response <- degsample$response
line <- degsample$line
line_binary <- degsample$line_binary
exonicread <- degsample$exonic_read
interval <- degsample$interval_binary

deg_mat$group <- group
deg_mat$treatment <- treatment
deg_mat$response <- response
deg_mat$line <- line
deg_mat$line_binary <- line_binary
deg_mat$exonicread <- exonicread
deg_mat$interval <- interval

deg_mat$response <- factor(deg_mat$response, levels = c(0, 1))

df <- deg_mat[,1:(ncol(deg_mat)-7)]

##plot PC1 and PC2
#autoplot(prcomp(df), data = deg_mat, colour = 'group',label = TRUE) + scale_color_manual(values = c("#578DD5","#D86868")) +theme_bw()
#autoplot(prcomp(deg_mat), data = deg_mat, label = TRUE) +theme_bw()

plot = autoplot(prcomp(df), data = deg_mat, colour = 'exonicread', label = TRUE, frame = TRUE, frame.type = 'norm') + theme_bw()
plot
# + theme(
#   text = element_text(family = "Arial"),
#   axis.title = element_text(family = "Arial"),
#   axis.text = element_text(family = "Arial")
# )

ggsave(filename = "/home/jiye/jiye/copycomparison/gDUTresearch/202309_analysis/genelevel_pca_interval.pdf", plot = plot, dpi = 300, width = 12, height = 7)


##variance explained by PCs
deg_pca <- prcomp(df)

## get genes that matches to top PCs
rotation_absolute <- as.data.frame(abs(deg_pca$rotation))  # take the absolute value of the loadings
max_loading_index <- apply(rotation_absolute, 1, which.max)  # find the index of the maximum value in each row
max_loading_pc <- paste0("PC", max_loading_index)  # convert index to PC

result <- data.frame(Gene = rownames(deg_pca$rotation), PC = max_loading_pc)

gene_names_top_pc <- unique(subset(result, PC %in% c("PC2"))$Gene)

write(gene_names_top_pc, file = '/home/jiye/jiye/copycomparison/gDUTresearch/202310_analysis/pca/gene_PC2.txt')

pc1_2 = rotation_absolute[,c('PC1','PC2')]
write.csv(pc1_2, '/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/pca/gene_fil_pc1_2.txt',row.names = TRUE)

##get contribution of each variable with FactoMineR / factoextra
#fviz_cos2(deg_pca, choice = "var", axes = 1:2, sort.val="desc")
pcaresult <- get_pca_var(deg_pca)
cos2_pcaresult <- pcaresult$cos2

#write.csv(cos2_pcaresult, '/home/jiye/jiye/copycomparison/gDUTresearch/DESeq2/response/PCA_whole_cos2.txt',row.names = TRUE)


##PC variance with ggfortify
pc_eigenvalues <- deg_pca$sdev^2
pc_eigenvalues <- tibble(PC = factor(1:length(pc_eigenvalues)), 
                         variance = pc_eigenvalues) %>% 
  # add a new column with the percent variance
  mutate(pct = variance/sum(variance)*100) %>% 
  # add another column with the cumulative variance explained
  mutate(pct_cum = cumsum(pct))

# print the result
pc_eigenvalues

pc_eigenvalues %>% 
  ggplot(aes(x = PC)) +
  ggtitle("gene expression data")+
  geom_col(aes(y = pct)) +
  geom_line(aes(y = pct_cum, group = 1)) + 
  geom_point(aes(y = pct_cum)) +
  labs(x = "Principal component", y = "Fraction variance explained") +
  coord_cartesian(xlim = c(1, 30))+
  theme_bw()

# PC1 & PC2 genes?
pc_loadings <- deg_pca$rotation
pc_loadings <- pc_loadings %>% 
  as_tibble(rownames = "gene")

top_genes <- pc_loadings %>% 
  # select only the PCs we are interested in
  dplyr::select(gene, PC1, PC2) %>%
  # convert to a "long" format
  pivot_longer(matches("PC"), names_to = "PC", values_to = "loading") %>% 
  # for each PC
  group_by(PC) %>% 
  # arrange by descending order of loading
  arrange(desc(abs(loading))) %>% 
  # take the 10 top rows
  slice(1:10) %>% 
  # pull the gene column as a vector
  pull(gene) %>% 
  # ensure only unique genes are retained
  unique()

top_genes
top_loadings <- pc_loadings %>% 
  dplyr::filter(gene %in% top_genes)

ggplot(data = top_loadings) +
  geom_segment(aes(x = 0, y = 0, xend = PC1, yend = PC2), 
               arrow = arrow(length = unit(0.1, "in")),
               colour = "brown") +
  geom_text(aes(x = PC1, y = PC2, label = gene),
            nudge_y = 0.005, size = 3) +
  scale_x_continuous(expand = c(0.02, 0.02))

## tidying the PCA results

pc_eigen <- tidy(deg_pca, matrix = "eigenvalues") # PC variances (eigen values)
tidy(deg_pca, matrix = "loadings") # variable loading 

#write.csv(pc_eigen, '/home/jiye/jiye/copycomparison/gDUTresearch/DESeq2/response/PCA_whole_pcresult.txt',row.names = TRUE)






################################################
################################################
################   DTU   #######################
################################################
################################################




#dtu <- read.csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/final_202306_discovery_transcript_exp.txt', header = TRUE, sep = "\t")
#dtu <- read.csv('/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/input/discovery/response/onlydtu_inputdata.csv', header = TRUE, sep = ",")
#dtusample <- read.csv('/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/input/discovery/response/metadata.csv', header = TRUE, sep = ",")

dtu <- read.csv('//home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_transcript_exp.txt', header = TRUE, sep = "\t", row.names = 1)
dtusample <- read.csv('//home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.csv', header = TRUE, sep = ",", row.names=2)

# Assuming dut_input is a data frame in R
filtered_dtu <- data.frame(dtu[rowSums(dtu > 0) >= ncol(dtu)*0.6, ])
#rownames(dtu) <- NULL

#dtu_mat <- dtu %>% 
#  column_to_rownames("gene_ENST") %>% 
#  as.matrix() %>% 
#  t() %>% 
#  as.data.frame()

dtu_mat <- filtered_dtu %>% 
  as.matrix() %>% 
  t() %>% 
  as.data.frame()

dtu_mat <- dtu %>% 
  as.matrix() %>% 
  t() %>% 
  as.data.frame()

group <- degsample$group
treatment <- degsample$treatment
response <- degsample$response
line <- degsample$line
line_binary <- degsample$line_binary
exonicread <- degsample$exonic_read
interval <- degsample$interval_binary

dtu_mat$group <- group
dtu_mat$treatment <- treatment
dtu_mat$response <- response
dtu_mat$line <- line
dtu_mat$line_binary <- line_binary
dtu_mat$exonicread <- exonicread
dtu_mat$response <- factor(dtu_mat$response, levels = c(0, 1))
dtu_mat$interval <- interval

df2 <- dtu_mat[,1:(ncol(dtu_mat)-7)]

##plot PC1 and PC2
#autoplot(prcomp(dtu_mat), data = dtu_mat, label=TRUE) + scale_color_manual(values = c("#578DD5")) +theme_bw()
plot2 = autoplot(prcomp(df2), data = dtu_mat, label=TRUE, color='interval', frame = TRUE, frame.type = 'norm') + theme_bw()
ggsave(filename = "/home/jiye/jiye/copycomparison/gDUTresearch/202309_analysis/fil_transcriptlevel_pca_interval.pdf", plot = plot2, dpi = 300, width = 12, height = 7)

##variance explained by PCs
dtu_pca <- prcomp(df2)


## get genes that matches to top PCs
rotation_absolute2 <- as.data.frame(abs(dtu_pca$rotation))  # take the absolute value of the loadings
max_loading_index2 <- apply(rotation_absolute2, 1, which.max)  # find the index of the maximum value in each row
max_loading_pc2 <- paste0("PC", max_loading_index2)  # convert index to PC


result2 <- data.frame(Transcript = rownames(dtu_pca$rotation), PC = max_loading_pc2)

trans_names_top_pc <- unique(subset(result2, PC %in% c("PC1", "PC2","PC3","PC4","PC5")) $Transcript)

write(trans_names_top_pc, file = '/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/pca/transcript_topPC.txt')

pc1_2 = rotation_absolute2[,c('PC1','PC2')]
write.csv(pc1_2, '/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/pca/transcript_pc1_2.txt',row.names = TRUE)




##get contribution of each variable with FactoMineR / factoextra
#fviz_cos2(deg_pca, choice = "var", axes = 1:2, sort.val="desc")
pcaresult2 <- get_pca_var(dtu_pca)
cos2_pcaresult2 <- pcaresult2$cos2

#write.csv(cos2_pcaresult2, '/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/input/discovery/response/PCA_whole_cos2.csv',row.names = TRUE)


##PCvariance with ggfortify
pc_eigenvalues2 <- dtu_pca$sdev^2
pc_eigenvalues2 <- tibble(PC = factor(1:length(pc_eigenvalues2)), 
                         variance = pc_eigenvalues2) %>% 
  # add a new column with the percent variance
  mutate(pct = variance/sum(variance)*100) %>% 
  # add another column with the cumulative variance explained
  mutate(pct_cum = cumsum(pct))

# print the result
pc_eigenvalues2

pc_eigenvalues2 %>% 
  ggplot(aes(x = PC)) +
  ggtitle("transcript expression data")+
  geom_col(aes(y = pct)) +
  geom_line(aes(y = pct_cum, group = 1)) + 
  geom_point(aes(y = pct_cum)) +
  labs(x = "Principal component", y = "Fraction variance explained") +
  coord_cartesian(xlim = c(1, 30))+
  theme_bw()

# PC1 & PC2 genes?
pc_loadings2 <- dtu_pca$rotation
pc_loadings2 <- pc_loadings2 %>% 
  as_tibble(rownames = "transcript")

top_genes2 <- pc_loadings2 %>% 
  # select only the PCs we are interested in
  dplyr::select(transcript, PC1, PC2) %>%
  # convert to a "long" format
  pivot_longer(matches("PC"), names_to = "PC", values_to = "loading") %>% 
  # for each PC
  group_by(PC) %>% 
  # arrange by descending order of loading
  arrange(desc(abs(loading))) %>% 
  # take the 10 top rows
  slice(1:10) %>% 
  # pull the gene column as a vector
  pull(transcript) %>% 
  # ensure only unique genes are retained
  unique()

top_genes2

## tidying the PCA results

pc_eigen2 <- tidy(dtu_pca, matrix = "eigenvalues") # PC variances (eigen values)
tidy(dtu_pca, matrix = "loadings") # variable loading 

#write.csv(pc_eigen2, '/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/input/discovery/response/PCA_whole_pcresult.csv',row.names = TRUE)


set.seed(1)
par(mfrow=c(1,1))
dtu_mat$group <- I(as.character(dtu_mat$group))
g_colors <- list("Responder" = "red", "Non-Responder" = "blue")
autoplot(kmeans(df2, 3), data = dtu_mat, frame=TRUE)
autoplot(kmeans(df2, 2), data = dtu_mat, frame=TRUE, label = TRUE, label.size = 3) + scale_fill_manual(values=g_colors)

