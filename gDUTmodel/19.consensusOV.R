####consensusOV for OC subtyping####

library(consensusOV)
library(stringr)
library(Biobase)
library(genefu)
#tmm <- read.csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/202306_newRNAseq/processed/final_202308_discovery_gene_TMM.txt', header = TRUE, sep = ",",row.names = 1)

tmm_df <- read.csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM.txt', header = TRUE, sep="\t", row.names = 1)
sample_info <-read.csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep = '\t', header = TRUE)
tmm <- tmm_df
symbolid <- row.names(tmm_df)
symbol_ensg <- stringr::str_extract(symbolid, "^[^.]*")
colnames(tmm) <- gsub("\\.", "-", colnames(tmm))

response_1_samples <- sample_info$sample_full[sample_info$response == 1]
response_0_samples <- sample_info$sample_full[sample_info$response == 0]

# Filter columns based on response
df_response_1 <- tmm[, colnames(tmm) %in% response_1_samples]
df_response_0 <- tmm[, colnames(tmm) %in% response_0_samples]

########################
tmm <- df_response_1
########################

library(org.Hs.eg.db)

cols <- c("ENTREZID", "SYMBOL", "ENSEMBL", "GENENAME")
org <- AnnotationDbi::select(org.Hs.eg.db, keys=symbol_ensg, columns=cols, keytype="ENSEMBL")
org <- as.data.frame(org)

tmm$symbol_ensg <- symbol_ensg
tmm$geneid <- row.names(tmm)

merged_df <- merge(tmm, org, by.x = "symbol_ensg", by.y = "ENSEMBL")

df_unique <- merged_df[!duplicated(merged_df$ENTREZID), ]
df_clean <- na.omit(df_unique)
entrezid <- df_clean$ENTREZID
new_rownames <- paste0("geneid.", df_clean$ENTREZID)

# Assign the new rownames back to the dataframe
rownames(df_clean) <- new_rownames

df_clean <- df_clean[, 2:(ncol(df_clean)-4)]


inputmatrix <- as.matrix(df_clean)

################## START SUBTYPING #####################
Bentink.subtypes <- get.subtypes(inputmatrix, entrezid, method = "Bentink")
Konecny.subtypes <- get.subtypes(inputmatrix, entrezid, method = "Konecny")
Helland.subtypes <- get.subtypes(inputmatrix, entrezid, method = "Helland")
Verhaak.subtypes <- get.subtypes(inputmatrix, entrezid, method = "Verhaak")
Consensus.subtypes <- get.subtypes(inputmatrix, entrezid, method = "consensusOV")



################## PLOT!!!!!!!! #####################
#######
data_values <- Consensus.subtypes$consensusOV.subtypes
#######

sample_1 <- data_values[seq(1, length(data_values), by = 2)]
sample_2 <- data_values[seq(2, length(data_values), by = 2)]

data <- data.frame(sample_1 = as.factor(sample_1), sample_2 = as.factor(sample_2))
data <- data[c("sample_2", "sample_1")]

write.csv(data, "/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ConsensusOV_AR.txt", row.names = FALSE)

library(RColorBrewer)


p <- ggplot(data, aes(x = 1, xend = 2, y = as.numeric(sample_2), yend = as.numeric(sample_1))) +
  geom_segment(aes(color = sample_2), size = 1) + 
  geom_point(aes(y = as.numeric(sample_2), color = sample_2), size = 3) +
  geom_point(aes(x = 2, y = as.numeric(sample_1), color = sample_1), size = 3) +
  scale_x_continuous(name = "Samples", breaks = c(1, 2), labels = c("Pre-Treatment", "Post-Treatment")) +
  scale_y_continuous(name = "Ovarian Cancer Subtypes", breaks = 1:4, labels = levels(data$sample_2)) +
  scale_color_manual(values = c("#884A39", "#C38154", "#FFC26F","#F9E0BB")) +
  theme_minimal() +theme(
    legend.position = "none",
    axis.title.x = element_text(size = 11, face = "bold", color = "black"),
    axis.title.y = element_text(vjust = 5, size = 11, face = "bold", color = "black"),
    axis.text.x = element_text(size = 10, color = "black"),
    axis.text.y = element_text(size = 10, color = "black"),
    plot.margin = unit(c(1, 1, 1, 1), "cm")
    
  )

p

# Since we have four unique subtypes and they should be the same for source and target, we create a list of nodes for each
# We then append the list of target nodes to the source nodes to create a complete list
unique_subtypes <- unique(c(data$sample_1, data$sample_2))
nodes <- data.frame(name=c(unique_subtypes, unique_subtypes)) # duplicated for source and target

# Map the subtypes in sample_1 and sample_2 to their respective indices in the nodes data frame
# The source index will map to the first instance of the subtype, and the target to the second
data$source <- match(data$sample_1, nodes$name[1:4]) - 1
data$target <- match(data$sample_2, nodes$name[5:8]) + 3
# Create a placeholder column in 'data' for counting. Let's name it 'count'.
data$count <- 1

# Use aggregate to sum this 'count' for each source-target combination.
links <- aggregate(count ~ source + target, data = data, FUN = sum)

# Rename the aggregated column to 'value' as expected for sankey diagrams.
names(links)[which(names(links) == "count")] <- "value"
# Create links by counting the number of transitions between each pair of subtypes

# Assign 'group' based on whether the subtypes in sample_1 and sample_2 match
# This does not rely on adjusted indices but directly on subtype names
links$group <- ifelse(data$sample_1[match(paste(links$source, links$target), paste(data$source, data$target))] == data$sample_2[match(paste(links$source, links$target), paste(data$source, data$target))], "same", "diff")

my_color <- 'd3.scaleOrdinal() .domain(["DIF_consensus", "IMR_consensus", "MES_consensus", "PRO_consensus", "diff", "same"]) .range(["#EA5C2B", "#FF7F3F", "#F6D860", "#95CD41", "#F2D9BF", "#F6F3EE"])'

# Make the Network
sankey <- sankeyNetwork(Links = links, Nodes = nodes, Source = "source", 
                        Target = "target", Value = "value", NodeID = "name",
                        height=500, width=600,
                        fontSize = 16, nodeWidth = 30, colourScale=my_color, LinkGroup = "group")
sankey

ggsave(filename = "/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/AR_consensusOV.pdf", plot = p, dpi = 300, width = 9, height = 7)
