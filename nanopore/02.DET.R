library(DESeq2)

# Load your data
# Assuming your data is in a tab-separated file named 'readcounts.txt'
read_counts <- read.delim("/home/jiye/jiye/nanopore/FINALDATA/137_transcript_readcount.txt", header = TRUE, row.names = 1,check.names = FALSE)
colData <- read.delim("/home/jiye/jiye/nanopore/FINALDATA/batchcorrection/137_batch_type_info.txt", header = TRUE, row.names = 1,check.names = FALSE)

# Count the number of samples
num_samples <- ncol(read_counts)

# Calculate the 30% threshold
threshold <- 0.2 * num_samples

# Filter out transcripts not expressed in at least 40% of the samples
# Here, we assume "expression" means having a non-zero count
filtered_counts <- read_counts[rowSums(read_counts > 0) >= threshold, ]

# Create metadata
sample_info <- data.frame(
  condition = rep(c("Normal", "Tumor"), times = num_samples / 2), # assuming alternating N/T pairs
  pair = rep(1:(num_samples / 2), each = 2),
  batch = colData$batch # pairing each N with its respective T
)
sample_info$pair <- factor(sample_info$pair)
sample_info$condition <- factor(sample_info$condition)
sample_info$batch <- factor(sample_info$batch)

rownames(sample_info) <- colnames(filtered_counts)

# Check if everything is correct
head(sample_info)

# Create DESeq2 dataset
dds <- DESeqDataSetFromMatrix(
  countData = filtered_counts,
  colData = sample_info,
  design = ~ pair + condition
)

# Run DESeq2 differential expression analysis
dds <- DESeq(dds)

res <- results(dds)

# Extract results for the contrast of interest (Tumor vs. Normal)
res <- results(dds, contrast = c("condition", "Tumor", "Normal"))

# View summary of results
summary(res)

# You can also save the results to a file
write.csv(as.data.frame(res), "/home/jiye/jiye/nanopore/202411_analysis/whole_DETresult.txt")

# Convert results to a data frame
res_df <- as.data.frame(res)

# Create a new column to highlight significant genes
# Here we assume a threshold of padj < 0.05 (adjusted p-value) and log2FoldChange > |1|
res_df$significance <- with(res_df, ifelse(padj < 0.05 & abs(log2FoldChange) > 2, "Significant", "Not significant"))

# Remove any NA values for plotting
res_df <- na.omit(res_df)
library(ggplot2)
# Create the volcano plot
volcano_plot <- ggplot(res_df, aes(x = log2FoldChange, y = -log10(padj))) +
  geom_point(aes(color = significance), alpha = 0.6, size = 1.5) +
  scale_color_manual(values = c("Significant" = "red", "Not significant" = "grey")) +
  theme_minimal() +
  labs(
    title = "Volcano Plot",
    x = "Log2 Fold Change",
    y = "-Log10 Adjusted p-value"
  ) +
  theme(legend.title = element_blank())

# Display the plot
print(volcano_plot)

############ decide threshold #############
significant_transcripts_fc <- res[which(res$padj < 0.01 & abs(res$log2FoldChange) >1.5), ]
num_significant_fc <- nrow(significant_transcripts_fc)
print(paste("Number of significant transcripts (padj < 0.05 & abs(log2FoldChange) > 2):", num_significant_fc))

##################### TPM!!!!!!!!!!! ####################
TPM <- read.delim("/home/jiye/jiye/nanopore/FINALDATA/137_transcript_TPM.txt", header = TRUE, row.names = 1,check.names = FALSE)

################## N vs. T only novel transcripts ##################
significant_novel_transcripts <- rownames(significant_transcripts_fc)[grepl("^MSTRG", rownames(significant_transcripts_fc))]

# Step 2: Filter the expression matrix to include only these novel transcripts

#expr_matrix_novel <- as.matrix(TPM[rownames(TPM) %in% significant_trans, ]) ##novel
#expr_matrix <- as.matrix(TPM[rownames(TPM) %in% rownames(significant_transcripts_fc), ]) ##all sig
expr_matrix <- as.matrix(TPM[rownames(TPM) %in% rownames(filtered_counts), ]) ##all sig

# Step 3:Zscore normalization
expr_matrix_scaled <- t(apply(expr_matrix, 1, scale))
colnames(expr_matrix_scaled) <- colnames(expr_matrix)  # Preserve column names
rownames(expr_matrix_scaled) <- rownames(expr_matrix)
#expr_matrix_scaled<- log2(expr_matrix + 1)

normal_samples <- grep("-N$", colnames(expr_matrix))
tumor_samples <- grep("-T$", colnames(expr_matrix))
expr_matrix_scaled <- expr_matrix_scaled[, c(normal_samples, tumor_samples)]

# Step 4: Create simple annotations for Tumor and Normal
# Assuming your column names have '-N' and '-T' to distinguish between normal and tumor samples
sample_type <- ifelse(grepl("-N$", colnames(expr_matrix_scaled)), "Normal", "Tumor")
annotation <- HeatmapAnnotation(
  SampleType = sample_type,
  col = list(SampleType = c("Normal" = "blue", "Tumor" = "red"))
)

# Step 5: Create the heatmap
library(ComplexHeatmap)
Heatmap(
  expr_matrix_scaled,  # Log2-normalized expression matrix
  name = "Expression",
  top_annotation = annotation,
  show_row_names = FALSE,
  show_column_names = FALSE,
  cluster_columns = FALSE,  # Optional clustering
  cluster_rows = FALSE,
  clustering_method_columns = "ward.D",
  clustering_method_rows = "ward.D",# Optional clustering
  #column_title = "Tumor vs Normal"
)

###################### delta exp: complex heatmap#######################
library(ComplexHeatmap)
library(circlize)
#expr_matrix <- as.matrix(TPM[rownames(TPM) %in% rownames(significant_transcripts_fc), ]) ##all sig
expr_matrix <- as.matrix(TPM[rownames(TPM) %in% significant_novel_transcripts, ])

# Step 2: Separate Normal and Tumor columns
normal_samples <- grep("-N$", colnames(expr_matrix))  # Assuming '-N' suffix for normal samples
tumor_samples <- grep("-T$", colnames(expr_matrix))   # Assuming '-T' suffix for tumor samples

# Ensure that Normal and Tumor are paired correctly
if (length(normal_samples) != length(tumor_samples)) {
  stop("Number of normal and tumor samples do not match. Please check your data.")
}

# Step 3: Calculate delta values (Tumor - Normal)
expr_matrix_delta <- expr_matrix[, tumor_samples] - expr_matrix[, normal_samples]
colnames(expr_matrix_delta) <- gsub("-T$", "", colnames(expr_matrix[, tumor_samples]))  # Simplify column names

# Step 4: Optional: Z-score normalization for better visualization
expr_matrix_delta_scaled <- t(apply(expr_matrix_delta, 1, scale))
colnames(expr_matrix_delta_scaled) <- colnames(expr_matrix_delta)  # Preserve column names
rownames(expr_matrix_delta_scaled) <- rownames(expr_matrix_delta)  # Preserve row names

metadata <- read.table("/home/jiye/jiye/nanopore/FINALDATA/137_clinicaldata.txt", header = TRUE, sep = "\t")

age_color_scale <- colorRamp2(c(min(metadata$age, na.rm = TRUE), max(metadata$age, na.rm = TRUE)), c("white", "black"))

# Create annotations
sample_annotation <- HeatmapAnnotation(
  df = metadata[, c("age","sex", "MSI_status", "KRAS_mut",
                    "lymphatic_invasion", "venous_invasion", "perineural_invasion", "location",
                    "stage","DFS")],  # Adjust columns as needed
  col = list(
    
    age = age_color_scale,
    sex = c("Male" = "#ABC9FF", "Female" = "#FF8A8A"),
    MSI_status = c("MSS" = "#F5F0BB", "MSI-H" = "#61A3BA", "MSI-L" = "#D2DE32"),
    KRAS_mut = c("WT" = "#F8F4E1", "MT" = "#AF8F6F", "ND" = "grey"),
    lymphatic_invasion = c('0' = "#FAF7F0", '1' = "#AB886D"),  # Assuming 0 = no invasion, 1 = invasion
    venous_invasion = c('0' = "#FAF7F0", '1' = "#AB886D"),
    perineural_invasion = c('0' = "#FAF7F0", '1' = "#AB886D"),
    stage = c('1' = "#F1F1E8", '2' = "#BFDCAE", '3' = "#81B214", '4' = "#206A5D")  # Assuming stages 1-4
  ),
  annotation_height = unit(c(2, 2, 2, 2, 2), "mm"),
  na_col = "grey"  # Optionally, specify a color for NA values
)


ht <- Heatmap(
  expr_matrix_delta_scaled,  # Use the delta matrix for visualization
  name = "Delta Expression",
  top_annotation = sample_annotation,  # Assuming sample_annotation is aligned with delta columns
  show_row_names = FALSE,
  show_column_names = FALSE,
  cluster_columns = TRUE,
  cluster_rows = TRUE,
  clustering_method_columns = "ward.D",
  clustering_method_rows = "ward.D",# Enable column clustering
  #column_title = "Delta Expression Heatmap (Tumor - Normal)"
)
# Draw the heatmap
#pdf("/home/jiye/jiye/nanopore/202411_analysis/fc2_1993_heatmap.pdf", width = 8, height = 8)  # Adjust width and height as needed
draw(ht)
dev.off()
