library(ComplexHeatmap)

TPM <- read.delim("/home/jiye/jiye/nanopore/FINALDATA/137_transcript_TPM.txt", header = TRUE, row.names = 1,check.names = FALSE)
det <- read.delim("/home/jiye/jiye/nanopore/202411_analysis/wilcoxon_DET/Wilcoxon_DETresult.txt", header = TRUE, row.names = 1,check.names = FALSE)
novellist <- read.delim("/home/jiye/jiye/nanopore/202411_analysis/sqantioutput/stringtie_novel_list.txt", header = TRUE, row.names = 1,check.names = FALSE)
dut <- read.delim("/home/jiye/jiye/nanopore/202411_analysis/isoformswitchanalyzer/DUTresult.txt", header = TRUE, row.names = 1,check.names = FALSE)
f <- readLines("/home/jiye/jiye/nanopore/202411_analysis/isoformswitchanalyzer/tumorspcificlist.txt")
t_list <- unlist(strsplit(f, ","))
dut_list <- dut$isoform_id
#TPM <- read.delim("/home/jiye/jiye/nanopore/FINALDATA/137_gene_TPM.txt", header = TRUE, row.names = 1,check.names = FALSE)
#TPM = subset(TPM, select = -c(ENSG_id) )
#det <- read.delim("/home/jiye/jiye/nanopore/202411_analysis/wilcoxon_DEG/Wilcoxon_DEGresult.txt", header = TRUE, row.names = 1,check.names = FALSE)

################################ IsoformSwitchAnalyzeR DUT heatmap ##############################
expr_matrix <- TPM
#expr_matrix$transcript_id_split <- sub("-.*", "", rownames(expr_matrix))
#rownames(expr_matrix) <- expr_matrix$transcript_id_split
#expr_matrix$transcript_id_split <- NULL
expr_matrix <- as.matrix(expr_matrix[rownames(expr_matrix) %in% t_list, ]) ##dut_list vs.. t_list

expr_matrix_scaled <- t(apply(expr_matrix, 1, scale))
colnames(expr_matrix_scaled) <- colnames(expr_matrix)  # Preserve column names
rownames(expr_matrix_scaled) <- rownames(expr_matrix)  # Preserve row names

expr_matrix_scaled <- expr_matrix_scaled[, seq(2, ncol(expr_matrix_scaled), by = 2)]
colnames(expr_matrix_scaled) <- gsub("-T$", "", colnames(expr_matrix_scaled))  # Simplify column names


metadata <- read.table("/home/jiye/jiye/nanopore/FINALDATA/137_clinicaldata_final.txt", header = TRUE, sep = "\t", check.names = FALSE,row.names = 1)
metadata <- metadata[match(colnames(expr_matrix_scaled), rownames(metadata)), ]

age_color_scale <- colorRamp2(c(min(metadata$age, na.rm = TRUE), max(metadata$age, na.rm = TRUE)), c("white", "black"))

if (nrow(metadata) != ncol(expr_matrix_scaled)) {
  stop("Mismatch between number of metadata rows and matrix columns")
}
metadata$MSI_status <- as.factor(metadata$MSI_status)
metadata$sex <- as.factor(metadata$sex)
metadata$age <- as.numeric(metadata$age)
metadata$CMS <- as.factor(metadata$CMS)
metadata$Mstage <- as.factor(metadata$Mstage)

# Create annotations
sample_annotation <- HeatmapAnnotation(
  df = metadata[, c("sex",
                    "MSI_status", "KRAS_mut",
                    "lymphatic_invasion", "venous_invasion", "perineural_invasion","stage","CMS","Mstage"
  )],  # Adjust columns as needed
  col = list(
    #age = age_color_scale,
    sex = c("Male" = "#ABC9FF", "Female" = "#FF8A8A"),
    
    MSI_status = c("MSS" = "#F5F0BB", "MSI-H" = "#61A3BA", "MSI-L" = "#D2DE32"),
    KRAS_mut = c("WT" = "#F8F4E1", "MT" = "#AF8F6F", "ND" = "grey"),
    lymphatic_invasion = c('0' = "#FAF7F0", '1' = "#AB886D"),  # Assuming 0 = no invasion, 1 = invasion
    venous_invasion = c('0' = "#FAF7F0", '1' = "#AB886D"),
    perineural_invasion = c('0' = "#FAF7F0", '1' = "#AB886D"),
    stage = c('1' = "#F1F1E8", '2' = "#BFDCAE", '3' = "#81B214", '4' = "#206A5D"),
    CMS = c('CMS1' = "#F1F1E8", 'CMS2' = "#BFDCAE", 'CMS3' = "#81B214", 'CMS4' = "#206A5D")
  ),
  #annotation_height = unit(c(2, 2, 2, 2), "mm"),
  na_col = "grey"  # Optionally, specify a color for NA values
)


ht <- Heatmap(
  expr_matrix_scaled,  # Use the delta matrix for visualization
  name = "DUT expression in tumor",
  top_annotation = sample_annotation,  # Assuming sample_annotation is aligned with delta columns
  show_row_names = FALSE,
  show_column_names = FALSE,
  cluster_columns = TRUE,
  cluster_rows = TRUE,
  clustering_method_columns = "ward.D2",
  clustering_method_rows = "ward.D2",# Enable column clustering
  #column_title = "Delta Expression Heatmap (Tumor - Normal)"
)

# Draw the heatmap
pdf("/home/jiye/jiye/nanopore/202411_analysis/figures/tonly_tumor_heatmap.pdf", width = 8, height = 9)  # Adjust width and height as needed
draw(ht)
dev.off()










#############################################################











significant_transcripts_fc <- det[which(det$'p-adjusted' < 0.05 & abs(det$log2FC) >1.5), ]
num_significant_fc <- nrow(significant_transcripts_fc)
print(paste("Number of significant transcripts (padj < 0.05 & abs(log2FoldChange) > 2):", num_significant_fc))


################## N vs. T only novel transcripts ##################
significant_novel_transcripts <- rownames(significant_transcripts_fc)[grepl("^MSTRG", rownames(significant_transcripts_fc))]


#expr_matrix <- as.matrix(TPM[rownames(TPM) %in% significant_novel_transcripts, ]) ##novel
expr_matrix <- as.matrix(TPM[rownames(TPM) %in% rownames(significant_transcripts_fc), ]) ##all sig
prefix_df1 <- sub("-.*", "", rownames(expr_matrix))
expr_matrix <- expr_matrix[prefix_df1 %in% rownames(novellist), ]

# Step 3:Zscore normalization
expr_matrix_scaled <- t(apply(expr_matrix, 1, scale))
colnames(expr_matrix_scaled) <- colnames(expr_matrix)  # Preserve column names
rownames(expr_matrix_scaled) <- rownames(expr_matrix)

normal_samples <- grep("-N$", colnames(expr_matrix))
tumor_samples <- grep("-T$", colnames(expr_matrix))
expr_matrix_scaled <- expr_matrix_scaled[, c(normal_samples, tumor_samples)]

# Step 4: Create simple annotations for Tumor and Normal
# Assuming your column names have '-N' and '-T' to distinguish between normal and tumor samples
sample_type <- ifelse(grepl("-N$", colnames(expr_matrix_scaled)), "Normal", "Tumor")
annotation <- HeatmapAnnotation(
  SampleType = sample_type,
  col = list(SampleType = c("Normal" = "#0D0CB5", "Tumor" = "#C82121"))
)

# Step 5: Create the heatmap

ht <- Heatmap(
  expr_matrix_scaled,  # Log2-normalized expression matrix
  name = "Expression",
  top_annotation = annotation,
  show_row_names = FALSE,
  show_column_names = FALSE,
  cluster_columns = TRUE,  # Optional clustering
  cluster_rows = TRUE,
  clustering_method_columns = "ward.D",
  clustering_method_rows = "ward.D",# Optional clustering
  #column_title = "Tumor vs Normal"
)

pdf("/home/jiye/jiye/nanopore/202411_analysis/wilcoxon_DET/DET_NT_onlynovel_ward_heatmap.pdf", width = 7, height = 8)
draw(ht)
dev.off()


################################## delta TPM with clinical info #############################################
library(ComplexHeatmap)
library(circlize)
expr_matrix <- as.matrix(TPM[rownames(TPM) %in% rownames(significant_transcripts_fc), ]) ##all sig
##expr_matrix <- as.matrix(TPM[rownames(TPM) %in% significant_novel_transcripts, ])

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

metadata <- read.table("/home/jiye/jiye/nanopore/FINALDATA/137_clinicaldata.txt", header = TRUE, sep = "\t", check.names = FALSE,row.names = 1)
metadata <- metadata[match(colnames(expr_matrix_delta_scaled), rownames(metadata)), ]

age_color_scale <- colorRamp2(c(min(metadata$age, na.rm = TRUE), max(metadata$age, na.rm = TRUE)), c("white", "black"))

if (nrow(metadata) != ncol(expr_matrix_delta_scaled)) {
  stop("Mismatch between number of metadata rows and matrix columns")
}
metadata$MSI_status <- as.factor(metadata$MSI_status)
metadata$sex <- as.factor(metadata$sex)
metadata$age <- as.numeric(metadata$age)
# Create annotations
sample_annotation <- HeatmapAnnotation(
  df = metadata[, c("sex"
                    #,"sex", "MSI_status", "KRAS_mut"
                    #"lymphatic_invasion", "venous_invasion", "perineural_invasion","DFS",
                    #"stage"
                    )],  # Adjust columns as needed
  col = list(
    #age = age_color_scale,
    sex = c("Male" = "#ABC9FF", "Female" = "#FF8A8A")
    #,
    #MSI_status = c("MSS" = "#F5F0BB", "MSI-H" = "#61A3BA", "MSI-L" = "#D2DE32"),
    #KRAS_mut = c("WT" = "#F8F4E1", "MT" = "#AF8F6F", "ND" = "grey")
    #lymphatic_invasion = c('0' = "#FAF7F0", '1' = "#AB886D"),  # Assuming 0 = no invasion, 1 = invasion
    #venous_invasion = c('0' = "#FAF7F0", '1' = "#AB886D"),
    ##perineural_invasion = c('0' = "#FAF7F0", '1' = "#AB886D"),
    #stage = c('1' = "#F1F1E8", '2' = "#BFDCAE", '3' = "#81B214", '4' = "#206A5D")  # Assuming stages 1-4
  ),
  #annotation_height = unit(c(2, 2, 2, 2), "mm"),
  na_col = "grey"  # Optionally, specify a color for NA values
)

sample_annotation <- HeatmapAnnotation(
  age = metadata$age,
  sex = metadata$sex,
  MSI_status = metadata$MSI_status,
  KRAS_mut = metadata$KRAS_mut,
  stage = metadata$stage,
  col = list(
    age = age_color_scale,
    sex = c("Male" = "#ABC9FF", "Female" = "#FF8A8A"),
    MSI_status = c("MSS" = "#F5F0BB", "MSI-H" = "#61A3BA", "MSI-L" = "#D2DE32"),
    KRAS_mut = c("WT" = "#F8F4E1", "MT" = "#AF8F6F", "ND" = "grey"),
    stage = c('1' = "#F1F1E8", '2' = "#BFDCAE", '3' = "#81B214", '4' = "#206A5D")
  ),
  annotation_height = unit(c(2, 2, 2, 2, 2), "mm"),
  na_col = "grey"  # Optionally specify a color for NA values
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
pdf("/home/jiye/jiye/nanopore/202411_analysis/wilcoxon_DET/DET_deltaTPM_onlynovel_ward_heatmap.pdf", width = 8, height = 9)  # Adjust width and height as needed
draw(ht)
dev.off()

sample_annotation <- HeatmapAnnotation(
  sex = metadata$sex,
  col = list(
    sex = c("Male" = "#ABC9FF", "Female" = "#FF8A8A")
  ),
  na_col = "grey"
)
Heatmap(
  expr_matrix_delta_scaled,
  top_annotation = sample_annotation,
  show_row_names = FALSE,
  show_column_names = FALSE
)




################################## tumor TPM with clinical info #############################################
library(ComplexHeatmap)
library(circlize)
expr_matrix <- as.matrix(TPM[, (1:ncol(TPM)) %% 2 == 0]) 
expr_matrix <- expr_matrix[rowSums(expr_matrix > 0) >= 42, ]

##expr_matrix <- as.matrix(TPM[rownames(TPM) %in% significant_novel_transcripts, ])


# Step 3: Calculate delta values (Tumor - Normal)
colnames(expr_matrix) <- gsub("-T$", "", colnames(expr_matrix))  # Simplify column names
expr_matrix_delta <- expr_matrix

# Step 4: Optional: Z-score normalization for better visualization
expr_matrix_delta_scaled <- t(apply(expr_matrix_delta, 1, scale))
colnames(expr_matrix_delta_scaled) <- colnames(expr_matrix_delta)  # Preserve column names
rownames(expr_matrix_delta_scaled) <- rownames(expr_matrix_delta)  # Preserve row names

metadata <- read.table("/home/jiye/jiye/nanopore/FINALDATA/137_clinicaldata_new.txt", header = TRUE, sep = "\t", check.names = FALSE,row.names = 1)
metadata <- metadata[match(colnames(expr_matrix_delta_scaled), rownames(metadata)), ]

age_color_scale <- colorRamp2(c(min(metadata$age, na.rm = TRUE), max(metadata$age, na.rm = TRUE)), c("white", "black"))

if (nrow(metadata) != ncol(expr_matrix_delta_scaled)) {
  stop("Mismatch between number of metadata rows and matrix columns")
}
metadata$MSI_status <- as.factor(metadata$MSI_status)
metadata$sex <- as.factor(metadata$sex)
metadata$age <- as.numeric(metadata$age)
# Create annotations
sample_annotation <- HeatmapAnnotation(
  df = metadata[, c("sex"
                    #,"sex", "MSI_status", "KRAS_mut"
                    #"lymphatic_invasion", "venous_invasion", "perineural_invasion","DFS",
                    #"stage"
  )],  # Adjust columns as needed
  col = list(
    #age = age_color_scale,
    sex = c("Male" = "#ABC9FF", "Female" = "#FF8A8A")
    #,
    #MSI_status = c("MSS" = "#F5F0BB", "MSI-H" = "#61A3BA", "MSI-L" = "#D2DE32"),
    #KRAS_mut = c("WT" = "#F8F4E1", "MT" = "#AF8F6F", "ND" = "grey")
    #lymphatic_invasion = c('0' = "#FAF7F0", '1' = "#AB886D"),  # Assuming 0 = no invasion, 1 = invasion
    #venous_invasion = c('0' = "#FAF7F0", '1' = "#AB886D"),
    ##perineural_invasion = c('0' = "#FAF7F0", '1' = "#AB886D"),
    #stage = c('1' = "#F1F1E8", '2' = "#BFDCAE", '3' = "#81B214", '4' = "#206A5D")  # Assuming stages 1-4
  ),
  #annotation_height = unit(c(2, 2, 2, 2), "mm"),
  na_col = "grey"  # Optionally, specify a color for NA values
)

sample_annotation <- HeatmapAnnotation(
  age = metadata$age,
  sex = metadata$sex,
  MSI_status = metadata$MSI_status,
  CMS = metadata$CMS,
  KRAS_mut = metadata$KRAS_mut,
  stage = metadata$stage,
  col = list(
    age = age_color_scale,
    sex = c("Male" = "#ABC9FF", "Female" = "#FF8A8A"),
    MSI_status = c("MSS" = "#F5F0BB", "MSI-H" = "#61A3BA", "MSI-L" = "#D2DE32"),
    KRAS_mut = c("WT" = "#F8F4E1", "MT" = "#AF8F6F", "ND" = "grey"),
    stage = c('1' = "#F1F1E8", '2' = "#BFDCAE", '3' = "#81B214", '4' = "#206A5D"),
    CMS = c('CMS1'='#D91656','CMS2'='#F6D6D6','CMS3'='#FB9EC6','CMS4'='#F6D6D6','nan'="grey")
  ),
  annotation_height = unit(c(2, 2, 2, 2, 2,2), "mm"),
  na_col = "grey"  # Optionally specify a color for NA values
)

ht <- Heatmap(
  expr_matrix_delta_scaled,  # Use the delta matrix for visualization
  name = "Tumor Transcript Expression",
  top_annotation = sample_annotation,  # Assuming sample_annotation is aligned with delta columns
  show_row_names = FALSE,
  show_column_names = FALSE,
  cluster_columns = TRUE,
  cluster_rows = FALSE,
  clustering_method_columns = "ward.D",
  clustering_method_rows = "ward.D",# Enable column clustering
  #column_title = "Delta Expression Heatmap (Tumor - Normal)"
)

# Draw the heatmap
pdf("/home/jiye/jiye/nanopore/202411_analysis/wilcoxon_DET/onlytumor_TPM_ward_heatmap.pdf", width = 8, height = 9)  # Adjust width and height as needed
draw(ht)
dev.off()

sample_annotation <- HeatmapAnnotation(
  sex = metadata$sex,
  col = list(
    sex = c("Male" = "#ABC9FF", "Female" = "#FF8A8A")
  ),
  na_col = "grey"
)
Heatmap(
  expr_matrix_delta_scaled,
  top_annotation = sample_annotation,
  show_row_names = FALSE,
  show_column_names = FALSE
)

