if (!require("pheatmap")) install.packages("pheatmap", dependencies=TRUE)
if (!require("RColorBrewer")) install.packages("RColorBrewer", dependencies=TRUE)
if (!require("colorspace")) install.packages("colorspace", dependencies=TRUE)
if (!require("scales")) install.packages("scales", dependencies=TRUE)
if (!require("wesanderson")) install.packages("wesanderson", dependencies=TRUE)
if (!require("ggsci")) install.packages("ggsci", dependencies=TRUE)

library(pheatmap)
library(RColorBrewer)
library(colorspace)
library(scales)
library(wesanderson)
library(ggsci)

# Load the data
#tu_data <- read.csv("/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/featureselection/32_deltaTU_input.txt", sep="\t", header=TRUE, check.names = F, row.names = 1)
tu_data <- read.csv("/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/featureselection/rf_49_deltaTU_input.txt", sep="\t", header=TRUE, check.names = F, row.names = 1)

metadata <- read.csv("/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo_new.txt", sep="\t", header=TRUE, check.names = F,)
metadata <- metadata[seq(1, nrow(metadata), by = 2), ]
rownames(metadata) <- metadata$sample_id


# Check for and handle missing values
metadata <- na.omit(metadata)

# Ensure all data in tu_data is numeric
#tu_data <- apply(tu_data, 2, as.numeric)

# Select the column from metadata for the binary annotation
binary_annotation <- metadata[, "response"]  # Replace with your actual column name

# Select additional columns for annotations
#additional_annotations <- metadata[, c("BRCAmut","PFI")]
additional_annotations <- metadata[, c("line_binary", "exonic_read", "BRCAmut", "drug","purpose", "PFI")]

# Ensure all categorical variables are factors
#additional_annotations$line_binary <- as.factor(additional_annotations$line_binary)
additional_annotations$exonic_read <- as.factor(additional_annotations$exonic_read)
additional_annotations$BRCAmut <- as.factor(additional_annotations$BRCAmut)
#additional_annotations$survival <- as.factor(additional_annotations$survival)
additional_annotations$drug <- as.factor(additional_annotations$drug)
additional_annotations$purpose <- as.factor(additional_annotations$purpose)
#additional_annotations$line <- as.factor(additional_annotations$line)

# Ensure all numeric variables are numeric
additional_annotations$PFI <- as.numeric(additional_annotations$PFI)
#additional_annotations$OS <- as.numeric(additional_annotations$OS)
#percentile_75 <- quantile(additional_annotations$PFI, 0.75)
#additional_annotations$PFI_upper25 <- ifelse(additional_annotations$PFI >= percentile_75, 'long', 'short')


# Combine the annotations
annotations <- data.frame(response = binary_annotation, additional_annotations)
annotations$response <- as.factor(annotations$response)


# Define a function to get a palette with a minimum of 3 colors
#get_palette <- function(n) {
#  if (n <= 12) {
#    return(brewer.pal(n = n, name = "Set3"))
#  } else {
#    return(qualitative_hcl(n, palette = "Set 3"))  # Use colorspace for more than 12 colors
#  }
#}

library(viridis)

# Define the color palette for annotations
#line_levels <- levels(annotations$line)
#line_colors <- setNames(get_palette(length(line_levels)), line_levels)

annotation_colors <- list(
  response = c("0" = "#11468F", "1" = "#EB455F"),  # Adjust colors if needed
  #line_binary = c("FL" = "lightblue", "N-FL" = "pink"),
  #line = line_colors,
  PFI = scales::seq_gradient_pal("white", "black")(seq(0, 1, length.out = 100)),
  #exonic_read = c("low" = "lightblue", "high" = "pink"),
  BRCAmut = c("0" = "#FCDC2A", "1" = "#87A922")
  #survival = c("0" = "lightblue", "1" = "pink"),
  #OS = scales::seq_gradient_pal("black", "white")(seq(0, 1, length.out = 100)),
  #drug = c("Olaparib" = "lightblue", "Niraparib" = "pink", "Rucaparib" = "green"),
  #purpose = c("maintenance" = "lightblue", "salvage" = "pink")
)


# Create the heatmap with hierarchical clustering and annotations
hm <- pheatmap(
  tu_data,
  clustering_method = "average",  # Ward method
  annotation_col = annotations,
  annotation_colors = annotation_colors,
  scale = "row",  # Adjust scaling as needed ("none", "row", "column")
  show_rownames = TRUE,
  show_colnames = FALSE,
  cluster_cols = TRUE,
  cluster_rows = TRUE,# Adjust if you want to show column names
  fontsize_row = 9
)

library(cluster)
dist_matrix <- dist(tu_data)  # Compute distance matrix for columns (samples)
hc <- hclust(dist_matrix, method = "single")  # Perform hierarchical clustering
# Cut the tree to form clusters (choose an appropriate number of clusters, e.g., k = 4)
cluster_assignments <- cutree(hc, k = 4)
# Calculate silhouette scores
silhouette_scores <- silhouette(cluster_assignments, dist_matrix)
# Plot silhouette scores
plot(silhouette_scores, main = "Silhouette Plot")


library(ggplot2)
hm
ggsave(filename = "/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/fig2/32_heatmap.pdf", plot = hm, dpi = 300, width = 5, height = 7)
