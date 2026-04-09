#%%
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from neuroCombat import neuroCombat

#%%
# Assuming your 3 datasets are stored in pandas dataframes df1, df2, and df3
df1 = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/POLO/83_POLO_transcript_exp.txt', sep='\t', index_col=0)
df1 = df1.drop(columns=['target_gene'])
df2 = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_transcript_exp.txt', sep='\t', index_col=0)
df3 = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_TPM.txt', sep='\t', index_col=0)
df3 = df3.iloc[:-2, :]
df2 = df2.iloc[:,:-1]
#%%
from sklearn.preprocessing import StandardScaler

# 1. Align the rows by reindexing to ensure the same order
common_index = df1.index  # Use df1's index as the common order
df1_aligned = df1.reindex(common_index)
df2_aligned = df2.reindex(common_index)
df3_aligned = df3.reindex(common_index)

# 2. Add a 'Dataset' column for labeling
df1_aligned['Dataset'] = 'POLO'
df2_aligned['Dataset'] = 'Pre-Post'
df3_aligned['Dataset'] = 'Validation'

# 3. Concatenate the datasets along the rows after alignment
df_combined = pd.concat([df1_aligned, df2_aligned, df3_aligned], axis=1)

# 3. Create the 'y' labels for the columns (samples)
y = (['POLO'] * df1_aligned.shape[1]) + (['Pre-Post'] * df2_aligned.shape[1]) + (['Validation'] * df3_aligned.shape[1])

# 4. Transpose the DataFrame to have samples as rows and transcripts as columns for PCA
X = df_combined.T

# 5. Handle NaN, Inf, and large values
X = X.apply(pd.to_numeric, errors='coerce')  # Convert all data to numeric, setting errors to NaN
X_cleaned = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN and Inf with 0

# Optional: Clip large values to a reasonable range
large_value_threshold = 1e6
X_cleaned = np.clip(X_cleaned, -large_value_threshold, large_value_threshold)

# 6. Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cleaned)

# 7. Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 8. Plot PCA results
plt.figure(figsize=(8, 6))
for dataset, color in zip(['POLO', 'Pre-Post', 'Validation'], ['r', 'g', 'b']):
    mask = np.array(y) == dataset  # Create a boolean mask for each dataset
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=dataset, alpha=0.6, color=color)

plt.title('PCA Plot of 3 Datasets')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)')
plt.legend()
plt.show()



# %% #^^ Validation vs. POLO
# 1. Align the rows (transcripts/genes) by index to ensure the same order
common_index = df1.index  # Use df1's index as the common order
df1_aligned = df1.reindex(common_index)
df3_aligned = df3.reindex(common_index)

# 2. Concatenate the datasets along the columns (axis=1)
df_combined = pd.concat([df1_aligned, df3_aligned], axis=1)

# 3. Create the 'y' labels for the columns (samples)
y = (['POLO'] * df1_aligned.shape[1]) + (['Validation'] * df3_aligned.shape[1])

# 4. Transpose the DataFrame to have samples as rows and transcripts as columns for PCA
X = df_combined.T

# 5. Handle NaN, Inf, and large values
X = X.apply(pd.to_numeric, errors='coerce')  # Convert all data to numeric, setting errors to NaN
X_cleaned = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN and Inf with 0

# Optional: Clip large values to a reasonable range
large_value_threshold = 1e6
X_cleaned = np.clip(X_cleaned, -large_value_threshold, large_value_threshold)

# 6. Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cleaned)

# 7. Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)


# 8. Plot PCA results for df1 (POLO) and df3 (Validation), with sample names
samples = df_combined.columns

plt.figure(figsize=(8, 6))
for dataset, color in zip(['POLO', 'Validation'], ['r', 'b']):
    mask = np.array(y) == dataset  # Create a boolean mask for each dataset
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=dataset, alpha=0.6, color=color)

# Add sample labels to the plot
# for i, sample in enumerate(samples):
#     plt.text(X_pca[i, 0], X_pca[i, 1], sample, fontsize=8, alpha=0.75)

plt.title('PCA Plot of df1 (POLO) and df3 (Validation) with Sample Labels')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)')
plt.legend()
plt.show()

#%%
# %%
##^^ batch correction - whole dataset

# Step 1: Align genes (rows) between df1 and df3 to ensure they match
common_index = df3.index  # Use df3's index as the common order
df1_aligned = df1.reindex(common_index)
df3_aligned = df3.reindex(common_index)

# Step 2: Transpose df1 to (genes as rows, samples as columns) for ComBat
df1_transposed = df1_aligned.T

# Step 3: Create batch labels (all df1 samples belong to batch 0)
covars = {
    'batch': [0] * df1_transposed.shape[1]  # One batch for all df1 samples
}
covars = pd.DataFrame(covars)

# Step 4: Apply ComBat batch correction to df1 only
combat_corrected_df1 = neuroCombat(
    dat=df1_transposed,  # Genes as rows, samples as columns
    covars=covars,
    batch_col='batch'
)['data']

# Step 5: Transpose corrected df1 back to the original shape
df1_corrected = pd.DataFrame(combat_corrected_df1).T
df1_corrected.index = df1_aligned.index  # Restore gene index
df1_corrected.columns = df1_aligned.columns  # Restore sample names

# Step 6: Concatenate corrected df1 with the original df3 along columns
df_combined = pd.concat([df1_corrected, df3], axis=1)

# Step 7: Transpose combined data for PCA (samples as rows, genes as columns)
X = df_combined.T  # Now shape (samples, genes)

# Step 8: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 9: Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 10: Create sample labels
y = np.array(['POLO'] * df1_aligned.shape[1] + ['Validation'] * df3_aligned.shape[1])

# Step 11: Plot the PCA results
plt.figure(figsize=(8, 6))
for dataset, color in zip(['POLO', 'Validation'], ['r', 'b']):
    mask = (y == dataset)  # Boolean mask for samples
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=dataset, alpha=0.6, color=color)

plt.title('PCA Plot After ComBat Batch Correction on df1 Only')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)')
plt.legend()
plt.show()


#%%
##^^batch correction with harmony
import harmonypy as hm

# Perform PCA before applying Harmony
pca = PCA(n_components=20)  # Use more components to capture latent space
X_pca = pca.fit_transform(X_scaled)

# Run Harmony to align batch effects
harmony_out = hm.run_harmony(X_pca, y, 'Dataset')

# Extract Harmony-corrected embeddings
X_harmony = harmony_out.Z_corr

# Plot the Harmony-corrected PCA
plt.figure(figsize=(8, 6))
for dataset, color in zip(['POLO', 'Validation'], ['r', 'b']):
    mask = (y == dataset)
    plt.scatter(X_harmony[mask, 0], X_harmony[mask, 1], label=dataset, alpha=0.6, color=color)

plt.title('PCA Plot After Harmony Batch Correction')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


# %%
#%%
###^^ Check Cluster ####
from sklearn.cluster import KMeans

# 1. Perform PCA as before (we assume you've already done this and have X_pca)

# 2. Apply K-means clustering to the PCA results for the "Validation" dataset (blue points)
validation_mask = np.array(y) == 'Validation'  # Boolean mask for Validation samples

# Use K-means clustering to divide the "Validation" dataset into two clusters
kmeans = KMeans(n_clusters=2, random_state=42)
validation_clusters = kmeans.fit_predict(X_pca[validation_mask])

# 3. Get the sample names that belong to each cluster
validation_samples = np.array(samples)[validation_mask]  # Validation sample names

# Cluster 0 and Cluster 1
cluster_0_samples = validation_samples[validation_clusters == 0]
cluster_1_samples = validation_samples[validation_clusters == 1]

# Print the sample names in each cluster
print("Cluster 0 samples:")
print(cluster_0_samples)

print("\nCluster 1 samples:")
print(cluster_1_samples)

# 4. Plot PCA results with cluster labels
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[validation_mask, 0], X_pca[validation_mask, 1], c=validation_clusters, cmap='coolwarm', alpha=0.6)
plt.title('Validation Dataset Clustering')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)')
for i, sample in enumerate(validation_samples):
    plt.text(X_pca[validation_mask][i, 0], X_pca[validation_mask][i, 1], sample, fontsize=8, alpha=0.75)
plt.show()




# %%
###^^^ cluster 1+3 (POLO + Validation) / top 30% gene #####

# 1. Align the rows (transcripts/genes) by index to ensure the same order
common_index = df1.index  # Use df1's index as the common order
df1_aligned = df1.reindex(common_index)
df3_aligned = df3.reindex(common_index)

# 2. Concatenate the datasets along the columns (axis=1)
df_combined = pd.concat([df1_aligned, df3_aligned], axis=1)

# 3. Calculate the mean expression per gene (row-wise mean)
gene_means = df_combined.mean(axis=1)

# 4. Select the top 30% most highly expressed genes
top_30_percent_threshold = np.percentile(gene_means, 70)  # 70th percentile
top_30_percent_genes = gene_means[gene_means >= top_30_percent_threshold].index

# 5. Filter the dataset to include only the top 30% most highly expressed genes
df_top_30_percent = df_combined.loc[top_30_percent_genes]

# 6. Transpose the DataFrame to have samples as rows and transcripts as columns for PCA
X = df_top_30_percent.T

# 7. Handle NaN, Inf, and large values
X = X.apply(pd.to_numeric, errors='coerce')  # Convert all data to numeric, setting errors to NaN
X_cleaned = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN and Inf with 0

# Optional: Clip large values to a reasonable range
large_value_threshold = 1e6
X_cleaned = np.clip(X_cleaned, -large_value_threshold, large_value_threshold)

# 8. Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cleaned)

# 9. Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 10. Create labels for the samples (POLO and Validation)
y = (['POLO'] * df1_aligned.shape[1]) + (['Validation'] * df3_aligned.shape[1])

# 11. Plot PCA results for df1 (POLO) and df3 (Validation)
plt.figure(figsize=(8, 6))
for dataset, color in zip(['POLO', 'Validation'], ['r', 'b']):
    mask = np.array(y) == dataset  # Create a boolean mask for each dataset
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=dataset, alpha=0.6, color=color)

plt.title('PCA Plot of Top 30% Expressed Genes')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)')
plt.legend()
plt.show()
# %%
##^^ analyze PC1 and PC2

# 4. Extract PCA Loadings (components_)
loadings = pd.DataFrame(pca.components_.T, 
                        index=df_top_30_percent.index, 
                        columns=['PC1', 'PC2'])

# 5. Get the top 10 transcripts contributing to PC1 and PC2
top_PC1_genes = loadings['PC1'].abs().sort_values(ascending=False).head(20)
top_PC2_genes = loadings['PC2'].abs().sort_values(ascending=False).head(20)

# Print the results
print("Top 20 Transcripts Contributing to PC1:")
print(top_PC1_genes)

print("\nTop 20 Transcripts Contributing to PC2:")
print(top_PC2_genes)
# %
# 

#%%
##^^^^ batch correction with Pycombat (Top  30%)

# Step 1: Merge datasets and create batch labels
common_index = df1.index  # Align by gene index
df1_aligned = df1.reindex(common_index)
df3_aligned = df3.reindex(common_index)

# Concatenate along columns
df_combined = pd.concat([df1_aligned, df3_aligned], axis=1)

# 3. Calculate the mean expression per gene (row-wise mean)
gene_means = df_combined.mean(axis=1)

# 4. Select the top 30% most highly expressed genes
top_30_percent_threshold = np.percentile(gene_means, 70)  # 70th percentile
top_30_percent_genes = gene_means[gene_means >= top_30_percent_threshold].index

# 5. Filter the dataset to include only the top 30% most highly expressed genes
df_top_30_percent = df_combined.loc[top_30_percent_genes]

# Create batch labels: 0 for POLO, 1 for Validation
batch_labels = [0] * df1_aligned.shape[1] + [1] * df3_aligned.shape[1]

# Step 2: Transpose the DataFrame (samples as rows, transcripts as columns)
X = df_top_30_percent.T

# Create a dictionary for batch information
covars = {
    'batch': batch_labels  # Batch labels (e.g., [0, 0, 0, ..., 1, 1, 1])
}

covars = pd.DataFrame(covars)

# Apply ComBat batch effect correction
combat_corrected = neuroCombat(
    dat=X.T,  # Transpose the data (genes as rows, samples as columns)
    covars=covars,  # Dictionary containing the batch information
    batch_col='batch',  # Name of the column in covars with batch info
)['data']

# Step 4: Transpose back (samples as rows, genes as columns) for PCA
X_corrected = combat_corrected.T

# Step 5: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_corrected)

# Step 6: Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 7: Create labels for the samples
y = ['POLO'] * df1_aligned.shape[1] + ['Validation'] * df3_aligned.shape[1]

# Step 8: Plot PCA results after batch correction
plt.figure(figsize=(8, 6))
for dataset, color in zip(['POLO', 'Validation'], ['r', 'b']):
    mask = np.array(y) == dataset
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=dataset, alpha=0.6, color=color)

plt.title('PCA Plot After ComBat Batch Effect Correction')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)')
plt.legend()
plt.show()



# %%
##^^^^ batch correction (correct only POLO) with Pycombat (Top  30%)

# Step 1: Align genes (rows) between df1 and df3 to ensure they match
common_index = df3.index  # Use df3's index as the common order
df1_aligned = df1.reindex(common_index)
df3_aligned = df3.reindex(common_index)

# Step 2: Compute the mean expression for each transcript across both datasets
mean_expression = pd.concat([df1_aligned, df3_aligned], axis=1).mean(axis=1)

# Step 3: Select the top 30% most expressed transcripts
top_30_percent_genes = mean_expression.nlargest(int(len(mean_expression) * 0.3)).index

# Step 4: Filter df1 to only include the top 30% expressed transcripts
df1_filtered = df1_aligned.loc[top_30_percent_genes]

# Step 5: Transpose df1 (samples as rows, genes as columns)
df1_transposed = df1_filtered.T  # Now shape: (samples, top 30% genes)

# Step 6: Create batch labels for df1
batch_labels = [0] * df1_transposed.shape[0]  # All samples belong to batch 0

# Step 7: Create the covariates DataFrame
covars = pd.DataFrame({'batch': batch_labels})
# Step 8: Apply ComBat batch correction
combat_corrected_df1 = neuroCombat(
    dat=df1_transposed.T,  # Transpose again to (genes, samples) for neuroCombat
    covars=covars,         # Covariates with batch info
    batch_col='batch'      # Specify batch column
)['data']

df1_corrected = pd.DataFrame(combat_corrected_df1).T
print(f"df1_corrected shape: {df1_corrected.shape}")  # (samples, genes)

# Check df1_filtered shape
print(f"df1_filtered shape: {df1_filtered.shape}")  # (genes, samples)

# Ensure dimensions align before setting index and columns
df1_corrected.index = df1_filtered.columns  # Restore sample names
df1_corrected.columns = df1_filtered.index  # Restore gene index

print("Index and columns restored successfully.")

# Step 10: Concatenate corrected df1 with the original df3 (same top 30% genes)
df_combined = pd.concat([df1_corrected, df3_aligned.loc[top_30_percent_genes]], axis=1)

# Step 11: Transpose combined data for PCA (samples as rows, genes as columns)
X = df_combined.T  # Now shape: (samples, genes)

# Step 12: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 13: Perform PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 14: Create sample labels aligned with the columns of df_combined
y = np.array(['POLO'] * df1_filtered.shape[1] + ['Validation'] * df3_aligned.loc[top_30_percent_genes].shape[1])

# Check the number of labels and PCA samples for consistency
print(f"Number of labels: {len(y)}, PCA samples: {X_pca.shape[0]}")

# Step 15: Plot the PCA results
plt.figure(figsize=(8, 6))
for dataset, color in zip(['POLO', 'Validation'], ['r', 'b']):
    mask = (y == dataset)  # Boolean mask for samples
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=dataset, alpha=0.6, color=color)

plt.title('PCA Plot After ComBat Batch Correction on df1 Only (Top 30% Transcripts)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}% variance)')
plt.legend()
plt.show()
# %%
