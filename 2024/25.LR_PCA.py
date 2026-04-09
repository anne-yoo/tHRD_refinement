#%%
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.multitest as ssm
import scipy as sp
import pickle
import sys
import os
import re
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import plotly.express as px
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 12,    # x축 틱 라벨 글꼴 크기
'ytick.labelsize': 12,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 13,
'legend.title_fontsize': 13, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})

#%%
########## MAKE gene TPM FILE#####################################
# # Step 1: Load the raw read counts matrix (rows = ENSG IDs, columns = samples)
# counts_df = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_gene_readcount.txt', sep='\t', index_col=0)  # Replace with your counts file path

# # Step 2: Load the gene length file
# gene_info = pd.read_csv('/home/jiye/jiye/copycomparison/GC_POLO_tHRD/rawfiles/trim/2nd_align/readcounts/PL-OV-P049_readcounts_tpm.txt', sep='\t')

# # Step 3: Extract relevant columns (ENSG ID and Length) from the gene info file
# lengths_df = gene_info[['Geneid', 'Length']].set_index('Geneid')

# # Step 5: Merge counts with gene lengths on ENSG ID
# merged_df = counts_df.merge(lengths_df, left_index=True, right_index=True, how='inner')

# # Step 6: Separate the Length column and the counts matrix
# lengths = merged_df['Length']
# counts_matrix = merged_df.drop('Length', axis=1)

# # Step 7: Calculate RPK (Reads Per Kilobase)
# rpk = counts_matrix.div(lengths / 1000, axis=0)

# # Step 8: Calculate scaling factors (sum of RPKs per sample)
# scaling_factors = rpk.sum(axis=0) / 1e6  # Divide by 1 million

# # Step 9: Calculate TPM by dividing RPK by the scaling factor for each sample
# tpm = rpk.div(scaling_factors, axis=1)

# # Step 10: Save the TPM matrix to a CSV file
# #tpm.to_csv('tpm_results.csv')

# # Optional: Check the first few rows of the TPM matrix
# print(tpm.head())

# #tpm.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_gene_TPM.txt', sep='\t', header=True)






#%%
#^^ val+dis pre TU ########

# Step 1: Load the dataset
tu = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/156_validation_TU.txt', sep='\t', index_col=0)
clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_validation_clinicalinfo.txt', sep='\t')
res = tu.T
res = res.iloc[:,:-2]
res['response'] = clin['response'].to_list()
res['y'] = clin['interval'].to_list()
res = res.loc[res['response']==1,:]

# Step 2: Separate features (X) and target (y)
X = res.iloc[:, :-2]  # All columns except the last one (transcripts)
y = res.iloc[:, -1]   # Last column (Duration)
X = X.dropna(axis=1)

n_components = 10  # Select the top 10 components (PC1~PC10)
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

loading_scores = np.abs(pca.components_).sum(axis=0)

# Step 4: Select the top-K features based on the summed loadings
K = 70  # Number of features to select (adjust as needed)
top_feature_indices = np.argsort(loading_scores)[-K:]  # Indices of top-K features

# Step 5: Subset the original dataset to keep only the top-K features
X_selected = X.iloc[:, top_feature_indices]

# Display the shape of the selected dataset
print(f"Selected dataset shape: {X_selected.shape}")

# Step 3: Fit the Linear Regression model
model = LinearRegression()
model.fit(X_selected, y)

# Step 4: Make predictions using the fitted model
y_pred = model.predict(X_selected)

# Step 5: Calculate Total Variance (V_D)
y_mean = np.mean(y)
V_D = np.sum((y - y_mean) ** 2)

# Step 6: Calculate Explained Variance (V_D_hat)
y_pred_mean = np.mean(y_pred)
V_D_hat = np.sum((y_pred - y_pred_mean) ** 2)

# Step 7: Compute the Explained Variance Ratio
explained_variance_ratio = V_D_hat / V_D

# Step 8: Display the Results
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")
print(f"Explained Variance Ratio: {explained_variance_ratio:.4f}")

# Optional: Save predicted results for analysis
res['Predicted_Duration'] = y_pred
#res.to_csv('predicted_results.csv', index=False)


#%%
#^^ val+dis pre TU - only major transcripts########

# Step 1: Load the dataset
tu = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/156_validation_TU.txt', sep='\t', index_col=0)
clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_validation_clinicalinfo.txt', sep='\t')
transcript_info = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')

major_transcripts = transcript_info[transcript_info['type'] == 'major']['gene_ENST']

res = tu.T
res = res.iloc[:,:-2]
res['response'] = clin['response'].to_list()
res['y'] = clin['interval'].to_list()
res = res.loc[res['response']==1,:]

# Step 2: Separate features (X) and target (y)
X = res.iloc[:, :-2]  # All columns except the last one (transcripts)
y = res.iloc[:, -1]   # Last column (Duration)
X = X[major_transcripts]
X = X.dropna(axis=1)

n_components = 10  # Select the top 10 components (PC1~PC10)
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

loading_scores = np.abs(pca.components_).sum(axis=0)

# Step 4: Select the top-K features based on the summed loadings
K = 100  # Number of features to select (adjust as needed)
top_feature_indices = np.argsort(loading_scores)[-K:]  # Indices of top-K features

# Step 5: Subset the original dataset to keep only the top-K features
X_selected = X.iloc[:, top_feature_indices]

# Display the shape of the selected dataset
print(f"Selected dataset shape: {X_selected.shape}")

# Step 3: Fit the Linear Regression model
model = LinearRegression()
model.fit(X_selected, y)

# Step 4: Make predictions using the fitted model
y_pred = model.predict(X_selected)

# Step 5: Calculate Total Variance (V_D)
y_mean = np.mean(y)
V_D = np.sum((y - y_mean) ** 2)

# Step 6: Calculate Explained Variance (V_D_hat)
y_pred_mean = np.mean(y_pred)
V_D_hat = np.sum((y_pred - y_pred_mean) ** 2)

# Step 7: Compute the Explained Variance Ratio
explained_variance_ratio = V_D_hat / V_D

# Step 8: Display the Results
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")
print(f"Explained Variance Ratio: {explained_variance_ratio:.4f}")

# Optional: Save predicted results for analysis
res['Predicted_Duration'] = y_pred
#res.to_csv('predicted_results.csv', index=False)

# %%
#%%

#^^ val+dis pre gene TPM ########

# Step 1: Load the dataset
gene_116 = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_gene_TPM.txt', sep='\t', index_col=0)
gene_dis = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM.txt', sep='\t', index_col=0)
gene_dis = gene_dis.iloc[:,1::2]

merged_gene = pd.concat([gene_116,gene_dis], axis=1)

clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_validation_clinicalinfo.txt', sep='\t')
res = merged_gene.T
res['response'] = clin['response'].to_list()
res['y'] = clin['interval'].to_list()
res = res.loc[res['response']==1,:]

# Step 2: Separate features (X) and target (y)
X = res.iloc[:, :-2]  # All columns except the last one (transcripts)
y = res.iloc[:, -1]   # Last column (Duration)
X = X.dropna(axis=1)

n_components = 10  # Select the top 10 components (PC1~PC10)
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

loading_scores = np.abs(pca.components_).sum(axis=0)

# Step 4: Select the top-K features based on the summed loadings
K = 70  # Number of features to select (adjust as needed)
top_feature_indices = np.argsort(loading_scores)[-K:]  # Indices of top-K features

# Step 5: Subset the original dataset to keep only the top-K features
X_selected = X.iloc[:, top_feature_indices]

# Display the shape of the selected dataset
print(f"Selected dataset shape: {X_selected.shape}")


# Step 3: Fit the Linear Regression model
model = LinearRegression()
model.fit(X_selected, y)

# Step 4: Make predictions using the fitted model
y_pred = model.predict(X_selected)

# Step 5: Calculate Total Variance (V_D)
y_mean = np.mean(y)
V_D = np.sum((y - y_mean) ** 2)

# Step 6: Calculate Explained Variance (V_D_hat)
y_pred_mean = np.mean(y_pred)
V_D_hat = np.sum((y_pred - y_pred_mean) ** 2)

# Step 7: Compute the Explained Variance Ratio
explained_variance_ratio = V_D_hat / V_D

# Step 8: Display the Results
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")
print(f"Explained Variance Ratio: {explained_variance_ratio:.4f}")

# Optional: Save predicted results for analysis
res['Predicted_Duration'] = y_pred
#res.to_csv('predicted_results.csv', index=False)



# %%
####** + POLO #############
polo_tu = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/POLO/83_POLO_TU.txt', sep='\t', index_col=0)
polo_gene = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/POLO/83_POLO_gene_exp_TPM.txt', sep='\t', index_col=0)


#%%
#####** PCA TU #################
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Step 2: Standardize the datasets (PCA works better with standardized data)
######################
df1 = tu
df1 = df1.iloc[:-2,:]
df1 = df1.apply(pd.to_numeric, errors='coerce') 
#df1 = df1.loc[major_transcripts.tolist()] ##only major transcripts
df1 = df1.dropna(axis=0)

tu_variances = df1.var(axis=1)
top_tu = tu_variances.nlargest(10000).index #whole:10000, major:3000
df1 = df1.loc[top_tu]
df1 = df1.T

#df2 = merged_gene
df2 = gene_116
gene_variances = df2.var(axis=1)
top_genes = gene_variances.nlargest(3000).index
df2 = df2.loc[top_genes]
df2 = df2.T

#%%
##^ only responders ####
# reslist = list(clin[clin['response']==1]['sample_id'])
# df1.index = df1.index.str.replace('-bfD', '', regex=False)
# df2.index = df2.index.str.replace('-bfD', '', regex=False)
# df1 = df1.loc[reslist]
# df2 = df2.loc[reslist]
######################

#####################
genelist = merged_gene.index.to_list()

#df1 = polo_tu
df1 = df1.dropna(axis=0)  # Remove cols with NaN
df2 = polo_gene
df2 = df2.loc[genelist]
df2 = df2.dropna(axis=0)

tu_variances = df1.var(axis=1)
top_tu = tu_variances.nlargest(10000).index
df1 = df1.loc[top_tu]
df1 = df1.T
gene_variances = df2.var(axis=1)
top_genes = gene_variances.nlargest(3000).index
df2 = df2.loc[top_genes]
df2 = df2.T
#####################

#%%
# Step 2: Standardize both datasets using StandardScaler (more robust than manual)
scaler = StandardScaler()
df1_scaled = scaler.fit_transform(df1)
df2_scaled = scaler.fit_transform(df2)



# Step 3: Run PCA on both datasets
pca1 = PCA()
pca2 = PCA()

pca1_results = pca1.fit_transform(df1_scaled)
pca2_results = pca2.fit_transform(df2_scaled)

# Step 4: Create scree plots for both datasets
# def scree_plot(pca, dataset_name, max_components=50):
#     plt.figure()
#     num_components = min(max_components, len(pca.explained_variance_ratio_))
#     plt.plot(np.arange(1, num_components + 1), 
#             pca.explained_variance_ratio_[:num_components], marker='o')
#     plt.title(f'Scree Plot - {dataset_name}')
#     plt.xlabel('Principal Component')
#     plt.ylabel('Explained Variance Ratio')
#     plt.show()

# scree_plot(pca1, 'Dataset 1')
# scree_plot(pca2, 'Dataset 2')

# Step 5: Determine the number of PCs to keep (n) based on cumulative variance
n=10
# Step 6: Extract the top `n` PCs from both datasets
pc1_topn = pca1_results[:, :n]
pc2_topn = pca2_results[:, :n]

# Step 7: Compute correlation between the PCs of the two datasets
corr_matrix = np.corrcoef(pc1_topn.T, pc2_topn.T)[:n, n:]

# Step 8: Display the correlation matrix as a heatmap
plt.figure(figsize=(8, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
            xticklabels=[f'TU_PC{i+1}' for i in range(n)],
            yticklabels=[f'gene_PC{i+1}' for i in range(n)])
plt.title('Correlation between PCs of TU and gene exp')
plt.show()

# Step 4: Split the data into training and test sets
model = LinearRegression()
model.fit(pc1_topn, y)
y_pred = model.predict(pc1_topn)

# Step 3: Calculate the total variance (V_D)
y_mean = np.mean(y)  # Mean of the target values
V_D = np.sum((y - y_mean) ** 2)  # Total variance in y

# Step 4: Calculate the residual variance (V_D_hat)
residual_variance = np.sum((y - y_pred) ** 2)

# Step 5: Calculate the explained variance ratio
explained_variance_ratio = 1 - (residual_variance / V_D)

# Step 6: Display the results
print(f'TU Model Coefficients: {model.coef_}')
print(f'TU Model Intercept: {model.intercept_}')
print(f'TU Explained Variance Ratio: {explained_variance_ratio:.4f}')

# Repeat the same for the Gene Expression PC-based model
model = LinearRegression()
model.fit(pc2_topn, y)
y_pred = model.predict(pc2_topn)  # Fit using all data

# Step 3: Calculate the total variance (V_D)
y_mean = np.mean(y)  # Mean of the target values
V_D = np.sum((y - y_mean) ** 2)  # Total variance in y

# Step 4: Calculate the residual variance (V_D_hat)
residual_variance = np.sum((y - y_pred) ** 2)

# Step 5: Calculate the explained variance ratio
explained_variance_ratio = 1 - (residual_variance / V_D)

# Step 6: Display the results
print(f'Gene exp Model Coefficients: {model.coef_}')
print(f'Gene exp Model Intercept: {model.intercept_}')
print(f'Gene exp Explained Variance Ratio: {explained_variance_ratio:.4f}')


# %%
###^^ gene level vs. transcript level #######
#####**** (1) PCA #################################################
df1 = tu
df1 = df1.iloc[:-2,:]
#df1 = df1.loc[major_transcripts.tolist()]
df1 = df1.dropna(axis=0)
df1 = df1.apply(pd.to_numeric, errors='coerce') 
tu_variances = df1.var(axis=1)
top_tu = tu_variances.nlargest(10000).index #whole:10000, major:3000
df1 = df1.loc[top_tu]

#df2 = merged_gene
df2 = gene_116
gene_variances = df2.var(axis=1)
top_genes = gene_variances.nlargest(2000).index
df2 = df2.loc[top_genes]

pca_gene = PCA(n_components=2).fit_transform(df2)
pca_transcript = PCA(n_components=2).fit_transform(df1)

#drug_response = pd.Series(list(clin['response']), index=df1.index)

# Plot PCA results for both datasets side by side

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].scatter(pca_gene[:, 0], pca_gene[:, 1], alpha=0.7, s=8, c='#0C356A')
axes[0].set_title('Gene Exp Level')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

axes[1].scatter(pca_transcript[:, 0], pca_transcript[:, 1], alpha=0.7, s=8, c='#347928')
axes[1].set_title('Transcript Usage Level')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')

plt.tight_layout()
plt.show()

# %%
#####^^^^ Correlaiotn Heatmap ##############################
import seaborn as sns
from sklearn.preprocessing import StandardScaler

tu = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_TU.txt', sep='\t', index_col=0)
df1 = tu
df1 = df1.iloc[:-2,:]
#df1 = df1.loc[major_transcripts.tolist()]
df1 = df1.dropna(axis=0)
df1 = df1.apply(pd.to_numeric, errors='coerce')  # Convert all values to numeric, turning non-numeric entries into NaN

tu_variances = df1.var(axis=1)
top_tu = tu_variances.nlargest(10000).index #whole:10000, major:3000
df1 = df1.loc[top_tu]
df1 = df1.apply(pd.to_numeric, errors='coerce')
scaler = StandardScaler()
# df1_norm = pd.DataFrame(scaler.fit_transform(df1.T), 
#                             index=df1.columns,
#                             columns=df1.index).T

gene_116 = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_gene_TPM.txt', sep='\t', index_col=0)
df2 = gene_116
gene_variances = df2.var(axis=1)
top_genes = gene_variances.nlargest(2000).index 
df2 = df2.loc[top_genes]
df2 = df2.apply(pd.to_numeric, errors='coerce')
df2.columns = df2.columns.str.replace('-bfD', '', regex=False)
scaler = StandardScaler()
# df2_norm = pd.DataFrame(scaler.fit_transform(df2.T), 
#                             index=df2.columns,
#                             columns=df2.index).T


gene_corr = df2.corr()  # Gene-level correlation
transcript_corr = df1.corr()  # Transcript-level correlation

# 4. Plot heatmaps for both matrices
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

sns.heatmap(gene_corr, ax=axes[0], cmap='coolwarm', center=0.5)
axes[0].set_title('Gene Exp Level Corr.')

sns.heatmap(transcript_corr, ax=axes[1], cmap='coolwarm', center=0.5)
axes[1].set_title('Transcript Exp Level Corr.')

plt.tight_layout()
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/intro/gene_trans_TPM_exp_corr.pdf',dpi=300, format='pdf', bbox_inches='tight', transparent=True, pad_inches=0.1)
plt.show()

# %%
####^^ K-means + sample heatmap #######################
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import contingency

tu = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/156_validation_TU.txt', sep='\t', index_col=0)
df1 = tu
df1 = df1.iloc[:-2,:]
#df1 = df1.loc[major_transcripts.tolist()]
df1 = df1.dropna(axis=0)
df1 = df1.apply(pd.to_numeric, errors='coerce')

# Step 2: Transpose both datasets to have samples as rows (necessary for clustering)
gene_data_T = df2.T  # (samples, genes)
transcript_data_T = df1.T  # (samples, transcripts)

# Step 3: Apply K-means clustering (using K=4 as an example; adjust based on your data)
k = 3  # Set the number of clusters (you can use other values too)
kmeans_gene = KMeans(n_clusters=k, random_state=42).fit(gene_data_T)
kmeans_transcript = KMeans(n_clusters=k, random_state=42).fit(transcript_data_T)

# Step 4: Get cluster labels for both datasets
gene_clusters = kmeans_gene.labels_
transcript_clusters = kmeans_transcript.labels_

# Step 5: Create a contingency table to compare the cluster assignments
cont_table = pd.crosstab(gene_clusters, transcript_clusters)
print("Contingency Table (Gene vs Transcript Clustering):")
print(cont_table)

# Step 6: Calculate clustering similarity metrics (ARI and NMI)
ari_score = adjusted_rand_score(gene_clusters, transcript_clusters)
nmi_score = normalized_mutual_info_score(gene_clusters, transcript_clusters)

print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}")


# Step 7: Visualize the contingency table as a heatmap
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5))
sns.heatmap(cont_table, annot=True, cmap='Blues', fmt='d')
plt.title('Contingency Table: Gene Exp vs Transcript Usage Clustering')
plt.xlabel('Transcript Usage Clusters')
plt.ylabel('Gene Exp Clusters')
# plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/contingency_gene_TU.pdf', 
#             dpi=300,               # High resolution (300+ dpi)
#             format='pdf',           # Save as vector-based PDF for quality
#             bbox_inches='tight',    # Removes extra whitespace
#             transparent=True,       # Background transparency (optional)
#             pad_inches=0.1)         # Small padding around the plot
plt.show()

from sklearn.metrics import adjusted_rand_score

ari_score = adjusted_rand_score(gene_clusters, transcript_clusters)
print(f"Adjusted Rand Index (ARI): {ari_score:.4f}")

from sklearn.metrics import normalized_mutual_info_score

nmi_score = normalized_mutual_info_score(gene_clusters, transcript_clusters)
print(f"Normalized Mutual Information (NMI): {nmi_score:.4f}")

#%%
sns.set_theme(style='ticks')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Arial"

pca = PCA(n_components=2)
pca_result = pca.fit_transform(transcript_data_T)

cluster_labels = kmeans_transcript.labels_

# Step 4: Plot the PCA results, coloring points by their cluster
plt.figure(figsize=(6, 6))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], 
                hue=cluster_labels, palette='tab10', 
                style=cluster_labels, s=80, legend='full')
plt.title('K-means Clustering (K=3) in PCA Space: Transcript Usage')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/kmeans_geneexp.pdf', 
#             dpi=300,               # High resolution (300+ dpi)
#             format='pdf',           # Save as vector-based PDF for quality
#             bbox_inches='tight',    # Removes extra whitespace
#             transparent=True,       # Background transparency (optional)
#             pad_inches=0.1)         # Small padding around the plot
plt.show()

pca = PCA(n_components=2)
pca_result = pca.fit_transform(gene_data_T)

cluster_labels = kmeans_gene.labels_

# Step 4: Plot the PCA results, coloring points by their cluster
plt.figure(figsize=(6, 6))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], 
                hue=cluster_labels, palette='tab10', 
                style=cluster_labels, s=80, legend='full')
plt.title('K-means Clustering (K=3) in PCA Space: Gene Expression')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/kmeans_TU.pdf', 
#             dpi=300,               # High resolution (300+ dpi)
#             format='pdf',           # Save as vector-based PDF for quality
#             bbox_inches='tight',    # Removes extra whitespace
#             transparent=True,       # Background transparency (optional)
#             pad_inches=0.1) 
plt.show()
# %%
