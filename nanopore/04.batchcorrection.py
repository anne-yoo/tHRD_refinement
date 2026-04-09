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
from combat.pycombat import pycombat

# %%
readcount = pd.read_csv('/home/jiye/jiye/nanopore/FINALDATA/137_gene_readcount.txt', sep='\t', index_col=0)
tpm = pd.read_csv('/home/jiye/jiye/nanopore/FINALDATA/137_gene_TPM.txt', sep='\t', index_col=0)
readcount = readcount.apply(pd.to_numeric, errors='coerce')  # Convert all values to numeric, setting non-convertible values to NaN
tpm = tpm.iloc[:,1:]
tpm = tpm.apply(pd.to_numeric, errors='coerce')  # Convert all values to numeric, setting non-convertible values to NaN
tpm = tpm.dropna()
readcount = readcount.dropna()
tpm = tpm.loc[tpm.nunique(axis=1) > 1]
readcount = readcount.loc[readcount.nunique(axis=1) > 1]

# Predefined list of GridION samples
gridion_samples = [
    'PM-AU-0002-N', 'PM-AU-0004-N', 'PM-AU-0006-N', 'PM-AU-0007-N', 'PM-AU-0008-N',
    'PM-AU-0024-N', 'PM-AU-0025-N', 'PM-AU-0033-N', 'PM-AU-0037-N', 'PM-AU-0038-N',
    'PM-PA-1001-N', 'PM-PA-1004-N', 'PM-PS-0001-N', 'PM-PS-0005-N', 'PM-PS-0009-N',
    'PM-PS-0011-N', 'PM-PS-0013-N', 'PM-PS-0014-N', 'PM-PS-0016-N', 'PM-PS-0017-N',
    'PM-PS-0018-N', 'PM-PS-0019-N', 'PM-PS-0024-N', 'PM-PS-0026-N', 'PM-PS-0027-N',
    'PM-PS-0028-N', 'PM-PS-0456-N', 'PM-PS-0485-N', 'PM-PU-1002-N', 'PM-PU-1004-N',
    'PM-PU-1005-N', 'PM-PU-1009-N', 'PM-PU-1010-N', 'PM-PU-1011-N', 'PM-PU-1012-N',
    'PM-PU-1013-N', 'PM-PU-1014-N', 'PM-PU-1015-N', 'PM-PU-1024-N', 'PM-PU-1025-N',
    'PM-PU-1026-N', 'PM-PU-1027-N',
    'PM-AU-0002-T', 'PM-AU-0004-T', 'PM-AU-0006-T', 'PM-AU-0007-T', 'PM-AU-0008-T',
    'PM-AU-0024-T', 'PM-AU-0025-T', 'PM-AU-0033-T', 'PM-AU-0037-T', 'PM-AU-0038-T',
    'PM-PA-1001-T', 'PM-PA-1004-T', 'PM-PS-0001-T', 'PM-PS-0005-T', 'PM-PS-0009-T',
    'PM-PS-0011-T', 'PM-PS-0013-T', 'PM-PS-0014-T', 'PM-PS-0016-T', 'PM-PS-0017-T',
    'PM-PS-0018-T', 'PM-PS-0019-T', 'PM-PS-0024-T', 'PM-PS-0026-T', 'PM-PS-0027-T',
    'PM-PS-0028-T', 'PM-PS-0456-T', 'PM-PS-0485-T', 'PM-PU-1002-T', 'PM-PU-1004-T',
    'PM-PU-1005-T', 'PM-PU-1009-T', 'PM-PU-1010-T', 'PM-PU-1011-T', 'PM-PU-1012-T',
    'PM-PU-1013-T', 'PM-PU-1014-T', 'PM-PU-1015-T', 'PM-PU-1024-T', 'PM-PU-1025-T',
    'PM-PU-1026-T', 'PM-PU-1027-T'
]

# Full list of samples (replace with your actual list of all samples)
all_samples = readcount.columns.to_list()

# Creating the batch vector: 0 if in GridION samples, 1 otherwise
batches = [0 if sample in gridion_samples else 1 for sample in all_samples]

# Output the batch vector
#print(batches)

corrected_readcount = pycombat(readcount, batch=batches)
corrected_tpm = pycombat(tpm, batch=batches)

corrected_readcount.to_csv('/home/jiye/jiye/nanopore/FINALDATA/batchcorrection/137_gene_readcount.txt', sep='\t')
corrected_tpm.to_csv('/home/jiye/jiye/nanopore/FINALDATA/batchcorrection/137_gene_TPM.txt', sep='\t')

#%%
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Assuming `expr_matrix` is your data matrix with samples as columns and transcripts as rows
# Transpose the matrix to have samples as rows and transcripts as columns for PCA
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 13,    # x축 틱 라벨 글꼴 크기

'ytick.labelsize': 13,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 11,
'legend.title_fontsize': 11, # 범례 글꼴 크기
'figure.titlesize': 16    # figure 제목 글꼴 크기
})
sns.set_style("white")

###################
expr_matrix = readcount
###################
num_samples = expr_matrix.shape[1]  # Total number of samples (columns)
min_samples_required = int(0.4 * num_samples)  # 40% of the total samples

# Filter transcripts that are expressed above the threshold in at least 40% of the samples
filtered_expr_matrix = expr_matrix[(expr_matrix > 0).sum(axis=1) >= min_samples_required]

expr_matrix_T = filtered_expr_matrix.T

# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(expr_matrix_T)

# Create a DataFrame for the PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
pca_df['Sample'] = expr_matrix.columns  # Add sample names

#%%
plt.figure(figsize=(8, 6))
color_map = {'Normal': '#187498', 'Tumor':'#EB5353'}
pca_df['Color'] = np.where(pca_df['Sample'].str.endswith('-N'), '#187498', 
                            np.where(pca_df['Sample'].str.endswith('-T'), '#EB5353', 'grey'))

for i, row in pca_df.iterrows():
    plt.scatter(row['PC1'], row['PC2'], color=row['Color'], alpha=0.8, s=60)

# Add labels for samples
# for i, row in tsne_df.iterrows():
#     plt.text(row['t-SNE 1'], row['t-SNE 2'], row['Sample'], fontsize=8, ha='right', va='bottom')


# Adding a legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=11, label=label)
                for label, color in color_map.items()]
plt.legend(handles=legend_elements, title="Sample Type")

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('readcount PCA before batch correction')
#plt.grid()
plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/PCA_readcount_beforecorrection_NT.pdf', dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)

plt.show()

#%%
# Add a column to indicate if a sample is in the list of interest
pca_df['Color'] = np.where(pca_df['Sample'].isin(gridion_samples), 'red', 'lightgrey')  # 'red' for interest, 'blue' otherwise

# Plotting PCA
plt.figure(figsize=(8, 6))
for i, row in pca_df.iterrows():
    plt.scatter(row['PC1'], row['PC2'], color=row['Color'], alpha=0.7, label=row['Sample'] if row['Sample'] in gridion_samples else "")
    # if row['Sample'] in samples_of_interest:
    #     plt.text(row['PC1'], row['PC2'], row['Sample'], fontsize=9, ha='right')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('readcount PCA before batch correction')
plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/PCA_readcount_beforecorrection.pdf', dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)

plt.show()# %%

#%%
import umap
# Perform UMAP
reducer = umap.UMAP(n_components=2, random_state=42)  # You can tune parameters as needed
umap_result = reducer.fit_transform(expr_matrix_T)

# Create a DataFrame for the UMAP results
umap_df = pd.DataFrame(data=umap_result, columns=['UMAP 1', 'UMAP 2'])
umap_df['Sample'] = expr_matrix.columns  # Add sample names
umap_df['Color'] = np.where(umap_df['Sample'].str.endswith('-N'), '#187498', 
                            np.where(umap_df['Sample'].str.endswith('-T'), '#EB5353', 'grey'))  # Default to 'grey' if neither '-N' nor '-T'


# Plotting t-SNE
plt.figure(figsize=(8, 6))
for i, row in umap_df.iterrows():
    plt.scatter(row['UMAP 1'], row['UMAP 2'], color=row['Color'], alpha=0.8, s=40)


# Adding a legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=11, label=label)
                for label, color in color_map.items()]
plt.legend(handles=legend_elements, title="Sample Type")

plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('readcount UMAP before batch correction')
plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/UMAP_readcount_beforecorrection_NT.pdf', dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)

#plt.grid()
plt.show()
#%%
umap_df['Color'] = np.where(umap_df['Sample'].isin(gridion_samples), 'red', 'lightgrey')  # 'red' for interest, 'blue' otherwise

# Plotting PCA
plt.figure(figsize=(8, 6))
for i, row in umap_df.iterrows():
    plt.scatter(row['UMAP 1'], row['UMAP 2'], color=row['Color'], alpha=0.7, label=row['Sample'] if row['Sample'] in gridion_samples else "")
    # if row['Sample'] in samples_of_interest:
    #     plt.text(row['PC1'], row['PC2'], row['Sample'], fontsize=9, ha='right')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('readcount UMAP before batch correction')
plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/UMAP_readcount_beforecorrection.pdf', dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)

plt.show()# %%

# %%
#corrected_readcount.to_csv('/home/jiye/jiye/nanopore/FINALDATA/batchcorrection/137_transcript_readcount.txt', sep='\t')
corrected_tpm.to_csv('/home/jiye/jiye/nanopore/FINALDATA/batchcorrection/137_transcript_TPM.txt', sep='\t')

# %%
sampleinfo = pd.DataFrame({'batch':batches, 'sample_type':['Normal','Tumor']*137, 'sample':tpm.columns.to_list()})
sampleinfo.to_csv('/home/jiye/jiye/nanopore/FINALDATA/batchcorrection/137_batch_type_info.txt', sep='\t')
# %%
