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
# %%

clin = pd.read_csv('/home/jiye/jiye/nanopore/FINALDATA/137_clinicaldata.txt', sep='\t')
tpm = pd.read_csv('/home/jiye/jiye/nanopore/FINALDATA/137_transcript_TPM.txt', sep='\t', index_col=0)

#%%
det = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/whole_DETresult.txt', sep=',', index_col=0)
detlist = list(det.loc[(det['padj']<0.01) & (det['log2FoldChange']>1.5)].index)
filtered_expr_matrix = tpm.loc[detlist,:]
# %%
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Assuming `expr_matrix` is your data matrix with samples as columns and transcripts as rows
# Transpose the matrix to have samples as rows and transcripts as columns for PCA

expr_matrix = tpm
num_samples = expr_matrix.shape[1]  # Total number of samples (columns)
min_samples_required = int(0.4 * num_samples)  # 40% of the total samples

# Filter transcripts that are expressed above the threshold in at least 40% of the samples
filtered_expr_matrix = expr_matrix[(expr_matrix > 0).sum(axis=1) >= min_samples_required]

#%%
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
plt.title('PCA with transcript TPM values')
#plt.grid()
#plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/PCA_normal_tumor.pdf', dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)

plt.show()
#%%
samples_of_interest = ['PM-AU-0002-N',
'PM-AU-0006-N',
'PM-AU-0006-T',
'PM-AU-0008-N',
'PM-AU-0008-T',
'PM-AU-0033-N',
'PM-AU-0033-T',
'PM-AU-0037-N',
'PM-AU-0037-T',
'PM-AU-0038-N',
'PM-AU-0038-T',
'PM-PA-1004-N',
'PM-PA-1004-T',
'PM-PS-0005-N',
'PM-PS-0005-T',
'PM-PS-0009-N',
'PM-PS-0011-N',
'PM-PS-0013-N',
'PM-PS-0014-N',
'PM-PS-0014-T',
'PM-PS-0016-N',
'PM-PS-0016-T',
'PM-PS-0019-N',
'PM-PS-0024-N',
'PM-PS-0024-T',
'PM-PS-0026-N',
'PM-PS-0026-T',
'PM-PS-0027-N',
'PM-PS-0028-N',
'PM-PS-0028-T',
'PM-PU-1002-N',
'PM-PU-1002-T',
'PM-PU-1004-N',
'PM-PU-1004-T',
'PM-PU-1009-N',
'PM-PU-1009-T',
'PM-PU-1010-N',
'PM-PU-1010-T',
'PM-PU-1012-N',
'PM-PU-1012-T',
'PM-PU-1013-N',
'PM-PU-1013-T',
'PM-PU-1014-N',
'PM-PU-1014-T',
'PM-PU-1015-N',
'PM-PU-1024-N',
'PM-PU-1024-T',
'PM-PU-1025-N',
'PM-PU-1025-T',
'PM-PU-1026-N',
'PM-PU-1026-T',
'PM-PU-1027-N',
'PM-PU-1027-T']
#%%
# Add a column to indicate if a sample is in the list of interest
pca_df['Color'] = np.where(pca_df['Sample'].isin(samples_of_interest), 'red', 'lightgrey')  # 'red' for interest, 'blue' otherwise

# Plotting PCA
plt.figure(figsize=(8, 6))
for i, row in pca_df.iterrows():
    plt.scatter(row['PC1'], row['PC2'], color=row['Color'], alpha=0.7, label=row['Sample'] if row['Sample'] in samples_of_interest else "")
    # if row['Sample'] in samples_of_interest:
    #     plt.text(row['PC1'], row['PC2'], row['Sample'], fontsize=9, ha='right')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA with TPM values')
#plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/PCA_outliers.pdf', dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)

plt.show()# %%

# %%
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# Perform t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)  # You can tune perplexity as needed
tsne_result = tsne.fit_transform(expr_matrix_T)

# Create a DataFrame for the t-SNE results
tsne_df = pd.DataFrame(data=tsne_result, columns=['t-SNE 1', 't-SNE 2'])
tsne_df['Sample'] = expr_matrix.columns  # Add sample names
#%%
# Determine the color based on sample name ending
tsne_df['Color'] = np.where(tsne_df['Sample'].str.endswith('-N'), '#187498', 
                            np.where(tsne_df['Sample'].str.endswith('-T'), '#EB5353', 'grey'))  # Default to 'grey' if neither '-N' nor '-T'


# Plotting t-SNE
# Mapping for legend
color_map = {'Normal': '#187498', 'Tumor':'#EB5353'}
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
#sns.set_style("ticks")

# Plotting t-SNE
plt.figure(figsize=(8, 6))
for i, row in tsne_df.iterrows():
    plt.scatter(row['t-SNE 1'], row['t-SNE 2'], color=row['Color'], alpha=0.8, s=60)

# Add labels for samples
# for i, row in tsne_df.iterrows():
#     plt.text(row['t-SNE 1'], row['t-SNE 2'], row['Sample'], fontsize=8, ha='right', va='bottom')


# Adding a legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=11, label=label)
                for label, color in color_map.items()]
plt.legend(handles=legend_elements, title="Sample Type")

plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE with transcript TPM values')
#plt.grid()
##plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/tSNE_normal_tumor.pdf', dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)

plt.show()
#%%
#samples_of_interest = ['PM-PB-1084-N','PM-PB-1085-N','PM-PB-1086-N','PM-PB-1087-N','PM-PB-1088-N','PM-PB-1089-N','PM-PB-1090-N','PM-AU-1002-N','PM-AU-1006-N','PM-PB-1084-T','PM-PB-1085-T','PM-PB-1086-T','PM-PB-1087-T','PM-PB-1088-T','PM-PB-1089-T','PM-PB-1090-T','PM-AU-1002-T','PM-AU-1006-T']
# Add a column to indicate if a sample is in the list of interest
tsne_df['Color'] = np.where(tsne_df['Sample'].isin(samples_of_interest), 'red', 'lightgrey')  # 'red' for interest, 'blue' otherwise

# Plotting PCA
plt.figure(figsize=(8, 6))
for i, row in tsne_df.iterrows():
    plt.scatter(row['t-SNE 1'], row['t-SNE 2'], color=row['Color'], alpha=0.7, label=row['Sample'] if row['Sample'] in samples_of_interest else "")
    # if row['Sample'] in samples_of_interest:
    #     plt.text(row['PC1'], row['PC2'], row['Sample'], fontsize=9, ha='right')

plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.title('t-SNE with transcript TPM values')
#plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/tSNE_outliers.pdf', dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)

plt.show()# %%
# %%
import umap
# Perform UMAP
reducer = umap.UMAP(n_components=2, random_state=42)  # You can tune parameters as needed
umap_result = reducer.fit_transform(expr_matrix_T)

# Create a DataFrame for the UMAP results
umap_df = pd.DataFrame(data=umap_result, columns=['UMAP 1', 'UMAP 2'])
umap_df['Sample'] = expr_matrix.columns  # Add sample names

# Determine the color based on sample name ending
umap_df['Color'] = np.where(umap_df['Sample'].str.endswith('-N'), '#187498', 
                            np.where(umap_df['Sample'].str.endswith('-T'), '#EB5353', 'grey'))  # Default to 'grey' if neither '-N' nor '-T'


# Plotting t-SNE
# Mapping for legend
color_map = {'Normal': '#187498', 'Tumor':'#EB5353'}
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
#sns.set_style("ticks")
#%%
# Plotting t-SNE
plt.figure(figsize=(20, 15))
for i, row in umap_df.iterrows():
    plt.scatter(row['UMAP 1'], row['UMAP 2'], color=row['Color'], alpha=0.8, s=40)

##Add labels for samples
# for i, row in umap_df.iterrows():
#     plt.text(row['UMAP 1'], row['UMAP 2'], row['Sample'], fontsize=10, ha='right', va='bottom')


# Adding a legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=11, label=label)
                for label, color in color_map.items()]
plt.legend(handles=legend_elements, title="Sample Type")

plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('UMAP with transcript TPM values')
#plt.grid()
plt.show()

#%%
#samples_of_interest = ['PM-PB-1084-N','PM-PB-1085-N','PM-PB-1086-N','PM-PB-1087-N','PM-PB-1088-N','PM-PB-1089-N','PM-PB-1090-N','PM-AU-1002-N','PM-AU-1006-N','PM-PB-1084-T','PM-PB-1085-T','PM-PB-1086-T','PM-PB-1087-T','PM-PB-1088-T','PM-PB-1089-T','PM-PB-1090-T','PM-AU-1002-T','PM-AU-1006-T']
# Add a column to indicate if a sample is in the list of interest
umap_df['Color'] = np.where(umap_df['Sample'].isin(samples_of_interest), 'red', 'lightgrey')  # 'red' for interest, 'blue' otherwise

# Plotting PCA
plt.figure(figsize=(8, 6))
for i, row in umap_df.iterrows():
    plt.scatter(row['UMAP 1'], row['UMAP 2'], color=row['Color'], alpha=0.7, label=row['Sample'] if row['Sample'] in samples_of_interest else "")
    # if row['Sample'] in samples_of_interest:
    #     plt.text(row['PC1'], row['PC2'], row['Sample'], fontsize=9, ha='right')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('UMAP with transcript TPM values')
#plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/tSNE_outliers.pdf', dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)

plt.show()# %%
# %%
samplelist = ['PM-AU-0002','PM-AU-0006','PM-PB-1084','PM-PB-1085','PM-PB-1086','PM-PB-1087','PM-PB-1088','PM-PB-1089','PM-PB-1090','PM-AU-1002','PM-AU-1006',]
tmp = clin[clin['sample'].isin(samplelist)]




# %%
clin = pd.read_csv('/home/jiye/jiye/nanopore/FINALDATA/137_clinicaldata_new.txt', sep='\t')
clintmp = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/clinical_data_cleaned.csv', sep=',')
# %%
clintmp = pd.DataFrame(clintmp.loc[:,['Tstage','Nstage','Mstage','sample_normal']])
clintmp = clintmp.replace('PM-PM-1072-N', 'PM-PU-1072-N')
clintmp['sample'] = clintmp['sample_normal'].str.rsplit("-",1).str[0]
clintmp = clintmp.drop(['sample_normal'],axis=1)
merged = clin.merge(clintmp, on='sample')

# %%
merged.to_csv('/home/jiye/jiye/nanopore/FINALDATA/137_clinicaldata_final.txt', sep='\t', index=False)
# %%
