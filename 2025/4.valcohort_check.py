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
import re
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.stats.multitest import multipletests
from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42  
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'Helvetica'

plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 13,    # x축 틱 라벨 글꼴 크기

'ytick.labelsize': 13,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 13,
'legend.title_fontsize': 13, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})
sns.set_style("ticks")

# %%
val = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/156_validation_TU.txt', sep='\t', index_col=0)
valinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_validation_clinicalinfo_new.txt', sep='\t', index_col=0)
typelist1 = list(valinfo['type']) #CR AR IR
typelist2 = ['R' if x=='CR' else x for x in typelist1]
typelist2 = ['R' if x=='AR' else x for x in typelist2]
val = val.iloc[:-2,:]
# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# Load the data (replace with actual file path)
df = val.T  # Transpose to sample × transcript
df = df.dropna(axis='columns')

# Sample labels (replace with actual lists)
group_labels = typelist1
binary_labels = typelist2

# Standardize data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

selector = VarianceThreshold(threshold=0.1)  # Remove transcripts with low variance
df_scaled = selector.fit_transform(df_scaled)

# PCA transformation
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)

# UMAP transformation
umap_result = umap.UMAP(n_components=2, random_state=42).fit_transform(df_scaled)

#%%
# Create DataFrames for plotting
pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
pca_df["Group"] = group_labels
umap_df = pd.DataFrame(umap_result, columns=["UMAP1", "UMAP2"])
umap_df["Group"] = group_labels

# Plot PCA
plt.figure(figsize=(7, 5))
sns.scatterplot(x="PC1", y="PC2", hue="Group", data=pca_df, palette={"AR": "#FFCC29", "IR": "#81B214","CR":"#409EDD"}, alpha=0.8)
plt.title("PCA Projection")
plt.legend(title="Group")
plt.show()

# Plot UMAP
plt.figure(figsize=(7, 5))
sns.scatterplot(x="UMAP1", y="UMAP2", hue="Group", data=umap_df, palette={"AR": "#FFCC29", "IR": "#81B214","CR":"#409EDD"}, alpha=0.8)
plt.title("UMAP Projection")
plt.legend(title="Group")
plt.show()
#"R": "#A482BB", "IR": "#81B214"

# %%
#%%
# Create DataFrames for plotting
pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
pca_df["Group"] = binary_labels
umap_df = pd.DataFrame(umap_result, columns=["UMAP1", "UMAP2"])
umap_df["Group"] = binary_labels

plt.figure(figsize=(7, 5))
sns.scatterplot(x="PC1", y="PC2", hue="Group", data=pca_df, palette={"R": "#A482BB", "IR": "#81B214"}, alpha=0.8)
plt.title("PCA Projection")
plt.legend(title="Group")
plt.show()

# Plot UMAP
plt.figure(figsize=(7, 5))
sns.scatterplot(x="UMAP1", y="UMAP2", hue="Group", data=umap_df, palette={"R": "#A482BB", "IR": "#81B214"}, alpha=0.8)
plt.title("UMAP Projection")
plt.legend(title="Group")
plt.show()

# %%
