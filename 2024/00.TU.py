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


sns.set(font = 'Arial')
sns.set_style("whitegrid")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# %%
TU = TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0)
TU['Gene Symbol'] = TU.index.str.split("-",2).str[1]
TU['TranscriptID'] = TU.index.str.split("-",2).str[0]

trans_gene = TU[['TranscriptID','Gene Symbol']]
trans_gene = trans_gene.set_index('TranscriptID')

newtpm = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/hg19/tpm_matrix.csv', index_col=0)

# %%
merged = pd.merge(newtpm,trans_gene, how='inner',left_index=True, right_index=True)
# %%
df_usage_percentage = pd.DataFrame()

# Calculate the usage percentage for each sample
for sample in merged.columns[:-1]:  # Exclude the last column ('Gene')
    # Group by gene and calculate the sum of TPMs for each gene
    gene_tpm_sum = merged.groupby('Gene Symbol')[sample].transform('sum')
    # Calculate the usage percentage
    df_usage_percentage[sample] = merged[sample] / gene_tpm_sum

# Set the Gene column in the usage percentage DataFrame
df_usage_percentage['Gene Symbol'] = merged['Gene Symbol']

#%%
# Optionally, save the usage percentage DataFrame to a new TSV file
col = TU.columns
col = col[:-1]
df_usage_percentage = df_usage_percentage[col]

# %%
