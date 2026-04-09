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
import pandas as pd
from scipy.stats import wilcoxon
sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
responder = sampleinfo.loc[(sampleinfo['response']==1),'sample_full'].to_list()
nonresponder = sampleinfo.loc[(sampleinfo['response']==0),'sample_full'].to_list()

TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0)
TU['GeneName'] = TU.index.str.split("-",2).str[1]
df = TU
df_nogene = df.iloc[:,:-1]

################## AR vs. IR ###########################3
df_nogene = df_nogene[responder]
########################################################

# Separate the expression data by pre- and post-treatment
pre_columns = df_nogene.columns[1::2]  
post_columns = df_nogene.columns[0::2]  

# Create a list to store the results
results = []
for gene, group in df.groupby('GeneName'):
    # Flatten to compare all transcripts' expression for pre- vs. post-treatment
    pre_values = group[pre_columns].values.flatten()
    post_values = group[post_columns].values.flatten()
    
    # Calculate the differences
    differences = post_values - pre_values
    
    # Check if there are any non-zero differences
    if np.any(differences != 0):
        # Perform the Wilcoxon signed-rank test
        stat, p = wilcoxon(differences)
        results.append((gene, stat, p))
    else:
        # If all differences are zero, append NaN or a placeholder value
        results.append((gene, np.nan, np.nan))

# Convert to DataFrame
results_df = pd.DataFrame(results, columns=['GeneName', 'Statistic', 'P-value'])

# Filtering or handling NaN values as needed
# For example, you might choose to consider these as non-significant directly,
# Or handle them in a specific way based on your analysis requirements

# Adjusting for multiple comparisons as previously described
from statsmodels.stats.multitest import multipletests
_, p_adjusted, _, _ = multipletests(results_df['P-value'].fillna(1), alpha=0.05, method='fdr_bh')
results_df['Adjusted P-value'] = p_adjusted

#%%
# Filter based on adjusted p-value, handling NaN values appropriately
significant_genes = results_df[results_df['P-value'] < 0.05]

print(significant_genes)

# %%
gDUTlist = significant_genes['GeneName']
with open('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUTgene/IR_DUTgenelist.txt', "w") as file:
    for item in gDUTlist:
        file.write("%s\n" % item)








# %%
#######^^ intersection with old / new !? ####

with open('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUTgene/AR_DUTgenelist.txt') as file:
    newgDUTlist = [line.rstrip() for line in file]

ARdut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t')
IRdut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/nonresponder_stable_DUT_Wilcoxon.txt', sep='\t')

major = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = major[major['type']=='major']

gDUTlist = list(set(ARdut.loc[(ARdut['p_value']<0.05) & (ARdut['log2FC']>1.5), 'Gene Symbol']))
DUTlist = list(ARdut.loc[(ARdut['p_value']<0.05) & (ARdut['log2FC']>1.5), 'gene_ENST'])
enr = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202401_analysis/DUT/responder_stable_Wilcoxon_GOenrichment_FC_15.txt', sep='\t')
enr = enr[(enr["Term"].str.contains("repair", case=False)) | (enr["Term"].str.contains("DNA Damage", case=False))]
enr = enr[enr["Adjusted P-value"]<=0.01]
genelist = [gene for sublist in enr['Genes'].str.split(';') for gene in sublist]
genelist = list(set(genelist))
enrmajorlist = majorlist[majorlist['genename'].isin(genelist)]['gene_ENST']
enrmajordutlist = set(enrmajorlist).intersection(set(DUTlist))

# %%
AR_intersect = list(set(newgDUTlist).intersection(set(gDUTlist)))
with open('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUTgene/AR_intersect_oldnew.txt', "w") as file:
    for item in AR_intersect:
        file.write("%s\n" % item)
# %%
