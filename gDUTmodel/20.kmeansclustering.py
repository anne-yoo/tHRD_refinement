#%%
#! this code is.....
"""
<K-meansclutering>
Step 1: Choose the number K of clusters with KElbowVisualizer
Step 2: Select random K points as cluster centers
Step 3: Assign each data point to the nearest centroid
Step 4: Compute and place the new centroid of each cluster
Step 5: Repeat step 4 until no observations change cluster

(1) gene-level TPM data
(2) transcript-level exp data 
"""


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

sns.set(font = 'Arial')
sns.set_style("whitegrid")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# %%
geneexp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM_symbol.txt',sep='\t', index_col=0)
genedf = geneexp.iloc[:,:-1]
filtered_gene = genedf.loc[(genedf > 0).sum(axis=1) >= genedf.shape[1]*0.6]
filtered_gene = filtered_gene.transpose()
genet = genedf.transpose()
#scaler = MinMaxScaler()
#data_scale = scaler.fit_transform(geneexp.iloc[:,:-1])

# %%
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,10))
visualizer.fit(filtered_gene)
visualizer.finalize()
#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202309_analysis/gene_kmeans_elbow.pdf", bbox_inches="tight")


#%%
transexp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_transcript_exp.txt',sep='\t', index_col=0)
filtered_trans = transexp.loc[(transexp > 0).sum(axis=1) >= transexp.shape[1]*0.6]
filtered_trans = filtered_trans.transpose()
transt = transexp.transpose()

#scaler2 = MinMaxScaler()
#data_scale2 = scaler2.fit_transform(transexp)
# %%
model = KMeans()
visualizer2 = KElbowVisualizer(model, k=(1,10))
visualizer2.fit(filtered_trans)
visualizer2.finalize()
#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202309_analysis/transcript_kmeans_elbow.pdf", bbox_inches="tight")

# %%
###^ gene data kmeans
k = 3
model = KMeans(n_clusters = k, random_state = 10)
model.fit(genet)
genet['cluster'] = model.fit_predict(genet)


# %%
###^ transcript / gene filtering!!!!!!!!!
geneexp_n = geneexp.iloc[:,:-1]
geneexp_n['Zero_Count'] = ((geneexp_n != 0).sum(axis=1))*1.25

transexp_n = transexp
transexp_n['Zero_Count'] = ((transexp_n != 0).sum(axis=1))*1.25


plt.figure(figsize=(10,6))

sns.set(font = 'Arial')
sns.set_style("whitegrid")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#sns.histplot(geneexp_n, x="Zero_Count", binwidth=4, color='darkolivegreen')
sns.histplot(geneexp_n, x="Zero_Count", kde=False, cumulative=True, binwidth=4, color='darkolivegreen', stat='percent')

plt.xlabel('% of non-zeros')
plt.ylabel('% of genes')


plt.axvline(x=60, color='red', linestyle='--', linewidth=1.5)

plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202309_analysis/cum_gene_countofzeros_80.pdf", bbox_inches="tight")
plt.show()



# %%
###^ transcript data kmeans
k = 3
model2 = KMeans(n_clusters = k, random_state = 10)
model2.fit(transt)
transt['cluster'] = model2.fit_predict(transt)
# %%

trans = pd.DataFrame(filtered_trans.index)
trans.columns = ['transcript_id']
trans['gene'] = trans['transcript_id'].str.split("-",1).str[1]
trans['trans_cluster'] = list(filtered_trans.iloc[:,-1])

gene_tmp = geneexp[['Gene Symbol']]
gene_tmp['gene_id']= gene_tmp.index
gene = pd.DataFrame(filtered_gene.index)
gene.columns = ['gene_id']
gene['gene_cluster'] = list(filtered_gene.iloc[:,-1])
gene = pd.merge(gene,gene_tmp, how='inner')

gene.columns = ['gene_id','gene_cluster', 'gene']

#%%
merged = pd.merge(gene,trans,left_on='gene', right_on='gene')
contingency_table = pd.crosstab(merged['gene_cluster'], merged['trans_cluster'])

# %%
plt.figure(figsize=(10,6))

sns.set(font = 'Arial')
sns.set_style("whitegrid")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sns.displot(genet, x="cluster", color='darkolivegreen',height=6, aspect=1.5)
# %%
