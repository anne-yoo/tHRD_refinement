###%%
#! this code is.....
## 20230525
"""
CCA: DEG data vs. DTU data

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
import matplotlib.cm as cm
from matplotlib.pyplot import gcf
from sklearn.cross_decomposition import CCA

# %%
deg = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/DESeq2/response/onlydeg_countdata.csv', index_col=0)
dtu = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/input/discovery/response/onlydtu_inputdata.csv', index_col=0)
# %%
col = dtu.columns
deg = deg[col]
# %%
deg_T = deg.T
dtu_T = dtu.T



# %%
from umap import UMAP
import plotly.express as px

dtusample = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/input/discovery/response/metadata.csv', index_col=0)

#%%

dtu_f = pd.merge(dtu_T, dtusample, left_index=True, right_index=True, how='inner')

umap_2d = UMAP(n_components=2, init='random', random_state=0)
proj_2d = umap_2d.fit_transform(dtu_f.iloc[:,:-1])
fig_2d = px.scatter(
    proj_2d, x=0, y=1,
    color=dtu_f.group, labels={'color': 'group'}
)
fig_2d.show()
fig_2d.write_image("/home/jiye/jiye/copycomparison/gDUTresearch/figures/dtu_UMAP.pdf")
# %%
deg_f = pd.merge(deg_T, dtusample, left_index=True, right_index=True, how='inner')

umap_2d = UMAP(n_components=2, init='random', random_state=0)
proj_2d = umap_2d.fit_transform(deg_f.iloc[:,:-1])
fig_2d = px.scatter(
    proj_2d, x=0, y=1,
    color=deg_f.group, labels={'color': 'group'}
)
fig_2d.show()
fig_2d.write_image("/home/jiye/jiye/copycomparison/gDUTresearch/figures/deg_UMAP.pdf")
# %%
from sklearn.cluster import KMeans

def find_best_clusters(df, maximum_K):
    
    clusters_centers = []
    k_values = []
    
    for k in range(1, maximum_K):
        
        kmeans_model = KMeans(n_clusters = k)
        kmeans_model.fit(df)
        
        clusters_centers.append(kmeans_model.inertia_)
        k_values.append(k)
        
    
    return clusters_centers, k_values

def generate_elbow_plot(clusters_centers, k_values):
    
    figure = plt.subplots(figsize = (12, 6))
    plt.plot(k_values, clusters_centers, 'o-', color = 'orange')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Cluster Inertia")
    plt.title("Elbow Plot of KMeans")
    plt.show()
    
clusters_centers, k_values = find_best_clusters(deg_f.iloc[:,:-1], 12)

generate_elbow_plot(clusters_centers, k_values)
# %%
