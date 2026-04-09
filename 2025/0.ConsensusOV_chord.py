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
from yellowbrick.cluster import KElbowVisualizer
import plotly.express as px
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
AR = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ConsensusOV_AR.txt', sep=',')
IR = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ConsensusOV_IR.txt', sep=',')
AR.columns = ['pre','post']
IR.columns = ['pre','post']

# %%
df = AR
transition_matrix = df.groupby(["pre", "post"]).size().unstack(fill_value=0)
transition_matrix.index = ['DIF','IMR','MES','PRO']
transition_matrix.columns = ['DIF','IMR','MES','PRO']

data = [
    (row_index, col_index, value)
    for row_index, row in transition_matrix.iterrows()
    for col_index, value in row.items()
    if value > 0  # Include only non-zero values
]

import holoviews as hv
from holoviews import opts
hv.extension('bokeh')
hv.output(size=120)

# Step 2: Convert to DataFrame for links (edges)
links = pd.DataFrame(data, columns=['source', 'target', 'value'])

# Step 3: Create unique node mapping and ensure consistent types
unique_nodes = sorted(set(links['source']).union(set(links['target'])))
node_mapping = {name: idx for idx, name in enumerate(unique_nodes)}

# Map source and target to numeric indices
links['source'] = links['source'].map(node_mapping)
links['target'] = links['target'].map(node_mapping)

# Step 4: Create nodes DataFrame with consistent types
nodes = pd.DataFrame({'index': range(len(unique_nodes)), 'name': unique_nodes})

# Step 5: Create the Chord diagram
chord = hv.Chord((links, hv.Dataset(nodes, 'index')))
chord = chord.select(value=(1, None))  # Filter edges with value >= 1
chord.opts(
    opts.Chord(
        cmap='Category20',                  # Use a categorical colormap
        edge_cmap='Category20',             # Match edge color to the colormap
        edge_color='source',                # Set edge color based on the source node
        labels='name',                      # Use node names as labels
        node_color='index',                 # Set node colors based on the index
    )
)

chord


# %%
