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
####** select exon file ####
count19 = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/featurecounts/exonreadcount/BRIP1_exon19.txt', sep='\t', header=None)
count20 = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/featurecounts/exonreadcount/BRIP1_exon20.txt', sep='\t', header=None)

count19.columns = ['sample','count']
count20.columns = ['sample','count']

countsum = count19['count'] + count20['count']

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')

count = count19
count['treatment'] = count['sample'].apply(lambda x: 'post' if '-atD' in x else 'pre')

####** sum of exon19+20 ####
count['count'] = countsum
####*#######################

###** only AR/IR samples ###
ARlist = list(sampleinfo.loc[sampleinfo['response']==1,'sample_full'])
IRlist = list(sampleinfo.loc[sampleinfo['response']==0,'sample_full'])

count = count[count['sample'].isin(IRlist)]
#####*###################

count['sampleid'] = count['sample'].str[:-4]

# %%
plt.figure(figsize=(5,8))
sns.set_theme(style='ticks', palette='pastel', font_scale=1)

ax = sns.boxplot(y='count', x='treatment', data=count, order=['pre','post'],
            showmeans=False, 
            #meanprops = {"marker":"o", "color":"blue","markeredgecolor": "blue", "markersize":"10"}
            )

from statannot import add_stat_annotation
add_stat_annotation(ax, data=count, x='treatment', y='count',
                    box_pairs=[("pre", "post")],
                    test='Wilcoxon',  text_format='simple', loc='inside')

for p in count['sampleid'].unique():
    subset = count[count['sampleid']==p]
    x = [list(subset['treatment'])[1], list(subset['treatment'])[0]]
    y = [list(subset['count'])[1], list(subset['count'])[0]]
    plt.plot(x,y, marker="o", markersize=4, color='grey', linestyle="--", linewidth=0.7)
plt.title('BRIP1 exon19+exon20')
# %%
