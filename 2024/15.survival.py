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
import textwrap


sns.set(font = 'Arial')
sns.set_style("whitegrid")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# %%
###^ PFS comparison ####
#####* (1) exon 19 / 20 up down #######
####** select exon file ####
count19 = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/featurecounts/exonreadcount/BRIP1_exon19.txt', sep='\t', header=None)
count20 = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/featurecounts/exonreadcount/BRIP1_exon20.txt', sep='\t', header=None)

count19.columns = ['sample_full','count']
count20.columns = ['sample_full','count']

countsum = count19['count'] + count20['count']

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')

##################
count = count19
#count['count'] = countsum
##################

count['treatment'] = count['sample_full'].apply(lambda x: 'post' if '-atD' in x else 'pre')
count['sample_id'] = count['sample_full'].str[:-4]
count['interval'] = sampleinfo['interval']
count['status'] = 'True'
count['status'] = count['status'].astype(bool)
# Group by 'sample_id' and pivot the 'treatment' values to create separate columns for 'pre' and 'post'
pivot_df = count.pivot_table(index=['sample_id', 'interval', 'status'], columns='treatment', values='count', aggfunc='first')

# Calculate the 'deltacount' as the difference between 'post' and 'pre'
pivot_df['deltacount'] = pivot_df['post'] - pivot_df['pre']

# Reset the index to turn multi-index into columns
result_df = pivot_df.reset_index()


######### only AR/IR samples ############
ARlist = list(sampleinfo.loc[sampleinfo['response']==1,'sample_id'])
IRlist = list(sampleinfo.loc[sampleinfo['response']==0,'sample_id'])

result_df = result_df[result_df['sample_id'].isin(ARlist)]
#########################################

mean_count = result_df['deltacount'].mean()
median_count = result_df['deltacount'].median()

# Create 'mean_cluster' column
result_df['mean_cluster'] = result_df['deltacount'].apply(lambda x: 'high' if x > mean_count else 'low')

# Create 'median_cluster' column
result_df['median_cluster'] = result_df['deltacount'].apply(lambda x: 'high' if x > median_count else 'low')

new_sample = result_df



T = new_sample["interval"]
E = new_sample["status"]
plt.hist(T, bins = 50)
plt.show()

from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

kmf = KaplanMeierFitter()

# Divide the dataset into 'up' and 'down' groups based on the 'cluster' column
###########################
cluster = 'mean_cluster'
###########################

new_sample['status'] = new_sample['status'].astype(bool)

# Convert 'interval' to numeric, if it's not already
new_sample['interval'] = pd.to_numeric(new_sample['interval'], errors='coerce')

# Now filter the groups based on the 'cluster' column being 'high'
m = (new_sample[cluster] == 'high')
T_up = new_sample['interval'][m]
E_up = new_sample['status'][m]
T_down = new_sample['interval'][~m]
E_down = new_sample['status'][~m]


# Fit the KM model for 'up' group
kmf.fit(durations = T_up, event_observed = E_up, label = "high")
ax = plt.subplot(111)
sns.set_style('whitegrid')
kmf.plot_survival_function(ax=ax, color="#5465FF")

# Fit the KM model for 'down' group
kmf.fit(durations = T_down, event_observed = E_down, label = "low")
kmf.plot_survival_function(ax=ax, at_risk_counts=False, color="#BC67C6")

# Perform the log-rank test
results = logrank_test(T_up, T_down, event_observed_A=E_up, event_observed_B=E_down)
p_value = results.p_value

# Annotate the plot with the p-value
plt.annotate(f'p-value = {p_value:.4f}', xy=(0.1, 0.2), xycoords='axes fraction')

plt.xlabel('treatment interval')
plt.show()


#%%
###** (2) exon 19+20 
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
#%%
kmf = KaplanMeierFitter()
kmf.fit(durations = T, event_observed = E)
#kmf.plot_survival_function()

ax = plt.subplot(111)
sns.set_style('whitegrid')
m = (new_sample["cluster"] == 'up')
kmf.fit(durations = T[m], event_observed = E[m], label = "up")
kmf.plot_survival_function(ax = ax, color="#5465FF")

kmf.fit(T[~m], event_observed = E[~m], label = "down")
kmf.plot_survival_function(ax = ax, at_risk_counts = False, color="#BC67C6")
plt.xlabel('treatment interval')
# %%
