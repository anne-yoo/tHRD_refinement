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
event = ['SE','MX','A5','A3','RI','AF','AL']
df = pd.DataFrame()
for e in event:
    input = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/SUPPA2/dpsi_stringtie_'+e+'_variable_10.ioe.dpsi', sep='\t', index_col=0)
    input.columns = ['dPSI','pval']
    input['event'] = e
    df = pd.concat([df,input])
# %%
# Count total events by type
event_counts = df["event"].value_counts()

# Count significant events by type (p-val < 0.05)
sig_event_counts = df[df["pval"] < 0.05]["event"].value_counts()

# Combine into a DataFrame for plotting
event_summary = pd.DataFrame({
    "Event Type": event_counts.index,
    "Total Events": event_counts.values,
    "Significant Events": sig_event_counts.reindex(event_counts.index, fill_value=0).values
})

# Add counts to event names for all events and significant events
event_summary["Labeled Event Type (All)"] = event_summary.apply(
    lambda row: f"{row['Event Type']} ({row['Total Events']})", axis=1
)
event_summary["Labeled Event Type (Sig)"] = event_summary.apply(
    lambda row: f"{row['Event Type']} ({row['Significant Events']})", axis=1
)

# Define a Seaborn color palette
palette = sns.color_palette("pastel", len(event_summary))

# Create subplots for the pie charts
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Increase label font size
label_fontsize = 14

# Pie chart for all events
axes[0].pie(event_summary["Total Events"], labels=event_summary["Labeled Event Type (All)"], autopct='%1.1f%%',
            startangle=140, colors=palette, textprops={'fontsize': label_fontsize})
axes[0].set_title("All Detected Events", fontsize=17)

# Pie chart for significant events
axes[1].pie(event_summary["Significant Events"], labels=event_summary["Labeled Event Type (Sig)"], autopct='%1.1f%%',
            startangle=140, colors=palette, textprops={'fontsize': label_fontsize})
axes[1].set_title("Significant Events (p < 0.05)", fontsize=17)

# Adjust layout and display the chart
plt.tight_layout()
#plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/SUPPA2/AS_piechart.pdf', bbox_inches='tight', dpi=300)
plt.show()

# %%
#df.to_csv('/home/jiye/jiye/nanopore/202411_analysis/SUPPA2/AS_summary.txt', sep='\t', index=True)
# %%

newdf = df.dropna()
# %%
######^^ isoform length R. event PSI ############

dpsi = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/SUPPA2/AS_event_genename.txt', sep='\t', index_col=0)
tpm = pd.read_csv('/home/jiye/jiye/nanopore/FINALDATA/matched_transcript_TPM.txt', sep='\t', index_col=0)
length = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/SUPPA2/transcript_length.txt', sep='\t', index_col=0)
normal_cols = tpm.iloc[:, ::2]  # Select even-indexed columns (Normal)
tumor_cols = tpm.iloc[:, 1::2]  # Select odd-indexed columns (Tumor)

# Step 2: Calculate mean TPM for normal and tumor samples
normal_means = normal_cols.mean(axis=1)
tumor_means = tumor_cols.mean(axis=1)

# Step 3: Calculate delta TPM (Tumor - Normal)
delta_tpm = pd.DataFrame(tumor_means - normal_means)
#%%
delta_tpm.columns = ['deltaTPM']
delta_tpm['gene'] = delta_tpm.index.str.split('-',0).str[1]  # Extract gene names

# Step 5: Select representative transcript for each gene (max |delta_tpm|)
representative_transcripts = (
    delta_tpm.loc[delta_tpm.groupby('gene')['deltaTPM'].apply(lambda x: x.abs().idxmax())]
)

# %%
representative_transcripts.index = representative_transcripts.index.str.split('-',0).str[0]
merged = representative_transcripts.merge(length, left_index=True, right_index=True)
# %%
dpsi_merged = pd.merge(dpsi, merged, left_on='genename', right_on='gene', how='left')
dpsi_merged = dpsi_merged[dpsi_merged['pval']<0.05]
dpsi_merged = dpsi_merged[(dpsi_merged['event']=='SE')] #|(dpsi_merged['event']=='AF')
dpsi_merged = dpsi_merged.dropna()

# Step 2: Plot dPSI vs. length
from scipy.stats import pearsonr

# Step 2: Calculate correlation (Pearson)
x = dpsi_merged['dPSI']
y = dpsi_merged['length']
correlation, pval = pearsonr(x, y)

# Step 3: Plot dPSI vs. length with correlation line
plt.figure(figsize=(8, 6))

# Scatter plot
plt.scatter(x, y, alpha=0.7, edgecolors='k', label='AS Events')
plt.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)

# Add correlation line
m, b = np.polyfit(x, y, 1)  # Linear regression (slope, intercept)
plt.plot(x, m * x + b, color='red', linestyle='-', label='Correlation Line')

# Add text for correlation and p-value
plt.text(0.05, max(y) * 0.9, f'R = {correlation:.2f}\np = {pval:.2e}', fontsize=12, color='red')

# Title and labels
plt.title("dPSI vs Transcript Length (Alternative Splicing Events)")
plt.xlabel("dPSI (Tumor vs Normal)")
plt.ylabel("Transcript Length (bp)")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# %%
###^^ shortest transcript #####
trans = pd.DataFrame(tpm.index)
trans['trans'] = trans['transcript_id'].str.split('-',0).str[0]
trans['gene'] = trans['transcript_id'].str.split('-',0).str[1]
trans = trans.set_index('trans') 
merged = trans.merge(length, left_index=True, right_index=True)
shortest_transcripts = merged.loc[merged.groupby('gene')['length'].idxmin()]

dpsi_merged = pd.merge(dpsi, shortest_transcripts, left_on='genename', right_on='gene', how='left')
dpsi_merged = dpsi_merged[dpsi_merged['pval']<0.05]
dpsi_merged = dpsi_merged[(dpsi_merged['event']=='AF')] #|(dpsi_merged['event']=='AF') #(dpsi_merged['event']=='SE')|
dpsi_merged = dpsi_merged.dropna()
#dpsi_merged = dpsi_merged[dpsi_merged['length']<30000]

from scipy.stats import pearsonr

# Step 2: Calculate correlation (Pearson)
x = dpsi_merged['dPSI']
y = dpsi_merged['length']
correlation, pval = pearsonr(x, y)

# Step 3: Plot dPSI vs. length with correlation line
plt.figure(figsize=(6, 6))

# Scatter plot
plt.scatter(x, y, alpha=0.7, edgecolors='k', label='Sig. AS Events')


# Add correlation line
m, b = np.polyfit(x, y, 1)  # Linear regression (slope, intercept)
plt.plot(x, m * x + b, color='red', linestyle='-', ) #label='Correlation Line'

# Add text for correlation and p-value
plt.text(-0.3, 55000, f'R = {correlation:.2f}\np = {pval:.2e}', fontsize=12, color='red')
sns.kdeplot(x=dpsi_merged['dPSI'], y=dpsi_merged['length'], levels=5, color='blue', alpha=0.3)

# Title and labels
#plt.title("dPSI vs Transcript Length (Alternative Splicing Events)")
plt.xlabel("dPSI")
plt.ylabel("Transcript Length (bp)")
plt.grid(alpha=0.3)
plt.ylim([-0.1,80000])
plt.legend()
plt.show()

# %%
###^^^ tumor-only transcripts #####
trans = pd.read_csv('/home/jiye/jiye/nanopore/FINALDATA/matched_transcript_TPM.txt', sep='\t', index_col=0)
trans = trans[trans.apply(lambda x: (x != 0).sum(), axis=1) >= trans.shape[1]*0.2]
tumor = trans.iloc[:,1::2]
normal = trans.iloc[:,0::2]
tumor = tumor[tumor.apply(lambda x: (x != 0).sum(), axis=1) >= tumor.shape[1]*0.2]
normal = normal[normal.apply(lambda x: (x != 0).sum(), axis=1) >= normal.shape[1]*0.2]

tumorlist = tumor.index
normallist = normal.index

results = set(tumorlist)-set(normallist)
# %%
tlist = list(results)
trans = pd.DataFrame(tlist)
trans.columns = ['transcript_id']
trans['trans'] = trans['transcript_id'].str.split('-',0).str[0]
trans['gene'] = trans['transcript_id'].str.split('-',0).str[1]
trans = trans.set_index('trans') 
#merged_tumor = trans.merge(length, left_index=True, right_index=True)
merged_tumor = trans
merged_tumor['type'] = 'tumor-specific'
#shortest_transcripts = merged.loc[merged.groupby('gene')['length'].idxmin()]



nlist = list(set(normallist)-set(tumorlist))
n = pd.DataFrame(nlist)
n.columns = ['transcript_id']
n['trans'] = n['transcript_id'].str.split('-',0).str[0]
n['gene'] = n['transcript_id'].str.split('-',0).str[1]
n = n.set_index('trans') 
#merged_normal = n.merge(length, left_index=True, right_index=True)
merged_normal = n
merged_normal['type'] = 'normal-specific'

merged = pd.concat([merged_tumor,merged_normal])
#%%
dpsi_merged = pd.merge(dpsi, merged, left_on='genename', right_on='gene', how='left')
dpsi_merged = dpsi_merged[dpsi_merged['pval']<0.05]
dpsi_merged = dpsi_merged[(dpsi_merged['event']=='AF')] #|(dpsi_merged['event']=='AF') #(dpsi_merged['event']=='SE')|
dpsi_merged = dpsi_merged.dropna()
#dpsi_merged = dpsi_merged[dpsi_merged['length']<300000]
#%%
# Scatter plot with different colors for 'type'
df = dpsi_merged

# Create a boxplot for length by type
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 12,    # x축 틱 라벨 글꼴 크기
'ytick.labelsize': 12,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 13,
'legend.title_fontsize': 13, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})
plt.figure(figsize=(4,5))
sns.set_style("whitegrid")
# Customize the plot
plt.title("AF", fontsize=13)
#plt.xlabel("Type", fontsize=12)
plt.ylabel("dPSI (Tumor vs Normal)", fontsize=12)
plt.grid(axis='y', alpha=0.3)


ax = sns.boxplot(x='type', y='dPSI', data=df, palette='Set2', showfliers=True, order=["tumor-specific", "normal-specific"])
sns.stripplot(x='type', y='dPSI', data=df, color='grey', alpha=0.7, jitter=True, s=4, order=["tumor-specific", "normal-specific"])

sns.despine()

from statannot import add_stat_annotation
add_stat_annotation(ax, data=df, x='type', y='dPSI',
                    box_pairs=[("tumor-specific", "normal-specific")], 
                    #comparisons_correction=None,
                    test='Mann-Whitney',  text_format='star', loc='inside', fontsize=15)
#plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/SUPPA2/SE_dPSI_boxplot.pdf', bbox_inches='tight', dpi=300)
plt.show()


# %%

up_df = df[df['dPSI']>0]
down_df = df[df['dPSI']<0]
# %%
