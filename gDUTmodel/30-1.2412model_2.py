#%%
####** step #####
"""
(1) use only AR DUT
(2) pre TU values of AR DUTs: corr between class vs. TU
-> use features with negative correlation
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
from scipy import stats
from statsmodels.stats.multitest import multipletests
import gseapy as gp


sns.set(font = 'Arial')
sns.set_style("whitegrid")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# %%
##^ step 1 : use HR or +a GO terms for DUT selection from resistance cohort ####
major = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = list(major[major['type']=='major']['gene_ENST'])
ratio = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/majorminor/major2minorratio.txt', sep='\t', index_col=0)
TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0)

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
AR_samples = list(set(sampleinfo.loc[(sampleinfo['response']==1),'sample_id']))
IR_samples = list(set(sampleinfo.loc[(sampleinfo['response']==0),'sample_id']))

AR_dut_df = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t')

#%%
AR_dut = set(AR_dut_df[(AR_dut_df['p_value']<0.01) & (np.abs(AR_dut_df['log2FC'])>1.5)]['gene_ENST'])
dutlist = list(AR_dut)

AR_dut_gene = set(AR_dut_df[(AR_dut_df['p_value']<0.01) & (np.abs(AR_dut_df['log2FC'])>1.5)]['Gene Symbol'])
gDUTlist = list(AR_dut_gene)

#%%

enr = gp.enrichr(gene_list=gDUTlist, # or "./tests/data/gene_list.txt",
                    gene_sets=['GO_Biological_Process_2021','Reactome_2022'], # 'Reactome_2022', 'GO_Biological_Process_2018'
                    organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                    outdir=None, # don't write to disk
    )

enrresult = enr.results.sort_values(by=['Adjusted P-value']) 

#%%
file = enrresult
file = file.sort_values('Adjusted P-value')
file['Term'] = file['Term'].str.rsplit(" ",1).str[0]
file = file[file['Adjusted P-value']<0.1]
file['Adjusted P-value'] = -np.log10(file['Adjusted P-value'])
#%%
# file = file[(file["Term"].str.contains("homologous recombination", case=False)) | (file["Term"].str.contains("DNA Damage response", case=False)) | (file["Term"].str.contains("cell cycle", case=False)) | (file["Term"].str.contains("Signaling by WNT in cancer", case=False)) | (file["Term"].str.contains("PI3K/AKT Signaling in Cancer", case=False)) | (file["Term"].str.contains("double strand break", case=False))| (file["Term"].str.contains("DNA repair", case=False))]

file = file[(file["Term"].str.contains("homologous recombination", case=False))|(file["Term"].str.contains("S/G2", case=False))|(file["Term"].str.contains("response to DNA damage", case=False))]

HRlist = list(set([gene for sublist in file['Genes'].str.split(';') for gene in sublist]))

input = TU.loc[dutlist,:]
input['gene'] = input.index.str.split("-",1).str[1]
input = input[input['gene'].isin(HRlist)]
input = input.iloc[:,:-1]

#%%
#input = input.loc[input.index.isin(majorlist),:]

print(file['Term'])
#%%
data = input


# Initialize a dictionary to store all features
features_dict = {}

# Process each DUT (row in DataFrame)
for transcript, row in data.iterrows():
    for sample in set([col[:-4] for col in data.columns if '-atD' in col or '-bfD' in col]):  # Remove '-atD'/'-bfD'
        pre_col = f"{sample}-bfD"
        post_col = f"{sample}-atD"
        
        if pre_col in data.columns and post_col in data.columns:
            pre_TU = row[pre_col]
            post_TU = row[post_col]
            delta_TU = post_TU - pre_TU
            interaction = delta_TU * pre_TU
            
            # Store features with transcript-feature as index and sample as column
            features_dict[(f"{transcript}-pre", sample)] = pre_TU
            features_dict[(f"{transcript}-delta", sample)] = delta_TU
            features_dict[(f"{transcript}-int", sample)] = interaction

# Convert the dictionary into a DataFrame
features_df = pd.DataFrame.from_dict(features_dict, orient='index', columns=["Value"])
features_df.reset_index(inplace=True)

# Split the multi-index (Feature, Sample)
features_df[['Feature', 'Sample']] = pd.DataFrame(features_df['index'].tolist(), index=features_df.index)
features_df.drop(columns='index', inplace=True)

# Pivot the DataFrame so that samples are columns and features are rows
final_df = features_df.pivot(index='Feature', columns='Sample', values='Value')

# Filter for AR and IR samples only
filtered_sample_ids = AR_samples + IR_samples
filtered_df = final_df[filtered_sample_ids]

#* delta vs. pre#
pre_only_features = [f for f in filtered_df.index if '-pre' in f]
filtered_df = filtered_df.loc[pre_only_features]
###*########

# Create the response variable (y)
response_labels = {sample: 1 for sample in AR_samples}  # Label AR as 1
response_labels.update({sample: 0 for sample in IR_samples})  # Label IR as 0

# Align the response labels with the columns in filtered_df
y = np.array([response_labels[sample] for sample in filtered_df.columns])

####^^ boxplot check 
plotdf= pd.DataFrame({'sample':filtered_df.columns, 'mean':filtered_df.mean(axis=0),'group':y})
plotdf['group'] = plotdf['group'].map({1: 'AR', 0:'IR'})

plt.figure(figsize=(4,6))
#sns.set_style("whitegrid")
sns.set_theme(style='ticks',palette='pastel')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 12,    # x축 틱 라벨 글꼴 크기
'ytick.labelsize': 12,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 11,
'legend.title_fontsize': 11, # 범례 글꼴 크기
'figure.titlesize': 14    # figure 제목 글꼴 크기
})


ax = sns.boxplot(y='mean', x='group', data=plotdf, 
            showfliers=False, order=['IR','AR'], 
            #color='#FFD92E'
            palette='vlag'
            #meanprops = {"marker":"o", "color":"blue","markeredgecolor": "blue", "markersize":"10"}
            )
#ax.set_ylim([0,0.01])
ax.set_ylabel('pre TU')
#ax.set_xticklabels(['IR','AR'])
#plt.title(transname)
sns.despine()

sns.stripplot(y='mean', x='group', data=plotdf, 
            order=['IR','AR'], 
            color='#7D7C7C',  # Color of the points
            size=4,         # Size of the points
            jitter=True,    # Adds some jitter to avoid overlapping
            alpha=0.8,
            ax=ax)


from statannot import add_stat_annotation


add_stat_annotation(ax, data=plotdf, x='group', y='mean',
                    box_pairs=[('IR','AR')], 
                    order = ['IR','AR'], 
                    comparisons_correction=None,
                    test='Mann-Whitney', text_format='simple', loc='inside', fontsize=12
                    )

plt.show()


#%%
##^ majorsum
majorsum = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/majorminor/majorsum.txt', sep='\t', index_col=0)

ratio = majorsum
HRratio = ratio.loc[HRlist,:]
data = HRratio.dropna()


# Initialize a dictionary to store all features
features_dict = {}

# Process each DUT (row in DataFrame)
for transcript, row in data.iterrows():
    for sample in set([col[:-4] for col in data.columns if '-atD' in col or '-bfD' in col]):  # Remove '-atD'/'-bfD'
        pre_col = f"{sample}-bfD"
        post_col = f"{sample}-atD"
        
        if pre_col in data.columns and post_col in data.columns:
            pre_TU = row[pre_col]
            post_TU = row[post_col]
            delta_TU = post_TU - pre_TU
            interaction = delta_TU * pre_TU
            
            # Store features with transcript-feature as index and sample as column
            features_dict[(f"{transcript}-pre", sample)] = pre_TU
            features_dict[(f"{transcript}-delta", sample)] = delta_TU
            features_dict[(f"{transcript}-int", sample)] = interaction

# Convert the dictionary into a DataFrame
features_df = pd.DataFrame.from_dict(features_dict, orient='index', columns=["Value"])
features_df.reset_index(inplace=True)

# Split the multi-index (Feature, Sample)
features_df[['Feature', 'Sample']] = pd.DataFrame(features_df['index'].tolist(), index=features_df.index)
features_df.drop(columns='index', inplace=True)

# Pivot the DataFrame so that samples are columns and features are rows
final_df = features_df.pivot(index='Feature', columns='Sample', values='Value')

# Filter for AR and IR samples only
filtered_sample_ids = AR_samples + IR_samples
filtered_df = final_df[filtered_sample_ids]

# %%

# Create the response variable (y)
response_labels = {sample: 1 for sample in AR_samples}  # Label AR as 1
response_labels.update({sample: 0 for sample in IR_samples})  # Label IR as 0

# Align the response labels with the columns in filtered_df
y = np.array([response_labels[sample] for sample in filtered_df.columns])

# Extract features (X)
X = filtered_df.T  # Transpose so samples are rows and features are columns

######^^ change features: pre / int / delta ########
pre_only_features = [f for f in X.columns if '-delta' in f]
X = X[pre_only_features]

#%%
####*** correlation AR vs. IR ######

# Compute correlation
correlations = [np.corrcoef(X[feature], y)[0, 1] for feature in X.columns]
correlation_df = pd.DataFrame({'Feature': X.columns, 'Correlation': correlations})
selected_features = correlation_df.sort_values(by='Correlation', key=abs, ascending=False).head(20)
print("Top Features by Correlation:")
print(selected_features)

#%%
####^^ boxplot check 
plotdf= pd.DataFrame({'sample':filtered_df.columns, 'mean':filtered_df.mean(axis=0),'group':y})
plotdf['group'] = plotdf['group'].map({1: 'AR', 0:'IR'})

plt.figure(figsize=(4,6))
#sns.set_style("whitegrid")
sns.set_theme(style='ticks',palette='pastel')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 12,    # x축 틱 라벨 글꼴 크기
'ytick.labelsize': 12,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 11,
'legend.title_fontsize': 11, # 범례 글꼴 크기
'figure.titlesize': 14    # figure 제목 글꼴 크기
})


ax = sns.boxplot(y='mean', x='group', data=plotdf, 
            showfliers=False, order=['IR','AR'], 
            #color='#FFD92E'
            palette='vlag'
            #meanprops = {"marker":"o", "color":"blue","markeredgecolor": "blue", "markersize":"10"}
            )
#ax.set_ylim([0,0.01])
ax.set_ylabel('pre TU')
#ax.set_xticklabels(['IR','AR'])
#plt.title(transname)
sns.despine()

# sns.stripplot(y='mean', x='group', data=plotdf, 
#             order=['IR','AR'], 
#             color='#7D7C7C',  # Color of the points
#             size=4,         # Size of the points
#             jitter=True,    # Adds some jitter to avoid overlapping
#             alpha=0.8,
#             ax=ax)


from statannot import add_stat_annotation


add_stat_annotation(ax, data=plotdf, x='group', y='mean',
                    box_pairs=[('IR','AR')], 
                    order = ['IR','AR'], 
                    comparisons_correction=None,
                    test='Mann-Whitney', text_format='simple', loc='inside', fontsize=12
                    )

plt.show()
#%%





from sklearn.feature_selection import mutual_info_classif

# Compute mutual information
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mi_scores})
mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)

# Plot top features
plt.figure(figsize=(10, 6))
plt.barh(mi_df['Feature'].head(30), mi_df['Mutual Information'].head(30), color='orange')
plt.xlabel('Mutual Information')
plt.ylabel('Feature')
plt.title('Top Features by Mutual Information')
plt.gca().invert_yaxis()
plt.show()

# Select top features
selected_features = mi_df[mi_df['Mutual Information'] > 0.05]  # Define a threshold



#%%
#######** PCA ###############################
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# Subset the data for resistance cohort only
X_resistance = X[(y == 1) | (y == 0)]  # AR = 1, IR = 0
y_resistance = y[(y == 1) | (y == 0)]  # AR = 1, IR = 0

# Perform PCA
pca = PCA(n_components=5, random_state=42)  # Keep the first 5 PCs
X_pca = pca.fit_transform(X_resistance)

# Visualize the first 2 principal components
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[y_resistance == 0, 0], X_pca[y_resistance == 0, 1], label='IR', alpha=0.7, c='orange')
plt.scatter(X_pca[y_resistance == 1, 0], X_pca[y_resistance == 1, 1], label='AR', alpha=0.7, c='blue')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Resistance Cohort (AR vs. IR)')
plt.legend()
plt.show()

# Print explained variance ratios
print("Explained Variance Ratios:", pca.explained_variance_ratio_)

# Extract top features contributing to PC1
pc1_loadings = pd.DataFrame(
    {'Feature': X.columns, 'Loading': pca.components_[0]}
).sort_values(by='Loading', key=abs, ascending=False)

# Top 10 features contributing to PC1
top_features_pc1 = pc1_loadings.head(10)
print("Top Features Contributing to PC1:\n", top_features_pc1)


#%%
########** LDA ########################################
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import pandas as pd

# Apply LDA to the resistance cohort
lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X, y)

# Visualize the LDA projection
plt.figure(figsize=(8, 6))
plt.hist(X_lda[y == 0], bins=20, alpha=0.7, label='IR', color='orange')
plt.hist(X_lda[y == 1], bins=20, alpha=0.7, label='AR', color='blue')
plt.xlabel('LDA Projection')
plt.ylabel('Frequency')
plt.title('LDA Projection of Resistance Cohort (AR vs. IR)')
plt.legend()
plt.show()

# Get the LDA coefficients
lda_coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': lda.coef_[0]})
selected_features = lda_coefficients[lda_coefficients['Coefficient'].abs() > 20]
X_selected = X[selected_features['Feature']]

lda2 = LinearDiscriminantAnalysis()
X_lda2 = lda2.fit_transform(X_selected, y)

# Visualize the LDA projection
plt.figure(figsize=(8, 6))
plt.hist(X_lda2[y == 0], bins=20, alpha=0.7, label='IR', color='orange')
plt.hist(X_lda2[y == 1], bins=20, alpha=0.7, label='AR', color='blue')
plt.xlabel('LDA Projection')
plt.ylabel('Frequency')
plt.title('LDA Projection of Resistance Cohort (AR vs. IR)')
plt.legend()
plt.show()

# Get the LDA coefficients
lda_coefficients2 = pd.DataFrame({'Feature': X_selected.columns, 'Coefficient': lda2.coef_[0]})
selected_features_pos = lda_coefficients2[lda_coefficients2['Coefficient'] > 0]
selected_features_neg = lda_coefficients2[lda_coefficients2['Coefficient'] < 0]

#%%
selected_transcripts = list(set([feature.rsplit('-',1)[0] for feature in pre_only_features]))

mean_tu_data = []

for sample in AR_samples + IR_samples:
    pre_values = []
    delta_values = []
    
    for transcript in selected_transcripts:
        # Define column names
        pre_col = f"{sample}-bfD"
        post_col = f"{sample}-atD"
        
        # Check if both columns exist in the data
        if pre_col in data.columns and post_col in data.columns:
            # Retrieve TU values
            pre_value = data.loc[transcript, pre_col]
            post_value = data.loc[transcript, post_col]
            
            # Exclude NaN or invalid values
            if not np.isnan(pre_value) and not np.isnan(post_value):
                pre_values.append(pre_value)
                delta_values.append(post_value - pre_value)
    
    # Verify number of values matches expected transcripts
    if len(pre_values) != len(selected_transcripts):
        print(f"Warning: Transcript count mismatch for {sample}. Expected {len(selected_transcripts)}, got {len(pre_values)}.")
    
    # Calculate mean TU values (if valid values exist)
    mean_pre_tu = np.mean(pre_values) if pre_values else np.nan
    mean_delta_tu = np.mean(delta_values) if delta_values else np.nan
    
    group = "AR" if sample in AR_samples else "IR"
    
    mean_tu_data.append({"Sample": sample, "Group": group, "Mean Pre TU": mean_pre_tu, "Mean Delta TU": mean_delta_tu})

# Convert to DataFrame
mean_tu_df = pd.DataFrame(mean_tu_data)

# Plot Mean Pre TU
plt.figure(figsize=(4,6))
#sns.set_style("whitegrid")
sns.set_theme(style='ticks',palette='pastel')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 12,    # x축 틱 라벨 글꼴 크기
'ytick.labelsize': 12,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 11,
'legend.title_fontsize': 11, # 범례 글꼴 크기
'figure.titlesize': 14    # figure 제목 글꼴 크기
})

ax = sns.boxplot(data=mean_tu_df, x="Group", y="Mean Pre TU", showfliers=False)
plt.title("Mean Pre TU: IR vs. AR")
plt.ylabel("Mean Pre TU")
plt.xlabel("")

sns.stripplot(y='Mean Pre TU', x='Group', data=mean_tu_df, 
            order=['AR','IR'], 
            color='#7D7C7C',  # Color of the points
            size=4,         # Size of the points
            jitter=True,    # Adds some jitter to avoid overlapping
            alpha=0.8,
            ax=ax)

from statannot import add_stat_annotation

add_stat_annotation(ax, data=mean_tu_df, x='Group', y='Mean Pre TU',
                    box_pairs=[('AR','IR')], 
                    order = ['AR','IR'], 
                    comparisons_correction=None,
                    test='Mann-Whitney', text_format='simple', loc='inside', fontsize=12
                    )
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/int_model/onlyint_preTU_boxplot.pdf', bbox_inches='tight', dpi=300)

plt.show()


plt.figure(figsize=(4,6))
#sns.set_style("whitegrid")
sns.set_theme(style='ticks',palette='pastel')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 12,    # x축 틱 라벨 글꼴 크기
'ytick.labelsize': 12,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 11,
'legend.title_fontsize': 11, # 범례 글꼴 크기
'figure.titlesize': 14    # figure 제목 글꼴 크기
})

ax = sns.boxplot(data=mean_tu_df, x="Group", y="Mean Delta TU", showfliers=False)
plt.title("Mean Delta TU: IR vs. AR")
plt.ylabel("Mean Delta TU")
plt.xlabel("")

sns.stripplot(y='Mean Delta TU', x='Group', data=mean_tu_df, 
            order=['AR','IR'], 
            color='#7D7C7C',  # Color of the points
            size=4,         # Size of the points
            jitter=True,    # Adds some jitter to avoid overlapping
            alpha=0.8,
            ax=ax)

from statannot import add_stat_annotation

add_stat_annotation(ax, data=mean_tu_df, x='Group', y='Mean Delta TU',
                    box_pairs=[('AR','IR')], 
                    order = ['AR','IR'], 
                    comparisons_correction=None,
                    test='Mann-Whitney', text_format='simple', loc='inside', fontsize=12
                    )

#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/int_model/onlyint_deltaTU_boxplot.pdf', bbox_inches='tight', dpi=300)
plt.show()

# %%
