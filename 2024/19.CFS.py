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


# %%

TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0)


sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
responder = list(set(sampleinfo.loc[(sampleinfo['response']==1),'sample_id']))
nonresponder = list(set(sampleinfo.loc[(sampleinfo['response']==0),'sample_id']))

import pickle

#%%
# Reactome2022_featurelist.pkl / BP2018_featurelist.pkl
with open('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/featureselection/Reactome2022_featurelist.pkl', 'rb') as file:
    featurelist = pickle.load(file)


finalresult_reac = []

for trans in featurelist:
    brip1_atD = TU.iloc[TU.index==trans,0::2].values[0]
    brip1_bfD = TU.iloc[TU.index==trans,1::2].values[0]

    x = brip1_atD - brip1_bfD ###deltaTU
    y = sampleinfo.iloc[0::2,3]
    #x = np.array(x).reshape(-1, 1)
    #y = np.array(y)

    from sklearn.linear_model import LogisticRegression
    df = pd.DataFrame({'x': x, 'y': y})
    import statsmodels.api as sm

    # Add a constant to the independent variable matrix
    X = sm.add_constant(df['x'])

    # Perform linear regression
    model = sm.OLS(df['y'], X).fit()

    # Get the correlation matrix
    from scipy.stats import pearsonr
    corr_value, _ = pearsonr(df['x'], df['y'])
    
    
    #save it to the result dict
    result = {
    'feature': trans,
    'pval': model.f_pvalue,
    'corr': corr_value
    }

    finalresult_reac.append(result)

finalresult_reac = pd.DataFrame(finalresult_reac)
# %%
# Reactome2022_featurelist.pkl / BP2018_featurelist.pkl
with open('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/featureselection/BP2018_featurelist.pkl', 'rb') as file:
    featurelist = pickle.load(file)


finalresult_bp = []

for trans in featurelist:
    brip1_atD = TU.iloc[TU.index==trans,0::2].values[0]
    brip1_bfD = TU.iloc[TU.index==trans,1::2].values[0]

    x = brip1_atD - brip1_bfD
    y = sampleinfo.iloc[0::2,3]
    #x = np.array(x).reshape(-1, 1)
    #y = np.array(y)

    from sklearn.linear_model import LogisticRegression
    df = pd.DataFrame({'x': x, 'y': y})
    import statsmodels.api as sm

    # Add a constant to the independent variable matrix
    X = sm.add_constant(df['x'])

    # Perform linear regression
    model = sm.OLS(df['y'], X).fit()

    # Get the correlation matrix
    from scipy.stats import pearsonr
    corr_value, _ = pearsonr(df['x'], df['y'])
    
    
    #save it to the result dict
    result = {
    'feature': trans,
    'pval': model.f_pvalue,
    'corr': corr_value
    }

    finalresult_bp.append(result)

finalresult_bp = pd.DataFrame(finalresult_bp)

# %%
#features = list(finalresult_reac[finalresult_reac['pval']<0.05]['feature']) #list(finalresult_reac['feature'])
#features = list(finalresult_reac['feature'])
features = set(finalresult_bp[finalresult_bp['pval']<0.05]['feature']).union(finalresult_reac[finalresult_reac['pval']<0.05]['feature'])
#features = set(finalresult_bp[finalresult_bp['pval']<0.05]['feature']).intersection(finalresult_reac[finalresult_reac['pval']<0.05]['feature'])
#features = set(finalresult_bp['feature']).union(finalresult_reac['feature'])
tu_atD = TU.iloc[TU.index.isin(features),0::2]
tu_bfD = TU.iloc[TU.index.isin(features),1::2]
tu_bfD.columns = tu_bfD.columns.str[:-4]
tu_atD.columns = tu_atD.columns.str[:-4]
df = tu_atD.subtract(tu_bfD)
df['std'] = df.std(axis=1)
df = df.sort_values(by=["std"], ascending=[False])
#df = df[df['std']>0.05]
df = df.iloc[:,:-1]
#%%
y = sampleinfo.iloc[0::2,3]
y.index = df.columns
#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc, roc_curve, plot_roc_curve, plot_precision_recall_curve
X = df.T
from sklearn.metrics import f1_score, make_scorer, classification_report, accuracy_score

# Initialize lists to store metrics and feature importances
f1_scores = []
roc_aucs = []
pr_aucs = []
feature_importance_list = []
accs = []

# Define a parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [10, 50, 100, 500, 1000],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Set up GridSearchCV with weighted F1 score
f1_scorer = make_scorer(f1_score, average='weighted')

# Loop over 50 random seeds
for seed in range(2):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=seed, stratify=y)
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=seed),
        param_grid=param_grid,
        cv=5,
        scoring=f1_scorer,
        verbose=0
    )
    
    # Fit GridSearchCV on the training data
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters
    best_rf = grid_search.best_estimator_
    
    # Predict on the test set
    y_pred = best_rf.predict(X_test)
    y_pred_prob = best_rf.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_auc = auc(recall, precision)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store metrics
    f1_scores.append(f1)
    roc_aucs.append(roc_auc)
    pr_aucs.append(pr_auc)
    accs.append(accuracy)
    
    # Plot ROC curve for each iteration
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure(1)
    plt.plot(fpr, tpr, color='grey', alpha=0.2)

    # Plot Precision-Recall curve for each iteration
    plt.figure(2)
    plt.plot(recall, precision, color='grey', alpha=0.2)

# Mean metrics
mean_f1 = np.mean(f1_scores)
mean_roc_auc = np.mean(roc_aucs)
mean_pr_auc = np.mean(pr_aucs)
mean_acc = np.mean(accs)

# Plot mean ROC curve
plt.figure(1)
mean_fpr, mean_tpr, _ = roc_curve(y_test, y_pred_prob)
plt.plot(mean_fpr, mean_tpr, color='red', label=f'Mean ROC curve (area = {mean_roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves over 50 Random Seeds')
plt.legend(loc='lower right')
plt.show()

# Plot mean Precision-Recall curve
plt.figure(2)
mean_precision, mean_recall, _ = precision_recall_curve(y_test, y_pred_prob)
plt.plot(mean_recall, mean_precision, color='blue', label=f'Mean PR curve (area = {mean_pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves over 50 Random Seeds')
plt.legend(loc='lower left')
plt.show()

# Save feature importance to a DataFrame
feature_importance_df = pd.DataFrame(feature_importance_list, columns=df.columns)
print(feature_importance_df)

# Print mean metrics
print(f'Mean F1 Score: {mean_f1}')
print(f'Mean ROC-AUC: {mean_roc_auc}')
print(f'Mean PR-AUC: {mean_pr_auc}')
print(f'Mean accuracy: {mean_acc}')



#%%
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler

# Step 1: Scale the data
scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

# Step 2: Winsorize the data (example: limit to the 5th and 95th percentiles)
winsorized_df = df.apply(lambda x: winsorize(x, limits=[0.05, 0.05]), axis=0)


# Define the clustering method (e.g., 'single', 'complete', 'average', 'centroid', 'median', 'ward')
method = 'ward'  # Change this to use a different clustering method

# Perform hierarchical clustering
linkage_matrix = linkage(df.T, method=method)  # Transpose to cluster samples

# Map binary values to colors
lut = {0: 'blue', 1: 'red'}
col_colors = y.map(lut)

# Create a clustermap without the color bar legend and y-axis label
clustermap = sns.clustermap(
    df  , 
    row_cluster=True, 
    col_cluster=True, 
    col_linkage=linkage_matrix, 
    cmap='coolwarm', 
    col_colors=col_colors,
    yticklabels=False,
    #standard_scale=0
    
    #cbar_pos=None  # Remove the color bar legend
)

# Customize the plot to remove the y-axis label
clustermap.ax_heatmap.set_ylabel('')

# Show the plot
plt.show()

# %%

#%%
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix, labels=df.columns)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()





#%%
linkage_matrix = linkage(df.T, method=method)

# Step 3: Extract flat clusters with a specified number of clusters
num_clusters = 10  # Specify the desired number of clusters
clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')

# Reassign small clusters to the nearest larger cluster
def reassign_small_clusters(clusters, linkage_matrix, min_size=2):
    cluster_sizes = pd.Series(clusters).value_counts()
    small_clusters = cluster_sizes[cluster_sizes < min_size].index
    new_clusters = clusters.copy()
    
    # Iterate over small clusters
    for small_cluster in small_clusters:
        sample_indices = np.where(clusters == small_cluster)[0]
        
        for sample_index in sample_indices:
            # Find the nearest cluster with size >= min_size
            min_distance = np.inf
            nearest_cluster = None
            
            for larger_cluster in cluster_sizes[cluster_sizes >= min_size].index:
                larger_cluster_indices = np.where(clusters == larger_cluster)[0]
                distance = np.mean([linkage_matrix[np.min([i, j]), 2] for i in larger_cluster_indices for j in sample_indices])
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_cluster = larger_cluster
            
            # Reassign the sample to the nearest larger cluster
            new_clusters[sample_index] = nearest_cluster
    
    return new_clusters

# Reassign single-sample clusters
clusters = reassign_small_clusters(clusters, linkage_matrix)

# Map real group colors (blue for 0, red for 1)
lut_real = {0: 'blue', 1: 'red'}
real_group_colors = y.map(lut_real)

# Map clusters to colors
num_clusters = len(np.unique(clusters))
cluster_colors = sns.color_palette("hsv", num_clusters)
cluster_colors_series = pd.Series(clusters, index=df.columns).map(dict(zip(range(1, num_clusters+1), cluster_colors)))

# Combine the real group colors and cluster colors into a DataFrame
col_colors_df = pd.DataFrame({'Real Group': real_group_colors, 'Cluster': cluster_colors_series})

# Create a clustermap without the color bar legend and y-axis label
clustermap = sns.clustermap(
    df, 
    row_cluster=False, 
    col_cluster=True, 
    col_linkage=linkage_matrix, 
    cmap='coolwarm', 
    col_colors=col_colors_df,
    yticklabels=False,
    cbar_pos=None  # Remove the color bar legend
)

# Customize the plot to remove the y-axis label and column color legend
clustermap.ax_heatmap.set_ylabel('')
clustermap.ax_col_dendrogram.set_ylabel('')
clustermap.ax_col_dendrogram.set_xlabel('')

# Show the plot
plt.show()

# Plot the dendrogram separately for closer inspection
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix, labels=df.columns)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()




#%%
df['response'] = y




#%% #######^^^^^^^^^^^^^^^^^^ only pre #######################
tu_atD = TU.iloc[TU.index.isin(features),0::2]
tu_bfD = TU.iloc[TU.index.isin(features),1::2]
tu_bfD.columns = tu_bfD.columns.str[:-4]
tu_atD.columns = tu_atD.columns.str[:-4]
df = tu_bfD
y = sampleinfo.iloc[0::2,3]
y.index = df.columns
# Define the clustering method (e.g., 'single', 'complete', 'average', 'centroid', 'median', 'ward')
metric='euclidean', #cityblock, minkowski, euclidean, cosine, correlation, hamming, jaccard, chebyshev, canberra, dice, rogerstanimoto, russellrao, sokalsneath
method = 'ward'  # Change this to use a different clustering method

# Perform hierarchical clustering
linkage_matrix = linkage(df.T, method=method)
# Map binary values to colors
lut = {0: 'blue', 1: 'red'}
col_colors = y.map(lut)

# Create a clustermap
sns.set_style('white')
plt.figure(figsize=(5,5))
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 8,    # x축 틱 라벨 글꼴 크기
'ytick.labelsize': 11,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 13,
'legend.title_fontsize': 13, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})

# Step 1: Scale the data
scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

# Step 2: Winsorize the data (example: limit to the 5th and 95th percentiles)
winsorized_df = df.apply(lambda x: winsorize(x, limits=[0.05, 0.05]), axis=0)

# Define the clustering method (e.g., 'single', 'complete', 'average', 'centroid', 'median', 'ward')
method = 'ward'  # Change this to use a different clustering method

# Perform hierarchical clustering
linkage_matrix = linkage(df.T, method=method)  # Transpose to cluster samples

# Map binary values to colors
lut = {0: 'blue', 1: 'red'}
col_colors = y.map(lut)

# Create a clustermap without the color bar legend and y-axis label
clustermap = sns.clustermap(
    df, 
    row_cluster=True, 
    col_cluster=True, 
    col_linkage=linkage_matrix, 
    cmap='coolwarm', 
    col_colors=col_colors,
    yticklabels=False,
    #cbar_pos=None  # Remove the color bar legend
)

# Customize the plot to remove the y-axis label
clustermap.ax_heatmap.set_ylabel('')

# Show the plot
plt.show()

# %%
