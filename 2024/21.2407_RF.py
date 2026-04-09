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
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

# %%
TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0)

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
responder = list(set(sampleinfo.loc[(sampleinfo['response']==1),'sample_id']))
nonresponder = list(set(sampleinfo.loc[(sampleinfo['response']==0),'sample_id']))

import pickle

# Reactome2022_featurelist.pkl / BP2018_featurelist.pkl
with open('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/featureselection/Reactome2022_featurelist.pkl', 'rb') as file:
    featurelist_reactome = pickle.load(file)

with open('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/featureselection/BP2018_featurelist.pkl', 'rb') as file:
    featurelist_bp = pickle.load(file)

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc, roc_curve, plot_roc_curve, plot_precision_recall_curve


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
plt.figure(figsize=(5,4))
sns.set_style("white")

set1 = set(featurelist_bp).union(set(featurelist_reactome))
set2 = set(featurelist_bp).intersection(set(featurelist_reactome))


###
featureset = set1
###
tu_atD = TU.iloc[TU.index.isin(featureset),0::2]
tu_bfD = TU.iloc[TU.index.isin(featureset),1::2]
tu_bfD.columns = tu_bfD.columns.str[:-4]
tu_atD.columns = tu_atD.columns.str[:-4]
df = tu_atD.subtract(tu_bfD)
df = df.T

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
y = sampleinfo.iloc[0::2,3]

#%%
########^^^^^^^^^^^^^^^^ RFE + GridSearchCV ################################################
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42, stratify=y)

# Define the Random Forest model
rf = RandomForestClassifier(random_state=42)

# Set up RFECV
rfe = RFECV(
    estimator=rf,
    step=1,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    verbose=1
)

# Perform RFE
rfe.fit(X_train, y_train)

# Get the optimal number of features
n_features_optimal = rfe.n_features_
print("Optimal number of features:", n_features_optimal)

# Get the selected features
selected_features = X_train.columns[rfe.support_]
print("Selected features:", selected_features)

plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross-validation score (F1)")
plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
plt.title("RFECV - Number of Features vs. Cross-Validation Score (F1)")
plt.show()

# Define a parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [10, 50, 100, 500, 1000],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    verbose=2
)

# Fit GridSearchCV on the selected features
grid_search.fit(X_train[selected_features], y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best parameters:", best_params)

# Evaluate the model with the best parameters on the test set
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test[selected_features])
accuracy = accuracy_score(y_test, y_pred)
print("Test set accuracy:", accuracy)

# Evaluate the model with the best parameters on the test set
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test[selected_features])
f1 = f1_score(y_test, y_pred)
print("Test set F1 score:", f1)

# Plot ROC-AUC
y_pred_prob = best_rf.predict_proba(X_test[selected_features])[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Plot Precision-Recall AUC
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recall, precision)
plt.figure()
plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall (PR) Curve')
plt.legend(loc='lower left')
plt.show()



#%%
###############^^^^^^^^^^ mean feature importance + gridsearchCV ############################################################



# Calculate feature importances using the whole dataset
rf = RandomForestClassifier(random_state=42)
rf.fit(df, y)

# Get feature importances
feature_importances = rf.feature_importances_
feature_names = df.columns

fi_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# Sort the DataFrame by importance
fi_df = fi_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances sorted from max to min
plt.figure(figsize=(4, 16))
plt.barh(fi_df['Feature'], fi_df['Importance'])
plt.xlabel('Feature Importance')
plt.title('Feature Importances from Random Forest')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
plt.yticks(fontsize=8.5)
plt.show()

# Select features based on mean importance
mean_importance = np.mean(feature_importances)
selected_features = feature_names[feature_importances > mean_importance]
print("Selected features:", selected_features)

#%%
# Split the data into training and testing sets using selected features
X_train, X_test, y_train, y_test = train_test_split(df[selected_features], y, test_size=0.3, random_state=42, stratify=y)

# Define a parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [10, 100, 500, 1000],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

from sklearn.metrics import f1_score, make_scorer, classification_report, accuracy_score
# Set up GridSearchCV with weighted F1 score
f1_scorer = make_scorer(f1_score, average='weighted')
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    #n_jobs=-1,
    scoring=f1_scorer,
    verbose=2
)

# Fit GridSearchCV on the selected features
grid_search.fit(X_train[selected_features], y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best parameters:", best_params)

# Evaluate the model with the best parameters on the test set
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test[selected_features])
accuracy = accuracy_score(y_test, y_pred)
print("Test set accuracy:", accuracy)

# Evaluate the model with the best parameters on the test set
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test[selected_features])
f1 = f1_score(y_test, y_pred)
print("Test set F1 score:", f1)

# Plot ROC-AUC
y_pred_prob = best_rf.predict_proba(X_test[selected_features])[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = roc_auc_score(y_test, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Plot Precision-Recall AUC
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = auc(recall, precision)
plt.figure()
plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall (PR) Curve')
plt.legend(loc='lower left')
plt.show()


#%%
#####^^^^^ random seed 100 iterations #############
from sklearn.metrics import f1_score, make_scorer, classification_report, accuracy_score
# Calculate feature importances using the whole dataset
rf = RandomForestClassifier(random_state=42)
rf.fit(df, y)

# Get feature importances
feature_importances = rf.feature_importances_
feature_names = df.columns

fi_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# Sort the DataFrame by importance
fi_df = fi_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances sorted from max to min
plt.figure(figsize=(4, 16))
plt.barh(fi_df['Feature'], fi_df['Importance'])
plt.xlabel('Feature Importance')
plt.title('Feature Importances from Random Forest')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
plt.yticks(fontsize=8.5)
plt.show()

# Select features based on mean importance
mean_importance = np.mean(feature_importances)
selected_features = feature_names[feature_importances > mean_importance]
print("Selected features:", selected_features)

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
    X_train, X_test, y_train, y_test = train_test_split(df[selected_features], y, test_size=0.3, random_state=seed, stratify=y)
    
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










# %%
####^^ PCA -> RF ############################
df = df.T
pca = PCA(n_components=3)
pca.fit(df)
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance)
# Get the loadings (components)
loadings = pca.components_

# Display the loadings
feature_names = df.columns
for i, component in enumerate(loadings):
    print(f"Principal Component {i+1}:")
    for feature, loading in zip(feature_names, component):
        print(f" - {feature}: {loading:.4f}")
selected_features = np.abs(loadings[0]) > 0.05  # Threshold of 0.5 for selection
selected_feature_names = feature_names[selected_features]
print("Selected Features:", selected_feature_names)

# Create a new DataFrame with selected features
selected_data = df[selected_feature_names]

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
y = sampleinfo.iloc[0::2,3]
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(selected_data, y, test_size=0.3, random_state=0)

# Train the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
