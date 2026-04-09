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
##^^241116 new code ####


# Parameter grid for Grid Search
# param_grid = {
#     'n_estimators': [50, 75, 100],  # Reducing the maximum number of trees
#     'max_depth': [4, 6, 8],  # Further limit tree depth to prevent complex fits
#     'min_samples_split': [10, 20, 30],  # More restrictive splitting
#     'min_samples_leaf': [7, 10, 15],  # Increase minimum samples in a leaf
#     'max_features': ['sqrt', 'log2']  # Limit features considered at each split
# }

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15], #15
    'min_samples_split': [5, 10, 15], #5
    'min_samples_leaf': [2, 4, 6], #2
    'max_features': ['sqrt', 'log2']
}

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc
rf = RandomForestClassifier(random_state=42,class_weight='balanced')
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

# Select features based on mean importance
mean_importance = np.median(feature_importances) ###median?
selected_features = feature_names[feature_importances > mean_importance]
print("Selected features:", selected_features)

X_train, X_test, y_train, y_test = train_test_split(df[selected_features], y, test_size=0.3, random_state=42, stratify=y)


# Initialize Random Forest model
rf = RandomForestClassifier(random_state=42,class_weight='balanced')

# Stratified k-fold cross-validation
cv = StratifiedKFold(n_splits=5)

# Grid Search with cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Best parameters from Grid Search
best_rf = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)


best_rf.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc, RocCurveDisplay, PrecisionRecallDisplay

# Predictions and probabilities
y_train_pred = best_rf.predict(X_train)
y_test_pred = best_rf.predict(X_test)
y_train_prob = best_rf.predict_proba(X_train)[:, 1]
y_test_prob = best_rf.predict_proba(X_test)[:, 1]

# Metrics calculations
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)
train_roc_auc = roc_auc_score(y_train, y_train_prob)
test_roc_auc = roc_auc_score(y_test, y_test_prob)

# Precision-Recall AUC
precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_prob)
precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_prob)
train_pr_auc = auc(recall_train, precision_train)
test_pr_auc = auc(recall_test, precision_test)

# Feature importance (based on RFE-selected features)
feature_importances = pd.DataFrame({
    'Feature': selected_features,  # Column names of X #X.columns
    'Importance': best_rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Print metrics
print("\nTrain Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
print("Train ROC-AUC:", train_roc_auc)
print("Test ROC-AUC:", test_roc_auc)
print("Train PR-AUC:", train_pr_auc)
print("Test PR-AUC:", test_pr_auc)
print("Train F1 Score:", train_f1)
print("Test F1 Score:", test_f1)
print("\nFeature Importances (RFE Selected):\n", feature_importances)

#%%

plt.figure(figsize=(6, 12))
plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='#133E87')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.ylim(-0.5, len(feature_importances) - 0.5)
plt.gca().invert_yaxis() 
plt.tight_layout()
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/40_deltaTU_featureimportance.pdf', dpi=300, format='pdf', bbox_inches='tight', transparent=True, pad_inches=0.1)
plt.show()

# Plot ROC Curve
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, roc_curve, precision_recall_curve
# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, color='#B43F3F', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)  # Diagonal reference line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/40_deltaTU_testROCAUC_curve.pdf',dpi=300, format='pdf', bbox_inches='tight', transparent=True, pad_inches=0.1)
plt.show()

# Calculate Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_test_prob)
pr_auc = auc(recall, precision)

# Plot Precision-Recall curve
plt.figure(figsize=(7, 5))
plt.plot(recall, precision, color='#B43F3F', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
positive_proportion = sum(y_test) / len(y_test)
plt.axhline(y=positive_proportion, linestyle='--', color='gray', lw=1, label='Baseline')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0,1.05])
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/40_deltaTU_testPRAUC_curve.pdf',dpi=300, format='pdf', bbox_inches='tight', transparent=True, pad_inches=0.1)
plt.show()

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size as needed
disp.plot(ax=ax, cmap=plt.cm.Greens, values_format='d')  # Use 'd' for integer values

# Change font size of text in confusion matrix
for text in disp.text_.ravel():
    text.set_fontsize(14) 

plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/40_deltaTU_confusionmatrix.pdf', dpi=300,format='pdf', bbox_inches='tight',transparent=True, pad_inches=0.1)
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

# # Plot the feature importances sorted from max to min
# plt.figure(figsize=(4, 16))
# plt.barh(fi_df['Feature'], fi_df['Importance'])
# plt.xlabel('Feature Importance')
# plt.title('Feature Importances from Random Forest')
# plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
# plt.yticks(fontsize=8.5)
# plt.show()

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
    'n_estimators': [10, 100, 500, 1000],
    'max_depth': [None, 15, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Set up GridSearchCV with weighted F1 score
f1_scorer = make_scorer(f1_score, average='weighted')

# Initialize figures for ROC and PR curves
plt.figure(figsize=(8, 6))  # ROC curve figure
plt.figure(figsize=(8, 6))  # PR curve figure


# Loop over 50 random seeds
for seed in range(100):
    # Split the data into training and testing sets
    print("Random Seed: ",seed)
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
    plt.plot(fpr, tpr, color='#CE9898', alpha=0.2)

    # Plot Precision-Recall curve for each iteration
    plt.figure(2)
    plt.plot(recall, precision, color='#8BBBC4', alpha=0.2)

# Mean metrics
mean_f1 = np.mean(f1_scores)
mean_roc_auc = np.mean(roc_aucs)
mean_pr_auc = np.mean(pr_aucs)
mean_acc = np.mean(accs)

# Plot mean ROC curve
plt.figure(1)
mean_fpr, mean_tpr, _ = roc_curve(y_test, y_pred_prob)
plt.plot(mean_fpr, mean_tpr, color='#D11313', label=f'Mean ROC curve (area = {mean_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey', alpha=0.7)  # Add diagonal line x=y
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves over 100 Random Seeds')
plt.legend(loc='lower right')
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202408_analysis/deltaTU_model/ROCcurve_100iter.pdf', bbox_inches="tight")


# Plot mean Precision-Recall curve
plt.figure(2)
mean_precision, mean_recall, _ = precision_recall_curve(y_test, y_pred_prob)
plt.plot(mean_recall, mean_precision, color='#0E3FA9', label=f'Mean PR curve (area = {mean_pr_auc:.2f})')
plt.axhline(y=0.575, color='grey', linestyle='--', alpha=0.7)  # Add horizontal line y=0.575
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves over 100 Random Seeds')
plt.legend(loc='lower left')
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202408_analysis/deltaTU_model/PRcurve_100iter.pdf', bbox_inches="tight")


# Save feature importance to a DataFrame
feature_importance_df = pd.DataFrame(feature_importance_list, columns=df[selected_features].columns)
feature_importance_df.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202408_analysis/deltaTU_model/FI_100iter.txt', sep='\t')
print(feature_importance_df)

# Print mean metrics
print(f'Mean F1 Score: {mean_f1}')
print(f'Mean ROC-AUC: {mean_roc_auc}')
print(f'Mean PR-AUC: {mean_pr_auc}')

# Save metrics to a text file
with open('/home/jiye/jiye/copycomparison/gDUTresearch/202408_analysis/deltaTU_model/results_100iter.txt', 'w') as file:
    file.write(f'Mean F1 Score: {mean_f1}\n')
    file.write(f'Mean ROC-AUC: {mean_roc_auc}\n')
    file.write(f'Mean PR-AUC: {mean_pr_auc}\n')
    file.write(f'Mean accuracy: {mean_acc}\n')