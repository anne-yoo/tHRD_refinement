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
###^^ pre AR vs. pre IR ###

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
featureset = set1.union(set2)
###

df_bfD = TU.iloc[TU.index.isin(featureset),1::2]
df = df_bfD.T

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
y = sampleinfo.iloc[0::2,3]


#%%
#^^241116 new code###
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42, stratify=y)

# Parameter grid for Grid Search
# param_grid = {
#     'n_estimators': [50, 75, 100],  # Reducing the maximum number of trees
#     'max_depth': [4, 6, 8],  # Further limit tree depth to prevent complex fits
#     'min_samples_split': [10, 20, 30],  # More restrictive splitting
#     'min_samples_leaf': [7, 10, 15],  # Increase minimum samples in a leaf
#     'max_features': ['sqrt', 'log2']  # Limit features considered at each split
# }

param_grid = {
    'n_estimators': [10, 100, 500, 1000],
    'max_depth': [None, 15, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc

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

from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE

rfecv = RFECV(estimator=best_rf, step=1, cv=StratifiedKFold(5), scoring='accuracy')
rfecv.fit(X_train, y_train)

print("Optimal number of features:", rfecv.n_features_)
selected_features = X_train.columns[rfecv.support_]
#print("Selected Features:", selected_features)

# Fit model with selected features
X_train_rfe = X_train[selected_features]
X_test_rfe = X_test[selected_features]
best_rf.fit(X_train_rfe, y_train)


from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc, RocCurveDisplay, PrecisionRecallDisplay

# Predictions and probabilities
y_train_pred = best_rf.predict(X_train_rfe)
y_test_pred = best_rf.predict(X_test_rfe)
y_train_prob = best_rf.predict_proba(X_train_rfe)[:, 1]
y_test_prob = best_rf.predict_proba(X_test_rfe)[:, 1]

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


plt.figure(figsize=(6, 12))
plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='#133E87')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.ylim(-0.5, len(feature_importances) - 0.5)
plt.gca().invert_yaxis() 
plt.tight_layout()
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/40_preTU_featureimportance.pdf', dpi=300, format='pdf', bbox_inches='tight', transparent=True, pad_inches=0.1)
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
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/40_preTU_testROCAUC_curve.pdf',dpi=300, format='pdf', bbox_inches='tight', transparent=True, pad_inches=0.1)
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
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/40_preTU_testPRAUC_curve.pdf',dpi=300, format='pdf', bbox_inches='tight', transparent=True, pad_inches=0.1)
plt.show()

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size as needed
disp.plot(ax=ax, cmap=plt.cm.Greens, values_format='d')  # Use 'd' for integer values

# Change font size of text in confusion matrix
for text in disp.text_.ravel():
    text.set_fontsize(14) 

plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/40_preTU_confusionmatrix.pdf', dpi=300,format='pdf', bbox_inches='tight',transparent=True, pad_inches=0.1)
plt.show()
#%%

# # %%
# ########^^^^^^^^^^^^^^^^ RFE + GridSearchCV ################################################
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42, stratify=y)

# # Define the Random Forest model
# rf = RandomForestClassifier(random_state=42)

# # Set up RFECV
# rfe = RFECV(
#     estimator=rf,
#     step=1,
#     cv=5,  # 5-fold cross-validation
#     scoring='f1',
#     verbose=1
# )

# # Perform RFE
# rfe.fit(X_train, y_train)

# # Get the optimal number of features
# n_features_optimal = rfe.n_features_
# print("Optimal number of features:", n_features_optimal)

# # Get the selected features
# selected_features = X_train.columns[rfe.support_]
# print("Selected features:", selected_features)

# plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("Cross-validation score (F1)")
# plt.plot(range(1, len(rfe.grid_scores_) + 1), rfe.grid_scores_)
# plt.title("RFECV - Number of Features vs. Cross-Validation Score (F1)")
# plt.show()

# # Define a parameter grid for hyperparameter tuning
# param_grid = {
#     'n_estimators': [10, 50, 100, 500, 1000],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'bootstrap': [True, False]
# }

# # Set up GridSearchCV
# grid_search = GridSearchCV(
#     estimator=rf,
#     param_grid=param_grid,
#     cv=5,
#     scoring='f1',
#     verbose=1
# )

# # Fit GridSearchCV on the selected features
# grid_search.fit(X_train[selected_features], y_train)

# # Get the best parameters
# best_params = grid_search.best_params_
# print("Best parameters:", best_params)

# # Evaluate the model with the best parameters on the test set
# best_rf = grid_search.best_estimator_
# y_pred = best_rf.predict(X_test[selected_features])
# accuracy = accuracy_score(y_test, y_pred)
# print("Test set accuracy:", accuracy)

# # Evaluate the model with the best parameters on the test set
# best_rf = grid_search.best_estimator_
# y_pred = best_rf.predict(X_test[selected_features])
# f1 = f1_score(y_test, y_pred)
# print("Test set F1 score:", f1)

# # Plot ROC-AUC
# y_pred_prob = best_rf.predict_proba(X_test[selected_features])[:, 1]
# fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
# roc_auc = roc_auc_score(y_test, y_pred_prob)
# plt.figure()
# plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()

# # Plot Precision-Recall AUC
# precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
# pr_auc = auc(recall, precision)
# plt.figure()
# plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall (PR) Curve')
# plt.legend(loc='lower left')
# plt.show()



# # %%
# # Calculate feature importances using the whole dataset
# rf = RandomForestClassifier(random_state=42)
# rf.fit(df, y)

# # Get feature importances
# feature_importances = rf.feature_importances_
# feature_names = df.columns

# fi_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance': feature_importances
# })

# # Sort the DataFrame by importance
# fi_df = fi_df.sort_values(by='Importance', ascending=False)

# # Plot the feature importances sorted from max to min
# plt.figure(figsize=(4, 16))
# plt.barh(fi_df['Feature'], fi_df['Importance'])
# plt.xlabel('Feature Importance')
# plt.title('Feature Importances from Random Forest')
# plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
# plt.yticks(fontsize=8.5)
# plt.show()

# # Select features based on mean importance
# mean_importance = np.mean(feature_importances)
# selected_features = feature_names[feature_importances > mean_importance]
# print("Selected features:", selected_features)

# # Split the data into training and testing sets using selected features
# X_train, X_test, y_train, y_test = train_test_split(df[selected_features], y, test_size=0.3, random_state=42, stratify=y)

# # Define a parameter grid for hyperparameter tuning
# param_grid = {
#     'n_estimators': [10, 100, 500, 1000],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'bootstrap': [True, False]
# }

# from sklearn.metrics import f1_score, make_scorer, classification_report, accuracy_score
# # Set up GridSearchCV with weighted F1 score
# f1_scorer = make_scorer(f1_score, average='weighted')
# grid_search = GridSearchCV(
#     estimator=rf,
#     param_grid=param_grid,
#     cv=5,
#     #n_jobs=-1,
#     scoring=f1_scorer,
#     verbose=2
# )

# # Fit GridSearchCV on the selected features
# grid_search.fit(X_train[selected_features], y_train)

# # Get the best parameters
# best_params = grid_search.best_params_
# print("Best parameters:", best_params)

# # Evaluate the model with the best parameters on the test set
# best_rf = grid_search.best_estimator_
# y_pred = best_rf.predict(X_test[selected_features])
# accuracy = accuracy_score(y_test, y_pred)
# print("Test set accuracy:", accuracy)

# # Evaluate the model with the best parameters on the test set
# best_rf = grid_search.best_estimator_
# y_pred = best_rf.predict(X_test[selected_features])
# f1 = f1_score(y_test, y_pred)
# print("Test set F1 score:", f1)

# # Plot ROC-AUC
# y_pred_prob = best_rf.predict_proba(X_test[selected_features])[:, 1]
# fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
# roc_auc = roc_auc_score(y_test, y_pred_prob)
# plt.figure()
# plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc='lower right')
# plt.show()

# # Plot Precision-Recall AUC
# precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
# pr_auc = auc(recall, precision)
# plt.figure()
# plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall (PR) Curve')
# plt.legend(loc='lower left')
# plt.show()
# %%
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
# param_grid = {
#     'n_estimators': [10, 100, 500, 1000],
#     'max_depth': [None, 15, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'bootstrap': [True, False]
# }
param_grid = {
    'n_estimators': [50, 75, 100],  # Reducing the maximum number of trees
    'max_depth': [4, 6, 8],  # Further limit tree depth to prevent complex fits
    'min_samples_split': [10, 20, 30],  # More restrictive splitting
    'min_samples_leaf': [7, 10, 15],  # Increase minimum samples in a leaf
    'max_features': ['sqrt', 'log2']  # Limit features considered at each split
}

# Set up GridSearchCV with weighted F1 score
f1_scorer = make_scorer(f1_score, average='weighted')

# Initialize figures for ROC and PR curves
plt.figure(figsize=(8, 6))  # ROC curve figure
plt.figure(figsize=(8, 6))  # PR curve figure


# Loop over 50 random seeds
for seed in [42]:
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
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202408_analysis/preTU_model/ROCcurve_100iter.pdf', bbox_inches="tight")


# Plot mean Precision-Recall curve
plt.figure(2)
mean_precision, mean_recall, _ = precision_recall_curve(y_test, y_pred_prob)
plt.plot(mean_recall, mean_precision, color='#0E3FA9', label=f'Mean PR curve (area = {mean_pr_auc:.2f})')
plt.axhline(y=0.575, color='grey', linestyle='--', alpha=0.7)  # Add horizontal line y=0.575
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202408_analysis/preTU_model/PRcurve_100iter.pdf', bbox_inches="tight")


# Save feature importance to a DataFrame
feature_importance_df = pd.DataFrame(feature_importance_list, columns=df[selected_features].columns)
#feature_importance_df.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202408_analysis/preTU_model/FI_100iter.txt', sep='\t')
print(feature_importance_df)

# Print mean metrics
print(f'Mean F1 Score: {mean_f1}')
print(f'Mean ROC-AUC: {mean_roc_auc}')
print(f'Mean PR-AUC: {mean_pr_auc}')


# #Save metrics to a text file
# with open('/home/jiye/jiye/copycomparison/gDUTresearch/202408_analysis/preTU_model/results_100iter.txt', 'w') as file:
#     file.write(f'Mean F1 Score: {mean_f1}\n')
#     file.write(f'Mean ROC-AUC: {mean_roc_auc}\n')
#     file.write(f'Mean PR-AUC: {mean_pr_auc}\n')
#     file.write(f'Mean accuracy: {mean_acc}\n')

# %%
# ##^^^ heatmap ? ###########################################


# from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
# from scipy.stats.mstats import winsorize
# from sklearn.preprocessing import StandardScaler
# features = ['ENST00000391831.1-AKT1S1', 'ENST00000482687.1-APTX',
#     'ENST00000404251.1-BAZ1B', 'MSTRG.49834.247-BRIP1',
#     'ENST00000310447.5-CASP2',
#     'ENST00000310232.6-CCDC13', 'ENST00000349780.4-CDK5RAP2',
#     'MSTRG.98879.3-CENPW', 'ENST00000368328.4-CENPW',
#     'ENST00000308987.5-CKS1B',
#     'MSTRG.14904.57-CTBP2', 'MSTRG.14904.52-CTBP2',
#     'ENST00000373573.3-HDAC8', 'ENST00000444376.2-HNRNPU',
#     'ENST00000353801.3-HSP90AB1',
#     'ENST00000392915.1-MAP3K19', 'ENST00000375321.1-MAP3K8',
#     'MSTRG.11829.3-MAP3K8',
#     'MSTRG.55994.22-MZT2A', 'MSTRG.18587.4-OTUB1',
#     'MSTRG.92133.3-PAK4', 
#     'MSTRG.1704.189-RCC1', 'ENST00000412779.2-RNASE1',
#     'MSTRG.52876.4-RPS27A',
#     'MSTRG.51515.3-SMC6', 'ENST00000453269.2-TAF6',
#     'MSTRG.30902.1-TFDP1',
#     'MSTRG.74585.63-TFDP2', 'ENST00000395958.2-YWHAZ',
#     'ENST00000395405.1-ZWINT']

# onlypremodel = ['ENST00000457016.1-APC',
# 'MSTRG.97711.85-CCND3',
# 'MSTRG.41549.4-CSNK2A2',
# 'MSTRG.25406.5-DYRK2',
# 'ENST00000545134.1-HELB',
# 'ENST00000344908.5-MAP3K2',
# 'MSTRG.12458.189-MCL1',
# 'ENST00000461554.1-PCBP4',
# 'MSTRG.65506.82-PIGU',
# 'MSTRG.7469.43-PIK3R3',
# 'MSTRG.126473.45-POM121',
# 'ENST00000358514.4-PSMB10',
# 'ENST00000520452.1-PTTG1',
# 'MSTRG.1704.180-RCC1',
# 'MSTRG.31105.1-RNASE1',
# 'MSTRG.17113.1-SFR1',
# 'ENST00000402989.1-SMC6',
# 'ENST00000321464.5-ZBTB38']

# onlydeltamodel = ['ENST00000302759.6-BUB1',
# 'ENST00000399334.3-CEP152',
# 'ENST00000375440.4-CUL4A',
# 'ENST00000342628.2-FOXM1',
# 'MSTRG.49465.142-GRB2',
# 'ENST00000454366.1-GTSE1',
# 'ENST00000373135.3-L3MBTL1',
# 'ENST00000382948.5-MBD1',
# 'MSTRG.39427.204-NRG4',
# 'ENST00000222254.8-PIK3R2',
# 'ENST00000261207.5-PPP1R12A',
# 'ENST00000498508.2-PROX1',
# 'ENST00000575111.1-RNF167',
# 'ENST00000254942.3-TERF2',
# 'ENST00000380122.5-TXLNG',
# 'ENST00000511817.1-XRCC4']

# #%%
# dfinput = df[features].T

# #dfinput = df.iloc[:-1,:].dropna()

# # Step 1: Scale the data
# scaler = StandardScaler()
# scaled_df = pd.DataFrame(scaler.fit_transform(dfinput), index=dfinput.index, columns=dfinput.columns)

# # Step 2: Winsorize the data (example: limit to the 5th and 95th percentiles)
# #winsorized_df = dfinput.apply(lambda x: winsorize(x, limits=[0.05, 0.05]), axis=0)


# # Define the clustering method (e.g., 'single', 'complete', 'average', 'centroid', 'median', 'ward')
# method = 'centroid'  # Change this to use a different clustering method

# # Perform hierarchical clustering
# linkage_matrix = linkage(scaled_df.T, method=method)  # Transpose to cluster samples

# # Map binary values to colors
# lut = {0: 'blue', 1: 'red'}
# y = sampleinfo.iloc[0::2,3]
# y.index = sampleinfo.iloc[1::2,1]
# col_colors = y.map(lut)

# plt.figure(figsize=(15,5))

# # Create a clustermap without the color bar legend and y-axis label
# clustermap = sns.clustermap(
#     scaled_df, 
#     row_cluster=True, 
#     col_cluster=True, 
#     col_linkage=linkage_matrix, 
#     cmap='coolwarm', 
#     col_colors=col_colors,
#     yticklabels=False,
#     xticklabels=False,
#     standard_scale=1,
#     #z_score=1
    
#     #cbar_pos=None  # Remove the color bar legend
# )

# # Customize the plot to remove the y-axis label
# clustermap.ax_heatmap.set_ylabel('')

# # Show the plot
# plt.show()


# %%
#######^^^^^^ boxplot check ##########
plotdf = TU.iloc[:,1::2].T

featurelist = ['ENST00000555055.1-XRCC3',
'MSTRG.11829.3-MAP3K8','MSTRG.17113.1-SFR1',
'ENST00000402989.1-SMC6','ENST00000349780.4-CDK5RAP2','ENST00000394224.3-SIPA1',
'MSTRG.12458.189-MCL1','ENST00000505397.1-LIN54',
'ENST00000373711.2-CDKN1A','MSTRG.114270.18-NSMCE2',
'MSTRG.92133.10-PAK4','ENST00000399334.3-CEP152',
'ENST00000575111.1-RNF167','ENST00000396355.1-NDE1','MSTRG.65357.56-ID1','MSTRG.11771.35-WAC',
'ENST00000498508.2-PROX1','ENST00000482687.1-APTX',
'ENST00000375321.1-MAP3K8','ENST00000539466.1-USP47',
'ENST00000540257.1-TUBB4A','ENST00000401393.3-INO80',
'MSTRG.92133.3-PAK4','ENST00000545134.1-HELB',
'ENST00000355640.3-XIAP','ENST00000321464.5-ZBTB38',
'ENST00000308987.5-CKS1B','ENST00000342628.2-FOXM1',
'ENST00000206765.6-TGM1','ENST00000382865.1-POLN',
'MSTRG.25406.5-DYRK2','ENST00000392915.1-MAP3K19',
'MSTRG.62156.213-AP5S1','MSTRG.17041.3-FBXL15',
'MSTRG.49834.247-BRIP1','ENST00000318407.3-BOK',
'MSTRG.49465.142-GRB2',
'ENST00000457016.1-APC',
'ENST00000433017.1-FIGNL1',
'MSTRG.62679.6-AURKA',
'MSTRG.11771.23-WAC',
'MSTRG.51515.3-SMC6',
'ENST00000531427.1-CUL5',
'ENST00000261207.5-PPP1R12A',
'MSTRG.18587.4-OTUB1',
'ENST00000382948.5-MBD1',
'ENST00000373135.3-L3MBTL1',
'MSTRG.52876.4-RPS27A',
'ENST00000404276.1-CHEK2',
'ENST00000370782.2-MMS19',
'ENST00000310447.5-CASP2',
'ENST00000445403.1-SRC',
'ENST00000380122.5-TXLNG',
'ENST00000572008.1-RPS15A',
'ENST00000219476.3-TSC2',
'ENST00000233027.5-NEK4',
'MSTRG.1704.78-PHACTR4',
'ENST00000376597.4-CEP78',
'MSTRG.104113.4-FIGNL1',
'MSTRG.19662.15-BIRC2',
'ENST00000310232.6-CCDC13',
'ENST00000257430.4-APC',
'ENST00000444376.2-HNRNPU',
'ENST00000344908.5-MAP3K2',
'ENST00000263201.1-CDC45',
'ENST00000361265.4-AJUBA',
'ENST00000453269.2-TAF6',
'ENST00000404251.1-BAZ1B',
'MSTRG.91105.1-NPM1',
'MSTRG.9436.8-NUCKS1',
'ENST00000375440.4-CUL4A',
'ENST00000395913.3-AURKA',
'ENST00000442510.2-PTPRC']

featurelist = ['ENST00000457016.1-APC', 'ENST00000257430.4-APC',
'ENST00000482687.1-APTX', 'MSTRG.62679.6-AURKA',
'ENST00000395913.3-AURKA', 'ENST00000404251.1-BAZ1B',
'MSTRG.19662.15-BIRC2', 'ENST00000318407.3-BOK',
'MSTRG.49834.247-BRIP1', 'ENST00000310447.5-CASP2',
'ENST00000310232.6-CCDC13', 'ENST00000263201.1-CDC45',
'ENST00000349780.4-CDK5RAP2', 'ENST00000373711.2-CDKN1A',
'ENST00000399334.3-CEP152', 'ENST00000376597.4-CEP78',
'ENST00000404276.1-CHEK2', 'ENST00000308987.5-CKS1B',
'ENST00000375440.4-CUL4A', 'ENST00000531427.1-CUL5',
'MSTRG.25406.5-DYRK2', 'ENST00000433017.1-FIGNL1',
'MSTRG.104113.4-FIGNL1', 'ENST00000342628.2-FOXM1',
'MSTRG.49465.142-GRB2', 'ENST00000545134.1-HELB',
'ENST00000444376.2-HNRNPU', 'MSTRG.65357.56-ID1',
'ENST00000373135.3-L3MBTL1', 'ENST00000505397.1-LIN54',
'ENST00000392915.1-MAP3K19', 'ENST00000344908.5-MAP3K2',
'ENST00000375321.1-MAP3K8', 'MSTRG.11829.3-MAP3K8',
'ENST00000382948.5-MBD1', 'MSTRG.12458.189-MCL1',
'ENST00000370782.2-MMS19', 'ENST00000233027.5-NEK4',
'MSTRG.91105.1-NPM1', 'MSTRG.114270.18-NSMCE2', 'MSTRG.9436.8-NUCKS1',
'MSTRG.18587.4-OTUB1', 'MSTRG.92133.10-PAK4', 'MSTRG.92133.3-PAK4',
'MSTRG.1704.78-PHACTR4', 'ENST00000382865.1-POLN',
'ENST00000261207.5-PPP1R12A', 'ENST00000498508.2-PROX1',
'ENST00000442510.2-PTPRC', 'ENST00000575111.1-RNF167',
'ENST00000572008.1-RPS15A', 'MSTRG.52876.4-RPS27A',
'MSTRG.17113.1-SFR1', 'ENST00000394224.3-SIPA1', 'MSTRG.51515.3-SMC6',
'ENST00000402989.1-SMC6', 'ENST00000445403.1-SRC',
'ENST00000453269.2-TAF6', 'ENST00000206765.6-TGM1',
'ENST00000219476.3-TSC2', 'ENST00000380122.5-TXLNG',
'ENST00000539466.1-USP47', 'MSTRG.11771.23-WAC', 'MSTRG.11771.35-WAC',
'ENST00000355640.3-XIAP', 'ENST00000555055.1-XRCC3'] 

featurelist = ['ENST00000457016.1-APC', 'ENST00000257430.4-APC',
'ENST00000482687.1-APTX', 'ENST00000404251.1-BAZ1B',
'MSTRG.49834.247-BRIP1', 'ENST00000310447.5-CASP2',
'ENST00000349780.4-CDK5RAP2', 'ENST00000376597.4-CEP78',
'ENST00000308987.5-CKS1B', 'ENST00000375440.4-CUL4A',
'ENST00000531427.1-CUL5', 'MSTRG.49465.142-GRB2',
'ENST00000545134.1-HELB', 'ENST00000444376.2-HNRNPU',
'ENST00000373135.3-L3MBTL1', 'ENST00000392915.1-MAP3K19',
'ENST00000375321.1-MAP3K8', 'MSTRG.11829.3-MAP3K8',
'ENST00000382948.5-MBD1', 'MSTRG.12458.189-MCL1',
'ENST00000233027.5-NEK4', 'MSTRG.18587.4-OTUB1', 'MSTRG.92133.10-PAK4',
'MSTRG.92133.3-PAK4', 'MSTRG.1704.78-PHACTR4',
'ENST00000261207.5-PPP1R12A', 'ENST00000575111.1-RNF167',
'MSTRG.52876.4-RPS27A', 'MSTRG.17113.1-SFR1', 'MSTRG.51515.3-SMC6',
'ENST00000453269.2-TAF6', 'ENST00000206765.6-TGM1',
'ENST00000219476.3-TSC2', 'ENST00000380122.5-TXLNG',
'ENST00000539466.1-USP47', 'ENST00000355640.3-XIAP']

featurelist = ['ENST00000391831.1-AKT1S1', 'ENST00000457016.1-APC',
       'ENST00000482687.1-APTX', 'ENST00000404251.1-BAZ1B',
       'ENST00000318407.3-BOK', 'MSTRG.49834.247-BRIP1',
       'ENST00000302759.6-BUB1', 'ENST00000310447.5-CASP2',
       'ENST00000310232.6-CCDC13', 'ENST00000308108.4-CCNE2',
       'ENST00000349780.4-CDK5RAP2', 'MSTRG.98879.3-CENPW',
       'ENST00000368328.4-CENPW', 'ENST00000399334.3-CEP152',
       'ENST00000376597.4-CEP78', 'ENST00000404276.1-CHEK2',
       'ENST00000308987.5-CKS1B', 'MSTRG.14904.57-CTBP2',
       'MSTRG.14904.52-CTBP2', 'ENST00000375440.4-CUL4A',
       'ENST00000531427.1-CUL5', 'ENST00000346618.3-E2F3',
       'ENST00000342628.2-FOXM1', 'MSTRG.49465.142-GRB2',
       'ENST00000454366.1-GTSE1', 'ENST00000373573.3-HDAC8',
       'ENST00000545134.1-HELB', 'ENST00000353801.3-HSP90AB1',
       'ENST00000373135.3-L3MBTL1', 'ENST00000505397.1-LIN54',
       'ENST00000392915.1-MAP3K19', 'ENST00000375321.1-MAP3K8',
       'MSTRG.11829.3-MAP3K8', 'ENST00000262815.8-MAU2',
       'ENST00000382948.5-MBD1', 'MSTRG.12458.189-MCL1',
       'MSTRG.55994.22-MZT2A', 'MSTRG.39427.204-NRG4',
       'MSTRG.114270.18-NSMCE2', 'MSTRG.18587.4-OTUB1', 'MSTRG.92133.10-PAK4',
       'MSTRG.92133.3-PAK4', 'ENST00000222254.8-PIK3R2',
       'ENST00000382865.1-POLN', 'MSTRG.126473.45-POM121',
       'ENST00000261207.5-PPP1R12A', 'ENST00000490777.2-PPP2R2D',
       'ENST00000498508.2-PROX1', 'ENST00000358514.4-PSMB10',
       'ENST00000442510.2-PTPRC', 'ENST00000520452.1-PTTG1',
       'MSTRG.1704.189-RCC1', 'ENST00000412779.2-RNASE1',
       'ENST00000575111.1-RNF167', 'MSTRG.52876.4-RPS27A',
       'MSTRG.17113.1-SFR1', 'MSTRG.51515.3-SMC6', 'ENST00000317296.5-STAG3',
       'ENST00000453269.2-TAF6', 'ENST00000254942.3-TERF2',
       'MSTRG.30902.1-TFDP1', 'MSTRG.74585.63-TFDP2',
       'ENST00000539466.1-USP47', 'ENST00000355640.3-XIAP',
       'ENST00000555055.1-XRCC3', 'ENST00000511817.1-XRCC4',
       'ENST00000395958.2-YWHAZ', 'ENST00000395405.1-ZWINT']
# Reactome2022_featurelist.pkl / BP2018_featurelist.pkl
with open('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/featureselection/Reactome2022_featurelist.pkl', 'rb') as file:
    featurelist_reactome = pickle.load(file)

with open('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/featureselection/BP2018_featurelist.pkl', 'rb') as file:
    featurelist_bp = pickle.load(file)

featurelist = set(featurelist_bp).union(set(featurelist_reactome))

plotdf = plotdf[featurelist]
plotdf['mean pre TU'] = plotdf.mean(axis=1)

plotdf['response'] = y.tolist()
plotdf['response'].replace({0: 'IR', 1: 'AR'}, inplace=True)

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


ax = sns.boxplot(y='mean pre TU', x='response', data=plotdf, 
            showfliers=True, order=['IR','AR'], 
            #color='#FFD92E'
            palette='vlag'
            #meanprops = {"marker":"o", "color":"blue","markeredgecolor": "blue", "markersize":"10"}
            )
#ax.set_ylim([0,0.01])
ax.set_ylabel('pre TU')
#ax.set_xticklabels(['IR','AR'])
#plt.title(transname)
sns.despine()

sns.stripplot(y='mean pre TU', x='response', data=plotdf, 
            order=['IR', 'AR'], 
            color='grey',  # Color of the points
            size=6,         # Size of the points
            jitter=True,    # Adds some jitter to avoid overlapping
            alpha=0.8,
            ax=ax)


from statannot import add_stat_annotation


add_stat_annotation(ax, data=plotdf, x='response', y='mean pre TU',
                    box_pairs=[('IR','AR')], 
                    comparisons_correction=None,
                    test='Mann-Whitney', text_format='simple', loc='inside', fontsize=13
                    )

#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/137features_discovery_boxplot.pdf', bbox_inches='tight', dpi=300)
plt.show()


# %%
####^^^ AR post vs. IR pre #################

post = TU.iloc[:,0::2].T
post = post[featureset]
post['response'] = y.tolist()
post['response'].replace({0: 'IR', 1: 'AR'}, inplace=True)
pre = TU.iloc[:,1::2].T
pre = pre[featureset]
pre['response'] = y.tolist()
pre['response'].replace({0: 'IR', 1: 'AR'}, inplace=True)
#%%
featurelist = ['ENST00000391831.1-AKT1S1', 'ENST00000457016.1-APC',
       'ENST00000482687.1-APTX', 'ENST00000404251.1-BAZ1B',
       'ENST00000318407.3-BOK', 'MSTRG.49834.247-BRIP1',
       'ENST00000302759.6-BUB1', 'ENST00000310447.5-CASP2',
       'ENST00000310232.6-CCDC13', 'ENST00000308108.4-CCNE2',
       'ENST00000349780.4-CDK5RAP2', 'MSTRG.98879.3-CENPW',
       'ENST00000368328.4-CENPW', 'ENST00000399334.3-CEP152',
       'ENST00000376597.4-CEP78', 'ENST00000404276.1-CHEK2',
       'ENST00000308987.5-CKS1B', 'MSTRG.14904.57-CTBP2',
       'MSTRG.14904.52-CTBP2', 'ENST00000375440.4-CUL4A',
       'ENST00000531427.1-CUL5', 'ENST00000346618.3-E2F3',
       'ENST00000342628.2-FOXM1', 'MSTRG.49465.142-GRB2',
       'ENST00000454366.1-GTSE1', 'ENST00000373573.3-HDAC8',
       'ENST00000545134.1-HELB', 'ENST00000353801.3-HSP90AB1',
       'ENST00000373135.3-L3MBTL1', 'ENST00000505397.1-LIN54',
       'ENST00000392915.1-MAP3K19', 'ENST00000375321.1-MAP3K8',
       'MSTRG.11829.3-MAP3K8', 'ENST00000262815.8-MAU2',
       'ENST00000382948.5-MBD1', 'MSTRG.12458.189-MCL1',
       'MSTRG.55994.22-MZT2A', 'MSTRG.39427.204-NRG4',
       'MSTRG.114270.18-NSMCE2', 'MSTRG.18587.4-OTUB1', 'MSTRG.92133.10-PAK4',
       'MSTRG.92133.3-PAK4', 'ENST00000222254.8-PIK3R2',
       'ENST00000382865.1-POLN', 'MSTRG.126473.45-POM121',
       'ENST00000261207.5-PPP1R12A', 'ENST00000490777.2-PPP2R2D',
       'ENST00000498508.2-PROX1', 'ENST00000358514.4-PSMB10',
       'ENST00000442510.2-PTPRC', 'ENST00000520452.1-PTTG1',
       'MSTRG.1704.189-RCC1', 'ENST00000412779.2-RNASE1',
       'ENST00000575111.1-RNF167', 'MSTRG.52876.4-RPS27A',
       'MSTRG.17113.1-SFR1', 'MSTRG.51515.3-SMC6', 'ENST00000317296.5-STAG3',
       'ENST00000453269.2-TAF6', 'ENST00000254942.3-TERF2',
       'MSTRG.30902.1-TFDP1', 'MSTRG.74585.63-TFDP2',
       'ENST00000539466.1-USP47', 'ENST00000355640.3-XIAP',
       'ENST00000555055.1-XRCC3', 'ENST00000511817.1-XRCC4',
       'ENST00000395958.2-YWHAZ', 'ENST00000395405.1-ZWINT']


post['mean TU'] = post.mean(axis=1)
pre['mean TU'] = pre.mean(axis=1)
post['sample'] = 'AR post'
post.loc[post['response']=='IR','sample'] = 'IR post'
pre['sample'] = 'AR pre'
pre.loc[pre['response']=='IR','sample'] = 'IR pre'

#%%
plotdf = pd.concat([pre,post],axis=0)
plotdf['sample'] = pd.Categorical(
    plotdf['sample'], 
    categories=['AR pre', 'AR post', 'IR pre', 'IR post'], 
    ordered=True
)
# %%
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


ax = sns.boxplot(y='mean TU', x='sample', data=plotdf, 
            showfliers=True, order=['AR pre', 'AR post','IR pre','IR post'], 
            #color='#FFD92E'
            palette='vlag'
            #meanprops = {"marker":"o", "color":"blue","markeredgecolor": "blue", "markersize":"10"}
            )
#ax.set_ylim([0,0.01])
ax.set_ylabel('TU')
#ax.set_xticklabels(['IR','AR'])
#plt.title(transname)
sns.despine()

sns.stripplot(y='mean TU', x='sample', data=plotdf, 
            order=['AR pre', 'AR post','IR pre','IR post'], 
            color='grey',  # Color of the points
            size=6,         # Size of the points
            jitter=True,    # Adds some jitter to avoid overlapping
            alpha=0.8,
            ax=ax)


from statannot import add_stat_annotation


add_stat_annotation(ax, data=plotdf, x='sample', y='mean TU',
                    box_pairs=[('AR post','IR pre'),('IR pre','IR post'),('AR pre','IR pre')], 
                    comparisons_correction=None,
                    test='Mann-Whitney', text_format='simple', loc='inside', fontsize=13
                    )

plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/137features_compare_all_boxplot.pdf', bbox_inches='tight', dpi=300)
plt.show()
# %%
