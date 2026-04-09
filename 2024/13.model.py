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
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV



sns.set(font = 'Arial')
sns.set_style("whitegrid")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler

#%%
##### ^^^^^^^^^ AR: pre vs. post #######################
input = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/model/enrmajordutTU.txt', sep='\t', index_col=0)
sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
responder = sampleinfo.loc[(sampleinfo['response']==1),'sample_full'].to_list()
nonresponder = sampleinfo.loc[(sampleinfo['response']==0),'sample_full'].to_list()
y = [1,0]*40
y = pd.DataFrame(y)
y.index = input.columns

input = input[responder]
X=input.T
y = y.loc[responder,:]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    #random_state=100
                                                    )

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predict probabilities for ROC and PR curves
y_train_pred_prob = model.predict_proba(X_train_scaled)[:, 1]
y_test_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
test_auc = roc_auc_score(y_test, y_test_pred_prob)
print(f"Test AUC: {test_auc:.2f}")


# Calculate AUC for training and test sets
train_auc = roc_auc_score(y_train, y_train_pred_prob)
test_auc = roc_auc_score(y_test, y_test_pred_prob)
print(f"Train AUC: {train_auc:.2f}")
print(f"Test AUC: {test_auc:.2f}")

# Calculate PR-AUC for training and test sets
train_ap = average_precision_score(y_train, y_train_pred_prob)
test_ap = average_precision_score(y_test, y_test_pred_prob)
print(f"Train Average Precision: {train_ap:.2f}")
print(f"Test Average Precision: {test_ap:.2f}")

# Plot ROC Curve
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for y_true, y_pred, set_name in [(y_train, y_train_pred_prob, 'Train'), (y_test, y_test_pred_prob, 'Test')]:
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, label=f'{set_name} (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# Plot Precision-Recall Curve
plt.subplot(1, 2, 2)
for y_true, y_pred, set_name in [(y_train, y_train_pred_prob, 'Train'), (y_test, y_test_pred_prob, 'Test')]:
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.plot(recall, precision, label=f'{set_name} (AP = {average_precision_score(y_true, y_pred):.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()

plt.tight_layout()
plt.show()

# Confusion Matrix for Training and Test Sets
for y_true, y_pred_prob, set_name in [(y_train, y_train_pred_prob, 'Train'), (y_test, y_test_pred_prob, 'Test')]:
    y_pred = np.where(y_pred_prob > 0.5, 1, 0)  # Using 0.5 as threshold
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(f"{set_name} Set Confusion Matrix:\n{conf_matrix}\n")
    # Plotting the confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cbar=True, cmap='Blues',)
    tick_marks = np.arange(2) + 0.5
    plt.xticks(tick_marks, ['False', 'True'], ha='center')
    plt.yticks(tick_marks, ['False', 'True'], va='center')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix at Best Threshold')
    plt.show()

# Plotting Feature Importances
feature_importances = pd.Series(model.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
top_n = 30  # You can adjust this to show more or fewer features
plt.figure(figsize=(8, 6))
feature_importances.iloc[:top_n].plot(kind='barh').invert_yaxis()  # Show top N features
plt.title('Top 30 Feature Importances')
plt.xlabel('Coefficient Magnitude')
plt.ylabel('Features (Transcripts)')
plt.show()

# %%
#%%
##### ^^^^^^^^^ deltaTU: IR vs. AR #######################
input = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/model/enrmajordutTU.txt', sep='\t', index_col=0)
sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
responder = sampleinfo.loc[(sampleinfo['response']==1),'sample_full'].to_list()
nonresponder = sampleinfo.loc[(sampleinfo['response']==0),'sample_full'].to_list()
y = sampleinfo['response']
y = pd.DataFrame(y)

delta_df = pd.DataFrame(index=input.index)

# Iterate over the columns in steps of 2 (since Post and Pre are adjacent)
for i in range(0, len(input.columns), 2):
    post_col = input.columns[i]
    pre_col = input.columns[i + 1]
    
    # Compute the delta and add to the new DataFrame
    # The new column name corresponds to the sample name without 'post' or 'pre'
    sample_name = post_col[:-4]  # Assuming the last 4 characters are 'post' or 'pre'
    delta_df[sample_name + 'delta'] = input[post_col] - input[pre_col]

X=delta_df.T
y = y.iloc[0::2]
y.index = X.index

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    #random_state=100
                                                    )

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predict probabilities for ROC and PR curves
y_train_pred_prob = model.predict_proba(X_train_scaled)[:, 1]
y_test_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
test_auc = roc_auc_score(y_test, y_test_pred_prob)
print(f"Test AUC: {test_auc:.2f}")


# Calculate AUC for training and test sets
train_auc = roc_auc_score(y_train, y_train_pred_prob)
test_auc = roc_auc_score(y_test, y_test_pred_prob)
print(f"Train AUC: {train_auc:.2f}")
print(f"Test AUC: {test_auc:.2f}")

# Calculate PR-AUC for training and test sets
train_ap = average_precision_score(y_train, y_train_pred_prob)
test_ap = average_precision_score(y_test, y_test_pred_prob)
print(f"Train Average Precision: {train_ap:.2f}")
print(f"Test Average Precision: {test_ap:.2f}")

# Plot ROC Curve
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for y_true, y_pred, set_name in [(y_train, y_train_pred_prob, 'Train'), (y_test, y_test_pred_prob, 'Test')]:
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, label=f'{set_name} (AUC = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# Plot Precision-Recall Curve
plt.subplot(1, 2, 2)
for y_true, y_pred, set_name in [(y_train, y_train_pred_prob, 'Train'), (y_test, y_test_pred_prob, 'Test')]:
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plt.plot(recall, precision, label=f'{set_name} (AP = {average_precision_score(y_true, y_pred):.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()

plt.tight_layout()
plt.show()

# Confusion Matrix for Training and Test Sets
for y_true, y_pred_prob, set_name in [(y_train, y_train_pred_prob, 'Train'), (y_test, y_test_pred_prob, 'Test')]:
    y_pred = np.where(y_pred_prob > 0.5, 1, 0)  # Using 0.5 as threshold
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(f"{set_name} Set Confusion Matrix:\n{conf_matrix}\n")
    # Plotting the confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cbar=True, )
    tick_marks = np.arange(2) + 0.5
    plt.xticks(tick_marks, ['False', 'True'], ha='center')
    plt.yticks(tick_marks, ['False', 'True'], va='center')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix at Best Threshold')
    plt.show()

# Plotting Feature Importances
feature_importances = pd.Series(model.coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
top_n = 30  # You can adjust this to show more or fewer features
plt.figure(figsize=(8, 6))
feature_importances.iloc[:top_n].plot(kind='barh').invert_yaxis()  # Show top N features
plt.title('Top 30 Feature Importances')
plt.xlabel('Coefficient Magnitude')
plt.ylabel('Features (Transcripts)')
plt.show()

























# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
#%%
#########***###################################
# input = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/model/enrmajordutTU.txt', sep='\t', index_col=0)
# sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
# responder = sampleinfo.loc[(sampleinfo['response']==1),'sample_full'].to_list()
# nonresponder = sampleinfo.loc[(sampleinfo['response']==0),'sample_full'].to_list()
# y = [1,0]*40
# y = pd.DataFrame(y)
# y.index = input.columns

# input = input[responder]
# X=input.T
# y = y.loc[responder,:]
###########*########################################

#######*############################################
input = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/model/enrmajordutTU.txt', sep='\t', index_col=0)
sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
responder = sampleinfo.loc[(sampleinfo['response']==1),'sample_full'].to_list()
nonresponder = sampleinfo.loc[(sampleinfo['response']==0),'sample_full'].to_list()
y = sampleinfo['response']
y = pd.DataFrame(y)

delta_df = pd.DataFrame(index=input.index)

# Iterate over the columns in steps of 2 (since Post and Pre are adjacent)
for i in range(0, len(input.columns), 2):
    post_col = input.columns[i]
    pre_col = input.columns[i + 1]
    
    # Compute the delta and add to the new DataFrame
    # The new column name corresponds to the sample name without 'post' or 'pre'
    sample_name = post_col[:-4]  # Assuming the last 4 characters are 'post' or 'pre'
    delta_df[sample_name + 'delta'] = input[post_col] - input[pre_col]

X=delta_df.T
y = y.iloc[0::2]
y.index = X.index
#########*##########################################

def plot_roc_curve(y_train_true, y_train_pred_prob, y_test_true, y_test_pred_prob):
    
    # Train ROC
    fpr_train, tpr_train, _ = roc_curve(y_train_true, y_train_pred_prob)
    auc_train = roc_auc_score(y_train_true, y_train_pred_prob)
    plt.plot(fpr_train, tpr_train, label=f'Train AUC = {auc_train:.2f}', color='blue')
    
    # Test ROC
    fpr_test, tpr_test, _ = roc_curve(y_test_true, y_test_pred_prob)
    auc_test = roc_auc_score(y_test_true, y_test_pred_prob)
    plt.plot(fpr_test, tpr_test, label=f'Test AUC = {auc_test:.2f}', color='red')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

def plot_precision_recall_curve(y_train_true, y_train_pred_prob, y_test_true, y_test_pred_prob):
    # Train Precision-Recall
    precision_train, recall_train, _ = precision_recall_curve(y_train_true, y_train_pred_prob)
    ap_train = average_precision_score(y_train_true, y_train_pred_prob)
    plt.plot(recall_train, precision_train, label=f'Train AP = {ap_train:.2f}', color='blue')
    
    # Test Precision-Recall
    precision_test, recall_test, _ = precision_recall_curve(y_test_true, y_test_pred_prob)
    ap_test = average_precision_score(y_test_true, y_test_pred_prob)
    plt.plot(recall_test, precision_test, label=f'Test AP = {ap_test:.2f}', color='red')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    
def plot_feature_importances(model, feature_names, top_n=30):
    # Extract feature importances
    importance_vals = model.coef_[0]
    feature_importances = pd.Series(importance_vals, index=feature_names)
    
    # Sort features by absolute importance
    feature_importances_sorted = feature_importances.sort_values(ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 6))
    feature_importances_sorted.plot(kind='barh')
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important at the top
    plt.title('Top Feature Importances')
    plt.xlabel('Absolute Importance')
    plt.ylabel('Features')
    plt.show()

def plot_confusion_matrix(y_true, y_pred_prob, threshold=0.5):
    # Convert probabilities to binary predictions based on the threshold
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Plot
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', square=True,
                xticklabels=['Pre', 'Post'], yticklabels=['Pre', 'Post'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define models and parameters for Grid Search
model = LogisticRegression(max_iter=1000)
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]
grid = dict(solver=solvers, penalty=penalty, C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid,  cv=cv, scoring='average_precision', error_score=0)
grid_result = grid_search.fit(X_train, y_train)

# Evaluate the best model from grid search on the test set
best_model = grid_result.best_estimator_

# Standardize features (use full dataset for feature names if X is a DataFrame)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Predict probabilities for ROC and PR curves using the best model
y_train_pred_prob = best_model.predict_proba(X_train_scaled)[:, 1]
y_test_pred_prob = best_model.predict_proba(X_test_scaled)[:, 1]

# Calculate AUC and PR-AUC for training and test sets
train_auc = roc_auc_score(y_train, y_train_pred_prob)
test_auc = roc_auc_score(y_test, y_test_pred_prob)
print(f"Train AUC: {train_auc:.2f}")
print(f"Test AUC: {test_auc:.2f}")

train_ap = average_precision_score(y_train, y_train_pred_prob)
test_ap = average_precision_score(y_test, y_test_pred_prob)
print(f"Train Average Precision: {train_ap:.2f}")
print(f"Test Average Precision: {test_ap:.2f}")

# Plotting ROC and Precision-Recall Curves
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_roc_curve(y_train, y_train_pred_prob, y_test, y_test_pred_prob)
plt.subplot(1, 2, 2)
plot_precision_recall_curve(y_train, y_train_pred_prob, y_test, y_test_pred_prob)
plt.tight_layout()
plt.show()

# Plotting Confusion Matrix for the Test Set
plot_confusion_matrix(y_test, y_test_pred_prob)

# If X was a DataFrame and not a result from make_blobs, you could plot feature importances
# Ensure X is a DataFrame with column names for this to work
if isinstance(X, pd.DataFrame):
    plot_feature_importances(best_model, X.columns)


# %%
# # #########***###################################
# input = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/model/ddr_repair_majordut_TU.txt', sep='\t', index_col=0)
# input = input.dropna()
# sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
# responder = sampleinfo.loc[(sampleinfo['response']==1),'sample_full'].to_list()
# nonresponder = sampleinfo.loc[(sampleinfo['response']==0),'sample_full'].to_list()
# y = [1,0]*40
# y = pd.DataFrame(y)
# y.index = input.columns

# input = input[responder]
# X=input.T
# y = y.loc[responder,:]
# ###########*########################################

######*############################################
# input = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/model/ddr_repair_majordut_TU.txt', sep='\t', index_col=0)
# input = input.dropna()
# sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
# responder = sampleinfo.loc[(sampleinfo['response']==1),'sample_full'].to_list()
# nonresponder = sampleinfo.loc[(sampleinfo['response']==0),'sample_full'].to_list()
# y = sampleinfo['response']
# y = pd.DataFrame(y)

# delta_df = pd.DataFrame(index=input.index)

# # Iterate over the columns in steps of 2 (since Post and Pre are adjacent)
# for i in range(0, len(input.columns), 2):
#     post_col = input.columns[i]
#     pre_col = input.columns[i + 1]
    
#     # Compute the delta and add to the new DataFrame
#     # The new column name corresponds to the sample name without 'post' or 'pre'
#     sample_name = post_col[:-4]  # Assuming the last 4 characters are 'post' or 'pre'
#     delta_df[sample_name + 'delta'] = input[post_col] - input[pre_col]

# X=delta_df.T
# y = y.iloc[0::2]
# y.index = X.index
#########*##########################################

# #########^##########################################
# from sklearn.utils import resample

# def bootstrap_resample(X, y, n_multiply):
#     # Convert to NumPy arrays if they're pandas DataFrames/Series to ensure compatibility
#     X = np.array(X) if not isinstance(X, np.ndarray) else X
#     y = np.array(y) if not isinstance(y, np.ndarray) else y

#     # Initialize lists to store resampled datasets
#     X_samples = [X]
#     y_samples = [y]
    
#     for _ in range(n_multiply - 1):
#         X_tmp, y_tmp = resample(X, y, replace=True)
#         X_samples.append(X_tmp)
#         y_samples.append(y_tmp)
    
#     # Use np.concatenate instead of np.vstack and np.hstack for clarity
#     X_resampled = np.concatenate(X_samples, axis=0)
#     y_resampled = np.concatenate(y_samples, axis=0)
    
#     return X_resampled, y_resampled

# X_resampled, y_resampled = bootstrap_resample(X, y, 5)
# #########^##########################################

from sklearn.pipeline import Pipeline

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Define pipeline and grid search parameters
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

param_grid = {
    'model__solver': ['liblinear','sag','saga'], # Using liblinear for simplicity
    'model__penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'model__C': [10, 1.0, 0.1],
}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', )
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_


def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Make probability predictions
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    train_auc = roc_auc_score(y_train, y_train_prob)
    test_auc = roc_auc_score(y_test, y_test_prob)
    print(f"Train AUC: {train_auc:.3f}, Test AUC: {test_auc:.3f}")
    
    train_ap = average_precision_score(y_train, y_train_prob)
    test_ap = average_precision_score(y_test, y_test_prob)
    print(f"Train AP: {train_ap:.3f}, Test AP: {test_ap:.3f}")
    
    # ROC Curve
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for y_true, y_prob, label in [(y_train, y_train_prob, 'Train'), (y_test, y_test_prob, 'Test')]:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y_true, y_prob):.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    # Precision-Recall Curve
    plt.subplot(1, 2, 2)
    for y_true, y_prob, label in [(y_train, y_train_prob, 'Train'), (y_test, y_test_prob, 'Test')]:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.plot(recall, precision, label=f"{label} (AP={average_precision_score(y_true, y_prob):.3f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/model/model1_aucfig.pdf', bbox_inches="tight")
    plt.show()
    
    # Confusion Matrix for Test Set
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pre', 'Post'], yticklabels=['Pre', 'Post'])
    #sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['IR', 'AR'], yticklabels=['IR', 'AR'])
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for Test Set')
    plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/model/model1_confusionmatrix.pdf', bbox_inches="tight")
    plt.show()


# Evaluate the model
evaluate_model(best_model, X_train, y_train, X_test, y_test)

def plot_feature_importances(coefs, feature_names):
    importance = pd.Series(coefs, index=feature_names)
    importance_sorted = importance.abs().sort_values(ascending=False)
    
    plt.figure(figsize=(8, 5))
    importance_sorted.head(20).plot(kind='barh', colormap='summer')
    plt.gca().invert_yaxis()
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importances')
    plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/model/model1_FI.pdf', bbox_inches="tight")
    plt.show()

# Access the logistic regression model directly
lr_model = best_model.named_steps['model']
plot_feature_importances(lr_model.coef_[0], X.columns)


# %%
