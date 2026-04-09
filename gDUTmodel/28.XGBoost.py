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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from xgboost import XGBClassifier 
import scipy.stats as stats

from skopt.space import Real, Integer

##### ^^^^^^^^^ VALIDATION COHORT #######################
input = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/modelinput/validation_TU.txt', sep='\t', index_col=0)
forfeature = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/featureselection/correlation_62.txt', sep='\t', index_col=0)
featurelist = list(forfeature.index)
input=input.T
X = input.loc[:,featurelist]
y = input.iloc[:,-1]
y = [int(float(num_str)) for num_str in y]
y = pd.DataFrame(y)
y.index = X.index






# %%

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import make_scorer, f1_score, auc, precision_recall_curve, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import average_precision_score


param_grid = {
    'n_estimators': [10, 50, 100, 500, 1000],
    'max_depth': [None, 10, 20, 30, 60, 80],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True,False]
}


# Number of Monte Carlo iterations
num_iterations = 30  # For example, 10 iterations

# Create a RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Define scorers
scorers = {
    'f1': 'f1',
    'roc_auc': 'roc_auc',
    'pr_auc': make_scorer(average_precision_score, needs_proba=True)
}

# Lists to store overall scores
all_f1_scores = []
all_roc_auc_scores = []
all_pr_auc_scores = []
roc_curves = []
pr_curves = []
best_params = []
feature_importances = {f'feature_{i}': [] for i in range(X.shape[1])}
f1_scores_and_splits = []

for iteration in range(num_iterations):
    # Split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=iteration, stratify=y)
    
    # Inner cross-validation for parameter tuning
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=iteration)
    
    # GridSearchCV for hyperparameter tuning
    clf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=inner_cv, scoring=scorers, refit='f1', verbose=1)
    clf.fit(X_train, y_train)
    
    # Best model evaluated on the separate test set
    best_model = clf.best_estimator_
    
    # Compute and store the scores
    # Compute the ROC curve and store it
    fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
    roc_curves.append((fpr, tpr))
    
    # Compute the precision-recall curve and store it
    precision, recall, _ = precision_recall_curve(y_test, best_model.predict_proba(X_test)[:, 1])
    pr_curves.append((recall, precision))
    
    y_scores = best_model.predict_proba(X_test)[:, 1]
    all_f1_scores.append(f1_score(y_test, best_model.predict(X_test)))
    all_roc_auc_scores.append(roc_auc_score(y_test, y_scores))
    all_pr_auc_scores.append(average_precision_score(y_test, y_scores))
    
    best_params.append({
        'params': clf.best_params_,
        'std_roc_auc': np.std(roc_curves[-1]),
        'std_pr_auc': np.std(pr_curves[-1])
    })
    
    # Get feature importances and save them
    for i, importance in enumerate(best_model.feature_importances_):
        feature_importances[f'feature_{i}'].append(importance)
    
    # Store the F1 score with the corresponding train/test split
    f1_scores_and_splits.append({
        'f1_score': f1_score(y_test, best_model.predict(X_test)),
        'train_index': X_train.index,
        'test_index': X_test.index,
        'params': clf.best_params_
    })


# Find the set of parameters with the minimum standard deviation for ROC-AUC and PR-AUC
min_std_roc_auc_params = min(best_params, key=lambda x: x['std_roc_auc'])['params']
min_std_pr_auc_params = min(best_params, key=lambda x: x['std_pr_auc'])['params']

# Decide which parameter set to choose
final_params = min_std_pr_auc_params  # or min_std_roc_auc_params, based on your preference

print(f"Parameters minimizing std ROC-AUC: {min_std_roc_auc_params}")
print(f"Parameters minimizing std PR-AUC: {min_std_pr_auc_params}")

###^ visualize std ###
# Assuming std_roc_auc_list and std_pr_auc_list are filled with the standard deviations
std_roc_auc_list = [param['std_roc_auc'] for param in best_params]
std_pr_auc_list = [param['std_pr_auc'] for param in best_params]

# Creating a DataFrame for the plot
std_df = pd.DataFrame({
    'ROC-AUC STD': std_roc_auc_list,
    'PR-AUC STD': std_pr_auc_list
})


###^ visualize mean feature importance ###
# Assuming 'feature_importances' is your dictionary with lists of importances
mean_importances = {int(feature.split('_')[1]): np.mean(importances) 
                    for feature, importances in feature_importances.items()}

# Sorting the mean importances based on the importance value
sorted_indices = sorted(mean_importances, key=mean_importances.get, reverse=True)

# Now we need to match these sorted indices with their respective names in 'X.columns'
# Assuming 'X.columns' is in the same order as your feature numbers
sorted_feature_names = [X.columns[i] for i in sorted_indices]

# Extract the sorted mean importances
sorted_mean_importances = [mean_importances[i] for i in sorted_indices]

# Now we plot using the sorted feature names and their corresponding mean importances
plt.figure(figsize=(15, 10))
plt.bar(sorted_feature_names, sorted_mean_importances, align='center')
plt.xticks(rotation=90)
plt.xlabel('Features')
plt.ylabel('Mean Importance')
plt.title('Mean Feature Importances')
plt.tight_layout()  # Adjusts the plot to ensure everything fits without overlapping
plt.show()


# Melting the DataFrame for seaborn
std_df = std_df.melt(var_name='Metric', value_name='Standard Deviation')

# Creating the jittered dot plot
plt.figure(figsize=(5, 7))
ax = sns.stripplot(data=std_df, x='Metric', y='Standard Deviation', jitter=True)

# Calculate the positions of the categories
category_positions = ax.get_xticks()

# Get the minimum STD for each Metric
min_std_df = std_df.loc[std_df.groupby('Metric')['Standard Deviation'].idxmin()]

# Plot the minimum points
# Use 'category_positions' to align the red dots correctly
sns.scatterplot(x=[category_positions[0]]*len(min_std_df[min_std_df['Metric']=='ROC-AUC STD']),
                y=min_std_df[min_std_df['Metric']=='ROC-AUC STD']['Standard Deviation'],
                color='red', label='Min STD ROC-AUC', s=100, ax=ax)
sns.scatterplot(x=[category_positions[1]]*len(min_std_df[min_std_df['Metric']=='PR-AUC STD']),
                y=min_std_df[min_std_df['Metric']=='PR-AUC STD']['Standard Deviation'],
                color='red', label='Min STD PR-AUC', s=100, ax=ax)
plt.ylim([0.24,0.45])
# Add legend and show plot
plt.legend()
plt.show()

#%%
##^ mean ROC/PR-AUC plot ##
from numpy import interp

mean_fpr = np.linspace(0, 1, 100)

# Interpolate each ROC curve to this common set of false positive rates
interpolated_tprs = []
for fpr, tpr in roc_curves:
    # The 'interp' function should have the new x-axis (mean_fpr) first,
    # followed by the old x-axis (fpr), and then the old y-axis (tpr)
    interpolated_tpr = np.interp(mean_fpr, fpr, tpr)
    interpolated_tpr[0] = 0.0  # Ensuring the curve starts at 0
    interpolated_tprs.append(interpolated_tpr)

mean_tpr = np.mean(interpolated_tprs, axis=0)
mean_tpr[-1] = 1.0  # Ensuring the curve ends at 1

# Calculate the standard deviation of the TPRs at each FPR
tpr_std = np.std(interpolated_tprs, axis=0)

# Mean ROC Curve
plt.figure(figsize=(7, 6))
for tpr in interpolated_tprs:
    plt.plot(mean_fpr, tpr, color='darkgreen', alpha=0.1)  # Adjust alpha for opacity
mean_roc_auc = np.mean(all_roc_auc_scores)
std_roc_auc = np.std(all_roc_auc_scores)
plt.plot(mean_fpr, mean_tpr, color='darkgreen', label=f'Mean ROC (AUC = {mean_roc_auc:.2f} ± {std_roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=0.5, color='red')  # Dashed diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Mean ROC Curve over All Iterations')
plt.legend(loc='lower right')
plt.show()


mean_recall = np.linspace(0, 1, 100)

# Interpolate each PR curve to this common set of recall values
interpolated_precisions = []
for recall, precision in pr_curves:
    # Reverse the recall and precision arrays to start from 0
    recall = recall[::-1]
    precision = precision[::-1]
    
    # Ensure that precision starts with 1.0 at recall 0.0
    if recall[0] != 0:
        recall = np.insert(recall, 0, 0.0)
        precision = np.insert(precision, 0, 1.0)
    
    # Ensure that precision ends with the last known precision value at recall 1.0
    if recall[-1] < 1:
        recall = np.append(recall, 1.0)
        precision = np.append(precision, precision[-1])
    
    interpolated_precision = np.interp(mean_recall, recall, precision)
    interpolated_precisions.append(interpolated_precision)

# Calculate the mean and standard deviation of the precision values
mean_precision = np.mean(interpolated_precisions, axis=0)


# Calculate the standard deviation of the precision values
precision_std = np.std(interpolated_precisions, axis=0)

# Mean PR Curve
plt.figure(figsize=(7, 6))
plt.axhline(y=89/130, color='red', linestyle='--', linewidth=0.5)

for interpolated_precision in interpolated_precisions:
    plt.plot(mean_recall, interpolated_precision, color='darkgreen', alpha=0.1)

mean_pr_auc = np.mean(all_pr_auc_scores)
std_pr_auc = np.std(all_pr_auc_scores)

# Plot the mean PR curve
plt.plot(mean_recall, mean_precision, color='darkgreen',
        label=f'Mean PR (AUC = {mean_pr_auc:.2f} ± {std_pr_auc:.2f})')

plt.xlabel('Recall')
plt.ylim([0,1])
plt.ylabel('Precision')
plt.title('Mean Precision-Recall Curve over All Iterations')
plt.legend(loc='lower left')
plt.show()


##^ final train/test set + visualize ROC-AUC / PR-AUC##
# Find the split with the highest F1 score using the final hyperparameter set
best_split = max([s for s in f1_scores_and_splits if s['params'] == final_params], key=lambda x: x['f1_score'])

# Use the best split to retrain the model and evaluate
X_train_final = X.loc[best_split['train_index']]
X_test_final = X.loc[best_split['test_index']]
y_train_final = y.loc[best_split['train_index']]
y_test_final = y.loc[best_split['test_index']]


final_model = RandomForestClassifier(**final_params, random_state=42)
final_model.fit(X_train_final, y_train_final)

y_probs_final = final_model.predict_proba(X_test_final)[:, 1]

# Compute ROC-AUC
roc_auc = roc_auc_score(y_test_final, y_probs_final)
print(f'ROC-AUC: {roc_auc}')

# Compute PR-AUC
pr_auc = average_precision_score(y_test_final, y_probs_final)
print(f'PR-AUC: {pr_auc}')

# Calculate the ROC curve points
fpr, tpr, _ = roc_curve(y_test_final, y_probs_final)

# Calculate the PR curve points
precision, recall, thresholds = precision_recall_curve(y_test_final, y_probs_final)
f1_scores = 2*recall*precision / (recall + precision)

best_index = np.argmax(f1_scores)
best_threshold = thresholds[best_index]


# Plot the ROC curve
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=0.5)  # Dashed diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')

# Plot the PR curve
plt.subplot(1, 2, 2)
plt.ylim([0,1])
plt.axhline(y=89/130, color='black', linestyle='--', linewidth=0.5)
plt.plot(recall, precision, label=f'PR curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')

plt.tight_layout()
plt.show()


#%%
##^ confusion matrix ##
from sklearn.metrics import confusion_matrix  # Import confusion_matrix

# Predict on the test set# Convert probabilities to binary predictions based on the threshold
best_threshold = 0.58
y_pred_threshold = (y_probs_final >= best_threshold).astype(int)

# Compute the confusion matrix using the true labels and your predictions
cm = confusion_matrix(y_test, y_pred_threshold)

# Plotting the confusion matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cbar=True, cmap='Blues',)
tick_marks = np.arange(2) + 0.5
plt.xticks(tick_marks, ['False', 'True'], ha='center')
plt.yticks(tick_marks, ['False', 'True'], va='center')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion Matrix at Best Threshold')
plt.show()

# %%

