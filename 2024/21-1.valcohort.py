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

# %%
dis = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_TU.txt', sep='\t', index_col=0)
val = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_TU.txt', sep='\t', index_col=0)
valgene = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_geneTPM.txt', sep='\t', index_col=0)
disinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo_new.txt', sep='\t', index_col=0)
valinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_validation_clinicalinfo.txt', sep='\t', index_col=0)
valinfo = valinfo.loc[list(val.columns),:]

dis_y = pd.DataFrame(disinfo.iloc[0::2,2])
dis_y.replace({0: 'IR', 1: 'AR'}, inplace=True)
valinfo['finalresponse'] = 'x'
valinfo.loc[(valinfo['ongoing']==1) | (valinfo['ongoing']==2) | (valinfo['ongoing']==4),'finalresponse'] = 'R'
valinfo.loc[(valinfo['ongoing']==0) & (valinfo['response']==1), 'finalresponse'] = 'AR'
valinfo.loc[(valinfo['ongoing']==3) & (valinfo['response']==1), 'finalresponse'] = 'AR'
valinfo.loc[(valinfo['ongoing']==0) & (valinfo['response']==0), 'finalresponse'] = 'IR'
valinfo.loc[(valinfo['ongoing']==3) & (valinfo['response']==0), 'finalresponse'] = 'IR'


# val_y = pd.DataFrame(valinfo.iloc[:,-1])

# merged = pd.merge(dis_y, val_y,left_index=True, right_index=True, how='outer')
# # Function to update finalresponse with response values if they are not NaN
# def update_finalresponse(row):
#     if not pd.isna(row['response']):
#         return row['response']
#     else:
#         return row['finalresponse']

# # Apply the function to the DataFrame
# merged['finalresponse'] = merged.apply(update_finalresponse, axis=1)

# dis_pre = dis.iloc[:,1::2]
# tu = pd.concat([dis_pre,val],axis=1)
# tu.columns = tu.columns.str.replace('-bfD', '', regex=False)
# cols = list(merged.index)
# tu = tu[cols]
# tu = tu.iloc[:-2,:]

# tu.loc['class'] = list(merged['finalresponse'])

#tu.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/dis_val_merged_TU.txt', sep='\t', index=True)

# %%
###^^^^^^^^^^^ 3 class classification ######################
df = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/dis_val_merged_TU.txt', sep='\t', index_col=0)

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

import pickle

# Reactome2022_featurelist.pkl / BP2018_featurelist.pkl
with open('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/featureselection/Reactome2022_featurelist.pkl', 'rb') as file:
    featurelist_reactome = pickle.load(file)

with open('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/featureselection/BP2018_featurelist.pkl', 'rb') as file:
    featurelist_bp = pickle.load(file)

set1 = set(featurelist_bp).union(set(featurelist_reactome))
set2 = set(featurelist_bp).intersection(set(featurelist_reactome))


###
features = set1
###


####*** with only 116 samples ####
df = val
y = pd.DataFrame(valinfo['finalresponse'])
#y = pd.DataFrame(df.iloc[-1,:])
y.columns=['class']
y['class'] = y['class'].map({'R': 0, 'IR': 1, 'AR': 2})
y = y['class']
X = df.loc[features,:]
X = X.T

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, precision_recall_curve, roc_curve, auc, f1_score, make_scorer
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import make_scorer, matthews_corrcoef, f1_score, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import label_binarize


# Calculate feature importances using the whole dataset
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
rf.fit(X, y)

# Get feature importances
feature_importances = rf.feature_importances_
feature_names = X.columns

# Plot feature importances

plt.figure(figsize=(6, 15))
plt.barh(feature_names, feature_importances)
plt.xlabel('Feature Importance')
plt.title('Feature Importances from Random Forest')
plt.show()

# Select features based on mean importance
mean_importance = np.mean(feature_importances)
selected_features = feature_names[feature_importances > mean_importance]
print("Selected features:", selected_features)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.3, random_state=42, stratify=y)

from imblearn.combine import SMOTETomek
###^^ oversampling ############################################################
# smote_tomek = SMOTETomek(random_state=42)
# X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)
##########^####################################################################

from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy={0: 100, 1: 80, 2: 40}, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define the Random Forest model
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# Define a parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100],  # Fewer trees
    'max_depth': [5, 10, 15],   # Limit tree depth to make trees more general
    'min_samples_split': [10, 20],  # Require more samples to split
    'min_samples_leaf': [5, 10],    # Require more samples in each leaf node
    'bootstrap': [True]             # Bootstrap to reduce variance
}

# Set up GridSearchCV with weighted F1 score
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

# Define MCC scorer and F1-Weighted scorer
mcc_scorer = make_scorer(matthews_corrcoef)
f1_scorer = make_scorer(f1_score, average='weighted')

# Compare metrics by setting up different grid searches
scorers = {
    'balanced_accuracy': make_scorer(balanced_accuracy_score),
    'f1_weighted': f1_scorer,
    'mcc': mcc_scorer
}

# Choose the scoring metric you want to focus on
chosen_scorer = 'mcc'  # Could be 'f1_weighted' or 'balanced_accuracy'

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=cv,
    scoring=scorers[chosen_scorer],
    verbose=2
)

# Fit GridSearchCV on the training data
grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best parameters and the best model
best_rf = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

# Predict on the test set
y_pred = best_rf.predict(X_test)
y_pred_prob = best_rf.predict_proba(X_test)

# Evaluate the model using chosen metrics
print("Test set MCC:", matthews_corrcoef(y_test, y_pred))
print("Test set F1-Weighted:", f1_score(y_test, y_pred, average='weighted'))
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred))


# Binarize the output for multiclass ROC curve calculation
y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))  # Convert y_test to binary format

# Step 1: Calculate the ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

n_classes = y_test_binarized.shape[1]  # Number of classes

for i in range(n_classes):
    fpr[i], tpr[i], thresholds = roc_curve(y_test_binarized[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Step 2: Plot ROC curve for each class
plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('One-vs-Rest ROC Curve for Each Class')
plt.legend(loc='lower right')
plt.show()

# Step 3: Choose the threshold for each class based on ROC curve
optimal_thresholds = []
for i in range(n_classes):
    optimal_idx = np.argmax(tpr[i] - fpr[i])  # Find the optimal index

    # Make sure that the optimal index is within bounds of the thresholds array
    if optimal_idx >= len(thresholds):
        optimal_idx = len(thresholds) - 1  # Correct the index if it's out of bounds

    optimal_thresholds.append(thresholds[optimal_idx])
    print(f"Optimal threshold for class {i}: {optimal_thresholds[i]}")


from sklearn.metrics import precision_recall_curve, auc

# Calculate precision-recall curve for each class
precision = dict()
recall = dict()
pr_auc = dict()

for i in range(n_classes):
    precision[i], recall[i], thresholds = precision_recall_curve(y_test_binarized[:, i], y_pred_prob[:, i])
    pr_auc[i] = auc(recall[i], precision[i])

# Plot Precision-Recall curve for each class
plt.figure()
for i in range(n_classes):
    plt.plot(recall[i], precision[i], label=f'Class {i} (area = {pr_auc[i]:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Each Class')
plt.legend(loc='lower left')
plt.show()

# Optional: Find the optimal threshold based on Precision-Recall curve
# optimal_thresholds = []
# for i in range(n_classes):
#     optimal_idx = np.argmax(tpr[i] - fpr[i])  # Find the optimal index

#     # Make sure that the optimal index is within bounds of the thresholds array
#     if optimal_idx >= len(thresholds):
#         optimal_idx = len(thresholds) - 1  # Correct the index if it's out of bounds

#     optimal_thresholds.append(thresholds[optimal_idx])
#     print(f"Optimal threshold for class {i}: {optimal_thresholds[i]}")


y_pred_custom = np.zeros_like(y_test)

# Step 2: Iterate over each class and apply the optimal threshold for each class
for i in range(len(np.unique(y_test))):  # Iterate over each class
    class_probabilities = y_pred_prob[:, i]  # Get predicted probabilities for class i
    y_pred_custom[class_probabilities > optimal_thresholds[i]] = i  # Apply custom threshold

# Step 3: Create the confusion matrix using the custom predictions
cm = confusion_matrix(y_test, y_pred_custom)

# Step 4: Visualize the confusion matrix
ConfusionMatrixDisplay(cm).plot()
plt.title('Confusion Matrix with Custom Thresholds')
plt.show()


#%%
# Step 1: Get the predicted probabilities for the training set
y_train_pred_prob = best_rf.predict_proba(X_train_resampled)

# Step 2: Predict the class using the default threshold of 0.5
# For multiclass classification, we use argmax to select the class with the highest probability
y_train_pred = np.argmax(y_train_pred_prob, axis=1)

# Step 3: Generate the confusion matrix for the training set with default threshold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm_train = confusion_matrix(y_train_resampled, y_train_pred)

# Step 4: Plot the confusion matrix
ConfusionMatrixDisplay(cm_train).plot()
plt.title('Confusion Matrix for Training Set with Threshold 0.5')
plt.show()

# Step 5: Evaluate other performance metrics (optional)
from sklearn.metrics import classification_report

print("Classification Report:\n", classification_report(y_train_resampled, y_train_pred))






#%%
#######^^^^^^^^^^^ XGBOOST###########################
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, f1_score, classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder


# Step 1: Fit XGBoost to the whole dataset for feature importance
le = LabelEncoder()
y = le.fit_transform(y)
X = X.apply(pd.to_numeric, errors='coerce')

xgb_model = xgb.XGBClassifier(random_state=77, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X, y)

# Get feature importances from XGBoost
feature_importances = xgb_model.feature_importances_
feature_names = X.columns

# Plot feature importances
plt.figure(figsize=(6, 15))
plt.barh(feature_names, feature_importances)
plt.xlabel('Feature Importance')
plt.title('Feature Importances from XGBoost')
plt.show()

# Step 2: Select features based on mean importance
mean_importance = np.mean(feature_importances)
selected_features = feature_names[feature_importances > mean_importance]
print("Selected features:", selected_features)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.3, random_state=77, stratify=y)

# Step 4: Apply SMOTE for resampling
#smote = SMOTE(sampling_strategy={0: 70, 1: 50, 2: 30}, random_state=42)
#X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
from imblearn.over_sampling import ADASYN

adasyn = ADASYN(sampling_strategy='auto', random_state=42)  # Automatically adjust weights
X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)

# Step 5: Define the XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

# Step 6: Define a **simpler parameter grid** for hyperparameter tuning
param_grid = {
    'n_estimators': [50],  # Reduce the number of trees to simplify the model
    'max_depth': [3, 5],   # Further reduce the depth to force the model to generalize
    'learning_rate': [0.05],  # Decrease the learning rate to make smaller adjustments
    'subsample': [0.7],    # Reduce subsample to make the model train on more diverse data
    'colsample_bytree': [0.7],  # Reduce feature sampling for each tree
    'reg_lambda': [10, 20],  # Increase L2 regularization
    'reg_alpha': [1, 5],     # Further increase L1 regularization
    'min_child_weight': [10, 20]  # Increase to force trees to generalize
}

# Use **StratifiedKFold** instead of repeated KFold to save time
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Use RandomizedSearchCV instead of GridSearchCV to sample from the parameter grid
grid_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    cv=cv,
    scoring='f1_weighted',  # Or use 'balanced_accuracy' or 'mcc' as needed
    verbose=2,
    n_iter=5  # Limit the number of parameter combinations evaluated
)

# Step 7: Fit RandomizedSearchCV on the training data with early stopping
grid_search.fit(X_train_resampled, y_train_resampled, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)


# Get the best parameters and the best model
best_xgb = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

# Step 8: Predict on the test set
y_pred = best_xgb.predict(X_test)
y_pred_prob = best_xgb.predict_proba(X_test)

# Step 9: Evaluate the model using chosen metrics
print("Test set MCC:", matthews_corrcoef(y_test, y_pred))
print("Test set F1-Weighted:", f1_score(y_test, y_pred, average='weighted'))
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))

# Classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 10: Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.title('Confusion Matrix')
plt.show()

# Step 11: Calculate ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

n_classes = y_test_binarized.shape[1]  # Number of classes

for i in range(n_classes):
    fpr[i], tpr[i], thresholds = roc_curve(y_test_binarized[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve for each class
plt.figure()
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('One-vs-Rest ROC Curve for Each Class')
plt.legend(loc='lower right')
plt.show()

# Step 12: Choose the threshold for each class based on ROC curve
optimal_thresholds = []
for i in range(n_classes):
    optimal_idx = np.argmax(tpr[i] - fpr[i])  # Find the optimal index

    # Make sure that the optimal index is within bounds of the thresholds array
    if optimal_idx >= len(thresholds):
        optimal_idx = len(thresholds) - 1  # Correct the index if it's out of bounds

    optimal_thresholds.append(thresholds[optimal_idx])
    print(f"Optimal threshold for class {i}: {optimal_thresholds[i]}")

# Step 13: Calculate Precision-Recall curve for each class
precision = dict()
recall = dict()
pr_auc = dict()

for i in range(n_classes):
    precision[i], recall[i], thresholds = precision_recall_curve(y_test_binarized[:, i], y_pred_prob[:, i])
    pr_auc[i] = auc(recall[i], precision[i])

# Plot Precision-Recall curve for each class
plt.figure()
for i in range(n_classes):
    plt.plot(recall[i], precision[i], label=f'Class {i} (area = {pr_auc[i]:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Each Class')
plt.legend(loc='lower left')
plt.show()

# Step 14: Apply the optimal threshold and create the confusion matrix
y_pred_custom = np.zeros_like(y_test)

# Iterate over each class and apply the optimal threshold for each class
for i in range(n_classes):  # Iterate over each class
    class_probabilities = y_pred_prob[:, i]  # Get predicted probabilities for class i
    y_pred_custom[class_probabilities > optimal_thresholds[i]] = i  # Apply custom threshold

# Create the confusion matrix using the custom predictions
cm = confusion_matrix(y_test, y_pred_custom)

# Visualize the confusion matrix
ConfusionMatrixDisplay(cm).plot()
plt.title('Confusion Matrix with Custom Thresholds')
plt.show()


#%%
# Step 1: Predict on the training set
y_train_pred = best_xgb.predict(X_train_resampled)
y_train_pred_prob = best_xgb.predict_proba(X_train_resampled)

# Step 2: Evaluate the model using chosen metrics on the training set
print("Train set MCC:", matthews_corrcoef(y_train_resampled, y_train_pred))
print("Train set F1-Weighted:", f1_score(y_train_resampled, y_train_pred, average='weighted'))
print("Train set Balanced Accuracy:", balanced_accuracy_score(y_train_resampled, y_train_pred))

# Classification report for the training set
print("Classification Report (Train Set):\n", classification_report(y_train_resampled, y_train_pred))

# Step 3: Plot the confusion matrix for the training set
cm_train = confusion_matrix(y_train_resampled, y_train_pred)
ConfusionMatrixDisplay(cm_train).plot()
plt.title('Confusion Matrix for Train Set')
plt.show()

# Step 4: Plot ROC curve for each class in the training set
y_train_binarized = label_binarize(y_train_resampled, classes=np.unique(y_train_resampled))

fpr_train = dict()
tpr_train = dict()
roc_auc_train = dict()

for i in range(n_classes):
    fpr_train[i], tpr_train[i], thresholds_train = roc_curve(y_train_binarized[:, i], y_train_pred_prob[:, i])
    roc_auc_train[i] = auc(fpr_train[i], tpr_train[i])

plt.figure()
for i in range(n_classes):
    plt.plot(fpr_train[i], tpr_train[i], label=f'Class {i} (area = {roc_auc_train[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('One-vs-Rest ROC Curve for Each Class (Train Set)')
plt.legend(loc='lower right')
plt.show()

# Step 5: Plot Precision-Recall curve for each class in the training set
precision_train = dict()
recall_train = dict()
pr_auc_train = dict()

for i in range(n_classes):
    precision_train[i], recall_train[i], _ = precision_recall_curve(y_train_binarized[:, i], y_train_pred_prob[:, i])
    pr_auc_train[i] = auc(recall_train[i], precision_train[i])

plt.figure()
for i in range(n_classes):
    plt.plot(recall_train[i], precision_train[i], label=f'Class {i} (area = {pr_auc_train[i]:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Each Class (Train Set)')
plt.legend(loc='lower left')
plt.show()

# Optional: Create a confusion matrix with custom thresholds for the training set
y_train_pred_custom = np.zeros_like(y_train_resampled)

# Apply optimal thresholds on the training set
for i in range(n_classes):
    class_probabilities_train = y_train_pred_prob[:, i]
    y_train_pred_custom[class_probabilities_train > optimal_thresholds[i]] = i

# Create the confusion matrix using the custom predictions for the training set
cm_train_custom = confusion_matrix(y_train_resampled, y_train_pred_custom)

# Plot confusion matrix with custom thresholds for the training set
ConfusionMatrixDisplay(cm_train_custom).plot()
plt.title('Confusion Matrix with Custom Thresholds (Train Set)')
plt.show()



# %%
########^^^ only IR AR pre^^^^^^^^^^^^^^^^^^^^^^^^^
y = pd.DataFrame(df.iloc[-1,:])
y.columns=['class']
y['class'] = y['class'].map({'R': 0, 'IR': 1, 'AR': 2})
y = y[(y['class']==1) | (y['class']==2)]
y['class'] = y['class'].map({1:0, 2:1})
y = y['class']
rows = y.index
X = df.loc[features,rows]
X = X.T

# Initialize lists to store metrics and feature importances
f1_scores = []
roc_aucs = []
pr_aucs = []
accs = []
feature_importance_list = []

# Calculate feature importances using the whole dataset
rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

# Get feature importances
feature_importances = rf.feature_importances_
feature_names = X.columns

# Plot feature importances
plt.figure(figsize=(6, 15))
plt.barh(feature_names, feature_importances)
plt.xlabel('Feature Importance')
plt.title('Feature Importances from Random Forest')
plt.show()

# Select features based on mean importance
mean_importance = np.mean(feature_importances)
selected_features = feature_names[feature_importances > mean_importance]
print("Selected features:", selected_features)


# Define a parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [10, 100, 500, 1000],
    'max_depth': [None, 10, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Set up GridSearchCV with weighted F1 score
f1_scorer = make_scorer(f1_score, average='weighted')

# Initialize figures for ROC and PR curves
plt.figure(figsize=(6, 6))
plt.figure(figsize=(6, 6))

# Loop over 100 random seeds
for seed in range(1):
    # Split the data into training and testing sets using selected features
    X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.3, random_state=seed, stratify=y)
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=seed),
        param_grid=param_grid,
        cv=5,
        scoring=f1_scorer,
        verbose=0
    )
    
    # Fit GridSearchCV on the selected features
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters
    best_rf = grid_search.best_estimator_
    
    # Predict on the test set
    y_pred = best_rf.predict(X_test)
    y_pred_prob = best_rf.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    f1 = f1_score(y_test, y_pred, average='binary')
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

# Plot mean ROC curve
plt.figure(1)
mean_fpr, mean_tpr, _ = roc_curve(y_test, y_pred_prob)
plt.plot(mean_fpr, mean_tpr, color='#EB5353', label=f'Mean ROC curve (area = {mean_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey', alpha=0.7)  # Add diagonal line x=y
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves over 100 Random Seeds')
plt.legend(loc='lower right')
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202408_analysis/preTU_model/ROCcurve_100iter.pdf', bbox_inches="tight")


# Plot mean Precision-Recall curve
plt.figure(2)
mean_precision, mean_recall, _ = precision_recall_curve(y_test, y_pred_prob)
plt.plot(mean_recall, mean_precision, color='#187498', label=f'Mean PR curve (area = {mean_pr_auc:.2f})')
plt.axhline(y=0.575, color='grey', linestyle='--', alpha=0.7)  # Add horizontal line y=0.575
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves over 100 Random Seeds')
plt.legend(loc='lower left')
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202408_analysis/preTU_model/PRcurve_100iter.pdf', bbox_inches="tight")



# Save feature importance to a DataFrame
feature_importance_df = pd.DataFrame(feature_importance_list, columns=feature_names)
print(feature_importance_df)

# Print mean metrics
print(f'Mean F1 Score: {mean_f1}')
print(f'Mean ROC-AUC: {mean_roc_auc}')
print(f'Mean PR-AUC: {mean_pr_auc}')


# %%
#########^^^^^^^^^^^^^^^^^^^^^^ validation cohort heatmap #################

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler

dfinput = X.T
#dfinput = df.iloc[:-1,:].dropna()

# Step 1: Scale the data
scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(dfinput), index=dfinput.index, columns=dfinput.columns)

# Step 2: Winsorize the data (example: limit to the 5th and 95th percentiles)
#winsorized_df = dfinput.apply(lambda x: winsorize(x, limits=[0.05, 0.05]), axis=0)


# Define the clustering method (e.g., 'single', 'complete', 'average', 'centroid', 'median', 'ward')
method = 'ward'  # Change this to use a different clustering method

# Perform hierarchical clustering
linkage_matrix = linkage(scaled_df, method=method)  # Transpose to cluster samples

# Map binary values to colors
lut = {0: 'green', 1: 'blue', 2:'red'}
col_colors = y.map(lut)

plt.figure(figsize=(15,5))

# Create a clustermap without the color bar legend and y-axis label
clustermap = sns.clustermap(
    scaled_df, 
    row_cluster=True, 
    col_cluster=True, 
    col_linkage=linkage_matrix, 
    cmap='coolwarm', 
    col_colors=col_colors,
    yticklabels=False,
    xticklabels=False,
    standard_scale=1,
    #z_score=1
    
    #cbar_pos=None  # Remove the color bar legend
)

# Customize the plot to remove the y-axis label
clustermap.ax_heatmap.set_ylabel('')

# Show the plot
plt.show()

# %%
#####^^^^ validation cohort boxplot check ###############
#* 1. 137 features (initial feature set) ##
interfeatures = ['ENST00000391831.1-AKT1S1', 'ENST00000482687.1-APTX',
    'ENST00000404251.1-BAZ1B', 'MSTRG.49834.247-BRIP1',
    'ENST00000310447.5-CASP2',
    'ENST00000310232.6-CCDC13', 'ENST00000349780.4-CDK5RAP2',
    'MSTRG.98879.3-CENPW', 'ENST00000368328.4-CENPW',
    'ENST00000308987.5-CKS1B',
    'MSTRG.14904.57-CTBP2', 'MSTRG.14904.52-CTBP2',
    'ENST00000373573.3-HDAC8', 'ENST00000444376.2-HNRNPU',
    'ENST00000353801.3-HSP90AB1',
    'ENST00000392915.1-MAP3K19', 'ENST00000375321.1-MAP3K8',
    'MSTRG.11829.3-MAP3K8',
    'MSTRG.55994.22-MZT2A', 'MSTRG.18587.4-OTUB1',
    'MSTRG.92133.3-PAK4', 
    'MSTRG.1704.189-RCC1', 'ENST00000412779.2-RNASE1',
    'MSTRG.52876.4-RPS27A',
    'MSTRG.51515.3-SMC6', 'ENST00000453269.2-TAF6',
    'MSTRG.30902.1-TFDP1',
    'MSTRG.74585.63-TFDP2', 'ENST00000395958.2-YWHAZ',
    'ENST00000395405.1-ZWINT']

onlydeltamodel = ['ENST00000302759.6-BUB1',
'ENST00000399334.3-CEP152',
'ENST00000375440.4-CUL4A',
'ENST00000342628.2-FOXM1',
'MSTRG.49465.142-GRB2',
'ENST00000454366.1-GTSE1',
'ENST00000373135.3-L3MBTL1',
'ENST00000382948.5-MBD1',
'MSTRG.39427.204-NRG4',
'ENST00000222254.8-PIK3R2',
'ENST00000261207.5-PPP1R12A',
'ENST00000498508.2-PROX1',
'ENST00000575111.1-RNF167',
'ENST00000254942.3-TERF2',
'ENST00000380122.5-TXLNG',
'ENST00000511817.1-XRCC4']

onlypremodel = ['ENST00000457016.1-APC',
'MSTRG.97711.85-CCND3',
'MSTRG.41549.4-CSNK2A2',
'MSTRG.25406.5-DYRK2',
'ENST00000545134.1-HELB',
'ENST00000344908.5-MAP3K2',
'MSTRG.12458.189-MCL1',
'ENST00000461554.1-PCBP4',
'MSTRG.65506.82-PIGU',
'MSTRG.7469.43-PIK3R3',
'MSTRG.126473.45-POM121',
'ENST00000358514.4-PSMB10',
'ENST00000520452.1-PTTG1',
'MSTRG.1704.180-RCC1',
'MSTRG.31105.1-RNASE1',
'MSTRG.17113.1-SFR1',
'ENST00000402989.1-SMC6',
'ENST00000321464.5-ZBTB38']

featurelist = interfeatures

########
plotdf = X.copy()
#plotdf = plotdf[featurelist]
plotdf = plotdf.apply(pd.to_numeric, errors='coerce')
########

plotdf['mean pre TU'] = plotdf.mean(axis=1)

plotdf['response'] = y.tolist()
plotdf['response'].replace({0: 'R', 1: 'IR', 2:'AR'}, inplace=True)

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
            showfliers=True, order=['R','IR','AR'], 
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
            order=['R','IR', 'AR'], 
            color='#7D7C7C',  # Color of the points
            size=4,         # Size of the points
            jitter=True,    # Adds some jitter to avoid overlapping
            alpha=0.8,
            ax=ax)


from statannot import add_stat_annotation


add_stat_annotation(ax, data=plotdf, x='response', y='mean pre TU',
                    box_pairs=[('IR','AR'),('R','AR'),('R','IR')], 
                    order = ['R','IR','AR'], 
                    comparisons_correction=None,
                    test='Mann-Whitney', text_format='simple', loc='inside', fontsize=12
                    )

#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202408_analysis/plot/137_initial_validation_116samples_boxplot.pdf', bbox_inches='tight', dpi=300)
plt.show()


# %%
#####** expHRD gene check #######
df = val #### use only 116 samples
y = valinfo['finalresponse']
with open('/home/jiye/jiye/copycomparison/gDUTresearch/202408_analysis/expHRD_comparison/expHRD_356_genes.txt', 'r') as file:
    explist = file.read()
explist = explist.strip("'").split("', '")
ensg2gene = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/annotation/ensg2symbol.dict', sep='\t', header=None)
ensg2gene.columns = ['ensg','genesymbol']
ensg2gene['ensg_parsed'] = ensg2gene['ensg'].str.split(".",1).str[0]

explistdf = pd.DataFrame({'ensg_parsed':explist})

merged = pd.merge(explistdf, ensg2gene, how='inner', left_on='ensg_parsed', right_on='ensg_parsed')

exp_X = df.T
exp_X = exp_X.loc[:, exp_X.columns.str.contains('|'.join(merged['genesymbol']))]


# %%
majorlist = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majortrans = list(majorlist[majorlist['type']=='major']['gene_ENST'])
major_exp_X = exp_X.loc[:,exp_X.columns.isin(majortrans)]

# %%
plotdf = major_exp_X
plotdf['mean pre TU'] = plotdf.mean(axis=1)

plotdf['response'] = y.tolist()
plotdf['response'].replace({0: 'R', 1: 'IR', 2:'AR'}, inplace=True)

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
            showfliers=True, order=['R','IR','AR'], 
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
            order=['R','IR', 'AR'], 
            color='#7D7C7C',  # Color of the points
            size=4,         # Size of the points
            jitter=True,    # Adds some jitter to avoid overlapping
            alpha=0.8,
            ax=ax)


from statannot import add_stat_annotation


add_stat_annotation(ax, data=plotdf, x='response', y='mean pre TU',
                    box_pairs=[('IR','AR'),('R','AR'),('R','IR')], 
                    order = ['R','IR','AR'], 
                    comparisons_correction=None,
                    test='Mann-Whitney', text_format='simple', loc='inside', fontsize=12
                    )

#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202408_analysis/plot/expHRD_majorTU_1177_boxplot.pdf', bbox_inches='tight', dpi=300)
plt.show()

# %%
####** AR DUT + major + expHRD #######
ARdut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t')
DUTlist = list(ARdut.loc[(ARdut['p_value']<0.05) & (ARdut['log2FC']>1.5), 'gene_ENST'])

plotdf = major_exp_X.loc[:,major_exp_X.columns.isin(DUTlist)]
plotdf['mean pre TU'] = plotdf.mean(axis=1)

plotdf['response'] = y.tolist()
plotdf['response'].replace({0: 'R', 1: 'IR', 2:'AR'}, inplace=True)

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
            showfliers=True, order=['R','IR','AR'], 
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
            order=['R','IR', 'AR'], 
            color='#7D7C7C',  # Color of the points
            size=4,         # Size of the points
            jitter=True,    # Adds some jitter to avoid overlapping
            alpha=0.8,
            ax=ax)


from statannot import add_stat_annotation


add_stat_annotation(ax, data=plotdf, x='response', y='mean pre TU',
                    box_pairs=[('IR','AR'),('R','AR'),('R','IR')], 
                    order = ['R','IR','AR'], 
                    comparisons_correction=None,
                    test='Mann-Whitney', text_format='simple', loc='inside', fontsize=12
                    )

#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202408_analysis/plot/expHRD_majorTU_DUT_44_116samples_boxplot.pdf', bbox_inches='tight', dpi=300)
plt.show()

# %%
# %%
####** ExpHRD gene check #######

plotdf = valgene.loc[valgene.index.isin(list(merged['ensg'])),:]
plotdf = plotdf.T
plotdf['mean pre gene exp'] = plotdf.mean(axis=1)
plotdf['mean pre gene exp'] = np.log2(plotdf['mean pre gene exp']+1)
y = valinfo['finalresponse']
plotdf['response'] = y.tolist()
plotdf['response'].replace({0: 'R', 1: 'IR', 2:'AR'}, inplace=True)

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


ax = sns.boxplot(y='mean pre gene exp', x='response', data=plotdf, 
            showfliers=True, order=['R','IR','AR'], 
            #color='#FFD92E'
            palette='vlag'
            #meanprops = {"marker":"o", "color":"blue","markeredgecolor": "blue", "markersize":"10"}
            )
#ax.set_ylim([0,0.01])

#ax.set_xticklabels(['IR','AR'])
#plt.title(transname)
sns.despine()

sns.stripplot(y='mean pre gene exp', x='response', data=plotdf, 
            order=['R','IR', 'AR'], 
            color='#7D7C7C',  # Color of the points
            size=4,         # Size of the points
            jitter=True,    # Adds some jitter to avoid overlapping
            alpha=0.8,
            ax=ax)

ax.set_ylabel('mean log2(TPM+1)')

from statannot import add_stat_annotation


add_stat_annotation(ax, data=plotdf, x='response', y='mean pre gene exp',
                    box_pairs=[('IR','AR'),('R','AR'),('R','IR')], 
                    order = ['R','IR','AR'], 
                    comparisons_correction=None,
                    test='Mann-Whitney', text_format='simple', loc='inside', fontsize=12
                    )

plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202408_analysis/plot/expHRD_geneexp_116samples_boxplot.pdf', bbox_inches='tight', dpi=300)
plt.show()

# %%
###^^^ expHRD random forest check #########
X = major_exp_X.iloc[:,:-2]
y = np.array(valinfo['finalresponse'].replace({'R':0, 'IR':1, 'AR':2}))
# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, precision_recall_curve, roc_curve, auc, f1_score, make_scorer
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from imblearn.over_sampling import SMOTE



# Scorers
f1_scorer = make_scorer(f1_score, average='weighted')
balanced_accuracy_scorer = make_scorer(balanced_accuracy_score)
mcc_scorer = make_scorer(matthews_corrcoef)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Use StratifiedKFold for cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define the pipeline with feature selection and RandomForest
pipeline = Pipeline([
    ('feature_selection', SelectFromModel(RandomForestClassifier(random_state=42))),
    ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# Correct the parameter grid to target the 'classifier' step in the pipeline
param_grid = {
    'classifier__n_estimators': [10, 100, 1000],
    'classifier__max_depth': [None, 10, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__bootstrap': [True, False]
}

# Define GridSearchCV
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,  # The updated parameter grid
    cv=cv,
    scoring=balanced_accuracy_scorer,
    verbose=2
)

# Fit the pipeline with cross-validation
grid_search.fit(X_resampled, y_resampled)

# Now predict on the test set
y_pred = grid_search.best_estimator_.predict(X_test)
y_pred_prob = grid_search.best_estimator_.predict_proba(X_test)

# Confusion matrix for multiclass classification
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()

# AUC-ROC for multiclass classification
roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
print("AUC-ROC score:", roc_auc)

# %%
# Access the fitted SelectFromModel step in the best pipeline
sfm = grid_search.best_estimator_.named_steps['feature_selection']

# Get the support (True for selected features, False for non-selected)
selected_features = sfm.get_support()

# Get the feature names from X_train based on the selected features
selected_feature_names = list(X_train.columns[selected_features])
print("Selected features:", len(selected_feature_names))

# %%
import xgboost as xgb

xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, random_state=42)
from sklearn.preprocessing import LabelEncoder

# If y contains string labels, convert to integers
le = LabelEncoder()
y = le.fit_transform(y)
X = X.apply(pd.to_numeric, errors='coerce')

from imblearn.over_sampling import SMOTE

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Perform SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train XGBoost on the balanced dataset
xgb_model = xgb.XGBClassifier(objective='multi:softprob', num_class=3, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_resampled, y_resampled)


# Predict on the original test set
y_pred_prob = xgb_model.predict_proba(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)



param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, cv=5, scoring='balanced_accuracy', verbose=1,)
grid_search_xgb.fit(X_train, y_train)


#Evaluate the model
y_pred_xgb = grid_search_xgb.best_estimator_.predict(X_test)


y_pred = np.argmax(y_pred_xgb, axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

# %%
