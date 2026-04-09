#! this code is.....
"""
Multivariate Regression! for predicting AR / IR group
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
import matplotlib.cm as cm
from matplotlib.pyplot import gcf

# %%
duts = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202312_analysis/finalDUTlist.txt',sep='\t')
parpi = list(duts['PARPi-derived_DUT'])
baseline = list(duts['Baseline_DUT'])
baseline = [x for x in baseline if type(x) is str]
whole = list(set(parpi+baseline))

#%%
intergenes = {'DDB1', 'POLD4', 'LIG1', 'POLR2B', 'SLC30A9', 'RPS27A', 'RFC5', 'USP45', 'UBC', 'USP7', 'MNAT1', 'POLK', 'BRIP1', 'COPS4', 'POLR2A', 'TCEA1', 'CUL4A', 'XAB2', 'CDK7', 'XPC', 'TP53', 'ERCC2', 'PNKP', 'GPS1', 'MMS19', 'COMMD1', 'COPS2', 'UVSSA', 'CCNH', 'RFC1'}

inter = [item for item in whole for gene in intergenes if gene in item]

# %%
tu = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/modelinput/validation_TU.txt', sep='\t', index_col=0)
input = tu.loc[tu.index.isin(inter),:]
input.columns = input.columns.str.replace("T", "P")

sampledf = pd.DataFrame({'GID':input.columns})

sample = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/validation_clinicalinfo.txt', sep='\t')
sample = sample[['GID','group']]
new_row = {'GID':'SV-OV-P056', 
            'group':1.0}
sample = sample.append(new_row, ignore_index=True)

sampleinfo = pd.merge(sampledf,sample,how='inner',left_on='GID',right_on='GID')
sampleinfo.columns = ['GID','response']

input = input.drop('ENST00000502706.1-RFC1')

#%%
#^ color mapping
#colors_dict = { 'pre_R':"#4998FF", 'pre_NR':"#FF4949", 'post_R':"#209422", 'post_NR':"#A81CBD"}
colors_dict = {0.0:"#4998FF", 1.0:"#FF4949"}
col_colors = sampleinfo['response'].map(colors_dict)

tmpinput = X_top_45.T
input_zscore = tmpinput.apply(lambda x: (x-x.mean())/x.std(), axis = 1)

##^ draw seaborn clustermap
# methods = ['single', 'complete', 'average', 'weighted', 'ward'] #5
methods = ['ward']
metrics = ['cityblock', 'minkowski', 'seuclidean', 'cosine', 'correlation', 'hamming', 'jaccard', 'chebyshev', 'canberra', 'dice', 'rogerstanimoto', 'russellrao', 'sokalsneath'] #13

for i in range(1):
    
    method = 'ward'

    g = sns.clustermap(input_zscore, col_colors=[col_colors], cmap="RdBu_r" , figsize=(6,9),vmin=-2,vmax=2,center=0,
                    method=method, #single, complete, average, weighted, ward
                    metric='euclidean', #cityblock, minkowski, euclidean, cosine, correlation, hamming, jaccard, chebyshev, canberra, dice, rogerstanimoto, russellrao, sokalsneath
                    #z_score=0,
                    #vmax=1, vmin=0,
                    #standard_scale=0,
                        linewidths=0, xticklabels=False, yticklabels=False, col_cluster=True, row_cluster=True)


    for group in  sampleinfo['response'].unique():
        g.ax_col_dendrogram.bar(0, 0, color=colors_dict[group],
                                label=group, linewidth=0)

    l1 = g.ax_col_dendrogram.legend(title='response', loc="center", ncol=2, bbox_to_anchor=(0.57, 1.01), bbox_transform=gcf().transFigure)

    # for event in event_info['event'].unique():
    #     g.ax_row_dendrogram.bar(0, 0, color=colors_dict2[event], label=event, linewidth=0);

    # l2 = g.ax_row_dendrogram.legend(title='Dataset', loc="center", ncol=2, bbox_to_anchor=(0.57, 1.06), bbox_transform=gcf().transFigure)

    #g.cax.set_position([.15, .2, .03, .45])
    ax = g.ax_heatmap
    # ax.set_xlabel("Samples")
    ax.set_ylabel("")
#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/fig3c_feature_response_clustermap.pdf", bbox_inches="tight")


# %%
#############^^^^^^^ model ..? #########
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from xgboost import XGBClassifier 
import scipy.stats as stats
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.ensemble import BalancedRandomForestClassifier


X = input.T
y = sampleinfo['response']
y = 1 - y
y.index = X.index

#%%
# Split your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=77)

#%%
# Define the Random Forest model
clf = RandomForestClassifier()

# Define hyperparameters grid
param_grid = {
    'n_estimators': [100, 200, 300, 500, 1000],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'max_features': ['auto', 'sqrt', 'log2', 0.5, 0.8],
    'bootstrap': [True],
    'class_weight': ['balanced', 'balanced_subsample']
}

# Create F1 scorer
f1_scorer = make_scorer(f1_score)

# Define GridSearchCV
grid_search = GridSearchCV(clf, param_grid, scoring=f1_scorer, cv=5, n_jobs=1, verbose=2)
grid_search.fit(X_train, y_train)



mean_f1_scores = grid_search.cv_results_['mean_test_score']
std_f1_scores = grid_search.cv_results_['std_test_score']



#%%%
######* min std #####
min_std_index = np.argmin(std_f1_scores)
highlight_index = min_std_index

def jitter(x, width=0.2):
    return x + np.random.rand(*x.shape) * width - width / 2

def plot_combined(data, title, nums, highlight_index):
    plt.figure(figsize=(4, 6))

    # Boxplot
    plt.boxplot(data, positions=[1], widths=0.6, showfliers=False)

    # Scatter points with jitter for x-coordinates
    y_vals = np.array(data) 
    x_vals = jitter(np.ones(y_vals.shape))
    plt.scatter(x_vals, y_vals, color='lightgray', s=50)

    # Highlight the point with minimum standard deviation
    plt.scatter(1, data[highlight_index], color='royalblue', s=50, zorder=5)

    plt.title(title)
    plt.tick_params(bottom = False, labelbottom=False) 
    #plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/val2/new_fig3d_F1score_'+str(nums)+'.pdf', )
    plt.show()

plot_combined(std_f1_scores, "Standard Deviation - F1 Scores", 1, min_std_index)
plot_combined(mean_f1_scores, "mean - F1 Scores",2,  min_std_index)



# Print hyperparameters of the model with the smallest std
best_params = grid_search.cv_results_['params'][min_std_index]
print("Hyperparameters of the model with the smallest std of F1 scores:", best_params)



#####* max mean f1 score
max_mean_f1_index = np.argmax(mean_f1_scores)
highlight_index = max_mean_f1_index

# You can use the same index for the std plot to highlight the same configuration

def plot_combined(data, title, nums, highlight_index, highlight_color='royalblue'):
    plt.figure(figsize=(4, 6))

    # Boxplot
    plt.boxplot(data, positions=[1], widths=0.6, showfliers=False)

    # Scatter points with jitter for x-coordinates
    y_vals = np.array(data)
    x_vals = jitter(np.ones(y_vals.shape))
    plt.scatter(x_vals, y_vals, color='lightgray', s=50)

    # Highlight the point with minimum standard deviation
    # Now also using it to highlight the maximum mean
    plt.scatter(1, data[highlight_index], color=highlight_color, s=50, zorder=5)

    plt.title(title)
    plt.tick_params(bottom=False, labelbottom=False)
    plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/val2/new_fig3d_F1score_'+str(nums)+'.pdf', )
    plt.show()

# Plotting the std F1 scores with the blue dot highlighting the configuration with the highest mean F1 score
plot_combined(std_f1_scores, "Standard Deviation - F1 Scores", 1, highlight_index)


# Plotting the mean F1 scores normally
plot_combined(mean_f1_scores, "Mean - F1 Scores", 2, highlight_index)

# Print hyperparameters of the model with the highest mean F1 score
best_params = grid_search.cv_results_['params'][max_mean_f1_index]
print("Hyperparameters of the model with the highest mean F1 score:", grid_search.cv_results_['params'][max_mean_f1_index])





















#%%
#################^^^ NEW PROTOCOL ######################
from scipy import interp
from scipy.interpolate import interp1d

# best_params = {'bootstrap': True, 'class_weight': 'balanced', 'max_depth': 30, 'max_features': 0.5, 'min_samples_leaf': 6, 'min_samples_split': 2, 'n_estimators': 100} 


# Initialize variables to store results
importances_list = []
mean_fpr = np.linspace(0, 1, 300)
tprs = []
aucs = []
precisions = []
recalls = []
whole_pred = []

# Fit the model and calculate metrics 100 times
for i in range(300):
    clf = RandomForestClassifier(**best_params)
    clf.fit(X_train, y_train)
    importances_list.append(clf.feature_importances_)
    
    # Predict probabilities and calculate ROC-AUC
    y_pred_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    aucs.append(auc(fpr, tpr))
    
    # Calculate Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    precisions.append(precision)
    recalls.append(recall)
    
    whole_pred.extend(y_pred_prob)
    

# Average feature importances and calculate standard deviation
importances_array = np.array(importances_list)
mean_importances = np.mean(importances_array, axis=0)
std_importances = np.std(importances_array, axis=0)


sorted_indices = np.argsort(mean_importances)[::-1]  # Sort features by importance
# Select the top 100 features
top_indices = sorted_indices[:100]

# Plot the feature importances for the top 100 features
plt.figure(figsize=(20, 10))  # You might want to adjust the figure size to fit 100 features
plt.title("Top 100 Average Feature Importances with Std Dev")
plt.bar(range(100), mean_importances[top_indices], color="r", yerr=std_importances[top_indices], align="center")
plt.xticks(range(100), X_train.columns[top_indices], rotation=90)
plt.tight_layout()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/val/new_fig3f_top100importance_new.pdf')
plt.show()

# Plotting ROC-AUC
roc_auc_std = np.std(aucs)

plt.figure(figsize=(6, 6))
plt.plot(mean_fpr, np.mean(tprs, axis=0), color='blue', lw=1.5, label=f'Mean ROC (AUC = {np.mean(aucs):.2f} $\pm$ {roc_auc_std:.2f} std)')
plt.fill_between(mean_fpr, np.mean(tprs, axis=0) - np.std(tprs, axis=0), np.mean(tprs, axis=0) + np.std(tprs, axis=0), color='blue', alpha=0.2)
plt.plot([0, 1], [0, 1], linestyle='--', lw=0.8, color='red', label='Chance', alpha=.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve')
plt.legend(loc="lower right")
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/val/new_fig3e_rocauc.pdf',bbox_inches="tight")
plt.show()


# Define a common recall threshold for interpolation
common_recall = np.linspace(0, 1, 300)
interpolated_precisions = []

for precision, recall in zip(precisions, recalls):
    # Ensure that the recall is sorted in ascending order
    sorted_indices = np.argsort(recall)
    sorted_recall = recall[sorted_indices]
    sorted_precision = precision[sorted_indices]

    # Use maximum accumulation to ensure precision is non-increasing
    accumulated_precision = np.maximum.accumulate(sorted_precision[::-1])[::-1]

    # Interpolate the accumulated precision values
    interp_func = interp1d(sorted_recall, accumulated_precision, bounds_error=False,
                        assume_sorted=True, fill_value=(accumulated_precision[-1], accumulated_precision[0]))
    interp_precision = interp_func(common_recall)
    interpolated_precisions.append(interp_precision)

# Compute the mean and standard deviation of the interpolated precisions
mean_precision = np.mean(interpolated_precisions, axis=0)
std_precision = np.std(interpolated_precisions, axis=0)

# Compute the average precision score (PR AUC)
average_precision_score = np.mean([auc(common_recall, p) for p in interpolated_precisions])
pr_auc_std = np.std([auc(common_recall, p) for p in interpolated_precisions])

# Plotting Precision-Recall curve
plt.figure(figsize=(6, 6))
plt.plot(common_recall, mean_precision, color='green', lw=2, label=f'Mean PR (AUC = {average_precision_score:.2f} $\pm$ {pr_auc_std:.2f})')
plt.fill_between(common_recall, mean_precision - std_precision, mean_precision + std_precision, color='green', alpha=0.2)
plt.axhline(y=0.684, color='red', linestyle='--', label='y = 0.684')
plt.ylim(bottom=0.35)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="best")
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/val/new_fig3e_prauc.pdf',bbox_inches="tight")
plt.show()

#%%
###^^ confusion matrix ###
whole_test = list(y_test)*300
best_threshold = 0.57
y_pred_best_threshold = (np.array(whole_pred) >= best_threshold).astype(int)

best_cm = confusion_matrix(whole_test, y_pred_best_threshold)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(np.eye(2), annot=best_cm, fmt='g', annot_kws={'size': 25},
            cmap=sns.color_palette(['#46900D5E', '#46900D'], as_cmap=True), cbar=False,
            yticklabels=['0', '1'], xticklabels=['0', '1'], ax=ax)
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.tick_params(labelsize=13, length=0)

ax.set_xlabel('Predicted Labels', size=13)
ax.set_ylabel('Actual Labels', size=13)

additional_texts = ['(True Negative)', '(False Positive)', '(False Negative)', '(True Positive)']
for text_elt, additional_text in zip(ax.texts, additional_texts):
    ax.text(*text_elt.get_position(), '\n' + additional_text, color=text_elt.get_color(),
            ha='center', va='top', size=13)
plt.tight_layout()
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/val/new_fig3k_confusion_whole.pdf', bbox_inches="tight")

plt.show()




#%%
# Now you can perform RFECV with the model using only the important features
important_features_mask = mean_importances > 0
X_train_important = X_train.iloc[:, important_features_mask]
X_test_important = X_test.iloc[:, important_features_mask]

min_features_to_select = 10  
# Minimum number of features to consider
rfecv = RFECV(
    estimator=clf,
    step=1,
    cv=StratifiedKFold(4),
    scoring="f1",
    min_features_to_select=min_features_to_select,
)
rfecv.fit(X_train_important, y_train)

print(f"Optimal number of features: {rfecv.n_features_}")

mean_scores = rfecv.cv_results_['mean_test_score']
std_dev = rfecv.cv_results_['std_test_score']
num_features = range(min_features_to_select, min_features_to_select + len(mean_scores))

plt.figure(figsize=(13,6))
plt.errorbar(num_features, mean_scores, yerr=std_dev, fmt='o-', color='g', lw=1.4, alpha=0.7, ms=4)
plt.title('RFECV Performance')
plt.xlabel('Number of Features Selected')
plt.ylabel('Cross-Validation Score (F1 Score)')
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/val/new_fig3g_rfecv.pdf', bbox_inches="tight")
plt.show()











#%%
####^^####################################################################
##################^^^^^^^^^^^^^ 45 features ##############################
###^^ ####################################################################
# Rank features by their importance scores
importance_ranked_indices = np.argsort(mean_importances)[::-1]

# Select the indices of the top 45 features
top_45_indices = importance_ranked_indices[:158]

# Use these indices to select the features from the original DataFrame
X_top_45 = X.iloc[:, top_45_indices]

#X_top_45.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202312_analysis/val_120_input.txt',sep='\t',index=True)
#X.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202312_analysis/val_200_input.txt',sep='\t',index=True)
X_top_45.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202312_analysis/val_nonzero_input.txt',sep='\t',index=True)

impdf = pd.DataFrame({'feature':X_train.columns[top_indices],'importance': mean_importances[top_indices] })
#impdf.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202312_analysis/val_top100_importance.txt',sep='\t',index=True)


impdf2= pd.DataFrame({'feature':X_train.columns[np.argsort(mean_importances)[::-1]],'importance': mean_importances[np.argsort(mean_importances)[::-1]] })
impdf2.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202312_analysis/new_val_200_importance.txt',sep='\t',index=True)


#%%
#########################################*************
# Split your data
X_train, X_test, y_train, y_test = train_test_split(X_top_45, y, test_size=0.25, stratify=y)
#%%
clf = RandomForestClassifier()

# Define hyperparameters grid
param_grid = {
    'n_estimators': [100, 200, 300, 500, 1000],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'max_features': ['auto', 'sqrt', 'log2', 0.5, 0.7, 0.9],
    'bootstrap': [True],
    'class_weight': ['balanced']
}
# Create F1 scorer
f1_scorer = make_scorer(f1_score)

# Define GridSearchCV
grid_search = GridSearchCV(clf, param_grid, scoring=f1_scorer, cv=5, n_jobs=1, verbose=2)
grid_search.fit(X_train, y_train)



mean_f1_scores = grid_search.cv_results_['mean_test_score']
std_f1_scores = grid_search.cv_results_['std_test_score']


#%%
######* min std #####
min_std_index = np.argmin(std_f1_scores)

def jitter(x, width=0.2):
    return x + np.random.rand(*x.shape) * width - width / 2

def plot_combined(data, title, nums, highlight_index):
    plt.figure(figsize=(4, 6))

    # Boxplot
    plt.boxplot(data, positions=[1], widths=0.6, showfliers=False)

    # Scatter points with jitter for x-coordinates
    y_vals = np.array(data) 
    x_vals = jitter(np.ones(y_vals.shape))
    plt.scatter(x_vals, y_vals, color='lightgray', s=50)

    # Highlight the point with minimum standard deviation
    plt.scatter(1, data[highlight_index], color='royalblue', s=50, zorder=5)

    plt.title(title)
    plt.tick_params(bottom = False, labelbottom=False) 
    plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/val/new_fig3h_lowest_std_F1score_'+str(nums)+'.pdf', )
    plt.show()
    
best_params = grid_search.cv_results_['params'][min_std_index]
print("Hyperparameters of the model with the lowest std F1 score:", best_params)

plot_combined(std_f1_scores, "Standard Deviation - F1 Scores", 1, min_std_index)
plot_combined(mean_f1_scores, "mean - F1 Scores", 2, min_std_index)



#####* max mean f1 score
max_mean_f1_index = np.argmax(mean_f1_scores)

# You can use the same index for the std plot to highlight the same configuration
highlight_index = max_mean_f1_index

def plot_combined(data, title, nums, highlight_index, highlight_color='royalblue'):
    plt.figure(figsize=(4, 6))

    # Boxplot
    plt.boxplot(data, positions=[1], widths=0.6, showfliers=False)

    # Scatter points with jitter for x-coordinates
    y_vals = np.array(data)
    x_vals = jitter(np.ones(y_vals.shape))
    plt.scatter(x_vals, y_vals, color='lightgray', s=50)

    # Highlight the point with minimum standard deviation
    # Now also using it to highlight the maximum mean
    plt.scatter(1, data[highlight_index], color=highlight_color, s=50, zorder=5)

    plt.title(title)
    plt.tick_params(bottom=False, labelbottom=False)
    plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/val/new_fig3h_highest_mean_F1score_'+str(nums)+'.pdf', )
    plt.show()

# Plotting the std F1 scores with the blue dot highlighting the configuration with the highest mean F1 score
plot_combined(std_f1_scores, "Standard Deviation - F1 Scores", 1, highlight_index)


# Plotting the mean F1 scores normally
plot_combined(mean_f1_scores, "Mean - F1 Scores", 2, highlight_index)

# Print hyperparameters of the model with the highest mean F1 score
#best_params = grid_search.cv_results_['params'][max_mean_f1_index]
print("Hyperparameters of the model with the highest mean F1 score:", best_params)























#%%
#################^^^ NEW PROTOCOL WITH SPLITTING THE X_TRAIN ######################
from scipy import interp
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix



# Initialize variables to store results
X_train_smaller, X_val, y_train_smaller, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
)

best_f1 = 0
best_clf = None
best_threshold = 0.5
importances_list = []
mean_fpr = np.linspace(0, 1, 300)
tprs = []
aucs = []
precisions = []
recalls = []


# Fit the model and calculate metrics 100 times
for i in range(300):
    current_clf = RandomForestClassifier(**best_params)
    current_clf.fit(X_train_smaller, y_train_smaller)
    importances_list.append(current_clf.feature_importances_)
    
    # Predict probabilities and calculate ROC-AUC
    y_val_prob = current_clf.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_val_prob)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    aucs.append(auc(fpr, tpr))
    
    # Calculate Precision-Recall
    precision, recall, _ = precision_recall_curve(y_val, y_val_prob)
    precisions.append(precision)
    recalls.append(recall)
    
    y_val_pred = current_clf.predict(X_val)
    current_f1 = f1_score(y_val, y_val_pred)
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_clf = current_clf

importances_array = np.array(importances_list)
mean_importances = np.mean(importances_array, axis=0)
std_importances = np.std(importances_array, axis=0)


sorted_indices = np.argsort(mean_importances)[::-1]  # Sort features by importance
# Select the top 100 features
top_indices = sorted_indices

# Plot the feature importances for the top 100 features
plt.figure(figsize=(20, 10))  # You might want to adjust the figure size to fit 100 features
plt.title("Average Feature Importances with Std Dev")
plt.bar(range(158), mean_importances[top_indices], color="r", yerr=std_importances[top_indices], align="center")
plt.xticks(range(158), X_train.columns[top_indices], rotation=90)
plt.tight_layout()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/val/new_fig3i_new_120importance.pdf')
plt.show()


# Plotting ROC-AUC
roc_auc_std = np.std(aucs)

plt.figure(figsize=(6, 6))
plt.plot(mean_fpr, np.mean(tprs, axis=0), color='blue', lw=1.5, label=f'Mean ROC (AUC = {np.mean(aucs):.2f} $\pm$ {roc_auc_std:.2f} std)')
plt.fill_between(mean_fpr, np.mean(tprs, axis=0) - np.std(tprs, axis=0), np.mean(tprs, axis=0) + np.std(tprs, axis=0), color='blue', alpha=0.2)
plt.plot([0, 1], [0, 1], linestyle='--', lw=0.8, color='red', label='Chance', alpha=.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve')
plt.legend(loc="lower right")
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/val/new_fig3j_new_rocauc.pdf',bbox_inches="tight")
plt.show()


# Define a common recall threshold for interpolation
common_recall = np.linspace(0, 1, 300)
interpolated_precisions = []

for precision, recall in zip(precisions, recalls):
    # Ensure that the recall is sorted in ascending order
    sorted_indices = np.argsort(recall)
    sorted_recall = recall[sorted_indices]
    sorted_precision = precision[sorted_indices]

    # Use maximum accumulation to ensure precision is non-increasing
    accumulated_precision = np.maximum.accumulate(sorted_precision[::-1])[::-1]

    # Interpolate the accumulated precision values
    interp_func = interp1d(sorted_recall, accumulated_precision, bounds_error=False,
                        assume_sorted=True, fill_value=(accumulated_precision[-1], accumulated_precision[0]))
    interp_precision = interp_func(common_recall)
    interpolated_precisions.append(interp_precision)

# Compute the mean and standard deviation of the interpolated precisions
mean_precision = np.mean(interpolated_precisions, axis=0)
std_precision = np.std(interpolated_precisions, axis=0)

# Compute the average precision score (PR AUC)
average_precision_score = np.mean([auc(common_recall, p) for p in interpolated_precisions])
pr_auc_std = np.std([auc(common_recall, p) for p in interpolated_precisions])

# Plotting Precision-Recall curve
plt.figure(figsize=(6, 6))
plt.plot(common_recall, mean_precision, color='green', lw=2, label=f'Mean PR (AUC = {average_precision_score:.2f} $\pm$ {pr_auc_std:.2f})')
plt.fill_between(common_recall, mean_precision - std_precision, mean_precision + std_precision, color='green', alpha=0.2)
plt.axhline(y=0.684, color='red', linestyle='--', label='y = 0.684')
plt.ylim(bottom=0.45)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="best")
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/val/new_fig3j_new_prauc.pdf',bbox_inches="tight")
plt.show()

#%%
#####^^^^^^^^^ Confusion Matrix ##############

y_scores = clf.predict_proba(X_test)[:, 1]

# Calculate precision and recall for various thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Calculate F1 score for each threshold
f1_scores = 2*precision*recall / (precision + recall)

# Find the index of the maximum F1 score
best_threshold_index = np.argmax(f1_scores)


# The best threshold value
best_threshold = thresholds[best_threshold_index]
best_threshold = 0.638
# Predict classes using the best threshold
y_pred_best_threshold = (y_scores >= best_threshold).astype(int)

# Compute the confusion matrix using predictions with the best threshold
best_cm = confusion_matrix(y_test, y_pred_best_threshold)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(np.eye(2), annot=best_cm, fmt='g', annot_kws={'size': 25},
            cmap=sns.color_palette(['#46900D5E', '#46900D'], as_cmap=True), cbar=False,
            yticklabels=['False', 'True'], xticklabels=['False', 'True'], ax=ax)
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')
ax.tick_params(labelsize=13, length=0)

ax.set_xlabel('Predicted Labels', size=13)
ax.set_ylabel('Actual Labels', size=13)

additional_texts = ['(False Positive)', '(False Negative)', '(True Negative)', '(True Positive)']
for text_elt, additional_text in zip(ax.texts, additional_texts):
    ax.text(*text_elt.get_position(), '\n' + additional_text, color=text_elt.get_color(),
            ha='center', va='top', size=13)
plt.tight_layout()
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/val/new_fig3k_confusion_whole.pdf', bbox_inches="tight")

plt.show()

#%%


#####* BalancedRandomForestClassifier ...??????
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

rf = BalancedRandomForestClassifier(n_estimators=200,
                                            random_state=42,
                                            n_jobs=1)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(rf, X_top_45, y, scoring='roc_auc', cv=cv, n_jobs=1)
#scores2 = cross_val_score(rf, X_top_45, y, scoring='pr_auc', cv=cv, n_jobs=1)
scores3 = cross_val_score(rf, X_top_45, y, scoring=f1_scorer, cv=cv, n_jobs=1)



rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_pred_prob = rf.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, marker='.', label='Logistic')
print(roc_auc_score(y_test, y_pred_prob))

lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_pred_prob)
lr_f1, lr_auc = f1_score(y_test, y_pred), auc(lr_recall, lr_precision)
plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))



#%%
########* Multivariate Regresion
import statsmodels.api as sm

newx = np.array(X_top_45)
newy = np.array(y)
newX = np.insert(newx, 0, np.ones((1,)), axis=1)
ols = sm.OLS(newy, newX).fit()
print(ols.summary())

#%%






















#%%























