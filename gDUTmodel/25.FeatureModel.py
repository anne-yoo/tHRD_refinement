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
from statsmodels.api import Logistic

from xgboost import XGBClassifier 
import scipy.stats as stats

from skopt.space import Real, Integer


#%%
########^^^  input dataset ########
input = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/modelinput/premodel_input_combined.txt', sep='\t', index_col=0) 

### only pre samples for response classification ###
input = input[input.index.str.contains('-bfD')]
input = input.iloc[:,:-1]

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.csv', sep=',')
sampleinfo = sampleinfo[sampleinfo['sample_full'].str.contains('-bfD')]
interval = list(sampleinfo['interval'])
response = list(sampleinfo['response'])

count = pd.DataFrame(input.astype(bool).sum(axis=0))
count['transcript'] = count.index
count.columns = ['num','transcript']

filtered_count = count[count['num']>=15]
filtered_list = list(filtered_count['transcript'])
input = input.loc[:,filtered_list]


#%%
input.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/modelinput/premodel_input_combined_filtered_530.txt', sep='\t')

#%%

corrlist = []
pvallist = []
corr_origin = []
for i in range(0,len(input.columns)):
    tu = list(input.iloc[:,i])
    r, p = stats.pointbiserialr(tu,response)
    corrlist.append(r**2)
    pvallist.append(p)
    corr_origin.append(r)

corr_result = pd.DataFrame({
    'corr':corrlist,
    'pval':pvallist,
    'corr_origin':corr_origin
})
corr_result.index = input.columns

final = corr_result[corr_result['pval']<0.1]
final['transcript'] = final.index
final = final.sort_values(by=['corr'],  ascending=False)

final.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/featureselection/pointbiserial_143.txt', sep='\t')


#%%
###^^ correlation example plot ####
plt.figure(figsize=(6,6))
sns.set_style("white")

final = final.sort_values(by=['corr_origin'],  ascending=False)


for i in range(0,20):
    ext = final.iloc[i,3]
    example = list(input.loc[:,ext])

    corr_df = pd.DataFrame({
        'Transcript Usage':example,
        'PARPi interval':interval,
        'response':response
    })

    g = sns.lmplot(data=corr_df, x='PARPi interval', y='Transcript Usage',hue = 'response',sharey=False, sharex=False)

    def annotate(data, **kws):
        r, p = stats.pearsonr(data['PARPi interval'], data['Transcript Usage'])
        ax = plt.gca()
        ax.text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
                transform=ax.transAxes)
        
    g.map_dataframe(annotate)
    plt.tight_layout()
    plt.xlabel("PFS (days)")
    #plt.title('ENST00000584322.1-BRIP1')
    plt.title(final.iloc[i,3])

    plt.show()



#%%
######^^ check major transcripts##
major = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/final_202306_discovery_major_TU.txt', sep='\t', index_col=0)


majortrans = set(major.index)
translist = set(final['transcript'])













#%%

featurelist = list(final.index)

tu = pd.read_csv('/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/merged_TU/whole_TU/230105_whole_TU.txt', sep='\t', index_col=0)
clinical = pd.read_csv('/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/clinical_info_gHRD/processed_info_with_originalID.txt', sep='\t')
tu = tu.loc[featurelist, :]

tu = tu.drop(["SV-OV-T035-RP-RNA_M", "SV-OV-T048-RP-RNA_M", "SV-OV-T050-RP-RNA_M", "SV-OV-T055-RP-RNA_M",\
"SV-OV-T059-RNA_M", "SV-OV-T065-RP-RNA_M", "SV-OV-T067-RNA_M", "SV-OV-P069-RP_M",\
"SV-OV-P070-RP_M", "SV_OV_HRD_SV-OV-P079_RP_rneasy_WRS_RS_S7", "SV_OV_HRD_SV-OV-P080_RP_rneasy_WRS_RS_S8",\
"SV-OV-T082-RNA_M", "SV-OV-T087-P-RNA_M", "SV_OV_HRD_SV-OV-P099_RP_rneasy_WRS_M",\
"SV-OV-T105-P-RNA_M", "SV-OV-T107-P-RNA_M", "SV-OV-P131-RP-RNA_M", "SMC_OV_OVLB_SV-OV-P134_RF_Rmini_WRS_M",\
"SV_OV_HRD_SV-OV-P137_RP_rneasy_WRS_M", "SMC_OV_OVLB_SV-OV-P142_RF_Rmini_WRS_M",\
"SMC_OV_OVLB_SV-OV-P143_RF_Rmini_WRS_M", "SV_OV_HRD_SV-OV-P164_RP_rneasy_WRS_M",\
"SV_OV_HRD_SV-OV-P219_RP_rneasy_WRS_S_G1_S2"], axis=1)    # pre-post 를 제외하는 경우
non_pre_post = tu[["SV_OV_HRD_SV-OV-P155-bfD_RP_rneasy_WRS_M","SV_OV_HRD_SV-OV-P157-atD_RP_rneasy_WRS_M", "SV_OV_HRD_SV-OV-P251-atD_RP"]]
## 약간 애매하긴 하지만, 애초에 pair 가 없는데 ID 가 이렇게 생성된 것이므로 학습에 이용
## 그렇다면 QC fail 인 matching sample 은..?
tu = tu.T
tu = tu[(~tu.index.str.contains("atD")) & (~tu.index.str.contains("bfD"))]
tu = pd.merge(tu.T, non_pre_post, left_index=True, right_index=True).T

tu.index = tu.index.str.replace("SMC_OV_OVLB_","")
tu.index = tu.index.str.replace("SV_OV_HRD_","")
tu.index = tu.index.str[:10]
tu.index = tu.index.str.replace("P","T")
## Combine clinical information
group_info = pd.read_csv(
    "/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/clinical_info_gHRD/processed_info_with_originalID.txt", 
    sep="\t", index_col="GID")
group_info = group_info[["OM/OS", "ORR", "drug"]]
group_info.columns = ["OM/OS", "group", "drug"]
group_info["drug"] = group_info["drug"].str.replace("Olapairb","Olaparib")
group_info["drug"] = group_info["drug"].str.replace("Olapairib","Olaparib")
group_info["drug"] = group_info["drug"].str.replace("olaparib","Olaparib")
group_info = group_info.dropna()

## 약제별 구분
group_info = group_info.drop("drug", axis=1)
group_info["10ID"] = group_info.index.str.replace("F","T").str[:10]
group_info = group_info.drop_duplicates()
group_info = group_info.set_index("10ID")
merged = pd.merge(tu, group_info, left_index=True,
                right_index=True, how="inner")










#%%
##### ^^^^^^^^^ VALIDATION COHORT #######################
input = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/modelinput/validation_TU.txt', sep='\t', index_col=0)
forfeature = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/featureselection/correlation_62.txt', sep='\t', index_col=0)
featurelist = list(forfeature.index)
input=input.T
X = input.loc[:,featurelist]
y = input.iloc[:,-1]
y = [int(float(num_str)) for num_str in y]


#%%
#############************* MonteCarlo ########################

#%%
# Split your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randomint)

# Define the Random Forest model
clf = RandomForestClassifier()

# Define hyperparameters grid
param_grid = {
    'n_estimators': [10, 50, 100, 200, 500],
    'max_depth': [None, 10, 20, 30, 60, 80],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': ['True','False']
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

def plot_combined(data, title, highlight_index):
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
    plt.show()

plot_combined(std_f1_scores, "Standard Deviation - F1 Scores", min_std_index)
plot_combined(mean_f1_scores, "mean - F1 Scores", min_std_index)



# Print hyperparameters of the model with the smallest std
best_params = grid_search.cv_results_['params'][min_std_index]
print("Hyperparameters of the model with the smallest std of F1 scores:", best_params)



#%%
#####* max mean f1 score
max_mean_f1_index = np.argmax(mean_f1_scores)

# You can use the same index for the std plot to highlight the same configuration
highlight_index = max_mean_f1_index

def plot_combined(data, title, highlight_index, highlight_color='royalblue'):
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
    plt.show()

# Plotting the std F1 scores with the blue dot highlighting the configuration with the highest mean F1 score
plot_combined(std_f1_scores, "Standard Deviation - F1 Scores", highlight_index)

# Plotting the mean F1 scores normally
plot_combined(mean_f1_scores, "Mean - F1 Scores", highlight_index)

# Print hyperparameters of the model with the highest mean F1 score
best_params = grid_search.cv_results_['params'][max_mean_f1_index]
print("Hyperparameters of the model with the highest mean F1 score:", best_params)

#%%
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Train model with best parameters (from earlier)
clf = RandomForestClassifier(**best_params)
clf.fit(X_train, y_train)

# Predict probabilities
y_pred_prob = clf.predict_proba(X_test)[:,1]


# ROC-AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC Curve')
plt.legend(loc="lower right")

# Precision-Recall
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
average_precision = average_precision_score(y_test, y_pred_prob)

plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (area = %0.2f)' % average_precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")

plt.tight_layout()
plt.show()

#%%

######^^^^^^^^^^^^^^^ RFECV #################
clf.fit(X_train, y_train)
importances = clf.feature_importances_


# Get the feature names from the columns of your dataframe
feature_names = np.array(X_train.columns)

# Sort the feature importances in descending order and take the top 100
indices = np.argsort(importances)[::-1]



# Now plot the feature importances for the top 100 features
plt.figure(figsize=(15, 10))
plt.title("Top 50 Feature Importances")
plt.bar(range(len(indices)), importances[indices], color="r", align="center")
plt.xticks(range(len(indices)), feature_names[indices], rotation=90)
plt.xlim([-1, len(indices)])
plt.tight_layout()
plt.show()

#%%

# Select features with importance > 0.001
important_features_mask = importances > 0
X_train_important = X_train.loc[:, important_features_mask]
X_test_important = X_test.loc[:, important_features_mask]

min_features_to_select = 23  # Minimum number of features to consider
clf = RandomForestClassifier(**best_params)
cv = StratifiedKFold(5)

rfecv = RFECV(
    estimator=clf,
    step=5,
    cv=cv,
    scoring="f1",
    min_features_to_select=min_features_to_select,
    n_jobs=1,
)
rfecv.fit(X_train_important, y_train)

print(f"Optimal number of features: {rfecv.n_features_}")

n_scores = len(rfecv.cv_results_["mean_test_score"])



sns.set_style("whitegrid")
plt.figure(figsize=(10,6))
plt.xlabel("Number of features selected")
plt.ylabel("Mean F1")


plt.errorbar(
    range(min_features_to_select, n_scores*5 + min_features_to_select, 5),
    rfecv.cv_results_["mean_test_score"],
    yerr=rfecv.cv_results_["std_test_score"], color='green', fmt='o'
)
plt.plot(range(min_features_to_select, n_scores*5 + min_features_to_select, 5),
    rfecv.cv_results_["mean_test_score"], color='green')

plt.title("RFECV result")
plt.show()


















#%%%
###^^^^^^^^^^^^^^^^^^^^ LASSO regression with PARPi-derived DUTs ############################


########^^^ make input dataset ########
input = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/modelinput/premodel_input_DUT_onlyrepair.txt', sep='\t', index_col=0) 

### only pre samples for response classification ###
input = input[input.index.str.contains('-bfD')]


##### Last column: pre(0)/post(1) , second to last column: responder(1)/nonresponder(0)
X = input.iloc[:,:-1]
y = input.iloc[:,-1]

from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV, KFold

### Parameters to be tested on GridSearchCV
params = {"alpha": np.arange(0.00001, 1, 0.001)}  # Adjust the step size

# Number of Folds and adding the random state for replication
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initializing the Model
lasso = Lasso()

# GridSearchCV with model, params, and folds.
lasso_cv = GridSearchCV(lasso, param_grid=params, cv=kf)
lasso_cv.fit(X, y)

# Get the best alpha value
best_alpha = lasso_cv.best_params_['alpha']
print(f"Best Params {lasso_cv.best_params_}")

#%%
# Column names (assuming X is a DataFrame)
input = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/modelinput/premodel_input_DUT_onlyrepair.txt', sep='\t', index_col=0) 

### only pre samples for response classification ###
input = input[input.index.str.contains('-bfD')]


##### Last column: pre(0)/post(1) , second to last column: responder(1)/nonresponder(0)
X = input.iloc[:,:-1]
y = input.iloc[:,-1]

names = X.columns

# Calling the model with the best parameter
#lasso1 = Lasso(alpha=best_alpha)
lasso1 = Lasso(alpha=0.0005)

lasso1.fit(X, y)  # You can also use X_train, y_train if you have split the data

# Using np.abs() to make coefficients positive
lasso1_coef = np.abs(lasso1.coef_)


# Filtering the coefficients based on the threshold (greater than 0 in this case)
selected_coef = lasso1_coef[lasso1_coef > 0]
selected_names = np.array(names)[lasso1_coef > 0]

sorted_indices = np.argsort(-selected_coef)  # Negative sign for descending order
sorted_coef = selected_coef[sorted_indices]
sorted_names = selected_names[sorted_indices]

# Plotting the sorted Column Names and Importance of Columns
plt.figure(figsize=(16,5))
sns.set_style("white")

plt.bar(sorted_names, sorted_coef) 
plt.xticks(rotation=90)
plt.grid()
plt.title("Sorted Feature Importance Based on Lasso")
plt.xlabel("Selected Features")
plt.ylabel("Importance")

plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/derived_lasso.pdf", bbox_inches="tight")

plt.show()
#%%







#%%
###^^^^^^^^^^^^^^^^^^^^ LASSO regression with Baseline DUTs ############################


# ### only pre samples for response classification ###
# input2 = input[input['response']==1]
# input2['pre/post'] = 1
# input2.loc[input2.index.str.contains('-bfD'),'pre/post'] = 0


# ##### Last column: pre(0)/post(1) , second to last column: responder(1)/nonresponder(0)
# X = input2.iloc[:,:-2]
# y = input2.iloc[:,-1]



########^^^ input dataset ########
input = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/modelinput/premodel_input_DUT_baseline.txt', sep='\t', index_col=0) 

### only pre samples for response classification ###
input = input[input.index.str.contains('-bfD')]


##### Last column: pre(0)/post(1) , second to last column: responder(1)/nonresponder(0)
X = input.iloc[:,:-1]
y = input.iloc[:,-1]

from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV, KFold

### Parameters to be tested on GridSearchCV
params = {"alpha": np.arange(0.00001, 1, 0.001)}  # Adjust the step size

# Number of Folds and adding the random state for replication
kf = KFold(n_splits=5, shuffle=True, random_state=28)

# Initializing the Model
lasso = Lasso()

# GridSearchCV with model, params, and folds.
lasso_cv = GridSearchCV(lasso, param_grid=params, cv=kf)
lasso_cv.fit(X, y)

# Get the best alpha value
best_alpha = lasso_cv.best_params_['alpha']
print(f"Best Params {lasso_cv.best_params_}")

#%%

########^^^ input dataset ########
input = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/modelinput/premodel_input_DUT_baseline.txt', sep='\t', index_col=0) 

### only pre samples for response classification ###
input = input[input.index.str.contains('-bfD')]


##### Last column: pre(0)/post(1) , second to last column: responder(1)/nonresponder(0)
X = input.iloc[:,:-1]
y = input.iloc[:,-1]

# Column names (assuming X is a DataFrame)
names = X.columns

# Calling the model with the best parameter
lasso1 = Lasso(alpha=best_alpha)
lasso1 = Lasso(alpha=0.0005)
lasso1.fit(X, y)  # You can also use X_train, y_train if you have split the data

# Using np.abs() to make coefficients positive
lasso1_coef = np.abs(lasso1.coef_)

# Filtering the coefficients based on the threshold (greater than 0 in this case)
selected_coef = lasso1_coef[lasso1_coef > 0]
selected_names = np.array(names)[lasso1_coef > 0]

sorted_indices = np.argsort(-selected_coef)  # Negative sign for descending order
sorted_coef = selected_coef[sorted_indices]
sorted_names2 = selected_names[sorted_indices]

#%%
# Plotting the sorted Column Names and Importance of Columns
plt.figure(figsize=(20,5))
sns.set_style("white")

plt.bar(sorted_names2, sorted_coef) 
plt.xticks(rotation=90)
plt.grid()
plt.title("Sorted Feature Importance Based on Lasso")
plt.xlabel("Selected Features")
plt.ylabel("Importance")

plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/baseline_lasso.pdf", bbox_inches="tight")

plt.show()
#%%














finalfeatures = list(set(sorted_names).union(set(sorted_names2)))


#%%
###########^^^ Random Forest for New Features: discovery plot #########


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


finalfeatures = list(set(sorted_names).union(set(sorted_names2)))

input = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/modelinput/premodel_input_combined_filtered_530.txt', sep='\t', index_col=0) 
#input = input[input.index.str.contains('-bfD')]


y = input.iloc[:,-1]

input2 = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0) 
input2 = input2.T
X = input2[finalfeatures]
#X = X[X.index.str.contains('-bfD')]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest with GridSearchCV
param_grid = {
    'n_estimators': [10, 50, 100, 200, 400, 500],
    'max_depth': [None, 4, 6, 10, 20, 30],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='roc_auc', n_jobs=1)
grid_search.fit(X_train, y_train)

# Best estimator
best_clf = grid_search.best_estimator_

# Predict probabilities
y_pred_prob = best_clf.predict_proba(X_test)[:,1]

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# PR curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
pr_auc = average_precision_score(y_test, y_pred_prob)

plt.figure()
plt.plot(recall, precision, color='b', label='PR curve (area = %0.2f)' % pr_auc)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")
plt.show()

# Test set score
test_score = best_clf.score(X_test, y_test)
print(f"Test Set Score: {test_score:.4f}")


feature_names = X_train.columns.tolist()

importances = best_clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)  # Rotate feature names for better readability
plt.xlim([-1, len(importances)])
plt.tight_layout()
plt.show()




# %%
###########^^ Validation Cohort #########
input = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/modelinput/validation_TU.txt', sep='\t', index_col=0)
old = pd.read_csv('/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/merged_TU/whole_TU/XGBoost/old_model_featrues.txt', sep='\t', index_col=0)
oldfeatures = list(old.index)
input=input.T
X = input.loc[:,oldfeatures]
y = input.iloc[:,-1]
y = [int(float(num_str)) for num_str in y]
# %%
