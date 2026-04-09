#%%
# -*- coding: utf-8 -*-
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
import xgboost as xgb
import warnings
import random
import SET_arial
from skopt import BayesSearchCV
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from sklearn.model_selection import cross_val_score
from xgboost import plot_importance, XGBClassifier
from sklearn import svm, model_selection
from scipy.stats import ttest_1samp, wilcoxon, ttest_ind, mannwhitneyu, \
    fisher_exact, binom_test, pearsonr
from numpy import set_printoptions, interp
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score,\
    roc_curve, auc, roc_auc_score, confusion_matrix, precision_recall_curve, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold,\
    train_test_split
warnings.simplefilter(action='ignore', category=Warning)
from functools import partial
from scipy.interpolate import splrep, splev
# import mkl
# mkl.set_num_threads(15) # Set the number of threads

"""
ML_predictor/06-2-2-10cv_draw.py 에서 그리는 curve 를 26명 (1set) 으로 그리지 않고
26 * 10 으로 그려보기 위한 스크립트, std 를 측정할 때에도 10cv merged matrix 로 확인했으므로
"""

## Find lowest model From 07_comp_nested_cv.py
DIR = "/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/merged_TU/whole_TU/XGBoost/bagging_features/w_best_cut/train_cut_off/AUC_model/"
# result_seed = sys.argv[1]
result_seed = 12
ncv = pd.read_csv(
                DIR+str(result_seed)+"_10cv_result.txt",
                sep="\t", index_col="Unnamed: 0"
                )
resam_set = ncv.iloc[:,:10]
ncv = ncv.iloc[:,10:13]
ncv.columns = ["ROC AUC", "PR AUC", "mean FI"]
data = ncv.reset_index()
report = {}
for seed in data["index"].unique(): # 1-100 의 params set >> overfitted 제외돼서 85개
    report[seed] = []
    temp = data[data["index"]==seed]
    report[seed].append(temp.std()[1])
    report[seed].append(temp.std()[2])
    report[seed].append(temp.mean()[1])
    report[seed].append(temp.mean()[2])

report = pd.DataFrame.from_dict(report, orient="index", columns=["ROC_std","PR_std","ROC_mean","PR_mean"])
report = report.sort_values(by="ROC_std")
lowest_seed = report.index.tolist()[0]  # the lowest std params set
# print(report.sort_values(by="ROC_mean", ascending=False))
resam_set = resam_set.reset_index()
resam_set = resam_set[resam_set["index"]==lowest_seed].drop("index", axis=1)
resam_set = pd.read_csv(
                        "/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/merged_TU/whole_TU/XGBoost/bagging_features/w_best_cut/train_cut_off/AUC_model/"+
                        "top40_f1_FI.txt",
                        sep="\t", header=None
                        )
print(resam_set)
resam_set = resam_set[0].tolist()
# %%
## Input processing & parameter setting
pos=28
DIR = "/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/merged_TU/whole_TU/XGBoost/"
OUT = DIR+"bagging_features/"
file_id = "230109_ex_pre-post_From_term_GENE_pos_sig_response_input.txt"
merged = pd.read_csv(
                    DIR+file_id,
                    sep="\t", index_col="Unnamed: 0"
                    ).T
feature_data = pd.read_csv(
                            OUT+"w_best_cut/train_cut_off/AUC_model/80-150_roc_auc_features_params_bagging_result.txt",
                            sep="\t"
                            )   # From 06_Final_tuning.py
best_model = pd.DataFrame(feature_data.iloc[int(lowest_seed)-1,:]).T
data = feature_data[["ROC AUC","PR AUC","mean test score","mean train score","inde test score"]]
data = data[data["mean test score"] >= 0.8]
sns.set_style("whitegrid")
plt.figure(figsize=(3,6))
sns.stripplot(data=data, zorder=0,
            palette={"PR AUC":"#FF4B4B", "ROC AUC":"#FFACAC",\
            "mean test score":"#56AAFF", "mean train score":"#56AAFF",\
            "inde test score":"#56AAFF", "best_acc":"#A3F784",\
            "best_F1":"#FFFF02","TNR":"#FF963F"})
# sns.stripplot(data=data[data["PR AUC"] == data["PR AUC"].max()],
#             zorder=0, color="#000000", size=9
#             )
plt.ylim(0.5,1.0)
#### Customized plot ####
data.loc["median"] = data.median()
data.loc["3Q"] = data.quantile(.75)
data.loc["1Q"] = data.quantile(.25)
sns.boxplot(data=data.loc[["median"]], linewidth=2.5, zorder=1)
sns.boxplot(data=data.loc[["3Q"]], width=0.4, linewidth=2.5, zorder=1)
sns.boxplot(data=data.loc[["1Q"]], width=0.4, linewidth=2.5, zorder=1)
plt.xticks(rotation=90, fontsize=15)
plt.yticks(fontsize=14)
plt.savefig("/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/merged_TU/whole_TU/XGBoost/bagging_features/w_best_cut/train_cut_off/AUC_model/figure/"
            +"overfitted_model_filter.pdf",
            bbox_inches="tight")
# %%
line=0
random_state = int(best_model.iloc[line,0])
feature_info = best_model.iloc[line,11:]
# print(random_state)

## Target selection
mean_IP=pd.read_csv(
                    DIR+"100BOP_tuning_importance.txt",
                    sep="\t", index_col="Unnamed: 0"
                    )
mean_IP.loc["mean"] = mean_IP.mean()
mean_IP = mean_IP.loc[["mean"]].T.sort_values(by="mean", ascending=False)
target = mean_IP.iloc[:pos,:].index.tolist() # Best feature set

## Estimation of the tuning model
def get_clf_eval(y_test, pred_proba, y_train, train_proba):
    """ 모델 평가지표 """
    ## Find best threshold based on train set
    pre,recall,threshold = precision_recall_curve(y_train, train_proba)
    fpr, tpr, roc_thres = roc_curve(y_train, train_proba)
    fpr_t, tpr_t, roc_thres_t = roc_curve(y_test, pred_proba)
    diff_rate = []
    for a,b in zip(tpr, fpr):
        diff_rate.append(np.abs(a-(1-b)))
    
    mat = pd.merge(y_train, pd.DataFrame(train_proba, index=y_train.index),
                left_index=True, right_index=True)
    high_TF = []
    for a in roc_thres:
        mat["temp"] = mat[0].apply(lambda x : 1 if x > a else 0)
        sens = mat[(mat["group"]==1) & (mat["temp"]==1)].shape[0] / (mat[(mat["group"]==1) & (mat["temp"]==1)].shape[0] + mat[(mat["group"]==1) & (mat["temp"]==0)].shape[0])
        spec = mat[(mat["group"]==0) & (mat["temp"]==0)].shape[0] / (mat[(mat["group"]==0) & (mat["temp"]==0)].shape[0] + mat[(mat["group"]==0) & (mat["temp"]==1)].shape[0])
        sum_val = sens + spec
        high_TF.append(sum_val)

    efscore = np.sqrt(tpr * (1-fpr))
    ix = np.argmax(efscore)
    ix2 = np.argmin(diff_rate)
    ix3 = np.argmax(high_TF)
    fscore = (2 * (pre * recall)) / (pre + recall)
    ix4 = np.argmax(fscore)
    best_cutoff = threshold[ix4]
    # best_cutoff = roc_thres[ix]
    best_cutoff2 = roc_thres[ix2]
    best_cutoff3 = roc_thres[ix3]

    ## The estimation of model performance at best threshold
    pred_best = pd.DataFrame(pred_proba, index=y_test.index)[0].apply(lambda x : 1 if x > best_cutoff2 else 0)
    acc = accuracy_score(y_test, pred_best)
    f1 = f1_score(y_test, pred_best)
    roc_auc = roc_auc_score(y_test, pred_proba)
    hrd_precision, hrd_recall, hrd_the = precision_recall_curve(
        y_test, pred_proba)
    pr_auc = auc(hrd_recall, hrd_precision)
        
    return roc_auc, pr_auc, best_cutoff, acc, f1, best_cutoff2, best_cutoff3, fpr_t, tpr_t

best_params = {
                "n_estimators": [feature_info[0].astype(int)],
                "learning_rate": [feature_info[1]],
                "max_depth": [feature_info[2].astype(int)],
                "subsample": [feature_info[3]],
                "gamma": [feature_info[4]],
                "colsample_bytree": [feature_info[5]]
                }

xgb = XGBClassifier(
                    random_state=random_state,
                    objective='binary:logistic', 
                    eval_metric="logloss",
                    use_label_encoder=False, 
                    n_jobs=4
                    )

xgb_grid = RandomizedSearchCV(
                            xgb, best_params,
                            n_jobs=15,
                            random_state=random_state,
                            scoring="f1", 
                            verbose=0, 
                            n_iter=1, 
                            return_train_score=True
                            )   # Fixed the model seed

## Save 10 cv results From resampling sets
merged_sub = pd.DataFrame() # score
merged_FI = pd.DataFrame()  # feature importance
# %%
tprs = []
aucs = []
plt.close()
plt.figure(figsize=(6,6))
for sampling_seed in resam_set:
    cv_fpr = []
    cv_tpr = []
    sampling_seed = int(sampling_seed)
    ## Make 10-fold dataset
    train_x, test_x, train_y, test_y = train_test_split(
                                                        merged[target].astype(float), 
                                                        merged["group"].astype(float), 
                                                        test_size=0.2, 
                                                        random_state=sampling_seed,
                                                        stratify=merged["group"]
                                                        )
    ## Training by the best parameters
    xgb_grid.fit(train_x, train_y)
    roc,pr,cutoff,acc,f1,cutoff2,cutoff3 = get_clf_eval(
                                                        test_y, 
                                                        xgb_grid.predict_proba(test_x)[:,1],
                                                        train_y,
                                                        xgb_grid.predict_proba(train_x)[:,1]
                                                        )[:-2]

    ## Added 230331
    # tpr, fpr = get_clf_eval(  # Using it when I drawing PR curves
    fpr, tpr = get_clf_eval(
                            test_y, 
                            xgb_grid.predict_proba(test_x)[:,1],
                            train_y,
                            xgb_grid.predict_proba(train_x)[:,1]
                            )[-2:]
    
    cv_fpr = np.hstack([cv_fpr,fpr])
    cv_tpr = np.hstack([cv_tpr,tpr])
    cv_fpr.sort()
    cv_tpr.sort()
    aucs.append(roc)
    plt.plot(cv_fpr, cv_tpr, color='green', linewidth=1, alpha=0.1, label=None)
    mean_fpr = np.linspace(0, 1, 100)
    tprs.append(interp(mean_fpr, cv_fpr, cv_tpr))
        
## Added 230331
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, linewidth=2, alpha=0.8, color='green',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc))
plt.ylim(bottom=0.0)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.33,
                label=r'$\pm$ 1 std. dev.')
plt.xlabel('FPR', fontsize=14)
plt.ylabel('TPR', fontsize=14)
x = [0.0, 1.0]
plt.plot(x, x, linestyle='dashed', color='red', linewidth=2.0, label='random')
plt.legend(fontsize=10, loc='best')

plt.savefig(
            "/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/merged_TU/whole_TU/XGBoost/bagging_features/w_best_cut/train_cut_off/AUC_model/figure/"+
            "ver2_interp_plot.pdf", bbox_inches="tight"
            )
###############
# %%
