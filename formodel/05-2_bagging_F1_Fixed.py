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
from skopt import BayesSearchCV
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from sklearn.model_selection import cross_val_score
from xgboost import plot_importance, XGBClassifier
from sklearn import svm, model_selection
from scipy.stats import ttest_1samp, wilcoxon, ttest_ind, mannwhitneyu, \
    fisher_exact, binom_test
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score,\
    roc_curve, auc, roc_auc_score, confusion_matrix, precision_recall_curve, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold,\
    train_test_split
warnings.simplefilter(action='ignore', category=Warning)
from functools import partial               # 함수 변수 고정
# import mkl
# mkl.set_num_threads(15) # Set the number of threads

"""
FROM: /home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/merged_TU/whole_TU/XGBoost/100BOP_tuning_importance.txt
Feature selection step 에서 100개의 tuning models 로 부터 얻어진 importance 값, 이를 기준으로 끝에서부터 하나씩 빼면서 개별 모델을 학습
개별 모델 학습시 features 에 의한 영향만을 보기 위해서 sample split, model, 그리고 10 cross-validation seed 는 모두 고정
04_230109_BOP.py 에서 수정
Filtering 이후 전체 featrues: 230109_ex_pre-post_From_term_GENE_pos_sig_response_input.txt >> old features 와 비교 목적
Old features 개수 맞춤: 230109_ex_pre-post_From_term_GENE_Filtered_pos_sig_response_input.txt >> To develop the best model
"""
## Input processing & parameter tuning
random_state=int(sys.argv[1]) # Fixed seed
DIR = "/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/merged_TU/whole_TU/XGBoost/"
OUT = DIR+"bagging_features/"
file_id = "230109_ex_pre-post_From_term_GENE_pos_sig_response_input.txt"
merged = pd.read_csv(
                    DIR
                    +file_id,
                    sep="\t", index_col="Unnamed: 0").T
features_info = pd.read_csv(
                            OUT+
                            "wo_best_cut/Fixed_"+str(random_state)+"_params_bagging_result_"+file_id,
                            sep="\t"
                            )
features_info = features_info.iloc[0,7:]
## Target selection (Bagging features)
mean_IP=pd.read_csv(
                    DIR+"100BOP_tuning_importance.txt",
                    sep="\t", index_col="Unnamed: 0"
                    )
mean_IP.loc["mean"] = mean_IP.mean()
mean_IP = mean_IP.loc[["mean"]].T.sort_values(by="mean", ascending=False)
for pos in range(mean_IP.shape[0]):
    pos=pos*-1
    if pos == 0:
        target = mean_IP.iloc[:,:].index.tolist()
    else:
        target = mean_IP.iloc[:pos,:].index.tolist()
    n_features = len(target)    # number of used features

    ## Set the input data
    train_x, test_x, train_y, test_y = train_test_split(
                                                        merged[target].astype(float), 
                                                        merged["group"].astype(float), 
                                                        test_size=0.2, 
                                                        random_state=random_state,
                                                        stratify=merged["group"]
                                                        )

    def get_clf_eval(y_test, pred_proba, pred, y_train, train_proba):
        """ 모델 평가지표 """
        pre,recall,threshold = precision_recall_curve(y_train, train_proba)
        fscore = (2 * (pre * recall)) / (pre + recall)
        ix = np.argmax(fscore)
        best_cutoff = threshold[ix]
        ## Train data 기반 best cut off 선정

        pred_best = pd.DataFrame(pred_proba, index=y_test.index)[0].apply(lambda x : 1 if x > best_cutoff else 0)
        print(pred_best)
        print(y_test)
        acc = accuracy_score(y_test, pred_best)
        f1 = f1_score(y_test, pred_best)
        roc_auc = roc_auc_score(y_test, pred_proba)
        hrd_precision, hrd_recall, hrd_the = precision_recall_curve(
            y_test, pred_proba)
        pr_auc = auc(hrd_recall, hrd_precision)
            
        return roc_auc, pr_auc, best_cutoff, acc, f1

    ## Estimation of the tuning model
    best_params = {
                    "n_estimators": [int(features_info[0])],
                    "learning_rate": [features_info[1]],
                    "max_depth": [int(features_info[2])],
                    "subsample": [features_info[3]],
                    "gamma": [features_info[4]],
                    "colsample_bytree": [features_info[5]]
                    }

    xgb = XGBClassifier(
                        random_state=random_state,
                        objective='binary:logistic', 
                        eval_metric="logloss",
                        use_label_encoder=False, 
                        n_jobs=4
                        )
    kfold = StratifiedKFold(n_splits=10, random_state=random_state, shuffle=True)   # Fixed the CV seed

    xgb_grid = RandomizedSearchCV(
                                xgb, best_params, cv=kfold, 
                                n_jobs=2, 
                                random_state=random_state,
                                scoring="f1", verbose=1, 
                                n_iter=1, 
                                return_train_score=True
                                )   # Fixed the model seed

    ## Training by the best parameters
    xgb_grid.fit(train_x, train_y)
    roc,pr,cutoff,acc,f1 = get_clf_eval(
                                        test_y, 
                                        xgb_grid.predict_proba(test_x)[:,1],
                                        xgb_grid.predict(test_x),
                                        train_y,
                                        xgb_grid.predict_proba(train_x)[:,1]
                                        )
    report = pd.DataFrame(columns=["ROC AUC", "PR AUC", "cutoff",
                                   "#Class", "mean test score",
                                   "mean train score", "inde test score",
                                   "best_acc", "best_F1"], 
                                   index=[n_features])
    report.iloc[0,0] = roc
    report.iloc[0,1] = pr
    report.iloc[0,2] = cutoff
    report.iloc[0,3] = len(set(xgb_grid.predict_proba(test_x)[:,1]))
    cv_result = pd.DataFrame(xgb_grid.cv_results_)
    
    ## indenpendent set 에 대한 검증, 이건 세트가 고정이라 best params 로 한 것임
    cv_result["test_score_added_by_hyeongu"] = xgb_grid.score(
                                                            test_x, test_y
                                                            )

    cv_result = cv_result.sort_values(by="mean_test_score")[[
                                                            "mean_test_score",
                                                            "mean_train_score",
                                                            "test_score_added_by_hyeongu"
                                                            ]]
    best = cv_result.iloc[-1,:] # 기준: mean test score
    report.iloc[0,4] = best[0]  # mean test score (from 10 cv)
    report.iloc[0,5] = best[1]  # mean train score (from 10 cv)
    report.iloc[0,6] = best[2]  # Unseen data (test set)
    report.iloc[0,7] = acc
    report.iloc[0,8] = f1

    ## Result check
    params = pd.DataFrame.from_dict(best_params, orient="columns")
    params.index = [n_features]
    df_rep = pd.concat([report, params], axis=1)

    test_y, xgb_grid.predict_proba(test_x)[:,1]

    if pos == 0:
        df_rep.to_csv(
            OUT+"w_best_cut/train_cut_off/F1_Fixed_"+str(random_state)+"_params_bagging_result_"+file_id, 
            mode="a", sep="\t"
            )
    else:
        df_rep.to_csv(
            OUT+"w_best_cut/train_cut_off/F1_Fixed_"+str(random_state)+"_params_bagging_result_"+file_id, 
            mode="a", sep="\t", header=None
            )
    break
# %%
