#%%
# -*- coding: utf-8 -*-
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import sklearn
import statsmodels.stats.multitest as ssm
import xgboost as xgb
from numpy import set_printoptions
from scipy.stats import (binom_test, fisher_exact, mannwhitneyu, ttest_1samp,
                        ttest_ind, wilcoxon)
from sklearn import model_selection, svm
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                            mean_squared_error, precision_recall_curve,
                            precision_score, recall_score, roc_auc_score,
                            roc_curve)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                    StratifiedKFold, train_test_split)
from xgboost import XGBClassifier, plot_importance

"""
transcript tpm 을 통한 gHRD 예측 모델 학습,
gradient boosting algorithm by xgboost
"""

import random

## Input processing
start = 99
while start < 100:
    random_state = random.randint(1,9999)
    random_state = 59
    # file_id = "feature_matrix_main_tu_filter.txt"
    # file_id = "feature_matrix_top1_main.txt"
    # file_id = "feature_matrix_after_down_main.txt"
    file_id = "feature_matrix_after_up_main.txt"

    DIR = "/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/merged_TU/whole_TU/"
    OUT = DIR+"XGBoost/"
    if os.path.exists(OUT) == False:
        os.mkdir(OUT)

    tu = pd.read_csv(
        OUT+file_id, sep="\t", index_col="gene_ENST")
    tu = tu.T
    tu.index = tu.index.str.replace("SMC_OV_OVLB_","")
    tu.index = tu.index.str.replace("SV_OV_HRD_","")
    tu.index = tu.index.str.replace("P","T")
    print("Used features :", tu.shape[1])

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
    # unseened = group_info[~group_info["drug"].str.contains("Ola")]  # Non-Olaparib
    # unseened = unseened.drop("drug", axis=1)
    # group_info = group_info[group_info["drug"].str.contains("Ola")]

    group_info = group_info.drop("drug", axis=1)
    group_info["10ID"] = group_info.index.str.replace("F","T").str[:10]
    group_info = group_info.drop_duplicates()
    group_info = group_info.set_index("10ID")
    merged = pd.merge(tu, group_info, left_index=True,
                    right_index=True, how="inner")
    
    # print(merged["group"].value_counts())
    # unseened_res = merged[merged["group"]==1].sample(n=26, random_state=random_state)
    # unseened_nr = merged[merged["group"]==0].sample(n=14, random_state=random_state)
    # unseened_df = pd.concat([unseened_res, unseened_nr])    # 원본 비율 맞춰서
    # unseened_df = merged.sample(n=40, random_state=random_state)
    # unseened_df = pd.merge(tu, unseened, left_index=True,\
    #         right_index=True, how="inner")

    # merged = merged.reindex(
    #                         list(set(merged.index.tolist()) - 
    #                         set(unseened_df.index.tolist()))
    #                         )   # Remove unseened sample from training + test set

    ## AMBITION 추가
    # clincal: processed_info_with_originalID.txt1
    # TU: /home/hyeongu/DATA5/hyeongu/AMBITION_copy/AMBITION/align/2nd_pass/OV_gtf_quantified/processed
    #     whole TU 계산 중

    print("Used samples :", merged.shape[0])
    # print("Unseened samples :", unseened_df.shape[0])
    print("-"*40)
    merged.to_csv(
            OUT+"add_clinical_"+file_id,
            sep="\t")

    train_x, test_x, train_y, test_y = train_test_split(
                                                        merged.iloc[:,:-2], 
                                                        merged.iloc[:,-1],
                                                        test_size=0.2, 
                                                        random_state=random_state,
                                                        stratify=merged.iloc[:,-1])

    ## XGBoost setting
    def get_clf_eval(true_y, pred_proba, pred, sw):
        """ 모델 평가지표 """
        accuracy = accuracy_score(true_y, pred)
        precision = precision_score(true_y, pred)
        recall = recall_score(true_y, pred)
        f1 = f1_score(true_y, pred)
        roc_auc = roc_auc_score(true_y, pred_proba)
        hrd_precision, hrd_recall, hrd_the = precision_recall_curve(
            true_y, pred_proba)
        pr_auc = auc(hrd_recall, hrd_precision)
        
        if sw == "on":
            print("\n",
                "ROC_auc: {0:.4f}, PR_auc: {1:.4f}"
                .format(roc_auc, pr_auc),\
                "ACC: {0:.4f}, Prec: {1:.4f}, Recal: {2:.4f}, F1: {3:.4f}"
                .format(accuracy, precision, recall, f1))
            
        return roc_auc, pr_auc, f1


    def sk_wrapper(random_state):
        xgb_sk = XGBClassifier(
                            random_state=random_state,
                            objective='binary:logistic', eval_metric="error",
                            use_label_encoder=False, n_jobs=4
                            )
    
        ## Hyper-parameter tuning
        # xgb_param_grid = {
        #                 "n_estimators": [int(x) for x in np.linspace(start=10, stop=100, num=10)],
        #                 "learning_rate": [0.001,0.005,0.01,0.02,0.03,0.04,0.05,0.06],    # [0-1] 클수록 overfitting
        #                 "max_depth": [2,3],
        #                 "subsample": [0.2,0.3,0.4,0.5,0.6,0.7,0.8], # [0-1] 각 Tree의 관측 데이터 샘플링 비율
        #                 "gamma": [4.5,5.0,5.5,6.0,6.5,7.0,7.5], # 분할을 수행하는데 필요한 최소 손실 감소를 지정, 클수록 overfitting 제어
        #                 "colsample_bytree": [0.2,0.3,0.4,0.5,0.6,0.7]   # 각 트리마다의 feature 샘플링 비율, 
        #                 }
        xgb_param_grid = {
                        "n_estimators": [79],
                        "learning_rate": [0.001],    # [0-1] 클수록 overfitting
                        "max_depth": [10],
                        "subsample": [0.2999776615654719], # [0-1] 각 Tree의 관측 데이터 샘플링 비율
                        "gamma": [1.0], # 분할을 수행하는데 필요한 최소 손실 감소를 지정, 클수록 overfitting 제어
                        "colsample_bytree": [0.25704377944573176]   # 각 트리마다의 feature 샘플링 비율, 
                        }
        kfold = StratifiedKFold(n_splits=10, random_state=random_state)

        ## Do learning
        xgb_grid = RandomizedSearchCV(
            xgb_sk, xgb_param_grid, cv=kfold, n_jobs=30, random_state=random_state,
            scoring="f1", verbose=1, n_iter=150, return_train_score=True
        )

        xgb_grid.fit(train_x, train_y)
        sk_pred_prob = pd.DataFrame(
                                    xgb_grid.predict_proba(train_x)[:,1]
                                    )
        sk_pred_prob.index = train_x.index
        if os.path.exists(OUT+"pred_result/") == False:
            os.mkdir(OUT+"pred_result/")

        ## Cross validation
        report = pd.DataFrame(columns=["ROC AUC", "PR AUC", "f1", "mean test score", \
                "mean train score", "inde test score", "seed"], index=[random_state])
        roc,pr,f1 = get_clf_eval(
                                test_y, 
                                xgb_grid.predict_proba(test_x)[:,1],
                                xgb_grid.predict(test_x),
                                "off"
                                )
        report.iloc[0,0] = roc
        report.iloc[0,1] = pr
        report.iloc[0,2] = f1
        cv_result = pd.DataFrame(xgb_grid.cv_results_)
        cv_result["test_score_added_by_hyeongu"] = xgb_grid.score(
            test_x, test_y)    # indenpendent set 에 대한 검증, 이건 세트가 고정이라 best params 로 한 것임
        
        cv_result = cv_result.sort_values(by="mean_test_score")[["mean_test_score","mean_train_score","test_score_added_by_hyeongu"]]
        best = cv_result.iloc[-1,:]
        report.iloc[0,3] = best[0]
        report.iloc[0,4] = best[1]
        report.iloc[0,5] = best[2]
        print(xgb_grid.best_params_)
        print("-"*40)
        # roc,pr,f1 = get_clf_eval(
        #                         unseened_df.iloc[:,-1],
        #                         xgb_grid.predict_proba(unseened_df.iloc[:,:-2])[:,1],
        #                         xgb_grid.predict(unseened_df.iloc[:,:-2]),"off"
        #                         )
        # report.iloc[0,6] = roc
        # report.iloc[0,7] = pr
        # report.iloc[0,8] = f1
        report.iloc[0,6] = random_state

        if len(set(xgb_grid.predict_proba(test_x)[:,1])) > 5:    
            print(report)
            report.to_csv(
                OUT+"/cv_result/BOP_except_55175_shuffled_"+file_id+"_report.txt", sep="\t", mode="a", index=None
            )

        def Draw():
            pre,recall,the = precision_recall_curve(
                                    test_y,
                                    xgb_grid.predict_proba(test_x)[:,1]
                                    )
            if len(recall) > 3:
                for x,y,t in zip(recall, pre, the):
                    if x >= 0.75 and y >= 0.8:
                        sns.set_style("whitegrid")
                        plt.figure(figsize=(10,6))
                        plt.plot(recall, pre)
                        plt.ylim(bottom=0.0)
                        plt.scatter(x,y,facecolors="None",edgecolors="#003CFF")
                        plt.text(x,y,t,rotation=45,fontsize=13)

                        adj_pred = pd.DataFrame(
                                                xgb_grid.predict_proba(test_x)[:,1],
                                                index=test_x.index
                                                )
                        adj_pred["bi"] = adj_pred[0].apply(
                                                            lambda x : 1 if x > 0.509042 else 0
                                                            )
                        confusion_df = pd.merge(
                                                test_y,adj_pred[["bi",0]],
                                                left_index=True,right_index=True
                                                )
                        print(confusion_df.sort_values(by=0))
                        confusion = confusion_matrix(
                                                    confusion_df["group"], confusion_df["bi"]
                                                    )
                        confusion = pd.DataFrame(
                                                confusion,
                                                columns=["NR","Res"],
                                                index=["NR","Res"]
                                                )
                        plt.figure(figsize=(5,5))
                        sns.heatmap(confusion, annot=True, cmap="mako")
                        plt.xlabel("Predicted label", fontsize=14)
                        plt.ylabel("True label", fontsize=14)
                        plt.title(random_state, fontsize=13)
                        accuracy = accuracy_score(confusion_df["group"], confusion_df["bi"])
                        f1 = f1_score(confusion_df["group"], confusion_df["bi"])
                        print("F1: %.2f, ACC: %.2f" % (f1, accuracy))
                        break

        Draw()
        # if os.path.exists(OUT+"cv_result/") == False:
        #     os.mkdir(OUT+"cv_result/")
        
        ## Save model
        # import joblib
        # if os.path.exists(OUT+"model/") == False:
        #     os.mkdir(OUT+"model/")
        # file_name = OUT+'model/'+str(random_state)+'_transcripts.h5'
        # print(file_name)
        # joblib.dump(xgb_grid, file_name)

        # pre,rec,the = precision_recall_curve(test_y, xgb_grid.predict_proba(test_x)[:,1])
        # sns.set_style("whitegrid")
        # plt.plot(rec,pre)
        # plt.ylim(bottom=0)
        # print(xgb_grid.predict_proba(test_x)[:,1])
        # print(test_x)

    sk_wrapper(random_state)
    start+=1
# %%
