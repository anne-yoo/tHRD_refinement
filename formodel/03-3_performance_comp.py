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
import warnings
warnings.simplefilter(action='ignore', category=Warning)
from functools import partial               # 함수 변수 고정
import random


""" 
screen -r hyeongu_model
모델 결과 정리
"""

DIR="/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/merged_TU/whole_TU/XGBoost/co_learning_cv_result/"
# file_id="230109_ex_pre-post_relaxed_pos_sig_response_input.txt"
# file_id="230109_ex_pre-post_old_input.txt"
file_id="230109_ex_pre-post_From_term_pos_sig_response_input.txt"
data = pd.read_csv(
                DIR+"co_learning_BOP_"+file_id,
                sep="\t"
                )   # BOP 결과 리스트
data = data[data["ROC AUC"]!="ROC AUC"] # 중복 header 제거
merged = pd.read_csv(
                    "/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/merged_TU/whole_TU/XGBoost/"
                    +file_id,
                    sep="\t"
                    )   # input TU data
merged.index = merged.iloc[:,0]
merged = merged.iloc[:,1:].T

for line in range(data.shape[0]):
    bopseed = data.iloc[[line]]["Unnamed: 0"].values[0]
    # bopseed = 8827922
    # Target seed for best parameters
    start=0
    try:
        os.remove(DIR+"/"+file_id+"_"+str(bopseed)+"_report.txt")
    except:
        pass

    while start < 30:
        random_state = random.randint(1,99999999)
        train_x, test_x, train_y, test_y = train_test_split(
                                                            merged.iloc[:,:-2].astype(float), 
                                                            merged.iloc[:,-1].astype(float),
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
                                eval_metric='logloss',
                                use_label_encoder=False, 
                                n_jobs=4, verbosity=0
                                )

            parameter = pd.read_csv(
                                    DIR+"co_learning_BOP_"+file_id,
                                # "/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/merged_TU/whole_TU/XGBoost/cv_result/"
                                # +"sample_except_pre-post_BOP_feature_matrix_response_group.txt",
                                    sep="\t"
                                    )   # BOP result
            parameter = data[data["Unnamed: 0"]==bopseed]
            xgb_param_grid = {
                            "n_estimators": [int(parameter["n_estimators"].values[0])],
                            "learning_rate": [float(parameter["learning_rate"].values[0])],    # [0-1] 클수록 overfitting
                            "max_depth": [int(parameter["max_depth"].values[0])],
                            "subsample": [float(parameter["subsample"].values[0])], # [0-1] 각 Tree의 관측 데이터 샘플링 비율
                            "gamma": [float(parameter["gamma"].values[0])], # 분할을 수행하는데 필요한 최소 손실 감소를 지정, 클수록 overfitting 제어
                            "colsample_bytree": [float(parameter["colsample_bytree"].values[0])]   # 각 트리마다의 feature 샘플링 비율, 
                            }
            
            kfold = StratifiedKFold(n_splits=2)

            ## Do learning
            xgb_grid = RandomizedSearchCV(
                                            xgb_sk, xgb_param_grid, cv=kfold, 
                                            n_jobs=5, random_state=random_state,
                                            scoring="f1", verbose=0, n_iter=1, 
                                            return_train_score=True
                                        )   # 정해졌으므로 n_iter=1

            xgb_grid.fit(train_x, train_y)
            sk_pred_prob = pd.DataFrame(
                                        xgb_grid.predict_proba(train_x)[:,1]
                                        )
            sk_pred_prob.index = train_x.index

            ## Cross validation
            report = pd.DataFrame(columns=["ROC AUC", "PR AUC", "f1", "mean test score", \
                    "mean train score", "inde test score", "#Class"], \
                    index=[random_state])
            roc,pr,f1 = get_clf_eval(
                                    test_y, 
                                    xgb_grid.predict_proba(test_x)[:,1],
                                    xgb_grid.predict(test_x),
                                    "off"
                                    )
            ## Ext val
            # eroc,epr,ef1 = get_clf_eval(
            #                         ext_df1["group"], 
            #                         xgb_grid.predict_proba(ext_df1.iloc[:,:-1])[:,1],
            #                         xgb_grid.predict(ext_df1.iloc[:,:-1]),
            #                         "off"
            #                         )
            #############
            report.iloc[0,0] = roc
            report.iloc[0,1] = pr
            report.iloc[0,2] = f1
            cv_result = pd.DataFrame(xgb_grid.cv_results_)
            cv_result["test_score_added_by_hyeongu"] = xgb_grid.score(
                test_x, test_y)    
            # indenpendent set 에 대한 검증, 이건 세트가 고정이라 best params 로 한 것임
            
            cv_result = cv_result.sort_values(by="mean_test_score")[["mean_test_score","mean_train_score","test_score_added_by_hyeongu"]]
            best = cv_result.iloc[-1,:]
            report.iloc[0,3] = best[0]
            report.iloc[0,4] = best[1]
            report.iloc[0,5] = best[2]
            report.iloc[0,6] = len(set(xgb_grid.predict_proba(test_x)[:,1].tolist()))

            def Draw():
                pre,recall,the = precision_recall_curve(
                                        test_y,
                                        xgb_grid.predict_proba(test_x)[:,1]
                                        )
                ## Ext val
                # fpr,tpr,ex_the = roc_curve(
                #                         ext_df1.iloc[:,-1],
                #                         xgb_grid.predict_proba(ext_df1.iloc[:,:-1])[:,1]
                #                         )

                # epre,erecall,ethe = precision_recall_curve(
                #                         ext_df1.iloc[:,-1],
                #                         xgb_grid.predict_proba(ext_df1.iloc[:,:-1])[:,1]
                #                         )
                ##############
                if len(recall) > 3:
                    for x,y,t in zip(recall, pre, the):
                        if pr > 0.8:
                            # sig.append(random_state)
                            ## Figure
                            # sns.set_style("whitegrid")
                            # plt.figure(figsize=(6,6))
                            # plt.plot(recall, pre)
                            # plt.text(0.05,0.08,"AUC={}".format(round(pr,3)),fontsize=13)
                            # plt.ylim(bottom=0.0)
                            # plt.xlabel("Recall", fontsize=13)
                            # plt.ylabel("Precision", fontsize=13)
                            ####

                            ## Find the best cut off
                            fscore = (2 * pre * recall) / (pre + recall)
                            # locate the index of the largest f score
                            ix = np.argmax(fscore)
                            ####
                            # plt.scatter(recall[ix], pre[ix], color="#000000")

                            # plt.figure(figsize=(6,6))
                            # # plt.plot(tpr, fpr)    # PR curve
                            # plt.plot(fpr, tpr)  # ROC curve
                            # plt.ylim(bottom=0.0)
                            # plt.text(0.05,0.08,"AUC={}".format(round(eroc,3)),fontsize=13)
                            # # plt.xlabel("Recall", fontsize=13)
                            # # plt.ylabel("Precision", fontsize=13)
                            # plt.xlabel("FPR", fontsize=13)
                            # plt.ylabel("TPR", fontsize=13)
                            ####

                            ## Find the best cut off
                            # efscore = np.sqrt(tpr * (1-fpr))
                            # locate the index of the largest f score
                            # ix2 = np.argmax(efscore)

                            ####
                            # plt.scatter(fpr[ix2], tpr[ix2], color="#000000")

                            # plt.figure(figsize=(6,6))
                            # plt.plot(erecall, epre)    # PR curve
                            # # plt.plot(fpr, tpr)  # ROC curve
                            # plt.ylim(bottom=0.0)
                            # plt.text(0.05,0.08,"AUC={}".format(round(epr,3)),fontsize=13)
                            # plt.xlabel("Recall", fontsize=13)
                            # plt.ylabel("Precision", fontsize=13)
                            ####
                            # plt.xlabel("FPR", fontsize=13)
                            # plt.ylabel("TPR", fontsize=13)

                            int_pred = pd.DataFrame(xgb_grid.predict_proba(test_x)[:,1])
                            int_pred["bi"] = int_pred[0].apply(lambda x : 1 if x >= the[ix] else 0)
                            # int_pred["bi"] = int_pred[0].apply(lambda x : 1 if x >= 0.528016 else 0)
                            int_pred.index = test_x.index
                            int_pred = pd.merge(int_pred, test_y, left_index=True, right_index=True)
                            # ext_pred = pd.DataFrame(xgb_grid.predict_proba(ext_df1.iloc[:,:-1])[:,1])
                            # ext_pred.index = ext_df1.index
                            # ext_pred["bi"] = ext_pred[0].apply(lambda x : 1 if x >= ex_the[ix2] else 0)
                            # # ext_pred["bi"] = ext_pred[0].apply(lambda x : 1 if x >= 0.603127 else 0)
                            # ext_pred = pd.merge(ext_pred, ext_df1["group"], left_index=True, right_index=True)
                            # # print(int_pred.sort_values(by=0))

                            sns.set_style("white")
                            cm = confusion_matrix(int_pred[["group"]], int_pred[["bi"]])
                            # cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, \
                            #     display_labels = [False, True])
                            # cm_display.plot()
                            # cm2 = confusion_matrix(ext_pred[["group"]], ext_pred[["bi"]])
                            # cm_display = ConfusionMatrixDisplay(confusion_matrix = cm2, \
                            #     display_labels = [False, True])
                            # cm_display.plot()
                        break
                
            Draw()
            report.to_csv(
                        DIR+"/"+file_id+"_"+str(bopseed)+"_report.txt", 
                        sep="\t", mode="a", index=None
                        )
            FI = pd.DataFrame(
                            xgb_grid.best_estimator_.feature_importances_,
                            index=merged.columns[:-2], columns=[bopseed])
            try:
                FI[FI[7]>0].to_csv(
                                    DIR+"From_term_important.txt", sep="\t"
                                    )
            except:
                pass

        sk_wrapper(random_state)
        start+=1

    def Analysis(seed=sys.argv):
        auc_check = {}
        auc_check[bopseed] = []
        try:
            data = pd.read_csv(
                                DIR+"/"+file_id+"_"+str(bopseed)+"_report.txt",
                                sep="\t"
                                )
            data = data.dropna()
            data = data[data["ROC AUC"]!="ROC AUC"]
            data = data.astype(float)
            data = data[["ROC AUC", "PR AUC", "mean test score", \
                        "mean train score", "inde test score"]]
            # data = data.drop("ext ROC", axis=1)
            if data["PR AUC"].mean() > 0.76:
                sns.set_style("whitegrid")
                plt.figure(figsize=(4,8))
                sns.stripplot(data=data, zorder=0,
                            palette={"PR AUC":"#FF4B4B", "ROC AUC":"#FFACAC",\
                            "mean test score":"#56AAFF", "mean train score":"#56AAFF",\
                            "inde test score":"#56AAFF"})
                plt.ylim(0.0,1.0)
                # sns.barplot(data=data, color="#4DA0FF")
                #### Customized plot ####
                data.loc["median"] = data.median()
                data.loc["3Q"] = data.quantile(.75)
                data.loc["1Q"] = data.quantile(.25)
                sns.boxplot(data=data.loc[["median"]], linewidth=2.5, zorder=1)
                sns.boxplot(data=data.loc[["3Q"]], width=0.4, linewidth=2.5, zorder=1)
                sns.boxplot(data=data.loc[["1Q"]], width=0.4, linewidth=2.5, zorder=1)
            auc_check[bopseed].append(data["PR AUC"].mean())
            auc_check[bopseed].append(data["ROC AUC"].mean())
            auc_check = pd.DataFrame.from_dict(auc_check).T
            auc_check.columns = ["PR", "ROC"]
            print(auc_check)
            if os.path.exists(DIR+"30times_PR_AUC_check.txt"):
                auc_check.to_csv(
                                DIR+"30times_PR_AUC_check.txt",
                                sep="\t", mode="a", header=None
                                )
            else:
                auc_check.to_csv(
                                DIR+"30times_PR_AUC_check.txt",
                                sep="\t", mode="a"
                                )

            ### Select the best model
            # sns.pointplot(
            #             data=data.loc[[6104]], color="#FF0000", 
            #             markers="o", zorder=2
            #             )
            #########################
            if data["PR AUC"].mean() > 0.76:
                point_df = data.melt()
                # sns.scatterplot(x="variable", y="value", data=point_df, edgecolor="None")
                plt.axhline(0.7, color="#B1B1B1", linestyle="--")
                plt.xticks(rotation=90, fontsize=15)
                plt.yticks(fontsize=14)
                plt.figure(figsize=(8,4))
                sns.kdeplot(point_df[point_df["variable"]=="PR AUC"]["value"])
            # plt.savefig(
            #             "/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/pre_post_TU/final_data/MW_result/figure/"
            #             +str(bopseed)+"_30times.pdf", 
            #             bbox_inches="tight"
            #             )
            num_ambi = data[data["AMBI_pr"]>0.7].shape[0]
            df = pd.DataFrame.from_dict({seed:num_ambi}, orient="index")
            df.columns = ["AMBI"]
        except:
            pass

    Analysis(bopseed)
    # break
# %%
