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
# import mkl
# mkl.set_num_threads(15) # Set the number of threads

"""
INPUT: /home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/code/whole_file/02_new_filtering_v2023.py
새로운 strategy 로 찾은 features 를 이용한 모델 학습 시작
/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/code/ML_predictor/03-2_BOP_co_learning.py 
위 코드에서 약간 수정한 Ver
이 스크립트의 결과에 기반해서 성능평가는 따로 진행해야 함, 이 스크립트는 단순히 well-tuning model 을 만들기 위함임
성능 평가: /home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/code/ML_predictor/03-3_performance_comp.py
"""

## Input processing & parameter tuning
start = 0
for random_state in range(1,301):
    import random
    random_state = random.randint(1,9999999)
    DIR = "/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/merged_TU/whole_TU/XGBoost/"
    file_id = sys.argv[1]   
    # Old features 개수 맞춤: 230109_ex_pre-post_From_term_GENE_pos_sig_response_input.txt
    # Filtering 이후 전체 featrues: 230109_ex_pre-post_From_term_GENE_Filtered_pos_sig_response_input.txt
    merged = pd.read_csv(
                        DIR
                        +file_id,
                        sep="\t", index_col="Unnamed: 0").T
    
    print("Used samples :", merged.shape[0])
    train_x, test_x, train_y, test_y = train_test_split(
                                                        merged.iloc[:,:-2].astype(float), 
                                                        merged.iloc[:,-1].astype(float), 
                                                        test_size=0.2, 
                                                        random_state=random_state,
                                                        stratify=merged.iloc[:,-1]
                                                        )
    # start += 1
    # xgb_parameter_bounds = [
    #                     Integer(20,100,name="n_estimators"),    # range (from, to)
    #                     Real(0.001,0.01,name="learning_rate"),  # 높을수록 overfitting
    #                     Integer(2,5,name="max_depth"),
    #                     Real(0.5,1.0,name="subsample"), # 각 트리마다의 데이터 샘플링 비율.
    #                     Real(0.01,4.0,name="gamma"), # 분할을 수행하는데 필요한 최소 손실 감소를 지정한다. 클수록 over 방지
    #                     Real(0.1,0.6,name="colsample_bytree")   # 각 트리마다의 feature 샘플링 비율
    #                     ]
    xgb_parameter_bounds = [
                        Integer(50,150,name="n_estimators"),    # range (from, to)
                        Real(0.06,0.09,name="learning_rate"),  # 높을수록 overfitting
                        Integer(2,3,name="max_depth"),
                        Real(0.3,0.35,name="subsample"), # 각 트리마다의 데이터 샘플링 비율.
                        Real(1.0,1.6,name="gamma"), # 분할을 수행하는데 필요한 최소 손실 감소를 지정한다. 클수록 over 방지
                        Real(0.1,0.3,name="colsample_bytree")   # 각 트리마다의 feature 샘플링 비율
                        ]
    @use_named_args(xgb_parameter_bounds)
    def objective(**xgb_parameter_bounds):
        xgb = XGBClassifier(
                            random_state=42, n_jobs=15,
                            use_label_encoder=False, 
                            eval_metric='error'
                            )
        kfold = StratifiedKFold(n_splits=10, random_state=random_state, shuffle=True)
        xgb.set_params(**xgb_parameter_bounds)

        return -np.mean(cross_val_score(xgb, train_x, train_y, cv=kfold, n_jobs=15,
                                        scoring="f1"))

    res_gp = gp_minimize(objective, xgb_parameter_bounds, n_calls=80,
            random_state=0, n_jobs=15, kappa=3,
            verbose=True)  # kappa default 1.96 값이 높아질수록 expoliation 보다 exploration 을 더 많이 함

    def get_clf_eval(y_test, pred_proba, pred, sw):
        """ 모델 평가지표 """
        # confusion = confusion_matrix(y_test, pred)
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred)
        recall = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        roc_auc = roc_auc_score(y_test, pred_proba)
        hrd_precision, hrd_recall, hrd_the = precision_recall_curve(
            y_test, pred_proba)
        pr_auc = auc(hrd_recall, hrd_precision)
        
        if sw == "on":
            print("\n",
                "ROC_auc: {0:.4f}, PR_auc: {1:.4f}"
                .format(roc_auc, pr_auc),\
                "ACC: {0:.4f}, Prec: {1:.4f}, Recal: {2:.4f}, F1: {3:.4f}"
                .format(accuracy, precision, recall, f1))
            
        return roc_auc, pr_auc

    from skopt.plots import plot_convergence
    # plot_convergence(res_gp)
    # plt.savefig(
    #             DIR+"co_learning_cv_result/"
    #             +file_id+"_converge_plot.pdf",
    #             bbox_inches="tight"
    #             )
    # print(res_gp)

    print("""Best score=%.4f
    Best parameters:
    - n_estimators=%d
    - learning_rate=%.6f
    - max_depth=%d
    - subsample=%.6f
    - gamma=%.6f
    - colsample_bytree=%.6f""" % (res_gp.fun,
                                res_gp.x[0], res_gp.x[1],
                                res_gp.x[2], res_gp.x[3],
                                res_gp.x[4], res_gp.x[5]))

    best_params = {
                    "n_estimators": [res_gp.x[0]],
                    "learning_rate": [res_gp.x[1]],
                    "max_depth": [res_gp.x[2]],
                    "subsample": [res_gp.x[3]],
                    "gamma": [res_gp.x[4]],
                    "colsample_bytree": [res_gp.x[5]]
                    }

    xgb = XGBClassifier(
                        random_state=random_state,
                        objective='binary:logistic', eval_metric="logloss",
                        use_label_encoder=False, n_jobs=4
                        )
    kfold = StratifiedKFold(n_splits=10, random_state=random_state, shuffle=True)

    ## Tuning model 에 대한 평가
    xgb_grid = RandomizedSearchCV(
                                xgb, best_params, cv=kfold, n_jobs=2, random_state=random_state,
                                scoring="f1", verbose=1, n_iter=1, return_train_score=True
                                )
    xgb_grid.fit(train_x, train_y)
    report = pd.DataFrame(columns=["ROC AUC", "PR AUC", "#Class", "mean test score", \
                "mean train score", "inde test score"], index=[random_state])
    roc,pr = get_clf_eval(test_y, xgb_grid.predict_proba(test_x)[:,1],\
        xgb_grid.predict(test_x),"off")
    report.iloc[0,0] = roc
    report.iloc[0,1] = pr
    report.iloc[0,2] = len(set(xgb_grid.predict_proba(test_x)[:,1]))
    cv_result = pd.DataFrame(xgb_grid.cv_results_)
    cv_result["test_score_added_by_hyeongu"] = xgb_grid.score(
        test_x, test_y)    # indenpendent set 에 대한 검증, 이건 세트가 고정이라 best params 로 한 것임

    cv_result = cv_result.sort_values(by="mean_test_score")[["mean_test_score","mean_train_score","test_score_added_by_hyeongu"]]
    best = cv_result.iloc[-1,:]
    report.iloc[0,3] = best[0]
    report.iloc[0,4] = best[1]
    report.iloc[0,5] = best[2]

    ## Result check
    print(xgb_grid.best_params_)
    print(report)
    params = pd.DataFrame.from_dict(best_params, orient="columns")
    params.index = [random_state]
    df_rep = pd.concat([report, params], axis=1)
    df_rep.to_csv(
        DIR+"co_learning_cv_result/"+
        "co_learning_BOP_"+file_id, mode="a",
        sep="\t"
        )

    ### importance 가 필요한 경우
    FI = pd.DataFrame(
                    xgb_grid.best_estimator_.feature_importances_,
                    index=merged.columns[:-2], columns=[random_state]
                    ).T
    try:
        FI.to_csv(
                DIR+"From_term_GENE_important.txt",
                sep="\t", mode="a"
                )
    except:
        pass

    ###
    start += 1
# %%


