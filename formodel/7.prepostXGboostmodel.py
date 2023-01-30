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

warnings.simplefilter(action='ignore', category=FutureWarning)
from functools import partial       

#%%
"""
pre-post stable gene + variable transcript usage
"""

import random
random_state = random.randint(1,99999)

#^ download input tu file
file_id = ['input_stable_mt_DTUresult.csv','input_stable_resint_DTUresult.csv']
DIR = "/home/jiye/jiye/copycomparison/OC_transcriptome/XGboost/"
OUT = DIR + "modeloutput/"
if os.path.exists(OUT) == False:
    os.mkdir(OUT)

tu = pd.read_csv(DIR+file_id[0], index_col=0)
tu['pre/post'] = tu.index.str[11:]
tu = tu.replace(['bfD','atD'],[0,1])
print("number of used features : ", tu.shape[1])
print("number of used samples :", tu.shape[0])

# merged.to_csv(
#         OUT+"add_clinical_"+file_id,
#         sep="\t")

#^ train, test set split
train_x, test_x, train_y, test_y = train_test_split(
    tu.iloc[:,:-2], tu.iloc[:,-1], test_size=0.2, random_state=random_state, stratify=tu.iloc[:,-1]
)

#^ XGboost parameter
xgb_parameter_bounds = [
                        Integer(2,70,name="n_estimators"),    # range (from, to)
                        Real(0.001,0.9,name="learning_rate"),
                        Integer(1,10,name="max_depth"),
                        Real(0.1,0.9,name="subsample"), # 각 트리마다의 데이터 샘플링 비율.
                        Real(1,20,name="gamma"), # 분할을 수행하는데 필요한 최소 손실 감소를 지정한다. 클수록 over 방지
                        Real(0.1,0.9,name="colsample_bytree")   # 각 트리마다의 feature 샘플링 비율
                        ]

@use_named_args(xgb_parameter_bounds)

def objective(**xgb_parameter_bounds):
    xgb = XGBClassifier(random_state=42, n_jobs=None)
    xgb.set_params(**xgb_parameter_bounds)

    return -np.mean(cross_val_score(xgb, train_x, train_y, cv=10, n_jobs=None, scoring="f1"))

res_gp = gp_minimize(objective, xgb_parameter_bounds, n_calls=100, random_state=0, n_jobs=None, n_initial_points=10, kappa=3.0, verbose=False)

#^ XGboost evaulation criteria
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
plot_convergence(res_gp)

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

#^ Run XGboost
xgb = XGBClassifier(
                    random_state=random_state,
                    objective='binary:logistic', eval_metric='mlogloss',
                    use_label_encoder=False, n_jobs=None
                    )
kfold = StratifiedKFold(n_splits=10, random_state=random_state, shuffle=True)
xgb_grid = RandomizedSearchCV(
                            xgb, best_params, cv=kfold, n_jobs=None, random_state=random_state,
                            scoring="f1", verbose=1, n_iter=1, return_train_score=True
                            )
xgb_grid.fit(train_x, train_y)

report = pd.DataFrame(columns=["ROC AUC", "PR AUC", "mean test score", \
            "mean train score", "inde test score"], index=[random_state])
roc,pr = get_clf_eval(test_y, xgb_grid.predict_proba(test_x)[:,1],\
    xgb_grid.predict(test_x),"off")
report.iloc[0,0] = roc
report.iloc[0,1] = pr
cv_result = pd.DataFrame(xgb_grid.cv_results_)
cv_result["test_score_added_by_hyeongu"] = xgb_grid.score(
    test_x, test_y)    # indenpendent set 에 대한 검증, 이건 세트가 고정이라 best params 로 한 것임

cv_result = cv_result.sort_values(by="mean_test_score")[["mean_test_score","mean_train_score","test_score_added_by_hyeongu"]]
best = cv_result.iloc[-1,:]
report.iloc[0,2] = best[0]
report.iloc[0,3] = best[1]
report.iloc[0,4] = best[2]

## Result check
print(xgb_grid.best_params_)
print(report)

params = pd.DataFrame.from_dict(best_params, orient="columns")
params.index = [random_state]
df_rep = pd.concat([report, params], axis=1)

df_rep.to_csv(
        OUT+"xgboost_result"+file_id, mode="a", #append mode
        sep="\t"
        )

# %%
