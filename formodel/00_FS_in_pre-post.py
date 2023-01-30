#%%
"""
transcript usage 을 통한 gHRD 예측 모델 학습,
gradient boosting algorithm by xgboost
"""
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
from xgboost import plot_importance, XGBClassifier
from sklearn import svm, model_selection
from scipy.stats import ttest_1samp, wilcoxon, ttest_ind, mannwhitneyu, fisher_exact, binom_test,\
    pearsonr, gmean, spearmanr, gaussian_kde
from numpy import set_printoptions
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score,\
    roc_curve, auc, roc_auc_score, confusion_matrix, precision_recall_curve, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold,\
    train_test_split
from sklearn.decomposition import NMF,PCA
from sklearn.inspection import permutation_importance


## Input processing
# random_state = int(sys.argv[1])
DIR = "/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/merged_TU/whole_TU/"
OUT = DIR+"XGBoost/"
if os.path.exists(OUT) == False:
    os.mkdir(OUT)

tu = pd.read_csv(
    "/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/pre_post_TU/final_data/whole_TU_highly_variable_only.txt",\
    sep="\t", index_col="gene_ENST")
samples = tu.columns.tolist()
for i in range(int(len(samples)/2)):
    i = i*2
    if "atD" in samples[i]:
        samples[i+1] = samples[i+1][:10]+"-bfD"
    elif "bfD" in samples[i]:
        samples[i+1] = samples[i+1][:10]+"-atD"
    elif "atD" in samples[i+1]:
        samples[i] = samples[i][:10]+"-bfD"
    elif "bfD" in samples[i+1]:
        samples[i] = samples[i][:10]+"-atD"

tu.columns = samples
try:
    tu = tu.drop(["SV-OV-P080-atD","SV-OV-P250-atD","SV-OV-P055-atD",\
            "SV-OV-P143-atD","SV-OV-P137-atD","SV-OV-P134-atD",\
            "SV-OV-P174-bfD","SV-OV-P164-atD"], axis=1)
except:
    print("Already exculded abnormal samples")
    pass

tu = tu.T
tu.index = tu.index.str.replace("P","T")
tu = tu.T

## Feature selection
## 일단은 DNA repair 전체를 대상으로 진행해보자, 일정 수준 검증이 되었다고 보고
## Highly variable transcripts
high_tu = tu
high_tu = high_tu[(high_tu==0).astype(int).sum(axis=1) <= 10]   # 90% 이상의 샘플에서 발현이 있는 경우
high_tu = high_tu.T[(high_tu==0).sum() < high_tu.shape[0]*0.5].T   # Abnormal samples filtering
high_tu["gene"] = high_tu.index.str.split("-",1).str[1]
dna_repair = pd.read_csv(
    "/home/hyeongu/DATA5/hyeongu/GSEA/gmt/merged_GO_NER.txt",
    sep="\t", header=None
)
for num,gene in enumerate(dna_repair[0].unique()):
    if num == 0:
        sub = high_tu[high_tu["gene"]==gene]
    else:
        sub = pd.concat([sub, high_tu[high_tu["gene"]==gene]])

### Jump to this gene selection step
sub = high_tu
###
dna_repair = pd.read_csv(
    "/home/hyeongu/DATA5/hyeongu/GSEA/gmt/vegf_repair-related_fgf.txt",
    sep="\t", header=None
)
for num,gene in enumerate(dna_repair[0].unique()):
    if num == 0:
        sub = high_tu[high_tu["gene"]==gene]
    else:
        sub = pd.concat([sub, high_tu[high_tu["gene"]==gene]])
sub = sub.drop("gene",axis=1)
print(sub.shape)
# %%
drop_set = []
for i in range(sub.shape[0]):
    upper = sub.iloc[i,:].quantile(.75)
    lower = sub.iloc[i,:].quantile(.25)
    iqr = upper - lower
    if iqr < 0.1:
        drop_set.append(sub.index.tolist()[i])
sub = sub.drop(drop_set)
print(sub.shape)
# %%
## 여기서 약간 새로운 접근 방식을 채택한다, DNA repair (SSB관련), VEGF, FGF 관련 term 의 TU pattern 을 이용
## 각 모델들의 prob 에 대해 weight 를 어떻게 줄 수 있을지
def Features_(data, goal, model):
    from sklearn.feature_selection import SelectFromModel
    group_info = pd.read_csv(
        "/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/clinical_info_gHRD/processed_info_with_originalID.txt", 
        sep="\t", index_col="GID")
    group_info = group_info[["OM/OS", "ORR", "interval"]]
    group_info.columns = ["OM/OS", "group", "interval"]
    group_info["10ID"] = group_info.index.str.replace("F","T").str[:10]
    group_info = group_info.set_index("10ID")
    group_info = group_info.dropna()
    data["pre-post"] = data.index.str[-3:]
    data["pre-post"] = data["pre-post"].apply(lambda x : 1 if x == "bfD" else 0)
    data.index = data.index.str[:10]

    if goal == "pre-post":
        X = data.iloc[:,:-1]
        Y = data.iloc[:,-1]

    else:
        data = data[data["pre-post"]==1]
        merged = pd.merge(data, group_info, left_index=True,
                        right_index=True, how="inner")
        merged = merged.drop_duplicates()
        X = merged.iloc[:,:-4]
        Y = merged.iloc[:,-2]

    if model == "tree":
        xgb_sk = XGBClassifier(random_state=55,
                    objective='binary:logistic', eval_metric="error",
                    use_label_encoder=False, n_jobs=4)
        selector = SelectFromModel(estimator=xgb_sk).fit(X,Y)
        score = selector.estimator_.feature_importances_    # When the estimator is classifier
        score = pd.DataFrame(score)
        score = score[score[0]>0]

    elif model == "self_tree":
        xgb_sk = XGBClassifier(random_state=55,
                    objective='binary:logistic', eval_metric="error",
                    use_label_encoder=False, n_jobs=4)
        ## Hyper-parameter tuning
        xgb_param_grid = {
            "n_estimators": [int(x) for x in np.linspace(start=3, stop=100, num=5)],
            "learning_rate": [0.01,0.02,0.03,0.04,0.05],    # [0-1] 클수록 overfitting
            "max_depth": [2,3,4,5,6],
            "subsample": [0.4,0.5,0.6,0.7,0.8], # [0-1] 각 Tree의 관측 데이터 샘플링 비율
            "gamma": [0.7,0.8,0.9,1.0], # 분할을 수행하는데 필요한 최소 손실 감소를 지정, 클수록 overfitting 제어
            "colsample_bytree": [0.2,0.3,0.4]   # 각 트리마다의 feature 샘플링 비율, 
            }

        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        ## Do learning
        xgb_grid = RandomizedSearchCV(
            xgb_sk, xgb_param_grid, cv=kfold, n_jobs=15, random_state=random_state,
            scoring="precision", verbose=1, n_iter=100, return_train_score=True
        )

        xgb_grid.fit(X, Y)
        cv_result = pd.DataFrame(xgb_grid.cv_results_)
        cv_result.loc["mean"] = cv_result.mean()
        cv_result = cv_result.iloc[-1,:]
        print(cv_result[["mean_test_score","mean_train_score"]])
        print(xgb_grid.best_params_)
        score = xgb_grid.best_estimator_.feature_importances_    # When the estimator is classifier
        score = pd.DataFrame(score)
        score = score[score[0]>0]
        score.columns = [random_state]

    select = pd.DataFrame(X.columns)
    select.columns = ["gene_ENST"]
    select = select.reindex(score.index.tolist())
    sub_X = X[select["gene_ENST"].tolist()]
    input_df = pd.merge(sub_X,Y,left_index=True,right_index=True)
    print(X.shape[0])
    print(input_df.shape[1])
    input_df.to_csv(
        OUT+goal+"_features.txt", 
        sep="\t"
        )

random_state = 42
score = Features_(sub.T,"pre-post","self_tree")
# %%
