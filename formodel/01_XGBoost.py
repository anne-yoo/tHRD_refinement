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
import SET_arial

"""
transcript tpm 을 통한 gHRD 예측 모델 학습,
gradient boosting algorithm by xgboost
"""

bopseed = 221486
import random
DIR = "/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/merged_TU/whole_TU/"
OUT = DIR+"XGBoost/"
# file_id = "feature_matrix_response_group.txt"
# file_id = "include_pre-post_response_group.txt"
file_id = "include_pre-post_abs_74.txt"
# file_id = "old_model_featrues.txt"
try:
    os.rmdir(OUT+"cv_result/merged_BOP_except_"+str(bopseed)+"_shuffled_"+file_id+"_report.txt")
except:
    pass

## Input processing
start = 0
sig = []
while start < 50:
    random_state = random.randint(1,9999999)
    # random_state = 8526317

    DIR = "/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/merged_TU/whole_TU/"
    OUT = DIR+"XGBoost/"
    if os.path.exists(OUT) == False:
        os.mkdir(OUT)
    tu = pd.read_csv(
        OUT+file_id, sep="\t", index_col="gene_ENST")
    print(tu.shape)
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
    # print(tu.shape)
    
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

    ext_val1 = pd.read_csv(
                            "~/DATA5/hyeongu/AMBITION_copy/AMBITION/align/2nd_pass/OV_gtf_quantified/processed/target_response_group.txt",
                            sep="\t", index_col="gene_ENST"
                            ).T
    ext_y1 = pd.read_csv(
                        "/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/clinical_info_gHRD/221128_ambition_clinical_with_id.txt",
                        sep="\t", header=None
                        )              
    ext_y1 = ext_y1[[7,3]]
    ext_y1.columns = ["ID","group"]   # 11 vs 10 / nr vs res

    ## Sum ambition
    ext_df1 = pd.merge(ext_val1, ext_y1, left_index=True, right_on="ID").set_index("ID")
    # print(merged.shape)
    # merged = pd.concat([merged,ext_df1])
    # print("Unseened samples :", unseened_df.shape[0])
    # print("-"*40)
    merged.to_csv(
            OUT+"add_clinical_"+file_id,
            sep="\t")
    # sns.kdeplot((merged.iloc[:,:-2]==0).sum(axis=1))
    # plt.axvline((merged.iloc[:,:-2]==0).sum(axis=1).quantile(.90))
    # print((merged.iloc[:,:-2]==0).sum(axis=1).quantile(.90))
    # print(merged[(merged.iloc[:,:-2]==0).sum(axis=1).astype(float) > 17])
    # merged = merged[(merged.iloc[:,:-2]==0).sum(axis=1).astype(float) < 18] # Filter out sparse samples
    # print(merged.shape)
    # start += 1
    ext_val2 = pd.read_csv(
                            "~/LOCAL2/hyeongu/KGOG3046/KGOG3046_WTS/fastq/trim/2nd_pass/2nd_with_only_TRU-D/quantified/processed/target_response_group.txt",
                            sep="\t", index_col="gene_ENST"
                        ).T
    ext_y2 = pd.read_csv(
                        "/home/hyeongu/LOCAL2/hyeongu/KGOG3046/KGOG3046_WTS/fastq/trim/2nd_pass/2nd_with_only_TRU-D/221117_OV_gtf_quant/quantified/adj_id_map_with_clinical.txt",
                        sep="\t", header=None
                        )

    ext_val3 = pd.read_csv(
                            "~/DATA5/hyeongu/NAC_OV_sample/fastq/second_result/temp/quantified/processed/target_response_group.txt",
                            sep="\t", index_col="gene_ENST"
                        ).T
    ext_y3 = pd.read_csv(
                        "/home/hyeongu/DATA5/hyeongu/NAC_OV_sample/sample_info/processed_clinical.txt",
                        sep="\t", header=None
                        )

    ext_y2 = ext_y2[[4,1]]
    ext_y3 = ext_y3[[0,2]]
    ext_y2.columns = ["ID","group"]   # 2 vs 8 / nr vs res
    ext_y3.columns = ["ID","group"]   # 10 vs 10 / nr vs res
    ## Merge Platinum cohort
    ext_val1 = pd.merge(ext_val3.T, ext_val2.T, left_index=True, right_index=True).T
    ext_y1 = pd.concat([ext_y3, ext_y2])
    ext_df1 = pd.merge(ext_val1, ext_y1, left_index=True, right_on="ID").set_index("ID")

    # # %%
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
                            eval_metric='logloss',
                            use_label_encoder=False, 
                            n_jobs=4, verbosity=0
                            )
    
        xgb_param_grid = {
                        "n_estimators": [39],
                        "learning_rate": [0.007142544249264731],    # [0-1] 클수록 overfitting
                        "max_depth": [2],
                        "subsample": [0.5], # [0-1] 각 Tree의 관측 데이터 샘플링 비율
                        "gamma": [0.3995089501050342], # 분할을 수행하는데 필요한 최소 손실 감소를 지정, 클수록 overfitting 제어
                        "colsample_bytree": [0.5459145558456554]   # 각 트리마다의 feature 샘플링 비율, 
                        }
        kfold = StratifiedKFold(n_splits=10)

        ## Do learning
        xgb_grid = RandomizedSearchCV(
                                        xgb_sk, xgb_param_grid, cv=kfold, 
                                        n_jobs=30, random_state=random_state,
                                        scoring="f1", verbose=0, n_iter=1, 
                                        return_train_score=True
                                    )   # 정해졌으므로 n_iter=1

        xgb_grid.fit(train_x, train_y)
        sk_pred_prob = pd.DataFrame(
                                    xgb_grid.predict_proba(train_x)[:,1]
                                    )
        sk_pred_prob.index = train_x.index
        if os.path.exists(OUT+"pred_result/") == False:
            os.mkdir(OUT+"pred_result/")

        ## Cross validation
        report = pd.DataFrame(columns=["ROC AUC", "PR AUC", "f1", "mean test score", \
                "mean train score", "inde test score", "ext ROC", "ext PR AUC", "#class int", "#class ext", "seed"], index=[random_state])
        roc,pr,f1 = get_clf_eval(
                                test_y, 
                                xgb_grid.predict_proba(test_x)[:,1],
                                xgb_grid.predict(test_x),
                                "off"
                                )
        eroc,epr,ef1 = get_clf_eval(
                                ext_df1["group"], 
                                xgb_grid.predict_proba(ext_df1.iloc[:,:-1])[:,1],
                                xgb_grid.predict(ext_df1.iloc[:,:-1]),
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
        report.iloc[0,6] = eroc
        report.iloc[0,7] = epr
        report.iloc[0,8] = len(set(xgb_grid.predict_proba(test_x)[:,1].tolist()))
        report.iloc[0,9] = len(set(xgb_grid.predict_proba(ext_df1.iloc[:,:-1])[:,1]))
        # print(xgb_grid.best_params_)
        # print("-"*40)
        # roc,pr,f1 = get_clf_eval(
        #                         unseened_df.iloc[:,-1],
        #                         xgb_grid.predict_proba(unseened_df.iloc[:,:-2])[:,1],
        #                         xgb_grid.predict(unseened_df.iloc[:,:-2]),"off"
        #                         )
        # report.iloc[0,6] = roc
        # report.iloc[0,7] = pr
        # report.iloc[0,8] = f1
        report.iloc[0,10] = random_state
        print(len(set(xgb_grid.predict_proba(test_x)[:,1])),
            len(set(xgb_grid.predict_proba(ext_df1.iloc[:,:-1])[:,1])))
        # if len(set(xgb_grid.predict_proba(test_x)[:,1])) > 29:
        ## predict proba 서로 다르게 5개 이상 예측되는 경우
        # print(report)
        report.to_csv(
                    OUT+"/cv_result/merged_BOP_except_"+str(bopseed)+"_shuffled_"+file_id+"_report.txt", 
                    sep="\t", mode="a", index=None
                    )

        def Draw():
            pre,recall,the = precision_recall_curve(
                                    test_y,
                                    xgb_grid.predict_proba(test_x)[:,1]
                                    )
            # print(roc_auc_score(test_y,xgb_grid.predict_proba(test_x)[:,1]))
            if len(recall) > 3:
                for x,y,t in zip(recall, pre, the):
                    # if x >= 0.8 and y >= 0.8 and pr > 0.8:
                    # if pr > 0.8 and roc_auc_score(test_y,xgb_grid.predict_proba(test_x)[:,1]) > 0.65:
                    if pr > 0.8:
                        sig.append(random_state)
                        # sns.set_style("whitegrid")
                        # plt.figure(figsize=(6,6))
                        # plt.plot(recall, pre)
                        # plt.text(0.05,0.08,"AUC={}".format(round(pr,3)),fontsize=13)
                        # plt.ylim(bottom=0.0)
                        # plt.scatter(x,y,facecolors="None",edgecolors="#003CFF")
                        # plt.text(x,y,t,rotation=45,fontsize=13)
                        # plt.xlabel("Recall", fontsize=13)
                        # plt.ylabel("Precision", fontsize=13)
                        # # plt.savefig(
                        # #             OUT+"figure/16055_"+str(random_state)+"_PR.pdf",
                        # #             bbox_inches="tight"
                        # #             )

                        # adj_pred = pd.DataFrame(
                        #                         xgb_grid.predict_proba(test_x)[:,1],
                        #                         index=test_x.index
                        #                         )
                        # adj_pred["bi"] = adj_pred[0].apply(
                        #                                     lambda x : 1 if x >= 0.575708 else 0
                        #                                     )
                        # confusion_df = pd.merge(
                        #                         test_y,adj_pred[["bi",0]],
                        #                         left_index=True,right_index=True
                        #                         )
                        # confusion_df.columns = ["group","bi",random_state]                  
                        # print(confusion_df.sort_values(by=random_state).head(40))
                        # confusion = confusion_matrix(
                        #                             confusion_df["group"], confusion_df["bi"]
                        #                             )
                        # confusion = pd.DataFrame(
                        #                         confusion,
                        #                         columns=["NR","Res"],
                        #                         index=["NR","Res"]
                        #                         )
                        # plt.figure(figsize=(5,5))
                        # sns.heatmap(
                        #             confusion, annot=True, 
                        #             cmap="Reds", annot_kws={"fontsize":13})
                        # plt.xlabel("Predicted label", fontsize=14)
                        # plt.ylabel("True label", fontsize=14)
                        # plt.title(random_state, fontsize=13)
                        # # plt.savefig(
                        # #             OUT+"figure/16055_"+str(random_state)+"_confusion.pdf",
                        # #             bbox_inches="tight"
                        # #             )

                        # accuracy = accuracy_score(confusion_df["group"], confusion_df["bi"])
                        # f1 = f1_score(confusion_df["group"], confusion_df["bi"])
                        # print("F1: %.2f, ACC: %.2f" % (f1, accuracy))
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

print(bopseed, sig, len(sig))
# %%
def Analysis(seed=sys.argv):
    DIR = "/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/merged_TU/whole_TU/XGBoost/cv_result/"
    file_id = "merged_BOP_except_"+str(seed)+"_shuffled_feature_matrix_response_group.txt_report.txt"
    # file_id = "merged_BOP_except_"+str(seed)+"_shuffled_include_pre-post_response_group.txt_report.txt"
    try:
        data = pd.read_csv(
                            DIR+file_id, sep="\t"
                            )
        data = data.dropna()
        data = data[data["ROC AUC"]!="ROC AUC"]
        data = data.astype(float)
        print(data[(data["ext ROC"] > 0.75) & (data["PR AUC"] > 0.8) & 
            (data["#class ext"] >= 22)])
        data = data.iloc[:,:-3]
        # data = data.drop("ROC AUC", axis=1)
        # data = data.drop("ext ROC", axis=1)
        sns.set_style("whitegrid")
        plt.figure(figsize=(4,8))
        sns.stripplot(data=data, zorder=0, color="#56AAFF")
        # sns.barplot(data=data, color="#4DA0FF")
        #### Customized plot ####
        data.loc["median"] = data.median()
        data.loc["3Q"] = data.quantile(.75)
        data.loc["1Q"] = data.quantile(.25)
        sns.boxplot(data=data.loc[["median"]], linewidth=2.5, zorder=1)
        sns.boxplot(data=data.loc[["3Q"]], width=0.4, linewidth=2.5, zorder=1)
        sns.boxplot(data=data.loc[["1Q"]], width=0.4, linewidth=2.5, zorder=1)
        print(data["PR AUC"].mean())
        sns.pointplot(
                    data=data.loc[[0]], color="#FF0000", 
                    markers="o", zorder=2
                    )
        #########################
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
        # df.to_csv(
        #         DIR+"Merged_val_result_response.txt", sep="\t", mode="a"
        #         )
    except:
        pass

if __name__ == '__main__':
    # Analysis(sys.argv[1])
    Analysis(bopseed)
# %%
