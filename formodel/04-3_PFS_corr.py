#%%
""" 
stable gene (pre vs post) 의 DTU
variable gene (pre vs post) 의 DTU 선별을 위한 script
"""
from sklearn.decomposition import NMF,PCA
from scipy.stats import ttest_1samp, wilcoxon, ttest_ind, mannwhitneyu, fisher_exact, binom_test, pearsonr, gmean, zscore
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
from statsmodels.stats.multitest import multipletests as ssm
import gseapy as gp
import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
import seaborn as sns
import SET_arial
import sys

DIR="/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/pre_post_TU/final_data/MW_result/"
# OUT="/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/pre_post_TU/final_data/221229_strategy/"
data = pd.read_csv(
            # DIR+"DNA_repair_related_term_FDR10_stable_DUTs.txt",
            DIR+"model_input/1st_filtered_TU.txt",
            sep="\t", index_col="Unnamed: 0"
            )   # Return input data
# DUT_rank = data.copy()
# DUT_rank["abs_tu_diff"] = np.abs(DUT_rank["tu_diff"])
# DUT_rank = DUT_rank.sort_values(by="abs_tu_diff", ascending=False)
# DUT_rank[["gene","abs_tu_diff"]].to_csv(
#                                     OUT+"DUT_sorting.txt",
#                                     sep="\t", index=False
#                                     )
target_gene = data["gene"].unique()

## Major TU 분석했던 동일한 유전자의 minor DUTs 로 시작
# %%
def Make_tu_df(d_type):
    if d_type == "whole":
        # DIR = '/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/pre_post_TU/final_data/'
        DIR = "/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/pre_post_TU/final_data/MW_result/model_input/"
        tu_df = pd.read_csv(
            # DIR+"minor_prop.txt",
            DIR+"MW_RES_INT_stable_diff_minor_prop.txt",
            sep="\t"
            )

        tu_df["gene"] = tu_df["gene_ENST"].str.split("-", 1).str[1]
        tu_df = tu_df.set_index("gene_ENST")
        samples = tu_df.columns.tolist()[:-1]
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
        
        samples.append("gene")
        tu_df.columns = samples
        #### 230107 
        target_atu = pd.DataFrame(columns=tu_df.columns)
        for gene in target_gene:
            target_atu = target_atu.append(tu_df[tu_df["gene"] == gene])
        tu_df = target_atu.copy()
        ####
        tu_df = tu_df.iloc[:,:-1]
    
    else:
        tu_df = pd.read_csv(
                            DIR+"RES_INT_MW_input.txt",
                            sep="\t", index_col="gene_ENST"
                            )
    # tu_df = tu_df.reindex(data.index)   # Responder (removed NR DUTs) stable DUTs 를 사용
    tu_df = tu_df.drop(["SV-OV-P055-atD","SV-OV-P080-atD","SV-OV-P134-atD","SV-OV-P137-atD",\
            "SV-OV-P143-atD","SV-OV-P164-atD","SV-OV-P174-bfD","SV-OV-P250-atD"], axis=1)
    tu_df = tu_df.T

    return tu_df

tu_df = Make_tu_df("whole")
# %%
## 각 aTU 중에서 interval 과 corr 이 강한 것을 선정
def add_clinic():
    """ TU matrix 에 clinical info 를 추가
    Returns: TU + clinical dataframe
    """

    tu_df["10ID"] = tu_df.index.str[:10]
    group_info = pd.read_csv(
                        "/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/clinical_info_gHRD/"+
                        "processed_info_with_originalID.txt", sep="\t", index_col="GID"
                        )
    group_info = group_info[["OM/OS", "ORR", "drug", "interval"]]
    group_info.columns = ["OM/OS", "group", "drug", "interval"]
    group_info["drug"] = group_info["drug"].str.replace("Olapairb","Olaparib")
    group_info["drug"] = group_info["drug"].str.replace("Olapairib","Olaparib")
    group_info["drug"] = group_info["drug"].str.replace("olaparib","Olaparib")
    group_info = group_info.dropna()
    group_info = group_info.drop("drug", axis=1)
    group_info["10ID"] = group_info.index.str.replace("F","T").str[:10]
    group_info["10ID"] = group_info["10ID"].str.replace("T","P")
    group_info = group_info.drop_duplicates()
    group_info = group_info.set_index("10ID")
    df = pd.merge(tu_df, group_info, left_on="10ID", right_index=True)
    df["pre-post"] = df.index.str[-3:]
    df["pre-post"] = df["pre-post"].apply(lambda x : 0 if x.startswith("bf") else 1)
    # 0 = bfD, 1 = atD

    return df


input_df = add_clinic()
input_df = input_df[input_df["pre-post"]==0]
## Select only responder group
# %%
def Do_corr(direction):
    """ PFS ~ minor DUTs corr

    Args:
        direction (_type_): pos (minor 증가(==기능성 감소) PFS는 증가), 
                            abs (방향성 무관, association 에 주목)
    """
    corr_result = {}
    k = 0
    for i in input_df.columns.tolist()[:-5]:    # Each transcripts
        corr_result[i] = []
        r,p = pearsonr(input_df[i], input_df["interval"])
        corr_result[i].append(r)
        corr_result[i].append(p)
        corr_result[i].append(np.mean(input_df[i]))
        # if k < 10 and np.abs(r) > 0.3:    # Condition 1
        # if k < 1 and np.abs(r) > 0.3 and np.mean(input_df[i]) > 0.1:    # Condition 2
        #     sns.set_style("whitegrid")
        #     plt.figure(figsize=(4,4))
        #     sns.regplot(x=i, y="interval", data=input_df)
        #     k += 1
        #     plt.show()
        #     plt.close()

    corr_result = pd.DataFrame.from_dict(corr_result, orient="index")
    corr_result.columns = ["r","p","mean_tu"]
    corr_result["abs R"] = np.abs(corr_result["r"])
    corr_result = corr_result.sort_values(by="abs R", ascending=False)
    # corr_result = corr_result[corr_result[2] >= 0.05]
    corr_result["gene"] = corr_result.index.str.split("-",1).str[1]

    if direction == "pos":
        corr_result = corr_result[corr_result["r"] > 0.0]
    else:
        corr_result = corr_result[corr_result["abs R"] > 0.15]
    
    corr_result.to_csv(
                        DIR+"model_input/"
                        +"PFS_"+direction+"_corr_minor_tu.txt", sep="\t"
                        )
    # corr_result.to_csv(
    #                 "/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/pre_post_TU/final_data/221229_strategy/"
    #                 +"enrichment_terms_"+direction+"_corr_minor_tu.txt", sep="\t"
    #                 )


Do_corr("pos")
Do_corr("abs")
# %%
sig_response = {}
for i in input_df.columns[:-5]:
    res = input_df[input_df["group"]==1][[i]]
    nr = input_df[input_df["group"]==0][[i]]
    u,p = mannwhitneyu(res, nr, alternative="greater")
    if p < 0.05:
        sig_response[i] = []
        sig_response[i].append(p)
        sig_response[i].append(np.mean(res)[0])
        sig_response[i].append(np.mean(nr)[0])
print(res.shape)
print(nr.shape)
sig_response = pd.DataFrame.from_dict(sig_response, orient="index")
## response 비교하여 feature selection 하는 함수 만들기
sig_response.columns = ["mwp","res_mean","nr_mean"]
sig_response.to_csv(
                    DIR+"model_input/response_associated_aTU.txt",
                    sep="\t"
                    )
# %%
