#%%
""" 
Stable gene 의 DTU functionality 확인하기
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
#import SET_arial
import sys

DIR = "/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/pre_post_TU/final_data/MW_result/"
# path = sys.argv[1]
# path_list = ["mw_stable_main/", "mw_stable_RES_INT/", "mw_stable_NR/",
#             "mw_variable_main/", "mw_variable_RES_INT/", "stable_mt_GOenrichment.csv",
#             "stable_resint_GOenrichment.csv", "stable_nr_GOenrichment.csv",
#             "variable_mt_GOenrichment.csv", "variable_resint_GOenrichment.csv"]

path_list = ["MW_RES_INT_stable/", "MW_RES_INT_variable/",
            "MW_NR_stable/", "MW_NR_variable/", "stable_R",
            "variable_R", "stable_NR", "variable_NR"]
merged_df = pd.DataFrame(columns=["logq","type"])
for path in path_list:
    if path.startswith("MW"):
        data = pd.read_csv(
                            DIR+path+"GO_Biological_Process_2021.enrichr.enrichr.reports.txt",
                            sep="\t"
        )
        data["logq"] = -np.log10(data["Adjusted P-value"])
    else:
        data = pd.read_csv(
                            # "/home/jiye/jiye/copycomparison/OC_transcriptome/satuRn_results/new/"+
                            "/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/output/discovery/"+
                            path+"_GOenrichment.csv", sep=","
                            )
        data["logq"] = -np.log10(data["Adjusted P-value"])
        data = data[data["Gene_set"]=="GO_Biological_Process_2021"]
    sig = data[data["Adjusted P-value"] <= 0.05]
    ## Select DNA repair related Terms 
    data = data[(data["Term"].str.contains("repair")) | (data["Term"].str.contains("DNA damage")) 
                | (data["Term"].str.contains("DNA metabolic")) | (data["Term"].str.contains("(GO:0006260)"))
                | (data["Term"].str.contains("DNA duplex unwinding"))]
    ##################################
    df = data[["logq"]]
    df = df.astype(float)
    #### Term 확인을 위한 scatter plot (FDR < 0.1 ONLY)
    if data[data["logq"]>=1.0].shape[0] > 0:
        sns.set_style("whitegrid")
        plt.figure(figsize=(3,6))
        ax = sns.scatterplot(
                            x="logq",y="Term",data=data[data["logq"]>=1.0],
                            s=130, edgecolor="None",
                            color="#FF8A8A"
                            )

        plt.xlim(left=1.0)
        print(data[data["logq"]>=1.0].shape[0], path)
        if data[data["logq"]>=1.0].shape[0] < 10:
            every_nth = 1
        elif 10 < data[data["logq"]>=1.0].shape[0] < 25:
            # https://stackoverflow.com/questions/6682784/reducing-number-of-plot-ticks
            every_nth = 4
        elif 25 < data[data["logq"]>=1.0].shape[0]:
            every_nth = 6
        for n, label in enumerate(ax.yaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)
        if path.split("/")[0] == "stable_R":
            plt.title("satuRn", fontsize=15)
        if path.split("/")[0] == "MW_RES_INT_stable":
            plt.title("Mann-whitney U", fontsize=15)
        plt.xlabel("$-Log_{10}$FDR", fontsize=13)
        plt.ylabel("")
        # plt.savefig(
        #             DIR+"figure/"+path.split("/")[0]+".pdf",
        #             bbox_inches="tight"
        #             )
    ####
    df["type"] = path   # DNA repair related Terms (regardless FDR)
    # This dataframe is used below def
    merged_df = merged_df.append(df)
merged_df = merged_df.reset_index()
merged_df = merged_df.drop("index", axis=1)
merged_df["analysis_type"] = merged_df["type"].apply(
                                                    lambda x : 1 if "stable" in x and not "NR" in x 
                                                    else (0 if "NR" in x else 0)
                                                    )
merged_df = merged_df.reset_index()
merged_df = merged_df.sort_values(by="analysis_type")
print(merged_df)
# %%
merged_df["type"] = merged_df["type"].str.replace("stable_NR", "satuRn NR stable")
merged_df["type"] = merged_df["type"].str.replace("stable_R", "satuRn R stable")
merged_df["type"] = merged_df["type"].str.replace("variable_NR", "satuRn NR variable")
merged_df["type"] = merged_df["type"].str.replace("variable_R", "satuRn R variable")
merged_df["type"] = merged_df["type"].str.replace("MW_RES_INT_stable/", "MW LT stable")
merged_df["type"] = merged_df["type"].str.replace("MW_NR_stable/", "MW ST stable")
merged_df["type"] = merged_df["type"].str.replace("MW_RES_INT_variable/", "MW LT variable")
merged_df["type"] = merged_df["type"].str.replace("MW_NR_variable/", "MW ST variable")
print(merged_df)
# %%
def Draw(tool):
    plt.figure(figsize=(13,5))
    if tool == "MW":
        col_annot = {"MW LT stable":"#FF3F3F", "MW LT variable":"#FF8181", 
                    "MW ST stable":"#FF9E2E", "MW ST variable":"#FFC37E"
                    }
        input_df = merged_df[merged_df["type"].str.startswith("MW")]
    else:
        col_annot = {"satuRn R variable":"#FF8181", "satuRn R stable":"#FF3F3F", 
                    "satuRn NR stable":"#FF9E2E","satuRn NR variable":"#FFC37E"}
        input_df = merged_df[merged_df["type"].str.startswith("satuRn")]
        input_df = input_df.sort_index()
        input_df = input_df.reset_index()
        input_df = input_df.drop("index",axis=1)
        print(input_df)

    sns.set_style("white")
    sns.scatterplot(
                    x=input_df.index.tolist(), 
                    y="logq", data=input_df,
                    s=150, alpha=.6,
                    edgecolor="None",
                    hue="type", palette=col_annot
                    )
    sns.scatterplot(
                    x=input_df[input_df["logq"]>1].index.tolist(), 
                    y="logq", data=input_df[input_df["logq"]>1],
                    s=150, alpha=1.,
                    edgecolor="#000000",
                    hue="type", palette=col_annot
                    )

    plt.legend("", frameon=False)
    sns.despine(
                top=True, right=True, trim=True
                )
    sns.despine(
                top=True, right=True, trim=True
                )
    plt.xticks([30,100,170,245],["Responder\nstabe genes","Responder\nvariabe genes",
                                "Non-responder\nstabe genes","Non-responder\nvariabe genes"])
    plt.ylabel("$-Log_{10}$FDR", fontsize=14)
    plt.xlabel(tool, fontsize=13)
    plt.axhline(1.0, color="#CACACA", linestyle="--")

    OUT = DIR+"figure/"
    # plt.savefig(
    #             OUT+"GO_merged_"+tool+".pdf",
    #             bbox_inches="tight"
    #             )

Draw("MW")
Draw("satuRn")
# %%
