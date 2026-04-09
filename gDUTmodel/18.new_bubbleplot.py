#%%
#! this code is.....
## 20230815
"""
R pre vs. NR pre: stable/variable gene group DUT GO term 확인 (뿅뿅이 그림)
0. Go enrichment
1. GO enrichment plot
2. FDR 뿅뿅이 plot
3. [BY SatuRn] acquired vs. innate: stable gene group에서 venn diagram
"""

#%%
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
import re
import matplotlib.cm as cm
from matplotlib.pyplot import gcf

# %%
##^^^ 0. Go enrichment

import gseapy as gp
###*enrichR GO enrichment analysis

path_list = ['acquired/DUT_acquired_stable','acquired/DUT_acquired_variable','innate/DUT_innate_stable','innate/DUT_innate_variable']

path2 = '/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/result_DUTDEG/'

for i in range(0,1):    
    results = pd.read_csv(path2+path_list[i]+'_SatuRnResult.txt')

    results.rename(columns = {"Unnamed: 0": "isoform_id"}, inplace = True)
    results['gene_id'] = results['isoform_id'].str.split("-",1).str[1]

    pcut = results[results['pval']<0.05]['gene_id']
    pcut = pcut.drop_duplicates()
    glist = pcut.squeeze().str.strip().to_list()

    enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                    gene_sets=['GO_Biological_Process_2021'], 
                    organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                    outdir=None, # don't write to disk
    )

    enrresult = enr.results.sort_values(by=['Adjusted P-value']) 

    ##* file saving
    #enrresult.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/result_DUTDEG/'+path_list[i]+'_GOenrichment.txt', index=False)                               

#%%
#^ 1. GO enrichment plot: acquired
path_list = ['acquired/DUT_acquired_stable','acquired/DUT_acquired_variable','innate/DUT_innate_stable','innate/DUT_innate_variable']
merged_df = pd.DataFrame(columns=["logq","type"])
genes = pd.DataFrame(columns=['Genes'])

for path in path_list:
    data = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/result_DUTDEG/'+path_list[i]+'_GOenrichment.txt',sep=',') #satuRn
    #data = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/MW/'+path+'_RpreNRpreMWDUT_GOenrichment.csv',sep=',') #MW
    
    data["logq"] = -np.log10(data["Adjusted P-value"])
    data = data[data["Gene_set"]=="GO_Biological_Process_2021"]
    
    sig = data[data["Adjusted P-value"] <= 0.05]
    
    ##* Select DNA repair related Terms 
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
        if path== "acquired_stable":
            plt.title("R pre vs. R post (stable genes)", fontsize=14, position=(0.5, 1.0+0.02))
        if path == "acquired_variable":
            plt.title("R pre vs. R post (variable genes)", fontsize=14, position=(0.5, 1.0+0.02))
        if path== "comp_acquired_stable":
            plt.title("NR pre vs. NR post (stable genes)", fontsize=14, position=(0.5, 1.0+0.02))
        if path== "comp_acquired_variable":
            plt.title("NR pre vs. NR post (variable genes)", fontsize=14, position=(0.5, 1.0+0.02))
        plt.xlabel("$-Log_{10}$FDR", fontsize=13)
        plt.ylabel("")
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(13)

        
        
        genedf = pd.DataFrame(data[data["logq"]>=1.0]['Genes'])
        genedf.columns = ['Genes']
        genes = pd.concat([genes,genedf])
        
        #plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/figures/GOenrichment_"+path+".pdf", bbox_inches="tight")
        plt.show()
    else:
        print("No significant data!!! in", path)
        
    df["type"] = path
    merged_df = merged_df.append(df)


merged_df = merged_df.reset_index()
merged_df = merged_df.drop("index", axis=1)
merged_df = merged_df.reset_index()
merged_df = merged_df.sort_values(by="type")
print(merged_df)
# %%
##^ 2. FDR 뿅뿅이 plot (acquired)

plt.figure(figsize=(13,5))
col_annot = {"acquired/DUT_acquired_stable":"#FF3F3F", "acquired/DUT_acquired_variable":"#FF8181", "innate/DUT_innate_stable":"#FF9E2E", "innate/DUT_innate_variable":"#FFC37E"}

input_df = merged_df
input_df = input_df.sort_index()
input_df = input_df.reset_index()
input_df = input_df.drop("index",axis=1)

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
plt.xticks([30,100, 170, 240],["R pre vs. R post (stable)", "R pre vs. R post (variable)","NR pre vs. NR post (stable)","NR pre vs.NR post (variable)"], fontsize=12)
plt.ylabel("$-Log_{10}$FDR", fontsize=14)
plt.axhline(1.0, color="#CACACA", linestyle="--")
plt.title("Enrichment of DNA repair genes", fontsize=15, position=(0.5, 1.0+0.02))

# %%
