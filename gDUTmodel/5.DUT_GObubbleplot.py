
#%%
#! this code is.....
## 20230518 | 20230920
"""
R pre vs. NR pre: stable/variable gene group DUT GO term 확인 (뿅뿅이 그림)
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
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import plotly.express as px
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.stats.multitest import multipletests


sns.set(font = 'Arial')
sns.set_style("whitegrid")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# %%
#^ 0. Go enrichment

import gseapy as gp

###*enrichR GO enrichment analysis

filelist = ['responder_stable','responder_variable','nonresponder_stable','nonresponder_variable']

for i in range(4):
    path2 = '/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/'
    path_new = '/home/jiye/jiye/copycomparison/gDUTresearch/202401_analysis/DUT/'

    results = pd.read_csv(path2+filelist[i]+'_DUT_Wilcoxon.txt', sep='\t')

    results.rename(columns = {"Unnamed: 0": "transcript"}, inplace = True)
    
    #pcut = results[results['pval']<0.05]['gene']
    pcut = results[(results['p_value']<0.05) & (np.abs(results['log2FC'])>1.5)]['Gene Symbol']
    pcut = pcut.drop_duplicates()
    glist = pcut.squeeze().str.strip().to_list()

    enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                    gene_sets=['GO_Biological_Process_2021'], #KEGG_2021_Human
                    organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                    outdir=None, # don't write to disk
    )

    enrresult = enr.results.sort_values(by=['Adjusted P-value']) 
    print(filelist[i], len(enrresult[enrresult['Adjusted P-value']<0.05]))
    enrresult.to_csv(path_new+filelist[i]+'_Wilcoxon_GOenrichment_FC_15.txt', sep='\t')
    
#%%
enrresult

#%%
#^ 1. GO enrichment plot: acquired

path_list = ['responder_stable','responder_variable','nonresponder_stable','nonresponder_variable']
merged_df = pd.DataFrame(columns=["logq","type"])
genes = pd.DataFrame(columns=['Genes'])

for path in path_list:
    #data = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/'+path+'_Wilcoxon_GOenrichment.txt',sep='\t') 
    data = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202401_analysis/DUT/'+path+'_Wilcoxon_GOenrichment_FC_15.txt',sep='\t') 
    #data = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/MW/'+path+'_RpreNRpreMWDUT_GOenrichment.csv',sep=',') #MW
    
    data["logq"] = -np.log10(data["Adjusted P-value"])
    data = data[data["Gene_set"]=="GO_Biological_Process_2021"]
    
    sig = data[data["Adjusted P-value"] <= 0.01]
    
    ##* Select DNA repair related Terms 
    #data = data[(data["Term"].str.contains("repair", case=False)) | (data["Term"].str.contains("DNA Damage")) 
    #             | (data["Term"].str.contains("DNA metabolic")) | (data["Term"].str.contains("(GO:0006260)"))
    #             | (data["Term"].str.contains("DNA duplex unwinding"))]
    
    data = data[(data["Term"].str.contains("repair", case=False)) | (data["Term"].str.contains("DNA Damage", case=False))]
    
    #data = data[(data["Term"].str.contains("repair", case=False))]
    # ##################################

    df = data[["logq"]]
    df = df.astype(float)
    
    #### Term 확인을 위한 scatter plot (FDR < 0.1 ONLY)
    if data[data["logq"]>=2.0].shape[0] > 0:
        
        if data[data["logq"]>=2.0].shape[0] < 3:
            every_nth = 1
            sns.set_style("whitegrid")
            plt.figure(figsize=(4,1))
            ax = sns.scatterplot(
                                x="logq",y="Term",data=data[data["logq"]>=2.0],
                                s=130, edgecolor="None",
                                color="#FF8A8A"
                                )

            plt.xlim(left=1.5, right=4.5)
            print(data[data["logq"]>=2.0].shape[0], path)
            print(data[data["logq"]>=2.0].shape[0], path)
        
        else:
            # https://stackoverflow.com/questions/6682784/reducing-number-of-plot-ticks
            every_nth = 1
            sns.set_style("whitegrid")
            plt.figure(figsize=(4,2))
            ax = sns.scatterplot(
                                x="logq",y="Term",data=data[data["logq"]>=2.0],
                                s=130, edgecolor="None",
                                color="#FF8A8A"
                                )

            plt.xlim(left=1.5, right=10)
            print(data[data["logq"]>=1.0].shape[0], path)
            print(data[data["logq"]>=1.0].shape[0], path)
        
        
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

        
        
        genedf = pd.DataFrame(data[data["logq"]>=2.0]['Genes'])
        genedf.columns = ['Genes']
        genes = pd.concat([genes,genedf])
        
        #plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/fig2d_GOenrichment_"+path+".pdf", bbox_inches="tight")
        
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
col_annot = {"responder_stable":"#FF3F3F", "responder_variable":"#FF8181", "nonresponder_stable":"#FF9E2E", "nonresponder_variable":"#FFC37E"}

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
                x=input_df[input_df["logq"]>2].index.tolist(), 
                y="logq", data=input_df[input_df["logq"]>2],
                s=150, alpha=1.,
                edgecolor="#000000",
                hue="type", palette=col_annot
                )

plt.legend("", frameon=False)
sns.despine(
            top=True, right=True
            )
sns.despine(
            top=True, right=True
            )
plt.xlim(-10,260)
plt.xticks([35,95,155,215],["AR stable", "AR variable","IR stable","IR variable"], fontsize=13)
plt.ylabel("$-Log_{10}$FDR", fontsize=14)
plt.axhline(2.0, color="#CACACA", linestyle="--")
#plt.title("Enrichment of DNA repair genes", fontsize=15, position=(0.5, 1.0+0.02))


plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/bubbleplot.pdf", bbox_inches="tight")

#%%

enr = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_Wilcoxon_GOenrichment.txt', sep='\t')

data = enr
data = data[data["Adjusted P-value"] <= 0.1]
data = data[(data["Term"].str.contains("repair", case=False)) | (data["Term"].str.contains("DNA Damage", case=False))]

#data = data[(data["Term"].str.contains("homologous recombination", case=False))  | (data["Term"].str.contains("GO:2000779"))]

dnarepairgenes = [gene for sublist in data['Genes'].str.split(';') for gene in sublist]
dnarepairgenes = set(dnarepairgenes)
#%%































#%%
#^ 0. Go enrichment

import gseapy as gp

###*enrichR GO enrichment analysis

filelist = ['responder','nonresponder']

for i in range(2):
    path2 = '/home/jiye/jiye/copycomparison/gDUTresearch/202310_analysis/DUT/'

    results = pd.read_csv(path2+filelist[i]+'_wholegene_DUT_Wilcoxon.txt', sep='\t')

    results.rename(columns = {"Unnamed: 0": "transcript"}, inplace = True)
    
    #pcut = results[results['pval']<0.05]['gene']
    pcut = results[results['p_value']<0.01]['Gene Symbol']
    pcut = pcut.drop_duplicates()
    glist = pcut.squeeze().str.strip().to_list()

    enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                    gene_sets=['GO_Biological_Process_2021','KEGG_2021_Human'], 
                    organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                    outdir=None, # don't write to disk
    )

    enrresult = enr.results.sort_values(by=['Adjusted P-value']) 
    print(filelist[i], len(enrresult[enrresult['Adjusted P-value']<0.05]))
    print(filelist[i], len(enrresult[enrresult['P-value']<0.05]))
    enrresult.to_csv(path2+filelist[i]+'_wholegene_Wilcoxon_GOenrichment.txt', sep='\t')
    
#%%
enrresult

#%%
#^ 1. GO enrichment plot: acquired

path_list = ['responder','nonresponder']
merged_df = pd.DataFrame(columns=["logq","type"])
genes = pd.DataFrame(columns=['Genes'])

for path in path_list:
    data = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202310_analysis/DUT/'+path+'_wholegene_Wilcoxon_GOenrichment.txt',sep='\t') 
    #data = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/MW/'+path+'_RpreNRpreMWDUT_GOenrichment.csv',sep=',') #MW
    
    data["logq"] = -np.log10(data["Adjusted P-value"])
    data = data[data["Gene_set"]=="GO_Biological_Process_2021"]
    
    sig = data[data["Adjusted P-value"] <= 0.05]
    
    ##* Select DNA repair related Terms 
    data = data[(data["Term"].str.contains("repair")) | (data["Term"].str.contains("DNA damage")) 
                | (data["Term"].str.contains("DNA metabolic")) | (data["Term"].str.contains("(GO:0006260)"))
                | (data["Term"].str.contains("DNA duplex unwinding"))]
    # ##################################

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
        
        plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202310_analysis/GOenrichment_wholegene_"+path+".pdf", bbox_inches="tight")
        
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
col_annot = {"responder":"#FF3F3F","nonresponder":"#FF9E2E"}

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
plt.xticks([30,170],["Responder", "Nonresponder"], fontsize=12)
plt.ylabel("$-Log_{10}$FDR", fontsize=14)
plt.axhline(1.0, color="#CACACA", linestyle="--")
plt.title("Enrichment of DNA repair genes", fontsize=15, position=(0.5, 1.0+0.02))

plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202310_analysis/wholegene_bubbleplot.pdf", bbox_inches="tight")

# %%
