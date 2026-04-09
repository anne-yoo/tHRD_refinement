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
from scipy.stats import mannwhitneyu



sns.set(font = 'Arial')
sns.set_style("whitegrid")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


# %%
#^ 1. GO enrichment plot: acquired

path_list = ['responder_stable','responder_variable','nonresponder_stable','nonresponder_variable']
merged_df = pd.DataFrame(columns=["logq","type","Adjusted P-value"])
genes = pd.DataFrame(columns=['Genes'])

for path in path_list:
    data = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/'+path+'_Wilcoxon_GOenrichment.txt',sep='\t') 
    #data = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/MW/'+path+'_RpreNRpreMWDUT_GOenrichment.csv',sep=',') #MW
    
    data["logq"] = -np.log10(data["Adjusted P-value"])
    data = data[data["Gene_set"]=="GO_Biological_Process_2021"]
    
    
    ##* Select DNA repair related Terms 
    # data = data[(data["Term"].str.contains("repair", case=False)) | (data["Term"].str.contains("DNA Damage")) 
    #             | (data["Term"].str.contains("DNA metabolic")) | (data["Term"].str.contains("(GO:0006260)"))
    #             | (data["Term"].str.contains("DNA duplex unwinding"))]
    
    #data = data[(data["Term"].str.contains("repair", case=False)) | (data["Term"].str.contains("DNA Damage", case=False))]
    
    data = data[(data["Term"].str.contains("repair", case=False))]
    # ##################################

    df = data[["logq","Adjusted P-value"]]
    df = df.astype(float)
    
    #### Term 확인을 위한 scatter plot (FDR < 0.1 ONLY)
    if data[data["Adjusted P-value"]<=0.01].shape[0] > 0:
        
        if data[data["Adjusted P-value"]<=0.01].shape[0] < 3:
            every_nth = 1
            sns.set_style("whitegrid")
            plt.figure(figsize=(6,1.5))
            ax = sns.scatterplot(
                                x="logq",y="Term",data=data[data["Adjusted P-value"]<=0.01],
                                s=130, edgecolor="None",
                                color="#FF8A8A"
                                )

            plt.xlim(left=1.0)
            print(data[data["Adjusted P-value"]<=0.01].shape[0], path)
            print(data[data["Adjusted P-value"]<=0.01].shape[0], path)
        
        else:
            # https://stackoverflow.com/questions/6682784/reducing-number-of-plot-ticks
            every_nth = 1
            sns.set_style("whitegrid")
            plt.figure(figsize=(6,6))
            ax = sns.scatterplot(
                                x="logq",y="Term",data=data[data["Adjusted P-value"]<=0.010],
                                s=130, edgecolor="None",
                                color="#FF8A8A"
                                )

            plt.xlim(left=1.0)
            print(data[data["Adjusted P-value"]<=0.01].shape[0], path)
            print(data[data["Adjusted P-value"]<=0.01].shape[0], path)
        
        
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
        
        plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/FDRcorrection/001_GOenrichment_"+path+".pdf", bbox_inches="tight")
        
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

plt.figure(figsize=(12,5))
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
                x=input_df[input_df["Adjusted P-value"]<=0.01].index.tolist(), 
                y="logq", data=input_df[input_df["Adjusted P-value"]<=0.01],
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
plt.xlim(-10,200)
plt.xticks([15,65,115,165],["R (stable)", "R (variable)","NR (stable)","NR (variable)"], fontsize=13)
plt.ylabel("$-Log_{10}$FDR", fontsize=14)
plt.axhline(2.0, color="#CACACA", linestyle="--")
plt.title("Enrichment of DNA repair genes", fontsize=15, position=(0.5, 1.0+0.02))


plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/FDRcorrection/001_bubbleplot.pdf", bbox_inches="tight")



#%%
#####^### DNA repair-related PARPi-derived DUTs ######

dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t')
enr = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_Wilcoxon_GOenrichment.txt', sep='\t')

dut.columns = ['transcript','pval','Gene Symbol', 'log2FC']
dut = dut[dut['pval']<0.05]
dutlist = dut['transcript'].tolist()

transexp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt',sep='\t', index_col=0)
#filtered_trans = transexp.loc[(transexp > 0).sum(axis=1) >= transexp.shape[1]*0.3]
transexp['Gene Symbol'] = transexp.index.str.split("-",1).str[1]
transexp = transexp[transexp['Gene Symbol']!='-']

data = enr
data = data[data["Adjusted P-value"] <= 0.01]
# data = data[(data["Term"].str.contains("repair")) | (data["Term"].str.contains("DNA damage")) 
#                 | (data["Term"].str.contains("DNA metabolic")) | (data["Term"].str.contains("(GO:0006260)"))
#                 | (data["Term"].str.contains("DNA duplex unwinding"))]

#data = data[(data["Term"].str.contains("repair", case=False)) | (data["Term"].str.contains("DNA Damage", case=False))]
data = data[(data["Term"].str.contains("repair", case=False))]

dnarepairgenes = [gene for sublist in data['Genes'].str.split(';') for gene in sublist]
dnarepairgenes = set(dnarepairgenes)

#%%
dut2 = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/nonresponder_stable_DUT_Wilcoxon.txt', sep='\t')
enr2 = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/nonresponder_stable_Wilcoxon_GOenrichment.txt', sep='\t')

dut2.columns = ['transcript','pval','Gene Symbol', 'log2FC']
dut2 = dut2[dut2['pval']<0.05]
dutlist2 = dut2['transcript'].tolist()

data2 = enr2
data2 = data2[data2["Adjusted P-value"] <= 0.01]
# data = data[(data["Term"].str.contains("repair")) | (data["Term"].str.contains("DNA damage")) 
#                 | (data["Term"].str.contains("DNA metabolic")) | (data["Term"].str.contains("(GO:0006260)"))
#                 | (data["Term"].str.contains("DNA duplex unwinding"))]

#data = data[(data["Term"].str.contains("repair", case=False)) | (data["Term"].str.contains("DNA Damage", case=False))]
data2 = data2[(data2["Term"].str.contains("repair", case=False))]

dnarepairgenes2 = [gene for sublist in data2['Genes'].str.split(';') for gene in sublist]
dnarepairgenes2 = set(dnarepairgenes2)

onlyrespondergenes = dnarepairgenes - dnarepairgenes2



#%%
dut_baseline = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/baseline/stable_DUT_MW.txt', sep='\t')
dut_baseline.columns = ['transcript','pval','Gene Symbol']
dut_baseline = dut_baseline[dut_baseline['pval']<0.05]

baseline_in_dnarepairgenes = set(dut_baseline.loc[dut_baseline['Gene Symbol'].isin(dnarepairgenes),'transcript'])


dut_parp = dut[dut['pval']<0.05]

parp_in_dnarepairgenes = set(dut_parp.loc[dut_parp['Gene Symbol'].isin(dnarepairgenes), 'transcript'])

#%%
#####****** Venn Diagram ################
from matplotlib_venn import venn2


plt.figure(figsize=(6,6))
sns.set_style("white")
vd2 = venn2([parp_in_dnarepairgenes, baseline_in_dnarepairgenes],set_labels=('PARPi-derived', 'Baseline'),set_colors=('#F8CF6A','#74B2D4'), alpha=0.8)

for text in vd2.set_labels:  #change label size
    text.set_fontsize(13)
for text in vd2.subset_labels:  #change number size
    text.set_fontsize(16)
    
vd2.get_patch_by_id('11').set_color('#C8C452')
vd2.get_patch_by_id('11').set_alpha(1)

#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/001_parp_baseline_Venn.pdf", bbox_inches="tight")
plt.show()
















#%%
transexp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt',sep='\t', index_col=0)

combined = list(parp_in_dnarepairgenes.union(baseline_in_dnarepairgenes))

dut_TU = transexp[transexp.index.isin(combined)]


inputTU = dut_TU
t_inputTU = inputTU.T

f_inputTU = t_inputTU
sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.csv', sep=',')
f_inputTU['response'] = list(sampleinfo['response'])

#####SAVE####
##f_inputTU.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/modelinput/premodel_input_combined.txt', sep='\t')


# %%
