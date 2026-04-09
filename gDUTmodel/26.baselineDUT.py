#%%
#! this code is.....
"""
R pre vs. NR pre
Mann-Whitney
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
from scipy.stats import mannwhitneyu



sns.set(font = 'Arial')
sns.set_style("whitegrid")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


# %%
#####^^^ Paired Wilcoxon DEG responder / non-responder ####

geneexp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_gene_exp_TPM_symbol.txt',sep='\t', index_col=0)

genesym = geneexp[['Gene Symbol']]
sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.csv', sep=',')
FL_list = list(sampleinfo[sampleinfo['line_binary']=='N-FL']['sample_full'])
sampleinfo = sampleinfo[sampleinfo['line_binary']=='N-FL']

geneexp = geneexp.loc[:,FL_list]

responder = sampleinfo[sampleinfo['response']==1]['sample_full']
nonresponder = sampleinfo[sampleinfo['response']==0]['sample_full']
responder = responder.to_list()
nonresponder = nonresponder.to_list()

R_df = geneexp[responder]
NR_df = geneexp[nonresponder]

R_pre = R_df[R_df.columns[1::2]]
NR_pre = NR_df[NR_df.columns[1::2]]


deg_pval = []

for gene in R_pre.index:
    u_stat, p_val = mannwhitneyu(R_pre.loc[gene], NR_pre.loc[gene], alternative='two-sided')
    deg_pval.append(p_val)

# Create a new DataFrame with geneid and respective p-values
result_df = pd.DataFrame({
    'p_value': deg_pval,
})

result_df.index = R_pre.index
result_df = pd.merge(result_df,genesym, how='inner', left_index=True, right_index=True)

#result_df.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DEG/baseline/non1L_MW_DEGresult.txt', sep='\t')
    




# %%
#####^^^ Paired Wilcoxon DUT responder / non-responder ####
degresult = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DEG/baseline/non1L_MW_DEGresult.txt', sep='\t')
DEGlist = set(degresult[degresult['p_value']<0.05]['Gene Symbol'])
nonDEGlist = set(degresult[degresult['p_value'] > 0.05]['Gene Symbol'])

print(" variable: ", len(DEGlist))
print(" stable: ", len(nonDEGlist))

filtered_trans = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0)

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.csv', sep=',')

FL_list = list(sampleinfo[sampleinfo['line_binary']=='N-FL']['sample_full'])
sampleinfo = sampleinfo[sampleinfo['line_binary']=='N-FL']

responder = sampleinfo[sampleinfo['response']==1]['sample_full']
nonresponder = sampleinfo[sampleinfo['response']==0]['sample_full']
responder = responder.to_list()
nonresponder = nonresponder.to_list()

filtered_trans = filtered_trans.loc[:,FL_list]

R_tu = filtered_trans[responder]
NR_tu = filtered_trans[nonresponder]

R_pre_tu = R_tu[R_tu.columns[1::2]]
R_pre_tu['Gene Symbol'] = R_pre_tu.index.str.split("-",2).str[1]

NR_pre_tu = NR_tu[NR_tu.columns[1::2]]
NR_pre_tu['Gene Symbol'] = NR_pre_tu.index.str.split("-",2).str[1]


##* variable
variable_R_pre = R_pre_tu[R_pre_tu['Gene Symbol'].isin(DEGlist)]
variable_NR_pre = NR_pre_tu[NR_pre_tu['Gene Symbol'].isin(DEGlist)]
variable_R_pre = variable_R_pre.iloc[:,:-1]
variable_NR_pre = variable_NR_pre.iloc[:,:-1]

variable_dut_pval = []

for tr in variable_R_pre.index:
    if set([a - b for a, b in zip(variable_R_pre.loc[tr,:].values, variable_NR_pre.loc[tr,:].values)]) != {0}: 
        u_stat, p_val = mannwhitneyu(variable_R_pre.loc[tr,:].values, variable_NR_pre.loc[tr,:].values, alternative='two-sided')
        variable_dut_pval.append(p_val)
    else:
        variable_dut_pval.append(1)
    
# Create a new DataFrame with geneid and respective p-values
variable_result = pd.DataFrame({
    'p_value': variable_dut_pval,
})

variable_result.index = variable_R_pre.index
variable_result['Gene Symbol'] = variable_result.index.str.split("-",1).str[1]


##* stable
stable_R_pre = R_pre_tu[R_pre_tu['Gene Symbol'].isin(nonDEGlist)]
stable_NR_pre = NR_pre_tu[NR_pre_tu['Gene Symbol'].isin(nonDEGlist)]
stable_R_pre = stable_R_pre.iloc[:,:-1]
stable_NR_pre = stable_NR_pre.iloc[:,:-1]

stable_dut_pval = []

for tr in stable_R_pre.index:
    if set([a - b for a, b in zip(stable_R_pre.loc[tr,:].values, stable_NR_pre.loc[tr,:].values)]) != {0}: 
        u_stat, p_val = mannwhitneyu(stable_R_pre.loc[tr,:].values, stable_NR_pre.loc[tr,:].values, alternative='two-sided')
        stable_dut_pval.append(p_val)
    else:
        stable_dut_pval.append(1)

# Create a new DataFrame with geneid and respective p-values
stable_result = pd.DataFrame({
    'p_value': stable_dut_pval,
})

stable_result.index = stable_R_pre.index
stable_result['Gene Symbol'] = stable_result.index.str.split("-",1).str[1]

variable_result.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/baseline/non1L_variable_DUT_MW.txt', sep='\t')
stable_result.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/baseline/non1L_stable_DUT_MW.txt', sep='\t')


# %%
######^^^^^^^^^^ GO plot ###################









# %%
#^ 0. Go enrichment
# %%
#^ 0. Go enrichment

import gseapy as gp

###*enrichR GO enrichment analysis

filelist = ['1L_stable','1L_variable','non1L_stable','non1L_variable']

for i in range(4):
    path2 = '/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/baseline/'

    results = pd.read_csv(path2+filelist[i]+'_DUT_MW.txt', sep='\t')

    results.rename(columns = {"Unnamed: 0": "transcript"}, inplace = True)
    
    #pcut = results[results['pval']<0.05]['gene']
    pcut = results[results['p_value']<0.05]['Gene Symbol']
    pcut = pcut.drop_duplicates()
    glist = pcut.squeeze().str.strip().to_list()

    enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                    gene_sets=['GO_Biological_Process_2021'], #KEGG_2021_Human
                    organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                    outdir=None, # don't write to disk
    )

    enrresult = enr.results.sort_values(by=['Adjusted P-value']) 
    print(filelist[i], len(glist))
    #print(filelist[i], len(enrresult[enrresult['Adjusted P-value']<0.05]))
    #enrresult.to_csv(path2+filelist[i]+'_MW_GOenrichment.txt', sep='\t')

#%%
enrresult

#%%
#^ 1. GO enrichment plot: baseline

path_list = ['1L_stable','1L_variable','non1L_stable','non1L_variable']
merged_df = pd.DataFrame(columns=["logq","type"])
genes = pd.DataFrame(columns=['Genes'])

for path in path_list:
    data = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/baseline/'+path+'_MW_GOenrichment.txt',sep='\t') 
    #data = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/MW/'+path+'_RpreNRpreMWDUT_GOenrichment.csv',sep=',') #MW
    
    data["logq"] = -np.log10(data["Adjusted P-value"])
    data = data[data["Gene_set"]=="GO_Biological_Process_2021"]
    
    sig = data[data["Adjusted P-value"] <= 0.05]
    
    ##* Select DNA repair related Terms 
    # data = data[(data["Term"].str.contains("repair", case=False)) | (data["Term"].str.contains("DNA Damage")) 
    #             | (data["Term"].str.contains("DNA metabolic")) | (data["Term"].str.contains("(GO:0006260)"))
    #             | (data["Term"].str.contains("DNA duplex unwinding"))]
    
    #data = data[(data["Term"].str.contains("repair", case=False)) | (data["Term"].str.contains("DNA Damage", case=False))]
    
    data = data[(data["Term"].str.contains("repair", case=False)) | (data["Term"].str.contains("replication fork", case=False))]
    # ##################################

    df = data[["logq"]]
    df = df.astype(float)
    
    #### Term 확인을 위한 scatter plot (FDR < 0.1 ONLY)
    if data[data["logq"]>=1.0].shape[0] > 0:
        
        if data[data["logq"]>=1.0].shape[0] < 4:
            every_nth = 1
            sns.set_style("whitegrid")
            plt.figure(figsize=(4,1))
            ax = sns.scatterplot(
                                x="logq",y="Term",data=data[data["logq"]>=1.0],
                                s=130, edgecolor="None",
                                color="#FF8A8A"
                                )

            plt.xlim(left=0.75, right=2)
            print(data[data["logq"]>=2.0].shape[0], path)
            print(data[data["logq"]>=2.0].shape[0], path)
        
        else:
            # https://stackoverflow.com/questions/6682784/reducing-number-of-plot-ticks
            every_nth = 1
            sns.set_style("whitegrid")
            plt.figure(figsize=(4,2))
            ax = sns.scatterplot(
                                x="logq",y="Term",data=data[data["logq"]>=1.0],
                                s=130, edgecolor="None",
                                color="#FF8A8A"
                                )

            plt.xlim(left=0.75, right=2)
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

        
        
        genedf = pd.DataFrame(data[data["logq"]>=1.0]['Genes'])
        genedf.columns = ['Genes']
        genes = pd.concat([genes,genedf])
        
        #plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/fig2e_GOenrichment_"+path+".pdf", bbox_inches="tight")
        
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

plt.figure(figsize=(11,5))
col_annot = {"1L_stable":"#FF3F3F","non1L_stable":"#FF9E2E", "non1L_variable":"#FFC37E"}

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
            top=True, right=True
            )
sns.despine(
            top=True, right=True
            )
plt.xlim(-10,150)
plt.xticks([15,65,115],["1L stable", "non-1L stable", "non-1L variable"], fontsize=13)
plt.ylabel("$-Log_{10}$FDR", fontsize=14)
plt.axhline(1.0, color="#CACACA", linestyle="--")
plt.title("Enrichment of DNA repair genes", fontsize=15, position=(0.5, 1.0+0.02))


plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/fig2e_bubbleplot.pdf", bbox_inches="tight")

#%%


















#%%
######^^ make input ##R###

##########^^^^ MAKE INPUT for preliminary model ##########
# dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/baseline/1L_stable_DUT_MW.txt', sep='\t')
# filtered_trans['Gene Symbol'] = filtered_trans.index.str.split("-",2).str[1]

# dut.columns = ['transcript','pval','Gene Symbol']
# dut = dut[dut['pval']<0.05]
# dutlist = dut['transcript'].tolist()

# repairexp = filtered_trans[filtered_trans['Gene Symbol'].isin(dnarepairgenes)]

# dut_TU = repairexp[repairexp.index.isin(dutlist)]


# inputTU = dut_TU.iloc[:,:-1]
# t_inputTU = inputTU.T
# f_inputTU = t_inputTU
# f_inputTU['response'] = list(sampleinfo['response'])

# #####SAVE####
# f_inputTU.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/modelinput/1L_premodel_input_DUT_baseline.txt', sep='\t')




# %%
####^^^ baseline vs. PARPi-derived DUTs####

baseline = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/modelinput/1L_premodel_input_DUT_baseline.txt', sep='\t', index_col=0)
derived = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/modelinput/premodel_input_DUT_onlyrepair.txt', sep='\t', index_col=0)

baseline = baseline.iloc[:,:-1]
derived = derived.iloc[:,:-1]

baseline_dut = set(baseline.columns)
derived_dut = set(derived.columns)

baseline_gene = set(baseline.columns.str.split("-",2).str[1])
derived_gene = set(derived.columns.str.split("-",2).str[1])

# %%
from matplotlib_venn import venn2

plt.figure(figsize=(6,6))
sns.set_style("white")
vd2 = venn2([baseline_dut, derived_dut],set_labels=('Baseline', 'PARPi-derived'), set_colors=('#F8CF6A','#74B2D4'), alpha=0.8)

for text in vd2.set_labels:  #change label size
    text.set_fontsize(13)
for text in vd2.subset_labels:  #change number size
    text.set_fontsize(16)
    
vd2.get_patch_by_id('11').set_color('#C8C452')
vd2.get_patch_by_id('11').set_alpha(1)

#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/1L_B_D_dut_Venn.pdf", bbox_inches="tight")
plt.show()

# %%
plt.figure(figsize=(6,6))
sns.set_style("white")
vd2 = venn2([baseline_gene, derived_gene],set_labels=('Baseline', 'PARPi-derived'), set_colors=('#F8CF6A','#74B2D4'), alpha=0.8)

for text in vd2.set_labels:  #change label size
    text.set_fontsize(13)
for text in vd2.subset_labels:  #change number size
    text.set_fontsize(16)
    
vd2.get_patch_by_id('11').set_color('#C8C452')
vd2.get_patch_by_id('11').set_alpha(1)

plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/1l_B_D_gene_Venn.pdf", bbox_inches="tight")
plt.show()
# %%
















