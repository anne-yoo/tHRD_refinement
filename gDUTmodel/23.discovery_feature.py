#%%
#! this code is.....
"""
<Discovery Model initial features>
20231004
1. FDR < 0.1 DNA repair gene의 모든 major / minor transcript 비교 (mean/median)
2. FDR < 0.1 DNA repair gene 중 stable DUT만 major / minor transcript 비교  (mean/median)

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


# %%
dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/nonresponder_stable_DUT_Wilcoxon.txt', sep='\t')
enr = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/nonresponder_stable_Wilcoxon_GOenrichment.txt', sep='\t')

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

dnarepairgenes2 = [gene for sublist in data['Genes'].str.split(';') for gene in sublist]
dnarepairgenes2 = set(dnarepairgenes2)


#%%
#####****** Venn Diagram ################
from matplotlib_venn import venn2


plt.figure(figsize=(6,6))
sns.set_style("white")
vd2 = venn2([dnarepairgenes, dnarepairgenes2],set_labels=('R (stable)', 'NR (stable)'),set_colors=('#FF8181','#FFB055'), alpha=1)

for text in vd2.set_labels:  #change label size
    text.set_fontsize(13)
for text in vd2.subset_labels:  #change number size
    text.set_fontsize(16)
    
vd2.get_patch_by_id('11').set_color('#ffa97f')
vd2.get_patch_by_id('11').set_alpha(1)

plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/001_R_NR_gDUT_Venn.pdf", bbox_inches="tight")
plt.show()

#%%
########^^^^^ R_stable - NR stable ??? ########
finalgenes = dnarepairgenes - dnarepairgenes2


#%%
##########^^^^ MAKE INPUT for preliminary model ##########
dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t')

dut.columns = ['transcript','pval','Gene Symbol', 'log2FC']
dut = dut[dut['pval']<0.05]
dutlist = dut['transcript'].tolist()

repairexp = transexp[transexp['Gene Symbol'].isin(dnarepairgenes)]

dut_TU = repairexp[repairexp.index.isin(dutlist)]


inputTU = dut_TU.iloc[:,:-1]
t_inputTU = inputTU.T

#%%
# Calculate the means of each column

#cols_to_drop = t_inputTU.columns[t_inputTU.mean() < 0.02]

# Drop these columns from the original dataframe
#f_inputTU = t_inputTU.drop(columns=cols_to_drop, inplace=False)


#%%
f_inputTU = t_inputTU
sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.csv', sep=',')
f_inputTU['response'] = list(sampleinfo['response'])
#f_inputTU['treatment'] = [0 if i % 2 else 1 for i, _ in enumerate(f_inputTU.index)]


#%%
#####SAVE####
f_inputTU.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/modelinput/premodel_input_DUT_onlyrepair.txt', sep='\t')


#%%%
len(set(t_inputTU.columns.str.split("-",2).str[-1]))


#%%
# df_whole = filtered_trans[filtered_trans['Gene Symbol'].isin(dnarepairgenes)]
# df_dut = filtered_trans[filtered_trans.index.isin(dutlist)]
# df_dut = df_dut[df_dut['Gene Symbol'].isin(dnarepairgenes)]




sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.csv', sep=',')

responder = sampleinfo[sampleinfo['response']==1]['sample_full']
nonresponder = sampleinfo[sampleinfo['response']==0]['sample_full']
responder = responder.to_list()
nonresponder = nonresponder.to_list()

###***######
#df_dut = dut_TU.iloc[:,:-1]
df_dut = repairexp.iloc[:,:-1]
#######*###


# df_dut = df_dut.iloc[:,:-1]
# df_whole = df_whole.iloc[:,:-1]

R_df_dut = df_dut[responder]
NR_df_dut = df_dut[nonresponder]

# R_df_whole = df_whole[responder]
# NR_df_whole = df_whole[nonresponder]

# ####
# mean_d = R_df_dut.mean()
# median_d = R_df_dut.median()
# quantile_75_d = R_df_dut.quantile(0.75)

# R_df_dut['mean'] = mean_d
# R_df_dut['median'] = median_d
# R_df_dut['quantile'] = quantile_75_d

# mean_t = R_df_whole.mean()
# median_t = R_df_whole.median()
# quantile_75_t = R_df_whole.quantile(0.75)

# R_df_whole['mean'] = mean_t
# R_df_whole['median'] = median_t
# R_df_whole['quantile'] = quantile_75_t
# ####


major = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/final_202306_discovery_major_TU.txt', sep='\t', index_col=0)
minor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/final_202306_discovery_minor_TU.txt', sep='\t', index_col=0)

majortrans = major.index.tolist()
minortrans = minor.index.tolist()

R_df_dut['type'] = R_df_dut.index.to_series().apply(lambda x: 'major' if x in majortrans else ('minor' if x in minortrans else None))
# R_df_whole['type'] = R_df_whole.index.to_series().apply(lambda x: 'major' if x in majortrans else ('minor' if x in minortrans else None))
NR_df_dut['type'] = NR_df_dut.index.to_series().apply(lambda x: 'major' if x in majortrans else ('minor' if x in minortrans else None))
# NR_df_whole['type'] = NR_df_whole.index.to_series().apply(lambda x: 'major' if x in majortrans else ('minor' if x in minortrans else None))





#%%
################^^^^^^^^ R_df_dut ##############

filelist = [R_df_dut, NR_df_dut]
namelist = ['R_df_dut','NR_df_dut']
namelist2 = ['Responder','Nonresponder']
typelist = ['major','minor']


for i in range(2):
    df = filelist[i]
    
    for j in range(2):
        new_df = df[df['type']==typelist[j]]
        new_df = new_df.iloc[:,:-1]
        
        df_bfD = new_df[[col for col in new_df.columns if '-bfD' in col]]
        df_atD = new_df[[col for col in new_df.columns if '-atD' in col]]
        
        mean_bfD = list(df_bfD.mean(axis=1))
        mean_atD = list(df_atD.mean(axis=1))
        meandict = {'pre': mean_bfD, 'post': mean_atD}
        meandf = pd.DataFrame(meandict)
        meandf.index = new_df.index
        
        med_bfD = list(df_bfD.median(axis=1))
        med_atD = list(df_atD.median(axis=1))
        meddict = {'pre': med_bfD, 'post': med_atD}
        meddf = pd.DataFrame(meddict)
        meddf.index = new_df.index
        
        q_bfD = list(df_bfD.quantile(0.75, axis=1))
        q_atD = list(df_atD.quantile(0.75, axis=1))
        qdict = {'pre': q_bfD, 'post': q_atD}
        qdf = pd.DataFrame(qdict)
        qdf.index = new_df.index
        
        plt.figure(figsize=(4,6))
        sns.set_style("white")
        
        col1 = ["#70A8F2","#F36464"]
        
        ##
        figureinput = meandf.melt(var_name='Treatment', value_name='Usage')
        ##
        
        if j==0:
            plt.ylim(0,0.2)
        else:
            plt.ylim(0,0.12)
            
        ax = sns.boxplot(x="Treatment", y="Usage", data=figureinput,  palette=col1, showfliers=False)
        

        from statannot import add_stat_annotation

        add_stat_annotation(ax, data=figureinput, x='Treatment', y='Usage',
                    box_pairs=[("pre", "post")],
                    test='Wilcoxon',  text_format='simple', loc='outside')
        plt.figtext(0.5, 1.005, namelist2[i] + ' - ' + typelist[j]+' transcript exp', ha='center', va='center', fontsize=13)
        #plt.title(namelist2[i] + ' - ' + typelist[j]+' transcript exp', fontsize=13, y=3)
        sns.despine()

        plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/boxplot/WHOLEDUT_mean_"+namelist[i]+"_"+typelist[j]+".pdf", bbox_inches="tight")
        plt.show()



# %%
####^ count major transcripts which usage went up after PARPi treatment: R vs. NR ####

count_list_R = []
count_list_NR = []

plotdf = R_df_dut[R_df_dut['type']=='major']
plotdf = plotdf.iloc[:,:-1]
# Loop over the pairs
for i in range(0, plotdf.shape[1], 2):  # Increment by 2 because each pair is 2 columns
    after_treatment = plotdf.iloc[:, i]
    before_treatment = plotdf.iloc[:, i + 1]
    
    # Count number of genes with higher expression after treatment
    count = sum(after_treatment > before_treatment)
    count_list_R.append(count)
    
plotdf2 = NR_df_dut[NR_df_dut['type']=='major']
plotdf2 = plotdf2.iloc[:,:-1]
# Loop over the pairs
for i in range(0, plotdf2.shape[1], 2):  # Increment by 2 because each pair is 2 columns
    after_treatment = plotdf2.iloc[:, i]
    before_treatment = plotdf2.iloc[:, i + 1]
    
    # Count number of genes with higher expression after treatment
    count = sum(after_treatment > before_treatment)
    count_list_NR.append(count)

# Create a new DataFrame to store the counts
count_df_R = pd.DataFrame({'sample': 'Responder',
                        'Count': count_list_R})

count_df_NR = pd.DataFrame({'sample': 'Non-Responder',
                        'Count': count_list_NR})

count_df = pd.concat([count_df_R, count_df_NR], axis=0)

# Create the swarm plot
plt.figure(figsize=(4,6))

sns.set(font_scale=1.1)
sns.set_style("whitegrid")
col2 = ["#2B9C09","#F2BA1F"]
col1 = ["#9AE584","#FFDF8F"]
ax = sns.boxplot(x='sample', y='Count', data=count_df, palette=col1)
sns.stripplot(x='sample', y='Count', data=count_df, palette=col2, s=9)
plt.xlabel('')
plt.ylabel('# increased in post (major TU)', fontsize=13)

#add_stat_annotation(ax, data=count_df, x='sample', y='Count',
                    # box_pairs=[("Responder", "Non-Responder")],
                    # test='Mann-Whitney',  text_format='simple', loc='outside')

plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/boxplot/WHOLEDUT_001_increasedmajorTU.pdf", bbox_inches="tight")


plt.show()







# %%
#################^^^^^ count sample number for each transcript ##########
trans_count_R = []
for j in range(0, plotdf2.shape[0]):
    count_list_R = []
    for i in range(0, plotdf2.shape[1], 2): 
        # Increment by 2 because each pair is 2 columns
        after_treatment = plotdf2.iloc[j, i]
        before_treatment = plotdf2.iloc[j, i + 1]
        
        # Count number of genes with higher expression after treatment
        count = after_treatment > before_treatment
        count_list_R.append(count)
    trans_count_R.append(sum(count_list_R))

trans_count_R_df = pd.DataFrame({'major DUT':R_df_dut[R_df_dut['type']=='major'].index, 'count':trans_count_R })
# %%
