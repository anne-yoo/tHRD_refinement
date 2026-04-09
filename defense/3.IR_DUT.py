
###^^^^^^ PARPi-derived DUT vs. baseline DUT ####

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
from matplotlib_venn import venn2
from matplotlib_venn import venn3
from venny4py.venny4py import *



sns.set(font = 'Arial')
sns.set_style("whitegrid")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# %%
####^ PARPi-derived DUTs #####

dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t')
enr = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_Wilcoxon_GOenrichment.txt', sep='\t')

dut.columns = ['transcript','pval','Gene Symbol', 'log2FC']
dut = dut[dut['pval']<0.05]
dutlist = dut['transcript'].tolist()


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


plt.figure(figsize=(4,4))
sns.set_style("white")
vd2 = venn2([dnarepairgenes, dnarepairgenes2],set_labels=('AR stable', 'IR stable'),set_colors=('#FF8181','#FFB055'), alpha=0.9)

for text in vd2.set_labels:  #change label size
    text.set_fontsize(13)
for text in vd2.subset_labels:  #change number size
    text.set_fontsize(16)
    
vd2.get_patch_by_id('11').set_color('#ffa97f')
vd2.get_patch_by_id('11').set_alpha(1)

#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/fig2d_parpiderivedDUT_venn.pdf", bbox_inches="tight")
plt.show()

# %%
#####^^^ PARPi-derived DUT Functionality Check ###########

transexp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt',sep='\t', index_col=0)
#filtered_trans = transexp.loc[(transexp > 0).sum(axis=1) >= transexp.shape[1]*0.3]
transexp['Gene Symbol'] = transexp.index.str.split("-",1).str[1]
transexp = transexp[transexp['Gene Symbol']!='-']

dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t')

dut.columns = ['transcript','pval','Gene Symbol', 'log2FC']
dut = dut[dut['pval']<0.05]
dutlist = dut['transcript'].tolist()

##
repairexp = transexp[transexp['Gene Symbol'].isin(dnarepairgenes)]
##

dut_TU = repairexp[repairexp.index.isin(dutlist)]


inputTU = dut_TU.iloc[:,:-1]
t_inputTU = inputTU.T

f_inputTU = t_inputTU
sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.csv', sep=',')
f_inputTU['response'] = list(sampleinfo['response'])
#f_inputTU['treatment'] = [0 if i % 2 else 1 for i, _ in enumerate(f_inputTU.index)]
#####SAVE####
#f_inputTU.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/modelinput/premodel_input_DUT_onlyrepair.txt', sep='\t')

len(set(t_inputTU.columns.str.split("-",2).str[-1]))

responder = sampleinfo[sampleinfo['response']==1]['sample_full']
nonresponder = sampleinfo[sampleinfo['response']==0]['sample_full']
responder = responder.to_list()
nonresponder = nonresponder.to_list()

###***######
df_dut = dut_TU.iloc[:,:-1]
#df_dut = repairexp.iloc[:,:-1]
#######*###


R_df_dut = df_dut[responder]
NR_df_dut = df_dut[nonresponder]

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


for i in range(1):
    df = filelist[i]
    
    for j in range(1,2):
        new_df = df[df['type']==typelist[j]]
        new_df = new_df.iloc[:,:-1]
        
        minor_AR = new_df
        print(new_df)
        
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
        plt.figtext(0.5, 1.005, 'AR - ' + typelist[j]+' DUT usage', ha='center', va='center', fontsize=13)
        #plt.title(namelist2[i] + ' - ' + typelist[j]+' transcript exp', fontsize=13, y=3)
        sns.despine()

        #plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/fig3a_boxplot_IR_Parpiderived_DUT_"+namelist[i]+"_"+typelist[j]+".pdf", bbox_inches="tight")
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
count_df_R = pd.DataFrame({'sample': 'AR',
                        'Count': count_list_R})

count_df_NR = pd.DataFrame({'sample': 'IR',
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

plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/fig3a_increasedmajorTU_IR.pdf", bbox_inches="tight")

plt.show()








#%%
#####**** 231221 meeting: check individual genes ####
#genelist = ['TP53BP1','BRCA1','BRCA2','BRIP1','CHEK1','CHEK2',]
genelist = ['ATRIP',]


major_AR['gene'] = major_AR.index.str.split("-",1).str[1]
minor_AR['gene'] = minor_AR.index.str.split("-",1).str[1]

major_check = major_AR[major_AR['gene'].isin(genelist)]
minor_check = minor_AR[minor_AR['gene'].isin(genelist)]

#%%
for i in major_check.index:
    major_bfD = major_check[[col for col in major_check.columns if '-bfD' in col]]
    major_atD = major_check[[col for col in major_check.columns if '-atD' in col]]
    
    bfD_list = major_bfD.loc[i,:].to_list()
    atD_list = major_atD.loc[i,:].to_list()
    
    bfD_df = pd.DataFrame({'sample': 'pre',
                        'TU': bfD_list})

    atD_df = pd.DataFrame({'sample': 'post',
                            'TU': atD_list})

    prepost_df = pd.concat([bfD_df, atD_df], axis=0)
    
    plt.figure(figsize=(4,6))

    sns.set(font_scale=1.1)
    sns.set_style("whitegrid")
    col2 = ["#2B9C09","#F2BA1F"]
    col1 = ["#9AE584","#FFDF8F"]
    ax = sns.boxplot(x='sample', y='TU', data=prepost_df, palette=col1)
    sns.stripplot(x='sample', y='TU', data=prepost_df, palette=col2, s=9)
    plt.xlabel('')
    plt.ylabel('')
    plt.title(i,fontsize=13)
    plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/genecheck/major/'+i+'.pdf', bbox_inches="tight")
    plt.show()


for i in minor_check.index:
    minor_bfD = minor_check[[col for col in minor_check.columns if '-bfD' in col]]
    minor_atD = minor_check[[col for col in minor_check.columns if '-atD' in col]]
    
    bfD_list = minor_bfD.loc[i,:].to_list()
    atD_list = minor_atD.loc[i,:].to_list()
    
    bfD_df = pd.DataFrame({'sample': 'pre',
                        'TU': bfD_list})

    atD_df = pd.DataFrame({'sample': 'post',
                            'TU': atD_list})

    prepost_df = pd.concat([bfD_df, atD_df], axis=0)
    
    plt.figure(figsize=(4,6))

    sns.set(font_scale=1.1)
    sns.set_style("whitegrid")
    col2 = ["#2B9C09","#F2BA1F"]
    col1 = ["#9AE584","#FFDF8F"]
    ax = sns.boxplot(x='sample', y='TU', data=prepost_df, palette=col1)
    sns.stripplot(x='sample', y='TU', data=prepost_df, palette=col2, s=9)
    plt.xlabel('')
    plt.ylabel('')
    plt.title(i,fontsize=13)
    plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/genecheck/minor/'+i+'.pdf', bbox_inches="tight")
    plt.show()







# %%
#####^^^^ Baseline DUTs #####

dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/baseline/1L_stable_DUT_MW.txt', sep='\t')
enr = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/baseline/1L_stable_MW_GOenrichment.txt', sep='\t')

dut.columns = ['transcript','pval','Gene Symbol']
dut = dut[dut['pval']<0.05]
dutlist = dut['transcript'].tolist()

data = enr
data = data[data["Adjusted P-value"] <= 0.1]
# data = data[(data["Term"].str.contains("repair")) | (data["Term"].str.contains("DNA damage")) 
#                 | (data["Term"].str.contains("DNA metabolic")) | (data["Term"].str.contains("(GO:0006260)"))
#                 | (data["Term"].str.contains("DNA duplex unwinding"))]

#data = data[(data["Term"].str.contains("repair", case=False)) | (data["Term"].str.contains("DNA Damage", case=False))]
data = data[(data["Term"].str.contains("repair", case=False))]

dnarepairgenes = [gene for sublist in data['Genes'].str.split(';') for gene in sublist]
dnarepairgenes = set(dnarepairgenes)


# %%
dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/baseline/non1L_stable_DUT_MW.txt', sep='\t')
enr = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/baseline/non1L_stable_MW_GOenrichment.txt', sep='\t')

dut.columns = ['transcript','pval','Gene Symbol']
dut = dut[dut['pval']<0.05]
dutlist = dut['transcript'].tolist()

data = enr
data = data[data["Adjusted P-value"] <= 0.1]
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


plt.figure(figsize=(4,4))
sns.set_style("white")
vd2 = venn2([dnarepairgenes, dnarepairgenes2],set_labels=('1L stable', 'non-1L stable'),set_colors=('#FF8181','#FFB055'), alpha=0.9)

for text in vd2.set_labels:  #change label size
    text.set_fontsize(13)
for text in vd2.subset_labels:  #change number size
    text.set_fontsize(16)
    
vd2.get_patch_by_id('11').set_color('#ffa97f')
vd2.get_patch_by_id('11').set_alpha(1)

plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/fig2e_BaselineDUT_venn.pdf", bbox_inches="tight")
plt.show()




# %%
#####^^^ Baseline Functionality Check ###########

transexp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0)
transexp['Gene Symbol'] = transexp.index.str.split("-",1).str[1]
transexp = transexp[transexp['Gene Symbol']!='-']

dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/baseline/non1L_stable_DUT_MW.txt', sep='\t')

dut.columns = ['transcript','pval','Gene Symbol']
dut = dut[dut['pval']<0.05]
dutlist = dut['transcript'].tolist()

repairexp = transexp[transexp['Gene Symbol'].isin(dnarepairgenes)] ### gene filtering

dut_TU = repairexp[repairexp.index.isin(dutlist)] ### DUT filtering

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.csv', sep=',')
FL_list = list(sampleinfo[sampleinfo['line_binary']=='FL']['sample_full'])
sampleinfo = sampleinfo[sampleinfo['line_binary']=='FL']

responder = sampleinfo[sampleinfo['response']==1]['sample_full']
nonresponder = sampleinfo[sampleinfo['response']==0]['sample_full']
responder = responder.to_list()
nonresponder = nonresponder.to_list()

###***######
df_dut = dut_TU.iloc[:,:-1]
#df_dut = repairexp.iloc[:,:-1]
#######*###

df_dut = df_dut.loc[:,FL_list]

R_tu = df_dut[responder]
NR_tu = df_dut[nonresponder]

R_pre_tu = R_tu[R_tu.columns[1::2]]
R_pre_tu['Gene Symbol'] = R_pre_tu.index.str.split("-",2).str[1]

NR_pre_tu = NR_tu[NR_tu.columns[1::2]]
NR_pre_tu['Gene Symbol'] = NR_pre_tu.index.str.split("-",2).str[1]

major = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/final_202306_discovery_major_TU.txt', sep='\t', index_col=0)
minor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/final_202306_discovery_minor_TU.txt', sep='\t', index_col=0)

majortrans = major.index.tolist()
minortrans = minor.index.tolist()

R_pre_tu['type'] = R_pre_tu.index.to_series().apply(lambda x: 'major' if x in majortrans else ('minor' if x in minortrans else None))
# R_df_whole['type'] = R_df_whole.index.to_series().apply(lambda x: 'major' if x in majortrans else ('minor' if x in minortrans else None))
NR_pre_tu['type'] = NR_pre_tu.index.to_series().apply(lambda x: 'major' if x in majortrans else ('minor' if x in minortrans else None))
# NR_df_whole['type'] = NR_df_whole.index.to_series().apply(lambda x: 'major' if x in majortrans else ('minor' if x in minortrans else None))





#%%
################^^^^^^^^ R_df_dut ##############

typelist = ['major','minor']

for i in range(2):
    new_R_pre = R_pre_tu[R_pre_tu['type']==typelist[i]]
    new_NR_pre = NR_pre_tu[NR_pre_tu['type']==typelist[i]]
    
    mean_R_pre = list(new_R_pre.mean(axis=1))
    mean_NR_pre = list(new_NR_pre.mean(axis=1))
    
    meandict = {'IR pre': mean_NR_pre, 'AR pre': mean_R_pre}
    meandf = pd.DataFrame(meandict)
    meandf.index = new_R_pre.index
    
    print(meandf)
    
    plt.figure(figsize=(4,6))
    sns.set_style("white")
    
    col1 = ["#70A8F2","#F36464"]
    col2 = ["#074AA2","#971010"]
    
    ##
    figureinput = meandf.melt(var_name='Treatment', value_name='Usage')
    ##
    
    if i==0:
        plt.ylim(0,0.2)
    else:
        plt.ylim(0,0.12)
        
    ax = sns.boxplot(x="Treatment", y="Usage", data=figureinput,  palette=col1, showfliers=False)
    sns.stripplot(x="Treatment", y="Usage", data=figureinput,  palette=col2, dodge=True)

    from statannot import add_stat_annotation

    add_stat_annotation(ax, data=figureinput, x='Treatment', y='Usage',
                box_pairs=[("IR pre", "AR pre")],
                test='Wilcoxon',  text_format='simple', loc='outside')
    plt.figtext(0.5, 1.005, '1L - ' + typelist[i]+' DUT usage', ha='center', va='center', fontsize=13)
    #plt.title(namelist2[i] + ' - ' + typelist[j]+' transcript exp', fontsize=13, y=3)
    sns.despine()

    #plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/fig3b_boxplot_1L_Baseline_DUT_"+typelist[i]+".pdf", bbox_inches="tight")
    plt.show()




# %%
######^^^^^^^^^^^ PARPi-derived vs. Baseline DUT ##################\
###* PARPi-derived ###
dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t')
enr = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_Wilcoxon_GOenrichment.txt', sep='\t')

dut.columns = ['transcript','pval','Gene Symbol', 'log2FC']
dut = dut[dut['pval']<0.05]
dutlist_p = dut['transcript'].tolist()


data = enr
data = data[data["Adjusted P-value"] <= 0.01]
# data = data[(data["Term"].str.contains("repair")) | (data["Term"].str.contains("DNA damage")) 
#                 | (data["Term"].str.contains("DNA metabolic")) | (data["Term"].str.contains("(GO:0006260)"))
#                 | (data["Term"].str.contains("DNA duplex unwinding"))]

#data = data[(data["Term"].str.contains("repair", case=False)) | (data["Term"].str.contains("DNA Damage", case=False))]
data = data[(data["Term"].str.contains("repair", case=False))]

dnarepairgenes_p = [gene for sublist in data['Genes'].str.split(';') for gene in sublist]
dnarepairgenes_p = set(dnarepairgenes_p)


###* 1L Baseline ###
dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/baseline/1L_stable_DUT_MW.txt', sep='\t')
enr = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/baseline/1L_stable_MW_GOenrichment.txt', sep='\t')

dut.columns = ['transcript','pval','Gene Symbol']
dut = dut[dut['pval']<0.05]
dutlist_b = dut['transcript'].tolist()

data = enr
data = data[data["Adjusted P-value"] <= 0.1]
# data = data[(data["Term"].str.contains("repair")) | (data["Term"].str.contains("DNA damage")) 
#                 | (data["Term"].str.contains("DNA metabolic")) | (data["Term"].str.contains("(GO:0006260)"))
#                 | (data["Term"].str.contains("DNA duplex unwinding"))]

#data = data[(data["Term"].str.contains("repair", case=False)) | (data["Term"].str.contains("DNA Damage", case=False))]
data = data[(data["Term"].str.contains("repair", case=False))]

dnarepairgenes_b = [gene for sublist in data['Genes'].str.split(';') for gene in sublist]
dnarepairgenes_b = set(dnarepairgenes_b)

###^^^ gene level: venn PARPi vs. 1L baseline #####

plt.figure(figsize=(6,6))
sns.set_style("white")
vd2 = venn2([dnarepairgenes_p, dnarepairgenes_b],set_labels=('PARPi-derived', 'Baseline'),set_colors=('#F8CF6A','#74B2D4'), alpha=0.8)

for text in vd2.set_labels:  #change label size
    text.set_fontsize(13)
for text in vd2.subset_labels:  #change number size
    text.set_fontsize(16)
    
vd2.get_patch_by_id('11').set_color('#C8C452')
vd2.get_patch_by_id('11').set_alpha(1)

plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/fig2f_gene_dutcomparison_venn.pdf", bbox_inches="tight")
plt.show()

#%%
##^^ ++ HR genes ######
HRlist = set(open('/home/jiye/jiye/copycomparison/gDUTresearch/data/HR_genes.txt').read().splitlines())
HRlist2 = {'C11orf30', 'RAD50','BRCA1','BRCA2', 'ATR', 'RAD51C', 'RAD51', 'FANCA', 'TP53', 'CHEK1', 'ATM', 'FANCE', 'CHEK2', 'NBN', 'FANCF', 'BRIP1', 'FANCC', 'PTEN', 'FANCB'}

plt.figure(figsize=(6,6))
sns.set_style("white")

sets = {
    'HR gene': HRlist2,
    'PARPi-derived': dnarepairgenes_p,
    'Baseline': dnarepairgenes_b}
    
venny4py(sets=sets)

plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/fig2f_HRDgene_dutcomparison_venn.pdf", bbox_inches="tight")
plt.show()

#%%
###^^^ trans level: venn PARPi vs. 1L baseline ###
parpi = set([item for item in dutlist_p for gene in dnarepairgenes_p if gene in item])
baseline = set([item for item in dutlist_b for gene in dnarepairgenes_b if gene in item])

plt.figure(figsize=(6,6))
sns.set_style("white")
vd2 = venn2([parpi, baseline],set_labels=('PARPi-derived', 'Baseline'),set_colors=('#F8CF6A','#74B2D4'), alpha=0.8)

for text in vd2.set_labels:  #change label size
    text.set_fontsize(13)
for text in vd2.subset_labels:  #change number size
    text.set_fontsize(16)
    
vd2.get_patch_by_id('11').set_color('#C8C452')
vd2.get_patch_by_id('11').set_alpha(1)

plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/fig2f_trans_dutcomparison_venn.pdf", bbox_inches="tight")
plt.show()




# %%
#######^^^ SAVE DUTs #########

padded_lists = []
all_lists = [list(parpi), list(baseline), list(dnarepairgenes_p), list(dnarepairgenes_b)]
for lst in all_lists:
    # Extend each list with NaNs to match the length of the longest list
    padded_list = lst + [np.nan] * (max(len(all_lists[0]), len(all_lists[1]), len(all_lists[2]), len(all_lists[3])) - len(lst))
    padded_lists.append(padded_list)
    
duts = pd.DataFrame(padded_lists).T
duts.columns = ['PARPi-derived_DUT','Baseline_DUT','PARPi-derived_gene','Baseline_gene']
duts.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202312_analysis/finalDUTlist.txt',sep='\t',index=False)

# %%
