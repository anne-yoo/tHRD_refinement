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
thrd1 = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/tHRD_result/ensemble_model1_tHRD.txt', sep='\t')
thrd2 = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/tHRD_result/ensemble_model2_tHRD.txt', sep='\t')

# %%
thrd1['treatment'] = 'pre'
thrd1.iloc[0::2,2] = 'post'

thrd2['treatment'] = 'pre'
thrd2.iloc[0::2,2] = 'post'

thrd1['group'] = 'negative'
thrd1.loc[thrd1['mean']>np.median(thrd1['mean']),'group'] = 'positive'

thrd2['group'] = 'negative'
thrd2.loc[thrd2['mean']>np.median(thrd2['mean']),'group'] = 'positive'

# %%
plt.figure(figsize=(4,6))
col1 = ["#E59C84","#8F8FFF"]
ax = sns.boxplot(x="treatment", y="mean", data=thrd1,  palette=col1, showfliers=False, order=['pre','post'])
sns.stripplot(x='treatment', y='mean', data=thrd1, palette=col1, s=9, order=['pre','post'])

# %%
plt.figure(figsize=(4,6))
col1 = ["#E59C84","#8F8FFF"]
ax = sns.boxplot(x="treatment", y="mean", data=thrd2,  palette=col1, showfliers=False, order=['pre','post'])
sns.stripplot(x='treatment', y='mean', data=thrd2, palette=col1, s=9, order=['pre','post'])

# %%
###^^ tHRD positive / negative ###

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')

####***########
model = thrd1
####***########

original_list = list(model.iloc[1::2,-1])
repeated_list = [item for item in original_list for _ in range(2)]

sampleinfo['thrd'] = repeated_list

####***########
thrd_positive = list(sampleinfo[sampleinfo['thrd']=='positive']['sample_full'])
thrd_negative = list(sampleinfo[sampleinfo['thrd']=='negative']['sample_full'])
####***########


#%%

datalist = [thrd_positive, thrd_negative]
namelist = ['BRCAmt', 'BRCAwt']

for i in range(2):
    degresult = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202312_analysis/DEG/whole_Wilcoxon_DEGresult_FC.txt', sep='\t')

    
    DEGlist = set(degresult[degresult['p_value']<0.05]['Gene Symbol'])
    nonDEGlist = set(degresult[degresult['p_value'] > 0.05]['Gene Symbol'])
    
    print(namelist[i]," variable: ", len(DEGlist))
    print(namelist[i]," stable: ", len(nonDEGlist))

    filtered_trans = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0)
    
    filtered_trans['Gene Symbol'] = filtered_trans.index.str.split("-",2).str[1]

    ####^ variable genes: DUT ####
    variable_trans = filtered_trans[filtered_trans['Gene Symbol'].isin(DEGlist)]
    variable_trans = variable_trans[datalist[i]]


    variable_dut_pval = []
    for index, row in variable_trans.iterrows():
        pre_samples = row[1::2].values  # Even-indexed columns are pre-treatment samples
        post_samples = row[::2].values  # Odd-indexed columns are post-treatment samples
        
        # Perform the paired Wilcoxon test and store the p-value        
        if set([a - b for a, b in zip(pre_samples, post_samples)]) != {0}: 
            w, p = stats.wilcoxon(post_samples, pre_samples)
            variable_dut_pval.append(p)
        else:
            variable_dut_pval.append(1)

    # Create a new DataFrame with geneid and respective p-values
    variable_result = pd.DataFrame({
        'p_value':variable_dut_pval,
    })
    variable_result.index = variable_trans.index
    variable_result['Gene Symbol'] = variable_result.index.str.split("-",1).str[1]

    ##### FC #####
    avg_pre = variable_trans.iloc[:, 1::2].mean(axis=1)
    avg_post = variable_trans.iloc[:, ::2].mean(axis=1)

    fold_change = np.log2(avg_post / avg_pre)
    variable_result['log2FC'] = fold_change
    ##############

    ####^ stable genes: DUT ####
    stable_trans = filtered_trans[filtered_trans['Gene Symbol'].isin(nonDEGlist)]
    stable_trans = stable_trans[datalist[i]]

    stable_dut_pval = []
    
    for index, row in stable_trans.iterrows():
        pre_samples = row[1::2].values  # Even-indexed columns are pre-treatment samples
        post_samples = row[::2].values  # Odd-indexed columns are post-treatment samples
        
        # Perform the paired Wilcoxon test and store the p-value
        if set([a - b for a, b in zip(pre_samples, post_samples)]) != {0}: 
            w, p = stats.wilcoxon(post_samples, pre_samples)
            stable_dut_pval.append(p)
        else:
            stable_dut_pval.append(1)

    # Create a new DataFrame with geneid and respective p-values
    stable_result = pd.DataFrame({
        'p_value': stable_dut_pval,
    })
    stable_result.index = stable_trans.index
    stable_result['Gene Symbol'] = stable_result.index.str.split("-",1).str[1]
    
    ##### FC #####
    avg_pre = stable_trans.iloc[:, 1::2].mean(axis=1)
    avg_post = stable_trans.iloc[:, ::2].mean(axis=1)

    fold_change = np.log2(avg_post / avg_pre)
    stable_result['log2FC'] = fold_change
    ##############
    
    
    #############
    variable_DUT = variable_result[variable_result['p_value'] < 0.05]['Gene Symbol']
    stable_DUT = stable_result[stable_result['p_value'] < 0.05]['Gene Symbol']
    
    print('variable DUT: ', len(variable_DUT))
    print('stable DUT: ', len(stable_DUT))
    
    variable_result.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202401_analysis/brca/'+namelist[i]+'_variable_DUT_Wilcoxon.txt', sep='\t')
    stable_result.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202401_analysis/brca/'+namelist[i]+'_stable_DUT_Wilcoxon.txt', sep='\t')

# %%
#^ 0. Go enrichment

import gseapy as gp

###*enrichR GO enrichment analysis

filelist = ['BRCAmt_stable','BRCAmt_variable','BRCAwt_stable','BRCAwt_variable']

for i in range(4):
    path2 = '/home/jiye/jiye/copycomparison/gDUTresearch/202401_analysis/brca/'

    results = pd.read_csv(path2+filelist[i]+'_DUT_Wilcoxon.txt', sep='\t')

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
    print(filelist[i], len(enrresult[enrresult['Adjusted P-value']<0.05]))
    enrresult.to_csv(path2+filelist[i]+'_Wilcoxon_GOenrichment.txt', sep='\t')
    

#%%
#^ 1. GO enrichment plot: acquired

path_list = ['BRCAmt_stable','BRCAmt_variable','BRCAwt_stable','BRCAwt_variable']
merged_df = pd.DataFrame(columns=["logq","type"])
genes = pd.DataFrame(columns=['Genes'])

for path in path_list:
    data = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202401_analysis/brca/'+path+'_Wilcoxon_GOenrichment.txt',sep='\t') 
    #data = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/MW/'+path+'_RpreNRpreMWDUT_GOenrichment.csv',sep=',') #MW
    
    data["logq"] = -np.log10(data["Adjusted P-value"])
    data = data[data["Gene_set"]=="GO_Biological_Process_2021"]
    
    sig = data[data["Adjusted P-value"] <= 0.05]
    
    ##* Select DNA repair related Terms 
    # data = data[(data["Term"].str.contains("repair", case=False)) | (data["Term"].str.contains("DNA Damage")) 
    #             | (data["Term"].str.contains("DNA metabolic")) | (data["Term"].str.contains("(GO:0006260)"))
    #             | (data["Term"].str.contains("DNA duplex unwinding"))]
    
    #data = data[(data["Term"].str.contains("repair", case=False)) | (data["Term"].str.contains("DNA Damage", case=False))]
    
    data = data[(data["Term"].str.contains("repair", case=False))]
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

            plt.xlim(left=1.5, right=3.5)
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

            plt.xlim(left=1.5, right=3.5)
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
        
        plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202401_analysis/brca/GOenrichment_"+path+".pdf", bbox_inches="tight")
        
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
col_annot = {"BRCAmt_stable":"#FF3F3F", "BRCAmt_variable":"#FF8181", "BRCAwt_stable":"#FF9E2E", "BRCAwt_variable":"#FFC37E"}

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
plt.xlim(-10,190)
plt.xticks([15,65,115,165],["BRCAmt stable", "BRCAmt variable","BRCAwt stable","BRCAwt variable"], fontsize=13)
plt.ylabel("$-Log_{10}$FDR", fontsize=14)
plt.axhline(2.0, color="#CACACA", linestyle="--")
plt.title("Enrichment of DNA repair genes", fontsize=15, position=(0.5, 1.0+0.02))


plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202401_analysis/brca/bubbleplot.pdf", bbox_inches="tight")



# %%
###^^ major TU: brca ####
enr = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202401_analysis/brca/BRCAwt_stable_Wilcoxon_GOenrichment.txt', sep='\t')

data = enr
data = data[data["Adjusted P-value"] <= 0.01]
data = data[(data["Term"].str.contains("repair", case=False))]

dnarepairgenes = [gene for sublist in data['Genes'].str.split(';') for gene in sublist]
dnarepairgenes = set(dnarepairgenes)

#dutlist = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202401_analysis/brca/BRCAwt_stable_DUT_Wilcoxon.txt', sep='\t')
#dutlist = dutlist[dutlist['p_value']<0.05]
#DUTlist = list(dutlist['gene_ENST'])
#dut_df = filtered_trans.loc[filtered_trans.index.isin(DUTlist),:]
filtered_trans = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0)

filtered_trans['Gene Symbol'] = filtered_trans.index.str.split("-",2).str[1]

dut_df = filtered_trans
repairexp = dut_df[dut_df['Gene Symbol'].isin(dnarepairgenes)]

df_dut = repairexp.iloc[:,:-1]
R_df_dut = df_dut[thrd_positive]
NR_df_dut = df_dut[thrd_negative]

major = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/final_202306_discovery_major_TU.txt', sep='\t', index_col=0)
minor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/final_202306_discovery_minor_TU.txt', sep='\t', index_col=0)

majortrans = major.index.tolist()
minortrans = minor.index.tolist()

R_df_dut['type'] = R_df_dut.index.to_series().apply(lambda x: 'major' if x in majortrans else ('minor' if x in minortrans else None))
NR_df_dut['type'] = NR_df_dut.index.to_series().apply(lambda x: 'major' if x in majortrans else ('minor' if x in minortrans else None))

filelist = [R_df_dut, NR_df_dut]
namelist = ['BRCAmt','BRCAwt']
namelist2 = ['BRCAmt','BRCAwt']
typelist = ['major','minor']


for i in range(1,2):
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
            plt.ylim(0,0.3)
        else:
            plt.ylim(0,0.12)
            
        ax = sns.boxplot(x="Treatment", y="Usage", data=figureinput,  palette=col1, showfliers=False)
        

        from statannot import add_stat_annotation

        add_stat_annotation(ax, data=figureinput, x='Treatment', y='Usage',
                    box_pairs=[("pre", "post")],
                    test='Wilcoxon',  text_format='simple', loc='outside')
        plt.figtext(0.5, 1.005, namelist2[i] + ' ' + typelist[j]+' TU', ha='center', va='center', fontsize=13)
        #plt.title(namelist2[i] + ' - ' + typelist[j]+' transcript exp', fontsize=13, y=3)
        sns.despine()

        plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202401_analysis/brca/wholemajor/DUT_mean_"+namelist[i]+"_"+typelist[j]+".pdf", bbox_inches="tight")
        plt.show()




#%%
# %%
######^^ major TU : tHRD ########
enr = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202401_analysis/thrd/thrd_negative_stable_model1_Wilcoxon_GOenrichment.txt', sep='\t')

data = enr
data = data[data["Adjusted P-value"] <= 0.01]
data = data[(data["Term"].str.contains("repair", case=False))]

dnarepairgenes = [gene for sublist in data['Genes'].str.split(';') for gene in sublist]
dnarepairgenes = set(dnarepairgenes)

#dutlist = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202401_analysis/brca/BRCAwt_stable_DUT_Wilcoxon.txt', sep='\t')
#dutlist = dutlist[dutlist['p_value']<0.05]
#DUTlist = list(dutlist['gene_ENST'])
#dut_df = filtered_trans.loc[filtered_trans.index.isin(DUTlist),:]
filtered_trans = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0)

filtered_trans['Gene Symbol'] = filtered_trans.index.str.split("-",2).str[1]

dut_df = filtered_trans
repairexp = dut_df[dut_df['Gene Symbol'].isin(dnarepairgenes)]

df_dut = repairexp.iloc[:,:-1]
R_df_dut = df_dut[thrd_positive]
NR_df_dut = df_dut[thrd_negative]

major = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/final_202306_discovery_major_TU.txt', sep='\t', index_col=0)
minor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/final_202306_discovery_minor_TU.txt', sep='\t', index_col=0)

majortrans = major.index.tolist()
minortrans = minor.index.tolist()

R_df_dut['type'] = R_df_dut.index.to_series().apply(lambda x: 'major' if x in majortrans else ('minor' if x in minortrans else None))
NR_df_dut['type'] = NR_df_dut.index.to_series().apply(lambda x: 'major' if x in majortrans else ('minor' if x in minortrans else None))

filelist = [R_df_dut, NR_df_dut]
namelist = ['tHRD+','tHRD-']
namelist2 = ['tHRD+','tHRD-']
typelist = ['major','minor']


for i in range(1,2):
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
            plt.ylim(0,0.30)
        else:
            plt.ylim(0,0.12)
            
        ax = sns.boxplot(x="Treatment", y="Usage", data=figureinput,  palette=col1, showfliers=False)
        

        from statannot import add_stat_annotation

        add_stat_annotation(ax, data=figureinput, x='Treatment', y='Usage',
                    box_pairs=[("pre", "post")],
                    test='Wilcoxon',  text_format='simple', loc='outside')
        plt.figtext(0.5, 1.005, namelist2[i]  + ' ' + typelist[j]+' TU', ha='center', va='center', fontsize=13)
        #plt.title(namelist2[i] + ' - ' + typelist[j]+' transcript exp', fontsize=13, y=3)
        sns.despine()

        plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202401_analysis/thrd/wholemajor/mean_"+namelist[i]+"_"+typelist[j]+".pdf", bbox_inches="tight")
        plt.show()




# %%
###############^^^^ tRHD pre vs. post barplot ##########

df = thrd1
df['treatment'] = pd.Categorical(df['treatment'], categories=['pre', 'post'], ordered=True)
df['group'] = list(sampleinfo['response'])


# Extract patient ID without treatment indication for plotting
df['patient_id'] = df['id'].str[:-4]

pal =  {'pre': '#FFC436', 'post': '#337CCF'}

# Plot using seaborn
plt.figure(figsize=(12, 6))
sns.barplot(x='patient_id', y='mean', hue='treatment', data=df[df['group']==1], hue_order=['pre', 'post'], palette=pal)
plt.title('AR group - tHRD',fontsize=13)
plt.xlabel('')
plt.ylabel('tHRD mean')
plt.legend(title='Treatment')
plt.ylim(0,0.55)
plt.xticks(rotation=45)  # Rotate x labels for better readability if needed
plt.tight_layout()  # Adjust layout to ensure everything fits without overlapping
plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202401_analysis/AR_tHRD.pdf", bbox_inches="tight")
plt.show()
# %%
