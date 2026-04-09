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
#%% #^ functions for DUT save + enrichR
#######^1. stable / variable DUT ####

def saveDUT():
    sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.csv', sep=',')

    responder = sampleinfo[sampleinfo['response']==1]['sample_full']
    nonresponder = sampleinfo[sampleinfo['response']==0]['sample_full']
    responder = responder.to_list()
    nonresponder = nonresponder.to_list()

    list = [responder, nonresponder]
    namelist = ['responder', 'nonresponder']

    for i in range(2):
        degresult = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DEG/'+namelist[i]+'_Wilcoxon_DEGresult_FC.txt', sep='\t')
        
        DEGlist = set(degresult[degresult['p_value']<0.05]['Gene Symbol'])
        nonDEGlist = set(degresult[degresult['p_value'] > 0.05]['Gene Symbol'])
        
        print(namelist[i]," variable: ", len(DEGlist))
        print(namelist[i]," stable: ", len(nonDEGlist))

        #######*** TU file #########################################################
        filtered_trans = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0)
        ##**#################################################################################
        
        filtered_trans['Gene Symbol'] = filtered_trans.index.str.split("-",2).str[1]
        
        ####^ filter only major transcripts ####
        major = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
        majortrans = major[major['type']=='major']['gene_ENST'].to_list()
        filtered_trans = filtered_trans.loc[filtered_trans.index.isin(majortrans)]

        ####^ variable genes: DUT ####
        variable_trans = filtered_trans[filtered_trans['Gene Symbol'].isin(DEGlist)]
        variable_trans = variable_trans[list[i]]


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
        stable_trans = stable_trans[list[i]]

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
        
        variable_result.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/majorDUT/'+namelist[i]+'_variable_DUT_Wilcoxon.txt', sep='\t')
        stable_result.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/majorDUT/'+namelist[i]+'_stable_DUT_Wilcoxon.txt', sep='\t')

#####^2. distbrution DUT gene ####

def saveDistributionDUTgene():
    from scipy.stats import wilcoxon
    sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
    responder = sampleinfo.loc[(sampleinfo['response']==1),'sample_full'].to_list()
    nonresponder = sampleinfo.loc[(sampleinfo['response']==0),'sample_full'].to_list()

    ############**************** TU file #####################################
    TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0)
    ###############*******#############################################################
    
    
    namelist = ['responder','nonresponder']
    variablelist1 = [responder, nonresponder] #list of AR , IR samples
    namelist1 = ['AR', 'IR']
    namelist2 = ['stable', 'variable']
    
    for i in range(2): ## AR vs. IR
        degresult = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DEG/'+namelist[i]+'_Wilcoxon_DEGresult_FC.txt', sep='\t')    
        DEGlist = set(degresult[degresult['p_value']<0.05]['Gene Symbol'])
        nonDEGlist = set(degresult[degresult['p_value'] > 0.05]['Gene Symbol'])
        
        tmpdf = TU[variablelist1[i]] ## TU df with only AR / IR samples
        tmpdf['GeneName'] = tmpdf.index.str.split("-",2).str[1]
        
        stabledf = tmpdf[tmpdf['GeneName'].isin(nonDEGlist)]
        variabledf = tmpdf[tmpdf['GeneName'].isin(DEGlist)]
        variablelist2 = [stabledf, variabledf]
        
        for j in range(2): ## stable vs. variable
            df = variablelist2[j]
            df_nogene = df.iloc[:,:-1] ## without the last column; for parsing pre vs. post
            
            # Separate the expression data by pre- and post-treatment
            pre_columns = df_nogene.columns[1::2]  
            post_columns = df_nogene.columns[0::2]  
            # Create a list to store the results
            results = []
            for gene, group in df.groupby('GeneName'):
                # Flatten to compare all transcripts' expression for pre- vs. post-treatment
                pre_values = group[pre_columns].values.flatten()
                post_values = group[post_columns].values.flatten()
                
                # Calculate the differences
                differences = post_values - pre_values
                
                # Check if there are any non-zero differences
                if np.any(differences != 0):
                    # Perform the Wilcoxon signed-rank test
                    stat, p = wilcoxon(differences)
                    results.append((gene, stat, p))
                else:
                    # If all differences are zero, append NaN or a placeholder value
                    results.append((gene, np.nan, np.nan))

            # Convert to DataFrame
            results_df = pd.DataFrame(results, columns=['GeneName', 'Statistic', 'P-value'])
            
            # Adjusting for multiple comparisons as previously described
            from statsmodels.stats.multitest import multipletests
            _, p_adjusted, _, _ = multipletests(results_df['P-value'].fillna(1), alpha=0.05, method='fdr_bh')
            results_df['Adjusted P-value'] = p_adjusted
            results_df.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/majorDUT/'+namelist1[i]+'_'+namelist2[j]+'_distribution_result.txt', sep='\t', index=False)
            
            significant_genes = results_df[results_df['P-value'] < 0.05]
            gDUTlist = significant_genes['GeneName']
            with open('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/majorDUT/'+namelist1[i]+'_'+namelist2[j]+'_distribution_DUTgenelist.txt', "w") as file:
                for item in gDUTlist:
                    file.write("%s\n" % item)

#%%
#####^3. enrichR#####
def enrichR():  
    import gseapy as gp

    path_list1 = ['responder_stable','responder_variable','nonresponder_stable','nonresponder_variable']
    path_list2 = ['AR_stable','AR_variable','IR_stable','IR_variable']

    path = '/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/majorDUT/'
    #path2 = '/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUT/'
    
    #### DUT ###########
    for i in range(4):    
        results = pd.read_csv(path+path_list1[i]+'_DUT_Wilcoxon.txt', sep='\t')
        pcut = results[(results['p_value']<0.05) & (np.abs(results['log2FC'])>1.5)]['Gene Symbol']
        pcut = pcut.drop_duplicates()
        glist = pcut.squeeze().str.strip().to_list()
        print(path_list1[i], len(glist))

        enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                        gene_sets=['GO_Biological_Process_2021'], 
                        organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                        outdir=None, # don't write to disk
        )

        enrresult = enr.results.sort_values(by=['Adjusted P-value']) 

        ## file save
        enrresult.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/majorDUT/'+path_list1[i]+'_DUT_GOenrichment.txt', index=False, sep='\t')         

    ### distribution ##############
    for i in range(4):    
        results = pd.read_csv(path+path_list2[i]+'_distribution_result.txt', sep='\t')
        pcut = results[(results['P-value']<0.05)]['GeneName']
        pcut = pcut.drop_duplicates()
        glist = pcut.squeeze().str.strip().to_list()

        enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                        gene_sets=['GO_Biological_Process_2021'], 
                        organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                        outdir=None, # don't write to disk
        )

        enrresult = enr.results.sort_values(by=['Adjusted P-value']) 

        ## file save
        enrresult.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/majorDUT/'+path_list2[i]+'_distDUT_GOenrichment.txt', index=False, sep='\t')  

#%% #^^ Enrich+Bubbleplot

path_list = ['responder_stable','responder_variable','nonresponder_stable','nonresponder_variable']
path_list2 = ['AR_stable','AR_variable','IR_stable','IR_variable']
merged_df = pd.DataFrame(columns=["logq","type"])
genes = pd.DataFrame(columns=['Genes'])

for i in range(4):
    ################################### DUT vs. distribution 3#########################################################
    path = path_list[i]
    data = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/majorDUT/'+path+'_DUT_GOenrichment.txt',sep='\t')  
    
    # path = path_list[i]
    # path2 = path_list2[i]
    # data = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/majorDUT/'+path2+'_distDUT_GOenrichment.txt',sep='\t')  
    ################################### DUT vs. distribution 3#########################################################
    
    data["logq"] = -np.log10(data["Adjusted P-value"])
    data = data[data["Gene_set"]=="GO_Biological_Process_2021"]
    
    sig = data[data["Adjusted P-value"] <= 0.1]
    
    ##* Select DNA repair related Terms     
    data = data[(data["Term"].str.contains("repair", case=False)) | (data["Term"].str.contains("DNA Damage", case=False))]

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

            plt.xlim(left=0.5, right=6)
            print(data[data["logq"]>=1.0].shape[0], path)
            print(data[data["logq"]>=1.0].shape[0], path)
        
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

            plt.xlim(left=0.5, right=10)
            print(data[data["logq"]>=1.0].shape[0], path)
            print(data[data["logq"]>=1.0].shape[0], path)
        
        
        for n, label in enumerate(ax.yaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)

        plt.xlabel("$-Log_{10}$FDR", fontsize=13)
        plt.ylabel("")
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(13)

        genedf = pd.DataFrame(data[data["logq"]>=2.0]['Genes'])
        genedf.columns = ['Genes']
        genes = pd.concat([genes,genedf])
        
        ################################### DUT vs. distribution 3#########################################################
        plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/GOenrichment_DUT_"+path+".pdf", bbox_inches="tight")
        #plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/GOenrichment_dist_DUT_"+path2+".pdf", bbox_inches="tight")
        ################################### DUT vs. distribution 3#########################################################
    else:
        print("No significant data!!! in", path)
        
    df["type"] = path
    merged_df = merged_df.append(df)

merged_df = merged_df.reset_index()
merged_df = merged_df.drop("index", axis=1)
merged_df = merged_df.reset_index()
merged_df = merged_df.sort_values(by="type")
print(merged_df)

#%%
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
plt.xlim(-10,270)
#plt.xticks([35,85,115,145],["AR stable", "AR variable","IR stable","IR variable"], fontsize=13)
plt.xticks([35,105,175,240],["AR stable", "AR variable","IR stable","IR variable"], fontsize=13)
plt.ylabel("$-Log_{10}$FDR", fontsize=14)
plt.axhline(2.0, color="#CACACA", linestyle="--")

################################### DUT vs. distribution 3#########################################################
#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/DUTplot"+path+"_DUT_bubbleplot.pdf", bbox_inches="tight")
plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/DUTplot"+path2+"_dist_DUT_bubbleplot.pdf", bbox_inches="tight")
################################### DUT vs. distribution 3#########################################################



#%% #^^ AR vs. IR venn diagram
plt.figure(figsize=(6,6))
sns.set_style("white")
from matplotlib_venn import venn2

###** DUT gene ###
###### whole DUT gene #####

AR = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/majorDUT/responder_stable_DUT_Wilcoxon.txt',  sep='\t')
ARdut = set(AR[(AR['p_value']<0.05) & (np.abs(AR['log2FC'])>1.5)]['Gene Symbol'])

IR = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/majorDUT/nonresponder_stable_DUT_Wilcoxon.txt',  sep='\t')
IRdut = set(IR[(IR['p_value']<0.05) & (np.abs(IR['log2FC'])>1.5)]['Gene Symbol'])

vd2 = venn2([ARdut, IRdut],set_labels=('AR', 'IR'),set_colors=('#F8CF6A','#74B2D4'), alpha=0.8)

for text in vd2.set_labels:  #change label size
    text.set_fontsize(13)
for text in vd2.subset_labels:  #change number size
    text.set_fontsize(16)
    
vd2.get_patch_by_id('11').set_color('#C8C452')
vd2.get_patch_by_id('11').set_alpha(1)

plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/wholeDUT_ARIR_Venn.pdf", bbox_inches="tight")
plt.show()

###### only DNA repair & DNA damage repair######
ARenr = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/majorDUT/responder_stable_DUT_GOenrichment.txt', sep='\t')
ARenr = ARenr[ARenr["Adjusted P-value"] <= 0.01]
ARenr = ARenr[(ARenr["Term"].str.contains("repair", case=False)) | (ARenr["Term"].str.contains("DNA Damage", case=False))]
ARlist = set([gene for sublist in ARenr['Genes'].str.split(';') for gene in sublist])

IRenr = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/majorDUT/nonresponder_stable_DUT_GOenrichment.txt', sep='\t')
IRenr = IRenr[IRenr["Adjusted P-value"] <= 0.01]
IRenr = IRenr[(IRenr["Term"].str.contains("repair", case=False)) | (IRenr["Term"].str.contains("DNA Damage", case=False))]
IRlist = set([gene for sublist in IRenr['Genes'].str.split(';') for gene in sublist])

vd2 = venn2([ARlist, IRlist],set_labels=('AR', 'IR'),set_colors=('#F8CF6A','#74B2D4'), alpha=0.8)

for text in vd2.set_labels:  #change label size
    text.set_fontsize(13)
for text in vd2.subset_labels:  #change number size
    text.set_fontsize(16)
    
vd2.get_patch_by_id('11').set_color('#C8C452')
vd2.get_patch_by_id('11').set_alpha(1)

plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/repairDUT_ARIR_Venn.pdf", bbox_inches="tight")
plt.show()

#%%
###** distribution DUT gene ###
###### whole DUT gene #####
AR = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUT/AR_stable_distribution_result.txt',  sep='\t')
ARdut = set(AR[(AR['P-value']<0.05) ]['GeneName'])

IR = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUT/IR_stable_distribution_result.txt',  sep='\t')
IRdut = set(IR[(IR['P-value']<0.05) ]['GeneName'])

vd2 = venn2([ARdut, IRdut],set_labels=('AR', 'IR'),set_colors=('#F8CF6A','#74B2D4'), alpha=0.8)

for text in vd2.set_labels:  #change label size
    text.set_fontsize(13)
for text in vd2.subset_labels:  #change number size
    text.set_fontsize(16)
    
vd2.get_patch_by_id('11').set_color('#C8C452')
vd2.get_patch_by_id('11').set_alpha(1)

plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUTplot/wholeDistribution_ARIR_Venn.pdf", bbox_inches="tight")
plt.show()

##### only DNA repair & DNA damage repair######
ARenr = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUT/AR_stable_distDUT_GOenrichment.txt', sep='\t')
ARenr = ARenr[ARenr["Adjusted P-value"] <= 0.1]
ARenr = ARenr[(ARenr["Term"].str.contains("repair", case=False)) | (ARenr["Term"].str.contains("DNA Damage", case=False))]
ARlist = set([gene for sublist in ARenr['Genes'].str.split(';') for gene in sublist])

IRenr = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUT/IR_stable_distDUT_GOenrichment.txt', sep='\t')
IRenr = IRenr[IRenr["Adjusted P-value"] <= 0.1]
IRenr = IRenr[(IRenr["Term"].str.contains("repair", case=False)) | (IRenr["Term"].str.contains("DNA Damage", case=False))]
IRlist = set([gene for sublist in IRenr['Genes'].str.split(';') for gene in sublist])

vd2 = venn2([ARlist, IRlist],set_labels=('AR', 'IR'),set_colors=('#F8CF6A','#74B2D4'), alpha=0.8)

for text in vd2.set_labels:  #change label size
    text.set_fontsize(13)
for text in vd2.subset_labels:  #change number size
    text.set_fontsize(16)
    
vd2.get_patch_by_id('11').set_color('#C8C452')
vd2.get_patch_by_id('11').set_alpha(1)

plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUTplot/repairDistribution_ARIR_Venn.pdf", bbox_inches="tight")
plt.show()



#%%
with open('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUTgene/AR_DUTgenelist.txt') as file:
    newgDUTlist = [line.rstrip() for line in file]

ARdut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t')
IRdut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/nonresponder_stable_DUT_Wilcoxon.txt', sep='\t')

major = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = major[major['type']=='major']

gDUTlist = list(set(ARdut.loc[(ARdut['p_value']<0.05) & (ARdut['log2FC']>1.5), 'Gene Symbol']))
DUTlist = list(ARdut.loc[(ARdut['p_value']<0.05) & (ARdut['log2FC']>1.5), 'gene_ENST'])
enr = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202401_analysis/DUT/responder_stable_Wilcoxon_GOenrichment_FC_15.txt', sep='\t')
enr = enr[(enr["Term"].str.contains("repair", case=False)) | (enr["Term"].str.contains("DNA Damage", case=False))]
enr = enr[enr["Adjusted P-value"]<=0.01]
genelist = [gene for sublist in enr['Genes'].str.split(';') for gene in sublist]
genelist = list(set(genelist))
enrmajorlist = majorlist[majorlist['genename'].isin(genelist)]['gene_ENST']
enrmajordutlist = set(enrmajorlist).intersection(set(DUTlist))


#%%

major = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/final_202306_discovery_major_TU.txt', sep='\t', index_col=0)
minor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/final_202306_discovery_minor_TU.txt', sep='\t', index_col=0)

majortrans = major.index.tolist()
minortrans = minor.index.tolist()

def process_transcripts(majortrans, minortrans):
    gene_dict = {}
    
    for item in majortrans:
        transcript, gene = item.rsplit('-', 1)
        if gene not in gene_dict:
            gene_dict[gene] = {'major': transcript, 'minor': []}
        else:
            gene_dict[gene]['minor'].append(transcript)
    
    for item in minortrans:
        transcript, gene = item.rsplit('-', 1)
        if gene in gene_dict:
            gene_dict[gene]['minor'].append(transcript)
        else:
            gene_dict[gene] = {'major': None, 'minor': [transcript]}
    
    data = []
    for gene, transcripts in gene_dict.items():
        major_transcript = transcripts['major']
        query_list = transcripts['minor']
        if major_transcript:
            data.append({'genename': gene, 'major_list': major_transcript, 'query_list': query_list})
    
    return pd.DataFrame(data)

# Create the dataframe
result_df = process_transcripts(majortrans, minortrans)
result_df.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/NMD_inputquery_whole.txt', sep='\t', index=False)

#%%
repair_damage = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/NMD_inputquery_repair_damage.txt', sep='\t')
genelist = list(repair_damage['genename'])

newdf = result_df[result_df['genename'].isin(genelist)]










#%%

def adjust_transcripts(row):
    # Check if major_list is not empty
    if row['major_list']:
        if len(row['major_list']) > 1:
            # Move all but the first major transcript to the minor list
            row['query_list'].extend(row['major_list'][1:])
        # Keep only the first major transcript and remove the list wrapper
        row['major_list'] = row['major_list'][0]
    else:
        # If major_list is empty, assign None or an empty string, depending on your preference
        row['major_list'] = None  # or row['major_list'] = ''
    return row

# Apply the function to each row
new_df = result_df.apply(adjust_transcripts, axis=1)
#%%
new_df = new_df.drop(['genename'], axis=1)

# %%
result_df.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/NMD_inputquery.txt', sep='\t', index=False)
new_df.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/NMD_inputquery2.txt', sep='\t', index=False)
# %%
