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
##^^ gene with major DUT & distribution gene: AR vs. IR
##^^ major list -> added list? or canonical list?
#%%
def read_go_file(file_path):
    go_terms = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into GO term and genes, assuming they are separated by tabs
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                go_term, genes = parts
                genes_list = genes.split()  # Splitting gene symbols which are separated by spaces
                go_terms[go_term] = genes_list
    return go_terms

go_terms = read_go_file('/home/jiye/jiye/copycomparison/OC_transcriptome/GO_Biological_Process_2021.txt')
TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_TU.txt', sep='\t')
ratio = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/majorminor/major2minorratio.txt', sep='\t') 
sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
responder = sampleinfo.loc[(sampleinfo['response']==1),'sample_full'].to_list()
nonresponder = sampleinfo.loc[(sampleinfo['response']==0),'sample_full'].to_list()
major = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')

majorlist = major[major['type']=='major']
#%%
##* AR
ARdut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t')
major = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = major[major['type']=='major']


###* AR ####
gDUTlist = list(set(ARdut.loc[(ARdut['p_value']<0.05) & (np.abs(ARdut['log2FC'])>1.5), 'Gene Symbol']))
DUTlist = list(ARdut.loc[(ARdut['p_value']<0.05) & (np.abs(ARdut['log2FC'])>1.5), 'gene_ENST'])

##############################################################
enr = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUT/responder_stable_DUT_GOenrichment.txt', sep='\t')
#enr = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUT/AR_stable_distDUT_GOenrichment.txt', sep='\t')
##############################################################
enr = enr[(enr["Term"].str.contains("repair", case=False)) | (enr["Term"].str.contains("DNA Damage", case=False))]
enr = enr[enr["Adjusted P-value"]<=0.01]
#%%

for i in range(7):
    enr2 = enr.iloc[i,:]
    genelist = enr2['Genes'].split(';')
    genelist = list(set(genelist))
    enrmajorlist = majorlist[majorlist['genename'].isin(genelist)]['gene_ENST']
    enrmajordutlist = set(enrmajorlist).intersection(set(DUTlist))

    filtered_TU = TU[TU['gene_ENST'].isin(enrmajordutlist)]

    new_df = ratio.set_index(['gene'])
    new_df = new_df[new_df.index.isin(genelist)]
    ####################responder / nonresponder############################
    new_df = new_df[responder]
    ########################################################################

    df_bfD = new_df[[col for col in new_df.columns if '-bfD' in col]]
    df_atD = new_df[[col for col in new_df.columns if '-atD' in col]]

    mean_bfD = list(df_bfD.mean(axis=1))
    mean_atD = list(df_atD.mean(axis=1))
    meandict = {'pre': mean_bfD, 'post': mean_atD}
    meandf = pd.DataFrame(meandict)
    meandf.index = new_df.index

    plt.figure(figsize=(4,6))
    sns.set_style("white")

    col1 = ["#70A8F2","#F36464"]

    tmplist = meandf.index.to_list()
    tmplist = tmplist*2

    ##
    figureinput = meandf.melt(var_name='Treatment', value_name='ratio')
    figureinput['gene'] = tmplist
    ##


    ax = sns.boxplot(x="Treatment", y="ratio", data=figureinput,  palette=col1, showfliers=False)
    plt.ylim(0,1.65)

    from statannot import add_stat_annotation

    add_stat_annotation(ax, data=figureinput, x='Treatment', y='ratio',
                box_pairs=[("pre", "post")],
                test='Wilcoxon', text_format='simple', loc='outside', comparisons_correction=None,
                )
    plt.figtext(0.5, 1.005, enr2[1], ha='center', va='center', fontsize=13)
    #plt.title(namelist2[i] + ' - ' + typelist[j]+' transcript exp', fontsize=13, y=3)
    medians = figureinput.groupby(['Treatment'])['ratio'].median()
    medians = medians.iloc[::-1]
    medians = medians.round(decimals=5)
    vertical_offset = figureinput['ratio'].median() * 0.3

    for xtick in ax.get_xticks():
        print(xtick)
        ax.text(xtick, medians[xtick] + vertical_offset, medians[xtick], 
                horizontalalignment='center',size='medium',color='w',weight='semibold')
        
    sns.despine()
#%%
##*AR
dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t')
###################################################################
dutdf= dut.loc[(dut['p_value']<0.05) & (dut['log2FC']>1.5)]
majorDUTgene = set(dutdf['Gene Symbol'])
#majordutdf = dutdf[dutdf['gene_ENST'].isin(majorlist)]
#majorDUTgene = set(majordutdf['Gene Symbol'])
###################################################################


with open('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUTgene/AR_DUTgenelist.txt') as file:
    distDUTlist = [line.rstrip() for line in file]

#AR_DUTgene = set(distDUTlist).union(majorDUTgene)
AR_DUTgene = majorDUTgene
print(len(distDUTlist), len(majorDUTgene))

#%%
##* IR
dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/nonresponder_stable_DUT_Wilcoxon.txt', sep='\t')

###################################################################
dutdf= dut.loc[(dut['p_value']<0.05) & (dut['log2FC']>1.5)]
majorDUTgene = set(dutdf['Gene Symbol'])
#majordutdf = dutdf[dutdf['gene_ENST'].isin(majorlist)]
#majorDUTgene = set(majordutdf['Gene Symbol'])
###################################################################

with open('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUTgene/IR_DUTgenelist.txt') as file:
    distDUTlist = [line.rstrip() for line in file]

#IR_DUTgene = set(distDUTlist).union(majorDUTgene)
IR_DUTgene = majorDUTgene

print(len(distDUTlist), len(majorDUTgene))

#################### MODEL INPUT ##########################
#bothlist = list(AR_DUTgene.union(IR_DUTgene))
#GO_DUTgene = list(set(bothlist).intersection(set(go_terms['double-strand break repair (GO:0006302)'])))

# with open('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUTgene/dsb_ARIR_both_DUTlist.txt', "w") as file:
#     for item in GO_DUTgene:
#         file.write("%s\n" % item)


# %%

term_list = ['cellular response to DNA damage stimulus (GO:0006974)', 'DNA repair (GO:0006281)', 'DNA damage response, signal transduction by p53 class mediator (GO:0030330)', 'mitotic G1 DNA damage checkpoint signaling (GO:0031571)', 'nucleotide-excision repair (GO:0006289)','double-strand break repair (GO:0006302)']


DUTlist = [AR_DUTgene, IR_DUTgene]
responder = sampleinfo.loc[(sampleinfo['response']==1),'sample_full'].to_list()
nonresponder = sampleinfo.loc[(sampleinfo['response']==0),'sample_full'].to_list()
samplelist = [responder,nonresponder]
namelist = ['AR','IR']

###^^^ Major Sum #########################
for term in term_list:
    GO_genelist = go_terms[term]
    for i in range(0,2):
        enrmajorlist = majorlist[majorlist['genename'].isin(GO_genelist)]['gene_ENST']
        new_df = TU.set_index(['gene_ENST'])
        new_df = new_df[new_df.index.isin(enrmajorlist)]

        ###################responder / nonresponder##########################
        new_df = new_df[samplelist[i]]
        #####################################################################

        ## summation of majot TUs in same gene ##
        new_df['gene'] = new_df.index.str.split("-",1).str[1]
        new_df = new_df.reset_index()
        new_df = new_df.groupby('gene').sum()
        new_df = new_df.reset_index().set_index('gene')

        df_bfD = new_df[[col for col in new_df.columns if '-bfD' in col]]
        df_atD = new_df[[col for col in new_df.columns if '-atD' in col]]

        mean_bfD = list(df_bfD.mean(axis=1))
        mean_atD = list(df_atD.mean(axis=1))
        meandict = {'pre': mean_bfD, 'post': mean_atD}
        meandf = pd.DataFrame(meandict)
        meandf.index = new_df.index

        plt.figure(figsize=(4,6))
        sns.set_style("white")

        col1 = ["#70A8F2","#F36464"]

        tmplist = meandf.index.to_list()
        tmplist = tmplist*2

        ##
        figureinput = meandf.melt(var_name='Treatment', value_name='TU')
        figureinput['gene'] = tmplist
        ##


        ax = sns.boxplot(x="Treatment", y="TU", data=figureinput,  palette=col1, showfliers=False)
        #plt.ylim(0,0.16)

        from statannot import add_stat_annotation

        add_stat_annotation(ax, data=figureinput, x='Treatment', y='TU',
                    box_pairs=[("pre", "post")],
                    test='Wilcoxon', text_format='simple', loc='outside', #comparisons_correction=None,
                    )
        plt.figtext(0.5, 1.005, term, ha='center', va='center', fontsize=13)
        #plt.title(namelist2[i] + ' - ' + typelist[j]+' transcript exp', fontsize=13, y=3)
        medians = figureinput.groupby(['Treatment'])['TU'].median()
        medians = medians.iloc[::-1]
        medians = medians.round(decimals=5)
        vertical_offset = figureinput['TU'].median() * 0.1

        for xtick in ax.get_xticks():
            print(xtick)
            ax.text(xtick, medians[xtick] + vertical_offset, medians[xtick], 
                    horizontalalignment='center',size='medium',color='w',weight='semibold')
            
        sns.despine()
        
        
#%%


#%%
ratio = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/majorminor/major2minorratio.txt', sep='\t') 
ratio = ratio.set_index('gene')

        
# %%
enr_formodel = enr.loc[(4,57),:]
genelist = [gene for sublist in enr_formodel['Genes'].str.split(';') for gene in sublist]

#enr_formodel = enr.loc[(177),:]
#genelist = enr_formodel['Genes'].split(';')
genelist = list(set(genelist))

modelinput = ratio[ratio.index.isin(genelist)]
#modelinput.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/model/ddr_dsb_ratio.txt', sep='\t')
# %%
enr_formodel = enr.loc[(4,57),:]
genelist = [gene for sublist in enr_formodel['Genes'].str.split(';') for gene in sublist]

#enr_formodel = enr.loc[(177),:]
#genelist = enr_formodel['Genes'].split(';')
genelist = list(set(genelist))
DUTlist = list(ARdut.loc[(ARdut['p_value']<0.05) & (ARdut['log2FC']>1.5), 'gene_ENST'])

enrmajorlist = majorlist[majorlist['genename'].isin(genelist)]['gene_ENST']
enrmajordutlist = set(enrmajorlist).intersection(set(DUTlist))
# %%
enr_formodel = enr.loc[(4,57),:]
genelist = [gene for sublist in enr_formodel['Genes'].str.split(';') for gene in sublist]

#enr_formodel = enr.loc[(177),:]
#genelist = enr_formodel['Genes'].split(';')
genelist = list(set(genelist))
#%%
modelinput2 = TU[TU['gene_ENST'].isin(enrmajordutlist)]
modelinput2.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/model/ddr_repair_majordut_TU.txt', sep='\t', index=False)

# %%
