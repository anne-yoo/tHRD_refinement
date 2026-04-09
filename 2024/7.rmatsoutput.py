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
# Read the data into a pandas DataFrame
df = pd.read_csv("/home/jiye/jiye/copycomparison/gDUTresearch/rMATS/individual/FDR_psi_jcec_summary.txt", sep='\t', header=None, names=["Sample", "AS_Event", "GeneSymbol"])



def count_common_genes(df, threshold):
    # Calculate the total number of unique samples
    total_samples = len(df["Sample"].unique())
    
    # Group by AS_Event, GeneSymbol, and GeneType, then count unique samples for each group
    gene_counts = df.groupby(["AS_Event", "GeneSymbol"])["Sample"].nunique().reset_index(name="SampleCount")
    
    # Calculate the percentage of samples for each gene
    gene_counts["Percentage"] = (gene_counts["SampleCount"] / total_samples)
    
    # Filter genes appearing in more than threshold% of the samples
    common_genes = gene_counts[gene_counts["Percentage"] > threshold]
    
    # Group by AS_Event and GeneType and count the genes
    final_counts = common_genes.groupby(["AS_Event"]).size().reset_index(name="GeneCount")
    
    genes_list_df = common_genes.drop(columns=["SampleCount", "Percentage"]).sort_values(by=["AS_Event", "GeneSymbol"])
    
    
    return final_counts, common_genes

# Function to plot the gene counts 
def plot_gene_counts(data, title):
    sns.set(style="whitegrid")
    plt.figure(figsize=(7, 3))
    ax = sns.barplot(x="GeneCount", y="AS_Event", data=data, palette="viridis")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Gene Count")
    ax.set_ylabel("AS Event")
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_width())}', 
                    (p.get_width(), p.get_y() + p.get_height() / 2), 
                    ha='center', va='center', 
                    fontsize=11.5, color='white',
                    xytext=(-16, 0), 
                    textcoords='offset points')
        
    plt.show()



###################plot ###################################################################
# Plot the common gene counts for a given threshold
threshold = 0.0  # You can change this threshold as needed
counts = count_common_genes(df, threshold)[0]
plot_gene_counts(counts, f"Common Gene Counts (Threshold: {threshold * 100}%)")
############################################################################################

################# save gene list ###########################################################
print(count_common_genes(df, threshold)[1])
############################################################################################
# %%
###^^^ AR / IR ####

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
responder = list(set(sampleinfo.loc[(sampleinfo['response']==1),'sample_id']))
nonresponder = list(set(sampleinfo.loc[(sampleinfo['response']==0),'sample_id']))

# %%
AR = df[df['Sample'].isin(responder)]
IR = df[df['Sample'].isin(nonresponder)]

#%%
threshold = 0.0

gene_counts = AR.groupby("AS_Event")["GeneSymbol"].agg(lambda x: len(set(x))).reset_index()
gene_counts.columns = ['AS_Event','GeneCount']
# You can change this threshold as needed
plot_gene_counts(gene_counts, "AR: Whole Gene Counts")
common_genes = count_common_genes(AR, threshold)
plot_gene_counts(common_genes, f"AR: Common Gene Counts (Threshold: {threshold * 100}%)")

gene_counts = IR.groupby("AS_Event")["GeneSymbol"].agg(lambda x: len(set(x))).reset_index()
gene_counts.columns = ['AS_Event','GeneCount']
# You can change this threshold as needed
plot_gene_counts(gene_counts, "IR: Whole Gene Counts")
common_genes = count_common_genes(IR, threshold)
plot_gene_counts(common_genes, f"IR: Common Gene Counts (Threshold: {threshold * 100}%)")

# %%
###^ save for enrichR

whole = count_common_genes(df, 0)[1]
ARgene = count_common_genes(AR, 0)[1]
IRgene = count_common_genes(IR, 0)[1]
# whole = set(whole['GeneSymbol'])
# ARgene = set(ARgene['GeneSymbol'])
# IRgene = set(IRgene['GeneSymbol'])





#%%
with open('/home/jiye/jiye/copycomparison/gDUTresearch/rMATS/individual/enrichr/whole_gene.txt', "w") as file:
    for item in whole:
        file.write("%s\n" % item)

with open('/home/jiye/jiye/copycomparison/gDUTresearch/rMATS/individual/enrichr/AR_gene.txt', "w") as file:
    for item in ARgene:
        file.write("%s\n" % item)

with open('/home/jiye/jiye/copycomparison/gDUTresearch/rMATS/individual/enrichr/IR_gene.txt', "w") as file:
    for item in IRgene:
        file.write("%s\n" % item)

# %%
onlyAR = list(set(ARgene) - set(IRgene))
onlyIR = list(set(IRgene) - set(ARgene))

with open('/home/jiye/jiye/copycomparison/gDUTresearch/rMATS/individual/enrichr/onlyAR_gene.txt', "w") as file:
    for item in onlyAR:
        file.write("%s\n" % item)

with open('/home/jiye/jiye/copycomparison/gDUTresearch/rMATS/individual/enrichr/onlyIR_gene.txt', "w") as file:
    for item in onlyIR:
        file.write("%s\n" % item)




# %%
ARdegresult = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DEG/responder_Wilcoxon_DEGresult_FC.txt', sep='\t')

ARvariable = set(ARdegresult[ARdegresult['p_value']<0.05]['Gene Symbol'])
ARstable = set(ARdegresult[ARdegresult['p_value'] > 0.05]['Gene Symbol'])

IRdegresult = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DEG/nonresponder_Wilcoxon_DEGresult_FC.txt', sep='\t')

IRvariable = set(IRdegresult[IRdegresult['p_value']<0.05]['Gene Symbol'])
IRstable = set(IRdegresult[IRdegresult['p_value'] > 0.05]['Gene Symbol'])

wholedegresult = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DEG/whole_Wilcoxon_DEGresult_FC.txt', sep='\t')

wholevariable = set(wholedegresult[wholedegresult['p_value']<0.05]['Gene Symbol'])
wholestable = set(wholedegresult[wholedegresult['p_value'] > 0.05]['Gene Symbol'])

def count_common_genes(df, stable_genes, variable_genes, threshold):
    # Calculate the total number of unique samples
    total_samples = len(df["Sample"].unique())
    
    # Identify each gene as stable or variable
    df["GeneType"] = df["GeneSymbol"].apply(lambda x: "Stable" if x in stable_genes else ("Variable" if x in variable_genes else None))
    
    # Keep only rows where GeneType is not None
    df = df[df["GeneType"].notna()]
    
    ### Group by AS_Event, GeneSymbol, and GeneType, then count unique samples for each group
    #gene_counts = df.groupby(["AS_Event", "GeneSymbol", "GeneType"])["Sample"].nunique().reset_index(name="SampleCount")
    gene_counts = df.groupby(["AS_Event", "GeneSymbol"])["Sample"].nunique().reset_index(name="SampleCount")
    
    # Calculate the percentage of samples for each gene
    gene_counts["Percentage"] = (gene_counts["SampleCount"] / total_samples)
    
    # Filter genes appearing in more than threshold% of the samples
    #common_genes = gene_counts[gene_counts["Percentage"] > threshold]
    common_genes = gene_counts[gene_counts["SampleCount"] > threshold]
    
    ### Group by AS_Event and GeneType and count the genes
    #final_counts = common_genes.groupby(["AS_Event", "GeneType"]).size().reset_index(name="GeneCount")
    final_counts = common_genes.groupby(["AS_Event"]).size().reset_index(name="GeneCount")
    
    return final_counts, common_genes

# Function to plot the gene counts with split bars for stable and variable genes
def plot_gene_counts(data, title):
    sns.set_style("whitegrid")
    plt.figure(figsize=(6, 4))
    
    plt.rcParams.update({
        'axes.titlesize': 13,     # 제목 글꼴 크기
        'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
        'xtick.labelsize': 12,    # x축 틱 라벨 글꼴 크기
        'ytick.labelsize': 12,    # y축 틱 라벨 글꼴 크기
        'legend.fontsize': 11,
        'legend.title_fontsize': 11, # 범례 글꼴 크기
        'figure.titlesize': 15    # figure 제목 글꼴 크기
        })
    # Plot the combined data with hue parameter
    ax = sns.barplot(x="GeneCount", y="AS_Event", hue="GeneType", data=data, palette={"Stable": "skyblue", "Variable": "salmon"})
    
    
    plt.legend()
    #ax.set_title(title)
    ax.set_xlabel("Gene Count")
    ax.set_ylabel("AS Event")
    plt.xlim(0,5000)
    sns.despine()
    
#%%
# Example threshold (you can change this as needed)
threshold = 10
# Count common genes based on threshold and genea lists
AR_common_genes = count_common_genes(AR, ARstable, ARvariable, threshold)[0]
IR_common_genes = count_common_genes(IR, IRstable, IRvariable, threshold)[0]

colors = {'Variable': '#AA8976', 'Stable': '#70AF85',}
sns.set_style("whitegrid")
plt.figure(figsize=(6, 4))

plt.rcParams.update({
    'axes.titlesize': 13,     # 제목 글꼴 크기
    'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
    'xtick.labelsize': 12,    # x축 틱 라벨 글꼴 크기
    'ytick.labelsize': 13,    # y축 틱 라벨 글꼴 크기
    'legend.fontsize': 11,
    'legend.title_fontsize': 11, # 범례 글꼴 크기
    'figure.titlesize': 15    # figure 제목 글꼴 크기
    })

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(5, 3), sharey=True, gridspec_kw={'wspace': 0})
sns.barplot(data=AR_common_genes, x='GeneCount', y='AS_Event', 
            #palette='Set2',
            color='#81A263',
            #hue='GeneType', palette=colors,
            ci=False, orient='horizontal',  ax=ax2)
ax2.yaxis.set_label_position('right')
ax2.tick_params(axis='y', labelright=True, right=True)
#ax2.set_title('  '+'AR', loc='left')
#plt.legend(title='Gene Class')

sns.barplot(data=IR_common_genes,x='GeneCount', y='AS_Event', 
            color='#81A263',
            #hue='GeneType',palette=colors,
            ci=False, orient='horizontal',  ax=ax1)

# optionally use the same scale left and right
xmax = max(ax1.get_xlim()[1], ax2.get_xlim()[1])
ax1.set_xlim(xmax=xmax)
ax2.set_xlim(xmax=xmax)
ax2.set_ylabel('')
ax2.set_xlabel('')
ax1.set_xlabel('')

ax1.invert_xaxis()  # reverse the direction
ax1.tick_params(labelleft=False, left=False)
ax1.set_ylabel('')
#ax1.set_title('IR'+'  ', loc='right')
#ax1.legend_.remove()

#ticks = range(0, int(xmax) + 1000, 3000)
#ax1.set_xticks(ticks)
#ax2.set_xticks(ticks)

plt.tight_layout()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/fig1/FDR_rmats_count_ARIR.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%import matplotlib.pyplot as plt
import seaborn as sns

# Assuming AR_common_genes and IR_common_genes are already defined DataFrames

colors = {'Variable': '#AA8976', 'Stable': '#70AF85',}
sns.set_style("whitegrid")

plt.rcParams.update({
    'axes.titlesize': 13,     # 제목 글꼴 크기
    'axes.labelsize': 12,     # x, y축 라벨 글꼴 크기
    'xtick.labelsize': 12,    # x축 틱 라벨 글꼴 크기
    'ytick.labelsize': 13,    # y축 틱 라벨 글꼴 크기
    'legend.fontsize': 11,
    'legend.title_fontsize': 11, # 범례 글꼴 크기
    'figure.titlesize': 15    # figure 제목 글꼴 크기
    })

AR_common_genes['sample_class'] = 'AR'
IR_common_genes['sample_class'] = 'IR'

fig, ax = plt.subplots(figsize=(4, 5))


figureinput = pd.concat([AR_common_genes,IR_common_genes], axis=0)
df_pivot = figureinput.pivot_table(values='GeneCount', index='sample_class', columns='AS_Event', aggfunc='sum').fillna(0)
df_pivot.plot(kind='bar', stacked=True, colormap='Set2',ax=ax)
plt.legend(title='AS Event', bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
ax.set_xlabel('Sample Class')
ax.set_ylabel('Gene Count')
sns.despine()
plt.tight_layout()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/fig1/FDR_rmats_count_ARIR_stacked_over10.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%
#%%
##^^^^^^^ GO enrichment ######
import gseapy as gp
threshold = 10
ARgene = count_common_genes(AR, ARstable, ARvariable, threshold)[1]
IRgene = count_common_genes(IR, IRstable, IRvariable, threshold)[1]

eventslist = ['MXE','SE','RI','A3SS','A5SS']

#%%
###
###
#glist = list(set(ARgene['GeneSymbol']))
for e in eventslist:
    glist =list(set(IRgene[IRgene['AS_Event']==e]['GeneSymbol']))

    enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                    gene_sets=[#'GO_Biological_Process_2021','GO_Biological_Process_2018',
                                'GO_Biological_Process_2023',
                                'Reactome_2022'
                                ], 
                    organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                    outdir=None, # don't write to disk
    )

    enrresult = enr.results.sort_values(by=['Adjusted P-value']) 

    file = enrresult
    def string_fraction_to_float(fraction_str):
        numerator, denominator = fraction_str.split('/')
        return float(numerator) / float(denominator)

    file['per'] = file['Overlap'].apply(string_fraction_to_float)

    file = file.sort_values('Adjusted P-value')
    #file = file.sort_values(by='Combined Score', ascending=False)
    ##remove mitochondrial ##
    file['Term'] = file['Term'].str.rsplit(" ",1).str[0]
    #file = file[~file['Term'].str.contains('mitochondrial')]
    #file = file.iloc[:10,:]
    file = file[file['Adjusted P-value']<0.05]
    file['Adjusted P-value'] = -np.log10(file['Adjusted P-value'])

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams.update({
    'axes.titlesize': 13,     # 제목 글꼴 크기
    'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
    'xtick.labelsize': 12,    # x축 틱 라벨 글꼴 크기
    'ytick.labelsize': 12,    # y축 틱 라벨 글꼴 크기
    'legend.fontsize': 13,
    'legend.title_fontsize': 13, # 범례 글꼴 크기
    'figure.titlesize': 15    # figure 제목 글꼴 크기
    })
    
    if file.shape[0]<13 and file.shape[0]>0:
        plt.figure(figsize=(5,2))
        sns.set_style("white")
        sns.barplot(
            data=file, x='Adjusted P-value', y='Term', color='#81A263',
            #hue='per', 
            #palette='coolwarm', 
            #edgecolor=None, legend=False, 
        )
        plt.xlabel('-log10(FDR)')
        plt.ylabel('')
        plt.title(e)
        #plt.yticks(fontsize=13)
        #plt.xscale('log')  # Log scale for better visualization

        # Expanding the plot layout to make room for GO term labels
        plt.gcf().subplots_adjust(left=0.5)
        # Set y-axis limits to remove vacant areas
        #plt.ylim(file['Term'].min(), file['Term'].max())
        #plt.gcf().subplots_adjust(right=0.8)

        sns.despine()
        plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/fig1/DSG_GO/2_IR_'+e+'.pdf', dpi=300, bbox_inches='tight')
        plt.show()
    
    elif file.shape[0]>=13 and file.shape[0]<25:
        plt.figure(figsize=(5,5))
        sns.set_style("white")
        sns.barplot(
            data=file, x='Adjusted P-value', y='Term', color='#81A263',
            #hue='per', 
            #palette='coolwarm', 
            #edgecolor=None, legend=False, 
        )
        plt.xlabel('-log10(FDR)')
        plt.ylabel('')
        plt.title(e)
        plt.gcf().subplots_adjust(left=0.5)
        sns.despine()
        plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/fig1/DSG_GO/2_IR_'+e+'.pdf', dpi=300, bbox_inches='tight')
        plt.show()
    
    elif file.shape[0]>=25:
        plt.figure(figsize=(5,14))
        sns.set_style("white")
        sns.barplot(
            data=file, x='Adjusted P-value', y='Term', color='#81A263',
            #hue='per',
            #edgecolor=None, legend=False, 
        )
        plt.xlabel('-log10(FDR)')
        plt.ylabel('')
        plt.title(e)
        plt.gcf().subplots_adjust(left=0.5)
        sns.despine()
        plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/fig1/DSG_GO/2_IR_'+e+'.pdf', dpi=300, bbox_inches='tight')
        plt.show()


#%%
# Plot the gene counts with split bars for stable and variable genes
plot_gene_counts(AR_common_genes, f"whole: Common Gene Counts (Threshold: {threshold * 100}%)")
#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/whole_50_AScount.pdf", bbox_inches="tight")

# plot_gene_counts(common_genes, f"AR: Whole Gene Counts")
# plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/AR_whole_AScount.pdf", bbox_inches="tight")

# %%
from matplotlib_venn import venn2

def get_stable_genes_over_threshold(df, stable_genes, threshold):
    # Calculate the total number of unique samples
    total_samples = len(df["Sample"].unique())
    
    # Identify each gene as stable or variable
    df["GeneType"] = df["GeneSymbol"].apply(lambda x: "Stable" if x in stable_genes else None)
    
    # Keep only rows where GeneType is 'Stable'
    df_stable = df[df["GeneType"] == "Stable"]
    
    # Group by GeneSymbol, then count unique samples for each gene
    gene_counts = df_stable.groupby("GeneSymbol")["Sample"].nunique().reset_index(name="SampleCount")
    
    # Calculate the percentage of samples for each gene
    gene_counts["Percentage"] = (gene_counts["SampleCount"] / total_samples)
    
    # Filter genes appearing in more than threshold% of the samples
    common_genes = gene_counts[gene_counts["Percentage"] > threshold]["GeneSymbol"].unique()
    
    return set(common_genes)

stable_genes_dataset1 = get_stable_genes_over_threshold(AR, ARstable, threshold)
stable_genes_dataset2 = get_stable_genes_over_threshold(IR, IRstable, threshold)

#%%