#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# %%
####& plot AF_genome ~ AF_RNA joint plot: MIN DEPTH = 10 ####

WES_RNA_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth10/WES_RNA_BRCA1.csv')
WES_RNA_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth10/WES_RNA_BRCA2.csv')
WGS_RNA_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth10/WGS_RNA_BRCA1.csv')
WGS_RNA_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth10/WGS_RNA_BRCA2.csv')

filelist = [WES_RNA_BRCA1, WES_RNA_BRCA2, WGS_RNA_BRCA1, WGS_RNA_BRCA2]
filenamelist = ['WES_BRCA1', 'WES_BRCA2', 'WGS_BRCA1', 'WGS_BRCA2']
#%%
for i in range(4):
    file = filelist[i]
    file['deltaAF'] = file['RNA_AF'] - file['genome_AF']
    
    res = file[file['response']=='responder']
    non_res = file[file['response']=='non-responder'] 
    
    fig, axes = plt.subplots(1,2, sharey=True, figsize=(12,6))
    sns.set_style('whitegrid')
    fig.suptitle('<'+filenamelist[i]+': RNA_AF - genome_AF by IMPACT>', fontsize=13)
    orders = ['HIGH','MODERATE','LOW','MODIFIER']
    # palette_dict={'HIGH':'red','MODERATE':'lightgreen','LOW':'pink','MODIFIER':'skyblue'}
    sns.boxplot(ax=axes[0], data=res, y='deltaAF', x="impact")
    sns.boxplot(ax=axes[1], data=non_res, y='deltaAF', x="impact")
    axes[0].set_title("Responder")
    axes[1].set_title("Non-Responder")
    plt.savefig('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth10/boxplot_depth10/'+filenamelist[i]+'.png', bbox_inches='tight', dpi=150)
    plt.show()
# %%











# %%
####& plot AF_genome ~ AF_RNA joint plot: MIN DEPTH = 5 ####

WES_RNA_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WES_RNA_BRCA1.csv')
WES_RNA_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WES_RNA_BRCA2.csv')
WGS_RNA_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WGS_RNA_BRCA1.csv')
WGS_RNA_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WGS_RNA_BRCA2.csv')

filelist = [WES_RNA_BRCA1, WES_RNA_BRCA2, WGS_RNA_BRCA1, WGS_RNA_BRCA2]
filenamelist = ['WES_BRCA1', 'WES_BRCA2', 'WGS_BRCA1', 'WGS_BRCA2']
#%%
for i in range(4):
    file = filelist[i]
    file['deltaAF'] = file['RNA_AF'] - file['genome_AF']
    
    res = file[file['response']=='responder']
    non_res = file[file['response']=='non-responder'] 
    
    fig, axes = plt.subplots(1,2, sharey=True, figsize=(12,6))
    sns.set_style('whitegrid')
    fig.suptitle('<'+filenamelist[i]+': RNA_AF - genome_AF by IMPACT>', fontsize=13)
    orders = ['HIGH','MODERATE','LOW','MODIFIER']
    # palette_dict={'HIGH':'red','MODERATE':'lightgreen','LOW':'pink','MODIFIER':'skyblue'}
    sns.boxplot(ax=axes[0], data=res, y='deltaAF', x="impact")
    sns.boxplot(ax=axes[1], data=non_res, y='deltaAF', x="impact")
    axes[0].set_title("Responder")
    axes[1].set_title("Non-Responder")
    plt.savefig('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/boxplot_depth5/'+filenamelist[i]+'.png', bbox_inches='tight', dpi=150)
    plt.show()
# %%
