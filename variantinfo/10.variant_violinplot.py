#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# %%
####& plot AF_genome ~ AF_RNA joint plot: MIN DEPTH = 10 ####

WES_RNA_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/modifiedAF/WES_RNA_BRCA1.csv')
WES_RNA_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/modifiedAF/WES_RNA_BRCA2.csv')
WGS_RNA_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/modifiedAF/WGS_RNA_BRCA1.csv')
WGS_RNA_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/modifiedAF/WGS_RNA_BRCA2.csv')

filelist = [WES_RNA_BRCA1, WES_RNA_BRCA2, WGS_RNA_BRCA1, WGS_RNA_BRCA2]
filenamelist = ['WES_BRCA1', 'WES_BRCA2', 'WGS_BRCA1', 'WGS_BRCA2']
#%%
for i in range(4):
    file = filelist[i]
    file['deltaAF'] = file['RNA_AF'] - file['genome_AF']
    file['resistance'] = 'Acquired'
    con = (file['duration']<181)
    file.loc[con,'resistance'] = 'Innate'

    sns.set(rc = {'figure.figsize':(8,5)})
    sns.set_style('whitegrid')
    plt.suptitle('<'+filenamelist[i]+': RNA_AF - genome_AF by IMPACT>', fontsize=13)
    orders = ['HIGH','MODERATE','LOW','MODIFIER']
    sns.violinplot(data=file, y='deltaAF', x="impact", order=orders, hue='response', split=True, palette='hls')
    plt.legend(loc='upper left')
    plt.savefig('/home/omics/DATA6/jiye/copycomparison/variantinfo/modifiedAF/modifiedfigures/violinplot/'+filenamelist[i]+'.png', bbox_inches='tight', dpi=150)
    plt.show()











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
    file['resistance'] = 'Acquired'
    con = (file['duration']<181)
    file.loc[con,'resistance'] = 'Innate'

    sns.set(rc = {'figure.figsize':(8,5)})
    sns.set_style('whitegrid')
    plt.suptitle('<'+filenamelist[i]+': RNA_AF - genome_AF by IMPACT>', fontsize=13)
    orders = ['HIGH','MODERATE','LOW','MODIFIER']
    sns.violinplot(data=file, y='deltaAF', x="impact", order=orders, hue='response', split=True, palette='hls')
    plt.legend(loc='upper left')
    plt.savefig('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/violin_depth5/'+filenamelist[i]+'.png', bbox_inches='tight', dpi=150)
    plt.show()