#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# %%
####& plot AF_genome ~ AF_RNA joint plot: MIN DEPTH = 5 ####

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
    condition1 = (0.4<=file['genome_AF']) & (file['genome_AF'] <= 0.6)
    condition2 = (0<file['genome_AF']) & (file['genome_AF'] < 0.4)

    germline = file.loc[condition1,:]
    somatic = file.loc[condition2,:]

    sns.set(rc = {'figure.figsize':(8,5)})
    sns.set_style('whitegrid')
    plt.suptitle('<'+filenamelist[i]+': RNA_AF - genome_AF (germline only)>', fontsize=13)
    orders = ['HIGH','MODERATE','LOW','MODIFIER']
    sns.violinplot(data=germline, y='deltaAF', x="impact", order=orders, hue='response', split=True, palette='hls')
    plt.legend(loc='upper left')
    plt.savefig('/home/omics/DATA6/jiye/copycomparison/variantinfo/modifiedAF/modifiedfigures/violinplot/germline-'+filenamelist[i]+'.png', bbox_inches='tight', dpi=150)
    plt.show()














# %%
for i in range(4):
    file = filelist[i]
    file['deltaAF'] = file['RNA_AF'] - file['genome_AF']
    file['resistance'] = 'Acquired'
    con = (file['duration']<181)
    file.loc[con,'resistance'] = 'Innate'
    condition1 = (0.4<=file['genome_AF']) & (file['genome_AF'] <= 0.6)
    condition2 = (0<file['genome_AF']) & (file['genome_AF'] < 0.4)

    germline = file.loc[condition1,:]
    somatic = file.loc[condition2,:]

    innate = file[file['resistance']=='Innate']
    acquired = file[file['resistance']=='Acquired']

    fig, axes = plt.subplots(1,2, sharey=True, figsize=(12,6))
    sns.set_style('whitegrid')
    fig.suptitle('<'+filenamelist[i]+': RNA_AF - genome_AF by IMPACT>', fontsize=13)
    orders = ['HIGH','MODERATE','LOW','MODIFIER']
    # palette_dict={'HIGH':'red','MODERATE':'lightgreen','LOW':'pink','MODIFIER':'skyblue'}
    sns.violinplot(ax=axes[0], data=innate, y='deltaAF', x="impact", order=orders, hue='response', split=True, palette='hls')
    sns.violinplot(ax=axes[1], data=acquired, y='deltaAF', x="impact", order=orders, hue='response', split=True, palette='hls')
    axes[0].set_title("Innate Resistance")
    axes[1].set_title("Acquired Resistance")
    plt.show()



    
# %%
