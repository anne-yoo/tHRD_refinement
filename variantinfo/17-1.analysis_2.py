#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu 


# %%
###########^ data download ##########

#* each location's AF / variant impact + consequence
WES_RNA_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WES_RNA_BRCA1.csv')
WES_RNA_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WES_RNA_BRCA2.csv')
WGS_RNA_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WGS_RNA_BRCA1.csv')
WGS_RNA_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WGS_RNA_BRCA2.csv')

WES_genome_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WES_genome_BRCA1.csv')
WES_genome_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WES_genome_BRCA2.csv')
WGS_genome_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WGS_genome_BRCA1.csv')
WGS_genome_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WGS_genome_BRCA2.csv')

#* clinical data + filename information
clinical = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/control_PD/clinicaldata_final.csv')
clinical = clinical.dropna()
WES_filelist = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/control_PD/WES_RNA_filelist_final.csv')
WGS_filelist = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/control_PD/WGS_RNA_filelist_final.csv')

filelist = [WES_RNA_BRCA1, WES_RNA_BRCA2, WGS_RNA_BRCA1, WGS_RNA_BRCA2]
filenamelist = ['WES_BRCA1', 'WES_BRCA2', 'WGS_BRCA1', 'WGS_BRCA2']




# %%
###########^ simple boxplot code ###########

def drawboxplot(inputnumber):
    file = filelist[inputnumber]
    file['deltaAF'] = file['RNA_AF'] - file['genome_AF']
    df = pd.merge(file,clinical,on='sample_ID')

    df['ctr_exp'] = 'none'
    con1 = (df['Purpose']=='Maintenance') & (df['Seq_time']=='before')
    con2 = (df['Purpose']=='Maintenance') & (df['Seq_time']=='after') & (df['Stop_reason']=='1')
    con3 = (df['Purpose']=='Salvage') 

    df.loc[con1,'ctr_exp'] = 'ctr'
    df.loc[con2,'ctr_exp'] = 'exp'
    df.loc[con3, 'ctr_exp'] = 'exp'

    df = df[df['ctr_exp']!= 'none']
    
    sns.set(rc = {'figure.figsize':(6.5,6)})
    sns.set_style('whitegrid')
    ax = sns.boxplot(data=df, y='deltaAF', x="ctr_exp", hue='response', palette='hls',showfliers = False)
    handles, labels = ax.get_legend_handles_labels()

    sns.stripplot(data=df, y='deltaAF', x="ctr_exp", hue='response', palette=['#C62D3C','#00A5AA'], dodge=True, size=5, alpha=0.6)
    plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0. );
    plt.suptitle('< Analysis 2 | ' + filenamelist[inputnumber] +' deltaAF' + ': ctr vs. exp >',y=0.96, x=0.55,fontsize=13)
    
    # plt.savefig('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/violin_depth5/'+filenamelist[i]+'.png', bbox_inches='tight', dpi=150)
    plt.show()




# %%
drawboxplot(2)
# %%
