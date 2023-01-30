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

    main = df[df['Purpose']=='Maintenance']
    sal = df[df['Purpose']=='Salvage']

    main['ctr_exp'] = 'none'
    sal['ctr_exp'] = 'none'

    main.loc[main['Seq_time']=='before','ctr_exp'] = 'ctr_1'
    con1 = (main['Seq_time']=='after') & (main['Stop_reason']=='1')
    main.loc[con1,'ctr_exp'] = 'exp_1'

    sal.loc[sal['Seq_time']=='before','ctr_exp'] = 'ctr_2'
    con2 = (sal['Seq_time']=='after') & (sal['Stop_reason']=='1')
    sal.loc[con2,'ctr_exp'] = 'exp_2'     

    main = main[main['ctr_exp']!='none']
    sal = sal[sal['ctr_exp']!='none']

    # print(main[(main['ctr_exp']=='exp_1') & (main['response']=='responder')]['sample_ID'].unique())

    fig, axes = plt.subplots(1,2, sharey=True, figsize=(12,6))
    fig.subplots_adjust(wspace=0.1)
    sns.set_style('whitegrid')
    plt.suptitle('< Analysis 1 | ' + filenamelist[inputnumber] +' deltaAF' + ': ctr vs. exp >',y=0.96, x=0.55,fontsize=13)
    
    sns.stripplot(ax=axes[0], data=main, y='deltaAF', x="ctr_exp", hue='response',palette=['#C62D3C','#00A5AA'], dodge=True, size=5, alpha=0.6)
    b = sns.boxplot(ax=axes[0], data=main, y='deltaAF', x="ctr_exp", hue='response', palette='hls', showfliers = False)
    b.legend().set_visible(False)

    sns.stripplot(ax=axes[1], data=sal, y='deltaAF', x="ctr_exp", hue='response', palette=['#C62D3C','#00A5AA'], dodge=True, size=5, alpha=0.6)
    ax = sns.boxplot(ax=axes[1], data=sal, y='deltaAF', x="ctr_exp", hue='response', palette='hls',showfliers = False )
    
    axes[0].set_title("Maintenance")
    axes[1].set_title("Salvage")
    handles, labels = ax.get_legend_handles_labels()
    
    plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0. );
    # plt.savefig('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/violin_depth5/'+filenamelist[i]+'.png', bbox_inches='tight', dpi=150)
    plt.show()




# %%
drawboxplot(2)
# %%
