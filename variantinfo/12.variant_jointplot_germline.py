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
    file['resistance'] = 'Acquired'
    con = (file['duration']<181)
    file.loc[con,'resistance'] = 'Innate'

    innate = file[file['resistance']=='Innate']
    acquired = file[file['resistance']=='Acquired']
    syn = file[file['consequence']=='synonymous_variant']
    filekind = [innate, acquired, syn]
    filekindname = ['Innate Resistance', 'Acquired Resistance', 'Control (synonymous variants)']
    forsaving = ['innate','acquired','syn']

    sns.set_style('whitegrid')
    sns.set(rc = {'figure.figsize':(9,10)})
    orders=['responder','non-responder']
    g=sns.jointplot(data=file, x='genome_AF', y='RNA_AF', hue="response", hue_order=orders, palette="hls")
    plt.suptitle('< ' + filenamelist[i] +'>',y=1.02, fontsize=12)
    plt.tight_layout()
    plt.xlim([0.3,0.7])
    x0, x1 = g.ax_joint.get_xlim()
    y0, y1 = g.ax_joint.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    g.ax_joint.plot(lims, lims, '--g')
    # plt.savefig('/home/omics/DATA6/jiye/copycomparison/variantinfo/modifiedAF/whole_jointplot_depth10/'+filenamelist[i]+'.png', bbox_inches='tight', dpi=150)

    plt.show()


#%%
for i in range(4):
    file = filelist[i]
    file['resistance'] = 'Acquired'
    con = (file['duration']<181)
    file.loc[con,'resistance'] = 'Innate'

    condition1 = (0.4<=file['genome_AF']) & (file['genome_AF'] <= 0.6)
    germline = file.loc[condition1,:]

    sns.set_style('whitegrid')
    sns.set(rc = {'figure.figsize':(9,10)})
    orders=['responder','non-responder']
    g= sns.jointplot(data=germline, x='genome_AF', y='RNA_AF', hue="response", hue_order=orders, palette="hls")
    plt.suptitle('< ' + filenamelist[i] +'>',y=1.02, fontsize=12)
    plt.tight_layout()

    x0, x1 = g.ax_joint.get_xlim()
    y0, y1 = g.ax_joint.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    g.ax_joint.plot(lims, lims, '--g')
    # plt.savefig('/home/omics/DATA6/jiye/copycomparison/variantinfo/modifiedAF/whole_germline_jointplot_depth10/'+filenamelist[i]+'.png', bbox_inches='tight', dpi=150)

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
    file['resistance'] = 'Acquired'
    con = (file['duration']<181)
    file.loc[con,'resistance'] = 'Innate'

    innate = file[file['resistance']=='Innate']
    acquired = file[file['resistance']=='Acquired']
    syn = file[file['consequence']=='synonymous_variant']
    filekind = [innate, acquired, syn]
    filekindname = ['Innate Resistance', 'Acquired Resistance', 'Control (synonymous variants)']
    forsaving = ['innate','acquired','syn']

    sns.set_style('whitegrid')
    sns.set(rc = {'figure.figsize':(9,10)})
    orders=['responder','non-responder']
    g=sns.jointplot(data=file, x='genome_AF', y='RNA_AF', hue="response", hue_order=orders, palette="hls")
    plt.suptitle('< ' + filenamelist[i] +'>',y=1.02, fontsize=12)
    plt.tight_layout()
    x0, x1 = g.ax_joint.get_xlim()
    y0, y1 = g.ax_joint.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    g.ax_joint.plot(lims, lims, '--g')
    plt.savefig('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/whole_jointplot_depth5/'+filenamelist[i]+'.png', bbox_inches='tight', dpi=150)

    plt.show()


#%%
for i in range(4):
    file = filelist[i]
    file['resistance'] = 'Acquired'
    con = (file['duration']<181)
    file.loc[con,'resistance'] = 'Innate'

    condition1 = (0.4<=file['genome_AF']) & (file['genome_AF'] <= 0.6)
    germline = file.loc[condition1,:]

    sns.set_style('whitegrid')
    sns.set(rc = {'figure.figsize':(9,10)})
    orders=['responder','non-responder']
    g= sns.jointplot(data=germline, x='genome_AF', y='RNA_AF', hue="response", hue_order=orders, palette="hls")
    plt.suptitle('< ' + filenamelist[i] +'>',y=1.02, fontsize=12)
    plt.tight_layout()
    x0, x1 = g.ax_joint.get_xlim()
    y0, y1 = g.ax_joint.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    g.ax_joint.plot(lims, lims, '--g')
    plt.savefig('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/whole_germline_jointplot_depth5/'+filenamelist[i]+'.png', bbox_inches='tight', dpi=150)

    plt.show()
# %%
