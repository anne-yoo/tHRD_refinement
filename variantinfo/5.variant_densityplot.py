#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# %%
####& plot AF_RNA - AF_genome density plot: MIN DEPTH = 10 ####

WES_RNA_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth10/WES_RNA_BRCA1.csv')
WES_RNA_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth10/WES_RNA_BRCA2.csv')
WGS_RNA_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth10/WGS_RNA_BRCA1.csv')
WGS_RNA_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth10/WGS_RNA_BRCA2.csv')

WES_genome_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth10/WES_genome_BRCA1.csv')
WES_genome_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth10/WES_genome_BRCA2.csv')
WGS_genome_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth10/WGS_genome_BRCA1.csv')
WGS_genome_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth10/WGS_genome_BRCA2.csv')


# %%
####^ by IMPACT ####
WES_RNA_BRCA1['delta_AF'] = WES_RNA_BRCA1['RNA_AF'] - WES_RNA_BRCA1['genome_AF']
plt.figure(dpi=150)
sns.set(rc = {'figure.figsize':(8,5)})
sns.set_style('whitegrid')
sns.kdeplot(data=WES_RNA_BRCA1, x='delta_AF', hue="impact", fill=True, common_norm=False, alpha=0.4)
plt.title('< WES_BRCA1: RNA_AF - genome_AF by IMPACT>', pad=10)
plt.savefig('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth10/figures_depth10/WES_BRCA1_IMPACT.png',dpi=150)
plt.show()

# %%
####^ by response ####
WES_RNA_BRCA1['delta_AF'] = WES_RNA_BRCA1['RNA_AF'] - WES_RNA_BRCA1['genome_AF']
plt.figure(dpi=150)
sns.set(rc = {'figure.figsize':(8,5)})
sns.set_style('whitegrid')
sns.kdeplot(data=WES_RNA_BRCA1, x='delta_AF', hue="response", fill=True, common_norm=False, alpha=0.4)
plt.title('< WES_BRCA1: RNA_AF - genome_AF by RESPONSE>', pad=10)
plt.savefig('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth10/figures_depth10/WES_BRCA1_RESPONSE.png',dpi=150)
plt.show()


























# %%
####& plot AF_RNA - AF_genome density plot: MIN DEPTH = 5 ####

WES_RNA_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WES_RNA_BRCA1.csv')
WES_RNA_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WES_RNA_BRCA2.csv')
WGS_RNA_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WGS_RNA_BRCA1.csv')
WGS_RNA_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WGS_RNA_BRCA2.csv')

WES_genome_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WES_genome_BRCA1.csv')
WES_genome_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WES_genome_BRCA2.csv')
WGS_genome_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WGS_genome_BRCA1.csv')
WGS_genome_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WGS_genome_BRCA2.csv')


# %%
####^ by IMPACT ####
WGS_RNA_BRCA1['delta_AF'] = WGS_RNA_BRCA1['RNA_AF'] - WGS_RNA_BRCA1['genome_AF']
plt.figure(dpi=150)
sns.set(rc = {'figure.figsize':(8,5)})
sns.set_style('whitegrid')
sns.kdeplot(data=WGS_RNA_BRCA1, x='delta_AF', hue="impact", fill=True, common_norm=False, alpha=0.4)
plt.title('< WGS_BRCA1: RNA_AF - genome_AF by IMPACT>', pad=10)
plt.savefig('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/figures_depth5/WGS_BRCA1_IMPACT.png',dpi=150)
plt.show()
# %%
####^ by response ####
WGS_RNA_BRCA1['delta_AF'] = WGS_RNA_BRCA1['RNA_AF'] - WGS_RNA_BRCA1['genome_AF']
plt.figure(dpi=150)
sns.set(rc = {'figure.figsize':(8,5)})
sns.set_style('whitegrid')
sns.kdeplot(data=WGS_RNA_BRCA1, x='delta_AF', hue="response", fill=True, common_norm=False, alpha=0.4)
plt.title('< WGS_BRCA1: RNA_AF - genome_AF by RESPONSE>', pad=10)
plt.savefig('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/figures_depth5/WGS_BRCA1_RESPONSE.png',dpi=150)
plt.show()

# %%
