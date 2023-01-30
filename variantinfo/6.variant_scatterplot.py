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

#%%
####^ by IMPACT ####
plt.figure(dpi=150)
ax = plt.gca()
ax.set_aspect(0.7)
sns.set(rc = {'figure.figsize':(8,6)})
sns.set_style('whitegrid')
sns.lineplot(x=[0,1],y=[0,1], size=1, linestyle='--', color='black', alpha=0.8, legend=None)
sns.scatterplot(data=WGS_RNA_BRCA2, x='genome_AF', y='RNA_AF', hue="impact", palette="hls")
plt.title('< WGS_BRCA2: genome_AF ~ RNA AF by IMPACT>', pad=10)
plt.savefig('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth10/scatterplot_depth10/WGS_BRCA2_IMPACT.png',dpi=150)
plt.show()


#%%
####^ by RESPONSE ####
plt.figure(dpi=150)
ax = plt.gca()
ax.set_aspect(0.7)
sns.set(rc = {'figure.figsize':(8,5)})
sns.set_style('whitegrid')
sns.lineplot(x=[0,1],y=[0,1], size=1, linestyle='--', color='black', alpha=0.8, legend=None)
sns.scatterplot(data=WGS_RNA_BRCA2, x='genome_AF', y='RNA_AF', hue="response", palette="hls")
plt.title('< WGS_BRCA2: genome_AF ~ RNA AF by RESPONSE>', pad=10)
plt.savefig('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth10/scatterplot_depth10/WGS_BRCA2_RESPONSE.png',dpi=150)
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

#%%
####^ by IMPACT ####
plt.figure(dpi=150)
ax = plt.gca()
ax.set_aspect(0.7)
sns.set(rc = {'figure.figsize':(8,6)})
sns.set_style('whitegrid')
sns.lineplot(x=[0,1],y=[0,1], size=1, linestyle='--', color='black', alpha=0.8, legend=None)
sns.scatterplot(data=WES_RNA_BRCA2, x='genome_AF', y='RNA_AF', hue="impact", palette="hls")
plt.title('< WES_BRCA2: genome_AF ~ RNA AF by IMPACT>', pad=10)
plt.savefig('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/scatterplot_depth5/WES_BRCA2_IMPACT.png',dpi=150)
plt.show()


#%%
####^ by RESPONSE ####
plt.figure(dpi=150)
ax = plt.gca()
ax.set_aspect(0.7)
sns.set(rc = {'figure.figsize':(8,5)})
sns.set_style('whitegrid')
sns.lineplot(x=[0,1],y=[0,1], size=1, linestyle='--', color='black', alpha=0.8, legend=None)
sns.scatterplot(data=WES_RNA_BRCA2, x='genome_AF', y='RNA_AF', hue="response", palette="hls")
plt.title('< WES_BRCA2: genome_AF ~ RNA AF by RESPONSE>', pad=10)
plt.savefig('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/scatterplot_depth5/WES_BRCA2_RESPONSE.png',dpi=150)
plt.show()

# %%
