#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# %%
WES_RNA_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WES_RNA_BRCA1.csv')
WES_RNA_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WES_RNA_BRCA2.csv')
WGS_RNA_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WGS_RNA_BRCA1.csv')
WGS_RNA_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WGS_RNA_BRCA2.csv')

WES_genome_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WES_genome_BRCA1.csv')
WES_genome_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WES_genome_BRCA2.csv')
WGS_genome_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WGS_genome_BRCA1.csv')
WGS_genome_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WGS_genome_BRCA2.csv')

# %%

cldata = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/newdata/RNA_clinical.csv')
cldata = cldata.loc[:,['sample','mutation','reason']]
cldata.columns = ['sample_ID','BRCA_status','reason']

files = [WES_RNA_BRCA1, WES_RNA_BRCA2, WGS_RNA_BRCA1, WGS_RNA_BRCA2]
filenames = ['WES_RNA_BRCA1', 'WES_RNA_BRCA2', 'WGS_RNA_BRCA1', 'WGS_RNA_BRCA2']

# for i in range(4):
#     newfile = pd.merge(cldata, files[i], on='sample_ID')
#     newfile = newfile[newfile['mut']=='wt']
#     newfile.to_csv('/home/jiye/jiye/copycomparison/variantinfo/withsalvage/depth5/onlyRNAwt/'+ filenames[i]+'.csv', index=False)






#%%
WES_RNA_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WES_RNA_BRCA1.csv')
WES_RNA_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WES_RNA_BRCA2.csv')
WGS_RNA_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WGS_RNA_BRCA1.csv')
WGS_RNA_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WGS_RNA_BRCA2.csv')