#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

WES_RNA_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WES_RNA_BRCA1.csv')
WES_RNA_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WES_RNA_BRCA2.csv')
WGS_RNA_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WGS_RNA_BRCA1.csv')
WGS_RNA_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WGS_RNA_BRCA2.csv')

#%%
BRCAstat = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/newdata/sampleID_BRCAstatus.csv')

# %%
brca1 = (BRCAstat['BRCA_status']=='gBRCA1+') | (BRCAstat['BRCA_status']=='M(BRCA1/2)')
brca2 = (BRCAstat['BRCA_status']=='gBRCA2+') | (BRCAstat['BRCA_status']=='M(BRCA1/2)')

BRCA1 = BRCAstat.loc[brca1,:]
BRCA2 = BRCAstat.loc[brca2,:]

germline_WES_RNA_BRCA1 = pd.merge(BRCA1, WES_RNA_BRCA1, on='sample_ID')
germline_WES_RNA_BRCA2 = pd.merge(BRCA2, WES_RNA_BRCA2, on='sample_ID')
germline_WGS_RNA_BRCA1 = pd.merge(BRCA1, WGS_RNA_BRCA1, on='sample_ID')
germline_WGS_RNA_BRCA2 = pd.merge(BRCA1, WGS_RNA_BRCA2, on='sample_ID')


# %%
dt = germline_WGS_RNA_BRCA2
mt = dt[dt['mut']=='mt']
wt = dt[dt['mut']=='wt']

print(dt['sample_ID'].nunique())

impactlist = ['HIGH','MODERATE','LOW','MODIFIER']
for impact in impactlist:
    print( mt[mt['impact']==impact].shape[0])

for impact in impactlist:
    print( wt[wt['impact']==impact].shape[0])


# %%
WES_genome_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WES_genome_BRCA1.csv')
WES_genome_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WES_genome_BRCA2.csv')
WGS_genome_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WGS_genome_BRCA1.csv')
WGS_genome_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WGS_genome_BRCA2.csv')


germline_WES_genome_BRCA1 = pd.merge(BRCA1, WES_genome_BRCA1, on='sample_ID')
germline_WES_genome_BRCA2 = pd.merge(BRCA2, WES_genome_BRCA2, on='sample_ID')
germline_WGS_genome_BRCA1 = pd.merge(BRCA1, WGS_genome_BRCA1, on='sample_ID')
germline_WGS_genome_BRCA2 = pd.merge(BRCA1, WGS_genome_BRCA2, on='sample_ID')

# %%
dt = germline_WES_genome_BRCA1
print(dt['sample_ID'].nunique())
impactlist = ['HIGH','MODERATE','LOW','MODIFIER']
for impact in impactlist:
    print( dt[dt['impact']==impact].shape[0])