#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# %%
####* BRCA1 depth distribution 1~10
filepath = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/newdata/RNA_filepath.csv')
plt.figure(dpi=130)
sns.set(rc = {'figure.figsize':(8,4)})
sns.set_style('whitegrid')
for i in range(filepath.shape[0]):
    filename = filepath.iloc[i,1]
    filename = filename.replace('.vcf','')
    depth = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/newdata/depth_RNA_BRCA1/brca1-' + filename + '-depth' , sep='\t', header=None, on_bad_lines='skip')
    depth.columns = ['chr','loc','read depth']
    depth = depth[depth['read depth']!=0]
    depth = depth[depth['read depth']<11]
    sns.kdeplot(data=depth, x='read depth')
    
    
# %%
####* BRCA1 depth==0 proportion
filepath = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/newdata/RNA_filepath.csv')
countzero = []
countwhole = []

for i in range(filepath.shape[0]):
    filename = filepath.iloc[i,1]
    filename = filename.replace('.vcf','')
    depth = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/newdata/depth_RNA_BRCA1/brca1-' + filename + '-depth' , sep='\t', header=None, on_bad_lines='skip')
    depth.columns = ['chr','loc','read depth']
    count = depth[depth['read depth']==0].shape[0]
    countzero.append(count)
    countwhole.append(depth.shape[0])

#%%
countdf = pd.DataFrame(list(zip(countzero,countwhole)), columns=['countzero','countwhole'])
countdf['percentage'] = (countdf['countzero']/countdf['countwhole'])*100
countdf['percentage'] = countdf['percentage'].round(2)
countdf['nonzero'] = countdf['countwhole'] - countdf['countzero']





# %%
####^ Count mut / wt RNAseq : MIN DEPTH = 10 ####

WES_RNA_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth10/WES_RNA_BRCA1.csv')
WES_RNA_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth10/WES_RNA_BRCA2.csv')
WGS_RNA_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth10/WGS_RNA_BRCA1.csv')
WGS_RNA_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth10/WGS_RNA_BRCA2.csv')

mt = WES_RNA_BRCA1[WES_RNA_BRCA1['mut']=='mt']
wt = WES_RNA_BRCA1[WES_RNA_BRCA1['mut']=='wt']







# %%
####^ Count mut / wt RNAseq : MIN DEPTH = 5 ####

WES_RNA_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WES_RNA_BRCA1.csv')
WES_RNA_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WES_RNA_BRCA2.csv')
WGS_RNA_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WGS_RNA_BRCA1.csv')
WGS_RNA_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WGS_RNA_BRCA2.csv')


mt = WES_RNA_BRCA1[WES_RNA_BRCA1['mut']=='mt']
wt = WES_RNA_BRCA1[WES_RNA_BRCA1['mut']=='wt']

condition1 = (0.4<=mt['genome_AF']) & (mt['genome_AF'] <= 0.6)
condition2 = (0.4<=wt['genome_AF']) & (wt['genome_AF'] <= 0.6)

mt = mt.loc[condition1,:]
wt = wt.loc[condition2,:]

impactlist = ['HIGH','MODERATE','LOW','MODIFIER']
for impact in impactlist:
    print( mt[mt['impact']==impact].shape[0])

for impact in impactlist:
    print( wt[wt['impact']==impact].shape[0])



#%%
WES_RNA_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WES_RNA_BRCA1.csv')
WES_RNA_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WES_RNA_BRCA2.csv')
WGS_RNA_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WGS_RNA_BRCA1.csv')
WGS_RNA_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WGS_RNA_BRCA2.csv')

WES_genome_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WES_genome_BRCA1.csv')
WES_genome_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WES_genome_BRCA2.csv')
WGS_genome_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WGS_genome_BRCA1.csv')
WGS_genome_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WGS_genome_BRCA2.csv')

#%%
dt = WGS_genome_BRCA2
condition1 = (0.4<=dt['genome_AF']) & (dt['genome_AF'] <= 0.6)
dt = dt.loc[condition1,:]
print(dt['sample_ID'].nunique())
impactlist = ['HIGH','MODERATE','LOW','MODIFIER']
for impact in impactlist:
    print( dt[dt['impact']==impact].shape[0])






#%%
WES_genome_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WES_genome_BRCA1.csv')
WES_genome_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WES_genome_BRCA2.csv')
WGS_genome_BRCA1 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WGS_genome_BRCA1.csv')
WGS_genome_BRCA2 = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/variantinfo/withsalvage/depth5/WGS_genome_BRCA2.csv')








# %%
