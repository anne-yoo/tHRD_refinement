#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.stats import pearsonr
import random


#%%
def makeplotdf(i):
        
    WES_matched = pd.read_csv('//home/jiye/jiye/copycomparison/HRgenes_data/AFdata/pre-post-WES-HR')
    WGS_matched = pd.read_csv('//home/jiye/jiye/copycomparison/HRgenes_data/AFdata/pre-post-WGS-HR')

    filelist = [WES_matched, WGS_matched]

    plotnamelist = ['WES_matched', 'WGS_matched']

    clinical = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/final_clinicaldata')

    onlypre_WES_matched_samplelist = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/only-pre_WES_matched_samplelist', sep="\t", header=None)
    onlypre_WGS_matched_samplelist = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/only-pre_WGS_matched_samplelist', sep="\t", header=None)
    onlypre_WES_matched_samplelist.columns = ['sample_ID','pre/post','wes','rna']
    onlypre_WGS_matched_samplelist.columns = ['sample_ID','pre/post','wgs','rna']
    matchedlist = [onlypre_WES_matched_samplelist,onlypre_WGS_matched_samplelist]

    unique_clinical = clinical.drop(['Seq_time'],axis=1)
    unique_clinical['sample_ID']=unique_clinical['sample_ID'].str.replace('-atD','')
    unique_clinical['sample_ID']=unique_clinical['sample_ID'].str.replace('-bfD','')
    unique_clinical.drop_duplicates(['sample_ID'],keep='first',ignore_index=True, inplace=True)
    unique_clinical.rename(columns = {'sample_ID':'sample'}, inplace=True)


    if i==0:
        file = filelist[i]
        file['sample'] = file['sample_ID']
        file['sample'] = file['sample'].str.replace('-atD','')
        file['sample'] = file['sample'].str.replace('-bfD','')

        matchedfile = pd.merge(matchedlist[0], file, on='sample_ID')
        mergedfile = pd.merge(matchedfile, unique_clinical, on='sample')
        duration_sorted = mergedfile.sort_values('Duration')
        duration_sorted['deltaAF'] = duration_sorted['RNA_AF'] - duration_sorted['genome_AF']

        duration_sorted['therapy_duration'] = 'tmp'
        # duration_sorted['Duration'] = duration_sorted['Duration'].astype(int)
        duration_sorted.loc[duration_sorted['Duration']<=180,'therapy_duration'] = 'short'
        duration_sorted.loc[(duration_sorted['Duration']>180) & (duration_sorted['Duration']<=365),'therapy_duration'] ='medium'
        duration_sorted.loc[duration_sorted['Duration']>365,'therapy_duration'] = 'long'
        
    elif i==1:
        file = filelist[i]
        file['sample'] = file['sample_ID']
        file['sample'] = file['sample'].str.replace('-atD','')
        file['sample'] = file['sample'].str.replace('-bfD','')

        matchedfile = pd.merge(matchedlist[1], file, on='sample_ID')
        mergedfile = pd.merge(matchedfile, unique_clinical, on='sample')
        duration_sorted = mergedfile.sort_values('Duration')
        duration_sorted['deltaAF'] = duration_sorted['RNA_AF'] - duration_sorted['genome_AF']

        duration_sorted['therapy_duration'] = 'tmp'
        # duration_sorted['Duration'] = duration_sorted['Duration'].astype(int)
        duration_sorted.loc[duration_sorted['Duration']<=180,'therapy_duration'] = 'short'
        duration_sorted.loc[(duration_sorted['Duration']>180) & (duration_sorted['Duration']<=365),'therapy_duration'] ='medium'
        duration_sorted.loc[duration_sorted['Duration']>365,'therapy_duration'] = 'long'

    return duration_sorted


#%%
#^^ +) hypothesis 1-1: correlation matrix delta AF ~ Duration
geneid = pd.read_csv('/home/jiye/jiye/copycomparison/HRgeneid_ENSG.csv',header=None,sep=',')
geneid.columns=['geneid','ENSG']
wesplot = makeplotdf(0)
wgsplot = makeplotdf(1)
dflist = [wesplot,wgsplot]
filenamelist = ['WES pre samples', 'WGS pre samples']

wesdf = wesplot
wesdf = wesdf[wesdf['pre/post']=='pre']
wesdf = wesdf[wesdf['Purpose']=='Maintenance'] 
condition1 = (0.4<=wesdf['genome_AF']) & (wesdf['genome_AF'] <= 0.6)
wesdf = wesdf.loc[condition1,:]

wesdf = pd.merge(wesdf,geneid,on='ENSG')
###&########
wesdf = wesdf[wesdf['geneid']=='RAD51D']
###&########
wesdf = wesdf[['sample_ID','geneid','genome_AF','deltaAF','Duration']]
meandf = pd.DataFrame(wesdf.groupby('sample_ID')['deltaAF'].mean())
duration = wesdf[['sample_ID','Duration']]
duration= duration.drop_duplicates(subset=['sample_ID'])

finalwes = pd.merge(meandf,duration,on='sample_ID')
# print(wesdf.groupby('geneid')[['Duration','deltaAF']].corr().unstack().iloc[:,1])
sns.set(rc = {'figure.figsize':(4,4)})
sns.set_style('whitegrid')
# plt.suptitle('<'+filenamelist[i]+': duration ~ delta AF>', fontsize=11)
# sns.scatterplot(data=corrdf, x="Duration", y="deltaAF", color='red')
g=sns.lmplot(data=finalwes, x="Duration", y="deltaAF")
g.set(ylim=(-0.7, 0.7))
plt.title('<'+filenamelist[0]+': duration ~ delta AF>')
plt.show()
#%%
wgsdf = wgsplot
wgsdf = wgsdf[wgsdf['pre/post']=='pre']
wgsdf = wgsdf[wgsdf['Purpose']=='Maintenance'] 
condition1 = (0.4<=wgsdf['genome_AF']) & (wgsdf['genome_AF'] <= 0.6)
wgsdf = wgsdf.loc[condition1,:]
wgsdf = pd.merge(wgsdf,geneid,on='ENSG')
###&########
wgsdf = wgsdf[wgsdf['geneid']=='RAD51D']
###&########
wgsdf = wgsdf[['sample_ID','geneid','genome_AF','deltaAF','Duration']]

meandf = pd.DataFrame(wgsdf.groupby('sample_ID')['deltaAF'].mean())
duration = wgsdf[['sample_ID','Duration']]
duration= duration.drop_duplicates(subset=['sample_ID'])
finalwgs = pd.merge(meandf,duration,on='sample_ID')
# print(wgsdf.groupby('geneid')[['Duration','deltaAF']].corr().unstack().iloc[:,1])
sns.set(rc = {'figure.figsize':(4,4)})
sns.set_style('whitegrid')
# plt.suptitle('<'+filenamelist[i]+': duration ~ delta AF>', fontsize=11)
# sns.scatterplot(data=corrdf, x="Duration", y="deltaAF", color='red')
g=sns.lmplot(data=finalwgs, x="Duration", y="deltaAF")
g.set(ylim=(-0.7, 0.7))
plt.title('<'+filenamelist[1]+': duration ~ delta AF>')
plt.show()


#%%
#^^ +) hypothesis 1-3: correlation matrix AI ~ duration
geneid = pd.read_csv('/home/jiye/jiye/copycomparison/HRgeneid_ENSG.csv',header=None,sep=',')
geneid.columns=['geneid','ENSG']
wesplot = makeplotdf(0)
wgsplot = makeplotdf(1)
dflist = [wesplot,wgsplot]
filenamelist = ['WES pre samples', 'WGS pre samples']

wesdf = wesplot
wesdf = wesdf[wesdf['pre/post']=='pre']
wesdf = wesdf[wesdf['Purpose']=='Maintenance'] 
wesdf['AI'] = wesdf['RNA_AF'] - 0.5
condition1 = (0.4<=wesdf['genome_AF']) & (wesdf['genome_AF'] <= 0.6)
wesdf = wesdf.loc[condition1,:]
wesdf = pd.merge(wesdf,geneid,on='ENSG')
###&########
wesdf = wesdf[wesdf['geneid']=='RAD51D']
###&########
wesdf = wesdf[['sample_ID','geneid','genome_AF','AI','Duration']]
meandf = pd.DataFrame(wesdf.groupby('sample_ID')['AI'].mean())
duration = wesdf[['sample_ID','Duration']]
duration= duration.drop_duplicates(subset=['sample_ID'])

finalwes = pd.merge(meandf,duration,on='sample_ID')
# print(wesdf.groupby('geneid')[['Duration','deltaAF']].corr().unstack().iloc[:,1])
sns.set(rc = {'figure.figsize':(4,4)})
sns.set_style('whitegrid')
# plt.suptitle('<'+filenamelist[i]+': duration ~ delta AF>', fontsize=11)
# sns.scatterplot(data=corrdf, x="Duration", y="deltaAF", color='red')
g=sns.lmplot(data=finalwes, x="Duration", y="AI")
g.set(ylim=(-0.7, 0.7))
plt.title('<'+filenamelist[0]+': duration ~ AI>')
plt.show()


wgsdf = wgsplot
wgsdf = wgsdf[wgsdf['pre/post']=='pre']
wgsdf = wgsdf[wgsdf['Purpose']=='Maintenance'] 
wgsdf['AI'] = wgsdf['RNA_AF'] - 0.5
condition1 = (0.4<=wgsdf['genome_AF']) & (wgsdf['genome_AF'] <= 0.6)
wgsdf = wgsdf.loc[condition1,:]
wgsdf = pd.merge(wgsdf,geneid,on='ENSG')
###&########
wgsdf = wgsdf[wgsdf['geneid']=='RAD51D']
###&########
wgsdf = wgsdf[['sample_ID','geneid','genome_AF','AI','Duration']]
meandf = pd.DataFrame(wgsdf.groupby('sample_ID')['AI'].mean())
duration = wgsdf[['sample_ID','Duration']]
duration= duration.drop_duplicates(subset=['sample_ID'])

finalwgs = pd.merge(meandf,duration,on='sample_ID')
# print(wesdf.groupby('geneid')[['Duration','deltaAF']].corr().unstack().iloc[:,1])
sns.set(rc = {'figure.figsize':(4,4)})
sns.set_style('whitegrid')
# plt.suptitle('<'+filenamelist[i]+': duration ~ delta AF>', fontsize=11)
# sns.scatterplot(data=corrdf, x="Duration", y="deltaAF", color='red')
g=sns.lmplot(data=finalwgs, x="Duration", y="AI")
g.set(ylim=(-0.7, 0.7))
plt.title('<'+filenamelist[1]+': duration ~ AI>')
plt.show()

# %%
