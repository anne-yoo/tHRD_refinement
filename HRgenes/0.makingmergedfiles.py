
#? columns: loc / genome_AF / impact / patient_ID / response / duration
#? 

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import os.path


# %%
##########^ make new genome AF file  #########
#* 1. 샘플 별로 각각 만들기: chr | position | variantID | genome variant AF(ref) | RNA AF(ref) | consequence | impact
#* 2. 1번 파일들 머지해서 통으로 파일 하나 만들기: sample ID | chr | position | variantID | genome variant AF(ref) | RNA AF(ref) | consequence | impact 
#* 3. 샘플별로 각각 1번파일 만드는거 4번 matched로 하기 (genome pre file =genome post file)
#* 4. 5번 파일들 머지해서 통으로 파일 하나 만들기! 
#* AF = refCount/totalCount로 통일 

#%%
wesfilelist = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/WES_RNA_filelist_final',sep='\t',header=None)
wgsfilelist = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/WGS_RNA_filelist_final',sep='\t', header=None)

wesfilelist.columns = ['sample','genomefile','RNAfile']
wgsfilelist.columns = ['sample','genomefile','RNAfile']


# %%
#* 1. 샘플 별로 각각 만들기: chr | position | variantID | genome variant AF(ref) | RNA AF(ref) | consequence | impact
##^ WES_whole
for i in range(wesfilelist.shape[0]):
    # making chr | position | varinatID | RNA AF
    sample= wesfilelist.iloc[i,0]
    
    path = '/home/jiye/jiye/copycomparison/HRgenes_data/readcountdata/whole-sample/WESdata/readcount-WES-HR-'+sample
    if os.path.exists(path): #only non-empty file
        readcountdata = pd.read_csv('/home/jiye/jiye/copycomparison/HRgenes_data/readcountdata/whole-sample/WESdata/readcount-WES-HR-'+sample, sep='\t')
        if readcountdata.shape[0]>=1: 
            RNA_AF_df = readcountdata[['contig','position','variantID','refCount','totalCount']]
            RNA_AF_df['RNA_AF'] = RNA_AF_df.loc[:,'refCount']/RNA_AF_df.loc[:,'totalCount']
            RNA_AF_df['newpos'] = RNA_AF_df.agg('{0[contig]}:{0[position]}'.format, axis=1)

            RNA_AF = RNA_AF_df[['newpos','variantID','RNA_AF']]
            RNA_AF.columns = ['position','variantID','RNA_AF']

            # making genome AF | consequence | impact
            wesfile = wesfilelist.iloc[i,1]
            genome_AF_df = pd.read_csv('/home/jiye/jiye/copycomparison/HRgenes_data/genomeAFdata/WESdata/AF-nodup-filtered-ms-biallelic-HRgenes-'+wesfile+'.vep.vcf',sep='\t', header=None)
            genome_AF_df.columns =['contig','position','refCount','altCount']
            genome_AF_df['genome_AF'] = genome_AF_df.loc[:,'refCount']/(genome_AF_df.loc[:,'refCount']+genome_AF_df.loc[:,'altCount'])
            genome_AF_df['newpos'] = genome_AF_df.agg('{0[contig]}:{0[position]}'.format, axis=1)
            genome_AF = genome_AF_df[['newpos','genome_AF']]
            genome_AF.columns = ['position','genome_AF']

            variantinfo = pd.read_csv('~/jiye/copycomparison/HRgenes_data/WESdata/vep/final-vep-HRgenes-' + wesfilelist.iloc[i,1] + '.vep.vcf', sep='\t', header=None, on_bad_lines='skip')
            variantinfo.columns =['position','ENSG','ENST','consequence','impact']
            variantinfo['impact'] = variantinfo['impact'].str.replace('IMPACT=','')
            variantinfo = variantinfo[~variantinfo['position'].str.contains('-')] #only SNV
            variantinfo = variantinfo.loc[:,['position','ENSG','impact','consequence']]


            RNA_genome_merged = pd.merge(RNA_AF,genome_AF,on='position')
            final_merged = pd.merge(RNA_genome_merged,variantinfo, on='position')

            final_merged.to_csv('/home/jiye/jiye/copycomparison/HRgenes_data/AFdata/whole-sample/WESdata/AF_WES_HR-'+sample, index=False)
            

#%%
##^ WGS_whole
for i in range(wgsfilelist.shape[0]):
    # making chr | position | varinatID | RNA AF
    sample= wgsfilelist.iloc[i,0]
    
    path = '/home/jiye/jiye/copycomparison/HRgenes_data/readcountdata/whole-sample/WGSdata/readcount-WGS-HR-'+sample
    if os.path.exists(path): #only non-empty file
        readcountdata = pd.read_csv('/home/jiye/jiye/copycomparison/HRgenes_data/readcountdata/whole-sample/WGSdata/readcount-WGS-HR-'+sample, sep='\t')
        if readcountdata.shape[0]>=1: 
            RNA_AF_df = readcountdata[['contig','position','variantID','refCount','totalCount']]
            RNA_AF_df['RNA_AF'] = RNA_AF_df.loc[:,'refCount']/RNA_AF_df.loc[:,'totalCount']
            RNA_AF_df['newpos'] = 'pos'
            RNA_AF_df['newpos'] = RNA_AF_df.agg('{0[contig]}:{0[position]}'.format, axis=1)

            RNA_AF = RNA_AF_df[['newpos','variantID','RNA_AF']]
            RNA_AF.columns = ['position','variantID','RNA_AF']

            # making genome AF | consequence | impact
            wgsfile = wgsfilelist.iloc[i,1]
            genome_AF_df = pd.read_csv('/home/jiye/jiye/copycomparison/HRgenes_data/genomeAFdata/WGSdata/AF-nodup-filtered-ms-biallelic-HRgenes-'+wgsfile+'.vcf',sep='\t', header=None)
            genome_AF_df.columns =['contig','position','refCount','altCount']
            genome_AF_df['genome_AF'] = genome_AF_df.loc[:,'refCount']/(genome_AF_df.loc[:,'refCount']+genome_AF_df.loc[:,'altCount'])
            genome_AF_df['newpos'] = genome_AF_df.agg('{0[contig]}:{0[position]}'.format, axis=1)
            genome_AF = genome_AF_df[['newpos','genome_AF']]
            genome_AF.columns = ['position','genome_AF']

            variantinfo = pd.read_csv('~/jiye/copycomparison/HRgenes_data/WGSdata/vep/final-vep-HRgenes-' + wgsfilelist.iloc[i,1] + '.vcf', sep='\t', header=None, on_bad_lines='skip')
            variantinfo.columns =['position','ENSG','ENST','consequence','impact']
            variantinfo['impact'] = variantinfo['impact'].str.replace('IMPACT=','')
            variantinfo = variantinfo[~variantinfo['position'].str.contains('-')] #only SNV
            variantinfo = variantinfo.loc[:,['position','ENSG','impact','consequence']]

            RNA_genome_merged = pd.merge(RNA_AF,genome_AF,on='position')
            final_merged = pd.merge(RNA_genome_merged,variantinfo, on='position')

            final_merged.to_csv('/home/jiye/jiye/copycomparison/HRgenes_data/AFdata/whole-sample/WGSdata/AF_WGS_HR-'+sample, index=False)



############################################################################################&





############################################################################################&





# %%
#* 2. 1번 파일들 머지해서 통으로 파일 하나 만들기: sample ID | chr | position | variantID | genome variant AF(ref) | RNA AF(ref) | consequence | impact 
##^ WES_whole
import os.path

def makedf(i):
    sample = wesfilelist.iloc[i,0]
    tmpdf = pd.read_csv('/home/jiye/jiye/copycomparison/HRgenes_data/AFdata/whole-sample/WESdata/AF_WES_HR-'+sample, sep=',')
    tmpdf['sample_ID'] = sample
    return tmpdf

concatdfwhole = makedf(0)
for i in range(1, wesfilelist.shape[0]):
    sample = wesfilelist.iloc[i,0]
    path = '/home/jiye/jiye/copycomparison/HRgenes_data/AFdata/whole-sample/WESdata/AF_WES_HR-'+sample
    if os.path.exists(path):        
        tmpdf = makedf(i)
        concatdfwhole = pd.concat([concatdfwhole,tmpdf], ignore_index=True)

concatdfwhole.to_csv('/home/jiye/jiye/copycomparison/HRgenes_data/AFdata/whole-sample-WES-HR', index=False)

    
# %%
##^ WGS_whole
import os.path

def makedf(i):
    sample = wgsfilelist.iloc[i,0]
    tmpdf = pd.read_csv('/home/jiye/jiye/copycomparison/HRgenes_data/AFdata/whole-sample/WGSdata/AF_WGS_HR-'+sample, sep=',')
    tmpdf['sample_ID'] = sample
    return tmpdf

concatdf = makedf(0)
for i in range(1, wgsfilelist.shape[0]):
    sample = wgsfilelist.iloc[i,0]
    path = '/home/jiye/jiye/copycomparison/HRgenes_data/AFdata/whole-sample/WGSdata/AF_WGS_HR-'+sample
    if os.path.exists(path):        
        tmpdf = makedf(i)
        concatdf = pd.concat([concatdf,tmpdf], ignore_index=True)

concatdf.to_csv('/home/jiye/jiye/copycomparison/HRgenes_data/AFdata/whole-sample-WGS-HR', index=False)



############################################################################################&






############################################################################################&



# %%
#* 3. 샘플별로 각각 1번파일 만드는거 matched로 하기 (genome pre file = genome post file)

onlypre_WES_matched_samplelist = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/only-pre_WES_matched_samplelist', sep="\t", header=None)
onlypre_WGS_matched_samplelist = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/only-pre_WGS_matched_samplelist', sep="\t", header=None)
onlypre_WES_matched_samplelist.columns = ['sample_ID','pre/post','wes','rna']
onlypre_WGS_matched_samplelist.columns = ['sample_ID','pre/post','wes','rna']
matchedlist = [onlypre_WES_matched_samplelist,onlypre_WGS_matched_samplelist]


############################################################################################&
##^ WES_matched
for i in range(onlypre_WES_matched_samplelist.shape[0]):
    # making chr | position | varinatID | RNA AF
    sample= onlypre_WES_matched_samplelist.iloc[i,0]
    
    path = '/home/jiye/jiye/copycomparison/HRgenes_data/readcountdata/pre-post/WESdata/readcount-WES-HR-'+sample
    if os.path.exists(path): #only non-empty file
        readcountdata = pd.read_csv('/home/jiye/jiye/copycomparison/HRgenes_data/readcountdata/pre-post/WESdata/readcount-WES-HR-'+sample, sep='\t')
        if readcountdata.shape[0]>=1: 
            RNA_AF_df = readcountdata[['contig','position','variantID','refCount','totalCount']]
            RNA_AF_df['RNA_AF'] = RNA_AF_df.loc[:,'refCount']/RNA_AF_df.loc[:,'totalCount']
            RNA_AF_df['newpos'] = RNA_AF_df.agg('{0[contig]}:{0[position]}'.format, axis=1)

            RNA_AF = RNA_AF_df[['newpos','variantID','RNA_AF']]
            RNA_AF.columns = ['position','variantID','RNA_AF']

            # making genome AF | consequence | impact
            wesfile = onlypre_WES_matched_samplelist.iloc[i,2]
            genome_AF_df = pd.read_csv('/home/jiye/jiye/copycomparison/HRgenes_data/genomeAFdata/WESdata/AF-nodup-filtered-ms-biallelic-HRgenes-'+wesfile+'.vep.vcf',sep='\t', header=None)
            genome_AF_df.columns =['contig','position','refCount','altCount']
            genome_AF_df['genome_AF'] = genome_AF_df.loc[:,'refCount']/(genome_AF_df.loc[:,'refCount']+genome_AF_df.loc[:,'altCount'])
            genome_AF_df['newpos'] = genome_AF_df.agg('{0[contig]}:{0[position]}'.format, axis=1)
            genome_AF = genome_AF_df[['newpos','genome_AF']]
            genome_AF.columns = ['position','genome_AF']

            variantinfo = pd.read_csv('~/jiye/copycomparison/HRgenes_data/WESdata/vep/final-vep-HRgenes-' + wesfile + '.vep.vcf', sep='\t', header=None, on_bad_lines='skip')
            variantinfo.columns =['position','ENSG','ENST','consequence','impact']
            variantinfo['impact'] = variantinfo['impact'].str.replace('IMPACT=','')
            variantinfo = variantinfo[~variantinfo['position'].str.contains('-')] #only SNV
            variantinfo = variantinfo.loc[:,['position','ENSG','impact','consequence']]


            RNA_genome_merged = pd.merge(RNA_AF,genome_AF,on='position')
            final_merged = pd.merge(RNA_genome_merged,variantinfo, on='position')

            final_merged.to_csv('/home/jiye/jiye/copycomparison/HRgenes_data/AFdata/pre-post/WESdata/AF_WES_HR-'+sample, index=False)

#%%
##^ WGS_matched
for i in range(onlypre_WGS_matched_samplelist.shape[0]):
    # making chr | position | varinatID | RNA AF
    sample= onlypre_WGS_matched_samplelist.iloc[i,0]
    
    path = '/home/jiye/jiye/copycomparison/HRgenes_data/readcountdata/pre-post/WGSdata/readcount-WGS-HR-'+sample
    if os.path.exists(path): #only non-empty file
        readcountdata = pd.read_csv('/home/jiye/jiye/copycomparison/HRgenes_data/readcountdata/pre-post/WGSdata/readcount-WGS-HR-'+sample, sep='\t')
        if readcountdata.shape[0]>=1: 
            RNA_AF_df = readcountdata[['contig','position','variantID','refCount','totalCount']]
            RNA_AF_df['RNA_AF'] = RNA_AF_df.loc[:,'refCount']/RNA_AF_df.loc[:,'totalCount']
            RNA_AF_df['newpos'] = RNA_AF_df.agg('{0[contig]}:{0[position]}'.format, axis=1)

            RNA_AF = RNA_AF_df[['newpos','variantID','RNA_AF']]
            RNA_AF.columns = ['position','variantID','RNA_AF']

            # making genome AF | consequence | impact
            wgsfile = onlypre_WGS_matched_samplelist.iloc[i,2]
            genome_AF_df = pd.read_csv('/home/jiye/jiye/copycomparison/HRgenes_data/genomeAFdata/WGSdata/AF-nodup-filtered-ms-biallelic-HRgenes-'+wgsfile+'.vcf',sep='\t', header=None)
            genome_AF_df.columns =['contig','position','refCount','altCount']
            genome_AF_df['genome_AF'] = genome_AF_df.loc[:,'refCount']/(genome_AF_df.loc[:,'refCount']+genome_AF_df.loc[:,'altCount'])
            genome_AF_df['newpos'] = genome_AF_df.agg('{0[contig]}:{0[position]}'.format, axis=1)
            genome_AF = genome_AF_df[['newpos','genome_AF']]
            genome_AF.columns = ['position','genome_AF']

            variantinfo = pd.read_csv('~/jiye/copycomparison/HRgenes_data/WGSdata/vep/final-vep-HRgenes-' + wgsfile + '.vcf', sep='\t', header=None, on_bad_lines='skip')
            variantinfo.columns =['position','ENSG','ENST','consequence','impact']
            variantinfo['impact'] = variantinfo['impact'].str.replace('IMPACT=','')
            variantinfo = variantinfo[~variantinfo['position'].str.contains('-')] #only SNV
            variantinfo = variantinfo.loc[:,['position','ENSG','impact','consequence']]

            RNA_genome_merged = pd.merge(RNA_AF,genome_AF,on='position')
            final_merged = pd.merge(RNA_genome_merged,variantinfo, on='position')

            final_merged.to_csv('/home/jiye/jiye/copycomparison/HRgenes_data/AFdata/pre-post/WGSdata/AF_WGS_HR-'+sample, index=False)


#%%
#* 4. 3번 파일들 머지해서 통으로 파일 하나 만들기! 
##^ WES_matched


def makedf(i):
    sample = onlypre_WES_matched_samplelist.iloc[i,0]
    tmpdf = pd.read_csv('/home/jiye/jiye/copycomparison/HRgenes_data/AFdata/pre-post/WESdata/AF_WES_HR-'+sample, sep=',')
    tmpdf['sample_ID'] = sample
    return tmpdf

concatdf = makedf(0)
for i in range(1, onlypre_WES_matched_samplelist.shape[0]):
    sample = onlypre_WES_matched_samplelist.iloc[i,0]
    path = '/home/jiye/jiye/copycomparison/HRgenes_data/AFdata/pre-post/WESdata/AF_WES_HR-'+sample
    if os.path.exists(path):
        tmpdf = makedf(i)
        concatdf = pd.concat([concatdf,tmpdf], ignore_index=True)

concatdf.to_csv('/home/jiye/jiye/copycomparison/HRgenes_data/AFdata/pre-post-WES-HR', index=False)

#%%
##^ WGS_matched


def makedf(i):
    sample = onlypre_WGS_matched_samplelist.iloc[i,0]
    tmpdf = pd.read_csv('/home/jiye/jiye/copycomparison/HRgenes_data/AFdata/pre-post/WGSdata/AF_WGS_HR-'+sample, sep=',')
    tmpdf['sample_ID'] = sample
    return tmpdf

concatdf = makedf(0)
for i in range(1, onlypre_WGS_matched_samplelist.shape[0]):
    sample = onlypre_WGS_matched_samplelist.iloc[i,0]
    path = '/home/jiye/jiye/copycomparison/HRgenes_data/AFdata/pre-post/WGSdata/AF_WGS_HR-'+sample
    if os.path.exists(path):
        tmpdf = makedf(i)
        concatdf = pd.concat([concatdf,tmpdf], ignore_index=True)

concatdf.to_csv('/home/jiye/jiye/copycomparison/HRgenes_data/AFdata/pre-post-WGS-HR', index=False)

# %%

