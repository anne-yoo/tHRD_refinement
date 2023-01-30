
#? columns: loc / genome_AF / impact / patient_ID / response / duration
#? 

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu


# %%
##########^ make new genome AF file  #########
#* 1. 샘플 별로 각각 만들기: chr | position | variantID | genome variant AF(ref) | RNA AF(ref) | consequence | impact
#* 2. clinical data: sample ID | response | BRCAstatus | therapy interval
#* 3. 1번 파일들 머지해서 통으로 파일 하나 만들기: sample ID | chr | position | variantID | genome variant AF(ref) | RNA AF(ref) | consequence | impact 
#* 4. matched sample pre-post file directory : genome-all pre RNA-pre/post
#* 5. 샘플별로 각각 1번파일 만드는거 4번 matched로 하기 (genome pre file =genome post file)
#* 6. 5번 파일들 머지해서 통으로 파일 하나 만들기! 
#* AF = refCount/totalCount로 통일 

#%%
wesfilelist = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/WES_RNA_filelist_final',sep='\t',header=None)
wgsfilelist = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/WGS_RNA_filelist_final',sep='\t', header=None)

wesfilelist.columns = ['sample','genomefile','RNAfile']
wgsfilelist.columns = ['sample','genomefile','RNAfile']


# %%
#* 1. 샘플 별로 각각 만들기: chr | position | variantID | genome variant AF(ref) | RNA AF(ref) | consequence | impact
##^ WES_BRCA1
for i in range(wesfilelist.shape[0]):
    # making chr | position | varinatID | RNA AF
    sample= wesfilelist.iloc[i,0]
    readcountdata = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/readcountdata/WES_BRCA1/readcount-WES-BRCA1-'+sample, sep='\t')
    if readcountdata.shape[0]>=1: #only non-empty file
        RNA_AF_df = readcountdata[['contig','position','variantID','refCount','totalCount']]
        RNA_AF_df['RNA_AF'] = RNA_AF_df.loc[:,'refCount']/RNA_AF_df.loc[:,'totalCount']
        RNA_AF = RNA_AF_df[['contig','position','variantID','RNA_AF']]

        # making genome AF | consequence | impact
        wesfile = wesfilelist.iloc[i,1]
        genome_AF_df = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/forgenomeAF/WES_BRCA1/AF-filtered-ms-biallelic-BRCA1-'+wesfile+'.vep.vcf',sep='\t', header=None)
        genome_AF_df.columns =['contig','position','refCount','altCount']
        genome_AF_df['genome_AF'] = genome_AF_df.loc[:,'refCount']/(genome_AF_df.loc[:,'refCount']+genome_AF_df.loc[:,'altCount'])
        genome_AF = genome_AF_df[['position','genome_AF']]

        variantinfo = pd.read_csv('~/jiye/copycomparison/newdata/vep_WES_BRCA1/BRCA1-' + wesfilelist.iloc[i,1] + '.vep.vcf-vep-filtered0-canonical', sep='\t', header=None, on_bad_lines='skip')
        variantinfo.columns =['loc','ENSG','ENST','consequence','impact']
        variantinfo['loc'] = variantinfo['loc'].str.replace('chr17:','') # BRCA1
        variantinfo['impact'] = variantinfo['impact'].str.replace('IMPACT=','')
        variantinfo = variantinfo[~variantinfo['loc'].str.contains('-')] #only SNV
        variantinfo['position'] = pd.to_numeric(variantinfo['loc'])
        variantinfo = variantinfo.loc[:,['position','impact','consequence']]

        RNA_genome_merged = pd.merge(RNA_AF,genome_AF,on='position')
        final_merged = pd.merge(RNA_genome_merged,variantinfo, on='position')

        final_merged.to_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/WES_BRCA1/AF_WES_BRCA1-'+sample, index=False)


#%%
##^ WES_BRCA2
for i in range(wesfilelist.shape[0]):
    # making chr | position | varinatID | RNA AF
    sample= wesfilelist.iloc[i,0]
    readcountdata = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/readcountdata/WES_BRCA2/readcount-WES-BRCA2-'+sample, sep='\t')
    if readcountdata.shape[0]>=1: #only non-empty file
        RNA_AF_df = readcountdata[['contig','position','variantID','refCount','totalCount']]
        RNA_AF_df['RNA_AF'] = RNA_AF_df.loc[:,'refCount']/RNA_AF_df.loc[:,'totalCount']
        RNA_AF = RNA_AF_df[['contig','position','variantID','RNA_AF']]

        # making genome AF | consequence | impact
        wesfile = wesfilelist.iloc[i,1]
        genome_AF_df = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/forgenomeAF/WES_BRCA2/AF-filtered-ms-biallelic-BRCA2-'+wesfile+'.vep.vcf',sep='\t', header=None)
        genome_AF_df.columns =['contig','position','refCount','altCount']
        genome_AF_df['genome_AF'] = genome_AF_df.loc[:,'refCount']/(genome_AF_df.loc[:,'refCount']+genome_AF_df.loc[:,'altCount'])
        genome_AF = genome_AF_df[['position','genome_AF']]

        variantinfo = pd.read_csv('~/jiye/copycomparison/newdata/vep_WES_BRCA2/BRCA2-' + wesfilelist.iloc[i,1] + '.vep.vcf-vep-filtered0-canonical', sep='\t', header=None, on_bad_lines='skip')
        variantinfo.columns =['loc','ENSG','ENST','consequence','impact']
        variantinfo['loc'] = variantinfo['loc'].str.replace('chr13:','') # BRCA1
        variantinfo['impact'] = variantinfo['impact'].str.replace('IMPACT=','')
        variantinfo = variantinfo[~variantinfo['loc'].str.contains('-')] #only SNV
        variantinfo['position'] = pd.to_numeric(variantinfo['loc'])
        variantinfo = variantinfo.loc[:,['position','impact','consequence']]

        RNA_genome_merged = pd.merge(RNA_AF,genome_AF,on='position')
        final_merged = pd.merge(RNA_genome_merged,variantinfo, on='position')

        final_merged.to_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/WES_BRCA2/AF_WES_BRCA2-'+sample, index=False)




# %%

##^ WGS_BRCA1
for i in range(wgsfilelist.shape[0]):
    # making chr | position | varinatID | RNA AF
    sample= wgsfilelist.iloc[i,0]
    readcountdata = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/readcountdata/WGS_BRCA1/readcount-WGS-BRCA1-'+sample, sep='\t')
    if readcountdata.shape[0]>=1: #only non-empty file
        RNA_AF_df = readcountdata[['contig','position','variantID','refCount','totalCount']]
        RNA_AF_df['RNA_AF'] = RNA_AF_df.loc[:,'refCount']/RNA_AF_df.loc[:,'totalCount']
        RNA_AF = RNA_AF_df[['contig','position','variantID','RNA_AF']]

        # making genome AF | consequence | impact
        wgsfile = wgsfilelist.iloc[i,1]
        genome_AF_df = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/forgenomeAF/WGS_BRCA1/AF-filtered-ms-biallelic-BRCA1-'+wgsfile+'.vcf',sep='\t', header=None)
        genome_AF_df.columns =['contig','position','refCount','altCount']
        genome_AF_df['genome_AF'] = genome_AF_df.loc[:,'refCount']/(genome_AF_df.loc[:,'refCount']+genome_AF_df.loc[:,'altCount'])
        genome_AF = genome_AF_df[['position','genome_AF']]

        variantinfo = pd.read_csv('~/jiye/copycomparison/newdata/vep_WGS_BRCA1/BRCA1-' + wgsfilelist.iloc[i,1] + '.vcf-vep-filtered0-canonical', sep='\t', header=None, on_bad_lines='skip')
        variantinfo.columns =['loc','ENSG','ENST','consequence','impact']
        variantinfo['loc'] = variantinfo['loc'].str.replace('chr17:','') # BRCA1
        variantinfo['impact'] = variantinfo['impact'].str.replace('IMPACT=','')
        variantinfo = variantinfo[~variantinfo['loc'].str.contains('-')] #only SNV
        variantinfo['position'] = pd.to_numeric(variantinfo['loc'])
        variantinfo = variantinfo.loc[:,['position','impact','consequence']]

        RNA_genome_merged = pd.merge(RNA_AF,genome_AF, on='position')
        final_merged = pd.merge(RNA_genome_merged,variantinfo, on='position')

        final_merged.to_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/WGS_BRCA1/AF_WGS_BRCA1-'+sample, index=False)



# %%

##^ WGS_BRCA2
for i in range(wgsfilelist.shape[0]):
    # making chr | position | varinatID | RNA AF
    sample= wgsfilelist.iloc[i,0]
    readcountdata = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/readcountdata/WGS_BRCA2/readcount-WGS-BRCA2-'+sample, sep='\t')
    if readcountdata.shape[0]>=1: #only non-empty file
        RNA_AF_df = readcountdata[['contig','position','variantID','refCount','totalCount']]
        RNA_AF_df['RNA_AF'] = RNA_AF_df.loc[:,'refCount']/RNA_AF_df.loc[:,'totalCount']
        RNA_AF = RNA_AF_df[['contig','position','variantID','RNA_AF']]

        # making genome AF | consequence | impact
        wgsfile = wgsfilelist.iloc[i,1]
        genome_AF_df = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/forgenomeAF/WGS_BRCA2/AF-filtered-ms-biallelic-BRCA2-'+wgsfile+'.vcf',sep='\t', header=None)
        genome_AF_df.columns =['contig','position','refCount','altCount']
        genome_AF_df['genome_AF'] = genome_AF_df.loc[:,'refCount']/(genome_AF_df.loc[:,'refCount']+genome_AF_df.loc[:,'altCount'])
        genome_AF = genome_AF_df[['position','genome_AF']]

        variantinfo = pd.read_csv('~/jiye/copycomparison/newdata/vep_WGS_BRCA2/BRCA2-' + wgsfilelist.iloc[i,1] + '.vcf-vep-filtered0-canonical', sep='\t', header=None, on_bad_lines='skip')
        variantinfo.columns =['loc','ENSG','ENST','consequence','impact']
        variantinfo['loc'] = variantinfo['loc'].str.replace('chr13:','') # BRCA2
        variantinfo['impact'] = variantinfo['impact'].str.replace('IMPACT=','')
        variantinfo = variantinfo[~variantinfo['loc'].str.contains('-')] #only SNV
        variantinfo['position'] = pd.to_numeric(variantinfo['loc'])
        variantinfo = variantinfo.loc[:,['position','impact','consequence']]

        RNA_genome_merged = pd.merge(RNA_AF,genome_AF, on='position')
        final_merged = pd.merge(RNA_genome_merged,variantinfo, on='position')

        final_merged.to_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/WGS_BRCA2/AF_WGS_BRCA2-'+sample, index=False)





############################################################################################&




# %%
#* 2. clinical data: sample ID | response | BRCAstatus | therapy interval | 
##^ clinical data for all samples
clinical = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/control_PD/clinicaldata_final.csv')
clinical = clinical.drop_duplicates(['sample_ID'], keep = 'first', ignore_index = True)
samplelist = clinical[['sample_ID']]
maintenance = clinical[clinical['Purpose']=='Maintenance']
salvagetmp = clinical[clinical['Purpose']=='Salvage']
salvage = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/control_PD/Salvage_new_ORR.csv')
salvage.columns = ['sample_ID','purpose','ORR']

maintenance['response'] = 'non-responder'
for i in range(maintenance.shape[0]):
    if maintenance.iloc[i,2] == 'Mt': #BRCA mutation
        if maintenance.iloc[i,3] == '1L':
            if maintenance.iloc[i,1] >= 540:
                maintenance.iloc[i,7] = 'responder'
        else:
            if maintenance.iloc[i,1] >=360:
                maintenance.iloc[i,7] = 'responder'
    else: #no BRCA mutation
        if maintenance.iloc[i,3] == '1L':
            if maintenance.iloc[i,1] >= 360:
                maintenance.iloc[i,7] = 'responder'
        else:
            if maintenance.iloc[i,1] >=180:
                maintenance.iloc[i,7] = 'responder'

salvage['response'] = 'non-responder'

for i in range(salvage.shape[0]):
    if salvage.iloc[i,2] == ('PR' or 'CR'):
        salvage.iloc[i,3] = 'responder'
    elif salvage.iloc[i,2] == 'NE':
        salvage.iloc[i,3] = 'unclear'
    
finalsalvage = pd.merge(salvage, salvagetmp, on='sample_ID')
new_clinical = pd.concat([maintenance,finalsalvage])

finalclinical = pd.merge(samplelist,new_clinical,on='sample_ID')
finalclinical = finalclinical.drop(['purpose','ORR'],axis=1)

finalclinical.to_csv('/home/jiye/jiye/copycomparison/control_PD/final_clinicaldata', index=False)





############################################################################################&





# %%
#* 3. 1번 파일들 머지해서 통으로 파일 하나 만들기: sample ID | chr | position | variantID | genome variant AF(ref) | RNA AF(ref) | consequence | impact 
##^ WES_BRCA1

def makedf(i):
    sample = wesfilelist.iloc[i,0]
    tmpdf = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/WES_BRCA1/AF_WES_BRCA1-'+sample, sep=',')
    tmpdf['sample_ID'] = sample
    return tmpdf

concatdf = makedf(0)
for i in range(1, wesfilelist.shape[0]):
    sample = wesfilelist.iloc[i,0]
    readcountdata = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/readcountdata/WES_BRCA1/readcount-WES-BRCA1-'+sample, sep='\t')
    if readcountdata.shape[0]>=1:
        tmpdf = makedf(i)
        concatdf = pd.concat([concatdf,tmpdf], ignore_index=True)

concatdf.to_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/sample-merged/WES-BRCA1-whole-sample', index=False)

    
# %%
##^ WES_BRCA2

def makedf(i):
    sample = wesfilelist.iloc[i,0]
    tmpdf = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/WES_BRCA2/AF_WES_BRCA2-'+sample, sep=',')
    tmpdf['sample_ID'] = sample
    return tmpdf

concatdf = makedf(0)
for i in range(1, wesfilelist.shape[0]):
    sample = wesfilelist.iloc[i,0]
    readcountdata = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/readcountdata/WES_BRCA2/readcount-WES-BRCA2-'+sample, sep='\t')
    if readcountdata.shape[0]>=1:
        tmpdf = makedf(i)
        concatdf = pd.concat([concatdf,tmpdf], ignore_index=True)

concatdf.to_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/sample-merged/WES-BRCA2-whole-sample', index=False)

# %%
##^ WGS_BRCA1

def makedf(i):
    sample = wgsfilelist.iloc[i,0]
    tmpdf = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/WGS_BRCA1/AF_WGS_BRCA1-'+sample, sep=',')
    tmpdf['sample_ID'] = sample
    return tmpdf

concatdf = makedf(0)
for i in range(1, wgsfilelist.shape[0]):
    sample = wgsfilelist.iloc[i,0]
    readcountdata = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/readcountdata/WGS_BRCA1/readcount-WGS-BRCA1-'+sample, sep='\t')
    if readcountdata.shape[0]>=1:
        tmpdf = makedf(i)
        concatdf = pd.concat([concatdf,tmpdf], ignore_index=True)

concatdf.to_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/sample-merged/WGS-BRCA1-whole-sample', index=False)

# %%
##^ WGS_BRCA2

def makedf(i):
    sample = wgsfilelist.iloc[i,0]
    tmpdf = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/WGS_BRCA2/AF_WGS_BRCA2-'+sample, sep=',')
    tmpdf['sample_ID'] = sample
    return tmpdf

concatdf = makedf(1)
for i in range(2, wgsfilelist.shape[0]):
    sample = wgsfilelist.iloc[i,0]
    readcountdata = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/readcountdata/WGS_BRCA2/readcount-WGS-BRCA2-'+sample, sep='\t')
    if readcountdata.shape[0]>=1:
        tmpdf = makedf(i)
        concatdf = pd.concat([concatdf,tmpdf], ignore_index=True)

concatdf.to_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/sample-merged/WGS-BRCA2-whole-sample', index=False)




############################################################################################&




# %%
#* 4. matched sample pre-post file directory : genome-all pre RNA-pre/post

WES_matched_samplelist = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/WES_matched_samplelist.csv', header=None)
WGS_matched_samplelist = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/WGS_matched_samplelist.csv', header=None)
WES_matched_samplelist.columns = ['sample','pre/post']
WGS_matched_samplelist.columns = ['sample','pre/post']

matched_wesfilelist = pd.merge(WES_matched_samplelist, wesfilelist, on='sample')
matched_wgsfilelist = pd.merge(WGS_matched_samplelist, wgsfilelist, on='sample')

for i in range(10):
    matched_wesfilelist.iloc[10+i,2] = matched_wesfilelist.iloc[i,2]

for i in range(14):
    matched_wgsfilelist.iloc[14+i,2] = matched_wgsfilelist.iloc[i,2]

matched_wesfilelist.to_csv('/home/jiye/jiye/copycomparison/control_PD/only-pre_WES_matched_samplelist.csv', index=False)
matched_wgsfilelist.to_csv('/home/jiye/jiye/copycomparison/control_PD/only-pre_WGS_matched_samplelist.csv', index=False)





############################################################################################&



# %%
#* 5. 샘플별로 각각 1번파일 만드는거 4번 matched로 하기 (genome pre file = genome post file)

onlypre_WES_matched_samplelist = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/only-pre_WES_matched_samplelist', sep="\t", header=None)
onlypre_WGS_matched_samplelist = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/only-pre_WGS_matched_samplelist', sep="\t", header=None)
onlypre_WES_matched_samplelist.columns = ['sample_ID','pre/post','wes','rna']
onlypre_WGS_matched_samplelist.columns = ['sample_ID','pre/post','wes','rna']
matchedlist = [onlypre_WES_matched_samplelist,onlypre_WGS_matched_samplelist]

##^ WES_BRCA1
for i in range(onlypre_WES_matched_samplelist.shape[0]):
    # making chr | position | variantID | RNA AF
    sample= onlypre_WES_matched_samplelist.iloc[i,0]
    readcountdata = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/readcountdata/pre-post/WES_BRCA1/readcount-WES-BRCA1-'+sample, sep='\t')
    if readcountdata.shape[0]>=1: #only non-empty file
        RNA_AF_df = readcountdata[['contig','position','variantID','refCount','totalCount']]
        RNA_AF_df['RNA_AF'] = RNA_AF_df.loc[:,'refCount']/RNA_AF_df.loc[:,'totalCount']
        RNA_AF = RNA_AF_df[['contig','position','variantID','RNA_AF']]

        # making genome AF | consequence | impact
        wesfile = onlypre_WES_matched_samplelist.iloc[i,2]
        genome_AF_df = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/forgenomeAF/WES_BRCA1/AF-filtered-ms-biallelic-BRCA1-'+wesfile+'.vep.vcf',sep='\t', header=None)
        genome_AF_df.columns =['contig','position','refCount','altCount']
        genome_AF_df['genome_AF'] = genome_AF_df.loc[:,'refCount']/(genome_AF_df.loc[:,'refCount']+genome_AF_df.loc[:,'altCount'])
        genome_AF = genome_AF_df[['position','genome_AF']]

        variantinfo = pd.read_csv('~/jiye/copycomparison/newdata/vep_WES_BRCA1/BRCA1-' + wesfile + '.vep.vcf-vep-filtered0-canonical', sep='\t', header=None, on_bad_lines='skip')
        variantinfo.columns =['loc','ENSG','ENST','consequence','impact']
        variantinfo['loc'] = variantinfo['loc'].str.replace('chr17:','') # BRCA1
        variantinfo['impact'] = variantinfo['impact'].str.replace('IMPACT=','')
        variantinfo = variantinfo[~variantinfo['loc'].str.contains('-')] #only SNV
        variantinfo['position'] = pd.to_numeric(variantinfo['loc'])
        variantinfo = variantinfo.loc[:,['position','impact','consequence']]

        RNA_genome_merged = pd.merge(RNA_AF,genome_AF,on='position')
        final_merged = pd.merge(RNA_genome_merged,variantinfo, on='position')

        final_merged.to_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/pre-post/WES_BRCA1/AF_WES_BRCA1-'+sample, index=False)

##^ WES_BRCA2
for i in range(onlypre_WES_matched_samplelist.shape[0]):
    # making chr | position | variantID | RNA AF
    sample= onlypre_WES_matched_samplelist.iloc[i,0]
    readcountdata = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/readcountdata/pre-post/WES_BRCA2/readcount-WES-BRCA2-'+sample, sep='\t')
    if readcountdata.shape[0]>=1: #only non-empty file
        RNA_AF_df = readcountdata[['contig','position','variantID','refCount','totalCount']]
        RNA_AF_df['RNA_AF'] = RNA_AF_df.loc[:,'refCount']/RNA_AF_df.loc[:,'totalCount']
        RNA_AF = RNA_AF_df[['contig','position','variantID','RNA_AF']]

        # making genome AF | consequence | impact
        wesfile = onlypre_WES_matched_samplelist.iloc[i,2]
        genome_AF_df = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/forgenomeAF/WES_BRCA2/AF-filtered-ms-biallelic-BRCA2-'+wesfile+'.vep.vcf',sep='\t', header=None)
        genome_AF_df.columns =['contig','position','refCount','altCount']
        genome_AF_df['genome_AF'] = genome_AF_df.loc[:,'refCount']/(genome_AF_df.loc[:,'refCount']+genome_AF_df.loc[:,'altCount'])
        genome_AF = genome_AF_df[['position','genome_AF']]

        variantinfo = pd.read_csv('~/jiye/copycomparison/newdata/vep_WES_BRCA2/BRCA2-' + wesfile + '.vep.vcf-vep-filtered0-canonical', sep='\t', header=None, on_bad_lines='skip')
        variantinfo.columns =['loc','ENSG','ENST','consequence','impact']
        variantinfo['loc'] = variantinfo['loc'].str.replace('chr13:','') # BRCA1
        variantinfo['impact'] = variantinfo['impact'].str.replace('IMPACT=','')
        variantinfo = variantinfo[~variantinfo['loc'].str.contains('-')] #only SNV
        variantinfo['position'] = pd.to_numeric(variantinfo['loc'])
        variantinfo = variantinfo.loc[:,['position','impact','consequence']]

        RNA_genome_merged = pd.merge(RNA_AF,genome_AF,on='position')
        final_merged = pd.merge(RNA_genome_merged,variantinfo, on='position')

        final_merged.to_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/pre-post/WES_BRCA2/AF_WES_BRCA2-'+sample, index=False)

##^ WGS_BRCA1
for i in range(onlypre_WGS_matched_samplelist.shape[0]):
    # making chr | position | variantID | RNA AF
    sample= onlypre_WGS_matched_samplelist.iloc[i,0]
    readcountdata = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/readcountdata/pre-post/WGS_BRCA1/readcount-WGS-BRCA1-'+sample, sep='\t')
    if readcountdata.shape[0]>=1: #only non-empty file
        RNA_AF_df = readcountdata[['contig','position','variantID','refCount','totalCount']]
        RNA_AF_df['RNA_AF'] = RNA_AF_df.loc[:,'refCount']/RNA_AF_df.loc[:,'totalCount']
        RNA_AF = RNA_AF_df[['contig','position','variantID','RNA_AF']]

        # making genome AF | consequence | impact
        wgsfile = onlypre_WGS_matched_samplelist.iloc[i,2]
        genome_AF_df = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/forgenomeAF/WGS_BRCA1/AF-filtered-ms-biallelic-BRCA1-'+wgsfile+'.vcf',sep='\t', header=None)
        genome_AF_df.columns =['contig','position','refCount','altCount']
        genome_AF_df['genome_AF'] = genome_AF_df.loc[:,'refCount']/(genome_AF_df.loc[:,'refCount']+genome_AF_df.loc[:,'altCount'])
        genome_AF = genome_AF_df[['position','genome_AF']]

        variantinfo = pd.read_csv('~/jiye/copycomparison/newdata/vep_WGS_BRCA1/BRCA1-' + wgsfile + '.vcf-vep-filtered0-canonical', sep='\t', header=None, on_bad_lines='skip')
        variantinfo.columns =['loc','ENSG','ENST','consequence','impact']
        variantinfo['loc'] = variantinfo['loc'].str.replace('chr17:','') # BRCA1
        variantinfo['impact'] = variantinfo['impact'].str.replace('IMPACT=','')
        variantinfo = variantinfo[~variantinfo['loc'].str.contains('-')] #only SNV
        variantinfo['position'] = pd.to_numeric(variantinfo['loc'])
        variantinfo = variantinfo.loc[:,['position','impact','consequence']]

        RNA_genome_merged = pd.merge(RNA_AF,genome_AF,on='position')
        final_merged = pd.merge(RNA_genome_merged,variantinfo, on='position')

        final_merged.to_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/pre-post/WGS_BRCA1/AF_WGS_BRCA1-'+sample, index=False)

##^ WGS_BRCA2
for i in range(onlypre_WGS_matched_samplelist.shape[0]):
    # making chr | position | variantID | RNA AF
    sample= onlypre_WGS_matched_samplelist.iloc[i,0]
    readcountdata = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/readcountdata/pre-post/WGS_BRCA2/readcount-WGS-BRCA2-'+sample, sep='\t')
    if readcountdata.shape[0]>=1: #only non-empty file
        RNA_AF_df = readcountdata[['contig','position','variantID','refCount','totalCount']]
        RNA_AF_df['RNA_AF'] = RNA_AF_df.loc[:,'refCount']/RNA_AF_df.loc[:,'totalCount']
        RNA_AF = RNA_AF_df[['contig','position','variantID','RNA_AF']]

        # making genome AF | consequence | impact
        wgsfile = onlypre_WGS_matched_samplelist.iloc[i,2]
        genome_AF_df = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/forgenomeAF/WGS_BRCA2/AF-filtered-ms-biallelic-BRCA2-'+wgsfile+'.vcf',sep='\t', header=None)
        genome_AF_df.columns =['contig','position','refCount','altCount']
        genome_AF_df['genome_AF'] = genome_AF_df.loc[:,'refCount']/(genome_AF_df.loc[:,'refCount']+genome_AF_df.loc[:,'altCount'])
        genome_AF = genome_AF_df[['position','genome_AF']]

        variantinfo = pd.read_csv('~/jiye/copycomparison/newdata/vep_WGS_BRCA2/BRCA2-' + wgsfile + '.vcf-vep-filtered0-canonical', sep='\t', header=None, on_bad_lines='skip')
        variantinfo.columns =['loc','ENSG','ENST','consequence','impact']
        variantinfo['loc'] = variantinfo['loc'].str.replace('chr13:','') # BRCA2
        variantinfo['impact'] = variantinfo['impact'].str.replace('IMPACT=','')
        variantinfo = variantinfo[~variantinfo['loc'].str.contains('-')] #only SNV
        variantinfo['position'] = pd.to_numeric(variantinfo['loc'])
        variantinfo = variantinfo.loc[:,['position','impact','consequence']]

        RNA_genome_merged = pd.merge(RNA_AF,genome_AF,on='position')
        final_merged = pd.merge(RNA_genome_merged,variantinfo, on='position')

        final_merged.to_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/pre-post/WGS_BRCA2/AF_WGS_BRCA2-'+sample, index=False)



############################################################################################&



#%%
#* 6. 5번 파일들 머지해서 통으로 파일 하나 만들기! 
##^ WES_BRCA1

def makedf(i):
    sample = onlypre_WES_matched_samplelist.iloc[i,0]
    tmpdf = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/pre-post/WES_BRCA1/AF_WES_BRCA1-'+sample, sep=',')
    tmpdf['sample_ID'] = sample
    return tmpdf

concatdf = makedf(0)
for i in range(1, onlypre_WES_matched_samplelist.shape[0]):
    sample = onlypre_WES_matched_samplelist.iloc[i,0]
    readcountdata = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/readcountdata/pre-post/WES_BRCA1/readcount-WES-BRCA1-'+sample, sep='\t')
    if readcountdata.shape[0]>=1:
        tmpdf = makedf(i)
        concatdf = pd.concat([concatdf,tmpdf], ignore_index=True)

concatdf.to_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/pre-post/sample-merged/WES-BRCA1-whole-sample', index=False)


#%%
##^ WES_BRCA2

def makedf(i):
    sample = onlypre_WES_matched_samplelist.iloc[i,0]
    tmpdf = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/pre-post/WES_BRCA2/AF_WES_BRCA2-'+sample, sep=',')
    tmpdf['sample_ID'] = sample
    return tmpdf

concatdf = makedf(0)
for i in range(1, onlypre_WES_matched_samplelist.shape[0]):
    sample = onlypre_WES_matched_samplelist.iloc[i,0]
    readcountdata = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/readcountdata/pre-post/WES_BRCA2/readcount-WES-BRCA2-'+sample, sep='\t')
    if readcountdata.shape[0]>=1:
        tmpdf = makedf(i)
        concatdf = pd.concat([concatdf,tmpdf], ignore_index=True)

concatdf.to_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/pre-post/sample-merged/WES-BRCA2-whole-sample', index=False)

#%%
##^ WGS_BRCA1

def makedf(i):
    sample = onlypre_WGS_matched_samplelist.iloc[i,0]
    tmpdf = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/pre-post/WGS_BRCA1/AF_WGS_BRCA1-'+sample, sep=',')
    tmpdf['sample_ID'] = sample
    return tmpdf

concatdf = makedf(0)
for i in range(1, onlypre_WGS_matched_samplelist.shape[0]):
    sample = onlypre_WGS_matched_samplelist.iloc[i,0]
    readcountdata = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/readcountdata/pre-post/WGS_BRCA1/readcount-WGS-BRCA1-'+sample, sep='\t')
    if readcountdata.shape[0]>=1:
        tmpdf = makedf(i)
        concatdf = pd.concat([concatdf,tmpdf], ignore_index=True)

concatdf.to_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/pre-post/sample-merged/WGS-BRCA1-whole-sample', index=False)
#%%
##^ WGS_BRCA2

def makedf(i):
    sample = onlypre_WGS_matched_samplelist.iloc[i,0]
    tmpdf = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/pre-post/WGS_BRCA2/AF_WGS_BRCA2-'+sample, sep=',')
    tmpdf['sample_ID'] = sample
    return tmpdf

concatdf = makedf(0)
for i in range(1, onlypre_WGS_matched_samplelist.shape[0]):
    sample = onlypre_WGS_matched_samplelist.iloc[i,0]
    readcountdata = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/readcountdata/pre-post/WGS_BRCA2/readcount-WGS-BRCA2-'+sample, sep='\t')
    if readcountdata.shape[0]>=1:
        tmpdf = makedf(i)
        concatdf = pd.concat([concatdf,tmpdf], ignore_index=True)

concatdf.to_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/pre-post/sample-merged/WGS-BRCA2-whole-sample', index=False)
# %%
