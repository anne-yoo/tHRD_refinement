
#? columns: loc / genome_AF / impact / patient_ID / response / duration
#? 

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu


# %%

######* 
def response_separation():
        ref = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/newdata/sample_reference2.csv')
        cldata = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/newdata/RNA_clinical.csv')
        salvage = pd.read_csv('/home/omics/DATA6/jiye/copycomparison/newdata/Salvage_ORR.csv')
        salvage.columns = ['sample','purpose2','ORR']
        cldata = cldata.drop(['purpose'],axis=1)
        merged = pd.merge(cldata,ref,on='sample')
        merged2 = pd.merge(merged,salvage,on='sample',how='left')

        con1 = (merged2['purpose2']=='maintenance')
        merged2.loc[con1,'purpose'] = 'Maintenance'
        merged2 = merged2.drop(['purpose2'], axis=1)        
        maintenance = merged2[merged2['purpose']=='Maintenance']
        salvage = merged2[merged2['purpose']=='salvage']

        maintenance['response'] = 'non-responder'
        for i in range(maintenance.shape[0]):
            if maintenance.iloc[i,1] == 'Mt': #BRCA mutation
                if maintenance.iloc[i,5] == '1L':
                    if maintenance.iloc[i,3] >= 540:
                        maintenance.iloc[i,4] = 'responder'
                else:
                    if maintenance.iloc[i,3] >=360:
                        maintenance.iloc[i,4] = 'responder'
            else: #no BRCA mutation
                if maintenance.iloc[i,5] == '1L':
                    if maintenance.iloc[i,3] >= 360:
                        maintenance.iloc[i,4] = 'responder'
                else:
                    if maintenance.iloc[i,3] >=180:
                        maintenance.iloc[i,4] = 'responder'

        salvage = salvage[salvage['ORR']!='NE']
        salvage['response'] = 'non-responder'
        salvage = salvage.dropna()

        for i in range(salvage.shape[0]):
            if salvage.iloc[i,7] == ('PR' or 'CR'):
                salvage.iloc[i,4] = 'responder'

        final = pd.concat([maintenance,salvage])

        return final


wes_filelist = pd.read_csv('~/jiye/copycomparison/newdata/WES_finalfilelist.csv', sep=',')
wes_filelist.columns = ['sample','WESfile','RNAfile']

merged = response_separation() 

cl_data = merged.iloc[:,[0,3,4]]
sampletmp = pd.merge(wes_filelist,cl_data,on='sample')
sampletmp.columns = ['sample_ID','WESfile','RNAfile','duration','response']
sampleinfo = sampletmp.loc[:,['sample_ID','response','duration']] ##* WES~RNA sample clinical info
filelist = sampletmp.loc[:,['sample_ID','WESfile','RNAfile']] ##* WES~RNA matched file name


# %%
######^ DEPTH 5 ######
##* for RNA matching, first filter out unempty files
unemptyfilelist = []
for i in filelist.index:
    try:
        pd.read_csv('~/jiye/copycomparison/newdata/vep_WES_BRCA2/BRCA2-' + filelist.iloc[i,1] + '-vep-filtered0-canonical', sep='\t', header=None, on_bad_lines='skip')
        unemptyfilelist.append(i)
    except pd.errors.EmptyDataError:
        pass

filelist = filelist.iloc[unemptyfilelist]
filelist = filelist.reset_index(drop=True)

##* make concat df 
def makedf(i):
    wes_AF = pd.read_csv('~/jiye/copycomparison/control_PD/WES_BRCA2/final-BRCA2-' + filelist.iloc[i,1], sep='\t', header=None, on_bad_lines='skip')
    wes_AF = wes_AF.rename(columns={1:'loc', 4:'genotype'})
    wes_AF['genome_AF'] = 1 - wes_AF.iloc[:,5]/wes_AF.iloc[:,5:].sum(axis=1) ##* Alternative Allele Frequency calculation 
    wes_AF = wes_AF[wes_AF['genotype'] != '1/1']
    wes_AF['sample_ID'] = filelist.iloc[i,0]
    wes_AF['response'] = sampleinfo.iloc[i,1]
    wes_AF['duration'] = sampleinfo.iloc[i,2]
    wes_AF = wes_AF.dropna() ##* allele frequency = x/0 
    wes_AF = wes_AF.loc[:,['loc','genome_AF','sample_ID','response','duration']]

    variantinfo = pd.read_csv('~/jiye/copycomparison/newdata/vep_WES_BRCA2/BRCA2-' + filelist.iloc[i,1] + '-vep-filtered0-canonical', sep='\t', header=None, on_bad_lines='skip')
    variantinfo.columns =['loc','ENSG','ENST','consequence','impact']
    variantinfo['loc'] = variantinfo['loc'].str.replace('chr13:','') ##* BRCA1
    variantinfo['impact'] = variantinfo['impact'].str.replace('IMPACT=','')
    variantinfo = variantinfo[~variantinfo['loc'].str.contains('-')] #only SNV
    variantinfo['loc'] = pd.to_numeric(variantinfo['loc'])
    variantinfo = variantinfo.loc[:,['loc','impact','consequence']]

    final_wes = pd.merge(wes_AF, variantinfo, on='loc')

    rna_depth10_file = filelist.iloc[i,2]
    rna_depth10_file = rna_depth10_file.replace('.vcf','')
    rna_depth10 = pd.read_csv('~/jiye/copycomparison/newdata/depth_RNA_BRCA2/brca2-' + rna_depth10_file + '-depth', sep='\t', header=None, on_bad_lines='skip')

    rna_depth10.columns = ['chr','loc','depth']
    rna_depth10 = rna_depth10[rna_depth10['depth']>=5]

    wes_rna = pd.merge(rna_depth10, final_wes, on='loc')
    wes_rna = wes_rna.drop(['chr','depth'], axis=1)

    rna = pd.read_csv('~/jiye/copycomparison/newdata/RNA_BRCA2/final-brca2-' + filelist.iloc[i,2], sep='\t', header=None, on_bad_lines='skip')
    rna = rna.rename(columns={1:'loc', 4:'genotype'})
    rna['RNA_AF'] = 1 - rna.iloc[:,5]/rna.iloc[:, 5:].sum(axis=1)
    rna = rna.loc[:,['loc','RNA_AF']]
    rna = rna.dropna()
    rna_matched = pd.merge(wes_rna, rna, on='loc', how='left')
        
    return rna_matched

concatdf = makedf(0)

for i in range(1, filelist.shape[0]):
    tmpdf = makedf(i)
    concatdf = pd.concat([concatdf,tmpdf], ignore_index=True)

finaloutput = concatdf

finaloutput['mut'] = 'mt'
con=finaloutput['RNA_AF'].isna()
finaloutput.loc[con,'mut'] = 'wt'
finaloutput = finaloutput.fillna(0)

#%%
# finaloutput.to_csv('/home/jiye/jiye/copycomparison/variantinfo/withsalvage/depth5/WES_RNA_BRCA2.csv', index=False)

#%%
finaloutput.to_csv('/home/jiye/jiye/copycomparison/variantinfo/modifiedAF/WES_RNA_BRCA2.csv', index=False)



















# %%
######^ DEPTH 10 ######
##* for RNA matching, first filter out unempty files
unemptyfilelist = []
for i in filelist.index:
    try:
        pd.read_csv('~/jiye/copycomparison/newdata/vep_WES_BRCA2/BRCA2-' + filelist.iloc[i,1] + '-vep-filtered0-canonical', sep='\t', header=None, on_bad_lines='skip')
        unemptyfilelist.append(i)
    except pd.errors.EmptyDataError:
        pass

filelist = filelist.iloc[unemptyfilelist]
filelist = filelist.reset_index(drop=True)

##* make concat df 
def makedf(i):
    wes_AF = pd.read_csv('~/jiye/copycomparison/newdata/WES_BRCA2/final-BRCA2-' + filelist.iloc[i,1], sep='\t', header=None, on_bad_lines='skip')
    wes_AF = wes_AF.iloc[:,:-1]
    wes_AF = wes_AF.rename(columns={1:'loc', 4:'genotype'})
    wes_AF['genome_AF'] = 1 - wes_AF.iloc[:,5]/wes_AF.iloc[:,5:].sum(axis=1) ##* Alternative Allele Frequency calculation 
    wes_AF = wes_AF[wes_AF['genotype'] != '1/1']
    wes_AF['sample_ID'] = filelist.iloc[i,0]
    wes_AF['response'] = sampleinfo.iloc[i,1]
    wes_AF['duration'] = sampleinfo.iloc[i,2]
    wes_AF = wes_AF.dropna() ##* allele frequency = x/0 
    wes_AF = wes_AF.loc[:,['loc','genome_AF','sample_ID','response','duration']]

    variantinfo = pd.read_csv('~/jiye/copycomparison/newdata/vep_WES_BRCA2/BRCA2-' + filelist.iloc[i,1] + '-vep-filtered0-canonical', sep='\t', header=None, on_bad_lines='skip')
    variantinfo.columns =['loc','ENSG','ENST','consequence','impact']
    variantinfo['loc'] = variantinfo['loc'].str.replace('chr13:','') ##* BRCA1
    variantinfo['impact'] = variantinfo['impact'].str.replace('IMPACT=','')
    variantinfo = variantinfo[~variantinfo['loc'].str.contains('-')] #only SNV
    variantinfo['loc'] = pd.to_numeric(variantinfo['loc'])
    variantinfo = variantinfo.loc[:,['loc','impact','consequence']]

    final_wes = pd.merge(wes_AF, variantinfo, on='loc')

    rna_depth10_file = filelist.iloc[i,2]
    rna_depth10_file = rna_depth10_file.replace('.vcf','')
    rna_depth10 = pd.read_csv('~/jiye/copycomparison/newdata/depth10_RNA_BRCA2/brca2-' + rna_depth10_file + '-depth10', sep='\t', header=None, on_bad_lines='skip')

    rna_depth10.columns = ['chr','loc','depth']
    wes_rna = pd.merge(rna_depth10, final_wes, on='loc')
    wes_rna = wes_rna.drop(['chr','depth'], axis=1)

    rna = pd.read_csv('~/jiye/copycomparison/newdata/RNA_BRCA2/final-brca2-' + filelist.iloc[i,2], sep='\t', header=None, on_bad_lines='skip')
    rna = rna.rename(columns={1:'loc', 4:'genotype'})
    rna['RNA_AF'] = 1 - rna.iloc[:,5]/rna.iloc[:, 5:].sum(axis=1)
    rna = rna.loc[:,['loc','RNA_AF']]
    rna = rna.dropna()
    rna_matched = pd.merge(wes_rna, rna, on='loc', how='left')
        
    return rna_matched

concatdf = makedf(0)

for i in range(1, filelist.shape[0]):
    tmpdf = makedf(i)
    concatdf = pd.concat([concatdf,tmpdf], ignore_index=True)

finaloutput = concatdf

finaloutput['mut'] = 'mt'
con=finaloutput['RNA_AF'].isna()
finaloutput.loc[con,'mut'] = 'wt'
finaloutput = finaloutput.fillna(0)

#%%
finaloutput.to_csv('/home/jiye/jiye/copycomparison/variantinfo/withsalvage/depth10/WES_RNA_BRCA2.csv', index=False)

#%%
#finaloutput.to_csv('/home/jiye/jiye/copycomparison/variantinfo/WES_RNA_BRCA2.csv', index=False)