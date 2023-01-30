
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


wgs_filelist = pd.read_csv('~/jiye/copycomparison/newdata/new_WGS_RNA_filelist2.csv', sep=',')
wgs_filelist.columns = ['sample','WGSfile','RNAfile']
wgs_filelist.drop([63], axis=0, inplace=True)

merged = response_separation() 

cl_data = merged.iloc[:,[0,3,4]]
sampletmp = pd.merge(wgs_filelist,cl_data,on='sample')
sampletmp.columns = ['sample_ID','WGSfile','RNAfile','duration','response']
sampleinfo = sampletmp.loc[:,['sample_ID','response','duration']] ##* WES~RNA sample clinical info
filelist = sampletmp.loc[:,['sample_ID','WGSfile','RNAfile']] ##* WES~RNA matched file name

#%%
def makedf(i):
    wgs_AF = pd.read_csv('~/jiye/copycomparison/control_PD/new_WGS_BRCA2_header/final-BRCA2-' + filelist.iloc[i,1], sep='\t', header=None, on_bad_lines='skip')
    wgs_AF = wgs_AF.rename(columns={1:'loc', 4:'genotype'})
    wgs_AF['genome_AF'] = 1 - wgs_AF.iloc[:,5]/wgs_AF.iloc[:,5:].sum(axis=1) ##* Alternative Allele Frequency calculation 
    wgs_AF = wgs_AF[wgs_AF['genotype'] != '1/1']
    wgs_AF['sample_ID'] = filelist.iloc[i,0]
    wgs_AF['response'] = sampleinfo.iloc[i,1]
    wgs_AF['duration'] = sampleinfo.iloc[i,2]
    wgs_AF = wgs_AF.dropna() ##* allele frequency = x/0 
    wgs_AF = wgs_AF.loc[:,['loc','genome_AF','sample_ID','response','duration']]

    variantinfo = pd.read_csv('~/jiye/copycomparison/newdata/vep_WGS_BRCA2/BRCA2-' + filelist.iloc[i,1] + '-vep-filtered0-canonical', sep='\t', header=None, on_bad_lines='skip')
    variantinfo.columns =['loc','ENSG','ENST','consequence','impact']
    variantinfo['loc'] = variantinfo['loc'].str.replace('chr13:','') ##* BRCA2
    variantinfo['impact'] = variantinfo['impact'].str.replace('IMPACT=','')
    variantinfo = variantinfo[~variantinfo['loc'].str.contains('-')] #only SNV
    variantinfo['loc'] = pd.to_numeric(variantinfo['loc'])
    variantinfo = variantinfo.loc[:,['loc','impact','consequence']]

    final_wes = pd.merge(wgs_AF, variantinfo, on='loc')

    return final_wes

concatdf = makedf(0)

for i in range(1, filelist.shape[0]):
    tmpdf = makedf(i)
    concatdf = pd.concat([concatdf,tmpdf], ignore_index=True)

finaloutput = concatdf
    
#%%
finaloutput.to_csv('/home/jiye/jiye/copycomparison/variantinfo/modifiedAF/WGS_genome_BRCA2.csv', index=False)

#%%
# finaloutput.to_csv('/home/jiye/jiye/copycomparison/variantinfo/withsalvage/depth10/WGS_genome_BRCA2.csv', index=False)

#%%
#finaloutput.to_csv('/home/jiye/jiye/copycomparison/variantinfo/WGS_genome_BRCA2.csv', index=False)