#%%
#! this code is.....
####^^
##^ 1. transcript usage (whole_prop.txt) / 2. transcript read count (merged.txt) / 3. gene expression TPM (_tpm.txt)
#####*** added num = 32
##* 230706_Pentamedix n=8
#["YUHS_PRE-02_TF","YUHS_PRE-03_TF","YUHS-POST-01-TF","YUHS-POST-02-TF","YUHS_POST-16_TT_Lib-2","YUHS_POST-17_TT_Lib","YUHS_POST-18_TT_Lib-1","YUHS_POST-19_TT_Lib"]
#["SV-OV-P107-bfD","SV-OV-P158-bfD","SV-OV-P087-atD","SV-OV-P107-atD","YUHS-179-atD","YUHS-182-atD","SV-OV-P154-atD","SV-OV-P178-atD"]

##* 230622_Pentamedix n=10
#["YUHS-PRE-6-TF","YUHS-PRE-9-TF","YUHS-PRE-13-TF","YUHS_PRE_14_TF","YUHS-PRE-15-TF","YUHS_PRE_16_TT_Lib","YUHS_POST_06_TF","YUHS-POST-07-TT","YUHS_POST_15_TF","YUHS_PRE_18_TT_Lib"]
#["SV-OV-P040-bfD","SV-OV-P124-bfD","SV-OV-P182-bfD","SV-OV-P221-bfD","YUHS-181-bfD","YUHS-179-bfD","SV-OV-P040-atD","SV-OV-P042-atD","YUHS-181-atD","SV-OV-P154-bfD"]


##* HG_230524_Pentamedix n=13
#["YUHS_PRE-01_TF","YUHS_PRE-04_TT","YUHS_PRE-07_TT","YUHS-PRE-10-TT","YUHS_PRE-11_TF","YUHS-PRE-12-TT","YUHS_POST-3-TT","YUHS_POST-4-TT","YUHS_POST-05_TF","YUHS_POST-09_TF","YUHS_POST-10-TT","YUHS_POST-12-TT","YUHS_POST-14_TF"]
#["SV-OV-P087-bfD","SV-OV-P174-bfD","SV-OV-P042-bfD","SV-OV-P163-bfD","SV-OV-P178-bfD","SV-OV-P181-bfD","SV-OV-P158-atD","SV-OV-P174-atD","YUHS-180-atD","SV-OV-P124-atD","SV-OV-P163-atD","SV-OV-P181-atD","SV-OV-P221-atD"]

##* 230609_Pentamedix n=1
#["YUHS-PRE-05-TF"]
#["YUHS-180-bfD"]

####^^###################################################################################################################
##* After Qualimap QC + 겹치는 샘플 제외 num = 20
##* 230706_Pentamedix n=4
#sam_0706 = ["YUHS_POST-16_TT_Lib-2","YUHS_POST-17_TT_Lib","YUHS_POST-18_TT_Lib-1","YUHS_POST-19_TT_Lib"]
#name_0706 = ["YUHS-179-atD","YUHS-182-atD","SV-OV-P154-atD","SV-OV-P178-atD"]

##* 230622_Pentamedix n=5
#sam_0622 = ["YUHS_PRE_16_TT_Lib","YUHS_POST_06_TF","YUHS-POST-07-TT","YUHS_POST_15_TF","YUHS_PRE_18_TT_Lib"]
#name_0622 = ["YUHS-179-bfD","SV-OV-P040-atD","SV-OV-P042-atD","YUHS-181-atD","SV-OV-P154-bfD"]


##* HG_230524_Pentamedix n=11
#sam_0524 = ["YUHS_PRE-04_TT","YUHS_PRE-07_TT","YUHS-PRE-10-TT","YUHS_PRE-11_TF","YUHS-PRE-12-TT","YUHS_POST-3-TT","YUHS_POST-4-TT","YUHS_POST-09_TF","YUHS_POST-10-TT","YUHS_POST-12-TT","YUHS_POST-14_TF"]
#name_0524 = ["SV-OV-P174-bfD","SV-OV-P042-bfD","SV-OV-P163-bfD","SV-OV-P178-bfD","SV-OV-P181-bfD","SV-OV-P158-atD","SV-OV-P174-atD", "SV-OV-P124-atD","SV-OV-P163-atD","SV-OV-P181-atD","SV-OV-P221-atD"]

##* 230609_Pentamedix n=0

####^^###################################################################################################################
##* Only Pre-Post Matched Samples (No QC cut, n=26) (230904)
##* 230706_Pentamedix n=6
#sam_0706 = ["YUHS_PRE-03_TF", "YUHS_PRE-09_TF", "YUHS_PRE-16_TT_Lib", "YUHS_POST-19_TT_Lib", "YUHS_POST-16_TT_Lib-2", "YUHS_POST-18_TT_Lib-1"]
#name_0706 = ["SV-OV-P158-bfD","SV-OV-P124-bfD","YUHS-179-bfD","SV-OV-P178-atD","YUHS-179-atD","SV-OV-P154-atD"]

##* 230622_Pentamedix n=8
#sam_0622 = ["YUHS-PRE-6-TF","YUHS_PRE_14_TF","YUHS-PRE-15-TF","YUHS_PRE_18_TT_Lib","YUHS_POST_05_TF","YUHS_POST_06_TF","YUHS-POST-07-TT","YUHS_POST_15_TF"]
#name_0622 = ["SV-OV-P040-bfD","SV-OV-P221-bfD","YUHS-181-bfD","SV-OV-P154-bfD","YUHS-180-atD","SV-OV-P040-atD","SV-OV-P042-atD","YUHS-181-atD"]

##* HG_230524_Pentamedix n=11
#sam_0524 = ["YUHS_PRE-04_TT","YUHS_PRE-07_TT","YUHS-PRE-10-TT","YUHS_PRE-11_TF","YUHS-PRE-12-TT","YUHS_POST-3-TT","YUHS_POST-4-TT","YUHS_POST-09_TF","YUHS_POST-10-TT","YUHS_POST-12-TT","YUHS_POST-14_TF"]
#name_0524 = ["SV-OV-P174-bfD","SV-OV-P042-bfD","SV-OV-P163-bfD","SV-OV-P178-bfD","SV-OV-P181-bfD","SV-OV-P158-atD","SV-OV-P174-atD","SV-OV-P124-atD","SV-OV-P163-atD","SV-OV-P181-atD","SV-OV-P221-atD"]

##* 230609_Pentamedix n=1
#sam_0609 = ["YUHS-PRE-05-TF"]
#name_0609 = ["YUHS-180-bfD"]


#######^^^@@@@@@#########################################################################################################
####^^###################################################################################################################
##* Only Pre-Post Matched Samples (No QC cut, total n=40) (230912)
##* 230706_Pentamedix n=6
sam_0706 = ["YUHS_PRE-03_TF", "YUHS_PRE-09_TF", "YUHS_PRE-16_TT_Lib", "YUHS_POST-19_TT_Lib", "YUHS_POST-16_TT_Lib-2", "YUHS_POST-18_TT_Lib-1"]
name_0706 = ["SV-OV-P158-bfD","SV-OV-P124-bfD","YUHS-179-bfD","SV-OV-P178-atD","YUHS-179-atD","SV-OV-P154-atD"]

##* 230622_Pentamedix n=8
sam_0622 = ["YUHS-PRE-6-TF","YUHS_PRE_14_TF","YUHS-PRE-15-TF","YUHS_PRE_18_TT_Lib","YUHS_POST_05_TF","YUHS_POST_06_TF","YUHS-POST-07-TT","YUHS_POST_15_TF"]
name_0622 = ["SV-OV-P040-bfD","SV-OV-P221-bfD","YUHS-181-bfD","SV-OV-P154-bfD","YUHS-180-atD","SV-OV-P040-atD","SV-OV-P042-atD","YUHS-181-atD"]

##* HG_230524_Pentamedix n=11
sam_0524 = ["YUHS_PRE-04_TT","YUHS_PRE-07_TT","YUHS-PRE-10-TT","YUHS_PRE-11_TF","YUHS-PRE-12-TT","YUHS_POST-3-TT","YUHS_POST-4-TT","YUHS_POST-09_TF","YUHS_POST-10-TT","YUHS_POST-12-TT","YUHS_POST-14_TF"]
name_0524 = ["SV-OV-P174-bfD","SV-OV-P042-bfD","SV-OV-P163-bfD","SV-OV-P178-bfD","SV-OV-P181-bfD","SV-OV-P158-atD","SV-OV-P174-atD","SV-OV-P124-atD","SV-OV-P163-atD","SV-OV-P181-atD","SV-OV-P221-atD"]

##* 230609_Pentamedix n=1
sam_0609 = ["YUHS-PRE-05-TF"]
name_0609 = ["YUHS-180-bfD"]

##* 230824_Pentamedix n=6
sam_0824 = ["YUHS-POST-20-TF-RNASQ-Lib","YUHS-POST-22-TF-RNASQ-Lib","YUHS-POST-21-TF-RNASQ-Lib","YUHS-POST-23-TF-RNASQ-Lib","YUHS-POST-25-TF-RNASQ-Lib","YUHS-PRE-20-TF-RNASQ-Lib"]
name_0824 = ["YUHS-183-atD","SV-OV-P134-atD","SV-OV-P045-atD","SV-OV-P137-atD","SV-OV-P143-atD","YUHS-183-bfD"]

##* 230904_Pentamedix n=1
sam_0904 = ["YUHS-POST-26-TF-RNASQ-Lib"]
name_0904 = ["SV-OV-P164-atD"]

##* samples to exclude from the original dataset
exc = ['SV-OV-P174-bfD','SV-OV-P174-atD','SV-OV-P250-atD','SV-OV-P250-bfD','SV-OV-P059-bfD','SV-OV-P059-atD','SV-OV-P134-atD','SV-OV-P137-atD','SV-OV-P143-atD','SV-OV-P164-atD']

##* sample to include from the validation cohort
# ['SV-OV-P045']
# directory: /home/jiye/jiye/copycomparison/data/readcount/20DNB79-B/SV-OV-T045-FAligned.sortedByCoord.out.bam


#%%
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.multitest as ssm
import scipy as sp
import pickle
import sys
import os
import re

#%%
######^^^^ 1. Transcript Usage - whole_prop.txt #######

tu_0706 = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/202306_newRNAseq/230706_Pentamedix/trim/2nd_align/quantified/processed/whole_prop.txt',sep='\t')
tu_0622 = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/202306_newRNAseq/230622_Pentamedix/trim/2nd_align/quantified/processed/whole_prop.txt',sep='\t')
tu_0524 = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/202306_newRNAseq/HG_230524_Pentamedix/trim/2nd_align/quantified/processed/whole_prop.txt',sep='\t')
tu_0609 = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/202306_newRNAseq/230609_Pentamedix/trim/2nd_align/quantified/processed/whole_prop.txt',sep='\t')
tu_0824 = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/202306_newRNAseq/230824_Pentamedix/trim/2nd_align/quantified/processed/whole_prop.txt',sep='\t')
tu_0904 = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/202306_newRNAseq/230904_Pentamedix/trim/2nd_align/quantified/processed/whole_prop.txt',sep='\t')

tu_org = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/validation_TU.txt', sep='\t')

# %%
a_0706 = tu_0706[sam_0706]
a_0622 = tu_0622[sam_0622]
a_0524 = tu_0524[sam_0524]
a_0609 = tu_0609[sam_0609]
a_0824 = tu_0824[sam_0824]
a_0904 = tu_0904[sam_0904]


a_0706.columns = name_0706
a_0622.columns = name_0622
a_0524.columns = name_0524
a_0609.columns = name_0609
a_0824.columns = name_0824
a_0904.columns = name_0904

# %%
enstid = tu_0524['gene_ENST']
merged_tu = pd.concat([enstid,a_0706,a_0622,a_0524,a_0609,a_0824,a_0904], axis=1)
merged_tu.set_index(merged_tu.columns[0], inplace=True)
merged_tu = merged_tu.sort_index(axis=1)

#%%
#merged_tu.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/tmp/33_discovery_TU.txt', sep='\t')
#%%




#%%
#%%
######^^^^ 2. transcript read count (merged.txt) #######

tpm_0706 = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/202306_newRNAseq/230706_Pentamedix/trim/2nd_align/quantified/processed/merged.txt',sep='\t')
tpm_0622 = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/202306_newRNAseq/230622_Pentamedix/trim/2nd_align/quantified/processed/merged.txt',sep='\t')
tpm_0524 = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/202306_newRNAseq/HG_230524_Pentamedix/trim/2nd_align/quantified/processed/merged.txt',sep='\t')
tpm_0609 = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/202306_newRNAseq/230609_Pentamedix/trim/2nd_align/quantified/processed/merged.txt',sep='\t')
tpm_0824 = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/202306_newRNAseq/230824_Pentamedix/trim/2nd_align/quantified/processed/merged.txt',sep='\t')
tpm_0904 = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/202306_newRNAseq/230904_Pentamedix/trim/2nd_align/quantified/processed/merged.txt',sep='\t')

tu_org = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/validation_transcript_exp.txt', sep='\t')

# %%
m_0706 = tpm_0706[sam_0706]
m_0622 = tpm_0622[sam_0622]
m_0524 = tpm_0524[sam_0524]
m_0609 = tpm_0609[sam_0609]
m_0824 = tpm_0824[sam_0824]
m_0904 = tpm_0904[sam_0904]

m_0706.columns = name_0706
m_0622.columns = name_0622
m_0524.columns = name_0524
m_0609.columns = name_0609
m_0904.columns = name_0904
m_0824.columns = name_0824

# %%
enstid = tpm_0524['gene_ENST']
merged_tpm = pd.concat([enstid,m_0706,m_0622,m_0524,m_0609,m_0824,m_0904], axis=1)
merged_tpm.set_index(merged_tpm.columns[0], inplace=True)
merged_tpm = merged_tpm.sort_index(axis=1)

# %%
#merged_tpm.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/tmp/33_discovery_transcript_exp.txt', sep='\t')

#%%
######^^^^ 3. gene expression readcount (.txt) #######

first_tpm = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/202306_newRNAseq/readcounts/230706_Pentamedix/'+sam_0706[0]+'_readcounts.txt', sep='\t', skiprows=1)
first = first_tpm.iloc[:,[0,6]]
first.columns = ['Geneid',name_0706[0]]
tpm = first
# %%
for i in range(1,len(sam_0706)):
    tmp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/202306_newRNAseq/readcounts/230706_Pentamedix/'+sam_0706[i]+'_readcounts.txt', sep='\t', skiprows=1)
    tmp = pd.DataFrame(tmp.iloc[:,-1])
    tmp.columns = [name_0706[i]]
    tpm = pd.concat([tpm, tmp], axis=1)

for i in range(0,len(sam_0622)):
    tmp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/202306_newRNAseq/readcounts/230622_Pentamedix/'+sam_0622[i]+'_readcounts.txt', sep='\t', skiprows=1)
    tmp = pd.DataFrame(tmp.iloc[:,-1])
    tmp.columns = [name_0622[i]]
    tpm = pd.concat([tpm, tmp], axis=1)
    
for i in range(0,len(sam_0524)):
    tmp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/202306_newRNAseq/readcounts/230524_Pentamedix/'+sam_0524[i]+'_readcounts.txt', sep='\t', skiprows=1)
    tmp = pd.DataFrame(tmp.iloc[:,-1]) 
    tmp.columns = [name_0524[i]]
    tpm = pd.concat([tpm, tmp], axis=1)

for i in range(0,len(sam_0609)):
    tmp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/202306_newRNAseq/readcounts/230609_Pentamedix/'+sam_0609[i]+'_readcounts.txt', sep='\t', skiprows=1)
    tmp = pd.DataFrame(tmp.iloc[:,-1]) 
    tmp.columns = [name_0609[i]]
    tpm = pd.concat([tpm, tmp], axis=1)
    
for i in range(0,len(sam_0824)):
    tmp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/202306_newRNAseq/readcounts/230824_Pentamedix/'+sam_0824[i]+'_readcounts.txt', sep='\t', skiprows=1)
    tmp = pd.DataFrame(tmp.iloc[:,-1]) 
    tmp.columns = [name_0824[i]]
    tpm = pd.concat([tpm, tmp], axis=1)

for i in range(0,len(sam_0904)):
    tmp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/202306_newRNAseq/readcounts/230904_Pentamedix/'+sam_0904[i]+'_readcounts.txt', sep='\t', skiprows=1)
    tmp = pd.DataFrame(tmp.iloc[:,-1]) 
    tmp.columns = [name_0904[i]]
    tpm = pd.concat([tpm, tmp], axis=1)
# %%
tpm.set_index(tpm.columns[0], inplace=True)
tpm = tpm.sort_index(axis=1)

#%%%
#tpm.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/tmp/33_discovery_gene_exp_readcount.txt', sep='\t')



#%%
######^^^^ 4. gene expression TPM (_tpm.txt) #######

first_tpm = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/202306_newRNAseq/readcounts/230706_Pentamedix/'+sam_0706[0]+'_readcounts_tpm.txt', sep='\t', skiprows=1)
first = first_tpm.iloc[:,[0,6]]
first.columns = ['Geneid',name_0706[0]]
tpm = first
# %%
for i in range(1,len(sam_0706)):
    tmp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/202306_newRNAseq/readcounts/230706_Pentamedix/'+sam_0706[i]+'_readcounts_tpm.txt', sep='\t', skiprows=1)
    tmp = pd.DataFrame(tmp.iloc[:,-1])
    tmp.columns = [name_0706[i]]
    tpm = pd.concat([tpm, tmp], axis=1)

for i in range(0,len(sam_0622)):
    tmp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/202306_newRNAseq/readcounts/230622_Pentamedix/'+sam_0622[i]+'_readcounts_tpm.txt', sep='\t', skiprows=1)
    tmp = pd.DataFrame(tmp.iloc[:,-1])
    tmp.columns = [name_0622[i]]
    tpm = pd.concat([tpm, tmp], axis=1)
    
for i in range(0,len(sam_0524)):
    tmp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/202306_newRNAseq/readcounts/230524_Pentamedix/'+sam_0524[i]+'_readcounts_tpm.txt', sep='\t', skiprows=1)
    tmp = pd.DataFrame(tmp.iloc[:,-1]) 
    tmp.columns = [name_0524[i]]
    tpm = pd.concat([tpm, tmp], axis=1)

for i in range(0,len(sam_0609)):
    tmp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/202306_newRNAseq/readcounts/230609_Pentamedix/'+sam_0609[i]+'_readcounts_tpm.txt', sep='\t', skiprows=1)
    tmp = pd.DataFrame(tmp.iloc[:,-1]) 
    tmp.columns = [name_0609[i]]
    tpm = pd.concat([tpm, tmp], axis=1)
    
for i in range(0,len(sam_0824)):
    tmp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/202306_newRNAseq/readcounts/230824_Pentamedix/'+sam_0824[i]+'_readcounts_tpm.txt', sep='\t', skiprows=1)
    tmp = pd.DataFrame(tmp.iloc[:,-1]) 
    tmp.columns = [name_0824[i]]
    tpm = pd.concat([tpm, tmp], axis=1)

for i in range(0,len(sam_0904)):
    tmp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/202306_newRNAseq/readcounts/230904_Pentamedix/'+sam_0904[i]+'_readcounts_tpm.txt', sep='\t', skiprows=1)
    tmp = pd.DataFrame(tmp.iloc[:,-1]) 
    tmp.columns = [name_0904[i]]
    tpm = pd.concat([tpm, tmp], axis=1)
# %%
tpm.set_index(tpm.columns[0], inplace=True)
tpm = tpm.sort_index(axis=1)

#%%%
#tpm.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/tmp/33_discovery_gene_exp_TPM.txt', sep='\t')


# %%
######^^^^ 4. merge old + new files #######

##########*** TU ##########
old_majortu = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/discovery_majorTU.txt', sep='\t', index_col=0)
old_minortu = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/discovery_minorTU.txt', sep='\t', index_col=0)
new_tu = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/tmp/33_discovery_TU.txt', sep='\t', index_col=0)

tu_val = pd.read_csv('/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/merged_TU/whole_TU/include_pre-post_filtered_data.txt', sep='\t', index_col=0)
tu_add = tu_val[['SV-OV-T045-RP-RNA_M']]
tu_add.columns = ['SV-OV-P045-bfD']

old_wholetu = pd.concat([old_majortu,old_minortu])
old_wholetu = old_wholetu.reindex(new_tu.index)
old_wholetu = old_wholetu.drop(exc, axis=1)

final_wholetu = pd.concat([old_wholetu, new_tu], axis=1)
final_wholetu = pd.concat([final_wholetu,tu_add], axis=1)
final_wholetu = final_wholetu.sort_index(axis=1)

#%%
#final_wholetu.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_TU.txt', sep='\t')




# %%
#########*** Transcript expression (TMM) ###########
old_transexp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/discovery_transcript_exp.txt', sep='\t', index_col=0)
new_transexp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/tmp/33_discovery_transcript_exp.txt', sep='\t', index_col=0)

transexp_val = pd.read_csv ('/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/merged_TU/230407_ex_pre-post_TPM_sig_response_input.txt', sep='\t', index_col=0)
trans_add = transexp_val[['SV-OV-T045']]
trans_add.columns = ['SV-OV-P045-bfD']

sample = old_transexp.columns.tolist()# %%
samples = sample

for i in range(int(len(samples)/2)):
    i = i*2
    if "atD" in samples[i]:
        samples[i+1] = samples[i+1][:10]+"-bfD"
    elif "bfD" in samples[i]:
        samples[i+1] = samples[i+1][:10]+"-atD"
    elif "atD" in samples[i+1]:
        samples[i] = samples[i][:10]+"-bfD"
    elif "bfD" in samples[i+1]:
        samples[i] = samples[i][:10]+"-atD"

old_transexp.columns = samples

old_transexp = old_transexp.reindex(new_transexp.index)
old_transexp = old_transexp.drop(exc, axis=1)


final_transexp = pd.concat([old_transexp,new_transexp], axis=1)
final_transexp = pd.merge(final_transexp,trans_add, how='inner', left_index=True, right_index=True)
final_transexp = final_transexp.sort_index(axis=1)
#%%
#final_transexp.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_transcript_exp.txt', sep='\t')






#%%
#########*** gene expression (readcounts) ###########
old_geneexp = pd.read_csv('/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/readcount/final_data/230406_final_pre_post_samples_raw_counts.txt', sep='\t', index_col=0)
new_geneexp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/tmp/33_discovery_gene_exp_readcount.txt', sep='\t', index_col=0)


geneexp_val = pd.read_csv ('/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/readcount/220923_whole_raw_counts.txt', sep='\t', index_col=0)
geneexp_add = geneexp_val[['SV-OV-T045-F']]
geneexp_add.columns = ['SV-OV-P045-bfD']

old_geneexp.rename(columns = {'SV-OV-T105-P': 'SV-OV-P105-bfD'}, inplace = True)
old_geneexp = old_geneexp.drop(exc, axis=1)

final_geneexp = pd.merge(old_geneexp,new_geneexp, left_index=True, right_index=True)
final_geneexp = pd.merge(final_geneexp, geneexp_add, how='inner', left_index=True, right_index=True)

final_geneexp = final_geneexp.sort_index(axis=1)
# %%
#final_geneexp.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_readcount.txt', sep='\t')



#%%
#########*** gene expression (readcounts) with gene symbol ###########
symbol = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/ensgidtosymbol.dict', sep='\t', header=None, index_col=0)
symbol.columns = ['Gene Symbol']

sym_geneexp = pd.merge(final_geneexp, symbol, left_index=True, right_index=True)
#%%
#sym_geneexp.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_readcount_symbol.txt', sep='\t')
# %%









#%%
#########*** gene expression (TPM) ###########
old_geneexp = pd.read_csv('/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/readcount/final_data/TMM_230406_final_pre_post_samples_raw_counts.txt', sep='\t', index_col=0)
new_geneexp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/tmp/33_discovery_gene_exp_TPM.txt', sep='\t', index_col=0)


geneexp_val = pd.read_csv('/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/readcount/merged_tpm.txt', sep='\t', index_col=0)
geneexp_add = geneexp_val[['..20DNB79.B.SV.OV.T045.FAligned.sortedByCoord.out.bam']]
geneexp_add.columns = ['SV-OV-P045-bfD']

old_geneexp.rename(columns = {'SV-OV-T105-P': 'SV-OV-P105-bfD'}, inplace = True)
old_geneexp = old_geneexp.drop(exc, axis=1)

final_geneexp = pd.merge(old_geneexp,new_geneexp, left_index=True, right_index=True)
final_geneexp = pd.merge(final_geneexp, geneexp_add, how='inner', left_index=True, right_index=True)

final_geneexp = final_geneexp.sort_index(axis=1)
# %%
#final_geneexp.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM.txt', sep='\t')



#%%
#########*** gene expression (readcounts) with gene symbol ###########
symbol = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/ensgidtosymbol.dict', sep='\t', header=None, index_col=0)
symbol.columns = ['Gene Symbol']

sym_geneexp = pd.merge(final_geneexp, symbol, left_index=True, right_index=True)
#%%
#sym_geneexp.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM_symbol.txt', sep='\t')
# %%
