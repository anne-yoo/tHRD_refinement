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
import matplotlib.cm as cm
from matplotlib.pyplot import gcf
# %%
discovery_TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', index_col=0, sep='\t')
discovery_TPM = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_transcript_exp_symbol.txt', index_col=0, sep='\t')
discovery_gene = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_gene_exp_TPM_symbol.txt', index_col=0, sep='\t')

# %%
discovery_clinical = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.csv', sep=',')
validation_TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/modelinput/validation_TU.txt', sep='\t', index_col=0)
validation_TPM = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/validation_transcript_exp.txt', sep='\t', index_col=0)
validation_clinical = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/validation_clinicalinfo.csv', sep=',')



# %%
omitlist = ['SV-OV-P055','SV-OV-P079','SV-OV-P045','SV-OV-P131','SV-OV-P134','SV-OV-P137','SV-OV-P142','SV-OV-P143','SV-OV-P158','SV-OV-P163','SV-OV-P181','SV-OV-P221','SV-OV-P035','SV-OV-P040','SV-OV-P042','SV-OV-P048','SV-OV-P049','SV-OV-P050','SV-OV-P065','SV-OV-P069','SV-OV-P080','SV-OV-P082','SV-OV-P099','SV-OV-P100','SV-OV-P107','SV-OV-P124','SV-OV-P128','SV-OV-P154','SV-OV-P164','SV-OV-P173','SV-OV-P174','SV-OV-P175','SV-OV-P219','SV-OV-P068','SV-OV-P111','SV-OV-P105','SV-OV-P058','SV-OV-P067','SV-OV-P070','SV-OV-P072','SV-OV-P087','SV-OV-P109']

addlist = ['SV-OV-P055','SV-OV-P079','SV-OV-P045','SV-OV-P131','SV-OV-P134','SV-OV-P137','SV-OV-P142','SV-OV-P143','SV-OV-P158','SV-OV-P163','SV-OV-P181','SV-OV-P221','SV-OV-P035','SV-OV-P040','SV-OV-P042','SV-OV-P048','SV-OV-P050','SV-OV-P065','SV-OV-P069','SV-OV-P080','SV-OV-P082','SV-OV-P099','SV-OV-P107','SV-OV-P124','SV-OV-P154','SV-OV-P164','SV-OV-P173','SV-OV-P174','SV-OV-P175','SV-OV-P219','SV-OV-P068','SV-OV-P105','SV-OV-P067','SV-OV-P070','SV-OV-P087']

val_col = validation_TPM.columns

dis_col = set(discovery_clinical['sample_id'])

val_omit = set(omitlist).intersection(set(val_col))

#%%
validation_TU.columns = validation_TU.columns.str.replace("T", "P")
validation_TPM.columns = validation_TPM.columns.str.replace("T", "P")

validation_TPM = validation_TPM.drop(columns=list(val_omit))
validation_TU = validation_TU.drop(columns=list(val_omit))




# %%
validation_TPM.sort_index(axis=1, inplace=True)
validation_TU.sort_index(axis=1, inplace=True)

#%%
###** validation cohort gene expression preprocessing ####

#val_gene = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/tmpforgenetpm/220722_request_tpm.txt', sep='\t', index_col=0)
val_gene = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/validation_gene_exp_new.txt', sep='\t', index_col=0)
val_gene.columns = val_gene.columns.str.replace("SMC_OV_OVLB_","")
val_gene.columns = val_gene.columns.str.replace("SV_OV_HRD_","")
val_gene.columns = val_gene.columns.str[:10]
val_gene.columns = val_gene.columns.str.replace("F","P")
val_gene.columns = val_gene.columns.str.replace("T","P")

val_samplelist = val_gene.columns

collist = list(validation_TU.columns)
#collist = list(add_val_TU.columns)
#rowlist = list(discovery_gene.index)

val_gene = val_gene.loc[:,val_gene.columns.isin(collist)]
#val_gene = val_gene[val_gene.index.isin(rowlist)]
val_gene_final = val_gene.loc[:, ~val_gene.columns.duplicated()]
val_gene_final.sort_index(axis=1, inplace=True)

#%%
#val_gene_final.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_geneTPM.txt', sep='\t')
# %%
#validation_TU.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_TU.txt', sep='\t')
#validation_TPM.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_TPM.txt', sep='\t')

# %%
onlypre_TPM = discovery_TPM.iloc[:,1::2]
onlypre_TPM.columns = onlypre_TPM.columns.str[:-4]

add_val_TPM = pd.concat([validation_TPM, onlypre_TPM], axis=1)

onlypre_TU = discovery_TU.iloc[:,1::2]
onlypre_TU.columns = onlypre_TU.columns.str[:-4]

add_val_TU = pd.concat([validation_TU, onlypre_TU], axis=1)

# %%
#add_val_TU.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/156_validation_TU.txt', sep='\t')
#add_val_TPM.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/156_validation_TPM.txt', sep='\t')

# %%
discovery_clinical2 = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/discovery_clinicalinfo.csv')
validation_clinical = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/validation_clinicalinfo.csv', sep=',')
validation_clinical.columns = ['sample_id','OM/OS','response','drug','interval','BRCAmut','line','time','line_binary']
validation_clinical = validation_clinical.replace("Maintenance","maintenance")
# %%
discovery_clinical2.sort_values('sampleID', inplace=True, ascending=True)
dis_brca = list(discovery_clinical2['BRCAmut'])
dis_omos = list(discovery_clinical2['OM/OS'])
#%%
final_discovery_clinical = discovery_clinical.iloc[0::2,[0,2,3,4,6]]
final_discovery_clinical['BRCAmut'] = dis_brca
final_discovery_clinical['OM/OS'] = dis_omos

#%%
original_list = list(final_discovery_clinical['BRCAmut'])
repeated_list = [item for item in original_list for _ in range(2)]

discovery_clinical['BRCAmut'] = repeated_list

#%%
discovery_clinical.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t', index=False)

# %%
wholesamplelist = add_val_TPM.columns

# %%
validation_clinical2 = validation_clinical.iloc[:,[0,6,2,4,8,5,1]]

inter1 = set(wholesamplelist).intersection(set(validation_clinical2['sample_id']))
inter2 = set(wholesamplelist).intersection(set(final_discovery_clinical['sample_id']))


validation_clinical_156 = pd.concat([validation_clinical2[validation_clinical2['sample_id'].isin(inter1)], final_discovery_clinical[final_discovery_clinical['sample_id'].isin(inter2)]], axis=0)

validation_clinical_156 = validation_clinical_156[validation_clinical_156['sample_id'].isin(wholesamplelist)]
validation_clinical_156 = validation_clinical_156.sort_values(['sample_id'])
validation_clinical_156 = validation_clinical_156.drop_duplicates(subset='sample_id')

# %%
validation_clinical_156 = validation_clinical_156.replace(regex=['LM'],value='L')
# %%
#validation_clinical_156.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_validation_clinicalinfo.txt', sep='\t', index=False)
# %%
