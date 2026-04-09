#%%
#! this code is.....
"""
202308까지 추가된 데이터들로 DTU/DEG input data 몽땅 만들기 

<source data>
"/home/jiye/jiye/copycomparison/gDUTresearch/data/final_202306_discovery_~~""

<input directory>
"/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/input_DUTDEG/innate/~"
"/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/input_DUTDEG/acquired/~"

<result directory>
"/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/result_DUTDEG/innate/~"
"/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/result_DUTDEG/acquired/~"

step 1. innate / acquired resistance DEG
step 2. save stable gene / variable gene from step 1
step 3. innate / acquired resistance DTU
        1) innate-stable
        2) innate-variable
        3) acquired-stable
        4) acquired-variable
step 4. get DEGs(padj <0.05) and DUTs(empirical pval <0.005) list 
step 5. run SUPPA2! with innate / acquired resistance
step 6. run PCA with normalized gene readcounts + transcript tpm
step 7. exonic read histogram
"""



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
import matplotlib


# %%
###^ step 1. innate / acqruied resistance DEG DESeq2 input ###

###* innate resistance DEG input ###
clinical = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/discovery_clinicalinfo.txt', sep='\t')
readcount = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/final_202306_discovery_gene_readcount.txt', sep='\t', index_col=0)
clinical = clinical.iloc[:,:4]

# %%
pre_df = readcount.loc[:, readcount.columns.str.contains('bfD')]
pre_list = pre_df.columns
# %%
# Melt the dataframe B to long format
melted_clinical = clinical.melt(id_vars='Response', value_vars=['sampleID','bfD','atD'])

# Filter out rows which have their value in list A
filtered = melted_clinical[melted_clinical['value'].isin(pre_list)]

# Drop duplicates if there are any
filtered = filtered.drop_duplicates('value')

# Create the new dataframe
innate_meta = pd.DataFrame({
    'samples': pre_list,
    'Response': filtered.set_index('value')['Response'].reindex(pre_list).values
})

innate_meta['group'] = np.where(innate_meta['Response'] == 1, 'pre_R', 'pre_NR')

innate_meta = innate_meta[['samples','group']]

# %%
#pre_df.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/input_DUTDEG/innate/DEG_innate_countdata.txt', sep='\t')

#innate_meta.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/input_DUTDEG/innate/DUT_innate_metadata.txt', sep='\t', index=False)



# %%
###* acquired resistance DEG input ###
clinical = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/discovery_clinicalinfo.txt', sep='\t')
readcount = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/final_202306_discovery_gene_readcount.txt', sep='\t', index_col=0)
clinical = clinical.iloc[:,:4]

# %%
pre_list = readcount.columns
# Melt the dataframe B to long format
melted_clinical = clinical.melt(id_vars='Response', value_vars=['sampleID','bfD','atD'])

# Filter out rows which have their value in list A
filtered = melted_clinical[melted_clinical['value'].isin(pre_list)]

# Drop duplicates if there are any
filtered = filtered.drop_duplicates('value')

# Create the new dataframe
acquired_meta = pd.DataFrame({
    'samples': pre_list,
    'Response': filtered.set_index('value')['Response'].reindex(pre_list).values
})

acquired_meta['timing'] = 'bfD'
acquired_meta.loc[acquired_meta['samples'].str.contains("atD"), 'timing'] = 'atD'

acquired_meta = acquired_meta[acquired_meta['Response']==1]

acquired_meta['group'] = np.where(acquired_meta['timing'] == 'bfD', 'pre_R', 'post_R')

acquired_meta = acquired_meta[['samples','group']]

acquired_list = list(acquired_meta.samples)
acquired_df = readcount[acquired_list]
# %%
#acquired_df.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/input_DUTDEG/acquired/DEG_acquired_countdata.txt', sep='\t')
#acquired_meta.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/input_DUTDEG/acquired/DEG_acquired_metadata.txt', sep='\t', index=False)






# %%
###^ step 2. save stable gene / variable gene from step 1 ###

acquired_deg = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/result_DUTDEG/acquired/DEG_acquired_result.txt')
innate_deg = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/result_DUTDEG/innate/DEG_innate_result.txt')

idtosym = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/ensgidtosymbol.dict', sep='\t', header=None)
idtosym.columns = ['GeneID','GeneSymbol']

acquired_deg = acquired_deg.iloc[:,[0,6]]
innate_deg = innate_deg.iloc[:,[0,6]]
acquired_deg.columns = ['GeneID','padj']
innate_deg.columns = ['GeneID','padj']

acquired_deg_05 = acquired_deg[acquired_deg['padj']<0.05]
innate_deg_05 = innate_deg[innate_deg['padj']<0.05]

acquired_deg_05 = pd.merge(acquired_deg_05,idtosym, on='GeneID', how='inner')
innate_deg_05 = pd.merge(innate_deg_05,idtosym, on='GeneID', how='inner')











#%%    
###^ step 3. innate / acquired resistance DTU input ###
# %%
###* whole gene ###

dut_input = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/final_202306_discovery_transcript_exp.txt',sep='\t', index_col=0)
dut_input = dut_input[(dut_input>0).astype(int).sum(axis=1) > dut_input.shape[1]*0.6]
dut_input['GeneSymbol'] = dut_input.index.str.split("-",1).str[1]
dut_input = dut_input[dut_input['GeneSymbol']!= '-']

#%%
###* stable/variable
di_list = list(innate_meta['samples'])
dut_count = dut_input[di_list]

da_list = list(acquired_meta['samples'])
dut_count_ac = dut_input[da_list]

dut_innate_geneinfo = pd.DataFrame(dut_count.index.str.split("-",1).str[1])
dut_acquired_geneinfo = pd.DataFrame(dut_count_ac.index.str.split("-",1).str[1])
dut_innate_geneinfo['transcript'] = dut_count.index
dut_acquired_geneinfo['transcript'] = dut_count_ac.index
dut_innate_geneinfo.columns = ['geneid','transcript']
dut_acquired_geneinfo.columns = ['geneid','transcript']

dut_innate_geneinfo = dut_innate_geneinfo[['transcript','geneid']]
dut_acquired_geneinfo = dut_acquired_geneinfo[['transcript','geneid']]

#%%

# dut_count.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/input_DUTDEG/innate/DUT_innate_whole_countdata.txt', sep='\t')
# dut_count_ac.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/input_DUTDEG/acquired/DUT_acquired_whole_countdata.txt', sep='\t')
# dut_innate_geneinfo.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/input_DUTDEG/innate/DUT_innate_whole_geneinfo.txt', sep='\t', index=False) 
# dut_acquired_geneinfo.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/input_DUTDEG/acquired/DUT_acquired_whole_geneinfo.txt', sep='\t', index=False) 



#%%
dut_count['GeneSymbol'] = dut_count.index.str.split("-",1).str[1]
dut_count_ac['GeneSymbol'] = dut_count_ac.index.str.split("-",1).str[1]


t_list = list(dut_count['GeneSymbol'])
ac_variable = list(set(t_list).intersection(set(acquired_deg_05['GeneSymbol'])))
ac_stable = list(set(t_list)-set(ac_variable))

in_variable =  list(set(t_list).intersection(set(innate_deg_05['GeneSymbol'])))
in_stable = list(set(t_list)-set(in_variable))


# %%
innate_stable_count = dut_count.loc[dut_count['GeneSymbol'].isin(in_stable)]
innate_variable_count = dut_count.loc[dut_count['GeneSymbol'].isin(in_variable)]
acquired_stable_count = dut_count_ac.loc[dut_count_ac['GeneSymbol'].isin(ac_stable)]
acquired_variable_count = dut_count_ac.loc[dut_count_ac['GeneSymbol'].isin(ac_variable)]

innate_stable_geneinfo = pd.DataFrame(innate_stable_count['GeneSymbol'])
innate_variable_geneinfo = pd.DataFrame(innate_variable_count['GeneSymbol'])
acquired_stable_geneinfo = pd.DataFrame(acquired_stable_count['GeneSymbol'])
acquired_variable_geneinfo = pd.DataFrame(acquired_variable_count['GeneSymbol'])

innate_stable_count = innate_stable_count.drop(['GeneSymbol'],axis=1)
innate_variable_count = innate_variable_count.drop(['GeneSymbol'],axis=1)
acquired_stable_count = acquired_stable_count.drop(['GeneSymbol'],axis=1)
acquired_variable_count = acquired_variable_count.drop(['GeneSymbol'],axis=1)

# %%
innate_stable_count.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/input_DUTDEG/innate/DUT_innate_stable_countdata.txt', sep='\t')
innate_variable_count.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/input_DUTDEG/innate/DUT_innate_variable_countdata.txt', sep='\t')
acquired_stable_count.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/input_DUTDEG/acquired/DUT_acquired_stable_countdata.txt', sep='\t')
acquired_variable_count.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/input_DUTDEG/acquired/DUT_acquired_variable_countdata.txt', sep='\t')

innate_stable_geneinfo.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/input_DUTDEG/innate/DUT_innate_stable_geneinfo.txt', sep='\t') 
innate_variable_geneinfo.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/input_DUTDEG/innate/DUT_innate_variable_geneinfo.txt', sep='\t') 
acquired_stable_geneinfo.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/input_DUTDEG/acquired/DUT_acquired_stable_geneinfo.txt', sep='\t') 
acquired_variable_geneinfo.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/input_DUTDEG/acquired/DUT_acquired_variable_geneinfo.txt', sep='\t') 

innate_meta.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/input_DUTDEG/innate/DUT_innate_metadata.txt', sep='\t', index=False) 
acquired_meta.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/input_DUTDEG/acquired/DUT_acquired_metadata.txt', sep='\t', index=False) 







#%%

###^^  step 4. get DEGs(padj <0.05) and DUTs(empirical pval <0.005) list ++++ Venn Diagram

innate_deg = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/result_DUTDEG/innate/DEG_innate_result.txt', index_col=0)
innate_dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/result_DUTDEG/innate/DUT_innate_whole_SatuRnResult.txt', index_col=0)
acquired_deg = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/result_DUTDEG/acquired/DEG_acquired_result.txt', index_col=0)
acquired_dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/result_DUTDEG/acquired/DUT_acquired_whole_SatuRnResult.txt', index_col=0)


idtosym = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/ensgidtosymbol.dict', sep='\t', header=None, index_col=0)
idtosym.columns = ['gene_id']
innate_deg = pd.merge(innate_deg,idtosym,left_index=True, right_index=True)
innate_fil_deg = pd.DataFrame(innate_deg.loc[innate_deg['padj']<0.05, 'gene_id'])

acquired_deg = pd.merge(acquired_deg,idtosym,left_index=True, right_index=True)
acquired_fil_deg = pd.DataFrame(acquired_deg.loc[acquired_deg['padj']<0.05, 'gene_id'])

innate_fil_dut = pd.DataFrame(innate_dut.loc[innate_dut['empirical_pval']<0.005,'gene_id'])
acquired_fil_dut = pd.DataFrame(acquired_dut.loc[acquired_dut['empirical_pval']<0.005,'gene_id'])


# # %%
# innate_fil_deg.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/result_DUTDEG/innate/innate_deg_filtered.txt', sep='\t')
# innate_fil_dut.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/result_DUTDEG/innate/innate_dut_filtered.txt', sep='\t')
# acquired_fil_deg.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/result_DUTDEG/acquired/acquired_deg_filtered.txt', sep='\t')
# acquired_fil_dut.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/result_DUTDEG/acquired/acquired_dut_filtered.txt', sep='\t')

# %% 
####* Venn Diagram (innate/acquired)####
from matplotlib_venn import venn2
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

ac_dtugenes = set(acquired_fil_dut['gene_id'])
in_dtugenes = set(innate_fil_dut['gene_id'])
ac_degenes = set(acquired_fil_deg['gene_id'])
in_degenes = set(innate_fil_deg['gene_id'])

plt.figure(figsize=(6,6))
sns.set_style("white")
plt.title("Genes with Differentially Used Transcripts", fontsize=14)

vd2 = venn2([in_dtugenes, ac_dtugenes],set_labels=('Innate', 'Acquired'),set_colors=('#F8CF6A','#74B2D4'), alpha=0.8)

for text in vd2.set_labels:  #change label size
    text.set_fontsize(13)
for text in vd2.subset_labels:  #change number size
    text.set_fontsize(16)
    
vd2.get_patch_by_id('11').set_color('#C8C452')
vd2.get_patch_by_id('11').set_alpha(1)

plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/figures/gDUT_in_ac_Venn.pdf", bbox_inches="tight")
plt.show()









# %%
####^^^ step 5. run SUPPA2! with innate / acquired d

# %%
tdata = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/final_202306_discovery_transcript_exp.txt',sep='\t', index_col=0)

# acc = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/suppa2/202306_acquired.txt', index_col=0, sep='\t')
# inn = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/suppa2/202306_innate.txt', index_col=0, sep='\t')

ac_meta = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/suppa2/DUT_acquired_metadata.txt',  sep='\t')
in_meta = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/suppa2/DUT_innate_metadata.txt',  sep='\t')

acc = tdata[list(ac_meta['samples'])]
inn = tdata[list(in_meta['samples'])]

# %%
ac_post_R = list(ac_meta.loc[ac_meta['group']=='post_R','samples'])
ac_pre_R = list(ac_meta.loc[ac_meta['group']=='pre_R','samples'])

in_pre_R = list(in_meta.loc[in_meta['group']=='pre_R','samples'])
in_pre_NR = list(in_meta.loc[in_meta['group']=='pre_NR','samples'])

ac_post_R_df = acc[ac_post_R]
ac_pre_R_df = acc[ac_pre_R]
in_pre_R_df = inn[in_pre_R]
in_pre_NR_df = inn[in_pre_NR] 


# # %%
# ac_post_R_df.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/suppa2/acquired_post_R.txt', sep='\t')
# ac_pre_R_df.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/suppa2/acquired_pre_R.txt', sep='\t')

# in_pre_R_df.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/suppa2/innate_pre_R.txt', sep='\t')
# in_pre_NR_df.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/suppa2/innate_pre_NR.txt', sep='\t')



# %%
# %%
####^^^ step 7. exonic read histogram #####
#er = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/202306_newRNAseq/exonicread.txt')
fileinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.csv')
# %%

interval = fileinfo[['sample_id', 'interval', 'response']]
interval = interval.drop_duplicates()

plt.figure(figsize=(10,6))

sns.set(font = 'Arial')
sns.set_style("whitegrid")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sns.displot(interval, x="interval", binwidth=50, color='darkolivegreen',height=6, aspect=1.5)
plt.axvline(x=180, color='red', linestyle='--', linewidth=1.5)

# plt.xlabel('exonic read (%)')
plt.title("treatment interval", fontsize=13)
# plt.xticks(ticks=range(0, 71, 5))


plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202309_analysis/interval_80.pdf", bbox_inches="tight")
plt.show()
# %%


# %%
####^^^ step 8. validation cohort add mutation + line data #####

group_info = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/clinicaldata.txt', sep="\t", index_col="GID")

#* clinicaldata processing (056 -> 105)
group_info = group_info[["OM/OS", "ORR", "drug", "interval", "BRCA_binary"]]
group_info.columns = ["OM/OS", "group", "drug", "interval", "BRCAmut"]
group_info["drug"] = group_info["drug"].str.replace("Olapairb","Olaparib")
group_info["drug"] = group_info["drug"].str.replace("Olapairib","Olaparib")
group_info["drug"] = group_info["drug"].str.replace("olaparib","Olaparib")
group_info["GID"] = group_info.index.str.replace("F","T").str[:10]
group_info["GID"] = group_info["GID"].str.replace("T","P")
group_info["GID"] = group_info["GID"].str.replace("SV-OV-P056", "SV-OV-P105")
group_info = group_info.dropna()
group_info = group_info.drop_duplicates()
group_info = group_info.set_index("GID")

clinical = group_info
# %%
#clinical.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/validation_clinicalinfo.csv')
# %%

vc = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/validation_clinicalinfo.csv', index_col=0)

plt.figure(figsize=(10,6))

sns.set(font = 'Arial')
sns.set_style("whitegrid")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

lineorder = ['1L','1LM','2L','2LM','3L','3LM','4L','4LM','5L','5LM','6L','6LM','7L','7LM','8L','14L']
vc['Lines'] =  pd.Categorical(vc['Lines'], lineorder)

sns.histplot(vc, x="Lines", color='darkblue', binwidth=1)
plt.xlabel('Lines')
plt.title("validation cohort line info", fontsize=13)

plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/figures/validationcohort_lineinfo.pdf", bbox_inches="tight")
plt.show()

# %%
dc = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/discovery_clinicalinfo.csv', index_col=0)
# %%
plt.figure(figsize=(10,6))

sns.set(font = 'Arial')
sns.set_style("whitegrid")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

lineorder = ['1LM', '2LM', '3L', '3LM', '4L', '4LM', '5LM', '6L', '6LM', '7L', '7LM']
dc['Lines'] =  pd.Categorical(dc['Lines'], lineorder)

sns.histplot(dc, x="Lines", color='darkblue', binwidth=1)
plt.xlabel('Lines')
plt.title("discovery cohort line info", fontsize=13)

plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/figures/discoverycohort_lineinfo.pdf", bbox_inches="tight")
plt.show()



# %%


##########################^^^^^^^^^^^^^^^^ OV subtype change ############

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

helland = pd.read_csv("/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/subtype/Helland.txt", sep=' ', header=None)
verhaak = pd.read_csv("/home/jiye/jiye/copycomparison/gDUTresearch/202308_analysis/subtype/Verhaak.txt", sep=' ', header=None)

cat_to_num_mapping = {'DIF': 1, 'MES': 2, 'IMR': 3, 'PRO': 4}

def cat_to_num(cat):
    return int(cat[1:])


values = list(verhaak.iloc[0,:])
#values = [cat_to_num(val) for val in values]
values = [cat_to_num_mapping[val] for val in values]

post_treatment = values[::2]
pre_treatment = values[1::2]

# Check if lengths are consistent
if len(post_treatment) != len(pre_treatment):
    print("Warning: unequal number of pre and post-treatment samples.")

# Create a DataFrame
df = pd.DataFrame({
    'sample': range(len(pre_treatment)),
    'pre': pre_treatment,
    'post': post_treatment
})
df_melted = df.melt(id_vars=['sample'], value_vars=['pre', 'post'])

# Create the plot
plt.figure(figsize=(10,6))

sns.set(font = 'Arial')
sns.set_style("whitegrid")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.figure(figsize=(10, 6))
g = sns.pointplot(data=df_melted, x='variable', y='value', hue='sample', join=True, palette='crest')
plt.legend([], [], frameon=False)
plt.title('Verhaak Subtype')
plt.xlabel('Treatment')
plt.ylabel('OC Subtype')
#plt.yticks([1, 2, 3,4,5], ['C1', 'C2', 'C3', 'C4', 'C5'])
plt.yticks([1, 2, 3,4], ['DIF','MES','IMR','PRO'])

plt.show()


# %%
