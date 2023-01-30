#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.stats import pearsonr
import random
from scipy.stats import ttest_ind
from scipy.stats import ranksums

# %%
####*################## 1. TU pre vs. post boxplots ###############################

#^ non-matched RES vs. NR BRIP1 minor TU 
major_tu = pd.read_csv("/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/pre_post_TU/final_data/230105_major_prop.txt", sep='\t', index_col=0) #major TU
minor_tu = pd.read_csv("/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/pre_post_TU/final_data/230105_minor_prop.txt", sep='\t', index_col=0) #major TU

filelist = [major_tu, minor_tu]

whole_tu = pd.DataFrame()

for file in filelist:
    tu = file
    samples = tu.columns.to_list()
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
    tu.columns = samples


    tu = tu.drop(["SV-OV-P080-atD","SV-OV-P250-atD","SV-OV-P055-atD",\
                "SV-OV-P143-atD","SV-OV-P137-atD","SV-OV-P134-atD",\
                "SV-OV-P174-bfD","SV-OV-P164-atD"], axis=1)

    samples = tu.columns

    tu['gene_id'] = tu.index.str.split("-",1).str[1]
    tu['gene_ENST'] = tu.index

    briptu = tu[(tu['gene_ENST']=='MSTRG.49834.283-BRIP1') | (tu['gene_ENST']=='MSTRG.49834.288-BRIP1') | (tu['gene_ENST']=='ENST00000577598.1-BRIP1')| (tu['gene_ENST']=='ENST00000259008.2-BRIP1')]
    # briptu = tu[(tu['gene_ENST']=='MSTRG.49834.283-BRIP1')  | (tu['gene_ENST']=='ENST00000577598.1-BRIP1')| (tu['gene_ENST']=='ENST00000259008.2-BRIP1')] # exclude 288


    pre_list = [x for x in samples if 'bfD' in x ]
    post_list = [x for x in samples if 'atD' in x ]

    pre_tu = briptu.filter(pre_list)
    post_tu = briptu.filter(post_list)

    post_stacked = post_tu.stack().reset_index()
    pre_stacked = pre_tu.stack().reset_index()

    post_stacked.columns = ['gene_ENST','sample','transcript usage']
    pre_stacked.columns = ['gene_ENST','sample','transcript usage']

    post_stacked['pre/post'] = 'post'
    pre_stacked['pre/post'] = 'pre'


    final_stacked = pd.concat([pre_stacked, post_stacked])
    final_stacked['sample'] = final_stacked['sample'].str[:10]

    whole_tu = pd.concat([whole_tu, final_stacked])


#%%
#^ merge clinical data
group_info = pd.read_csv('/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/clinical_info_gHRD/processed_info_with_originalID.txt', sep="\t", index_col="GID")

group_info = group_info[["OM/OS", "ORR", "drug", "interval"]]
group_info.columns = ["OM/OS", "group", "drug", "interval"]
group_info["drug"] = group_info["drug"].str.replace("Olapairb","Olaparib")
group_info["drug"] = group_info["drug"].str.replace("Olapairib","Olaparib")
group_info["drug"] = group_info["drug"].str.replace("olaparib","Olaparib")
group_info = group_info.dropna()
group_info = group_info.drop("drug", axis=1)
group_info["GID"] = group_info.index.str.replace("F","T").str[:10]
group_info["GID"] = group_info["GID"].str.replace("T","P")
group_info = group_info.drop_duplicates()
group_info = group_info.set_index("GID")

clinical = group_info
clinical['sample'] = clinical.index

cl_final = pd.merge(whole_tu, clinical, how='inner', on='sample')
cl_final['group'] = cl_final['group'].replace([1], 'Responder')
cl_final['group'] = cl_final['group'].replace([0], 'Non-Responder') 



#^ mungtaengi box plot
cl_final['gene_ENST'] = cl_final['gene_ENST'].str[:-6]
# orders = ['ENST00000259008.2', 'ENST00000577598.1', 'MSTRG.49834.283','MSTRG.49834.288']
orders = ['ENST00000259008.2', 'ENST00000577598.1', 'MSTRG.49834.283']

sns.set(rc = {'figure.figsize':(8,4)})
sns.set_style('whitegrid')
sns.boxplot(data=cl_final, x="gene_ENST", y="transcript usage", hue="pre/post", showfliers = True, order=orders)
plt.suptitle('<pre vs. post BRIP1 TU>', size=11)
plt.xticks(rotation=0, size=9)
plt.show()

#^ mungtaengi box plot: only responders
sns.set(rc = {'figure.figsize':(8,4)})
sns.set_style('whitegrid')
sns.boxplot(data=cl_final[cl_final['group']=='Responder'], x="gene_ENST", y="transcript usage", hue="pre/post", showfliers = True, order=orders)
plt.suptitle('<pre vs. post BRIP1 TU: Responder>', size=11)
plt.xticks(rotation=0, size=9)
plt.show()

#^ mungtaengi box plot: only non-responders
sns.set(rc = {'figure.figsize':(8,4)})
sns.set_style('whitegrid')
sns.boxplot(data=cl_final[cl_final['group']=='Non-Responder'], x="gene_ENST", y="transcript usage", hue="pre/post", showfliers = True, order=orders)
plt.suptitle('<pre vs. post BRIP1 TU: Non-Responder>', size=11)
plt.xticks(rotation=0, size=9)
plt.show()



#%%
##*#################### 2. TU pre vs. post matched pointplot ###############################
cat_minor = cl_final[cl_final['gene_ENST']=='MSTRG.49834.288']
cat_major = cl_final[cl_final['gene_ENST']=='ENST00000259008.2']

sns.set(rc = {'figure.figsize':(6,4)})
sns.set_style('whitegrid')
sns.catplot(x="pre/post", y="transcript usage", hue="sample", kind="point", data=cat_major, palette='tab10', height = 3.6, aspect = 1.4)
plt.suptitle('<Trancript Usage: ENST00000259008.2>', y=1.03, size=11)
plt.show()

#%%
##*#################### 3. TU pre vs. post binary heatmap ###############################
##^ only responders, sample-matched
cl_r = cat_minor[cat_minor['group']=='Responder'] #only responders
pre = cl_r[cl_r['pre/post']=='pre']
post = cl_r[cl_r['pre/post']=='post']
pre = pre[['sample','transcript usage']]
post = post[['sample','transcript usage']]

prepost = pd.merge(pre,post,how='inner',on='sample')
prepost.columns = ['sample','TU_pre','TU_post']

##^ heatmap
for i in range(prepost.shape[0]):
    if prepost.iloc[i,1]>prepost.iloc[i,2]:
        prepost.iloc[i,1] = 1
        prepost.iloc[i,2] = 0
    else:
        prepost.iloc[i,1] = 0
        prepost.iloc[i,2] = 1


prepost = prepost.set_index('sample')
prepost.index.name = None

sns.set(rc = {'figure.figsize':(4,5)})
sns.set_style('whitegrid')
sns.heatmap(prepost,cmap=['#EEDAD5','#F25027'], cbar=False,  linewidth=0.005)
plt.title("MSTRG.49834.288", fontsize =11)
# plt.title("ENST00000259008.2", fontsize =11)
plt.show()
#%%














# %%
##*#################### 4. TPM pre vs. post boxplots ###############################

#^ non-matched RES vs. NR BRIP1 minor TU 
tpm = pd.read_csv('/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/pre_post_TU/final_data/230106_new_final_pre_post_samples_TU_input.txt', sep='\t', index_col=0)

# remove transcript-gene unmatched transcripts
tpm = tpm[tpm['target_gene']!= '-']

# sample name change
sample = tpm.columns.tolist()# %%
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

tpm.columns = samples
try:
    tpm = tpm.drop(["SV-OV-P080-atD","SV-OV-P250-atD","SV-OV-P055-atD",\
            "SV-OV-P143-atD","SV-OV-P137-atD","SV-OV-P134-atD",\
            "SV-OV-P174-bfD","SV-OV-P164-atD"], axis=1)
except:
    

    pass


tpm = tpm.drop(['target_gene'],axis=1)
tpm['gene_ENST'] = tpm.index

briptpm = tpm[(tpm['gene_ENST']=='MSTRG.49834.283-BRIP1') | (tpm['gene_ENST']=='MSTRG.49834.288-BRIP1') | (tpm['gene_ENST']=='ENST00000577598.1-BRIP1')| (tpm['gene_ENST']=='ENST00000259008.2-BRIP1')]
# briptpm = tpm[(tpm['gene_ENST']=='MSTRG.49834.283-BRIP1')  | (tpm['gene_ENST']=='ENST00000577598.1-BRIP1')| (tpm['gene_ENST']=='ENST00000259008.2-BRIP1')] # exclude 288

samples = briptpm.columns
pre_list = [x for x in samples if 'bfD' in x ]
post_list = [x for x in samples if 'atD' in x ]

pre_tpm = briptpm.filter(pre_list)
post_tpm = briptpm.filter(post_list)

post_stacked = post_tpm.stack().reset_index()
pre_stacked = pre_tpm.stack().reset_index()

post_stacked.columns = ['gene_ENST','sample','TPM']
pre_stacked.columns = ['gene_ENST','sample','TPM']

post_stacked['pre/post'] = 'post'
pre_stacked['pre/post'] = 'pre'


final_stacked = pd.concat([pre_stacked, post_stacked])
final_stacked['sample'] = final_stacked['sample'].str[:10]

final_stacked['TPM'] = np.log2(final_stacked['TPM']+1)

final_stacked.columns = ['gene_ENST','sample','log2(TPM+1)','pre/post']


#^ merge clinical data
group_info = pd.read_csv('/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/clinical_info_gHRD/processed_info_with_originalID.txt', sep="\t", index_col="GID")

group_info = group_info[["OM/OS", "ORR", "drug", "interval"]]
group_info.columns = ["OM/OS", "group", "drug", "interval"]
group_info["drug"] = group_info["drug"].str.replace("Olapairb","Olaparib")
group_info["drug"] = group_info["drug"].str.replace("Olapairib","Olaparib")
group_info["drug"] = group_info["drug"].str.replace("olaparib","Olaparib")
group_info = group_info.dropna()
group_info = group_info.drop("drug", axis=1)
group_info["GID"] = group_info.index.str.replace("F","T").str[:10]
group_info["GID"] = group_info["GID"].str.replace("T","P")
group_info = group_info.drop_duplicates()
group_info = group_info.set_index("GID")

clinical = group_info
clinical['sample'] = clinical.index

#%%
cl_final = pd.merge(final_stacked, clinical, how='inner', on='sample')
cl_final['group'] = cl_final['group'].replace([1], 'Responder')
cl_final['group'] = cl_final['group'].replace([0], 'Non-Responder')



#^ mungtaengi box plot
cl_final['gene_ENST'] = cl_final['gene_ENST'].str[:-6]
orders = ['ENST00000259008.2', 'ENST00000577598.1', 'MSTRG.49834.283','MSTRG.49834.288']
# orders = ['ENST00000259008.2', 'ENST00000577598.1', 'MSTRG.49834.283']

sns.set(rc = {'figure.figsize':(8,4)})
sns.set_style('whitegrid')
sns.boxplot(data=cl_final, x="gene_ENST", y="log2(TPM+1)", hue="pre/post", showfliers = False, order=orders)
plt.suptitle('<pre vs. post BRIP1 TPM>', size=11)
plt.xticks(rotation=0, size=9)
plt.show()

#^ mungtaengi box plot: only responders
sns.set(rc = {'figure.figsize':(8,4)})
sns.set_style('whitegrid')
sns.boxplot(data=cl_final[cl_final['group']=='Responder'], x="gene_ENST", y="log2(TPM+1)", hue="pre/post", showfliers = False, order=orders)
plt.suptitle('<pre vs. post BRIP1 TPM: Responder>', size=11)
plt.xticks(rotation=0, size=9)
plt.show()

#^ mungtaengi box plot: only non-responders
sns.set(rc = {'figure.figsize':(8,4)})
sns.set_style('whitegrid')
sns.boxplot(data=cl_final[cl_final['group']=='Non-Responder'], x="gene_ENST", y="log2(TPM+1)", hue="pre/post", showfliers = False, order=orders)
plt.suptitle('<pre vs. post BRIP1 TPM: Non-Responder>', size=11)
plt.xticks(rotation=0, size=9)
plt.show()


#%%
##*#################### 5. TPM pre vs. post matched pointplot ###############################
cat_minor = cl_final[cl_final['gene_ENST']=='MSTRG.49834.288']
cat_major = cl_final[cl_final['gene_ENST']=='ENST00000259008.2']

sns.set(rc = {'figure.figsize':(6,4)})
sns.set_style('whitegrid')
sns.catplot(x="pre/post", y="log2(TPM+1)", hue="sample", kind="point", data=cat_minor, palette='tab10', height = 3.6, aspect = 1.4)
plt.suptitle('<TPM: MSTRG.49834.288>', y=1.03, size=11)
plt.show()


# %%
##*#################### 6. TPM pre vs. post binary heatmap ###############################
##^ only responders, sample-matched
cl_r = cat_minor[cat_minor['group']=='Responder'] #only responders
pre = cl_r[cl_r['pre/post']=='pre']
post = cl_r[cl_r['pre/post']=='post']
pre = pre[['sample','log2(TPM+1)']]
post = post[['sample','log2(TPM+1)']]

prepost = pd.merge(pre,post,how='inner',on='sample')
prepost.columns = ['sample','log2(TPM+1)_pre','log2(TPM+1)_post']

# %%
##^ heatmap
for i in range(prepost.shape[0]):
    if prepost.iloc[i,1]>prepost.iloc[i,2]:
        prepost.iloc[i,1] = 1
        prepost.iloc[i,2] = 0
    else:
        prepost.iloc[i,1] = 0
        prepost.iloc[i,2] = 1


prepost = prepost.set_index('sample')
prepost.index.name = None
#%%
sns.set(rc = {'figure.figsize':(4,5)})
sns.set_style('whitegrid')
sns.heatmap(prepost,cmap=['#EEDAD5','#F25027'], cbar=False,  linewidth=0.005)
plt.title("MSTRG.49834.288", fontsize =11)
# plt.title("ENST00000259008.2", fontsize =11)
plt.show()

# %%
