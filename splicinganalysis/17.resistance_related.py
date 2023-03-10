#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.stats import pearsonr
import gseapy as gp

# %%
#######* resistance-related gene TU analysis #######
tu = pd.read_csv("/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/pre_post_TU/final_data/230105_major_prop.txt", sep='\t', index_col=0) #major TU
whole_tu = pd.DataFrame()

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

#### resistance-related gene list
# genelist = ['BRIP1','CHEK1','CHEK2','TP53','RAD17','ATM','ATR','PTEN','MLH1','RAD51','WEE1','TP53BP1','MAD2L2','SMARCAL1','ERCC1','ERCC4','FANCD2']
genelist = ['BRIP1','CHEK1','CHEK2','TP53','RAD17','ATM','ATR','PTEN','MLH1','RAD51','WEE1','TP53BP1','MAD2L2','SMARCAL1','FANCD2']

####

briptu = tu[tu['gene_id'].isin(genelist)]

pre_list = [x for x in samples if 'bfD' in x ]
post_list = [x for x in samples if 'atD' in x ]

pre_tu = briptu.filter(pre_list)
post_tu = briptu.filter(post_list)

post_stacked = post_tu.stack().reset_index()
pre_stacked = pre_tu.stack().reset_index()

post_stacked.columns = ['gene_ENST','sample','TU']
pre_stacked.columns = ['gene_ENST','sample','TU']


post_stacked['pre/post'] = 'post'
pre_stacked['pre/post'] = 'pre'

post_stacked['gene id'] = post_stacked['gene_ENST'].str.split("-",1).str[1]
pre_stacked['gene id'] = pre_stacked['gene_ENST'].str.split("-",1).str[1]

final_stacked = pd.concat([pre_stacked, post_stacked])
final_stacked['sample'] = final_stacked['sample'].str[:10]

whole_tu = pd.concat([whole_tu, final_stacked])

major_list = list(set(whole_tu['gene_ENST']))

#%%
##^ merge clinical data: only responders
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

cl_final = cl_final[cl_final['group']=='Responder']

#%%
##^ boxplots
sns.set(rc = {'figure.figsize':(12,8)})
sns.set_style('whitegrid')
sns.boxplot(data=cl_final, x="gene id", y="TU", hue="pre/post", showfliers = False, order=genelist, palette={'pre':'#4998FF', 'post':'#FF4949'})
plt.title('<pre vs. post major TU>', size=11)
plt.xticks(rotation=0, size=9)
plt.ylabel('Transcript Usage')

plt.show()

#%%
# %%
##*#################### TU pre vs. post binary heatmap ###############################
##^ only responders, sample-matched
pre = cl_final[cl_final['pre/post']=='pre']
post = cl_final[cl_final['pre/post']=='post']
pre = pre[['sample','TU','gene_ENST', 'gene id']]
post = post[['sample','TU','gene_ENST', 'gene id']]

#%%
prepost = pd.merge(pre,post,how='inner',on=['sample','gene_ENST','gene id'])
prepost.columns = ['sample','TU_pre','gene_ENST','gene id','TU_post']

##^ heatmap
prepost = prepost[['sample','TU_pre','TU_post','gene id']]
prepost = prepost.groupby(['sample','gene id']).mean()

for i in range(prepost.shape[0]):
    if prepost.iloc[i,0]>prepost.iloc[i,1]:
        prepost.iloc[i,0] = 1
        prepost.iloc[i,1] = 0
    else:
        prepost.iloc[i,0] = 0
        prepost.iloc[i,1] = 1

prepost = prepost[['TU_post']]
prepost_df = prepost.reset_index()
prepost_pivot = prepost_df.pivot(index='gene id', columns='sample')['TU_post']

prepost_pivot.index.name = None

sns.set(rc = {'figure.figsize':(5,8)})
sns.set_style('whitegrid')
cmap=['#EEDAD5','#F25027']
ax=sns.heatmap(prepost_pivot, cmap=['#EEDAD5','#F25027'], cbar=False,  linewidth=0.005, annot_kws={'rotation':180})
ax.set_xlabel('')
# ax.set_xticklabels(ax.get_xticklabels(),rotation=30)

from matplotlib.patches import Patch
legend_handles = [Patch(color=cmap[True], label='pre < post'),  
                Patch(color=cmap[False], label='pre > post')]  
plt.legend(handles=legend_handles, ncol=2, bbox_to_anchor=[0.5, 1.02], loc='lower center', fontsize=8, handlelength=.8)
plt.title("major transcripts TU change", fontsize =11, pad=40)

plt.tight_layout()
plt.show()

























#%%
#######* resistance-related gene TPM analysis #######
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
samples = tpm.columns
tpm['gene_id'] = tpm.index.str.split("-",1).str[1]
tpm['transcript_id'] = tpm.index

#### resistance-related gene list
# genelist = ['BRIP1','CHEK1','CHEK2','TP53','RAD17','ATM','ATR','PTEN','MLH1','RAD51','WEE1','TP53BP1','MAD2L2','SMARCAL1','ERCC1','ERCC4','FANCD2']
genelist = ['BRIP1','CHEK1','CHEK2','TP53','RAD17','ATM','ATR','PTEN','MLH1','RAD51','WEE1','TP53BP1','MAD2L2','SMARCAL1','FANCD2']
####

briptpm = tpm[tpm['transcript_id'].isin(major_list)]

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

post_stacked['gene id'] = post_stacked['gene_ENST'].str.split("-",1).str[1]
pre_stacked['gene id'] = pre_stacked['gene_ENST'].str.split("-",1).str[1]

final_stacked = pd.concat([pre_stacked, post_stacked])
final_stacked['sample'] = final_stacked['sample'].str[:10]

whole_tpm = final_stacked

whole_tpm['TPM'] = np.log2(whole_tpm['TPM']+1)

#%%
##^ merge clinical data: only responders
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

cl_final = pd.merge(whole_tpm, clinical, how='inner', on='sample')
cl_final['group'] = cl_final['group'].replace([1], 'Responder')
cl_final['group'] = cl_final['group'].replace([0], 'Non-Responder') 

cl_final = cl_final[cl_final['group']=='Responder']

#%%

##^ boxplots
sns.set(rc = {'figure.figsize':(12,8)})
sns.set_style('whitegrid')
sns.boxplot(data=cl_final, x="gene id", y="TPM", hue="pre/post", showfliers = False, order = genelist, palette={'pre':'#4998FF', 'post':'#FF4949'})
plt.title('<pre vs. post major transcript TPM>', size=11)
plt.xticks(rotation=0, size=9)
plt.ylabel('log2(TPM+1)')
plt.show()
# %%







# %%
##*#################### TPM pre vs. post binary heatmap ###############################
##^ only responders, sample-matched
pre = cl_final[cl_final['pre/post']=='pre']
post = cl_final[cl_final['pre/post']=='post']
pre = pre[['sample','TPM','gene_ENST', 'gene id']]
post = post[['sample','TPM','gene_ENST', 'gene id']]

#%%
prepost = pd.merge(pre,post,how='inner',on=['sample','gene_ENST','gene id'])
prepost.columns = ['sample','TPM_pre','gene_ENST','gene id','TPM_post']

##^ heatmap
prepost = prepost[['sample','TPM_pre','TPM_post','gene id']]
prepost = prepost.groupby(['sample','gene id']).mean()

for i in range(prepost.shape[0]):
    if prepost.iloc[i,0]>prepost.iloc[i,1]:
        prepost.iloc[i,0] = 1
        prepost.iloc[i,1] = 0
    else:
        prepost.iloc[i,0] = 0
        prepost.iloc[i,1] = 1

prepost = prepost[['TPM_post']]
prepost_df = prepost.reset_index()
prepost_pivot = prepost_df.pivot(index='gene id', columns='sample')['TPM_post']

prepost_pivot.index.name = None

sns.set(rc = {'figure.figsize':(5,8)})
sns.set_style('whitegrid')
cmap=['#EEDAD5','#F25027']
ax=sns.heatmap(prepost_pivot, cmap=['#EEDAD5','#F25027'], cbar=False,  linewidth=0.005, annot_kws={'rotation':180})
ax.set_xlabel('')
# ax.set_xticklabels(ax.get_xticklabels(),rotation=30)

from matplotlib.patches import Patch
legend_handles = [Patch(color=cmap[True], label='pre < post'),  
                Patch(color=cmap[False], label='pre > post')]  
plt.legend(handles=legend_handles, ncol=2, bbox_to_anchor=[0.5, 1.02], loc='lower center', fontsize=8, handlelength=.8)
plt.title("major transcripts TPM change", fontsize =11, pad=40)

plt.tight_layout()
plt.show()
# %%
