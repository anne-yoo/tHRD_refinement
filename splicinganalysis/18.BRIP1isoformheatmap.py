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
whole_tu = pd.DataFrame()

major_tu = pd.read_csv("/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/pre_post_TU/final_data/230105_major_prop.txt", sep='\t', index_col=0) #major TU
minor_tu = pd.read_csv("/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/pre_post_TU/final_data/230105_minor_prop.txt", sep='\t', index_col=0) #minor TU

filelist = [major_tu,minor_tu]

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

    #### resistance-related gene list
    genelist = ['BRIP1']

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
sns.boxplot(data=cl_final, x="gene_ENST", y="TU", hue="pre/post", showfliers = False, palette={'pre':'#EDDFDF', 'post':'#FF4949'})
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
prepost = prepost[['sample','TU_pre','TU_post','gene_ENST']]
prepost = prepost.groupby(['sample','gene_ENST']).mean()

for i in range(prepost.shape[0]):
    if prepost.iloc[i,0]>prepost.iloc[i,1]:
        prepost.iloc[i,0] = 1
        prepost.iloc[i,1] = 0
    elif prepost.iloc[i,0]==prepost.iloc[i,1]:
        prepost.iloc[i,0] = 2
        prepost.iloc[i,1] = 2
    else:
        prepost.iloc[i,0] = 0
        prepost.iloc[i,1] = 1

prepost = prepost[['TU_post']]
prepost_df = prepost.reset_index()
prepost_pivot = prepost_df.pivot(index='gene_ENST', columns='sample')['TU_post']

prepost_pivot.index.name = None

#%%
majordf = prepost_pivot.loc[['ENST00000259008.2-BRIP1']]
minordf = prepost_pivot.loc[['ENST00000577598.1-BRIP1','MSTRG.49834.288-BRIP1','MSTRG.49834.293-BRIP1']]
minordf.index = minordf.index.str.split('-',1).str[0]
majordf.index = majordf.index.str.split('-',1).str[0]

#%%
##^minor
sns.set(rc = {'figure.figsize':(7,3)})
sns.set_style('whitegrid')
cmap=['#2D2DE9','#EFEFEF','#CDCDCD']
ax=sns.heatmap(minordf, cmap=cmap, cbar=False,  linewidth=0.005, annot_kws={'rotation':270})
ax.set_xlabel('')
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 9)
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 13)
ax.set_xticklabels(ax.get_xticklabels(),rotation=30)

from matplotlib.patches import Patch
legend_handles = [Patch(color=cmap[1], label='pre < post'),  
                Patch(color=cmap[0], label='pre > post'),]  
plt.legend(handles=legend_handles, ncol=2, bbox_to_anchor=[0.5, 1.02], loc='lower center', fontsize=8, handlelength=.8)

plt.tight_layout()
plt.show()
# %%
#%%
##^major
sns.set(rc = {'figure.figsize':(7,1)})
sns.set_style('whitegrid')
cmap=['#EFEFEF','#EFEFEF','#E43D27']
ax=sns.heatmap(majordf, cmap=cmap, cbar=False,  linewidth=0.005, annot_kws={'rotation':270})
ax.set_xlabel('')
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 9)
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 14)
ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
ax.set_yticklabels(ax.get_yticklabels(),rotation=0)

from matplotlib.patches import Patch
legend_handles = [Patch(color=cmap[2], label='pre < post'),  
                Patch(color=cmap[0], label='pre > post'),]  
plt.legend(handles=legend_handles, ncol=2, bbox_to_anchor=[0.5, 1.02], loc='lower center', fontsize=8, handlelength=.8)

plt.tight_layout()
plt.show()
# %%
