
#%%
#! this code is.....
## 20230525
"""
1) R pre vs. R post (acquired/innate): whole gene DEG + clustermap
2) R pre vs. R post (acquired/innate): whole gene DTU + clustermap
3) R pre vs. R post (acquired/innate): DEG vs. DSG Venn Diagram
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
import matplotlib.cm as cm
from matplotlib.pyplot import gcf

# %%
#^ 1) R pre vs. R post (acquired): whole gene DEG + clustermap
##** make dataset for DESeq2 input

countdata = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/DESeq2/countdata.csv', sep = ",", index_col=0)

# clinical data
group_info = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/clinicaldata.txt', sep="\t", index_col="GID")


# clinicaldata processing (056 -> 105)
group_info = group_info[["OM/OS", "ORR", "drug", "interval"]]
group_info.columns = ["OM/OS", "group", "drug", "interval"]
group_info["drug"] = group_info["drug"].str.replace("Olapairb","Olaparib")
group_info["drug"] = group_info["drug"].str.replace("Olapairib","Olaparib")
group_info["drug"] = group_info["drug"].str.replace("olaparib","Olaparib")
group_info["GID"] = group_info.index.str.replace("F","T").str[:10]
group_info["GID"] = group_info["GID"].str.replace("T","P")
group_info["GID"] = group_info["GID"].str.replace("SV-OV-P056", "SV-OV-P105")
group_info = group_info.dropna()
group_info = group_info.drop("drug", axis=1)
group_info = group_info.drop_duplicates()
group_info = group_info.set_index("GID")

clinical = group_info

mTU = countdata

wholesample = pd.DataFrame(mTU.columns, columns=['sample'])
##res
res = clinical[clinical['group']==1]
mtsample = pd.DataFrame(res.index)
wholesample['GID'] = wholesample['sample'].str[:-4]
onlymt = pd.merge(wholesample, mtsample, on='GID', how='inner')
mtsamplelist = onlymt['sample'].tolist()
major_res = mTU[mtsamplelist]
##nonres
nonres = clinical[clinical['group']==0]
mtsample = pd.DataFrame(nonres.index)
wholesample['GID'] = wholesample['sample'].str[:-4]
onlymt = pd.merge(wholesample, mtsample, on='GID', how='inner')
mtsamplelist = onlymt['sample'].tolist()
major_nonres = mTU[mtsamplelist]

#then split into R_pre, R_post, NR_pre, NR_post
major_R_pre = major_res.loc[:, major_res.columns.str.contains('bfD')]
major_info_1 = pd.DataFrame(major_R_pre.columns)
major_info_1['group'] = 'R_pre'

major_R_post = major_res.loc[:, major_res.columns.str.contains('atD')]
major_info_2 = pd.DataFrame(major_R_post.columns)
major_info_2['group'] = 'R_post'

major_NR_pre = major_nonres.loc[:, major_nonres.columns.str.contains('bfD')]
major_info_3 = pd.DataFrame(major_NR_pre.columns)
major_info_3['group'] = 'NR_pre'

major_NR_post = major_nonres.loc[:, major_nonres.columns.str.contains('atD')]
major_info_4 = pd.DataFrame(major_NR_post.columns)
major_info_4['group'] = 'NR_post'

#concat 4 data
final_major = pd.concat([major_R_pre,major_R_post,major_NR_pre,major_NR_post],axis=1)
final_major_info = pd.concat([major_info_1,major_info_2,major_info_3,major_info_4])
final_major_info.columns = ['samples','group']


# filter out only R pre vs. R post (acquired)
inputmeta = final_major_info[ (final_major_info['group'] == 'R_pre') | (final_major_info['group'] == 'R_post') ]
inputsamples = list(inputmeta['samples'])
inputcount = final_major[inputsamples]

# save data
#inputmeta.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/DESeq2/resistance/acquired_metadata.csv',index=False)
#inputcount.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/DESeq2/resistance/acquired_countdata.csv',index=True)

# filter out only R pre vs. NR pre (innate)
inputmeta2 = final_major_info[ (final_major_info['group'] == 'R_pre') | (final_major_info['group'] == 'NR_pre') ]
inputsamples2 = list(inputmeta2['samples'])
inputcount2 = final_major[inputsamples2]

#save data
#inputmeta2.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/DESeq2/resistance/innate_metadata.csv',index=False)
#inputcount2.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/DESeq2/resistance/innate_countdata.csv',index=True)


# %%
##** acquired/innate clustermap
ac_deg = pd.read_csv('//home/jiye/jiye/copycomparison/gDUTresearch/DESeq2/resistance/acquired_DEG_result.csv', sep = ",", index_col=0)
in_deg = pd.read_csv('//home/jiye/jiye/copycomparison/gDUTresearch/DESeq2/resistance/innate_DEG_result.csv', sep = ",", index_col=0)

#########^#####
#deg = ac_deg ##acquired
deg = in_deg ##innate
#########^#####


#########^#####
#deg = deg[(deg['pvalue']<0.05) & (deg['padj']<0.01)]
deg = deg[(deg['pvalue']<0.05)]
#########^#####

degenes = list(deg.index)


#########^#####
#forplot = inputcount.loc[degenes,:] ##acquired
forplot = inputcount2.loc[degenes,:] ##innate
#########^#####

## z-score normalization
forplot_z =  forplot.apply(lambda x: (x-x.mean())/x.std(), axis = 1)
forplot_z = forplot_z.dropna(axis=0)

#draw clustermap
#########^#####
#colors_dict = {"R_pre":"#FF9595", "R_post":"#A54A4A"} ##acquired
colors_dict = { "R_pre":"#FF9595", "NR_pre":"#7CB5FF"} ##innate
#########^#####


#########^#####
#col_colors = inputmeta['group'].map(colors_dict) ##acquired
col_colors = inputmeta2['group'].map(colors_dict) ##innate
#########^#####

g = sns.clustermap(forplot_z, col_colors=[col_colors], cmap="RdBu_r" , figsize=(7,9),vmin=-2,vmax=2, center=0,
                method='ward', #single, complete, average, weighted, ward
                metric='euclidean', #cityblock, minkowski, euclidean, cosine, correlation, hamming, jaccard, chebyshev, canberra, dice, rogerstanimoto, russellrao, sokalsneath
                linewidths=0, xticklabels=False, yticklabels=False, col_cluster=True,row_cluster=True)

g.fig.subplots_adjust(right=0.7)
g.ax_cbar.set_position((0.73, .2, .03, .4))

for group in inputmeta2['group'].unique(): #######inputmeta2: innate
    g.ax_col_dendrogram.bar(0, 0, color=colors_dict[group],
                            label=group, linewidth=0)

l1 = g.ax_col_dendrogram.legend(title='sample type', loc="center", ncol=2, bbox_to_anchor=(0.8, 0.96), bbox_transform=gcf().transFigure) #bbox_to_anchor=(0.42, 0.88),

ax = g.ax_heatmap
# ax.set_xlabel("Samples")
ax.set_ylabel("")

#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/figures/innate_deg_clustermap_pval.pdf", bbox_inches="tight")

plt.show()
# %%
##* acquired/innate DEG genelist
#########^#####
#ac_deg_data = ac_deg[((ac_deg['pvalue']<0.05) & (ac_deg['padj']<0.01))]
#in_deg_data = in_deg[((in_deg['pvalue']<0.05) & (in_deg['padj']<0.01))]

ac_deg_data = ac_deg[(ac_deg['pvalue']<0.05)]
in_deg_data = in_deg[(in_deg['pvalue']<0.05)]
#########^#####

ac_degenes = list(ac_deg_data.index)
in_degenes = list(in_deg_data.index)

# %%












































#%%
#^ 2) R pre vs. R post (acquired/innate): whole gene DTU + clustermap
##* acquired satuRn input generation
#* data download
texp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/discovery_transcript_exp.txt',  sep='\t', index_col=0)
group_info = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/clinicaldata.txt', sep="\t", index_col="GID")

#* clinicaldata processing (056 -> 105)
group_info = group_info[["OM/OS", "ORR", "drug", "interval"]]
group_info.columns = ["OM/OS", "group", "drug", "interval"]
group_info["drug"] = group_info["drug"].str.replace("Olapairb","Olaparib")
group_info["drug"] = group_info["drug"].str.replace("Olapairib","Olaparib")
group_info["drug"] = group_info["drug"].str.replace("olaparib","Olaparib")
group_info["GID"] = group_info.index.str.replace("F","T").str[:10]
group_info["GID"] = group_info["GID"].str.replace("T","P")
group_info["GID"] = group_info["GID"].str.replace("SV-OV-P056", "SV-OV-P105")
group_info = group_info.dropna()
group_info = group_info.drop("drug", axis=1)
group_info = group_info.drop_duplicates()
group_info = group_info.set_index("GID")

clinical = group_info

#* remove transcript-gene unmatched transcripts
texp['target_gene'] = texp.index.str.split("-",1).str[1]
TUdata = texp[texp['target_gene']!= '-']

#* sample name change
sample = TUdata.columns.tolist()# %%
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

TUdata.columns = samples
try:
    TUdata = TUdata.drop(["SV-OV-P080-atD","SV-OV-P250-atD","SV-OV-P055-atD",\
            "SV-OV-P143-atD","SV-OV-P137-atD","SV-OV-P134-atD",\
            "SV-OV-P174-bfD","SV-OV-P164-atD"], axis=1)
except:
    

    pass

#* 60% 이상 filtering
TUdata = TUdata.drop(['target_gene'],axis=1)
TUdata = TUdata[(TUdata>0).astype(int).sum(axis=1) > TUdata.shape[1]*0.6]

#* sample filtering: R / NR

wholesample = pd.DataFrame(TUdata.columns, columns=['sample'])

###&&&&################################################
res_int = clinical[clinical['group']==1]
###&&&&#################################################

mtsample = pd.DataFrame(res_int.index)
wholesample['GID'] = wholesample['sample'].str[:-4]

onlymt = pd.merge(wholesample, mtsample, on='GID', how='inner')

mtsamplelist = onlymt['sample'].tolist()

TUdata = TUdata[mtsamplelist]


#* gene filtering: stable / variable (only with p-value)

tu_data = TUdata

#* sample info = metadata
sampleinfo = pd.DataFrame(tu_data.columns, columns=['samples'])
sampleinfo['group'] = 'R_pre'
sampleinfo.loc[sampleinfo['samples'].str.contains("atD"),'group'] = "R_post"

sampleinfo1 = sampleinfo

#* geneinfo
geneinfo = pd.DataFrame(columns=['transcript','gene'])
geneinfo["gene"] = tu_data.index.str.split("-",1).str[1]
geneinfo["transcript"] = tu_data.index
geneinfo = geneinfo.set_index("transcript", drop=False)

tu_data1 = tu_data
# %%
#* input data export
path = '/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/input/discovery/resistance/'
#sampleinfo.to_csv(path+'acquired_metadata.csv',index=False)
#tu_data.to_csv(path +'acquired_inputdata.csv',index=True)
#geneinfo.to_csv(path+'acquired_geneinfo.csv',index=True)












# %%
#^ 2) R pre vs. R post (acquired/innate): whole gene DTU + clustermap
##* innate!!!!! satuRn input generation
#* data download
texp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/discovery_transcript_exp.txt',  sep='\t', index_col=0)
group_info = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/clinicaldata.txt', sep="\t", index_col="GID")

#* clinicaldata processing (056 -> 105)
group_info = group_info[["OM/OS", "ORR", "drug", "interval"]]
group_info.columns = ["OM/OS", "group", "drug", "interval"]
group_info["drug"] = group_info["drug"].str.replace("Olapairb","Olaparib")
group_info["drug"] = group_info["drug"].str.replace("Olapairib","Olaparib")
group_info["drug"] = group_info["drug"].str.replace("olaparib","Olaparib")
group_info["GID"] = group_info.index.str.replace("F","T").str[:10]
group_info["GID"] = group_info["GID"].str.replace("T","P")
group_info["GID"] = group_info["GID"].str.replace("SV-OV-P056", "SV-OV-P105")
group_info = group_info.dropna()
group_info = group_info.drop("drug", axis=1)
group_info = group_info.drop_duplicates()
group_info = group_info.set_index("GID")

clinical = group_info

#* remove transcript-gene unmatched transcripts
texp['target_gene'] = texp.index.str.split("-",1).str[1]
TUdata = texp[texp['target_gene']!= '-']

#* sample name change
sample = TUdata.columns.tolist()# %%
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

TUdata.columns = samples
try:
    TUdata = TUdata.drop(["SV-OV-P080-atD","SV-OV-P250-atD","SV-OV-P055-atD",\
            "SV-OV-P143-atD","SV-OV-P137-atD","SV-OV-P134-atD",\
            "SV-OV-P174-bfD","SV-OV-P164-atD"], axis=1)
except:
    

    pass

#* 60% 이상 filtering
TUdata = TUdata.drop(['target_gene'],axis=1)
TUdata = TUdata[(TUdata>0).astype(int).sum(axis=1) > TUdata.shape[1]*0.6]

#* sample filtering: R / NR

wholesample = pd.DataFrame(TUdata.columns, columns=['sample'])

###&&&&################################################
res_int = clinical[clinical['group']==1]
nr = clinical[clinical['group']==0]
###&&&&#################################################

mtsample = pd.DataFrame(res_int.index)
wholesample['GID'] = wholesample['sample'].str[:-4]

onlymt = pd.merge(wholesample, mtsample, on='GID', how='inner')

mtsamplelist = onlymt['sample'].tolist()

mtsample2 = pd.DataFrame(nr.index)

onlynr = pd.merge(wholesample, mtsample2, on='GID', how='inner')

nrsamplelist = onlynr['sample'].tolist()

TUdata_R = TUdata[mtsamplelist]
TUdata_NR = TUdata[nrsamplelist]

TU_R = TUdata_R.filter(regex='-bfD')
sample1 = pd.DataFrame(TU_R.columns, columns=['samples'])
sample1['group'] = 'R_pre'

TU_NR = TUdata_NR.filter(regex='-bfD')
sample2 = pd.DataFrame(TU_NR.columns, columns=['samples'])
sample2['group'] = 'NR_pre'

tu_data = pd.concat([TU_R,TU_NR],axis=1)
sampleinfo= pd.concat([sample1,sample2])

tu_data2 = tu_data
sampleinfo2 = sampleinfo
#* geneinfo
geneinfo = pd.DataFrame(columns=['transcript','gene'])
geneinfo["gene"] = tu_data.index.str.split("-",1).str[1]
geneinfo["transcript"] = tu_data.index
geneinfo = geneinfo.set_index("transcript", drop=False)


# %%
#* input data export
path = '/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/input/discovery/resistance/'
#sampleinfo.to_csv(path+'innate_metadata.csv',index=False)
#tu_data.to_csv(path +'innate_inputdata.csv',index=True)
#geneinfo.to_csv(path+'innate_geneinfo.csv',index=True)

# %%









# %%
##** acquired/innate clustermap DTU
ac_dtu = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/output/discovery/resistance/acquired_satuRnresult.csv', sep = ",", index_col=0)
in_dtu = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/output/discovery/resistance/innate_satuRnresult.csv', sep = ",", index_col=0)

#########^#####
#dtu = ac_dtu ##acquired
dtu = in_dtu ##innate
#########^#####

#########^#####
#dtu = dtu[(dtu['pval']<0.05) & (dtu['regular_FDR']<0.1)]
dtu = dtu[(dtu['pval']<0.005)]
#########^#####

dtugenes = list(dtu.index)


#########^#####
#forplot = tu_data1.loc[dtugenes,:] ##acquired
forplot = tu_data2.loc[dtugenes,:] ##innate
#########^#####

## z-score normalization
forplot_z =  forplot.apply(lambda x: (x-x.mean())/x.std(), axis = 1)
forplot_z = forplot_z.dropna(axis=0)

#draw clustermap
#########^#####
#colors_dict = {"R_pre":"#FF9595", "R_post":"#A54A4A"} ##acquired
colors_dict = { "R_pre":"#FF9595", "NR_pre":"#7CB5FF"} ##innate
#########^#####


#########^#####
#col_colors = sampleinfo1['group'].map(colors_dict) ##acquired
col_colors = sampleinfo2['group'].map(colors_dict) ##innate
#########^#####

g = sns.clustermap(forplot_z, col_colors=[col_colors], cmap="RdBu_r" , figsize=(7,9),vmin=-2,vmax=2, center=0,
                method='ward', #single, complete, average, weighted, ward
                metric='euclidean', #cityblock, minkowski, euclidean, cosine, correlation, hamming, jaccard, chebyshev, canberra, dice, rogerstanimoto, russellrao, sokalsneath
                linewidths=0, xticklabels=False, yticklabels=False, col_cluster=True,row_cluster=True)

g.fig.subplots_adjust(right=0.7)
g.ax_cbar.set_position((0.73, .2, .03, .4))

for group in sampleinfo2['group'].unique(): #######inputmeta2: innate
    g.ax_col_dendrogram.bar(0, 0, color=colors_dict[group],
                            label=group, linewidth=0)

l1 = g.ax_col_dendrogram.legend(title='sample type', loc="center", ncol=2, bbox_to_anchor=(0.8, 0.96), bbox_transform=gcf().transFigure) #bbox_to_anchor=(0.42, 0.88),

ax = g.ax_heatmap
# ax.set_xlabel("Samples")
ax.set_ylabel("")

#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/figures/innate_dtu_clustermap.pdf", bbox_inches="tight")

plt.show()
# %%
##* acquired/innate DTU genelist
#########^#####
ac_dtu_data = ac_dtu[(ac_dtu['pval']<0.005)]
in_dtu_data = in_dtu[(in_dtu['pval']<0.005)]
#########^#####

ac_dtugenes = list(ac_dtu_data.index)
in_dtugenes = list(in_dtu_data.index)

ac_dtugenes = [i.split('-', 1)[1] for i in ac_dtugenes]
in_dtugenes = [i.split('-', 1)[1] for i in in_dtugenes]










# %%
#^ 3) R pre vs. R post (acquired/innate): DEG vs. DSG Venn Diagram

ensgtosymbol = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/ensgidtosymbol.dict', sep='\t', header=None)
ensgtosymbol.columns = ['id','symbol']
ac_degdf = pd.DataFrame(ac_degenes, columns=['id'])
in_degdf = pd.DataFrame(in_degenes, columns=['id'])

ac_tmp = pd.merge(ac_degdf, ensgtosymbol, on='id', how='inner')
in_tmp = pd.merge(in_degdf, ensgtosymbol, on='id', how='inner')

#%%
from matplotlib_venn import venn2

ac_dtugenes = set(ac_dtugenes)
in_dtugenes = set(in_dtugenes)
ac_degenes = set(ac_tmp['symbol'])
in_degenes = set(in_tmp['symbol'])

plt.figure(figsize=(6,6))
sns.set_style("white")
vd2 = venn2([ac_dtugenes, ac_degenes],set_labels=('acquired_DTU', 'acquired_DEG'),set_colors=('#F8CF6A','#74B2D4'), alpha=0.8)

for text in vd2.set_labels:  #change label size
    text.set_fontsize(13)
for text in vd2.subset_labels:  #change number size
    text.set_fontsize(16)
    
vd2.get_patch_by_id('11').set_color('#C8C452')
vd2.get_patch_by_id('11').set_alpha(1)

#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/figures/acquired_DTU_DEG_Venn.pdf", bbox_inches="tight")
plt.show()

#############

plt.figure(figsize=(6,6))
sns.set_style("white")
vd2 = venn2([in_dtugenes, in_degenes],set_labels=('innate_DTU', 'innate_DEG'),set_colors=('#F8CF6A','#74B2D4'), alpha=0.8)

for text in vd2.set_labels:  #change label size
    text.set_fontsize(13)
for text in vd2.subset_labels:  #change number size
    text.set_fontsize(16)
    
vd2.get_patch_by_id('11').set_color('#C8C452')
vd2.get_patch_by_id('11').set_alpha(1)

#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/figures/innate_DTU_DEG_Venn.pdf", bbox_inches="tight")
plt.show()
# %%
