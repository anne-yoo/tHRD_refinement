#%%
#! this code is.....
## 20230525
"""
1) R vs. NR : whole gene DEG + clustermap
2) R vs. NR : whole gene DTU + clustermap
3) R vs. NR : DEG vs. DTU Venn Diagram

pre / post 상관없이 response로만 확인해보기

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
#^ 1) R vs. NR : whole gene DEG + clustermap

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
countres = mTU[mtsamplelist]
##nonres
nonres = clinical[clinical['group']==0]
mtsample = pd.DataFrame(nonres.index)
wholesample['GID'] = wholesample['sample'].str[:-4]
onlymt = pd.merge(wholesample, mtsample, on='GID', how='inner')
mtsamplelist = onlymt['sample'].tolist()
countnonres = mTU[mtsamplelist]

nonres_info = pd.DataFrame(countnonres.columns)
nonres_info['group'] = 'Non-Responder'

res_info = pd.DataFrame(countres.columns)
res_info['group'] = 'Responder'

finalinput = pd.concat([countres, countnonres],axis=1)
inputmeta = pd.concat([res_info, nonres_info])

inputmeta.columns = ['sample','group']


#inputmeta.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/DESeq2/response/metadata.csv',index=False)
#finalinput.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/DESeq2/response/countdata.csv',index=True)


#%%
###* DEG clustermap
deg = pd.read_csv('//home/jiye/jiye/copycomparison/gDUTresearch/DESeq2/response/DEG_result.csv', sep = ",", index_col=0)

deg = deg[(deg['pvalue']<0.05)]
degenes = list(deg.index)

forplot = finalinput.loc[degenes,:] ##only deg
#forplot.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/DESeq2/response/onlydeg_countdata.csv',index=True)

#%%
## z-score normalization
forplot_z =  forplot.apply(lambda x: (x-x.mean())/x.std(), axis = 1)
forplot_z = forplot_z.dropna(axis=0)
colors_dict = { "Responder":"#D86868", "Non-Responder":"#578DD5"} 

col_colors = inputmeta['group'].map(colors_dict)

g = sns.clustermap(forplot_z, col_colors=[col_colors], cmap="RdBu_r" , figsize=(7,9),vmin=-2,vmax=2, center=0,
                method='ward', #single, complete, average, weighted, ward
                metric='euclidean', #cityblock, minkowski, euclidean, cosine, correlation, hamming, jaccard, chebyshev, canberra, dice, rogerstanimoto, russellrao, sokalsneath
                linewidths=0, xticklabels=False, yticklabels=False, col_cluster=True,row_cluster=True)

g.fig.subplots_adjust(right=0.7)
g.ax_cbar.set_position((0.73, .2, .03, .4))

for group in inputmeta['group'].unique(): #######inputmeta2: innate
    g.ax_col_dendrogram.bar(0, 0, color=colors_dict[group],
                            label=group, linewidth=0)

l1 = g.ax_col_dendrogram.legend(title='sample type', loc="center", ncol=2, bbox_to_anchor=(0.75, 0.96), bbox_transform=gcf().transFigure) #bbox_to_anchor=(0.42, 0.88),

ax = g.ax_heatmap
# ax.set_xlabel("Samples")
ax.set_ylabel("")

#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/figures/response_deg_clustermap.pdf", bbox_inches="tight")

plt.show()









# %%
#^ 2) R vs. NR : whole gene DTU + clustermap
##* satuRn input generation

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

res = clinical[clinical['group']==1]
nonres = clinical[clinical['group']==0]

rsample = pd.DataFrame(res.index)
nrsample = pd.DataFrame(nonres.index)
wholesample['GID'] = wholesample['sample'].str[:-4]

rtmp = pd.merge(wholesample, rsample, on='GID', how='inner')
nrtmp = pd.merge(wholesample, nrsample, on='GID', how='inner')

rTU = TUdata[list(rtmp['sample'])]
nrTU = TUdata[list(nrtmp['sample'])]

r_sampleinfo = pd.DataFrame(rTU.columns, columns=['samples'])
r_sampleinfo['group'] = 'Responder'

nr_sampleinfo = pd.DataFrame(nrTU.columns, columns=['samples'])
nr_sampleinfo['group'] = 'Non-Responder'

finaltu = pd.concat([rTU,nrTU], axis=1)
sampleinfo = pd.concat([r_sampleinfo,nr_sampleinfo])

geneinfo = pd.DataFrame(columns=['transcript','gene'])
geneinfo["gene"] = finaltu.index.str.split("-",1).str[1]
geneinfo["transcript"] = finaltu.index

# %%
#* input data export
# path = '/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/input/discovery/response/'
# sampleinfo.to_csv(path+'metadata.csv',index=False)
# finaltu.to_csv(path +'inputdata.csv',index=True)
# geneinfo.to_csv(path+'geneinfo.csv',index=True)




#%%
###* DEG clustermap
dtu = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/output/discovery/response/satuRnresult.csv', sep = ",", index_col=0)

dtu = dtu[(dtu['pval']<0.005)]
dtugenes = list(dtu.index)

forplot = finaltu.loc[dtugenes,:] ##innate
#forplot.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/input/discovery/response/onlydtu_inputdata.csv', index=True)

#%%

## z-score normalization
forplot_z =  forplot.apply(lambda x: (x-x.mean())/x.std(), axis = 1)
forplot_z = forplot_z.dropna(axis=0)
colors_dict = { "Responder":"#D86868", "Non-Responder":"#578DD5"} 

col_colors = sampleinfo['group'].map(colors_dict)

g = sns.clustermap(forplot_z, col_colors=[col_colors], cmap="RdBu_r" , figsize=(7,9),vmin=-2,vmax=2, center=0,
                method='ward', #single, complete, average, weighted, ward
                metric='euclidean', #cityblock, minkowski, euclidean, cosine, correlation, hamming, jaccard, chebyshev, canberra, dice, rogerstanimoto, russellrao, sokalsneath
                linewidths=0, xticklabels=False, yticklabels=False, col_cluster=True,row_cluster=True)

g.fig.subplots_adjust(right=0.7)
g.ax_cbar.set_position((0.73, .2, .03, .4))

for group in sampleinfo['group'].unique(): #######inputmeta2: innate
    g.ax_col_dendrogram.bar(0, 0, color=colors_dict[group],
                            label=group, linewidth=0)

l1 = g.ax_col_dendrogram.legend(title='sample type', loc="center", ncol=2, bbox_to_anchor=(0.75, 0.96), bbox_transform=gcf().transFigure) #bbox_to_anchor=(0.42, 0.88),

ax = g.ax_heatmap
# ax.set_xlabel("Samples")
ax.set_ylabel("")

#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/figures/response_dtu_clustermap.pdf", bbox_inches="tight")

plt.show()





# %%
##^3) R vs. NR : DEG vs. DTU Venn Diagram

deg = pd.read_csv("/home/jiye/jiye/copycomparison/gDUTresearch/DTUDEGcomp/DESeq2_DEG_result_symbol.txt",sep = ",", index_col=1)


dtu_data = dtu[(dtu['pval']<0.005)]
dtugene = list(dtu_data.index)
dtugene = [i.split('-', 1)[1] for i in dtugene]

deg_data = deg[(deg['pvalue']<0.05)]
deggene = list(deg_data.index)

deg_genes = set(deggene)
dtu_genes = set(dtugene)

from matplotlib_venn import venn2

plt.figure(figsize=(6,6))
sns.set_style("white")
vd2 = venn2([dtu_genes, deg_genes],set_labels=('response_DTU', 'response_DEG'),set_colors=('#F8CF6A','#74B2D4'), alpha=0.8)

for text in vd2.set_labels:  #change label size
    text.set_fontsize(13)
for text in vd2.subset_labels:  #change number size
    text.set_fontsize(16)
    
vd2.get_patch_by_id('11').set_color('#C8C452')
vd2.get_patch_by_id('11').set_alpha(1)

plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/figures/response_DTU_DEG_Venn.pdf", bbox_inches="tight")
plt.show()

# %%
