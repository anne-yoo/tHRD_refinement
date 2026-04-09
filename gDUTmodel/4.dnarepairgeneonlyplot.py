#%%
#! this code is.....
## R pre vs. R post vs. NR pre vs. NR post 비교를 위해 인트로에서 clustering 및 heatmap으로 전체적인 양상을 먼저 확인하기! (Discovery Cohort) 230516
## ****ONLY DNA REPAIR VERSION**** 
##1) major/minor transcript expression heatmap
##2) minor/major transcript expression boxplot
##3) major/minor transcript usage heatmap
##4) minor/major transcript usage boxplot
##5) kdeplot major/minor TU
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
#* gene list download
# repair = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/DNARepairRelatedGenes.txt',  sep='\t', header=None)

# genelist = [repair.iloc[i,2:] for i in range(0,5)]
# geneset = [gene for sublist in genelist for gene in sublist]
# geneset = list(set(geneset))
# repairgenelist = [item for item in geneset if str(item) != 'nan']
#print(repairgenelist)#geneset = geneset.remove(np.nan)

# #* gene list save
# import pickle

# filePath = '/home/jiye/jiye/copycomparison/gDUTresearch/data/DNARepairGeneList.txt'
# with open(filePath, 'wb') as lf:
#     pickle.dump(repairgenelist, lf)


#%%
##^ 1) major/minor transcript expression heatmap
#* data download
repair_path = '/home/jiye/jiye/copycomparison/gDUTresearch/data/DNARepairGeneList.txt' #repair gene list as python list
with open(repair_path, 'rb') as lf:
    repairgenes = pickle.load(lf)

#%%
#with open('/home/jiye/jiye/copycomparison/gDUTresearch/data/newDNARepairGeneList.txt', 'w+') as file:
#    file.write('\n'.join(repairgenes))
#%%
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


#* split to major and minor
majorlist = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/discovery_majorTU.txt',  sep='\t', index_col=0)
major_transcript_list = majorlist.index.to_list()

majorTU = TUdata.loc[major_transcript_list]
minorTU = TUdata.loc[~TUdata.index.isin(major_transcript_list)]

###########* filter only repair genes!!!!!!!!#######################
majorTU = majorTU[majorTU['target_gene'].isin(repairgenes)]
minorTU = minorTU[minorTU['target_gene'].isin(repairgenes)]
##*#################################################################

#* filter 60%
majorTU = majorTU.drop(['target_gene'],axis=1)
minorTU = minorTU.drop(['target_gene'],axis=1)

majorTU = majorTU[(majorTU>0).astype(int).sum(axis=1) > majorTU.shape[1]*0.6]
minorTU = minorTU[(minorTU>0).astype(int).sum(axis=1) > minorTU.shape[1]*0.6]
# %%
#%%
#* split major TU data into R_pre, R_post, NR_pre, NR_post
#* first into R and NR
########&&&######################
mTU = minorTU
####&&###########################

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

#* then split into R_pre, R_post, NR_pre, NR_post
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

#* concat 4 data
final_major = pd.concat([major_R_pre,major_R_post,major_NR_pre,major_NR_post],axis=1)
final_major_info = pd.concat([major_info_1,major_info_2,major_info_3,major_info_4])
final_major_info.columns = ['samples','group']

#* data zscore normalization: one gene <-> across all samples!!
major_zscore =  final_major.apply(lambda x: (x-x.mean())/x.std(), axis = 1)
major_zscore = major_zscore.dropna(axis=0)
#%%
#* color mapping
colors_dict = { "R_pre":"#FF9595", "R_post":"#C0C0C0", "NR_pre":"#7CB5FF", "NR_post":"#C0C0C0"} #Rpre vs. NRpre
#colors_dict = { "R_pre":"#FF9595", "R_post":"#A54A4A", "NR_pre":"#C0C0C0", "NR_post":"#C0C0C0"} #Rpre vs. Rpost
col_colors = final_major_info['group'].map(colors_dict)

#* draw seaborn clustermap
methods = ['single', 'complete', 'average', 'weighted', 'ward'] #5
metrics = ['cityblock', 'minkowski', 'seuclidean', 'cosine', 'correlation', 'hamming', 'jaccard', 'chebyshev', 'canberra', 'dice', 'rogerstanimoto', 'russellrao', 'sokalsneath'] #13

g = sns.clustermap(major_zscore, col_colors=[col_colors], cmap="RdBu_r" , figsize=(8,10),vmin=-2,vmax=2, center=0,
                method='ward', #single, complete, average, weighted, ward
                metric='euclidean', #cityblock, minkowski, euclidean, cosine, correlation, hamming, jaccard, chebyshev, canberra, dice, rogerstanimoto, russellrao, sokalsneath
                linewidths=0, xticklabels=False, yticklabels=False, col_cluster=True,row_cluster=True)

g.fig.subplots_adjust(right=0.7)
g.ax_cbar.set_position((0.73, .2, .03, .4))

for group in  final_major_info['group'].unique():
    g.ax_col_dendrogram.bar(0, 0, color=colors_dict[group],
                            label=group, linewidth=0)

l1 = g.ax_col_dendrogram.legend(title='sample type', loc="center", ncol=2, bbox_to_anchor=(0.46, 0.86), bbox_transform=gcf().transFigure) #bbox_to_anchor=(0.42, 0.88),

ax = g.ax_heatmap
# ax.set_xlabel("Samples")
ax.set_ylabel("")

# %%
#^ 2) mean minor/major transcript expression boxplot
mean_R_pre = pd.DataFrame(major_R_pre.apply(lambda x: x.mean(), axis=1))
mean_R_pre['group'] = 'R_pre'

mean_R_post = pd.DataFrame(major_R_post.apply(lambda x: x.mean(), axis=1))
mean_R_post['group'] = 'R_post'

mean_NR_pre = pd.DataFrame(major_NR_pre.apply(lambda x: x.mean(), axis=1))
mean_NR_pre['group'] = 'NR_pre'

mean_NR_post = pd.DataFrame(major_NR_post.apply(lambda x: x.mean(), axis=1))
mean_NR_post['group'] = 'NR_post'

mean_df = pd.concat([mean_R_pre,mean_R_post,mean_NR_pre,mean_NR_post])
sns.set(rc = {'figure.figsize':(10,5)})
sns.set_style('whitegrid')
plt.suptitle('<Mean Minor Transcript Expression>', fontsize=13)
orders = ['R_pre','R_post','NR_pre','NR_post']

g = sns.boxplot(data=mean_df, y=0, x="group",palette='hls',showfliers=False)
g.set_ylabel("transcript expression")
plt.show()



# %%
#^ 2-1) minor/major transcript expression boxplot : z-score

major_R_pre = major_R_pre.apply(lambda x: (x-x.mean())/x.std(), axis = 1)
stacked_R_pre = major_R_pre.stack().reset_index()
stacked_R_pre['group'] = 'R_pre'
stacked_R_pre = stacked_R_pre.iloc[:,2:]

major_R_post = major_R_post.apply(lambda x: (x-x.mean())/x.std(), axis = 1)
stacked_R_post = major_R_post.stack().reset_index()
stacked_R_post['group'] = 'R_post'
stacked_R_post = stacked_R_post.iloc[:,2:]

major_NR_pre = major_NR_pre.apply(lambda x: (x-x.mean())/x.std(), axis = 1)
stacked_NR_pre = major_NR_pre.stack().reset_index()
stacked_NR_pre['group'] = 'NR_pre'
stacked_NR_pre = stacked_NR_pre.iloc[:,2:]

major_NR_post = major_NR_post.apply(lambda x: (x-x.mean())/x.std(), axis = 1)
stacked_NR_post = major_NR_post.stack().reset_index()
stacked_NR_post['group'] = 'NR_post'
stacked_NR_post = stacked_NR_post.iloc[:,2:]

stacked_df = pd.concat([stacked_R_pre,stacked_R_post,stacked_NR_pre,stacked_NR_post])

sns.set(rc = {'figure.figsize':(10,5)})
sns.set_style('whitegrid')
plt.suptitle('<Minor Transcript Expression z-score>', fontsize=13)
orders = ['R_pre','R_post','NR_pre','NR_post']
g = sns.boxplot(data=stacked_df, y=0, x="group",palette='hls',showfliers=False)
g.set_ylabel("transcript expression")
plt.show()
# %%















#%%
#^ 3) major/minor TU heatmap

#* data download
texp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/discovery_minorTU.txt',  sep='\t', index_col=0) #minor
##texp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/discovery_majorTU.txt',  sep='\t', index_col=0) #major
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

###########* filter only repair genes!!!!!!!!#######################
TUdata = TUdata[TUdata['target_gene'].isin(repairgenes)]
##*#################################################################

#* filter with 60%
TUdata = TUdata.drop(['target_gene'],axis=1)
TUdata = TUdata[(TUdata>0).astype(int).sum(axis=1) > TUdata.shape[1]*0.6]



#* split into R_pre, R_post, NR_pre, NR_post
#* first into R and NR
########&&&######################
mTU = TUdata
####&&###########################

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

#* then split into R_pre, R_post, NR_pre, NR_post
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

#* concat 4 data
final_major = pd.concat([major_R_pre,major_R_post,major_NR_pre,major_NR_post],axis=1)
final_major_info = pd.concat([major_info_1,major_info_2,major_info_3,major_info_4])
final_major_info.columns = ['samples','group']

#* data zscore normalization: one gene <-> across all samples!!
major_zscore =  final_major.apply(lambda x: (x-x.mean())/x.std(), axis = 1)
major_zscore = major_zscore.dropna(axis=0)

2#%%
#* color mapping
colors_dict = { "R_pre":"#FF9595", "R_post":"#C0C0C0", "NR_pre":"#7CB5FF", "NR_post":"#C0C0C0"} #Rpre vs. NRpre
#colors_dict = { "R_pre":"#FF9595", "R_post":"#A54A4A", "NR_pre":"#C0C0C0", "NR_post":"#C0C0C0"} #Rpre vs. Rpost
col_colors = final_major_info['group'].map(colors_dict)

#* draw seaborn clustermap
methods = ['single', 'complete', 'average', 'weighted', 'ward'] #5
metrics = ['cityblock', 'minkowski', 'seuclidean', 'cosine', 'correlation', 'hamming', 'jaccard', 'chebyshev', 'canberra', 'dice', 'rogerstanimoto', 'russellrao', 'sokalsneath'] #13

g = sns.clustermap(major_zscore, col_colors=[col_colors], cmap="RdBu_r" , figsize=(8,10),vmin=-2,vmax=2, center=0,
                method='ward', #single, complete, average, weighted, ward
                metric='euclidean', #cityblock, minkowski, euclidean, cosine, correlation, hamming, jaccard, chebyshev, canberra, dice, rogerstanimoto, russellrao, sokalsneath
                linewidths=0, xticklabels=False, yticklabels=False, col_cluster=True,row_cluster=True)

g.fig.subplots_adjust(right=0.7)
g.ax_cbar.set_position((0.73, .2, .03, .4))

for group in  final_major_info['group'].unique():
    g.ax_col_dendrogram.bar(0, 0, color=colors_dict[group],
                            label=group, linewidth=0)

l1 = g.ax_col_dendrogram.legend(title='sample type', loc="center", ncol=2, bbox_to_anchor=(0.46, 0.86), bbox_transform=gcf().transFigure) #bbox_to_anchor=(0.42, 0.88),

ax = g.ax_heatmap
# ax.set_xlabel("Samples")
ax.set_ylabel("")


# %%
#^ 4) mean minor/major transcript usage boxplot
mean_R_pre = pd.DataFrame(major_R_pre.apply(lambda x: x.mean(), axis=1))
mean_R_pre['group'] = 'R_pre'

mean_R_post = pd.DataFrame(major_R_post.apply(lambda x: x.mean(), axis=1))
mean_R_post['group'] = 'R_post'

mean_NR_pre = pd.DataFrame(major_NR_pre.apply(lambda x: x.mean(), axis=1))
mean_NR_pre['group'] = 'NR_pre'

mean_NR_post = pd.DataFrame(major_NR_post.apply(lambda x: x.mean(), axis=1))
mean_NR_post['group'] = 'NR_post'

mean_df = pd.concat([mean_R_pre,mean_R_post,mean_NR_pre,mean_NR_post])
sns.set(rc = {'figure.figsize':(10,5)})
sns.set_style('whitegrid')
orders = ['R_pre','R_post','NR_pre','NR_post']
g = sns.boxplot(data=mean_df, y=0, x="group",palette='hls',showfliers=False)
g.set_ylabel("transcript usage")
plt.suptitle('<Mean Major Transcript Usage >', fontsize=13)
plt.show()
# %%
#^ 4-1) mean minor/major transcript usage boxplot
major_R_pre = major_R_pre.apply(lambda x: (x-x.mean())/x.std(), axis = 1)
stacked_R_pre = major_R_pre.stack().reset_index()
stacked_R_pre['group'] = 'R_pre'
stacked_R_pre = stacked_R_pre.iloc[:,2:]

major_R_post = major_R_post.apply(lambda x: (x-x.mean())/x.std(), axis = 1)
stacked_R_post = major_R_post.stack().reset_index()
stacked_R_post['group'] = 'R_post'
stacked_R_post = stacked_R_post.iloc[:,2:]

major_NR_pre = major_NR_pre.apply(lambda x: (x-x.mean())/x.std(), axis = 1)
stacked_NR_pre = major_NR_pre.stack().reset_index()
stacked_NR_pre['group'] = 'NR_pre'
stacked_NR_pre = stacked_NR_pre.iloc[:,2:]

major_NR_post = major_NR_post.apply(lambda x: (x-x.mean())/x.std(), axis = 1)
stacked_NR_post = major_NR_post.stack().reset_index()
stacked_NR_post['group'] = 'NR_post'
stacked_NR_post = stacked_NR_post.iloc[:,2:]

stacked_df = pd.concat([stacked_R_pre,stacked_R_post,stacked_NR_pre,stacked_NR_post])
sns.set(rc = {'figure.figsize':(10,5)})
sns.set_style('whitegrid')
plt.suptitle('<Major Transcript Usage z-score>', fontsize=13)
orders = ['R_pre','R_post','NR_pre','NR_post']
g = sns.boxplot(data=stacked_df, y=0, x="group",palette='hls',showfliers=False)
g.set_ylabel("transcript usage")
plt.show()


#%%
#^ 5) major/minor TU z-score kdeplot

stacked_df = stacked_df.reset_index()
k=sns.kdeplot(
    data=stacked_df, x=0, hue="group",
    fill=True, common_norm=False, palette="hls",
    alpha=.5, linewidth=0,
)
k.set_xlabel("transcript usage z-score")
plt.suptitle('<Major Transcript Usage z-score>', fontsize=13)
plt.show()

# %%
