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
import re
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.stats.multitest import multipletests
from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42  
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'Helvetica'

plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 13,    # x축 틱 라벨 글꼴 크기

'ytick.labelsize': 13,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 13,
'legend.title_fontsize': 13, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})
sns.set_style("ticks")

# %%
with open('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARup_tlist.txt', 'r') as file:
    tlist = [line.strip() for line in file]

AR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t', index_col=0)
IR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUT/nonresponder_stable_DUT_Wilcoxon.txt', sep='\t', index_col=0)

ARdutlist = AR_dut.loc[(AR_dut['p_value']<0.05) & (np.abs(AR_dut['log2FC'])>1.5)].index.to_list()
IRdutlist = IR_dut.loc[(IR_dut['p_value']<0.05) & (np.abs(IR_dut['log2FC'])>1.5)].index.to_list()

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
ARlist = list(set(sampleinfo.loc[(sampleinfo['response']==1),'sample_id']))
IRlist = list(set(sampleinfo.loc[(sampleinfo['response']==0),'sample_id']))

TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0)
TU.columns = TU.columns.str[:-4]
preTU = TU.iloc[:,1::2]
postTU = TU.iloc[:,0::2]

majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = majorminor[majorminor['type']=='major']['gene_ENST'].to_list()
minorlist = majorminor[majorminor['type']=='minor']['gene_ENST'].to_list()

#%%
######
tlist = list(set(IRdutlist).intersection(set(minorlist)))
# with open('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARup_tlist.txt', 'r') as file:
#     tlist = [line.strip() for line in file]
######

#%%
##^^^ plot by sample
ARpre = preTU.loc[tlist,ARlist]
IRpre = preTU.loc[tlist,IRlist]
ARpost = postTU.loc[tlist,ARlist]
predf = pd.DataFrame({'mean TU':ARpre.mean(), 'time':'pre', 'sample':ARpre.columns})
postdf = pd.DataFrame({'mean TU':ARpost.mean(), 'time':'post', 'sample':ARpre.columns})

finaldf = pd.concat([predf,postdf],axis=0)
finaldf = finaldf.reset_index()

plt.figure(figsize=(4, 5))
ax = sns.boxplot(y='mean TU', x='time', data=finaldf, 
                        showfliers=False, order=['pre','post'], palette={"pre": "#FFCC29", "post": "#FFAE4B"}
                        #meanprops = {"marker":"o", "color":"blue","markeredgecolor": "blue", "markersize":"10"}
                        )
#ax.set_ylim([0,0.01])
ax.set_ylabel('mean TU per sample')
sns.despine()

from statannotations.Annotator import Annotator

pairs = [("pre","post")]
annot = Annotator(ax, pairs, data=finaldf, x="time", y="mean TU")
annot.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=1, ) #comparisons_correction="Bonferroni"
annot.apply_and_annotate()

for p in finaldf['sample'].unique():
    subset = finaldf[finaldf['sample']==p]
    x = [list(subset['time'])[0], list(subset['time'])[1]]
    y = [list(subset['mean TU'])[0], list(subset['mean TU'])[1]]
    plt.plot(x,y, marker="o", markersize=4, color='grey', linestyle="--", linewidth=0.7)

plt.xlabel("")
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/sampleboxplot_ARminordutlist.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%
##^^ plot by transcript
ARpre = preTU.loc[tlist,ARlist]
IRpre = preTU.loc[tlist,IRlist]

ARpre_mean = ARpre.mean(axis=1)  # transcript 별 평균 TU (pre)
ARpost_mean = ARpost.mean(axis=1)  # transcript 별 평균 TU (post)

predf = pd.DataFrame({'mean TU': ARpre_mean, 'time': 'pre', 'transcript': ARpre.index})
postdf = pd.DataFrame({'mean TU': ARpost_mean, 'time': 'post', 'transcript': ARpost.index})

finaldf = pd.concat([predf, postdf], axis=0).reset_index(drop=True)

plt.figure(figsize=(4, 5))
ax = sns.boxplot(y='mean TU', x='time', data=finaldf, showfliers=False, 
                 order=['pre', 'post'], palette={"pre": "#FFCC29", "post": "#FFAE4B"})


# for t in finaldf['transcript'].unique():
#     subset = finaldf[finaldf['transcript'] == t]
#     x = subset['time'].tolist()
#     y = subset['mean TU'].tolist()
#     plt.plot(x, y, marker="o", markersize=3, color='grey', linestyle="--", linewidth=0.7, alpha=0.6)
# plt.ylim([-0.005,0.223])

pairs = [("pre", "post")]
annot = Annotator(ax, pairs, data=finaldf, x="time", y="mean TU")
annot.configure(test="Mann-Whitney", text_format="star", loc="outside", verbose=1, )
annot.apply_and_annotate()

ax.set_ylabel('mean TU per transcript')
sns.despine()
plt.xlabel("")
ax.set_xticklabels(["AR pre","AR post"])
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/transboxplot_ARminordutlist.pdf', dpi=300, bbox_inches='tight')
plt.show()



#%%
##^^ AR pre vs. IR pre by sample #######
ARpre = preTU.loc[tlist,ARlist]
IRpre = preTU.loc[tlist,IRlist]
ARpost = postTU.loc[tlist,ARlist]
predf = pd.DataFrame({'mean TU':ARpre.mean(), 'time':'AR', 'sample':ARpre.columns})
postdf = pd.DataFrame({'mean TU':IRpre.mean(), 'time':'IR', 'sample':IRpre.columns})

finaldf = pd.concat([predf,postdf],axis=0)
finaldf = finaldf.reset_index()

plt.figure(figsize=(4, 5))
ax = sns.boxplot(y='mean TU', x='time', data=finaldf, 
                        showfliers=False, order=['AR','IR'], palette={"AR": "#FFCC29", "IR": "#81B214"}
                        #meanprops = {"marker":"o", "color":"blue","markeredgecolor": "blue", "markersize":"10"}
                        )
#ax.set_ylim([0,0.01])
ax.set_ylabel('mean TU per sample')
sns.despine()

ax = sns.swarmplot(y='mean TU', x='time', data=finaldf, 
                        order=['AR','IR'],size=4, color='grey'
                        #meanprops = {"marker":"o", "color":"blue","markeredgecolor": "blue", "markersize":"10"}
                        )


from statannotations.Annotator import Annotator

pairs = [("AR","IR")]
annot = Annotator(ax, pairs, data=finaldf, x="time", y="mean TU")
annot.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=1,)
annot.apply_and_annotate()
plt.xlabel("")
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/sampleboxplot_ARpreIRpre_ARminordutlist.pdf', dpi=300, bbox_inches='tight')
plt.show()


#%%
##^^ AR pre vs. IR pre by transcript #######

ARpre = preTU.loc[tlist,ARlist]
IRpre = preTU.loc[tlist,IRlist]

ARpre_mean = ARpre.mean(axis=1) 
IRpre_mean = IRpre.mean(axis=1) 

predf = pd.DataFrame({'mean TU': ARpre_mean, 'time': 'AR', 'transcript': ARpre.index})
postdf = pd.DataFrame({'mean TU': IRpre_mean, 'time': 'IR', 'transcript': IRpre.index})

finaldf = pd.concat([predf, postdf], axis=0).reset_index(drop=True)

# 시각화
plt.figure(figsize=(4, 5))
ax = sns.boxplot(y='mean TU', x='time', data=finaldf,showfliers=False,
                 order=['AR', 'IR'], palette={"AR": "#FFCC29", "IR": "#81B214"})
#sns.swarmplot(y='mean TU', x='time', data=finaldf, order=['AR', 'IR'], palette={"AR": "#FFCC29", "IR": "#81B214"}, size=4)


# for t in finaldf['transcript'].unique():
#     subset = finaldf[finaldf['transcript'] == t]
#     x = subset['time'].tolist()
#     y = subset['mean TU'].tolist()
#     plt.plot(x, y, marker="o", markersize=3, color='grey', linestyle="--", linewidth=0.7, alpha=0.6) 

# plt.ylim([-0.003, 0.073])

# 통계 검정 (Mann-Whitney U test)
pairs = [("AR", "IR")]
annot = Annotator(ax, pairs, data=finaldf, x="time", y="mean TU")
annot.configure(test="Mann-Whitney", text_format="star", loc="outside", verbose=1, )
annot.apply_and_annotate()
ax.set_ylabel('mean TU per transcript')
ax.set_xticklabels(["AR pre","IR pre"])
sns.despine()
plt.xlabel("")
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARIRpretransboxplot_ARminordutlist.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
####^^^ validation cohort check ########
val = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_TU.txt', sep='\t', index_col=0)
valinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_validation_clinicalinfo_new.txt', sep='\t', index_col=0)
vallist = list(val.columns)
valinfo = valinfo.loc[vallist,:]

#%%
filteredval = val.loc[tlist,:]
filteredval = filteredval.apply(pd.to_numeric, errors='coerce')

###^ val cohort check by sample ######
meanval = pd.DataFrame({'mean TU':filteredval.mean(),'type':valinfo['type'],'gHRDscore':valinfo['gHRDscore'],'BRCAmut':valinfo['BRCAmut']})

plt.figure(figsize=(5, 5))
ax = sns.boxplot(y='mean TU', x='type', data=meanval, 
                        showfliers=False, palette={"AR": "#FFCC29", "IR": "#81B214","CR":"#409EDD"}
                        )
ax = sns.swarmplot(y='mean TU', x='type', data=meanval, 
                        order=['CR','AR','IR'],size=4, color='grey', alpha=0.5
                        #meanprops = {"marker":"o", "color":"blue","markeredgecolor": "blue", "markersize":"10"}
                        )
ax.set_ylabel('mean TU by sample')
plt.xlabel("")
sns.despine()

pairs = [("CR","AR"),("CR","IR"),("AR","IR")]
annot = Annotator(ax, pairs, data=meanval, x="type", y="mean TU")
annot.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=1)
annot.apply_and_annotate()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/validation_ARminordutlist_boxplot_sample.pdf', dpi=300, bbox_inches='tight')
plt.show()

meanval = meanval.replace('CR', 'R')
meanval = meanval.replace('AR', 'R')

plt.figure(figsize=(4, 5))
ax = sns.boxplot(y='mean TU', x='type', data=meanval, 
                        showfliers=False, palette={"R": "#955DB3", "IR": "#81B214"}
                        )
# ax = sns.swarmplot(y='mean TU', x='type', data=meanval, 
#                         order=['CR','AR','IR'],size=4, color='grey'
#                         #meanprops = {"marker":"o", "color":"blue","markeredgecolor": "blue", "markersize":"10"}
#                         )
#ax.set_ylim([0,0.01])
ax.set_ylabel('mean TU')
plt.xlabel("")
sns.despine()
pairs = [("CR","AR"),("CR","IR"),("AR","IR")]
pairs = [("R","IR")]
annot = Annotator(ax, pairs, data=meanval, x="type", y="mean TU")
annot.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=1, correction_format="Bonferroni")
annot.apply_and_annotate()
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/validation_RNR_ARmajordutlist_boxplot.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
###^ val cohort check by transcript ######
#filteredval = filteredval.drop(index='MSTRG.92133.3-PAK4')

filteredval = val.loc[tlist,:]
filteredval = filteredval.apply(pd.to_numeric, errors='coerce')

# Transcript 단위 평균 TU 계산
CR_mean = filteredval.loc[:, valinfo[valinfo['type'] == 'CR'].index].mean(axis=1)
AR_mean = filteredval.loc[:, valinfo[valinfo['type'] == 'AR'].index].mean(axis=1)
IR_mean = filteredval.loc[:, valinfo[valinfo['type'] == 'IR'].index].mean(axis=1)

# 새로운 DataFrame 생성 (Transcript별 TU 비교)
crdf = pd.DataFrame({'mean TU': CR_mean, 'type': 'CR', 'transcript': filteredval.index})
ardf = pd.DataFrame({'mean TU': AR_mean, 'type': 'AR', 'transcript': filteredval.index})
irdf = pd.DataFrame({'mean TU': IR_mean, 'type': 'IR', 'transcript': filteredval.index})

finaldf = pd.concat([crdf, ardf, irdf], axis=0).reset_index(drop=True)

# 시각화
plt.figure(figsize=(5, 5))
ax = sns.boxplot(y='mean TU', x='type', data=finaldf, showfliers=False,
                 palette={"AR": "#FFCC29", "IR": "#81B214", "CR": "#409EDD"})
#sns.swarmplot(y='mean TU', x='type', data=finaldf, color="grey", size=4, alpha=0.5)

# 통계 검정 (Mann-Whitney U test)
pairs = [("CR", "AR"), ("CR", "IR"), ("AR", "IR")]
annot = Annotator(ax, pairs, data=finaldf, x="type", y="mean TU")
annot.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=1, comparisons_correction="Bonferroni")
annot.apply_and_annotate()

ax.set_ylabel('mean TU per transcript')
sns.despine()
plt.xlabel("")
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/validation_IRminordutlist_boxplot_trans.pdf', dpi=300, bbox_inches='tight')

plt.show()

# Responder (R) vs. IR 비교
finaldf = finaldf.replace('CR', 'R')
finaldf = finaldf.replace('AR', 'R')

plt.figure(figsize=(4, 5))
ax = sns.boxplot(y='mean TU', x='type', data=finaldf, showfliers=False, palette={"R": "#955DB3", "IR": "#81B214"})
#sns.swarmplot(y='mean TU', x='type', data=finaldf, color="grey", size=3)

# 통계 검정 (Mann-Whitney U test)
pairs = [("R", "IR")]
annot = Annotator(ax, pairs, data=finaldf, x="type", y="mean TU")
annot.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=1, comparisons_correction="Bonferroni")
annot.apply_and_annotate()

ax.set_ylabel('mean TU per transcript')
sns.despine()
plt.xlabel("")
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/validation_RNR_IRminordutlist_boxplot_trans.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
