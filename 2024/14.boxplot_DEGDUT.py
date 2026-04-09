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
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import plotly.express as px
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.stats.multitest import multipletests



# %%
#### gene exp ####
geneexp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM_symbol.txt', sep='\t', index_col=0 )
geneexp = geneexp.set_index('Gene Symbol')

genename = 'CHEK2'

gene = geneexp.loc[genename,:]
genedf = pd.DataFrame(gene)
genedf.columns = ['TPM']
genedf['sample'] = genedf.index

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
genedf['Treatment'] = genedf['sample'].apply(lambda x: 'Post' if '-atD' in x else 'Pre')
genedf['sampleid'] = genedf['sample'].str[:-4]

# ###** only AR/IR samples ###
# ARlist = list(sampleinfo.loc[sampleinfo['response']==1,'sample_full'])
# IRlist = list(sampleinfo.loc[sampleinfo['response']==0,'sample_full'])

# count = count[count['sample'].isin(IRlist)]
# #####*###################



plt.figure(figsize=(3,5))
#sns.set_style("whitegrid")
sns.set_theme(style='ticks',palette='pastel')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 12,    # x축 틱 라벨 글꼴 크기
'ytick.labelsize': 12,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 11,
'legend.title_fontsize': 11, # 범례 글꼴 크기
'figure.titlesize': 14    # figure 제목 글꼴 크기
})
ax = sns.boxplot(y='TPM', x='Treatment', data=genedf, order=['Pre','Post'],
            showmeans=False, palette='vlag'
            #meanprops = {"marker":"o", "color":"blue","markeredgecolor": "blue", "markersize":"10"}
            )

from statannot import add_stat_annotation
add_stat_annotation(ax, data=genedf, x='Treatment', y='TPM',
                    box_pairs=[("Pre", "Post")],
                    test='Wilcoxon',  text_format='star', loc='inside', fontsize=14)
plt.title(genename, fontsize=14)
plt.ylabel('TPM')
sns.despine()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/fig1/CHEK2_DEG_boxplot.png', dpi=300, bbox_inches='tight')

plt.show()

#%%
for p in genedf['sampleid'].unique():
    subset = genedf[genedf['sampleid']==p]
    x = [list(subset['treatment'])[1], list(subset['treatment'])[0]]
    y = [list(subset['TPM'])[1], list(subset['TPM'])[0]]
    plt.plot(x,y, marker="o", markersize=4, color='grey', linestyle="--", linewidth=0.7)
plt.title(genename)

#%%
pre_mean = np.mean(genedf.iloc[1::2,0])
post_mean = np.mean(genedf.iloc[0::2,0])
tmplist = [pre_mean,post_mean]
genefigure = pd.DataFrame({'gene':'BRCA1','mean TPM':tmplist, 'treatment':['pre','post']})

plt.figure(figsize=(3,8))
sns.set_theme(style='ticks', palette='pastel', font_scale=1)

ax = sns.barplot(y='mean TPM', x='gene', data=genefigure, hue='treatment',
            palette='vlag'
            #meanprops = {"marker":"o", "color":"blue","markeredgecolor": "blue", "markersize":"10"}
            )
ax.set(xticklabels=[]) 


# %%
#### TU ####
TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_TU.txt', sep='\t', index_col=0 )
TU['Gene Symbol'] = TU.index.str.split("-",2).str[1]

#%%
ARlist = list(sampleinfo.loc[sampleinfo['response']==1,'sample_full'])
IRlist = list(sampleinfo.loc[sampleinfo['response']==0,'sample_full'])
genename = 'BRIP1'
transdf = TU[TU['Gene Symbol']==genename]
transdf = transdf.iloc[-13:,:]
############################
#transdf = transdf[IRlist]
############################
pre = transdf.iloc[:,1::2]
post = transdf.iloc[:,0::2]

predf = pd.DataFrame({'mean':pre.mean(axis=1), 'Treatment':'Pre', 'transcript':transdf.index})
postdf = pd.DataFrame({'mean':post.mean(axis=1), 'Treatment':'Post','transcript':transdf.index})
figuredf = pd.concat([predf,postdf])


plt.figure(figsize=(4,5))
#sns.set_style("whitegrid")
sns.set_theme(style='white',palette='pastel')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 12,    # x축 틱 라벨 글꼴 크기
'ytick.labelsize': 12,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 11,
'legend.title_fontsize': 11, # 범례 글꼴 크기
'figure.titlesize': 14    # figure 제목 글꼴 크기
})
sns.set_theme(style='ticks', palette='pastel', font_scale=1)

ax = sns.barplot(y='mean', x='transcript', data=figuredf, 
                hue='Treatment',
            palette='vlag'
            #meanprops = {"marker":"o", "color":"blue","markeredgecolor": "blue", "markersize":"10"}
            )
ax.set(xticklabels=[]) 
ax.set_ylabel('mean TU')
plt.title('BRCA1', fontsize=14)
plt.xlabel('')
sns.despine()
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/fig1/BRCA1_transcript_barplot.png', dpi=300)
plt.show()

#%%
### transcript boxplot ###

#transname = 'MSTRG.50413.1-BRCA1'
#transname = 'ENST00000467274.1-BRCA1'
#transname = 'MSTRG.67301.64-ASCC2'
#transname = 'MSTRG.49834.247-BRIP1'
#transname = 'ENST00000404276.1-CHEK2'
#transname = 'ENST00000456369.1-CHEK2'
transname = 'ENST00000348295.3-CHEK2'
#transname = 'MSTRG.67243.46-CHEK2'
#transname = 'ENST00000555055.1-XRCC3'


TU_new = TU.iloc[:,:-1]
prepost = ['Post','Pre']*40
transdf = pd.DataFrame({'TU':TU_new.loc[transname,:], 'Treatment':prepost, 'transcript':transname})
##################################
#transdf = transdf.loc[IRlist,:]
##################################
transdf['sampleid'] = transdf.index.str[:-4]

plt.figure(figsize=(3,4))
#sns.set_style("whitegrid")
sns.set_theme(style='ticks',palette='pastel')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 12,    # x축 틱 라벨 글꼴 크기
'ytick.labelsize': 12,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 11,
'legend.title_fontsize': 11, # 범례 글꼴 크기
'figure.titlesize': 14    # figure 제목 글꼴 크기
})

ax = sns.boxplot(y='TU', x='Treatment', data=transdf, 
            showfliers=False, order=['Pre','Post'], color='#FFD92E'
            #palette='vlag'
            #meanprops = {"marker":"o", "color":"blue","markeredgecolor": "blue", "markersize":"10"}
            )
#ax.set_ylim([0,0.01])
ax.set_ylabel('TU')
plt.title(transname)
sns.despine()

from statannot import add_stat_annotation
add_stat_annotation(ax, data=transdf, x='Treatment', y='TU',
                    box_pairs=[("Pre", "Post")], #comparisons_correction=None,
                    test='Wilcoxon',  text_format='star', loc='inside', fontsize=20)

for p in transdf['sampleid'].unique():
    subset = transdf[transdf['sampleid']==p]
    x = [list(subset['Treatment'])[1], list(subset['Treatment'])[0]]
    y = [list(subset['TU'])[1], list(subset['TU'])[0]]
    plt.plot(x,y, marker="o", markersize=4, color='grey', linestyle="--", linewidth=0.7)
    
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/fig1/CHEK2_mint_boxplot.png', bbox_inches='tight', dpi=300)
plt.show()
# %%
