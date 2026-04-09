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


sns.set(font = 'Arial')
sns.set_style("whitegrid")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#%%
#^ 0. Go enrichment

import gseapy as gp

###*enrichR GO enrichment analysis

TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_TU.txt', sep='\t', index_col=0)
with open('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUTgene/AR_DUTgenelist.txt') as file:
    gDUTlist = [line.rstrip() for line in file]
major = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = major[major['type']=='major']['gene_ENST'].to_list()
ARdut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t', index_col=0)
IRdut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/nonresponder_stable_DUT_Wilcoxon.txt', sep='\t', index_col=0)
ARdut = ARdut[(ARdut['p_value'] < 0.05) & (ARdut['log2FC'].abs() > 1.5)]
IRdut = IRdut[(IRdut['p_value'] < 0.05) & (IRdut['log2FC'].abs() > 1.5)]

ARdutlist = ARdut.index.to_list()
IRdutlist = IRdut.index.to_list()

#%%
#####
new_df = TU.loc[TU.index.isin(ARdutlist),:]
#####

new_df['gene'] = new_df.index.str.split("-",1).str[1]
#new_df = new_df.loc[new_df['gene'].isin(gDUTlist),:]
new_df = new_df.reset_index()
new_df = new_df.groupby('gene').max()
new_df = new_df.reset_index().set_index('gene')

new_df.ndex = new_df.index.str.upper()

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
responder = sampleinfo.loc[(sampleinfo['response']==1),'sample_full'].to_list()
nonresponder = sampleinfo.loc[(sampleinfo['response']==0),'sample_full'].to_list()


#%%
############## AR / IR ###################
input = new_df[responder]
##########################################
pre = input.iloc[:, 1::2]
post = input.iloc[:, 0::2]

fc = post.mean(axis=1)/pre.mean(axis=1)
log2FC = np.log2(fc)

df_rank = pd.DataFrame({'Gene':list(input.index), 'Rank':log2FC.values})
#df_rank = df_rank.loc[df_rank['Gene'].isin(gDUTlist),:]
df_rank = df_rank[~df_rank.isin([np.nan, np.inf, -np.inf]).any(1)]
df_rank = df_rank.sort_values(by='Rank', ascending=True)
df_rank.columns = ['Gene', 'Rank']
# %%


# %%
# GSEA 실행
gsea_res = gp.prerank(rnk=df_rank, gene_sets='GO_Biological_Process_2021', seed=1024, permutation_num=1000)

# 결과 저장 리스트
out_list = []
for term, result in gsea_res.results.items():
    p = result.get('pval', np.nan)
    fdr = result.get('fdr', np.nan)
    nes = result.get('nes', np.nan)
    es = result.get('es', np.nan)
    gene = result.get('lead_genes', '')
    hits = result.get('hits', [])  
    res = result.get('RES', [])   

    out_list.append([term, p, fdr, nes, es, gene, hits, res]) 

#%%
# DataFrame 생성 (gseaplot 실행할 때 필요한 컬럼 포함)
df_out = pd.DataFrame(out_list, columns=['Term', 'p_value', 'fdr', 'nes', 'es', 'gene', 'hits', 'res'])
df_out = df_out.sort_values('fdr').reset_index(drop=True)
df_out = df_out[df_out['p_value']<0.05]

#%%
# 첫 번째 GO term 선택 (혹은 원하는 term 선택)
term_to_plot = df_out.iloc[26]['Term']

# gseaplot 실행
from gseapy.plot import gseaplot

gseaplot(rank_metric=gsea_res.ranking, 
         term=term_to_plot, 
         nes=df_out[df_out['Term'] == term_to_plot]['nes'].values[0], 
         pval=df_out[df_out['Term'] == term_to_plot]['p_value'].values[0], 
         fdr=df_out[df_out['Term'] == term_to_plot]['fdr'].values[0], 
         es=df_out[df_out['Term'] == term_to_plot]['es'].values[0], 
         hits=df_out[df_out['Term'] == term_to_plot]['hits'].values[0], 
         RES=df_out[df_out['Term'] == term_to_plot]['res'].values[0]) 

# %%
