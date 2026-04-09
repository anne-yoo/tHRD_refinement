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
## DEG comparison #########################
deseq2_AR = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/DEGDUT/AR_prepost_DEG_DESeq2_raw.csv', sep=',', index_col=0)
wil_AR = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/DEGDUT/AR_Wilcoxon_DEGresult_FC.txt', sep='\t', index_col=0)
deseq2_IR = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/DEGDUT/IR_prepost_DEG_DESeq2_raw.csv', sep=',', index_col=0)
wil_IR = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/DEGDUT/IR_Wilcoxon_DEGresult_FC.txt', sep='\t', index_col=0)

# %%
from venn import venn 
# venn diagram plot
set1 = set(wil_IR[wil_IR['p_value']<0.01].index)
set2 = set(deseq2_IR[deseq2_IR['padj']<0.05].index)
labels = {'Wilcoxon':set1,'DESeq2':set2}
venn(labels)
# %%
from venn import venn 
# venn diagram plot
set1 = set(wil_IR[wil_IR['p_value']<0.05].index)
set2 = set(deseq2_IR[deseq2_IR['padj']<0.05].index)
labels = {'Wilcoxon':set1,'DESeq2':set2}
venn(labels)
# %%
t_info= pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/gencode_transcript_type.txt', sep='\t', index_col=0)
t_info = t_info[['genename','gene_type']]
t_info.drop_duplicates(inplace=True)
pcgene = set(t_info[t_info['gene_type']=='protein_coding']['genename'])

set1 = set(wil_IR[wil_IR['p_value']<0.05].index) & pcgene
set2 = set(deseq2_IR[deseq2_IR['padj']<0.05].index) & pcgene
labels = {'Wilcoxon':set1,'DESeq2':set2}
venn(labels)

# %%
