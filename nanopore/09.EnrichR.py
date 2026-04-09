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
import textwrap


sns.set(font = 'Arial')
sns.set_style("whitegrid")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

#%%
##### DO enrichR ! #######
import gseapy as gp
trans = pd.read_csv('/home/jiye/jiye/nanopore/FINALDATA/matched_transcript_TPM.txt', sep='\t', index_col=0)
trans = trans[trans.apply(lambda x: (x != 0).sum(), axis=1) >= trans.shape[1]*0.2]
# tumor = trans.iloc[:,1::2]
# normal = trans.iloc[:,0::2]
# tumor = tumor[tumor.apply(lambda x: (x != 0).sum(), axis=1) >= tumor.shape[1]*0.2]
# normal = normal[normal.apply(lambda x: (x != 0).sum(), axis=1) >= normal.shape[1]*0.2]
# tumorlist = tumor.index
# normallist = normal.index
# results = list(set(tumorlist)-set(normallist))
# glist = list(set([item.split('-', 1)[1] for item in results]))

#with open("/home/jiye/jiye/nanopore/202411_analysis/isoformswitchanalyzer/tumorspcificlist.txt", "w") as f:
#    f.write(",".join(map(str, results)))

#%%
###^ check gDET vs. DEG #####
translist = trans.index.to_list()
det = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/DESeq2/deseq2_det.txt', sep='\t', index_col=0)
det = det.loc[det.index.isin(translist),:]
det = det.loc[(np.abs(det['log2FoldChange'])>1.5) & (det['padj']<0.05),:]
det['genename'] = det.index.str.split("-",1).str[1]
det['trans'] = det.index.str.split("-",1).str[0]
gdetlist = set(det['genename'])

t = det.loc[det['log2FoldChange']>1.5,:]
n = det.loc[det['log2FoldChange']<-1.5,:]

deg = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/DESeq2/deseq2_deg.txt', sep='\t')
deg = deg.loc[(np.abs(deg['log2FoldChange'])>1.5) & (deg['padj']<0.05),:]
deglist = set(deg['gene_name'])

#%%
####^^ check NMD #####
nmdlist = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/TranSuite/NMDlist.txt')
nmdlist = nmdlist['Transcript_ID'].to_list()

print(len(set(det['trans']).intersection(set(nmdlist))))
print(len(set(t['trans']).intersection(set(nmdlist))))
print(len(set(n['trans']).intersection(set(nmdlist))))


#%%
results = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/isoformswitchanalyzer/gDUTlist.txt', sep='\t')
glist = results['gene_name'].to_list()


#with open("/home/jiye/jiye/nanopore/202411_analysis/isoformswitchanalyzer/gDUTlist_forSTRING.txt", "w") as f:
#    f.write(",".join(map(str, glist)))

#%%
translist = trans.index.to_list()
novellist = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/TSSanalysis/sqt_output_classification.txt', sep='\t')
novellist = novellist[(novellist['structural_category']=='novel_in_catalog') |(novellist['structural_category']=='novel_not_in_catalog') ]['isoform'].to_list()

results = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/DESeq2/deseq2_det.txt', sep='\t')

results['genename'] = results['transcript_id'].str.split('-',1).str[1]
results['transid'] = results['transcript_id'].str.split('-',1).str[0]

results = results[results['transcript_id'].isin(translist)] ##filtering
#results = results[~results['transid'].isin(novellist)] ##onlynovel

#glist = list(set(list(results.loc[(np.abs(results['log2FoldChange'])>2) & (results['padj']<0.01),:]['genename'])))
glist = list(set(list(results.loc[(results['log2FoldChange']>1.5) & (results['padj']<0.05),:]['genename'])))
#glist = list(set(list(results.loc[(results['log2FoldChange']>1.5) & (results['padj']<0.05),:]['gene_name'])))
print(len(glist))

#%%
import gseapy as gp
enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                gene_sets=[
                    #'GO_Biological_Process_2023',
                            #'Reactome_2022',
                            'GO_Biological_Process_2018',
                            ], 
                organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                outdir=None, # don't write to disk
)

enrresult = enr.results.sort_values(by=['Adjusted P-value']) 

#enrresult.to_csv('/home/jiye/jiye/nanopore/202411_analysis/DESeq2/DET_Whole_enrichment.txt',index=False, sep='\t')

# ########### enrichR output file #########
# file = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/rMATS/individual/enrichr/onlyIR_enrichr.txt', sep='\t')
# #########################################

#%%
file = enrresult
def string_fraction_to_float(fraction_str):
    numerator, denominator = fraction_str.split('/')
    return float(numerator) / float(denominator)

file['per'] = file['Overlap'].apply(string_fraction_to_float)

file = file.sort_values('Adjusted P-value')
#file = file.sort_values(by='Combined Score', ascending=False)
##remove mitochondrial ##
file['Term'] = file['Term'].str.rsplit(" ",1).str[0]
#file = file[~file['Term'].str.contains('mitochondrial')]
#file = file.iloc[:10,:]

file['Adjusted P-value'] = -np.log10(file['Adjusted P-value'])
file = file[file['Adjusted P-value']>3]
#%%
# plt.rcParams["font.family"] = "Arial"
# plt.rcParams.update({
# 'axes.titlesize': 13,     # 제목 글꼴 크기
# 'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
# 'xtick.labelsize': 12,    # x축 틱 라벨 글꼴 크기
# 'ytick.labelsize': 12,    # y축 틱 라벨 글꼴 크기
# 'legend.fontsize': 12,
# 'legend.title_fontsize': 12, # 범례 글꼴 크기
# 'figure.titlesize': 15    # figure 제목 글꼴 크기
# })

sns.set_style("ticks")

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
df = file.iloc[::-1]
df['Gene_count'] = df['Overlap'].str.split('/').str[0].astype(int)

go_terms = df['Term']
gene_counts = df['Gene_count']
adjusted_p_values = df['Adjusted P-value']

# # Add larger offsets to avoid overlap
# y_positions = list(range(len(go_terms)))  # Default positions
# y_offsets = [
#     pos + 0.4 if term == "collagen fibril organization" else pos - 0.4 if term == "extracellular matrix organization" else pos
#     for pos, term in zip(y_positions, go_terms)
# ]

# # Create the lollipop chart
# fig, ax = plt.subplots(figsize=(6, 2))  # Increase height for better spacing

# for y_pos, term, count, color in zip(y_offsets, go_terms, adjusted_p_values, colors):
#     ax.plot([0, count], [y_pos, y_pos], color=color, lw=3)
#     ax.scatter(count, y_pos, color=color, s=60, zorder=3)

# # Adjust the y-ticks and their positions
# ax.set_yticks(y_offsets)  # Use the adjusted positions for ticks
# ax.set_yticklabels(go_terms, fontsize=10)  # Keep the corresponding terms
# ax.set_xlabel("Adjusted P-value", fontsize=12)

# # Add a colorbar
# plt.colorbar(sm, label='Gene Count', ax=ax)

# # Add padding to prevent clipping
# plt.tight_layout(pad=0.2)
# sns.despine()
# #plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/figures/Novel_TumorUp_DET_GOenrichment.pdf', dpi=300, bbox_inches='tight')
# plt.show()

#%%

# Normalize the adjusted p-values for coloring
gene_count_min = gene_counts.min()
gene_count_max = gene_counts.max()
norm = Normalize(vmin=gene_count_min - (gene_count_max - gene_count_min)/2, vmax=gene_count_max)
norm = Normalize(vmin=0, vmax=gene_count_max)

sm = ScalarMappable(cmap='Blues', norm=norm)
colors = sm.to_rgba(gene_counts)

# Create the lollipop chart
fig, ax = plt.subplots(figsize=(8, 5))

for i, (term, count, color) in enumerate(zip(go_terms, adjusted_p_values, colors)):
    ax.plot([0, count], [i, i], color=color, lw=3)
    ax.scatter(count, i, color=color, s=60, zorder=3)

# Customize the axes
ax.set_yticks(range(len(go_terms)))
ax.set_yticklabels(go_terms)
ax.set_xlabel("Adjusted P-value", fontsize=12)
#ax.set_title("Up-regulated transcripts in tumor", fontsize=13)
plt.colorbar(sm, label='Gene Count', ax=ax)

# Adjust layout
plt.tight_layout()
sns.despine()
plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/figures/TumorUp_DET_GOenrichment.pdf', dpi=300, bbox_inches='tight')
plt.show()


#%%
ecm_list = df.loc[df['Term']=='extracellular matrix organization','Genes'].str.split(';').explode().tolist()

#%%
trans = pd.read_csv('/home/jiye/jiye/nanopore/FINALDATA/matched_transcript_TPM.txt', sep='\t', index_col=0)
trans['genename'] = trans.index.str.split('-',1).str[1]
trans.index = trans.index.str.split('-',1).str[0]
trans = trans[['genename']]
dutlist = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/isoformswitchanalyzer/DUTresult.txt', sep='\t')
merged = pd.merge(trans, dutlist,left_index=True, right_on='isoform_id',how='inner')
# %%
gDUT_list = list(set(merged['genename']))
# %%
both = set(gDUT_list).intersection(set(glist))
# %%
