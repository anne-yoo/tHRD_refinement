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
'legend.fontsize': 12,
'legend.title_fontsize': 12, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})
sns.set_style("ticks")
# %%
majiq = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/MAJIQ/output_condition_1/deltapsi/Responder-Nonresponder.deltapsi.tsv', sep='\t',comment="#")
# %%
lsv = majiq[(majiq['probability_changing']>0.95)]  #& (majiq['dpsi_mean'].abs()>0.1)
# %%
val_clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2025clinical/83_POLO_clinicalinfo.txt', sep='\t', index_col=0)

# %%
# 44 제외 
majiqheatmap =  pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/MAJIQ/output_condition_1/modulize/heatmap.tsv', sep='\t',comment="#")
majiqsummary =  pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/MAJIQ/output_condition_1/modulize/summary.tsv', sep='\t',comment="#")

# %%
######^^ IsoformSwitchAnalyzer REsult ############33
gDUT = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/IsoformSwitchAnalyer/isoformswitchanalyzer_SatuRn_gDUTlist.txt', sep='\t', index_col=0)

gDUT = gDUT.reset_index()

pcut = gDUT[['gene_name']]
pcut = pcut.drop_duplicates()
glist = pcut.squeeze().str.strip().to_list()
print(len(glist))
#%%
import gseapy as gp
enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                gene_sets=['GO_Biological_Process_2021',
                           #'Reactome_2022'
                           #'GO_Biological_Process_2018','GO_Biological_Process_2023','Reactome_2022'
                           ], 
                organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                outdir=None, # don't write to disk
)

enrresult = enr.results.sort_values(by=['Adjusted P-value']) 


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
file['Term'] = file['Term'].str.rsplit(" ",n=1).str[0]
#file = file[~file['Term'].str.contains('mitochondrial')]
#file = file.iloc[:10,:]
file = file[file['Adjusted P-value']<0.1]
file['Adjusted P-value'] = -np.log10(file['Adjusted P-value'])

plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 12,    # x축 틱 라벨 글꼴 크기
'ytick.labelsize': 12,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 13,
'legend.title_fontsize': 13, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})
plt.figure(figsize=(4,5))
sns.set_style("whitegrid")
scatter = sns.scatterplot(
    data=file, x='Adjusted P-value', y='Term', hue='per', palette='coolwarm', edgecolor=None, legend=False, s=80
)
plt.xlabel('-log10(FDR)')
plt.ylabel('')
#plt.yticks(fontsize=13)
#plt.xscale('log')  # Log scale for better visualization

# Expanding the plot layout to make room for GO term labels
plt.gcf().subplots_adjust(left=0.4)
#plt.gcf().subplots_adjust(right=0.8)

# Creating color bar
norm = plt.Normalize(file['per'].min(), file['per'].max())
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
sm.set_array([])

# Displaying color bar
cbar = plt.colorbar(sm)
#cbar.set_label('Overlap Percentage')A
# plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/fig1/GO_DEG_AR_fc1.pdf', dpi=300, bbox_inches='tight')
plt.show()
## %%

# %%
DDRlist = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/DDR_genelist_whole.txt', sep='\t')
DDRcorelist = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/DDR_genelist_core.txt', sep='\t')

from collections import OrderedDict

rename_map_1 = {'Nucleotide Excision Repair (NER) - includes TC-NER and GC-NER': 'Nucleotide Excision Repair (NER)'}
rename_map_2 = {'Nucleotide Excision Repair (NER, including TC-NER and GC-NER))': 'Nucleotide Excision Repair (NER)', 
                'Homologous Recomination (HR)': 'Homologous Recombination (HR)',}

ddr_gene = {}

for col in DDRlist.columns:
    # NaN 제거하고 리스트로 변환
    genes = DDRlist[col].dropna().tolist()
    ddr_gene[col] = genes

ddr_genelist = OrderedDict()
for k, v in ddr_gene.items(): 
    new_k = rename_map_1.get(k, k)
    ddr_genelist[new_k] = v

ddr_coregene = {}

for col in DDRcorelist.columns:
    # NaN 제거하고 리스트로 변환
    genes = DDRcorelist[col].dropna().tolist()
    ddr_coregene[col] = genes

ddr_coregenelist = OrderedDict()
for k, v in ddr_coregene.items():
    new_k = rename_map_2.get(k, k)
    ddr_coregenelist[new_k] = v
    
# %%
ddrgene = [item for sublist in ddr_genelist.values() for item in sublist]
# %%
studyoff = {'PL-OV-P005','PL-OV-P021','PL-OV-P044','PL-OV-P037'}



# %%
isa = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/IsoformSwitchAnalyer/condition1_isoformswitchanalyzer_SatuRn_isoformlist.txt', sep='\t')

glist = list(set(isa[isa['isoform_switch_q_value']<0.1]['gene_name']))

import gseapy as gp
enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                gene_sets=['GO_Biological_Process_2023',
                           #'Reactome_2022'
                           #'GO_Biological_Process_2018','GO_Biological_Process_2023','Reactome_2022'
                           ], 
                organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                outdir=None, # don't write to disk
)

enrresult = enr.results.sort_values(by=['Adjusted P-value'])

# %%
