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

plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 13,    # x축 틱 라벨 글꼴 크기

'ytick.labelsize': 13,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 13,
'legend.title_fontsize': 13, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})
sns.set_style("white")

# %%
#########^ DUT #############
trans = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_TU.txt', sep='\t', index_col=0)
trans = trans.drop(['OM/OS'])

last_row = trans.iloc[-1:]  # Selects the last row as a DataFrame
data_to_filter = trans.iloc[:-1]  # Selects all rows except the last one

minimum_samples = int(data_to_filter.shape[1] * 0.4) ########## 40% threshold

# Apply filtering based on non-zero values treated as "1"
filtered_data = data_to_filter[data_to_filter.apply(lambda x: (x != 0).sum(), axis=1) >= minimum_samples]

# Concatenate the filtered data with the last row
trans = pd.concat([filtered_data, last_row])

trans.loc['group',:] = trans.loc['group',:].astype('float')
trans.loc['group',:] = trans.loc['group',:].astype('int')
trans['Gene Symbol'] = trans.index.str.split("-",1).str[1]

##^^^^^^^ gDUT vs. DEG ######################
dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/MW_DUTresult_FC_stable.txt', sep='\t', index_col=0)
deg = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/MW_DEGresult_FC.txt', sep='\t', index_col=0)

deglist = set(deg.loc[(deg['p_value']<0.05) & (np.abs(deg['log2FC'])>0.4)]['Gene Symbol'])
gDUTlist = set(dut.loc[(dut['p_value']<0.05) & (np.abs(dut['log2FC'])>1)]['Gene Symbol'])

#%%
from matplotlib_venn import venn2

plt.figure(figsize=(6,6))
sns.set_style("white")
vd2 = venn2([deglist, gDUTlist],set_labels=('DEG', 'gDUT'),set_colors=('#F8CF6A','#74B2D4'), alpha=0.8)

###^^^^^^^^^^^^^^^ GO enrichment ####################3

import gseapy as gp

glist = list(gDUTlist)
enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                gene_sets=['GO_Biological_Process_2018',], 
                #gene_sets=['GO_Biological_Process_2021',], 
                organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                outdir=None, # don't write to disk
)

enrresult = enr.results.sort_values(by=['Adjusted P-value']) 

file = enrresult
#file['Term'] = file['Term'].str.rsplit(" ",1).str[0]
file = file[file['Adjusted P-value']<0.1]
file['FDR'] = -np.log10(file['Adjusted P-value'])
file = file[(file["Term"].str.contains("homologous recombination", case=False)) | (file["Term"].str.contains("DNA Damage", case=False)) | (file["Term"].str.contains("cell cycle", case=False)) | (file["Term"].str.contains("Signaling by WNT in cancer", case=False)) | (file["Term"].str.contains("PI3K/AKT Signaling in Cancer", case=False)) | (file["Term"].str.contains("double strand break", case=False))| (file["Term"].str.contains("DNA repair", case=False))]

parplist = file['Term']
parplist = enrresult[enrresult['Term'].isin(parplist)]
parpgene = list(set([gene for sublist in parplist['Genes'].str.split(';') for gene in sublist]))


dut_filtered = dut.loc[(dut['p_value']<0.05) & (np.abs(dut['log2FC'])>1)]

#%%
majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = majorminor[majorminor['type']=='major']['gene_ENST'].to_list()
minorlist = majorminor[majorminor['type']=='minor']['gene_ENST'].to_list()

#%%
#dutlist = dut_filtered.index.to_list() #every DUT
#dutlist = dut_filtered[dut_filtered['Gene Symbol'].isin(parpgene)].index.to_list() #PARP-related DUT
#gdutlist = dut_filtered[dut_filtered['Gene Symbol'].isin(parpgene)]['Gene Symbol'].to_list() #PARP-related gDUT's every major transcripts
#dutlist = trans[trans['Gene Symbol'].isin(gdutlist)].index.to_list()
dutlist = ['ENST00000462785.1-ANAPC1', 'ENST00000399921.1-BACH1',
'MSTRG.64434.27-BACH1', 'ENST00000438077.1-BCL6',
'MSTRG.76986.4-BOD1L1', 'MSTRG.70493.63-BRD4', 'MSTRG.107181.3-CASP2',
'MSTRG.107181.12-CASP2', 'MSTRG.40447.94-CCP110',
'MSTRG.118691.18-CDC14B', 'MSTRG.50566.130-CDC27',
'MSTRG.97471.5-CDKN1A', 'MSTRG.36583.8-CEP152',
'MSTRG.21311.136-CEP164', 'ENST00000495702.1-CEP41',
'ENST00000369258.4-CHD1L', 'MSTRG.25537.1-CNOT2',
'ENST00000482247.1-CNOT3', 'MSTRG.44564.41-CRLF3',
'MSTRG.44564.100-CRLF3', 'MSTRG.47423.131-CSNK1D',
'MSTRG.19857.22-CUL5', 'MSTRG.18408.57-DDB1',
'ENST00000392508.2-DYNLL1', 'MSTRG.92444.257-ERCC1',
'MSTRG.8730.1-EXO1', 'MSTRG.125415.9-FMR1', 'ENST00000433809.1-GNL1',
'MSTRG.49437.181-GRB2', 'MSTRG.39996.125-GSPT1', 'MSTRG.35807.1-GTF2H3',
'MSTRG.48052.10-HAUS1', 'MSTRG.45506.1-HIC1', 'MSTRG.35934.45-HSP90AA1',
'ENST00000543739.1-IGHMBP2', 'MSTRG.38105.141-INO80',
'MSTRG.35661.496-KNTC1', 'MSTRG.35661.564-KNTC1', 'MSTRG.98513.1-MCM9',
'MSTRG.119054.76-MELK', 'MSTRG.1112.3-MTOR', 'ENST00000368395.1-MUC1',
'MSTRG.58505.23-NABP1', 'ENST00000342673.5-NDE1',
'MSTRG.104799.39-NUDT1', 'MSTRG.60602.7-ORC2',
'ENST00000493653.1-PDS5B', 'MSTRG.99356.16-PLAGL1',
'MSTRG.39376.29-PML', 'MSTRG.21640.21-POLA2', 'MSTRG.97751.202-POLH',
'MSTRG.22049.10-PPME1', 'MSTRG.92882.30-PPP2R1A',
'ENST00000264639.4-PSMD3', 'MSTRG.119195.31-PSMD5',
'ENST00000539365.1-PTPN6', 'MSTRG.77282.347-RNF168',
'ENST00000539417.2-RPAIN', 'ENST00000571558.1-RPAIN',
'MSTRG.13789.3-SIRT1', 'ENST00000344921.6-SMARCB1',
'MSTRG.57826.59-SPAST', 'ENST00000449210.1-SPDYA',
'ENST00000519141.1-SPIDR', 'MSTRG.44461.251-TAOK1',
'ENST00000416441.2-TAOK2', 'ENST00000541186.1-TAOK3',
'ENST00000532437.1-TNKS1BP1', 'MSTRG.77574.395-UBA7',
'MSTRG.35853.44-UBC', 'ENST00000452012.1-UBE2E1',
'MSTRG.113884.25-UBR5', 'MSTRG.64424.11-USP16',
'ENST00000559771.1-USP3', 'MSTRG.3691.2-VAV3', 'MSTRG.119002.2-VCP',
'MSTRG.11771.50-WAC']

majordutlist = list(set(dutlist).intersection(set(majorlist))) ## major vs. minor
group = np.array(trans.iloc[-1,:-1])
group = list(group.astype(int))

############################
R = trans.loc[majordutlist,trans.iloc[-1,:]==1]
R = R.apply(pd.to_numeric, errors='coerce')
NR = trans.loc[majordutlist,trans.iloc[-1,:]==0]
NR = NR.apply(pd.to_numeric, errors='coerce')

Rdf = pd.DataFrame({'mean TU':R.mean(axis=1), 'Response':'R', 'transcript':R.index})
NRdf = pd.DataFrame({'mean TU':NR.mean(axis=1), 'Response':'NR','transcript':R.index})
figuredf = pd.concat([Rdf,NRdf])


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
plt.figure(figsize=(4,5))
ax = sns.boxplot(y='mean TU', x='Response', data=figuredf, order=['R','NR'],
            showmeans=False, palette='vlag', showfliers=True
            #meanprops = {"marker":"o", "color":"blue","markeredgecolor": "blue", "markersize":"10"}
            )
#ax.set_ylim([0,0.2])

from statannot import add_stat_annotation
add_stat_annotation(ax, data=figuredf, x='Response', y='mean TU',
                    box_pairs=[("R", "NR")],
                    test='Mann-Whitney',  text_format='star', loc='inside', fontsize=14, comparisons_correction=None)
plt.title('major DUT', fontsize=14)
plt.ylabel('mean TU')
sns.despine()


for p in figuredf['transcript'].unique():
    subset = figuredf[figuredf['transcript']==p]
    x = [list(subset['Response'])[1], list(subset['Response'])[0]]
    y = [list(subset['mean TU'])[1], list(subset['mean TU'])[0]]
    plt.plot(x,y, marker="o", markersize=4, color='grey', linestyle="--", linewidth=0.7)
    
    
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/fig1/CHEK2_DEG_boxplot.png', dpi=300, bbox_inches='tight')

plt.show()

# %%
# features_66 = ['ENST00000457016.1-APC', 'ENST00000257430.4-APC',
# 'ENST00000482687.1-APTX', 'MSTRG.62679.6-AURKA',
# 'ENST00000395913.3-AURKA', 'ENST00000404251.1-BAZ1B',
# 'MSTRG.19662.15-BIRC2', 'ENST00000318407.3-BOK',
# 'MSTRG.49834.247-BRIP1', 'ENST00000310447.5-CASP2',
# 'ENST00000310232.6-CCDC13', 'ENST00000263201.1-CDC45',
# 'ENST00000349780.4-CDK5RAP2', 'ENST00000373711.2-CDKN1A',
# 'ENST00000399334.3-CEP152', 'ENST00000376597.4-CEP78',
# 'ENST00000404276.1-CHEK2', 'ENST00000308987.5-CKS1B',
# 'ENST00000375440.4-CUL4A', 'ENST00000531427.1-CUL5',
# 'MSTRG.25406.5-DYRK2', 'ENST00000433017.1-FIGNL1',
# 'MSTRG.104113.4-FIGNL1', 'ENST00000342628.2-FOXM1',
# 'MSTRG.49465.142-GRB2', 'ENST00000545134.1-HELB',
# 'ENST00000444376.2-HNRNPU', 'MSTRG.65357.56-ID1',
# 'ENST00000373135.3-L3MBTL1', 'ENST00000505397.1-LIN54',
# 'ENST00000392915.1-MAP3K19', 'ENST00000344908.5-MAP3K2',
# 'ENST00000375321.1-MAP3K8', 'MSTRG.11829.3-MAP3K8',
# 'ENST00000382948.5-MBD1', 'MSTRG.12458.189-MCL1',
# 'ENST00000370782.2-MMS19', 'ENST00000233027.5-NEK4',
# 'MSTRG.91105.1-NPM1', 'MSTRG.114270.18-NSMCE2', 'MSTRG.9436.8-NUCKS1',
# 'MSTRG.18587.4-OTUB1', 'MSTRG.92133.10-PAK4', 'MSTRG.92133.3-PAK4',
# 'MSTRG.1704.78-PHACTR4', 'ENST00000382865.1-POLN',
# 'ENST00000261207.5-PPP1R12A', 'ENST00000498508.2-PROX1',
# 'ENST00000442510.2-PTPRC', 'ENST00000575111.1-RNF167',
# 'ENST00000572008.1-RPS15A', 'MSTRG.52876.4-RPS27A',
# 'MSTRG.17113.1-SFR1', 'ENST00000394224.3-SIPA1', 'MSTRG.51515.3-SMC6',
# 'ENST00000402989.1-SMC6', 'ENST00000445403.1-SRC',
# 'ENST00000453269.2-TAF6', 'ENST00000206765.6-TGM1',
# 'ENST00000219476.3-TSC2', 'ENST00000380122.5-TXLNG',
# 'ENST00000539466.1-USP47', 'MSTRG.11771.23-WAC', 'MSTRG.11771.35-WAC',
# 'ENST00000355640.3-XIAP', 'ENST00000555055.1-XRCC3'] ## from deltaTU model (resistance cohort) after RFE

# features_73 = ['ENST00000555055.1-XRCC3',
# 'MSTRG.11829.3-MAP3K8','MSTRG.17113.1-SFR1',
# 'ENST00000402989.1-SMC6','ENST00000349780.4-CDK5RAP2','ENST00000394224.3-SIPA1',
# 'MSTRG.12458.189-MCL1','ENST00000505397.1-LIN54',
# 'ENST00000373711.2-CDKN1A','MSTRG.114270.18-NSMCE2',
# 'MSTRG.92133.10-PAK4','ENST00000399334.3-CEP152',
# 'ENST00000575111.1-RNF167','ENST00000396355.1-NDE1','MSTRG.65357.56-ID1','MSTRG.11771.35-WAC',
# 'ENST00000498508.2-PROX1','ENST00000482687.1-APTX',
# 'ENST00000375321.1-MAP3K8','ENST00000539466.1-USP47',
# 'ENST00000540257.1-TUBB4A','ENST00000401393.3-INO80',
# 'MSTRG.92133.3-PAK4','ENST00000545134.1-HELB',
# 'ENST00000355640.3-XIAP','ENST00000321464.5-ZBTB38',
# 'ENST00000308987.5-CKS1B','ENST00000342628.2-FOXM1',
# 'ENST00000206765.6-TGM1','ENST00000382865.1-POLN',
# 'MSTRG.25406.5-DYRK2','ENST00000392915.1-MAP3K19',
# 'MSTRG.62156.213-AP5S1','MSTRG.17041.3-FBXL15',
# 'MSTRG.49834.247-BRIP1','ENST00000318407.3-BOK',
# 'MSTRG.49465.142-GRB2',
# 'ENST00000457016.1-APC',
# 'ENST00000433017.1-FIGNL1',
# 'MSTRG.62679.6-AURKA',
# 'MSTRG.11771.23-WAC',
# 'MSTRG.51515.3-SMC6',
# 'ENST00000531427.1-CUL5',
# 'ENST00000261207.5-PPP1R12A',
# 'MSTRG.18587.4-OTUB1',
# 'ENST00000382948.5-MBD1',
# 'ENST00000373135.3-L3MBTL1',
# 'MSTRG.52876.4-RPS27A',
# 'ENST00000404276.1-CHEK2',
# 'ENST00000370782.2-MMS19',
# 'ENST00000310447.5-CASP2',
# 'ENST00000445403.1-SRC',
# 'ENST00000380122.5-TXLNG',
# 'ENST00000572008.1-RPS15A',
# 'ENST00000219476.3-TSC2',
# 'ENST00000233027.5-NEK4',
# 'MSTRG.1704.78-PHACTR4',
# 'ENST00000376597.4-CEP78',
# 'MSTRG.104113.4-FIGNL1',
# 'MSTRG.19662.15-BIRC2',
# 'ENST00000310232.6-CCDC13',
# 'ENST00000257430.4-APC',
# 'ENST00000444376.2-HNRNPU',
# 'ENST00000344908.5-MAP3K2',
# 'ENST00000263201.1-CDC45',
# 'ENST00000361265.4-AJUBA',
# 'ENST00000453269.2-TAF6',
# 'ENST00000404251.1-BAZ1B',
# 'MSTRG.91105.1-NPM1',
# 'MSTRG.9436.8-NUCKS1',
# 'ENST00000375440.4-CUL4A',
# 'ENST00000395913.3-AURKA',
# 'ENST00000442510.2-PTPRC']

# features_50=['ENST00000391831.1-AKT1S1', 'ENST00000482687.1-APTX',
# 'ENST00000404251.1-BAZ1B', 'MSTRG.49834.247-BRIP1',
# 'ENST00000302759.6-BUB1', 'ENST00000310447.5-CASP2',
# 'ENST00000310232.6-CCDC13', 'ENST00000308108.4-CCNE2',
# 'ENST00000349780.4-CDK5RAP2', 'MSTRG.98879.3-CENPW',
# 'ENST00000368328.4-CENPW', 'ENST00000376597.4-CEP78',
# 'ENST00000404276.1-CHEK2', 'ENST00000308987.5-CKS1B',
# 'MSTRG.14904.57-CTBP2', 'MSTRG.14904.52-CTBP2',
# 'ENST00000531427.1-CUL5', 'ENST00000342628.2-FOXM1',
# 'ENST00000454366.1-GTSE1', 'ENST00000353801.3-HSP90AB1',
# 'ENST00000505397.1-LIN54', 'ENST00000392915.1-MAP3K19',
# 'ENST00000375321.1-MAP3K8', 'MSTRG.11829.3-MAP3K8',
# 'ENST00000382948.5-MBD1', 'MSTRG.55994.22-MZT2A',
# 'MSTRG.39427.204-NRG4', 'MSTRG.18587.4-OTUB1', 'MSTRG.92133.10-PAK4',
# 'MSTRG.92133.3-PAK4', 'ENST00000222254.8-PIK3R2',
# 'ENST00000261207.5-PPP1R12A', 'ENST00000498508.2-PROX1',
# 'ENST00000358514.4-PSMB10', 'ENST00000442510.2-PTPRC',
# 'MSTRG.1704.189-RCC1', 'ENST00000412779.2-RNASE1',
# 'ENST00000575111.1-RNF167', 'MSTRG.52876.4-RPS27A',
# 'MSTRG.17113.1-SFR1', 'MSTRG.51515.3-SMC6', 'ENST00000453269.2-TAF6',
# 'ENST00000254942.3-TERF2', 'MSTRG.30902.1-TFDP1',
# 'MSTRG.74585.63-TFDP2', 'ENST00000539466.1-USP47',
# 'ENST00000355640.3-XIAP', 'ENST00000511817.1-XRCC4',
# 'ENST00000395958.2-YWHAZ', 'ENST00000395405.1-ZWINT']

tmp =['ENST00000391831.1-AKT1S1', 'ENST00000457016.1-APC', #####FINAL 68 features
       'ENST00000482687.1-APTX', 'ENST00000404251.1-BAZ1B',
       'ENST00000318407.3-BOK', 'MSTRG.49834.247-BRIP1',
       'ENST00000302759.6-BUB1', 'ENST00000310447.5-CASP2',
       'ENST00000310232.6-CCDC13', 'ENST00000308108.4-CCNE2',
       'ENST00000349780.4-CDK5RAP2', 'MSTRG.98879.3-CENPW',
       'ENST00000368328.4-CENPW', 'ENST00000399334.3-CEP152',
       'ENST00000376597.4-CEP78', 'ENST00000404276.1-CHEK2',
       'ENST00000308987.5-CKS1B', 'MSTRG.14904.57-CTBP2',
       'MSTRG.14904.52-CTBP2', 'ENST00000375440.4-CUL4A',
       'ENST00000531427.1-CUL5', 'ENST00000346618.3-E2F3',
       'ENST00000342628.2-FOXM1', 'MSTRG.49465.142-GRB2',
       'ENST00000454366.1-GTSE1', 'ENST00000373573.3-HDAC8',
       'ENST00000545134.1-HELB', 'ENST00000353801.3-HSP90AB1',
       'ENST00000373135.3-L3MBTL1', 'ENST00000505397.1-LIN54',
       'ENST00000392915.1-MAP3K19', 'ENST00000375321.1-MAP3K8',
       'MSTRG.11829.3-MAP3K8', 'ENST00000262815.8-MAU2',
       'ENST00000382948.5-MBD1', 'MSTRG.12458.189-MCL1',
       'MSTRG.55994.22-MZT2A', 'MSTRG.39427.204-NRG4',
       'MSTRG.114270.18-NSMCE2', 'MSTRG.18587.4-OTUB1', 'MSTRG.92133.10-PAK4',
       'MSTRG.92133.3-PAK4', 'ENST00000222254.8-PIK3R2',
       'ENST00000382865.1-POLN', 'MSTRG.126473.45-POM121',
       'ENST00000261207.5-PPP1R12A', 'ENST00000490777.2-PPP2R2D',
       'ENST00000498508.2-PROX1', 'ENST00000358514.4-PSMB10',
       'ENST00000442510.2-PTPRC', 'ENST00000520452.1-PTTG1',
       'MSTRG.1704.189-RCC1', 'ENST00000412779.2-RNASE1',
       'ENST00000575111.1-RNF167', 'MSTRG.52876.4-RPS27A',
       'MSTRG.17113.1-SFR1', 'MSTRG.51515.3-SMC6', 'ENST00000317296.5-STAG3',
       'ENST00000453269.2-TAF6', 'ENST00000254942.3-TERF2',
       'MSTRG.30902.1-TFDP1', 'MSTRG.74585.63-TFDP2',
       'ENST00000539466.1-USP47', 'ENST00000355640.3-XIAP',
       'ENST00000555055.1-XRCC3', 'ENST00000511817.1-XRCC4',
       'ENST00000395958.2-YWHAZ', 'ENST00000395405.1-ZWINT']

# Reactome2022_featurelist.pkl / BP2018_featurelist.pkl
with open('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/featureselection/Reactome2022_featurelist.pkl', 'rb') as file:
    featurelist_reactome = pickle.load(file)

with open('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/featureselection/BP2018_featurelist.pkl', 'rb') as file:
    featurelist_bp = pickle.load(file)

featurelist = set(featurelist_bp).union(set(featurelist_reactome))

dis = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_TU.txt', sep='\t', index_col=0)
val = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_TU.txt', sep='\t', index_col=0)
disinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo_new.txt', sep='\t', index_col=0)
valinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_validation_clinicalinfo.txt', sep='\t', index_col=0)
valinfo = valinfo.loc[list(val.columns),:]

dis_y = pd.DataFrame(disinfo.iloc[0::2,2])
dis_y.replace({0: 'IR', 1: 'AR'}, inplace=True)
valinfo['finalresponse'] = 'x'

valinfo.loc[(valinfo['ongoing']==1) | (valinfo['ongoing']==2) | (valinfo['ongoing']==4),'finalresponse'] = 'CR'
valinfo.loc[(valinfo['ongoing']==0) & (valinfo['response']==1), 'finalresponse'] = 'AR'
valinfo.loc[(valinfo['ongoing']==3) & (valinfo['response']==1), 'finalresponse'] = 'AR'
valinfo.loc[(valinfo['response']==0), 'finalresponse'] = 'IR'


featurelist =tmp 
df = val
y = pd.DataFrame(valinfo['finalresponse'])
#y = pd.DataFrame(df.iloc[-1,:])
y.columns=['class']
y['class'] = y['class'].map({'CR': 0, 'IR': 1, 'AR': 2})
y = y['class']
X = df.loc[featurelist,:]
X = X.T
########
plotdf = X.copy()
#plotdf = plotdf[featurelist]
plotdf = plotdf.apply(pd.to_numeric, errors='coerce')
########

plotdf['mean pre TU'] = plotdf.mean(axis=1)

plotdf['response'] = y.tolist()
plotdf['response'].replace({0: 'CR', 1: 'IR', 2:'AR'}, inplace=True)
plotdf['response'].replace({'IR': 'NR', 'AR': 'NR'}, inplace=True)

# plotdf['response'] = pd.Categorical(
#     plotdf['response'], 
#     categories=['CR', 'IR', 'AR'], 
#     ordered=True
#)


plt.figure(figsize=(4,6))
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


ax = sns.boxplot(y='mean pre TU', x='response', data=plotdf, 
            showfliers=True, order=['CR','NR'], 
            #color='#FFD92E'
            palette='vlag'
            #meanprops = {"marker":"o", "color":"blue","markeredgecolor": "blue", "markersize":"10"}
            )
#ax.set_ylim([0,0.01])
ax.set_ylabel('pre TU')
#ax.set_xticklabels(['IR','AR'])
#plt.title(transname)
sns.despine()

sns.stripplot(y='mean pre TU', x='response', data=plotdf, 
            order=['CR','NR'], 
            color='#7D7C7C',  # Color of the points
            size=4,         # Size of the points
            jitter=True,    # Adds some jitter to avoid overlapping
            alpha=0.8,
            ax=ax)


from statannot import add_stat_annotation


add_stat_annotation(ax, data=plotdf, x='response', y='mean pre TU',
                    #box_pairs=[('IR','AR'),('CR','AR'),('CR','IR')], 
                    box_pairs = [('CR','NR')],
                    order = ['CR','NR'], 
                    comparisons_correction=None,
                    test='Mann-Whitney', text_format='simple', loc='inside', fontsize=12
                    )

###
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/68features_validation_CR_NR_boxplot.pdf', bbox_inches='tight', dpi=300)
plt.show()



# %%
##^^^^^^^^^ response model major TU check ###################
trans = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_TU.txt', sep='\t', index_col=0)
trans = trans.drop(['OM/OS'])

last_row = trans.iloc[-1:]  # Selects the last row as a DataFrame
data_to_filter = trans.iloc[:-1]  # Selects all rows except the last one

minimum_samples = int(data_to_filter.shape[1] * 0.4) ########## 40% threshold

# Apply filtering based on non-zero values treated as "1"
filtered_data = data_to_filter[data_to_filter.apply(lambda x: (x != 0).sum(), axis=1) >= minimum_samples]

# Concatenate the filtered data with the last row
trans = pd.concat([filtered_data, last_row])

trans.loc['group',:] = trans.loc['group',:].astype('float')
trans.loc['group',:] = trans.loc['group',:].astype('int')
trans['Gene Symbol'] = trans.index.str.split("-",1).str[1]
##^^^^^^^ gDUT vs. DEG ######################
dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/MW_DUTresult_FC_stable.txt', sep='\t', index_col=0)
deg = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/MW_DEGresult_FC.txt', sep='\t', index_col=0)

deglist = set(deg.loc[(deg['p_value']<0.05) & (np.abs(deg['log2FC'])>0.4)]['Gene Symbol'])
gDUTlist = set(dut.loc[(dut['p_value']<0.05) & (np.abs(dut['log2FC'])>1)]['Gene Symbol'])

from matplotlib_venn import venn2

plt.figure(figsize=(6,6))
sns.set_style("white")
vd2 = venn2([deglist, gDUTlist],set_labels=('DEG', 'gDUT'),set_colors=('#F8CF6A','#74B2D4'), alpha=0.8)

###^^^^^^^^^^^^^^^ GO enrichment ####################3

import gseapy as gp

glist = list(gDUTlist)
enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                gene_sets=['GO_Biological_Process_2018',], 
                #gene_sets=['GO_Biological_Process_2021',], 
                organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                outdir=None, # don't write to disk
)

enrresult = enr.results.sort_values(by=['Adjusted P-value']) 


file = enrresult
#file['Term'] = file['Term'].str.rsplit(" ",1).str[0]
#file = file[(file['Adjusted P-value']<0.01) & (file['Adjusted P-value']>0.001) ]
file['FDR'] = -np.log10(file['Adjusted P-value'])
file = file[file['FDR']>1 ]

file = file[(file["Term"].str.contains("homologous recombination", case=False)) | (file["Term"].str.contains("DNA Damage", case=False)) | (file["Term"].str.contains("cell cycle", case=False)) | (file["Term"].str.contains("Signaling by WNT in cancer", case=False)) | (file["Term"].str.contains("PI3K/AKT Signaling in Cancer", case=False)) | (file["Term"].str.contains("double strand break", case=False))| (file["Term"].str.contains("DNA repair", case=False))]
##3| (file["Term"].str.contains("repair", case=False)) ]
            ##(file["Term"].str.contains("response to stress", case=False))

#file = enrresult
#fdrcut = file[file['Adjusted P-value']<0.1]
#fdrcut['FDR'] = -np.log10(fdrcut['Adjusted P-value'])
fdrcut = file
terms = fdrcut['Term']
fdr_values = fdrcut['FDR']

plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 13,    # x축 틱 라벨 글꼴 크기

'ytick.labelsize': 10,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 10,
'legend.title_fontsize': 13, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})
sns.set_style("white")
# Create a horizontal bar plot
plt.figure(figsize=(3,20))  # Making the figure tall and narrow
bars = plt.barh(range(len(terms)), fdr_values, color='#859F3D')

# Set y-axis labels and position them on the right
plt.yticks(range(len(terms)), terms, ha='left')
plt.gca().yaxis.tick_right()  # Move y-axis ticks to the right
plt.yticks(fontsize=16)

# Add labels and title
plt.xlabel('-log10(FDR)')
plt.gca().invert_yaxis()  # Invert y-axis to have the first term on top

plt.tight_layout()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/DUT_BP2018_FDR2_barplot.pdf', dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)

plt.show()


###^^^ PARPi-related enriched GO terms ######
# parplist = ['chromatin remodeling (GO:0006338)', 'cell cycle G2/M phase transition (GO:0044839)', 
#             'regulation of transcription, DNA-templated (GO:0006355)', 
#             #'regulation of gene expression (GO:0010468)', 
#             'negative regulation of gene expression (GO:0010629)',
#             'regulation of mRNA stability (GO:0043488)'] ## negative ..?
parplist = file['Term']
parplist = enrresult[enrresult['Term'].isin(parplist)]
parpgene = list(set([gene for sublist in parplist['Genes'].str.split(';') for gene in sublist]))



#####^^^^^^^^^^^^^^^^^ Random Forest with PARPi-related enriched GO genes ####################################################

dut_filtered = dut.loc[(dut['p_value']<0.05) & (np.abs(dut['log2FC'])>1)]
featurelist = dut_filtered[dut_filtered['Gene Symbol'].isin(parpgene)].index.to_list()

y = np.array(trans.iloc[-1,:-1])
y = y.astype(int)
X = trans.loc[featurelist,:]
X = X.iloc[:,:-1]
X = X.T

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Parameter grid for Grid Search
# param_grid = {
#     'n_estimators': [50, 100, 150],  # Fewer trees to reduce complexity
#     'max_depth': [5, 10, 15],  # Restrict maximum depth of trees to prevent overfitting
#     'min_samples_split': [5, 10, 15],  # Increase minimum samples required to split a node
#     'min_samples_leaf': [2, 4, 6],  # Increase minimum samples required at each leaf node
#     'max_features': ['sqrt', 'log2']  # Control the number of features considered at each split
# }
param_grid = {
    'n_estimators': [50, 75, 100],  # Reducing the maximum number of trees
    'max_depth': [4, 6, 8],  # Further limit tree depth to prevent complex fits
    'min_samples_split': [10, 20, 30],  # More restrictive splitting
    'min_samples_leaf': [7, 10, 15],  # Increase minimum samples in a leaf
    'max_features': ['sqrt', 'log2']  # Limit features considered at each split
}

# param_grid = {
#     'n_estimators': [50, 100, 150],
#     'max_depth': [5, 10,], #15
#     'min_samples_split': [10, 15, 20], #5
#     'min_samples_leaf': [4, 6], #2
#     'max_features': ['sqrt', 'log2']
# }

# Initialize Random Forest model
rf = RandomForestClassifier(random_state=42,class_weight='balanced')

# Stratified k-fold cross-validation
cv = StratifiedKFold(n_splits=5)

# Grid Search with cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Best parameters from Grid Search
best_rf = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE

rfecv = RFECV(estimator=best_rf, step=1, cv=StratifiedKFold(5), scoring='accuracy')
rfecv.fit(X_train, y_train)

print("Optimal number of features:", rfecv.n_features_)
selected_features = X_train.columns[rfecv.support_]
#print("Selected Features:", selected_features)

# Fit model with selected features
X_train_rfe = X_train[selected_features]
X_test_rfe = X_test[selected_features]
best_rf.fit(X_train_rfe, y_train)


from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc, RocCurveDisplay, PrecisionRecallDisplay

# Predictions and probabilities
y_train_pred = best_rf.predict(X_train_rfe)
y_test_pred = best_rf.predict(X_test_rfe)
y_train_prob = best_rf.predict_proba(X_train_rfe)[:, 1]
y_test_prob = best_rf.predict_proba(X_test_rfe)[:, 1]

# Metrics calculations
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)
train_roc_auc = roc_auc_score(y_train, y_train_prob)
test_roc_auc = roc_auc_score(y_test, y_test_prob)

# Precision-Recall AUC
precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_prob)
precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_prob)
train_pr_auc = auc(recall_train, precision_train)
test_pr_auc = auc(recall_test, precision_test)

# Feature importance (based on RFE-selected features)
feature_importances = pd.DataFrame({
    'Feature': selected_features,  # Column names of X #X.columns
    'Importance': best_rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Print metrics
print("\nTrain Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
print("Train ROC-AUC:", train_roc_auc)
print("Test ROC-AUC:", test_roc_auc)
print("Train PR-AUC:", train_pr_auc)
print("Test PR-AUC:", test_pr_auc)
print("Train F1 Score:", train_f1)
print("Test F1 Score:", test_f1)
print("\nFeature Importances (RFE Selected):\n", feature_importances)

# %%
#*###### major TU check
dis = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_TU.txt', sep='\t', index_col=0)
val = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_TU.txt', sep='\t', index_col=0)
disinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo_new.txt', sep='\t', index_col=0)
valinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_validation_clinicalinfo.txt', sep='\t', index_col=0)
valinfo = valinfo.loc[list(val.columns),:]

majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = majorminor[majorminor['type']=='major']['gene_ENST'].to_list()
minorlist = majorminor[majorminor['type']=='minor']['gene_ENST'].to_list()

dis_y = pd.DataFrame(disinfo.iloc[0::2,2])
dis_y.replace({0: 'IR', 1: 'AR'}, inplace=True)
valinfo['finalresponse'] = 'x'
valinfo.loc[(valinfo['ongoing']==1) | (valinfo['ongoing']==2) | (valinfo['ongoing']==4),'finalresponse'] = 'CR'
valinfo.loc[(valinfo['ongoing']==0) & (valinfo['response']==1), 'finalresponse'] = 'AR'
valinfo.loc[(valinfo['ongoing']==3) & (valinfo['response']==1), 'finalresponse'] = 'AR'
valinfo.loc[(valinfo['response']==0), 'finalresponse'] = 'IR'

#%%
#############################################################################################
filtered_features = list(set(majorlist).intersection(set(selected_features))) ###### major / minor
##############################################################################################

df = val
y = pd.DataFrame(valinfo['finalresponse'])
#y = pd.DataFrame(df.iloc[-1,:])
y.columns=['class']
y['class'] = y['class'].map({'CR': 0, 'IR': 1, 'AR': 2})
y = y['class']
X = df.loc[filtered_features,:]
X = X.T
########
plotdf = X.copy()
#plotdf = plotdf[featurelist]
plotdf = plotdf.apply(pd.to_numeric, errors='coerce')
########

plotdf['mean pre TU'] = plotdf.mean(axis=1)

plotdf['response'] = y.tolist()
plotdf['response'].replace({0: 'CR', 1: 'IR', 2:'AR'}, inplace=True)

plt.figure(figsize=(4,6))
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


ax = sns.boxplot(y='mean pre TU', x='response', data=plotdf, 
            showfliers=True, order=['CR','IR','AR'], 
            #color='#FFD92E'
            palette='vlag'
            #meanprops = {"marker":"o", "color":"blue","markeredgecolor": "blue", "markersize":"10"}
            )
#ax.set_ylim([0,0.01])
ax.set_ylabel('pre TU')
#ax.set_xticklabels(['IR','AR'])
#plt.title(transname)
sns.despine()

sns.stripplot(y='mean pre TU', x='response', data=plotdf, 
            order=['CR','IR', 'AR'], 
            color='#7D7C7C',  # Color of the points
            size=4,         # Size of the points
            jitter=True,    # Adds some jitter to avoid overlapping
            alpha=0.8,
            ax=ax)


from statannot import add_stat_annotation


add_stat_annotation(ax, data=plotdf, x='response', y='mean pre TU',
                    box_pairs=[('IR','AR'),('CR','AR'),('CR','IR')], 
                    order = ['CR','IR','AR'], 
                    comparisons_correction=None,
                    test='Mann-Whitney', text_format='simple', loc='inside', fontsize=12
                    )

##plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/137features_validation_boxplot.pdf', bbox_inches='tight', dpi=300)

plt.show()

# %%
###^^^ R vs. NR

#############################################################################################
filtered_features = list(set(majorlist).intersection(set(featurelist))) ###### major / minor featurelist/selected_features
##############################################################################################

df = val
y = pd.DataFrame(valinfo['response'])
#y = pd.DataFrame(df.iloc[-1,:])
y.columns=['class']
y = y['class']
X = df.loc[filtered_features,:]
X = X.T
########
plotdf = X.copy()
#plotdf = plotdf[featurelist]
plotdf = plotdf.apply(pd.to_numeric, errors='coerce')
########

plotdf['mean pre TU'] = plotdf.mean(axis=1)

plotdf['response'] = y.tolist()
plotdf['response'].replace({0: 'NR', 1: 'R'}, inplace=True)

plt.figure(figsize=(4,6))
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


ax = sns.boxplot(y='mean pre TU', x='response', data=plotdf, 
            showfliers=True, order=['NR','R',], 
            #color='#FFD92E'
            palette='vlag'
            #meanprops = {"marker":"o", "color":"blue","markeredgecolor": "blue", "markersize":"10"}
            )
#ax.set_ylim([0,0.01])
ax.set_ylabel('pre TU')
#ax.set_xticklabels(['IR','AR'])
#plt.title(transname)
sns.despine()

sns.stripplot(y='mean pre TU', x='response', data=plotdf, 
            order=['NR','R',], 
            color='#7D7C7C',  # Color of the points
            size=4,         # Size of the points
            jitter=True,    # Adds some jitter to avoid overlapping
            alpha=0.8,
            ax=ax)


from statannot import add_stat_annotation


add_stat_annotation(ax, data=plotdf, x='response', y='mean pre TU',
                    box_pairs=[('NR','R')], 
                    order = ['NR','R'], 
                    comparisons_correction=None,
                    test='Mann-Whitney', text_format='simple', loc='inside', fontsize=12
                    )

plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/RNR_16onlymajor_140features_validation_boxplot.pdf', bbox_inches='tight', dpi=300)
plt.show()

# %%
