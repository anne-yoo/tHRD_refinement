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
import pandas as pd
from scipy.stats import wilcoxon
import numpy as np
from statsmodels.stats.multitest import multipletests

trans = pd.read_csv('/home/jiye/jiye/nanopore/FINALDATA/137_transcript_TPM.txt', sep='\t', index_col=0)
trans = trans[trans.apply(lambda x: (x != 0).sum(), axis=1) >= 83] ## 30% across samples in only promethion

#%%
clin = pd.read_csv('/home/jiye/jiye/nanopore/FINALDATA/137_clinicaldata.txt', sep='\t', index_col=0)
msslist = clin[clin['MSI_status']=='MSS'].index.to_list()
msilist = clin[(clin['MSI_status']=='MSI-H')|(clin['MSI_status']=='MSI-L')].index.to_list()
krasmtlist = clin[clin['KRAS_mut']=='MT'].index.to_list()
kraswtlist = clin[clin['KRAS_mut']=='WT'].index.to_list()

#%%
####^^ filter samplelist ###############################
prefix_list = kraswtlist
filtered_columns = [col for col in trans.columns if col.rsplit('-',1)[0] in prefix_list]
input = trans[filtered_columns]
########^^##############################################

#%%

T = input.iloc[:,1::2]
N = input.iloc[:,0::2]
T = T.apply(pd.to_numeric, errors='coerce')
N = N.apply(pd.to_numeric, errors='coerce')

# Calculate p-values and log2 fold changes
p_values = []
log2_fold_changes = []
translist = trans.index.to_list()


for t in translist:
    # Get the expression values for the transcript in both dataframes
    values1 = N.loc[t].values
    values2 = T.loc[t].values
    
    # Perform Mann-Whitney U test
    stat, p_value = wilcoxon(values1, values2, alternative='two-sided')
    p_values.append(p_value)
    
    # Calculate log2 fold change
    mean1 = np.mean(values1)
    mean2 = np.mean(values2)
    log2_fc = np.log2(mean2) - np.log2(mean1)  # T/N
    log2_fold_changes.append(log2_fc)

# Adjust p-values using Benjamini-Hochberg method (False Discovery Rate correction)
p_adjusted = multipletests(p_values, method='fdr_bh')[1]

# Create a results DataFrame
results = pd.DataFrame({
    'transcript_id': translist,
    'p_value': p_values,
    'p-adjusted': p_adjusted,
    'log2FC': log2_fold_changes,
})

results

# Display the results
print(results)
#results.to_csv('/home/jiye/jiye/nanopore/202411_analysis/DET_clingroup/KRAS_WT_DETresult.txt', sep='\t', index=False)




# %%
typelist = ['KRAS_MT','KRAS_WT','MSI','MSS']
typelist = ['MSI']
for i in range(1):
    dut = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/DET_clingroup/'+typelist[i]+'_DETresult.txt', sep='\t', index_col=0)
    dut['gene_name'] = dut.index.str.split("-",1).str[1]
    gDUTlist = set(dut.loc[(dut['p-adjusted']<0.05) & (np.abs(dut['log2FC'])>1.5)]['gene_name'])

    ###^^^^^^^^^^^^^^^ GO enrichment ####################3

    import gseapy as gp

    glist = list(gDUTlist)
    enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                    gene_sets=['GO_Biological_Process_2021',], 
                    #gene_sets=['GO_Biological_Process_2021',], 
                    organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                    outdir=None, # don't write to disk
    )

    enrresult = enr.results.sort_values(by=['Adjusted P-value']) 

    ## file save
    enrresult.to_csv('/home/jiye/jiye/nanopore/202411_analysis/DET_clingroup/'+typelist[i]+'_DET_GOenrichment.txt', index=False, sep='\t')


    file = enrresult
    #file['Term'] = file['Term'].str.rsplit(" ",1).str[0]

    file['FDR'] = -np.log10(file['Adjusted P-value'])
    file = file[file['FDR']>1]

    fdrcut = file
    terms = fdrcut['Term']
    fdr_values = fdrcut['FDR']

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
    # Create a horizontal bar plot
    plt.figure(figsize=(8,14))  # Making the figure tall and narrow
    bars = plt.barh(range(len(terms)), fdr_values, color='#859F3D')

    # Set y-axis labels and position them on the right
    plt.yticks(range(len(terms)), terms, ha='left')
    plt.gca().yaxis.tick_right()  # Move y-axis ticks to the right
    plt.yticks(fontsize=16)

    # Add labels and title
    plt.xlabel('-log10(FDR)')
    plt.gca().invert_yaxis()  # Invert y-axis to have the first term on top

    plt.tight_layout()
    plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/DET_clingroup/'+typelist[i]+'_DET_FDR1_enrichment.pdf', dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)

    plt.show()

# %%
#dut1 = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/DET_clingroup/KRAS_MT_DETresult.txt', sep='\t', index_col=0)
#dut2 = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/DET_clingroup/KRAS_WT_DETresult.txt', sep='\t', index_col=0)
dut1 = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/DET_clingroup/MSI_DETresult.txt', sep='\t', index_col=0)
dut2 = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/DET_clingroup/MSS_DETresult.txt', sep='\t', index_col=0)

tmp1 = set(dut1.loc[(dut1['p-adjusted']<0.05) & (np.abs(dut1['log2FC'])>1.5)].index.str.split("-",1).str[1])
tmp2 = set(dut2.loc[(dut2['p-adjusted']<0.05) & (np.abs(dut2['log2FC'])>1.5)].index.str.split("-",1).str[1])

from matplotlib_venn import venn2

plt.figure(figsize=(6,6))
sns.set_style("white")
aaa = venn2([tmp1, tmp2],set_labels=('MSI', 'MSS'),set_colors=('#F8CF6A','#74B2D4'), alpha=0.8)
# Set font sizes for the Venn diagram labelsㄴ
for text in aaa.set_labels:
    if text:  # Check if the text is not None
        text.set_fontsize(16)
for text in aaa.subset_labels:
    if text:  # Check if the text is not None
        text.set_fontsize(16)

# %%
#^^^ enrichment
glist = list(tmp2.intersection(tmp1))
enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                gene_sets=['GO_Biological_Process_2021',], 
                #gene_sets=['GO_Biological_Process_2021',], 
                organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                outdir=None, # don't write to disk
)

enrresult = enr.results.sort_values(by=['Adjusted P-value']) 

file = enrresult
#file['Term'] = file['Term'].str.rsplit(" ",1).str[0]

#%%
file['FDR'] = -np.log10(file['Adjusted P-value'])
file = file[file['FDR']>2
            ]

fdrcut = file
terms = fdrcut['Term']
fdr_values = fdrcut['FDR']

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
# Create a horizontal bar plot
plt.figure(figsize=(8,10))  # Making the figure tall and narrow
bars = plt.barh(range(len(terms)), fdr_values, color='#859F3D')

# Set y-axis labels and position them on the right
plt.yticks(range(len(terms)), terms, ha='left')
plt.gca().yaxis.tick_right()  # Move y-axis ticks to the right
plt.yticks(fontsize=16)

# Add labels and title
plt.xlabel('-log10(FDR)')
plt.gca().invert_yaxis()  # Invert y-axis to have the first term on top

plt.tight_layout()
#plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/DET_clingroup/'+typelist[i]+'_DET_FDR1_enrichment.pdf', dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)

plt.show()


# %%
#^^^ novel transcript #########
novellist = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/sqantioutput/stringtie_novel_list.txt', sep='\t', index_col=0)
novellist = novellist.index.to_list()
tpm = pd.read_csv('/home/jiye/jiye/nanopore/FINALDATA/137_transcript_TPM.txt', sep='\t', index_col=0)
filtered_rows = [row for row in tpm.index if row.split('-',1)[0] in novellist]
#%%
input = tpm.loc[filtered_rows] ## novel transcripts
# %%
df1 = input.iloc[:,1::2] ##tumor
df2 = input.iloc[:,0::2] ##normal

mask = (df1.sum(axis=1) != 0) & (df2.sum(axis=1) == 0) ## only in normal

# Get rows that satisfy the condition
result = df1[mask], df2[mask]


# %%
