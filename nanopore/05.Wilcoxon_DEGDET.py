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
# %%
# %%
import pandas as pd
from scipy.stats import wilcoxon
import numpy as np
from statsmodels.stats.multitest import multipletests

trans = pd.read_csv('/home/jiye/jiye/nanopore/FINALDATA/137_transcript_TPM.txt', sep='\t', index_col=0)
gene = pd.read_csv('/home/jiye/jiye/nanopore/FINALDATA/137_gene_TPM.txt', sep='\t', index_col=0)

#%%
gridion_samples = [
    'PM-AU-0002-N', 'PM-AU-0004-N', 'PM-AU-0006-N', 'PM-AU-0007-N', 'PM-AU-0008-N',
    'PM-AU-0024-N', 'PM-AU-0025-N', 'PM-AU-0033-N', 'PM-AU-0037-N', 'PM-AU-0038-N',
    'PM-PA-1001-N', 'PM-PA-1004-N', 'PM-PS-0001-N', 'PM-PS-0005-N', 'PM-PS-0009-N',
    'PM-PS-0011-N', 'PM-PS-0013-N', 'PM-PS-0014-N', 'PM-PS-0016-N', 'PM-PS-0017-N',
    'PM-PS-0018-N', 'PM-PS-0019-N', 'PM-PS-0024-N', 'PM-PS-0026-N', 'PM-PS-0027-N',
    'PM-PS-0028-N', 'PM-PS-0456-N', 'PM-PS-0485-N', 'PM-PU-1002-N', 'PM-PU-1004-N',
    'PM-PU-1005-N', 'PM-PU-1009-N', 'PM-PU-1010-N', 'PM-PU-1011-N', 'PM-PU-1012-N',
    'PM-PU-1013-N', 'PM-PU-1014-N', 'PM-PU-1015-N', 'PM-PU-1024-N', 'PM-PU-1025-N',
    'PM-PU-1026-N', 'PM-PU-1027-N',
    'PM-AU-0002-T', 'PM-AU-0004-T', 'PM-AU-0006-T', 'PM-AU-0007-T', 'PM-AU-0008-T',
    'PM-AU-0024-T', 'PM-AU-0025-T', 'PM-AU-0033-T', 'PM-AU-0037-T', 'PM-AU-0038-T',
    'PM-PA-1001-T', 'PM-PA-1004-T', 'PM-PS-0001-T', 'PM-PS-0005-T', 'PM-PS-0009-T',
    'PM-PS-0011-T', 'PM-PS-0013-T', 'PM-PS-0014-T', 'PM-PS-0016-T', 'PM-PS-0017-T',
    'PM-PS-0018-T', 'PM-PS-0019-T', 'PM-PS-0024-T', 'PM-PS-0026-T', 'PM-PS-0027-T',
    'PM-PS-0028-T', 'PM-PS-0456-T', 'PM-PS-0485-T', 'PM-PU-1002-T', 'PM-PU-1004-T',
    'PM-PU-1005-T', 'PM-PU-1009-T', 'PM-PU-1010-T', 'PM-PU-1011-T', 'PM-PU-1012-T',
    'PM-PU-1013-T', 'PM-PU-1014-T', 'PM-PU-1015-T', 'PM-PU-1024-T', 'PM-PU-1025-T',
    'PM-PU-1026-T', 'PM-PU-1027-T'
]
trans = trans.iloc[:,~trans.columns.isin(gridion_samples)]
trans = trans[trans.apply(lambda x: (x != 0).sum(), axis=1) >= 65] ## 30% across samples in only promethion

#trans = trans[trans.apply(lambda x: (x != 0).sum(), axis=1) >= 83] ## 30% across samples


#%%

T = trans.iloc[:,1::2]
N = trans.iloc[:,0::2]
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
results.to_csv('/home/jiye/jiye/nanopore/202411_analysis/wilcoxon_DET/onlyprometh_Wilcoxon_DETresult.txt', sep='\t', index=False)

# %%
##^^ DEG
gene = pd.read_csv('/home/jiye/jiye/nanopore/FINALDATA/137_gene_TPM.txt', sep='\t', index_col=0)
gene = gene.iloc[:,~gene.columns.isin(gridion_samples)]
geneid = gene['ENSG_id'].to_list()
gene = gene.iloc[:,1:]
T = gene.iloc[:,1::2]
N = gene.iloc[:,0::2]
T = T.apply(pd.to_numeric, errors='coerce')
N = N.apply(pd.to_numeric, errors='coerce')

# Calculate p-values and log2 fold changes
p_values = []
log2_fold_changes = []
genelist = gene.index.to_list()


# Filter out genes with identical expression values in N and T dataframes
filtered_genelist = [g for g in genelist if not np.array_equal(N.loc[g].values, T.loc[g].values)]

p_values = []
log2_fold_changes = []

for g in filtered_genelist:
    # Get the expression values for the transcript in both dataframes
    values1 = N.loc[g].values
    values2 = T.loc[g].values
    
    # Perform Wilcoxon signed-rank test
    stat, p_value = wilcoxon(values1, values2, alternative='two-sided')
    p_values.append(p_value)

    # Calculate log2 fold change (T/N)
    mean1 = np.mean(values1)
    mean2 = np.mean(values2)
    log2_fc = np.log2(mean2) - np.log2(mean1)  # Adding a small pseudocount to avoid log(0)
    log2_fold_changes.append(log2_fc)

# Adjust p-values using Benjamini-Hochberg method (False Discovery Rate correction)
p_adjusted = multipletests(p_values, method='fdr_bh')[1]

# Create a results DataFrame
results = pd.DataFrame({
    'gene_name': filtered_genelist,
    'p_value': p_values,
    'p-adjusted': p_adjusted,
    'log2FC': log2_fold_changes,
})

results

# Display the results
print(results)

results.to_csv('/home/jiye/jiye/nanopore/202411_analysis/wilcoxon_DEG/onlyprometh_Wilcoxon_DEGresult.txt', sep='\t', index=False)




# %%
# %%
##^^^^^^^ gDUT vs. DEG ######################
dut = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/wilcoxon_DET/onlyprometh_Wilcoxon_DETresult.txt', sep='\t', index_col=0)
deg = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/wilcoxon_DEG/onlyprometh_Wilcoxon_DEGresult.txt', sep='\t')
dut['gene_name'] = dut.index.str.split("-",1).str[1]
deglist = set(deg.loc[(deg['p-adjusted']<0.05) & (np.abs(deg['log2FC'])>1.5)]['gene_name'])
gDUTlist = set(dut.loc[(dut['p-adjusted']<0.05) & (np.abs(dut['log2FC'])>1.5)]['gene_name'])

#%%
from matplotlib_venn import venn2

plt.figure(figsize=(6,6))
sns.set_style("white")
vd2 = venn2([deglist, gDUTlist],set_labels=('DEG', 'DET gene'),set_colors=('#F8CF6A','#74B2D4'), alpha=0.8)
plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/wilcoxon_DET/DEG_DET_venn.pdf', dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)

#%%
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
enrresult.to_csv('/home/jiye/jiye/nanopore/202411_analysis/wilcoxon_DET/BP2021_DET_GOenrichment.txt', index=False, sep='\t')


file = enrresult
#file['Term'] = file['Term'].str.rsplit(" ",1).str[0]

file['FDR'] = -np.log10(file['Adjusted P-value'])
file = file[file['FDR']>2]

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
plt.figure(figsize=(8,16))  # Making the figure tall and narrow
bars = plt.barh(range(len(terms)), fdr_values, color='#859F3D')

# Set y-axis labels and position them on the right
plt.yticks(range(len(terms)), terms, ha='left')
plt.gca().yaxis.tick_right()  # Move y-axis ticks to the right
plt.yticks(fontsize=16)

# Add labels and title
plt.xlabel('-log10(FDR)')
plt.gca().invert_yaxis()  # Invert y-axis to have the first term on top

plt.tight_layout()
plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/wilcoxon_DET/BP2021_FDR2_barplot.pdf', dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)

plt.show()
# %%
