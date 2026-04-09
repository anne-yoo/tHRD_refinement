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
val_df = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/116_validation_TU_group.txt', sep='\t', index_col=0)
val_gene = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/116_validation_gene_TPM_symbol_group.txt', sep='\t', index_col=0)
val_clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_validation_clinicalinfo_new.txt', sep='\t', index_col=0)
val_df = val_df.iloc[:-1,:-1]
val_df = val_df.apply(pd.to_numeric, errors='coerce')
vallist = list(val_df.columns)
val_clin = val_clin.loc[vallist,:]
genesym = val_gene.iloc[:,-1]
val_gene = val_gene.iloc[:-1,:-1]
val_gene = val_gene.apply(pd.to_numeric, errors='coerce')

# %%
###^^ DEG / DUT for maintenance vs salvage groups #####

main_list = val_clin[val_clin['OM/OS']=='maintenance'].index.to_list()
sal_list = val_clin[val_clin['OM/OS']=='salvage'].index.to_list()

main = val_gene.loc[:,main_list]
sal = val_gene.loc[:,sal_list]

whole = val_gene.copy()
dflist = [main] #sal
#dflist = [whole]

namelist = ['maintenance'] #'salvage'
#namelist = ['whole']

for j in range(1):
    df = dflist[j]
    clin = val_clin.loc[df.columns,:]
    ##################################################
    clin = clin[clin['BRCAmut']==0] #BRCA mutation !!
    ##################################################
    res_list = clin[(clin['type']=='CR') | (clin['type']=='AR')].index.to_list()
    #res_list = clin[(clin['type']=='CR')].index.to_list()
    nonres_list = clin[(clin['type']=='IR')].index.to_list()
    resdf = df.loc[:,res_list]
    nonresdf = df.loc[:,nonres_list]
    
    deg_pval = []

    for i in range(resdf.shape[0]):
        r_samples = resdf.iloc[i,:]
        nr_samples = nonresdf.iloc[i,:]
        w, p = stats.mannwhitneyu(r_samples, nr_samples)
        deg_pval.append(p)

    # Create a new DataFrame with geneid and respective p-values
    result_df = pd.DataFrame({
        'p_value': deg_pval,
    })
    result_df.index = val_gene.index
    result_df = pd.merge(result_df,genesym, how='inner', left_index=True, right_index=True)

    ### add FC ###

    avg_r = resdf.mean(axis=1)
    avg_nr = nonresdf.mean(axis=1)

    # Calculate the fold change as the log2 ratio of average post-treatment to pre-treatment expression
    fold_change = np.log2(avg_r / avg_nr)
    result_df['log2FC'] = fold_change

    result_df.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/'+namelist[j]+'_BRCAwt_DEGresult_FC.txt', sep='\t')
# %%
####^ DUT
main_list = val_clin[val_clin['OM/OS']=='maintenance'].index.to_list()
sal_list = val_clin[val_clin['OM/OS']=='salvage'].index.to_list()

main = val_df.loc[:,main_list]
sal = val_df.loc[:,sal_list]

dflist = [main] #sal
namelist = ['maintenance'] #'salvage'

for j in range(1):
        
    degresult = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/'+namelist[j]+'_BRCAwt_DEGresult_FC.txt', sep='\t')
    DEGlist = set(degresult[degresult['p_value']<0.05]['Gene Symbol'])
    nonDEGlist = set(degresult[degresult['p_value'] >= 0.05]['Gene Symbol'])
    
    df = dflist[j]
    clin = val_clin.loc[df.columns,:]
    ##################################################
    clin = clin[clin['BRCAmut']==0] #BRCA mutation !!
    ##################################################
    res_list = clin[(clin['type']=='CR') | (clin['type']=='AR')].index.to_list()
    nonres_list = clin[(clin['type']=='IR')].index.to_list()
    r_df = df.loc[:,res_list]
    nr_df = df.loc[:,nonres_list]
    r_df['Gene Symbol'] = r_df.index.str.split("-",n=1).str[-1]
    nr_df['Gene Symbol'] = nr_df.index.str.split("-",n=1).str[-1]


    variable_NR = nr_df[nr_df['Gene Symbol'].isin(DEGlist)]
    variable_R = r_df[r_df['Gene Symbol'].isin(DEGlist)]
    stable_NR = nr_df[nr_df['Gene Symbol'].isin(nonDEGlist)]
    stable_R = r_df[r_df['Gene Symbol'].isin(nonDEGlist)]

    variable_NR = variable_NR.iloc[:,:-1]
    variable_R = variable_R.iloc[:,:-1]
    stable_NR = stable_NR.iloc[:,:-1]
    stable_R = stable_R.iloc[:,:-1]

    variable_dut_pval = []
    for i in range(len(variable_NR.index)):
        NR_samples = variable_NR.iloc[i,:].values 
        R_samples = variable_R.iloc[i,:].values 
        
        w, p = stats.mannwhitneyu(NR_samples, R_samples)
        variable_dut_pval.append(p)

    # Create a new DataFrame with geneid and respective p-values
    variable_result = pd.DataFrame({
        'p_value':variable_dut_pval,
    })
    variable_result.index = variable_NR.index
    variable_result['Gene Symbol'] = variable_result.index.str.split("-",n=1).str[-1]

    ##### FC #####
    avg_NR = variable_NR.mean(axis=1)
    avg_R = variable_R.mean(axis=1)

    fold_change = np.log2(avg_R / avg_NR)
    variable_result['log2FC'] = fold_change

    #### stable
    stable_dut_pval = []
    for i in range(len(stable_NR.index)):
        NR_samples = stable_NR.iloc[i,:].values 
        R_samples = stable_R.iloc[i,:].values 
        
        w, p = stats.mannwhitneyu(NR_samples, R_samples)
        stable_dut_pval.append(p)

    # Create a new DataFrame with geneid and respective p-values
    stable_result = pd.DataFrame({
        'p_value': stable_dut_pval,
    })
    stable_result.index = stable_NR.index
    stable_result['Gene Symbol'] = stable_result.index.str.split("-",n=1).str[-1]

    ##### FC #####
    avg_NR = stable_NR.mean(axis=1)
    avg_R = stable_R.mean(axis=1)

    fold_change = np.log2(avg_R / avg_NR)
    stable_result['log2FC'] = fold_change

    variable_result.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/'+namelist[j]+'_BRCAwt_variable_DUT_FC.txt', sep='\t')
    stable_result.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/'+namelist[j]+'_BRCAwt_stable_DUT_FC.txt', sep='\t')

# %%
###^ venn diagram salvage vs. maintenance DUT ######

majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = majorminor[majorminor['type']=='major']['gene_ENST'].to_list()

# sal_variable = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/noAR_salvagevariable_DUT_FC.txt', sep='\t', index_col=0)
# main_variable = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/noAR_maintenancevariable_DUT_FC.txt', sep='\t', index_col=0)
sal_stable = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/maintenance_BRCAwt_stable_DUT_FC.txt', sep='\t', index_col=0)
main_stable = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/maintenance_BRCAmt_stable_DUT_FC.txt', sep='\t', index_col=0)
tmp_stable = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/salvagestable_DUT_FC.txt', sep='\t', index_col=0)

from matplotlib_venn import venn2, venn3

# set1 = set(sal_variable[(sal_variable['p_value']<0.05) & (np.abs(sal_variable['log2FC'])>1.5)].index)
# set2 = set(main_variable[(main_variable['p_value']<0.05) & (np.abs(main_variable['log2FC'])>1.5)].index)

# plt.figure(figsize=(5, 5))
# venn2([set1, set2], set_labels=('Salvage', 'Maintenance'))
# plt.title('Variable DUT')
# plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/2505figs/noAR_variableDUTvenn_SM.pdf', dpi=300, bbox_inches='tight')
# plt.show()

set1 = set(sal_stable[(sal_stable['p_value']<0.05) & (np.abs(sal_stable['log2FC'])>1.5)].index)
set2 = set(main_stable[(main_stable['p_value']<0.05) & (np.abs(main_stable['log2FC'])>1.5)].index)
set3 = set(tmp_stable[(tmp_stable['p_value']<0.05) & (np.abs(tmp_stable['log2FC'])>1.5)].index)

plt.figure(figsize=(5, 5))
venn3([set1, set2, set3], set_labels=('BRCAwt Main', 'BRCAmt Main','Salvage'))
plt.title('Stable DUT')
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/2505figs/noAR_stableDUTvenn_SM.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
##^^ DEG GO ########
import gseapy as gp
saldeg = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/noAR_salvage_DEGresult_FC.txt', sep='\t',index_col=0)
maindeg = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/noAR_maintenance_DEGresult_FC.txt', sep='\t', index_col=0)

set1 = set(saldeg[(saldeg['p_value']<0.05) & (np.abs(saldeg['log2FC'])>1)]['Gene Symbol'])
set2 = set(maindeg[(maindeg['p_value']<0.05) & (np.abs(maindeg['log2FC'])>1)]['Gene Symbol'])

plt.figure(figsize=(3, 3))
venn2([set1, set2], set_labels=('Salvage', 'Maintenance'))
plt.title('DEG')
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/2505figs/noAR_DEGvenn_SM.pdf', dpi=300, bbox_inches='tight')
plt.show()

enr = gp.enrichr(gene_list=list(set2), # or "./tests/data/gene_list.txt",
                gene_sets=['GO_Biological_Process_2021',
                           #'Reactome_2022'
                           #'GO_Biological_Process_2018','GO_Biological_Process_2023','Reactome_2022'
                           ], 
                organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                outdir=None, # don't write to disk
)

file = enr.results.sort_values(by=['Adjusted P-value']) 
file['FDR'] = -np.log10(file['Adjusted P-value'])
file = file[file['FDR']>1 ]
file = file.iloc[:20,:]
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
plt.figure(figsize=(3,7))  # Making the figure tall and narrow
bars = plt.barh(range(len(terms)), fdr_values,  color='#C1EFDD') #C1EFDD for maintenance / #FDAA9D for salvage

# Set y-axis labels and position them on the right
plt.yticks(range(len(terms)), terms, ha='left')
plt.gca().yaxis.tick_right()  # Move y-axis ticks to the right
plt.yticks(fontsize=13)

# Add labels and title
plt.xlabel('-log10(FDR)')
plt.gca().invert_yaxis()  # Invert y-axis to have the first term on top

plt.tight_layout()
sns.despine()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/2505figs/withAR_maintenance_DEG_GO.pdf', dpi=300, bbox_inches='tight')

plt.show()
# %%

#^^ sample comparison:R .NR S/M

from venny4py.venny4py import *

tmpclin = val_clin[val_clin['type']!='AR']

# set1 = set(val_clin[val_clin['OM/OS']=='maintenance'].index.to_list())
# set2 = set(val_clin[val_clin['OM/OS']=='salvage'].index.to_list())
# set3 = set(val_clin[(val_clin['type']=='CR') | (val_clin['type']=='AR')].index.to_list())
# set4 = set(val_clin[(val_clin['type']=='IR')].index.to_list())

set1 = set(tmpclin[tmpclin['OM/OS']=='maintenance'].index.to_list())
set2 = set(tmpclin[tmpclin['OM/OS']=='salvage'].index.to_list())
set3 = set(tmpclin[(tmpclin['type']=='CR')].index.to_list())
set4 = set(tmpclin[(tmpclin['type']=='IR')].index.to_list())

#dict of sets
sets = {
    'maintenance': set1,
    'salvage': set2,
    'responder': set3,
    'nonresponder': set4}
    
venny4py(sets=sets)
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/2505figs/noAR_venn_4groups.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
deg = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/whole_DEGresult_FC.txt', sep='\t',index_col=0)
deglist = set(deg[(deg['p_value']<0.05) & (np.abs(deg['log2FC'])>1)]['Gene Symbol'])


# %%
##^^^ DUT GO enrichment #######################

import gseapy as gp

stable_result = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/maintenance_BRCAwt_stable_DUT_FC.txt', sep='\t', index_col=0)
#glist = list(DEGlist)
glist = list(set(stable_result[(stable_result['p_value']<0.05) & (np.abs(stable_result['log2FC']>1.5))]['Gene Symbol']))
##glist = list(set(variable_result[(variable_result['p_value']<0.05) & (np.abs(variable_result['log2FC']>1.5))]['Gene Symbol']))
enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                gene_sets=['GO_Biological_Process_2021',
                        #'Reactome_2022'
                        #'GO_Biological_Process_2018','GO_Biological_Process_2023','Reactome_2022'
                        ], 
                organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                outdir=None, # don't write to disk
)

enrresult = enr.results.sort_values(by=['Adjusted P-value']) 
file = enrresult
def string_fraction_to_float(fraction_str):
    numerator, denominator = fraction_str.split('/')
    return float(numerator) / float(denominator)

file['per'] = file['Overlap'].apply(string_fraction_to_float)

file = file.sort_values('Adjusted P-value')
#file = file.sort_values(by='Combined Score', ascending=False)
##remove mitochondrial ##
#file['Term'] = file['Term'].str.rsplit(" ",1).str[0]
#file = file[~file['Term'].str.contains('mitochondrial')]
#file = file.iloc[:10,:]
file = file[file['Adjusted P-value']<0.05]
file = file.iloc[:30,:]
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
sns.set_style("ticks")

fig, ax = plt.subplots(figsize=(6, 6))

# Create a horizontal bar plot
bars = ax.barh(file['Term'], file['Adjusted P-value'], color='#154674')
ax.set_xlabel('-log10(adjp)')

# Adjust y-axis tick labels: set font size and ensure labels are fully visible
ax.set_yticklabels(file['Term'], fontsize=10)

# Invert y-axis to have the lowest p-value at the top
ax.invert_yaxis()

# Adjust subplot parameters to give more room for y-axis labels
plt.subplots_adjust(left=0.65)  # Adjust the left margin as needed
sns.despine()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/whole_DUT_116_GOplot.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%
###^^ major / minor DUT KDE plot #################
stable_result = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/maintenance_BRCAwt_stable_DUT_FC.txt', sep='\t', index_col=0)

r_list = val_clin[(val_clin['OM/OS']=='maintenance') & (val_clin['type']!='IR') & (val_clin['BRCAmut']==0)].index.to_list()
nr_list = val_clin[(val_clin['OM/OS']=='maintenance') & (val_clin['type']=='IR') & (val_clin['BRCAmut']==0)].index.to_list()

r_df = val_df.loc[:,r_list]
nr_df = val_df.loc[:,nr_list]

r_df['Gene Symbol'] = r_df.index.str.split("-",n=1).str[-1]
nr_df['Gene Symbol'] = nr_df.index.str.split("-",n=1).str[-1]

majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = majorminor[majorminor['type']=='major']['gene_ENST'].to_list()

dutlist = stable_result.loc[(stable_result['p_value']<0.05) & (np.abs(stable_result['log2FC'])>1.5)].index.to_list()

#########################################################
R_major = r_df.loc[dutlist,r_list]
NR_major = nr_df.loc[dutlist,nr_list]

# R_major = r_df.loc[:,r_list] #### whole transcripts
# NR_major = nr_df.loc[:,nr_list] ##### whole transcripts 
#########################################################

R_major = R_major.loc[R_major.index.isin(majorlist),:] #major vs. minor
NR_major = NR_major.loc[NR_major.index.isin(majorlist),:] #major vs. minor

R_major_mean = R_major.mean(axis=1)
NR_major_mean = NR_major.mean(axis=1)

palette = {"Responder": "#FFA498", "Non-responder": "#A2B0FF"}

# Create DataFrame for plotting
df_whole = pd.DataFrame({
    'TU': pd.concat([R_major_mean, NR_major_mean]),
    'Condition': ['Responder'] * len(R_major_mean) + ['Non-responder'] * len(NR_major_mean)
})

# Plot for AR (All Transcripts)
plt.figure(figsize=(6,5))
sns.kdeplot(data=df_whole, x='TU', hue='Condition', fill=True, palette=palette)
plt.title('major DUTs')
#plt.xlim(right=0.44)
#plt.xticks(np.arange(0,0.5,0.1))
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/maintenance_BRCAwt_majorDUT_KDEplot.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
####^^^ DNA repair genes check #######################
import gseapy as gp
go_results = gp.get_library("GO_Biological_Process_2021", organism="Human")
repairgenes = go_results['double-strand break repair (GO:0006302)']
repairgenes = go_results['DNA repair (GO:0006281)']

stable_result = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/stable_DUT_FC.txt', sep='\t', index_col=0)
dutlist = stable_result.loc[(stable_result['p_value']<0.05) & (np.abs(stable_result['log2FC'])>1.5)].index.to_list()

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
ARlist = list(set(sampleinfo.loc[(sampleinfo['response']==1),'sample_id']))
IRlist = list(set(sampleinfo.loc[(sampleinfo['response']==0),'sample_id']))


majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = majorminor[majorminor['type']=='major']['gene_ENST'].to_list()
minorlist = majorminor[majorminor['type']=='minor']['gene_ENST'].to_list()

HRRgenes = ['GEN1', 'BARD1', 'RAD50', 'SHFM1', 'XRCC2', 'NBN', 'MUS81', 'MRE11A', 'RAD52', 'BRCA2', 'XRCC3', 'RAD51C', 'RAD51D', 'TP53BP1', 'BLM', 'SLX1A', 'PALB2', 'TOP3A', 'BRCA1', 'EME1', 'BRIP1', 'RBBP8']

#%%
from statannotations.Annotator import Annotator
##* every minor/major transcripts ##########
val_clin['group'] = val_clin['type'].replace({'CR': 'R', 'IR': 'NR','AR': 'R'})

main_list = val_clin[val_clin['OM/OS']=='maintenance'].index.to_list()
sal_list = val_clin[val_clin['OM/OS']=='salvage'].index.to_list()

main = val_df.loc[:,main_list]
sal = val_df.loc[:,sal_list]

main_clin = val_clin[val_clin['OM/OS']=='maintenance']
sal_clin= val_clin[val_clin['OM/OS']=='salvage']

def plot_transcriptwise_group_mean(val_df, val_clin, transcript_list, repairgenes, title, omos):
    # transcript에서 gene 이름 추출
    val_df['gene'] = val_df.index.str.split("-", n=1).str[-1]

    # transcript filtering
    selected = val_df.loc[val_df.index.isin(transcript_list) & val_df['gene'].isin(repairgenes)].drop(columns='gene')
    print(f"# of transcripts selected: {selected.shape[0]}")

    # 각 transcript에 대해 R/NR 그룹의 TU 평균을 계산
    group_info = val_clin['group']
    grouped_means = []

    for tx in selected.index:
        expr = selected.loc[tx]
        group_expr = expr.groupby(group_info).mean()  # R/NR 평균
        for group in ['R', 'NR']:
            if group in group_expr:
                grouped_means.append({'transcript': tx, 'group': group, 'mean_TU': group_expr[group]})

    # long dataframe
    df_mean = pd.DataFrame(grouped_means)

    # plot
    plt.figure(figsize=(4, 5))
    palette = {"R": "#FFBBBB", "NR": "#8DADFF"}
    ax = sns.boxplot(x='group', y='mean_TU', data=df_mean, palette=palette, showfliers=False)
    #sns.stripplot(x='group', y='mean_TU', data=df_mean, color='gray', size=3, alpha=0.6, jitter=True)

    # p-value
    annotator = Annotator(ax, [("R", "NR")], data=df_mean, x="group", y="mean_TU")
    annotator.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=1)
    annotator.apply_and_annotate()

    plt.title(f"{title} transcripts: HRR genes")
    plt.xlabel('')
    plt.tight_layout()
    sns.despine()

    #plt.savefig(f'/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/{omos}_{title}_whole_HRR_boxplot.pdf',
                # dpi=300, bbox_inches='tight')
    plt.show()


plot_transcriptwise_group_mean(sal, sal_clin, majorlist, HRRgenes, title='Major', omos='salvage')
plot_transcriptwise_group_mean(sal, sal_clin, minorlist, HRRgenes, title='Minor', omos='salvage')


#%%
from statannotations.Annotator import Annotator
##* only minor/major DUTs ##########
def plot_transcriptwise_group_mean_tu(val_df, val_clin, transcript_list, repairgenes, title, omos):
    # load DUT-filtered transcript list
    stable_result = pd.read_csv(
        f'/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/{omos}stable_DUT_FC.txt',
        sep='\t', index_col=0)
    dutlist = stable_result.loc[
        (stable_result['p_value'] < 0.05) & (np.abs(stable_result['log2FC']) > 1.5)
    ].index.to_list()

    # gene info 붙이기
    val_df['gene'] = val_df.index.str.split("-", n=1).str[-1]

    # filter transcript list
    tlist = list(set(transcript_list).intersection(dutlist))
    selected = val_df.loc[val_df.index.isin(tlist) & val_df['gene'].isin(repairgenes)].drop(columns='gene')
    print(f"# of transcripts: {selected.shape[0]}, # of samples: {selected.shape[1]}")

    # transcript별로 R/NR 그룹의 평균 TU 계산
    group_info = val_clin['group']
    result_rows = []
    for tx in selected.index:
        expr = selected.loc[tx]
        group_means = expr.groupby(group_info).mean()
        for g in ['R', 'NR']:
            if g in group_means:
                result_rows.append({'transcript': tx, 'group': g, 'mean_TU': group_means[g]})

    df_groupmean = pd.DataFrame(result_rows)

    # plot
    plt.figure(figsize=(4, 5))
    palette = {"R": "#FFBBBB", "NR": "#8DADFF"}
    ax = sns.boxplot(x='group', y='mean_TU', data=df_groupmean, palette=palette, showfliers=False)
    #sns.stripplot(x='group', y='mean_TU', data=df_groupmean, color='gray', size=3, alpha=0.6, jitter=True)

    # p-value annotation
    annotator = Annotator(ax, [("R", "NR")], data=df_groupmean, x="group", y="mean_TU")
    annotator.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=1)
    annotator.apply_and_annotate()

    plt.title(f"{title} DUTs: HRR core genes")
    plt.xlabel('')
    plt.tight_layout()
    sns.despine()

    #plt.savefig(
    #     f'/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/{omos}_{title}_DUT_HRR_boxplot.pdf',
    #     dpi=300, bbox_inches='tight'
    # )
    plt.show()

plot_transcriptwise_group_mean(sal, sal_clin, majorlist, HRRgenes, title='Major', omos='salvage')
plot_transcriptwise_group_mean(sal, sal_clin, minorlist, HRRgenes, title='Minor', omos='salvage')

#%%%%%5

#####^^ 요기 위로는 transcript-wise, 아래로는 sample-wise ##########





# %%
##* every minor/major transcripts ##########
val_clin['group'] = val_clin['type'].replace({'CR': 'R', 'IR': 'NR','AR': 'R'})

main_list = val_clin[val_clin['OM/OS']=='maintenance'].index.to_list()
sal_list = val_clin[val_clin['OM/OS']=='salvage'].index.to_list()

main = val_df.loc[:,main_list]
sal = val_df.loc[:,sal_list]

main_clin = val_clin[val_clin['OM/OS']=='maintenance']
sal_clin = val_clin[val_clin['OM/OS']=='salvage']

def plot_sample_mean_tu(val_df, val_clin, transcript_list, repairgenes, title, omos):
    # transcript에서 gene 이름 추출
    val_df['gene'] = val_df.index.str.split("-", n=1).str[-1]
    
    # major/minor 리스트 안에 있고, gene이 repairgene에 포함되는 transcript만 선택
    selected = val_df.loc[val_df.index.isin(transcript_list) & val_df['gene'].isin(repairgenes)].drop(columns='gene')
    
    # sample별 평균 TU값 계산
    sample_mean_tu = selected.T.mean(axis=1).to_frame(name='mean_TU')
    sample_mean_tu['group'] = val_clin['group'].values
    
    # boxplot
    palette = {"R": "#FFBBBB", "NR": "#8DADFF"}
    plt.figure(figsize=(4, 5))
    ax = sns.boxplot(x='group', y='mean_TU', data=sample_mean_tu, palette=palette, showfliers=False)
    sns.stripplot(x='group', y='mean_TU', data=sample_mean_tu, color='grey', size=3, alpha=0.6, jitter=True)

    # p-value annotation
    pairs = [("R", "NR")]
    annotator = Annotator(ax, pairs, data=sample_mean_tu, x="group", y="mean_TU")
    annotator.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=1)
    annotator.apply_and_annotate()

    plt.title(f"{title} transcripts: HRR genes")
    plt.xlabel('')
    plt.tight_layout()
    sns.despine()
    plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/'+omos+'_'+title+'_whole_HRR_samplewise_boxplot.pdf', dpi=300, bbox_inches='tight')
    plt.show()

plot_sample_mean_tu(main, main_clin, majorlist, HRRgenes, title='Major', omos='maintenance')
plot_sample_mean_tu(main, main_clin, minorlist, HRRgenes, title='Minor', omos='maintenance')
plot_sample_mean_tu(sal, sal_clin, majorlist, HRRgenes, title='Major', omos='salvage')
plot_sample_mean_tu(sal, sal_clin, minorlist, HRRgenes, title='Minor', omos='salvage')
#%%
##* only minor/major DUTs ##########
def plot_sample_mean_tu(val_df, val_clin, transcript_list, repairgenes, title, omos):
    stable_result = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/'+omos+'stable_DUT_FC.txt', sep='\t', index_col=0)

    dutlist = stable_result.loc[(stable_result['p_value']<0.05) & (np.abs(stable_result['log2FC'])>1.5)].index.to_list()
    
    # transcript에서 gene 이름 추출
    val_df['gene'] = val_df.index.str.split("-", n=1).str[-1]
    tlist = list(set(transcript_list).intersection(dutlist))
    # major/minor 리스트 안에 있고, gene이 repairgene에 포함되는 transcript만 선택
    selected = val_df.loc[val_df.index.isin(tlist) & val_df['gene'].isin(repairgenes)].drop(columns='gene')
    print(selected.shape)
    # sample별 평균 TU값 계산
    sample_mean_tu = selected.T.mean(axis=1).to_frame(name='mean_TU')
    sample_mean_tu['group'] = val_clin['group'].values

    # boxplot
    palette = {"R": "#FFBBBB", "NR": "#8DADFF"}
    plt.figure(figsize=(4, 5))
    ax = sns.boxplot(x='group', y='mean_TU', data=sample_mean_tu, palette=palette, showfliers=False)
    sns.stripplot(x='group', y='mean_TU', data=sample_mean_tu, color='grey', size=3, alpha=0.6, jitter=True)

    # p-value annotation
    pairs = [("R", "NR")]
    annotator = Annotator(ax, pairs, data=sample_mean_tu, x="group", y="mean_TU")
    annotator.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=1)
    annotator.apply_and_annotate()

    plt.title(f"{title} DUTs: HRR genes")
    plt.xlabel('')
    plt.tight_layout()
    sns.despine()
    plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/'+omos+'_'+title+'_DUT_HRR_samplewise_boxplot.pdf', dpi=300, bbox_inches='tight')
    
    plt.show()

#plot_sample_mean_tu(main, main_clin, majorlist, HRRgenes, title='Major', omos='maintenance')
#plot_sample_mean_tu(main, main_clin, minorlist, HRRgenes, title='Minor', omos='maintenance')
plot_sample_mean_tu(sal, sal_clin, majorlist, HRRgenes, title='Major', omos='salvage')
plot_sample_mean_tu(sal, sal_clin, minorlist, HRRgenes, title='Minor', omos='salvage')



# %%
#######^^ salvage vs. maintenance volcano plot ###########
from scipy.stats import mannwhitneyu
stable_result = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/salvagestable_DUT_FC.txt', sep='\t', index_col=0)
saldutlist = stable_result.loc[(stable_result['p_value']<0.05) & (np.abs(stable_result['log2FC'])>1.5)].index.to_list()

stable_result = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/maintenancestable_DUT_FC.txt', sep='\t', index_col=0)
maindutlist = stable_result.loc[(stable_result['p_value']<0.05) & (np.abs(stable_result['log2FC'])>1.5)].index.to_list()


main_df = val_df.loc[:,main_list]
sal_df = val_df.loc[:,sal_list]


salmajor_R = sal_df.loc[sal_df.index.isin(list(set(saldutlist).intersection(set(majorlist)))),sal_clin[sal_clin['group']=='R'].index]
salmajor_NR = sal_df.loc[sal_df.index.isin(list(set(saldutlist).intersection(set(majorlist)))),sal_clin[sal_clin['group']=='NR'].index]
mainmajor_R = main_df.loc[main_df.index.isin(list(set(maindutlist).intersection(set(majorlist)))),main_clin[main_clin['group']=='R'].index]
mainmajor_NR = main_df.loc[main_df.index.isin(list(set(maindutlist).intersection(set(majorlist)))),main_clin[main_clin['group']=='NR'].index]

# ΔTU + paired t-test
def compute_delta_pval(pre, post):
    delta = post.mean(axis=1) - pre.mean(axis=1)
    pvals = []
    for i in range(pre.shape[0]):
        _, p = mannwhitneyu(post.iloc[i, :], pre.iloc[i, :])
        pvals.append(p)
    return delta, np.array(pvals)

# salvage
delta_sal, pval_sal = compute_delta_pval(salmajor_R, salmajor_NR)
df_sal = pd.DataFrame({
    'transcript': salmajor_R.index,
    'delta': delta_sal,
    'neg_log10_pval': -np.log10(pval_sal + 1e-10),
})


# maintenance
delta_main, pval_main = compute_delta_pval(mainmajor_R, mainmajor_NR)
df_main = pd.DataFrame({
    'transcript': mainmajor_R.index,
    'delta': delta_main,
    'neg_log10_pval': -np.log10(pval_main + 1e-10),
})

# Subplot Grid (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

# AR Plot (왼쪽)
axes[0].scatter(df_sal['delta'], df_sal['neg_log10_pval'], color="#FFCC29", alpha=0.5, s=20)
axes[0].axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
axes[0].set_title('TU change of major DUTs: salvage')
axes[0].set_xlabel('ΔTU (NR - R)')
axes[0].set_ylabel('-log10(pval)')

# IR Plot (오른쪽)
axes[1].scatter(df_main['delta'], df_main['neg_log10_pval'], color="#81B214", alpha=0.5, s=20)
axes[1].axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
axes[1].set_title('TU change of major DUTs: maintenance')
axes[1].set_xlabel('ΔTU (NR - R)')

plt.tight_layout()
sns.despine()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/SM_majorDUTvolcano.pdf', dpi=300, bbox_inches='tight')
plt.show()


#%%
##############^^^^ GO-level volcano plot ###########################
import gseapy as gp
go_results = gp.get_library("GO_Biological_Process_2021", organism="Human")
GO_terms = list(go_results.keys())
#%%
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, wilcoxon
import matplotlib.pyplot as plt
val_clin['group'] = val_clin['type'].replace({'CR': 'R', 'IR': 'NR','AR': 'R'})

main_list = val_clin[val_clin['OM/OS']=='maintenance'].index.to_list()
sal_list = val_clin[val_clin['OM/OS']=='salvage'].index.to_list()

main = val_df.loc[:,main_list]
sal = val_df.loc[:,sal_list]

main_clin = val_clin[val_clin['OM/OS']=='maintenance']
sal_clin = val_clin[val_clin['OM/OS']=='salvage']

main_df = val_df.loc[:,main_list]
sal_df = val_df.loc[:,sal_list]


salmajor_R = sal_df.loc[sal_df.index.isin(list(set(saldutlist).intersection(set(majorlist)))),sal_clin[sal_clin['group']=='R'].index]
salmajor_NR = sal_df.loc[sal_df.index.isin(list(set(saldutlist).intersection(set(majorlist)))),sal_clin[sal_clin['group']=='NR'].index]
mainmajor_R = main_df.loc[main_df.index.isin(list(set(maindutlist).intersection(set(majorlist)))),main_clin[main_clin['group']=='R'].index]
mainmajor_NR = main_df.loc[main_df.index.isin(list(set(maindutlist).intersection(set(majorlist)))),main_clin[main_clin['group']=='NR'].index]

# 🧬 유전자 이름 추출 (예: ENST00000373993.1-A1CF → A1CF)
transcript_genes = salmajor_R.index.to_series().str.split("-", n=1).str[-1]
transcript_to_gene = pd.Series(transcript_genes.values, index=salmajor_R.index)

# transcript_genes = mainmajor_R.index.to_series().str.split("-", n=1).str[-1]
# transcript_to_gene = pd.Series(transcript_genes.values, index=mainmajor_R.index)

# 🎯 GO term마다 ΔTU, p-value 계산
go_summary = []

for go_term, gene_list in go_results.items():
    # 해당 GO term에 속한 transcript 찾기
    matching_transcripts = transcript_to_gene[transcript_to_gene.isin(gene_list)].index

    if len(matching_transcripts) < 3:
        continue  # transcript 수가 너무 적으면 스킵

    pre = salmajor_R.loc[matching_transcripts]
    post = salmajor_NR.loc[matching_transcripts]
    # pre = mainmajor_R.loc[matching_transcripts]
    # post = mainmajor_NR.loc[matching_transcripts]

    # transcript별 ΔTU 평균
    mean_R = pre.mean(axis=0)    # R 그룹 sample별 mean TU
    mean_NR = post.mean(axis=0)  # NR 그룹 sample별 mean TU

    # 그룹 평균끼리 빼기
    delta = mean_NR.mean() - mean_R.mean()  # ✅ 최종 기능 수준 차이
    #delta = post.mean(axis=1) - pre.mean(axis=1)
    
    # 그룹 평균끼리 paired t-test
    try:
        _, pval = mannwhitneyu(post.mean(axis=0), pre.mean(axis=0))
    except:
        continue

    go_summary.append({
        'GO_term': go_term,
        'delta': delta.mean(),
        'p_value': pval,
        'transnum': len(matching_transcripts),
        'genelist': matching_transcripts,
        'R_tu': pre
    })

# 🧾 결과 정리
go_df = pd.DataFrame(go_summary)
go_df['neg_log10_pval'] = -np.log10(go_df['p_value'] + 1e-10)

# 🔥 Volcano Plot 그리기
highlight_terms = ["double-strand break repair via homologous recombination (GO:0000724)","double-strand break repair (GO:0006302)" ] #"replication fork processing (GO:0031297)", "positive regulation of cell cycle (GO:0045787)", "cell cycle G1/S phase transition (GO:0044843)","regulation of G2/M transition of mitotic cell cycle (GO:0010389)","positive regulation of Wnt signaling pathway (GO:0030177)"]  # label 달고 싶은 GO term 리스트
# Create a color column based on whether the GO_term is in highlight_terms
go_df['color'] = go_df['GO_term'].apply(lambda x: '#90D6F0' if x in highlight_terms else '#FFCC29')

#%%
# Plot non-highlighted points first
plt.figure(figsize=(6, 6))

# 기준선
plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
        
plt.scatter(
    go_df.loc[go_df['color'] == '#FFCC29', 'delta'],
    go_df.loc[go_df['color'] == '#FFCC29', 'neg_log10_pval'],
    color="#FFCC29",#'#FFCC29', #"#81B214",
    alpha=0.5,
)

# Overlay highlighted points

# for i, row in go_df.iterrows():
#     if row['GO_term'] in highlight_terms:
#         plt.text(row['delta'], row['neg_log10_pval'] -0.25,  # y값 살짝 위로
#                  row['GO_term'], fontsize=9, ha='center', color='black')
# plt.scatter(
#     go_df.loc[go_df['color'] == '#90D6F0', 'delta'],
#     go_df.loc[go_df['color'] == '#90D6F0', 'neg_log10_pval'],
#     color='#FF4B4B',
#     alpha=0.9,
#     label='Highlighted'
# )

plt.xlabel('Mean ΔTU')
plt.ylabel('-log10(pval)')
plt.title('major DUTs - GO term level: salvage')
plt.xlim([-0.45, 0.54])
#plt.xlim([-0.18, 0.18])
plt.tight_layout()
sns.despine()
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/salvage_GOterm_volcano.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%
# Sort the DataFrame by p-value in ascending order and select the top 30
top_go_terms = go_df[go_df['transnum']>10] #salvage:10?
top_go_terms = top_go_terms[top_go_terms['delta'] > 0]
top_go_terms = top_go_terms[top_go_terms['p_value'] < 0.05]  # p-value < 0.05 필터링
#top_go_terms = go_df
top_go_terms = top_go_terms.sort_values(
    by=['p_value', 'delta'], 
    ascending=[True, False]  # p_value는 오름차순, delta는 내림차순
).head(50)# Normalize 'delta' values for color mapping
norm = plt.Normalize(top_go_terms['delta'].min(), top_go_terms['delta'].max())
cmap = plt.get_cmap('coolwarm')  #

fig, ax = plt.subplots(figsize=(7, 9))

# Create a horizontal bar plot
bars = ax.barh(top_go_terms['GO_term'], -np.log10(top_go_terms['p_value']), color='#154674')
ax.set_xlabel('-log10(pval)')

# bars = ax.barh(top_go_terms['GO_term'], top_go_terms['delta'], color='#154674')
# ax.set_xlabel('ΔTU')


# Adjust y-axis tick labels: set font size and ensure labels are fully visible
ax.set_yticklabels(top_go_terms['GO_term'], fontsize=10)

# Invert y-axis to have the lowest p-value at the top
ax.invert_yaxis()

# Adjust subplot parameters to give more room for y-axis labels
plt.subplots_adjust(left=0.65)  # Adjust the left margin as needed
sns.despine()
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/salvage_top50GOterm_barplot_filtered.pdf', dpi=300, bbox_inches='tight')

plt.show()# %%


# %%
###^^^ cibersort maintenance vs. salvage #########
ciber = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/cibersort/valcohort_immucellai.txt', sep='\t', index_col=0)
clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_validation_clinicalinfo_new.txt', sep='\t', index_col=0)

vallist = list(val_df.columns)
ciber = ciber.iloc[:,:-3]

ciber['group'] = clin['type']
ciber['OM/OS'] = clin['OM/OS']
ciber['SampleID'] = ciber.index
#ciber['ongoing'] = clin['ongoing']
#ciber = ciber[~((ciber['ongoing']==1) & (ciber['group']=='CR'))] ##without ongoing samples
#ciber = ciber[ciber['group']!='AR']
ciber['group'] = ciber['group'].replace({'CR': 'R', 'AR': 'R', 'IR': 'NR'})

##### salvage vs. maintenance ##########
ciber = ciber[ciber['OM/OS']=='salvage'] 
##### salvage vs. maintenance ##########


ciber.drop(columns=['OM/OS'], inplace=True)  # 'OM/OS' 열 제거
ciber_melted = ciber.melt(
    id_vars=['SampleID', 'group'],
    var_name='CellType',
    value_name='Proportion'
)

from statannotations.Annotator import Annotator

plt.figure(figsize=(18, 7))
ax = sns.boxplot(data=ciber_melted, x='CellType', y='Proportion', hue='group', palette={"R": "#FFBBBB", "NR": "#8DADFF"})
plt.title('salvage group')
plt.xlabel('')
plt.ylabel('Proportion')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Group',)

cell_types = ciber_melted['CellType'].unique()
pairs = [((cell, 'R'), (cell, 'NR')) for cell in cell_types]

annotator = Annotator(
    ax, pairs,
    data=ciber_melted,
    x='CellType', y='Proportion', hue='group',
    hue_order=['R', 'NR'],
)
annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
annotator.apply_and_annotate()

plt.tight_layout()
sns.despine()

plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/immucellai_salvage_boxplot.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
