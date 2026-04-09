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
####^^^ validation cohort check ########
val_df = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_TU.txt', sep='\t', index_col=0)
val_gene = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_gene_TPM.txt', sep='\t', index_col=0)
val_clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_validation_clinicalinfo_new.txt', sep='\t', index_col=0)
val_df = val_df.apply(pd.to_numeric, errors='coerce')

vallist = list(val_df.columns)
val_clin = val_clin.loc[vallist,:]

genesymbol =  pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM_symbol.txt', sep='\t', index_col=0)
genesymbol = genesymbol.iloc[:,-1]

##### omit list ######
omitlist = val_clin[(val_clin['response']==0)&(val_clin['ongoing']==2)].index.to_list()
val_clin = val_clin[~val_clin.index.isin(omitlist)]
val_df =  val_df.iloc[:-2,val_df.columns.isin(val_clin.index.to_list())]
val_gene =  val_gene.iloc[:,val_gene.columns.isin(val_clin.index.to_list())]
val_gene = pd.merge(val_gene,genesymbol,left_index=True,right_index=True)
##### only CR + IR ######

# %%
##^^ DEG ##############
r_list = val_clin[(val_clin['type']!='IR')].index.to_list()
nr_list = val_clin[(val_clin['type']=='IR')].index.to_list()

f_gene = val_gene.iloc[:,:-1]
genesym = val_gene[['Gene Symbol']]

deg_pval = []
r_df = val_gene.loc[:,r_list]
nr_df = val_gene.loc[:,nr_list]

for i in range(r_df.shape[0]):
    r_samples = r_df.iloc[i,:]
    nr_samples = nr_df.iloc[i,:]
    w, p = stats.mannwhitneyu(r_samples, nr_samples)
    deg_pval.append(p)

# Create a new DataFrame with geneid and respective p-values
result_df = pd.DataFrame({
    'p_value': deg_pval,
})
result_df.index = f_gene.index
result_df = pd.merge(result_df,genesym, how='inner', left_index=True, right_index=True)

### add FC ###

avg_r = r_df.mean(axis=1)
avg_nr = nr_df.mean(axis=1)

# Calculate the fold change as the log2 ratio of average post-treatment to pre-treatment expression
fold_change = np.log2(avg_r / avg_nr)
result_df['log2FC'] = fold_change

result_df.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/110_DEGresult_FC.txt', sep='\t')

#%%
##^^^ DUT #############

degresult = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/110_DEGresult_FC.txt', sep='\t')
DEGlist = set(degresult[degresult['p_value']<0.05]['Gene Symbol'])
nonDEGlist = set(degresult[degresult['p_value'] >= 0.05]['Gene Symbol'])
r_df = val_df.loc[:,r_list]
nr_df = val_df.loc[:,nr_list]

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

#%%
variable_result.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/110_variable_DUT_FC.txt', sep='\t')
stable_result.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/110_stable_DUT_FC.txt', sep='\t')

# %%
##^^^ DEG / DUT GO enrichment #######################

import gseapy as gp

stable_result = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/stable_DUT_FC.txt', sep='\t', index_col=0)
#glist = list(DEGlist)
glist = list(set(stable_result[(stable_result['p_value']<0.05) & (np.abs(stable_result['log2FC'])>1)]['Gene Symbol']))
#glist = list(set(variable_result[(variable_result['p_value']<0.05) & (np.abs(variable_result['log2FC']>1.5))]['Gene Symbol']))
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
file = file[file['Adjusted P-value']<0.01]
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
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/DEG_downregulated_GOplot.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%
###^^ major / minor DUT  KDE plot #################
stable_result = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/stable_DUT_FC.txt', sep='\t', index_col=0)
degresult = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/DEGresult_FC.txt', sep='\t')
DEGlist = set(degresult[degresult['p_value']<0.05]['Gene Symbol'])
nonDEGlist = set(degresult[degresult['p_value'] >= 0.05]['Gene Symbol'])

r_list = val_clin[(val_clin['type']=='CR')].index.to_list()
nr_list = val_clin[(val_clin['type']=='IR')].index.to_list()

r_df = val_df.loc[:,r_list]
nr_df = val_df.loc[:,nr_list]

r_df['Gene Symbol'] = r_df.index.str.split("-",n=1).str[-1]
nr_df['Gene Symbol'] = nr_df.index.str.split("-",n=1).str[-1]

majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = majorminor[majorminor['type']=='major']['gene_ENST'].to_list()

dutlist = stable_result.loc[(stable_result['p_value']<0.05) & (np.abs(stable_result['log2FC'])>1.5)].index.to_list()

R_major = r_df.loc[dutlist,r_list]
NR_major = nr_df.loc[dutlist,nr_list]

# R_major = r_df.loc[:,r_list] #### whole transcripts
# NR_major = nr_df.loc[:,nr_list] ##### whole transcripts 

R_major = R_major.loc[R_major.index.isin(majorlist),:]
NR_major = NR_major.loc[NR_major.index.isin(majorlist),:]

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
plt.xlim(right=0.44)
plt.xticks(np.arange(0,0.5,0.1))
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/majorDUT_KDEplot.pdf', dpi=300, bbox_inches='tight')
plt.show()


# %%
###^^^ cibersort #########
ciber = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/cibersort/valcohort_cibersort.txt', sep='\t', index_col=0)
clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_validation_clinicalinfo_new.txt', sep='\t', index_col=0)

vallist = list(val_df.columns)
ciber = ciber.iloc[:,:-3]

ciber['group'] = clin['type']
ciber['SampleID'] = ciber.index
#ciber['ongoing'] = clin['ongoing']
#ciber = ciber[~((ciber['ongoing']==1) & (ciber['group']=='CR'))] ##without ongoing samples
ciber = ciber[ciber['group']!='AR']
ciber['group'] = ciber['group'].replace({'CR': 'R', 'IR': 'NR'})


ciber_melted = ciber.melt(
    id_vars=['SampleID', 'group'],
    var_name='CellType',
    value_name='Proportion'
)

from statannotations.Annotator import Annotator

plt.figure(figsize=(18, 7))
ax = sns.boxplot(data=ciber_melted, x='CellType', y='Proportion', hue='group', palette={"R": "#FFBBBB", "NR": "#8DADFF"})
plt.title('R vs. NR')
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

#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/cibersortboxplot_RNR.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%
val_clin['group'] = val_clin['type'].replace({'CR': 'R', 'IR': 'NR'})
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
def plot_sample_mean_tu(val_df, val_clin, transcript_list, repairgenes, title):
    # transcript에서 gene 이름 추출
    val_df = val_df.copy()
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

    plt.title(f"{title} transcripts: DNA repair genes")
    plt.xlabel('')
    plt.tight_layout()
    sns.despine()
    plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/'+title+'_repair_boxplot.pdf', dpi=300, bbox_inches='tight')
    plt.show()


plot_sample_mean_tu(val_df, val_clin, majorlist, repairgenes, title='Major')
plot_sample_mean_tu(val_df, val_clin, minorlist, repairgenes, title='Minor')


# %%
#%%
from statannotations.Annotator import Annotator
##* only minor/major DUTs ##########
def plot_sample_mean_tu(val_df, val_clin, transcript_list, repairgenes, title):
    # transcript에서 gene 이름 추출
    val_df = val_df.copy()
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

    plt.title(f"{title} DUTs: DNA repair genes")
    plt.xlabel('')
    plt.tight_layout()
    sns.despine()
    plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/'+title+'_DUT_repair_boxplot.pdf', dpi=300, bbox_inches='tight')
    
    plt.show()


plot_sample_mean_tu(val_df, val_clin, majorlist, repairgenes, title='Major')
plot_sample_mean_tu(val_df, val_clin, minorlist, repairgenes, title='Minor')

# %%
