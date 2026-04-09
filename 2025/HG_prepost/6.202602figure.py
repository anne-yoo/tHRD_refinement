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
from statannotations.Annotator import Annotator
from statannot import add_stat_annotation
rcParams['pdf.fonttype'] = 42  
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'Helvetica'

plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 13,    # x축 틱 라벨 글꼴 크기

'ytick.labelsize': 13,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 13,
'legend.title_fontsize': 13, # 범례 글꼴 크기|
'figure.titlesize': 15    # figure 제목 글꼴 크기
})
sns.set_style("ticks")

# %%
####^^ (1-1) group 1 증가 GO enrichment######
sqanti = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/sqantioutput/sqanti_hg19_classification.txt', sep='\t')
sqanti.dropna(axis=1, how='all', inplace=True)

AR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/whole_AR_stable_DUT_Wilcoxon_delta_withna.txt', sep='\t', index_col=0)
IR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/whole_IR_stable_DUT_Wilcoxon_delta_withna.txt', sep='\t', index_col=0)
# AR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/maintenance/AR_stable_DUT_Wilcoxon_delta_withna.txt', sep='\t', index_col=0) #^Only maintenance
# IR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/maintenance/IR_stable_DUT_Wilcoxon_delta_withna.txt', sep='\t', index_col=0) #^Only maintenance

ARdutlist = AR_dut.loc[(AR_dut['p_value']<0.05) & (np.abs(AR_dut['delta_TU'])>0.05)].index.to_list()
IRdutlist = IR_dut.loc[(IR_dut['p_value']<0.05) & (np.abs(IR_dut['delta_TU'])>0.05)].index.to_list()

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost_80_clinicalinfo.txt', sep='\t', index_col=0)
sampleinfo_full = sampleinfo.copy()
#sampleinfo = sampleinfo[sampleinfo['purpose']=='maintenance'] #^ Only maintenance

ARlist = list(set(sampleinfo.loc[(sampleinfo['response']==1),'sample_full']))
IRlist = list(set(sampleinfo.loc[(sampleinfo['response']==0),'sample_full']))

transexp = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_80_transcript_TPM.txt',sep='\t', index_col=0)
transexp = transexp.iloc[:,:-1]
transexp = transexp.loc[(transexp > 0).sum(axis=1) >= 8] #20% 이상에서는 나오긴 해야됨 ...

#transexp = transexp.loc[(transexp > 0).sum(axis=1) >= transexp.shape[1]*0.3]
transexp["gene"] = transexp.index.str.split("-", n=1).str[-1]
gene_sum = transexp.groupby("gene").transform("sum")
filtered_trans = transexp.iloc[:, :-1].div(gene_sum)
TU = filtered_trans.copy()

mainlist =(list(set(sampleinfo[sampleinfo['purpose']=='maintenance']['sample_full'])))
sallist = (list(set(sampleinfo_full[sampleinfo_full['purpose']=='salvage']['sample_full'])))

main_TU = TU[mainlist]
sal_TU = TU[sallist]

main_TU = main_TU.sort_index(axis=1)
main_TU.columns = main_TU.columns.str[:-4]

sal_TU = sal_TU.sort_index(axis=1)
sal_TU.columns = sal_TU.columns.str[:-4]

majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_80_majorminorlist.txt',sep='\t')
majorlist = majorminor[majorminor['type']=='major']['Transcript-Gene'].to_list()
minorlist = majorminor[majorminor['type']=='minor']['Transcript-Gene'].to_list()

sampleinfo = sampleinfo.iloc[::2,:]

proteincoding = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM_symbol.txt', sep='\t', index_col=0)
proteincodinglist = proteincoding['Gene Symbol'].to_list()

TU = filtered_trans.copy()
#TU = TU.loc[:,TU.columns.isin(mainlist)] #^Only maintenance
TU = TU[TU.index.str.split("-", n=1).str[-1].isin(proteincodinglist)]
TU.columns = TU.columns.str[:-4] 

# 인덱스 정리 (Transcript ID만 남김)
TU.index = TU.index.str.split("-", n=1).str[0]

# Pre/Post 분리
preTU = TU.iloc[:, 1::2] 
postTU = TU.iloc[:, 0::2]

preTU = preTU.fillna(0)
postTU = postTU.fillna(0)

# ---------------------------------------------------------
# 2. 샘플 분리 및 리스트 전처리
# ---------------------------------------------------------
# (1) 샘플 분리
ar_samples = sampleinfo[sampleinfo['response'] == 1].index.intersection(preTU.columns)
ir_samples = sampleinfo[sampleinfo['response'] == 0].index.intersection(preTU.columns)

# (2) DUT List 전처리 (ID만 추출)
ar_isoforms_clean = [x.split('-', 1)[0] for x in ARdutlist]
ir_isoforms_clean = [x.split('-', 1)[0] for x in IRdutlist]

# ---------------------------------------------------------
# 3. 데이터 집계 (Aggregation): 샘플별 평균 계산
# ---------------------------------------------------------
# Coding 여부(True/False)와 Response(AR/IR)에 따라 반복문을 돌며 평균을 계산합니다.

# df_cat의 isoform 컬럼도 ID만 남도록 정리 필요 (안전장치)
df_cat = sqanti[['isoform','structural_category','subcategory','within_CAGE_peak','coding']]
df_cat['isoform_clean'] = df_cat['isoform'].astype(str).str.split('-', n=1).str[0]
majorlist_set = set([x.split('-', 1)[0] for x in majorlist])
coding_set = set(sqanti[sqanti['coding']=='coding']['isoform'])
majorlist_set = majorlist_set.intersection(coding_set)
df_cat['major'] = df_cat['isoform'].isin(majorlist_set)
####filter#####
valid_cat = {"full-splice_match", "novel_in_catalog"}
df_cat['major'] = df_cat['major'] & df_cat['structural_category'].isin(valid_cat)
df_cat.set_index('isoform', inplace=True)
###############

group1 = df_cat[df_cat['major']==True]['isoform_clean'].to_list()
group2 = df_cat[(df_cat['major']==False)&(df_cat['coding']=='coding')]['isoform_clean'].to_list()
group3 = df_cat[(df_cat['major']==False)&(df_cat['coding']=='non_coding')]['isoform_clean'].to_list()

class1 = majorminor[majorminor['transcriptid'].isin(group1)]['Transcript-Gene'].to_list()
class2 = majorminor[majorminor['transcriptid'].isin(group2)]['Transcript-Gene'].to_list()
class3 = majorminor[majorminor['transcriptid'].isin(group3)]['Transcript-Gene'].to_list()

#%%
####^^ sampleinfo fig ########

## piechart for BRCAmut
counts = sampleinfo[sampleinfo['response']==0]['BRCAmut'].value_counts().sort_index()
labels = ['BRCAwt', 'BRCAmt']  # 0, 1 순서

colors = sns.color_palette('husl', n_colors=len(counts))
plt.figure(figsize=(3, 3))
wedges, texts, autotexts = plt.pie(
    counts,
    labels=None,   # label 제거
    autopct='%1.1f%%',
    startangle=90,
    textprops={'fontsize': 12},
    labeldistance=1.2,
    colors=colors
)
plt.axis('equal')  # 원형 유지
plt.legend(wedges, labels, loc='center left', bbox_to_anchor=(1, 0.5))
#plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/sampleinfo_IR_BRCAmut_piechart.pdf', dpi=300, bbox_inches='tight')
plt.show()

## piechart for drug
counts = sampleinfo[sampleinfo['response']==1]['drug'].value_counts()
colors = sns.color_palette('husl', n_colors=len(counts))
labels = ['Olaparib','Niraparib','Rucaparib']  # 0, 1 순서
plt.figure(figsize=(3, 3))
wedges, texts, autotexts = plt.pie(
    counts,
    labels=None,   # label 제거
    autopct='%1.1f%%',
    startangle=90,
    textprops={'fontsize': 12},
    labeldistance=0.5,
    colors=colors
)
plt.axis('equal')
plt.legend(wedges, labels, loc='center left', bbox_to_anchor=(1, 0.5))
#plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/sampleinfo_IR_drug_piechart.pdf', dpi=300, bbox_inches='tight')
plt.show()

## line info
sampleinfo['line_num'] = sampleinfo['line'].str.extract(r'(\d+)').astype(int)
sampleinfo['line_num'] = pd.Categorical(
    sampleinfo['line_num'],
    categories=sorted(sampleinfo['line_num'].dropna().unique()),
    ordered=True
)
sampleinfo['Group'] = sampleinfo['response'].map({0: 'IR', 1: 'AR'})
plt.figure(figsize=(5.5, 3))
ax=sns.countplot(
    data=sampleinfo,
    x='line_num',
    hue='Group',
    palette={"AR": "#FEB24C", "IR": "#5AAE61"}
)
from matplotlib.ticker import MaxNLocator
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.xlabel('Line')
plt.ylabel('Count')
plt.tight_layout()
sns.despine()
#plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/sampleinfo_line_countplot.pdf', dpi=300, bbox_inches='tight')
plt.show()

##interval
palette = sns.color_palette('Set2', 2)

plt.figure(figsize=(5, 4))

ax = sns.boxplot(
    data=sampleinfo,
    x='interval',
    y='Group',
    order=['IR', 'AR'],
    whis=1.5,
    linewidth=1.5,
    fliersize=4,
    width=0.6,
    palette={"AR": "#FEB24C", "IR": "#5AAE61"},
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5)
)

plt.xlabel('Days')
plt.ylabel('Group')
plt.tight_layout()
sns.despine()
#plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/sampleinfo_interval_boxplot.pdf', dpi=300, bbox_inches='tight')
plt.show()

## class 1 /3 venn diagram
AR_dutlist = set(ARdutlist).intersection(set(class1))
IR_dutlist = set(IRdutlist).intersection(set(class1))
from matplotlib_venn import venn2
plt.figure(figsize=(4,4))

v = venn2(
    [AR_dutlist, IR_dutlist],
    set_labels=('', '')  # label 제거
)

# 색 지정
v.get_patch_by_id('10').set_color('#FEB24C')  # AR only
v.get_patch_by_id('01').set_color('#5AAE61')  # IR only
v.get_patch_by_id('11').set_color('#ACB056')  # overlap (원하면 변경)

# 투명도 (겹치는 느낌 살리기)
for patch in ['10', '01', '11']:
    if v.get_patch_by_id(patch):
        v.get_patch_by_id(patch).set_alpha(0.7)
for text in v.subset_labels:
    if text:
        text.set_fontsize(16)
        text.set_weight('bold')
plt.tight_layout()
#plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/Class1DUT_venn.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%
## DUT pre / post upregulated barplot
ARuplist = IR_dut.loc[(IR_dut['p_value']<0.05) & (IR_dut['delta_TU']>0.05)].index.to_list()
ARdownlist = IR_dut.loc[(IR_dut['p_value']<0.05) & (IR_dut['delta_TU']<-0.05)].index.to_list()

class_map = {}

for t in class1:
    class_map[t] = 'Class1'
for t in class2:
    class_map[t] = 'Class2'
for t in class3:
    class_map[t] = 'Class3'

# up
up_df = pd.DataFrame({
    'transcript': ARuplist,
    'Class': [class_map.get(t, None) for t in ARuplist],
    'Direction': 'Post-up'
})

# down
down_df = pd.DataFrame({
    'transcript': ARdownlist,
    'Class': [class_map.get(t, None) for t in ARdownlist],
    'Direction': 'Pre-up'
})

plot_df = pd.concat([up_df, down_df], ignore_index=True)

# class 없는 애들 제거
plot_df = plot_df.dropna(subset=['Class'])

palette = {
    'Post-up': '#E66C5C',
    'Pre-up': '#5780D8'
}

plt.figure(figsize=(6,3))

sns.countplot(
    data=plot_df,
    x='Class',
    hue='Direction',
    order=['Class1', 'Class2', 'Class3'],
    hue_order=['Post-up', 'Pre-up'],
    palette=palette,)

plt.xlabel('')
plt.ylabel('Count')
plt.legend(title='')
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
sns.despine()
plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/IR_Class1DUT_barplot.pdf', dpi=300, bbox_inches='tight')
plt.show()


# %%

# %%

#%%
####^^ (1-1) group 1 증가 GO enrichment######
import gseapy as gp

ARuplist = AR_dut.loc[(AR_dut['p_value']<0.05) & (AR_dut['delta_TU']>0.05)].index.to_list()
ARdownlist = AR_dut.loc[(AR_dut['p_value']<0.05) & (AR_dut['delta_TU']<-0.05)].index.to_list()
IRuplist = IR_dut.loc[(IR_dut['p_value']<0.01) & (IR_dut['delta_TU']<-0.05)].index.to_list()

tlist = set(ARuplist).intersection(set(class1))
tlist2 = set(ARdownlist).intersection(set(class3))

glist = list(set([x.split('-', 1)[-1] for x in list(tlist)]))
glist2 = list(set([x.split('-', 1)[-1] for x in list(tlist2)]))

print(len(glist))

enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                gene_sets=['GO_Biological_Process_2021',
                           'Reactome_2022'
                           #'GO_Biological_Process_2018','GO_Biological_Process_2023','Reactome_2022','Reactome_Pathways_2024',
                           #'MSigDB_Hallmark_2020'
                           ], 
                organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                outdir=None, # don't write to disk
)

enrresult = enr.results.sort_values(by=['Adjusted P-value']) 
print(enrresult[enrresult['Adjusted P-value']<0.1]['Term'])
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
file = file.iloc[:20,:]
file = file[file['Adjusted P-value']<0.1]
file['Adjusted P-value'] = -np.log10(file['Adjusted P-value'])

cellcycle = set(
    gene
    for genes in file.iloc[:4, :]['Genes']
    for gene in genes.split(';')
)
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
plt.figure(figsize=(7,11)) #7,11
sns.set_style("whitegrid")
scatter = sns.scatterplot(
    data=file, x='Adjusted P-value', y='Term', hue='per', palette='coolwarm', edgecolor=None, legend=False, s=80
)
plt.xlabel('-log10(FDR)')
plt.ylabel('')
#plt.yticks(fontsize=13)
#plt.xscale('log')  # Log scale for better visualizationf

# Expanding the plot layout to make room for GO term labels
plt.gcf().subplots_adjust(left=0.4)
#plt.gcf().subplots_adjust(right=0.8)

# Creating color bar
norm = plt.Normalize(file['per'].min(), file['per'].max())
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
sm.set_array([])

# Displaying color bar
cbar = plt.colorbar(sm)
#cbar.set_label('Overlap Percentage')
#plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/40_upclass1_GOtop20.pdf', dpi=300, bbox_inches='tight')
plt.show()

enr = gp.enrichr(gene_list=glist2, # or "./tests/data/gene_list.txt",
                gene_sets=['GO_Biological_Process_2021',
                           'Reactome_2022',
                           #'GO_Biological_Process_2018','GO_Biological_Process_2023','Reactome_2022',
                           #'MSigDB_Hallmark_2020'
                           ], 
                organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                outdir=None, # don't write to disk
)

enrresult = enr.results.sort_values(by=['Adjusted P-value']) 
print(enrresult[enrresult['Adjusted P-value']<0.1]['Term'])
file = enrresult
def string_fraction_to_float(fraction_str):
    numerator, denominator = fraction_str.split('/')
    return float(numerator) / float(denominator)

file['per'] = file['Overlap'].apply(string_fraction_to_float)

file = file.sort_values('Adjusted P-value')
#file = file.sort_values(by='Combined Score', ascending=False)
##remove mitochondrial ##
file['Term'] = file['Term'].str.rsplit(" ",n=1).str[0] #for reactome2022
#file = file[~file['Term'].str.contains('mitochondrial')]
file = file.iloc[:20,:]
file = file[file['Adjusted P-value']<0.1]
file['Adjusted P-value'] = -np.log10(file['Adjusted P-value'])

# dsbrepair = set(
#     gene
#     for genes in file.iloc[[0,1,3,6], :]['Genes']
#     for gene in genes.split(';')
# )

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
plt.figure(figsize=(7,11)) #7,11
sns.set_style("whitegrid")
scatter = sns.scatterplot(
    data=file, x='Adjusted P-value', y='Term', hue='per', palette='coolwarm', edgecolor=None, legend=False, s=80
)
plt.xlabel('-log10(FDR)')
plt.ylabel('')
#plt.yticks(fontsize=13)
#plt.xscale('log')  # Log scale for better visualizationf

# Expanding the plot layout to make room for GO term labels
plt.gcf().subplots_adjust(left=0.4)
#plt.gcf().subplots_adjust(right=0.8)

# Creating color bar
norm = plt.Normalize(file['per'].min(), file['per'].max())
sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
sm.set_array([])

# Displaying color bar
cbar = plt.colorbar(sm)
#cbar.set_label('Overlap Percentage')
#plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/40_downclass3_GOtop20.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%

##^^^^ top 5 from class 1 / 3 ###########

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ARuplist = AR_dut.loc[(AR_dut['p_value']<0.05) & (AR_dut['delta_TU']>0.05)].index.to_list()
ARdownlist = AR_dut.loc[(AR_dut['p_value']<0.05) & (AR_dut['delta_TU']<-0.05)].index.to_list()
IRuplist = IR_dut.loc[(IR_dut['p_value']<0.05) & (IR_dut['delta_TU']>0.05)].index.to_list()
IRdownlist = IR_dut.loc[(IR_dut['p_value']<0.05) & (IR_dut['delta_TU']<-0.05)].index.to_list()

tlist = set(ARuplist).intersection(set(class1))
tlist2 = set(ARdownlist).intersection(set(class3))

glist = list(set([x.split('-', 1)[-1] for x in list(tlist)]))
glist2 = list(set([x.split('-', 1)[-1] for x in list(tlist2)]))

def get_top_enrichment(gene_list, label):
    # Reactome_2022로 분석 수행
    enr = gp.enrichr(gene_list=gene_list,
                    gene_sets=['GO_Biological_Process_2021','Reactome_2022',], 
                    organism='human',
                    outdir=None)
    
    res = enr.results.copy()
    # P-value 기준 정렬 및 데이터 가공
    res = res.sort_values(by='Adjusted P-value')
    res['Term'] = res['Term'].str.rsplit(" ", n=1).str[0] # 뒤쪽 ID 제거
    res['-log10(FDR)'] = -np.log10(res['Adjusted P-value'])
    
    # 상위 5개 추출 및 라벨링
    top5 = res.head(5).copy()
    top5['Group'] = label
    return top5

# 1. 데이터 가져오기
top5_up = get_top_enrichment(glist, 'AR Upregulated (Class 1)')
top5_down = get_top_enrichment(glist2, 'AR Downregulated (Class 3)')

# 1. 시각화를 위한 통합 데이터프레임 생성
df_plot = pd.concat([top5_up, top5_down]).reset_index(drop=True)

# 2. 그래프 설정
plt.rcParams["font.family"] = "Arial"
fig, ax = plt.subplots(figsize=(10, 8))

# 3. 바 플롯 그리기
sns.barplot(
    data=df_plot, 
    x='-log10(FDR)', 
    y='Term', 
    palette=["#FF9616"]*5+["#1E9652"]*5,
    ax=ax
)

# 4. 커스텀 범례(Legend) 생성
legend_elements = [
    Patch(facecolor='#FF9616', label='Upregulated Class 1 DUT'),
    Patch(facecolor='#1E9652', label='Downregulated Class 3 DUT')
]

# 왼쪽 상단(upper left)에 범례 추가
ax.legend(handles=legend_elements, 
          loc='upper right', 
          bbox_to_anchor=(-0.9, 1), 
          frameon=False, 
          fontsize=12, 
          handlelength=1.5)
plt.subplots_adjust(left=0.5)

# 5. 스타일링 및 마무리
ax.set_xlabel('-log10(FDR)', fontsize=13)
ax.set_ylabel('', fontsize=14)
ax.grid(axis='x', linestyle='--', alpha=0.5)

# x축 범위를 데이터에 맞춰 자동 조절 (FDR 차이가 커도 반영됨)
ax.set_xlim(0, max(df_plot['-log10(FDR)']) * 1.1)

sns.despine()
plt.tight_layout()
#plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/40_top5_GObarplot.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%
##^^ top 5 그룹별 delta TU boxplot ####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("ticks")

go_input_transcript_sets = {
    "AR Upregulated (Class 1)": sorted(set(ARuplist) & set(class1)),
    "AR Downregulated (Class 3)": sorted(set(ARdownlist) & set(class3)),
    "IR Upregulated (Class 1)": sorted(set(IRuplist) & set(class1)),
    "IR Downregulated (Class 3)": sorted(set(IRdownlist) & set(class3)),
}

def build_term_sample_deltatu_df_filtered(
    df_plot,
    delta_TU_gene,
    ar_samples,
    ir_samples,
    go_input_transcript_sets,
    min_genes_matched=1
):

    common_samples = delta_TU_gene.columns.intersection(list(ar_samples) + list(ir_samples))
    ar_samples_use = [s for s in ar_samples if s in common_samples]
    ir_samples_use = [s for s in ir_samples if s in common_samples]

    idx_series = pd.Series(delta_TU_gene.index, index=delta_TU_gene.index)
    gene_from_index = idx_series.str.split("-", n=1).str[1]

    rows = []

    for _, row in df_plot.iterrows():
        term = row["Term"]
        group = row["Group"]

        if group not in go_input_transcript_sets:
            continue

        term_genes = [g.strip() for g in str(row["Genes"]).split(";") if g.strip() != ""]
        allowed_transcripts = set(go_input_transcript_sets[group])

        # 1) GO input에 실제 사용된 transcript만 남김
        in_input_mask = idx_series.isin(allowed_transcripts)

        # 2) 그 중에서 term gene에 해당하는 transcript만 남김
        gene_match_mask = gene_from_index.isin(term_genes)

        final_mask = in_input_mask & gene_match_mask
        sub = delta_TU_gene.loc[final_mask].copy()

        matched_transcripts = idx_series[final_mask].tolist()
        matched_genes = sorted(set(gene_from_index[final_mask].dropna()))

        n_matched_genes = len(matched_genes)
        n_features = sub.shape[0]

        if n_matched_genes < min_genes_matched or n_features == 0:
            continue

        sample_mean = sub.mean(axis=0)

        for s in ar_samples_use:
            if s in sample_mean.index:
                rows.append({
                    "Term": term,
                    "Group": group,
                    "sample": s,
                    "response": "AR",
                    "mean_delta_TU": sample_mean.loc[s],
                    "n_features": n_features,
                    "matched_genes": ";".join(matched_genes),
                    "matched_transcripts": ";".join(matched_transcripts)
                })

        for s in ir_samples_use:
            if s in sample_mean.index:
                rows.append({
                    "Term": term,
                    "Group": group,
                    "sample": s,
                    "response": "IR",
                    "mean_delta_TU": sample_mean.loc[s],
                    "n_features": n_features,
                    "matched_genes": ";".join(matched_genes),
                    "matched_transcripts": ";".join(matched_transcripts)
                })

    plot_df = pd.DataFrame(rows)
    return plot_df

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("ticks")

def plot_term_sample_deltatu_boxplot(
    plot_df,
    term_order=None,
    save_path=None,
    figsize=(14, 5)
):
    if plot_df.empty:
        raise ValueError("plot_df is empty.")

    df = plot_df.copy()

    if term_order is None:
        term_order = df["Term"].drop_duplicates().tolist()

    palette = {"AR": "#FEB24C", "IR": "#5AAE61"}

    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=df,
        x="Term",
        y="mean_delta_TU",
        hue="response",
        order=term_order,
        hue_order=["AR", "IR"],
        palette=palette,
        showfliers=False,
        width=0.75,
        ax=ax
    )

    sns.stripplot(
        data=df,
        x="Term",
        y="mean_delta_TU",
        hue="response",
        order=term_order,
        hue_order=["AR", "IR"],
        dodge=True,
        palette=palette,
        size=4,
        alpha=0.75,
        edgecolor="none",
        ax=ax
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], frameon=False, title="", loc="upper right")

    ax.set_xlabel("")
    ax.set_ylabel("Mean delta TU")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

plot_df_term = build_term_sample_deltatu_df_filtered(
    df_plot=df_plot,
    delta_TU_gene=delta_TU_gene,
    ar_samples=ar_samples,
    ir_samples=ir_samples,
    go_input_transcript_sets=go_input_transcript_sets,
    min_genes_matched=1
)

print(plot_df_term.shape)
print(plot_df_term[["Term", "Group", "n_features", "matched_genes"]].drop_duplicates().head(10))

term_order = df_plot["Term"].drop_duplicates().tolist()

plot_term_sample_deltatu_boxplot(
    plot_df_term,
    term_order=term_order,
    # save_path="/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/top10_term_mean_deltaTU_boxplot_filtered.pdf"
)

#%%
#####^^ 모든 샘플 공통 DUT ##################
import numpy as np
import pandas as pd

def find_majority_direction_dut_transcripts(
    delta_TU_gene,
    transcript_list,
    samples,
    direction="up",
    min_agree_frac=0.8,
    min_non_na_frac=1.0,
    allow_zero=False
):
    use_ids = [x for x in transcript_list if x in delta_TU_gene.index]
    use_samples = [s for s in samples if s in delta_TU_gene.columns]

    X = delta_TU_gene.loc[use_ids, use_samples].copy()

    n_samples = len(use_samples)
    n_non_na = X.notna().sum(axis=1)
    non_na_frac = n_non_na / n_samples

    if direction == "up":
        sign_mask = X.ge(0) if allow_zero else X.gt(0)
    elif direction == "down":
        sign_mask = X.le(0) if allow_zero else X.lt(0)
    else:
        raise ValueError("direction must be 'up' or 'down'")

    agree_frac = sign_mask.mean(axis=1)

    summary_df = pd.DataFrame(index=X.index)
    summary_df["n_samples"] = n_samples
    summary_df["n_non_na"] = n_non_na
    summary_df["non_na_frac"] = non_na_frac
    summary_df["agree_frac"] = agree_frac
    summary_df["mean_delta_TU"] = X.mean(axis=1)
    summary_df["median_delta_TU"] = X.median(axis=1)
    summary_df["min_delta_TU"] = X.min(axis=1)
    summary_df["max_delta_TU"] = X.max(axis=1)

    keep = (summary_df["non_na_frac"] >= min_non_na_frac) & (summary_df["agree_frac"] >= min_agree_frac)
    ids = summary_df.index[keep].tolist()

    return ids, summary_df.loc[ids].sort_values("agree_frac", ascending=False)

AR_class1_dut = sorted(set(ARuplist) & set(class1))
AR_class3_dut = sorted(set(ARdownlist) & set(class3))

AR_class1_majority_up, AR_class1_majority_up_df = find_majority_direction_dut_transcripts(
    delta_TU_gene=delta_TU_gene,
    transcript_list=AR_class1_dut,
    samples=ar_samples,
    direction="up",
    min_agree_frac=0.8
)

AR_class3_majority_down, AR_class3_majority_down_df = find_majority_direction_dut_transcripts(
    delta_TU_gene=delta_TU_gene,
    transcript_list=AR_class3_dut,
    samples=ar_samples,
    direction="down",
    min_agree_frac=0.8
)

#%%
#revigo = enrresult[enrresult['Adjusted P-value']<0.1]
#revigo = revigo[['Term','Adjusted P-value']]
# revigo['Term'] = revigo['Term'].str.extract(r'\((.*?)\)')
#revigo.set_index('Term', inplace=True)
# revigo.to_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/maintenance/AR_increased_group1DUT_GO_005.txt', sep='\t', index=False)


#%%
####^^ group 123 volcano ########

from adjustText import adjust_text  # 라벨 겹침 방지 (없으면 pip install adjustText)
sns.set_style("ticks")
def plot_volcano_grid_internal_legend(datasets, groups, x_col='delta_TU', y_col='p_value', 
                                      fc_cutoff=0.05, p_cutoff=0.05, save_path=None):
    """
    각 서브플롯 내부에 개별 legend를 표시하는 버전
    """
    fig, axes = plt.subplots(3, 2, figsize=(11, 13)) #(18, 11) (11,13)
    sns.set_style("ticks")
    
    data_keys = ['AR', 'IR']
    colors = {'Up': '#E66C5C', 'Down': '#5780D8', 'Sig_Only': 'grey', 'NS': '#D3D3D3'}
    alphas = {'Up': 0.8, 'Down': 0.8, 'Sig_Only': 0.4, 'NS': 0.2}

    for i, d_key in enumerate(data_keys):
        for j, group in enumerate(groups):
            ax = axes[j, i]
            
            # 데이터 처리
            df_raw = datasets[d_key].copy()
            df_raw.index = df_raw.index.str.split("-", n=1).str[0]
            df = df_raw.loc[df_raw.index.isin(group), :].copy()
            
            if df.empty:
                ax.text(0.5, 0.5, "No Data", ha='center')
                continue

            df['nlog10'] = -np.log10(df[y_col])
            conditions = [
                (df[y_col] < p_cutoff) & (df[x_col] > fc_cutoff),
                (df[y_col] < p_cutoff) & (df[x_col] < -fc_cutoff),
                (df[y_col] < p_cutoff) & (df[x_col].abs() <= fc_cutoff),
                (df[y_col] >= p_cutoff)
            ]
            choices = ['Up', 'Down', 'Sig_Only', 'NS']
            df['category'] = np.select(conditions, choices, default='NS')

            # 산점도 그리기
            for cat in ['NS', 'Sig_Only', 'Down', 'Up']:
                subset = df[df['category'] == cat]
                if subset.empty: continue
                
                # Sig_Only는 범례에서 제외 (기존 코드 유지)
                label_name = cat if cat != 'Sig_Only' else '_nolegend_'
                
                ax.scatter(x=subset[x_col], y=subset['nlog10'], 
                           c=colors[cat], alpha=alphas[cat], 
                           linewidth=0, s=20,
                           label=label_name)

            # 임계선 추가
            ax.axhline(-np.log10(p_cutoff), linestyle='--', color='black', alpha=0.3, linewidth=1)
            ax.axvline(fc_cutoff, linestyle='--', color='black', alpha=0.3, linewidth=1)
            ax.axvline(-fc_cutoff, linestyle='--', color='black', alpha=0.3, linewidth=1)

            # --- 범례 설정 (각 subplot 내부) ---
            # frameon=False로 테두리 제거, loc='upper right' 혹은 'best'
            ax.legend(loc='upper right', frameon=False, fontsize=11, markerscale=1.2)

            # 축 및 제목 설정
            ax.set_title(f'{d_key} (Class {j+1})', fontsize=14, fontweight='bold')
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(-0.2, 5.2)
            ax.set_xlabel(r'$\Delta$ TU', fontsize=13)
            ax.set_ylabel(r'$-log_{10}p$', fontsize=13)
            
            sns.despine(ax=ax)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()

# 실행_path는 필요에 따라 지정하세요
datasets = {'AR': AR_dut, 'IR': IR_dut}
groups = [group1, group2, group3]
plot_volcano_grid_internal_legend(datasets, groups, save_path='/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/AR+IR_Class_volcano.png')#save_path='/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/40_DUTvolcano.pdf')

#%%
####^^ Class1~3 DUT boxplot #########

import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

sns.set_style("ticks")

palette = {
    "AR_Pre": "#FFEDA0",
    "AR_Post": "#FEB24C",
    "IR_Pre": "#D9F0D3",
    "IR_Post": "#5AAE61"
}

def p_to_star(p):
    if pd.isna(p):
        return "ns"
    if p < 1e-4:
        return "****"
    elif p < 1e-3:
        return "***"
    elif p < 1e-2:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def extract_pair_id(sample_name):
    """
    Example:
      SV-OV-P035-atD -> SV-OV-P035
      SV-OV-P035-bfD -> SV-OV-P035
    """
    sample_name = str(sample_name)
    sample_name = re.sub(r"-(atD|bfD)$", "", sample_name, flags=re.IGNORECASE)
    return sample_name


def split_time_cols(cols, pre_label="bfD", post_label="atD"):
    """
    Default:
      Pre  = bfD
      Post = atD

    If your biological meaning is opposite, just swap:
      pre_label="atD", post_label="bfD"
    """
    cols = [str(c) for c in cols]
    pre_cols = [c for c in cols if c.lower().endswith(f"-{pre_label.lower()}")]
    post_cols = [c for c in cols if c.lower().endswith(f"-{post_label.lower()}")]

    if len(pre_cols) == 0 or len(post_cols) == 0:
        raise ValueError(
            f"Could not detect paired columns with suffixes "
            f"pre='-{pre_label}', post='-{post_label}'.\n"
            f"Found pre_cols={len(pre_cols)}, post_cols={len(post_cols)}"
        )
    return pre_cols, post_cols


def restrict_samples_by_group(df, sample_list):
    sample_set = set(sample_list)

    keep_cols = [
        c for c in df.columns
        if extract_pair_id(c) in sample_set
    ]

    return df[keep_cols].copy()


def compute_paired_sample_meanTU(
    filtered_trans,
    transcript_list,
    sample_list,
    fillna=False,
    pre_label="bfD",
    post_label="atD"
):
    """
    filtered_trans: transcript x sample dataframe
    transcript_list: selected transcripts for this panel
    sample_list: ar_samples or ir_samples
    fillna=False -> transcript-level NaN ignored
    fillna=True  -> transcript-level NaN filled with 0 first

    Returns:
      paired_df with columns [PairID, Pre, Post]
    """
    # transcript subset
    tx = [t for t in transcript_list if t in filtered_trans.index]
    if len(tx) == 0:
        return pd.DataFrame(columns=["PairID", "Pre", "Post"])

    sub = filtered_trans.loc[tx].copy()

    # sample subset (AR panel gets only AR samples; IR panel gets only IR samples)
    sub = restrict_samples_by_group(sub, sample_list)
    if sub.shape[1] == 0:
        return pd.DataFrame(columns=["PairID", "Pre", "Post"])

    if fillna:
        sub = sub.fillna(0)

    pre_cols, post_cols = split_time_cols(
        sub.columns,
        pre_label=pre_label,
        post_label=post_label
    )

    # sample-wise mean across selected transcripts
    mean_pre = sub[pre_cols].mean(axis=0)
    mean_post = sub[post_cols].mean(axis=0)

    pre_df = pd.DataFrame({
        "SampleCol": mean_pre.index,
        "Pre": mean_pre.values
    })
    pre_df["PairID"] = pre_df["SampleCol"].map(extract_pair_id)

    post_df = pd.DataFrame({
        "SampleCol": mean_post.index,
        "Post": mean_post.values
    })
    post_df["PairID"] = post_df["SampleCol"].map(extract_pair_id)

    paired = pd.merge(
        pre_df[["PairID", "Pre"]],
        post_df[["PairID", "Post"]],
        on="PairID",
        how="inner"
    ).dropna(subset=["Pre", "Post"])

    return paired


def add_stat_annotation(ax, x1, x2, y, h, text, fontsize=11):
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, c="dimgray")
    ax.text(
        (x1 + x2) / 2,
        y + h,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color="dimgray"
    )


def plot_meanTU_box_grid_by_class(
    filtered_trans,
    ARdutlist,
    IRdutlist,
    class1,
    class2,
    class3,
    ar_samples,
    ir_samples,
    fillna=False,
    pre_label="bfD",
    post_label="atD",
    save_path=None
):
    """
    3x2 grid:
      AR Class1 | IR Class1
      AR Class2 | IR Class2
      AR Class3 | IR Class3

    For each panel:
      AR Class k uses transcripts in (ARdutlist ∩ classk) and samples in ar_samples
      IR Class k uses transcripts in (IRdutlist ∩ classk) and samples in ir_samples

    y-axis is shared only within each row (same class), not globally.
    """

    class_groups = [class1, class2, class3]

    fig, axes = plt.subplots(3, 2, figsize=(11, 13))

    for row_idx, class_tx in enumerate(class_groups):
        # -------- class-specific y-range (shared only between AR/IR in this row) --------
        row_vals = []

        # AR side
        ar_tx = list(set(ARdutlist).intersection(set(class_tx)))
        ar_paired = compute_paired_sample_meanTU(
            filtered_trans=filtered_trans,
            transcript_list=ar_tx,
            sample_list=ar_samples,
            fillna=fillna,
            pre_label=pre_label,
            post_label=post_label
        )
        if not ar_paired.empty:
            row_vals.extend(ar_paired["Pre"].tolist())
            row_vals.extend(ar_paired["Post"].tolist())

        # IR side
        ir_tx = list(set(IRdutlist).intersection(set(class_tx)))
        ir_paired = compute_paired_sample_meanTU(
            filtered_trans=filtered_trans,
            transcript_list=ir_tx,
            sample_list=ir_samples,
            fillna=fillna,
            pre_label=pre_label,
            post_label=post_label
        )
        if not ir_paired.empty:
            row_vals.extend(ir_paired["Pre"].tolist())
            row_vals.extend(ir_paired["Post"].tolist())

        if len(row_vals) == 0:
            y_min, y_max = 0, 1
        else:
            y_min = np.nanmin(row_vals)
            y_max = np.nanmax(row_vals)
            if np.isclose(y_min, y_max):
                margin = 0.05 if y_min == 0 else abs(y_min) * 0.05
            else:
                margin = (y_max - y_min) * 0.08
            y_min -= margin
            y_max += margin

        # -------- plot AR / IR for this class --------
        for col_idx, (label, dutlist, sample_list) in enumerate([
            ("AR", ARdutlist, ar_samples),
            ("IR", IRdutlist, ir_samples)
        ]):
            ax = axes[row_idx, col_idx]

            tx_list = list(set(dutlist).intersection(set(class_tx)))

            paired_df = compute_paired_sample_meanTU(
                filtered_trans=filtered_trans,
                transcript_list=tx_list,
                sample_list=sample_list,
                fillna=fillna,
                pre_label=pre_label,
                post_label=post_label
            )

            if paired_df.empty:
                ax.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=12)
                ax.set_title(f"{label}: (Class {row_idx+1})", fontsize=14, fontweight="bold")
                ax.set_ylim(y_min, y_max)
                ax.set_xlabel("Time", fontsize=13)
                ax.set_ylabel("Mean TU", fontsize=13)
                sns.despine(ax=ax)
                continue

            long_df = paired_df.melt(
                id_vars="PairID",
                value_vars=["Pre", "Post"],
                var_name="Time",
                value_name="MeanTU"
            )

            # paired lines first
            for _, r in paired_df.iterrows():
                ax.plot(
                    [0, 1],
                    [r["Pre"], r["Post"]],
                    color="gray",
                    alpha=0.3,
                    linewidth=1,
                    zorder=1
                )

            sns.boxplot(
                data=long_df,
                x="Time",
                y="MeanTU",
                order=["Pre", "Post"],
                palette=[palette[f"{label}_Pre"], palette[f"{label}_Post"]],
                width=0.8,
                fliersize=0,
                linewidth=1.4,
                ax=ax
            )

            sns.stripplot(
                data=long_df,
                x="Time",
                y="MeanTU",
                order=["Pre", "Post"],
                color="gray",
                alpha=0.75,
                size=4,
                jitter=0.08,
                ax=ax,
                zorder=2
            )

            # stat annotation
            try:
                stat, pval = wilcoxon(paired_df["Pre"], paired_df["Post"])
                star = p_to_star(pval)
            except ValueError:
                pval = np.nan
                star = "ns"

            yr = y_max - y_min
            ymax_panel = np.nanmax(paired_df[["Pre", "Post"]].to_numpy())
            line_y = ymax_panel + yr * 0.04
            h = yr * 0.015
            add_stat_annotation(ax, 0, 1, line_y, h, star, fontsize=11)

            ax.set_title(f"{label} DUT (Class {row_idx+1})", fontsize=1, fontweight="bold")
            ax.set_xlabel("Time", fontsize=13)
            ax.set_ylabel("Mean TU", fontsize=13)
            ax.set_ylim(y_min, y_max+0.02)

            sns.despine(ax=ax)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()
    
plot_meanTU_box_grid_by_class(
    filtered_trans=filtered_trans,
    ARdutlist=ARdutlist,
    IRdutlist=IRdutlist,
    class1=class1,
    class2=class2,
    class3=class3,
    ar_samples=ar_samples,
    ir_samples=ir_samples,
    fillna=False,
    pre_label="bfD",
    post_label="atD",
    save_path="/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/AR_IR_Class_boxplot.pdf"
)
#%%
#####^^ delta TU로 rank sum -> GO check ######################

import pandas as pd
import numpy as np
from scipy.stats import ranksums
import gseapy as gp

def run_sample_aware_ranksum(delta_df, ids, samples, dut, ascending=False):
    """
    delta_df: Transcript x Sample 형태의 delta_TU 데이터프레임
    ids: 분석 대상 Transcript ID 리스트 (Class 1 또는 Class 3)
    samples: 분석 대상 샘플 리스트 (AR 또는 IR)
    ascending: False(내림차순, 증가량 큰 순), True(오름차순, 감소량 큰 순)
    """
    # 1. 해당 클래스와 샘플에 해당하는 데이터 추출
    subset = delta_df.loc[delta_df.index.isin(ids), samples]
    subset = subset.loc[subset.index.isin(dut), :]
    
    # 2. 샘플별 랭킹 매기기 (각 컬럼 내에서 순위 산출)
    # 증가량이 큰 것이 1등이 되게 하려면 ascending=False
    sample_ranks = subset.rank(ascending=ascending, method='min', axis=0)
    
    # 3. Transcript별로 모든 샘플의 랭킹을 합산 (Rank Sum)
    # 값이 작을수록 모든 샘플에서 공통적으로 상위권에 있었다는 뜻
    final_rank_sum = sample_ranks.sum(axis=1).sort_values()
    
    # 4. Transcript ID를 Gene Symbol로 변환 (Pathway 분석용)
    rank_df = pd.DataFrame(final_rank_sum, columns=['RankSum'])
    rank_df['gene_symbol'] = rank_df.index.str.split('-').str[-1]
    
    # 동일 유전자 내 가장 순위가 높은(값이 작은) 것을 대표로 선택
    gene_rank_sum = rank_df.groupby('gene_symbol')['RankSum'].mean().sort_values() #^min vs. mean
    
    return gene_rank_sum

delta_TU_gene = filtered_trans.copy()
delta_TU_gene = delta_TU_gene[delta_TU_gene.index.str.split("-", n=1).str[-1].isin(proteincodinglist)]

pre_TU_gene = delta_TU_gene.iloc[:, 1::2]
post_TU_gene = delta_TU_gene.iloc[:, 0::2]

pre_TU_gene.columns = pre_TU_gene.columns.str[:-4] 
post_TU_gene.columns = post_TU_gene.columns.str[:-4] 

delta_TU_gene = post_TU_gene.copy()
delta_TU_gene.index = pre_TU_gene.index 
delta_TU_gene = post_TU_gene.values - pre_TU_gene.values
delta_TU_gene = pd.DataFrame(delta_TU_gene, index=post_TU_gene.index, columns=pre_TU_gene.columns)
delta_TU_gene = delta_TU_gene.dropna(0) #dropna(), fillna(0)


import gseapy as gp
import matplotlib.pyplot as plt

def run_go_enrichment_for_top_n(gene_rank_sum, top_n, db=['GO_Biological_Process_2021','Reactome_2022'], title_suffix=""):
    """
    RankSum 상위 n개 유전자를 추출하여 GO Enrichment 분석 수행
    """
    # 1. 상위 n개 유전자 추출 (RankSum이 작을수록 상위권)
    top_genes = gene_rank_sum.head(top_n).index.tolist()
    
    print(f"--- Running GO Analysis for Top {len(top_genes)} Genes ({title_suffix}) ---")
    
    # 2. Enrichr 실행
    # cutoff 0.05를 기준으로 유의미한 Pathway를 찾습니다.
    enr = gp.enrichr(gene_list=top_genes,
                     gene_sets=db,
                     organism='Human',
                     outdir=None)

    # 3. 결과 정리
    res = enr.results
    sig_res = res[res['Adjusted P-value'] < 0.01].sort_values(by='Combined Score', ascending=False)
    
    if sig_res.empty:
        print("⚠️ 유의미한 GO term이 발견되지 않았습니다. top_n을 늘리거나 DB를 확인하세요.")
        return sig_res
    # 3. -log10(FDR) 계산
    # Adjusted P-value가 0인 경우를 대비해 아주 작은 값을 더해줍니다.
    sig_res['nlog10_FDR'] = -np.log10(sig_res['Adjusted P-value'] + 1e-10)
    sig_res = sig_res.sort_values(by='nlog10_FDR', ascending=False)

    # 4. 시각화 (상위 10개)
    plt.figure(figsize=(6, 8))
    top_10 = sig_res.head(10)
    
    # 색상을 한 가지로 통일 (예: 'steelblue' 또는 'salmon')
    main_color = '#458B73' # 깔끔한 블루 톤
    top_10['Term'] = top_10['Term'].str.rsplit(" ", n=1).str[0]  # 뒤쪽 ID 제거
    
    sns.barplot(data=top_10, x='nlog10_FDR', y='Term', color=main_color)
    
    # 축 및 제목 설정
    plt.title(f"{title_suffix}\n(top {top_n} genes by ranksum)", fontsize=14, fontweight='bold')
    plt.xlabel('-log10(FDR)', fontsize=13) # 요청하신 xlabel 적용
    plt.ylabel('')
    
    # 가독성을 위한 그리드 추가 (선택 사항)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    sns.despine()
    plt.tight_layout()
    #plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/40_ARclass1_ranksum_top1000_GOenrichment.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    return sig_res

# --- 실행 예시 ---'

# Class 1: 증가량이 큰 순서 (ascending=False)
ranks_c1 = run_sample_aware_ranksum(delta_TU_gene, class1, ar_samples,ARdutlist, ascending=False)
# Class 3: 감소량이 큰 순서 (ascending=True, -0.5가 -0.1보다 높은 순위가 됨)
ranks_c3 = run_sample_aware_ranksum(delta_TU_gene, class3, ar_samples, ARdutlist, ascending=True)

# 1. Class 1 (증가량 상위 200개)
res_c1_go = run_go_enrichment_for_top_n(ranks_c1, top_n=1000, title_suffix="AR: Class 1")
# 2. Class 3 (감소량 상위 200개)
res_c3_go = run_go_enrichment_for_top_n(ranks_c3, top_n=1000, title_suffix="AR: Class 3")


# # Class 1: 증가량이 큰 순서 (ascending=False)
ranks_c1 = run_sample_aware_ranksum(delta_TU_gene, class1, ir_samples, IRdutlist, ascending=False)
# Class 3: 감소량이 큰 순서 (ascending=True, -0.5가 -0.1보다 높은 순위가 됨)
ranks_c3 = run_sample_aware_ranksum(delta_TU_gene, class3, ir_samples, IRdutlist,ascending=True)

# 1. Class 1 (증가량 상위 200개)
res_c1_go = run_go_enrichment_for_top_n(ranks_c1, top_n=1000, title_suffix="IR Class 1 (Increased)")
# 2. Class 3 (감소량 상위 200개)
res_c3_go = run_go_enrichment_for_top_n(ranks_c3, top_n=1000, title_suffix="IR Class 3 (Decreased)")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_ranksum_distribution_fixed(gene_rank_sum, top_n=1000, title_suffix=""):
    """
    StopIteration 에러를 방지하기 위해 Overlay 방식으로 랭킹 분포를 시각화
    """
    # 1. 데이터 준비
    df_plot = pd.DataFrame(gene_rank_sum).reset_index()
    df_plot.columns = ['Gene', 'RankSum']
    df_plot['x'] = 0  # stripplot의 x축 고정 위치
    
    # 2. 데이터 분리
    top_df = df_plot.iloc[:top_n].copy()
    others_df = df_plot.iloc[top_n:].copy()
    
    # 3. 시각화 시작
    plt.figure(figsize=(4, 8))
    sns.set_style("ticks")
    
    # [Step 1] 전체(Others) 분포를 연한 회색으로 먼저 그림
    sns.stripplot(data=others_df, y='RankSum', x='x', color='#D3D3D3', 
                  jitter=0.3, size=2, alpha=0.3, zorder=1)
    
    # [Step 2] 상위(Top n) 분포를 주황색으로 그 위에 덧그림
    sns.stripplot(data=top_df, y='RankSum', x='x', color='#458B73', 
                  jitter=0.3, size=4, alpha=0.7, zorder=2)
    
    # 4. 가이드라인 및 Cutoff 표시
    cutoff_value = gene_rank_sum.iloc[top_n-1]
    plt.axhline(cutoff_value, color='#D25353', linestyle='--', linewidth=2, alpha=0.6)
    plt.text(0.5, cutoff_value, f' top {top_n} cutoff', 
             color='#D25353', va='bottom', fontsize=10, fontweight='bold')

    # 5. 축 및 라벨 설정
    #plt.title(f"Gene Ranking Distribution\n({title_suffix})", fontsize=14, fontweight='bold')
    plt.ylabel("rank sum")
    plt.xticks([]) # x축 눈금 제거
    #plt.xlabel(f"Total Genes: {len(df_plot)}", fontsize=11)
    plt.xlabel("")
    plt.ylim(df_plot['RankSum'].min() - 10000, 370000)
    
    # 수동 범례 추가 (Optional)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f'Top {top_n}',
               markerfacecolor='#458B73', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Others',
               markerfacecolor='#D3D3D3', markersize=8)
    ]
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    
    sns.despine(bottom=True)
    plt.tight_layout()
    #plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/40_ranksum_scatter.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# --- 실행 ---
plot_ranksum_distribution_fixed(ranks_c1, top_n=1000, title_suffix="AR Class 1")



#%%
###^^^^ 유전자 내부에서 transcript rank !!!!######

###^^ major ######
ARpre = pre_TU_gene.loc[:,ar_samples]
IRpre = pre_TU_gene.loc[:,ir_samples]
ARpost = post_TU_gene.loc[:,ar_samples]
IRpost = post_TU_gene.loc[:,ir_samples]


def compute_rank(df):
    """
    Compute transcript rank within each gene for each sample.
    - df: DataFrame with transcripts as index and samples as columns
    - Assumes transcript names are in format 'ENSTxxxx-Gene'
    """
    df = df.copy()
    df['Gene'] = df.index.str.split('-', n=1).str[1]  # Extract gene name
    ranked_df = df.groupby("Gene").rank(method="average", ascending=False)  # Rank within gene
    return ranked_df  # Drop gene column to keep only ranks

# Compute ranks for each dataset
ARpre_rank = compute_rank(ARpre)
IRpre_rank = compute_rank(IRpre)
ARpost_rank = compute_rank(ARpost)
IRpost_rank = compute_rank(IRpost)

import gseapy as gp
from statannotations.Annotator import Annotator

AR_COLOR = "#FEA435"  # orange
IR_COLOR = "#168A29"  # green

def dut_in_class(dutlist, class_list):
    return list(set(dutlist) & set(class_list))

def delta_rank_mean(pre_rank, post_rank, tx_list):
    # transcript x sample -> transcript mean(pre-post)
    pre = pre_rank.loc[tx_list, :]
    post = post_rank.loc[tx_list, :]
    return (pre - post).mean(axis=1, skipna=True)

def make_melted_df(delta_ar_mean, delta_ir_mean):
    df_ar = pd.DataFrame({"Group": "AR", "deltaRank": delta_ar_mean}).reset_index()
    df_ir = pd.DataFrame({"Group": "IR", "deltaRank": delta_ir_mean}).reset_index()
    df = pd.concat([df_ar, df_ir], ignore_index=True)
    return df

def plot_ar_vs_ir_boxplot(melted_df, ylabel, title=None, add_stats=True):
    plt.figure(figsize=(4, 5))
    plt.axhline(0, linestyle="--", color="grey", alpha=0.6)

    ax = sns.boxplot(
        x="Group", y="deltaRank", data=melted_df,
        palette={"AR": AR_COLOR, "IR": IR_COLOR},
        showfliers=False
    )

    # (optional) add strip for density
    # sns.stripplot(
    #     x="Group", y="deltaRank", data=melted_df,
    #     color="#6e6e6e", jitter=0.25, size=2.2, alpha=0.25, ax=ax
    # )

    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    if add_stats:
        pairs = [("AR", "IR")]
        annot = Annotator(ax, pairs, data=melted_df, x="Group", y="deltaRank")
        annot.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=0)
        annot.apply_and_annotate()

    sns.despine()
    plt.tight_layout()
    plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/'+title+'TUrankshift_boxplot.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def plot_top_transcript_cutoff(
    delta_series,
    top_n=300,
    ascending=False,
    top_color="#5C8D76",
    other_color="#9C9C9C",
    ylabel="rank shift",
    title=None,
    save_path=None
):
    s = delta_series.dropna().sort_values(ascending=ascending).copy()

    if s.empty:
        print("No valid values.")
        return None

    plot_df = pd.DataFrame({
        "deltaRank": s.values
    })

    plot_df["TopGroup"] = "Others"
    plot_df.iloc[:top_n, plot_df.columns.get_loc("TopGroup")] = f"Top {top_n}"

    cutoff_y = plot_df.iloc[top_n - 1]["deltaRank"]

    plt.figure(figsize=(3.2, 5))
    ax = plt.gca()

    # ⭐ 핵심: 왼쪽으로 shift
    jitter_strength = 0.03
    x_center = -0.12

    other_df = plot_df[plot_df["TopGroup"] == "Others"]
    top_df = plot_df[plot_df["TopGroup"] == f"Top {top_n}"]

    x_other = np.random.normal(x_center, jitter_strength, size=len(other_df))
    x_top = np.random.normal(x_center, jitter_strength, size=len(top_df))

    # Others
    ax.scatter(
        x_other,
        other_df["deltaRank"],
        s=11, alpha=0.6,
        color=other_color,
        edgecolor="none",
        label="Others"
    )

    # Top
    ax.scatter(
        x_top,
        top_df["deltaRank"],
        s=11, alpha=0.9,
        color=top_color,
        edgecolor="none",
        label=f"Top {top_n}"
    )

    # cutoff
    ax.hlines(
    y=cutoff_y,
    xmin=-0.25,
    xmax=0.05,   # 여기까지 선 그려짐
    linestyle="--",
    linewidth=1,
    color="gray",
    alpha=0.4
)

    # ax.text(
    #     -0.02, cutoff_y,
    #     f"top {top_n} cutoff",
    #     color="#D65F4A",
    #     fontsize=10,
    #     va="bottom"
    # )

    # ⭐ 오른쪽 공간 확보
    ax.set_xlim(-0.25, 0.2)
    ax.set_xticks([])
    ax.set_ylabel(ylabel)

    ax.legend(frameon=False, loc="lower right")

    sns.despine(bottom=True)
    plt.tight_layout()

    save_path='/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/'+title+'rank_scatter.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return plot_df

def genes_from_top_transcripts(delta_ar_mean, top_n, ascending):
    """
    delta_ar_mean: Series indexed by transcript ids
    ascending=False -> class1 (largest deltaRank)
    ascending=True  -> class3 (smallest deltaRank)
    """
    top_tx = delta_ar_mean.sort_values(ascending=ascending).head(top_n)

    genes = (
        top_tx.index.astype(str)
        .to_series()
        .str.split("-", n=1).str[1]   # transcriptid-genename
        .dropna()
        .unique()
        .tolist()
    )
    return genes, top_tx

# def enrichr_and_plot_bar(
#     genes,
#     title,
#     color,
#     gene_sets=('GO_Biological_Process_2021', 'Reactome_2022'),
#     fdr_cutoff=0.1,
#     top_terms=10
# ):
#     enr = gp.enrichr(
#         gene_list=genes,
#         gene_sets=list(gene_sets),
#         organism="Human",
#         outdir=None
#     )
#     res = enr.results.copy()
#     sig = res[res["Adjusted P-value"] < fdr_cutoff].copy()
#     if sig.empty:
#         print(f"[{title}] No significant terms at FDR<{fdr_cutoff}.")
#         return sig

#     sig["nlog10_FDR"] = -np.log10(sig["Adjusted P-value"] + 1e-300)
#     sig = sig.sort_values("nlog10_FDR", ascending=False)

#     top = sig.head(top_terms).copy()
#     top["Term"] = top["Term"].astype(str).str.rsplit(" ", n=1).str[0]

#     plt.figure(figsize=(8, 4))
#     sns.set_style("ticks")
#     ax = sns.barplot(data=top, x="nlog10_FDR", y="Term", color=color)
#     ax.set_title("")
#     ax.set_xlabel("-log10(FDR)")
#     ax.set_ylabel("")
#     ax.grid(axis="x", linestyle="--", alpha=0.3)
#     ax.set_xlim(0, float(top["nlog10_FDR"].max()) * 1.1)  # 너가 원한 xlim
#     plt.subplots_adjust(left=0.45)   # ← 고정값
#     ax.set_position([0.45, 0.1, 0.5, 0.8])  # 완전 고정 레이아웃 (더 강력)
#     sns.despine()
#     plt.tight_layout()
#     plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/'+title+'GObarplot.pdf', dpi=300, bbox_inches='tight')
    
#     plt.show()

#     return sig

def enrichr_and_plot_bar(
    genes,
    title,
    color,
    gene_sets=('GO_Biological_Process_2021', 'Reactome_2022'),
    fdr_cutoff=0.1,
    top_terms=10,
    xlim_max=None   # 여러 plot 공통 x축 최대값
):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import gseapy as gp

    enr = gp.enrichr(
        gene_list=genes,
        gene_sets=list(gene_sets),
        organism="Human",
        outdir=None
    )
    res = enr.results.copy()
    sig = res[res["Adjusted P-value"] < fdr_cutoff].copy()
    if sig.empty:
        print(f"[{title}] No significant terms at FDR<{fdr_cutoff}.")
        return sig

    sig["nlog10_FDR"] = -np.log10(sig["Adjusted P-value"] + 1e-300)
    sig = sig.sort_values("nlog10_FDR", ascending=False)

    top = sig.head(top_terms).copy()
    top["Term"] = top["Term"].astype(str).str.rsplit(" ", n=1).str[0]

    sns.set_style("whitegrid")

    fig = plt.figure(figsize=(5, 3.4))
    ax = fig.add_axes([0.10, 0.18, 0.2, 0.72])  
    # [left, bottom, width, height]
    # 이 width를 고정해서 bar 영역 길이를 plot마다 같게 만듦

    ax = sns.barplot(
        data=top,
        x="nlog10_FDR",
        y="Term",
        color=color,
        ax=ax
    )

    # bar thickness 줄이기
    for patch in ax.patches:
        old_h = patch.get_height()
        new_h = 0.7
        patch.set_height(new_h)
        patch.set_y(patch.get_y() + (old_h - new_h) / 2)

    ax.set_title("")
    ax.set_xlabel("-log10(FDR)")
    ax.set_ylabel("")

    # 오른쪽에 term label
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    for label in ax.get_yticklabels():
        label.set_horizontalalignment("left")

    # x축 범위 고정
    if xlim_max is None:
        xlim_max = float(top["nlog10_FDR"].max()) * 1.1
    ax.set_xlim(0, xlim_max)

    # 왼쪽 세로축 선 다시 보이게
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["left"].set_color("0.4")

    # 오른쪽 spine은 숨기고 싶으면
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # x축 grid만 약하게
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.grid(axis="y", visible=False)

    # x=0 기준선 강조하고 싶으면 이것도 가능
    # ax.axvline(0, color="0.4", linewidth=0.8)

    savepath = (
        '/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/'
        'merged_cov5_analysis/0210figures/' + title + 'GObarplot.pdf'
    )

    # tight_layout, bbox_inches='tight'는 쓰지 않음
    plt.savefig(savepath, dpi=300)
    plt.show()

    return sig

def run_ar_class_figure(
    class_name,
    class_list,
    ARdutlist, IRdutlist,
    ARpre_rank, ARpost_rank, IRpre_rank, IRpost_rank,
    top_n_tx_for_go=100,
    ascending_for_top=False,   # class1=False, class3=True
    fdr_cutoff=0.1
):
    # --- DUT ∩ class (AR/IR 각각 자기 리스트 유지)
    AR_tx = dut_in_class(ARdutlist, class_list)
    IR_tx = dut_in_class(IRdutlist, class_list)

    print(f"[{class_name}] AR tx={len(AR_tx)} | IR tx={len(IR_tx)}")

    # --- deltaRank mean
    delta_ar = delta_rank_mean(ARpre_rank, ARpost_rank, AR_tx)
    delta_ir = delta_rank_mean(IRpre_rank, IRpost_rank, IR_tx)

    # --- Boxplot + stripplot
    melted = make_melted_df(delta_ar, delta_ir)
    plot_ar_vs_ir_boxplot(
        melted_df=melted,
        ylabel="TU rank shift",
        title=f"{class_name}",
        add_stats=True
    )

    # --- AR 전체 rank shift에서 top cutoff 시각화
    cutoff_df = plot_top_transcript_cutoff(
        delta_series=delta_ar,
        top_n=top_n_tx_for_go,
        ascending=ascending_for_top,
        top_color="#118797",
        other_color="#AAAAAA",
        ylabel="rank shift",
        title = f"{class_name} AR",
    )

    # --- AR-only top transcripts -> genes -> enrichment
    genes_ar, top_tx_ar = genes_from_top_transcripts(
        delta_ar, top_n_tx_for_go, ascending=ascending_for_top
    )

    print(f"[{class_name}] AR top {top_n_tx_for_go} tx -> {len(genes_ar)} unique genes")

    sig = enrichr_and_plot_bar(
        genes_ar,
        title=f"AR_{class_name}",
        color="#118797",
        fdr_cutoff=fdr_cutoff,
        top_terms=5
    )

    return {
        "delta_ar": delta_ar,
        "delta_ir": delta_ir,
        "top_tx_ar": top_tx_ar,
        "genes_ar": genes_ar,
        "enrich_sig": sig,
        "cutoff_df": cutoff_df
    }

# Class 1: 올라가는 게 좋음 -> 큰 deltaRank가 top -> ascending=False
out_ar_c1 = run_ar_class_figure(
    class_name="Class 1",
    class_list=class1,
    ARdutlist=ARdutlist,
    IRdutlist=IRdutlist,
    ARpre_rank=ARpre_rank, ARpost_rank=ARpost_rank,
    IRpre_rank=IRpre_rank, IRpost_rank=IRpost_rank,
    top_n_tx_for_go=300,
    ascending_for_top=False,   # IMPORTANT
    fdr_cutoff=0.1
)

# Class 3: 내려가는 게 좋음 -> 작은 deltaRank가 top -> ascending=True
out_ar_c3 = run_ar_class_figure(
    class_name="Class 3",
    class_list=class3,
    ARdutlist=ARdutlist,
    IRdutlist=IRdutlist,
    ARpre_rank=ARpre_rank, ARpost_rank=ARpost_rank,
    IRpre_rank=IRpre_rank, IRpost_rank=IRpost_rank,
    top_n_tx_for_go=300,
    ascending_for_top=True,    # IMPORTANT
    fdr_cutoff=0.1
)



#%%


#%%
####^^ (2) splice factor gene list : https://doi.org/10.1016/j.celrep.2018.01.088 서플 파일 사용 ######

SF_genes = [
    'ACIN1', 'AGGF1', 'ALYREF', 'AQR', 'ARGLU1', 'BAG2', 'BCAS1', 'BCAS2', 'BUB3', 'BUD13',
    'BUD31', 'C17orf85', 'C19orf43', 'C1orf55', 'C1QBP', 'C9orf78', 'CACTIN', 'CCAR1', 'CCDC12', 'CCDC130',
    'CCDC75', 'CCDC94', 'CD2BP2', 'CDC40', 'CDC5L', 'CDK10', 'CDK11A', 'CDK12', 'CELF1', 'CELF2',
    'CELF3', 'CELF4', 'CELF5', 'CELF6', 'CFAP20', 'CHERP', 'CIRBP', 'CLASRP', 'CLK1', 'CLK2',
    'CLK3', 'CLK4', 'CLNS1A', 'CPSF6', 'CRNKL1', 'CSN3', 'CTNNBL1', 'CWC15', 'CWC22', 'CWC25',
    'CWC27', 'CXorf56', 'DDX1', 'DDX17', 'DDX18', 'DDX19A', 'DDX19B', 'DDX20', 'DDX21', 'DDX23',
    'DDX26B', 'DDX27', 'DDX39A', 'DDX39B', 'DDX3X', 'DDX3Y', 'DDX41', 'DDX42', 'DDX46', 'DDX5',
    'DDX50', 'DDX6', 'DGCR14', 'DHX15', 'DHX16', 'DHX30', 'DHX34', 'DHX35', 'DHX36', 'DHX38',
    'DHX40', 'DHX57', 'DHX8', 'DHX9', 'DNAJC6', 'DNAJC8', 'EEF1A1', 'EFTUD2', 'EIF2S2', 'EIF3A',
    'EIF4A3', 'ELAVL1', 'ELAVL2', 'ELAVL3', 'ELAVL4', 'FAM32A', 'FAM50A', 'FAM50B', 'FAM58A', 'FMR1',
    'FRA10AC1', 'FRG1', 'FUBP1', 'FUBP3', 'FUS', 'GEMIN2', 'GEMIN5', 'GNB2L1', 'GPATCH1', 'GPATCH3',
    'GPATCH8', 'GPKOW', 'GRSF1', 'HNRNPA0', 'HNRNPA1', 'HNRNPA2B1', 'HNRNPA3', 'HNRNPAB', 'HNRNPC', 'HNRNPCL1',
    'HNRNPD', 'HNRNPDL', 'HNRNPF', 'HNRNPH1', 'HNRNPH2', 'HNRNPH3', 'HNRNPK', 'HNRNPL', 'HNRNPLL', 'HNRNPM',
    'HNRNPR', 'HNRNPU', 'HNRNPUL1', 'HNRNPUL2', 'HSPA1A', 'HSPA1B', 'HSPA5', 'HSPA8', 'HSPB1', 'HTATSF1',
    'IGF2BP3', 'IK', 'ILF2', 'ILF3', 'INTS1', 'INTS3', 'INTS4', 'INTS5', 'INTS6', 'INTS7',
    'ISY1', 'JUP', 'KHDRBS1', 'KHDRBS3', 'KHSRP', 'KIAA1429', 'KIAA1967', 'KIN', 'LENG1', 'LOC649330',
    'LSM1', 'LSM10', 'LSM2', 'LSM3', 'LSM4', 'LSM5', 'LSM6', 'LSM7', 'NAA38', 'LSMD1',
    'LUC7L', 'LUC7L2', 'LUC7L3', 'MAGOH', 'MATR3', 'MBNL1', 'MBNL2', 'MBNL3', 'MFAP1', 'MFSD11',
    'MOV10', 'MSI1', 'MSI2', 'MYEF2', 'NCBP1', 'NCBP2', 'NELFE', 'NKAP', 'NONO', 'NOSIP',
    'NOVA1', 'NOVA2', 'NRIP2', 'NSRP1', 'NUDT21', 'NUMA1', 'PABPC1', 'PAXBP1', 'PCBP1', 'PCBP2',
    'PCBP3', 'PCBP4', 'PDCD7', 'PHF5A', 'PLRG1', 'PNN', 'PPIE', 'PPIG', 'PPIH', 'PPIL1',
    'PPIL2', 'PPIL3', 'PPIL4', 'PPM1G', 'PPP1CA', 'PPP1R8', 'PPWD1', 'PQBP1', 'PRCC', 'PRMT5',
    'PRPF18', 'PRPF19', 'PRPF3', 'PRPF31', 'PRPF38A', 'PRPF38B', 'PRPF39', 'PRPF4', 'PRPF40A', 'PRPF40B',
    'PRPF4B', 'PRPF6', 'PRPF8', 'PSEN1', 'PSIP1', 'PTBP1', 'PTBP2', 'PTBP3', 'PUF60', 'QKI',
    'RALY', 'RALYL', 'RAVER1', 'RAVER2', 'RBBP6', 'RBFOX2', 'RBM10', 'RBM14', 'RBM15', 'RBM15B',
    'RBM17', 'RBM22', 'RBM23', 'RBM25', 'RBM26', 'RBM27', 'RBM3', 'RBM39', 'RBM4', 'RBM42',
    'RBM45', 'RBM47', 'RBM4B', 'RBM5', 'RBM7', 'RBM8A', 'RBMS1', 'RBMX', 'RBMX2', 'RBMXL1',
    'RBMXL2', 'RNF113A', 'RNF20', 'RNF213', 'RNF34', 'RNF40', 'RNPC3', 'RNPS1', 'RNU1-1', 'RNU2-1',
    'RNU4-1', 'RNU5A-1', 'RNU6-1', 'SAP18', 'SAP30BP', 'SART1', 'SEC31B', 'SF1', 'SF3A1', 'SF3A2',
    'SF3A3', 'SF3B1', 'SF3B2', 'SF3B3', 'SF3B4', 'SF3B5', 'SF3B6', 'SFPQ', 'SKIV2L2', 'SLU7',
    'SMN1', 'SMNDC1', 'SMU1', 'SNIP1', 'SNRNP200', 'SNRNP25', 'SNRNP27', 'SNRNP35', 'SNRNP40', 'SNRNP48',
    'SNRNP70', 'SNRPA', 'SNRPA1', 'SNRPB', 'SNRPB2', 'SNRPC', 'SNRPD1', 'SNRPD2', 'SNRPD3', 'SNRPE',
    'SNRPF', 'SNRPG', 'SNRPN', 'NHP2L1', 'SNURF', 'SNW1', 'SPEN', 'SREK1', 'SRPK1', 'SRPK2',
    'SRPK3', 'SRRM1', 'SRRM2', 'SRRT', 'SRSF1', 'SRSF10', 'SRSF11', 'SRSF12', 'SRSF2', 'SRSF3',
    'SRSF4', 'SRSF5', 'SRSF6', 'SRSF7', 'SRSF8', 'SRSF9', 'SSB', 'SUGP1', 'SYF2', 'SYNCRIP',
    'TAF15', 'TCERG1', 'TFIP11', 'THOC1', 'THOC2', 'THOC3', 'THOC5', 'THOC6', 'THOC7', 'THRAP3',
    'TIA1', 'TIAL1', 'TNPO1', 'TOE1', 'TOP1MT', 'TOPORS', 'TRA2A', 'TRA2B', 'TRIM24', 'TTC14',
    'TXNL4A', 'U2AF1', 'U2AF1L4', 'U2AF2', 'U2SURP', 'UBL5', 'USP39', 'WBP11', 'WBP4', 'WDR77',
    'WDR83', 'WTAP', 'XAB2', 'YBX1', 'YBX3', 'ZC3H11A', 'ZC3H13', 'ZC3H18', 'ZC3H4', 'ZC3HAV1',
    'ZCCHC10', 'ZCCHC8', 'ZCRB1', 'ZFR', 'ZMAT2', 'ZMAT5', 'ZMYM3', 'ZNF131', 'ZNF207', 'ZNF326',
    'ZNF346', 'ZNF830', 'ZRSR1', 'ZRSR2'
]

NMD_genes = ['UPF1', 'UPF2', 'UPF3A','UPF3B','SMG1','SMG5','SMG6','SMG7']

geneexp = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_80_gene_TPM.txt', sep='\t', index_col=0)
#geneexp = geneexp.loc[geneexp.index.isin(SF_genes),geneexp.columns.isin(mainlist)]
geneexp = geneexp.loc[geneexp.index.isin(SF_genes),:]
geneexp = geneexp.loc[:,geneexp.columns.isin(ARlist)]
pre_gene = geneexp.iloc[:, 1::2]
post_gene = geneexp.iloc[:, 0::2]
pre_gene.columns = pre_gene.columns.str[:-4]
post_gene.columns = post_gene.columns.str[:-4]


# ---------------------------------------------------------
# 1. 통계 분석 및 16개 유전자 선별
# ---------------------------------------------------------

stats_results = []
common_genes = pre_gene.index.intersection(post_gene.index)

print("통계 분석 진행 중...")

for gene in common_genes:
    pre_vals = pre_gene.loc[gene].values
    post_vals = post_gene.loc[gene].values
    
    # Wilcoxon Test
    try:
        stat, pval = stats.wilcoxon(pre_vals, post_vals)
    except ValueError:
        pval = 1.0 
        
    # Log2 Fold Change
    mean_pre = np.mean(pre_vals)
    mean_post = np.mean(post_vals)
    # 0으로 나누기 방지 (+1 pseudo-count)
    log2fc = np.log2(mean_post + 1) - np.log2(mean_pre + 1)
    
    stats_results.append({
        'Gene': gene,
        'p_val': pval,
        'log2FC': log2fc
    })

res_df = pd.DataFrame(stats_results).set_index('Gene')

# 필터링: p < 0.05 AND |log2FC| > 1
# (증가/감소 모두 포함하려면 abs(), 증가만 보려면 abs() 제거)
target_genes = res_df[
    (res_df['p_val'] < 0.05) & 
    (res_df['log2FC'].abs() > 1)
].sort_values('p_val').index.tolist()

num_genes = len(target_genes)
print(f"조건을 만족하는 유전자 수: {num_genes}개")

# ---------------------------------------------------------
# 2. 8x2 Grid Plot 그리기
# ---------------------------------------------------------

if num_genes > 0:
    # 8행 2열의 서브플롯 생성 (높이를 충분히 줌)
    rows = 2
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8)) 
    axes_flat = axes.flatten() # 2차원 배열을 1차원으로 펴서 인덱싱 편하게 함

    for i, ax in enumerate(axes_flat):
        if i < num_genes:
            gene = target_genes[i]
            
            # 데이터 준비 (Long format)
            sample_ids = pre_gene.columns.tolist()
            plot_df = pd.DataFrame({
                'Sample_ID': sample_ids * 2,
                'Time': ['Pre'] * len(sample_ids) + ['Post'] * len(sample_ids),
                'Expression': np.concatenate([pre_gene.loc[gene].values, post_gene.loc[gene].values])
            })
            
            # A. Boxplot
            sns.boxplot(x='Time', y='Expression', data=plot_df, 
                        order=['Pre', 'Post'], palette=['#E6B0AA', '#D35400'], 
                        width=0.5, showfliers=False, ax=ax)
            
            # B. Stripplot
            sns.stripplot(x='Time', y='Expression', data=plot_df, 
                          order=['Pre', 'Post'], color='black', alpha=0.6, size=4, ax=ax)
            
            # C. Connecting Lines (핵심)
            for sample in sample_ids:
                sample_data = plot_df[plot_df['Sample_ID'] == sample]
                pre_val = sample_data[sample_data['Time'] == 'Pre']['Expression'].values[0]
                post_val = sample_data[sample_data['Time'] == 'Post']['Expression'].values[0]
                ax.plot([0, 1], [pre_val, post_val], color='gray', linewidth=0.8, alpha=0.4)
            
            # D. StatAnnotation
            from statannotations.Annotator import Annotator

            pairs = [("Pre", "Post")]

            annot = Annotator(
                ax, pairs, data=plot_df, x='Time', y='Expression',
                order=['Pre', 'Post']
            )
            annot.configure(test='Wilcoxon', text_format='star', loc='inside', verbose=0)
            annot.apply_and_annotate()
            
            # E. Title & Style
            pval_txt = res_df.loc[gene, 'p_val']
            fc_txt = res_df.loc[gene, 'log2FC']
            # 제목에 Gene 이름, p-value, FC 표시
            ax.set_title(f"{gene}", fontsize=11, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('') # 공간 절약을 위해 y축 라벨 생략 (필요시 추가)
            sns.despine(ax=ax)
            
        else:
            # 유전자가 16개보다 적을 경우 남는 칸은 숨김 처리
            ax.axis('off')

    # 전체 레이아웃 조정
    plt.tight_layout()
    plt.subplots_adjust(top=0.96) # 전체 제목 공간 확보
    #fig.suptitle('Splicing Factor Changes in Acquired Resistance (AR)', fontsize=16, fontweight='bold')
    
    # 저장 및 출력

#plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/40_top5_GObarplot.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%
####^^^ fig3 ################
sns.set_style("ticks")
def save_figure_a(AR_dut, class1, class3, save_path=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 10))
    sns.set_style("ticks")
    # Class 1: Up 강조
    data1 = AR_dut[AR_dut.index.isin(class1)].copy()
    ax1.scatter(data1['delta_TU'], -np.log10(data1['p_value']), c='lightgrey', s=15, alpha=0.3)
    up = data1[(data1['delta_TU'] > 0.05) & (data1['p_value'] < 0.05)]
    ax1.scatter(up['delta_TU'], -np.log10(up['p_value']), c='#FEA435', s=20, label='Up in Class 1')
    ax1.set_title("AR: Class 1", fontsize=13, fontweight='bold')

    # Class 3: Down 강조
    data3 = AR_dut[AR_dut.index.isin(class3)].copy()
    ax2.scatter(data3['delta_TU'], -np.log10(data3['p_value']), c='lightgrey', s=15, alpha=0.3)
    down = data3[(data3['delta_TU'] < -0.05) & (data3['p_value'] < 0.05)]
    ax2.scatter(down['delta_TU'], -np.log10(down['p_value']), c='#168A48', s=20, label='Down in Class 3')
    ax2.set_title("AR: Class 3", fontsize=13, fontweight='bold')

    for ax in [ax1, ax2]:
        ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.2, 4.5)
        ax.axhline(-np.log10(0.05), color='black', ls='--', lw=0.8, alpha=0.5)
        ax.set_xlabel('$\Delta$ TU'); ax.set_ylabel('$-log_{10}(P)$')
        sns.despine(ax=ax)
    
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_figure_c(delta_TU, class1_ids, class3_ids, ar_samples, ir_samples, save_path=None):
    sorted_samples = list(ar_samples) + list(ir_samples)
    
    # 40개 샘플 x 2개 Row (Mean Delta TU)
    hm_c1 = delta_TU.loc[delta_TU.index.isin(class1_ids), sorted_samples].mean()
    hm_c3 = delta_TU.loc[delta_TU.index.isin(class3_ids), sorted_samples].mean()
    hm_df = pd.DataFrame([hm_c1, hm_c3], index=['Class 1 (Functional ↑)', 'Class 3 (Non-coding ↓)'])

    plt.figure(figsize=(12, 2))
    ax = sns.heatmap(hm_df, cmap='RdBu_r', center=0, vmin=-0.2, vmax=0.2, xticklabels=False,
                     cbar_kws={"label": "mean $\Delta$TU", "orientation": "horizontal", "pad": 0.1})
    
    # AR/IR 구분선 (40명 중 AR 샘플 수 지점)
    ax.axvline(len(ar_samples), color='grey', lw=0.5)
    
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

from statannotations.Annotator import Annotator

def save_figure_d(isoform_id, preTU, postTU, ar_samples, ir_samples, gene_name="", save_path=None):
    ar_list = list(ar_samples)
    ir_list = list(ir_samples)
    
    # 1. 데이터 추출 및 결측치 확인
    ar_pre = preTU.loc[isoform_id, ar_list].values
    ar_post = postTU.loc[isoform_id, ar_list].values
    ir_pre = preTU.loc[isoform_id, ir_list].values
    ir_post = postTU.loc[isoform_id, ir_list].values

    # [DEBUG] NaN 존재 여부 확인
    print(f"--- Check for NaNs in {gene_name} ---")
    print(f"AR Pre NaN: {np.isnan(ar_pre).sum()}, Post NaN: {np.isnan(ar_post).sum()}")
    print(f"IR Pre NaN: {np.isnan(ir_pre).sum()}, Post NaN: {np.isnan(ir_post).sum()}")

    # 2. 데이터프레임 구축 (NaN이 포함된 행은 나중에 Annotator가 에러를 낼 수 있음)
    df = pd.DataFrame({
        'Usage': np.concatenate([ar_pre, ar_post, ir_pre, ir_post]),
        'Time': (['Pre'] * len(ar_list) + ['Post'] * len(ar_list) + 
                 ['Pre'] * len(ir_list) + ['Post'] * len(ir_list)),
        'Group': (['AR'] * (len(ar_list) * 2) + ['IR'] * (len(ir_list) * 2)),
        'Subject': ar_list + ar_list + ir_list + ir_list 
    })

    # 결측치(NaN)가 있는 샘플 쌍은 통계 계산에서 제외하는 것이 안전함
    df = df.dropna(subset=['Usage'])

    plt.figure(figsize=(6, 6))
    my_palette = {'Pre': '#FFD350', 'Post': '#68ACDD'}
    my_palette2 = {'Pre': '#FF9501', 'Post': '#218FDE'}
    ax = sns.boxplot(data=df, x='Group', y='Usage', hue='Time', 
                     palette=my_palette, showfliers=False, order=['AR', 'IR'], hue_order=['Pre', 'Post'])
    sns.stripplot(data=df, x='Group', y='Usage', hue='Time', 
                  palette=my_palette2, dodge=True, alpha=0.5, ax=ax, 
                  order=['AR', 'IR'], hue_order=['Pre', 'Post'])

    # 3. Matched line plot (Subject 기준으로 짝이 맞는 경우만 그림)
    for group_idx, group in enumerate(['AR', 'IR']):
        sub = df[df['Group'] == group]
        x_pre = group_idx - 0.2
        x_post = group_idx + 0.2
        for s in sub['Subject'].unique():
            s_data = sub[sub['Subject'] == s]
            if len(s_data) == 2: # Pre, Post 둘 다 존재할 때만 선을 그림
                ax.plot([x_pre, x_post], 
                        [s_data[s_data['Time']=='Pre']['Usage'].values[0], 
                         s_data[s_data['Time']=='Post']['Usage'].values[0]],
                        color='grey', linestyle='-', alpha=0.5, lw=1)

    # 4. 통계 주석 (에러 핸들링 강화)
    pairs = [(("AR", "Pre"), ("AR", "Post")), (("IR", "Pre"), ("IR", "Post"))]
    
    # Wilcoxon이 실패할 경우 t-test_paired로 자동 전환하거나 skip
    try:
        annot = Annotator(ax, pairs, data=df, x='Group', y='Usage', hue='Time', 
                          order=['AR', 'IR'], hue_order=['Pre', 'Post'])
        # zero_method='pratt' 등을 설정할 수 있으나 기본적으로 데이터 정제가 우선
        annot.configure(test='Wilcoxon', text_format='star', loc='inside')
        annot.apply_and_annotate()
    except Exception as e:
        print(f"⚠️ Wilcoxon failed. Trying Paired T-test... Error: {e}")
        try:
            annot.configure(test='t-test_paired', text_format='star', loc='inside')
            annot.apply_and_annotate()
        except Exception as e2:
            print(f"❌ All statistical tests failed: {e2}")

    plt.title(f"{gene_name} ({isoform_id})", fontsize=13)
    plt.xlabel('')
    plt.ylabel('Transcript Usage')
    if ax.get_legend():
        ax.get_legend().remove()
    sns.despine()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# ---------------------------------------------------------
# 1. delta_TU 계산 (Post - Pre)
# ---------------------------------------------------------
# TU columns가 [Post1, Pre1, Post2, Pre2, ...] 순서라면:
# postTU = TU.iloc[:, 0::2], preTU = TU.iloc[:, 1::2] (기존 코드 참고)

# 샘플 이름을 맞춰서 차이를 계산 (Transcriptid x Sample 형태)
delta_TU = postTU.copy()
delta_TU.columns = preTU.columns # 이름을 맞춰서 뺄셈 가능하게 함
delta_TU = postTU.values - preTU.values 
delta_TU = pd.DataFrame(delta_TU, index=postTU.index, columns=preTU.columns)
delta_TU = delta_TU.dropna(0) #fillna(0) 

# ---------------------------------------------------------
# 2. Figure A 실행: Volcano Plots (개별 저장)
# ---------------------------------------------------------

save_figure_a(AR_dut, class1, class3, 
              save_path='/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/fig3A_volcano.png')

# ---------------------------------------------------------
# 3. Figure C 실행: Sample-wise Heatmap (개별 저장)
# ---------------------------------------------------------
# ar_samples, ir_samples는 sampleinfo에서 추출한 인덱스 리스트
# Class 1 (Up) 관련 GO 유전자들
up_go_genes = set()
for g_str in df_plot[df_plot['Group'] == 'AR Upregulated (Class 1)']['Genes']:
    up_go_genes.update(g_str.split(';'))

# Class 3 (Down) 관련 GO 유전자들
down_go_genes = set()
for g_str in df_plot[df_plot['Group'] == 'AR Downregulated (Class 3)']['Genes']:
    down_go_genes.update(g_str.split(';'))

# 여기서는 Transcript-Gene 형태일 때 유전자를 뽑아내는 방식입니다.
target_c1_ids = [x for x in class1 if x.split('-')[-1] in up_go_genes]
target_c3_ids = [x for x in class3 if x.split('-')[-1] in down_go_genes]

# Heatmap용 ID (Transcript ID만)
hm_c1_ids = [x.split('-')[0] for x in target_c1_ids]
hm_c3_ids = [x.split('-')[0] for x in target_c3_ids]


save_figure_c(delta_TU, hm_c1_ids, hm_c3_ids, ar_samples, ir_samples, 
              save_path='/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/fig3C_heatmap.pdf')


# ---------------------------------------------------------
# 4. Figure D 실행: Representative Examples (개별 저장)
# ---------------------------------------------------------
# (1) Class 1 (Functional) 예시: AR_dut 기준 delta_TU가 가장 큰 것
up_go_genes = set(";".join(df_plot[df_plot['Group'].str.contains('Up')]['Genes']).split(";"))
down_go_genes = set(";".join(df_plot[df_plot['Group'].str.contains('Down')]['Genes']).split(";"))


# ---------------------------------------------------------
# 2. Top GO 유전자 중 대표 예시 선정 (Figure D용)
# ---------------------------------------------------------
# (1) Class 1 (Functional Up): Up GO 패스웨이에 속한 유전자 중 delta_TU 최대값
target_class1 = AR_dut[AR_dut.index.isin(class1)]
target_class1 = target_class1[(target_class1['p_value']<0.05)&(target_class1['delta_TU']>0.05)]

# 인덱스의 Gene symbol이 up_go_genes에 포함되는 것만 필터링
target_class1 = target_class1[target_class1.index.map(lambda x: x.split('-')[-1] in up_go_genes)]

if not target_class1.empty:
    example_up = target_class1.sort_values('delta_TU', ascending=False).index[0]
    example_up = 'ENST00000270861.5-PLK4' #ENST00000310955.6-CDC20	#ENST00000263201.1-CDC45 #ENST00000450114.2-WEE1 ENST00000372991.4-CCND3	ENST00000270861.5-PLK4	
    iso_up = example_up.split('-')[0] 
    gene_up = example_up.split('-')[-1]
    
    print(f"Selected Class 1 Example (from GO): {gene_up} ({iso_up})")
    save_figure_d(iso_up, preTU, postTU, ar_samples, ir_samples, gene_name=gene_up,
                save_path='/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/fig3D_class1_PLK4.pdf')

# (2) Class 3 (Non-coding Down): Down GO 패스웨이에 속한 유전자 중 delta_TU 최소값
target_class3 = AR_dut[AR_dut.index.isin(class3)]
target_class3 = target_class3[(target_class3['p_value']<0.05)&(target_class3['delta_TU']<-0.05)]
# 인덱스의 Gene symbol이 down_go_genes에 포함되는 것만 필터링
target_class3 = target_class3[target_class3.index.map(lambda x: x.split('-')[-1] in down_go_genes)]

if not target_class3.empty:
    example_down = target_class3.sort_values('delta_TU', ascending=True).index[0]
    example_down = 'MSTRG.460760.3-RNF8' #MSTRG.101999.43-CHEK1 #MSTRG.443838.1-RAD50 #MSTRG.98060.1-ATM #ENST00000478276.1-STAT3	MSTRG.460760.3-RNF8	
    iso_down = example_down.split('-')[0]
    gene_down = example_down.split('-')[-1]
    
    print(f"Selected Class 3 Example (from GO): {gene_down} ({iso_down})")
    save_figure_d(iso_down, preTU, postTU, ar_samples, ir_samples, gene_name=gene_down,
                save_path='/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/fig3D_class3_RNF8.pdf')


#%%
###^^^^^^^^^^ fig 2######################################

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from statannotations.Annotator import Annotator

def generate_functional_plots(TU, sampleinfo, class_lists, save_path=None):
    """
    TU: 전체 Transcript Usage 데이터프레임
    sampleinfo: 샘플 정보 (response: 1=AR, 0=IR)
    class_lists: [class1_ids, class2_ids, class3_ids] 리스트
    """
    
    # 1. 샘플 및 데이터 전처리
    ar_samples = sampleinfo[sampleinfo['response'] == 1].index.intersection(TU.columns[1::2]) # Pre 기준 추출
    ir_samples = sampleinfo[sampleinfo['response'] == 0].index.intersection(TU.columns[1::2])
    
    preTU = TU.iloc[:, 1::2]
    postTU = TU.iloc[:, 0::2]
    
    # 색상 팔레트 설정 (이미지 테마 반영)
    pal_ar = {'Pre': '#FFEDA0', 'Post': '#FEB24C'} # 노란색 계열
    pal_ir = {'Pre': '#D9F0D3', 'Post': '#5AAE61'} # 초록색 계열

    # Figure 설정 (3행 4열: B(Density) 2열 + C(Boxplot) 2열)
    fig, axes = plt.subplots(3, 4, figsize=(18, 14))
    plt.rcParams["font.family"] = "Arial"

    for row, ids in enumerate(class_lists):
        class_label = f"Class {row + 1}"
        
        # --- [Fig B: Density Plots] ---
        for col, (group_name, samples, palette) in enumerate([('AR', ar_samples, pal_ar), ('IR', ir_samples, pal_ir)]):
            ax = axes[row, col]
            
            # 해당 클래스 데이터 추출
            data_pre = preTU.loc[preTU.index.isin(ids), samples].values.flatten()
            data_post = postTU.loc[postTU.index.isin(ids), samples].values.flatten()
            
            # 결측치 제거
            data_pre = data_pre[~np.isnan(data_pre)]
            data_post = data_post[~np.isnan(data_post)]

            # KDE Plot (Density)
            sns.kdeplot(data_pre, ax=ax, fill=True, color=palette['Pre'], label='Pre', alpha=0.5)
            sns.kdeplot(data_post, ax=ax, fill=True, color=palette['Post'], label='Post', alpha=0.5)
            
            ax.set_title(f"{group_name} ({class_label})", fontsize=14, fontweight='bold')
            ax.set_xlabel("TU" if row == 2 else "")
            ax.set_ylabel("Density" if col == 0 else "")
            ax.set_xlim(-0.1, 1.1)
            if row == 0 and col == 0: ax.legend(frameon=False)

        # --- [Fig C: Boxplots (Sample-wise Mean)] ---
        for col_idx, (group_name, samples, palette) in enumerate([('AR', ar_samples, pal_ar), ('IR', ir_samples, pal_ir)]):
            ax = axes[row, col_idx + 2] # 3, 4번째 열 사용
            
            # 샘플별 평균 계산
            mean_pre = preTU.loc[preTU.index.isin(ids), samples].mean()
            mean_post = postTU.loc[postTU.index.isin(ids), samples].mean()
            
            # 데이터프레임 구축
            df_box = pd.DataFrame({
                'Mean_TU': pd.concat([mean_pre, mean_post]),
                'Time': ['Pre']*len(samples) + ['Post']*len(samples),
                'Subject': list(samples) * 2
            })

            # Boxplot + Stripplot (가로로 넓적한 비율)
            sns.boxplot(data=df_box, x='Time', y='Mean_TU', ax=ax, palette=palette, showfliers=False, width=0.5)
            sns.stripplot(data=df_box, x='Time', y='Mean_TU', ax=ax, color='.3', alpha=0.4, dodge=True)
            
            # Matched Line 추가
            for s in samples:
                vals = df_box[df_box['Subject'] == s]['Mean_TU'].values
                ax.plot([0, 1], vals, color='gray', linestyle='-', alpha=0.15, lw=1)

            # 통계 테스트
            annot = Annotator(ax, [("Pre", "Post")], data=df_box, x='Time', y='Mean_TU')
            annot.configure(test='Wilcoxon', text_format='star', loc='inside')
            annot.apply_and_annotate()

            ax.set_title(f"{group_name} {class_label} (Mean)", fontsize=14)
            ax.set_xlabel("")
            ax.set_ylabel("Mean TU" if col_idx == 0 else "")
            sns.despine(ax=ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# --- 실행 코드 ---

ar_dut_filtered = AR_dut.loc[
    (AR_dut['p_value'] < 0.05) & (AR_dut['delta_TU'].abs() > 0.05)
].index.to_list()

c1_dut_ids = [x.split('-')[0] for x in class1 if x in ar_dut_filtered]
c2_dut_ids = [x.split('-')[0] for x in class2 if x in ar_dut_filtered]
c3_dut_ids = [x.split('-')[0] for x in class3 if x in ar_dut_filtered]

# 디버깅: 필터링 후 남은 개수 확인
print(f"DUT Filtered - Class 1: {len(c1_dut_ids)}, Class 2: {len(c2_dut_ids)}, Class 3: {len(c3_dut_ids)}")

# 3. 통합 시각화 실행 (B + C)
class_lists_to_plot = [c1_dut_ids, c2_dut_ids, c3_dut_ids]

generate_functional_plots(
    TU=TU, 
    sampleinfo=sampleinfo, 
    class_lists=class_lists_to_plot, 
    #save_path='Figure_BC_Integrated_A4.pdf'
)
#%%
####^^ (2-1) Splice Factor NMF clustering? ##############

from sklearn.decomposition import NMF
from scipy.stats import wilcoxon

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from scipy.stats import wilcoxon, mannwhitneyu

# ---- load & subset once
geneexp_all = pd.read_csv(
    '/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_80_gene_TPM.txt',
    sep='\t', index_col=0
)

genes = [g for g in SF_genes if g in geneexp_all.index]

def make_X(sample_list):
    cols = [c for c in sample_list if c in geneexp_all.columns]
    X = geneexp_all.loc[genes, cols].copy()          # genes × samples
    X = np.log1p(X)                                  # log1p(TPM)
    X = X.T                                          # samples × genes
    X.index.name = "Sample"
    return X

X_AR = make_X(ARlist)  # (40 × 397)
X_IR = make_X(IRlist)  # (2*nIR × 397)

gene_mean = X_AR.mean(axis=0) + 1e-8
X_AR = X_AR / gene_mean
X_IR = X_IR / gene_mean

def parse_sample(s):
    # expects: SV-OV-P035-atD / SV-OV-P035-bfD
    m = re.match(r"(.+)-(atD|bfD)$", s)
    if m is None:
        raise ValueError(f"Unexpected sample name: {s}")
    patient = m.group(1)
    time = m.group(2)  # atD=post, bfD=pre
    return patient, time

def make_meta(X):
    patients, times = zip(*[parse_sample(s) for s in X.index])
    meta = pd.DataFrame({"patient": patients, "time": times}, index=X.index)
    return meta

meta_AR = make_meta(X_AR)
meta_IR = make_meta(X_IR)

# sanity: each patient should have bfD & atD once
assert (meta_AR.groupby("patient")["time"].nunique() == 2).all()
assert (meta_IR.groupby("patient")["time"].nunique() == 2).all()

k = 4
nmf = NMF(
    n_components=k,
    init="nndsvda",
    max_iter=3000,
    random_state=0
)

W_AR = nmf.fit_transform(X_AR.values)     # (AR samples × k)
H_genes = nmf.components_                # (k × genes)

W_AR = pd.DataFrame(W_AR, index=X_AR.index, columns=[f"C{i+1}" for i in range(k)])
H_genes = pd.DataFrame(H_genes, index=[f"C{i+1}" for i in range(k)], columns=X_AR.columns)

# project IR samples into the same component space
W_IR = nmf.transform(X_IR.values)         # (IR samples × k)
W_IR = pd.DataFrame(W_IR, index=X_IR.index, columns=W_AR.columns)

def patient_pre_post_delta(W, meta):
    # W: sample × component
    pre = W.loc[meta[meta["time"] == "bfD"].index].copy()
    post = W.loc[meta[meta["time"] == "atD"].index].copy()

    pre["patient"] = meta.loc[pre.index, "patient"].values
    post["patient"] = meta.loc[post.index, "patient"].values

    pre = pre.set_index("patient")
    post = post.set_index("patient")

    pre = pre.sort_index()
    post = post.sort_index()

    delta = post - pre
    return pre, post, delta

AR_pre, AR_post, AR_delta = patient_pre_post_delta(W_AR, meta_AR)
IR_pre, IR_post, IR_delta = patient_pre_post_delta(W_IR, meta_IR)

rows = []
for comp in W_AR.columns:
    d = AR_delta[comp].dropna()
    frac_pos = (d > 0).mean()

    try:
        _, p = wilcoxon(d)
    except ValueError:
        p = 1.0

    rows.append({
        "Component": comp,
        "Mean_Delta": d.mean(),
        "Frac_Positive": frac_pos,
        "P_value": p
    })

comp_stats = pd.DataFrame(rows).set_index("Component")
comp_stats = comp_stats.sort_values(["Frac_Positive", "Mean_Delta"], ascending=False)
comp_stats

target_comp = comp_stats.index[0]
print("Selected component:", target_comp)

def plot_AR_IR_component(comp):
    # scatter baseline vs delta
    plt.figure(figsize=(6,6))
    plt.scatter(AR_pre[comp], AR_delta[comp], label="AR", alpha=0.8)
    plt.scatter(IR_pre[comp], IR_delta[comp], label="IR", alpha=0.8)
    plt.axhline(0, linestyle="--", color="gray", linewidth=1)
    plt.xlabel("Pre (baseline component score)")
    plt.ylabel("Δ (post - pre)")
    plt.title(f"{comp}: baseline vs induction (AR vs IR)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # boxplots: baseline and delta
    data_baseline = [AR_pre[comp].values, IR_pre[comp].values]
    data_delta = [AR_delta[comp].values, IR_delta[comp].values]

    plt.figure(figsize=(6,4))
    plt.boxplot(data_baseline, labels=["AR_pre", "IR_pre"], showfliers=False)
    plt.title(f"{comp}: baseline comparison")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,4))
    plt.boxplot(data_delta, labels=["AR_Δ", "IR_Δ"], showfliers=False)
    plt.axhline(0, linestyle="--", color="gray", linewidth=1)
    plt.title(f"{comp}: induction (Δ) comparison")
    plt.tight_layout()
    plt.show()

plot_AR_IR_component(target_comp)


#%%
####^^ (3) Splice Factor gene x functional TU correlation ######
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------
# 1. 데이터 준비 (Delta 값 계산)
# ---------------------------------------------------------

# 가정: 
# 1. pre_gene, post_gene: (Row: Gene, Col: Sample) - 컬럼 순서 동일해야 함
# 2. pre_major_dut, post_major_dut: (Row: 1575 Transcripts, Col: Sample) - 컬럼 순서 동일해야 함
# 3. target_genes: 앞서 선별한 16개 SF 유전자 리스트

# A. X축 데이터: Delta SF Expression (Post - Pre)
# log2 변환된 값이면 뺄셈이 곧 Fold Change 의미를 가짐 (logA - logB = log(A/B))
target_genes = ["ALYREF","DDX1","CELF5","DHX30","MYEF2","DDX23","NAA38","PPM1G","PPP1CA","RBM47","ZNF207","EIF3A","LSM6","SNRPB","PCBP1","RBM15"]

delta_sf_exp = post_gene.loc[target_genes] - pre_gene.loc[target_genes]

ARdut_up = AR_dut.loc[(AR_dut['p_value']<0.05) & np.abs(AR_dut['delta_TU']<-0.05)].index.to_list()
ARdut_up = [x for x in ARdut_up if x in class3] #class1 vs. class3
# ARdut_up = [
#     s for s in ARdut_up
#     if any(gene in s for gene in cellcycle) #cellcycle vs. dsbrepair
# ]
ARdut_up = set([x.split('-', 1)[0] for x in ARdut_up])
tlist = set(ARdut_up)
post_major_dut = postTU.loc[postTU.index.isin(tlist),postTU.columns.isin(ar_samples)]
pre_major_dut = preTU.loc[preTU.index.isin(tlist),preTU.columns.isin(ar_samples)]

# B. Y축 데이터: Mean Delta TU (Post - Pre)
# 1575개 Major Transcript 각각의 Delta TU를 구한 뒤, 샘플별로 평균을 냄
delta_tu_matrix = post_major_dut - pre_major_dut
mean_delta_tu = delta_tu_matrix.mean(axis=0) # 결과: 샘플 개수만큼의 1D Series

# 데이터프레임으로 합치기 (분석용)
# 컬럼명이 샘플 ID라고 가정
samples = delta_sf_exp.columns
analysis_df = delta_sf_exp.T # Row: Sample, Col: 16 SF Genes
analysis_df['Mean_Delta_TU'] = mean_delta_tu.values

# ---------------------------------------------------------
# 2. 4x4 Grid Correlation Plot 그리기
# ---------------------------------------------------------

fig, axes = plt.subplots(4, 4, figsize=(20, 20)) # 16개니까 4x4
axes_flat = axes.flatten()

print("Delta Correlation 분석 시작...")

for i, gene in enumerate(target_genes):
    ax = axes_flat[i]
    
    # 데이터 추출
    x_vals = analysis_df[gene]
    y_vals = analysis_df['Mean_Delta_TU']
    
    # 상관계수 계산 (Pearson)
    # 샘플 수가 적거나 Outlier가 심하면 spearmanr로 변경 고려
    corr, p_val = pearsonr(x_vals, y_vals)
    
    # 1. Scatter Plot (점 찍기)
    sns.scatterplot(x=x_vals, y=y_vals, color='#D35400', s=100, alpha=0.7, ax=ax)
    
    # 2. Regression Line (회귀선)
    sns.regplot(x=x_vals, y=y_vals, scatter=False, color='gray', 
                line_kws={'linestyle': '--', 'linewidth': 1.5}, ci=None, ax=ax)
    
    # 3. 꾸미기
    # 배경에 상관계수에 따라 색을 입혀줄 수도 있음 (Optional)
    if p_val < 0.05 :
        ax.set_facecolor('#FDF2E9') # 유의한 양의 상관관계면 살짝 붉은 배경
        #title_color = 'red'
        #weight = 'bold'
    else:
        title_color = 'black'
        weight = 'normal'

    ax.set_title(f"{gene}\nR={corr:.2f}, p={p_val:.3f}", 
                 fontsize=14, color=title_color, fontweight=weight)
    
    ax.set_xlabel(f'\u0394 Expression ({gene})', fontsize=10) # Delta 기호
    if i % 4 == 0: # 맨 왼쪽 열에만 Y축 라벨 표시
        ax.set_ylabel('Mean \u0394 TU', fontsize=12)
    else:
        ax.set_ylabel('')
        
    sns.despine(ax=ax)

# 전체 레이아웃
plt.tight_layout()
plt.subplots_adjust(top=0.95)

# 저장
#save_path = "Delta_Correlation_16_Panel.png"
#plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/maintenance/ARdownregulated_SF_corr.pdf', dpi=300, bbox_inches='tight')
plt.show()


#%%
full_geneexp = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_80_gene_TPM.txt', sep='\t', index_col=0)

# 1-2. 타겟 유전자 16개 정의
target_genes = ["ALYREF","DDX1","CELF5","DHX30","MYEF2","DDX23","NAA38","PPM1G",
                "PPP1CA","RBM47","ZNF207","EIF3A","LSM6","SNRPB","PCBP1","RBM15"]

# 1-3. AR Pre 데이터 추출
# ARlist에 있는 컬럼만 가져온 뒤, 홀수번째 컬럼(1::2)이 Pre라고 가정 (기존 코드 로직 따름)
ar_exp = full_geneexp.loc[full_geneexp.index.isin(target_genes), full_geneexp.columns.isin(ARlist)]
ar_pre_gene = ar_exp.iloc[:, 1::2] 
ar_pre_gene.columns = ar_pre_gene.columns.str[:-4] # 컬럼명 정리 (_Pre 등 접미사 제거 가정)

# 1-4. IR Pre 데이터 추출
# ★ 중요: IRlist가 정의되어 있어야 합니다. (IR 그룹 샘플 ID 리스트)
# 만약 IRlist가 아직 없다면, ARlist와 같은 방식으로 미리 정의해주세요.
ir_exp = full_geneexp.loc[full_geneexp.index.isin(target_genes), full_geneexp.columns.isin(IRlist)]
ir_pre_gene = ir_exp.iloc[:, 1::2] # 동일하게 홀수번째 컬럼이 Pre라고 가정
ir_pre_gene.columns = ir_pre_gene.columns.str[:-4]

print(f"Target Genes: {len(target_genes)}")
print(f"AR Pre Samples: {ar_pre_gene.shape[1]}")
print(f"IR Pre Samples: {ir_pre_gene.shape[1]}")

# ---------------------------------------------------------
# 2. 8x2 Grid Plot 그리기 (AR Pre vs IR Pre)
# ---------------------------------------------------------

# 2행 8열 서브플롯 생성
rows = 2
cols = 8
fig, axes = plt.subplots(rows, cols, figsize=(24, 8)) 
axes_flat = axes.flatten()

for i, ax in enumerate(axes_flat):
    if i < len(target_genes):
        gene = target_genes[i]
        
        # 해당 유전자가 데이터에 있는지 확인 (결측 방지)
        if gene in ar_pre_gene.index and gene in ir_pre_gene.index:
            ar_vals = ar_pre_gene.loc[gene].values
            ir_vals = ir_pre_gene.loc[gene].values
            
            # 시각화를 위한 DataFrame 생성 (Long format)
            plot_df = pd.DataFrame({
                'Expression': np.concatenate([ar_vals, ir_vals]),
                'Group': ['AR Pre'] * len(ar_vals) + ['IR Pre'] * len(ir_vals)
            })
            
            # A. Boxplot
            # AR(Pre)는 기존 색상(#E6B0AA), IR(Pre)는 구분을 위해 파란색 계열(#AED6F1) 사용
            sns.boxplot(x='Group', y='Expression', data=plot_df, 
                        order=['AR Pre', 'IR Pre'], 
                        palette={'AR Pre': '#E6B0AA', 'IR Pre': '#AED6F1'}, 
                        width=0.6, showfliers=False, ax=ax)
            
            # B. Stripplot (개별 샘플 점 찍기)
            sns.stripplot(x='Group', y='Expression', data=plot_df, 
                          order=['AR Pre', 'IR Pre'], 
                          color='black', alpha=0.5, size=3, jitter=True, ax=ax)
            
            # C. 통계 검정 (Mann-Whitney U Test)
            # 서로 다른 그룹(Unpaired)이므로 Wilcoxon이 아닌 Mann-Whitney 사용
            add_stat_annotation(ax, data=plot_df, x='Group', y='Expression',
                                box_pairs=[("AR Pre", "IR Pre")],
                                test='Mann-Whitney', text_format='star', 
                                loc='inside', verbose=0)
            
            # D. 평균값 차이(Log2FC) 계산 및 제목 표시
            # 양수면 IR이 더 높음, 음수면 AR이 더 높음
            mean_ar = np.mean(ar_vals)
            mean_ir = np.mean(ir_vals)
            # 0으로 나누기 방지 (+1)
            fc_val = np.log2(mean_ir + 1) - np.log2(mean_ar + 1)
            
            # E. 스타일 꾸미기
            # 제목에 유전자 이름과 Fold Change (IR/AR) 표시
            ax.set_title(f"{gene}\n(FC: {fc_val:.2f})", fontsize=11, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('') 
            sns.despine(ax=ax)
            
        else:
            # 데이터가 없는 경우 빈 칸 처리
            ax.set_title(f"{gene} (No Data)", fontsize=10)
            ax.axis('off')
            
    else:
        # 16개 이후 남는 칸 숨김 (여기서는 딱 16개라 해당 없음)
        ax.axis('off')

# 전체 레이아웃 조정
plt.tight_layout()
plt.subplots_adjust(top=0.90) # 전체 제목 공간 확보
fig.suptitle('Baseline Expression Comparison: AR Pre vs. IR Pre', fontsize=16, fontweight='bold')

# 저장 경로 설정 및 저장
#save_path = '/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/maintenance/AR_vs_IR_Baseline_16_Panel.pdf'
#plt.savefig(save_path, dpi=300, bbox_inches='tight')
#print(f"그림이 저장될 경로: {save_path}")

plt.show()


#%%
#^^^^^^########################## pseudotime PCA ##############################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_pca_on_transcripts(
    pre_TU_gene,
    post_TU_gene,
    ar_samples,
    ir_samples,
    transcript_list,
    title=""
):
    # --- subset transcripts
    tx = list(set(transcript_list) & set(pre_TU_gene.index))
    
    pre = pre_TU_gene.loc[tx,  list(ar_samples) + list(ir_samples)]
    post = post_TU_gene.loc[tx, list(ar_samples) + list(ir_samples)]

    # --- stack pre & post as separate samples
    pre_df = pre.copy()
    post_df = post.copy()

    pre_df.columns = [c + "_pre" for c in pre_df.columns]
    post_df.columns = [c + "_post" for c in post_df.columns]

    combined = pd.concat([pre_df, post_df], axis=1)

    # --- transpose: samples x transcripts
    X = combined.T.fillna(0)

    # --- scale
    X_scaled = StandardScaler().fit_transform(X)

    # --- PCA
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"], index=X.index)

    # --- metadata
    
    all_samples = list(ar_samples) + list(ir_samples)

    # sample 이름만 추출 (_pre/_post 제거)
    base_sample = pca_df.index.str.replace("_pre", "").str.replace("_post", "")

    pca_df["Group"] = np.where(
        base_sample.isin(ar_samples),
        "AR",
        "IR"
    )

    pca_df["Time"] = np.where(
        pca_df.index.str.contains("_post"),
        "Post",
        "Pre"
    )


    # --- plot
    plt.figure(figsize=(6,5))
    sns.scatterplot(
        data=pca_df,
        x="PC1", y="PC2",
        hue="Group",
        style="Time",
        palette={"AR":"#FFCC29", "IR":"#81B214"},
        s=80
    )
    plt.title(title)
    plt.axhline(0, color="grey", alpha=0.3)
    plt.axvline(0, color="grey", alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("Explained variance:", pca.explained_variance_ratio_)
    return pca_df

# transcript groups
tx_class1 = list(set(ARdutlist) & set(class1))
tx_class3 = list(set(ARdutlist) & set(class3))

# Class 1 PCA
pca_c1 = run_pca_on_transcripts(
    pre_TU_gene,
    post_TU_gene,
    ar_samples,
    ir_samples,
    tx_class1,
    title="PCA – ARdut ∩ Class1"
)

# Class 3 PCA
pca_c3 = run_pca_on_transcripts(
    pre_TU_gene,
    post_TU_gene,
    ar_samples,
    ir_samples,
    tx_class3,
    title="PCA – ARdut ∩ Class3"
)

def plot_pc1_distribution(pca_df, title):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(5,5))
    sns.boxplot(
        data=pca_df,
        x="Time",
        y="PC1",
        hue="Group",
        palette={"AR":"#FFCC29", "IR":"#81B214"}
    )
    plt.title(title)
    plt.axhline(0, linestyle="--", color="grey", alpha=0.4)
    plt.tight_layout()
    plt.show()
    
plot_pc1_distribution(pca_c1, "PC1 distribution – Class1")
plot_pc1_distribution(pca_c3, "PC1 distribution – Class3")

#%%
import umap
from sklearn.preprocessing import StandardScaler

def get_top_variable_within_group(
    pre_TU_gene,
    post_TU_gene,
    ar_samples,
    ir_samples,
    transcript_group,   # e.g. ARdut ∩ class1
    top_n=500
):
    all_samples = list(ar_samples) + list(ir_samples)

    tx = list(set(transcript_group) & set(pre_TU_gene.index))

    pre = pre_TU_gene.loc[tx, all_samples]
    post = post_TU_gene.loc[tx, all_samples]

    combined = pd.concat([pre, post], axis=1)

    var = combined.var(axis=1, skipna=True)

    top_tx = var.sort_values(ascending=False).head(top_n).index.tolist()

    print(f"Selected {len(top_tx)} transcripts from group (variance filtered)")
    return top_tx

import umap
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def run_umap_group_usage(
    pre_TU_gene,
    post_TU_gene,
    ar_samples,
    ir_samples,
    tx_list,
    title=""
):
    all_samples = list(ar_samples) + list(ir_samples)

    pre = pre_TU_gene.loc[tx_list, all_samples]
    post = post_TU_gene.loc[tx_list, all_samples]

    pre.columns = [c+"_pre" for c in pre.columns]
    post.columns = [c+"_post" for c in post.columns]

    combined = pd.concat([pre, post], axis=1)

    X = combined.T.fillna(0)

    X_scaled = StandardScaler().fit_transform(X)

    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(X_scaled)

    df = pd.DataFrame(embedding, columns=["UMAP1","UMAP2"], index=X.index)

    base_sample = df.index.str.replace("_pre","").str.replace("_post","")

    df["Group"] = np.where(base_sample.isin(ar_samples),"AR","IR")
    df["Time"] = np.where(df.index.str.contains("_post"),"Post","Pre")

    plt.figure(figsize=(6,5))
    sns.scatterplot(
        data=df,
        x="UMAP1", y="UMAP2",
        hue="Group",
        style="Time",
        palette={"AR":"#FFCC29","IR":"#81B214"},
        s=90
    )

    # trajectory lines
    for sample in all_samples:
        pre_name = sample + "_pre"
        post_name = sample + "_post"
        if pre_name in df.index and post_name in df.index:
            plt.plot(
                [df.loc[pre_name,"UMAP1"], df.loc[post_name,"UMAP1"]],
                [df.loc[pre_name,"UMAP2"], df.loc[post_name,"UMAP2"]],
                color="grey", alpha=0.3
            )

    plt.title(title)
    plt.tight_layout()
    plt.show()

    return df

# group 정의
tx_class1 = list(set(ARdutlist) & set(class1))

# group 내부 variance 필터
top_tx_class1 = get_top_variable_within_group(
    pre_TU_gene,
    post_TU_gene,
    ar_samples,
    ir_samples,
    tx_class1,
    top_n=300
)

tx_class3 = list(set(ARdutlist) & set(class3))

# group 내부 variance 필터
top_tx_class3 = get_top_variable_within_group(
    pre_TU_gene,
    post_TU_gene,
    ar_samples,
    ir_samples,
    tx_class3,
    top_n=500
)

# UMAP
df_class1 = run_umap_group_usage(
    pre_TU_gene,
    post_TU_gene,
    ar_samples,
    ir_samples,
    top_tx_class1,
    title="UMAP – ARdut ∩ Class1 (variance filtered)"
)

df_class3 = run_umap_group_usage(
    pre_TU_gene,
    post_TU_gene,
    ar_samples,
    ir_samples,
    top_tx_class3,
    title="UMAP – ARdut ∩ Class1 (variance filtered)"
)

def plot_umap1_box(df, title=""):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # 4개 그룹 label 만들기
    df = df.copy()
    df["Condition"] = df["Group"] + "_" + df["Time"]
    
    order = ["AR_Pre", "AR_Post", "IR_Pre", "IR_Post"]
    
    plt.figure(figsize=(6,5))
    
    ax = sns.boxplot(
        data=df,
        x="Condition",
        y="UMAP1",
        order=order,
        palette={
            "AR_Pre": "#FFCC29",
            "AR_Post": "#FFCC29",
            "IR_Pre": "#81B214",
            "IR_Post": "#81B214"
        },
        showfliers=False
    )
    
    sns.stripplot(
        data=df,
        x="Condition",
        y="UMAP1",
        order=order,
        color="black",
        alpha=0.4,
        jitter=0.2,
        size=4
    )
    
    plt.title(title)
    plt.ylabel("UMAP1")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()


plot_umap1_box(
    df_class1, 
    title="UMAP1 distribution – ARdut ∩ Class1"
)

plot_umap1_box(
    df_class3,
    title="UMAP1 distribution – ARdut ∩ Class3"
)



# %%
#####^^^^^^^^ (4) Splice psi ######################
ar_psi = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/suppaoutput/AR/MW_psi_7events.txt',sep='\t')
ir_psi = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/suppaoutput/IR/MW_psi_7events.txt',sep='\t')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_event_grid_volcano(ar_df, ir_df, 
                            x_col='d_psi', y_col='pval', event_col='event',
                            save_path="AR_vs_IR_Splicing_Volcano_Grid.png"):
    
    # 1. 공통으로 존재하는 이벤트 타입 추출 및 정렬
    # (데이터가 많은 순서대로 정렬하면 보기 좋습니다)
    ar_counts = ar_df[event_col].value_counts()
    event_types = ar_counts.index.tolist()
    
    # 2. Grid 설정 (행: 이벤트 타입 수, 열: 2개 [AR, IR])
    n_rows = len(event_types)
    n_cols = 2
    
    # 그림 크기: 행이 많을 수 있으므로 세로 길이를 동적으로 조절
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3.5 * n_rows), sharex=True)
    
    # 시각적 스타일 설정
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    
    # 3. 이벤트 별로 반복하며 그리기
    for i, event in enumerate(event_types):
        
        # --- Row 별 데이터 필터링 ---
        # AR 데이터
        sub_ar = ar_df[ar_df[event_col] == event].copy()
        sub_ar['nlog10'] = -np.log10(sub_ar[y_col].replace(0, 1e-300))
        
        # IR 데이터
        sub_ir = ir_df[ir_df[event_col] == event].copy()
        sub_ir['nlog10'] = -np.log10(sub_ir[y_col].replace(0, 1e-300))
        
        # y축 최대값 통일 (같은 행끼리 비교하기 쉽게)
        max_y = max(sub_ar['nlog10'].max(), sub_ir['nlog10'].max()) if not sub_ar.empty and not sub_ir.empty else 10
        
        # --- [왼쪽 열] AR 그리기 ---
        ax_ar = axes[i, 0]
        if not sub_ar.empty:
            # Color logic: dPSI > 0 (Red), dPSI < 0 (Blue)
            colors = np.where(sub_ar[x_col] > 0, '#E64B35', '#3C5488')
            
            ax_ar.scatter(sub_ar[x_col], sub_ar['nlog10'], c=colors, 
                          alpha=0.6, s=15, edgecolors='none')
            
            # 통계 텍스트 (Up/Down 개수)
            n_up = sum(sub_ar[x_col] > 0)
            n_down = sum(sub_ar[x_col] < 0)
            ax_ar.text(0.05, 0.9, f'Inclusion (+): {n_up}\nExclusion (-): {n_down}', 
                       transform=ax_ar.transAxes, fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        ax_ar.axvline(0, color='black', linestyle='--', linewidth=0.8)
        ax_ar.set_title(f"AR : {event}", fontweight='bold')
        ax_ar.set_ylabel(r'$-log_{10}(P)$')
        ax_ar.set_ylim(1, max_y * 1.05)
        ax_ar.set_xlim(-0.5, 0.5) # PSI는 -1 ~ 1 범위

        # --- [오른쪽 열] IR 그리기 ---
        ax_ir = axes[i, 1]
        if not sub_ir.empty:
            colors = np.where(sub_ir[x_col] > 0, '#E64B35', '#3C5488')
            
            ax_ir.scatter(sub_ir[x_col], sub_ir['nlog10'], c=colors, 
                          alpha=0.6, s=15, edgecolors='none')
            
            n_up = sum(sub_ir[x_col] > 0)
            n_down = sum(sub_ir[x_col] < 0)
            ax_ir.text(0.05, 0.9, f'Inclusion (+): {n_up}\nExclusion (-): {n_down}', 
                       transform=ax_ir.transAxes, fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        ax_ir.axvline(0, color='black', linestyle='--', linewidth=0.8)
        ax_ir.set_title(f"IR : {event}", fontweight='bold')
        ax_ir.set_ylim(1, max_y * 1.05)
        ax_ir.set_xlim(-0.5, 0.5) # PSI는 -1 ~ 1 범위
        
        # 맨 마지막 행에만 X축 라벨
        if i == n_rows - 1:
            ax_ar.set_xlabel(r'$\Delta$PSI (Post - Pre)')
            ax_ir.set_xlabel(r'$\Delta$PSI (Post - Pre)')

    # 전체 타이틀
    plt.suptitle('Splicing Alterations by Event Type: AR vs. IR', fontsize=16, y=1.005)
    plt.tight_layout()
    
    #plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# --- 실행 예시 ---
# ar_psi, ir_psi는 선생님이 만든 데이터프레임 변수명이라고 가정합니다.
plot_event_grid_volcano(ar_psi, ir_psi)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation # pip install statannot 필요

# 1. 데이터 준비 (기존과 동일)ㄷ
ar_ri = ar_psi[ar_psi['event'] == 'RI'].copy()
ar_ri['Group'] = 'AR'

ir_ri = ir_psi[ir_psi['event'] == 'RI'].copy()
ir_ri['Group'] = 'IR'

combined_ri = pd.concat([ar_ri, ir_ri])

# 2. 그림 그리기 설정
fig, ax = plt.subplots(figsize=(6, 6))

# --- A. Violin Plot (배경) ---
# inner=None: 바이올린 안에 박스플롯이나 쿼타일 선을 그리지 않음 (점과 겹침 방지)
sns.violinplot(data=combined_ri, x='Group', y='d_psi', order=['AR', 'IR'],
               palette={'AR': '#FF7B00', 'IR': '#6FB80A'}, 
               inner=None, linewidth=1.5, ax=ax)

# ★ Violin 투명도 조절 (핵심) ★
# seaborn violinplot에는 alpha 파라미터가 직접 적용되지 않아서 이렇게 해야 함
for collection in ax.collections:
    collection.set_alpha(0.3) # 0.3 정도로 투명하게 설정

# --- B. Stripplot (점 찍기) ---
# jitter=True: 점들이 겹치지 않게 옆으로 살짝 퍼뜨림
sns.stripplot(data=combined_ri, x='Group', y='d_psi', order=['AR', 'IR'],
              palette={'AR': '#FF7B00', 'IR': '#6FB80A'}, size=3, alpha=0.6, jitter=0.2, ax=ax)

# --- C. Reference Line (0) ---
ax.axhline(0, color='gray', linestyle='--', linewidth=1, zorder=0)

# --- D. StatAnnotation (Mann-Whitney U) ---
add_stat_annotation(ax, data=combined_ri, x='Group', y='d_psi', order=['AR', 'IR'],
                    box_pairs=[("AR", "IR")],
                    test='Mann-Whitney', text_format='star', # 'star' or 'simple' (p=0.001 등)
                    loc='inside', verbose=2)

# 3. 꾸미기
ax.set_title('$\Delta$PSI Distribution: Retained Introns (RI)', fontsize=13)
ax.set_ylabel('$\Delta$PSI (Post - Pre)', fontsize=12)
ax.set_xlabel('')

sns.despine()
plt.tight_layout()
plt.show()

import pandas as pd

def count_inclusion_exclusion(df, event_col='event', dpsi_col='d_psi'):
    """
    Return a dataframe:
    index = event
    columns = ['Inclusion', 'Exclusion']
    """
    counts = (
        df
        .assign(
            Direction=pd.cut(
                df[dpsi_col],
                bins=[-1, 0, 1],
                labels=['Exclusion', 'Inclusion'],
                include_lowest=True
            )
        )
        .groupby([event_col, 'Direction'])
        .size()
        .unstack(fill_value=0)
    )

    # event 순서를 원래 데이터 기준으로 유지
    event_order = df[event_col].value_counts().index
    counts = counts.loc[event_order]

    return counts

import matplotlib.pyplot as plt

def plot_stacked_bar(count_df, title, ax,
                     colors={'Inclusion': '#69B1C2', 'Exclusion': '#BF716F'}):
    bottom = None

    for col in ['Exclusion', 'Inclusion']:
        ax.bar(
            count_df.index,
            count_df[col],
            bottom=bottom,
            label=col,
            color=colors[col],
            edgecolor='black',
            linewidth=0.5
        )
        bottom = count_df[col] if bottom is None else bottom + count_df[col]

    ax.set_title(title, fontweight='bold')
    ax.set_ylabel('Number of Events')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(frameon=False)
# 집계
ar_counts = count_inclusion_exclusion(ar_psi)
ir_counts = count_inclusion_exclusion(ir_psi)

# 플롯
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
sns.set_style("ticks")
plot_stacked_bar(ar_counts, 'AR', axes[0])
plot_stacked_bar(ir_counts, 'IR', axes[1])

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_diverging_bar(count_df, title, ax, 
                         colors={'Inclusion': '#69B1C2', 'Exclusion': '#BF716F'}):
    # Exclusion 개수를 음수로 변환하여 왼쪽으로 뻗게 만듦
    exclusion_values = -count_df['Exclusion']
    inclusion_values = count_df['Inclusion']
    events = count_df.index
    y_pos = np.arange(len(events))

    # Bar 그리기
    ax.barh(y_pos, exclusion_values, color=colors['Exclusion'], edgecolor='black', linewidth=0.5, label='Pre')
    ax.barh(y_pos, inclusion_values, color=colors['Inclusion'], edgecolor='black', linewidth=0.5, label='Post')

    # 중앙선 (0) 및 꾸미기
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(events)
    ax.invert_yaxis()  # 상단에 첫 번째 이벤트가 오도록 설정
    
    # X축 범위를 대칭으로 맞추기 (선택 사항)
    max_val = max(abs(exclusion_values).max(), inclusion_values.max())
    ax.set_xlim(-max_val * 2, max_val * 2)
    
    # 절대값으로 X축 라벨 표시
    ticks = ax.get_xticks()
    ax.set_xticklabels([int(abs(t)) for t in ticks])

    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xlabel('Number of Events')
    ax.legend(loc='lower right', frameon=False)
    sns.despine(ax=ax)
    

# --- 실행 ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

plot_diverging_bar(ar_counts, 'AR', axes[0])
plot_diverging_bar(ir_counts, 'IR', axes[1])

plt.tight_layout()
#plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/40_spliceevent_barplot.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
####^^^^ gDUT list vs. AS list ########

ARdeg = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/whole_AR_Wilcoxon_DEGresult_FC.txt', sep='\t')
ARdeglist = set(ARdeg[(ARdeg['p_value'] < 0.05)]['gene_name']) #&(ARdeg['log2FC']>1)
IRdeg = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/whole_IR_Wilcoxon_DEGresult_FC.txt', sep='\t')
IRdeglist = set(IRdeg[(IRdeg['p_value'] < 0.05)]['gene_name']) #&(ARdeg['log2FC']>1)

transinfo = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/cov5_filtered_transcripts_with_gene_info.tsv', sep='\t')
geneinfo = transinfo[['mstrg_gene_id','gene_name']].drop_duplicates()

# 데이터를 담을 리스트 초기화
categories = ['AR-Pre', 'AR-Post', 'IR-Pre', 'IR-Post']
overlap_counts = []
remaining_counts = []

# 분석 루프 (AR/IR 및 dpsi 기준에 따라 4번 수행)
# 각 조건에 맞게 ar_psi/ir_psi와 ar_isoforms_clean/ir_isoforms_clean을 사용한다고 가정합니다.
conditions = [
    (ar_psi, ar_isoforms_clean, ARdeglist, 'RI', -1), # AR Exclusion
    (ar_psi, ar_isoforms_clean, ARdeglist, 'RI', 1),  # AR Inclusion
    (ir_psi, ir_isoforms_clean, IRdeglist, 'RI', -1), # IR Exclusion
    (ir_psi, ir_isoforms_clean, IRdeglist, 'RI', 1)   # IR Inclusion
]

for psi_df, iso_clean, deglist, event_type, direction in conditions:
    # 1. AS_glist (Splicing Event Genes) 추출
    cond_psi = psi_df[psi_df['event'] == event_type].copy()
    if direction < 0:
        cond_psi = cond_psi[cond_psi['d_psi'] < 0]
    else:
        cond_psi = cond_psi[cond_psi['d_psi'] > 0]
        
    cond_psi = pd.merge(cond_psi, geneinfo, left_on='gene_id', right_on='mstrg_gene_id', how='inner')
    cond_psi = cond_psi.drop_duplicates()
    AS_glist = (set(cond_psi['gene_name']) - {np.nan}) & set(proteincodinglist)
    
    # 2. DUT_glist (Reference Gene Group - AR or IR) 추출
    DUT_glist = set(transinfo[transinfo['transcript_id'].isin(iso_clean)]['gene_name']) - {np.nan}
    
    # 3. 교집합 및 나머지 계산
    intersection = AS_glist & DUT_glist ###^^ DUT_glist vs. deglist
    inter_count = len(intersection)
    remain_count = len(AS_glist) - inter_count
    
    overlap_counts.append(inter_count)
    remaining_counts.append(remain_count)

import matplotlib.pyplot as plt
import pandas as pd

# 1. 시각화용 데이터프레임 구성 (위의 계산 코드 결과 활용)
plot_df = pd.DataFrame({
    'Category': ['AR-Pre', 'AR-Post', 'IR-Pre', 'IR-Post'],
    'Overlap': overlap_counts,  # [AR-Exc-overlap, AR-Inc-overlap, ...]
    'Others': remaining_counts,
    'Type': ['Exclusion', 'Inclusion', 'Exclusion', 'Inclusion']
})

# 2. 색상 설정
colors_map = {'Inclusion': '#69B1C2', 'Exclusion': '#BF716F'}
others_color = '#D3D3D3' # 연한 회색

plt.figure(figsize=(8, 5))

# 3. 막대 그래프 그리기
x = range(len(plot_df['Category']))

for i, row in plot_df.iterrows():
    # 하단: Overlap (Inclusion/Exclusion 색상 적용)
    plt.bar(i, row['Overlap'], color=colors_map[row['Type']], 
            edgecolor='black', linewidth=0.5, label='Overlap with DUT' if i < 2 else "")
    
    # 상단: Others (회색 고정)
    plt.bar(i, row['Others'], bottom=row['Overlap'], color=others_color, 
            edgecolor='black', linewidth=0.5, label='Others' if i == 0 else "")

    # 개수 라벨링 (옵션: 가독성을 위해)
    plt.text(i, row['Overlap']/2, f"{row['Overlap']}", ha='center', va='center', 
             color='white', fontweight='bold',fontsize=11)
    plt.text(i, row['Overlap'] + row['Others']/2, f"{row['Others']}", ha='center', va='center', 
             color='black', fontsize=11)

# 4. 그래프 디테일 설정
plt.xticks(x, plot_df['Category'])
plt.ylabel('Number of Genes')
plt.title('Retained Intron Events')

# 범례 중복 제거 설정
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#BF716F', lw=4, label='DUT gene overlap'),
    Line2D([0], [0], color='#69B1C2', lw=4, label='DUT gene overlap'),
    Line2D([0], [0], color='#D3D3D3', lw=4, label='No overlap')
]
plt.legend(handles=legend_elements, loc='upper right', frameon=False)

# 테두리 정리 (Despine)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
#plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/40_intronretention_gDUT_barplot.pdf', dpi=300, bbox_inches='tight')
plt.show()

# 1. 비율 계산을 위한 데이터 가공
plot_df['Total'] = plot_df['Overlap'] + plot_df['Others']
plot_df['Overlap_Pct'] = (plot_df['Overlap'] / plot_df['Total']) * 100
plot_df['Others_Pct'] = (plot_df['Others'] / plot_df['Total']) * 100

plt.figure(figsize=(8, 6))
x = range(len(plot_df['Category']))

# 2. 색상 설정
colors_map = {'Inclusion': '#69B1C2', 'Exclusion': '#BF716F'}
others_color = '#D3D3D3'

# 3. 막대 그래프 그리기 (비율 기준)
for i, row in plot_df.iterrows():
    # 하단: Overlap 비율
    plt.bar(i, row['Overlap_Pct'], color=colors_map[row['Type']], 
            edgecolor='black', linewidth=0.8)
    
    # 상단: Others 비율 (하단 위에 쌓음)
    plt.bar(i, row['Others_Pct'], bottom=row['Overlap_Pct'], color=others_color, 
            edgecolor='black', linewidth=0.8)

    # 4. 퍼센트 및 실제 개수 라벨링
    # Overlap 부분에 %와 (실제 개수) 표시
    plt.text(i, row['Overlap_Pct']/2, f"{row['Overlap_Pct']:.1f}%\n({row['Overlap']})", 
             ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    
    # 상단 Others 부분에 실제 개수 표시
    plt.text(i, row['Overlap_Pct'] + row['Others_Pct']/2, f"({row['Others']})", 
             ha='center', va='center', color='black', fontsize=10)

# 5. 그래프 디테일 설정
plt.xticks(x, plot_df['Category'], fontsize=12)
plt.yticks(range(0, 110, 20), [f'{i}%' for i in range(0, 110, 20)]) # y축 % 표기
plt.ylabel('Percentage of Genes (%)', fontsize=12)
plt.title('Retained Intron Events (Percentage)', fontsize=14, pad=15)

# 범례 설정
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#BF716F', lw=6, label='DUT gene overlap'),
    Line2D([0], [0], color='#69B1C2', lw=6, label='DUT gene overlap'),
    Line2D([0], [0], color='#D3D3D3', lw=6, label='No overlap')
]
plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), frameon=False)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
# 파일명에 _percentage 추가하여 저장
#plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/40_intronretention_gDUT_percentage_barplot.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%
# 결과를 저장할 딕셔너리
overlap_gene_dict = {}
AS_gene_dict = {}

# 분석 조건 설정 (AR/IR 및 dpsi 기준)
conditions = [
    ('AR-Exclusion', ar_psi, ar_isoforms_clean, 'RI', -1),
    ('AR-Inclusion', ar_psi, ar_isoforms_clean, 'RI', 1),
    ('IR-Exclusion', ir_psi, ir_isoforms_clean, 'RI', -1),
    ('IR-Inclusion', ir_psi, ir_isoforms_clean, 'RI', 1)
]

for label, psi_df, iso_clean, event_type, direction in conditions:
    # 1. AS_glist 추출
    cond_psi = psi_df[psi_df['event'] == event_type].copy()
    if direction < 0:
        cond_psi = cond_psi[cond_psi['d_psi'] < 0]
    else:
        cond_psi = cond_psi[cond_psi['d_psi'] > 0]
        
    cond_psi = pd.merge(cond_psi, geneinfo, left_on='gene_id', right_on='mstrg_gene_id', how='inner')
    cond_psi = cond_psi.drop_duplicates()
    AS_glist = (set(cond_psi['gene_name']) - {np.nan}) & set(proteincodinglist)
    
    # 2. DUT_glist 추출
    DUT_glist = set(transinfo[transinfo['transcript_id'].isin(iso_clean)]['gene_name']) - {np.nan}
    
    # 3. 교집합(겹치는 유전자) 리스트 저장
    intersection_list = sorted(list(AS_glist & DUT_glist))
    overlap_gene_dict[label] = intersection_list
    AS_gene_dict[label] = sorted(list(AS_glist))

overlap_gene_dict['AR-Exclusion']

#%%
##^ check gene list for DNA repair? ##

from gseapy.parser import get_library

# -----------------------------------
# input gene set
# -----------------------------------
my_genes = set(overlap_gene_dict['AR-Exclusion']) ##^ change here

# Enrichr libraries usually use uppercase gene symbols
my_genes_upper = {g.upper() for g in my_genes if pd.notna(g)}

# -----------------------------------
# libraries to search
# -----------------------------------
libraries = ["GO_Biological_Process_2021", "Reactome_2022"]

# DNA repair-related keyword pattern
# 필요하면 여기 키워드 더 추가 가능
pattern = re.compile(
    r"(dna repair|double strand break|single strand break|base excision repair|break repair"
    #r"nucleotide excision repair|mismatch repair|homologous recombination|"
    #r"non homologous end joining|nhej|fanconi|interstrand crosslink|"
    r"damage response|dna damage checkpoint|strand break)",
    #r"(alternative splicing)",
    flags=re.IGNORECASE
)

all_results = []
selected_gene_sets = {}

for lib in libraries:
    # download/load gene set library as dict: {term: [genes...]}
    gs_dict = get_library(name=lib, organism="Human")
    
    # keep only DNA-repair-related terms
    dna_repair_sets = {
        term: genes for term, genes in gs_dict.items()
        if pattern.search(term)
    }
    selected_gene_sets[lib] = dna_repair_sets
    
    for term, genes in dna_repair_sets.items():
        term_genes = {g.upper() for g in genes if pd.notna(g)}
        overlap = my_genes_upper & term_genes
        
        all_results.append({
            "Library": lib,
            "GeneSet": term,
            "TermSize": len(term_genes),
            "MySetSize": len(my_genes_upper),
            "OverlapCount": len(overlap),
            "OverlapGenes": ";".join(sorted(overlap))
        })

overlap_df = pd.DataFrame(all_results)

# overlap 있는 것만 보고 싶으면
overlap_df_nonzero = overlap_df[overlap_df["OverlapCount"] > 0].copy()

# 많이 겹치는 순으로 정렬
overlap_df_nonzero = overlap_df_nonzero.sort_values(
    ["OverlapCount", "Library", "GeneSet"],
    ascending=[False, True, True]
).reset_index(drop=True)

dna_repair_union = set()

for lib, gs_dict in selected_gene_sets.items():
    for term, genes in gs_dict.items():
        dna_repair_union |= {g.upper() for g in genes if pd.notna(g)}

overlap_union = my_genes_upper & dna_repair_union

print(f"Number of AR-Exclusion genes overlapping any DNA-repair term: {len(overlap_union)}")
print(sorted(overlap_union))


##^^ GO enrichment #####
genes_incl = set(overlap_gene_dict['AR-Inclusion'])
genes_excl = set(overlap_gene_dict['AR-Exclusion'])

gene_set = genes_incl | genes_excl

enr = gp.enrichr(
    gene_list=list(gene_set),
    gene_sets=["GO_Biological_Process_2021", "Reactome_2022"],
    organism="Human",
    outdir=None,   # 파일 저장 안함
    cutoff=0.1    # adj p-value cutoff
)

# 결과 dataframe
res = enr.results

#%%
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

ddrgenes = [item for sublist in ddr_genelist.values() for item in sublist]
set(ar_ri['gene_name']).intersection(set(ddrgenes))



# %%
##^^ (5) pre_cohort splice gene check #########
target_genes = ["ALYREF","DDX1","MYEF2","DDX23","PPM1G","RBM47","ZNF207","LSM6","SNRPB","PCBP1"]
target_genes = [
    "PRPF39",
    "HNRNPUL1",
    "NHP2L1",
    "ALYREF",
    "SRSF1",
    "DHX38",
    "PRPF8",
    "SNRNP40",
    "KHDRBS3",
    "INTS3",
    "RBM14",
    "C17orf85",
    "U2AF2",
    "DDX17",
    "RBM4B",
    "CLNS1A",
    "NUDT21",
    "CELF6"
]
newcohort = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_pre/111_pre/forval_111_gene_TPM.txt', sep='\t', index_col=0)
sfgeneexp = newcohort.loc[newcohort.index.isin(target_genes), :]
clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/112_PARPi_clinicalinfo.txt', sep='\t', index_col=0)
clin = clin.loc[clin.index.isin(sfgeneexp.columns),:]
#clin = clin[clin['setting']=='maintenance']
clin['group'] = 'i'
clin.loc[(clin['response']==1)&(clin['recur']==1),'group'] = 'AR'
clin.loc[(clin['response']==0),'group'] = 'IR'
clin.loc[(clin['response']==1)&(clin['recur']==0),'group'] = 'CR'

#clin = clin[clin['group']!='AR']

#clin = clin[clin['response']==1]
# clin = clin[clin['line']!='1L']
sfgeneexp = sfgeneexp.loc[:,sfgeneexp.columns.isin(clin.index)]

# newcohort = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO/83_gene_exp.txt', sep='\t', index_col=0)
# clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/83_POLO_clinicalinfo.txt', sep='\t', index_col=0)
# sfgeneexp = newcohort.loc[newcohort.index.isin(target_genes), :]


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test


# A. 전치 (Transpose): (Gene x Sample) -> (Sample x Gene)
# 분석을 위해 행이 샘플이 되도록 돌립니다.
df_t = sfgeneexp.T 

# B. Z-score Normalization (각 유전자별로 정규화)
# (값 - 평균) / 표준편차
df_z = (df_t - df_t.mean()) / df_t.std()

# C. Score 계산 (샘플별 Z-score의 평균)
# "Splicing Efficiency Score" 생성
splicing_score = df_t.mean(axis=1)
splicing_score.name = 'Splicing_Score'

# ---------------------------------------------------------
# 3. 임상 데이터와 병합 (Merge)
# ---------------------------------------------------------

# clin 데이터에 Score 추가 (인덱스 기준 병합)
# 만약 Sample ID가 컬럼으로 있다면 set_index('SampleID') 후 진행
merged_df = clin[['PFS', 'recur']].merge(splicing_score, left_index=True, right_index=True)

# 결측치 제거 (혹시 모를 오류 방지)
merged_df = merged_df.dropna()

print(f"분석 대상 환자 수: {len(merged_df)}명")

# ---------------------------------------------------------
# 4. Cox Proportional Hazards Model (Continuous Variable)
# ---------------------------------------------------------
# 점수 자체가 1단위 증가할 때 위험도가 얼마나 증가하는지 봅니다.

cph = CoxPHFitter()
cph.fit(merged_df, duration_col='PFS', event_col='recur')

print("\n[Cox Proportional Hazards Model Results]")
cph.print_summary()  # 통계 결과 요약 출력 (p-value, HR 확인)

# ---------------------------------------------------------
# 5. 시각화 1: Cox Forest Plot (Hazard Ratio)
# ---------------------------------------------------------
plt.figure(figsize=(8, 4))
cph.plot()
plt.title("Hazard Ratio of Splicing Score (Continuous)", fontsize=14)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 6. 시각화 2: Kaplan-Meier Curve (High vs. Low Group)
# ---------------------------------------------------------
# 시각화를 위해 점수 기준으로 High/Low 그룹을 나눕니다 (Median Cut)

median_score = merged_df['Splicing_Score'].median()
merged_df['Group'] = np.where(merged_df['Splicing_Score'] >= median_score, 'High Score', 'Low Score')

kmf = KaplanMeierFitter()
plt.figure(figsize=(5, 6))

# High Group (Score 높음 -> 내성 예상 -> 예후 나쁨 예상)
mask_high = merged_df['Group'] == 'High Score'
kmf.fit(merged_df[mask_high]['PFS'], event_observed=merged_df[mask_high]['recur'], label='High Splicing Score')
kmf.plot_survival_function(color='#E64B35', ci_show=False) # 빨강

# Low Group (Score 낮음 -> 예후 좋음 예상)
mask_low = merged_df['Group'] == 'Low Score'
kmf.fit(merged_df[mask_low]['PFS'], event_observed=merged_df[mask_low]['recur'], label='Low Splicing Score')
kmf.plot_survival_function(color='#3C5488', ci_show=False) # 파랑

# Log-rank Test (두 그룹 간 차이 검정)
results = logrank_test(merged_df[mask_high]['PFS'], merged_df[mask_low]['PFS'], 
                       event_observed_A=merged_df[mask_high]['recur'], 
                       event_observed_B=merged_df[mask_low]['recur'])

# 꾸미기
plt.title(f"CR+IR (p = {results.p_value:.4f})", fontsize=15, fontweight='bold')
plt.xlabel("Progression-Free Survival (Months)", fontsize=12)
plt.ylabel("Survival Probability", fontsize=12)
plt.grid(True, alpha=0.3)
sns.despine()
plt.tight_layout() 
plt.show()

prolif_genes = ['MKI67', 'PCNA', 'TOP2A', 'CCNB1']
results = {}

for gene in prolif_genes:
    if gene in newcohort.index:
        corr = splicing_score.corr(newcohort.loc[gene])
        results[gene] = corr

print("Correlation with Proliferation Markers:")
print(results)

merged_forfig = clin[['PFS', 'recur', 'group']].merge(splicing_score, left_index=True, right_index=True)

plt.figure(figsize=(6,6))
sns.set(style='ticks')
# 1. Boxplot을 그릴 때 ax 객체를 저장합니다.
ax = sns.boxplot(x='group', y='Splicing_Score', data=merged_forfig, 
                 order=['CR', 'IR','AR'],palette={"AR": "#FEB24C", "IR": "#5AAE61", "CR":"#58C1EE"}, showfliers=False)
sns.stripplot(x='group', y='Splicing_Score', data=merged_forfig,
              order=['CR', 'IR','AR'], 
              color='#545454',       # 점 색상
              alpha=0.3,           # 투명도 (0~1)
              jitter=0.1,          # 점들이 겹치지 않게 좌우로 흩뿌림
              size=5,              # 점 크기
              ax=ax)               # 같은 축(ax) 사용
# 2. 비교할 그룹의 쌍(pair)을 정의합니다.
pairs = [("CR", "AR"),("CR","IR"),("AR","IR")]

# 3. Annotator 설정 및 적용
annotator = Annotator(ax, pairs, data=merged_forfig, x='group', y='Splicing_Score', order=['CR','IR','AR'])

# test='t-test_ind' (t-검정) 또는 'Mann-Whitney' (비모수 검정) 중 선택
annotator.configure(test='Mann-Whitney', text_format='full', loc='inside', verbose=2, pvalue_format_string='{:.2f}')
annotator.apply_and_annotate()

sns.despine()
plt.ylabel('Splicing Score')
plt.title('Baseline Splicing Score') # 제목 추가 추천
#plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_pre/111_pre/figures/group1_splicingscore_CRvsAR.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
####^^^ 앞에서 사용한 group 1 AR DUT를 validation cohort에서 확인하기 #################



val_tpm = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_pre/111_pre/forval_111_transcript_TPM.txt', sep='\t', index_col=0)
val_tpm = val_tpm.loc[:, val_tpm.columns.isin(clin.index)]

val_tpm = val_tpm.loc[(val_tpm > 0).sum(axis=1) >= 15] #20% 이상에서는 나오긴 해야됨 ...
val_tpm["gene"] = val_tpm.index.str.split("-", n=1).str[-1]
gene_sum = val_tpm.groupby("gene").transform("sum")
val_tu = val_tpm.iloc[:, :-1].div(gene_sum)

# #%%
# plt.figure(figsize=(6,6))
# sns.set(style='ticks')
# # 1. Boxplot을 그릴 때 ax 객체를 저장합니다.
# ax = sns.boxplot(x='group', y='mean_group1_DUT_exp', data=merged_forfig, 
#                  order=['CR', 'AR' ,'IR'], palette=['#1681CD', '#E56007'], showfliers=False)
# sns.stripplot(x='group', y='mean_group1_DUT_exp', data=merged_forfig,
#               order=['CR', 'AR'], 
#               color='#545454',       # 점 색상
#               alpha=0.3,           # 투명도 (0~1)
#               jitter=0.1,          # 점들이 겹치지 않게 좌우로 흩뿌림
#               size=5,              # 점 크기
#               ax=ax)               # 같은 축(ax) 사용
# # 2. 비교할 그룹의 쌍(pair)을 정의합니다.
# pairs = [("CR", "AR")]

# # 3. Annotator 설정 및 적용
# annotator = Annotator(ax, pairs, data=merged_forfig, x='group', y='mean_group1_DUT_exp', order=['CR', 'AR'])

# # test='t-test_ind' (t-검정) 또는 'Mann-Whitney' (비모수 검정) 중 선택
# annotator.configure(test='Mann-Whitney', text_format='full', loc='inside', verbose=2, pvalue_format_string='{:.2f}')
# annotator.apply_and_annotate()

# sns.despine()
# plt.ylabel('mean_group1_DUT_exp')
# plt.title('group1 mean exp: CR vs AR') # 제목 추가 추천
# plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_pre/111_pre/figures/group1_DUT_exp_CRvsAR.pdf', dpi=300, bbox_inches='tight')
# plt.show()

#%%
ARdut_forval = AR_dut.loc[(AR_dut['p_value']<0.05) & np.abs(AR_dut['delta_TU']>0.05)].index.to_list()
ARgroup1dut = set(ARdut_forval).intersection(set(class1))

val_exp = val_tpm.loc[val_tpm.index.isin(ARgroup1dut), val_tpm.columns.isin(clin.index)]
val_dut = val_tu.loc[val_tu.index.isin(ARgroup1dut), val_tu.columns.isin(clin.index)]

merged_forfig['mean_group1_DUT_TU'] = val_dut.mean(axis=0)
merged_forfig['mean_group1_DUT_exp'] = val_exp.mean(axis=0)

plt.figure(figsize=(6,6))
sns.set(style='ticks')
# 1. Boxplot을 그릴 때 ax 객체를 저장합니다.
ax = sns.boxplot(x='group', y='mean_group1_DUT_TU', data=merged_forfig, 
                 order=['CR','IR','AR'], palette={"AR": "#FEB24C", "IR": "#5AAE61", "CR":"#58C1EE"}, showfliers=False)
sns.stripplot(x='group', y='mean_group1_DUT_TU', data=merged_forfig,
              order=['CR','IR','AR'], 
              color='#545454',       # 점 색상
              alpha=0.3,           # 투명도 (0~1)
              jitter=0.1,          # 점들이 겹치지 않게 좌우로 흩뿌림
              size=5,              # 점 크기
              ax=ax)               # 같은 축(ax) 사용
# 2. 비교할 그룹의 쌍(pair)을 정의합니다.
pairs = [("CR", "AR"),("CR","IR"),("AR","IR")]

# 3. Annotator 설정 및 적용
annotator = Annotator(ax, pairs, data=merged_forfig, x='group', y='mean_group1_DUT_TU', order=['CR','IR','AR'])

# test='t-test_ind' (t-검정) 또는 'Mann-Whitney' (비모수 검정) 중 선택
annotator.configure(test='Mann-Whitney', text_format='full', loc='inside', verbose=2, pvalue_format_string='{:.2f}')
annotator.apply_and_annotate()

sns.despine()
plt.ylabel('mean_group1_DUT_TU')
plt.title('Class1 AR DUT') # 제목 추가 추천
#plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_pre/111_pre/figures/group1_DUT_TU_CRvsARvsIR.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
merged_forfig = merged_forfig[merged_forfig['group'] != 'CR'] 
median_score = merged_forfig['mean_group1_DUT_exp'].median()
merged_forfig['Group'] = np.where(merged_forfig['mean_group1_DUT_exp'] >= median_score, 'High Exp', 'Low Exp')

kmf = KaplanMeierFitter()
plt.figure(figsize=(6, 6))

# High Group (Score 높음 -> 내성 예상 -> 예후 나쁨 예상)
mask_high = merged_forfig['Group'] == 'High Exp'
kmf.fit(merged_forfig[mask_high]['PFS'], event_observed=merged_forfig[mask_high]['recur'], label='High Group1 Exp')
kmf.plot_survival_function(color='#E64B35', ci_show=False) # 빨강

# Low Group (Score 낮음 -> 예후 좋음 예상)
mask_low = merged_forfig['Group'] == 'Low Exp'
kmf.fit(merged_forfig[mask_low]['PFS'], event_observed=merged_forfig[mask_low]['recur'], label='Low Group1 Exp')
kmf.plot_survival_function(color='#3C5488', ci_show=False) # 파랑

# Log-rank Test (두 그룹 간 차이 검정)
results = logrank_test(merged_forfig[mask_high]['PFS'], merged_forfig[mask_low]['PFS'], 
                       event_observed_A=merged_forfig[mask_high]['recur'], 
                       event_observed_B=merged_forfig[mask_low]['recur'])

# 꾸미기
plt.title(f"AR vs. IR (p = {results.p_value:.4f})", fontsize=15, fontweight='bold')
plt.xlabel("Progression-Free Survival (Months)", fontsize=12)
plt.ylabel("Survival Probability", fontsize=12)
plt.grid(True, alpha=0.3)
sns.despine()
plt.tight_layout()
plt.show()


median_score = merged_forfig['mean_group1_DUT_TU'].median()
merged_forfig['Group'] = np.where(merged_forfig['mean_group1_DUT_TU'] >= median_score, 'High TU', 'Low TU')

kmf = KaplanMeierFitter()
plt.figure(figsize=(6, 6))

# High Group (Score 높음 -> 내성 예상 -> 예후 나쁨 예상)
mask_high = merged_forfig['Group'] == 'High TU'
kmf.fit(merged_forfig[mask_high]['PFS'], event_observed=merged_forfig[mask_high]['recur'], label='High Group1 TU')
kmf.plot_survival_function(color='#E64B35', ci_show=False) # 빨강

# Low Group (Score 낮음 -> 예후 좋음 예상)
mask_low = merged_forfig['Group'] == 'Low TU'
kmf.fit(merged_forfig[mask_low]['PFS'], event_observed=merged_forfig[mask_low]['recur'], label='Low Group1 TU')
kmf.plot_survival_function(color='#3C5488', ci_show=False) # 파랑

# Log-rank Test (두 그룹 간 차이 검정)
results = logrank_test(merged_forfig[mask_high]['PFS'], merged_forfig[mask_low]['PFS'], 
                       event_observed_A=merged_forfig[mask_high]['recur'], 
                       event_observed_B=merged_forfig[mask_low]['recur'])

# 꾸미기
plt.title(f"AR vs. IR (p = {results.p_value:.4f})", fontsize=15, fontweight='bold')
plt.xlabel("Progression-Free Survival (Months)", fontsize=12)
plt.ylabel("Survival Probability", fontsize=12)
plt.grid(True, alpha=0.3)
sns.despine()
plt.tight_layout()
plt.show()

# %%
####^^ sampleinfo fig ########

## piechart for BRCAmut
counts = clin[clin['group']=='IR']['BRCAmt'].value_counts().sort_index()
labels = ['BRCAwt', 'BRCAmt']  # 0, 1 순서

colors = sns.color_palette('husl', n_colors=len(counts))
plt.figure(figsize=(3, 3))
wedges, texts, autotexts = plt.pie(
    counts,
    labels=None,   # label 제거
    autopct='%1.1f%%',
    startangle=90,
    textprops={'fontsize': 12},
    labeldistance=1.2,
    colors=colors
)
plt.axis('equal')  # 원형 유지
plt.legend(wedges, labels, loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/val_sampleinfo_IR_BRCAmut_piechart.pdf', dpi=300, bbox_inches='tight')
plt.show()

## piechart for drug
counts = clin[clin['group']=='IR']['drug'].value_counts()
colors = sns.color_palette('husl', n_colors=len(counts))
labels = ['Olaparib','Niraparib','Rucaparib']  # 0, 1 순서
plt.figure(figsize=(3, 3))
wedges, texts, autotexts = plt.pie(
    counts,
    labels=None,   # label 제거
    autopct='%1.1f%%',
    startangle=90,
    textprops={'fontsize': 12},
    labeldistance=0.5,
    colors=colors
)
plt.axis('equal')
plt.legend(wedges, labels, loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/val_sampleinfo_IR_drug_piechart.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%
## line info
clin['line_num'] = clin['line'].str.extract(r'(\d+)').astype(int)
clin['line_num'] = pd.Categorical(
    clin['line_num'],
    categories=sorted(clin['line_num'].dropna().unique()),
    ordered=True
)
plt.figure(figsize=(6, 3))
ax=sns.countplot(
    data=clin,
    x='line_num',
    hue='group',
    palette={"AR": "#FEB24C", "IR": "#5AAE61", "CR":"#58C1EE"}
)
from matplotlib.ticker import MaxNLocator
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.xlabel('Line')
plt.ylabel('Count')
plt.tight_layout()
sns.despine()
plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/val_sampleinfo_line_countplot.pdf', dpi=300, bbox_inches='tight')
plt.show()

##interval
palette = sns.color_palette('Set2', 2)

plt.figure(figsize=(6, 3))

ax = sns.boxplot(
    data=clin,
    x='PFS',
    y='group',
    order=['CR','IR','AR'],
    whis=1.5,
    linewidth=1.5,
    fliersize=4,
    width=0.6,
    palette={"AR": "#FEB24C", "IR": "#5AAE61", "CR":"#58C1EE"},
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5)
)

plt.xlabel('Days')
plt.ylabel('Group')
plt.tight_layout()
sns.despine()
plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/val_sampleinfo_interval_boxplot.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
