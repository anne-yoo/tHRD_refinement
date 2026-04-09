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
sqanti = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/sqantioutput/sqanti_classification.txt', sep='\t')
sqanti.dropna(axis=1, how='all', inplace=True)

# %%
##^^ simple histogram: length, exons

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
# 예: df라는 DataFrame에서 'age' 컬럼으로 히스토그램
sns.histplot(data=sqanti, x='length', bins=500, kde=False, color="#628141")
plt.xlabel("Transcript Length")
plt.ylabel("Count")
plt.xlim(0, 15000)
sns.despine()
plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/sqantioutput/figures/length_histogram.pdf', bbox_inches='tight', dpi=300)
plt.show()

plt.figure(figsize=(4,6))
sns.boxplot(data=sqanti, y='length', color="#628141",)
plt.ylabel("Transcript Length")
plt.ylim(0, 15000)
sns.despine()
plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/sqantioutput/figures/length_boxplot.pdf', bbox_inches='tight', dpi=300)
plt.show()


plt.figure(figsize=(6,4))
# 예: df라는 DataFrame에서 'age' 컬럼으로 히스토그램
sns.histplot(data=sqanti, x='exons', bins=200, kde=False, color="#628141")
plt.xlabel("Number of Exons")
plt.ylabel("Count")
plt.xlim(0, 40)
sns.despine()
plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/sqantioutput/figures/exon_histogram.pdf', bbox_inches='tight', dpi=300)
plt.show()

plt.figure(figsize=(4,6))
sns.boxplot(data=sqanti, y='exons', color="#628141",)
plt.ylabel("Number of Exons")
plt.ylim(0, 30)
sns.despine()
plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/sqantioutput/figures/exon_boxplot.pdf', bbox_inches='tight', dpi=300)
plt.show()

# %%
###^^ barplot: structural category ####
plt.figure(figsize=(6,4))
sns.countplot(
    data=sqanti,
    y='structural_category',
    order=sqanti['structural_category'].value_counts().index,
    palette="husl"
)
sns.despine()
plt.xlabel("Count")
plt.ylabel("Structural Category")
plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/sqantioutput/figures/category_barplot.pdf', bbox_inches='tight', dpi=300)
plt.show()


plt.figure(figsize=(6,4))
sns.countplot(
    data=sqanti[sqanti['isoform'].str.contains('MSTRG')],
    y='structural_category',
    order=sqanti['structural_category'].value_counts().index,
    palette="husl"
)
sns.despine()
plt.xlabel("Count")
plt.ylabel("Structural Category")
plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/sqantioutput/figures/mstrg_category_barplot.pdf', bbox_inches='tight', dpi=300)
plt.show()
# %%
plt.figure(figsize=(6,4))
sns.countplot(
    data=sqanti,
    y='subcategory',
    order=sqanti['subcategory'].value_counts().index,
    palette="husl"
)
sns.despine()
plt.xlabel("Count")
plt.ylabel("Subcategory")
plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/sqantioutput/figures/subcategory_barplot.pdf', bbox_inches='tight', dpi=300)
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(
    data=sqanti[sqanti['isoform'].str.contains('MSTRG')],
    y='subcategory',
    order=sqanti['subcategory'].value_counts().index,
    palette="husl"
)
sns.despine()
plt.xlabel("Count")
plt.ylabel("Subcategory")
plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/sqantioutput/figures/mstrg_subcategory_barplot.pdf', bbox_inches='tight', dpi=300)
plt.show()

#%%
# %%
val_clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2025clinical/83_POLO_clinicalinfo.txt', sep='\t', index_col=0)

# %%
AR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/maintenance/AR_stable_DUT_Wilcoxon_delta_withna.txt', sep='\t', index_col=0)
IR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/maintenance/IR_stable_DUT_Wilcoxon_delta_withna.txt', sep='\t', index_col=0)

ARdutlist = AR_dut.loc[(AR_dut['p_value']<0.05) & (np.abs(AR_dut['delta_TU'])>0.05)].index.to_list()
IRdutlist = IR_dut.loc[(IR_dut['p_value']<0.05) & (np.abs(IR_dut['delta_TU'])>0.05)].index.to_list()

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost_80_clinicalinfo.txt', sep='\t', index_col=0)
sampleinfo_full = sampleinfo.copy()
sampleinfo = sampleinfo[sampleinfo['purpose']=='maintenance']

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

TU.columns = TU.columns.str[:-4]
TU.index = TU.index.str.split("-", n=1).str[0]
preTU = TU.iloc[:,1::2]
postTU = TU.iloc[:,0::2]

# ARlist = [col[:-4] for col in ARlist]   
# IRlist = [col[:-4] for col in IRlist]
# ARpre = preTU.loc[ARdutlist,ARlist]
# ARpost = postTU.loc[ARdutlist,ARlist]
# IRpre = preTU.loc[IRdutlist,IRlist]
# IRpost = postTU.loc[IRdutlist,IRlist]

majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_80_majorminorlist.txt',sep='\t')
majorlist = majorminor[majorminor['type']=='major']['Transcript-Gene'].to_list()
minorlist = majorminor[majorminor['type']=='minor']['Transcript-Gene'].to_list()

sampleinfo = sampleinfo.iloc[::2,:]

#%%
##^^ cpat vs. cpc2

cpc2 = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/sqantioutput/cpc2output.txt', sep='\t', index_col=0)
cpat = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/sqantioutput/cpat.ORF_prob.best.tsv', sep='\t', index_col=0)

cpc2_noncoding = set(cpc2.loc[cpc2['label']=='noncoding'].index.to_list())
cpat_noncoding =set(cpat.loc[cpat['Coding_prob']<0.364].index.to_list())

from matplotlib_venn import venn2

plt.figure(figsize=(6,6))
sns.set_style("white")

vd2 = venn2([cpc2_noncoding, cpat_noncoding],set_labels=('cpc2', 'cpat'),set_colors=('#F8CF6A','#74B2D4'), alpha=0.8)

for text in vd2.set_labels:  #change label size
    text.set_fontsize(13)
for text in vd2.subset_labels:  #change number size
    text.set_fontsize(16)
    
vd2.get_patch_by_id('11').set_color('#C8C452')
vd2.get_patch_by_id('11').set_alpha(1)

#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/figures/001_variable_AR_IR_DUT_Venn.pdf", bbox_inches="tight")
plt.show()

noncodingtrans = cpc2_noncoding.intersection(cpat_noncoding)


#%% ####^^^ maintenance whole vs. DUT by sample#######################

from statannotations.Annotator import Annotator
df_cat = sqanti[['isoform','structural_category','subcategory','within_CAGE_peak','coding']]

# 1. 데이터 병합 (Merge)
# isoform을 기준으로 병합합니다.
pre_annot = preTU.merge(df_cat[['isoform', 'coding', 'within_CAGE_peak']],
                        left_index=True, right_on='isoform')
post_annot = postTU.merge(df_cat[['isoform', 'coding', 'within_CAGE_peak']],
                          left_index=True, right_on='isoform')

# 2. 그룹별 평균 계산 (수정됨)
# 중요: groupby().mean() 계산 시 'within_CAGE_peak' 등 숫자형 메타데이터가
# 샘플 데이터와 함께 평균 내어지는 것을 방지하기 위해 미리 drop 합니다.

# (A) Protein-coding 기준 그룹 평균
# isoform(문자열)과 within_CAGE_peak(숫자형)를 제외하고 coding으로만 그룹화
cols_to_drop_coding = ['within_CAGE_peak', 'isoform']
pre_coding = pre_annot.drop(columns=cols_to_drop_coding, errors='ignore').groupby('coding').mean().T
post_coding = post_annot.drop(columns=cols_to_drop_coding, errors='ignore').groupby('coding').mean().T

# (B) CAGE peak 기준 그룹 평균
# isoform(문자열)과 coding(불리언/숫자형)을 제외하고 within_CAGE_peak로만 그룹화
cols_to_drop_cage = ['coding', 'isoform']
pre_cage = pre_annot.drop(columns=cols_to_drop_cage, errors='ignore').groupby('within_CAGE_peak').mean().T
post_cage = post_annot.drop(columns=cols_to_drop_cage, errors='ignore').groupby('within_CAGE_peak').mean().T

# 3. 데이터 형태 변환 (Wide to Long) 및 Response 매핑

# --- Coding Data ---
df_pre_coding = pre_coding.assign(time='pre').reset_index().melt(
    id_vars=['index', 'time'], var_name='coding', value_name='TU'
)
df_post_coding = post_coding.assign(time='post').reset_index().melt(
    id_vars=['index', 'time'], var_name='coding', value_name='TU'
)
df_coding_long = pd.concat([df_pre_coding, df_post_coding])
df_coding_long = df_coding_long.rename(columns={'index': 'sample'})

# Response 매핑 (.map 사용이 더 안전하고 간결합니다)
df_coding_long['response'] = df_coding_long['sample'].map(sampleinfo['response']).map({1: 'AR', 0: 'IR'})

# --- CAGE Data ---
df_pre_cage = pre_cage.assign(time='pre').reset_index().melt(
    id_vars=['index', 'time'], var_name='CAGE', value_name='TU'
)
df_post_cage = post_cage.assign(time='post').reset_index().melt(
    id_vars=['index', 'time'], var_name='CAGE', value_name='TU'
)
df_cage_long = pd.concat([df_pre_cage, df_post_cage])
df_cage_long = df_cage_long.rename(columns={'index': 'sample'})

# Response 매핑
df_cage_long['response'] = df_cage_long['sample'].map(sampleinfo['response']).map({1: 'AR', 0: 'IR'})

# 4. 시각화 (Plotting)
plt.figure(figsize=(10, 5))
# 1. Catplot으로 그리기 (col='coding'으로 패널 분리)
g = sns.catplot(
    data=df_coding_long,
    x='response',
    y='TU',
    hue='time',
    col='coding',       # <--- 핵심: coding 여부에 따라 그래프를 쪼갭니다
    kind='box',
    palette=['#B2D085','#588513'],
    order=['AR', 'IR'],
    hue_order=['pre', 'post'],
    height=5, aspect=0.7, legend=False # 그래프 크기 조절
)
g.set_xlabels("")
# 2. 각 패널(subplot)마다 통계 검정 수행
# catplot은 여러 개의 ax를 가지므로 반복문으로 각각 그려줘야 합니다.
pairs = [
    (('AR', 'pre'), ('AR', 'post')),
    (('IR', 'pre'), ('IR', 'post')),
]

# g.axes.flat은 분할된 모든 그래프(ax)를 순회합니다
for ax in g.axes.flat:
    # 현재 ax에 그려진 데이터의 subset만 가져오는 게 아니라,
    # Annotator가 알아서 hue/x에 맞춰 매칭하지만,
    # 해당 ax의 title(예: coding = True)에 맞는 데이터만 필터링 되도록 주의해야 합니다.
    # 하지만 Annotator는 기본적으로 전체 데이터를 넘겨도 ax에 그려진 순서를 인식합니다.
    # 안전하게 하기 위해 plot_data=df_coding_long을 그대로 씁니다.
    
    # *중요*: FacetGrid와 Annotator를 같이 쓸 때는 
    # 각 ax가 어떤 데이터를 담고 있는지(coding=True인지 False인지) 알기 어렵기 때문에
    # 가장 간단한 방법은 각 ax에 대해 직접 annotate를 호출하는 것입니다.
    
    annot = Annotator(
        ax, pairs,
        data=df_coding_long, 
        x='response', y='TU', hue='time',
        order=['AR', 'IR'],
        hue_order=['pre', 'post']
    )
    annot.configure(test='Wilcoxon', text_format='star', loc='inside')
    
    # 3. 각 subplot 제목 설정 등 디자인 다듬기
    # subplot 제목에서 'coding = True' 등을 좀 더 깔끔하게 바꿀 수 있습니다.
    title = ax.get_title() 
    # 예: coding = True -> Coding, coding = False -> Non-coding
    # if 'non' in title or '1' in title:
    #     ax.set_title("Non-coding")
    # else:
    #     ax.set_title("Protein coding")
        
    annot.apply_and_annotate()
    
g.add_legend(
    title='time',
    loc='center left',   # 범례 상자의 기준점을 왼쪽 중앙으로 설정
    bbox_to_anchor=(1, 0.5) # 기준점을 전체 그림의 (1, 0.5) 위치에 배치
)
plt.tight_layout()
sns.despine()
#plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/sqantioutput/figures/ARIR_coding_boxplot.pdf', bbox_inches='tight', dpi=300)
plt.show()

#%%
####^^ DUT #####
from statannotations.Annotator import Annotator

proteincoding = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM_symbol.txt', sep='\t', index_col=0)
proteincodinglist = proteincoding['Gene Symbol'].to_list()

TU = filtered_trans.copy()
TU = TU.loc[:,mainlist]
TU.columns = TU.columns.str[:-4]
TU = TU[TU.index.str.split("-", n=1).str[-1].isin(proteincodinglist)]
TU.index = TU.index.str.split("-", n=1).str[0]
preTU = TU.iloc[:,1::2]
postTU = TU.iloc[:,0::2]

# (1) Pre 데이터 변환
df_pre_all = preTU.reset_index().rename(columns={'Transcript-Gene': 'isoform'})
df_post_all = postTU.reset_index().rename(columns={'Transcript-Gene': 'isoform'})
df_pre_all = df_pre_all.melt(
    id_vars='isoform', 
    var_name='sample', 
    value_name='TU'
)
df_pre_all['time'] = 'pre'
df_post_all = df_post_all.melt(
    id_vars='isoform', 
    var_name='sample', 
    value_name='TU'
)
df_post_all['time'] = 'post'

# (3) 합치기
df_all_long = pd.concat([df_pre_all, df_post_all], ignore_index=True)

# 2. 메타데이터 매핑 (Category & Response)

# (1) Coding / Non-coding 정보 매핑
# df_cat은 isoform을 인덱스나 컬럼으로 가지고 있어야 함
# 여기서는 df_cat이 isoform 컬럼을 가지고 있다고 가정하고 merge
df_all_long = df_all_long.merge(
    df_cat[['isoform', 'coding']], 
    on='isoform', 
    how='left'
)

sampleinfo_sal = sampleinfo_full[sampleinfo_full['purpose']=='salvage']
sampleinfo_sal = sampleinfo_sal.iloc[::2,:]
# (2) Response 정보 매핑
# sampleinfo['response']가 1(AR), 0(IR)로 되어있다고 가정
df_all_long['response'] = df_all_long['sample'].map(sampleinfo['response']).map({1: 'AR', 0: 'IR'})

# 3. 데이터가 너무 클 경우 시각화 팁 (Boxenplot 추천)
# 데이터가 수십만 건이면 boxplot은 뭉개져 보입니다. boxenplot이 분포를 더 잘 보여줍니다.

plt.figure(figsize=(10, 6))

g = sns.catplot(
    data=df_all_long,
    x='response',
    y='TU',
    hue='time',
    col='coding',       
    kind='box',       # <--- 데이터가 많을 때는 box 대신 boxen 추천 (또는 violin)
    palette=['#B2D085','#588513'],
    order=['AR', 'IR'],
    hue_order=['pre', 'post'],
    height=6, aspect=0.7,
    legend=False,
    showfliers=False    # <--- 이상치(점)가 너무 많으면 그래프가 안 보이니 끄는 옵션 고려
)

g.set_xlabels("")

# 4. 통계 검정 (주의: 데이터가 많으면 계산 오래 걸림)
pairs = [
    (('AR', 'pre'), ('AR', 'post')),
    (('IR', 'pre'), ('IR', 'post')),
]

for ax in g.axes.flat:
    # 제목 설정
    title = ax.get_title()
    if 'True' in title or '= coding' in title:
        ax.set_title("Protein-coding AR DUT (all gene)")
    else:
        ax.set_title("Non-coding AR DUT(all gene)")

    # 데이터 수가 너무 많으면(수십만 개) Wilcoxon 계산이 매우 오래 걸릴 수 있습니다.
    # 필요하다면 subsample을 하거나 p-value 표시를 생략해야 할 수도 있습니다.
    try:
        annot = Annotator(
            ax, pairs,
            data=df_all_long, 
            x='response', y='TU', hue='time',
            order=['AR', 'IR'],
            hue_order=['pre', 'post']
        )
        # 데이터가 많을 때는 't-test_ind' (독립표본) 혹은 'Mann-Whitney'를 쓰는 게 일반적입니다.
        # Wilcoxon(대응표본)을 쓰려면 샘플별/아이소폼별 짝이 정확히 맞아야 하는데, 
        # Long format에서 그 순서를 보장하기 까다로울 수 있습니다.
        # 전체 분포 비교이므로 Mann-Whitney를 추천합니다.
        annot.configure(test='Mann-Whitney', text_format='star', loc='inside')
        annot.apply_and_annotate()
    except Exception as e:
        print(f"Stats calculation skipped due to error or size: {e}")

# 범례 추가
g.add_legend(
    title='time',
    loc='center left',
    bbox_to_anchor=(1, 0.5)
)

plt.tight_layout()
#plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/sqantioutput/figures/main_ARIR_coding_boxplot_allgene_bytrans.pdf', bbox_inches='tight', dpi=300)
plt.show()

#%% ##^^ with ARdutlist and IRdutlist #################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator

# ---------------------------------------------------------
# 1. 데이터 로드 및 기본 전처리 (기존과 동일)
# ---------------------------------------------------------
proteincoding = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM_symbol.txt', sep='\t', index_col=0)
proteincodinglist = proteincoding['Gene Symbol'].to_list()

TU = filtered_trans.copy()
TU = TU.loc[:, TU.columns.isin(mainlist)]
TU.columns = TU.columns.str[:-4] # _TPM 제거 가정

# (옵션) Protein coding 필터링을 먼저 할지, 나중에 dutlist와 교집합할지 결정.
# 여기서는 일단 베이스로 남겨둡니다.
TU = TU[TU.index.str.split("-", n=1).str[-1].isin(proteincodinglist)]
TU.index = TU.index.str.split("-", n=1).str[0] # 인덱스 정리

# Pre/Post 분리 (컬럼 순서 기반)
preTU = TU.iloc[:, 1::2]
postTU = TU.iloc[:, 0::2]

# ---------------------------------------------------------
# 2. AR / IR 샘플 분리 및 개별 리스트(DUT list) 적용 (수정된 부분)
# ---------------------------------------------------------

# (1) 샘플 정보에서 AR과 IR 샘플 이름 추출
# sampleinfo의 인덱스가 샘플명과 일치한다고 가정합니다.
# map({1:'AR', 0:'IR'})을 사용하여 그룹을 나눕니다.
# (1) 샘플 컬럼 분리
ar_samples = sampleinfo[sampleinfo['response'] == 1].index.intersection(preTU.columns)
ir_samples = sampleinfo[sampleinfo['response'] == 0].index.intersection(preTU.columns)

# (2) --- [수정] DUT List 전처리 (ID만 추출) ---
# ARdutlist, IRdutlist가 'ID-Symbol' 형태이므로 '-'로 자르고 앞부분([0])만 가져옵니다.
ar_isoforms_clean = [x.split('-', 1)[0] for x in ARdutlist]
ir_isoforms_clean = [x.split('-', 1)[0] for x in IRdutlist]

# (3) AR 데이터 처리 (Clean된 리스트 적용)
# intersection을 통해 데이터프레임에 실제 존재하는 ID만 남깁니다.
target_genes_ar = preTU.index.intersection(ar_isoforms_clean)

df_pre_ar = preTU.loc[target_genes_ar, ar_samples].reset_index().rename(columns={'Transcript-Gene': 'isoform'})
df_post_ar = postTU.loc[target_genes_ar, ar_samples].reset_index().rename(columns={'Transcript-Gene': 'isoform'})

# Melt (AR)
df_pre_ar_long = df_pre_ar.melt(id_vars='isoform', var_name='sample', value_name='TU')
df_pre_ar_long['time'] = 'pre'
df_pre_ar_long['response'] = 'AR'

df_post_ar_long = df_post_ar.melt(id_vars='isoform', var_name='sample', value_name='TU')
df_post_ar_long['time'] = 'post'
df_post_ar_long['response'] = 'AR'

# (4) IR 데이터 처리 (Clean된 리스트 적용)
target_genes_ir = preTU.index.intersection(ir_isoforms_clean)

df_pre_ir = preTU.loc[target_genes_ir, ir_samples].reset_index().rename(columns={'Transcript-Gene': 'isoform'})
df_post_ir = postTU.loc[target_genes_ir, ir_samples].reset_index().rename(columns={'Transcript-Gene': 'isoform'})

# Melt (IR)
df_pre_ir_long = df_pre_ir.melt(id_vars='isoform', var_name='sample', value_name='TU')
df_pre_ir_long['time'] = 'pre'
df_pre_ir_long['response'] = 'IR'

df_post_ir_long = df_post_ir.melt(id_vars='isoform', var_name='sample', value_name='TU')
df_post_ir_long['time'] = 'post'
df_post_ir_long['response'] = 'IR'

# (5) 모든 데이터 합치기
df_all_long = pd.concat([df_pre_ar_long, df_post_ar_long, df_pre_ir_long, df_post_ir_long], ignore_index=True)

# ---------------------------------------------------------
# 3. 메타데이터 병합 (Coding 여부 등)
# ---------------------------------------------------------
# df_cat도 'isoform' 컬럼이 'ID-Symbol' 형태가 아니라 'ID'만 있어야 매칭됩니다.
# 만약 df_cat의 isoform도 정리가 안 되어 있다면 아래처럼 미리 정리해야 합니다.
# df_cat['isoform'] = df_cat['isoform'].astype(str).str.split('-', n=1).str[0]

df_all_long = df_all_long.merge(
    df_cat[['isoform', 'coding']], 
    on='isoform', 
    how='left'
)

# ---------------------------------------------------------
# 4. 시각화 (Plotting)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

g = sns.catplot(
    data=df_all_long,
    x='response',
    y='TU',
    hue='time',
    col='coding',       
    kind='box',         # 데이터 포인트가 많으면 'boxen' 추천
    palette=['#B2D085','#588513'],
    order=['AR', 'IR'],
    hue_order=['pre', 'post'],
    height=6, aspect=0.7,
    legend=False,
    showfliers=False
)

g.set_xlabels("")

# 5. 통계 검정
pairs = [
    (('AR', 'pre'), ('AR', 'post')),
    (('IR', 'pre'), ('IR', 'post')),
]

for ax in g.axes.flat:
    title = ax.get_title()
    # 타이틀 설정
    if 'True' in title or '1' in title or '= coding' in title:
        ax.set_title("Protein-coding DUTs")
    else:
        ax.set_title("Non-coding DUTs")

    try:
        annot = Annotator(
            ax, pairs,
            data=df_all_long, 
            x='response', y='TU', hue='time',
            order=['AR', 'IR'],
            hue_order=['pre', 'post']
        )
        # 데이터가 서로 다른 유전자 풀(pool)이므로 Mann-Whitney 사용
        annot.configure(test='Mann-Whitney', text_format='star', loc='inside')
        annot.apply_and_annotate()
    except Exception as e:
        print(f"Stats calculation skipped: {e}")

# 범례 추가
g.add_legend(
    title='time',
    loc='center left',
    bbox_to_anchor=(1, 0.5)
)

plt.tight_layout()
#plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/sqantioutput/figures/main_ARIR_DUT_coding_boxplot_allgene_bytrans.pdf', bbox_inches='tight', dpi=300)
plt.show()

#%%
####################^^ by sample ##############################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator

# ---------------------------------------------------------
# 1. 데이터 로드 및 기본 전처리
# ---------------------------------------------------------
proteincoding = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM_symbol.txt', sep='\t', index_col=0)
proteincodinglist = proteincoding['Gene Symbol'].to_list()

TU = filtered_trans.copy()
TU = TU.loc[:, TU.columns.isin(mainlist)]
TU = TU[TU.index.str.split("-", n=1).str[-1].isin(proteincodinglist)]
TU.columns = TU.columns.str[:-4] 

# 인덱스 정리 (Transcript ID만 남김)
TU.index = TU.index.str.split("-", n=1).str[0]

# Pre/Post 분리
preTU = TU.iloc[:, 1::2]
postTU = TU.iloc[:, 0::2]

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

df_list = []

# Coding / Non-coding 루프
for coding_status in ['coding', 'non_coding']: 
    # 해당 coding status에 맞는 유전자 ID 추출
    target_cat_genes = df_cat[df_cat['coding'] == coding_status]['isoform_clean']
    
    # AR / IR 루프
    for group_name in ['AR', 'IR']:
        if group_name == 'AR':
            current_samples = ar_samples
            current_dut_list = ar_isoforms_clean
        else:
            current_samples = ir_samples
            current_dut_list = ir_isoforms_clean
            
        # [핵심] 교집합 구하기: (PreTU 인덱스) & (DUT 리스트) & (Coding/Noncoding)
        ##^ #current_dut_list = set(TU.index.to_list())
        #current_dut_list = TU.index.to_list()
        final_genes = preTU.index.intersection(current_dut_list).intersection(target_cat_genes)
        final_genes = set(final_genes) - majorlist_set
        #final_genes = set(final_genes) - set(df_cat[df_cat['major'] == True]['isoform_clean'])
        print(coding_status,group_name, len(final_genes))
        # 유전자가 하나도 없는 경우 에러 방지
        if len(final_genes) == 0:
            continue
            
        # 평균 계산 (axis=0: 유전자들의 평균 -> 결과는 샘플별 값)
        # Pre
        mean_pre = preTU.loc[final_genes, current_samples].mean(axis=0)
        temp_pre = pd.DataFrame({
            'sample': mean_pre.index,
            'TU': mean_pre.values,
            'time': 'pre',
            'response': group_name,
            'coding': coding_status
        })
        
        # Post
        mean_post = postTU.loc[final_genes, current_samples].mean(axis=0)
        temp_post = pd.DataFrame({
            'sample': mean_post.index,
            'TU': mean_post.values,
            'time': 'post',
            'response': group_name,
            'coding': coding_status
        })
        
        df_list.extend([temp_pre, temp_post])

# 전체 데이터 병합
df_mean_long = pd.concat(df_list, ignore_index=True)
df_mean_long['color_group'] = df_mean_long['response'] + '_' + df_mean_long['time']
custom_palette = {
    'AR_pre': '#F1B08F', 'AR_post': '#EE7824',
    'IR_pre': '#B2D085', 'IR_post': '#588513'
}
rows = ['AR', 'IR']
cols = ['coding', 'non_coding'] # Coding 순서 (catplot의 col_order가 없으면 자동 정렬됨)
# 만약 catplot 생성 시 col_order, row_order를 안 줬다면 알파벳/숫자 순입니다.
# False(0), True(1) 순서일 가능성이 높습니다.
# 안전하게 order 지정:
g = sns.catplot(
    data=df_mean_long,
    x='time', y='TU',
    row='coding',       # 행: Coding (Major) / Non-coding (Minor)
    col='response',     # 열: AR / IR
    kind='box',
    order=['pre', 'post'],
    row_order=['coding', 'non_coding'], # 위: Major, 아래: Minor
    col_order=['AR', 'IR'],             # 왼쪽: AR, 오른쪽: IR
    hue='color_group',      # 1) hue를 새로 만든 그룹으로 지정
    palette=custom_palette, # 2) 커스텀 팔레트 적용
    dodge=False,
    height=4, aspect=1.2,
    sharey='row', legend=False# <--- 핵심: 같은 행끼리만 Y축 공유
)

# Lineplot 겹쳐 그리기
g.map_dataframe(sns.lineplot, x='time', y='TU', units='sample', estimator=None, color='grey', alpha=0.5, lw=1)

pairs = [('pre', 'post')]

# 2. 통계 적용 루프
# 배치가 바뀌었으므로 루프 순서도 row(coding) -> col(response) 순으로 접근해야 합니다.
row_vals = ['coding', 'non_coding'] # Major, Minor
col_vals = ['AR', 'IR']             # Groups

for i, row_val in enumerate(row_vals):
    for j, col_val in enumerate(col_vals):
        ax = g.axes[i, j]
        
        # 해당 패널 데이터 필터링
        # row_val이 coding 컬럼, col_val이 response 컬럼에 대응됨
        panel_data = df_mean_long[
            (df_mean_long['coding'] == row_val) & 
            (df_mean_long['response'] == col_val)
        ]
        
        if len(panel_data) == 0: continue

        # 통계 검정 (Wilcoxon)
        annot = Annotator(
            ax, pairs,
            data=panel_data, 
            x='time', y='TU',
            order=['pre', 'post']
        )
        
        annot.configure(test='Wilcoxon', text_format='star', loc='inside')
        annot.apply_and_annotate()
        
        # 제목 재설정 (사용자가 원하는 Major/Minor 표기 반영)
        type_str = "Protein-coding minor DUT" if (row_val == 'coding') else "Non-coding minor DUT"
        ax.set_title(f"{col_val} : {type_str}")
        
        # 0점 기준선
        ax.axhline(0, color='black', linewidth=1, linestyle='--')

plt.tight_layout()
#plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/sqantioutput/figures/main_ARIR_alltrans_coding_boxplot_allgene_bysample_005.pdf', bbox_inches='tight', dpi=300)
plt.show()
#%%

###& major minor
df_list = []

# Coding / Non-coding 루프
for coding_status in [True, False]: 
    # 해당 coding status에 맞는 유전자 ID 추출
    target_cat_genes = df_cat[df_cat['major'] == coding_status]['isoform_clean']
    
    # AR / IR 루프
    for group_name in ['AR', 'IR']:
        if group_name == 'AR':
            current_samples = ar_samples
            current_dut_list = ar_isoforms_clean
        else:
            current_samples = ir_samples
            current_dut_list = ir_isoforms_clean
            
        # [핵심] 교집합 구하기: (PreTU 인덱스) & (DUT 리스트) & (Coding/Noncoding)
        ##^ #current_dut_list = set(TU.index.to_list())
        final_genes = preTU.index.intersection(current_dut_list).intersection(target_cat_genes)
        
        # 유전자가 하나도 없는 경우 에러 방지
        if len(final_genes) == 0:
            continue
            
        # 평균 계산 (axis=0: 유전자들의 평균 -> 결과는 샘플별 값)
        # Pre
        mean_pre = preTU.loc[final_genes, current_samples].mean(axis=0)
        temp_pre = pd.DataFrame({
            'sample': mean_pre.index,
            'TU': mean_pre.values,
            'time': 'pre',
            'response': group_name,
            'major': coding_status
        })
        
        # Post
        mean_post = postTU.loc[final_genes, current_samples].mean(axis=0)
        temp_post = pd.DataFrame({
            'sample': mean_post.index,
            'TU': mean_post.values,
            'time': 'post',
            'response': group_name,
            'major': coding_status
        })
        
        df_list.extend([temp_pre, temp_post])

# 전체 데이터 병합
df_mean_long = pd.concat(df_list, ignore_index=True)
df_mean_long['color_group'] = df_mean_long['response'] + '_' + df_mean_long['time']
custom_palette = {
    'AR_pre': '#F1B08F', 'AR_post': '#EE7824',
    'IR_pre': '#B2D085', 'IR_post': '#588513'
}
rows = ['AR', 'IR']
cols = [True, False] # Coding 순서 (catplot의 col_order가 없으면 자동 정렬됨)
# 만약 catplot 생성 시 col_order, row_order를 안 줬다면 알파벳/숫자 순입니다.
# False(0), True(1) 순서일 가능성이 높습니다.
# 안전하게 order 지정:
g = sns.catplot(
    data=df_mean_long,
    x='time', y='TU',
    row='major',       # 행: Coding (Major) / Non-coding (Minor)
    col='response',     # 열: AR / IR
    kind='box',
    order=['pre', 'post'],
    row_order=[True,False], # 위: Major, 아래: Minor
    col_order=['AR', 'IR'],             # 왼쪽: AR, 오른쪽: IR
    hue='color_group',      # 1) hue를 새로 만든 그룹으로 지정
    palette=custom_palette, # 2) 커스텀 팔레트 적용
    dodge=False,
    height=4, aspect=1.2,
    sharey='row', legend=False# <--- 핵심: 같은 행끼리만 Y축 공유
)

# Lineplot 겹쳐 그리기
g.map_dataframe(sns.lineplot, x='time', y='TU', units='sample', estimator=None, color='grey', alpha=0.5, lw=1)

pairs = [('pre', 'post')]

# 2. 통계 적용 루프
# 배치가 바뀌었으므로 루프 순서도 row(coding) -> col(response) 순으로 접근해야 합니다.
row_vals = [True,False] # Major, Minor
col_vals = ['AR', 'IR']             # Groups

for i, row_val in enumerate(row_vals):
    for j, col_val in enumerate(col_vals):
        ax = g.axes[i, j]
        
        # 해당 패널 데이터 필터링
        # row_val이 coding 컬럼, col_val이 response 컬럼에 대응됨
        panel_data = df_mean_long[
            (df_mean_long['major'] == row_val) & 
            (df_mean_long['response'] == col_val)
        ]
        
        if len(panel_data) == 0: continue

        # 통계 검정 (Wilcoxon)
        annot = Annotator(
            ax, pairs,
            data=panel_data, 
            x='time', y='TU',
            order=['pre', 'post']
        )
        
        annot.configure(test='Wilcoxon', text_format='star', loc='inside')
        annot.apply_and_annotate()
        
        # 제목 재설정 (사용자가 원하는 Major/Minor 표기 반영)
        type_str = "Major DUT" if (row_val) else "Minor DUT"
        ax.set_title(f"{col_val} : {type_str}")
        
        # 0점 기준선
        ax.axhline(0, color='black', linewidth=1, linestyle='--')

plt.tight_layout()
plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/sqantioutput/figures/main_ARIR_DUT_filtered_majorminor_boxplot_allgene_bysample.pdf', bbox_inches='tight', dpi=300)
plt.show()

#%%
#%%
####^^^^^^ heatmap ################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# 1. 데이터 카테고리 정의 (순서 정렬을 위한 준비)
# ---------------------------------------------------------
# df_cat은 isoform을 인덱스로 가지고 있다고 가정합니다.
# 인덱스가 안 맞으면: df_cat = df_cat.set_index('isoform_clean') 등을 수행해야 함

# 조건에 따른 Category 컬럼 생성
# 1: Protein Coding Major
# 2: Protein Coding Minor
# 3: Non-coding Minor (나머지 Non-coding)


def assign_category(row):
    # 조건 1: Coding Major
    if (row['coding'] == 'coding') and (row['major'] == True):
        return 'Group1: Coding Major'
    # 조건 2: Coding Minor
    elif (row['coding'] == 'coding') and (row['major'] == False):
        return 'Group2: Coding Minor'
    # 조건 3: Non-coding (여기서는 Minor로 간주)
    elif row['coding'] == 'non_coding':
        return 'Group3: Noncoding Minor'
    else:
        return '4_Others' # 혹시 모를 예외

# df_cat에 적용 (인덱스 매칭 주의)
# 전사체 ID만 있는 컬럼이나 인덱스를 사용하세요.
# 예시에서는 인덱스가 'transcript_id'라고 가정하고 진행합니다.
df_cat['heatmap_group'] = df_cat.apply(assign_category, axis=1)

# Heatmap Row Colors 지정을 위한 색상 딕셔너리
category_colors = {
    'Group1: Coding Major': '#D62728',  # 빨강 (강조)
    'Group2: Coding Minor': '#FF7F0E',  # 주황
    'Group3: Noncoding Minor': '#1F77B4', # 파랑
    '4_Others': 'lightgrey'
}

# ---------------------------------------------------------
# 2. Heatmap 그리기 함수 정의
# ---------------------------------------------------------
def draw_custom_heatmap(group_name, dut_list, sample_list):
    """
    group_name: 'AR' or 'IR'
    dut_list: ARdutlist or IRdutlist (List of isoform IDs)
    sample_list: 해당 그룹의 sample ID list
    """
    
    # 1. DUT 리스트 전처리 (ID만 추출)
    clean_dut = [x.split('-', 1)[0] for x in dut_list]
    
    # 2. 데이터 추출 (Pre/Post 합치기)
    # preTU, postTU는 인덱스가 Transcript ID로 되어 있어야 함
    # 샘플 리스트 교집합 확인
    valid_samples = sample_list.intersection(preTU.columns)
    
    # 해당 그룹의 Pre, Post 데이터 가져오기 (행: 유전자, 열: 샘플)
    df_pre_sub = preTU.loc[clean_dut, valid_samples]
    df_post_sub = postTU.loc[clean_dut, valid_samples]
    
    # 3. 데이터 결합 (Left: Pre, Right: Post)
    # 컬럼명에 구분을 위해 suffix 추가 가능하지만, heatmap x축 라벨을 위해 일단 둠
    # 순서를 확실히 하기 위해 명시적으로 concat
    # (주의: 같은 샘플 순서로 pre와 post가 붙어야 1:1 비교가 시각적으로 잘 보임)
    combined_exp = pd.concat([df_pre_sub, df_post_sub], axis=1)
    
    # 4. 행(Row) 정렬: Category 순서대로
    # 현재 combined_exp의 인덱스(유전자)에 해당하는 category 정보를 가져옴
    row_cats = df_cat.loc[combined_exp.index, 'heatmap_group']
    
    # 정렬 키: 1. 그룹(Major->Minor->NC), 2. 유전자ID(알파벳순 - 선택사항)
    # 데이터를 정렬하기 위해 임시 데이터프레임 생성
    sort_df = pd.DataFrame({'group': row_cats})
    sort_df = sort_df.sort_values(by='group')
    
    # 정렬된 인덱스로 Expression Matrix 재정렬
    sorted_exp = combined_exp.loc[sort_df.index]
    
    # 5. Z-score Normalization (Row-wise)
    # 시각화를 위해 유전자별로 표준화 (Pre/Post 변화를 잘 보여줌)
    # axis=1 (행 방향 계산) -> 결과는 Transpose 되므로 다시 .T 필요하지만
    # scipy zscore나 sklearn scaler를 쓰면 편함. 여기서는 수동 계산:
    sorted_exp_z = sorted_exp.sub(sorted_exp.mean(axis=1), axis=0).div(sorted_exp.std(axis=1), axis=0)
    
    # 6. Row Colors 바 생성 (왼쪽에 붙을 색상 띠)
    row_colors = sort_df['group'].map(category_colors)
    
    # 7. 시각화
    plt.figure(figsize=(10, 10))
    
    # sns.clustermap을 쓰되 clustering을 끕니다 (우리가 정한 순서 유지를 위해)
    # row_cluster=False, col_cluster=False
    row_colors.name = ''
    g = sns.clustermap(
        sorted_exp_z,
        row_cluster=False, # Row 정렬 유지
        col_cluster=False, # Column 정렬 유지 (Pre -> Post 순서)
        row_colors=row_colors,
        cmap='vlag',       # 파랑-흰색-빨강 (발현량 변화에 적합)
        center=0,          # 0을 기준으로 (Z-score 0 = 평균)
        yticklabels=False, # 유전자 이름이 너무 많으면 끕니다.
        xticklabels=False, # 샘플 이름 필요하면 True
        cbar_pos=(0.02, 0.55, 0.03, 0.1), # 컬러바 위치 조절,
        vmin=-1.5, vmax=1.5, 
    )
    
    # 제목 및 레이아웃 설정
    g.ax_heatmap.set_title(f"{group_name}", fontsize=15, pad=20)
    g.ax_heatmap.set_ylabel("")
    # X축 구분선 추가 (Pre와 Post 사이)
    # 전체 컬럼 수의 절반 위치에 선 긋기
    mid_point = df_pre_sub.shape[1]
    g.ax_heatmap.axvline(mid_point, color='black', lw=2, linestyle='--')
    
    # 하단에 Pre / Post 텍스트 추가
    g.ax_heatmap.text(mid_point/2, sorted_exp_z.shape[0] + (sorted_exp_z.shape[0]*0.05), 
                      'Pre-treatment', ha='center', va='top', fontsize=12, fontweight='bold')
    g.ax_heatmap.text(mid_point + mid_point/2, sorted_exp_z.shape[0] + (sorted_exp_z.shape[0]*0.05), 
                      'Post-treatment', ha='center', va='top', fontsize=12, fontweight='bold')
    
    for label, color in category_colors.items():
        if label == '4_Others': continue
        g.ax_col_dendrogram.bar(0, 0, color=color, label=label, linewidth=0)
    
    # bbox_to_anchor의 두 번째 값(y)을 1.25 -> 1.05 정도로 낮춰서 히트맵 바로 위로 당깁니다.
    g.ax_col_dendrogram.legend(
        loc="lower center", 
        ncol=3, 
        bbox_to_anchor=(0.5, 1.01), # 그래프 바로 위쪽
        frameon=False,
        fontsize=11
    )
    plt.savefig(f'/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/maintenance/figures/{group_name}_DUT_heatmap_005.pdf', bbox_inches='tight', dpi=300)
    plt.show()

# ---------------------------------------------------------
# 3. 실행 (AR 및 IR 그룹에 대해 각각 실행)
# ---------------------------------------------------------

# AR 그룹 실행
# ar_samples는 이전 코드에서 구한 AR 환자 샘플 ID 리스트
print("Generating AR Heatmap...")
draw_custom_heatmap('Acquired Resistance Group', ARdutlist, ar_samples) #TU.index.to_list()

# IR 그룹 실행
# ir_samples는 이전 코드에서 구한 IR 환자 샘플 ID 리스트
print("Generating IR Heatmap...")
draw_custom_heatmap('Innate Resistance Group', IRdutlist, ir_samples) #TU.index.to_list()

#%%
###^^^^^ majorlist vs. codinglist #########
from matplotlib_venn import venn2
majorlist_set = set([x.split('-', 1)[0] for x in majorlist])
pcg_trans = set(TU.index.to_list())
coding_set = set(sqanti[sqanti['structural_category']=='full-splice_match']['isoform'])

plt.figure(figsize=(6,6))
sns.set_style("white")
vd2 = venn2([majorlist_set & pcg_trans, coding_set],set_labels=('major', 'FSM'),set_colors=('#F8CF6A','#74B2D4'), alpha=0.8)

for text in vd2.set_labels:  #change label size
    text.set_fontsize(13)
for text in vd2.subset_labels:  #change number size
    text.set_fontsize(16)
    
vd2.get_patch_by_id('11').set_color('#C8C452')
vd2.get_patch_by_id('11').set_alpha(1)

#plt.savefig("/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/sqantioutput/figures/major_vs_sqantiFSM_Venn.pdf", bbox_inches="tight")
plt.show()

#%%
tmp = sqanti[sqanti['isoform'].isin(majorlist_set)]















# %%
geneexp = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_80_gene_TPM.txt',sep='\t', index_col=0)
proteincoding = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM_symbol.txt', sep='\t', index_col=0)
proteincodinglist = proteincoding['Gene Symbol'].to_list()
geneexp = geneexp.loc[geneexp.index.isin(proteincodinglist)]
main_gene = geneexp[mainlist]
sal_gene = geneexp[sallist]

main_gene = main_gene.sort_index(axis=1)
main_gene.columns = main_gene.columns.str[:-4]

sal_gene = sal_gene.sort_index(axis=1)
sal_gene.columns = sal_gene.columns.str[:-4]


###^^ maintenance vs. salvage #################

main_pre = main_TU.iloc[:,1::2]
main_post = main_TU.iloc[:,0::2]
main_delta = main_post - main_pre

sal_pre = sal_TU.iloc[:,1::2]
sal_post = sal_TU.iloc[:,0::2]
sal_delta = sal_post - sal_pre

main_info = sampleinfo_full[sampleinfo_full['purpose']=='maintenance']
sal_info = sampleinfo_full[sampleinfo_full['purpose']=='salvage']

combined = pd.concat([main_pre, sal_pre], axis=1)
combined = combined[combined.index.str.split("-", n=1).str[-1].isin(proteincodinglist)]

label1 = (
    ["maintenance"] * main_pre.shape[1] +
    ["salvage"] * sal_pre.shape[1]
)

label2 = main_info.iloc[::2,-6].to_list() + sal_info.iloc[::2,-6].to_list() #BRCA
label3 = main_info.iloc[::2,2].to_list() + sal_info.iloc[::2,2].to_list() #response
label4 = main_info.iloc[::2,-3].to_list() + sal_info.iloc[::2,-3].to_list() #survival
label5 = main_info.iloc[::2,-5].to_list() + sal_info.iloc[::2,-5].to_list() #drug
label6 = main_info.iloc[::2,5].to_list() + sal_info.iloc[::2,5].to_list() #line
label7 = main_info.iloc[::2,4].to_list() + sal_info.iloc[::2,4].to_list() #exonicreads

import umap
from sklearn.preprocessing import StandardScaler

combined_log = np.log2(combined + 1)
X_scaled = StandardScaler().fit_transform(combined_log)
X_scaled = pd.DataFrame(X_scaled)
tu_var = X_scaled.var(axis=1)
top_transcripts = tu_var.sort_values(ascending=False).head(20000).index
X = X_scaled.loc[top_transcripts]
X = X.astype(float)
X = X.loc[~(X == 0).all(axis=1)]

# 3. CLR transform (compositional -> Euclidean)
X = X + 1e-6
gm = np.exp(np.log(X).mean(axis=0))
clr = np.log(X / gm)
clr = clr.T

# clr = X.T
reducer = umap.UMAP(random_state=42)
umap_emb = reducer.fit_transform(clr)

umap_df = pd.DataFrame({
    "UMAP1": umap_emb[:,0],
    "UMAP2": umap_emb[:,1],
    "label": label1
})

plt.figure(figsize=(6,5))
sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='label')
plt.title("UMAP")
plt.show()

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pc = pca.fit_transform(clr)

pca_df = pd.DataFrame({
    "PC1": pc[:,0],
    "PC2": pc[:,1],
    "purpose": label1
})

plt.figure(figsize=(6,5))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='purpose', palette="Set2")
plt.title("PCA")
plt.show()

# %%
from sklearn.metrics import silhouette_score
score = silhouette_score(clr, label1)
print(score)

#%%
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# 1. log2 transform
# ----------------------------
combined = pd.concat([main_pre, sal_pre], axis=1)
combined_log = np.log2(combined + 1)

# ----------------------------
# 2. top variable genes 선택 (axis=1 X, axis=0 O)
# ----------------------------
gene_var = combined_log.var(axis=1)      # gene variance across samples
top_genes = gene_var.sort_values(ascending=False).head(1000).index

X = combined_log.loc[top_genes]          # gene x sample

# ----------------------------
# 3. z-score scaling (sample axis)
# PCA는 반드시 scaling 필요
# ----------------------------
X_scaled = StandardScaler().fit_transform(X.T)   # sample x gene
X_scaled = pd.DataFrame(X_scaled, index=X.columns, columns=X.index)

# ----------------------------
# 4. PCA
# ----------------------------
pca = PCA(n_components=2)
pc = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PC1": pc[:,0],
    "PC2": pc[:,1],
    "purpose": label1
})

plt.figure(figsize=(6,5))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='purpose', palette="Set2")
plt.title("PCA")
plt.show()

# ----------------------------
# 5. UMAP (metric='euclidean' 권장)
# ----------------------------
reducer = umap.UMAP(random_state=42, metric="euclidean")
umap_emb = reducer.fit_transform(X_scaled)

umap_df = pd.DataFrame({
    "UMAP1": umap_emb[:,0],
    "UMAP2": umap_emb[:,1],
    "purpose": label1
})

plt.figure(figsize=(6,5))
sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='purpose', palette="Set2")
plt.title("UMAP")
plt.show()

# from sklearn.manifold import TSNE

# tsne = TSNE(
#     n_components=2,
#     perplexity=4,        # 샘플 수 적으면 낮게 설정
#     n_iter=2000,
#     random_state=42
# )

# tsne_emb = tsne.fit_transform(X_scaled)

# tsne_df = pd.DataFrame({
#     "tSNE1": tsne_emb[:,0],
#     "tSNE2": tsne_emb[:,1],
#     "label": label1
# })

# plt.figure(figsize=(6,5))
# sns.scatterplot(data=tsne_df, x='tSNE1', y='tSNE2', hue='label')
# plt.title("t-SNE")
# plt.show()

#%%
import numpy as np
import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler

# log-transform
combined_log = np.log2(combined + 1)

# remove transcripts that are all-zero across samples BEFORE scaling
combined_log = combined_log.loc[~(combined_log == 0).all(axis=1)]

# top variance selection
tu_var = combined_log.var(axis=1)
top_transcripts = tu_var.sort_values(ascending=False).head(1000).index
X = combined_log.loc[top_transcripts].astype(float)

# compositional correction (CLR)
X = X + 1e-6
gm = np.exp(np.log(X).mean(axis=0))
clr = np.log(X / gm)       # shape: transcripts × samples
clr = clr.T                # shape: samples × transcripts

# remove any remaining NaN/Inf (should be none)
clr = clr.replace([np.inf, -np.inf], np.nan)
clr = clr.dropna(axis=1, how='any')   # drop features with issues
clr = clr.dropna(axis=0, how='any')   # drop samples if needed

# scale AFTER CLR (good practice)
clr_scaled = StandardScaler().fit_transform(clr)

# UMAP
reducer = umap.UMAP(random_state=42)
umap_emb = reducer.fit_transform(clr_scaled)

# result
umap_df = pd.DataFrame({
    "UMAP1": umap_emb[:,0],
    "UMAP2": umap_emb[:,1],
    "purpose": label1  # just in case
})

plt.figure(figsize=(6,5))
sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='purpose', palette="Set2")
plt.title("UMAP")
plt.show()

pca = PCA(n_components=2)
pc = pca.fit_transform(clr_scaled)

pca_df = pd.DataFrame({
    "PC1": pc[:,0],
    "PC2": pc[:,1],
    "purpose": label1
})

plt.figure(figsize=(6,5))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='purpose', palette="Set2")
plt.title("PCA")
plt.show()




