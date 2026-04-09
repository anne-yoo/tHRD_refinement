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
# with open('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/group1_new_translist.txt', 'r') as file:
#     repair_transcripts = [line.strip() for line in file]
# with open('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/group2_new_translist.txt', 'r') as file:
#     immune_transcripts = [line.strip() for line in file]
# with open('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/group3_new_translist.txt', 'r') as file:
#     developmental_transcripts = [line.strip() for line in file]
with open('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/hrr_translist.txt', 'r') as file:
    hrr_transcripts = [line.strip() for line in file]
with open('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/cycle_translist.txt', 'r') as file:
    cycle_transcripts = [line.strip() for line in file]
with open('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/pathway_translist.txt', 'r') as file:
    pathway_transcripts = [line.strip() for line in file]

# %%
AR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t', index_col=0)
IR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUT/nonresponder_stable_DUT_Wilcoxon.txt', sep='\t', index_col=0)

ARdutlist = AR_dut.loc[(AR_dut['p_value']<0.05) & (np.abs(AR_dut['log2FC'])>1.5)].index.to_list()
IRdutlist = IR_dut.loc[(IR_dut['p_value']<0.05) & (np.abs(IR_dut['log2FC'])>1.5)].index.to_list()

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
ARlist = list(set(sampleinfo.loc[(sampleinfo['response']==1),'sample_id']))
IRlist = list(set(sampleinfo.loc[(sampleinfo['response']==0),'sample_id']))

TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0)
TU.columns = TU.columns.str[:-4]
preTU = TU.iloc[:,1::2]
postTU = TU.iloc[:,0::2]

ARpre = preTU.loc[ARdutlist,ARlist]
ARpost = postTU.loc[ARdutlist,ARlist]
IRpre = preTU.loc[:,IRlist]
IRpost = postTU.loc[:,IRlist]

majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = majorminor[majorminor['type']=='major']['gene_ENST'].to_list()
minorlist = majorminor[majorminor['type']=='minor']['gene_ENST'].to_list()

ARpremajor = ARpre.loc[ARpre.index.isin(majorlist),:]
ARpostmajor = ARpost.loc[ARpost.index.isin(majorlist),:]
IRpremajor = IRpre.loc[IRpre.index.isin(majorlist),:]
IRpostmajor = IRpost.loc[IRpost.index.isin(majorlist),:]

# %%
###^^ AR subtype ####################

import pandas as pd
import numpy as np

# AR 환자들에 대한 ΔTU vector 생성 (환자 수 = n)
# repair_delta = ARpostmajor.loc[repair_transcripts].mean(axis=0) - ARpremajor.loc[repair_transcripts].mean(axis=0)
# immune_delta = ARpostmajor.loc[immune_transcripts].mean(axis=0) - ARpremajor.loc[immune_transcripts].mean(axis=0)
# devel_delta = ARpostmajor.loc[developmental_transcripts].mean(axis=0) - ARpremajor.loc[developmental_transcripts].mean(axis=0)

hrr_delta = ARpostmajor.loc[hrr_transcripts].mean(axis=0) - ARpremajor.loc[hrr_transcripts].mean(axis=0)
cycle_delta = ARpostmajor.loc[cycle_transcripts].mean(axis=0) - ARpremajor.loc[cycle_transcripts].mean(axis=0)
pathway_delta = ARpostmajor.loc[pathway_transcripts].mean(axis=0) - ARpremajor.loc[pathway_transcripts].mean(axis=0)


# 2. Feature matrix (3D)
X = np.vstack([hrr_delta.values, cycle_delta.values, pathway_delta.values]).T
patient_ids = ARpremajor.columns
df_feat = pd.DataFrame(X, columns=['hrr_delta', 'cycle_delta', 'pathway_delta'], index=patient_ids)

# 3. K-means clustering (3D)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X)
df_feat['subtype_kmeans'] = kmeans_labels
df_feat['subtype'] = df_feat['subtype_kmeans'].map({
    1: 'weak resistance',
    0: 'strong resistance'
})

# 4. 3D 시각화
fig = plt.figure(figsize=(6, 6))
plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 12,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 11,    # x축 틱 라벨 글꼴 크기

'ytick.labelsize': 12,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 12,
'legend.title_fontsize': 12, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})
ax = fig.add_subplot(111, projection='3d')

colors = {'weak resistance': '#7EC3FF', 'strong resistance': '#FF9BB1'}

for subtype in df_feat['subtype'].unique():
    subset = df_feat[df_feat['subtype'] == subtype]
    ax.scatter(
        subset['hrr_delta'], subset['cycle_delta'], subset['pathway_delta'],
        c=colors[subtype], label=subtype, s=80, edgecolor='k'
    )

ax.set_xlabel("ΔTU (Axis 1)", fontsize=11)
ax.set_ylabel("ΔTU (Axis 2)", fontsize=11)
ax.set_zlabel("ΔTU (Axis 3)", fontsize=11)
ax.set_title("AR subtype")
ax.legend(loc='upper left')
ax.set_box_aspect(None, zoom=0.85)
plt.tight_layout()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/3Dkmeans_hrr.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%
## [repair_delta, immune_delta] 2D feature matrix
X = np.vstack([hrr_delta.values, cycle_delta.values]).T  # shape: (n_samples, 2)

# index를 환자 ID로
patient_ids = ARpremajor.columns
df_feat = pd.DataFrame(X, columns=['hrr_delta', 'cycle_delta'], index=patient_ids)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X)

df_feat['subtype_kmeans'] = kmeans_labels  # 0 or 1

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5))
df_feat['subtype'] = df_feat['subtype_kmeans'].map({
    1: 'weak resistance',
    0: 'strong resistance'
})
sns.scatterplot(data=df_feat, x='hrr_delta', y='cycle_delta', hue='subtype', palette={'weak resistance':'#7EC3FF', 'strong resistance':'#FF9BB1'}, s=100)
plt.legend(loc='upper left')
plt.axhline(0, ls='--', color='gray', alpha=0.3)
plt.axvline(0, ls='--', color='gray', alpha=0.3)
plt.title("AR Subtype")
plt.xlabel("ΔTU (Homologous Recombination Restoration)")
plt.ylabel("ΔTU (Cell Cycle Checkpoint Activation)")
plt.tight_layout()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype_kmeans_hrrcycle.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%
####^^^ trajectory #################

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo_new.txt', sep='\t', index_col=0)
clin_df = sampleinfo.loc[df_feat.index]
clin_df['subtype'] = df_feat['subtype']
clin_df = clin_df.iloc[0::2,:]

# repair_post_mean = ARpostmajor.loc[repair_transcripts].mean(axis=0)
# repair_pre_mean = ARpremajor.loc[repair_transcripts].mean(axis=0)
# immune_post_mean = ARpostmajor.loc[immune_transcripts].mean(axis=0)
# immune_pre_mean = ARpremajor.loc[immune_transcripts].mean(axis=0)
# devel_post_mean = ARpostmajor.loc[developmental_transcripts].mean(axis=0)
# devel_pre_mean = ARpremajor.loc[developmental_transcripts].mean(axis=0)

hrr_post_mean = ARpostmajor.loc[hrr_transcripts].mean(axis=0)
hrr_pre_mean = ARpremajor.loc[hrr_transcripts].mean(axis=0)
cycle_post_mean = ARpostmajor.loc[cycle_transcripts].mean(axis=0)
cycle_pre_mean = ARpremajor.loc[cycle_transcripts].mean(axis=0)
pathway_post_mean = ARpostmajor.loc[pathway_transcripts].mean(axis=0)
pathway_pre_mean = ARpremajor.loc[pathway_transcripts].mean(axis=0)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

for sample in hrr_pre_mean.index:
    # 시작점 (pre-treatment)
    x0 = hrr_pre_mean.loc[sample]
    y0 = cycle_pre_mean.loc[sample]
    z0 = pathway_pre_mean.loc[sample]
    
    # 변화량 (post - pre)
    dx = hrr_post_mean.loc[sample] - x0
    dy = cycle_post_mean.loc[sample] - y0
    dz = pathway_post_mean.loc[sample] - z0

    # 색상 지정
    subtype = clin_df.loc[sample, 'subtype']
    color = '#FF9BB1' if subtype == 'strong resistance' else '#7EC3FF'
    interval = clin_df.loc[sample, 'interval']
    # 화살표 그리기
    ax.quiver(x0, y0, z0, dx, dy, dz,
              color=color, linewidth=1.5, arrow_length_ratio=0.05, alpha=0.85)
    x1, y1, z1 = x0 + dx +0.003, y0 + dy+0.003, z0 + dz+0.003
    interval = clin_df.loc[sample, 'interval']
    ax.text(x1, y1, z1, str(interval),
            fontsize=7, color='black', ha='center', va='center')

# 축 라벨 & 타이틀
ax.set_xlim(0, 0.15)
ax.set_ylim(0, 0.16)
ax.set_zlim(0, 0.2)
ax.set_xlabel("ΔTU (Axis 1)")
ax.set_ylabel("ΔTU (Axis 2)")
ax.set_zlabel("ΔTU (Axis 3)")
#ax.set_title("Pre → Post TU Trajectory (3D)")

#ax.view_init(elev=20, azim=0)
ax.set_box_aspect(None, zoom=0.9)

plt.tight_layout()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/3Dtrajectory_hrr_interval.pdf', dpi=300, bbox_inches='tight')

plt.show()
#%%
#############2D###########
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5))

for sample in hrr_pre_mean.index:
    x0 = hrr_pre_mean.loc[sample]   # ✅ X축: repair
    y0 = cycle_pre_mean.loc[sample]   # ✅ Y축: immune
    dx = hrr_post_mean.loc[sample] - x0
    dy = cycle_post_mean.loc[sample] - y0

    subtype = clin_df.loc[sample, 'subtype']
    color = '#FF9BB1' if subtype == 'strong resistance' else '#7EC3FF'
    interval = clin_df.loc[sample, 'interval']
    plt.arrow(x0, y0, dx, dy,
              head_width=0.003, head_length=0.003,
              fc=color, ec=color, alpha=0.8,
              length_includes_head=True)
    plt.text(x0 + dx+0.001, y0 + dy+0.001, str(int(interval)),
             fontsize=7, color='black', ha='center', va='center')


plt.ylabel('Cell Cycle Checkpoint Activation')
plt.xlabel('Homololgous Recombination Restoration')
plt.title('Pre → Post TU Trajectory')
plt.grid(True)
plt.tight_layout()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype_trajectory_hrrcycle.pdf', dpi=300, bbox_inches='tight')
plt.show()




# %%
sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo_new.txt', sep='\t', index_col=0)
clin_df = sampleinfo.loc[df_feat.index]
clin_df['subtype'] = df_feat['subtype']
clin_df = clin_df.iloc[0::2,:]

# %%
##^^^ interval
from statannotations.Annotator import Annotator

plt.figure(figsize=(4, 5))
ax = sns.boxplot(data=clin_df, x='subtype', y='interval',showfliers=False, palette={'weak resistance':'#7EC3FF', 'strong resistance':'#FF9BB1'})
sns.stripplot(data=clin_df, x='subtype', y='interval', color='gray', jitter=True)
plt.ylabel("treatment interval")
plt.xlabel("")
plt.tight_layout()
sns.despine()

pairs = [('weak resistance', 'strong resistance')]

annotator = Annotator(
    ax, pairs,
    data=clin_df,
    x='subtype', y='interval',
    order=['weak resistance', 'strong resistance'],
)
annotator.configure(test='Mann-Whitney', text_format='simple', loc='inside')
annotator.apply_and_annotate()
plt.title('treatment interval')
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/ARsubtype_interval_boxplot_hrrcycle.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
###^^^ cibersort #####
ARpre = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/cibersort/ARpre_immucellai.txt', sep='\t', index_col=0)
ARpost = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/cibersort/ARpost_immucellai.txt', sep='\t', index_col=0)


ARpre.index = ARpre.index.str[:-4]
ARpost.index = ARpost.index.str[:-4]

# ARpre = ARpre.iloc[:,:-1] #infiltration
# ARpost = ARpost.iloc[:,:-1]
ARpre = pd.DataFrame(ARpre.iloc[:,-1])
ARpost = pd.DataFrame(ARpost.iloc[:,-1])

ARpre['SampleID'] = ARpre.index
ARpost['SampleID'] = ARpost.index
ARpre['subtype'] = df_feat['subtype']
ARpost['subtype'] = df_feat['subtype']
ARpre['state'] = 'Pre'
ARpost['state'] = 'Post'

AR_df = pd.concat([ARpre, ARpost], ignore_index=True)


AR_melted = AR_df.melt(
    id_vars=['SampleID', 'subtype', 'state'],
    var_name='CellType',
    value_name='Proportion'
)

AR_strong = AR_melted[AR_melted['subtype']=='strong resistance']
AR_weak = AR_melted[AR_melted['subtype']=='weak resistance']

# 그래프 스타일 설정
sns.set(style="white")

from statannotations.Annotator import Annotator
# AR 조건에 대한 박스플롯 생성 및 p-value 추가
plt.figure(figsize=(5, 6)) #16,5
ax = sns.boxplot(data=AR_strong, x='CellType', y='Proportion', hue='state', palette={"Pre": "#FFBBCA", "Post": "#FF7C99"})
plt.title('Strong Resistance')
plt.xlabel('')
plt.ylabel('Proportion')
#plt.xticks(rotation=45, ha='right')
cell_types = AR_melted['CellType'].unique()
pairs = [((cell, 'Pre'), (cell, 'Post')) for cell in cell_types]

annotator = Annotator(
    ax, pairs,
    data=AR_strong,
    x='CellType', y='Proportion', hue='state',
    hue_order=['Pre', 'Post'],
    pairwise_controls={'subject': 'SampleID'}
)
annotator.configure(test='Wilcoxon', text_format='star', loc='inside')
annotator.apply_and_annotate()

plt.legend(title='State')
plt.tight_layout()
sns.despine()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/SR_ImmuCellAI_infiltration.pdf', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(5, 6))
ax = sns.boxplot(data=AR_weak, x='CellType', y='Proportion', hue='state', palette={"Pre": "#A2D4FF", "Post": "#51AEFF"})
plt.title('Weak Resistance')
plt.xlabel('')
plt.ylabel('Proportion')
#plt.xticks(rotation=45, ha='right')
cell_types = AR_melted['CellType'].unique()
pairs = [((cell, 'Pre'), (cell, 'Post')) for cell in cell_types]

annotator = Annotator(
    ax, pairs,
    data=AR_weak,
    x='CellType', y='Proportion', hue='state',
    hue_order=['Pre', 'Post'],
    pairwise_controls={'subject': 'SampleID'}
)
annotator.configure(test='Wilcoxon', text_format='star', loc='inside')
annotator.apply_and_annotate()

plt.legend(title='State')
plt.tight_layout()
sns.despine()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/WR_ImmuCellAI_infiltration.pdf', dpi=300, bbox_inches='tight')
plt.show()
# %%

clin_df['immune_delta'] = immune_delta
clin_df['repair_delta'] = repair_delta

plt.figure(figsize=(5, 4))
ax = sns.scatterplot(
    data=clin_df, x='repair_delta', y='immune_delta',
    hue='interval', palette='crest', s=100
)
for i in clin_df.index:
    x = clin_df.loc[i, 'repair_delta']
    y = clin_df.loc[i, 'immune_delta']
    interval = int(clin_df.loc[i, 'interval'])
    ax.text(x + 0.002, y + 0.002, str(interval), fontsize=8)
plt.xlabel("ΔTU (DNA Repair)")
plt.ylabel("ΔTU (Immune Response)")
plt.title("AR Subtypes by ΔTU with Treatment Interval Labels")
plt.axhline(0, ls=':', color='gray')
plt.axvline(0, ls=':', color='gray')
plt.legend(title='subtype')
plt.tight_layout()
plt.show()

#%%
#######^^^ gHRD ########
plusinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_validation_clinicalinfo_new.txt', sep='\t', index_col=0)
gHRD = plusinfo.loc[clin_df.index,'gHRDscore']
clin_df['gHRD'] = gHRD

plt.figure(figsize=(4, 5))
ax = sns.boxplot(data=clin_df, x='subtype', y='gHRD', palette={'weak resistance':'#7EC3FF', 'strong resistance':'#FF9BB1'})
sns.stripplot(data=clin_df, x='subtype', y='gHRD', color='gray', jitter=True)
plt.ylabel("gHRD score")
plt.xlabel("")
plt.tight_layout()
sns.despine()

pairs = [('weak resistance', 'strong resistance')]

annotator = Annotator(
    ax, pairs,
    data=clin_df,
    x='subtype', y='gHRD',
    order=['weak resistance', 'strong resistance'],
)
annotator.configure(test='t-test_ind', text_format='simple', loc='inside')
annotator.apply_and_annotate()
plt.title('gHRD score')
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/ARsubtype_gHRDscore_boxplot.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%
#########^^^ Stemmness index######################
ar_stem = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/stemness/AR_stemnessindex.txt', sep='\t', index_col=0)
ir_stem = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/stemness/IR_stemnessindex.txt', sep='\t', index_col=0)

ar_stem.columns = ['Stemness']
ir_stem.columns = ['Stemness']
ar_stem.index = ar_stem.index.astype(str)
ir_stem.index = ir_stem.index.astype(str)

# 🔧 Stemness DF 전처리
def prep_stemness_df(df, condition):
    df = df.copy()
    df['SampleID'] = df.index.str.replace('-atD', '', regex=False).str.replace('-bfD', '', regex=False)
    idx = df.index.astype(str) 
    df['State'] = idx.str.extract('-(atD|bfD)', expand=False).map({'bfD': 'Pre', 'atD': 'Post'})
    df['Condition'] = condition
    df = df.rename(columns={df.columns[0]: 'Stemness'})
    return df

ar_df = prep_stemness_df(ar_stem, 'AR')
ir_df = prep_stemness_df(ir_stem, 'IR')

# 🔁 박스플롯 그리는 함수
def draw_stemness_box(df, condition, palette):
    plt.figure(figsize=(4, 5))
    ax = sns.boxplot(data=df, x='State', y='Stemness', palette=palette, order=['Pre','Post'])
    sns.stripplot(data=df, x='State', y='Stemness', color='gray', jitter=True)
    plt.title(f"{condition} Stemness (Pre vs Post)")
    annotator = Annotator(ax, [('Pre', 'Post')], data=df, x='State', y='Stemness')
    annotator.configure(test='Wilcoxon', text_format='star', loc='inside')
    annotator.apply_and_annotate()
    plt.tight_layout()
    plt.show()

# 🎨 색 설정
ar_palette = {"Pre": "#FFCC29", "Post": "#FFAE4B"}
ir_palette = {"Pre": "#81B214", "Post": "#5F8805"}

# 📊 그리기
draw_stemness_box(ar_df, "AR", ar_palette)
draw_stemness_box(ir_df, "IR", ir_palette)

# strong/weak subtype 정보 붙이기
subtype_map = clin_df['subtype'].to_dict()
ar_df['Subtype'] = ar_df['SampleID'].map(subtype_map)

# 🔁 subtype별 boxplot
def draw_subtype_stemness(df, subtype, palette):
    df_sub = df[df['Subtype'] == subtype]
    plt.figure(figsize=(4, 5))
    ax = sns.boxplot(data=df_sub, x='State', y='Stemness', palette=palette,order=['Pre','Post'])
    sns.stripplot(data=df_sub, x='State', y='Stemness', color='gray', jitter=True)
    plt.title(f"{subtype.capitalize()} Stemness (Pre vs Post)")
    annotator = Annotator(ax, [('Pre', 'Post')], data=df_sub, x='State', y='Stemness')
    annotator.configure(test='Wilcoxon', text_format='star', loc='inside')
    annotator.apply_and_annotate()
    plt.tight_layout()
    plt.show()

# 🎨 팔레트 재사용
draw_subtype_stemness(ar_df, "strong resistance", ar_palette)
draw_subtype_stemness(ar_df, "weak resistance", ar_palette)

#%%
######^^^ Stemness Gene Marker ################
stemlist =['POU5F1','KIT','ABCG2','ABCB1','SNAI1','SNAI2','CDH2','CDH1']
stemlist = ['PROM1','KIT',]
genetpm = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM_symbol.txt', sep='\t', index_col=0)
stemtpm = genetpm[genetpm['Gene Symbol'].isin(stemlist)]
stemtpm.columns = stemtpm.columns.str[:-4]
prestem = stemtpm.iloc[:,1::2]
poststem = stemtpm.iloc[:,0::2]

ARprestem = prestem.loc[:,ARlist]
ARpoststem = poststem.loc[:,ARlist]
IRprestem = prestem.loc[:,IRlist]
IRpoststem = poststem.loc[:,IRlist]

AR_stem_melt = pd.concat([
    ARprestem.melt(var_name="Sample", value_name="TPM").assign(Group="Pre"),
    ARpoststem.melt(var_name="Sample", value_name="TPM").assign(Group="Post")
])
plt.figure(figsize=(4, 5))
ax = sns.boxplot(data=AR_stem_melt, x="Group", y="TPM", palette={"Pre": "#FFCC29", "Post": "#FFAE4B"})
sns.stripplot(data=AR_stem_melt, x="Group", y="TPM", color="gray", jitter=True)
annot = Annotator(ax, [("Pre", "Post")], data=AR_stem_melt, x="Group", y="TPM")
annot.configure(test='Wilcoxon', text_format='star', loc='inside')
annot.apply_and_annotate()
plt.title("AR Stemness Markers")
plt.tight_layout()
plt.show()

IR_stem_melt = pd.concat([
    IRprestem.melt(var_name="Sample", value_name="TPM").assign(Group="Pre"),
    IRpoststem.melt(var_name="Sample", value_name="TPM").assign(Group="Post")
])
plt.figure(figsize=(4, 5))
ax = sns.boxplot(data=IR_stem_melt, x="Group", y="TPM", palette={"Pre": "#81B214", "Post": "#5F8805"})
sns.stripplot(data=IR_stem_melt, x="Group", y="TPM", color="gray", jitter=True)
annot = Annotator(ax, [("Pre", "Post")], data=IR_stem_melt, x="Group", y="TPM")
annot.configure(test='t-test_paired', text_format='star', loc='inside')
annot.apply_and_annotate()
plt.title("IR Stemness Markers")
plt.tight_layout()
plt.show()

for subtype, color in {
    "weak resistance": ("#7EC3FF", "#FF9BB1"),
    "strong resistance": ("#7EC3FF", "#FF9BB1")
}.items():
    # 해당 subtype의 sample ID
    subtype_ids = clin_df[clin_df['subtype'] == subtype].index.tolist()
    subtype_ids = [pid for pid in subtype_ids if pid in prestem.columns and pid in poststem.columns]

    pre_val = prestem[subtype_ids].mean(axis=0)
    post_val = poststem[subtype_ids].mean(axis=0)

    df = pd.DataFrame({
        "TPM": pd.concat([pre_val, post_val]),
        "Group": ["Pre"] * len(pre_val) + ["Post"] * len(post_val)
    })

    plt.figure(figsize=(4, 5))
    ax = sns.boxplot(data=df, x="Group", y="TPM", palette={"Pre": color[0], "Post": color[1]})
    sns.stripplot(data=df, x="Group", y="TPM", color="gray", jitter=True)
    annot = Annotator(ax, [("Pre", "Post")], data=df, x="Group", y="TPM")
    annot.configure(test='Wilcoxon', text_format='star', loc='inside')
    annot.apply_and_annotate()
    plt.title(f"AR Stemness Markers ({subtype})")
    plt.tight_layout()
    plt.show()



#%%
#####^^^ BRCAmut############3
plusinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_validation_clinicalinfo_new.txt', sep='\t', index_col=0)
gHRD = plusinfo.loc[clin_df.index,'gHRDscore']
clin_df['gHRD'] = gHRD

plt.figure(figsize=(4, 5))
ax = sns.boxplot(data=clin_df, x='subtype', y='BRCAmut', palette={'weak resistance':'#7EC3FF', 'strong resistance':'#FF9BB1'})
sns.stripplot(data=clin_df, x='subtype', y='BRCAmut', color='gray', jitter=True)
plt.ylabel("gHRD score")
plt.xlabel("")
plt.tight_layout()
sns.despine()

pairs = [('weak resistance', 'strong resistance')]

annotator = Annotator(
    ax, pairs,
    data=clin_df,
    x='subtype', y='BRCAmut',
    order=['weak resistance', 'strong resistance'],
)
annotator.configure(test='t-test_ind', text_format='simple', loc='inside')
annotator.apply_and_annotate()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype_gHRDscore_boxplot.pdf', dpi=300, bbox_inches='tight')
plt.show()
#%%
#^^^^^ Pre-treatment check in discovery cohort ##########
# 준비: df_feat에 pre TU 정보 추가
clin_df['repair_pre'] = ARpremajor.loc[repair_transcripts].mean(axis=0)
clin_df['immune_pre'] = ARpremajor.loc[immune_transcripts].mean(axis=0)
pairs = [('weak resistance', 'strong resistance')]

# repair TU boxplot
plt.figure(figsize=(4, 5))
ax = sns.boxplot(data=clin_df, x='subtype', y='repair_pre',
            palette={'weak resistance':'#7EC3FF', 'strong resistance':'#FF9BB1'})
sns.stripplot(data=clin_df, x='subtype', y='repair_pre', color='gray', jitter=True)
plt.title("DNA Repair: pre-treatment")
plt.ylabel("Mean TU")
plt.xlabel("")
plt.tight_layout()

annotator = Annotator(
    ax, pairs,
    data=clin_df,
    x='subtype', y='repair_pre',
    order=['weak resistance', 'strong resistance'],
)
annotator.configure(test='t-test_ind', text_format='star', loc='inside')
annotator.apply_and_annotate()
sns.despine()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype_DNArepair_preTU_boxplot.pdf', dpi=300, bbox_inches='tight')
plt.show()

# immune TU boxplot
plt.figure(figsize=(4, 5))
ax = sns.boxplot(data=clin_df, x='subtype', y='immune_pre',
            palette={'weak resistance':'#7EC3FF', 'strong resistance':'#FF9BB1'})
sns.stripplot(data=clin_df, x='subtype', y='immune_pre', color='gray', jitter=True)
plt.title("Immune Response: pre-treatment")
plt.ylabel("Mean TU")
plt.xlabel("")
plt.tight_layout()

annotator = Annotator(
    ax, pairs,
    data=clin_df,
    x='subtype', y='immune_pre',
    order=['weak resistance', 'strong resistance'],
)
annotator.configure(test='t-test_ind', text_format='star', loc='inside')
annotator.apply_and_annotate()
sns.despine()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype_immune_preTU_boxplot.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
######******** AR vs. IR ################################
# AR
AR_repair_pre = ARpremajor.loc[repair_transcripts].mean()
AR_repair_post = ARpostmajor.loc[repair_transcripts].mean()
AR_immune_pre = ARpremajor.loc[immune_transcripts].mean()
AR_immune_post = ARpostmajor.loc[immune_transcripts].mean()
AR_devel_pre = ARpremajor.loc[developmental_transcripts].mean()
AR_devel_post = ARpostmajor.loc[developmental_transcripts].mean()

# IR
IR_repair_pre = IRpremajor.loc[repair_transcripts].mean()
IR_repair_post = IRpostmajor.loc[repair_transcripts].mean()
IR_immune_pre = IRpremajor.loc[immune_transcripts].mean()
IR_immune_post = IRpostmajor.loc[immune_transcripts].mean()
IR_devel_pre = IRpremajor.loc[developmental_transcripts].mean()
IR_devel_post = IRpostmajor.loc[developmental_transcripts].mean()

def make_long_format(pre, post, label, category):
    df = pd.DataFrame({
        'SampleID': pre.index,
        'Pre': pre.values,
        'Post': post.values
    }).melt(id_vars='SampleID', var_name='State', value_name='Mean_TU')
    df['Condition'] = label
    df['Category'] = category
    return df

# 각각 생성
AR_repair_df = make_long_format(AR_repair_pre, AR_repair_post, 'AR', 'GO Group 1')
IR_repair_df = make_long_format(IR_repair_pre, IR_repair_post, 'IR', 'GO Group 1')
AR_immune_df = make_long_format(AR_immune_pre, AR_immune_post, 'AR', 'GO Group 2')
IR_immune_df = make_long_format(IR_immune_pre, IR_immune_post, 'IR', 'GO Group 2')
AR_devel_df = make_long_format(AR_devel_pre, AR_devel_post, 'AR', 'GO Group 3')
IR_devel_df = make_long_format(IR_devel_pre, IR_devel_post, 'IR', 'GO Group 3')

# 통합
df_plot = pd.concat([AR_repair_df, IR_repair_df, AR_immune_df, IR_immune_df, AR_devel_df, IR_devel_df])

import seaborn as sns
import matplotlib.pyplot as plt
from statannotations.Annotator import Annotator
palette_dict = {
    'AR': {"Pre": "#FFCC29", "Post": "#FFAE4B"},
    'IR': {"Pre": "#81B214", "Post": "#5F8805"}
}
for category in ['GO Group 1', 'GO Group 2', 'GO Group 3']:
    for cond in ['AR', 'IR']:
        plot_df = df_plot[(df_plot['Category'] == category) & (df_plot['Condition'] == cond)]

        plt.figure(figsize=(4, 5))
        ax = sns.boxplot(
            data=plot_df,
            x='State', y='Mean_TU',
            palette=palette_dict[cond]
        )
        sns.stripplot(data=plot_df, x='State', y='Mean_TU', color='gray', jitter=True)

        annotator = Annotator(ax, [('Pre', 'Post')], data=plot_df, x='State', y='Mean_TU')
        annotator.configure(test='Wilcoxon', text_format='simple', loc='inside')
        annotator.apply_and_annotate()

        plt.title(f"{cond} - {category}")
        plt.ylabel("Mean TU")
        plt.xlabel("")
        plt.tight_layout()
        sns.despine()
        plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/prepostboxplot_'+ f"{cond}_{category}"+'.pdf', dpi=300, bbox_inches='tight')
        plt.show()

#%%
# 두 그룹의 pre만 따로 합치기
pre_repair_df = pd.DataFrame({
    'SampleID': AR_repair_pre.index.tolist() + IR_repair_pre.index.tolist(),
    'Condition': ['AR'] * len(AR_repair_pre) + ['IR'] * len(IR_repair_pre),
    'Mean_TU': list(AR_repair_pre.values) + list(IR_repair_pre.values),
    'Category': 'GO Group 1'
})

pre_immune_df = pd.DataFrame({
    'SampleID': AR_immune_pre.index.tolist() + IR_immune_pre.index.tolist(),
    'Condition': ['AR'] * len(AR_immune_pre) + ['IR'] * len(IR_immune_pre),
    'Mean_TU': list(AR_immune_pre.values) + list(IR_immune_pre.values),
    'Category': 'GO Group 2'
})

pre_devel_df = pd.DataFrame({
    'SampleID': AR_devel_pre.index.tolist() + IR_devel_pre.index.tolist(),
    'Condition': ['AR'] * len(AR_devel_pre) + ['IR'] * len(IR_devel_pre),
    'Mean_TU': list(AR_devel_pre.values) + list(IR_devel_pre.values),
    'Category': 'GO Group 3'
})

for df in [pre_repair_df, pre_immune_df, pre_devel_df]:
    plt.figure(figsize=(4, 5))
    ax = sns.boxplot(data=df, x='Condition', y='Mean_TU', palette={"AR": "#FFCC29", "IR": "#81B214"})
    sns.stripplot(data=df, x='Condition', y='Mean_TU', color='gray', jitter=True)
    plt.title(f"{df['Category'].iloc[0]}: pre-treatment")
    plt.ylabel("Mean TU")
    plt.xlabel("")

    annotator = Annotator(ax, [('AR', 'IR')], data=df, x='Condition', y='Mean_TU')
    annotator.configure(test='Mann-Whitney', text_format='simple', loc='inside')
    annotator.apply_and_annotate()
    plt.tight_layout()
    sns.despine()
    plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/preARIRboxplot_'+ f"{df['Category'].iloc[0]}"+'.pdf', dpi=300, bbox_inches='tight')
    plt.show()
#%%

gHRD = list(plusinfo.loc[pre_repair_df['SampleID'],'gHRDscore'])
BRCAmut = list(plusinfo.loc[pre_repair_df['SampleID'],'BRCAmut'])
pre_repair_df['gHRD'] = gHRD
pre_repair_df['BRCAmut'] = BRCAmut
plt.figure(figsize=(4, 5))
sns.boxplot(data=pre_repair_df, x='Condition', y='gHRD', palette={"AR": "#FFCC29", "IR": "#81B214"})
sns.stripplot(data=pre_repair_df, x='Condition', y='gHRD', color='gray', jitter=True)
plt.xlabel("")



# %%
##########^^^ Transcript 별 heatmap #################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

def generate_delta_heatmap(transcript_list, ARpre, ARpost, IRpre, IRpost, title="ΔTU Heatmap"):
    summary = []

    for t in transcript_list:
        if t not in ARpre.index or t not in ARpost.index or t not in IRpre.index or t not in IRpost.index:
            continue

        # AR
        pre_AR = ARpre.loc[t]
        post_AR = ARpost.loc[t]
        delta_AR = post_AR.mean() - pre_AR.mean()
        try:
            p_AR = wilcoxon(post_AR, pre_AR).pvalue
        except:
            p_AR = 1.0

        # IR
        pre_IR = IRpre.loc[t]
        post_IR = IRpost.loc[t]
        delta_IR = post_IR.mean() - pre_IR.mean()
        try:
            p_IR = wilcoxon(post_IR, pre_IR).pvalue
        except:
            p_IR = 1.0

        summary.append({
            "transcript": t,
            "AR": delta_AR,
            "AR_pval": p_AR,
            "IR": delta_IR,
            "IR_pval": p_IR
        })

    df = pd.DataFrame(summary).set_index("transcript")
    heat_data = df[["AR", "IR"]]

    def get_star(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return ""

    annot = [
        [get_star(df.loc[t, "AR_pval"]), get_star(df.loc[t, "IR_pval"])]
        for t in df.index
    ]

    # Plot
    plt.figure(figsize=(5, len(df)*0.4))
    ax = sns.heatmap(
        heat_data, cmap='RdBu_r', center=0,
        annot=annot, fmt='', linewidths=0.5,
        cbar_kws={"label": "ΔTU"}
    )
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/pathwayheatmap.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    return df

generate_delta_heatmap(pathway_transcripts, ARpremajor, ARpostmajor, IRpremajor, IRpostmajor)

# %%
