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

#%%

AR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/DEGDUT/AR_stable_DUT_Wilcoxon.txt', sep='\t', index_col=0)
IR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/DEGDUT/IR_stable_DUT_Wilcoxon.txt', sep='\t', index_col=0)

ARdutlist = AR_dut.loc[(AR_dut['p_value']<0.05) & (np.abs(AR_dut['log2FC'])>1.5)].index.to_list()
IRdutlist = IR_dut.loc[(IR_dut['p_value']<0.05) & (np.abs(IR_dut['log2FC'])>1.5)].index.to_list()

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
ARlist = list(set(sampleinfo.loc[(sampleinfo['response']==1),'sample_id']))
IRlist = list(set(sampleinfo.loc[(sampleinfo['response']==0),'sample_id']))

TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost_80_transcript_TU.txt', sep='\t', index_col=0)
TU.columns = TU.columns.str[:-4]
preTU = TU.iloc[:,1::2]
postTU = TU.iloc[:,0::2]

ARpre = preTU.loc[ARdutlist,ARlist]
ARpost = postTU.loc[ARdutlist,ARlist]
IRpre = preTU.loc[IRdutlist,IRlist]
IRpost = postTU.loc[IRdutlist,IRlist]

majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/majorlist.txt', sep='\t')
majorlist = majorminor['Transcript-Gene'].to_list()
minorlist = TU.index.difference(majorlist).to_list()

#%%
#^^ major / minor #########
from scipy.stats import wilcoxon, ttest_rel

ARpremajor = ARpre.loc[ARpre.index.isin(majorlist),:]
ARpostmajor = ARpost.loc[ARpost.index.isin(majorlist),:]
IRpremajor = IRpre.loc[IRpre.index.isin(majorlist),:]
IRpostmajor = IRpost.loc[IRpost.index.isin(majorlist),:]

# ARpremajor = ARpre.loc[ARpre.index.isin(minorlist),:]
# ARpostmajor = ARpost.loc[ARpost.index.isin(minorlist),:]
# IRpremajor = IRpre.loc[IRpre.index.isin(minorlist),:]
# IRpostmajor = IRpost.loc[IRpost.index.isin(minorlist),:]

from scipy.stats import ttest_rel

# ΔTU 및 p-value 계산 함수
def compute_delta_pval(pre, post):
    delta = post.mean(axis=1) - pre.mean(axis=1)
    pvals = []
    for i in range(pre.shape[0]):
        _, p = wilcoxon(post.iloc[i, :], pre.iloc[i, :])
        pvals.append(p)
    return delta, np.array(pvals)

# AR 그룹 계산
delta_AR, pval_AR = compute_delta_pval(ARpremajor, ARpostmajor)
df_AR = pd.DataFrame({
    'transcript': ARpremajor.index,
    'delta': delta_AR,
    'neg_log10_pval': -np.log10(pval_AR + 1e-10),  # log(0) 방지
    'group': 'AR'
})

# IR 그룹 계산
delta_IR, pval_IR = compute_delta_pval(IRpremajor, IRpostmajor)
df_IR = pd.DataFrame({
    'transcript': IRpremajor.index,
    'delta': delta_IR,
    'neg_log10_pval': -np.log10(pval_IR + 1e-10),
    'group': 'IR'
})

# 두 그룹 합치기
df_plot = pd.concat([df_AR, df_IR], ignore_index=True)

# volcano plot 그리기
plt.figure(figsize=(6, 6))
colors = {"AR": "#FFCC29", "IR": "#81B214"}

for group in df_plot['group'].unique():
    sub = df_plot[df_plot['group'] == group]
    plt.scatter(sub['delta'], sub['neg_log10_pval'], label=group,
                color=colors[group], alpha=0.4, s=22)

plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
plt.xlabel('ΔTU')
plt.ylabel('-log10(pval)')
plt.title('ΔTU of major DUTs')
plt.legend()
plt.tight_layout()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/ARIRmajorDUTvolcano.pdf', dpi=300, bbox_inches='tight')
plt.show()


# %%
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt

ARpremajor = ARpre.loc[ARpre.index.isin(majorlist),:]
ARpostmajor = ARpost.loc[ARpost.index.isin(majorlist),:]
IRpremajor = IRpre.loc[IRpre.index.isin(majorlist),:]
IRpostmajor = IRpost.loc[IRpost.index.isin(majorlist),:]

# ARpremajor = ARpre.loc[ARpre.index.isin(minorlist),:]
# ARpostmajor = ARpost.loc[ARpost.index.isin(minorlist),:]
# IRpremajor = IRpre.loc[IRpre.index.isin(minorlist),:]
# IRpostmajor = IRpost.loc[IRpost.index.isin(minorlist),:]

# ΔTU + paired t-test
def compute_delta_pval(pre, post):
    delta = post.mean(axis=1) - pre.mean(axis=1)
    pvals = []
    for i in range(pre.shape[0]):
        _, p = wilcoxon(post.iloc[i, :], pre.iloc[i, :])
        pvals.append(p)
    return delta, np.array(pvals)

# AR
delta_AR, pval_AR = compute_delta_pval(ARpremajor, ARpostmajor)
df_AR = pd.DataFrame({
    'transcript': ARpremajor.index,
    'delta': delta_AR,
    'neg_log10_pval': -np.log10(pval_AR + 1e-10),
})

# IR
delta_IR, pval_IR = compute_delta_pval(IRpremajor, IRpostmajor)
df_IR = pd.DataFrame({
    'transcript': IRpremajor.index,
    'delta': delta_IR,
    'neg_log10_pval': -np.log10(pval_IR + 1e-10),
})

# Subplot Grid (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

# AR Plot (왼쪽)
axes[0].scatter(df_AR['delta'], df_AR['neg_log10_pval'], color="#FFCC29", alpha=0.5, s=20)
axes[0].axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
axes[0].set_title('TU change of AR major DUTs')
axes[0].set_xlabel('ΔTU')
axes[0].set_ylabel('-log10(pval)')

# IR Plot (오른쪽)
axes[1].scatter(df_IR['delta'], df_IR['neg_log10_pval'], color="#81B214", alpha=0.5, s=20)
axes[1].axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
axes[1].set_title('TU change of IR major DUTs')
axes[1].set_xlabel('ΔTU')

plt.tight_layout()
sns.despine()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/ARIRmajorDUTvolcano_ver2.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%
###^^^^ AR GO check #########
import gseapy as gp

AR_up = df_AR[df_AR['delta']>0]
AR_up['gene'] = AR_up['transcript'].str.split("-",n=1).str[-1]
glist = list(set(AR_up['gene']))
enr = gp.enrichr(
        gene_list=glist,
        gene_sets='GO_Biological_Process_2021',
        organism='Human',
        outdir=None
    )
enrresult = enr.results.sort_values(by=['Adjusted P-value']) 

#%%
###^^ every GO term check #####
import gseapy as gp
go_results = gp.get_library("GO_Biological_Process_2021", organism="Human")
GO_terms = list(go_results.keys())

from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests


ARpremajor = ARpre.loc[ARpre.index.isin(majorlist),:]
ARpostmajor = ARpost.loc[ARpost.index.isin(majorlist),:]
IRpremajor = IRpre.loc[IRpre.index.isin(majorlist),:]
IRpostmajor = IRpost.loc[IRpost.index.isin(majorlist),:]

# 🧬 유전자 이름 추출 (예: ENST00000373993.1-A1CF → A1CF)
transcript_genes = ARpremajor.index.to_series().str.split("-", n=1).str[-1]
transcript_to_gene = pd.Series(transcript_genes.values, index=ARpremajor.index)

# 🎯 GO term마다 ΔTU, p-value 계산
go_summary = []

for go_term, gene_list in go_results.items():
    # 해당 GO term에 속한 transcript 찾기
    matching_transcripts = transcript_to_gene[transcript_to_gene.isin(gene_list)].index

    if len(matching_transcripts) < 2:
        continue  # transcript 수가 너무 적으면 스킵

    pre = ARpremajor.loc[matching_transcripts]
    post = ARpostmajor.loc[matching_transcripts]

    # transcript별 ΔTU 평균
    delta = post.mean(axis=0) - pre.mean(axis=0)

    # 그룹 평균끼리 paired t-test
    try:
        _, pval = wilcoxon(post.mean(axis=0), pre.mean(axis=0))
    except:
        continue

    go_summary.append({
        'GO_term': go_term,
        'delta': delta.mean(),
        'p_value': pval,
        'transnum': len(matching_transcripts),
        'genelist': matching_transcripts
    })

# 🧾 결과 정리
go_df = pd.DataFrame(go_summary)
go_df['neg_log10_pval'] = -np.log10(go_df['p_value'] + 1e-10)
_, fdr, _, _ = multipletests(go_df['p_value'], method='fdr_bh')
go_df['FDR'] = fdr
go_df['neg_log10_FDR'] = -np.log10(go_df['FDR'] + 1e-10)

# 🔥 Volcano Plot 그리기
highlight_terms = ["regulation of double-strand break repair via homologous recombination (GO:0010569)",] #"double-strand break repair (GO:0006302)", "Wnt signaling pathway (GO:0016055)", "mitotic cell cycle phase transition (GO:0044772)" 
go_df['color'] = go_df['GO_term'].apply(lambda x: '#90D6F0' if x in highlight_terms else '#FFCC29')

#go_df = go_df[go_df['transnum'] > 10]  # transnum이 10 이상인 것만 필터링
##########################################################################3
highlight_keywords = [
    "homologous recombination","double-strand break repair","DNA damage",
    "cell cycle", "checkpoint", #"Wnt",
    #"phosphatidylinositol 3-kinase", "DNA damage"
]

go_df['color'] = go_df['GO_term'].apply(
    lambda x: '#90D6F0' if ('negative' not in x and any(keyword in x for keyword in highlight_keywords)) else '#FFCC29'
)
#############################################################################


# Plot non-highlighted points first
plt.figure(figsize=(6, 6))

# 기준선
plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)

#label 달기
for i, row in go_df.iterrows():
    if row['GO_term'] in highlight_terms:
        plt.text(row['delta'], row['neg_log10_FDR'] -0.25,  # y값 살짝 위로
                 row['GO_term'], fontsize=9, ha='center', color='black')
        
plt.scatter(
    go_df.loc[go_df['color'] == '#FFCC29', 'delta'],
    go_df.loc[go_df['color'] == '#FFCC29', 'neg_log10_FDR'],
    color='#FFCC29', #"#81B214",
    alpha=0.5,
)

# Overlay highlighted points
plt.scatter(
    go_df.loc[go_df['color'] == '#90D6F0', 'delta'],
    go_df.loc[go_df['color'] == '#90D6F0', 'neg_log10_FDR'],
    color='#ED4848',
    alpha=0.9, s=25,
    label='Highlighted'
)

plt.xlabel('Mean ΔTU')
plt.ylabel('-log10(pval)')
plt.title('AR major DUTs - GO term level')
plt.tight_layout()
sns.despine()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/IR_GOterm_volcano_dsb.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%
####^^ Barplot for AR GO terms #########

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# GO term 필터링
filtered_go_df = go_df[
    go_df['GO_term'].apply(
        lambda x: ('negative' not in x) and any(keyword in x for keyword in highlight_keywords)
    )
].copy()

# p-value log 변환
filtered_go_df['neglog10_p'] = -np.log10(filtered_go_df['p_value'])
filtered_go_df['neglog10_FDR'] = -np.log10(filtered_go_df['FDR'])

filtered_go_df = filtered_go_df[filtered_go_df['neglog10_FDR'] > 1]  # p-value < 0.05
filtered_go_df = filtered_go_df[filtered_go_df['transnum']>0]

# delta 정규화
norm = mcolors.Normalize(vmin=filtered_go_df['delta'].min(), vmax=filtered_go_df['delta'].max())
cmap = cm.get_cmap("Oranges")
colors = cmap(norm(filtered_go_df['delta']))

# 정렬
filtered_go_df = filtered_go_df.sort_values('neglog10_p', ascending=False)

# Plot
fig, ax = plt.subplots(figsize=(14, 7))

bars = ax.bar(
    x=filtered_go_df['GO_term'],
    height=filtered_go_df['neglog10_FDR'],
    color=colors,
    width=0.6
)

ax.set_xticklabels(
    filtered_go_df['GO_term'],
    rotation=30,
    ha='right',
    fontsize=10  # ← 여기를 원하는 크기로 조절!
)
ax.set_ylabel("-log10FDR")

# Colorbar 추가
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # matplotlib 3.1 이상에서 필요
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Mean ΔTU")

plt.tight_layout()
sns.despine()
plt.show()

#%%
#%%
# Sort the DataFrame by p-value in ascending order and select the top 30
top_go_terms = go_df[go_df['transnum']>10]
top_go_terms = top_go_terms.sort_values(by='p_value').head(50)
# Normalize 'delta' values for color mapping
norm = plt.Normalize(top_go_terms['delta'].min(), top_go_terms['delta'].max())
cmap = plt.get_cmap('coolwarm')  #

fig, ax = plt.subplots(figsize=(7, 9))

# Create a horizontal bar plot
bars = ax.barh(top_go_terms['GO_term'], -np.log10(top_go_terms['p_value']), color='#154674')
ax.set_xlabel('-log10(pval)')

# Adjust y-axis tick labels: set font size and ensure labels are fully visible
ax.set_yticklabels(top_go_terms['GO_term'], fontsize=10)

# Invert y-axis to have the lowest p-value at the top
ax.invert_yaxis()

# Adjust subplot parameters to give more room for y-axis labels
plt.subplots_adjust(left=0.65)  # Adjust the left margin as needed
sns.despine()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/AR_top50GOterm_barplot_filltered.pdf', dpi=300, bbox_inches='tight')

plt.show()

#%%
###^^ Group1 / Group2 check ####################3
import gseapy as gp
go_results = gp.get_library("GO_Biological_Process_2021", organism="Human")
GO_terms = list(go_results.keys())
ARpremajor = ARpre.loc[ARpre.index.isin(majorlist),:]
ARpostmajor = ARpost.loc[ARpost.index.isin(majorlist),:]
IRpremajor = IRpre.loc[IRpre.index.isin(majorlist),:]
IRpostmajor = IRpost.loc[IRpost.index.isin(majorlist),:]

# 🧬 유전자 이름 추출 (예: ENST00000373993.1-A1CF → A1CF)
transcript_genes = ARpremajor.index.to_series().str.split("-", n=1).str[-1]
transcript_to_gene = pd.Series(transcript_genes.values, index=ARpremajor.index)

group1_terms = [
    #'recombinational repair (GO:0000725)',
    'double-strand break repair (GO:0006302)',
    'double-strand break repair via homologous recombination (GO:0000724)',
    'mitotic cell cycle phase transition (GO:0044772)',
    'positive regulation of DNA-binding transcription factor activity (GO:0051091)'
]

group2_terms = [
    ##'positive regulation of inflammatory response (GO:0050729)',
    #'neutrophil degranulation (GO:0043312)',
    #'neutrophil activation involved in immune response (GO:0002283)',
    #'neutrophil mediated immunity (GO:0002446)',
    #'response to interleukin-1 (GO:0070555)',
    'cellular response to interleukin-1 (GO:0071347)',
    'inflammatory response (GO:0006954)',
    'neutrophil activation involved in immune response (GO:0002283)'
]

group3_terms = [
    #'embryonic organ morphogenesis (GO:0048562)',
    'positive regulation of developmental process (GO:0051094)',
    'positive regulation of cell differentiation (GO:0045597)',
    'positive regulation of canonical Wnt signaling pathway (GO:0090263)'
]

tmp =['embryonic organ morphogenesis (GO:0048562)','skeletal system morphogenesis (GO:0048705)']

repair_terms = ['double-strand break repair via homologous recombination (GO:0000724)']
cycle_terms = ['cell cycle G2/M phase transition (GO:0044839)','DNA damage checkpoint signaling (GO:0000077)']
pathway_terms = ['positive regulation of Wnt signaling pathway (GO:0030177)','positive regulation of phosphatidylinositol 3-kinase signaling (GO:0014068)','NIK/NF-kappaB signaling (GO:0038061)']

def get_transcripts_for_group(go_terms, go_results, transcript_to_gene):
    selected_transcripts = set()
    for go_term in go_terms:
        genes = go_results.get(go_term, [])
        transcripts = transcript_to_gene[transcript_to_gene.isin(genes)].index
        selected_transcripts.update(transcripts)
    return list(selected_transcripts)

# ▶️ 추출
group1_transcripts = get_transcripts_for_group(group1_terms, go_results, transcript_to_gene)
group2_transcripts = get_transcripts_for_group(group2_terms, go_results, transcript_to_gene)
group3_transcripts = get_transcripts_for_group(group3_terms, go_results, transcript_to_gene)

tmplist = get_transcripts_for_group(tmp, go_results, transcript_to_gene)

hrr_transcripts = get_transcripts_for_group(repair_terms, go_results, transcript_to_gene)
cycle_transcripts = get_transcripts_for_group(cycle_terms, go_results, transcript_to_gene)
pathway_transcripts = get_transcripts_for_group(pathway_terms, go_results, transcript_to_gene)
#%%
# with open('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/group1_new_translist.txt', 'w') as f:
#     for line in group1_transcripts:
#         f.write(f"{line}\n")
        
# with open('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/group2_new_translist.txt', 'w') as f:
#     for line in group2_transcripts:
#         f.write(f"{line}\n")
        
# with open('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/group3_new_translist.txt', 'w') as f:
#     for line in group3_transcripts:
#         f.write(f"{line}\n")
        
with open('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/hrr_translist.txt', 'w') as f:
    for line in hrr_transcripts:
        f.write(f"{line}\n")
with open('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/cycle_translist.txt', 'w') as f:
    for line in cycle_transcripts:
        f.write(f"{line}\n")
with open('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/pathway_translist.txt', 'w') as f:
    for line in pathway_transcripts:
        f.write(f"{line}\n")
#%%








#%%
##^^ cibersort /ImmucellAI######################

ARpre = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/cibersort/ARpre_immucellai.txt', sep='\t', index_col=0)
ARpost = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/cibersort/ARpost_immucellai.txt', sep='\t', index_col=0)
IRpre = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/cibersort/IRpre_immucellai.txt', sep='\t', index_col=0)
IRpost = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/cibersort/IRpost_immucellai.txt', sep='\t', index_col=0)

ARpre.index = ARpre.index.str[:-4]
ARpost.index = ARpost.index.str[:-4]
IRpre.index = IRpre.index.str[:-4]
IRpost.index = IRpost.index.str[:-4]

####################################
ARpre = pd.DataFrame(ARpre.iloc[:,-1]) #InfiltrationScore
ARpost = pd.DataFrame(ARpost.iloc[:,-1])
IRpre = pd.DataFrame(IRpre.iloc[:,-1])
IRpost = pd.DataFrame(IRpost.iloc[:,-1])
####################################

ARpre['SampleID'] = ARpre.index
ARpost['SampleID'] = ARpost.index
IRpre['SampleID'] = IRpre.index
IRpost['SampleID'] = IRpost.index

# 각 데이터프레임에 'Condition' 및 'State' 열 추가
ARpre['Condition'] = 'AR'
ARpre['State'] = 'Pre'
ARpost['Condition'] = 'AR'
ARpost['State'] = 'Post'
IRpre['Condition'] = 'IR'
IRpre['State'] = 'Pre'
IRpost['Condition'] = 'IR'
IRpost['State'] = 'Post'


# 3. concat
AR_df = pd.concat([ARpre, ARpost], ignore_index=True)

# 4. melt
AR_melted = AR_df.melt(
    id_vars=['SampleID', 'Condition', 'State'],
    var_name='CellType',
    value_name='Proportion'
)

IR_df = pd.concat([IRpre, IRpost], ignore_index=True)

# 4. melt
IR_melted = IR_df.melt(
    id_vars=['SampleID', 'Condition', 'State'],
    var_name='CellType',
    value_name='Proportion'
)

# 그래프 스타일 설정
sns.set(style="white")

from statannotations.Annotator import Annotator
# AR 조건에 대한 박스플롯 생성 및 p-value 추가
plt.figure(figsize=(5, 6)) #16,5
ax = sns.boxplot(data=AR_melted, x='CellType', y='Proportion', hue='State', palette={"Pre": "#FFCC29", "Post": "#FFAE4B"})
#plt.title('AR: Immune Cell Proportions Pre vs Post PARPi treatment')
plt.title('AR')
plt.xlabel('') #Immune Cell Type 
plt.ylabel('Proportion')
# plt.xticks(rotation=45, ha='right')
# plt.ylim([-0.02, 0.4])
cell_types = AR_melted['CellType'].unique()
pairs = [((cell, 'Pre'), (cell, 'Post')) for cell in cell_types]

annotator = Annotator(
    ax, pairs,
    data=AR_melted,
    x='CellType', y='Proportion', hue='State',
    hue_order=['Pre', 'Post'],
    pairwise_controls={'subject': 'SampleID'}
)
annotator.configure(test='Wilcoxon', text_format='star', loc='inside')
annotator.apply_and_annotate()

plt.legend(title='State')
plt.tight_layout()
sns.despine()

plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ImmuCellAI_AR_boxplot_infiltration.pdf', dpi=300, bbox_inches='tight')
plt.show()

# IR 조건에 대한 박스플롯 생성 및 p-value 추가
plt.figure(figsize=(5, 6))
ax = sns.boxplot(data=IR_melted, x='CellType', y='Proportion', hue='State', palette={"Pre": "#81B214", "Post": "#5F8805"})
# plt.title('IR: Immune Cell Proportions Pre vs Post PARPi treatment')
plt.title('IR')
plt.xlabel('') #Immune Cell Type 
plt.ylabel('Proportion')
# plt.xticks(rotation=45, ha='right')
#plt.ylim([-0.02, 0.4])
cell_types = IR_melted['CellType'].unique()
pairs = [((cell, 'Pre'), (cell, 'Post')) for cell in cell_types]

annotator = Annotator(
    ax, pairs,
    data=IR_melted,
    x='CellType', y='Proportion', hue='State',
    hue_order=['Pre', 'Post'],
    pairwise_controls={'subject': 'SampleID'}
)
annotator.configure(test='Wilcoxon', text_format='star', loc='inside')
annotator.apply_and_annotate()

plt.legend(title='State')
plt.tight_layout()
sns.despine()

plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ImmuCellAI_IR_boxplot_infiltration.pdf', dpi=300, bbox_inches='tight')
plt.show()
#%%




















# %%
####^^^^^^^^^^^^^^^^^^^^^ VALIDATION COHORT ##########################################
val = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_TU.txt', sep='\t', index_col=0)
valinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_validation_clinicalinfo_new.txt', sep='\t', index_col=0)
valinfo = valinfo[~((valinfo['ongoing']==1) & (valinfo['type']=='CR'))] #without ongoing check!!
typelist1 = list(valinfo['type']) #CR AR IR
typelist2 = ['R' if x=='CR' else x for x in typelist1]
typelist2 = ['R' if x=='AR' else x for x in typelist2]
valdata = val.iloc[:-2,val.columns.isin(valinfo.index.to_list())]
# %%
#%%
with open('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARup_tlist.txt', 'r') as file:
    tlist = [line.strip() for line in file]

ARpremajor = ARpre.loc[ARpre.index.isin(majorlist),:] ### major vs. minor
filteredval = valdata.loc[ARpremajor.index.to_list(),:]
filteredval = valdata.loc[tlist,:]
filteredval = filteredval.apply(pd.to_numeric, errors='coerce')

from statannotations.Annotator import Annotator

###^ val cohort check by sample ######
meanval = pd.DataFrame({'mean TU':filteredval.mean(),'type':valinfo['type'],'gHRDscore':valinfo['gHRDscore'],'BRCAmut':valinfo['BRCAmut']})

plt.figure(figsize=(5, 5))
ax = sns.boxplot(y='mean TU', x='type', data=meanval, 
                        showfliers=False, palette={"AR": "#FFCC29", "IR": "#81B214","CR":"#409EDD"}
                        )
# ax = sns.swarmplot(y='mean TU', x='type', data=meanval, 
#                         order=['CR','AR','IR'],size=4, color='grey', alpha=0.5
#                         #meanprops = {"marker":"o", "color":"blue","markeredgecolor": "blue", "markersize":"10"}
#                         )
ax.set_ylabel('mean TU by sample')
plt.xlabel("")
sns.despine()

pairs = [("CR","AR"),("CR","IR"),("AR","IR")]
annot = Annotator(ax, pairs, data=meanval, x="type", y="mean TU")
annot.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=1)
annot.apply_and_annotate()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/validation_ARminordutlist_boxplot_sample.pdf', dpi=300, bbox_inches='tight')
plt.show()

meanval = meanval.replace('CR', 'R')
meanval = meanval.replace('AR', 'R')

plt.figure(figsize=(4, 5))
ax = sns.boxplot(y='mean TU', x='type', data=meanval, 
                        showfliers=False, palette={"R": "#955DB3", "IR": "#81B214"}
                        )
# ax = sns.swarmplot(y='mean TU', x='type', data=meanval, 
#                         order=['CR','AR','IR'],size=4, color='grey'
#                         #meanprops = {"marker":"o", "color":"blue","markeredgecolor": "blue", "markersize":"10"}
#                         )
#ax.set_ylim([0,0.01])
ax.set_ylabel('mean TU')
plt.xlabel("")
sns.despine()
pairs = [("CR","AR"),("CR","IR"),("AR","IR")]
pairs = [("R","IR")]
annot = Annotator(ax, pairs, data=meanval, x="type", y="mean TU")
annot.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=1, ) ##correction_format="Bonferroni"
annot.apply_and_annotate()
plt.show()

# %%
###^ val cohort check by transcript ######
#filteredval = filteredval.drop(index='MSTRG.92133.3-PAK4')

#filteredval = valdata.loc[ARpremajor.index.to_list(),:]
filteredval = valdata.loc[tlist,:]
filteredval = filteredval.apply(pd.to_numeric, errors='coerce')

# Transcript 단위 평균 TU 계산
CR_mean = filteredval.loc[:, filteredval.columns.isin(valinfo[valinfo['type'] == 'CR'].index)].mean(axis=1)
AR_mean = filteredval.loc[:, filteredval.columns.isin(valinfo[valinfo['type'] == 'AR'].index)].mean(axis=1)
IR_mean = filteredval.loc[:, filteredval.columns.isin(valinfo[valinfo['type'] == 'IR'].index)].mean(axis=1)

# 새로운 DataFrame 생성 (Transcript별 TU 비교)
crdf = pd.DataFrame({'mean TU': CR_mean, 'type': 'CR', 'transcript': filteredval.index})
ardf = pd.DataFrame({'mean TU': AR_mean, 'type': 'AR', 'transcript': filteredval.index})
irdf = pd.DataFrame({'mean TU': IR_mean, 'type': 'IR', 'transcript': filteredval.index})

finaldf = pd.concat([crdf, ardf, irdf], axis=0).reset_index(drop=True)

# 시각화
plt.figure(figsize=(5, 5))
ax = sns.boxplot(y='mean TU', x='type', data=finaldf, showfliers=False,
                 palette={"AR": "#FFCC29", "IR": "#81B214", "CR": "#409EDD"})
#sns.swarmplot(y='mean TU', x='type', data=finaldf, color="grey", size=4, alpha=0.5)

# 통계 검정 (Mann-Whitney U test)
pairs = [("CR", "AR"), ("CR", "IR"), ("AR", "IR")]
annot = Annotator(ax, pairs, data=finaldf, x="type", y="mean TU")
annot.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=1, )#comparisons_correction="Bonferroni"
annot.apply_and_annotate()

ax.set_ylabel('mean TU per transcript')
sns.despine()
plt.xlabel("")
plt.show()

# Responder (R) vs. IR 비교
finaldf = finaldf.replace('CR', 'R')
finaldf = finaldf.replace('AR', 'R')

plt.figure(figsize=(4, 5))
ax = sns.boxplot(y='mean TU', x='type', data=finaldf, showfliers=False, palette={"R": "#955DB3", "IR": "#81B214"})
#sns.swarmplot(y='mean TU', x='type', data=finaldf, color="grey", size=3)

# 통계 검정 (Mann-Whitney U test)
pairs = [("R", "IR")]
annot = Annotator(ax, pairs, data=finaldf, x="type", y="mean TU")
annot.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=1, comparisons_correction="Bonferroni")
annot.apply_and_annotate()

ax.set_ylabel('mean TU per transcript')
sns.despine()
plt.xlabel("")
plt.show()

# %%
