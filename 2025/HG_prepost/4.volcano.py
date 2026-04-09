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
AR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/maintenance/AR_stable_DUT_Wilcoxon_delta_withna.txt', sep='\t', index_col=0)
IR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/maintenance/IR_stable_DUT_Wilcoxon_delta_withna.txt', sep='\t', index_col=0)

ARdutlist = AR_dut.loc[(AR_dut['p_value']<0.05) & (np.abs(AR_dut['delta_TU'])>0.1)].index.to_list()
IRdutlist = IR_dut.loc[(IR_dut['p_value']<0.05) & (np.abs(IR_dut['delta_TU'])>0.1)].index.to_list()

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost_80_clinicalinfo.txt', sep='\t', index_col=0)

sampleinfo = sampleinfo[sampleinfo['purpose']=='maintenance']
ARlist = list(set(sampleinfo.loc[(sampleinfo['response']==1),'sample_full']))
IRlist = list(set(sampleinfo.loc[(sampleinfo['response']==0),'sample_full']))

transexp = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_80_transcript_TPM.txt',sep='\t', index_col=0)
#transexp = transexp.iloc[:,:-1]

transexp = transexp[[col for col in transexp.columns if col in ARlist + IRlist]]
transexp = transexp.loc[(transexp > 0).sum(axis=1) >= 14] 

#transexp = transexp.loc[(transexp > 0).sum(axis=1) >= transexp.shape[1]*0.3]
transexp["gene"] = transexp.index.str.split("-", n=1).str[-1]
gene_sum = transexp.groupby("gene").transform("sum")
filtered_trans = transexp.iloc[:, :-1].div(gene_sum.iloc[:, :-1])
filtered_trans.fillna(0, inplace=True)
TU = filtered_trans
TU.columns = TU.columns.str[:-4]
preTU = TU.iloc[:,1::2]
postTU = TU.iloc[:,0::2]

ARlist = [col[:-4] for col in ARlist]   
IRlist = [col[:-4] for col in IRlist]
ARpre = preTU.loc[ARdutlist,ARlist]
ARpost = postTU.loc[ARdutlist,ARlist]
IRpre = preTU.loc[IRdutlist,IRlist]
IRpost = postTU.loc[IRdutlist,IRlist]

majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_80_majorminorlist.txt',sep='\t')
majorlist = majorminor[majorminor['type']=='major']['Transcript-Gene'].to_list()
minorlist = majorminor[majorminor['type']=='minor']['Transcript-Gene'].to_list()


#%% 
# ###^boxplot deltaTU check #########
from statannotations.Annotator import Annotator

sns.set_style('ticks')
# Add major/minor annotation
AR_dut['type'] = np.where(AR_dut.index.isin(majorlist), 'major', 'minor')
IR_dut['type'] = np.where(IR_dut.index.isin(majorlist), 'major', 'minor')

# Filter by significance
AR_sig = AR_dut[AR_dut['p_value'] < 0.05]
IR_sig = IR_dut[IR_dut['p_value'] < 0.05]

# Add cohort labels
AR_sig['cohort'] = 'AR'
IR_sig['cohort'] = 'IR'

# Combine
sig_all = pd.concat([AR_sig, IR_sig])

colors = {'AR': '#FFCC29', 'IR': '#81B214'}
pairs = [
    (("major", "AR"), ("major", "IR")),
    (("minor", "AR"), ("minor", "IR"))
]

plt.figure(figsize=(10, 6))
ax = sns.boxplot(
    data=sig_all,
    x='type',          # major / minor
    y='delta_TU',
    hue='cohort',      # AR / IR
    palette=colors,
    order=['major', 'minor']
)
plt.xlabel('ΔTU distribution (p < 0.05)')
plt.axhline(0, linestyle='--', color='gray')
plt.ylabel('ΔTU')
# Prepare annotation structure
annot = Annotator(ax, pairs, 
                  data=sig_all,
                  x='type', y='delta_TU',
                  hue='cohort', order=['major', 'minor'])

annot.configure(test='Mann-Whitney', text_format='star', loc='outside')
annot.apply_and_annotate()
sns.despine()
plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/maintenance/figures/sig_deltaTUboxplot.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Add cohort labels
AR_dut['cohort'] = 'AR'
IR_dut['cohort'] = 'IR'

all_data = pd.concat([AR_dut, IR_dut])

plt.figure(figsize=(10, 6))
ax = sns.boxplot(
    data=all_data,
    x='type',
    y='delta_TU',
    hue='cohort',
    palette=colors,
    order=['major', 'minor']
)
annot = Annotator(ax, pairs,
                  data=all_data,
                  x='type', y='delta_TU',
                  hue='cohort', order=['major', 'minor'])

annot.configure(test='Mann-Whitney', text_format='star', loc='outside')
annot.apply_and_annotate()

plt.xlabel('ΔTU distribution (all transcripts)')
plt.axhline(0, linestyle='--', color='gray')
plt.ylabel('ΔTU')
sns.despine()
plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/maintenance/figures/whole_deltaTUboxplot.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%
###^^ pval / log2FC volcano plot #########
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

ARmajordutlist = AR_dut.loc[(AR_dut['p_value']<0.05) & (AR_dut.index.isin(majorlist))].index.to_list()
IRmajordutlist = IR_dut.loc[(IR_dut['p_value']<0.05) & (IR_dut.index.isin(majorlist))].index.to_list()

# AR Plot (왼쪽)
axes[0].scatter(AR_dut.loc[ARmajordutlist,'delta_TU'], -np.log10(AR_dut.loc[ARmajordutlist,'p_value'] + 1e-10), color="#FFCC29", alpha=0.5, s=20)
axes[0].axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
axes[0].set_title('AR major DUT')
axes[0].set_xlabel('ΔTU')
axes[0].set_ylabel('-log10(pval)')
#axes[0].set_xlim(-10,10)

# IR Plot (오른쪽)
axes[1].scatter(IR_dut.loc[IRmajordutlist,'delta_TU'], -np.log10(IR_dut.loc[IRmajordutlist,'p_value'] + 1e-10), color="#81B214", alpha=0.5, s=20)
axes[1].axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
axes[1].set_title('IR major DUT')
axes[1].set_xlabel('ΔTU')
#axes[1].set_xlim(-10,10)

plt.tight_layout()
sns.despine()
plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/maintenance/figures/ARIR_majorDUT_volcano_deltaTU.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%
##^^ boxplot ###########



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
        _, p = wilcoxon(post.iloc[i, :], pre.iloc[i, :], zero_method='pratt')
        pvals.append(p)
    return delta, np.array(pvals)



# AR 그룹 계산
delta_AR, pval_AR = compute_delta_pval(ARpremajor, ARpostmajor) #ARpremajor/ARpostmajor
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
        _, p = wilcoxon(post.iloc[i, :], pre.iloc[i, :], zero_method='pratt')
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
axes[0].set_title('TU change of AR minor DUTs')
axes[0].set_xlabel('ΔTU')
axes[0].set_ylabel('-log10(pval)')

# IR Plot (오른쪽)
axes[1].scatter(df_IR['delta'], df_IR['neg_log10_pval'], color="#81B214", alpha=0.5, s=20)
axes[1].axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
axes[1].set_title('TU change of IR minor DUTs')
axes[1].set_xlabel('ΔTU')

plt.tight_layout()
sns.despine()
plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/maintenance/figures/ARIRminorDUTvolcano.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%
###^^^^ AR GO check #########
import gseapy as gp

# AR_up = AR_dut[AR_dut['p_value']<0.05]    #df_AR[df_AR['delta']>0]
# AR_up['gene'] = AR_up['transcript'].str.split("-",n=1).str[-1]
#glist = list(set(AR_dut[AR_dut['p_value']<0.05]['gene_name']))
glist = list(set(AR_dut.loc[(AR_dut['p_value']<0.05) & (AR_dut['delta_TU']>0.01) & (AR_dut.index.isin(majorlist))]['gene_name']))
enr = gp.enrichr(
        gene_list=glist,
        gene_sets='GO_Biological_Process_2021',
        organism='Human',
        outdir=None
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
file['Term'] = file['Term'].str.rsplit(" ",1).str[0]
#file = file[~file['Term'].str.contains('mitochondrial')]
file = file.iloc[:20,:]
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
plt.figure(figsize=(7,11))
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
#cbar.set_label('Overlap Percentage')
#plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/maintenance/majorDUT_plusdelta_GOenrichment.pdf', dpi=300, bbox_inches='tight')
plt.show()
#%%
###^^ every GO term check #####
import gseapy as gp
go_results = gp.get_library("GO_Biological_Process_2021", organism="Human")
GO_terms = list(go_results.keys())

from scipy.stats import ttest_rel
from statsmodels.stats.multitest import multipletests


# ARpremajor = ARpre.loc[ARpre.index.isin(minorlist),:]
# ARpostmajor = ARpost.loc[ARpost.index.isin(minorlist),:]
# IRpremajor = IRpre.loc[IRpre.index.isin(minorlist),:]
# IRpostmajor = IRpost.loc[IRpost.index.isin(minorlist),:]

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
        _, pval = wilcoxon(post.mean(axis=0), pre.mean(axis=0), zero_method='pratt')
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
sns.set_style("whitegrid")
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
    color='#FFCC29', #"#81B214",#FFCC29
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
plt.xlim(-0.5, 0.5)
sns.despine()
plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/maintenance/majorDUT_AR_GOlevel_withlabel.pdf', dpi=300, bbox_inches='tight')
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
filtered_go_df = filtered_go_df[filtered_go_df['transnum']>5]

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
top_go_terms = go_df[go_df['transnum']>5]
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

# ----- major transcript만 사용 -----
ARpremajor = ARpre.loc[ARpre.index.isin(majorlist)].copy()
ARpostmajor = ARpost.loc[ARpost.index.isin(majorlist)].copy()
IRpremajor = IRpre.loc[IRpre.index.isin(majorlist)].copy()
IRpostmajor = IRpost.loc[IRpost.index.isin(majorlist)].copy()

# ΔTU matrix (transcript × samples)
delta_AR = ARpostmajor - ARpremajor
delta_IR = IRpostmajor - IRpremajor

# transcript → gene 매핑 (index 형식: ENST...-GENE)
transcript_to_gene = ARpremajor.index.to_series().str.split("-", n=1).str[-1]
transcript_to_gene.index.name = "transcript"

GO_HRR  = "regulation of double-strand break repair via homologous recombination (GO:0010569)"
GO_NHEJ = "double-strand break repair via nonhomologous end joining (GO:0006303)"

def transcripts_for_go(go_term, transcript_to_gene, go_results):
    genes_in_go = set(go_results[go_term])
    mask = transcript_to_gene.isin(genes_in_go)
    return transcript_to_gene.index[mask]

HRR_trans  = transcripts_for_go(GO_HRR,  transcript_to_gene, go_results)
NHEJ_trans = transcripts_for_go(GO_NHEJ, transcript_to_gene, go_results)

print("AR HRR major transcript count:",  len(ARpremajor.index.intersection(HRR_trans)))
print("AR NHEJ major transcript count:", len(ARpremajor.index.intersection(NHEJ_trans)))


def make_prepost_long(pre_df, post_df, trans_list, label):
    common = pre_df.index.intersection(post_df.index).intersection(trans_list)
    if len(common) == 0:
        return pd.DataFrame()

    df = pd.DataFrame({
        "pre":  pre_df.loc[common].mean(axis=0),
        "post": post_df.loc[common].mean(axis=0)
    })

    df_long = df.melt(var_name="time", value_name="TU")
    df_long["pathway"] = label

    return df_long

AR_HRR_long  = make_prepost_long(ARpremajor, ARpostmajor, HRR_trans,  "HRR")
AR_NHEJ_long = make_prepost_long(ARpremajor, ARpostmajor, NHEJ_trans, "NHEJ")

plot_df = pd.concat([AR_HRR_long, AR_NHEJ_long], ignore_index=True)

def paired_pval(pre_df, post_df, trans_list):
    common = pre_df.index.intersection(post_df.index).intersection(trans_list)
    if len(common) == 0:
        return np.nan
    pre = pre_df.loc[common].mean(axis=1)
    post = post_df.loc[common].mean(axis=1)
    # Wilcoxon paired
    stat, p = wilcoxon(post, pre, zero_method="pratt")
    return p

p_HRR  = paired_pval(ARpremajor, ARpostmajor, HRR_trans)
p_NHEJ = paired_pval(ARpremajor, ARpostmajor, NHEJ_trans)

print("HRR pre vs post p-value:", p_HRR)
print("NHEJ pre vs post p-value:", p_NHEJ)
plt.figure(figsize=(8,6))
ax = sns.boxplot(
    data=plot_df,
    x="pathway",
    y="TU",
    hue="time",
    palette={"pre": "#8EC6FF", "post": "#FFB347"},
    order=["HRR","NHEJ"],
)

pairs = [
    (("HRR","pre"), ("HRR","post")),
    (("NHEJ","pre"), ("NHEJ","post"))
]

annot = Annotator(
    ax,
    pairs,
    data=plot_df,
    x="pathway",
    y="TU",
    hue="time",
    order=["HRR","NHEJ"],
)

annot.configure(
    test="Wilcoxon",
    text_format="star",
    loc="outside"
)
annot.apply_and_annotate()

plt.title("AR cohort — Pre vs Post TU (HRR vs NHEJ major DUTs)")
plt.ylabel("Transcript Usage (TU)")
sns.despine()
plt.tight_layout()
plt.show()

#%%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
def shannon_entropy_matrix(TU_df, gene_series):
    """
    TU_df: transcript × samples (already normalized)
    gene_series: transcript → gene mapping (same index as TU_df)
    
    Returns: gene × samples entropy matrix
    """
    # 1) log 처리 (0 log 0 방지)
    TU = TU_df.copy()
    TU = TU.replace(0, np.nan)
    logTU = np.log(TU)
    term = -(TU * logTU)

    # 2) gene-wise sum → entropy
    term["gene"] = gene_series
    entropy = term.groupby("gene").sum().fillna(0)

    return entropy

# transcript → gene 매핑
transcript_to_gene = preTU.index.to_series().str.split("-", n=1).str[-1]

# AR entropy pre/post
AR_entropy_pre  = shannon_entropy_matrix(preTU.loc[:,ARlist],  transcript_to_gene)
AR_entropy_post = shannon_entropy_matrix(postTU.loc[:,ARlist], transcript_to_gene)

# IR entropy pre/post
IR_entropy_pre  = shannon_entropy_matrix(preTU.loc[:,IRlist],  transcript_to_gene)
IR_entropy_post = shannon_entropy_matrix(postTU.loc[:,IRlist], transcript_to_gene)

def entropy_long(ent_pre, ent_post, cohort_label):
    df_pre = pd.DataFrame({
        "entropy": ent_pre.mean(axis=0),  # average entropy per patient
        "time": "pre",
        "cohort": cohort_label
    })
    df_post = pd.DataFrame({
        "entropy": ent_post.mean(axis=0),
        "time": "post",
        "cohort": cohort_label
    })
    return pd.concat([df_pre, df_post], ignore_index=True)

AR_long = entropy_long(AR_entropy_pre, AR_entropy_post, "AR")
IR_long = entropy_long(IR_entropy_pre, IR_entropy_post, "IR")

plot_df = pd.concat([AR_long, IR_long], ignore_index=True)
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

for i, cohort in enumerate(["AR", "IR"]):
    plt.subplot(1, 2, i+1)
    sub = plot_df[plot_df["cohort"] == cohort]

    ax = sns.kdeplot(
        data=sub, x="entropy", hue="time",
        fill=True, alpha=0.35, linewidth=2,
        palette={"pre": "#4B9CD3", "post": "#F4A259"},
    )

    plt.title(f"{cohort} cohort — Shannon entropy distribution")
    plt.xlabel("Shannon entropy")
    plt.ylabel("Density")
    plt.xlim(0, None)

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles=handles, labels=["pre", "post"], title="Time")


plt.tight_layout()
plt.show()


# %%


# major-only 필터링
# ARpre_major  = ARpre.loc[ARpre.index.iin(majorlist)]
# ARpost_major = ARpost.loc[ARpost.index.isin(majorlist)]
# IRpre_major  = IRpre.loc[IRpre.index.isin(majorlist)]
# IRpost_major = IRpost.loc[IRpost.index.isin(majorlist)]

ARpre_major  = ARpre.loc[ARpre.index.isin(minorlist)]
ARpost_major = ARpost.loc[ARpost.index.isin(minorlist)]
IRpre_major  = IRpre.loc[IRpre.index.isin(minorlist)]
IRpost_major = IRpost.loc[IRpost.index.isin(minorlist)]

# delta TU 계산
#delta_major = ARpost_major - ARpre_major
delta_major = IRpost_major - IRpre_major

# pre 평균 TU
#pre_mean = ARpre_major.mean(axis=1)
pre_mean = IRpre_major.mean(axis=1)

# delta TU 평균
delta_mean = delta_major.mean(axis=1)

df = pd.DataFrame({
    "pre_major_TU": pre_mean,
    "delta_major_TU": delta_mean,
})

# scatter plot
plt.figure(figsize=(6,5))
sns.scatterplot(data=df, x="pre_major_TU", y="delta_major_TU", alpha=0.5, color='#F4A259') # #81B214 #F4A259

# Optional: linear regression trend line
sns.regplot(
    data=df,
    x="pre_major_TU",
    y="delta_major_TU",
    scatter=False,
    ci=None,
    color="black",
    label="Linear fit"
)
rho, pval = stats.spearmanr(df["pre_major_TU"], df["delta_major_TU"])
text = f"Spearman r = {rho:.3f}\nP-value = {pval:.2e}"

plt.text(
    0.65,                     # x position (axes fraction)
    0.95,                     # y position (axes fraction)
    text,
    ha='left',
    va='top',
    transform=plt.gca().transAxes,
    fontsize=12,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
)

plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Pre-treatment major TU")
plt.ylabel("ΔTU (post - pre)")
plt.title("AR cohort")
sns.despine()
#plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/maintenance/figures/AR_delta_pre_scatterplot.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse

# =============================================
# Function: draw GMM covariance ellipse
# =============================================
def plot_cov_ellipse(mean, cov, ax, color, alpha=0.2):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]

    # 95% confidence ellipse scale
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(vals * 5.991)  # chi-square for 2 df at 95%

    ell = Ellipse(xy=mean, width=width, height=height, angle=theta,
                  color=color, alpha=alpha, lw=2)
    ax.add_patch(ell)


# =============================================
# Fit 2-component GMM
# =============================================
X = df[["pre_major_TU", "delta_major_TU"]].values

gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm.fit(X)

labels = gmm.predict(X)
df["cluster"] = labels

# Cluster colors
cluster_colors = {0: "#FFAA33", 1: "#3366CC"}

# =============================================
# Plotting
# =============================================
fig, ax = plt.subplots(figsize=(8, 7))

# ① Density background
sns.kdeplot(
    data=df, x="pre_major_TU", y="delta_major_TU",
    levels=10, fill=True, cmap="Blues", alpha=0.4, ax=ax
)

# ② Scatter with cluster colors
sns.scatterplot(
    data=df,
    x="pre_major_TU",
    y="delta_major_TU",
    hue="cluster",
    palette=cluster_colors,
    s=30,
    alpha=0.7,
    ax=ax
)

# ③ GMM ellipses
for k in range(2):
    mean = gmm.means_[k]
    cov = gmm.covariances_[k]
    plot_cov_ellipse(mean, cov, ax, color=cluster_colors[k], alpha=0.2)

# ④ Aesthetics
ax.axhline(0, linestyle="--", color="gray", alpha=0.6)
ax.set_xlabel("Pre-treatment major TU", fontsize=12)
ax.set_ylabel("ΔTU (post - pre)", fontsize=12)
ax.set_title("IR minor: GMM clustering", fontsize=14)

plt.legend(title="Cluster")
sns.despine()
plt.tight_layout()
plt.savefig('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/maintenance/figures/IR_minorDUT_GMM_clustering.pdf', dpi=300, bbox_inches='tight')
plt.show()

# =============================================
# Return GMM model + df with cluster label
# =============================================
#df.to_csv("AR_GMM_clusters_majorTU.csv")
print("Saved: AR_GMM_clusters_majorTU.csv")

#%%
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from scipy.stats import ranksums
from gseapy import enrichr


df_ar = df   # 이전 단계에서 만든 df (pre_major_TU, delta_major_TU, cluster 포함)

cluster0_genes = df_ar[df_ar.cluster == 0].index
cluster1_genes = df_ar[df_ar.cluster == 1].index

print("Cluster 0 transcripts:", len(cluster0_genes))
print("Cluster 1 transcripts:", len(cluster1_genes))


# ===============================================================
# 1) GO enrichment (Biological Process)
# ===============================================================

def run_go_enrichment(transcripts, gene_map):
    genes = gene_map.loc[transcripts].unique().tolist()

    enr = enrichr(
        gene_list = genes,
        gene_sets = "GO_Biological_Process_2021",
        organism = "Human",
        outdir = None
    )
    return enr.results

# transcript → gene 매핑
transcript_to_gene = df_ar.index.to_series().str.split("-", n=1).str[-1]
gene_map = pd.Series(transcript_to_gene.values, index=df_ar.index)

go_cluster0 = run_go_enrichment(cluster0_genes, gene_map)
go_cluster1 = run_go_enrichment(cluster1_genes, gene_map)

go_cluster0.head(), go_cluster1.head()


# ===============================================================
# 2) ΔTU matrix 생성 (transcript × sample)
# ===============================================================

delta_AR_matrix = ARpost_major.loc[df_ar.index] - ARpre_major.loc[df_ar.index]


# ===============================================================
# 3) 환자별 cluster load score 계산
# ===============================================================

cluster0_load = delta_AR_matrix.loc[cluster0_genes].mean(axis=0)
cluster1_load = delta_AR_matrix.loc[cluster1_genes].mean(axis=0)

cluster_load_df = pd.DataFrame({
    "cluster0_load": cluster0_load,
    "cluster1_load": cluster1_load
})
cluster_load_df.index.name = "sample"


# ===============================================================
# 4) clinical 데이터 결합
# ===============================================================

clin = sampleinfo.copy()
clin = clin.iloc[::2,:]
clin = clin.loc[cluster_load_df.index]

merged = pd.concat([cluster_load_df, clin], axis=1)

# survival info
merged["PFS"] = merged["PFI"]  # PFI 컬럼 사용
merged["event"] = merged["survival"]  # recurrence 1/0


# ===============================================================
# 5) Survival analysis (Kaplan–Meier)
# ===============================================================

def km_plot(load, name):
    median_cut = merged[load].median()
    merged["group"] = (merged[load] >= median_cut).astype(int)

    T = merged["PFS"]
    E = merged["event"]

    km = KaplanMeierFitter()

    plt.figure(figsize=(6,5))
    for group in [0,1]:
        ix = merged["group"] == group
        km.fit(T[ix], E[ix], label=f"{name} {'high' if group==1 else 'low'}")
        km.plot(ci_show=False)

    res = logrank_test(
        T[merged.group==0], T[merged.group==1],
        E[merged.group==0], E[merged.group==1]
    )

    plt.title(f"PFS by {name} load (p={res.p_value:.3e})")
    plt.xlabel("Time")
    plt.ylabel("Survival probability")
    plt.grid(alpha=0.3)
    plt.show()

    print(f"{name} log-rank p-value:", res.p_value)

km_plot("cluster0_load", "Cluster 0")
km_plot("cluster1_load", "Cluster 1")


# ===============================================================
# 6) Recurrence(0/1) vs cluster load 분포 비교 (Wilcoxon)
# ===============================================================

for c in ["cluster0_load", "cluster1_load"]:
    rec = merged[merged.event == 1][c]
    nonrec = merged[merged.event == 0][c]
    stat, p = ranksums(rec, nonrec)
    print(f"{c}: recurrence difference Wilcoxon p={p:.3e}")


# ===============================================================
# 7) Logistic regression (optional)
# ===============================================================

logreg_df = merged[["cluster0_load", "cluster1_load", "event"]].copy()

# simple logistic regression using statsmodels
import statsmodels.api as sm
X = logreg_df[["cluster0_load", "cluster1_load"]]
X = sm.add_constant(X)
y = logreg_df["event"]

model = sm.Logit(y, X).fit()
print(model.summary())

#%%
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

# merged: contains columns ["cluster0_load", "cluster1_load", "event"]
X = merged[["cluster0_load", "cluster1_load"]].values
y = merged["event"].values

# Add intercept
X2 = np.column_stack((np.ones(X.shape[0]), X))

# -------------------------------------------------------
# Logistic function
# -------------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# -------------------------------------------------------
# Negative log-likelihood function
# -------------------------------------------------------
def neg_log_likelihood(beta, X, y):
    z = X @ beta
    p = sigmoid(z)
    eps = 1e-12
    return -np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

# -------------------------------------------------------
# Fit logistic regression via MLE
# -------------------------------------------------------
beta_init = np.zeros(X2.shape[1])
result = minimize(neg_log_likelihood, beta_init, args=(X2, y), method='BFGS')

beta_hat = result.x
cov_matrix = result.hess_inv  # inverse Hessian = covariance matrix
se = np.sqrt(np.diag(cov_matrix))  # standard errors

# -------------------------------------------------------
# z-statistic and p-values
# -------------------------------------------------------
z_stats = beta_hat / se
p_values = 2 * (1 - norm.cdf(np.abs(z_stats)))

# -------------------------------------------------------
# Output summary
# -------------------------------------------------------
param_names = ["Intercept", "cluster0_load", "cluster1_load"]
summary = pd.DataFrame({
    "coef": beta_hat,
    "std_err": se,
    "z": z_stats,
    "p_value": p_values
}, index=param_names)

print(summary)

#%%
# ===============================================================
# 1) PreTU 기반 cluster load 계산
# ===============================================================

pre_AR_matrix = ARpre_major.loc[df_ar.index]  # transcript × AR patients

cluster0_preload = pre_AR_matrix.loc[cluster0_genes].mean(axis=0)
cluster1_preload = pre_AR_matrix.loc[cluster1_genes].mean(axis=0)

cluster_preload_df = pd.DataFrame({
    "cluster0_preload": cluster0_preload,
    "cluster1_preload": cluster1_preload
})
cluster_preload_df.index.name = "sample"

# ===============================================================
# 2) merge with clinical data
# ===============================================================
clin = sampleinfo.copy()
clin = clin.iloc[::2,:]
clin = clin.loc[cluster_load_df.index]
merged_pre = pd.concat([cluster_preload_df, clin], axis=1)
merged_pre.drop_duplicates(inplace=True)

merged_pre["PFS"] = merged_pre["PFI"]
merged_pre["event"] = merged_pre["survival"]

# ===============================================================
# 3) survival analysis
# ===============================================================
def km_plot(df, feature, label):
    from lifelines import KaplanMeierFitter
    import matplotlib.pyplot as plt

    km = KaplanMeierFitter()

    median_val = df[feature].median()

    high = df[df[feature] >= median_val]
    low  = df[df[feature] <  median_val]

    plt.figure(figsize=(6,5))

    km.fit(high["OS"], event_observed=high["event"], label=f"{label} High")
    ax = km.plot(ci_show=False)

    km.fit(low["OS"], event_observed=low["event"], label=f"{label} Low")
    km.plot(ci_show=False, ax=ax)

    plt.title(f"{label} — Survival curve")
    plt.xlabel("Overall Survival (days)")
    plt.ylabel("Survival probability")
    plt.grid(alpha=0.3)
    plt.show()


km_plot(merged_pre, "cluster0_preload", "Cluster0_preLoad")
km_plot(merged_pre, "cluster1_preload", "Cluster1_preLoad")


# ===============================================================
# 4) recurrence difference
# ===============================================================

for c in ["cluster0_preload", "cluster1_preload"]:
    rec  = merged_pre[merged_pre.event==1][c]
    nonr = merged_pre[merged_pre.event==0][c]
    stat, p = ranksums(rec, nonr)
    print(f"{c} recurrence Wilcoxon p={p:.3e}")

# ===============================================================
# 5) logistic regression (scipy version)
# ===============================================================

X = merged_pre[["cluster0_preload", "cluster1_preload"]].values
y = merged_pre["event"].values

X2 = np.column_stack([np.ones(X.shape[0]), X])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def negLL(beta, X, y):
    z = X @ beta
    p = sigmoid(z)
    eps = 1e-9
    return -np.sum(y * np.log(p+eps) + (1-y)*np.log(1-p+eps))

beta_init = np.zeros(X2.shape[1])
res = minimize(negLL, beta_init, args=(X2,y), method='BFGS')

beta = res.x
cov  = res.hess_inv
se   = np.sqrt(np.diag(cov))

z = beta / se
p = 2*(1-norm.cdf(np.abs(z)))

summary_pre = pd.DataFrame({
    "coef": beta,
    "std_err": se,
    "z": z,
    "p_value": p
}, index=["Intercept","cluster0_preload","cluster1_preload"])

print(summary_pre)

from lifelines import CoxPHFitter

df = merged_pre.copy()

# Cox model에 넣을 칼럼만 선택
cox_df = df[["OS", "event", "cluster0_preload", "cluster1_preload"]].dropna()

cph = CoxPHFitter()
cph.fit(cox_df, duration_col="OS", event_col="event")

cph.print_summary()   # coef, HR, p-values, CI 다 나옴


# %%
