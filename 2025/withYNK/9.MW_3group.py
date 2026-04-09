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

# %% ###^^ cohort preprocessing ####################################

tpm_df = pd.read_csv("/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_TPM.txt", sep="\t", index_col=0)
gene_df = pd.read_csv("/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_gene_TPM.txt", sep="\t", index_col=0)
tpm_df = tpm_df.iloc[:-2,:]
#tpm_df = tpm_df.drop(columns=['target_gene'])  # gene_name 컬럼 제거
tu_df = pd.read_csv("/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_TU.txt", sep="\t", index_col=0)
tu_df = tu_df.iloc[:-2,:]
clin_df = pd.read_csv("/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/112_PARPi_clinicalinfo.txt", sep="\t", index_col=0)
BRCAinfo = pd.read_excel("/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/BRCAinfo.xls", index_col=0, header=1)
BRCAinfo.index = BRCAinfo['GCgenome']
BRCAinfo = BRCAinfo[['gBRCA1','gBRCA2','tBRCA1','tBRCA2',]]
clin_df = clin_df.join(BRCAinfo, how='left')

nmdlist = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/TranSuite/merged/merged_TranSuite_output/merged_transfeat/merged_transfeat_NMD_features.csv', sep=',')
nmdlist = set(nmdlist['Transcript_ID'].to_list())
majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorminor['genename'] = majorminor['gene_ENST'].str.split("-",n=1).str[-1]
majorminor['NMD'] = majorminor['transcriptid'].apply(lambda x: 'NMD' if x in nmdlist else 'non-NMD')
majorminor.loc[(majorminor['type'] == 'major') & (majorminor['NMD'] == 'NMD'),'type'] = 'minor'

majorlist = majorminor[majorminor['type']=='major']['gene_ENST'].to_list()
minorlist = majorminor[majorminor['type']=='minor']['gene_ENST'].to_list()

clin_df['firstline'] = clin_df['line'].apply(lambda x: 1 if x=='1L' else 0)

tpm_df_sev = tpm_df[clin_df.index]
tu_df_sev = tu_df[clin_df.index]
gene_df_sev = gene_df[clin_df.index]
clin_df_sev = clin_df[['response','BRCAmt','drug','recur','PFS','firstline','setting']]
clin_df_sev['cohort'] = 'SEV'

genesymbol =  pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM_symbol.txt', sep='\t', index_col=0)
genesymbol = pd.DataFrame(genesymbol.iloc[:,-1])
genesymbol.columns = ['genesymbol']

vallist = list(clin_df_sev.index)
gene_df_sev = gene_df_sev.loc[genesymbol.index,vallist]
gene_df_sev.index = genesymbol.loc[gene_df_sev.index,'genesymbol']

tpm_df = pd.read_csv("/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/83_POLO_transcript_exp.txt", sep="\t", index_col=0)
gene_df = pd.read_csv("/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/83_POLO_gene_exp.txt", sep="\t", index_col=0)
tpm_df = tpm_df.drop(columns=['target_gene'])  # gene_name 컬럼 제거
tu_df = pd.read_csv("/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/83_POLO_transcript_usage.txt", sep="\t", index_col=0)
clin_df = pd.read_csv("/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/83_POLO_clinicalinfo.txt", sep="\t", index_col=0)
clin_df['BRCAmt'] = clin_df['BRCAmut']
clin_df['drug'] = 'Niraparib'
clin_df['setting'] = clin_df['OM/OS']
clin_df_snu = clin_df[['response','BRCAmt','drug','recur','PFS','setting']]
clin_df_snu['firstline'] = 1
clin_df_snu['cohort'] = 'SNU'

tpm_df_snu = tpm_df[clin_df_snu.index]
tu_df_snu = tu_df[clin_df_snu.index]
gene_df_snu = gene_df[clin_df_snu.index]

genesymbol =  pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM_symbol.txt', sep='\t', index_col=0)
genesymbol = pd.DataFrame(genesymbol.iloc[:,-1])
genesymbol.columns = ['genesymbol']

vallist = list(clin_df_snu.index)
gene_df_snu = gene_df_snu.loc[genesymbol.index,vallist]
gene_df_snu.index = genesymbol.loc[gene_df_snu.index,'genesymbol']

final_gene_exp = pd.merge(gene_df_sev, gene_df_snu, left_index=True, right_index=True, how='inner')
final_trans_exp = pd.merge(tpm_df_sev, tpm_df_snu, left_index=True, right_index=True, how='inner')
final_trans_usage = pd.merge(tu_df_sev, tu_df_snu, left_index=True, right_index=True, how='inner')
final_clin = pd.concat([clin_df_sev, clin_df_snu], axis=0)
final_clin['BRCAmt'] = final_clin['BRCAmt'].astype('int')
final_clin['firstline'] = final_clin['BRCAmt'].astype('int')

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

dnarepairgenes = {gene for gene_list in ddr_coregenelist.values() for gene in gene_list}
dnarepairgenes = list(dnarepairgenes)
    
final_gene_exp = final_gene_exp.loc[final_gene_exp.index.intersection(dnarepairgenes), :]
final_trans_exp = final_trans_exp.loc[final_trans_exp.index.str.contains('|'.join(dnarepairgenes)), :]
final_trans_usage = final_trans_usage.loc[final_trans_usage.index.str.contains('|'.join(dnarepairgenes)), :]

# %%
final_trans_usage = final_trans_usage.astype('float')
final_gene_exp = final_gene_exp.astype('float')
sal_list = final_clin[final_clin['setting']=='salvage'].index.to_list()
main_BRCAmt_list = final_clin[(final_clin['setting']!='salvage') & (final_clin['BRCAmt']==1)].index.to_list()
main_BRCAwt_list = final_clin[(final_clin['setting']!='salvage') & (final_clin['BRCAmt']==0)].index.to_list()
polo_list = final_clin[final_clin['cohort']=='SNU'].index.to_list()

sal_deg = ['MDC1']
main_mt_deg = ['APEX1', 'BRCA2', 'EME1', 'LIG4', 'XRCC4']
main_wt_deg = ['ATM', 'CUL5', 'POLM', 'RAD51', 'TP53BP1']


sal_df = final_trans_usage.loc[~final_trans_usage.index.str.contains('|'.join(sal_deg)),sal_list]
main_BRCAmt_df = final_trans_usage.loc[~final_trans_usage.index.str.contains('|'.join(main_mt_deg)),main_BRCAmt_list]
main_BRCAwt_df = final_trans_usage.loc[~final_trans_usage.index.str.contains('|'.join(main_wt_deg)),main_BRCAwt_list]
polo_df = final_trans_usage[polo_list]
# sal_df = final_gene_exp[sal_list]
# main_BRCAmt_df = final_gene_exp[main_BRCAmt_list]
# main_BRCAwt_df = final_gene_exp[main_BRCAwt_list]

#%%
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

# --- 1️⃣ 공통 함수 ---

def mw_test(df, clin_df):
    """df: transcript x sample, clin_df: includes 'response' (1/0)"""
    # 공통 sample만 사용
    common_samples = df.columns.intersection(clin_df.index)
    df = df[common_samples]
    clin_df = clin_df.loc[common_samples]

    res = []
    for tx in df.index:
        group1 = df.loc[tx, clin_df[clin_df['response'] == 1].index].dropna()
        group0 = df.loc[tx, clin_df[clin_df['response'] == 0].index].dropna()
        if len(group1) >= 3 and len(group0) >= 3:
            stat, p = mannwhitneyu(group1, group0, alternative='two-sided')
            res.append((tx, stat, p))
    if not res:
        return pd.DataFrame(columns=['transcript', 'stat', 'pval', 'pval_adj'])
    res_df = pd.DataFrame(res, columns=['transcript', 'stat', 'pval'])
    res_df['pval_adj'] = res_df['pval'] * len(res_df)
    res_df = res_df.sort_values('pval')
    return res_df


def normalize(df):
    """z-score normalize per transcript"""
    return (df - df.mean(axis=1).values.reshape(-1,1)) / df.std(axis=1).values.reshape(-1,1)


def plot_heatmap(df, title, clin_df):
    """좌: response=1 / 우: response=0 순서로 샘플 정렬"""
    common_samples = df.columns.intersection(clin_df.index)
    df = df[common_samples]
    clin_df = clin_df.loc[common_samples]

    order_pos = clin_df[clin_df['response'] == 1].index
    order_neg = clin_df[clin_df['response'] == 0].index
    ordered = [c for c in order_pos.tolist() + order_neg.tolist() if c in df.columns]

    df = df.loc[:, ordered]
    sns.clustermap(df, cmap='vlag', col_cluster=False, xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.show()


# --- 2️⃣ MW test 각 그룹 수행 ---

res_sal = mw_test(sal_df, final_clin)
res_mt  = mw_test(main_BRCAmt_df, final_clin)
res_wt  = mw_test(main_BRCAwt_df, final_clin)
res_polo = mw_test(polo_df, final_clin)

sig_sal = set(res_sal[res_sal['pval'] < 0.05]['transcript'])
sig_mt  = set(res_mt[res_mt['pval'] < 0.05]['transcript'])
sig_wt  = set(res_wt[res_wt['pval'] < 0.05]['transcript'])
sig_polo = set(res_polo[res_polo['pval'] < 0.05]['transcript'])

# --- 3️⃣ 세 그룹 간 Venn diagram ---

# plt.figure(figsize=(5,5))
# venn3([sig_sal, sig_mt, sig_wt],
#       set_labels=('Salvage', 'Maintenance BRCAmt', 'Maintenance BRCAwt'))
# plt.title('Significant transcripts (p < 0.05)')
# plt.show()



# %%
def plot_split_heatmap(df, title, clin_df):
    common_samples = df.columns.intersection(clin_df.index)
    df = df[common_samples]
    clin_df = clin_df.loc[common_samples]

    pos_samples = clin_df[clin_df['response'] == 1].index
    neg_samples = clin_df[clin_df['response'] == 0].index
    ordered = [c for c in pos_samples.tolist() + neg_samples.tolist() if c in df.columns]

    df = df.loc[:, ordered]

    plt.figure(figsize=(8,5))
    sns.heatmap(
        df, cmap='RdBu_r', center=0,
        vmin=-2, vmax=2,  # contrast 강화
        cbar_kws={'label': 'z-score'},
        xticklabels=False,
        yticklabels=False
    )
    plt.axvline(len(pos_samples), color='black', lw=2)
    plt.title(title)
    plt.xlabel(f'Responder ({len(pos_samples)}) | Nonresponder ({len(neg_samples)})')
    plt.ylabel('Transcripts')
    plt.tight_layout()
    plt.show()



def plot_sig_heatmaps(df, sig_set, clin_df, cohort_name):
    sig_major = list(sig_set.intersection(majorlist))
    sig_minor = list(sig_set.intersection(minorlist))

    if len(sig_major) > 2:
        major_norm = normalize(df.loc[df.index.intersection(sig_major)])
        plot_split_heatmap(major_norm, f'{cohort_name} - Significant Major transcripts', clin_df)
    else:
        print(f'⚠️ {cohort_name}: Not enough significant major transcripts ({len(sig_major)})')

    if len(sig_minor) > 2:
        minor_norm = normalize(df.loc[df.index.intersection(sig_minor)])
        plot_split_heatmap(minor_norm, f'{cohort_name} - Significant Minor transcripts', clin_df)
    else:
        print(f'⚠️ {cohort_name}: Not enough significant minor transcripts ({len(sig_minor)})')


# plot_sig_heatmaps(main_BRCAwt_df, sig_wt,final_clin, 'Maintenance BRCAwt')
# plot_sig_heatmaps(main_BRCAmt_df, sig_mt,final_clin, 'Maintenance BRCAmt')
# plot_sig_heatmaps(sal_df, sig_sal,final_clin, 'Salvage')
plot_sig_heatmaps(polo_df, sig_polo,final_clin, 'POLO')
# %%
tpm_df = pd.merge(tpm_df_sev, tpm_df_snu, left_index=True, right_index=True, how='inner')
tpm_df = tpm_df.astype('float')
gene_map = pd.Series([x.split("-",1)[-1] for x in tpm_df.index], index=tpm_df.index)

# -----------------------------
# 2. Make gene-level matrices
# -----------------------------
# (i) Total TPM (all transcripts)
expr_total = tpm_df.groupby(gene_map).sum()

# (ii) Major TPM (filter first → then sum)
expr_major = tpm_df.loc[tpm_df.index.isin(majorlist)] #*minorlist
gene_map_major = gene_map.loc[expr_major.index]
expr_major_gene = expr_major.groupby(gene_map_major).sum()

# log transform
expr_total_log = np.log2(expr_total + 1)
expr_major_log = np.log2(expr_major_gene + 1)

clin_df = final_clin[final_clin['setting']=='maintenance']

expr_major_log = expr_major_log[clin_df.index]
expr_total_log = expr_total_log[clin_df.index]

# %%
import gseapy as gp
res_total = gp.ssgsea(
    data=expr_total_log,
    gene_sets="MSigDB_Hallmark_2020", #GO_Biological_Process_2021, MSigDB_Hallmark_2020
    outdir=None,
    sample_norm_method="rank",
    permutation_num=0
)
df_total = res_total.res2d
score_total = df_total.pivot(index="Term", columns="Name", values="ES")

res_major = gp.ssgsea(
    data=expr_major_log,
    gene_sets="MSigDB_Hallmark_2020", #GO_Biological_Process_2021, MSigDB_Hallmark_2020
    outdir=None,
    sample_norm_method="rank",
    permutation_num=0
)
df_major = res_major.res2d
score_major = df_major.pivot(index="Term", columns="Name", values="ES")

# %%
from lifelines import CoxPHFitter
import numpy as np

results = []


###^^^# score_total or score_major 중 선택
score_df = score_major # 예시: major TPM 기반
score_df = score_df[clin_df[clin_df['cohort']=='SEV'].index]

for term in score_df.index:
    # pathway score 불러오기
    vals = score_df.loc[term]

    # median split
    group = np.where(vals > vals.median(), "High", "Low")

    # Cox 모델용 데이터프레임
    tmp = clin_df[["PFS", "recur",]].copy() # "firstline","BRCAmt","drug"
    tmp = tmp.loc[score_df.columns,:]
    tmp["group"] = (group == "High").astype(int)  # High=1, Low=0

    # 결측치 제거
    #tmp = tmp.dropna(subset=["PFS", "recur","firstline", "group","BRCAmt","drug"]) # "firstline", "group","BRCAmt","drug"
    #tmp = pd.get_dummies(tmp, columns=["drug"], drop_first=True)
    

    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(tmp, duration_col="PFS", event_col="recur")

        hr = np.exp(cph.params_["group"])
        ci_low, ci_high = np.exp(cph.confidence_intervals_.loc["group"]).tolist()
        pval = cph.summary.loc["group", "p"]

        results.append({
            "Pathway": term,
            "HR": hr,
            "CI_low": ci_low,
            "CI_high": ci_high,
            "pval": pval
        })
    except Exception as e:
        print(f"Skipped {term} due to error: {e}")

# 결과 DataFrame
res_df = pd.DataFrame(results)

# 필터링: High 위험 (HR>1) + p<0.05
sig_df_high = res_df[(res_df["HR"] > 1) & (res_df["pval"] < 0.05)].sort_values("pval")
sig_df_low = res_df[(res_df["HR"] < 1) & (res_df["pval"] < 0.05)].sort_values("pval")


##^^ sig HR barplot #######
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

df_low = sig_df_low.nsmallest(20, "pval") 
df_high = sig_df_high.nsmallest(20, "pval") 
df_low["log2HR"] = np.log2(df_low["HR"]) 
df_high["log2HR"] = np.log2(df_high["HR"]) 
df_low["neglog10p"] = -np.log10(df_low["pval"]) 
df_high["neglog10p"] = -np.log10(df_high["pval"])

# 정렬: 각각 HR 내림차순
df_high = df_high.sort_values("HR", ascending=True)
df_low = df_low.sort_values("HR", ascending=False)

# y축 Pathway는 high 먼저 + low 나중
pathways = list(df_low["Pathway"]) + list(df_high["Pathway"])
y_pos_low = np.arange(len(df_low))
y_pos_high = np.arange(len(df_high)) + len(df_low)

fig, ax = plt.subplots(figsize=(10, 8))
ax.tick_params(axis="y", which="major", pad=4)
# 색상맵
cmap_high = plt.cm.Reds
cmap_low = plt.cm.Blues
norm_high = mpl.colors.Normalize(vmin=df_high["neglog10p"].min(),
                                 vmax=df_high["neglog10p"].max())
norm_low = mpl.colors.Normalize(vmin=df_low["neglog10p"].min(),
                                vmax=df_low["neglog10p"].max())

# bar plot
ax.barh(y_pos_high, df_high["log2HR"],
        color=cmap_high(norm_high(df_high["neglog10p"])), label="High-risk (HR>1)")
ax.barh(y_pos_low, df_low["log2HR"],
        color=cmap_low(norm_low(df_low["neglog10p"])), label="Protective (HR<1)")

# y축 라벨
ax.set_yticks(np.arange(len(pathways)))
ax.set_yticklabels(pathways, fontsize=10)

# 기준선
ax.set_xlabel("log2(HR)")

# 테두리: 좌/하만 남김
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ylim 꽉 차게
ax.set_ylim(-0.5, len(pathways)-0.5)

sm_high = mpl.cm.ScalarMappable(cmap=cmap_high, norm=norm_high)
sm_low = mpl.cm.ScalarMappable(cmap=cmap_low, norm=norm_low)
sm_high.set_array([])
sm_low.set_array([])

# High-risk colorbar (위쪽, 짧게)
cax_high = fig.add_axes([0.92, 0.60, 0.02, 0.25])
cbar_high = fig.colorbar(sm_high, cax=cax_high)
cbar_high.ax.tick_params(labelsize=10)

# Protective colorbar (아래쪽, 짧게)
cax_low = fig.add_axes([0.92, 0.20, 0.02, 0.25])
cbar_low = fig.colorbar(sm_low, cax=cax_low)
cbar_low.ax.tick_params(labelsize=10)


plt.tight_layout(rect=[0, 0, 0.9, 1])  # 오른쪽 colorbar 공간 확보
plt.subplots_adjust(left=0.55)
sns.despine()
#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLOanalysis/figures/SEV_GOBPsig_pathway_barplot_majortrans.pdf", dpi=300,bbox_inches='tight')
plt.show()

# %%
