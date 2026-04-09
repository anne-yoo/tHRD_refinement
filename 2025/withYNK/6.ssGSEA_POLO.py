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
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, statistics
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Data 준비
# -----------------------------

# tpm_df, tu_df: transcript x sample
# clin_df: sample x clinical info
# row index = sample ID, column order = colnames of TPM/TU

tpm_df = pd.read_csv("/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/merged_83_transcript_TPM.txt", sep="\t", index_col=0)
tpm_df = tpm_df.drop(columns=['gene_name'])  # gene_name 컬럼 제거
tu_df = pd.read_csv("/home/jiye/jiye/copycomparison/GENCODEquant/POLO/83_transcript_TU.txt", sep="\t", index_col=0)
clin_df = pd.read_csv("/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/83_POLO_clinicalinfo.txt", sep="\t", index_col=0)

tpm_df = tpm_df[clin_df.index]
tu_df = tu_df[clin_df.index]

# majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/gencode_majorminorlist.txt', sep='\t')
# majorlist = majorminor.loc[majorminor['type']=='major','Transcript-Gene'].to_list()
# minorlist = tu_df.index.difference(majorlist).to_list()

majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/merged_83_cov5_majorminorlist.txt', sep='\t')
majorlist = majorminor[majorminor['type']=='major']['Transcript-Gene'].to_list()
minorlist = majorminor[majorminor['type']=='minor']['Transcript-Gene'].to_list()

sqanti = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/sqantioutput/sqanti_classification.txt', sep='\t')
sqanti.dropna(axis=1, how='all', inplace=True)

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

group1_list = majorminor[majorminor['transcriptid'].isin(group1)]['Transcript-Gene'].to_list()
group2_list = majorminor[majorminor['transcriptid'].isin(group2)]['Transcript-Gene'].to_list()
group3_list = majorminor[majorminor['transcriptid'].isin(group3)]['Transcript-Gene'].to_list()

#%%
###^^ gHRD -> recur / coxPH KM plot ###########

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# 색상 정의
colors = {
    "gHRD+": "#DA4343",   # red
    "gHRD−": "#3396D3"    # blue
}

# ----------------------------
# 예시 clin_df (함의 데이터 포맷)
# ----------------------------
# clin_df = pd.read_csv("clinical_data.tsv", sep="\t")

# gHRD status 라벨
clin_df["gHRD_status"] = clin_df["gHRD"].map({1: "gHRD+", 0: "gHRD−"})

# ====================================================
# ① gHRD vs 재발 여부 (barplot)
# ====================================================
plt.figure(figsize=(4,4))
recur_rate = (
    clin_df.groupby("gHRD_status")["recur"]
    .mean()
    .reset_index()
    .rename(columns={"recur": "Recurrence_Rate"})
)

sns.barplot(
    data=recur_rate,
    x="gHRD_status",
    y="Recurrence_Rate",
    palette=colors
)
plt.title("Recurrence rate by gHRD status", fontsize=12, fontweight="bold")
plt.ylabel("Proportion of recurrence")
plt.xlabel("")
plt.tight_layout()
sns.despine()
#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/gHRD_recur.pdf", dpi=300)
plt.show()

# ====================================================
# ② Kaplan–Meier curve + CoxPH + p-value 표시
# ====================================================
kmf = KaplanMeierFitter()

plt.figure(figsize=(6,5))
for status, color in colors.items():
    mask = clin_df["gHRD_status"] == status
    T = clin_df.loc[mask, "PFS"]
    E = clin_df.loc[mask, "recur"]  # 1=event, 0=censored
    
    kmf.fit(T, E, label=status)
    kmf.plot_survival_function(
        ci_show=False,
        color=color,
        linewidth=2,
        show_censors=True,   # ✅ censor 자동 표시 (+)
    )


plt.title("Progression-free survival by gHRD status", fontsize=12, fontweight="bold")
plt.xlabel("Days")
plt.ylabel("Survival probability")

# Log-rank test
group_pos = clin_df[clin_df["gHRD_status"] == "gHRD+"]
group_neg = clin_df[clin_df["gHRD_status"] == "gHRD−"]
lr = logrank_test(group_pos["PFS"], group_neg["PFS"],
                  event_observed_A=group_pos["recur"],
                  event_observed_B=group_neg["recur"])

# CoxPH for HR
cox_df = clin_df[["PFS", "recur", "gHRD"]].copy()
cph = CoxPHFitter()
cph.fit(cox_df, duration_col="PFS", event_col="recur")
summary = cph.summary.loc["gHRD"]
hr = summary["exp(coef)"]
ci_low = summary["exp(coef) lower 95%"]
ci_high = summary["exp(coef) upper 95%"]
pval = summary["p"]

# Plot annotation
plt.text(
    0.55 * clin_df["PFS"].max(), 0.8,
    f"HR = {hr:.2f} (95% CI {ci_low:.2f}–{ci_high:.2f})\n"
    f"p = {pval:.3e}",
    fontsize=11,
    bbox=dict(facecolor="white", alpha=0.7, boxstyle="round")
)

plt.legend(title="gHRD status", frameon=False)
plt.tight_layout()
sns.despine()
#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/gHRD_KMplot.pdf", dpi=300)
plt.show()

print(f"Log-rank p-value: {lr.p_value:.4f}")
print(cph.summary)

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gseapy as gp
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
# -----------------------------
# 0. Input
# -----------------------------
# tpm_df: transcript x sample TPM dataframe
# tu_df: transcript x sample TU dataframe (동일 index, col)
# clin_df: sample x clinical info dataframe
#   columns: PFS, recur, OS_time, OS_event
# majorlist: list of transcript IDs

# ensure sample order alignment
assert all(tpm_df.columns == tu_df.columns)
assert all(tpm_df.columns == clin_df.index)

# -----------------------------
# 1. Transcript → Gene mapping
# -----------------------------
# transcript IDs look like "ENSTxxxx-A1BG"
gene_map = pd.Series([x.split("-")[-1] for x in tpm_df.index], index=tpm_df.index)

# -----------------------------
# 2. Make gene-level matrices
# -----------------------------
# (i) Total TPM (all transcripts)
expr_total = tpm_df.groupby(gene_map).sum()

# (ii) Major TPM (filter first → then sum)
expr_major = tpm_df.loc[tpm_df.index.isin(group1_list)] #*minorlist
gene_map_major = gene_map.loc[expr_major.index]
expr_major_gene = expr_major.groupby(gene_map_major).sum()

# log transform
expr_total_log = np.log2(expr_total + 1)
expr_major_log = np.log2(expr_major_gene + 1)

#%%
# # -----------------------------
# # 3. Batch correction & PCA (if combining cohorts later)
# # -----------------------------
# # 여기서는 Cohort B만이라서 batch correction 불필요
# # PCA 예시 (gene-level TPM 사용)
# scaler = StandardScaler()
# X = scaler.fit_transform(expr_total_log.T)  # sample x gene
# pca = PCA(n_components=2)
# pc = pca.fit_transform(X)

# plt.scatter(pc[:,0], pc[:,1])
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.title("PCA (Cohort B)")
# plt.show()

# -----------------------------
# 4. ssGSEA
# -----------------------------
res_total = gp.ssgsea(
    data=expr_total_log,
    gene_sets="GO_Biological_Process_2021", #GO_Biological_Process_2021, MSigDB_Hallmark_2020
    outdir=None,
    sample_norm_method="rank",
    permutation_num=0
)
df_total = res_total.res2d
score_total = df_total.pivot(index="Term", columns="Name", values="ES")

res_major = gp.ssgsea(
    data=expr_major_log,
    gene_sets="GO_Biological_Process_2021",
    outdir=None,
    sample_norm_method="rank",
    permutation_num=0
)
df_major = res_major.res2d
score_major = df_major.pivot(index="Term", columns="Name", values="ES")

#%%
# -----------------------------
# 5. Survival analysis (DNA Repair example)
# -----------------------------
dna_total = score_total.loc["double-strand break repair via homologous recombination (GO:0000724)"] #double-strand break repair via homologous recombination (GO:0000724) #DNA Repair
dna_major = score_major.loc["double-strand break repair via homologous recombination (GO:0000724)"]

clin_df["dna_total_group"] = np.where(dna_total > dna_total.median(), "High", "Low")
clin_df["dna_major_group"] = np.where(dna_major > dna_major.median(), "High", "Low")

kmf = KaplanMeierFitter()


for group in ["dna_total_group", "dna_major_group"]:
    plt.figure(figsize=(6,5))
    ax = plt.subplot(111)

    colors = {
        "High": "#DA4343",  # 주황색
        "Low": "#3396D3",
        # 필요하면 더 추가
    }

    for label in clin_df[group].unique():
        idx = clin_df[group] == label
        kmf.fit(clin_df.loc[idx, "PFS"], clin_df.loc[idx, "recur"], label=label)
        kmf.plot_survival_function(
            ax=ax, ci_show=True, color=colors.get(label, "gray")
        )


    # log-rank test
    idx_high = clin_df[group] == "High"
    res = logrank_test(
        clin_df.loc[idx_high, "PFS"],
        clin_df.loc[~idx_high, "PFS"],
        clin_df.loc[idx_high, "recur"],
        clin_df.loc[~idx_high, "recur"]
    )

    # Cox regression (HR, CI)
    tmp = clin_df[[ "PFS", "recur", group ]].copy()
    tmp[group] = (tmp[group] == "High").astype(int)
    cph = CoxPHFitter()
    cph.fit(tmp, duration_col="PFS", event_col="recur")
    hr = np.exp(cph.params_)[0]
    ci = cph.confidence_intervals_.iloc[0].tolist()

    # plot title & annotation
    plt.title(f"DNA Repair") #G2-M Checkpoint
    plt.text(0.05, 0.05, f"Log-rank p = {res.p_value:.3}\nHR = {hr:.2f} ({ci[0]:.2f} - {ci[1]:.2f})",
             transform=ax.transAxes, fontsize=10)
    plt.xlabel('time')
    plt.ylabel('survival probability')
    sns.despine()
    #plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLOanalysis/figures/sev_G2MCheckpoint_KMplot_"+group+".pdf", dpi=300)
    plt.show()

#%%
###^^^ 전체 pathway CoxPH #######
from lifelines import CoxPHFitter
import numpy as np

results = []


###^^^# score_total or score_major 중 선택
score_df = score_major # 예시: major TPM 기반

for term in score_df.index:
    # pathway score 불러오기
    vals = score_df.loc[term]

    # median split
    group = np.where(vals > vals.median(), "High", "Low")

    # Cox 모델용 데이터프레임
    tmp = clin_df[["PFS", "recur"]].copy()
    tmp["group"] = (group == "High").astype(int)  # High=1, Low=0

    try:
        cph = CoxPHFitter()
        cph.fit(tmp, duration_col="PFS", event_col="recur")
        hr = np.exp(cph.params_)[0]
        ci_low, ci_high = np.exp(cph.confidence_intervals_.iloc[0]).tolist()
        pval = cph.summary["p"].values[0]

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

#%%
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
#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLOanalysis/figures/POLO_GOBPsig_pathway_barplot_majortrans.pdf", dpi=300,bbox_inches='tight')
plt.show()

#%%
####^^ only high HR for SNU #########################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# --- df_high 준비는 기존과 동일하다고 가정 ---
df_high = sig_df_high.nsmallest(20, "pval").copy()
df_high["log2HR"] = np.log2(df_high["HR"])
df_high["neglog10p"] = -np.log10(df_high["pval"])
df_high = df_high.sort_values("HR", ascending=False)

# --- X축 설정 ---
pathways = list(df_high["Pathway"])
x_pos_high = np.arange(len(df_high))

# --- Figure 설정 ---
fig, ax = plt.subplots(figsize=(12, 6))

# 색상맵 (Reds)
cmap_high = plt.cm.Reds
norm_high = mpl.colors.Normalize(
    vmin=df_high["neglog10p"].min(),
    vmax=df_high["neglog10p"].max()
)

# --- Bar plot ---
ax.bar(
    x_pos_high,
    df_high["log2HR"],
    color=cmap_high(norm_high(df_high["neglog10p"])),
    label="High HR"
)

# --- 축 라벨 및 눈금 설정 ---
ax.set_xticks(x_pos_high)
ax.set_xticklabels(
    pathways,
    fontsize=10,
    rotation=45,
    ha='right',
    rotation_mode='anchor'
)
ax.set_ylabel("log2(HR)")
ax.set_xlim(-0.5, len(pathways) - 0.5)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

# --- Colorbar ---
sm_high = mpl.cm.ScalarMappable(cmap=cmap_high, norm=norm_high)
sm_high.set_array([])
cax_high = fig.add_axes([0.92, 0.2, 0.015, 0.6])
cbar_high = fig.colorbar(sm_high, cax=cax_high)
cbar_high.ax.tick_params(labelsize=10)
cbar_high.set_label("-log10(p)", loc='top', fontsize=10)

# --- 마무리 ---
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.subplots_adjust(bottom=0.3, right=0.9)
sns.despine()
#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/GOBP_HR_barplot_onlyhighHR.pdf", dpi=300,bbox_inches='tight')
plt.show()


#%%
###^^ barplot 가로로 길게 수정 #########3
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- 데이터 준비 (원본 코드와 동일) ---
# (res_df가 이미 생성되었다고 가정)

# 필터링
sig_df_high = res_df[(res_df["HR"] > 1) & (res_df["pval"] < 0.05)].sort_values("pval")
sig_df_low = res_df[(res_df["HR"] < 1) & (res_df["pval"] < 0.05)].sort_values("pval")

# 상위 20개씩 선택
df_low = sig_df_low.nsmallest(20, "pval") 
df_high = sig_df_high.nsmallest(20, "pval") 

# 계산
df_low["log2HR"] = np.log2(df_low["HR"]) 
df_high["log2HR"] = np.log2(df_high["HR"]) 
df_low["neglog10p"] = -np.log10(df_low["pval"]) 
df_high["neglog10p"] = -np.log10(df_high["pval"])

# --- 정렬 순서 변경 ---
# X축에 나열하기 위해 Protective(Low) -> High-risk(High) 순으로 정렬
# (HR이 낮은 것부터 높은 것 순으로)
df_low = df_low.sort_values("HR", ascending=True) # Protective (HR < 1)
df_high = df_high.sort_values("HR", ascending=True) # High-risk (HR > 1)

# X축 Pathway 리스트 (Low -> High 순서)
pathways = list(df_low["Pathway"]) + list(df_high["Pathway"])
x_pos_low = np.arange(len(df_low))
x_pos_high = np.arange(len(df_high)) + len(df_low)

# --- ⬇️ Figure 크기 및 플롯 방식 수정 ⬇️ ---

# 가로로 길게 (e.g., width=18) / 세로는 적당히 (e.g., height=8)
fig, ax = plt.subplots(figsize=(18, 8)) 

ax.tick_params(axis="x", which="major", pad=4)
# 색상맵
cmap_high = plt.cm.Reds
cmap_low = plt.cm.Blues
norm_high = mpl.colors.Normalize(vmin=df_high["neglog10p"].min(),
                                 vmax=df_high["neglog10p"].max())
norm_low = mpl.colors.Normalize(vmin=df_low["neglog10p"].min(),
                                vmax=df_low["neglog10p"].max())

# 세로 bar plot (ax.bar 사용)
ax.bar(x_pos_high, df_high["log2HR"],
       color=cmap_high(norm_high(df_high["neglog10p"])), label="High HR")
ax.bar(x_pos_low, df_low["log2HR"],
      color=cmap_low(norm_low(df_low["neglog10p"])), label="Low HR")

# X축 라벨 (pathway 이름)
ax.set_xticks(np.arange(len(pathways)))
ax.set_xticklabels(
    pathways, 
    fontsize=11, 
    rotation=45,     # 1. 45도 대신 90도로 회전
    ha='right',      # 2. 라벨 정렬 기준을 오른쪽으로 (틱 중앙에 맞춤)
    rotation_mode='anchor' # 3. 정렬 기준점(anchor)을 기준으로 회전
)
# Y축 라벨
ax.set_ylabel("log2(HR)")

# X축 범위 꽉 차게
ax.set_xlim(-0.5, len(pathways)-0.5)

# 0 기준선 (Y축)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

# 테두리
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Colorbar Mappables
sm_high = mpl.cm.ScalarMappable(cmap=cmap_high, norm=norm_high)
sm_low = mpl.cm.ScalarMappable(cmap=cmap_low, norm=norm_low)
sm_high.set_array([])
sm_low.set_array([])

# Colorbar 위치 (플롯 오른쪽에 세로로 배치)
# [left, bottom, width, height]
cax_high = fig.add_axes([0.92, 0.55, 0.015, 0.3]) 
cbar_high = fig.colorbar(sm_high, cax=cax_high)
cbar_high.ax.tick_params(labelsize=10)
cbar_high.set_label("-log10(p)", loc='top', fontsize=10)

cax_low = fig.add_axes([0.92, 0.15, 0.015, 0.3])
cbar_low = fig.colorbar(sm_low, cax=cax_low)
cbar_low.ax.tick_params(labelsize=10)
cbar_low.set_label("-log10(p)", loc='top', fontsize=10)


#plt.tight_layout(rect=[0.05, 0.1, 0.9, 0.9]) # 레이아웃 조절 (하단 라벨 공간, 우측 컬러바 공간 확보)
plt.subplots_adjust(bottom=0.3) # X축 라벨이 잘리지 않도록 하단 여백 확보
sns.despine()

#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/GOBP2021_HR_barplot.pdf", dpi=300,bbox_inches='tight')
plt.show()


#%%
import matplotlib.pyplot as plt
import numpy as np

res_df["logHR"] = np.log2(res_df["HR"])
res_df["neglogP"] = -np.log10(res_df["pval"])

plt.figure(figsize=(8,6))
plt.scatter(res_df["logHR"], res_df["neglogP"], color="grey", alpha=0.6)

sig = (res_df["pval"] < 0.05)
plt.scatter(res_df.loc[sig & (res_df["HR"]>1), "logHR"],
            res_df.loc[sig & (res_df["HR"]>1), "neglogP"], color="red")
plt.scatter(res_df.loc[sig & (res_df["HR"]<1), "logHR"],
            res_df.loc[sig & (res_df["HR"]<1), "neglogP"], color="blue")

plt.axhline(-np.log10(0.05), ls="--", color="black")
plt.axvline(0, ls="--", color="black")
plt.xlabel("log2(HR)")
plt.ylabel("-log10(p)")
plt.title("Pathway-level CoxPH analysis")
plt.show()

#%%
###^^^^ tHRD : scatterplot ###################
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure categorical columns
clin_df["dna_major_group"] = clin_df["dna_major_group"].astype("category")
clin_df["recur"] = clin_df["recur"].astype("category")
clin_df["gHRD"] = clin_df["gHRD"].astype("category")

plt.figure(figsize=(16, 4))

# Plot 1: color by recur
plt.subplot(1, 3, 1)
sns.scatterplot(
    data=clin_df,
    x="tHRD",
    y="PFS",
    hue="recur",
    palette="Set1"
)
plt.title("PFS vs tHRD (color = recur)")

# Plot 2: color by gHRD
plt.subplot(1, 3, 2)
sns.scatterplot(
    data=clin_df,
    x="tHRD",
    y="PFS",
    hue="gHRD",
    palette="Set1"
)
plt.title("gHRD (LOH score)")

# Plot 3: color by dna_major_group
plt.subplot(1, 3, 3)
sns.scatterplot(
    data=clin_df,
    x="tHRD",
    y="PFS",
    hue="dna_major_group",
    palette="Set1"
)
plt.title("DNA repair group")

plt.tight_layout()
plt.show()


#%%
# -----------------------------
# 6. Heatmap (pathway clustering)
# -----------------------------
# 예: total TPM 기반 ssGSEA score heatmap

score_total = score_total.apply(pd.to_numeric, errors="coerce")
score_major = score_major.apply(pd.to_numeric, errors="coerce")


sns.clustermap(score_total, metric="euclidean", method="ward",
               cmap="RdBu_r", z_score=0,
               figsize=(12,10))
plt.suptitle("Pathway activity heatmap (Total TPM)")
plt.show()

# major TPM 기반 heatmap도 가능
sns.clustermap(score_major, metric="euclidean", method="ward",
               cmap="RdBu_r", z_score=0,
               figsize=(12,10))
plt.suptitle("Pathway activity heatmap (Major TPM)")
plt.show()


#%%
#####^^^ tHRD score vs. recur ####################
thrd = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO/RF_result/lane5/predicted_group_proba_based_on_120_OV_model_proba.txt', sep='\t', index_col=1)

clin_df['tHRD'] = thrd[['pred_HRD']]


#%%
####^^ cibersort & immucellai #################

import pandas as pd
import numpy as np
import os
import re # 정규표현식(re)은 필수입니다.
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

cibersort_df = pd.read_csv("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/cibersort/POLO_cibersort.txt", sep="\t", index_col=0)
immucellai_df = pd.read_csv("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/immucellai/POLO_immucellai.txt", sep="\t", index_col=0)
immucellai_df.index = immucellai_df.index.str.replace(r'\.', '-', regex=True)
cibersort_df.index = cibersort_df.index.str.replace(r'\.', '-', regex=True)
cibersort_df.columns = cibersort_df.columns.str.replace(r'\s+', '_', regex=True)
cibersort_df.columns = cibersort_df.columns.str.replace(r'[^\w\-_]', '', regex=True)

# --- 3. 분석 환경 설정 ---
TIME_COL = 'PFS'
EVENT_COL = 'recur'
P_VALUE_THRESHOLD = 1

# 저장 경로
IMMUCELLAI_PATH = "/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/immucellai"
CIBERSORT_PATH = "/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/cibersort"

# 분석에서 제외할 컬럼 목록 (정리된 이름 기준)
EXCLUDE_COLS = ['P_value', 'Correlation', 'RMSE', 'InfiltrationScore'] # . -> _ 변경됨

# --- 4. 핵심 분석 및 플로팅 함수 (이전과 동일) ---

def analyze_and_plot_survival(cell_df, clin_df, output_dir, df_name):
    """
    세포 구성 DataFrame을 받아 CoxPH 분석 및 유의미한 KM 플롯을 저장합니다.
    (이 함수는 이제 DataFrame이 이미 정리되었다고 가정합니다)
    """
    print(f"--- {df_name} 분석 시작 ---")
    
    # 0. 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 분석할 세포 타입 컬럼만 선택
    # (EXCLUDE_COLS에 포함되지 않은 컬럼만 선택)
    cell_types = [col for col in cell_df.columns if col not in EXCLUDE_COLS]
    
    # 2. 임상 데이터와 병합 (공통 샘플만 사용)
    # (cell_df와 clin_df의 인덱스 형식이 'PL-OV-P004'로 일치해야 함)
    merged_df = clin_df[[TIME_COL, EVENT_COL]].join(cell_df[cell_types], how='inner')
    
    if merged_df.empty:
        print(f"경고: {df_name}과 clin_df 간에 일치하는 샘플이 없습니다. 인덱스 형식을 확인하세요.")
        return

    significant_count = 0
    
    # 3. 각 세포 타입에 대해 반복 분석
    for cell_type in cell_types:
        analysis_df = merged_df[[TIME_COL, EVENT_COL, cell_type]].copy().dropna()
        
        # 데이터가 없거나 모두 동일한 값이면 건너뛰기
        if analysis_df.empty or analysis_df[cell_type].nunique() < 2:
            continue
            
        # 4. 중앙값(Median) 기준으로 그룹 분리
        median_val = analysis_df[cell_type].median()
        analysis_df['group'] = (analysis_df[cell_type] > median_val).astype(int) # 1=High, 0=Low
        
        # High/Low 그룹이 모두 존재해야 함
        if analysis_df['group'].nunique() < 2:
            continue
            
        # 5. CoxPH 모델 피팅
        try:
            cph = CoxPHFitter()
            cph.fit(analysis_df[[TIME_COL, EVENT_COL, 'group']], 
                    duration_col=TIME_COL, 
                    event_col=EVENT_COL,
                    formula="group")
            
            pval = cph.summary.loc['group', 'p']
            
            # 6. 유의미한 경우 (p < 0.05) KM 플롯 생성 및 저장
            if pval < P_VALUE_THRESHOLD:
                significant_count += 1
                print(f"  [유의미] {cell_type} (p = {pval:.4f}) ... 플롯 저장 중")
                
                # Log-rank test
                high_data = analysis_df[analysis_df['group'] == 1]
                low_data = analysis_df[analysis_df['group'] == 0]
                lr_results = logrank_test(high_data[TIME_COL], low_data[TIME_COL], 
                                          high_data[EVENT_COL], low_data[EVENT_COL])
                lr_pval = lr_results.p_value

                # CoxPH 결과
                hr = cph.hazard_ratios_['group']
                ci_low, ci_high = np.exp(cph.confidence_intervals_.loc['group']).values
                
                # 플로팅
                plt.figure(figsize=(7, 6))
                ax = plt.gca()
                kmf = KaplanMeierFitter()

                kmf.fit(low_data[TIME_COL], event_observed=low_data[EVENT_COL], label=f"Low (n={len(low_data)})")
                kmf.plot_survival_function(ax=ax, color='blue', ci_show=False)
                
                kmf.fit(high_data[TIME_COL], event_observed=high_data[EVENT_COL], label=f"High (n={len(high_data)})")
                kmf.plot_survival_function(ax=ax, color='red', ci_show=False)
                
                plt.title(f"{df_name}: {cell_type}", fontsize=14)
                plt.xlabel(TIME_COL)
                plt.ylabel("Survival Probability")
                
                # 통계 텍스트 추가
                text_str = (
                            f"CoxPH p = {pval:.3e}\n"
                            f"HR = {hr:.2f} (95% CI {ci_low:.2f}–{ci_high:.2f})")
                plt.text(0.05, 0.05, text_str, transform=ax.transAxes, 
                         bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
                
                # 7. 파일 저장
                # 컬럼명이 이미 정리되었으므로 파일 이름도 깔끔하게 저장됨
                filename = os.path.join(output_dir, f"KM_Plot_{cell_type}_p{pval:.4f}.pdf")
                plt.savefig(filename, bbox_inches='tight')
                plt.close()

        except Exception as e:
            # lifelines에서 수렴 실패 등 오류 발생 시
            print(f"  [오류] {cell_type} 분석 실패: {e}")
            
    print(f"--- {df_name} 분석 완료. 총 {significant_count}개의 유의미한 플롯 저장 ---")


# --- 5. 코드 실행 ---

if 'clin_df' in locals():
    # ImmuCellAI 분석 실행
    if 'immucellai_df' in locals():
        analyze_and_plot_survival(immucellai_df, clin_df, IMMUCELLAI_PATH, "ImmuCellAI")
    
    # CIBERSORT 분석 실행
    if 'cibersort_df' in locals():
        analyze_and_plot_survival(cibersort_df, clin_df, CIBERSORT_PATH, "CIBERSORT")
    
    print("\n=== 모든 분석이 완료되었습니다. ===")
else:
    print("\n!!! 'clin_df'를 로드한 후 이 코드를 실행해주세요. !!!")

# %%
###^ immune cell and ssGSEA inflammation correlation 

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import pandas as pd

# --- 1. 데이터 준비 ---
# (score_major, cibersort_df, immucellai_df가 이미 로드되었다고 가정)

# 1.1. 분석할 세 개의 데이터 시리즈(Series) 추출
try:
    x_series = score_major.loc['Inflammatory Response', :].astype(float)
    y1_series = cibersort_df.loc[:, 'T_cells_CD4_memory_resting']
    y2_series = immucellai_df.loc[:, 'Tfh']
except KeyError as e:
    print(f"!!! 데이터 추출 오류: {e}")
    print("DataFrame에 해당 'Key'가 존재하는지 확인하세요.")
    # 오류 발생 시 중단 (또는 적절한 예외 처리)
    raise e

# 1.2. 세 시리즈를 하나의 DataFrame으로 결합 (인덱스 기준 정렬)
# join='inner' : 세 데이터 모두에 존재하는 공통 샘플만 사용
combined_df = pd.concat([x_series, y1_series, y2_series], 
                        axis=1, 
                        join='inner')

# 1.3. 컬럼 이름 변경 (간결하게)
combined_df.columns = ['Inflammatory_Response', 'T_cells_CD4_memory_resting', 'Tfh']

# 1.4. 결측치(NA) 제거 (필수는 아니지만 안전장치)
combined_df = combined_df.dropna()

print(f"총 {len(combined_df)}개의 공통 샘플로 플롯을 생성합니다.")


# --- 2. 플롯 생성 (1행 2열) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- 플롯 1: Inflammatory Response vs. T_cells_CD4_memory_resting ---
ax1 = axes[0]
sns.regplot(
    data=combined_df,
    x='Inflammatory_Response',
    y='T_cells_CD4_memory_resting',
    ax=ax1,
    scatter_kws={'alpha': 0.7, 's': 20, 'color': 'blue'}, # 점 스타일
    line_kws={'color': 'black', 'linestyle': '--'}      # 선 스타일
)

# Spearman 상관계수 계산
r1, p1 = spearmanr(combined_df['Inflammatory_Response'], combined_df['T_cells_CD4_memory_resting'])

# 제목에 r값, p값 표시
ax1.set_title(f"Spearman r = {r1:.3f} (p = {p1:.2e})", fontsize=14)
ax1.set_xlabel("Inflammatory Response Score", fontsize=12)
ax1.set_ylabel("T_cells_CD4_memory_resting (CIBERSORT)", fontsize=12)

# --- 플롯 2: Inflammatory Response vs. Tfh ---
ax2 = axes[1]
sns.regplot(
    data=combined_df,
    x='Inflammatory_Response',
    y='Tfh',
    ax=ax2,
    scatter_kws={'alpha': 0.7, 's': 20, 'color': 'green'}, # 점 스타일
    line_kws={'color': 'black', 'linestyle': '--'}       # 선 스타일
)

# Spearman 상관계수 계산
r2, p2 = spearmanr(combined_df['Inflammatory_Response'], combined_df['Tfh'])

# 제목에 r값, p값 표시
ax2.set_title(f"Spearman r = {r2:.3f} (p = {p2:.2e})", fontsize=14)
ax2.set_xlabel("Inflammatory Response Score", fontsize=12)
ax2.set_ylabel("Tfh (ImmuCellAI)", fontsize=12)

# --- 3. 최종 표시 ---
plt.tight_layout()
plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/inflammation_immunecell_corr.pdf", dpi=300,bbox_inches='tight')
plt.show()

# %%
####^^ ssGSEA score correlation heatmap ############
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
    gene_sets="GO_Biological_Process_2021",
    outdir=None,
    sample_norm_method="rank",
    permutation_num=0
)
df_major = res_major.res2d
score_major = df_major.pivot(index="Term", columns="Name", values="ES")


results = []


###^^^# score_total or score_major 중 선택
score_df = score_major # 예시: major TPM 기반

for term in score_df.index:
    # pathway score 불러오기
    vals = score_df.loc[term]

    # median split
    group = np.where(vals > vals.median(), "High", "Low")

    # Cox 모델용 데이터프레임
    tmp = clin_df[["PFS", "recur"]].copy()
    tmp["group"] = (group == "High").astype(int)  # High=1, Low=0

    try:
        cph = CoxPHFitter()
        cph.fit(tmp, duration_col="PFS", event_col="recur")
        hr = np.exp(cph.params_)[0]
        ci_low, ci_high = np.exp(cph.confidence_intervals_.iloc[0]).tolist()
        pval = cph.summary["p"].values[0]

        results.append({
            "Pathway": term,
            "HR": hr,
            "CI_low": ci_low,
            "CI_high": ci_high,
            "pval": pval
        })
    except Exception as e:
        print(f"Skipped {term} due to error: {e}")

#%%
# 결과 DataFrame
res_df = pd.DataFrame(results)

# 필터링: High 위험 (HR>1) + p<0.05
sig_df_high = res_df[(res_df["HR"] > 1) & (res_df["pval"] < 0.05)].sort_values("pval")
sig_df_low = res_df[(res_df["HR"] < 1) & (res_df["pval"] < 0.05)].sort_values("pval")

# 상위 20개씩 선택
df_low = sig_df_low.nsmallest(20, "pval") 
df_high = sig_df_high.nsmallest(20, "pval") 

# 계산
df_low["log2HR"] = np.log2(df_low["HR"]) 
df_high["log2HR"] = np.log2(df_high["HR"]) 
df_low["neglog10p"] = -np.log10(df_low["pval"]) 
df_high["neglog10p"] = -np.log10(df_high["pval"])

# --- 정렬 순서 변경 ---
# X축에 나열하기 위해 Protective(Low) -> High-risk(High) 순으로 정렬
# (HR이 낮은 것부터 높은 것 순으로)
df_low = df_low.sort_values("HR", ascending=True) # Protective (HR < 1)
df_high = df_high.sort_values("HR", ascending=True) # High-risk (HR > 1)

try:
    high_pathway_list = df_high['Pathway'].tolist()
    low_pathway_list = df_low['Pathway'].tolist()
except NameError:
    print("!!! 'df_high' 또는 'df_low' DataFrame이 없습니다.")
    print("!!! 이전 단계의 barplot 코드를 먼저 실행해야 합니다.")
    # (실행 중단)
    raise

# 1.2. score_major에서 해당 pathway들의 ssGSEA 점수만 필터링
# (score_major는 pathway가 index, 샘플이 column이라고 가정)
high_scores = score_major.loc[high_pathway_list, :]
low_scores = score_major.loc[low_pathway_list, :]

# --- 2. 상관관계 계산 (Spearman) ---

# pandas의 .corr() 함수는 *컬럼(column)*을 기준으로 작동합니다.
# 현재 데이터(score_major)는 pathway가 *행(row)*에 있으므로,
# 계산 전에 .T (Transpose, 전치)를 사용해 행과 열을 바꿔줍니다.
# (샘플 = 행, pathway = 컬럼)

high_scores_T = high_scores.T
low_scores_T = low_scores.T

# 2.1. 두 DataFrame을 합친 후 전체 상관관계 매트릭스 계산
combined_scores_T = pd.concat([high_scores_T, low_scores_T], axis=1)
corr_matrix_full = combined_scores_T.corr(method='spearman')

# 2.2. High-Risk (행) x Low-Risk (열) 부분만 잘라내기
corr_matrix_final = corr_matrix_full.loc[high_pathway_list, low_pathway_list]

print(f"상관관계 매트릭스 생성 완료 (Shape: {corr_matrix_final.shape})")

# --- 3. Clustermap 시각화 ---
print("Clustermap을 그리는 중...")

# 폰트 크기 등 시각적 요소 조절
sns.set_context("notebook", font_scale=0.8)

g = sns.clustermap(
    corr_matrix_final,
    method='average',     # 클러스터링 방식
    cmap='RdBu_r',        # 색상 맵 (Red-White-Blue, r=reversed)
                          # 빨강 = 양의 상관, 파랑 = 음의 상관
    center=0,             # 색상 맵의 중앙을 0으로 설정
    vmin=-1, vmax=1,      # 색상 범위 고정
    figsize=(14, 14),     # 플롯 크기 (정사각형 권장)
    annot=False,          # (annot=True로 하면 숫자가 너무 많아 보이지 않음)
    linewidths=0.5,
    linecolor='lightgray'
)

# 제목 추가
g.fig.suptitle('Spearman Correlation: High-HR vs Low-HR Pathways', 
               fontsize=16, y=1.02)

# X축, Y축 라벨 크기 조절
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.show()

# %%
####^^^ sample clustering ########################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch 
import matplotlib.colors as mcolors 
import matplotlib.cm as cm 

# --- 1. & 2. 데이터 준비 (동일) ---
try:
    high_pathway_list = df_high['Pathway'].tolist()
    low_pathway_list = df_low['Pathway'].tolist()
    all_pathway_list = high_pathway_list + low_pathway_list
except NameError:
    print("!!! 'df_high' 또는 'df_low' DataFrame이 없습니다.")
    raise

data_for_heatmap = score_major.loc[all_pathway_list]
samples_to_use = data_for_heatmap.columns 
data_for_heatmap = data_for_heatmap.apply(pd.to_numeric, errors='coerce') # (이전 코드에서 가져옴)

# --- 3. 데이터 준비: Annotation (수정) ---

# 3.1. Row (Pathway) Annotation (동일)
row_annot_df = pd.DataFrame(index=all_pathway_list)
row_annot_df['Risk Type'] = 'Low-risk'
row_annot_df.loc[high_pathway_list, 'Risk Type'] = 'High-risk'
risk_palette = {'High-risk': '#E73724', 'Low-risk': '#0B55CB'}
row_colors_mapped = row_annot_df['Risk Type'].map(risk_palette)

# 3.2. Column (Sample) Annotation (수정)
ghrd_palette = {0: '#BEBEBE', 1: 'black'}
ghrd_colors = clin_df.loc[samples_to_use, 'gHRD'].map(ghrd_palette)
ghrd_colors.name = 'gHRD'

# --- (수정) PFS Colormap 변경 (e.g., plasma) ---
pfs_series = pd.to_numeric(clin_df.loc[samples_to_use, 'PFS'], errors='coerce')
norm = mcolors.Normalize(vmin=pfs_series.min(), vmax=pfs_series.max())
cmap = cm.Greys  # cm.Greys -> cm.plasma (또는 cm.viridis)
# -----------------------------------------------

pfs_colors = [mcolors.to_hex(cmap(norm(v))) for v in pfs_series]
pfs_colors = pd.Series(pfs_colors, index=pfs_series.index, name='PFS')
col_colors_list = [ghrd_colors, pfs_colors]


# --- 4. Clustermap 그리기 (수정) ---
print("Clustermap 생성을 시작합니다...")

g = sns.clustermap(
    data_for_heatmap,
    
    # --- (수정) 클러스터링 방법 변경 ---
    method='ward',      # 'average'(기본값) -> 'ward'로 변경
    # ---------------------------------
    
    col_cluster=True,  
    row_cluster=True,  
    
    z_score=0, 
    cmap='vlag',      
    center=0,         
    
    # --- (수정) 색상 범위 고정 ---
    vmin=-2.5,          # Z-score -2.5 이하
    vmax=2.5,           # Z-score +2.5 이상
    # -----------------------------
    
    row_colors=row_colors_mapped,
    col_colors=col_colors_list,
    
    figsize=(16, 14),      
    xticklabels=False,     
    yticklabels=True,     
    colors_ratio=0.03,     
    cbar_pos=(0.02, 0.8, 0.03, 0.15),
)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(), fontsize = 12)

# --- 5. 범례(Legend) 추가 (동일) ---
row_legend_handles = [
    Patch(facecolor=risk_palette['High-risk'], label='High HR'),
    Patch(facecolor=risk_palette['Low-risk'], label='Low HR')
]
ghrd_legend_handles = [
    Patch(facecolor=ghrd_palette[0], label='gHRD-'),
    Patch(facecolor=ghrd_palette[1], label='gHRD+')
]
g.ax_heatmap.legend(
    handles=row_legend_handles + ghrd_legend_handles, 
    title='',
    loc='upper left', 
    bbox_to_anchor=(1.02, 1.2), 
    frameon=False,
    fontsize=12
)
g.cax.set_ylabel("Z-score", fontsize=12)
    
#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/clustermap_top20.pdf", dpi=300,bbox_inches='tight')
plt.show()

# %%
####^^ c-index #####################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter
from sklearn.model_selection import GridSearchCV
from sksurv.metrics import concordance_index_censored
from sklearn.metrics import make_scorer

# ✅ 핵심: CoxnetSurvivalAnalysis를 linear_model에서 import
from sksurv.linear_model import CoxnetSurvivalAnalysis


# --- 0. 가정: 변수 로드 ---
# df_high, df_low, score_major, clin_df, TIME_COL, EVENT_COL

print("--- 1. 데이터 준비 (Data Preparation) ---")
try:
    high_pathways = df_high['Pathway'].tolist()
    low_pathways = df_low['Pathway'].tolist()
    all_pathways = high_pathways + low_pathways
except NameError:
    print("!!! 'df_high' 또는 'df_low' DataFrame이 없습니다.")
    raise

# feature matrix (samples x pathways)
X = score_major.loc[all_pathways].T

# append survival columns
data_for_ml = X.join(clin_df[[TIME_COL, EVENT_COL]])

# drop rows missing either features or survival info
data_for_ml = data_for_ml.dropna(subset=all_pathways + [TIME_COL, EVENT_COL])

X_final = data_for_ml[all_pathways]
y_final_df = data_for_ml[[TIME_COL, EVENT_COL]]

# y for sksurv: structured array [('event', bool), ('time', float)]
y_sksurv = np.array(
    list(zip(
        y_final_df[EVENT_COL].astype(bool),
        y_final_df[TIME_COL].astype(float)
    )),
    dtype=[('event', bool), ('time', float)]
)

# scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

print(f"총 {X_scaled.shape[0]}개의 샘플과 {X_scaled.shape[1]}개의 feature로 분석 시작.")


print("\n--- 2. Lasso-Cox (CoxnetSurvivalAnalysis + GridSearchCV) ---")

# CoxnetSurvivalAnalysis는 내부적으로 알파 경로 전체를 학습할 수 있는데,
# GridSearchCV로도 alpha sweep을 할 수 있다.
# l1_ratio=1.0 -> pure L1 (lasso)
base_model = CoxnetSurvivalAnalysis(l1_ratio=1.0)

# alpha를 수동으로 줄 수도 있고, CoxnetSurvivalAnalysis는 자체적으로 alphas 경로를 추정도 가능.
# 여기서는 명시적으로 alphas 리스트를 넘겨 GridSearchCV에서 고르게 할게.
alphas_logspace = 10.0 ** np.arange(-4, 1, 0.1)  # 1e-4 ... 1e0
param_grid = {
    "alphas": [ [a] for a in alphas_logspace ]
    # CoxnetSurvivalAnalysis expects a *list of alphas*,
    # so we wrap each scalar a into [a]
}

# concordance_index_censored returns (c_index, ...),
# but make_scorer expects scalar ⇒ 우리가 score_func 래핑해줘야 함.
def cindex_scorer(y_true, y_pred_risk):
    # y_pred_risk: higher means more risk (worse survival)
    event = y_true["event"]
    time = y_true["time"]
    c_index, _ = concordance_index_censored(
        event, time, -y_pred_risk  # concordance_index_censored expects *higher survival* better,
                                   # Coxnet gives risk scores (higher risk = worse), so negate
    )
    return c_index

custom_scorer = make_scorer(cindex_scorer, greater_is_better=True)

gcv = GridSearchCV(
    base_model,
    param_grid=param_grid,
    scoring=custom_scorer,
    cv=10,
    n_jobs=1,
    verbose=1
)

print("GridSearchCV를 시작합니다 (최적의 alpha 탐색 중)...")
gcv.fit(X_scaled, y_sksurv)

best_model = gcv.best_estimator_
best_alpha_list = gcv.best_params_['alphas']  # this is a list with one alpha
best_alpha = best_alpha_list[0]
c_index_cv = gcv.best_score_

print(f"\nGridSearchCV 완료.")
print(f"최적의 Alpha: {best_alpha:.5f}")
print(f"Lasso-Cox 모델 C-index (Cross-Validation): {c_index_cv:.4f}")

# training C-index on full data using best_model
# best_model.score(X, y) already returns concordance index internally
train_c_index = best_model.score(X_scaled, y_sksurv)
print(f"Lasso-Cox 모델 C-index (Full Training Data): {train_c_index:.4f}")


print("\n--- 3. Feature 선택 결과 ---")
# best_model.coef_ shape: (n_features,)

# --- 3. Feature 선택 결과 ---
# best_model.coef_ shape: (n_features, 1)
coefs = best_model.coef_

# --- ⬇️ 수정된 부분 ⬇️ ---
# coefs가 (40, 1) shape의 2D 배열이므로 .flatten()으로 1D (40,)로 변경
selected_features_series = pd.Series(coefs.flatten(), index=all_pathways)
# --- ⬆️ 수정된 부분 ⬆️ ---


final_features_df = (
    selected_features_series[selected_features_series != 0]
    .sort_values(ascending=False)
)

print(f"총 {len(final_features_df)}개의 feature가 선택되었습니다.")
print("선택된 Features와 계수:")
print(final_features_df)

final_feature_list = final_features_df.index.tolist()


if not final_feature_list:
    # (이하 4번 섹션은 동일하게 실행됩니다)
    print("\n경고: Lasso가 모든 feature를 0으로 만들었습니다. 선택된 feature가 없습니다.")
else:
    print("\n--- 4. lifelines CoxPHFitter로 최종 모델 재학습 ---")
    lifelines_X = X_final[final_feature_list]
    lifelines_df = lifelines_X.join(y_final_df)

    # lifelines formula는 컬럼명이 특수문자 있을 경우 백틱 필요
    formula = " + ".join(f"`{col}`" for col in final_feature_list)
    print(f"Formula: {formula}")

    cph_final = CoxPHFitter()
    cph_final.fit(
        lifelines_df,
        duration_col=TIME_COL,
        event_col=EVENT_COL,
        formula=formula
    )

    print("\n--- 최종 모델 요약 (lifelines) ---")
    cph_final.print_summary()

    c_index_lifelines = cph_final.concordance_index_
    print(f"\n최종 'lifelines' 모델 C-index: {c_index_lifelines:.4f}")

# %%
###^^ randomsurvivalforest ############
import numpy as np
import pandas as pd

from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer
from sklearn.inspection import permutation_importance
from lifelines import CoxPHFitter


########################################
# 0. 기본 정보 출력
########################################
print(f"RSF train samples: {X_final.shape[0]}, features: {X_final.shape[1]}")


########################################
# 1. C-index scorer 정의
########################################
def rsf_cindex_scorer(y_true, y_pred):
    event = y_true["event"]
    time = y_true["time"]
    y_pred = np.asarray(y_pred).squeeze()
    c_index = concordance_index_censored(event, time, y_pred)[0]
    return c_index

cindex_scorer = make_scorer(rsf_cindex_scorer, greater_is_better=True)


########################################
# 2. RSF + GridSearchCV로 최적 모델 찾기
########################################
base_rsf = RandomSurvivalForest(
    n_estimators=500,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
)

param_grid = {
    "n_estimators":       [200, 500, 1000],
    "min_samples_split":  [2, 5, 10],
    "min_samples_leaf":   [1, 3, 5],
    "max_features":       ["sqrt", 0.5, 0.8],
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

gcv_rsf = GridSearchCV(
    estimator=base_rsf,
    param_grid=param_grid,
    scoring=cindex_scorer,
    cv=cv,
    n_jobs=-1,
    verbose=1,
)

print("\n--- Fitting Random Survival Forest (GridSearchCV) ---")
gcv_rsf.fit(X_final, y_sksurv)

best_rsf = gcv_rsf.best_estimator_
best_params = gcv_rsf.best_params_
best_cv_cindex = gcv_rsf.best_score_

print("\n[RSF 모델 선택 결과]")
print("Best params:", best_params)
print(f"Best CV C-index: {best_cv_cindex:.4f}")

# train 성능
y_pred_train = np.asarray(best_rsf.predict(X_final)).squeeze()
train_cindex = concordance_index_censored(
    y_sksurv["event"],
    y_sksurv["time"],
    y_pred_train
)[0]
print(f"Train C-index (RSF): {train_cindex:.4f}")


########################################
# 3. Permutation importance 계산
########################################
print("\n--- Computing permutation importance ---")
perm_result = permutation_importance(
    estimator=best_rsf,
    X=X_final,
    y=y_sksurv,
    scoring=cindex_scorer,   # drop in C-index
    n_repeats=15,
    random_state=42,
)

importances_mean = pd.Series(
    perm_result.importances_mean,
    index=X_final.columns
).sort_values(ascending=False)

importances_std = pd.Series(
    perm_result.importances_std,
    index=X_final.columns
).loc[importances_mean.index]

print("\n[Permutation Importance] mean ΔC-index (bigger = more important):")
print(importances_mean)

topN = 20
top_features_raw = importances_mean.head(topN).index.tolist()

print(f"\nTop {topN} features before collinearity filtering:")
for rank, feat in enumerate(top_features_raw, start=1):
    mean_drop = importances_mean.loc[feat]
    std_drop = importances_std.loc[feat]
    print(f"{rank:2d}. {feat:30s}  ΔC-index={mean_drop:.4f} ± {std_drop:.4f}")


########################################
# 4. 상관계수 기반 중복 제거
#    - 상관계수(abs) > 0.85 인 feature 쌍 중,
#      더 덜 중요한 놈을 버린다.
########################################
corr = X_final[top_features_raw].corr().abs()

# upper triangle만 보고 높은 상관 찾기
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

to_drop = set()
for col in upper.columns:
    # col과 상관도 높은 애들 찾기
    highly_corr = upper[col][upper[col] > 0.85].index.tolist()
    for hc in highly_corr:
        # 둘 중 중요도가 낮은 feature를 버린다
        imp_col = importances_mean[col]
        imp_hc  = importances_mean[hc]
        if imp_col < imp_hc:
            to_drop.add(col)
        else:
            to_drop.add(hc)

filtered_features = [f for f in top_features_raw if f not in to_drop]

print("\n[Collinearity filtering]")
print("Dropped due to high correlation (>0.85):", list(to_drop))
print("Remaining features:", filtered_features)
print(f"Remaining feature count: {len(filtered_features)}")


########################################
# 5. CoxPHFitter 재학습 (penalizer로 안정화)
########################################
# lifelines용 데이터프레임 구성
lifelines_df = X_final[filtered_features].join(y_final_df)

# lifelines formula (백틱으로 감싸서 안전하게)
cox_formula = " + ".join(f"`{col}`" for col in filtered_features)
print("\nCox formula after filtering:")
print(cox_formula)

print("\n--- Fitting penalized Cox model on RSF-selected, de-correlated features ---")
cph_final = CoxPHFitter()  # ridge-like regularization
cph_final.fit(
    lifelines_df,
    duration_col=TIME_COL,
    event_col=EVENT_COL,
    formula=cox_formula
)

print("\n[Cox summary (after collinearity filtering + ridge penalization)]")
cph_final.print_summary()

print(f"\nFinal Cox C-index on full data: {cph_final.concordance_index_:.4f}")


# %%
