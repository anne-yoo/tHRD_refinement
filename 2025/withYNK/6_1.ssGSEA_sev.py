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

rcParams['pdf.fonttype'] = 42  ;
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
import gseapy as gp
from lifelines import KaplanMeierFitter, statistics
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Data 준비
# -----------------------------

# tpm_df, tu_df: transcript x sample
# clin_df: sample x clinical info
# row index = sample ID, column order = colnames of TPM/TU

tpm_df = pd.read_csv("/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/112_PARPi_transcript_exp.txt", sep="\t", index_col=0)
#tpm_df = tpm_df.drop(columns=['target_gene'])  # gene_name 컬럼 제거
tu_df = pd.read_csv("/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/112_PARPi_transcript_usage.txt", sep="\t", index_col=0)
clin_df = pd.read_csv("/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/112_PARPi_clinicalinfo.txt", sep="\t", index_col=0)

majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = majorminor[majorminor['type']=='major']['gene_ENST'].to_list()

BRCAinfo = pd.read_excel("/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/BRCAinfo.xls", index_col=0, header=1)
BRCAinfo.index = BRCAinfo['GCgenome']
BRCAinfo = BRCAinfo[['gBRCA1','gBRCA2','tBRCA1','tBRCA2',]]
clin_df = clin_df.join(BRCAinfo, how='left')


clin_df = clin_df[clin_df['setting']=='maintenance']
tpm_df = tpm_df[clin_df.index]
tu_df = tu_df[clin_df.index]


#%%
#%%
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gseapy as gp
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import scipy.stats as stats

# -----------------------------
# 0. Input
# -----------------------------
# tpm_df, tu_df, clin_df, majorlist 불러오기 전제
assert all(tpm_df.columns == tu_df.columns)
assert all(tpm_df.columns == clin_df.index)

# -----------------------------
# 1. Transcript → Gene mapping
# -----------------------------
gene_map = pd.Series([x.split("-")[-1] for x in tpm_df.index], index=tpm_df.index)

# -----------------------------
# 2. Gene-level matrix
# -----------------------------
expr_total = tpm_df.groupby(gene_map).sum()
expr_major = tpm_df.loc[tpm_df.index.isin(majorlist)]
gene_map_major = gene_map.loc[expr_major.index]
expr_major_gene = expr_major.groupby(gene_map_major).sum()

expr_total_log = np.log2(expr_total + 1)
expr_major_log = np.log2(expr_major_gene + 1)

# -----------------------------
# 3. ssGSEA function
# -----------------------------
def run_ssgsea(expr_matrix, db="hallmark"):
    if db.lower() == "hallmark":
        geneset = "MSigDB_Hallmark_2020"
    elif db.lower() == "bp2021":
        geneset = "GO_Biological_Process_2021"
    else:
        raise ValueError("db must be 'hallmark' or 'bp2021'")
    
    res = gp.ssgsea(
        data=expr_matrix,
        gene_sets=geneset,
        outdir=None,
        sample_norm_method="rank",
        permutation_num=0
    )
    df = res.res2d
    score = df.pivot(index="Term", columns="Name", values="ES")
    return score

# -----------------------------
# 4. Run ssGSEA (choose expr + db)
# -----------------------------
##^^ ###########################################
expr_choice = expr_major_log   # <--- major vs total
db_choice = "bp2021"         # <--- hallmark vs bp2021
##^^ ###########################################
score_df = run_ssgsea(expr_choice, db=db_choice)

# -----------------------------
# 5. BRCA annotation (0=WT,1=Pathogenic,2=VUS)
# -----------------------------
palette_brca = {"0":"#B3E061","1":"#EC741E","2":"#F1CA0A"}
brca_colors = pd.DataFrame({
    col: clin_df[col].astype(str).map(palette_brca)
    for col in ["gBRCA1","gBRCA2","tBRCA1","tBRCA2"]
}, index=clin_df.index)


# response (binary: 0=white, 1=black)
palette_resp = {0:"white", 1:"black"}
resp_colors = clin_df["response"].map(palette_resp)

# PFS (continuous: grayscale)
norm_pfs = mpl.colors.Normalize(vmin=clin_df["PFS"].min(), vmax=clin_df["PFS"].max())
cmap_pfs = plt.cm.Greys
pfs_colors = [mpl.colors.to_hex(cmap_pfs(norm_pfs(x))) for x in clin_df["PFS"]]

# (4) line (1L vs ~1L → binary, 흑백)
clin_df["line_binary"] = np.where(clin_df["line"]=="1L", 0, 1)
palette_line = {0:"white", 1:"black"}
line_colors = clin_df["line_binary"].map(palette_line)

# (5) BRCA_binary (네 카테고리 전부 0 vs 나머지, 흑백)
clin_df["BRCA_binary"] = np.where(
    (clin_df[["gBRCA1","gBRCA2","tBRCA1","tBRCA2"]] == 0).all(axis=1), 0, 1
)
palette_brca_bin = {0:"white", 1:"black"}
brca_bin_colors = clin_df["BRCA_binary"].map(palette_brca_bin)

# (6) col_colors 합치기
col_colors = pd.concat(
    [brca_colors,
     pd.DataFrame({
         "response": resp_colors,
         "PFS": pfs_colors,
         "line": line_colors,
         "BRCA_binary": brca_bin_colors
     }, index=clin_df.index)],
    axis=1
)

# heatmap
sns.clustermap(
    score_df.apply(pd.to_numeric, errors="coerce"),
    metric="euclidean", method="ward",
    cmap="RdBu_r", z_score=0,
    col_colors=col_colors, figsize=(14,12)
)
plt.suptitle(f"ssGSEA Heatmap ({db_choice}, {'Major' if expr_choice is expr_major_log else 'Total'})", y=1.02)
plt.show()

#%%
# -----------------------------
# 7. Survival analysis (BRCA WT only, DNA Repair)
# -----------------------------
# WT = BRCA1/2 전부 0인 경우만
brca_wt_idx = (clin_df[["gBRCA1","gBRCA2","tBRCA1","tBRCA2"]] == 0).all(axis=1)
clin_wt = clin_df.loc[brca_wt_idx]

dna_score = score_df.loc["Myc Targets V1", clin_wt.index]
# double-strand break repair via homologous recombination (GO:0000724)
clin_wt["dna_group"] = np.where(dna_score > dna_score.median(), "High", "Low")

kmf = KaplanMeierFitter()
plt.figure(figsize=(6,5))
ax = plt.subplot(111)
colors = {"High":"#DA4343","Low":"#3396D3"}
for label in clin_wt["dna_group"].unique():
    idx = clin_wt["dna_group"] == label
    kmf.fit(clin_wt.loc[idx, "PFS"], clin_wt.loc[idx, "recur"], label=label)
    kmf.plot_survival_function(ax=ax, ci_show=True, color=colors.get(label,"gray"))

idx_high = clin_wt["dna_group"] == "High"
res = logrank_test(
    clin_wt.loc[idx_high,"PFS"], clin_wt.loc[~idx_high,"PFS"],
    clin_wt.loc[idx_high,"recur"], clin_wt.loc[~idx_high,"recur"]
)
tmp = clin_wt[["PFS","recur","dna_group"]].copy()
tmp["dna_group"] = (tmp["dna_group"]=="High").astype(int)
cph = CoxPHFitter()
cph.fit(tmp, duration_col="PFS", event_col="recur")
hr = np.exp(cph.params_)[0]
ci = cph.confidence_intervals_.iloc[0].tolist()

plt.title("Myc Targets V1 (BRCA WT only)")
plt.text(0.05,0.05,f"Log-rank p={res.p_value:.3}\nHR={hr:.2f} ({ci[0]:.2f}-{ci[1]:.2f})",
         transform=ax.transAxes,fontsize=10)
plt.show()

#%%
# -----------------------------
# 8. Pathway vs BRCA mutation (WT=0, Pathogenic=1, VUS=2)
# -----------------------------
def plot_pathway_vs_brca(score_df, clin_df, term, brca_col):
    vals = pd.to_numeric(score_df.loc[term], errors="coerce")
    groups = pd.to_numeric(clin_df[brca_col], errors="coerce")
    df_plot = pd.DataFrame({"vals": vals, "groups": groups}).dropna()
    df_plot["groups"] = df_plot["groups"].astype(int).astype(str)

    # WT vs Pathogenic만 Mann-Whitney U
    x = df_plot.loc[df_plot["groups"]=="1","vals"]
    y = df_plot.loc[df_plot["groups"]=="0","vals"]
    if len(x) > 0 and len(y) > 0:
        stat, pval = stats.mannwhitneyu(x,y,alternative="two-sided")
    else:
        pval = np.nan

    sns.boxplot(x="groups", y="vals", data=df_plot,
                order=["0","1","2"],
                palette={"0":"#91D611","1":"#EC741E","2":"#F1CA0A"})
    sns.stripplot(x="groups", y="vals", data=df_plot,
                  color="black", alpha=0.5)
    plt.title(f"{term} vs {brca_col} (p={pval:.2e})\n0=WT,1=Pathogenic,2=VUS")
    plt.show()

# 예시 실행
for col in ["gBRCA1","gBRCA2","tBRCA1","tBRCA2"]:
    plot_pathway_vs_brca(score_df, clin_df, term="DNA Repair", brca_col=col)

# -----------------------------
# 9. Survival analysis by BRCA subtype WT only (2x2 subplot)
# -----------------------------
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

term = "Myc Targets V1"
cols = ["gBRCA1","gBRCA2","tBRCA1","tBRCA2"]
kmf = KaplanMeierFitter()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, col in enumerate(cols):
    ax = axes[i]
    subset = clin_df[clin_df[col] == 0].copy()  # WT만 선택
    if subset.empty:
        ax.set_title(f"{col}=WT only (no samples)")
        ax.axis("off")
        continue

    dna_score = pd.to_numeric(score_df.loc[term, subset.index], errors="coerce")
    subset["dna_group"] = np.where(dna_score > dna_score.median(), "High", "Low")

    colors = {"High":"#DA4343","Low":"#3396D3"}
    for label in subset["dna_group"].unique():
        idx = subset["dna_group"] == label
        kmf.fit(subset.loc[idx,"PFS"], subset.loc[idx,"recur"], label=label)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=colors.get(label,"gray"))

    idx_high = subset["dna_group"] == "High"
    if idx_high.sum() > 0 and (~idx_high).sum() > 0:
        res = logrank_test(
            subset.loc[idx_high,"PFS"],
            subset.loc[~idx_high,"PFS"],
            subset.loc[idx_high,"recur"],
            subset.loc[~idx_high,"recur"]
        )
        tmp = subset[["PFS","recur","dna_group"]].copy()
        tmp["dna_group"] = (tmp["dna_group"]=="High").astype(int)
        cph = CoxPHFitter()
        cph.fit(tmp, duration_col="PFS", event_col="recur")
        hr = np.exp(cph.params_)[0]
        ci = cph.confidence_intervals_.iloc[0].tolist()
        text = f"p={res.p_value:.3}\nHR={hr:.2f} ({ci[0]:.2f}-{ci[1]:.2f})"
    else:
        text = "Not enough samples"

    ax.set_title(f"{term} survival ({col}=WT only)")
    ax.text(0.05,0.05,text, transform=ax.transAxes, fontsize=9)
    ax.set_xlabel("time")
    ax.set_ylabel("survival probability")

plt.tight_layout()
plt.show()

#%%

# -----------------------------
# 10. Pathway-level Cox HR barplots by BRCA subtype WT only
# -----------------------------
from lifelines import CoxPHFitter
import matplotlib as mpl
cols = ["gBRCA1","gBRCA2","tBRCA1","tBRCA2"]

for col in cols:
    subset = clin_df[clin_df[col] == 0].copy()  # WT만 선택
    if subset.empty:
        print(f"{col}=WT only: no samples, skipped")
        continue

    results = []
    for term in score_df.index:
        vals = pd.to_numeric(score_df.loc[term, subset.index], errors="coerce")
        tmp = subset[["PFS","recur"]].copy()
        tmp["group"] = (vals > vals.median()).astype(int)  # High=1, Low=0

        # sample 수가 충분해야 Cox 가능
        if tmp["group"].nunique() < 2:
            continue

        try:
            cph = CoxPHFitter()
            cph.fit(tmp, duration_col="PFS", event_col="recur")
            hr = np.exp(cph.params_)[0]
            ci_low, ci_high = np.exp(cph.confidence_intervals_.iloc[0]).tolist()
            pval = cph.summary["p"].values[0]
            results.append({"Pathway": term, "HR": hr, "CI_low": ci_low, "CI_high": ci_high, "pval": pval})
        except Exception as e:
            continue

    res_df = pd.DataFrame(results)
    if res_df.empty:
        print(f"{col}=WT only: no valid Cox results")
        continue

    sig_df_high = res_df[(res_df["HR"] > 1) & (res_df["pval"] < 0.05)].sort_values("pval")
    sig_df_low = res_df[(res_df["HR"] < 1) & (res_df["pval"] < 0.05)].sort_values("pval")

    df_low = sig_df_low.nsmallest(20, "pval").copy()
    df_high = sig_df_high.nsmallest(20, "pval").copy()
    df_low["log2HR"] = np.log2(df_low["HR"])
    df_high["log2HR"] = np.log2(df_high["HR"])
    df_low["neglog10p"] = -np.log10(df_low["pval"])
    df_high["neglog10p"] = -np.log10(df_high["pval"])

    df_high = df_high.sort_values("HR", ascending=True)
    df_low = df_low.sort_values("HR", ascending=False)

    pathways = list(df_low["Pathway"]) + list(df_high["Pathway"])
    y_pos_low = np.arange(len(df_low))
    y_pos_high = np.arange(len(df_high)) + len(df_low)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.tick_params(axis="y", which="major", pad=4)

    cmap_high = plt.cm.Reds
    cmap_low = plt.cm.Blues
    norm_high = mpl.colors.Normalize(vmin=df_high["neglog10p"].min() if not df_high.empty else 0,
                                     vmax=df_high["neglog10p"].max() if not df_high.empty else 1)
    norm_low = mpl.colors.Normalize(vmin=df_low["neglog10p"].min() if not df_low.empty else 0,
                                    vmax=df_low["neglog10p"].max() if not df_low.empty else 1)

    if not df_high.empty:
        ax.barh(y_pos_high, df_high["log2HR"],
                color=cmap_high(norm_high(df_high["neglog10p"])), label="High-risk (HR>1)")
    if not df_low.empty:
        ax.barh(y_pos_low, df_low["log2HR"],
                color=cmap_low(norm_low(df_low["neglog10p"])), label="Protective (HR<1)")

    ax.set_yticks(np.arange(len(pathways)))
    ax.set_yticklabels(pathways, fontsize=10)
    ax.set_xlabel("log2(HR)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(-0.5, len(pathways)-0.5)

    sm_high = mpl.cm.ScalarMappable(cmap=cmap_high, norm=norm_high)
    sm_low = mpl.cm.ScalarMappable(cmap=cmap_low, norm=norm_low)
    sm_high.set_array([])
    sm_low.set_array([])

    cax_high = fig.add_axes([0.92, 0.60, 0.02, 0.25])
    cbar_high = fig.colorbar(sm_high, cax=cax_high)
    cbar_high.ax.tick_params(labelsize=10)
    cax_low = fig.add_axes([0.92, 0.20, 0.02, 0.25])
    cbar_low = fig.colorbar(sm_low, cax=cax_low)
    cbar_low.ax.tick_params(labelsize=10)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.subplots_adjust(left=0.55)
    sns.despine()
    plt.suptitle(f"Pathway Cox HR (WT only: {col})", y=1.02)
    plt.show()

# %%
# -----------------------------
# 11. Pathway-level Cox HR barplot (All WT vs Mut_or_VUS)
# -----------------------------
from lifelines import CoxPHFitter
import matplotlib as mpl

# -----------------------------
# 11. Pathway-level Cox HR barplots by BRCA_binary groups
# -----------------------------
from lifelines import CoxPHFitter

# All WT vs Mut_or_VUS binary 정의
clin_df["BRCA_binary"] = np.where(
    (clin_df[["gBRCA1","gBRCA2","tBRCA1","tBRCA2"]] == 0).all(axis=1),
    "All_WT", "Mut_or_VUS"
)

def pathway_hr_barplot(subset, label, topn=20):
    results = []
    for term in score_df.index:
        vals = pd.to_numeric(score_df.loc[term, subset.index], errors="coerce")
        tmp = subset[["PFS","recur"]].copy()
        tmp["group"] = (vals > vals.median()).astype(int)  # High=1, Low=0
        if tmp["group"].nunique() < 2:
            continue
        try:
            cph = CoxPHFitter()
            cph.fit(tmp, duration_col="PFS", event_col="recur")
            hr = np.exp(cph.params_)[0]
            ci_low, ci_high = np.exp(cph.confidence_intervals_.iloc[0]).tolist()
            pval = cph.summary["p"].values[0]
            results.append({"Pathway": term, "HR": hr, "CI_low": ci_low, "CI_high": ci_high, "pval": pval})
        except Exception:
            continue
    
    res_df = pd.DataFrame(results)
    if res_df.empty:
        print(f"{label}: no valid Cox results")
        return
    
    sig_df_high = res_df[(res_df["HR"] > 1) & (res_df["pval"] < 0.05)].sort_values("pval")
    sig_df_low = res_df[(res_df["HR"] < 1) & (res_df["pval"] < 0.05)].sort_values("pval")

    df_low = sig_df_low.nsmallest(topn, "pval").copy()
    df_high = sig_df_high.nsmallest(topn, "pval").copy()
    df_low["log2HR"] = np.log2(df_low["HR"])
    df_high["log2HR"] = np.log2(df_high["HR"])
    df_low["neglog10p"] = -np.log10(df_low["pval"])
    df_high["neglog10p"] = -np.log10(df_high["pval"])

    df_high = df_high.sort_values("HR", ascending=True)
    df_low = df_low.sort_values("HR", ascending=False)

    pathways = list(df_low["Pathway"]) + list(df_high["Pathway"])
    y_pos_low = np.arange(len(df_low))
    y_pos_high = np.arange(len(df_high)) + len(df_low)

    fig, ax = plt.subplots(figsize=(5, 4)) #(10, 8)
    ax.tick_params(axis="y", which="major", pad=4)

    cmap_high = plt.cm.Reds
    cmap_low = plt.cm.Blues
    norm_high = mpl.colors.Normalize(vmin=df_high["neglog10p"].min() if not df_high.empty else 0,
                                     vmax=df_high["neglog10p"].max() if not df_high.empty else 1)
    norm_low = mpl.colors.Normalize(vmin=df_low["neglog10p"].min() if not df_low.empty else 0,
                                    vmax=df_low["neglog10p"].max() if not df_low.empty else 1)

    if not df_high.empty:
        ax.barh(y_pos_high, df_high["log2HR"],
                color=cmap_high(norm_high(df_high["neglog10p"])), label="High-risk (HR>1)")
    if not df_low.empty:
        ax.barh(y_pos_low, df_low["log2HR"],
                color=cmap_low(norm_low(df_low["neglog10p"])), label="Protective (HR<1)")

    ax.set_yticks(np.arange(len(pathways)))
    ax.set_yticklabels(pathways, fontsize=10)
    ax.set_xlabel("log2(HR)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(-0.5, len(pathways)-0.5)

    sm_high = mpl.cm.ScalarMappable(cmap=cmap_high, norm=norm_high)
    sm_low = mpl.cm.ScalarMappable(cmap=cmap_low, norm=norm_low)
    sm_high.set_array([])
    sm_low.set_array([])

    cax_high = fig.add_axes([0.92, 0.60, 0.02, 0.25])
    cbar_high = fig.colorbar(sm_high, cax=cax_high)
    cbar_high.ax.tick_params(labelsize=10)
    cax_low = fig.add_axes([0.92, 0.20, 0.02, 0.25])
    cbar_low = fig.colorbar(sm_low, cax=cax_low)
    cbar_low.ax.tick_params(labelsize=10)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.subplots_adjust(left=0.55)
    sns.despine()
    plt.suptitle(f"Pathway Cox HR ({label})", y=1.02)
    plt.show()

# 실행: All_WT 그룹 / Mut_or_VUS 그룹 각각
subset_allwt = clin_df[clin_df["BRCA_binary"]=="All_WT"]
subset_mutvus = clin_df[clin_df["BRCA_binary"]=="Mut_or_VUS"]

pathway_hr_barplot(subset_allwt, "All_WT")
pathway_hr_barplot(subset_mutvus, "Mut_or_VUS")

# %%
# -----------------------------
# 13. Venn diagram of BRCA WT samples
# -----------------------------
from matplotlib_venn import venn2, venn3, venn3_circles
from matplotlib import pyplot as plt
from matplotlib_venn import venn3, venn3_circles
from matplotlib_venn import venn2, venn2_circles
import matplotlib_venn as mv

# 4-set Venn은 venn4가 없고, UpSetPlot을 쓰거나 matplotlib_venn의 venn4처럼 커스텀 구현 필요
# 여기서는 UpSetPlot으로 대체
from upsetplot import UpSet, from_memberships

# 각 BRCA 타입별 WT=0인 sample set
sets = {
    "gBRCA1_WT": set(clin_df.index[clin_df["gBRCA1"] == 0]),
    "gBRCA2_WT": set(clin_df.index[clin_df["gBRCA2"] == 0]),
    "tBRCA1_WT": set(clin_df.index[clin_df["tBRCA1"] == 0]),
    "tBRCA2_WT": set(clin_df.index[clin_df["tBRCA2"] == 0])
}

# UpSet input
memberships = []
for sample in clin_df.index:
    mem = []
    for key in sets:
        if sample in sets[key]:
            mem.append(key)
    memberships.append(mem)

data = from_memberships(memberships)

# Plot
up = UpSet(data, subset_size="count", show_counts=True)
up.plot()
plt.suptitle("BRCA WT overlaps (UpSet plot)")
plt.show()

# %%
