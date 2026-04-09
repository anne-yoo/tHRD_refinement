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
tpm_df = pd.read_csv("/home/jiye/jiye/copycomparison/GENCODEquant/SEV_pre/111_pre/111_transcript_TPM.txt", sep="\t", index_col=0)
#tpm_df = tpm_df.drop(columns=['target_gene'])  # gene_name 컬럼 제거
tu_df = pd.read_csv("/home/jiye/jiye/copycomparison/GENCODEquant/SEV_pre/111_pre/111_transcript_TU.txt", sep="\t", index_col=0)
clin_df = pd.read_csv("/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/112_PARPi_clinicalinfo.txt", sep="\t", index_col=0)
BRCAinfo = pd.read_excel("/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/BRCAinfo.xls", index_col=0, header=1)
BRCAinfo.index = BRCAinfo['GCgenome']
BRCAinfo = BRCAinfo[['gBRCA1','gBRCA2','tBRCA1','tBRCA2',]]
clin_df = clin_df.join(BRCAinfo, how='left')

tu_df = tu_df.loc[:,tu_df.columns.isin(clin_df.index)]

# nmdlist = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/TranSuite/merged/merged_TranSuite_output/merged_transfeat/merged_transfeat_NMD_features.csv', sep=',')
# nmdlist = set(nmdlist['Transcript_ID'].to_list())
# majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
# majorminor['genename'] = majorminor['gene_ENST'].str.split("-",n=1).str[-1]
# majorminor['NMD'] = majorminor['transcriptid'].apply(lambda x: 'NMD' if x in nmdlist else 'non-NMD')
# majorminor.loc[(majorminor['type'] == 'major') & (majorminor['NMD'] == 'NMD'),'type'] = 'minor'

# majorlist = majorminor[majorminor['type']=='major']['gene_ENST'].to_list()
# minorlist = majorminor[majorminor['type']=='minor']['gene_ENST'].to_list()

majorlist = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/majorlist.txt',sep='\t')
majorlist = majorlist['Transcript-Gene']
minorlist = tu_df.index.difference(majorlist).to_list()

clin_df['line_binary'] = clin_df['line'].apply(lambda x: 'FL' if x=='1L' else 'N-FL')

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^################
clin_df = clin_df[clin_df['setting']=='maintenance']
#clin_df = clin_df[clin_df['BRCAmt']==0]
#clin_df = clin_df[clin_df['line_binary']=='N-FL']
############^^^^##################################

tpm_df = tpm_df.loc[:,tpm_df.columns.isin(clin_df.index)]
tu_df = tu_df.loc[:,tu_df.columns.isin(clin_df.index)]
clin_df = clin_df.loc[tu_df.columns,:]
# %%
import gseapy as gp
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# ensure sample order alignment
assert all(tpm_df.columns == tu_df.columns)
assert all(tpm_df.columns == clin_df.index)

# -----------------------------
# 1. Transcript → Gene mapping
# -----------------------------
# transcript IDs look like "ENSTxxxx-A1BG"
gene_map = pd.Series([x.split("-",1)[-1] for x in tpm_df.index], index=tpm_df.index)

# -----------------------------
# 2. Make gene-level matrices
# -----------------------------
# (i) Total TPM (all transcripts)
expr_total = tpm_df.groupby(gene_map).sum()

# (ii) Major TPM (filter first → then sum)
expr_major = tpm_df.loc[tpm_df.index.isin(majorlist)]
expr_minor = tpm_df.loc[tpm_df.index.isin(minorlist)]
gene_map_major = gene_map.loc[expr_major.index]
expr_major_gene = expr_major.groupby(gene_map_major).sum()
gene_map_minor = gene_map.loc[expr_minor.index]
expr_minor_gene = expr_minor.groupby(gene_map_minor).mean()

# log transform
expr_total_log = np.log2(expr_total + 1)
expr_major_log = np.log2(expr_major_gene + 1)
expr_hr_log = np.log2((expr_minor_gene+0.001)/(expr_major_gene+0.001))
# 0️⃣ 공통 gene만 사용
common_genes = expr_minor_gene.index.intersection(expr_major_gene.index)

expr_minor_gene = expr_minor_gene.loc[common_genes].copy()
expr_major_gene = expr_major_gene.loc[common_genes].copy()

# 1️⃣ total 계산
expr_total = expr_minor_gene + expr_major_gene

def q10_pos(x):
    x = x[x > 0]
    return np.percentile(x, 1) if len(x) else 0.0

q10_minor = expr_minor_gene.apply(q10_pos, axis=1)
q10_major = expr_major_gene.apply(q10_pos, axis=1)

eps_minor = (0.1 * q10_minor).clip(lower=1e-4)
eps_major = (0.1 * q10_major).clip(lower=1e-4)

eps_minor_mat = pd.DataFrame(np.repeat(eps_minor.values[:,None], expr_minor_gene.shape[1], axis=1),
                             index=expr_minor_gene.index, columns=expr_minor_gene.columns)
eps_major_mat = pd.DataFrame(np.repeat(eps_major.values[:,None], expr_major_gene.shape[1], axis=1),
                             index=expr_major_gene.index, columns=expr_major_gene.columns)

expr_ratio = np.log2((expr_minor_gene + eps_minor_mat) /
                     (expr_major_gene + eps_major_mat))

# -----------------------------
# 4. ssGSEA
# -----------------------------
res_total = gp.ssgsea(
    data=expr_ratio,
    gene_sets="MSigDB_Hallmark_2020", #GO_Biological_Process_2021, MSigDB_Hallmark_2020, KEGG_2021_Human #gp.get_library_name()
    outdir=None,
    sample_norm_method="rank",
    permutation_num=0
)
df_total = res_total.res2d
score_total = df_total.pivot(index="Term", columns="Name", values="ES")

###^^^ 전체 pathway CoxPH #######
from lifelines import CoxPHFitter
import numpy as np
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

results = []


###^^^# score_total or score_major 중 선택
score_df = score_total # 예시: major TPM 기반

for term in score_df.index:
    # pathway score 불러오기
    vals = score_df.loc[term]

    # median split
    group = np.where(vals > vals.median(), "High", "Low")

    # Cox 모델용 데이터프레임
    tmp = clin_df[[
        "PFS", "recur",  ]].copy()
#"drug","BRCAmt", "setting","line_binary",]]
    

    tmp["group"] = (group == "High").astype(int)  # High=1, Low=0
#     for col in ["gBRCA1", "gBRCA2", "tBRCA1", "tBRCA2"]:
#         tmp[col] = (
#             tmp[col]
#             .astype(str)
#             .str.strip()
#             .replace({"-": "0", "2": "0"})   # VUS, missing → 0
#             .replace({"nan": "0", "NaN": "0", "NA": "0", "None": "0"})
#         )
#     tmp["line"] = (
#     tmp["line"]
#     .astype(str)
#     .str.extract(r"(\d+)")[0]   # 숫자 부분만 추출
#     .fillna("0")
#     .astype(int)
# )
#     # 이제 문자열 전부 숫자로 변환 (에러는 NaN으로)
#     tmp[col] = pd.to_numeric(tmp[col], errors="coerce").fillna(0).astype(int)
    # tmp = pd.get_dummies(tmp, columns=["drug"], drop_first=True, dtype=float)
    # tmp = pd.get_dummies(tmp, columns=["setting"], drop_first=True, dtype=float)
    # tmp = pd.get_dummies(tmp, columns=["line_binary"], drop_first=True, dtype=float)
    # tmp = pd.get_dummies(tmp, columns=["BRCAmt"], drop_first=True, dtype=float)


    # 필수 변수 결측 제거
    tmp = tmp.dropna(subset=["PFS", "recur", "group"])    
    
    try:
        cph = CoxPHFitter()
        cph.fit(tmp, duration_col="PFS", event_col="recur")

        hr = np.exp(cph.params_.loc["group"])
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

# %%

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

from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# --- Key survival columns ---
TIME_COL = 'PFS'
EVENT_COL = 'recur'  # 1=event, 0=censored

# --- Prepare Figure for KM Plots ---
n_rows = 2
n_cols = 5
fig_km, axes_km = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows),
                               squeeze=False, facecolor='white')

print("========= CoxPH & Kaplan-Meier Analysis Results =========")

n_sets = len(ddr_genelist)
if n_sets > 10:
    print(f"WARNING: ddr_genelist has {n_sets} items, but layout is fixed to 4x5 for 10 items. Only the first 10 will be plotted.")

for i, (name, genes) in enumerate(ddr_genelist.items()):
    if i >= 10:
        break

    # Base clinical data
    try:
        survival_df = clin_df[[TIME_COL, EVENT_COL, 'BRCAmt', 'drug', 'line_binary']].copy()
    except KeyError as e:
        print(f"!!! ERROR: Missing column: {e} !!!")
        break

    # Convert categorical columns to dummy variables
    survival_df = pd.get_dummies(
        survival_df,
        columns=['BRCAmt', 'drug', 'line_binary'],
        drop_first=True,
        dtype=float
    )

    # Calculate mean expression for valid genes
    valid_genes_major = list(set(genes) & set(expr_minor_gene.index))
    survival_df['mean_major_exp'] = expr_minor_gene.loc[valid_genes_major].mean(axis=0)

    # Drop rows with NA in essential columns
    analysis_df = survival_df.dropna(subset=[TIME_COL, EVENT_COL, 'mean_major_exp'])

    print(f"\n--- Results for Gene Set: {name} (Plot {i+1}/10) ---")

    # --- Plot position ---
    plot_col = i % n_cols
    plot_row = i // n_cols

    # --- Kaplan–Meier Plot ---
    var_to_analyze = 'mean_major_exp'
    ax = axes_km[plot_row, plot_col]
    kmf = KaplanMeierFitter()

    median_val = analysis_df[var_to_analyze].median()
    analysis_df['group'] = (analysis_df[var_to_analyze] > median_val).astype(int)

    high_data = analysis_df[analysis_df['group'] == 1]
    low_data = analysis_df[analysis_df['group'] == 0]

    kmf.fit(low_data[TIME_COL], event_observed=low_data[EVENT_COL], label=f"Low (n={len(low_data)})")
    kmf.plot_survival_function(ax=ax, ci_show=False, color='blue', show_censors=True)
    kmf.fit(high_data[TIME_COL], event_observed=high_data[EVENT_COL], label=f"High (n={len(high_data)})")
    kmf.plot_survival_function(ax=ax, ci_show=False, color='red', show_censors=True)

    lr_result = logrank_test(high_data[TIME_COL], low_data[TIME_COL],
                             high_data[EVENT_COL], low_data[EVENT_COL])
    p_val_km = lr_result.p_value

    ax.set_title(f"{name} - Mean Expr", fontsize=15)
    ax.set_xlabel(TIME_COL)
    if plot_col == 0:
        ax.set_ylabel("Survival Probability")

    # --- Cox Proportional Hazards Model (with covariates) ---
    cph = CoxPHFitter()

    # Extract all numeric columns for regression
    covariates = [c for c in analysis_df.columns if c not in [TIME_COL, EVENT_COL, 'group']]
    try:
        cph.fit(analysis_df[[TIME_COL, EVENT_COL] + covariates],
                duration_col=TIME_COL, event_col=EVENT_COL)
        summary = cph.summary.loc[var_to_analyze]
        hr, ci_low, ci_high, p_val_cox = summary[["exp(coef)", "exp(coef) lower 95%",
                                                  "exp(coef) upper 95%", "p"]]
    except Exception as e:
        print(f"❌ CoxPH fitting error for {name}: {e}")
        continue

    # Annotate results
    ax.text(
        0.05, 0.05,
        f"CoxPH (continuous):\n"
        f"HR = {hr:.2f} (95% CI {ci_low:.2f}–{ci_high:.2f})\n"
        f"Log-rank p = {p_val_km:.3f}\n"
        f"Cox p = {p_val_cox:.3e}",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, boxstyle="round")
    )

    print(f"[{name}] KM log-rank p = {p_val_km:.4f}")
    print(f"[{name}] CoxPH p = {p_val_cox:.4f} | HR = {hr:.3f}")
    if p_val_cox < 0.05:
        print("  -> SIGNIFICANT (CoxPH)")
        print(cph.summary)

plt.tight_layout(pad=2.0)
plt.show()


# %%
from lifelines import CoxPHFitter
import numpy as np
import pandas as pd

# clinical dataframe
clin_df = main_clin[(main_clin['BRCAmut'] == 0)][[TIME_COL, EVENT_COL]].copy()

results = []  # 모든 유전자 결과 저장
significant = []  # p < 0.05만 저장

for name, genes in ddr_genelist.items():
    for gene in genes:
        if gene not in expr_ratio.index:
            continue
        
        df = clin_df.copy()
        df['expr'] = expr_ratio.loc[gene, df.index]
        df = df.dropna(subset=['expr', TIME_COL, EVENT_COL])
        if len(df) < 10:
            continue
        
        try:
            cph = CoxPHFitter()
            cph.fit(df[[TIME_COL, EVENT_COL, 'expr']],
                    duration_col=TIME_COL, event_col=EVENT_COL)
            summary = cph.summary.loc['expr']
            hr = summary["exp(coef)"]
            ci_low = summary["exp(coef) lower 95%"]
            ci_high = summary["exp(coef) upper 95%"]
            pval = summary["p"]
            logp = -np.log10(pval) if pval > 0 else np.inf

            res = {
                "GeneSet": name,
                "Gene": gene,
                "HR": hr,
                "CI_low": ci_low,
                "CI_high": ci_high,
                "pval": pval,
                "log10p": logp,
                "n_samples": len(df)
            }
            results.append(res)
            if pval < 0.05:
                significant.append(res)

        except Exception as e:
            continue

# --- Convert to DataFrame ---
cox_results_df = pd.DataFrame(results)
sig_df = pd.DataFrame(significant).sort_values("pval")


print(f"Total genes analyzed: {len(cox_results_df)}")
print(f"Significant genes (p < 0.05): {len(sig_df)}\n")
print(sig_df[["GeneSet", "Gene", "HR", "pval", "log10p"]].to_string(index=False))