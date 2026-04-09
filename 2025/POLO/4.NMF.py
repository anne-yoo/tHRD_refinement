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
from sklearn.decomposition import NMF
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# %%
val_clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2025clinical/83_POLO_clinicalinfo.txt', sep='\t', index_col=0)
LOHscore = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2025clinical/LOHscore_ACT.txt', sep='\t', index_col=0)
geneexp = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/merged_83_gene_TPM.txt', sep='\t', index_col=0)
transexp = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/merged_83_transcript_TPM.txt', sep='\t', index_col=0)

val_clin = pd.merge(val_clin, LOHscore, left_index=True, right_index=True, how='left')
majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/merged_83_cov5_majorminorlist.txt', sep='\t')
sqanti = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/sqantioutput/sqanti_classification.txt', sep='\t')
sqanti.dropna(axis=1, how='all', inplace=True)

coding = sqanti[['isoform','coding','structural_category']]
majorminor = pd.merge(majorminor, coding, left_on='transcriptid', right_on='isoform', how='left')

group1 = majorminor[(majorminor['type']=='major')&(majorminor['coding']=='coding') & (majorminor['structural_category'].isin(['full-splice_match','novel_in_catalog']))]['Transcript-Gene'].to_list()
group2 = majorminor[(majorminor['type']=='minor')&(majorminor['coding']=='coding')]['Transcript-Gene'].to_list()
group3 = majorminor[(majorminor['type']=='minor')&(majorminor['coding']!='coding')]['Transcript-Gene'].to_list()

# %%
res_list = val_clin[(val_clin['PFS']>=365)].index.to_list()
nonres_list = val_clin[(val_clin['recur']==1)&(val_clin['PFS']<365)].index.to_list()

tx_counts = transexp.copy()
tx_counts = tx_counts.iloc[:,:-1]
tx_meta = majorminor.copy() 

#############################** major coding vs. just coding ##########################################
functional_tx = majorminor[(majorminor['type']=='major') & (majorminor['coding']=='coding') & (majorminor['structural_category'].isin(['full-splice_match','novel_in_catalog']))]['Transcript-Gene'].to_list()

# functional_tx = majorminor[(majorminor['coding']=='coding') & (majorminor['structural_category'].isin(['full-splice_match','novel_in_catalog']))]['transcriptid'].to_list()

#functional_tx = majorminor[(majorminor['coding']=='non_coding') ]['Transcript-Gene'].to_list()
###########################**##########################################################################

# Filter the count matrix
######################################################** whole vs. only major ##########################
tx_counts_filtered = tx_counts.loc[tx_counts.index.intersection(functional_tx)]
#tx_counts_filtered = tx_counts.copy()
#########################**#############################################################################

print(f"Original Transcripts: {tx_counts.shape[0]}")
print(f"Functional Transcripts: {tx_counts_filtered.shape[0]}")

# 3. Aggregate (Sum) to Gene Level
# Map transcript index to gene ID
tx_to_gene_map = tx_meta.set_index('Transcript-Gene')['genename']
tx_counts_filtered['genename'] = tx_counts_filtered.index.map(tx_to_gene_map)

# Group by Gene ID and Sum
gene_counts_functional = tx_counts_filtered.groupby('genename').sum()

print(f"Final Functional Genes: {gene_counts_functional.shape[0]}")

# ==========================================
# PART 2: Differential Expression (PyDESeq2)
# ==========================================

# 1. Prepare Metadata (Clinical Data)
# Ensure the index matches the columns of gene_counts_functional
clinical_df = val_clin.copy()
clinical_df = clinical_df.loc[clinical_df.index.isin(res_list+nonres_list),:]
clinical_df['group'] = 'responder'
clinical_df.loc[clinical_df.index.isin(nonres_list),'group'] = 'nonresponder'

# Filter clinical data to match the samples in counts (N=45)
common_samples = gene_counts_functional.columns.intersection(clinical_df.index)
counts_matrix = gene_counts_functional[common_samples].T 
metadata = clinical_df.loc[common_samples]

#%%
####^^ ssGSEA scoring ################
import gseapy as gp
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# geneexpdf = geneexp.copy()
# geneexpdf = geneexpdf.loc[counts_matrix.columns.to_list(),common_samples]
# geneexpdf = geneexpdf.T

codinggene = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM_symbol.txt', sep='\t')
codinggenelist = codinggene['Gene Symbol'].to_list()
geneexpdf_full = geneexp.copy()
geneexpdf_full = geneexpdf_full.loc[geneexpdf_full.index.isin(codinggenelist),common_samples]
geneexpdf_full = geneexpdf_full.T

expr_full_log = np.log2(geneexpdf_full + 0.1)
expr_full_log = expr_full_log.T

expr_major_log = np.log2(counts_matrix + 0.1)
expr_major_log = expr_major_log.T

res_major = gp.ssgsea(
    data=expr_major_log,
    gene_sets="GO_Biological_Process_2021", #GO_Biological_Process_2021, MSigDB_Hallmark_2020, Reactome_2022
    outdir=None,
    sample_norm_method="rank",
    permutation_num=0
)
df_major = res_major.res2d
score_major = df_major.pivot(index="Term", columns="Name", values="ES")

res_gene = gp.ssgsea(
    data=expr_full_log,
    gene_sets="GO_Biological_Process_2021", #GO_Biological_Process_2021, MSigDB_Hallmark_2020
    outdir=None,
    sample_norm_method="rank",
    permutation_num=0
)
df_gene = res_gene.res2d
score_gene = df_gene.pivot(index="Term", columns="Name", values="ES")

#%%
clin_df = metadata.copy()
dna_major = score_gene.loc["alternative mRNA splicing, via spliceosome (GO:0000380)"] ##^score_major vs. score_gene
#double-strand break repair via homologous recombination (GO:0000724) #DNA Repair #alternative mRNA splicing, via spliceosome (GO:0000380), #HDR Thru Homologous Recombination (HRR) R-HSA-5685942

clin_df['AS Score'] = dna_major
clin_df['HRR Score'] = score_major.loc["double-strand break repair via homologous recombination (GO:0000724)"]
clin_df["dna_major_group"] = np.where(dna_major > dna_major.median(), "High", "Low")

from lifelines.plotting import add_at_risk_counts  # <--- Essential import

for group in ["dna_major_group"]:
    plt.figure(figsize=(6, 5)) 
    ax = plt.subplot(111)

    colors = {"High": "#DA4343", "Low": "#3396D3"}
    fitters = [] 

    # Enforce order "High" -> "Low"
    target_labels = ["High", "Low"] 

    for label in target_labels:
        kmf = KaplanMeierFitter()
        idx = clin_df["dna_major_group"] == label
        
        # Fit data
        kmf.fit(clin_df.loc[idx, "PFS"], clin_df.loc[idx, "recur"], label=label)
        
        # Plot
        kmf.plot_survival_function(
            ax=ax, 
            ci_show=False, 
            color=colors.get(label, "gray"),
            linewidth=2,
            show_censors=True
        )
        fitters.append(kmf)

    # --- Statistics (Log-rank & Cox) ---
    idx_high = clin_df["dna_major_group"] == "High"
    res = logrank_test(
        clin_df.loc[idx_high, "PFS"], clin_df.loc[~idx_high, "PFS"],
        clin_df.loc[idx_high, "recur"], clin_df.loc[~idx_high, "recur"]
    )

    # Cox HR
    tmp = clin_df[["PFS", "recur", "dna_major_group"]].copy()
    tmp["dna_major_group"] = (tmp["dna_major_group"] == "High").astype(int)
    cph = CoxPHFitter()
    cph.fit(tmp, duration_col="PFS", event_col="recur")
    hr = np.exp(cph.params_)[0]
    ci = cph.confidence_intervals_.iloc[0].tolist()

    # --- Annotations ---
    plt.title("DSB Repair via HR (gene abundunce)") #double-strand break repair via homologous recombination (GO:0000724) #alternative mRNA splicing, via spliceosome (GO:0000380) #HDR Thru Homologous Recombination (HRR) R-HSA-5685942
    plt.text(0.05, 0.15, f"Log-rank p = {res.p_value:.4f}\nHR = {hr:.2f} ({ci[0]:.2f} - {ci[1]:.2f})",
            transform=ax.transAxes, fontsize=11, fontweight='regular')

    # Hide the default x-label because it overlaps with the table
    plt.ylabel('PFS Probability')
    plt.xlabel('')
    # # --- Add Risk Table (Customized) ---
    # # 1. Select rows: 'at_risk' is standard. 'events' counts recurrences. 
    # #    We exclude 'censored' to save space and avoid confusion.
    # add_at_risk_counts(*fitters, ax=ax, rows_to_show=['At risk', 'Events'])

    # # --- Styling the Risk Table & Layout ---

    # # 1. Adjust margins to prevent cutting off the table
    # #    'bottom=0.25' reserves 25% of the figure height for the table
    # plt.subplots_adjust(bottom=0.1) 
    # ax.spines['bottom'].set_visible(False)

    # # 2. Iterate through axes to find the table and adjust Font Size
    # #    The table is added as new axes to the figure.
    # for a in plt.gcf().axes:
    #     if a is not ax: # Identify the table axes (they are not the main ax)
    #         # Adjust tick labels (The numbers)
    #         a.tick_params(axis='x', labelsize=11) 
    #         # Adjust y-labels (The row names like "At risk")
    #         a.tick_params(axis='y', labelsize=11)
    #         # Rename row labels if you want specific Korean/English terms
    #         # (This is tricky but modifying the text objects works)
    #         yticks = a.get_yticklabels()
    #         for label in yticks:
    #             if "At risk" in label.get_text():
    #                 label.set_text("No. at Risk")
    #             elif "Events" in label.get_text():
    #                 label.set_text("Cumulative Recur") # Rename 'Events' to 'Recur'
    #         a.set_yticklabels(yticks)
    #         a.spines["top"].set_visible(False)
    #         a.spines["bottom"].set_visible(False)
    #         a.spines["left"].set_visible(False)
    #         a.spines["right"].set_visible(False)
    # # 3. Manually place the X-label at the very bottom
    sns.despine()
    #plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/fixed_HR_gene.pdf", dpi=300, bbox_inches='tight')
    plt.show()


#%% ###^^ correlation plot LOH score ~ splicing pathway score ##############
clin_df['gHRDscore'] = clin_df['ACT_LOHscore'].astype(float)
clin_df['gHRD_new'] = np.where(clin_df['gHRDscore'] >= 0.4, 1, 0)


# 색상 정의
colors = {
    "gHRD+": "#3396D3",   # red
    "gHRD−": "#DA4343"    # blue
}

# ----------------------------
# 예시 clin_df (함의 데이터 포맷)
# ----------------------------
# clin_df = pd.read_csv("clinical_data.tsv", sep="\t")

# gHRD status 라벨
clin_df["gHRD_status"] = clin_df["gHRD_new"].map({1: "gHRD+", 0: "gHRD−"})
clin_df["AS_status"] = clin_df["dna_major_group"].map({"High": 1, "Low": 0})

# ====================================================
# ① gHRD vs 재발 여부 (barplot)
# ====================================================
plt.figure(figsize=(3,4))
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
#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/78_gHRD_recur.pdf", dpi=300)
plt.show()

# ====================================================
# ② Kaplan–Meier curve + CoxPH + p-value 표시
# ====================================================
kmf = KaplanMeierFitter()

plt.figure(figsize=(5,5))
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


plt.title("gHRD status", fontsize=12, fontweight="bold")
plt.xlabel("Days")
plt.ylabel("Survival probability")

# Log-rank test
group_pos = clin_df[clin_df["gHRD_status"] == "gHRD+"]
group_neg = clin_df[clin_df["gHRD_status"] == "gHRD−"]
lr = logrank_test(group_pos["PFS"], group_neg["PFS"],
                  event_observed_A=group_pos["recur"],
                  event_observed_B=group_neg["recur"])

# CoxPH for HR
cox_df = clin_df[["PFS", "recur", "gHRD_new"]].copy()
cph = CoxPHFitter()
cph.fit(cox_df, duration_col="PFS", event_col="recur")
summary = cph.summary.loc["gHRD_new"]
hr = summary["exp(coef)"]
ci_low = summary["exp(coef) lower 95%"]
ci_high = summary["exp(coef) upper 95%"]
pval = summary["p"]

# Plot annotation
plt.text(
    0.55 * clin_df["PFS"].max(), 0.8,
    f"HR = {hr:.2f} (95% CI {ci_low:.2f}–{ci_high:.2f})\n"
    f"Log-rank p = {lr.p_value:.4f}",
    fontsize=11,
    bbox=dict(facecolor="white", alpha=0.7, boxstyle="round")
)

plt.legend(title="gHRD status", frameon=False)
plt.tight_layout()
sns.despine()
#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/78_gHRD_KMplot.pdf", dpi=300)
plt.show()

print(f"Log-rank p-value: {lr.p_value:.4f}")
print(cph.summary)

kmf = KaplanMeierFitter()
# 색상 정의
colors = {
    "gHRD+": "#3396D3",   # red 
    "gHRD−": "#DA4343"    # blue
}

colors = {
    "Low": "#3396D3",   # red 
    "High": "#DA4343"    # blue
}

plt.figure(figsize=(6,5))
for status, color in colors.items():
    mask = clin_df["dna_major_group"] == status
    T = clin_df.loc[mask, "PFS"]
    E = clin_df.loc[mask, "recur"]  # 1=event, 0=censored
    
    kmf.fit(T, E, label=status)
    kmf.plot_survival_function(
        ci_show=False,
        color=color,
        linewidth=2,
        show_censors=True,   # ✅ censor 자동 표시 (+)
    )


plt.title("ssGSEA AS score", fontsize=12, fontweight="bold")
plt.xlabel("Days")
plt.ylabel("Survival probability")

# Log-rank test
group_pos = clin_df[clin_df["dna_major_group"] == "High"]
group_neg = clin_df[clin_df["dna_major_group"] == "Low"]
lr = logrank_test(group_pos["PFS"], group_neg["PFS"],
                  event_observed_A=group_pos["recur"],
                  event_observed_B=group_neg["recur"])

# CoxPH for HR
cox_df = clin_df[["PFS", "recur", "AS_status"]].copy()
cph = CoxPHFitter()
cph.fit(cox_df, duration_col="PFS", event_col="recur")
summary = cph.summary.loc["AS_status"]
hr = summary["exp(coef)"]
ci_low = summary["exp(coef) lower 95%"]
ci_high = summary["exp(coef) upper 95%"]
pval = summary["p"]

# Plot annotation
plt.text(
    0.55 * clin_df["PFS"].max(), 0.6,
    f"HR = {hr:.2f} (95% CI {ci_low:.2f}–{ci_high:.2f})\n"
    f"Log-rank p = {lr.p_value:.4f}",
    fontsize=11,
    bbox=dict(facecolor="white", alpha=0.7, boxstyle="round")
)

plt.legend(title="AS score", frameon=False)
plt.tight_layout()
sns.despine()
#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/geneexp_ASscore_KMplot.pdf", dpi=300)
plt.show()

print(f"Log-rank p-value: {lr.p_value:.4f}")
print(cph.summary)

#%%
###^ timepoint dependent ROC curve ###
import numpy as np
import matplotlib.pyplot as plt
from sksurv.metrics import cumulative_dynamic_auc

# 1. scikit-survival 형식에 맞게 데이터 준비 (structured array)
# 'event'는 반드시 bool 타입이어야 안정적으로 작동해.
y_train = np.array([(bool(status), time) for status, time in zip(clin_df['recur'], clin_df['PFS'])],
                   dtype=[('event', 'bool'), ('time', 'float')])

# 2. 분석을 진행할 시점(Time points) 설정
# 데이터의 PFS 범위 내에서 50일 간격으로 AUC를 계산해.
# 데이터가 적은 극초반과 극후반은 제외하는 것이 통계적으로 더 정확해.
times = np.arange(100, clin_df['PFS'].max() - 100, 50) 

# 3. 각 점수별 Time-dependent AUC 계산
# gHRD와 AS_score는 수치형(float) 데이터여야 해.
auc_gHRD, mean_auc_gHRD = cumulative_dynamic_auc(y_train, y_train, clin_df['gHRD'].values, times)
auc_AS, mean_auc_AS = cumulative_dynamic_auc(y_train, y_train, clin_df['AS Score'].values, times)

# 4. 시각화 (Figure 1 - Panel h)
plt.figure(figsize=(5, 5))

# gHRD: 파란색 점선 (기존 지표)
plt.plot(times, auc_gHRD, marker="o", markersize=4, linestyle="-", color="#3182bd", 
         label=f"gHRD (Mean={mean_auc_gHRD:.2f})")

# AS score: 빨간색 실선 (우리의 새로운 지표)
plt.plot(times, auc_AS, marker="s", markersize=4, linestyle="-", color="#de2d26", 
         label=f"AS score (Mean={mean_auc_AS:.2f})")

plt.axhline(0.5, color='gray', linestyle=':', alpha=0.5) # 무작위 예측 기준선
plt.xlabel("PFS", fontsize=11)
plt.ylabel("Time-dependent AUC", fontsize=11)
plt.title("Time-dependent ROC", fontsize=11, fontweight="bold")
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.2)
plt.ylim(0, 1.0) # 차이를 명확히 보여주기 위해 범위를 0.4~1.0으로 설정

plt.tight_layout()
sns.despine()
#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/geneexp_timedependtROC.pdf", dpi=300)
plt.show()


#%%
###^ 2x2 Quadrant Survival Analysis (gHRD vs AS score) ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

# 1. 그룹 나누기 (Median 기준 또는 특정 Cut-off 사용)
# gHRD가 이미 High/Low로 되어있다면 그대로 쓰고, 점수라면 아래처럼 변환해.
clin_df['gHRD_status'] = np.where(clin_df['gHRD_new']==1 , 'gHRD+', 'gHRD-')

# 2. 4개의 Quadrant 그룹 생성
def define_quadrant(row):
    return f"{row['gHRD_status']} / {row['dna_major_group']}"

clin_df['Quadrant'] = clin_df.apply(define_quadrant, axis=1)
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

# 1. 시각화 설정
plt.figure(figsize=(6, 5))
kmf = KaplanMeierFitter()

# 네가 정의한 Quadrant 이름에 맞춰 색상 매핑 (가장 드라마틱한 대조를 위해)
colors = {
    'gHRD+ / Low': '#3396D3',   # Genomic 손상 + Splicing 낮음 (Best Responder 예상)
    'gHRD- / Low': '#094E78',    # Genomic 정상 + Splicing 낮음
    'gHRD+ / High': '#DA4343',  # Genomic 손상 + Splicing 높음 (Compensated)
    'gHRD- / High': '#7E1414'    # Genomic 정상 + Splicing 높음 (Worst)
}

n_counts = {
    'gHRD+ / High': 32,
    'gHRD+ / Low': 23,
    'gHRD- / Low': 16,
    'gHRD- / High': 7
}

# 2. 각 그룹별 KM Curve 그리기
for group, color in colors.items():
    mask = (clin_df['Quadrant'] == group)
    n = n_counts.get(group, mask.sum())
    if mask.sum() > 0:
        kmf.fit(clin_df.loc[mask, 'PFS'], clin_df.loc[mask, 'recur'], label=group)
        kmf.plot_survival_function(ax=plt.gca(), color=color, ci_show=False, linewidth=2, show_censors=True)
        last_time = kmf.survival_function_.index[-1]
        last_prob = kmf.survival_function_.iloc[-1, 0]
        
        # 텍스트 추가 (약간의 x축 여백을 줌)
        plt.text(last_time + 15, last_prob-0.02, f"n={n}", 
                 fontsize=9, color='black', va='center')

# 3. Multivariate Log-rank Test (4개 그룹 전체 비교)
results = multivariate_logrank_test(clin_df['PFS'], clin_df['Quadrant'], clin_df['recur'])
p_val = results.p_value

# 4. 그래프 디테일 설정
plt.title("gHRD & HRR Score", fontsize=14, fontweight='bold')
plt.xlabel("Days", fontsize=12)
plt.ylabel("PFS Probability", fontsize=12)
plt.legend(title='gHRD / HRR Score', loc='lower left', fontsize=10, title_fontsize=10, frameon=True)

# # P-value 및 샘플 수 정보 표기
# plt.text(clin_df['PFS'].max()*0.55, 0.75, 
#          f"Overall Log-rank p = {p_val:.4f}\n"
#          f"n = {len(clin_df)}", 
#          fontsize=12, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

plt.tight_layout()
sns.despine()
plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/HRRscore_quandrant_survival.pdf", dpi=300)
plt.show()

# 1. 데이터 준비 및 Z-score 정규화
# 분석용 데이터프레임 복사
cox_interaction_df = clin_df[['PFS', 'recur', 'AS Score', 'gHRD']].copy()

# AS_score를 Z-score로 변환 (해석의 용이성을 위해)
# Z-score = (x - mean) / std
mean_val = cox_interaction_df['AS Score'].mean()
std_val = cox_interaction_df['AS Score'].std()
cox_interaction_df['score_z'] = (cox_interaction_df['AS Score'] - mean_val) / std_val

# 2. 상호작용 항(Interaction Term) 생성
# score_z : gHRD
cox_interaction_df['interaction_term'] = cox_interaction_df['score_z'] * cox_interaction_df['gHRD']

# 3. Cox Proportional Hazards Model Fit
cph = CoxPHFitter()
# 공식: PFS ~ score_z + gHRD + interaction_term
cph.fit(cox_interaction_df[['PFS', 'recur', 'score_z', 'gHRD', 'interaction_term']], 
        duration_col='PFS', event_col='recur')

# 4. 결과 출력
print("--- Interaction Model Summary ---")
cph.print_summary()

# 5. 주요 지표 추출
summary = cph.summary
interaction_p = summary.loc['interaction_term', 'p']
score_z_hr = summary.loc['score_z', 'exp(coef)']

print(f"\nInteraction p-value: {interaction_p:.4f}")

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

# =========================
# input
# =========================
x = clin_df['AS Score']
y = clin_df['ACT_LOHscore']

# drop NaN pairwise
mask = x.notna() & y.notna()
x = x[mask].values
y = y[mask].values

# =========================
# statistics
# =========================
rho, pval = spearmanr(x, y)

# linear regression (for visualization only)
X = x.reshape(-1, 1)
model = LinearRegression()
model.fit(X, y)

xx = np.linspace(x.min()+1600, x.max(), 200)
yy = model.predict(xx.reshape(-1, 1))

# =========================
# plot
# =========================
fig, ax = plt.subplots(figsize=(4.5, 4.5))

ax.scatter(
    x, y,
    s=40,
    alpha=0.8,
    linewidth=0,
    color='#467707'  # blue shade
)

ax.plot(
    xx, yy,
    linestyle='--',
    linewidth=0.8,
    color='#1D3400'
)

ax.set_xlabel('AS Score', fontsize=12)
ax.set_ylabel('gHRD score', fontsize=12)

# annotation
text = (
    f"Spearman r = {rho:.2f}\n"
    f"p = {pval:.2e}"
)

ax.text(
    0.05, 0.95,
    text,
    transform=ax.transAxes,
    ha='left',
    va='top',
    fontsize=11
)

ax.tick_params(labelsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlim(4300,6800)
plt.tight_layout()
plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/gHRD_ASscore_corr.pdf", dpi=300)
plt.show()



#%%
###^^^ 전체 pathway CoxPH #######
from lifelines import CoxPHFitter
import numpy as np
import matplotlib as mpl

results = []

###^^^# score_total or score_major 중 선택
score_df = score_gene # 예시: major TPM 기반

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

# results = []
# for term in score_gene.index:
#     # pathway score 불러오기
#     vals = score_gene.loc[term]

#     # median split
#     group = np.where(vals > vals.median(), "High", "Low")

#     # Cox 모델용 데이터프레임
#     tmp = clin_df[["PFS", "recur"]].copy()
#     tmp["group"] = (group == "High").astype(int)  # High=1, Low=0

#     try:
#         cph = CoxPHFitter()
#         cph.fit(tmp, duration_col="PFS", event_col="recur")
#         hr = np.exp(cph.params_)[0]
#         ci_low, ci_high = np.exp(cph.confidence_intervals_.iloc[0]).tolist()
#         pval = cph.summary["p"].values[0]

#         results.append({
#             "Pathway": term,
#             "HR": hr,
#             "CI_low": ci_low,
#             "CI_high": ci_high,
#             "pval": pval
#         })
#     except Exception as e:
#         print(f"Skipped {term} due to error: {e}")

# # 결과 DataFrame
# res_gene = pd.DataFrame(results)

# 필터링: High 위험 (HR>1) + p<0.05
sig_df_high = res_df[(res_df["HR"] > 1) & (res_df["pval"] < 0.05)].sort_values("pval")
sig_df_low = res_df[(res_df["HR"] < 1) & (res_df["pval"] < 0.05)].sort_values("pval")


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

#%%

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
    rotation=30,     # 1. 45도 대신 90도로 회전
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


plt.subplots_adjust(bottom=0.3) # X축 라벨이 잘리지 않도록 하단 여백 확보
sns.despine()

#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/GOBP2021_major_barplot_codinggene_Top20.pdf", dpi=300,bbox_inches='tight')
plt.show()

#%%
###^^ only high risk ##################
####^^ only high HR for SNU #########################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# --- df_high 준비는 기존과 동일하다고 가정 ---
df_high = sig_df_high.nsmallest(10, "pval").copy()
df_high["log2HR"] = np.log2(df_high["HR"])
df_high["neglog10p"] = -np.log10(df_high["pval"])
df_high = df_high.sort_values("HR", ascending=False)

# --- X축 설정 ---
pathways = list(df_high["Pathway"])
x_pos_high = np.arange(len(df_high))

# --- Figure 설정 ---
fig, ax = plt.subplots(figsize=(8, 6))

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
    rotation=20,
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
plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/GOBP_major_barplot_onlyhighHR_Top10.pdf", dpi=300,bbox_inches='tight')
plt.show()


#%%
################^^^^^^^^^^^^ forest plot gene vs. transcript exp ####################################
res_df['Metric'] = 'Functional Transcript'
res_gene['Metric'] = 'Gene Expression'

res_func = res_df.copy()
res_func["Metric"] = "Functional Transcript"

# Top 20 by HR (or pval, 네가 쓰던 기준 그대로)
top20 = (
    res_func
    .sort_values("HR", ascending=False)
    .head(20)["Pathway"]
    .tolist()
)

res_gene_20 = res_gene[res_gene["Pathway"].isin(top20)].copy()
res_gene_20 = res_gene_20.drop_duplicates()

res_all = pd.concat([res_func[res_func["Pathway"].isin(top20)], res_gene_20], axis=0)

# log scale
res_all["logHR"] = np.log(res_all["HR"])
res_all["logCI_low"] = np.log(res_all["CI_low"])
res_all["logCI_high"] = np.log(res_all["CI_high"])
order = (
    res_func
    .set_index("Pathway")
    .loc[top20]              # 순서 보존
    .sort_values("HR", ascending=False)
    .index
    .tolist()
)

res_all["Pathway"] = pd.Categorical(
    res_all["Pathway"],
    categories=order,
    ordered=True
)
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8.5, 9))

y_base = np.arange(len(order))
offset = 0.18

colors = {
    "Functional Transcript": "#2ca02c",  # green
    "Gene Expression": "#ff7f0e"          # orange
}

for metric in ["Functional Transcript", "Gene Expression"]:
    df_m = (
        res_all
        [res_all["Metric"] == metric]
        .set_index("Pathway")
        .loc[order]
    )

    x = df_m["logHR"].values
    y = y_base + (offset if metric == "Functional Transcript" else -offset)

    ax.errorbar(
        x,
        y,
        xerr=[
            x - df_m["logCI_low"].values,
            df_m["logCI_high"].values - x
        ],
        fmt='o',
        color=colors[metric],
        capsize=3,
        markersize=5,
        linewidth=1.2,
        label=metric
    )

# HR = 1 기준선
ax.axvline(0, color='black', linestyle='--', linewidth=1)

# ===== 핵심 조정 =====
ax.set_xlim(-1, 2.7)              # CI가 숨 쉬게
ax.set_yticks(y_base)
ax.set_yticklabels(order, fontsize=9)
ax.set_xlabel("log(Hazard Ratio)", fontsize=10)

# Spine 정리
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 레이아웃 여백 확보 (legend + 긴 pathway)
plt.subplots_adjust(left=0.45, right=0.78)
#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/gene_major_forestplot.pdf", dpi=300,bbox_inches='tight')
plt.show()

#%%
#####^^^ 
dna_major = score_gene.loc["double-strand break repair via homologous recombination (GO:0000724)"] ##^score_major vs. score_gene
#double-strand break repair via homologous recombination (GO:0000724) #DNA Repair #alternative mRNA splicing, via spliceosome (GO:0000380), #HDR Thru Homologous Recombination (HRR) R-HSA-5685942

clin_df['AS Score'] = score_gene.loc["alternative mRNA splicing, via spliceosome (GO:0000380)"]
clin_df['HRR Score'] = score_major.loc["double-strand break repair via homologous recombination (GO:0000724)"]
clin_df['AS Score'] = pd.to_numeric(clin_df['AS Score'], errors='coerce')
clin_df['HRR Score'] = pd.to_numeric(clin_df['HRR Score'], errors='coerce')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr # 데이터 분포가 비모수적일 가능성이 커서 Spearman 추천

# 1. 상관계수 및 p-value 계산
# AS Score (Gene-level Splicing) vs HRR Score (Functional-level HRR)
rho, pval = spearmanr(clin_df['AS Score'], clin_df['HRR Score'])

# 2. 시각화 (Scatter plot)
plt.figure(figsize=(5, 5))
sns.regplot(
    data=clin_df, x='AS Score', y='HRR Score',
    scatter_kws={'alpha':0.7, 'color':'#1D8D17', 'edgecolor':None, 's':40, 'linewidths':0},
    line_kws={'color':'#0B5000', 'linestyle':'--', 'linewidth':0.7}
)

# 3. 그래프 디테일 설정
plt.xlabel("AS Score (gene exp)", fontsize=11)
plt.ylabel("HRR Score (functional tx exp)", fontsize=11)

# 통계치 표기 (Spearman r과 p-value)
stats_text = f"Spearman r = {rho:.2f}\np = {pval:.2e}"
plt.text(
    0.05, 0.95, stats_text, 
    transform=plt.gca().transAxes, 
    fontsize=11, verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.7, boxstyle='round')
)

plt.xlim(4500,6700)
plt.ylim(0,4300)
plt.tight_layout()
sns.despine()
#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/AS_HRR_corr.pdf", dpi=300,bbox_inches='tight')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter

# 1. 데이터 준비 및 Z-score 표준화 (per 1 SD)
# PFS, recur, AS Score, HRR Score가 들어있는 clin_df 사용
cox_multi_df = clin_df[['PFS', 'recur', 'AS Score', 'HRR Score']].copy()

# Z-score 변환 함수
def standardize(series):
    return (series - series.mean()) / series.std()

cox_multi_df['AS_z'] = standardize(cox_multi_df['AS Score'])
cox_multi_df['HRR_FTE_z'] = standardize(cox_multi_df['HRR Score'])

# 2. Multivariate Cox PH Model Fit
cph = CoxPHFitter()
cph.fit(cox_multi_df[['PFS', 'recur', 'AS_z', 'HRR_FTE_z']], 
        duration_col='PFS', event_col='recur')

# 3. Forest Plot 시각화
plt.figure(figsize=(6, 3))
cph.plot(hazard_ratios=True,  marker='s') # Hazard Ratio 모드로 그리기

# 4. 그래프 디테일 및 수치 표기 (HR, CI, p-val)
summary = cph.summary
y_ticks = [0, 1]
labels = ['HRR_FTE_z', 'AS_z']

for i, label in enumerate(labels):
    hr = summary.loc[label, 'exp(coef)']
    lower = summary.loc[label, 'exp(coef) lower 95%']
    upper = summary.loc[label, 'exp(coef) upper 95%']
    p = summary.loc[label, 'p']
    
    text = f"HR = {hr:.2f} ({lower:.2f}-{upper:.2f}), p = {p:.4e}"
    plt.text(upper + 0.1, i, text, va='center', fontsize=11, fontweight='bold')

plt.axvline(1, color='red', linestyle='--') # HR=1 기준선
plt.title("Multivariate Cox Regression: AS vs. Functional HRR", fontsize=13, fontweight='bold', pad=20)
plt.xlabel("Hazard Ratio (95% CI) per 1 SD increase", fontsize=11)
plt.yticks(y_ticks, labels)
plt.grid(axis='x', alpha=0.3)
sns.despine()
plt.show()

#%%
####^^^^^^^^ Random Survival Forest ##################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import cumulative_dynamic_auc
from sklearn.model_selection import KFold, GridSearchCV
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import cumulative_dynamic_auc
from sklearn.model_selection import KFold, GridSearchCV
from lifelines import KaplanMeierFitter, CoxPHFitter

# 1. 모델 피처 셋 정의
model_sets = {
    "M1: gHRD": ["gHRDscore"],
    "M2: AS+HRR": ["AS Score", "HRR Score"],
    "M3: Combined": ["gHRDscore", "AS Score", "HRR Score"]
}

# 2. RSF 하이퍼파라미터 튜닝
param_grid = {'n_estimators': [500, 1000], 'max_depth': [3, 5], 'min_samples_leaf': [9, 12, 15]}
gscv = GridSearchCV(RandomSurvivalForest(random_state=42), param_grid, cv=5, n_jobs=-1)
gscv.fit(X, y)
best_rsf = gscv.best_estimator_

# 3. 교차 검증 및 성능 측정
kf = KFold(n_splits=5, shuffle=True, random_state=42)
c_indices = {name: [] for name in model_sets.keys()}
cv_risk_scores = np.zeros(len(X))

for name, feat_list in model_sets.items():
    for train_idx, test_idx in kf.split(X):
        best_rsf.fit(X.iloc[train_idx][feat_list], y[train_idx])
        c_indices[name].append(best_rsf.score(X.iloc[test_idx][feat_list], y[test_idx]))
        if name == "M3: Combined":
            cv_risk_scores[test_idx] = best_rsf.predict(X.iloc[test_idx][feat_list])

# ==========================================
# Panel A: C-index 비교 저장
# ==========================================
plt.figure(figsize=(6, 6))
sns.barplot(data=pd.DataFrame(c_indices), color = "#5C974D", capsize=.1)
plt.title("Predictive Accuracy (C-index)",  fontsize=12)
plt.ylabel("Harrell's C-index", fontsize=12)
plt.ylim(0.5, 0.85)
plt.grid(axis='y', alpha=0.3)
sns.despine()
plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/Fig4_cindex.pdf", dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# Panel B: Time-dependent AUC 비교 저장
# ==========================================
plt.figure(figsize=(6, 6))
times = np.arange(300, clin_df['PFS'].max() - 100, 50)
colors = ["#3182bd", "#fd8d3c", "#de2d26"] # 모델 수에 맞춰 색상 조정

for (name, feat_list), color in zip(model_sets.items(), colors):
    best_rsf.fit(X[feat_list], y)
    auc, _ = cumulative_dynamic_auc(y, y, best_rsf.predict(X[feat_list]), times)
    plt.plot(times, auc, "o-", color=color, label=f"{name} (Mean={np.mean(auc):.2f})", lw=2)

plt.ylim(0.3, 1)
plt.axhline(0.5, color='gray', linestyle='--', lw=0.7)
plt.title("Time-dependent AUC", fontsize=12)
plt.xlabel("Days", fontsize=11)
plt.ylabel("Cumulative Dynamic AUC", fontsize=12)
plt.legend(fontsize=10, loc='lower right')
plt.grid(alpha=0.2)
sns.despine()
plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/Fig4_timeAUC.pdf", dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# Panel D: 최종 모델 Risk Stratification 저장
# ==========================================
plt.figure(figsize=(6, 6))
clin_df['Final_Risk'] = cv_risk_scores
median_risk = clin_df['Final_Risk'].median()
clin_df['Risk_Group'] = np.where(clin_df['Final_Risk'] > median_risk, 'High', 'Low')

# Cox 모델 통계치 계산
cph = CoxPHFitter()
stat_df = clin_df[['PFS', 'recur', 'Risk_Group']].copy()
stat_df['is_high'] = (stat_df['Risk_Group'] == 'High').astype(int)
cph.fit(stat_df[['PFS', 'recur', 'is_high']], 'PFS', 'recur')
hr = cph.hazard_ratios_[0]
p_val = cph.summary.loc['is_high', 'p']
ci_lower, ci_upper = np.exp(cph.confidence_intervals_.iloc[0])

# KM Plot
kmf = KaplanMeierFitter()
for g, c in zip(['High', 'Low'], ['#de2d26', '#3182bd']):
    mask = clin_df['Risk_Group'] == g
    kmf.fit(clin_df.loc[mask, 'PFS'], clin_df.loc[mask, 'recur'], label=f"{g} Risk")
    kmf.plot_survival_function(color=c, ci_show=False, linewidth=2.5, show_censors=True)

# 통계치 주석
# Panel D: KM Plot 내부 텍스트 위치 수정
ax = plt.gca() # 현재 활성화된 축 가져오기

# x=0.05, y=0.05는 왼쪽 하단 구석 (0~1 범위)
# transform=ax.transAxes를 넣어야 데이터 값이 아닌 '그래프 비율'로 배치됨
plt.text(0.05, 0.05, 
         f"HR = {hr:.2f} (95% CI {ci_lower:.2f}-{ci_upper:.2f})\nLog-rank p = {p_val:.4e}", 
         transform=ax.transAxes,
         fontsize=11, 
         verticalalignment='bottom',
         horizontalalignment='left',
         bbox=False)
plt.title("RSF Model Risk Stratification (CV-based)", fontsize=12)
plt.xlabel("Days", fontsize=12)
plt.ylabel("PFS Probability", fontsize=12)
sns.despine()
plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/Fig4_riskstrat.pdf", dpi=300, bbox_inches='tight')
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sksurv.ensemble import RandomSurvivalForest

# 1. 평가 시점 설정 (데이터의 마지막 유효 시점, 예: 900일)
# 83명 코호트의 PFS 범위를 고려하여 마지막 시점으로 고정
eval_time = 730

# 2. 평가 시점 기준 Binary Label 생성 (Censoring 고려)
# eval_time 이전에 재발한 사람(1) vs eval_time 이후 재발하거나 그때까지 재발 안 한 사람(0)
# (단, eval_time 이전에 Censored된 데이터는 정확한 레이블링이 불가능하므로 제외하는 것이 일반적)
mask = (clin_df['PFS'] >= eval_time) | (clin_df['recur'] == 1)
eval_df = clin_df[mask].copy()
y_binary = ((eval_df['PFS'] <= eval_time) & (eval_df['recur'] == 1)).astype(int)

# 3. 모델별 ROC 데이터 계산 및 시각화
plt.figure(figsize=(7, 7))
model_colors = {
    "M1: gHRD": "#3182bd",      # Blue
    "M4: AS-only": "#31a354",  # Green
    "M2: AS+HRR": "#fd8d3c",    # Orange
    "M3: Combined": "#de2d26"   # Red
}

for name, feat_list in model_sets.items():
    # 모델 재학습 및 리스크 점수 예측
    best_rsf.fit(X[feat_list], y)
    # eval_df에 있는 환자들의 리스크 점수만 추출
    risk_scores = best_rsf.predict(eval_df[feat_list])
    
    # FPR, TPR 계산
    fpr, tpr, _ = roc_curve(y_binary, risk_scores)
    roc_auc = auc(fpr, tpr)
    
    # 시각화
    plt.plot(fpr, tpr, color=model_colors[name], lw=2.5,
             label=f"{name} (AUC = {roc_auc:.2f})")

# 4. 그래프 디테일 설정
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1) # Random line
plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title(f'ROC Curve Comparison at {eval_time} Days', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

#%%
# ============================================================
# Figure 4: Internal validation (bootstrap OOB + repeated CV)
# Features: AS Score, HRR Score, gHRDscore
# Endpoint: PFS (days), recur (1=event, 0=censor)
# Models: Cox (several), Random Survival Forest (RSF)
# Panels:
#   a) Bootstrap OOB C-index (mean ± 95% CI)
#   b) Bootstrap OOB time-dependent AUC curves
#   c) Repeated-CV out-of-fold risk stratification KM
#   d) Cox(all) forest plot (HR per 1 SD)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold

# scikit-survival
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc

# lifelines (for KM + logrank + Cox forest plot)
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test


# -----------------------------
# CONFIG (edit if needed)
# -----------------------------
TIME_COL = "PFS"         # progression-free survival in days
EVENT_COL = "recur"      # 1=recurrence event, 0=censored

FEATURE_COLS = ["AS Score", "HRR Score", "gHRDscore"]

# time-dependent AUC: choose fixed clinical timepoints (days)
TIMES_FIXED = np.array([365, 730])  # 1y, 2y (3y=1095는 risk set 적으면 불안정)

# time-dependent AUC curve grid
AUC_STEP = 30  # days
MIN_AT_RISK = 10
MIN_EVENTS_BY_T = 5

N_BOOT = 500           # bootstrap replicates (500~2000)
MIN_OOB = 10           # skip bootstrap replicate if OOB too small
CV_SPLITS = 5
CV_REPEATS = 50        # repeated CV for stable out-of-fold risk
RANDOM_STATE = 0


# -----------------------------
# Helpers
# -----------------------------
def _clean_df(clin_df: pd.DataFrame) -> pd.DataFrame:
    df = clin_df.copy()

    needed = [TIME_COL, EVENT_COL] + FEATURE_COLS
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in clin_df: {missing}")

    # numeric conversion
    for c in [TIME_COL] + FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[EVENT_COL] = pd.to_numeric(df[EVENT_COL], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=needed).copy()

    # enforce types
    df[TIME_COL] = df[TIME_COL].astype(float)
    df[EVENT_COL] = df[EVENT_COL].astype(int)
    if not set(df[EVENT_COL].unique()).issubset({0, 1}):
        raise ValueError(f"{EVENT_COL} must be 0/1. Found: {sorted(df[EVENT_COL].unique())}")

    return df


def _make_xy(df: pd.DataFrame):
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    
    # sksurv.util.Surv.from_dataframe은 입력된 컬럼명을 필드명으로 사용함 (예: 'recur', 'PFS')
    y = Surv.from_dataframe(EVENT_COL, TIME_COL, df)
    
    # 코드 내부의 다른 함수들이 ['event'], ['time'] 필드명을 사용하므로 강제로 이름을 변경
    # y.dtype.names를 통해 필드명을 ("event", "time")으로 고정
    y.dtype.names = ("event", "time")
    
    return X, y


def _filter_supported_times(df: pd.DataFrame, times: np.ndarray,
                            min_at_risk: int = MIN_AT_RISK,
                            min_events_by_t: int = MIN_EVENTS_BY_T) -> np.ndarray:
    t = df[TIME_COL].to_numpy(dtype=float)
    e = df[EVENT_COL].to_numpy(dtype=int).astype(bool)

    keep = []
    for tt in times:
        at_risk = np.sum(t >= tt)
        events_by_t = np.sum(e & (t <= tt))
        keep.append((at_risk >= min_at_risk) and (events_by_t >= min_events_by_t))
    times = np.array(times, dtype=float)
    return times[np.array(keep, dtype=bool)]


def _make_auc_time_grid(df: pd.DataFrame) -> np.ndarray:
    # candidate grid: from 90 days to ~90th percentile to avoid tail instability
    t_max = float(np.quantile(df[TIME_COL].to_numpy(dtype=float), 0.9))
    grid = np.arange(90, max(120, int(t_max)), AUC_STEP, dtype=float)

    # add fixed timepoints (1y/2y) too
    times = np.unique(np.concatenate([grid, TIMES_FIXED.astype(float)]))
    times = np.sort(times)

    # keep only supported times
    times = _filter_supported_times(df, times)
    return times


def build_estimator(kind: str, random_state: int = 0):
    """
    kind: 'cox' or 'rsf'
    We scale inputs so HR per 1 SD is natural for Cox;
    scaling doesn't hurt RSF with 3 features.
    """
    if kind == "cox":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", CoxPHSurvivalAnalysis()),
        ])

    if kind == "rsf":
        # With only 3 predictors, keep RSF simple and conservative
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomSurvivalForest(
                n_estimators=1500,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features=1.0,      # use all features at splits (important when p=3)
                n_jobs=-1,
                random_state=random_state,
            ))
        ])

    raise ValueError(f"Unknown kind: {kind}")


def bootstrap_oob_eval(X, y, feat_idx, kind, times, n_boot=500, min_oob=10, random_state=0):
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    times = np.array(times, dtype=float)
    auc_mat = []
    cindex_list = []

    Xsub = X[:, feat_idx]

    for b in range(n_boot):
        train_idx = rng.integers(0, n, size=n)
        unique_train = np.unique(train_idx)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[unique_train] = False
        test_idx = np.where(oob_mask)[0]

        if len(test_idx) < min_oob:
            continue

        est = build_estimator(kind, random_state=random_state + b)
        est.fit(Xsub[train_idx], y[train_idx])
        risk_test = est.predict(Xsub[test_idx])

        # 1. C-index 계산 (이건 시간 범위에 덜 민감함)
        ci = concordance_index_censored(y[test_idx]["event"], y[test_idx]["time"], risk_test)[0]
        cindex_list.append(ci)

        # 2. Time-dependent AUC 필터링 강화 (핵심 수정 부분)
        # sksurv는 평가 시간이 test data의 [min_time, max_time) 사이에 있어야 함
        t_min_test = y[test_idx]["time"].min()
        t_max_test = y[test_idx]["time"].max()
        
        # 기존 times < t_max에 더해 times > t_min 조건 추가
        times_ok = times[(times > t_min_test) & (times < t_max_test)]
        
        row = np.full(len(times), np.nan, dtype=float)

        if len(times_ok) >= 2:
            try:
                out = cumulative_dynamic_auc(y[train_idx], y[test_idx], risk_test, times_ok)
                auc = out[0]
                pos = np.searchsorted(times, times_ok)
                row[pos] = auc
            except ValueError:
                # 필터링을 해도 데이터 불균형으로 에러가 날 경우 해당 회차는 skip
                pass

        auc_mat.append(row)

    return np.array(cindex_list, dtype=float), np.vstack(auc_mat) if len(auc_mat) else np.empty((0, len(times)))


def summarize_bootstrap(metric_array):
    metric_array = np.array(metric_array, dtype=float)
    metric_array = metric_array[~np.isnan(metric_array)]
    return {
        "mean": float(np.mean(metric_array)),
        "lo": float(np.quantile(metric_array, 0.025)),
        "hi": float(np.quantile(metric_array, 0.975)),
        "n": int(metric_array.size),
    }


def repeated_cv_oof_risk(X, y, feat_idx, kind, n_splits=5, n_repeats=50, random_state=0):
    """
    Repeated stratified CV (stratified by event) to get stable out-of-fold risk score per patient.
    Returns: risk_oof (length n)
    """
    n = X.shape[0]
    Xsub = X[:, feat_idx]

    events = y["event"].astype(int)
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

    risk_sum = np.zeros(n, dtype=float)
    risk_cnt = np.zeros(n, dtype=float)

    for fold, (tr, te) in enumerate(rkf.split(Xsub, events)):
        est = build_estimator(kind, random_state=random_state + fold)
        est.fit(Xsub[tr], y[tr])
        risk = est.predict(Xsub[te])

        risk_sum[te] += risk
        risk_cnt[te] += 1.0

    risk_oof = risk_sum / np.maximum(risk_cnt, 1.0)
    return risk_oof


def plot_km_from_risk(ax, df, risk, title):
    # median split on out-of-fold risk
    thr = np.median(risk)
    high = risk >= thr

    kmf = KaplanMeierFitter()

    t = df[TIME_COL].to_numpy(dtype=float)
    e = df[EVENT_COL].to_numpy(dtype=int)

    kmf.fit(t[~high], event_observed=e[~high], label=f"Low risk (n={np.sum(~high)})")
    kmf.plot(ax=ax, ci_show=False)

    kmf.fit(t[high], event_observed=e[high], label=f"High risk (n={np.sum(high)})")
    kmf.plot(ax=ax, ci_show=False)

    # log-rank
    lr = logrank_test(t[high], t[~high], event_observed_A=e[high], event_observed_B=e[~high])
    ax.set_title(title)
    ax.set_xlabel("Days")
    ax.set_ylabel("PFS probability")
    ax.text(0.02, 0.02, f"log-rank p = {lr.p_value:.3g}", transform=ax.transAxes)


def plot_cox_forest(ax, df, z_cols, title):
    # lifelines Cox with standardized predictors (HR per 1 SD)
    tmp = df[[TIME_COL, EVENT_COL] + z_cols].copy()
    tmp = tmp.rename(columns={TIME_COL: "time", EVENT_COL: "event"})

    cph = CoxPHFitter()
    cph.fit(tmp, duration_col="time", event_col="event")

    summ = cph.summary.copy()
    # columns typically include: 'exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p'
    # robust column lookup:
    hr_col = "exp(coef)"
    lo_col = next((c for c in summ.columns if "lower" in c and "exp" in c), None)
    hi_col = next((c for c in summ.columns if "upper" in c and "exp" in c), None)
    p_col  = "p"

    feats = list(summ.index)
    hrs = summ[hr_col].to_numpy()
    los = summ[lo_col].to_numpy() if lo_col else np.exp(summ["coef"].to_numpy() - 1.96 * summ["se(coef)"].to_numpy())
    his = summ[hi_col].to_numpy() if hi_col else np.exp(summ["coef"].to_numpy() + 1.96 * summ["se(coef)"].to_numpy())
    ps  = summ[p_col].to_numpy()

    # order top-to-bottom
    y_pos = np.arange(len(feats))[::-1]

    ax.hlines(y=y_pos, xmin=los, xmax=his, lw=2)
    ax.plot(hrs, y_pos, "s")

    ax.axvline(1.0, ls="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feats)
    ax.set_xlabel("Hazard Ratio (95% CI) per 1 SD increase")
    ax.set_title(title)

    # annotate HR and p
    for i, (hr, lo, hi, p) in enumerate(zip(hrs, los, his, ps)):
        ax.text(max(his)*1.05, y_pos[i], f"HR={hr:.2f} ({lo:.2f}-{hi:.2f}), p={p:.3g}",
                va="center", fontsize=9)

    ax.set_xlim(left=max(0.05, min(los)*0.9), right=max(his)*1.45)


# -----------------------------
# Main: build Figure 4
# -----------------------------
def make_figure4(clin_df: pd.DataFrame, out_png="Figure4_internal_validation.png"):
    df = _clean_df(clin_df)

    # Short names for plotting/cox forest
    df = df.rename(columns={
        "AS Score": "AS",
        "HRR Score": "HRR",
        "gHRDscore": "gHRD",
    })
    feat_cols_short = ["AS", "HRR", "gHRD"]

    global FEATURE_COLS
    FEATURE_COLS = feat_cols_short  # update global

    X, y = _make_xy(df)

    times = _make_auc_time_grid(df)

    # model specs: (kind, feature indices)
    specs = {
        "Cox_gHRD": ("cox", [2]),
        "Cox_AS": ("cox", [0]),
        "Cox_HRR": ("cox", [1]),
        "Cox_AS+HRR": ("cox", [0, 1]),
        "Cox_all": ("cox", [0, 1, 2]),
        "RSF_all": ("rsf", [0, 1, 2]),
    }

    # --- Bootstrap OOB evaluation for each model
    boot = {}
    for name, (kind, idx) in specs.items():
        cidx, auc_mat = bootstrap_oob_eval(
            X, y, feat_idx=idx, kind=kind, times=times,
            n_boot=N_BOOT, min_oob=MIN_OOB, random_state=RANDOM_STATE
        )
        boot[name] = {"cindex": cidx, "auc_mat": auc_mat}

    # Summaries for panel (a)
    perf_rows = []
    for name in specs.keys():
        s = summarize_bootstrap(boot[name]["cindex"])
        perf_rows.append([name, s["mean"], s["lo"], s["hi"], s["n"]])
    perf = pd.DataFrame(perf_rows, columns=["model", "cindex_mean", "cindex_lo", "cindex_hi", "n_boot_eff"])

    # --- Panel (b): choose 3 models for AUC curve to avoid clutter
    auc_models = ["Cox_gHRD", "Cox_all", "RSF_all"]

    # --- Panel (c): repeated CV out-of-fold risk for final model
    FINAL_MODEL = "Cox_all"  # change to "RSF_all" if you want
    kind_final, idx_final = specs[FINAL_MODEL]
    risk_oof = repeated_cv_oof_risk(
        X, y, feat_idx=idx_final, kind=kind_final,
        n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE
    )

    # --- Panel (d): Cox forest plot (AS/HRR/gHRD) per 1 SD
    # Standardize for interpretability per 1 SD increase
    z_cols = []
    for c in ["AS", "HRR", "gHRD"]:
        zc = f"{c}_z"
        df[zc] = (df[c] - df[c].mean()) / df[c].std(ddof=0)
        z_cols.append(zc)

    # -----------------------------
    # Plot: 2x2 panels
    # -----------------------------
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axA, axB = axs[0, 0], axs[0, 1]
    axC, axD = axs[1, 0], axs[1, 1]

    # (a) Bootstrap OOB C-index dot + CI
    perf_sorted = perf.sort_values("cindex_mean", ascending=True)
    y_pos = np.arange(len(perf_sorted))
    axA.hlines(y=y_pos, xmin=perf_sorted["cindex_lo"], xmax=perf_sorted["cindex_hi"], lw=2)
    axA.plot(perf_sorted["cindex_mean"], y_pos, "o")
    axA.set_yticks(y_pos)
    axA.set_yticklabels(perf_sorted["model"])
    axA.axvline(0.5, ls="--", color="gray")
    axA.set_xlabel("Bootstrap OOB C-index (mean ± 95% CI)")
    axA.set_title("Fig4a. Internal validation (bootstrap OOB)")

    # annotate n effective
    for i, (_, row) in enumerate(perf_sorted.iterrows()):
        axA.text(row["cindex_hi"] + 0.01, i, f"n={int(row['n_boot_eff'])}", va="center", fontsize=9)

    # (b) Bootstrap OOB time-dependent AUC curve (mean + CI band)
    for m in auc_models:
        auc_mat = boot[m]["auc_mat"]
        if auc_mat.shape[0] == 0:
            continue
        mean_auc = np.nanmean(auc_mat, axis=0)
        lo_auc = np.nanquantile(auc_mat, 0.025, axis=0)
        hi_auc = np.nanquantile(auc_mat, 0.975, axis=0)

        axB.plot(times, mean_auc, label=m)
        axB.fill_between(times, lo_auc, hi_auc, alpha=0.15)

    axB.axhline(0.5, ls="--", color="gray")
    axB.set_xlabel("Time (days)")
    axB.set_ylabel("Time-dependent AUC")
    axB.set_title("Fig4b. Time-dependent AUC (bootstrap OOB)")
    axB.legend(loc="lower right")

    # (c) CV OOF risk stratification KM
    plot_km_from_risk(axC, df, risk_oof, title=f"Fig4c. Repeated-CV OOF risk stratification ({FINAL_MODEL})")

    # (d) Cox forest plot (per 1 SD)
    plot_cox_forest(axD, df, z_cols=z_cols, title="Fig4d. Cox model (per 1 SD)")

    # Panel letters
    for ax, letter in zip([axA, axB, axC, axD], ["a", "b", "c", "d"]):
        ax.text(-0.12, 1.05, letter, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    plt.tight_layout()
    #fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.show()
    return fig, perf, times, boot


# -----------------------------
# Run (clin_df should already exist)
# -----------------------------
fig, perf_table, auc_times, boot_res = make_figure4(clin_df, out_png="Figure4.png")
print(perf_table.sort_values("cindex_mean", ascending=False))
# %%
#######^^^^^^^^ NMF clustering ##################################
major_deg = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/analysis/majorexp_DEG.txt', sep='\t', index_col=0)
deglist = major_deg[(major_deg['padj'] < 0.1) & (abs(major_deg['log2FoldChange']) > 0.58)].index.to_list()
df_exp = counts_matrix.T
var_genes = df_exp.var(axis=1).sort_values(ascending=False).head(3000).index
X = df_exp.loc[deglist].T
clinical = metadata.copy()

if X.min().min() < 0:
    X = X + abs(X.min().min())

# ==========================================
# 2. Perform NMF Clustering
# ==========================================

print("Running NMF Clustering...")
n_clusters = 2  # We expect 2 groups: Sensitive vs Resistant

# Initialize NMF
model = NMF(n_components=n_clusters, init='random', random_state=42, max_iter=1000)

# Fit and Transform
W = model.fit_transform(X) # (Samples x Features) -> (Samples x Clusters)
H = model.components_      # (Clusters x Features) -> Basis matrix

# Assign Cluster Labels (Who belongs to which group?)
# We take the column with the maximum value for each sample
cluster_labels = np.argmax(W, axis=1)

# Add to Clinical Data
# clinical = clinical.loc[X.index] # Ensure alignment
clinical['NMF_Cluster'] = cluster_labels
clinical['NMF_Cluster'] = clinical['NMF_Cluster'].map({0: 'Cluster A', 1: 'Cluster B'})

print(clinical['NMF_Cluster'].value_counts())

# ==========================================
# 3. Survival Analysis (Kaplan-Meier)
# ==========================================

kmf = KaplanMeierFitter()
plt.figure(figsize=(10, 6))

colors = ['blue', 'red']
groups = sorted(clinical['NMF_Cluster'].unique())

for i, group in enumerate(groups):
    mask = (clinical['NMF_Cluster'] == group)
    kmf.fit(
        durations=clinical.loc[mask, 'PFS'], 
        event_observed=clinical.loc[mask, 'recur'], 
        label=group
    )
    kmf.plot_survival_function(ci_show=False, color=colors[i], linewidth=2.5)

# Statistics (Log-Rank Test)
results = logrank_test(
    clinical.loc[clinical['NMF_Cluster'] == groups[0], 'PFS'],
    clinical.loc[clinical['NMF_Cluster'] == groups[1], 'PFS'],
    event_observed_A=clinical.loc[clinical['NMF_Cluster'] == groups[0], 'recur'],
    event_observed_B=clinical.loc[clinical['NMF_Cluster'] == groups[1], 'recur']
)

plt.title(f"PFS by NMF Clustering (p-value: {results.p_value:.4f})")
plt.xlabel("Days")
plt.ylabel("Survival Probability")
plt.grid(True, alpha=0.3)
plt.show()

# ==========================================
# 4. Heatmap of NMF Clusters (Visual Validation)
# ==========================================

# Create a heatmap sorted by NMF Cluster
sorted_samples = clinical.sort_values("NMF_Cluster").index
X_sorted = X.loc[sorted_samples].T # (Genes x Samples)

# Normalize per gene (Z-score) for visualization
row_mean = X_sorted.mean(axis=1)
row_std = X_sorted.std(axis=1)

X_zscore = X_sorted.sub(row_mean, axis=0).div(row_std, axis=0)

# Plot
plt.figure(figsize=(12, 10))
sns.heatmap(
    X_zscore, 
    cmap="RdBu_r", 
    center=0,
    xticklabels=False,
    yticklabels=True, # Show gene names if using 46 genes
    cbar_kws={'label': 'Z-Score'}
)
plt.title("Expression Heatmap Ordered by NMF Cluster")
plt.show()

# %%
##^^ splicing ##########################
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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter

###^^ 개별 SF gene cox regression ####
# 1. SF 유전자들의 발현 데이터만 추출 (행: 샘플, 열: 유전자)
sf_expr = expr_full_log.loc[expr_full_log.index.isin(SF_genes)].T 

# 2. 임상 데이터와 병합 (clin_df에 'PFS', 'recur'가 있다고 가정)
# clin_df의 index와 sf_expr의 index가 샘플 ID로 일치해야 함
analysis_df = sf_expr.join(clin_df[['PFS', 'recur']])

# 3. 개별 유전자별 Cox Regression 실행
results = []
for gene in SF_genes:
    try:
        # 단일 유전자에 대한 데이터프레임 구성
        temp_df = analysis_df[[gene, 'PFS', 'recur']].dropna()
        
        cph = CoxPHFitter()
        cph.fit(temp_df, duration_col='PFS', event_col='recur')
        
        summary = cph.summary.iloc[0]
        results.append({
            'gene': gene,
            'HR': summary['exp(coef)'],
            'lower_CI': summary['exp(coef) lower 95%'],
            'upper_CI': summary['exp(coef) upper 95%'],
            'p_val': summary['p']
        })
    except:
        # 수렴하지 않는 유전자는 제외
        continue

cox_results = pd.DataFrame(results)
cox_results['-log10(p)'] = -np.log10(cox_results['p_val'])

#%%
plt.figure(figsize=(6, 5))
# p-value 0.05 기준선 설정
sns.scatterplot(data=cox_results, x='HR', y='-log10(p)', alpha=0.6, color='#60685B', edgecolor=None, linewidth=0)

# 유의미한 유전자 강조 (예: p < 0.05)
significant = cox_results[cox_results['p_val'] < 0.05]
plt.scatter(significant['HR'], significant['-log10(p)'], color='#467707', s=20)

# 상위 10개 유전자 이름 표기
for i, row in significant.sort_values('p_val').head(10).iterrows():
    plt.text(row['HR'], row['-log10(p)'], row['gene'], fontsize=9)

plt.axhline(-np.log10(0.05), color='gray', linestyle='--', alpha=0.7, linewidth=0.8)
plt.axvline(1, color='gray', linestyle='--', alpha=0.7,linewidth=0.8)
plt.xlabel("Hazard Ratio (HR)")
plt.ylabel("-log10(p)")
sns.despine()
#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/SFgeneexp_volcanoplot.pdf", dpi=300,bbox_inches='tight')
plt.show()

# p-value 순으로 정렬하여 상위 20개 선택
top_sf = cox_results.sort_values('p_val').head(20)

plt.figure(figsize=(5, 8))
plt.errorbar(top_sf['HR'], range(len(top_sf)), 
             xerr=[top_sf['HR'] - top_sf['lower_CI'], top_sf['upper_CI'] - top_sf['HR']],
             fmt='o', color='#305402', capsize=3)

plt.yticks(range(len(top_sf)), top_sf['gene'])
plt.axvline(1, color='black', linestyle='--', linewidth=0.8)
plt.xlabel("Hazard Ratio (95% CI)")
plt.gca().invert_yaxis() # p-value 낮은 순이 위로 오게
sns.despine()
plt.show()
#%%
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

onlymajor = counts_matrix.copy()
onlymajor = onlymajor.T

splicegene = geneexp.loc[geneexp.index.isin(SF_genes),clinical_df.index]

# 결과 저장용
results = []

# 두 그룹 인덱스 추출
responder_idx = clinical_df[clinical_df["group"] == "responder"].index
nonresponder_idx = clinical_df[clinical_df["group"] == "nonresponder"].index

# 각 gene에 대해 MW-test 수행
for gene in splicegene.index:
    expr_responder = splicegene.loc[gene, responder_idx]
    expr_nonresponder = splicegene.loc[gene, nonresponder_idx]
    
    # MW test (two-sided)
    stat, p = mannwhitneyu(expr_responder, expr_nonresponder, alternative='two-sided')
    
    # log2FC (optional)
    log2fc = (expr_responder.mean() + 1e-6) / (expr_nonresponder.mean() + 1e-6)
    log2fc = np.log2(log2fc)

    results.append([gene, p, log2fc])

# 결과 DataFrame
mw_df = pd.DataFrame(results, columns=["gene", "p_value", "log2FC"])

# FDR correction (Benjamini-Hochberg)
mw_df["padj"] = multipletests(mw_df["p_value"], method='fdr_bh')[1]

# 정렬 (가장 유의한 순)
mw_df = mw_df.sort_values("padj")

mw_df.head()

#%%
import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation
import pandas as pd

finalSFlist = ['SNRNP25','SRSF9','UBL5','U2AF2','PTBP1','RBM42','YBX1','PRPF31'] ##with gene exp
#finalSFlist = ['CTNNBL1','SNRNP200','SNRNP25','SRSF9','SNRPE']
#finalSFlist = ['SRSF7','SRSF9','SRSF2','THOC3','SNRNP25']

# Melt 형식으로 변환
plot_df = splicegene.loc[finalSFlist].T  # samples × 8 genes
plot_df['group'] = clinical_df['group']  # add group info
plot_df = plot_df.reset_index().rename(columns={"index": "sample"})

# setup
sns.set(style="whitegrid")

fig, axes = plt.subplots(1, 5, figsize=(16, 4)) #plt.subplots(2, 4, figsize=(12, 6))
axes = axes.flatten()

for ax, gene in zip(axes, finalSFlist):

    # gene-specific df
    sub = plot_df[['group', gene]]
    
    # 메인 boxplot
    sns.boxplot(
        data=sub, 
        x='group', y=gene, 
        ax=ax,
        palette={"responder": "#4575b4", "nonresponder": "#d73027"},
        width=0.6,
        fliersize=0
    )
    
    # stripplot(점)
    sns.stripplot(
        data=sub,
        x='group', y=gene,
        ax=ax,
        color='black',
        dodge=True,
        jitter=True,
        size=3,
        alpha=0.3
    )
    
    from statannotations.Annotator import Annotator

    pairs = [("responder", "nonresponder")]

    annot = Annotator(
        ax, pairs,
        data=sub,
        x='group', y=gene
    )

    annot.configure(
        test='Mann-Whitney',
        text_format='star',
        show_test_name=False,
    )
    annot.apply_and_annotate()


    ax.set_title(gene, fontsize=12)
    ax.set_xlabel("")
    sns.despine()
    ax.set_ylabel("gene exp")

# 빈 subplot이 남으면 숨김
for i in range(len(finalSFlist), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


# %%

splicescore = geneexp.loc[geneexp.index.isin(finalSFlist),clinical_df.index].mean()
#splicescore = onlymajor.loc[onlymajor.index.isin(finalSFlist),clinical_df.index].mean()

clin_df["splicing score"] = np.where(splicescore > splicescore.median(), "High", "Low")

from lifelines.plotting import add_at_risk_counts  # <--- Essential import
sns.set_style("ticks")
for group in ["splicing score"]:
    plt.figure(figsize=(7, 5)) 
    ax = plt.subplot(111)

    colors = {"High": "#DA4343", "Low": "#3396D3"}
    fitters = [] 

    # Enforce order "High" -> "Low"
    target_labels = ["High", "Low"] 

    for label in target_labels:
        kmf = KaplanMeierFitter()
        idx = clin_df["splicing score"] == label
        
        # Fit data
        kmf.fit(clin_df.loc[idx, "PFS"], clin_df.loc[idx, "recur"], label=label)
        
        # Plot
        kmf.plot_survival_function(
            ax=ax, 
            ci_show=True, 
            color=colors.get(label, "gray"),
            linewidth=2
        )
        fitters.append(kmf)

    # --- Statistics (Log-rank & Cox) ---
    idx_high = clin_df["splicing score"] == "High"
    res = logrank_test(
        clin_df.loc[idx_high, "PFS"], clin_df.loc[~idx_high, "PFS"],
        clin_df.loc[idx_high, "recur"], clin_df.loc[~idx_high, "recur"]
    )

    # Cox HR
    tmp = clin_df[["PFS", "recur", "splicing score"]].copy()
    tmp["splicing score"] = (tmp["splicing score"] == "High").astype(int)
    cph = CoxPHFitter()
    cph.fit(tmp, duration_col="PFS", event_col="recur")
    hr = np.exp(cph.params_)[0]
    ci = cph.confidence_intervals_.iloc[0].tolist()

    # --- Annotations ---
    plt.title("Splicing Score")
    plt.text(0.05, 0.15, f"Log-rank p = {res.p_value:.4f}\nHR = {hr:.2f} ({ci[0]:.2f} - {ci[1]:.2f})",
            transform=ax.transAxes, fontsize=11, fontweight='regular')

    # Hide the default x-label because it overlaps with the table
    plt.ylabel('PFS Probability')
    plt.xlabel('')
    # --- Add Risk Table (Customized) ---
    # 1. Select rows: 'at_risk' is standard. 'events' counts recurrences. 
    #    We exclude 'censored' to save space and avoid confusion.
    add_at_risk_counts(*fitters, ax=ax, rows_to_show=['At risk', 'Events'])

    # --- Styling the Risk Table & Layout ---

    # 1. Adjust margins to prevent cutting off the table
    #    'bottom=0.25' reserves 25% of the figure height for the table
    plt.subplots_adjust(bottom=0.1) 
    ax.spines['bottom'].set_visible(False)

    # 2. Iterate through axes to find the table and adjust Font Size
    #    The table is added as new axes to the figure.
    for a in plt.gcf().axes:
        if a is not ax: # Identify the table axes (they are not the main ax)
            # Adjust tick labels (The numbers)
            a.tick_params(axis='x', labelsize=11) 
            # Adjust y-labels (The row names like "At risk")
            a.tick_params(axis='y', labelsize=11)
            # Rename row labels if you want specific Korean/English terms
            # (This is tricky but modifying the text objects works)
            yticks = a.get_yticklabels()
            for label in yticks:
                if "At risk" in label.get_text():
                    label.set_text("No. at Risk")
                elif "Events" in label.get_text():
                    label.set_text("Cumulative Recur") # Rename 'Events' to 'Recur'
            a.set_yticklabels(yticks)
            a.spines["top"].set_visible(False)
            a.spines["bottom"].set_visible(False)
            a.spines["left"].set_visible(False)
            a.spines["right"].set_visible(False)
    # 3. Manually place the X-label at the very bottom
    sns.despine(ax=ax)
    #plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLOanalysis/figures/sev_G2MCheckpoint_KMplot_"+group+".pdf", dpi=300)
    plt.show()

# %%
########********* random forest ###########################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os 

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (roc_curve, auc, confusion_matrix, 
                             precision_recall_curve, average_precision_score)
os.environ['NUMEXPR_MAX_THREADS'] = '16'
os.environ['NUMEXPR_NUM_THREADS'] = '16'

res_list = val_clin[(val_clin['PFS']>540)].index.to_list()
nonres_list = val_clin[(val_clin['recur']==1)&(val_clin['PFS']<540)].index.to_list()

clinical_df = val_clin.copy()
clinical_df = clinical_df.loc[clinical_df.index.isin(res_list+nonres_list),:]
clinical_df['group'] = 'responder'
clinical_df.loc[clinical_df.index.isin(nonres_list),'group'] = 'nonresponder'
clin_df = clinical_df.copy()

final_score = score_df.loc[score_df.index.isin(df_high['Pathway']), clinical_df.index]

# [중요] 데이터 전처리
# 1. Transpose: (20 features x 78 samples) -> (78 samples x 20 features)
X = final_score.T 

# 2. Label Encoding
# Nonresponder (Resistant)를 1로, Responder (Sensitive)를 0으로 설정
# (그래야 '내성 확률'을 예측하는 모델이 됩니다)
y = clin_df['group'].map({'nonresponder': 1, 'responder': 0}).values
feature_names = np.array(X.columns)

print(f"Data Shape: X={X.shape}, y={y.shape}")
print(f"Class Distribution: Nonresponder(1)={sum(y)}, Responder(0)={len(y)-sum(y)}")

# ==========================================
# 2. RFE: 최적의 피처 개수 찾기 (Feature Selection)
# ==========================================
print("\n[Step 1] Running Recursive Feature Elimination (RFE)...")

# 모델 설정 (Random Forest)
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)

# RFECV: CV를 통해 최적의 개수를 자동으로 찾음
# step=1: 한 번에 1개씩 제거
# cv=5: 5-fold CV로 검증
min_features = 3 # 최소한 3개는 남기자
rfecv = RFECV(estimator=rf_selector, step=1, cv=StratifiedKFold(5), 
              scoring='roc_auc', min_features_to_select=min_features, n_jobs=1)
rfecv.fit(X, y)

# 선택된 피처 확인
selected_features = feature_names[rfecv.support_]
print(f"Optimal number of features: {rfecv.n_features_}")
print(f"Selected Features: {selected_features}")


# --- [Figure 1] RFE Elbow Plot ---
plt.figure(figsize=(8, 5))
n_features_range = range(min_features, min_features + len(rfecv.cv_results_['mean_test_score']))
plt.plot(n_features_range, rfecv.cv_results_['mean_test_score'], 'o-', color='darkred')
plt.xlabel("Number of Features Selected")
plt.ylabel("Cross-Validated AUC Score")
plt.title("Recursive Feature Elimination (RFE) Performance")
plt.grid(True)
plt.show()

# ==========================================
# 3. Model Validation with Hyperparameter Tuning (Nested CV)
# ==========================================
print("\n[Step 2] Validating Model with Nested CV & Tuning...")

selected_features = ['double-strand break repair via homologous recombination (GO:0000724)','regulation of mitotic metaphase/anaphase transition (GO:0030071)','spliceosomal snRNP assembly (GO:0000387)','regulation of mitotic sister chromatid separation (GO:0010965)','nuclear transport (GO:0051169)']
selected_features = feature_names[rfecv.support_]
feature_names = selected_features.copy()
X_selected = X[selected_features].values

# 5-Fold를 20번 반복 (총 100번 루프)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)

# 튜닝할 파라미터 그리드 (N수가 적으므로 보수적으로 설정)
param_grid = {
    'n_estimators': [100, 200],      # 나무의 개수
    'max_depth': [3, 5],          # 나무의 깊이 (작을수록 과적합 방지)
    'min_samples_leaf': [3, 5],      # 잎사귀 노드의 최소 샘플 수 (클수록 보수적)
    'max_features': ['sqrt']         # 피처 참조 방식
}

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
importances_list = []
best_params_list = [] # 각 폴드마다 어떤 파라미터가 뽑혔는지 기록
confusion_matrices = []
precisions, ap_scores = [], []
mean_recall = np.linspace(0, 1, 100) # PR Curve의 x축 기준 (Recall)

plt.figure(figsize=(12, 5))

# --- [ROC Curve Loop] ---
plt.subplot(1, 2, 1)

# Nested CV 시작
for i, (train, test) in enumerate(cv.split(X_selected, y)):
    
    # 1. Inner Loop: Grid Search로 최적 파라미터 찾기 (Train 데이터만 사용!)
    rf = RandomForestClassifier(random_state=i, n_jobs=-1)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                               cv=3, scoring='roc_auc', n_jobs=16) # Inner CV는 3-fold 정도로 가볍게
    grid_search.fit(X_selected[train], y[train])
    
    # 2. Best Model 추출
    best_model = grid_search.best_estimator_
    best_params_list.append(grid_search.best_params_) # 기록용
    
    # 3. Outer Loop: Test 데이터로 평가
    probas_ = best_model.predict_proba(X_selected[test])
    
    # ROC 계산
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    
    precision, recall, _ = precision_recall_curve(y[test], probas_[:, 1])
    # Recall을 기준으로 보간(Interpolation)하여 평균 계산 준비
    precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
    ap_scores.append(average_precision_score(y[test], probas_[:, 1]))
    
    # Feature Importance 수집
    importances_list.append(best_model.feature_importances_)
    y_pred = (probas_[:, 1] > 0.5).astype(int)
    confusion_matrices.append(confusion_matrix(y[test], y_pred))
    
    # 그래프 그리기 (흐릿하게)
    plt.plot(fpr, tpr, lw=1, alpha=0.05, color='gray')

    # 진행상황 모니터링 (너무 오래 걸리면 주석 처리)
    if i % 20 == 0:
        print(f"Iter {i}/100 - Best Params: {grid_search.best_params_}")


# --- [Average ROC Curve] ---
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
# --- [Figure 1] ROC Curve ---
plt.figure(figsize=(6, 6))
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

plt.plot(mean_fpr, mean_tpr, color='darkred', lw=2, alpha=.8,
         label=r'Mean ROC (AUC = %0.3f $\pm$ %0.2f)' % (mean_auc, std_auc))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy', label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve (Top {len(selected_features)} Features)')
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
#plt.savefig('roc_curve.png', dpi=300)
plt.show()
print("Saved: roc_curve.png")

# --- [Figure 2] PR Curve (⭐️ 추가된 부분) ---
plt.figure(figsize=(6, 6))
mean_precision = np.mean(precisions, axis=0)
mean_ap = np.mean(ap_scores)
std_ap = np.std(ap_scores)
baseline = sum(y) / len(y) # 랜덤 모델의 기준선 (소수 클래스 비율)

plt.plot(mean_recall, mean_precision, color='#006400', lw=2, alpha=.8, # 진한 녹색
         label=r'Mean PR (AP = %0.3f $\pm$ %0.2f)' % (mean_ap, std_ap))
plt.plot([0, 1], [baseline, baseline], linestyle='--', lw=2, color='navy', label=f'Baseline ({baseline:.2f})')
plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision (PPV)')
plt.title(f'Precision-Recall Curve (Top {len(selected_features)} Features)')
plt.legend(loc="best")
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tight_layout()
#plt.savefig('pr_curve.png', dpi=300)
plt.show()
print("Saved: pr_curve.png")

# --- [Figure 3] Feature Importance ---
plt.figure(figsize=(8, 6))
mean_importances = np.mean(importances_list, axis=0)
indices = np.argsort(mean_importances)

plt.title('Feature Importance (Bootstrap Aggregated)')
plt.barh(range(len(indices)), mean_importances[indices], color='#B22222', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.tight_layout()
#plt.savefig('feature_importance.png', dpi=300)
plt.show()
print("Saved: feature_importance.png")

# --- [Figure 4] Confusion Matrix ---
plt.figure(figsize=(6, 5))
mean_cm = np.sum(confusion_matrices, axis=0) / (5 * 20) # 평균 계산

sns.heatmap(mean_cm, annot=True, fmt='.1f', cmap='Reds', cbar=False,
            xticklabels=['Pred: Sensitive(0)', 'Pred: Resistant(1)'],
            yticklabels=['Actual: Sensitive(0)', 'Actual: Resistant(1)'])
plt.title("Average Confusion Matrix")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
#plt.savefig('confusion_matrix.png', dpi=300)
plt.show()
print("Saved: confusion_matrix.png")

print("\nAll plots have been saved successfully.")

# %%
import pandas as pd
import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sklearn.model_selection import cross_val_score, RepeatedKFold

# ==========================================
# 1. 타겟 피처 선택 (Data Preparation)
# ==========================================
target_term = 'double-strand break repair via homologous recombination (GO:0000724)'

# [중요] 1개의 피처라도 모델에 넣을 땐 반드시 2D DataFrame 형태여야 함 (Sample x Feature)
# final_score는 (Gene x Sample) 형태라고 가정하므로, 뽑아서 Transpose + DataFrame화
X_single = final_score.loc[target_term, :].to_frame() 

# Survival 라벨 생성
y = Surv.from_arrays(event=clin_df['recur'].astype(bool), time=clin_df['PFS'])

# ==========================================
# 2. CV 실행 (Repeated K-Fold)
# ==========================================
# 모델 정의
estimator = CoxPHSurvivalAnalysis(alpha=0.0001)

# CV 설정 (5-Fold x 20번 반복 = 총 100번 테스트)
cv = RepeatedKFold(n_splits=5, n_repeats=20, random_state=42)

# 검증 실행 (scoring='mypath'는 c-index를 의미)
scores = cross_val_score(estimator, X_single, y, cv=cv)

# ==========================================
# 3. 결과 출력
# ==========================================
mean_c = np.mean(scores)
std_c = np.std(scores)

print(f"Target Feature: {target_term}")
print("-" * 50)
print(f"Mean C-index: {mean_c:.4f} (± {std_c:.4f})")

if mean_c > 0.7:
    print(">> 평가: 매우 강력한 단일 예측 인자입니다 (Excellent Predictor).")
elif mean_c > 0.6:
    print(">> 평가: 유의미한 예측력을 가지고 있습니다 (Good Predictor).")
else:
    print(">> 평가: 예측력이 다소 약합니다 (Weak Predictor).")
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy.interpolate import interp1d

# ==========================================
# 1. 데이터 및 설정
# ==========================================
target_term = 'double-strand break repair via homologous recombination (GO:0000724)'
X_vec = final_score.loc[target_term, :].values
y_event = clin_df['recur'].astype(bool).values
y_time = clin_df['PFS'].values

# CV 설정
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)
mean_times = np.linspace(0, y_time.max(), 100)

# 저장소
surv_probs_high, surv_probs_low = [], []
c_indices = []  # C-index 저장용

# ==========================================
# 2. CV Loop (C-index 계산 추가)
# ==========================================
print("Running CV KM with Metrics...")

for i, (train_idx, test_idx) in enumerate(cv.split(X_vec, y_event)):
    # Threshold 결정 (Training set 기준)
    train_median = np.median(X_vec[train_idx])
    
    # Test set 나누기
    mask_high = X_vec[test_idx] > train_median
    
    # --- [New] C-index 계산 (Fold 별 예측력) ---
    # 점수가 높을수록 위험(Risk)하다고 가정하므로 그대로 입력
    try:
        c_index = concordance_index_censored(
            y_event[test_idx], 
            y_time[test_idx], 
            X_vec[test_idx] # Risk Score
        )[0]
        c_indices.append(c_index)
    except:
        pass # Event가 하나도 없는 극단적인 경우 패스

    # KM Curve 데이터 수집 (기존과 동일)
    if mask_high.sum() > 0:
        times_h, probs_h = kaplan_meier_estimator(y_event[test_idx][mask_high], y_time[test_idx][mask_high])
        f_h = interp1d(times_h, probs_h, kind='previous', fill_value="extrapolate")
        surv_probs_high.append(f_h(mean_times))
        
    if (~mask_high).sum() > 0:
        times_l, probs_l = kaplan_meier_estimator(y_event[test_idx][~mask_high], y_time[test_idx][~mask_high])
        f_l = interp1d(times_l, probs_l, kind='previous', fill_value="extrapolate")
        surv_probs_low.append(f_l(mean_times))

# ==========================================
# 3. 통계치 계산
# ==========================================
# Mean C-index
mean_c_index = np.mean(c_indices)
std_c_index = np.std(c_indices)

# Mean Survival Curve
mean_curve_high = np.mean(surv_probs_high, axis=0)
mean_curve_low = np.mean(surv_probs_low, axis=0)

# Median Survival Time (확률 0.5가 되는 지점 찾기)
def get_median_survival(times, probs):
    # 확률이 0.5 이하로 떨어지는 첫 지점
    idx = np.where(probs <= 0.5)[0]
    if len(idx) > 0:
        return times[idx[0]]
    else:
        return np.inf # 0.5까지 안 떨어짐 (Not Reached)

med_high = get_median_survival(mean_times, mean_curve_high)
med_low = get_median_survival(mean_times, mean_curve_low)

# ==========================================
# 4. 시각화 (Text 추가)
# ==========================================
plt.figure(figsize=(9, 7))

# --- Plotting Curves ---
# Spaghetti lines
for surv in surv_probs_high: plt.step(mean_times, surv, where="post", color='firebrick', alpha=0.03, lw=1)
for surv in surv_probs_low: plt.step(mean_times, surv, where="post", color='steelblue', alpha=0.03, lw=1)

# Mean Lines & Shadow
plt.step(mean_times, mean_curve_high, where="post", color='firebrick', lw=3, label='High Risk (HR Pathway High)')
plt.fill_between(mean_times, 
                 np.maximum(0, mean_curve_high - np.std(surv_probs_high, axis=0)), 
                 np.minimum(1, mean_curve_high + np.std(surv_probs_high, axis=0)), 
                 step="post", alpha=0.2, color='firebrick')

plt.step(mean_times, mean_curve_low, where="post", color='steelblue', lw=3, label='Low Risk (HR Pathway Low)')
plt.fill_between(mean_times, 
                 np.maximum(0, mean_curve_low - np.std(surv_probs_low, axis=0)), 
                 np.minimum(1, mean_curve_low + np.std(surv_probs_low, axis=0)), 
                 step="post", alpha=0.2, color='steelblue')

# --- [New] Statistics Text Box ---
stats_text = (
    f"Model Performance (CV)\n"
    f"--------------------------------\n"
    f"C-index : {mean_c_index:.3f} (±{std_c_index:.3f})\n\n"
    f"Median PFS (Days)\n"
    f"--------------------------------\n"
    f"High Risk : {med_high:.0f} days\n"
    f"Low Risk  : {'> Max F/U' if med_high == np.inf else f'{med_low:.0f} days'}"
)

# 그래프 빈 공간(보통 왼쪽 아래)에 텍스트 박스 삽입
plt.text(0.05, 0.05, stats_text, transform=plt.gca().transAxes, 
         fontsize=11, verticalalignment='bottom', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

# Decoration
plt.title(f'Cross-Validated Kaplan-Meier Analysis\nTarget: {target_term.split("(")[0]}', fontsize=13, fontweight='bold')
plt.xlabel("Progression-Free Survival (Days)")
plt.ylabel("Survival Probability")
plt.ylim(0, 1.05)
plt.legend(loc="upper right")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

print("Median PFS High:", med_high)
print("Median PFS Low:", med_low)
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxPHSurvivalAnalysis # HR 계산용
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy.interpolate import interp1d

# ==========================================
# 1. 데이터 및 설정
# ==========================================
target_term = 'double-strand break repair via homologous recombination (GO:0000724)'
X_vec = final_score.loc[target_term, :].values
y_event = clin_df['recur'].astype(bool).values
y_time = clin_df['PFS'].values

# CV 설정
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)
mean_times = np.linspace(0, y_time.max(), 100)

surv_probs_high, surv_probs_low = [], []
c_indices = []
hazard_ratios = [] # HR 저장용 리스트

# ==========================================
# 2. CV Loop (HR 계산 추가)
# ==========================================
print("Running CV KM with HR & C-index...")

for i, (train_idx, test_idx) in enumerate(cv.split(X_vec, y_event)):
    # Threshold 결정 (Training set 기준)
    train_median = np.median(X_vec[train_idx])
    
    # Test set 나누기
    mask_high = X_vec[test_idx] > train_median
    
    # 그룹이 쪼개지지 않는 극단적인 경우(한쪽에 몰림) 방지
    if mask_high.sum() == 0 or (~mask_high).sum() == 0:
        continue

    # --- [1] C-index 계산 ---
    try:
        c_index = concordance_index_censored(
            y_event[test_idx], 
            y_time[test_idx], 
            X_vec[test_idx]
        )[0]
        c_indices.append(c_index)
    except: pass

    # --- [2] Hazard Ratio (HR) 계산 ---
    # Test set 안에서 High(1) vs Low(0)의 위험도 차이 계산
    try:
        # 그룹 변수 생성 (High=1, Low=0)
        group_var = mask_high.astype(int).reshape(-1, 1)
        
        # Univariate CoxPH 실행
        cox = CoxPHSurvivalAnalysis(alpha=0.0001)
        cox.fit(group_var, Surv.from_arrays(event=y_event[test_idx], time=y_time[test_idx]))
        
        # Coef를 HR로 변환 (exp)
        hr = np.exp(cox.coef_[0])
        hazard_ratios.append(hr)
    except: pass

    # --- [3] KM Curve 데이터 수집 ---
    times_h, probs_h = kaplan_meier_estimator(y_event[test_idx][mask_high], y_time[test_idx][mask_high])
    f_h = interp1d(times_h, probs_h, kind='previous', fill_value="extrapolate")
    surv_probs_high.append(f_h(mean_times))
        
    times_l, probs_l = kaplan_meier_estimator(y_event[test_idx][~mask_high], y_time[test_idx][~mask_high])
    f_l = interp1d(times_l, probs_l, kind='previous', fill_value="extrapolate")
    surv_probs_low.append(f_l(mean_times))

# ==========================================
# 3. 통계치 요약
# ==========================================
mean_c_index = np.mean(c_indices)
std_c_index = np.std(c_indices)

mean_hr = np.mean(hazard_ratios)
std_hr = np.std(hazard_ratios)

mean_curve_high = np.mean(surv_probs_high, axis=0)
mean_curve_low = np.mean(surv_probs_low, axis=0)

# ==========================================
# 4. 시각화 (Text 수정됨)
# ==========================================
plt.figure(figsize=(9, 7))

# Spaghetti lines
for surv in surv_probs_high: plt.step(mean_times, surv, where="post", color='firebrick', alpha=0.03, lw=1)
for surv in surv_probs_low: plt.step(mean_times, surv, where="post", color='steelblue', alpha=0.03, lw=1)

# Mean Lines & Shadow
plt.step(mean_times, mean_curve_high, where="post", color='firebrick', lw=3, label='High Risk (HR Pathway High)')
plt.fill_between(mean_times, 
                 np.maximum(0, mean_curve_high - np.std(surv_probs_high, axis=0)), 
                 np.minimum(1, mean_curve_high + np.std(surv_probs_high, axis=0)), 
                 step="post", alpha=0.2, color='firebrick')

plt.step(mean_times, mean_curve_low, where="post", color='steelblue', lw=3, label='Low Risk (HR Pathway Low)')
plt.fill_between(mean_times, 
                 np.maximum(0, mean_curve_low - np.std(surv_probs_low, axis=0)), 
                 np.minimum(1, mean_curve_low + np.std(surv_probs_low, axis=0)), 
                 step="post", alpha=0.2, color='steelblue')

# --- [Updated] Statistics Text Box ---
# HR과 C-index만 깔끔하게 표시
stats_text = (
    f"Cross-Validation Metrics\n"
    f"--------------------------------\n"
    f"Mean HR : {mean_hr:.2f} (±{std_hr:.2f})\n"
    f"C-index : {mean_c_index:.3f} (±{std_c_index:.3f})"
)

plt.text(0.05, 0.05, stats_text, transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='bottom', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

# Decoration
plt.title(f'Cross-Validated Kaplan-Meier Analysis\nTarget: {target_term.split("(")[0]}', fontsize=13, fontweight='bold')
plt.xlabel("Progression-Free Survival (Days)")
plt.ylabel("Survival Probability")
plt.ylim(0, 1.05)
plt.legend(loc="upper right", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('cv_km_curve_final.png', dpi=300)
plt.show()

print(f"Final Mean HR: {mean_hr:.4f}")
# %%
#########^^^^^^^^^^ Bootstrap #####################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sklearn.utils import resample

# ==========================================
# 1. 데이터 준비
# ==========================================
target_term = 'double-strand break repair via homologous recombination (GO:0000724)'
# # target_term = 'spliceosomal snRNP assembly (GO:0000387)' 

# X_vec = final_score.loc[target_term, :].values
# #X_vec = geneexp.loc[geneexp.index.isin(finalSFlist),clinical_df.index].mean().values

X_vec  = clin_df['HRR Score'].values
y_event = clin_df['recur'].astype(bool).values
y_time = clin_df['PFS'].values

# 구조화된 y 배열 (sksurv용)
y_structured = Surv.from_arrays(event=y_event, time=y_time)

# ==========================================
# 2. Bootstrap 설정
# ==========================================
n_bootstraps = 1000  # 1000번 반복 (표준)
bootstrap_hrs = []
bootstrap_c_indices = []

print(f"Running {n_bootstraps} Bootstraps for Robust Estimation...")

# ==========================================
# 3. Bootstrap Loop
# ==========================================
for i in range(n_bootstraps):
    # [1] Resampling (복원 추출)
    # indices를 뽑아서 In-Bag(학습용)과 Out-of-Bag(검증용)을 나눔
    indices = np.arange(len(X_vec))
    boot_idx = resample(indices, replace=True, n_samples=len(indices), random_state=i)
    oob_idx = np.setdiff1d(indices, boot_idx) # 뽑히지 않은 나머지 (약 36.8%)
    
    # 데이터셋 구성
    X_boot = X_vec[boot_idx]
    y_boot = y_structured[boot_idx]
    
    # 예외 처리: 한 종류의 이벤트만 뽑히면 스킵 (거의 안 일어나지만 안전장치)
    if y_boot['event'].sum() == 0 or y_boot['event'].sum() == len(y_boot):
        continue

    # [2] HR 계산 (In-Bag Sample)
    # Median 기준으로 그룹 나누기
    median_val = np.median(X_boot)
    group_var = (X_boot > median_val).astype(int).reshape(-1, 1) # High=1, Low=0
    
    # 한 그룹이 비어있으면 스킵
    if group_var.sum() == 0 or group_var.sum() == len(group_var):
        continue
        
    try:
        cox = CoxPHSurvivalAnalysis(alpha=0.0001) # L2 규제 살짝 줌 (안정성)
        cox.fit(group_var, y_boot)
        hr = np.exp(cox.coef_[0])
        
        # 비정상적인 값(너무 큰 값) 필터링 (Bootstrap에서도 간혹 발생 가능)
        if hr < 100 and hr > 0.01:
            bootstrap_hrs.append(hr)
    except:
        continue
        
    # [3] C-index 검증 (Out-of-Bag Sample) - 이게 진짜 검증 성능임
    if len(oob_idx) > 0:
        X_oob = X_vec[oob_idx]
        y_oob_event = y_event[oob_idx]
        y_oob_time = y_time[oob_idx]
        
        try:
            # C-index 계산
            c_idx = concordance_index_censored(
                y_oob_event, y_oob_time, X_oob
            )[0]
            bootstrap_c_indices.append(c_idx)
        except:
            pass

# ==========================================
# 4. 결과 집계 (95% CI 계산)
# ==========================================
# HR 통계
hr_median = np.median(bootstrap_hrs)
hr_lower = np.percentile(bootstrap_hrs, 2.5)  # 95% CI Lower
hr_upper = np.percentile(bootstrap_hrs, 97.5) # 95% CI Upper

# C-index 통계
c_mean = np.mean(bootstrap_c_indices)
c_std = np.std(bootstrap_c_indices)
c_lower = np.percentile(bootstrap_c_indices, 2.5)
c_upper = np.percentile(bootstrap_c_indices, 97.5)

print("\n" + "="*40)
print(f" Target: {target_term.split('(')[0]}")
print("="*40)
print(f"[Hazard Ratio (Bootstrap N={len(bootstrap_hrs)})]")
print(f"  Median HR : {hr_median:.4f}")
print(f"  95% CI    : [{hr_lower:.4f} - {hr_upper:.4f}]")
print("-" * 40)
print(f"[C-index (Out-of-Bag Validation)]")
print(f"  Mean C-idx: {c_mean:.4f} (±{c_std:.4f})")
print(f"  95% CI    : [{c_lower:.4f} - {c_upper:.4f}]")
print("="*40)

# ==========================================
# 5. 분포 시각화 (논문용 Figure)
# ==========================================
plt.figure(figsize=(5, 5))
sns.histplot(bootstrap_hrs, kde=True, color='firebrick', element="step")
plt.axvline(1, color='gray', linestyle=':', label='No Effect (HR=1)')
plt.axvline(hr_median, color='black', linestyle='--', label=f'Median HR={hr_median:.2f}')

plt.title(f'Bootstrap Distribution of Hazard Ratio\n(Median: {hr_median:.2f}, 95% CI: {hr_lower:.2f}-{hr_upper:.2f})')
plt.xlabel('Hazard Ratio')
plt.legend()
plt.tight_layout()
sns.despine()
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/HRR_bootstrap_distribution.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: C-index Distribution
# plt.subplot(1, 2, 2)
plt.figure(figsize=(5, 5))
sns.histplot(bootstrap_c_indices, kde=True, color='steelblue', element="step")
plt.axvline(c_mean, color='black', linestyle='--', label=f'Mean C-index={c_mean:.2f}')
plt.axvline(0.5, color='gray', linestyle=':', label='Random (0.5)')
plt.title(f'Bootstrap Distribution of C-index (OOB)\n(Mean: {c_mean:.2f}, 95% CI: {c_lower:.2f}-{c_upper:.2f})')
plt.xlabel('Concordance Index')
plt.legend()
plt.tight_layout()
plt.xlim(0.0, 1.0)
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/HRR_bootstrap_Cindex_distribution.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%

plt.figure(figsize=(5,5))
ax = sns.boxplot(
        data=val_clin, 
        x='recur', y='PFS', 
        width=0.6,
        fliersize=0,
        palette='husl',
    )
sns.stripplot(val_clin, y='PFS',x='recur',color='black',
        dodge=True,
        jitter=True,
        size=5,
        ax = ax,
        alpha=0.3)
sns.despine()
plt.show()


# %%
###^^ SF ~ HR gene ################################
lib = gp.get_library(name="GO_Biological_Process_2021", organism="Human")
genes_hr = lib["double-strand break repair via homologous recombination (GO:0000724)"]
hr_major = counts_matrix.T.copy()
hr_major = hr_major.loc[hr_major.index.isin(genes_hr), :]
threshold = 0.1  # 10%

hr_major = hr_major.loc[
    (hr_major != 0).sum(axis=1) >= threshold * hr_major.shape[1]
]


# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

# ---------------------------------------------------------
# 1. 데이터 준비 (선생님의 데이터프레임 이름으로 교체하세요)
# ---------------------------------------------------------
# 예시: 공통된 샘플만 남기고 Transpose (Correlation은 sample이 row여야 함)
# hr_major = ... (Load your data)
# splicegene = ... (Load your data)

# 공통 샘플 찾기
common_samples = hr_major.columns.intersection(splicegene.columns)
df_hr = hr_major[common_samples].T  # 행: 샘플, 열: 유전자
df_sf = splicegene[common_samples].T # 행: 샘플, 열: 유전자

# ---------------------------------------------------------
# 2. Correlation Matrix 계산 (Spearman 권장)
# ---------------------------------------------------------
# SF와 HR 유전자 간의 상관계수만 계산
correlation_matrix = pd.DataFrame(
    index=df_sf.columns, 
    columns=df_hr.columns
)

for sf in df_sf.columns:
    for hr in df_hr.columns:
        # Spearman 상관계수 계산
        corr = df_sf[sf].corr(df_hr[hr], method='spearman')
        correlation_matrix.loc[sf, hr] = corr

correlation_matrix = correlation_matrix.astype(float)

# ---------------------------------------------------------
# 3. Figure 1: Clustered Heatmap
# ---------------------------------------------------------
# 보기 좋게 하기 위해 상관성이 너무 낮은 건 제외하거나, 
# 관심 있는 유전자 리스트가 있다면 그들만 slicing해서 그리세요.

g = sns.clustermap(
    correlation_matrix,
    cmap="RdBu_r",        # Red=Positive, Blue=Negative
    center=0,
    figsize=(8, 6),
    #dendrogram_ratio=(.1, .1),
    #cbar_pos=(0.02, 0.8, 0.03, 0.15),
    yticklabels=False,
    xticklabels=False
)

g.ax_heatmap.set_xlabel("HR genes", fontsize=10)
g.ax_heatmap.set_ylabel("SF genes", fontsize=10)
plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/functionalHR_SF_corr_heatmap_wide.pdf", dpi=300, bbox_inches='tight')
plt.show() # savefig 시: plt.savefig("Fig3_Heatmap.pdf")

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# 가정: correlation_matrix는 이미 계산됨 (Result 3 코드 참고)
# correlation_matrix rows = Splicing Factors, cols = HR Genes

# ---------------------------------------------------------
# 1. "Master Regulator" 순위 매기기
# ---------------------------------------------------------
# 각 Splicing Factor별로 HR 유전자들과의 평균 상관계수를 구함
sf_importance = correlation_matrix.mean(axis=1).sort_values(ascending=False)

print("Top 10 Master Splicing Factors:")
print(sf_importance.head(10))

# Top 10만 따로 시각화 (Bar plot)
plt.figure(figsize=(5, 10))
sns.barplot(x=sf_importance.head(20).values, y=sf_importance.head(20).index, palette="Reds_r")
plt.xlabel("mean correlation")
plt.ylabel("")
sns.despine()
#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/AS_corr_top20_barplot.pdf", dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------
# 2. Elite Network 그리기 (Threshold 상향)
# ---------------------------------------------------------
elite_threshold = 0.7  # 기준을 0.7로 대폭 상향

G = nx.Graph()
# ... (이전 코드와 노드 추가 방식 동일) ...

# 엣지 추가 (높은 Threshold 적용)
edge_count = 0
for sf in correlation_matrix.index:
    for hr in correlation_matrix.columns:
        corr_val = correlation_matrix.loc[sf, hr]
        if corr_val > elite_threshold:
            G.add_edge(sf, hr, weight=corr_val)
            edge_count += 1

print(f"Threshold {elite_threshold} 적용 시 연결 개수: {edge_count}")

# 만약 연결이 여전히 너무 많으면 0.75, 0.8로 올리면서 확인
if edge_count > 0:
    # ... (네트워크 그리기 코드, 이전 답변 참고) ...
    pass




#%%
# ---------------------------------------------------------
# 4. Figure 2: Co-expression Network
# ---------------------------------------------------------
# 상관계수가 특정 threshold (예: 0.5) 이상인 연결만 그림

threshold = 0.7
G = nx.Graph()

# 노드 추가
for gene in df_hr.columns:
    G.add_node(gene, type='HR', color='#98DD88')
for gene in df_sf.columns:
    G.add_node(gene, type='SF', color='#93C7CA')

# 엣지 추가 (Threshold 넘는 것만)
for sf in correlation_matrix.index:
    for hr in correlation_matrix.columns:
        corr_val = correlation_matrix.loc[sf, hr]
        if corr_val > threshold:
            G.add_edge(sf, hr, weight=corr_val)

# 고립된 노드 제거 (깔끔하게 보이기 위해)
isolates = list(nx.isolates(G))
G.remove_nodes_from(isolates)

# 그리기 설정
pos = nx.spring_layout(G, k=0.5, iterations=50) # k값 조절로 노드 간격 조정
node_colors = [nx.get_node_attributes(G, 'color')[node] for node in G.nodes()]
edges = G.edges()
weights = [G[u][v]['weight'] * 2 for u,v in edges] # 선 굵기 = 상관계수 비례

plt.figure(figsize=(10, 10))
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.9)
nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')

# 범례 추가
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Functional HR Gene', markerfacecolor='#98DD88', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Splicing Factor', markerfacecolor='#93C7CA', markersize=10)
]
plt.legend(handles=legend_elements, loc='upper right')
plt.title(f"Co-expression Network (Spearman rho > {threshold})")
plt.axis('off')
plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/AS_corr_coexpmatrix.pdf", dpi=300, bbox_inches='tight')
plt.show()

# ---------------------------------------------------------
# 5. Figure 3: Top Pairs Scatter Plot (검증용)
# ---------------------------------------------------------
# 가장 상관관계가 높은 Top pair 찾기
stacked_corr = correlation_matrix.stack()
top_pair = stacked_corr.idxmax() # (SF_name, HR_name)
max_corr = stacked_corr.max()

sf_name, hr_name = top_pair

plt.figure(figsize=(6, 6))
sns.regplot(
    x=df_sf[sf_name], 
    y=df_hr[hr_name], 
    scatter_kws={'alpha':0.6}, 
    line_kws={'color':'red'}
)
plt.xlabel(f"{sf_name} Expression (TPM)")
plt.ylabel(f"{hr_name} Functional Expression")
plt.title(f"Top Correlated Pair (R = {max_corr:.2f})")
plt.show()


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import numpy as np
import math

# -------------------------------------------------------------------------
# 1. Grid 설정 (5칸 x 3줄)
# -------------------------------------------------------------------------
# sig_genes는 이전 단계에서 찾아낸 15개 유전자 리스트라고 가정합니다.
# 만약 리스트가 없다면, 앞선 코드의 sig_genes 결과를 그대로 쓰시면 됩니다.
# 예시: sig_genes = ['SRSF2', 'NONO', 'HNRNPK', ... ] (총 15개)

n_cols = 5
n_rows = 3
total_plots = n_cols * n_rows

# 피겨 사이즈 조절 (가로를 넓게)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 12))
axes = axes.flatten() # 반복문 돌리기 편하게 1차원으로 폅니다.

sns.set_style("ticks")
colors = {"High": "#DA4343", "Low": "#3396D3"} # 빨강/파랑

# -------------------------------------------------------------------------
# 2. 반복문으로 15개 그리기
# -------------------------------------------------------------------------
for i, ax in enumerate(axes):
    # 그릴 유전자가 없으면 빈 칸으로 남김 (혹시 15개보다 적을 경우 대비)
    if i >= len(sig_genes):
        ax.set_visible(False)
        continue
        
    gene = sig_genes[i]
    
    # --- 데이터 준비 ---
    try:
        gene_exp = splicegene.loc[gene]
    except KeyError:
        continue

    tmp_df = clin_df.copy()
    common_samples = tmp_df.index.intersection(gene_exp.index)
    tmp_df = tmp_df.loc[common_samples]
    current_exp = gene_exp.loc[common_samples]

    # Median Cutoff
    cutoff = current_exp.median()
    tmp_df["group"] = np.where(current_exp > cutoff, "High", "Low")
    
    # --- 통계 계산 (HR, P-value) ---
    # 1. Log-rank
    idx_high = tmp_df["group"] == "High"
    results = logrank_test(
        tmp_df.loc[idx_high, "PFS"], tmp_df.loc[~idx_high, "PFS"],
        tmp_df.loc[idx_high, "recur"], tmp_df.loc[~idx_high, "recur"]
    )
    p_val = results.p_value

    # 2. Cox HR
    cox_data = tmp_df[["PFS", "recur", "group"]].copy()
    cox_data["group"] = (cox_data["group"] == "High").astype(int)
    cph = CoxPHFitter()
    try:
        cph.fit(cox_data, duration_col="PFS", event_col="recur")
        hr = cph.hazard_ratios_["group"]
        ci = cph.confidence_intervals_.iloc[0]
    except:
        hr, ci = np.nan, [np.nan, np.nan]

    # --- Plotting (KM Curve) ---
    for label in ["High", "Low"]:
        kmf = KaplanMeierFitter()
        idx = tmp_df["group"] == label
        kmf.fit(tmp_df.loc[idx, "PFS"], tmp_df.loc[idx, "recur"], label=label)
        
        kmf.plot_survival_function(
            ax=ax, 
            ci_show=True, 
            color=colors.get(label, "gray"),
            linewidth=2
        )

    # --- 디자인 다듬기 ---
    ax.set_title(gene, fontsize=14, fontweight='bold')
    ax.set_xlabel("Days", fontsize=10) # 요청하신대로 Days
    
    # Y축 라벨은 맨 왼쪽 열에만 표시 (공간 절약)
    if i % n_cols == 0:
        ax.set_ylabel("PFS Probability", fontsize=10)
    else:
        ax.set_ylabel("")
        
    ax.set_ylim(0, 1.05)
    
    # 범례(Legend) 제거 (깔끔하게 하려면 제거하거나, 첫 번째 그래프에만 표시)
    ax.get_legend().remove()
    
    # 통계 수치 텍스트 (그래프 안쪽)
    text_str = f"p={p_val:.3f}\nHR={hr:.2f}"
    ax.text(0.05, 0.1, text_str, transform=ax.transAxes, fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    # 테두리 정리
    sns.despine(ax=ax)

# -------------------------------------------------------------------------
# 3. 마무리 (전체 범례 추가 및 저장)
# -------------------------------------------------------------------------
# 전체 그림에 대한 공통 범례를 상단이나 하단에 하나만 추가
handles = [plt.Line2D([0], [0], color=colors["High"], lw=2, label='High'),
           plt.Line2D([0], [0], color=colors["Low"], lw=2, label='Low')]
fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, 1.02), frameon=False)

plt.tight_layout()
plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/top15_SF_survival.pdf", dpi=300, bbox_inches='tight')
plt.show()

#%%
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test

# -------------------------------------------------------------------------
# 1. 모든 HR 유전자에 대해 Cox Regression 수행
# -------------------------------------------------------------------------
# 결과 저장을 위한 리스트
cox_results = []

print(f"Total genes to analyze: {hr_major.shape[0]}")

for gene in hr_major.index:
    try:
        # 1. 데이터 준비
        gene_exp = hr_major.loc[gene]
        
        # Clinical Data와 샘플 매칭
        tmp_df = clin_df.copy()
        common_samples = tmp_df.index.intersection(gene_exp.index)
        
        if len(common_samples) < 10: # 샘플 수가 너무 적으면 스킵
            continue
            
        tmp_df = tmp_df.loc[common_samples]
        current_exp = gene_exp.loc[common_samples]

        # 2. 그룹 나누기 (Median Cutoff)
        cutoff = current_exp.median()
        # 발현량이 모두 같거나(0인 경우 등) 분산이 없으면 스킵
        if current_exp.std() == 0:
            continue
            
        tmp_df["group"] = np.where(current_exp > cutoff, "High", "Low")
        
        # 3. Cox Proportional Hazards Model
        # High=1, Low=0으로 변환
        cox_data = tmp_df[["PFS", "recur", "group"]].copy()
        cox_data["group"] = (cox_data["group"] == "High").astype(int)
        
        cph = CoxPHFitter()
        cph.fit(cox_data, duration_col="PFS", event_col="recur")
        
        # 4. 결과 추출
        hr = cph.hazard_ratios_["group"]
        p_val = cph.summary.loc["group", "p"]
        ci_lower = np.exp(cph.confidence_intervals_.iloc[0, 0])
        ci_upper = np.exp(cph.confidence_intervals_.iloc[0, 1])
        
        # 결과 저장
        cox_results.append({
            "Gene": gene,
            "HR": hr,
            "P_value": p_val,
            "CI_Lower": ci_lower,
            "CI_Upper": ci_upper
        })
        
    except Exception as e:
        # 수렴 실패 등의 에러 발생 시 그냥 넘어감
        continue

# -------------------------------------------------------------------------
# 2. 유의한 유전자 필터링 및 확인
# -------------------------------------------------------------------------
# DataFrame으로 변환
res_df = pd.DataFrame(cox_results)

# P-value 0.05 미만인 것만 추출
sig_df = res_df[res_df["P_value"] < 0.05].sort_values("P_value")

print("-" * 30)
print(f"분석 완료된 유전자 수: {len(res_df)}")
print(f"유의한 유전자 (P < 0.05) 개수: {len(sig_df)}")
print("-" * 30)

print("\nTop 10 Significant Genes:")
print(sig_df[["Gene", "HR", "P_value"]].head(17))

# 유의한 유전자 이름 리스트 추출 (다음 시각화 단계용)
sig_hr_genes = sig_df["Gene"].tolist()

import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import numpy as np
import pandas as pd

# -------------------------------------------------------------------------
# 1. Grid 설정 (5칸 x 4줄 = 총 20칸)
# -------------------------------------------------------------------------
# 이전 단계에서 추출한 17개 유전자 리스트
# (만약 리스트 변수명이 다르다면 아래 sig_hr_genes를 해당 변수명으로 바꿔주세요)
genes_to_plot = sig_hr_genes 
n_genes = len(genes_to_plot)

# 레이아웃 설정: 가로 5개, 세로 4개
n_cols = 5
n_rows = 4

# 전체 그림 크기: 세로를 조금 더 길게 (줄당 4인치 * 4줄 = 16인치)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 16))
axes = axes.flatten() # 1차원 배열로 펼치기

sns.set_style("ticks")
colors = {"High": "#DA4343", "Low": "#3396D3"}

print(f"Plotting {n_genes} genes in a {n_rows}x{n_cols} grid...")

# -------------------------------------------------------------------------
# 2. 반복문 실행
# -------------------------------------------------------------------------
for i, ax in enumerate(axes):
    # 유전자가 17개이므로, 18번째 칸(인덱스 17)부터는 빈칸 처리
    if i >= n_genes:
        ax.set_visible(False)
        continue
        
    gene = genes_to_plot[i]
    
    # --- 데이터 준비 ---
    try:
        gene_exp = hr_major.loc[gene]
    except KeyError:
        print(f"Skipping {gene}: Not found in expression data")
        ax.set_visible(False)
        continue

    # Clinical Data 매칭
    tmp_df = clin_df.copy()
    common_samples = tmp_df.index.intersection(gene_exp.index)
    
    if len(common_samples) == 0:
        ax.set_visible(False)
        continue
        
    tmp_df = tmp_df.loc[common_samples]
    current_exp = gene_exp.loc[common_samples]

    # Median Cutoff로 그룹 나누기
    cutoff = current_exp.median()
    tmp_df["group"] = np.where(current_exp > cutoff, "High", "Low")
    
    # --- 통계 계산 (HR, P-value) ---
    # 1. Log-rank Test
    idx_high = tmp_df["group"] == "High"
    results = logrank_test(
        tmp_df.loc[idx_high, "PFS"], tmp_df.loc[~idx_high, "PFS"],
        tmp_df.loc[idx_high, "recur"], tmp_df.loc[~idx_high, "recur"]
    )
    p_val = results.p_value

    # 2. Cox Proportional Hazards (HR 계산용)
    cox_data = tmp_df[["PFS", "recur", "group"]].copy()
    cox_data["group"] = (cox_data["group"] == "High").astype(int)
    cph = CoxPHFitter()
    try:
        cph.fit(cox_data, duration_col="PFS", event_col="recur")
        hr = cph.hazard_ratios_["group"]
    except:
        hr = np.nan # 수렴 실패 시 NaN 처리

    # --- Plotting (KM Curve) ---
    for label in ["High", "Low"]:
        kmf = KaplanMeierFitter()
        idx = tmp_df["group"] == label
        if idx.sum() > 0:
            kmf.fit(tmp_df.loc[idx, "PFS"], tmp_df.loc[idx, "recur"], label=label)
            kmf.plot_survival_function(
                ax=ax, 
                ci_show=True, 
                color=colors.get(label, "gray"),
                linewidth=2
            )

    # --- 디자인 다듬기 ---
    ax.set_title(gene, fontsize=14, fontweight='bold')
    ax.set_xlabel("Days", fontsize=10) # X축 라벨 통일
    
    # Y축 라벨: 맨 왼쪽 열(0, 5, 10, 15번째)에만 표시
    if i % n_cols == 0:
        ax.set_ylabel("PFS Probability", fontsize=10)
    else:
        ax.set_ylabel("")
        
    ax.set_ylim(0, 1.05)
    
    # 개별 범례 제거
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    
    # 통계 수치 텍스트 (P-value, HR)
    # HR이 NaN이면 P-value만 표시
    if pd.isna(hr):
        text_str = f"p={p_val:.3f}"
    else:
        text_str = f"p={p_val:.3f}\nHR={hr:.2f}"
        
    ax.text(0.05, 0.1, text_str, transform=ax.transAxes, fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

    # 테두리 정리
    sns.despine(ax=ax)

# -------------------------------------------------------------------------
# 3. 마무리 (공통 범례 및 레이아웃)
# -------------------------------------------------------------------------
# 상단 중앙에 공통 범례 추가
handles = [plt.Line2D([0], [0], color=colors["High"], lw=2, label='High'),
           plt.Line2D([0], [0], color=colors["Low"], lw=2, label='Low')]

fig.legend(handles=handles, loc='upper center', ncol=2, fontsize=14, 
           bbox_to_anchor=(0.5, 1.02), frameon=False)

plt.tight_layout()
plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/top17_HRmajor_survival.pdf", dpi=300, bbox_inches='tight')
plt.show()

#%%
################^^ SEV pre data ###########################
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

sev = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_pre/111_pre/111_gene_exp.txt', sep='\t', index_col=0)
sfchecklist = ['SRSF2','MAGOH','SRSF7','HNRNPA2B1','SNRPG','BUB3','CLK2','SNRNP40','YBX1','HNRNPA1','SF3B4','SNRPE','DDX20','SRSF3'] #'SF3B6',
#sfchecklist = ['DDX39A','GPKOW','MAGOH','PPP1R8','CELF5','DHX30','WDR83','SRSF9','ISY1','GEMIN2']
sev_clin_ = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2025clinical/112_validation_clinicalinfo_new.txt', sep='\t', index_col=0)
sev_clin_ = sev_clin_.loc[sev_clin_.index.isin(sev.columns), :]

sev_clin = sev_clin_.copy()
#####
sev_clin_ = sev_clin_[(sev_clin_['BRCAmt']==0)&(sev_clin_['setting']=='maintenance')] #&(sev_clin_['line'].isin(['2L']))]
#####

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

# 1) align samples and compute SF mean
for g in range(len(sfchecklist)):
    gene = sfchecklist[g]
    sev_sf = sev.loc[gene, sev.columns.isin(sev_clin.index)]
    sev_clin = sev_clin_.copy()
    sev_clin["mean_SF_exp"] = sev_sf

    # 2) create numeric group (0/1)
    sev_clin["SFgroup"] = (sev_clin["mean_SF_exp"] > sev_clin["mean_SF_exp"].median()).astype(int)

    # 3) ensure numeric types for lifelines
    sev_clin["PFS"] = pd.to_numeric(sev_clin["PFS"], errors="coerce")
    sev_clin["recur"] = pd.to_numeric(sev_clin["recur"], errors="coerce").astype("Int64")

    # drop bad rows (NaN in duration/event/group)
    df = sev_clin[["PFS", "recur", "SFgroup"]].dropna()
    df["recur"] = df["recur"].astype(int)

    # (optional) sanity: event must be 0/1
    df = df[df["recur"].isin([0, 1])]

    # 4) plotting colors keyed by group id
    colors = {0: "#3396D3", 1: "#DA4343"}
    labels = {0: "Low", 1: "High"}

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(6, 5))

    for g in [0, 1]:
        mask = df["SFgroup"] == g
        T = df.loc[mask, "PFS"]
        E = df.loc[mask, "recur"]

        kmf.fit(T, E, label=labels[g])
        kmf.plot_survival_function(
            ci_show=False,
            color=colors[g],
            linewidth=2,
            show_censors=True,
        )

    plt.title(gene, fontsize=12, fontweight="bold")
    plt.xlabel("Days")
    plt.ylabel("PFS probability")

    # 5) Log-rank
    group1 = df[df["SFgroup"] == 1]
    group0 = df[df["SFgroup"] == 0]
    lr = logrank_test(group1["PFS"], group0["PFS"],
                    event_observed_A=group1["recur"],
                    event_observed_B=group0["recur"])

    # 6) CoxPH
    cph = CoxPHFitter()
    cph.fit(df[["PFS", "recur", "SFgroup"]], duration_col="PFS", event_col="recur")
    summary = cph.summary.loc["SFgroup"]
    hr = summary["exp(coef)"]
    ci_low = summary["exp(coef) lower 95%"]
    ci_high = summary["exp(coef) upper 95%"]
    pval = summary["p"]

    plt.text(
        0.55 * df["PFS"].max(), 0.8,
        f"HR = {hr:.2f} (95% CI {ci_low:.2f}–{ci_high:.2f})\n"
        f"p = {pval:.3e}\n"
        f"log-rank p = {lr.p_value:.3e}",
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.7, boxstyle="round")
    )

    plt.legend(title="SFgroup", frameon=False)
    plt.tight_layout()
    sns.despine()
    plt.show()


# %%
##^^^ SEV pre validation with SF ssGSEA score ######
import gseapy as gp
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

codinggene = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM_symbol.txt', sep='\t')
codinggenelist = codinggene['Gene Symbol'].to_list()

sev = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_80_gene_TPM.txt', sep='\t', index_col=0)
sev = sev.loc[sev.index.isin(codinggenelist), :]
sev = sev.iloc[:,1::2]
sev.columns = sev.columns.str[:-4]

sev_clin_ = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo_new.txt', sep='\t', index_col=0)
sev_clin_ = sev_clin_.iloc[1::2,:]
sev_clin_ = sev_clin_.loc[sev_clin_.index.isin(sev.columns), :]

sev_clin_['PFS'] = sev_clin_['interval']
sev_clin_['recur'] = 1

# sfchecklist = ['SRSF2','MAGOH','SRSF7','HNRNPA2B1','SNRPG','BUB3','CLK2','SNRNP40','YBX1','HNRNPA1','SF3B4','SNRPE','DDX20','SRSF3'] #'SF3B6',
# sev = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_pre/111_pre/111_gene_exp.txt', sep='\t', index_col=0)
# sev_clin_ = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2025clinical/112_validation_clinicalinfo_new.txt', sep='\t', index_col=0)
# sev_clin_ = sev_clin_.loc[sev_clin_.index.isin(sev.columns), :]

# #####
# sev_clin_ = sev_clin_[(sev_clin_['BRCAmt']==0)&(sev_clin_['setting']=='maintenance')] #&(sev_clin_['setting']=='maintenance')&(sev_clin_['line'].isin(['1L','2L']))
# #####

sev_clin = sev_clin_.copy()
sev = sev.loc[:, sev_clin.index]
sev_log = np.log2(sev+0.1)

res_major = gp.ssgsea(
    data=sev_log,
    gene_sets="GO_Biological_Process_2021", #GO_Biological_Process_2021, MSigDB_Hallmark_2020, Reactome_2022
    outdir=None,
    sample_norm_method="rank",
    permutation_num=0
)
df_major = res_major.res2d
score_major = df_major.pivot(index="Term", columns="Name", values="ES")

clin_df = sev_clin.copy()
dna_major = score_major.loc["alternative mRNA splicing, via spliceosome (GO:0000380)"] #double-strand break repair via homologous recombination (GO:0000724) #DNA Repair #alternative mRNA splicing, via spliceosome (GO:0000380), #HDR Thru Homologous Recombination (HRR) R-HSA-5685942

clin_df['AS Score'] = dna_major
clin_df["dna_major_group"] = np.where(dna_major > dna_major.median(), "High", "Low")


from lifelines.plotting import add_at_risk_counts  # <--- Essential import

for group in ["dna_major_group"]:
    plt.figure(figsize=(7, 5)) 
    ax = plt.subplot(111)

    colors = {"High": "#DA4343", "Low": "#3396D3"}
    fitters = [] 

    # Enforce order "High" -> "Low"
    target_labels = ["High", "Low"] 

    for label in target_labels:
        kmf = KaplanMeierFitter()
        idx = clin_df["dna_major_group"] == label
        
        # Fit data
        kmf.fit(clin_df.loc[idx, "PFS"], clin_df.loc[idx, "recur"], label=label)
        
        # Plot
        kmf.plot_survival_function(
            ax=ax, 
            ci_show=True, 
            color=colors.get(label, "gray"),
            linewidth=2
        )
        fitters.append(kmf)

    # --- Statistics (Log-rank & Cox) ---
    idx_high = clin_df["dna_major_group"] == "High"
    res = logrank_test(
        clin_df.loc[idx_high, "PFS"], clin_df.loc[~idx_high, "PFS"],
        clin_df.loc[idx_high, "recur"], clin_df.loc[~idx_high, "recur"]
    )

    # Cox HR
    tmp = clin_df[["PFS", "recur", "dna_major_group"]].copy()
    tmp["dna_major_group"] = (tmp["dna_major_group"] == "High").astype(int)
    cph = CoxPHFitter()
    cph.fit(tmp, duration_col="PFS", event_col="recur")
    hr = np.exp(cph.params_)[0]
    ci = cph.confidence_intervals_.iloc[0].tolist()

    # --- Annotations ---
    plt.title("alternative mRNA splicing, via spliceosome (GO:0000380)") #double-strand break repair via homologous recombination (GO:0000724) #alternative mRNA splicing, via spliceosome (GO:0000380) #HDR Thru Homologous Recombination (HRR) R-HSA-5685942
    plt.text(0.05, 0.15, f"Log-rank p = {res.p_value:.4f}\nHR = {hr:.2f} ({ci[0]:.2f} - {ci[1]:.2f})",
            transform=ax.transAxes, fontsize=11, fontweight='regular')

    # Hide the default x-label because it overlaps with the table
    plt.ylabel('PFS Probability')
    plt.xlabel('')
    # --- Add Risk Table (Customized) ---
    # 1. Select rows: 'at_risk' is standard. 'events' counts recurrences. 
    #    We exclude 'censored' to save space and avoid confusion.
    add_at_risk_counts(*fitters, ax=ax, rows_to_show=['At risk', 'Events'])

    # --- Styling the Risk Table & Layout ---

    # 1. Adjust margins to prevent cutting off the table
    #    'bottom=0.25' reserves 25% of the figure height for the table
    plt.subplots_adjust(bottom=0.1) 
    ax.spines['bottom'].set_visible(False)

    # 2. Iterate through axes to find the table and adjust Font Size
    #    The table is added as new axes to the figure.
    for a in plt.gcf().axes:
        if a is not ax: # Identify the table axes (they are not the main ax)
            # Adjust tick labels (The numbers)
            a.tick_params(axis='x', labelsize=11) 
            # Adjust y-labels (The row names like "At risk")
            a.tick_params(axis='y', labelsize=11)
            # Rename row labels if you want specific Korean/English terms
            # (This is tricky but modifying the text objects works)
            yticks = a.get_yticklabels()
            for label in yticks:
                if "At risk" in label.get_text():
                    label.set_text("No. at Risk")
                elif "Events" in label.get_text():
                    label.set_text("Cumulative Recur") # Rename 'Events' to 'Recur'
            a.set_yticklabels(yticks)
            a.spines["top"].set_visible(False)
            a.spines["bottom"].set_visible(False)
            a.spines["left"].set_visible(False)
            a.spines["right"].set_visible(False)
    # 3. Manually place the X-label at the very bottom
    sns.despine(ax=ax)
    #plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/HR_gene.pdf", dpi=300, bbox_inches='tight')
    plt.show()
# %%
from lifelines import CoxPHFitter
import numpy as np
import matplotlib as mpl

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

#%%

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
    rotation=30,     # 1. 45도 대신 90도로 회전
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

plt.subplots_adjust(bottom=0.3) # X축 라벨이 잘리지 않도록 하단 여백 확보
sns.despine()

#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/GOBP2021_major_barplot_codinggene_Top20.pdf", dpi=300,bbox_inches='tight')
plt.show()
# %%
