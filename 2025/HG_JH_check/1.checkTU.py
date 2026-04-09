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
from scipy.stats import mannwhitneyu

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

#%%
##TSG & frame 고장난 것들##

check_qry = ['ENST00000413440.1',
'MSTRG.2439.51',
'ENST00000369535.4',
'MSTRG.5835.1',
'MSTRG.5835.45',
'MSTRG.7521.11',
'MSTRG.12697.4',
'MSTRG.21460.23',
'MSTRG.23113.36',
'MSTRG.23113.46',
'MSTRG.23113.55',
'MSTRG.23113.90',
'MSTRG.25177.1',
'ENST00000267163.4',
'MSTRG.27367.3',
'MSTRG.27367.6',
'ENST00000262713.2',
'MSTRG.29977.13',
'ENST00000439696.2',
'MSTRG.35428.1',
'MSTRG.35461.17',
'MSTRG.38510.146',
'ENST00000327483.5',
'ENST00000317276.4',
'ENST00000285071.4',
'ENST00000578112.1',
'ENST00000584437.1',
'ENST00000259008.2',
'MSTRG.42543.7',
'MSTRG.47914.205',
'ENST00000270279.3',
'ENST00000411977.2',
'ENST00000397062.3',
'MSTRG.59271.59',
'MSTRG.62002.10',
'ENST00000216181.5',
'ENST00000295760.7',
'MSTRG.71555.38',
'MSTRG.76851.1',
'MSTRG.77772.5',
'ENST00000296839.2',
'MSTRG.84722.9',
'MSTRG.92529.47',
'MSTRG.97812.21',
'MSTRG.97812.42',
'MSTRG.100449.1',
'MSTRG.100449.4',
'MSTRG.100593.20',
'ENST00000304494.5',
'MSTRG.110303.21'
]

check_qry2 = ['MSTRG.23113.90','MSTRG.97812.42']

CAGlist = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/CAGlist.txt', sep='\t', index_col=0, header=None)
CAGlist = list(set(CAGlist.index)) # CAGlist를 리스트로 변환
#%%
##^^^^^^^^^^^^^^^^^^^^^^ 현구/정훈 확인 ##########################33
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon  # 또는 ttest_rel
from statannotations.Annotator import Annotator

# 예시 DataFrame (row: transcript, columns: N1, T1, N2, T2, ...)
# df = pd.read_csv(...) or your expression matrix
transcript_id = "MSTRG.70394.21-NRG1"  # 보고 싶은 transcript

dut = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/cDUT.cDUT_COAD_hg38.gtf.tmap', sep='\t')

dutlist = list(dut[dut['class_code']=='=']['ref_id'])

#df = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/mixSRLR/quantification_v2/LR_283_transcript_TU_fillna_fromTPM.txt', sep='\t', index_col=0)

df = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/quantification/LR_238_transcript_TU_fillna_fromTMM.txt', sep='\t', index_col=0)
df.index = df.index.str.split("-", n=1).str[0]  # clean transcript ID

#df.index = df.index.str.split("-", n=1).str[0]  # clean transcript ID
df = df.loc[df.index.isin(dutlist)]  # keep only matched transcripts
df = df[df.sum(axis=1) > 0]  # 발현량이 모두 0인 transcript 제거


#%%
####^^^^^ all tumor vs. all normal #############

# Normal / Tumor 전부 모으기
all_normal = df.filter(like="-N").values.flatten()
all_tumor = df.filter(like="-T").values.flatten()

# long-format 데이터프레임 생성
plot_df = pd.DataFrame({
    "expression": list(all_normal) + list(all_tumor),
    "condition": ["Normal"] * len(all_normal) + ["Tumor"] * len(all_tumor)
})

# NaN 제거
plot_df = plot_df.dropna()

# boxplot
plt.figure(figsize=(5, 5))
ax = sns.boxplot(data=plot_df, x="condition", y="expression",
                 showfliers=False, palette=["#4B74B6", "#E33C89"])

# Annotator: Mann–Whitney U test
annot = Annotator(ax,
                  pairs=[("Normal", "Tumor")],
                  data=plot_df,
                  x="condition", y="expression")

annot.configure(test='Mann-Whitney',
                text_format='star',
                loc='inside',
                verbose=1,
                comparisons_correction=None)

annot.apply_and_annotate()

plt.title("All Tumor vs All Normal (unpaired)")
plt.tight_layout()
plt.xlabel("")
plt.ylabel("TU")
#plt.savefig("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/figures/cDUT_mannwhitney_boxplot.pdf", bbox_inches='tight', dpi=300)
plt.show()

# 직접 p-value 계산도 가능
stat, pval = mannwhitneyu(all_normal, all_tumor, alternative="two-sided")
print(f"Mann–Whitney U test p-value: {pval:.4e}")

from scipy.stats import mannwhitneyu
import numpy as np
import pandas as pd

results = []

for transcript in df.index:
    # 모든 normal / tumor 값 모으기
    normal_vals = df.filter(like="-N").loc[transcript].dropna().values
    tumor_vals = df.filter(like="-T").loc[transcript].dropna().values

    if len(normal_vals) >= 2 and len(tumor_vals) >= 2:
        # Mann–Whitney U test
        stat, pval = mannwhitneyu(tumor_vals, normal_vals, alternative="two-sided")
        
        mean_normal = np.mean(normal_vals)
        mean_tumor = np.mean(tumor_vals)
        log2fc = np.log2((mean_tumor + 1e-6) / (mean_normal + 1e-6))
        
        results.append({
            "transcript": transcript,
            "log2FC": log2fc,
            "pval": pval
        })

# DataFrame으로 변환 + -log10 p-value 계산
res_df = pd.DataFrame(results)
res_df["-log10(pval)"] = -np.log10(res_df["pval"])

#%%
# 3. extract sample ID
samples = [col.rsplit('-', 1)[0] for col in df.columns]
unique_samples = sorted(set(samples))

# 4. transcript별 normal/tumor 평균 계산
data = []

for transcript in df.index:
    normal_vals = []
    tumor_vals = []
    for sample in unique_samples:
        n_col = f"{sample}-N"
        t_col = f"{sample}-T"
        if n_col in df.columns and t_col in df.columns:
            normal_vals.append(df.at[transcript, n_col])
            tumor_vals.append(df.at[transcript, t_col])
    
    if normal_vals and tumor_vals:
        data.append({
            "transcript": transcript,
            "Normal": pd.Series(normal_vals).mean(),
            "Tumor": pd.Series(tumor_vals).mean()
        })

# long-format 변환
plot_df = pd.DataFrame(data).melt(id_vars="transcript", var_name="condition", value_name="expression")

# plot
plt.figure(figsize=(5, 5))
ax = sns.boxplot(data=plot_df, x="condition", y="expression", showfliers=False, palette=["#4B74B6", "#E33C89"])

# Annotator: Mann–Whitney 설정
annot = Annotator(ax,
                  pairs=[("Normal", "Tumor")],
                  data=plot_df,
                  x="condition", y="expression")

annot.configure(test='Wilcoxon', #'Mann-Whitney'
                text_format='star',
                loc='inside',
                verbose=1,
                comparisons_correction=None)

annot.apply_and_annotate()

plt.title("cDUT (per transcript)")
plt.tight_layout()
plt.xlabel("")
plt.ylabel("TU")
#plt.savefig("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/figures/cDUT_matched_wilcoxon_boxplot.pdf", bbox_inches='tight', dpi=300)
plt.show()


pivot_df = plot_df.pivot_table(index="transcript", columns="condition", values="expression", aggfunc='mean')
pivot_df["delta"] = pivot_df["Tumor"] - pivot_df["Normal"]
num_upregulated = (pivot_df["delta"] > 0).sum()

print(f"Tumor에서 발현량이 더 높은 transcript 개수: {num_upregulated}")

#%%
samplelist = df.columns.to_list()
normallist = [df.columns[i] for i in range(len(samplelist)) if samplelist[i].endswith('-N')]
tumorlist = [df.columns[i] for i in range(len(samplelist)) if samplelist[i].endswith('-T')]
matchedlist = list(set([col.rsplit('-', 1)[0] for col in normallist]) & set([col.rsplit('-', 1)[0] for col in tumorlist]))

#%%
###^^^ cDUT volcano plot 준비 ################
from scipy.stats import mannwhitneyu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

results = []

for transcript in df.index:
    normal_vals = []
    tumor_vals = []
    
    for sample in unique_samples:
        n_col = f"{sample}-N"
        t_col = f"{sample}-T"
        if n_col in df.columns and t_col in df.columns:
            normal_vals.append(df.at[transcript, n_col])
            tumor_vals.append(df.at[transcript, t_col])
            
    normal_vals = pd.Series(normal_vals).dropna()
    tumor_vals = pd.Series(tumor_vals).dropna()

    if len(normal_vals) >= 2 and len(tumor_vals) >= 2:
        if len(normal_vals) == len(tumor_vals) and (tumor_vals != normal_vals).any():
            stat, pval = wilcoxon(tumor_vals, normal_vals, alternative="two-sided")
        else:
            pval = 1.0
        mean_normal = np.mean(normal_vals)
        mean_tumor = np.mean(tumor_vals)
        log2fc = np.log2((mean_tumor + 1e-6) / (mean_normal + 1e-6))
        
        results.append({
            "transcript": transcript,
            "log2FC": log2fc,
            "pval": pval
        })

# DataFrame으로 변환 + -log10 p-value 계산
res_df = pd.DataFrame(results)
res_df["-log10(pval)"] = -np.log10(res_df["pval"])
from statsmodels.stats.multitest import multipletests
from scipy.stats import binomtest
# FDR 보정 (Benjamini-Hochberg)
res_df["FDR"] = multipletests(res_df["pval"], method="fdr_bh")[1]
res_df["-log10(FDR)"] = -np.log10(res_df["FDR"])
from scipy.stats import binomtest

def binomial_test_by_cutoff(df, col_fc="log2FC", col_sig="FDR", sig_thresh=0.05, cutoffs=[0,0.5,0.6,1,1.5,1.6,2], use_fdr=True):
    results = []
    for cutoff in cutoffs:
        if use_fdr:
            subset = df[(df[col_sig] < sig_thresh) & (df[col_fc].abs() > cutoff)]
        else:
            subset = df[df[col_fc].abs() > cutoff]
        
        n_up = (subset[col_fc] > 0).sum()
        n_down = (subset[col_fc] < 0).sum()
        n_total = n_up + n_down
        
        if n_total > 0:
            # binomial test: H0 = up:down = 50:50
            pval = binomtest(n_up, n_total, p=0.5, alternative="two-sided").pvalue
        else:
            pval = np.nan
        
        results.append({
            "cutoff": cutoff,
            "n_up": n_up,
            "n_down": n_down,
            "total": n_total,
            "binom_pval": pval
        })
    return pd.DataFrame(results)

# (1) FDR correction 적용
binom_fdr = binomial_test_by_cutoff(res_df, use_fdr=True)
print("Wilcoxon (FDR corrected)")
print(binom_fdr)

# (2) FDR correction 없이
binom_raw = binomial_test_by_cutoff(res_df, use_fdr=False)
print("Wilcoxon (raw p)")
print(binom_raw)

#%%
####^^ volcano mannwhitney #################
from scipy.stats import mannwhitneyu

results_mwu = []

for transcript in df.index:
    normal_vals = []
    tumor_vals = []
    
    for sample in unique_samples:
        n_col = f"{sample}-N"
        t_col = f"{sample}-T"
        if n_col in df.columns:
            normal_vals.append(df.at[transcript, n_col])
        if t_col in df.columns:
            tumor_vals.append(df.at[transcript, t_col])
            
    normal_vals = pd.Series(normal_vals).dropna()
    tumor_vals = pd.Series(tumor_vals).dropna()

    if len(normal_vals) >= 2 and len(tumor_vals) >= 2:
        # unpaired two-sided Mann–Whitney test
        stat, pval = mannwhitneyu(tumor_vals, normal_vals, alternative="two-sided")
        mean_normal = np.mean(normal_vals)
        mean_tumor = np.mean(tumor_vals)
        log2fc = np.log2((mean_tumor + 1e-6) / (mean_normal + 1e-6))
        
        results_mwu.append({
            "transcript": transcript,
            "log2FC": log2fc,
            "pval": pval
        })

# DataFrame으로 변환
res_df_mwu = pd.DataFrame(results_mwu)
res_df_mwu["-log10(pval)"] = -np.log10(res_df_mwu["pval"])

# FDR 보정
res_df_mwu["FDR"] = multipletests(res_df_mwu["pval"], method="fdr_bh")[1]
res_df_mwu["-log10(FDR)"] = -np.log10(res_df_mwu["FDR"])

# (1) FDR correction 적용
binom_fdr_mwu = binomial_test_by_cutoff(res_df_mwu, use_fdr=True)
print("Mann–Whitney (FDR corrected)")
print(binom_fdr_mwu)

# (2) FDR correction 없이
binom_raw_mwu = binomial_test_by_cutoff(res_df_mwu, use_fdr=False)
print("Mann–Whitney (raw p)")
print(binom_raw_mwu)




# %%
###^^^ Binomial Test ######################
from scipy.stats import binomtest

# p < 0.05 필터링
sig = res_df[res_df["pval"] < 0.05]

# log2FC > 0 vs < 0 개수 세기
pos = (sig["log2FC"] > 0).sum()
neg = (sig["log2FC"] < 0).sum()

# binomial test: 성공 횟수 = pos, 시도 횟수 = pos + neg, 기대확률 = 0.5
bt_1 = binomtest(pos, pos + neg, p=0.5, alternative='two-sided')
print("p < 0.05 기준 binomial test 결과:", bt_1)

# log2FC 절댓값이 1 이상인 애들만 필터링
sig_strong = sig[sig["log2FC"].abs() > 1]

# log2FC > 1 vs < -1
pos_strong = (sig_strong["log2FC"] > 0).sum()
neg_strong = (sig_strong["log2FC"] < 0).sum()

# binomial test
bt_2 = binomtest(pos_strong, pos_strong + neg_strong, p=0.5, alternative='two-sided')
print("p < 0.05 and |log2FC| > 1 기준 binomial test 결과:", bt_2)

#%%
###^^^^^^^^^^^^^^^ Volcano plot Plotting ##########################

checklist = dut[dut['qry_id'].isin(check_qry2)]['ref_id'].to_list()
CAG_checklist = dut[dut['qry_gene_id'].isin(CAGlist)]['ref_id'].to_list()

# 컬러 그룹 정의
def color_group(row):
    if row["pval"] < 0.05:
        if row["log2FC"] > 1.5:
            return "Cancer-preferred"
        elif row["log2FC"] < -1.5:
            return "Normal-preferred"
    return "Not significant"

res_df["group"] = res_df.apply(color_group, axis=1)
#res_df_mwu["group"] = res_df_mwu.apply(color_group, axis=1)

# 색상 맵
palette = {
    "Cancer-preferred": "#E33C89",        # 빨강
    "Normal-preferred": "#4B74B6",      # 파랑
    "Not significant": "lightgray"
}

# Volcano plot
plt.figure(figsize=(6, 6))
ax = sns.scatterplot(
    #data=res_df[res_df['transcript'].isin(checklist)],
    data=res_df,
    x="log2FC",
    y="-log10(pval)",
    hue="group",
    palette=palette,
    edgecolor=None,
    s=40,
    alpha=0.8,
    legend =False
)

# 기준선
plt.axhline(-np.log10(0.05), color="gray", linestyle="--", alpha=0.8, linewidth=0.8 )   # p = 0.05
plt.axvline(1.5, color="gray", linestyle="--", alpha=0.8, linewidth=0.8 )                 # log2FC = +1
plt.axvline(-1.5, color="gray", linestyle="--", alpha=0.8, linewidth=0.8)     # log2FC = -1

import matplotlib.patches as mpatches

handles = [
    mpatches.Patch(color="#E33C89", label="Cancer"),
    mpatches.Patch(color="#4B74B6", label="Normal")
]
ax.legend(handles=handles, title="", loc="upper left", frameon=False)


from adjustText import adjust_text

# transcript → gene 매핑
ref_to_gene = dut.set_index("ref_id")["qry_gene_id"].to_dict()

# # 라벨링 대상 (checklist + CAG_checklist + pval < 0.05)
# highlight_df = res_df[
#     (res_df["transcript"].isin(checklist)) &
#     (res_df["pval"] < 0.05) &
#     (res_df["log2FC"] > 0)
# ]


# for _, row in highlight_df.iterrows():
#     tx = row["transcript"]
#     gene = ref_to_gene.get(tx, "")
#     label = f"{tx} - {gene}" if gene else tx

#     ax.annotate(
#         label,
#         xy=(row["log2FC"], row["-log10(pval)"]),
#         xytext=(row["log2FC"] + 0.5, row["-log10(FDR)"] + 1.5),
#         textcoords='data',
#         arrowprops=dict(arrowstyle='->', color='black', lw=0.5),
#         fontsize=8,
#         ha='left'
#     )


# sns.scatterplot(
#     data=highlight_df,
#     x="log2FC",
#     y="-log10(FDR)",
#     color="#E33C89",    # 진한 분홍 (기존보다 더 진하게 하거나 바꿔도 됨)
#     edgecolor="black",
#     s=40,               # 마커 크기
#     alpha=1,
#     zorder=3,           # 위에 그려지게
#     ax=ax               # 기존 plot 위에 덧그리기
# )


# 기타 설정
plt.xlim([-5, 5])
plt.xlabel("log2FC")
plt.ylabel("-log10(pval)")
plt.tight_layout()
sns.despine()
plt.savefig("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/figures/cDUT_volcano_wilcoxon_pval_15.pdf", bbox_inches='tight', dpi=300)
plt.show()

#%%
ref_to_gene = dut.set_index("ref_id")["qry_gene_id"].to_dict()

# 2. label 컬럼: "ENST... - GENE" 형식
res_df["label"] = res_df["transcript"].apply(
    lambda tx: f"{tx} - {ref_to_gene.get(tx, '')}"
)

# 3. highlight 컬럼: checklist 또는 CAG_checklist 포함 여부
highlight_set = set(checklist + CAG_checklist)
# 조건을 모두 만족하는 row만 True
res_df["highlight"] = (
    res_df["transcript"].isin(highlight_set) &
    (res_df["pval"] < 0.05) &
    (res_df["log2FC"] > 0)
)

#res_df.to_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/cDUT_volcano_data.csv', sep='\t',index=False)
#%%
#####^^ TSG + Frameshift 확인 ##########

from statannotations.Annotator import Annotator
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

tsglist = ['ENST00000304494.10','ENST00000262713.7','ENST00000267163.6','ENST00000439696.3']

for tx_id in tsglist:
    expr_list = []

    for sample in unique_samples:
        n_col = f"{sample}-N"
        t_col = f"{sample}-T"
        if n_col in df.columns and t_col in df.columns:
            expr_list.append({"sample": sample, "condition": "Normal", "expression": df.at[tx_id, n_col]})
            expr_list.append({"sample": sample, "condition": "Tumor", "expression": df.at[tx_id, t_col]})

    # 데이터프레임 만들기
    plot_df = pd.DataFrame(expr_list).dropna()

    # box + swarm plot
    plt.figure(figsize=(4, 5))
    ax = sns.boxplot(data=plot_df, x="condition", y="expression", showfliers=False,
                     palette={"Normal": "#4B74B6", "Tumor": "#E33C89"}, order=["Normal", "Tumor"])
    sns.swarmplot(data=plot_df, x="condition", y="expression", color=".25", size=3)

    # 통계 Annotator
    annot = Annotator(ax, pairs=[("Normal", "Tumor")], data=plot_df,
                      x="condition", y="expression")
    annot.configure(test="Mann-Whitney", text_format='simple', loc='inside', comparisons_correction=None)
    annot.apply_and_annotate()

    # 제목 및 정리
    ax.set_title(f"{tx_id}")
    ax.set_ylabel("Transcript Usage")
    ax.set_xlabel("")
    sns.despine()
    plt.tight_layout()
    plt.show()


#%%
###^^^ Distribution vs. normal dist ################
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import norm, ks_2samp

# log2FC 값 (유의한 것만)
sig_vals = res_df[(res_df["pval"] < 0.05)]["log2FC"].dropna()

# 평균, 표준편차
mu = sig_vals.mean()
sigma = sig_vals.std()

# 이론 정규분포 정의
x = np.linspace(-3 * sigma, 3 * sigma, 1000)
y = norm.pdf(x, loc=0, scale=sigma)

normal_sample = np.random.normal(loc=0, scale=sigma, size=len(sig_vals))
ks_stat, ks_pval = ks_2samp(sig_vals, normal_sample)

from scipy.stats import skew

skewness = skew(sig_vals)
print("Skewness:", skewness)

# 플롯
plt.figure(figsize=(6, 5))

plt.axvline(0, color='black', linestyle='--', linewidth=0.5, alpha=0.8)
plt.plot(x, y, linestyle='--', color='#559502', linewidth=0.8, label='Normal Distribution', alpha=0.5)
plt.fill_between(x, y, color='#72C406', alpha=0.2)
sns.kdeplot(sig_vals, color='orange', fill=True, label='Observed log2FC', alpha=0.4, linewidth=1, linestyle='--')
plt.text(0.9, 0.7, f"K–S pval = {ks_pval:.2e}", transform=plt.gca().transAxes,
         ha='right', va='top', fontsize=11, color='black')
# 라벨링
plt.title("cDUT (p < 0.05)")
plt.xlabel("log2FC")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.xlim([-9, 9])
plt.ylim([0, 0.65])
sns.despine()
plt.savefig("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/figures/cDUT_log2FC_dist.pdf", bbox_inches='tight', dpi=300)
plt.show()

#%%
####^^ Histogram #############
import matplotlib.pyplot as plt
import seaborn as sns

sig_vals = res_df[res_df["pval"] < 0.05]["log2FC"].dropna()
pos_vals = sig_vals[sig_vals > 0]
neg_vals = sig_vals[sig_vals < 0]

plt.figure(figsize=(6, 4))
sns.histplot(pos_vals, color='#E33C89', bins=60, label='Cancer', alpha=1)
sns.histplot(neg_vals, color='#4B74B6', bins=60, label='Normal', alpha=1)
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.title("cDUT (p < 0.05)")
plt.xlabel("log2FC")
plt.ylabel("Count")
plt.legend()
plt.xlim([-5,5])
plt.tight_layout()
sns.despine()
#plt.savefig('/home/jiye/jiye/nanopore/HG_JH_check/cDUT/cDUT_hist.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

# 유의한 log2FC 값
sig_vals = res_df[res_df["pval"] < 0.05]["log2FC"].dropna()

# Histogram
fig, ax1 = plt.subplots(figsize=(7, 5))

# 왼쪽 y축: 히스토그램 (count)
sns.histplot(pos_vals, color='#E33C89', bins=60, label='Cancer', alpha=1, ax=ax1, zorder=3)
sns.histplot(neg_vals, color='#4B74B6', bins=60, label='Normal', alpha=1, ax=ax1, zorder=3)
ax1.set_xlabel("log2FC")
ax1.set_ylabel("Count")
ax1.tick_params(axis='y')

# 오른쪽 y축: 이론 정규분포 (density)
ax2 = ax1.twinx()
x = np.linspace(sig_vals.min()-1, sig_vals.max()+1, 1000)
mu = sig_vals.mean()
sigma = sig_vals.std()
pdf = norm.pdf(x, loc=mu, scale=sigma)

ax2.plot(x, pdf, color='grey', linestyle='--', label='Normal (μ, σ)', linewidth=1, alpha=0.2, zorder=2)
ax2.fill_between(x, 0, pdf, color='grey', alpha=0.1, zorder=1)
ax2.set_ylabel("Density")
ax2.tick_params(axis='y')
ax2.set_ylim(0, 0.6)

fig.suptitle("Histogram of log2FC (p < 0.05) with Normal Overlay", fontsize=14)
fig.tight_layout()
fig.subplots_adjust(top=0.9)

plt.xlim([-5, 5])

# 정규분포 최대값 (이론적)
max_density = norm.pdf(mu, loc=mu, scale=sigma)

# 히스토그램 최대값 (관측된 count 기준)
max_count = sig_vals.value_counts().max()

# 비율로 스케일 맞추기
scale_factor = max_count / max_density

# 오른쪽 y축 스케일 고정 (count에 맞춰서 정규분포 밀도 스케일 조절)
ax2.set_ylim(0, max_count * 1.2)

plt.show()


# %%

# %%

# %%

# %%

# %%
###^^^^ Per SAMPLE ####################
row = df.loc[df.index.isin(dutlist)]
row = row.mean()
# sample ID 추출
samples = [col.rsplit('-', 1)[0] for col in row.index]

# normal-tumor 매칭
data = []
for sample in set(samples):
    n_col = f"{sample}-N"
    t_col = f"{sample}-T"
    if n_col in row and t_col in row:
        data.append({
            "sample": sample,
            "Normal": row[n_col],
            "Tumor": row[t_col]
        })

# long format
plot_df = pd.DataFrame(data).melt(id_vars="sample", var_name="condition", value_name="expression")

# plot
plt.figure(figsize=(5, 5))
ax = sns.boxplot(data=plot_df, x="condition", y="expression", showfliers=False, palette=["#4B74B6", "#E33C89"])

# 점을 깔끔하게 찍기: stripplot (작게, jitter)
sns.stripplot(data=plot_df, x="condition", y="expression", color='black', size=3, jitter=False)

# paired line
for s in plot_df["sample"].unique():
    vals = plot_df[plot_df["sample"] == s].sort_values("condition")["expression"].values
    plt.plot(["Normal", "Tumor"], vals, color="gray", linewidth=0.5, alpha=0.3)

# 통계
normal_vals = plot_df[plot_df["condition"] == "Normal"]["expression"].values
tumor_vals = plot_df[plot_df["condition"] == "Tumor"]["expression"].values
stat, pval = wilcoxon(normal_vals, tumor_vals)
plt.title(f"{transcript_id} (p = {pval:.3e})")

# annotation
annot = Annotator(ax, pairs=[("Normal", "Tumor")], data=plot_df, x="condition", y="expression")
annot.configure(test='Wilcoxon', text_format='star', loc='inside')
annot.apply_and_annotate()

# legend 제거!
ax.get_legend().remove() if ax.get_legend() else None

plt.tight_layout()
plt.title("pDUT (per sample)")
plt.ylabel("TU")
plt.xlabel("")
plt.show()


# %%
###^^ Survival Analysis #############################
clinical = pd.read_csv('/home/jiye/jiye/nanopore/FINALDATA/clinical_data_relapse.txt', sep='\t')
clinical.index = clinical.iloc[:,0].str[:-2]  # clean sample ID 

TU_df = pd.read_csv('/home/jiye/jiye/nanopore/FINALDATA/matched_TU.txt', sep='\t', index_col=0)
TU_df.index = TU_df.index.str.split("-", n=1).str[0]  # clean transcript ID

tmap = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/pDUT/compare_out.pDUT_COAD_hg38.gtf.tmap', sep='\t')
tmap = tmap[tmap['class_code']=='=']  
ref_qry_map = dict(zip(tmap["qry_id"], tmap["ref_id"]))


# 1. pDUT_label: index가 qry_id임 → ref_id로 매핑
label_df = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/pDUT/pDUT_label.txt', sep='\t', index_col=0, header=None)
label_df.columns = ['label']
label_df["ref_id"] = label_df.index.map(ref_qry_map)

# 2. TU_df에 있는 transcript만 남기기
label_df = label_df[label_df["ref_id"].isin(TU_df.index)]

# 3. label_dict 만들기 (index: ref_id, value: label)
label_dict = label_df.set_index("ref_id")["label"].to_dict()

import pandas as pd
from lifelines.statistics import logrank_test

# 1. 준비
tu_df = TU_df.copy()
tu_df = tu_df.iloc[:,1::2]
tu_df.columns = tu_df.columns.str.replace('-T', '', regex=False)

# 3. clinical 정제 (OS + survival 사용)
available_samples = list(set(tu_df.columns) & set(clinical.index))

############################ DFS / OS ################################
clinical_sub = clinical.loc[available_samples].dropna(subset=["DFS", "Relapse"]).copy()

# "Alive"/"Dead" → 0/1 매핑
clinical_sub["event"] = clinical_sub["Relapse"].map({"No": 0, "Yes": 1})
############################ DFS / OS ################################


# 샘플 리스트 최종 확정
samples_final = clinical_sub.index

# 4. 생존 분석 루프
results = []

for tx_id, label in label_dict.items():
    if tx_id not in tu_df.index:
        continue

    expr = tu_df.loc[tx_id, samples_final]
    median = expr.median()
    group = (expr > median).map({True: "high", False: "low"})

    df = pd.DataFrame({
        "group": group,
        "DFS": clinical_sub["DFS"],
        "event": clinical_sub["event"]
    }).dropna()

    if df["group"].nunique() < 2:
        continue

    try:
        result = logrank_test(
            df[df.group == "high"]["DFS"],
            df[df.group == "low"]["DFS"],
            event_observed_A=df[df.group == "high"]["event"],
            event_observed_B=df[df.group == "low"]["event"]
        )
        pval = result.p_value
        high_event_rate = df[df.group == "high"]["event"].mean()
        low_event_rate = df[df.group == "low"]["event"].mean()
        better_group = "high" if high_event_rate < low_event_rate else "low"
        match = (label == f"better_{better_group}")
        
        results.append({
            "transcript": tx_id,
            "label": label,
            "better_group_in_survival": better_group,
            "pval": pval,
            "match": match
        })

    except:
        continue

# 5. 결과 요약
result_df = pd.DataFrame(results)

# %%
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts

add_result_df = pd.merge(result_df,tmap,left_on='transcript', right_on='ref_id', how='left')

#%%
def plot_km_for_top_transcripts(result_df, TU_df, clinical_sub, top_n=12):
    top_hits = result_df[result_df["pval"] < 0.05].sort_values("pval").head(top_n)

    for i, row in top_hits.iterrows():
        tx_id = row["transcript"]
        label = row["label"]
        pval = row["pval"]

        # 발현 + 생존 데이터 준비
        expr = TU_df.loc[tx_id].reindex(clinical_sub.index)
        median = expr.median()
        group = (expr > median).map({True: "high", False: "low"})

        df = pd.DataFrame({
            "group": group,
            "DFS": clinical_sub["DFS"],
            "event": clinical_sub["event"]
        }).dropna()

        if df["group"].nunique() < 2:
            print(f"⚠️ Skipping {tx_id}: only one group present.")
            continue

        mask_high = df["group"] == "high"
        mask_low = df["group"] == "low"

        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()

        kmf_high.fit(df[mask_high]["DFS"], df[mask_high]["event"], label=f"High")
        kmf_low.fit(df[mask_low]["DFS"], df[mask_low]["event"], label=f"Low")

        sns.set_style("white")
        fig, ax = plt.subplots(figsize=(6, 5))
        kmf_high.plot(ax=ax, show_censors=True, ci_show=False, color='red')
        kmf_low.plot(ax=ax, show_censors=True, ci_show=False, color='blue')
        ax.set_ylim(0, 1)
        table_ax = add_at_risk_counts(kmf_low, kmf_high, ax=ax, rows_to_show=["At risk"])
        table_ax.spines['top'].set_visible(False)
        
        gene_id = add_result_df[add_result_df['transcript']==tx_id]['qry_gene_id'].values[0]
        ref_id = add_result_df[add_result_df['transcript']==tx_id]['qry_id'].values[0]

        ax.set_title(f"{ref_id}-{gene_id} ({label})\nlog-rank p = {pval:.2e}")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("DFS (COAD)")
        
        
        plt.tight_layout()
        ax.spines["right"].set_visible(False)
        #plt.savefig('/home/jiye/jiye/nanopore/HG_JH_check/pDUT/'+f"{ref_id}_{gene_id}_survival.pdf", dpi=300, bbox_inches='tight')
        plt.show()




plot_km_for_top_transcripts(result_df, tu_df, clinical_sub, top_n=12)


#%%
###^^ transcript 별 sample boxplot #########
import seaborn as sns
import matplotlib.pyplot as plt

from statannotations.Annotator import Annotator

def plot_expression_boxplots_for_top_transcripts(result_df, TU_df, top_n=3):
    top_hits = result_df[result_df["pval"] < 0.05].sort_values("pval").head(top_n)

    for _, row in top_hits.iterrows():
        tx_id = row["transcript"]
        label = row["label"]

        # 발현값 추출
        all_samples = list(set([col.rsplit("-", 1)[0] for col in TU_df.columns]))

        data = []
        for sample in all_samples:
            n_col = f"{sample}-N"
            t_col = f"{sample}-T"
            if n_col in TU_df.columns and t_col in TU_df.columns:
                data.append({
                    "sample": sample,
                    "condition": "Normal",
                    "expression": TU_df.at[tx_id, n_col]
                })
                data.append({
                    "sample": sample,
                    "condition": "Tumor",
                    "expression": TU_df.at[tx_id, t_col]
                })

        plot_df = pd.DataFrame(data)

        # box + swarm plot
        plt.figure(figsize=(4, 5))
        ax = sns.boxplot(data=plot_df, x="condition", y="expression", showfliers=False,
                         palette={"Normal": "#4B74B6", "Tumor": "#E33C89"})
        sns.swarmplot(data=plot_df, x="condition", y="expression", color=".25", size=3)

        # 통계 Annotator
        annot = Annotator(ax, pairs=[("Normal", "Tumor")], data=plot_df, x="condition", y="expression")
        annot.configure(test="Mann-Whitney", text_format='simple', loc='inside', comparisons_correction=None)
        annot.apply_and_annotate()
        gene_id = add_result_df[add_result_df['transcript']==tx_id]['qry_gene_id'].values[0]
        ref_id = add_result_df[add_result_df['transcript']==tx_id]['qry_id'].values[0]

        # 제목 및 정리
        ax.set_title(f"{ref_id}-{gene_id} ({label})")
        ax.set_ylabel("Transcript Usage")
        ax.set_xlabel("")
        sns.despine()
        plt.tight_layout()
        #plt.savefig('/home/jiye/jiye/nanopore/HG_JH_check/pDUT/'+f"{ref_id}_{gene_id}_boxplot.pdf", dpi=300, bbox_inches='tight')
        plt.show()


plot_expression_boxplots_for_top_transcripts(result_df, TU_df, top_n=3)









# %%
###^^ MINJU DET list #######
detlist = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/DESeq2/deseq2_det.txt', sep='\t', index_col=0)
# %%
det = detlist[(detlist['padj']<0.01) & (detlist['log2FoldChange']>2)]
finallist = det.index.to_list()
with open("/home/jiye/jiye/nanopore/202411_analysis/DESeq2/minju/hard_tumor_up.txt", "w") as f:
    for item in finallist:
        f.write(item + "\n")
# %%
