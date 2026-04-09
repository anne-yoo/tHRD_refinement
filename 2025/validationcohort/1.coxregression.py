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
val_df = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/116_validation_TU_group.txt', sep='\t', index_col=0)
val_gene = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/116_validation_gene_TPM_symbol_group.txt', sep='\t', index_col=0)
val_clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_validation_clinicalinfo_new.txt', sep='\t', index_col=0)
val_df = val_df.iloc[:-1,:-1]
val_df = val_df.apply(pd.to_numeric, errors='coerce')
vallist = list(val_df.columns)
val_clin = val_clin.loc[vallist,:]
genesym = val_gene.iloc[:,-1]
val_gene = val_gene.iloc[:-1,:-1]
val_gene = val_gene.apply(pd.to_numeric, errors='coerce')



# %%
###^^^ Univariate Cox regression with forest plot###############

from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

clin = val_clin.copy()

pred_proba = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/HGmodel/116_HGmodel_proba.txt', sep='\t', index_col=1)
clin = pd.merge(clin, pred_proba, left_index=True, right_index=True)

# Step 1: event column 정의
clin["event"] = clin["ongoing"].map({
    0: 1,  # PD로 투약 종료 → progression
    3: 1,  # pre-post 코호트 내 resistance → progression
    1: 0,  # ongoing → censored
    2: 0,  # NER로 종료 → censored
    4: 0,  # 다른 치료로 종료 → censored
})

druginfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/validation_sampleinfo.txt', sep='\t', index_col=0)
druginfo = druginfo['drug']

clin = clin.merge(druginfo, left_index=True, right_index=True, how='left')

clin['gHRD'] = clin['gHRDscore'].apply(lambda x: 1 if x >= 42 else 0)

#############################################
clin = clin[clin['OM/OS']=='maintenance']
#############################################

# Step 1: 데이터 전처리
df = clin[["interval", "event", "BRCAmut", "gHRD", "pred_HRD"]].dropna()
df.columns = ["interval", "event", "BRCAmut", "gHRD", "tHRD"]

# tHRD z-score 변환
scaler = StandardScaler()
df["tHRD_scaled"] = scaler.fit_transform(df[["tHRD"]])
df = df.drop(columns=["tHRD"])

# Step 2: 변수 리스트
univ_vars = ["gHRD", "BRCAmut"]
label_map = {
    "gHRD": "gHRD",
    "BRCAmut": "BRCA mutation"
}

# Step 3: 각 변수에 대해 univariate Cox regression 수행
results = []
for var in univ_vars:
    temp = df[["interval", "event", var]]
    cph = CoxPHFitter()
    cph.fit(temp, duration_col="interval", event_col="event")
    row = cph.summary.loc[var]
    results.append({
        "label": label_map[var],
        "HR": row["exp(coef)"],
        "CI_lower": row["exp(coef) lower 95%"],
        "CI_upper": row["exp(coef) upper 95%"],
        "p": row["p"]
    })

plot_df = pd.DataFrame(results)

# Step 4: Forest plot
fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(plot_df["HR"], plot_df["label"],
            xerr=[plot_df["HR"] - plot_df["CI_lower"],
                  plot_df["CI_upper"] - plot_df["HR"]],
            fmt='o', color='black', ecolor='black', elinewidth=1.2, capsize=4)
ax.axvline(x=1, color='black', linestyle='--', alpha=0.4, linewidth=0.8)

for i, row in plot_df.iterrows():
    ax.text(row["HR"] + 0.05, i + 0.08, f'HR={row["HR"]:.2f}, p={row["p"]:.2f}', va='center', fontsize=9)

ax.set_xlabel("Hazard Ratio (95% CI)")
plt.tight_layout()
plt.grid(False)
plt.box(False)
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/univariate_coxregression_salvage.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
####^^^ Cox regression with OM/OS variable + BRCAmut + gHRD score ######
###^^^ Mutivriate Cox regression with forest plot###############

from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

clin = val_clin.copy()

# Step 1: event column 정의
clin["event"] = clin["ongoing"].map({
    0: 1,  # PD로 투약 종료 → progression
    3: 1,  # pre-post 코호트 내 resistance → progression
    1: 0,  # ongoing → censored
    2: 0,  # NER로 종료 → censored
    4: 0,  # 다른 치료로 종료 → censored
})

druginfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/validation_sampleinfo.txt', sep='\t', index_col=0)
druginfo = druginfo['drug']

clin = clin.merge(druginfo, left_index=True, right_index=True, how='left')
clin['gHRD'] = clin['gHRDscore'].apply(lambda x: 1 if x >= 42 else 0)
# 각 group별로 돌리기
grouped_results = []

for cohort in ['All', 'maintenance', 'salvage']:
    if cohort == 'All':
        df_sub = clin[["interval", "event", "BRCAmut", "gHRD", "OM/OS"]].dropna()
    else:
        df_sub = clin[clin["OM/OS"] == cohort][["interval", "event", "BRCAmut", "gHRD"]].dropna()

    temp = df_sub[["interval", "event", "BRCAmut", "gHRD"]]
    cph = CoxPHFitter()
    try:
        cph.fit(temp, duration_col="interval", event_col="event")
        for var in ["BRCAmut", "gHRD"]:
            row = cph.summary.loc[var]
            grouped_results.append({
                "label": var,
                "cohort": cohort,
                "HR": row["exp(coef)"],
                "CI_lower": row["exp(coef) lower 95%"],
                "CI_upper": row["exp(coef) upper 95%"],
                "p": row["p"]
            })
    except:
        continue

plot_df = pd.DataFrame(grouped_results)

#%%
import matplotlib.pyplot as plt

# plot_df는 다음 컬럼을 포함해야 함:
# ['label', 'cohort', 'HR', 'CI_lower', 'CI_upper', 'p', 'label_cohort']

# 색상: salvage는 빨강, 나머지는 회색
plot_df["color"] = plot_df["cohort"].map({
    "salvage": "#d62728",       # 빨간색
    "maintenance": "gray",
    "All": "gray"
})

# y축 순서를 위해 정렬
plot_df = plot_df.sort_values(by=["label", "cohort"], ascending=[True, False])
import matplotlib.pyplot as plt

# 1. label_cohort 컬럼 만들기
plot_df["label_cohort"] = plot_df["label"] + " (" + plot_df["cohort"] + ")"

# 2. 색상 지정
plot_df["color"] = plot_df["cohort"].map({
    "salvage": "#000000",       # 빨강
    "maintenance": "#000000",
    "All": "#6C6C6C"
})

# 3. label_cohort 기준 정렬
plot_df = plot_df.sort_values(by="label_cohort", ascending=False)
plot_df["y_pos"] = range(len(plot_df), 0, -1) 
# 4. Plot
fig, ax = plt.subplots(figsize=(7, 4))
# 수정된 text 위치: 아래쪽으로 조금 내림 (va='top' + y좌표 -0.1 보정)
for i, row in plot_df.iterrows():
    ax.errorbar(
        row["HR"], row["label_cohort"],
        xerr=[[row["HR"] - row["CI_lower"]], [row["CI_upper"] - row["HR"]]],
        fmt='o', color=row["color"], ecolor=row["color"], elinewidth=1.5, capsize=4
    )
    ax.text(row["CI_upper"] + 0.05, row["label_cohort"], f'HR={row["HR"]:.2f}, p={row["p"]:.2f}', 
        va='center', fontsize=10, color=row["color"])


# 기준선
ax.axvline(x=1, color='black', linestyle='--', linewidth=0.8, alpha=0.6)

# 스타일링
ax.set_xlabel("Hazard Ratio (95% CI)")
plt.title("")
plt.tight_layout()
plt.grid(False)
plt.box(False)
ytick_labels = ['gHRD (salvage)', 'gHRD (maintenance)', 'gHRD (whole)', 'BRCA mt (salvage)', 'BRCA mt (maintenance)', 'BRCA mt (whole)']
ax.set_yticks(range(len(ytick_labels)))
ax.set_yticklabels(ytick_labels)
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/correlation_brca_ghrd.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
clin['group'] = clin['type'].replace({'CR': 'R', 'IR': 'NR', 'AR': 'R'})

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 예시: clin 데이터프레임에 'group'과 'response' 컬럼이 있다고 가정
# clin = pd.read_csv("your_clinical_data.csv")

# group과 response 기준으로 count 집계
plot_df = clin.groupby(['OM/OS', 'group']).size().reset_index(name='count')

# barplot 그리기
plt.figure(figsize=(4, 5))
sns.barplot(
    data=plot_df,
    x='OM/OS',
    y='count',
    hue='group',
    hue_order=['R', 'NR'],
    palette={'R': '#FF8286', 'NR': '#77CDFF'}
)

# plot 꾸미기
plt.xlabel('')
plt.ylabel('count')
plt.legend(title='Response')
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
sns.despine()
plt.title('Sample Count')
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/samplecount_val.pdf', dpi=300, bbox_inches='tight')
plt.show()

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 그림 사이즈 조정
plt.figure(figsize=(4, 5))
sns.boxplot(
    data=clin,
    x='OM/OS',
    y='interval',
    hue='group',
    hue_order=['R', 'NR'],
    palette={'R': '#FF8286', 'NR': '#77CDFF'},
)
plt.xlabel('')
plt.ylabel('days')
plt.legend([], [], frameon=False)
sns.despine()
plt.title('treatment duration')
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/sampleinfo_duration_val.pdf', dpi=300, bbox_inches='tight')
plt.show()

# === 2. BRCAmut countplot by response ===
brca_mt = clin[clin['BRCAmut'] == True]

# barplot (countplot) - BRCA mutation 환자들의 R vs NR 비율
plt.figure(figsize=(4, 5))
sns.countplot(
    data=brca_mt,
    x='OM/OS',  # 혹은 x='response'로 바로 나눠도 됨
    hue='group',
    hue_order=['R', 'NR'],
    palette={'R': '#FF8286', 'NR': '#77CDFF'}
)
plt.xlabel('')
plt.ylabel('mutation count')
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.legend(title='Response')
sns.despine()
plt.legend([], [], frameon=False)
plt.title('BRCA mt')
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/sampleinfo_BRCAmt_val.pdf', dpi=300, bbox_inches='tight')
plt.show()

# === 3. gHRDscore violinplot by response ===
plt.figure(figsize=(4, 5))
sns.boxplot(
    data=clin,
    x='OM/OS',
    y='gHRDscore',
    hue='group',
    hue_order=['R', 'NR'],
    palette={'R': '#FF8286', 'NR': '#77CDFF'},

)
plt.ylabel('score')
plt.xlabel('')
plt.tight_layout()
plt.legend([], [], frameon=False)
sns.despine()
plt.title('gHRD score')
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/sampleinfo_gHRDscore_val.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
# === 4. gBRCA binary group ===
clin['gHRD'] = clin['gHRDscore'].apply(lambda x: 'HRD' if x >= 42 else 'HRP')
plt.figure(figsize=(4, 5))
sns.barplot(
    data=clin,
    x='OM/OS',
    y='gHRDscore',
    hue='gHRD',
    hue_order=['HRD', 'HRP'],
    palette={'HRD': '#FF8286', 'HRP': '#77CDFF'},

)
plt.ylabel('count')
plt.xlabel('')
plt.tight_layout()
sns.despine()
plt.title('gHRD binary')
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/sampleinfo_gHRDscore_val.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%

plt.rcParams.update({
'axes.titlesize': 11,     # 제목 글꼴 크기
'axes.labelsize': 11,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 11,    # x축 틱 라벨 글꼴 크기

'ytick.labelsize': 11,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 11,
'legend.title_fontsize': 11, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})

from upsetplot import UpSet, from_indicators
import matplotlib.pyplot as plt
clintmp = clin[clin['OM/OS']!='maintenance']

df_bool = pd.DataFrame(index=clintmp.index)
df_bool['BRCA mt'] = clintmp['BRCAmut'] == 1
df_bool['BRCA wt'] = clintmp['BRCAmut'] == 0
df_bool['gHRD'] = clintmp['gHRD'] == 'HRD'
df_bool['gHRP'] = clintmp['gHRD'] == 'HRP'

# Step 2: subset별 count를 MultiIndex Series로 만들기
subset_counts = df_bool.groupby(['gHRP', 'gHRD', 'BRCA wt', 'BRCA mt']).size()

subset_counts.index = subset_counts.index.reorder_levels(['gHRP', 'BRCA wt', 'gHRD', 'BRCA mt'])
subset_counts = subset_counts.sort_index()


from upsetplot import plot

fig = plt.figure(figsize=(7, 4))
plot(subset_counts, fig=fig, sort_by='cardinality',
    sort_categories_by=None, element_size=None)
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/salvage_upsetplot.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%

