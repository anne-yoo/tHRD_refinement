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
sev_tu = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/112_PARPi_transcript_usage.txt', sep='\t', index_col=0)
sev_clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/112_PARPi_clinicalinfo.txt', sep='\t', index_col=0)
polo_tu = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/83_POLO_transcript_usage.txt', sep='\t', index_col=0)
polo_clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/83_POLO_clinicalinfo.txt', sep='\t', index_col=0)

# %%
polo_clin_mod = polo_clin[['recur', 'PFS','OM/OS','BRCAmut','line','response']]
polo_clin_mod.columns = ['recur','PFS','setting','BRCAmt','line','response']
polo_clin_mod['drug'] = 'Niraparib'
sev_clin_mod = sev_clin[['recur', 'PFS','setting','BRCAmt','line','drug','response']]

final_tu = pd.concat([sev_tu, polo_tu], axis=1)
final_clin = pd.concat([sev_clin_mod, polo_clin_mod], axis=0)

majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = majorminor[majorminor['type']=='major']['gene_ENST'].to_list()
major_tu_df = final_tu.loc[final_tu.index.isin(majorlist), :]
major_tu_sev = sev_tu.loc[sev_tu.index.isin(majorlist), :]
major_tu_polo = polo_tu.loc[polo_tu.index.isin(majorlist), :]

# %%
from collections import OrderedDict

HR_restoration = [
    'AP5S1','APTX','BAZ1B','BRIP1','CDC45','CHEK2','DCLRE1C','FANCB','IFFO1',
    'INO80','LIG4','NSMCE2','NUCKS1','POLN','RAD21L1','REV3L','SFR1','SMARCAD1',
    'SMC6','XRCC3','XRCC4'
]

Cell_cycle_checkpoint_G2M = [
    'AJUBA','AURKA','CDK5RAP2','CDKN1A','CEP152','CEP78','CHEK2','FBXL15',
    'FOXM1','NDE1','PPP1R12A','TUBB4A'
]

Pro_survival_signaling = [
    'ACKR3','AMER1','CCL20','CD4','DUSP19','EIF2AK2','FBXW7','FGFR3','JUP',
    'KANK1','MAP3K11','NTRK2','PDGFA','PJA2','PRDM15','PRKD2','PSMB10','PTK2B',
    'PTPRC','RAP1A','RNF146','RNF220','SEMA4C','SPRY2','TBL1XR1','TLR2','TRAF2',
    'TRAF3','UNC5CL','USP47','WNT7A','XIAP'
]

AR_genelist = OrderedDict()
AR_genelist['HR_restoration'] = HR_restoration
AR_genelist['Cell_cycle_checkpoint_G2M'] = Cell_cycle_checkpoint_G2M
AR_genelist['Pro_survival_signaling'] = Pro_survival_signaling

HR_restoration_transcripts = [
 'ENST00000263201.1-CDC45','ENST00000324138.3-FANCB','ENST00000336604.4-IFFO1',
 'ENST00000368802.3-REV3L','ENST00000378278.2-DCLRE1C','ENST00000382865.1-POLN',
 'ENST00000401393.3-INO80','ENST00000402989.1-SMC6','ENST00000404251.1-BAZ1B',
 'ENST00000404276.1-CHEK2','ENST00000409241.1-RAD21L1','ENST00000482687.1-APTX',
 'ENST00000511817.1-XRCC4','ENST00000555055.1-XRCC3','MSTRG.114270.18-NSMCE2',
 'MSTRG.17113.1-SFR1','MSTRG.30687.2-LIG4','MSTRG.49834.247-BRIP1',
 'MSTRG.51515.3-SMC6','MSTRG.62156.213-AP5S1','MSTRG.80592.1-SMARCAD1',
 'MSTRG.9436.8-NUCKS1', 
] #'ENST00000259008.2-BRIP1'
Cell_cycle_checkpoint_G2M_transcripts = [
 'ENST00000261207.5-PPP1R12A','ENST00000342628.2-FOXM1',
 'ENST00000349780.4-CDK5RAP2','ENST00000361265.4-AJUBA',
 'ENST00000373711.2-CDKN1A','ENST00000376597.4-CEP78',
 'ENST00000395913.3-AURKA','ENST00000396355.1-NDE1',
 'ENST00000399334.3-CEP152','ENST00000404276.1-CHEK2',
 'ENST00000540257.1-TUBB4A','MSTRG.17041.3-FBXL15',
 'MSTRG.62679.6-AURKA'
]
Pro_survival_signaling_transcripts = [
 'ENST00000247668.2-TRAF2','ENST00000260795.2-FGFR3','ENST00000269844.3-PRDM15',
 'ENST00000277120.3-NTRK2','ENST00000309100.3-MAP3K11','ENST00000309649.3-RNF146',
 'ENST00000354221.4-DUSP19','ENST00000355640.3-XIAP','ENST00000358514.4-PSMB10',
 'ENST00000358813.4-CCL20','ENST00000361799.2-RNF220','ENST00000373164.1-UNC5CL',
 'ENST00000393930.1-JUP','ENST00000393956.3-FBXW7','ENST00000395127.2-EIF2AK2',
 'ENST00000403336.1-AMER1','ENST00000433867.1-PRKD2','ENST00000442510.2-PTPRC',
 'ENST00000539466.1-USP47','ENST00000540649.1-SPRY2','ENST00000544172.1-PTK2B',
 'ENST00000560371.1-TRAF3','MSTRG.104716.1-PDGFA','MSTRG.115621.11-KANK1',
 'MSTRG.23811.6-CD4','MSTRG.3783.10-RAP1A','MSTRG.54701.40-SEMA4C',
 'MSTRG.60771.1-ACKR3','MSTRG.67773.4-WNT7A','MSTRG.75665.10-TBL1XR1',
 'MSTRG.83135.5-TLR2','MSTRG.83135.6-TLR2','MSTRG.88422.7-PJA2',
 'MSTRG.98706.19-RNF146'
]

AR_txlist = OrderedDict()
AR_txlist['HR_restoration'] = HR_restoration_transcripts
AR_txlist['Cell_cycle_checkpoint_G2M'] = Cell_cycle_checkpoint_G2M_transcripts
AR_txlist['Pro_survival_signaling'] = Pro_survival_signaling_transcripts

# %%
import pandas as pd
from collections import OrderedDict
from lifelines import CoxPHFitter
import statsmodels.api as sm

# -------------------
# 1. Feature construction
# -------------------
# major_tu_df: transcript (row) x sample (col)
# final_clin: sample metadata with ['recur','PFS','setting','BRCAmt','line','drug']

def build_features(major_tu_df, final_clin, AR_genelist):
    features = pd.DataFrame(index=final_clin.index)
    
    for group, genes in AR_genelist.items():
        # transcript → gene mapping이 transcript_id-gene 형식이라고 가정
        matching_transcripts = [t for t in major_tu_df.index if any(g in t for g in genes)]
        if not matching_transcripts:
            continue
        group_tu = major_tu_df.loc[matching_transcripts].mean(axis=0)  # 평균 TU per sample
        features[group] = group_tu
    
    return features

def preprocess_covariates(clin):
    df = clin.copy()

    # Line: remove 'L' and convert to int
    if 'line' in df.columns:
        df['line'] = df['line'].astype(str).str.replace('L','',regex=False).str.replace('+','',regex=False)
        df['line'] = pd.to_numeric(df['line'], errors='coerce')
    # Drug: one-hot encode
    if 'drug' in df.columns:
        df = pd.get_dummies(df, columns=['drug'], drop_first=True)
    
    return df

# -------------------
# Maintenance Cox regression
# -------------------
def run_cox_regression(features, clin):
    # Merge only at this step
    df = features.join(clin[['PFS','recur','BRCAmt','line','drug']])
    df = preprocess_covariates(df)
    df = df.dropna()

    df = df.rename(columns={'PFS':'time','recur':'event'})
    
    from lifelines import CoxPHFitter
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df, duration_col='time', event_col='event')
    return cph

# -------------------
# 3. Salvage: Logistic regression for response
# -------------------
def run_logistic_regression(features, clin):
    df = features.join(clin[['response']])  # salvage에서 binary response
    df = df.dropna()
    
    y = df['response']
    X = df.drop(columns=['response'])
    X = sm.add_constant(X)
    
    model = sm.Logit(y, X).fit()
    return model

# -------------------
# Example usage
# -------------------
features = build_features(major_tu_df, final_clin, AR_txlist)
maint_idx = final_clin.query("setting == 'maintenance'").index
salv_idx = final_clin.query("setting == 'salvage'").index

maint_features = features.loc[maint_idx]
salv_features = features.loc[salv_idx]

cox_model = run_cox_regression(maint_features, final_clin.loc[maint_idx])
logit_model = run_logistic_regression(salv_features, final_clin.loc[salv_idx])

print(cox_model.summary)
print(logit_model.summary())

summary = cox_model.summary  # cox_model is fitted CoxPHFitter
coef = summary['coef']
ci_low = summary['coef lower 95%']
ci_high = summary['coef upper 95%']

plt.figure(figsize=(6,4))
plt.errorbar(coef.index, coef, 
             yerr=[coef - ci_low, ci_high - coef], 
             fmt='o', capsize=4)
plt.axhline(0, color='grey', linestyle='--')
plt.xticks(rotation=90)
plt.ylabel("Cox regression coefficient (log HR)")
plt.title("Forest plot of Cox model")
plt.show()


params = logit_model.params
conf = logit_model.conf_int()
odds = np.exp(params)
odds_ci = np.exp(conf)

plt.figure(figsize=(6,4))
plt.errorbar(params.index, odds, 
             yerr=[odds - odds_ci[0], odds_ci[1] - odds], 
             fmt='o', capsize=4)
plt.axhline(1, color='grey', linestyle='--')
plt.xticks(rotation=90)
plt.ylabel("Odds ratio")
plt.title("Forest plot of Logistic regression")
plt.show()


# %%
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter

# -------------------
# 1. Feature construction (transcript-level)
# -------------------
# major_tu_df: transcript (row) x sample (col)
# AR_txlist: dict[group_name -> transcript list]

def build_features_transcript_level(major_tu_df, final_clin, AR_txlist):
    features = pd.DataFrame(index=final_clin.index)
    
    for group, transcripts in AR_txlist.items():
        matching = [t for t in major_tu_df.index if t in transcripts]
        if not matching:
            continue
        # transcript-level feature
        group_df = major_tu_df.loc[matching].T  # samples x transcripts
        # prefix transcript IDs with group name to avoid collisions
        group_df = group_df.add_prefix(group + "_")
        features = features.join(group_df, how="outer")
    
    return features

# -------------------
# 2. Clinical preprocessing
# -------------------
def preprocess_covariates(clin):
    df = clin.copy()
    if 'line' in df.columns:
        df['line'] = df['line'].astype(str).str.replace('L','',regex=False).str.replace('+','',regex=False)
        df['line'] = pd.to_numeric(df['line'], errors='coerce')
    if 'drug' in df.columns:
        df = pd.get_dummies(df, columns=['drug'], drop_first=True)
    return df

# -------------------
# 3. Maintenance Cox regression
# -------------------
def run_cox_regression(features, clin):
    df = features.join(clin[['PFS','recur','BRCAmt','line','drug']])
    df = preprocess_covariates(df)
    df = df.dropna()

    df = df.rename(columns={'PFS':'time','recur':'event'})
    
    cph = CoxPHFitter(penalizer=0.1)  # penalizer to handle many correlated features
    cph.fit(df, duration_col='time', event_col='event')
    return cph

# -------------------
# 4. Salvage Logistic regression
# -------------------
def run_logistic_regression(features, clin):
    df = features.join(clin[['response']])
    df = df.dropna()
    
    y = df['response']
    X = df.drop(columns=['response'])
    X = sm.add_constant(X)
    
    # use regularized logistic regression if needed (many features vs few samples)
    model = sm.Logit(y, X).fit()
    return model


from sklearn.preprocessing import StandardScaler

def preprocess_features(features, var_thresh=1e-6, corr_thresh=0.95, scale=True):
    """
    Transcript-level feature matrix 전처리:
    1) variance filter
    2) correlation filter
    3) scaling (optional)
    """
    df = features.copy()
    
    # 1. Variance filter (remove near-constant features)
    df = df.loc[:, df.var() > var_thresh]
    
    # 2. Correlation filter (remove highly correlated features)
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > corr_thresh)]
    df = df.drop(columns=to_drop)
    
    # 3. Scaling (z-score)
    if scale:
        scaler = StandardScaler()
        df[:] = scaler.fit_transform(df)
    
    return df

from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler

def run_group_cox(features, clin, group_prefix, penalizer=0.1):
    # 해당 그룹 transcript만 추출
    group_cols = [c for c in features.columns if c.startswith(group_prefix + "_")]
    df = features[group_cols].join(clin[['PFS','recur','BRCAmt','line','drug']])
    
    # covariate 전처리
    df = preprocess_covariates(df)
    df = df.dropna()
    df = df.rename(columns={'PFS':'time','recur':'event'})
    
    # scaling (optional: transcript-level 값 안정화)
    scaler = StandardScaler()
    df[group_cols] = scaler.fit_transform(df[group_cols])
    
    # Cox model
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df, duration_col='time', event_col='event')
    
    return cph

# -------------------
# 그룹별 실행
# -------------------
cox_results = {}

for group in AR_txlist.keys():  # HR_restoration, Cell_cycle_checkpoint_G2M, Pro_survival_signaling
    cph = run_group_cox(maint_features, final_clin.loc[maint_idx], group_prefix=group, penalizer=0.1)
    cox_results[group] = cph
    print(f"\n=== {group} Cox summary ===")
    print(cph.summary[['coef','exp(coef)','p']].head(10))


# -------------------
# Example usage
# -------------------
features_t = build_features_transcript_level(major_tu_df, final_clin, AR_txlist)

# maintenance cohort
maint_idx = final_clin.query("setting == 'maintenance'").index
maint_features = preprocess_features(features_t.loc[maint_idx])

# salvage cohort
salv_idx = final_clin.query("setting == 'salvage'").index
salv_features = preprocess_features(features_t.loc[salv_idx])

cox_model = run_cox_regression(maint_features, final_clin.loc[maint_idx])
#logit_model = run_logistic_regression(salv_features, final_clin.loc[salv_idx])

print(cox_model.summary)
#print(logit_model.summary())

# -------------------
# 5. Visualization
# -------------------
# Cox forest plot
summary = cox_model.summary
coef = summary['coef']
ci_low = summary['coef lower 95%']
ci_high = summary['coef upper 95%']

plt.figure(figsize=(12,6))
plt.errorbar(coef.index, coef, 
             yerr=[coef - ci_low, ci_high - coef], 
             fmt='o', capsize=4)
plt.axhline(0, color='grey', linestyle='--')
plt.xticks(rotation=90)
plt.ylabel("Cox regression coefficient (log HR)")
plt.title("Forest plot of Cox model (transcript-level)")
plt.show()

# # Logistic forest plot
# params = logit_model.params
# conf = logit_model.conf_int()
# odds = np.exp(params)
# odds_ci = np.exp(conf)

# plt.figure(figsize=(6,8))
# plt.errorbar(params.index, odds, 
#              yerr=[odds - odds_ci[0], odds_ci[1] - odds], 
#              fmt='o', capsize=4)
# plt.axhline(1, color='grey', linestyle='--')
# plt.xticks(rotation=90)
# plt.ylabel("Odds ratio")
# plt.title("Forest plot of Logistic regression (transcript-level)")
# plt.show()

#%%

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

def run_group_rsf_transcripts(major_tu_df, clin, AR_txlist, maint_idx, random_state=77):
    results = {}

    for group, transcripts in AR_txlist.items():
        print(f"\n=== {group} ===")

        # 교집합 (major_tu_df 에 있는 transcript만 사용)
        group_cols = [t for t in transcripts if t in major_tu_df.index]
        if not group_cols:
            print("No transcripts for this group in TU matrix, skipping.")
            continue

        # transcript-level feature (sample x transcript)
        df = major_tu_df.loc[group_cols].T
        df = df.join(clin[['PFS','recur']])
        df = df.dropna().rename(columns={'PFS':'time','recur':'event'})

        # scaling transcript values
        scaler = StandardScaler()
        transcript_cols = [c for c in df.columns if c not in ["time","event"]]
        df[transcript_cols] = scaler.fit_transform(df[transcript_cols])

        X = df[transcript_cols]
        y = Surv.from_dataframe("event", "time", df)

        # stratified split by event
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=random_state, stratify=df["event"]
        )

        # RSF
        rsf = RandomSurvivalForest(
            n_estimators=500,
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=random_state
        )
        rsf.fit(X_train, y_train)

        # c-index
        train_pred = rsf.predict(X_train)
        test_pred = rsf.predict(X_test)

        c_index_train = concordance_index_censored(
            y_train["event"], y_train["time"], train_pred
        )[0]
        c_index_test = concordance_index_censored(
            y_test["event"], y_test["time"], test_pred
        )[0]

        print(f"RSF c-index (train): {c_index_train:.3f}")
        print(f"RSF c-index (test) : {c_index_test:.3f}")

        results[group] = {
            "rsf": rsf,
            "c_index_train": c_index_train,
            "c_index_test": c_index_test
        }

    return results

rsf_results = run_group_rsf_transcripts(major_tu_df, final_clin.loc[maint_idx], AR_txlist, maint_idx)

# %%
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def l1_feature_selection(X, y, cv=5, random_state=42):
    """
    L1-penalized logistic regression으로 중요한 feature 선택
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegressionCV(
            Cs=10, cv=cv, penalty="l1", solver="saga",
            random_state=random_state, max_iter=5000
        ))
    ])
    pipe.fit(X, y)
    coef = pipe.named_steps["clf"].coef_[0]
    selected = X.columns[coef != 0]
    return selected


def filter_features(X, var_thresh=1e-6, corr_thresh=0.95):
    # variance filter
    X = X.loc[:, X.var() > var_thresh]
    # correlation filter
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > corr_thresh)]
    X = X.drop(columns=to_drop)
    return X


from sklearn.model_selection import KFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
import numpy as np

def rsf_crossval(X, y, n_splits=5, random_state=42):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    c_indices = []
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        rsf = RandomSurvivalForest(
            n_estimators=500, min_samples_split=10, min_samples_leaf=5,
            n_jobs=-1, random_state=random_state
        )
        rsf.fit(X_train, y_train)
        preds = rsf.predict(X_test)
        c_index = concordance_index_censored(y_test["event"], y_test["time"], preds)[0]
        c_indices.append(c_index)
    return np.mean(c_indices), np.std(c_indices)

def run_group_rsf_pipeline(major_tu_df, clin, AR_txlist, maint_idx, n_splits=5):
    results = {}
    for group, transcripts in AR_txlist.items():
        print(f"\n=== {group} ===")

        group_cols = [t for t in transcripts if t in major_tu_df.index]
        df = major_tu_df.loc[group_cols].T.join(clin[['PFS','recur']])
        df = df.dropna().rename(columns={'PFS':'time','recur':'event'})

        X_raw = df.drop(columns=['time','event'])
        y = Surv.from_dataframe("event", "time", df)

        # 1. feature filtering
        X_filt = filter_features(X_raw)

        # 2. L1 selection 시도
        try:
            selected = l1_feature_selection(X_filt, df["event"])
            X_sel = X_filt[selected]
            if X_sel.shape[1] == 0:
                print("No features selected, fallback to filtered features.")
                X_sel = X_filt
        except Exception as e:
            print("L1 selection failed, fallback to filtered features.")
            X_sel = X_filt

        # 3. 최종 feature 비었으면 skip
        if X_sel.shape[1] == 0:
            print("No features left after filtering, skipping.")
            continue

        # 4. cross-validation RSF
        mean_c, std_c = rsf_crossval(X_sel, y, n_splits=n_splits)
        print(f"RSF {n_splits}-fold CV c-index: {mean_c:.3f} ± {std_c:.3f}")

        results[group] = {
            "features_used": X_sel.columns.tolist(),
            "cv_cindex_mean": mean_c,
            "cv_cindex_std": std_c
        }
    return results

from sklearn.decomposition import PCA

def build_group_scores(major_tu_df, clin, AR_txlist, method="mean"):
    """
    각 그룹별 transcript들을 집약해서 group-level score 생성
    method: "mean" or "pca"
    """
    group_scores = pd.DataFrame(index=clin.index)

    for group, transcripts in AR_txlist.items():
        group_cols = [t for t in transcripts if t in major_tu_df.index]
        if not group_cols:
            print(f"=== {group}: no transcripts found ===")
            continue

        print(f"\n=== {group} ({len(group_cols)} transcripts) ===")
        print(group_cols)

        df_group = major_tu_df.loc[group_cols].T  # sample x transcripts
        df_group = df_group.loc[clin.index]

        if method == "mean":
            score = df_group.mean(axis=1)
        elif method == "pca":
            pca = PCA(n_components=1, random_state=42)
            score = pca.fit_transform(df_group).flatten()
        else:
            raise ValueError("method must be 'mean' or 'pca'")

        group_scores[group] = score

    return group_scores


# -------------------
# RSF CV 평가 (그룹 score 기반)
# -------------------
def run_rsf_cv_groups(major_tu_df, clin, AR_txlist, maint_idx, method="mean", n_splits=5):
    # group score 만들기
    X = build_group_scores(major_tu_df, clin.loc[maint_idx], AR_txlist, method=method)
    df = X.join(clin.loc[maint_idx][['PFS','recur']]).dropna()
    df = df.rename(columns={'PFS':'time','recur':'event'})

    y = Surv.from_dataframe("event","time",df)
    X = df.drop(columns=['time','event'])

    mean_c, std_c = rsf_crossval(X, y, n_splits=n_splits)
    print(f"[{method}] RSF {n_splits}-fold CV c-index: {mean_c:.3f} ± {std_c:.3f}")
    return mean_c, std_c

print("=== Group score (mean) ===")
run_rsf_cv_groups(major_tu_df, final_clin, AR_txlist, maint_idx, method="mean", n_splits=5)

print("\n=== Group score (PCA) ===")
run_rsf_cv_groups(major_tu_df, final_clin, AR_txlist, maint_idx, method="pca", n_splits=5)


results = run_group_rsf_pipeline(major_tu_df, final_clin.loc[maint_idx], AR_txlist, maint_idx, n_splits=5)

# %%
