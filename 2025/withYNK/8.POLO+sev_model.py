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

###^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^################
clin_df = clin_df[clin_df['setting']=='maintenance']
############^^^^##################################

tpm_df_sev = tpm_df[clin_df.index]
tu_df_sev = tu_df[clin_df.index]
gene_df_sev = gene_df[clin_df.index]
clin_df_sev = clin_df[['response','BRCAmt','drug','recur','PFS','firstline']]
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
clin_df_snu = clin_df[['response','BRCAmt','drug','recur','PFS']]
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
dnarepairgenes = dnarepairgenes[40:50]
    
final_gene_exp = final_gene_exp.loc[final_gene_exp.index.intersection(dnarepairgenes), :]
final_trans_exp = final_trans_exp.loc[final_trans_exp.index.str.contains('|'.join(dnarepairgenes)), :]
final_trans_usage = final_trans_usage.loc[final_trans_usage.index.str.contains('|'.join(dnarepairgenes)), :]

# final_gene_exp = gene_df_snu.loc[gene_df_snu.index.intersection(dnarepairgenes), :]
# final_trans_exp = tpm_df_snu.loc[tpm_df_snu.index.str.contains('|'.join(dnarepairgenes)), :]
# final_trans_usage = tu_df_snu.loc[tu_df_snu.index.str.contains('|'.join(dnarepairgenes)), :]
# final_clin = final_clin.loc[final_clin.index.intersection(clin_df_snu.index), :]


# %% ##^^ pipeline #################################################


import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.utils import resample
from scipy.stats import wilcoxon
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def zscore_by_cohort(df, cohort: pd.Series):
    out = []
    for coh, idx in cohort.groupby(cohort).groups.items():
        sub = df.loc[:, idx]
        out.append((sub - sub.mean(axis=1).values[:,None]) / (sub.std(axis=1).values[:,None] + 1e-8))
    return pd.concat(out, axis=1)[df.columns]

final_gene_exp_z = zscore_by_cohort(final_gene_exp, final_clin["cohort"])


# ==== CONFIG ====
TIME_COL = "PFS"
EVENT_COL = "recur"
COVS_ALWAYS = [] #["cohort", "BRCAmt", "drug", "firstline"]
N_BOOT = 100
STRATA = ["cohort"]  # cohort는 strata로만 쓰고 공변량에서는 제외

# =====================================================================
# 1. Utility (safe ops)
# =====================================================================
def safe_entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if len(p) == 0:
        return 0.0
    return -np.sum(p * np.log(p))

def safe_div(num, den):
    num, den = np.asarray(num), np.asarray(den)
    out = np.full_like(num, np.nan, dtype=float)
    mask = np.isfinite(num) & np.isfinite(den) & (den > 0)
    out[mask] = num[mask] / den[mask]
    return out

def safe_log_ratio(a, b):
    a, b = np.asarray(a), np.asarray(b)
    out = np.full_like(a, np.nan, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    out[mask] = np.log2(a[mask] / b[mask])
    return out

def clr_rowsafe(mat, eps=1e-6):
    X = np.asarray(mat, float) + eps
    gm = np.exp(np.mean(np.log(X), axis=0, keepdims=True))
    return np.log(X / gm)

def tu_clr_by_gene(final_trans_usage, trans_ids_for_gene):
    sub = final_trans_usage.loc[trans_ids_for_gene].fillna(0.0)
    return pd.DataFrame(clr_rowsafe(sub.values), index=sub.index, columns=sub.columns)

def zscore_TU_by_cohort(clr_df, cohort):
    return zscore_by_cohort(clr_df, cohort)


# =====================================================================
# 2. Index normalization
# =====================================================================
# transcript index: 'ENST...-GENE' -> transcript id part only
_trans_ids = final_trans_usage.index.str.split("-", n=1).str[0]
_transid_to_full = dict(zip(_trans_ids, final_trans_usage.index))

# gene symbols: strip version suffix if any
final_gene_exp.index = final_gene_exp.index.str.replace(r"\.\d+$", "", regex=True)
majorminor["genename"] = majorminor["genename"].astype(str).str.replace(r"\.\d+$", "", regex=True)

# =====================================================================
# 3. Isoform features via majorminor (match by transcript id before '-')
# =====================================================================
from tqdm import tqdm

def compute_isoform_features(final_trans_usage, trans_exp, majorminor):
    mm = majorminor.copy()
    mm["transcriptid_clean"] = mm["transcriptid"].astype(str).str.split("-", n=1).str[0]

    # transcriptid_clean → full index
    _trans_ids = final_trans_usage.index.str.split("-", n=1).str[0]
    _transid_to_full = dict(zip(_trans_ids, final_trans_usage.index))

    gene_feat = {}
    for gene, sub in mm.groupby("genename"):
        # --- minor isoform만 선택 ---
        minors = [
            t for t in sub.loc[sub["type"] == "minor", "transcriptid_clean"]
            if t in _transid_to_full
        ]
        if len(minors) < 2:
            continue  # PCA 최소 2개 필요

        minors_full = [_transid_to_full[t] for t in minors]

        # --- (1) TU matrix for minors ---
        tu = final_trans_usage.loc[minors_full].fillna(0.0).astype(float)
        clr = pd.DataFrame(clr_rowsafe(tu.values), index=tu.index, columns=tu.columns)

        # --- (2) PCA (PC1, PC2) ---
        k = min(2, clr.shape[0], clr.shape[1])
        pca = PCA(n_components=k)
        pcs = pca.fit_transform(clr.T)  # samples x k
        pc_df = pd.DataFrame(
            pcs,
            index=clr.columns,
            columns=[f"minor_pc{i+1}" for i in range(k)]
        )

        # --- (3) Expression-based top2 minors ---
        exp_sub = trans_exp.loc[minors_full].fillna(0.0).astype(float)
        top2 = exp_sub.mean(axis=1).sort_values(ascending=False).head(2).index.tolist()
        top2_exp = exp_sub.loc[top2].T  # samples x 2
        top2_exp.columns = [f"minor_exp{i+1}" for i in range(len(top2))]

        # --- (4) Merge all features ---
        #feat = pc_df.join(top2_exp, how="left")
        feat = pc_df
        gene_feat[gene] = feat

    print(f"✅ generated minor-only features for {len(gene_feat)} genes")
    return gene_feat


isoform_feats = compute_isoform_features(final_trans_usage, final_trans_exp, majorminor)

# =====================================================================
# 4. Bootstrap per-gene Cox (robust to collinearity)
# =====================================================================
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from lifelines.utils import concordance_index
from sklearn.utils import resample
import pandas.api.types as pdt

def _drop_constant_cols(df):
    keep = df.columns[df.nunique(dropna=True) > 1]
    return df[keep]

def _make_dummies(train_df, test_df):
    # cohort는 strata로 사용할 것이므로 더미 변환하지 않음
    to_dummy = []
    for col in ["drug", "BRCAmt", "firstline"]:
        if col in train_df.columns and (pdt.is_string_dtype(train_df[col]) or pdt.is_object_dtype(train_df[col])):
            to_dummy.append(col)
    if to_dummy:
        train_df = pd.get_dummies(train_df, columns=to_dummy, drop_first=True)
        test_df  = pd.get_dummies(test_df,  columns=to_dummy, drop_first=True)
        test_df = test_df.reindex(columns=train_df.columns, fill_value=0)
    return train_df, test_df

def _safe_fit(cph, df, cols):
    try:
        tmp = df[cols + [TIME_COL, EVENT_COL]].copy()
        tmp = _drop_constant_cols(tmp)
        tmp = tmp.fillna(0)  # ✅ 모든 NaN → 0
        cols2 = [c for c in cols if c in tmp.columns]
        if len(cols2) == 0:
            return None
        cph.fit(
            tmp[cols2 + [TIME_COL, EVENT_COL]],
            duration_col=TIME_COL,
            event_col=EVENT_COL,
            strata=STRATA if all(s in tmp.columns for s in STRATA) else None
        )
        return cph, cols2
    except (ConvergenceError, np.linalg.LinAlgError, ValueError, FloatingPointError, TypeError):
        return None


# =====================================================================
# 4. Bootstrap per-gene Cox (robust to collinearity)
# =====================================================================
# (기존 _drop_constant_cols, _make_dummies, _safe_fit 함수는 그대로 둡니다)

def bootstrap_per_gene(final_gene_exp, isoform_feats, final_clin, B=N_BOOT, random_state=42):
    np.random.seed(random_state)
    samples = final_clin.index
    results = []

    for gene in tqdm(final_gene_exp.index, desc="Bootstrap per gene"):
        if gene not in isoform_feats:
            continue
        gene_vec = final_gene_exp.loc[gene, samples]
        iso_df = isoform_feats[gene].loc[samples]

        for b in range(B):
            boot_train = resample(samples, replace=True, stratify=final_clin["cohort"])
            boot_test = [s for s in samples if s not in boot_train]

            train_df = final_clin.loc[boot_train].copy()
            test_df  = final_clin.loc[boot_test].copy()

            # 더미 인코딩 (cohort 제외)
            train_df, test_df = _make_dummies(train_df, test_df)

            # strata로 쓰는 컬럼은 공변량에서 제외
            covs = [c for c in train_df.columns
                    if c not in ([TIME_COL, EVENT_COL, "gene"] + STRATA)]

            #--- M0 ---
            cph0 = CoxPHFitter(penalizer=0.1)
            fit0 = _safe_fit(cph0, train_df, covs)
            if fit0 is None:
                continue
            cph0, covs0 = fit0
            t0 = test_df.reindex(columns=train_df.columns, fill_value=0)

            # 🚨 [FIX-1] C0 계산 수정
            pred_scores_0 = -cph0.predict_partial_hazard(t0.fillna(0)) # 예측 시 0으로 채움
            event_times_0 = t0[TIME_COL]
            event_obs_0 = t0[EVENT_COL]
            
            valid_idx_0 = pd.Series(event_times_0).notna() & \
                          pd.Series(event_obs_0).notna() & \
                          pd.Series(pred_scores_0).notna()

            C0 = np.nan
            if valid_idx_0.sum() > 1: # C-index는 최소 2개 샘플 필요
                try:
                    C0 = concordance_index(
                        event_times_0[valid_idx_0],
                        pred_scores_0[valid_idx_0],
                        event_obs_0[valid_idx_0]
                    )
                except: pass # 예외 발생 시 C0 = np.nan 유지


            # --- M1 (gene) ---
            train_df1 = train_df.copy()
            test_df1  = test_df.copy()
            train_df1["gene"] = gene_vec.loc[boot_train].astype(float).values
            test_df1["gene"]  = gene_vec.loc[boot_test].astype(float).values
            covs1 = covs + ["gene"]
            cph1 = CoxPHFitter(penalizer=0.1)
            fit1 = _safe_fit(cph1, train_df1, covs1)
            if fit1 is None:
                continue
            cph1, covs1 = fit1
            t1 = test_df1.reindex(columns=train_df1.columns, fill_value=0)

            # 🚨 [FIX-2] C1 계산 수정
            pred_scores_1 = -cph1.predict_partial_hazard(t1.fillna(0)) # 예측 시 0으로 채움
            event_times_1 = t1[TIME_COL]
            event_obs_1 = t1[EVENT_COL]

            valid_idx_1 = pd.Series(event_times_1).notna() & \
                          pd.Series(event_obs_1).notna() & \
                          pd.Series(pred_scores_1).notna()

            C1 = np.nan
            if valid_idx_1.sum() > 1:
                try:
                    C1 = concordance_index(
                        event_times_1[valid_idx_1],
                        pred_scores_1[valid_idx_1],
                        event_obs_1[valid_idx_1]
                    )
                except: pass

            # --- M2 (isoform features) ---
            train_df2 = train_df1.copy()
            test_df2  = test_df1.copy()
            for col in iso_df.columns:
                train_df2[col] = iso_df.loc[boot_train, col].values
                test_df2[col]  = iso_df.loc[boot_test, col].values
            
            train_df2 = _drop_constant_cols(train_df2)
            test_df2  = test_df2.reindex(columns=train_df2.columns, fill_value=0)

            covs2 = [c for c in covs1 + list(iso_df.columns) if c in train_df2.columns]
            cph2 = CoxPHFitter(penalizer=0.1)
            fit2 = _safe_fit(cph2, train_df2, covs2)
            if fit2 is None:
                continue
            cph2, covs2 = fit2

            # 🚨 [FIX-3] C2 계산 수정 (오류 발생 지점)
            pred_scores_2 = -cph2.predict_partial_hazard(test_df2.fillna(0)) # 예측 시 0으로 채움
            event_times_2 = test_df2[TIME_COL]
            event_obs_2 = test_df2[EVENT_COL]

            valid_idx_2 = pd.Series(event_times_2).notna() & \
                          pd.Series(event_obs_2).notna() & \
                          pd.Series(pred_scores_2).notna()

            C2 = np.nan
            if valid_idx_2.sum() > 1:
                try:
                    C2 = concordance_index(
                        event_times_2[valid_idx_2],
                        pred_scores_2[valid_idx_2],
                        event_obs_2[valid_idx_2]
                    )
                except: pass

            results.append({
                "gene": gene, "bootstrap": b,
                "C0": C0, "C1": C1, "C2": C2, #"C0": C0, 
                "ΔC1": C1 - C0, "ΔC2": C2 - C0
            })

    return pd.DataFrame(results)

res_df = bootstrap_per_gene(final_gene_exp_z, isoform_feats, final_clin, B=N_BOOT)

# =====================================================================
# 5. Summarize (ΔC, p, FDR)
# =====================================================================
from scipy.stats import wilcoxon

def summarize_bootstrap(res_df):
    if res_df.empty or "gene" not in res_df.columns:
        raise ValueError("res_df empty or missing 'gene'. Check matching & features.")
    summary = (
        res_df.groupby("gene")
        .agg({"ΔC1": "median", "ΔC2": "median"}) # median은 NaN 무시(default)
        .sort_values("ΔC2", ascending=False)
    )
    pvals = []
    for g, sub in res_df.groupby("gene"):
        
        # 🚨 [FIX-4] Wilcoxon 테스트 전 NaN 처리
        # C-index 계산에서 NaN이 발생했을 수 있으므로,
        # 짝(pair)이 맞는 ΔC1, ΔC2 값만 추려서 테스트합니다.
        paired_data = sub[["ΔC1", "ΔC2"]].dropna()

        if len(paired_data) < 5: # NaN 제거 후 5개 미만이면 스킵
            continue
            
        try:
            # .values를 사용하거나 컬럼을 명시
            _, p = wilcoxon(paired_data["ΔC2"], paired_data["ΔC1"], alternative="greater")
        except ValueError:
            p = 1.0
        pvals.append((g, p))
    
    pvals_df = pd.DataFrame(pvals, columns=["gene", "p"]).sort_values("p")
    m = len(pvals_df)
    if m > 0:
        pvals_df["FDR"] = pvals_df["p"] * m / (np.arange(1, m + 1))
        pvals_df["FDR"] = np.minimum.accumulate(pvals_df["FDR"][::-1])[::-1]
        summary = summary.join(pvals_df.set_index("gene"), how="left")
    return summary

summary_df = summarize_bootstrap(res_df)
summary_df.sort_values("ΔC2", ascending=False).head(10)


gene_perf = (
    res_df.groupby("gene")[["C0", "C1", "C2"]] #"C0", 
    .agg(["median", "mean", "std"])
)
gene_perf.head(10)

# %%
###################^ pipeline 2 ##########################################

dnarepairgenes = ddr_coregenelist['Homologous Recombination (HR)']

final_gene_exp = pd.merge(gene_df_sev, gene_df_snu, left_index=True, right_index=True, how='inner')
final_trans_exp = pd.merge(tpm_df_sev, tpm_df_snu, left_index=True, right_index=True, how='inner')
final_trans_usage = pd.merge(tu_df_sev, tu_df_snu, left_index=True, right_index=True, how='inner')
final_gene_exp = final_gene_exp.loc[final_gene_exp.index.intersection(dnarepairgenes), :]
final_trans_exp = final_trans_exp.loc[final_trans_exp.index.str.contains('|'.join(dnarepairgenes)), :]
final_trans_usage = final_trans_usage.loc[final_trans_usage.index.str.contains('|'.join(dnarepairgenes)), :]

# =========================
# Residual stacking + IPCW
# =========================
import numpy as np
import pandas as pd
import pandas.api.types as pdt
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.cross_decomposition import PLSRegression
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv as skSurv
import warnings
warnings.filterwarnings("ignore")

def zscore_by_cohort(df, cohort: pd.Series):
    out = []
    for coh, idx in cohort.groupby(cohort).groups.items():
        sub = df.loc[:, idx]
        out.append((sub - sub.mean(axis=1).values[:,None]) / (sub.std(axis=1).values[:,None] + 1e-8))
    return pd.concat(out, axis=1)[df.columns]
final_gene_exp_z = zscore_by_cohort(final_gene_exp, final_clin["cohort"])

# -------- config --------
PENALIZER = 0.1
PLS_COMPONENTS = 2        # number of supervised TU components
USE_GENE_ZSCORE = False   # set True if you already have cohort-wise zscores
GENE_MATRIX = final_gene_exp_z  # or final_gene_exp_z if prepared
N_BOOT = 50
# ==== CONFIG ====
TIME_COL = "PFS"
EVENT_COL = "recur"
COVS_ALWAYS = [] #["cohort", "BRCAmt", "drug", "firstline"]
STRATA = ["cohort"]

# ---------- helpers ----------
def _drop_constant_cols(df):
    keep = df.columns[df.nunique(dropna=True) > 1]
    return df[keep]

def _make_dummies(train_df, test_df):
    # cohort is used as strata only (NOT dummied)
    to_dummy = []
    for col in ["drug", "BRCAmt", "firstline"]:
        if col in train_df.columns and (pdt.is_string_dtype(train_df[col]) or pdt.is_object_dtype(train_df[col])):
            to_dummy.append(col)
    if to_dummy:
        train_df = pd.get_dummies(train_df, columns=to_dummy, drop_first=True)
        test_df  = pd.get_dummies(test_df,  columns=to_dummy, drop_first=True)
        test_df = test_df.reindex(columns=train_df.columns, fill_value=0)
    return train_df, test_df

def _finite_scores(x, cap=50.0):
    x = np.asarray(x, float)
    x = np.where(np.isfinite(x), x, 0.0)
    return np.clip(x, -cap, cap)


def _safe_fit_cox(cph, df, covs):
    try:
        tmp = df[covs + [TIME_COL, EVENT_COL]].copy()
        tmp = _drop_constant_cols(tmp).fillna(0)
        covs2 = [c for c in covs if c in tmp.columns]
        if not covs2:
            return None
        cph.fit(
            tmp[covs2 + [TIME_COL, EVENT_COL]],
            duration_col=TIME_COL,
            event_col=EVENT_COL,
            strata=STRATA if all(s in tmp.columns for s in STRATA) else None
        )
        return cph, covs2
    except (ConvergenceError, np.linalg.LinAlgError, ValueError, FloatingPointError, TypeError):
        return None

def _clr_rowsafe(mat, eps=1e-6):
    X = np.asarray(mat, float) + eps
    gm = np.exp(np.mean(np.log(X), axis=0, keepdims=True))
    return np.log(X / gm)

# map clean transcript id -> full index used in final_trans_usage
_trans_ids_clean = final_trans_usage.index.str.split("-", n=1).str[0]
_transid_to_full = dict(zip(_trans_ids_clean, final_trans_usage.index))

def _minor_clr_matrix_for_gene(gene):
    """Return (X: samples x n_minor, list_of_trans) for THIS gene, minor-only CLR TU.
       If < 2 minors exist, returns (None, None)."""
    sub = majorminor[majorminor["genename"] == gene]
    minors = [
        t for t in sub.loc[sub["type"] == "minor", "transcriptid"].astype(str).str.split("-", n=1).str[0]
        if t in _transid_to_full
    ]
    if len(minors) < 2:
        return None, None
    minors_full = [_transid_to_full[t] for t in minors]
    tu = final_trans_usage.loc[minors_full].fillna(0.0).astype(float)        # iso x samples
    clr = pd.DataFrame(_clr_rowsafe(tu.values), index=tu.index, columns=tu.columns)
    # return samples x iso
    return clr.T, minors_full

def _to_sksurv_y(df):
    # sksurv structured array (event: bool, time: float)
    e = df[EVENT_COL].astype(bool).values
    t = df[TIME_COL].astype(float).values
    return skSurv.from_arrays(e, t)

def _ipcw_cindex(y_train, y_test, risk_scores):
    """Return integrated IPCW concordance index (float)."""
    out = concordance_index_ipcw(y_train, y_test, estimate=risk_scores)
    # Compatible with both old (tuple) and new (namedtuple/object) API
    if isinstance(out, tuple) or isinstance(out, list):
        return float(out[0])
    if hasattr(out, "concordance_index_"):
        return float(out.concordance_index_)
    if hasattr(out, "estimate"):
        return float(out.estimate)
    # Fallback: try first element
    try:
        return float(out[0])
    except Exception:
        raise ValueError(f"Unexpected return from concordance_index_ipcw: {type(out)}")


# ---------- main ----------
def bootstrap_with_plsU_IPCW(final_gene_exp, final_clin, B=N_BOOT, random_state=42):
    np.random.seed(random_state)
    samples = final_clin.index
    results = []

    for gene in tqdm(final_gene_exp.index, desc="Bootstrap per gene (PLS-U + IPCW)"):
        # build minor-CLR matrix for this gene (samples x isoforms)
        X_clr, minor_list = _minor_clr_matrix_for_gene(gene)
        if X_clr is None:
            continue

        # gene expression vector
        gvec = (GENE_MATRIX if not USE_GENE_ZSCORE else GENE_MATRIX).loc[gene, samples].astype(float)

        for b in range(B):
            # stratified bootstrap on cohort
            boot_train = resample(samples, replace=True, stratify=final_clin["cohort"])
            boot_test  = [s for s in samples if s not in boot_train]

            train_df = final_clin.loc[boot_train].copy()
            test_df  = final_clin.loc[boot_test].copy()
            train_df, test_df = _make_dummies(train_df, test_df)

            # ----- M0: clinical only -----
            covs0 = [c for c in train_df.columns if c not in [TIME_COL, EVENT_COL] + STRATA]
            cph0  = CoxPHFitter(penalizer=PENALIZER)
            fit0  = _safe_fit_cox(cph0, train_df, covs0)
            if fit0 is None:
                continue
            cph0, covs0 = fit0

            # align columns
            t0 = test_df.reindex(columns=train_df.columns, fill_value=0)

            # predictions for IPCW (risk scores: larger = riskier)
            # lifelines predicts partial hazard (>=0); use +partial_hazard as risk
            risk0 = cph0.predict_log_partial_hazard(t0).values.ravel()
            risk0 = _finite_scores(risk0)
            # y for IPCW
            y_tr = _to_sksurv_y(train_df)
            y_te = _to_sksurv_y(t0)

            C0_ipcw = _ipcw_cindex(y_tr, y_te, risk0)

            # minor-CLR design: samples x isoforms (already computed earlier)
            Xtr = X_clr.loc[boot_train].copy()
            Xte = X_clr.loc[boot_test].copy()

            # standardize columns using TRAIN only (robust to outliers: median/IQR)
            def robust_scale(df, eps=1e-9):
                med = df.median(axis=0)
                iqr = (df.quantile(0.75, axis=0) - df.quantile(0.25, axis=0)) + eps
                return (df - med) / iqr, med, iqr

            Xtr_s, med, iqr = robust_scale(Xtr)
            Xte_s = (Xte - med) / iqr

            # build Cox design frames with TU-only features
            # name features to avoid collision
            tu_cols = [f"TU_{j}" for j in range(Xtr_s.shape[1])]
            train_u = train_df.copy()
            test_u  = t0.copy()
            train_u = pd.concat([train_u, pd.DataFrame(Xtr_s.values, index=Xtr_s.index, columns=tu_cols)], axis=1)
            test_u  = pd.concat([test_u,  pd.DataFrame(Xte_s.values, index=Xte_s.index, columns=tu_cols)], axis=1)

            # quick inner split on TRAIN to choose penalizer (no leakage to outer test)
            rng = np.random.RandomState(123)
            mask = rng.rand(len(boot_train)) < 0.8
            inner_tr_ids = np.array(boot_train)[mask]
            inner_va_ids = np.array(boot_train)[~mask]

            pen_grid = [0.02, 0.05, 0.1, 0.2, 0.5]  # try a few; you can tweak
            best_pen, best_c = None, -np.inf
            for pen in pen_grid:
                cph_try = CoxPHFitter(penalizer=pen, l1_ratio=0.5)  # elastic-net (50% L1)
                covs_try = tu_cols  # TU-only
                fit_try = _safe_fit_cox(cph_try, train_u.loc[inner_tr_ids], covs_try)
                if fit_try is None:
                    continue
                cph_try, covs_ok = fit_try
                # validate on inner_va
                va = train_u.loc[inner_va_ids].reindex(columns=train_u.columns, fill_value=0)
                risk_va = cph_try.predict_partial_hazard(va).values.ravel()
                c_va = _ipcw_cindex(_to_sksurv_y(train_u.loc[inner_tr_ids]),
                                    _to_sksurv_y(va), risk_va)
                if c_va > best_c:
                    best_c, best_pen = c_va, pen

            # fallback if none fit
            if best_pen is None:
                continue

            # fit final TU-only Cox on full TRAIN with best penalizer
            cph_tu = CoxPHFitter(penalizer=best_pen, l1_ratio=0.5)
            fit_tu = _safe_fit_cox(cph_tu, train_u, tu_cols)
            if fit_tu is None:
                continue
            cph_tu, covs_tu = fit_tu

            # supervised TU signature = linear predictor (iso_lp)
            train_u["iso_lp"] = cph_tu.predict_log_partial_hazard(train_u).values.ravel()
            test_u["iso_lp"]  = cph_tu.predict_log_partial_hazard(test_u).values.ravel()
            train_u["iso_lp"] = _finite_scores(train_u["iso_lp"].values)
            test_u["iso_lp"]  = _finite_scores(test_u["iso_lp"].values)


            # ---------- M1-U (clinical + iso_lp) ----------
            covs_u = [c for c in covs0 + ["iso_lp"] if c in train_u.columns]
            cph_u  = CoxPHFitter(penalizer=PENALIZER)
            fit_u  = _safe_fit_cox(cph_u, train_u, covs_u)
            if fit_u is None:
                continue
            cph_u, covs_u = fit_u
            risk_u = cph_u.predict_log_partial_hazard(test_u).values.ravel()
            risk_u = _finite_scores(risk_u)

            Cu_ipcw = _ipcw_cindex(_to_sksurv_y(train_u), _to_sksurv_y(test_u), risk_u)

            # ---------- M1-G (clinical + gene) ----------
            train_g = train_df.copy(); test_g = t0.copy()
            train_g["gene"] = GENE_MATRIX.loc[gene, boot_train].values
            test_g["gene"]  = GENE_MATRIX.loc[gene, boot_test].values
            covs_g = [c for c in covs0 + ["gene"] if c in train_g.columns]
            cph_g  = CoxPHFitter(penalizer=PENALIZER)
            fit_g  = _safe_fit_cox(cph_g, train_g, covs_g)
            if fit_g is None:
                continue
            cph_g, covs_g = fit_g
            risk_g = cph_g.predict_log_partial_hazard(test_g).values.ravel()
            risk_g = _finite_scores(risk_g)
            Cg_ipcw = _ipcw_cindex(_to_sksurv_y(train_g), _to_sksurv_y(test_g), risk_g)


            # ---------- M1-G+U (clinical + gene + iso_lp) ----------
            train_gu = train_g.copy(); test_gu = test_g.copy()
            train_gu["iso_lp"] = train_u["iso_lp"]
            test_gu["iso_lp"]  = test_u["iso_lp"]
            covs_gu = [c for c in covs0 + ["gene","iso_lp"] if c in train_gu.columns]
            cph_gu  = CoxPHFitter(penalizer=PENALIZER)
            fit_gu  = _safe_fit_cox(cph_gu, train_gu, covs_gu)
            if fit_gu is None:
                continue
            cph_gu, covs_gu = fit_gu
            risk_gu = cph_gu.predict_log_partial_hazard(test_gu).values.ravel()
            risk_gu = _finite_scores(risk_gu)
            Cgu_ipcw = _ipcw_cindex(_to_sksurv_y(train_gu), _to_sksurv_y(test_gu), risk_gu)


            # record results (C0_ipcw already computed earlier)
            results.append({
                "gene": gene, "bootstrap": b,
                "C0_ipcw": C0_ipcw,
                "CG_ipcw": Cg_ipcw,
                "CU_ipcw": Cu_ipcw,
                "CGU_ipcw": Cgu_ipcw,
                "dG":  Cg_ipcw  - C0_ipcw,
                "dU":  Cu_ipcw  - C0_ipcw,
                "dGU": Cgu_ipcw - C0_ipcw,
                "n_U": len(tu_cols),
                "penU": best_pen,
            })

    return pd.DataFrame(results)

res_ipcw = bootstrap_with_plsU_IPCW(GENE_MATRIX, final_clin, B=N_BOOT)

# -------- summarize (medians + Wilcoxon one-sided) --------
from scipy.stats import wilcoxon

def summarize_ipcw(res_df):
    if res_df.empty:
        raise ValueError("No results; check your inputs.")
    # median C's and deltas
    summary = (res_df.groupby("gene")
               .agg(median_C0=("C0_ipcw", "median"),
                    median_CG=("CG_ipcw", "median"),
                    median_CU=("CU_ipcw", "median"),
                    median_CGU=("CGU_ipcw", "median"),
                    median_dG=("dG", "median"),
                    median_dU=("dU", "median"),
                    median_dGU=("dGU", "median"),
                    mean_nU=("n_U", "mean"))
               .sort_values("median_dU", ascending=False))

    # paired Wilcoxon (greater) for deltas
    pvals = []
    for g, sub in res_df.groupby("gene"):
        sub = sub.dropna(subset=["dG","dU","dGU","C0_ipcw"])
        if len(sub) < 8:
            continue
        # test that each added model > M0
        try:
            pG  = wilcoxon(sub["CG_ipcw"],  sub["C0_ipcw"],  alternative="greater").pvalue
        except ValueError: pG = 1.0
        try:
            pU  = wilcoxon(sub["CU_ipcw"],  sub["C0_ipcw"],  alternative="greater").pvalue
        except ValueError: pU = 1.0
        try:
            pGU = wilcoxon(sub["CGU_ipcw"], sub["C0_ipcw"], alternative="greater").pvalue
        except ValueError: pGU = 1.0
        pvals.append((g, pG, pU, pGU))
    if pvals:
        pdf = pd.DataFrame(pvals, columns=["gene","pG","pU","pGU"]).set_index("gene")
        # BH-FDR per family
        for col in ["pG","pU","pGU"]:
            m = pdf[col].rank(method="first").shape[0]
            order = pdf[col].sort_values().index
            q = (pdf.loc[order, col].values * np.arange(1, m+1) / m)
            q = np.minimum.accumulate(q[::-1])[::-1]
            pdf.loc[order, col.replace("p","q")] = q
        summary = summary.join(pdf, how="left")
    return summary

summary_ipcw = summarize_ipcw(res_ipcw)
summary_ipcw.head(20)

# %%
####^^ gene set ########################

# # dnarepairgenes = {gene for gene_list in ddr_coregenelist.values() for gene in gene_list}
# # dnarepairgenes = list(dnarepairgenes)
# dnarepairgenes = ddr_coregenelist['Homologous Recombination (HR)']

# final_gene_exp = pd.merge(gene_df_sev, gene_df_snu, left_index=True, right_index=True, how='inner')
# final_trans_exp = pd.merge(tpm_df_sev, tpm_df_snu, left_index=True, right_index=True, how='inner')
# final_trans_usage = pd.merge(tu_df_sev, tu_df_snu, left_index=True, right_index=True, how='inner')
# final_gene_exp = final_gene_exp.loc[final_gene_exp.index.intersection(dnarepairgenes), :]
# final_trans_exp = final_trans_exp.loc[final_trans_exp.index.str.contains('|'.join(dnarepairgenes)), :]
# final_trans_usage = final_trans_usage.loc[final_trans_usage.index.str.contains('|'.join(dnarepairgenes)), :]

from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv as skSurv

# --- step 1: iso_lp matrix 만들기 ---
# DNA repair gene 리스트 (이미 subset 되어있다고 가정)
repair_genes = list(final_gene_exp.index)   # 혹은 ['BRCA1','RAD51','MSH2',...]

iso_lp_df = pd.DataFrame(index=final_clin.index, columns=repair_genes, dtype=float)

for gene in repair_genes:
    X_clr, minor_list = _minor_clr_matrix_for_gene(gene)
    if X_clr is None:
        continue
    # 전체 샘플 대상으로 minor-CLR matrix
    # --- robust scaling (median / IQR) ---
    med = X_clr.median()
    iqr = X_clr.quantile(0.75) - X_clr.quantile(0.25)
    X_s = (X_clr - med) / (iqr + 1e-9)
    
    # train iso_lp model (elastic-net cox) on ALL samples for this gene
    tu_cols = [f"{gene}_TU{i}" for i in range(X_s.shape[1])]
    df = pd.concat([final_clin[[TIME_COL, EVENT_COL]], X_s], axis=1)
    df.columns = [TIME_COL, EVENT_COL] + tu_cols
    cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.5)
    try:
        cph.fit(df.fillna(0), duration_col=TIME_COL, event_col=EVENT_COL)
        lp = cph.predict_log_partial_hazard(df.fillna(0)).values.ravel()
        iso_lp_df.loc[df.index, gene] = lp
    except ConvergenceError:
        continue

iso_lp_df = iso_lp_df.fillna(0.0)
print(f"✅ iso_lp matrix built: {iso_lp_df.shape[1]} genes, {iso_lp_df.shape[0]} samples")

def zscore_by_cohort(df, cohort: pd.Series):
    out = []
    for coh, idx in cohort.groupby(cohort).groups.items():
        sub = df.loc[idx]
        z = (sub - sub.mean()) / (sub.std() + 1e-8)
        out.append(z)
    return pd.concat(out).loc[df.index]

iso_lp_z = zscore_by_cohort(iso_lp_df, final_clin["cohort"])
clin_covs = ["drug", "BRCAmt", "firstline"]  # 너가 쓰던 covariates
X = pd.get_dummies(final_clin[clin_covs], drop_first=True)
X = pd.concat([X, iso_lp_z], axis=1).fillna(0.0)

y = skSurv.from_arrays(
    event=final_clin[EVENT_COL].astype(bool).values,
    time=final_clin[TIME_COL].astype(float).values
)
import numpy as np
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.model_selection import KFold

alphas = 10 ** np.linspace(-3, 1, 50)  # penalty strength grid
coxnet = CoxnetSurvivalAnalysis(l1_ratio=0.5, alphas=alphas)

coxnet.fit(X, y)

print(f"✅ fitted Coxnet with {len(coxnet.coef_[0])} total features")
from sksurv.metrics import concordance_index_censored

kf = KFold(n_splits=5, shuffle=True, random_state=42)
c_scores = []

for train_idx, test_idx in kf.split(X):
    coxnet.fit(X.iloc[train_idx], y[train_idx])
    preds = -coxnet.predict(X.iloc[test_idx])
    result = concordance_index_censored(
        y[test_idx]["event"], y[test_idx]["time"], preds
    )
    c = result[0] if isinstance(result, tuple) else result.concordance_index_
    c_scores.append(c)

print(f"Mean 5-fold C-index: {np.mean(c_scores):.3f} ± {np.std(c_scores):.3f}")


print(f"Mean 5-fold C-index: {np.mean(c_scores):.3f} ± {np.std(c_scores):.3f}")
import numpy as np

coef_mat = np.vstack(coxnet.coef_)
mean_coef = np.median(coef_mat, axis=0)
coef_series = pd.Series(coxnet.coef_[-1], index=X.columns)
top_genes = coef_series.abs().sort_values(ascending=False).head(20)
top_genes

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,5))
sns.barplot(y=top_genes.index, x=top_genes.values, orient='h')
plt.title("Top DNA Repair Genes by Elastic-Net Cox Coefficient")
plt.xlabel("Coefficient (median across alphas)")
plt.tight_layout()
plt.show()

# %%
gene_map = pd.Series([x.split("-")[-1] for x in final_trans_exp.index], index=final_trans_exp.index)

# -----------------------------
# 2. Make gene-level matrices
# -----------------------------
# (i) Total TPM (all transcripts)
expr_total = final_trans_exp.groupby(gene_map).sum()

# (ii) Major TPM (filter first → then sum)
expr_major = final_trans_exp.loc[final_trans_exp.index.isin(majorlist)] #*minorlist
gene_map_major = gene_map.loc[expr_major.index]
expr_major_gene = pd.DataFrame(expr_major.groupby(gene_map_major).sum())
expr_major_gene = expr_major_gene.apply(pd.to_numeric, errors='coerce')
expr_major_log = np.log2(expr_major_gene + 0.01)

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score

from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored


# -------------------------------
# 0) Inputs
# -------------------------------
# expr_major_gene : your expression matrix (likely gene x sample)
# final_clin      : your clinical dataframe with columns: 'recur', 'PFS', 'cohort'
expr_df = expr_major_gene.copy()
clin_df = final_clin.copy()

# -------------------------------
# 1) Make sure samples are rows
#    (auto-detect by matching index/columns to clin_df.index)
# -------------------------------
idx_match = len(set(expr_df.index).intersection(clin_df.index))
col_match = len(set(expr_df.columns).intersection(clin_df.index))
if col_match > idx_match:
    expr_df = expr_df.T  # now rows are samples, columns are genes

# -------------------------------
# 2) Align samples
# -------------------------------
common = expr_df.index.intersection(clin_df.index)
expr_df = expr_df.loc[common].sort_index()
clin_df = clin_df.loc[common].sort_index()

# Defensive checks
assert {'recur','PFS','cohort'}.issubset(clin_df.columns), clin_df.columns.tolist()
assert expr_df.shape[0] == clin_df.shape[0]

# -------------------------------
# 3) Survival object
# -------------------------------
y = Surv.from_dataframe(event="recur", time="PFS", data=clin_df)

# -------------------------------
# 4) Z-score genes (per column)
# -------------------------------
X_z = pd.DataFrame(
    StandardScaler().fit_transform(expr_df.values),
    index=expr_df.index,
    columns=expr_df.columns,
)

# -------------------------------
# 5) Add cohort dummies (proxy for strata)
# -------------------------------
cohort_dv = pd.get_dummies(clin_df["cohort"], prefix="cohort", drop_first=True)

X_full = pd.concat([X_full, final_clin['BRCAmt']], axis=1)
# Final sanity check
assert X_full.shape[0] == len(y)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

# assume X_full, y (Surv object) already defined

# ---------------------------------
# Outer CV split (for final evaluation)
# ---------------------------------
outer_cv = KFold(n_splits=5, shuffle=True, random_state=77)
outer_scores = []
outer_best_params = []

# candidate l1_ratios to test
l1_candidates = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]  # 1.0=LASSO, 0.0=Ridge

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_full), 1):
    print(f"\n===== Outer Fold {fold_idx} =====")

    X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    best_cidx, best_alpha, best_l1 = -np.inf, None, None

    # -------------------------------
    # Inner CV: tune both alpha and l1_ratio
    # -------------------------------
    for l1_ratio in l1_candidates:
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=fold_idx)

        coxnet_path = CoxnetSurvivalAnalysis(
            l1_ratio=l1_ratio,
            alpha_min_ratio=0.05,
            n_alphas=100
        )
        coxnet_path.fit(X_train, y_train)

        # For each alpha along the path, compute C-index using inner CV
        mean_cidx_per_alpha = []
        for coef in coxnet_path.coef_.T:
            preds = np.dot(X_train, coef)
            cidx = concordance_index_censored(
                y_train["recur"], y_train["PFS"], preds
            )[0]
            mean_cidx_per_alpha.append(cidx)

        # find best alpha for this l1_ratio
        best_idx = np.argmax(mean_cidx_per_alpha)
        if mean_cidx_per_alpha[best_idx] > best_cidx:
            best_cidx = mean_cidx_per_alpha[best_idx]
            best_alpha = coxnet_path.alphas_[best_idx]
            best_l1 = l1_ratio

    print(f"Best inner params: l1_ratio={best_l1}, alpha={best_alpha:.4e}, inner C={best_cidx:.3f}")
    outer_best_params.append((best_l1, best_alpha))

    # -------------------------------
    # Refit with best params on outer train set
    # -------------------------------
    coxnet_best = CoxnetSurvivalAnalysis(l1_ratio=best_l1, alphas=[best_alpha])
    coxnet_best.fit(X_train, y_train)

    pred_test = coxnet_best.predict(X_test)
    cidx_test = concordance_index_censored(
        y_test["recur"], y_test["PFS"], pred_test
    )[0]

    print(f"Outer fold test C-index: {cidx_test:.3f}")
    outer_scores.append(cidx_test)

# ---------------------------------
# Summary
# ---------------------------------
print("\n==============================")
print(f"Mean outer test C-index: {np.mean(outer_scores):.3f} ± {np.std(outer_scores):.3f}")
print("Best params per fold:")
for i, (l1, a) in enumerate(outer_best_params, 1):
    print(f"  Fold {i}: l1_ratio={l1}, alpha={a:.4e}")


# %%
# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

# --------------------------------
# 0. Inputs
# --------------------------------
for i in range(len(ddr_coregenelist)):
    dnarepairgenes = ddr_coregenelist[list(ddr_coregenelist.keys())[i]]

    final_gene_exp = pd.merge(gene_df_sev, gene_df_snu, left_index=True, right_index=True, how='inner')
    final_trans_exp = pd.merge(tpm_df_sev, tpm_df_snu, left_index=True, right_index=True, how='inner')
    final_trans_usage = pd.merge(tu_df_sev, tu_df_snu, left_index=True, right_index=True, how='inner')
    final_gene_exp = final_gene_exp.loc[final_gene_exp.index.intersection(dnarepairgenes), :]
    final_trans_exp = final_trans_exp.loc[final_trans_exp.index.str.contains('|'.join(dnarepairgenes)), :]
    final_trans_usage = final_trans_usage.loc[final_trans_usage.index.str.contains('|'.join(dnarepairgenes)), :]


    gene_map = pd.Series([x.split("-")[-1] for x in final_trans_exp.index], index=final_trans_exp.index)

    # -----------------------------
    # 2. Make gene-level matrices
    # -----------------------------
    # (i) Total TPM (all transcripts)
    expr_total = final_trans_exp.groupby(gene_map).sum()

    # (ii) Major TPM (filter first → then sum)
    expr_major = final_trans_exp.loc[final_trans_exp.index.isin(majorlist)] #*minorlist
    gene_map_major = gene_map.loc[expr_major.index]
    expr_major_gene = pd.DataFrame(expr_major.groupby(gene_map_major).sum())
    expr_major_gene = expr_major_gene.apply(pd.to_numeric, errors='coerce')
    expr_major_log = np.log2(expr_major_gene + 0.01)


    expr_df = expr_major_gene.copy()
    clin_df = final_clin.copy()

    # Ensure rows = samples
    if len(set(expr_df.columns).intersection(clin_df.index)) > len(set(expr_df.index).intersection(clin_df.index)):
        expr_df = expr_df.T

    # Align
    common = expr_df.index.intersection(clin_df.index)
    expr_df = expr_df.loc[common].sort_index()
    clin_df = clin_df.loc[common].sort_index()
    y = Surv.from_dataframe(event="recur", time="PFS", data=clin_df)

    # --------------------------------
    # 1. Variance filter
    # --------------------------------
    var_thresh = 0.7  # tune if needed
    expr_var = expr_df.var(axis=0)
    expr_filtered = expr_df.loc[:, expr_var > var_thresh]
    print(f"Variance filter: {expr_df.shape[1]} → {expr_filtered.shape[1]} genes")

    # --------------------------------
    # 2. Combine with BRCAmt + firstline features
    # --------------------------------
    X_z = pd.DataFrame(
        StandardScaler().fit_transform(expr_filtered),
        index=expr_filtered.index,
        columns=expr_filtered.columns,
    )

    firstline_dv = pd.get_dummies(clin_df["firstline"], prefix="firstline", drop_first=True)
    X_clin = pd.concat([clin_df[["BRCAmt"]], firstline_dv], axis=1)
    X_full = pd.concat([X_z, X_clin], axis=1)

    # Cohort for stratified splitting (not used in regression)
    cohort_labels = clin_df["cohort"].astype(str)

    # --------------------------------
    # 3. Nested CV Coxnet
    # --------------------------------
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=77)
    outer_scores = []
    outer_best_params = []
    l1_candidates = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_full, clin_df["BRCAmt"]), 1):
        print(f"\n===== Outer Fold {fold_idx} =====")

        X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        best_cidx, best_alpha, best_l1 = -np.inf, None, None

        for l1_ratio in l1_candidates:
            inner_cv = KFold(n_splits=5, shuffle=True, random_state=fold_idx)
            coxnet_path = CoxnetSurvivalAnalysis(
                l1_ratio=l1_ratio,
                alpha_min_ratio=0.05,
                n_alphas=100
            )
            coxnet_path.fit(X_train, y_train)

            mean_cidx_per_alpha = []
            for coef in coxnet_path.coef_.T:
                preds = np.dot(X_train, coef)
                cidx = concordance_index_censored(
                    y_train["recur"], y_train["PFS"], preds
                )[0]
                mean_cidx_per_alpha.append(cidx)

            best_idx = np.argmax(mean_cidx_per_alpha)
            if mean_cidx_per_alpha[best_idx] > best_cidx:
                best_cidx = mean_cidx_per_alpha[best_idx]
                best_alpha = coxnet_path.alphas_[best_idx]
                best_l1 = l1_ratio

        print(f"Best inner params: l1_ratio={best_l1}, alpha={best_alpha:.4e}, inner C={best_cidx:.3f}")
        outer_best_params.append((best_l1, best_alpha))

        # Refit and test
        coxnet_best = CoxnetSurvivalAnalysis(l1_ratio=best_l1, alphas=[best_alpha])
        coxnet_best.fit(X_train, y_train)
        pred_test = coxnet_best.predict(X_test)
        cidx_test = concordance_index_censored(
            y_test["recur"], y_test["PFS"], pred_test
        )[0]

        print(f"Outer fold test C-index: {cidx_test:.3f}")
        outer_scores.append(cidx_test)

    # --------------------------------
    # 4. Summary
    # --------------------------------
    print("\n==============================")
    print(f"Mean outer test C-index: {np.mean(outer_scores):.3f} ± {np.std(outer_scores):.3f}")
    print("Best params per fold:")
    for i, (l1, a) in enumerate(outer_best_params, 1):
        print(f"  Fold {i}: l1_ratio={l1}, alpha={a:.4e}")


# %%
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
import warnings
from lifelines.exceptions import ConvergenceError

# --- 가정 ---
# ddr_coregenelist (dict): 미리 로드된 유전자 리스트 딕셔너리
# tu_df_sev, tu_df_snu (pd.DataFrame): 각 코호트의 Transcript Usage 데이터 (인덱스: transcript, 컬럼: sample)
# final_clin (pd.DataFrame): 'PFS', 'recur', 'BRCAmt', 'firstline', 'drug' 컬럼을
#                            포함하는, **미리 병합된** 전체 임상 데이터 (인덱스: sample)

# -----------------------------------------------------------------
# 1. 단일 전사체 Cox 회귀 분석을 위한 헬퍼 함수
# -----------------------------------------------------------------
def run_single_transcript_cox(cox_input_df, transcript_col_name, base_covariate_cols):
    """
    단일 전사체에 대해 Cox 회귀 분석을 수행합니다.

    Args:
        cox_input_df (pd.DataFrame): PFS, recur, 기본 공변량, 및 
                                     전사체 컬럼을 포함하는 전체 데이터
        transcript_col_name (str): 분석할 전사체 컬럼의 이름
        base_covariate_cols (list): 'BRCAmt_dummy', 'drug_dummy' 등 
                                     기본 공변량 컬럼 이름 리스트

    Returns:
        dict: 유의미한 경우 결과 (transcript, log10_p, HR), 그렇지 않으면 None
    """
    try:
        # 1. 전사체 데이터의 분산 확인 (거의 모든 샘플에서 값이 같으면 분석 불가)
        if cox_input_df[transcript_col_name].nunique(dropna=False) < 2:
            return None

        
        # Z-score로 TU 값을 변환 (평균 0, 표준편차 1)
        scaler = StandardScaler()
        # .values.reshape(-1, 1)을 통해 1D-array를 2D-array로 변환
        tu_zscored = scaler.fit_transform(cox_input_df[[transcript_col_name]])
        
        # 원본 cox_input_df를 복사하여 Z-score 값으로 대체
        df_to_fit = cox_input_df.copy()
        df_to_fit[transcript_col_name] = tu_zscored
        
        # 2. 유효한 기본 공변량 확인
        valid_base_covs = []
        for col in base_covariate_cols:
            if df_to_fit[col].nunique(dropna=False) > 1:
                valid_base_covs.append(col)

        # 3. CoxPHFitter 모델 준비
        cph = CoxPHFitter(penalizer=0.1) #* 공선성 문제가 해결되었다면 penalizer 제외 가능
        
        all_covs = valid_base_covs + [transcript_col_name]
        df_to_fit_final = df_to_fit[all_covs + ['PFS', 'recur']]
        
        # 4. 모델 피팅
        cph.fit(df_to_fit_final, duration_col='PFS', event_col='recur')
        
        # 5. 결과 추출 (이하 동일)
        summary = cph.summary.loc[transcript_col_name]
        p_val = summary['p']
        HR = summary['exp(coef)']
        
        if (p_val < 0.05) & (HR>1):
            return {
                'transcript': transcript_col_name,
                'log10_p': -np.log10(p_val),
                'HR': summary['exp(coef)'] # 이제 HR은 '1 표준편차'당 비율
            }
            
    except (ConvergenceError, np.linalg.LinAlgError, ValueError) as e:
        # print(f"ERROR (Fit Failed): {transcript_col_name}: {e}")
        return None
    except KeyError:
        # print(f"ERROR (KeyError): {transcript_col_name}")
        return None
        
    return None

# -----------------------------------------------------------------
# 2. 메인 분석 파이프라인
# -----------------------------------------------------------------

print("--- Cox 회귀 분석 시작 ---")

# 1. 기본 임상 데이터 및 더미 변수 *한 번만* 준비
#    (PFS, recur, BRCAmt, firstline, drug 컬럼이 있다고 가정)
base_clin_df = final_clin[['PFS', 'recur', 'BRCAmt','cohort' ]] #'firstline', 'drug'


base_formula_df = pd.get_dummies(base_clin_df, 
                                 columns=['BRCAmt', 'cohort'], #'firstline', 'drug'
                                 drop_first=True)

# 더미 변환된 기본 공변량 이름 목록 저장
base_cov_names = [col for col in base_formula_df.columns if col not in ['PFS', 'recur']]

# 2. 각 유전자 리스트(DDR 경로 등)에 대해 루프 실행
for i in range(len(ddr_coregenelist)):
    gene_list_name = list(ddr_coregenelist.keys())[i]
    dnarepairgenes = ddr_coregenelist[gene_list_name]
    
    print(f"\n===== 2. 유전자 리스트 분석 중: {gene_list_name} =====")
    
    # 3. 이 리스트에 해당하는 전사체 데이터 준비
    final_trans_usage = pd.merge(tu_df_sev, tu_df_snu, left_index=True, right_index=True, how='inner')
    final_trans_usage = final_trans_usage[final_trans_usage.index.isin(majorlist)]
    
    # 현재 유전자 리스트의 유전자를 포함하는 전사체만 필터링
    # (예: 'BRCA1'을 포함하는 'ENST...-BRCA1' 등을 찾음)
    gene_pattern = '|'.join(dnarepairgenes)
    final_trans_usage_filtered = final_trans_usage.loc[final_trans_usage.index.str.contains(gene_pattern, na=False)]
    
    if final_trans_usage_filtered.empty:
        print(f"'{gene_list_name}' 리스트에 해당하는 전사체를 찾을 수 없습니다.")
        continue
        
    # 4. 데이터 정렬 (가장 중요)
    # TU 데이터를 (샘플 x 전사체) 형태로 변환
    tu_transposed = final_trans_usage_filtered.T
    
    # 임상 더미 데이터와 전사체 데이터 간에 공통 샘플을 기준으로 정렬
    cox_input_df, tu_transposed_aligned = base_formula_df.align(tu_transposed, join='inner', axis=0)
    
    # 정렬된 두 데이터를 하나로 합침
    cox_input_df = pd.concat([cox_input_df, tu_transposed_aligned], axis=1)
    
    if cox_input_df.empty or tu_transposed_aligned.empty:
        print("임상 데이터와 전사체 데이터 간에 공통 샘플이 없습니다.")
        continue

    # 5. 현재 유전자 리스트의 *모든 전사체*에 대해 루프 실행
    print(f"총 {len(tu_transposed_aligned.columns)}개의 전사체에 대해 Cox 회귀 분석 수행...")
    significant_transcripts = []
    
    for transcript_name in tu_transposed_aligned.columns:
        # 헬퍼 함수를 호출하여 개별 회귀 분석 실행
        result = run_single_transcript_cox(
            cox_input_df, 
            transcript_col_name=transcript_name, 
            base_covariate_cols=base_cov_names.copy() 
        )
        
        # 유의미한 결과만 저장
        if result:
            significant_transcripts.append(result)
            
    # 6. 현재 유전자 리스트에 대한 결과 요약 출력
    if significant_transcripts:
        print(f"\n[결과] '{gene_list_name}'에서 {len(significant_transcripts)}개의 유의미한 전사체 발견 (p < 0.05):")
        
        # p-value가 낮은 순 (즉, -log10(p)가 높은 순)으로 정렬하여 출력
        for res in sorted(significant_transcripts, key=lambda x: x['log10_p'], reverse=True):
            print(f"  - {res['transcript']}")
            print(f"    HR = {res['HR']:.3f}, -log10(p) = {res['log10_p']:.3f}")
    else:
        print(f"\n[결과] '{gene_list_name}'에서 유의미한 전사체를 찾지 못했습니다 (p < 0.05).")

print("\n--- 모든 분석 완료 ---")

# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

# --------------------------------------------
# 1. Prepare data
# --------------------------------------------
hr_trans = [
    "MSTRG.19251.8-MRE11A",
    "ENST00000469165.2-RAD51B",
    "ENST00000396252.2-NBN",
    "ENST00000482007.1-RAD51C",
    "ENST00000399725.2-RBBP8",
    "MSTRG.109448.106-XRCC2"
]
final_trans_usage = pd.merge(tu_df_sev, tu_df_snu, left_index=True, right_index=True, how='inner')
final_trans_usage = final_trans_usage[final_trans_usage.index.isin(minorlist)]
expr_df = final_trans_usage.copy()
clin_df = final_clin.copy()

# ensure correct orientation
if len(set(expr_df.columns).intersection(clin_df.index)) > len(set(expr_df.index).intersection(clin_df.index)):
    expr_df = expr_df.T

# align samples
common = expr_df.index.intersection(clin_df.index)
expr_df = expr_df.loc[common].sort_index()
clin_df = clin_df.loc[common].sort_index()

# survival object
y = Surv.from_dataframe(event="recur", time="PFS", data=clin_df)

# select only HR transcripts
expr_sel = expr_df.loc[:, expr_df.columns.isin(hr_trans)]

# z-score normalize
X_z = pd.DataFrame(
    StandardScaler().fit_transform(expr_sel),
    index=expr_sel.index,
    columns=expr_sel.columns,
)

# add BRCAmt + firstline if desired
firstline_dv = pd.get_dummies(clin_df["firstline"], prefix="firstline", drop_first=True)
X_clin = pd.concat([clin_df[["BRCAmt"]], firstline_dv], axis=1)
X_full = pd.concat([X_z, X_clin], axis=1)

# --------------------------------------------
# 2. 5-fold cross-validation with tuning
# --------------------------------------------
l1_candidates = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
outer_cv = KFold(n_splits=5, shuffle=True, random_state=77)

results = []
for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_full), 1):
    print(f"\n===== Fold {fold_idx} =====")
    X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    best_cidx, best_alpha, best_l1 = -np.inf, None, None
    for l1_ratio in l1_candidates:
        coxnet_path = CoxnetSurvivalAnalysis(
            l1_ratio=l1_ratio,
            alpha_min_ratio=0.05,
            n_alphas=100
        )
        coxnet_path.fit(X_train, y_train)

        mean_cidx_per_alpha = []
        for coef in coxnet_path.coef_.T:
            preds = np.dot(X_train, coef)
            cidx = concordance_index_censored(
                y_train["recur"], y_train["PFS"], preds
            )[0]
            mean_cidx_per_alpha.append(cidx)

        best_idx = np.argmax(mean_cidx_per_alpha)
        if mean_cidx_per_alpha[best_idx] > best_cidx:
            best_cidx = mean_cidx_per_alpha[best_idx]
            best_alpha = coxnet_path.alphas_[best_idx]
            best_l1 = l1_ratio

    print(f"Best inner params: l1_ratio={best_l1}, alpha={best_alpha:.4e}, inner C={best_cidx:.3f}")

    # retrain with best params
    coxnet_best = CoxnetSurvivalAnalysis(l1_ratio=best_l1, alphas=[best_alpha])
    coxnet_best.fit(X_train, y_train)

    pred_test = coxnet_best.predict(X_test)
    cidx_test = concordance_index_censored(
        y_test["recur"], y_test["PFS"], pred_test
    )[0]

    results.append((fold_idx, best_l1, best_alpha, best_cidx, cidx_test))
    print(f"Fold {fold_idx} test C-index: {cidx_test:.3f}")

# --------------------------------------------
# 3. Summary
# --------------------------------------------
res_df = pd.DataFrame(results, columns=["Fold", "l1_ratio", "alpha", "Inner_C", "Test_C"])
print("\n==============================")
print(res_df)
print(f"\nMean Test C-index: {res_df['Test_C'].mean():.3f} ± {res_df['Test_C'].std():.3f}")

# %%
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

# --------------------------------------------
# 1. Prepare data
# --------------------------------------------
for i in range(len(ddr_coregenelist)):
    hr_genes = ddr_coregenelist[list(ddr_coregenelist.keys())[i]]

    # base expression and clinical
    final_trans_usage = pd.merge(tu_df_sev, tu_df_snu, left_index=True, right_index=True, how="inner")
    final_trans_usage = final_trans_usage[final_trans_usage.index.isin(minorlist)]
    expr_df = final_trans_usage.copy()
    clin_df = final_clin.copy()
    clin_df = clin_df[clin_df['BRCAmt']==0]
    expr_df = expr_df.loc[:,clin_df.index]

    # ensure orientation: rows = samples
    if len(set(expr_df.columns).intersection(clin_df.index)) > len(set(expr_df.index).intersection(clin_df.index)):
        expr_df = expr_df.T

    # align
    common = expr_df.index.intersection(clin_df.index)
    expr_df = expr_df.loc[common].sort_index()
    clin_df = clin_df.loc[common].sort_index()
    y = Surv.from_dataframe(event="recur", time="PFS", data=clin_df)

    # limit to HR transcripts (upper bound set)
    expr_df = expr_df.loc[:, expr_df.columns.str.contains('|'.join(hr_genes))]

    # add BRCAmt + firstline
    firstline_dv = pd.get_dummies(clin_df["firstline"], prefix="firstline", drop_first=True)
    X_clin_all = pd.concat([clin_df[["BRCAmt"]], firstline_dv], axis=1)

    # --------------------------------------------
    # 2. Nested CV (5 folds)
    # --------------------------------------------
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=77)
    l1_candidates = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    results = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(expr_df), 1):
        #print(f"\n===== Outer Fold {fold_idx} =====")
        # Split data
        expr_train, expr_test = expr_df.iloc[train_idx], expr_df.iloc[test_idx]
        clin_train, clin_test = clin_df.iloc[train_idx], clin_df.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ----------------------------------------
        # Step 1. Univariate feature selection (train only)
        # ----------------------------------------
        pvals = []
        hrs = []
        for g in expr_train.columns:
            tmp = pd.concat([clin_train[["PFS", "recur"]], expr_train[[g]]], axis=1)
            try:
                cph = CoxPHFitter(penalizer=0.1).fit(tmp, duration_col="PFS", event_col="recur")
                pvals.append(cph.summary.loc[g, "p"])
                hrs.append(cph.summary.loc[g, "exp(coef)"])  # hazard ratio
            except Exception:
                pvals.append(1.0)
                hrs.append(1.0)

        pval_series = pd.Series(pvals, index=expr_train.columns)
        hr_series = pd.Series(hrs, index=expr_train.columns)

        # Select only significant and protective (HR < 1) transcripts
        sig_genes = pval_series[(pval_series < 0.05) & (hr_series < 1)].index.tolist()

        if len(sig_genes) == 0:
            # fallback: use all HR transcripts
            sig_genes = expr_train.columns.tolist()

        #print(f"Selected {len(sig_genes)} features in fold {fold_idx}: {sig_genes}")

        # ----------------------------------------
        # Step 2. Combine with BRCAmt + firstline
        # ----------------------------------------
        # z-score on selected genes
        X_train_z = pd.DataFrame(
            StandardScaler().fit_transform(expr_train[sig_genes]),
            index=expr_train.index,
            columns=sig_genes
        )
        X_test_z = pd.DataFrame(
            StandardScaler().fit(expr_train[sig_genes]).transform(expr_test[sig_genes]),
            index=expr_test.index,
            columns=sig_genes
        )

        # clinical features
        firstline_dv_train = pd.get_dummies(clin_train["firstline"], prefix="firstline", drop_first=True)
        firstline_dv_test = pd.get_dummies(clin_test["firstline"], prefix="firstline", drop_first=True)
        cohort_dv_train = pd.get_dummies(clin_train["cohort"], prefix="cohort", drop_first=True)
        cohort_dv_test = pd.get_dummies(clin_test["cohort"], prefix="cohort", drop_first=True)
        X_clin_train = pd.concat([clin_train[["BRCAmt"]], firstline_dv_train], axis=1)
        X_clin_test = pd.concat([clin_test[["BRCAmt"]], firstline_dv_test], axis=1)
        X_clin_train = pd.concat([X_clin_train, cohort_dv_train], axis=1)
        X_clin_test = pd.concat([X_clin_test, cohort_dv_test], axis=1)

        X_train = pd.concat([X_train_z, X_clin_train], axis=1)
        X_test = pd.concat([X_test_z, X_clin_test], axis=1)
        
        # X_train = X_train_z
        # X_test = X_test_z

        # ----------------------------------------
        # Step 3. Inner tuning (alpha + l1_ratio)
        # ----------------------------------------
        best_cidx, best_alpha, best_l1 = -np.inf, None, None

        for l1_ratio in l1_candidates:
            coxnet_path = CoxnetSurvivalAnalysis(
                l1_ratio=l1_ratio,
                alpha_min_ratio=0.2,
                n_alphas=100
            )
            coxnet_path.fit(X_train, y_train)

            mean_cidx_per_alpha = []
            for coef in coxnet_path.coef_.T:
                preds = np.dot(X_train, coef)
                cidx = concordance_index_censored(
                    y_train["recur"], y_train["PFS"], preds
                )[0]
                mean_cidx_per_alpha.append(cidx)

            best_idx = np.argmax(mean_cidx_per_alpha)
            if mean_cidx_per_alpha[best_idx] > best_cidx:
                best_cidx = mean_cidx_per_alpha[best_idx]
                best_alpha = coxnet_path.alphas_[best_idx]
                best_l1 = l1_ratio

        #print(f"Best inner params: l1_ratio={best_l1}, alpha={best_alpha:.4e}, inner C={best_cidx:.3f}")

        # ----------------------------------------
        # Step 4. Refit and evaluate on test fold
        # ----------------------------------------
        coxnet_best = CoxnetSurvivalAnalysis(l1_ratio=best_l1, alphas=[best_alpha])
        coxnet_best.fit(X_train, y_train)
        pred_test = coxnet_best.predict(X_test)
        cidx_test = concordance_index_censored(
            y_test["recur"], y_test["PFS"], pred_test
        )[0]

        results.append((fold_idx, len(sig_genes), best_l1, best_alpha, best_cidx, cidx_test))
        #print(f"Fold {fold_idx} test C-index: {cidx_test:.3f}")

    # --------------------------------------------
    # 3. Summary
    # --------------------------------------------
    res_df = pd.DataFrame(results, columns=["Fold", "n_features", "l1_ratio", "alpha", "Inner_C", "Test_C"])
    print("\n=============================="+list(ddr_coregenelist.keys())[i])
    #print(res_df)
    print(f"\nMean Test C-index: {res_df['Test_C'].mean():.3f} ± {res_df['Test_C'].std():.3f}")

# %%
