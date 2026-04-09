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
val_clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2025clinical/112_validation_clinicalinfo_new.txt', sep='\t', index_col=0)
val_df = val_df.iloc[:-1,:-1]
val_df = val_df.apply(pd.to_numeric, errors='coerce')

genesymbol =  pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM_symbol.txt', sep='\t', index_col=0)
genesymbol = pd.DataFrame(genesymbol.iloc[:,-1])
genesymbol.columns = ['genesymbol']
vallist = list(val_clin.index)
val_df = val_df.loc[:,vallist]
val_gene = val_gene.apply(pd.to_numeric, errors='coerce')
val_gene = val_gene.iloc[:-1,:]
val_gene = val_gene.loc[:,vallist]
val_gene = pd.merge(val_gene,genesymbol,left_index=True,right_index=True, how='left')
val_gene.set_index('genesymbol', inplace=True)

majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = majorminor[majorminor['type']=='major']['gene_ENST'].to_list()

majorlist = [x for x in majorlist if 'ENST' in x]


from scipy.stats import spearmanr

olaparib_list = val_clin[val_clin['drug']=='Olaparib'].index.to_list()
niraparib_list = val_clin[val_clin['drug']=='Niraparib'].index.to_list()

main_list = val_clin[val_clin['setting']=='maintenance'].index.to_list()
sal_list = val_clin[val_clin['setting']=='salvage'].index.to_list()

main_gene = val_gene.loc[:,main_list]
sal_gene = val_gene.loc[:,sal_list]

main_tu = val_df.loc[:,main_list]
sal_tu = val_df.loc[:,sal_list]

main_clin = val_clin.loc[main_list,:]
sal_clin = val_clin.loc[sal_list,:]

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

major_df = val_df.loc[val_df.index.isin(majorlist),:]

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

# %%

from collections import OrderedDict, defaultdict
from scipy.stats import zscore
import statsmodels.api as sm
from statsmodels.tools import add_constant
from statsmodels.stats.multitest import multipletests

# -----------------------------
# Utilities
# -----------------------------
def align_by_samples(major_df: pd.DataFrame, val_clin: pd.DataFrame):
    """Align columns of major_df to rows of val_clin using intersection of sample IDs."""
    common = [s for s in major_df.columns if s in val_clin.index]
    X = major_df.loc[:, common].copy()
    clin = val_clin.loc[common].copy()
    return X, clin

def index_to_gene_series(transcript_index: pd.Index) -> pd.Series:
    """Extract gene symbol from transcript id like 'ENST...-GENE'."""
    return pd.Series(transcript_index).astype(str).str.split("-").str[-1].values

def map_gene_dict_to_transcripts(gene_dict: "OrderedDict[str, list[str]]",
                                 major_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a tidy mapping table: transcript, gene, term.
    A transcript can appear in multiple terms if the gene is listed in multiple terms.
    """
    gene_by_tx = index_to_gene_series(major_df.index)
    tx_df = pd.DataFrame({"transcript": major_df.index, "gene": gene_by_tx})
    
    rows = []
    for term, genes in gene_dict.items():
        gene_set = set(genes)
        mask = tx_df["gene"].isin(gene_set)
        if mask.any():
            sub = tx_df.loc[mask].copy()
            sub["term"] = term
            rows.append(sub)
    if not rows:
        return pd.DataFrame(columns=["transcript", "gene", "term"])
    return pd.concat(rows, axis=0, ignore_index=True)

def build_design_matrix(TU: pd.Series, cov: pd.DataFrame):
    """
    Build X (design matrix) with standardized TU and given covariates.
    Drop rows with NaNs in either X or y later in model functions.
    """
    X = cov.copy()
    x_tu = TU.astype(float)
    if x_tu.std(skipna=True) == 0 or x_tu.notna().sum() < 3:
        return None  # Not enough variation
    X = X.join((x_tu - x_tu.mean()) / x_tu.std(ddof=0), how="inner")
    X = X.rename(columns={X.columns[-1]: "TU"})  # last col is TU we just joined
    X = add_constant(X, has_constant="add")
    X = X.astype(float)   # ensure all numeric
    return X

def fdr_bh(pvals: pd.Series, alpha=0.05):
    """Benjamini-Hochberg FDR control."""
    p = pvals.values
    mask = np.isfinite(p)
    q = np.full_like(p, np.nan, dtype=float)
    if mask.sum() > 0:
        rej, qvals, *_ = multipletests(p[mask], method="fdr_bh", alpha=alpha)
        q[mask] = qvals
    return pd.Series(q, index=pvals.index)

# --- PATCH: per-transcript OLS with CI ---
def maintenance_per_transcript_ols(X_maint, clin_maint, mapping):
    import statsmodels.api as sm
    cov = prep_clin_covariates(clin_maint)
    y = pd.to_numeric(clin_maint["PFS"], errors="coerce")

    rows = []
    for tx in mapping["transcript"].unique():
        if tx not in X_maint.index:
            continue
        X = build_design_matrix(X_maint.loc[tx], cov)
        if X is None:
            continue
        dfm = pd.concat([y.rename("PFS"), X], axis=1).dropna()
        if dfm.shape[0] < 10:
            continue
        try:
            fit = sm.OLS(dfm["PFS"], dfm.drop(columns=["PFS"])).fit()
            coef = fit.params.get("TU", np.nan)
            pval = fit.pvalues.get("TU", np.nan)
            se   = fit.bse.get("TU", np.nan)
            lci, uci = fit.conf_int().loc["TU"] if "TU" in fit.params.index else (np.nan, np.nan)
            rows.append({"transcript": tx, "coef": coef, "se": se, "lci": lci, "uci": uci,
                         "pval": pval, "n": dfm.shape[0]})
        except Exception:
            continue

    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res = res.merge(mapping[["transcript","gene"]].drop_duplicates(), on="transcript", how="left")
    res["qval"] = fdr_bh(res["pval"])
    return res.sort_values(["qval","pval","coef"], ascending=[True,True,True])
    print(fit.summary())

# --- PATCH: per-transcript Logit with CI ---
def salvage_per_transcript_logit(X_salv, clin_salv, mapping):
    import statsmodels.api as sm
    cov = prep_clin_covariates(clin_salv)
    y = pd.to_numeric(clin_salv["response"], errors="coerce")

    rows = []
    for tx in mapping["transcript"].unique():
        if tx not in X_salv.index:
            continue
        X = build_design_matrix(X_salv.loc[tx], cov)
        if X is None:
            continue
        dfm = pd.concat([y.rename("response"), X], axis=1).dropna()
        if dfm.shape[0] < 10:
            continue
        yy = dfm["response"].astype(int)
        if yy.nunique() < 2:
            continue
        try:
            fit = sm.Logit(yy, dfm.drop(columns=["response"])).fit(disp=False, maxiter=200)
            coef = fit.params.get("TU", np.nan)
            pval = fit.pvalues.get("TU", np.nan)
            se   = fit.bse.get("TU", np.nan)
            if "TU" in fit.params.index:
                lci, uci = fit.conf_int().loc["TU"]
            else:
                lci, uci = (np.nan, np.nan)
            rows.append({"transcript": tx, "coef": coef, "se": se, "lci": lci, "uci": uci,
                         "pval": pval, "n": dfm.shape[0]})
        except Exception:
            continue

    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res = res.merge(mapping[["transcript","gene"]].drop_duplicates(), on="transcript", how="left")
    res["qval"] = fdr_bh(res["pval"])
    return res.sort_values(["qval","pval","coef"], ascending=[True,True,False])

def term_score_matrix(X: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """
    For each term, compute a per-sample score = mean z-scored TU across member transcripts.
    Returns DataFrame (rows=samples, cols=terms).
    """
    # z-score per transcript across samples
    Z = X.apply(lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) not in (0, np.nan) else s*0, axis=1)
    term_to_txs = mapping.groupby("term")["transcript"].apply(lambda s: list(set(s)))
    scores = {}
    for term, txs in term_to_txs.items():
        txs = [t for t in txs if t in Z.index]
        if len(txs) == 0:
            continue
        scores[term] = Z.loc[txs].mean(axis=0)
    if not scores:
        return pd.DataFrame(index=X.columns)
    S = pd.DataFrame(scores)
    return S

# --- PATCH: per-term OLS/Logit with CI ---
def maintenance_per_term_ols(X_maint, clin_maint, mapping):
    import statsmodels.api as sm
    cov = prep_clin_covariates(clin_maint)
    y = pd.to_numeric(clin_maint["PFS"], errors="coerce")
    S = term_score_matrix(X_maint, mapping)

    rows = []
    for term in S.columns:
        X = add_constant(cov.join(S[[term]]).rename(columns={term: "term_score"}), has_constant="add")
        X = X.astype(float) 
        dfm = pd.concat([y.rename("PFS"), X], axis=1).dropna()
        if dfm.shape[0] < 10:
            continue
        try:
            fit = sm.OLS(dfm["PFS"], dfm.drop(columns=["PFS"])).fit()
            coef = fit.params.get("term_score", np.nan)
            pval = fit.pvalues.get("term_score", np.nan)
            se   = fit.bse.get("term_score", np.nan)
            if "term_score" in fit.params.index:
                lci, uci = fit.conf_int().loc["term_score"]
            else:
                lci, uci = (np.nan, np.nan)
            rows.append({"term": term, "coef": coef, "se": se, "lci": lci, "uci": uci,
                         "pval": pval, "n": dfm.shape[0]})
        except Exception:
            continue
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["qval"] = fdr_bh(res["pval"])
    return res.sort_values(["qval","pval"])

def salvage_per_term_logit(X_salv, clin_salv, mapping):
    import statsmodels.api as sm
    cov = prep_clin_covariates(clin_salv)
    y = pd.to_numeric(clin_salv["response"], errors="coerce").astype("float")
    S = term_score_matrix(X_salv, mapping)

    rows = []
    for term in S.columns:
        X = add_constant(cov.join(S[[term]]).rename(columns={term: "term_score"}), has_constant="add")
        X = X.astype(float)
        dfm = pd.concat([y.rename("response"), X], axis=1).dropna()
        if dfm.shape[0] < 10:
            continue
        yy = dfm["response"].astype(int)
        if yy.nunique() < 2:
            continue
        try:
            fit = sm.Logit(yy, dfm.drop(columns=["response"])).fit(disp=False, maxiter=200)
            coef = fit.params.get("term_score", np.nan)
            pval = fit.pvalues.get("term_score", np.nan)
            se   = fit.bse.get("term_score", np.nan)
            if "term_score" in fit.params.index:
                lci, uci = fit.conf_int().loc["term_score"]
            else:
                lci, uci = (np.nan, np.nan)
            rows.append({"term": term, "coef": coef, "se": se, "lci": lci, "uci": uci,
                         "pval": pval, "n": dfm.shape[0]})
        except Exception:
            continue
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["qval"] = fdr_bh(res["pval"])
    return res.sort_values(["qval","pval"])

def run_covariate_adjusted_discovery_aligned(X: pd.DataFrame,
                                             clin: pd.DataFrame,
                                             gene_dict: "OrderedDict[str, list[str]]"):
    maint_mask = clin["setting"].astype(str).str.contains("maint", case=False, na=False)
    salv_mask  = clin["setting"].astype(str).str.contains("salv", case=False, na=False)

    X_maint, clin_maint = X.loc[:, maint_mask].copy(), clin.loc[maint_mask].copy()
    X_salv,  clin_salv  = X.loc[:, salv_mask].copy(),  clin.loc[salv_mask].copy()

    mapping = map_gene_dict_to_transcripts(gene_dict, X)

    maint_tx = maintenance_per_transcript_cox(X_maint, clin_maint, mapping) if not X_maint.empty else pd.DataFrame()
    salv_tx  = salvage_per_transcript_elasticnet(X_salv, clin_salv, mapping) if not X_salv.empty else pd.DataFrame()

    # 빈 DF라도 포함시켜서 pipeline 에러 안 나게
    maint_term = pd.DataFrame()
    salv_term  = pd.DataFrame()

    return {
        "maintenance_transcript": maint_tx,
        "salvage_transcript": salv_tx,
        "maintenance_term": maint_term,
        "salvage_term": salv_term,
        "mapping": mapping
    }



def annotate_transcript_terms(res_tx: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Attach semicolon-joined terms a transcript belongs to."""
    term_map = (mapping.groupby("transcript")["term"]
                .apply(lambda xs: "; ".join(sorted(set(xs)))))
    out = res_tx.merge(term_map.rename("terms"), on="transcript", how="left")
    return out

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------
# Top-hit tables
# --------------------
def make_top_table(res_df: pd.DataFrame, id_cols, top_n=20, alpha=0.05):
    if res_df is None or res_df.empty:
        return pd.DataFrame(columns=id_cols+["coef","lci","uci","pval","qval","n","sig"])
    df = res_df.copy()

    # ✅ pval/qval 없는 경우 dummy로 채우기
    if "pval" not in df.columns:
        df["pval"] = np.nan
    if "qval" not in df.columns:
        df["qval"] = np.nan

    # significance 플래그
    df["sig"] = (df["qval"] < alpha)
    df["abscoef"] = df["coef"].abs()

    # 정렬 기준: qval → pval → abscoef (있을 때만)
    sort_keys = [k for k in ["qval","pval","abscoef"] if k in df.columns]
    asc_flags = [True if k != "abscoef" else False for k in sort_keys]

    df = df.sort_values(sort_keys, ascending=asc_flags)

    cols_avail = [c for c in ["coef","lci","uci","pval","qval","n","sig"] if c in df.columns]
    out = df[id_cols + cols_avail].head(top_n)
    return out



# --------------------
# Volcano plot
# --------------------
def plot_volcano(res_df: pd.DataFrame, title: str, alpha=0.05, label_col=None, top_annot=15):
    if res_df is None or res_df.empty:
        print(f"[{title}] empty results")
        return
    df = res_df.copy()
    df = df[np.isfinite(df["coef"])]

    # fallback: if no pval, just set dummy
    if "pval" not in df.columns:
        df["pval"] = 1.0
    if "qval" not in df.columns:
        df["qval"] = 1.0

    df["mlog10p"] = -np.log10(df["pval"].clip(lower=1e-300))
    df["sig"] = df["qval"] < alpha

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.scatter(df.loc[~df["sig"], "coef"], df.loc[~df["sig"], "mlog10p"],
               s=18, alpha=0.65, label="NS")
    ax.scatter(df.loc[df["sig"], "coef"], df.loc[df["sig"], "mlog10p"],
               s=24, alpha=0.9, label=f"FDR<{alpha}")
    ax.axhline(-np.log10(0.05), linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Coefficient (TU effect)")
    ax.set_ylabel("-log10(p)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)

    if label_col and label_col in df.columns:
        top = df.sort_values("mlog10p", ascending=False).head(top_annot)
        for _, r in top.iterrows():
            ax.annotate(str(r[label_col])[:24],
                        (r["coef"], r["mlog10p"]),
                        xytext=(3,3), textcoords="offset points", fontsize=8)
    plt.tight_layout()
    plt.show()


# --------------------
# Forest plot
# --------------------
def plot_forest(res_df: pd.DataFrame, id_col: str, title: str, top_n=20, alpha=0.05, sort_by="qval"):
    if res_df is None or res_df.empty:
        print(f"[{title}] empty results")
        return
    df = res_df.copy()
    needed = ["coef","lci","uci", id_col]
    if any(c not in df.columns for c in needed):
        print(f"[{title}] missing CI columns; consider refit_for_forest().")
        return
    
    df["abscoef"] = df["coef"].abs()

    # ✅ fallback if sort_by not present
    if sort_by not in df.columns:
        sort_by = "coef"

    df = df.sort_values([sort_by, "abscoef"], ascending=[True, False]).head(top_n)
    df = df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(6.4, 0.36*len(df) + 1.8))
    y = np.arange(len(df))
    ax.errorbar(df["coef"], y,
                xerr=[df["coef"]-df["lci"], df["uci"]-df["coef"]],
                fmt="o", capsize=3, elinewidth=1, markersize=4)
    ax.axvline(0.0, linestyle="--", linewidth=1, alpha=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(df[id_col].astype(str).str.slice(0,32))
    ax.set_xlabel("Effect size (coef; 95% CI)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# --------------------
# (옵션) CI가 없는 경우 상위 K개만 리핏
# --------------------
def refit_for_forest(kind: str, id_list, X, clin, cov_builder, build_dm, is_term=False):
    """
    kind: "ols" or "logit"
    id_list: list of transcript ids or term names
    X: transcript TU (rows=transcripts) or term score matrix (rows=samples, cols=terms) when is_term=True
    clin: clinical subset DataFrame (maintenance or salvage)
    Returns DataFrame with coef,lci,uci,pval for those ids (skipping failures).
    """
    import statsmodels.api as sm
    rows = []
    cov = cov_builder(clin)
    target_col = "PFS" if kind=="ols" else "response"
    y = pd.to_numeric(clin[target_col], errors="coerce")
    for _id in id_list:
        try:
            if is_term:
                vec = X[_id].rename("feature")  # S matrix column
                Xd = add_constant(cov.join(vec), has_constant="add").rename(columns={"feature":"feat"})
            else:
                vec = X.loc[_id].rename("TU")
                Xd = build_dm(vec, cov)
            dfm = pd.concat([y.rename(target_col), Xd], axis=1).dropna()
            if dfm.shape[0] < 10:
                continue
            if kind == "ols":
                fit = sm.OLS(dfm[target_col], dfm.drop(columns=[target_col])).fit()
            else:
                yy = dfm[target_col].astype(int)
                if yy.nunique() < 2:
                    continue
                fit = sm.Logit(yy, dfm.drop(columns=[target_col])).fit(disp=False, maxiter=200)
            coef = fit.params.iloc[-1]  # last is TU/feat
            pval = fit.pvalues.iloc[-1]
            se   = fit.bse.iloc[-1]
            lci, uci = fit.conf_int().iloc[-1]
            rows.append({"id": _id, "coef": coef, "se": se, "lci": lci, "uci": uci, "pval": pval, "n": dfm.shape[0]})
        except Exception:
            continue
    return pd.DataFrame(rows)

def visualize_all(results, mapping, dict_name="DDR"):
    # Per-transcript tables
    maint_tx = annotate_transcript_terms(results["maintenance_transcript"], mapping) if not results["maintenance_transcript"].empty else results["maintenance_transcript"]
    salv_tx  = annotate_transcript_terms(results["salvage_transcript"], mapping)    if not results["salvage_transcript"].empty    else results["salvage_transcript"]

    top_maint_tx = make_top_table(maint_tx, ["transcript","gene","terms"], top_n=20)
    top_salv_tx  = make_top_table(salv_tx,  ["transcript","gene","terms"], top_n=20)

    # Per-term tables
    top_maint_term = make_top_table(results["maintenance_term"], ["term"], top_n=20)
    top_salv_term  = make_top_table(results["salvage_term"], ["term"], top_n=20)

    # Print heads (or save to CSV)
    print(f"\n[{dict_name}] Maintenance per-transcript (top 20):")
    print(top_maint_tx.head(20))
    print(f"\n[{dict_name}] Salvage per-transcript (top 20):")
    print(top_salv_tx.head(20))
    print(f"\n[{dict_name}] Maintenance per-term (top 20):")
    print(top_maint_term.head(20))
    print(f"\n[{dict_name}] Salvage per-term (top 20):")
    print(top_salv_term.head(20))

    # Volcano plots
    plot_volcano(maint_tx,  f"{dict_name} • Maintenance (per-transcript OLS)", alpha=0.05, label_col="gene", top_annot=15)
    plot_volcano(salv_tx,   f"{dict_name} • Salvage (per-transcript Logit)",  alpha=0.05, label_col="gene", top_annot=15)
    plot_volcano(results["maintenance_term"], f"{dict_name} • Maintenance (per-term OLS)",  alpha=0.05, label_col="term", top_annot=15)
    plot_volcano(results["salvage_term"],     f"{dict_name} • Salvage (per-term Logit)",   alpha=0.05, label_col="term", top_annot=15)

    # Forest plots (if CI exists; otherwise use refit_for_forest to build CI)
    if not maint_tx.empty and {"lci","uci"}.issubset(maint_tx.columns):
        plot_forest(maint_tx, "gene", f"{dict_name} • Maintenance Top Transcripts (Forest)", top_n=20)
    if not salv_tx.empty and {"lci","uci"}.issubset(salv_tx.columns):
        plot_forest(salv_tx, "gene", f"{dict_name} • Salvage Top Transcripts (Forest)", top_n=20)
    if not results["maintenance_term"].empty and {"lci","uci"}.issubset(results["maintenance_term"].columns):
        plot_forest(results["maintenance_term"], "term", f"{dict_name} • Maintenance Terms (Forest)", top_n=20)
    if not results["salvage_term"].empty and {"lci","uci"}.issubset(results["salvage_term"].columns):
        plot_forest(results["salvage_term"], "term", f"{dict_name} • Salvage Terms (Forest)", top_n=20)

def prep_clin_covariates(clin: pd.DataFrame):
    line_num = (
        clin["line"].astype(str)
        .str.extract(r"(\d+)")[0]
        .astype(float)
    )
    cov = pd.DataFrame(index=clin.index)
    cov["line_num"] = line_num
    cov["BRCAmt"] = pd.to_numeric(clin["BRCAmt"], errors="coerce")
    #cov["gHRDscore"] = pd.to_numeric(clin["gHRDscore"], errors="coerce")
    drug = clin["drug"].astype(str).fillna("Unknown")
    drug_dummies = pd.get_dummies(drug, prefix="drug", drop_first=True)

    cov = pd.concat([cov, drug_dummies], axis=1)

    # 🔑 ensure everything is float
    cov = cov.apply(pd.to_numeric, errors="coerce")
    return cov

from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

# --- Cox per-transcript (maintenance / PFS) ---
def maintenance_per_transcript_cox(X_maint, clin_maint, mapping):
    rows = []
    cov = prep_clin_covariates(clin_maint)
    for tx in mapping["transcript"].unique():
        if tx not in X_maint.index: 
            continue
        tu = X_maint.loc[tx].astype(float)
        df = cov.copy()
        df["TU"] = tu
        df["PFS"] = clin_maint["PFS"].astype(float)
        df["event"] = (clin_maint["recur"] == 1).astype(int)  # 1=PD event, others censored
        df = df.dropna()
        if df.shape[0] < 10:
            continue
        try:
            cph = CoxPHFitter()
            cph.fit(df, duration_col="PFS", event_col="event")
            s = cph.summary.loc["TU"]
            rows.append({
                "transcript": tx,
                "coef": s["coef"],
                "HR": s["exp(coef)"],
                "se": s["se(coef)"],
                "lci": s["coef lower 95%"],
                "uci": s["coef upper 95%"],
                "pval": s["p"],
                "n": df.shape[0]
            })
        except Exception as e:
            continue
    res = pd.DataFrame(rows)
    if res.empty: return res
    res = res.merge(mapping[["transcript","gene"]].drop_duplicates(), on="transcript", how="left")
    res["qval"] = fdr_bh(res["pval"])
    return res.sort_values(["qval","pval"])

# --- Elastic Net Logistic per-transcript (salvage / response) ---
def salvage_per_transcript_elasticnet(X_salv, clin_salv, mapping):
    rows = []
    cov = prep_clin_covariates(clin_salv)
    y = clin_salv["response"].astype(float)
    for tx in mapping["transcript"].unique():
        if tx not in X_salv.index:
            continue
        tu = X_salv.loc[tx].astype(float)
        df = cov.copy()
        df["TU"] = tu
        df = df.dropna()
        y_sub = y.loc[df.index]
        if df.shape[0] < 10 or y_sub.nunique() < 2:
            continue
        try:
            scaler = StandardScaler()
            X_std = scaler.fit_transform(df)
            model = LogisticRegressionCV(
                penalty="elasticnet", solver="saga",
                l1_ratios=[0.5], cv=3, max_iter=1000
            )
            model.fit(X_std, y_sub)
            coef = model.coef_[0][-1]  # TU의 계수
            rows.append({
                "transcript": tx,
                "coef": coef,
                "pval": np.nan,   # Elastic Net은 pval 제공 안 함
                "n": df.shape[0]
            })
        except Exception:
            continue
    res = pd.DataFrame(rows)
    if res.empty: return res
    res = res.merge(mapping[["transcript","gene"]].drop_duplicates(), on="transcript", how="left")
    return res.sort_values("coef", ascending=False)

# %%
# =========================
# Covariate-adjusted discovery: EXECUTION SCRIPT
# Prereqs:
#  - major_df: TU matrix (rows=transcripts, cols=samples)
#  - val_clin: clinical DataFrame (rows=samples)
#  - ddr_genelist: OrderedDict[str, list[str]]
#  - ddr_coregenelist: OrderedDict[str, list[str]]
#  - All helper functions from previous cells are already defined:
#    align_by_samples, map_gene_dict_to_transcripts, run_covariate_adjusted_discovery,
#    annotate_transcript_terms, make_top_table, plot_volcano, plot_forest, visualize_all, etc.
# =========================
# =========================
# EXECUTION SCRIPT
# =========================

import os
from datetime import datetime

# 0) Sanity checks
required_clin_cols = {"setting", "line", "BRCAmt", "gHRDscore", "drug", "PFS", "response"}
missing = required_clin_cols - set(val_clin.columns)
assert not missing, f"val_clin is missing required columns: {missing}"

# 1) Align samples
X, clin = align_by_samples(major_df, val_clin)
print(f"Aligned shapes: X {X.shape}, clin {clin.shape}")

# 2) Output directory
out_dir = "/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/covariate_analysis_2509"
os.makedirs(out_dir, exist_ok=True)

# 3) Run for both dictionaries
runs = [
    #("AR feature", AR_genelist),
    ("CoreDDR", ddr_coregenelist),
]

all_summaries = []

for dict_name, gene_dict in runs:
    print(f"\n=== Running covariate-adjusted discovery for: {dict_name} ===")

    # Execute discovery (already aligned)
    results = run_covariate_adjusted_discovery_aligned(X, clin, gene_dict)
    mapping = results["mapping"]

    # Save raw result tables
    for key in ["maintenance_transcript", "salvage_transcript","maintenance_term", "salvage_term"]:
        df = results[key]
        if df is not None and not df.empty:
            csv_path = os.path.join(out_dir, f"{dict_name}_{key}.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved: {csv_path} ({df.shape[0]} rows)")
        else:
            print(f"[{dict_name}] {key}: empty")

    # Annotate & top tables
    maint_tx = results["maintenance_transcript"]
    salv_tx  = results["salvage_transcript"]

    if not maint_tx.empty:
        maint_tx_anno = annotate_transcript_terms(maint_tx, mapping)
        maint_tx_top = make_top_table(maint_tx_anno, ["transcript","gene","terms"], top_n=20)
        maint_tx_top.to_csv(os.path.join(out_dir, f"{dict_name}_maintenance_transcript_top20.csv"), index=False)
    else:
        maint_tx_top = pd.DataFrame()

    if not salv_tx.empty:
        salv_tx_anno = annotate_transcript_terms(salv_tx, mapping)
        salv_tx_top = make_top_table(salv_tx_anno, ["transcript","gene","terms"], top_n=20)
        salv_tx_top.to_csv(os.path.join(out_dir, f"{dict_name}_salvage_transcript_top20.csv"), index=False)
    else:
        salv_tx_top = pd.DataFrame()

    # Terms
    maint_term = results["maintenance_term"]
    salv_term  = results["salvage_term"]

    if not maint_term.empty:
        maint_term_top = make_top_table(maint_term, ["term"], top_n=20)
        maint_term_top.to_csv(os.path.join(out_dir, f"{dict_name}_maintenance_term_top20.csv"), index=False)
    else:
        maint_term_top = pd.DataFrame()

    if not salv_term.empty:
        salv_term_top = make_top_table(salv_term, ["term"], top_n=20)
        salv_term_top.to_csv(os.path.join(out_dir, f"{dict_name}_salvage_term_top20.csv"), index=False)
    else:
        salv_term_top = pd.DataFrame()

    # Visualize
    visualize_all(results, mapping, dict_name=dict_name)

    # Collect for summary
    for tag, df in [("maintenance_term", maint_term), ("salvage_term", salv_term)]:
        if not df.empty:
            tmp = df.copy()
            tmp["where"] = tag
            tmp["dict"] = dict_name
            all_summaries.append(tmp[["dict","where","term","coef","pval","qval","n"]])

# 4) Cross-dict term comparison
if all_summaries:
    summary_df = pd.concat(all_summaries, axis=0, ignore_index=True)
    piv = (summary_df
           .pivot_table(index=["where","term"],
                        columns="dict",
                        values=["coef","pval","qval","n"]))
    piv.columns = [f"{a}__{b}" for a,b in piv.columns]
    piv = piv.reset_index()
    piv_path = os.path.join(out_dir, "term_level_comparison_DDR_vs_CoreDDR.csv")
    piv.to_csv(piv_path, index=False)
    print(f"\nSaved term-level comparison: {piv_path}")

print(f"\nAll done. Outputs in: {out_dir}")

# %%
import os
from datetime import datetime

# 0) Sanity checks
required_clin_cols = {"setting", "line", "BRCAmt", "gHRDscore", "drug", "PFS", "response"}
missing = required_clin_cols - set(val_clin.columns)
assert not missing, f"val_clin is missing required columns: {missing}"

# 1) Align samples
X, clin = align_by_samples(major_df, val_clin)
print(f"Aligned shapes: X {X.shape}, clin {clin.shape}")

# 2) Output directory
out_dir = "/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/covariate_analysis_2509"
os.makedirs(out_dir, exist_ok=True)

# 3) Run for both dictionaries
runs = [
    #("AR feature", AR_genelist),
    ("CoreDDR", ddr_coregenelist),
]

all_summaries = []

for dict_name, gene_dict in runs:
    print(f"\n=== Running covariate-adjusted discovery for: {dict_name} ===")

    # Sample split
    maint_mask = clin["setting"].astype(str).str.contains("maint", case=False, na=False)
    salv_mask  = clin["setting"].astype(str).str.contains("salv", case=False, na=False)

    X_maint, clin_maint = X.loc[:, maint_mask].copy(), clin.loc[maint_mask].copy()
    X_salv,  clin_salv  = X.loc[:, salv_mask].copy(),  clin.loc[salv_mask].copy()

    # Transcript ↔ gene ↔ term mapping
    mapping = map_gene_dict_to_transcripts(gene_dict, X)

    # --- Per-transcript ---
    maint_tx = maintenance_per_transcript_cox(X_maint, clin_maint, mapping) if not X_maint.empty else pd.DataFrame()
    salv_tx  = salvage_per_transcript_elasticnet(X_salv, clin_salv, mapping) if not X_salv.empty else pd.DataFrame()

    # --- Per-term ---
    maint_term = maintenance_per_term_ols(X_maint, clin_maint, mapping) if not X_maint.empty else pd.DataFrame()
    salv_term  = salvage_per_term_logit(X_salv, clin_salv, mapping) if not X_salv.empty else pd.DataFrame()

    results = {
        "maintenance_transcript": maint_tx,
        "salvage_transcript": salv_tx,
        "maintenance_term": maint_term,
        "salvage_term": salv_term,
        "mapping": mapping
    }

    # Save raw result tables
    for key, df in results.items():
        if key == "mapping":
            continue
        if df is not None and not df.empty:
            csv_path = os.path.join(out_dir, f"{dict_name}_{key}.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved: {csv_path} ({df.shape[0]} rows)")
        else:
            print(f"[{dict_name}] {key}: empty")

    # Annotate & top tables
    if not maint_tx.empty:
        maint_tx_anno = annotate_transcript_terms(maint_tx, mapping)
        maint_tx_top = make_top_table(maint_tx_anno, ["transcript","gene","terms"], top_n=20)
        maint_tx_top.to_csv(os.path.join(out_dir, f"{dict_name}_maintenance_transcript_top20.csv"), index=False)

    if not salv_tx.empty:
        salv_tx_anno = annotate_transcript_terms(salv_tx, mapping)
        salv_tx_top = make_top_table(salv_tx_anno, ["transcript","gene","terms"], top_n=20)
        salv_tx_top.to_csv(os.path.join(out_dir, f"{dict_name}_salvage_transcript_top20.csv"), index=False)

    if not maint_term.empty:
        maint_term_top = make_top_table(maint_term, ["term"], top_n=20)
        maint_term_top.to_csv(os.path.join(out_dir, f"{dict_name}_maintenance_term_top20.csv"), index=False)

    if not salv_term.empty:
        salv_term_top = make_top_table(salv_term, ["term"], top_n=20)
        salv_term_top.to_csv(os.path.join(out_dir, f"{dict_name}_salvage_term_top20.csv"), index=False)

    # Visualize
    visualize_all(results, mapping, dict_name=dict_name)

    # Collect for summary
    for tag, df in [("maintenance_term", maint_term), ("salvage_term", salv_term)]:
        if not df.empty:
            tmp = df.copy()
            tmp["where"] = tag
            tmp["dict"] = dict_name
            all_summaries.append(tmp[["dict","where","term","coef","pval","qval","n"]])

# 4) Cross-dict term comparison
if all_summaries:
    summary_df = pd.concat(all_summaries, axis=0, ignore_index=True)
    piv = (summary_df
           .pivot_table(index=["where","term"],
                        columns="dict",
                        values=["coef","pval","qval","n"]))
    piv.columns = [f"{a}__{b}" for a,b in piv.columns]
    piv = piv.reset_index()
    piv_path = os.path.join(out_dir, "term_level_comparison_DDR_vs_CoreDDR.csv")
    piv.to_csv(piv_path, index=False)
    print(f"\nSaved term-level comparison: {piv_path}")

print(f"\nAll done. Outputs in: {out_dir}")

# %%
def maintenance_multivariable_terms(X_maint, clin_maint, mapping):
    """
    Multi-variable OLS: PFS ~ clinical covariates + all term_scores
    Returns a tidy DataFrame with coef, se, CI, pval for all predictors.
    """
    import statsmodels.api as sm

    # clinical covariates
    cov = prep_clin_covariates(clin_maint)

    # term-level scores (z-scored TU mean per pathway)
    S = term_score_matrix(X_maint, mapping)

    # 표준화 (optional: term 간 scale 차이 줄이기)
    S = (S - S.mean()) / S.std(ddof=0)

    # design matrix
    X = cov.join(S)
    X = sm.add_constant(X, has_constant="add")
    X = X.astype(float)

    y = pd.to_numeric(clin_maint["PFS"], errors="coerce")

    dfm = pd.concat([y.rename("PFS"), X], axis=1).dropna()
    if dfm.shape[0] < 10:
        return pd.DataFrame()

    # fit OLS
    fit = sm.OLS(dfm["PFS"], dfm.drop(columns=["PFS"])).fit()

    # extract results
    summary = fit.summary2().tables[1]  # coef table
    out = summary.rename(columns={
        "Coef.": "coef",
        "Std.Err.": "se",
        "[0.025": "lci",
        "0.975]": "uci",
        "P>|t|": "pval"
    })
    out["n"] = dfm.shape[0]
    out = out.reset_index().rename(columns={"index": "variable"})

    # add q-values
    out["qval"] = fdr_bh(out["pval"])
    return out.sort_values("pval")

def salvage_multivariable_terms(X_salv, clin_salv, mapping, penalized=True):
    import statsmodels.api as sm
    cov = prep_clin_covariates(clin_salv)
    S = term_score_matrix(X_salv, mapping)
    S = (S - S.mean()) / S.std(ddof=0)

    X = cov.join(S)
    X = sm.add_constant(X, has_constant="add").astype(float)
    y = pd.to_numeric(clin_salv["response"], errors="coerce").astype(int)

    dfm = pd.concat([y.rename("response"), X], axis=1).dropna()
    if dfm.shape[0] < 10 or dfm["response"].nunique() < 2:
        return pd.DataFrame()

    try:
        if penalized:
            fit = sm.Logit(dfm["response"], dfm.drop(columns=["response"]))\
                    .fit_regularized(method="l1", alpha=0.1, disp=False)
        else:
            fit = sm.Logit(dfm["response"], dfm.drop(columns=["response"]))\
                    .fit(disp=False, maxiter=200)

        summary = fit.summary2().tables[1]
        out = summary.rename(columns={
            "Coef.": "coef",
            "Std.Err.": "se",
            "[0.025": "lci",
            "0.975]": "uci",
            "P>|z|": "pval"
        })
        out["n"] = dfm.shape[0]
        out = out.reset_index().rename(columns={"index": "variable"})
        out["qval"] = fdr_bh(out["pval"])
        return out.sort_values("pval")
    except Exception as e:
        print("Logit failed:", e)
        return pd.DataFrame()


# =========================
# EXECUTION SCRIPT (Multi-variable)
# =========================

# Align samples
X, clin = align_by_samples(major_df, val_clin)
print(f"Aligned shapes: X {X.shape}, clin {clin.shape}")

# Output directory
out_dir = "/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/covariate_analysis_multivar"
os.makedirs(out_dir, exist_ok=True)

runs = [
    #("CoreDDR", ddr_coregenelist),
    ("DDR full", ddr_genelist),
    #("AR feature", AR_genelist),
]

for dict_name, gene_dict in runs:
    print(f"\n=== Multi-variable regression for: {dict_name} ===")

    # Sample split
    maint_mask = clin["setting"].astype(str).str.contains("maint", case=False, na=False)
    salv_mask  = clin["setting"].astype(str).str.contains("salv", case=False, na=False)

    X_maint, clin_maint = X.loc[:, maint_mask].copy(), clin.loc[maint_mask].copy()
    X_salv,  clin_salv  = X.loc[:, salv_mask].copy(),  clin.loc[salv_mask].copy()

    # Mapping
    mapping = map_gene_dict_to_transcripts(gene_dict, X)

    # Multi-variable regression
    maint_multi = maintenance_multivariable_terms(X_maint, clin_maint, mapping) if not X_maint.empty else pd.DataFrame()
    salv_multi  = salvage_multivariable_terms(X_salv, clin_salv, mapping) if not X_salv.empty else pd.DataFrame()

    # Save results
    if not maint_multi.empty:
        path = os.path.join(out_dir, f"{dict_name}_maintenance_multivariable.csv")
        maint_multi.to_csv(path, index=False)
        print(f"Saved: {path} ({maint_multi.shape[0]} rows)")

    if not salv_multi.empty:
        path = os.path.join(out_dir, f"{dict_name}_salvage_multivariable.csv")
        salv_multi.to_csv(path, index=False)
        print(f"Saved: {path} ({salv_multi.shape[0]} rows)")

    # Quick view
    print("\nTop maintenance variables:")
    print(maint_multi.head(10) if not maint_multi.empty else "empty")
    print("\nTop salvage variables:")
    print(salv_multi.head(10) if not salv_multi.empty else "empty")

# %%
