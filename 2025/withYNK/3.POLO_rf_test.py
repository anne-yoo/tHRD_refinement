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
# %%
val_df = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/POLO/83_POLO_TU.txt', sep='\t', index_col=0)
val_gene = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/POLO/83_POLO_gene_exp_TPM.txt', sep='\t', index_col=0)
val_clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2025clinical/83_POLO_clinicalinfo.txt', sep='\t', index_col=0)

val_df = val_df.apply(pd.to_numeric, errors='coerce')
genesymbol =  pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM_symbol.txt', sep='\t', index_col=0)
genesymbol = pd.DataFrame(genesymbol.iloc[:,-1])
genesymbol.columns = ['genesymbol']

vallist = list(val_clin.index)
val_df = val_df.loc[:,vallist]
val_gene = val_gene.loc[genesymbol.index,vallist]
val_gene.index = genesymbol.loc[val_gene.index,'genesymbol']

val_clin['drug'] = 'Niraparib'

majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = majorminor[majorminor['type']=='major']['gene_ENST'].to_list()

from scipy.stats import spearmanr

olaparib_list = val_clin[val_clin['drug']=='Olaparib'].index.to_list()
niraparib_list = val_clin[val_clin['drug']=='Niraparib'].index.to_list()

main_list = val_clin[val_clin['OM/OS']=='maintenance'].index.to_list()
sal_list = val_clin[val_clin['OM/OS']=='salvage'].index.to_list()

main_gene = val_gene.loc[:,main_list]
sal_gene = val_gene.loc[:,sal_list]

main_tu = val_df.loc[:,main_list]
sal_tu = val_df.loc[:,sal_list]

main_clin = val_clin.loc[main_list,:]
sal_clin = val_clin.loc[sal_list,:]

# %%
transcripts = [
    "ENST00000242576.2-UNG",
    "ENST00000393094.2-CUL5",
    "ENST00000311895.7-ERCC4",
    "ENST00000355739.4-ERCC5",
    "ENST00000441310.2-PMS1",
    "ENST00000389301.3-FANCA",
    "ENST00000267430.5-FANCM",
    "ENST00000260947.4-BARD1",
    "ENST00000259008.2-BRIP1",
    "ENST00000308110.4-MUS81",
    "ENST00000553264.1-XRCC3",
    "ENST00000367505.2-SHPRH",
    "ENST00000350721.4-ATR",
    "ENST00000589866.1-RNMT",
    "ENST00000262173.3-RNMT",
    "ENST00000424901.1-AKT2",
    "ENST00000397124.1-MLST8",
    "MSTRG.73953.11-PIK3CB",
    "MSTRG.85941.2-RICTOR",
    "ENST00000225577.4-RPS6KB1",
    "ENST00000265724.3-ABCB1",
    "ENST00000370449.4-ABCC2",
    "ENST00000298139.5-WRN"
]
inputdf = val_df.loc[transcripts,:]

# %%
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# -------------------------
# 준비
# -------------------------
X = inputdf.T  # sample × transcript
y_reg = val_clin['PFS'].values
y_clf = val_clin['response'].values

# -------------------------
# (1) Random Forest Regression (PFS)
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.3, random_state=42)

reg = RandomForestRegressor(n_estimators=500, random_state=42)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

# Regression 평가
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("=== Random Forest Regression (PFS) ===")
print(f"R^2 Score: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")

# -------------------------
# (2) Random Forest Classification (response)
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_clf, test_size=0.3, random_state=42, stratify=y_clf)

clf = RandomForestClassifier(n_estimators=500, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:,1]

# Classification 평가
rocauc = roc_auc_score(y_test, y_prob)
prauc = average_precision_score(y_test, y_prob)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== Random Forest Classification (response) ===")
print(f"ROC AUC: {rocauc:.3f}")
print(f"PR AUC: {prauc:.3f}")
print(f"Accuracy: {acc:.3f}")
print(f"F1 Score: {f1:.3f}")

# %%
