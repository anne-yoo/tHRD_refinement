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

####^^^ validation cohort check ########
val_df = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_TU.txt', sep='\t', index_col=0)
val_gene = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_gene_TPM.txt', sep='\t', index_col=0)
val_clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_validation_clinicalinfo_new.txt', sep='\t', index_col=0)
val_df = val_df.apply(pd.to_numeric, errors='coerce')

vallist = list(val_df.columns)
val_clin = val_clin.loc[vallist,:]

genesymbol =  pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM_symbol.txt', sep='\t', index_col=0)
genesymbol = genesymbol.iloc[:,-1]

##### only CR + IR ######
val_clin = val_clin[val_clin['type']!='AR']
val_df =  val_df.iloc[:-2,val_df.columns.isin(val_clin.index.to_list())]
val_gene =  val_gene.iloc[:,val_gene.columns.isin(val_clin.index.to_list())]
val_gene = pd.merge(val_gene,genesymbol,left_index=True,right_index=True)
##### only CR + IR ######

# %%
####^^^ HG features RF #######################
import gseapy as gp
go_results = gp.get_library("GO_Biological_Process_2021", organism="Human")
repairgenes = go_results['double-strand break repair (GO:0006302)']
HG_features = pd.read_csv('/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/merged_TU/whole_TU/XGBoost/old_model_featrues.txt',sep='\t', index_col=0)
HG_features = list(HG_features.index)

X = val_df.loc[HG_features,:]
X = X.T
y = val_clin['type'].replace({'CR': 1, 'IR': 0})

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    roc_curve, auc, accuracy_score,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def run_rf_with_eval(X, y, random_state=42):
    # 1. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )

    # 2. RandomForest + GridSearchCV
    rf = RandomForestClassifier(class_weight='balanced', random_state=random_state)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
    }

    grid = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=4, verbose=1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # 3. Predict
    y_pred = best_model.predict(X_test)
    y_score = best_model.predict_proba(X_test)[:, 1]  # probability for class 1 (Responder)

    # 4. Metrics
    acc = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test, y_score)
    pr_auc = average_precision_score(y_test, y_score)

    # 5. Plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    axs[0].plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}', color='darkorange')
    axs[0].plot([0, 1], [0, 1], 'k--', label='Baseline')
    axs[0].set_title('ROC Curve')
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    axs[0].legend(loc='lower right')

    # PR
    axs[1].plot(recall, precision, label=f'AUPRC = {pr_auc:.3f}', color='darkblue')
    baseline = y_test.mean()
    axs[1].hlines(baseline, 0, 1, color='gray', linestyle='--', label='Baseline')
    axs[1].set_title('Precision-Recall Curve')
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')
    axs[1].set_ylim(0, 1)  # 항상 0부터 시작
    axs[1].legend(loc='upper right')

    plt.tight_layout()
    #plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/HGfeature_RF_ROCPRcurve.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # 6. 결과 출력
    print(f"Best Params: {grid.best_params_}")
    print(f"Test Accuracy: {acc:.3f}")
    print(f"Test ROC AUC: {roc_auc:.3f}")
    print(f"Test PR AUC: {pr_auc:.3f}")

    
    return best_model

model = run_rf_with_eval(X, y)

#%%
####^^^^ DNA repair #####################
HRRgenes = ['GEN1', 'BARD1', 'RAD50', 'SHFM1', 'XRCC2', 'NBN', 'MUS81', 'MRE11A', 'RAD52', 'BRCA2', 'XRCC3', 'RAD51C', 'RAD51D', 'TP53BP1', 'BLM', 'SLX1A', 'PALB2', 'TOP3A', 'BRCA1', 'EME1', 'BRIP1', 'RBBP8']
val_df2 = val_df.copy()
val_df2['gene'] = val_df2.index.str.split("-", n=1).str[-1]

import gseapy as gp
go_results = gp.get_library("GO_Biological_Process_2021", organism="Human")
#repairgenes = go_results['double-strand break repair (GO:0006302)']
repairgenes = go_results['DNA repair (GO:0006281)']

majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = majorminor[majorminor['type']=='major']['gene_ENST'].to_list()
minorlist = majorminor[majorminor['type']=='minor']['gene_ENST'].to_list()

stable_result = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/stable_DUT_FC.txt', sep='\t', index_col=0)
dutlist = list(stable_result[(stable_result['p_value']<0.05) & (stable_result['log2FC']>1.5)].index)

#%%
selected = val_df2.loc[val_df2.index.isin(minorlist) & val_df2['gene'].isin(repairgenes) & val_df2.index.isin(dutlist)].drop(columns='gene')
#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt

def run_rf_with_rfecv(X, y, random_state=42):
    # 1. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )

    # 2. RFECV
    base_estimator = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=random_state)
    selector = RFECV(
        estimator=base_estimator,
        step=0.1,
        cv=5,
        scoring='accuracy',
        n_jobs=4,
        verbose=1
    )
    selector.fit(X_train, y_train)

    X_train_sel = selector.transform(X_train)
    X_test_sel = selector.transform(X_test)

    print(f"RFECV selected {X_train_sel.shape[1]} features")

    # 3. GridSearchCV
    rf = RandomForestClassifier(class_weight='balanced', random_state=random_state)
    # param_grid = {
    #     'n_estimators': [100, 200],
    #     'max_depth': [3, 5, 7],  # 깊이 제한
    #     'min_samples_split': [2, 5],
    #     'min_samples_leaf': [1, 3, 5],
    #     'max_features': [None]# 리프 노드 최소 샘플 수 제한
    # }
    param_grid = {
    'n_estimators': [25, 50, 75, 100],  # Reducing the maximum number of trees
    'max_depth': [4, 6, 8],  # Further limit tree depth to prevent complex fits
    'min_samples_split': [5, 10, 20, 30],  # More restrictive splitting
    'min_samples_leaf': [5, 10, 15],  # Increase minimum samples in a leaf
    'max_features': ['sqrt', 'log2', None]  # Limit features considered at each split
}


    grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=4, verbose=1)
    grid.fit(X_train_sel, y_train)
    best_model = grid.best_estimator_

    # 4. Predict on train and test
    y_pred_test = best_model.predict(X_test_sel)
    y_score_test = best_model.predict_proba(X_test_sel)[:, 1]

    y_pred_train = best_model.predict(X_train_sel)
    y_score_train = best_model.predict_proba(X_train_sel)[:, 1]

    # 5. Metrics
    # Test
    acc_test = accuracy_score(y_test, y_pred_test)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_score_test)
    roc_auc_test = auc(fpr_test, tpr_test)

    precision_test, recall_test, _ = precision_recall_curve(y_test, y_score_test)
    pr_auc_test = average_precision_score(y_test, y_score_test)

    # Train
    acc_train = accuracy_score(y_train, y_pred_train)
    fpr_train, tpr_train, _ = roc_curve(y_train, y_score_train)
    roc_auc_train = auc(fpr_train, tpr_train)

    precision_train, recall_train, _ = precision_recall_curve(y_train, y_score_train)
    pr_auc_train = average_precision_score(y_train, y_score_train)

    # 6. Plot ROC + PR
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    axs[0].plot(fpr_test, tpr_test, label=f'Test AUC = {roc_auc_test:.3f}', color='#FF8D22')
    axs[0].plot(fpr_train, tpr_train, label=f'Train AUC = {roc_auc_train:.3f}', color='#FFBA79', linestyle='--')
    axs[0].plot([0, 1], [0, 1], 'k--', label='Baseline')
    axs[0].set_title('ROC Curve')
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    axs[0].legend(loc='lower right')

    # PR
    axs[1].plot(recall_test, precision_test, label=f'Test AUPRC = {pr_auc_test:.3f}', color='#4DAFFF')
    axs[1].plot(recall_train, precision_train, label=f'Train AUPRC = {pr_auc_train:.3f}', color='#9ED3FF', linestyle='--')
    baseline = y_test.mean()
    axs[1].hlines(baseline, 0, 1, color='gray', linestyle='--', label='Baseline')
    axs[1].set_ylim(-0.04, 1.04)
    axs[1].set_title('Precision-Recall Curve')
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')
    axs[1].legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    # 7. 결과 출력
    print(f"\nBest Params: {grid.best_params_}")
    print(f"[Test]  Accuracy: {acc_test:.3f} | ROC AUC: {roc_auc_test:.3f} | PR AUC: {pr_auc_test:.3f}")
    print(f"[Train] Accuracy: {acc_train:.3f} | ROC AUC: {roc_auc_train:.3f} | PR AUC: {pr_auc_train:.3f}")
    
    cm = confusion_matrix(y_test, y_pred_test)
    print("Confusion Matrix:")
    print(cm)
    return best_model, selector


X = selected.T
model, selector = run_rf_with_rfecv(X, y)


#%%
##^^^^ Regression ###################
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    accuracy_score, confusion_matrix
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def run_l1_logistic(X, y, random_state=42):
    # 1. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )

    # 2. L1-regularized LogisticRegression with CV
    model = LogisticRegressionCV(
        Cs=10,  # number of inverse regularization strengths to try
        cv=5,
        penalty='l1',
        solver='saga',
        class_weight='balanced',
        scoring='f1',
        random_state=random_state,
        max_iter=5000,
        n_jobs=4,
        refit=True
    )
    model.fit(X_train, y_train)

    # 3. Predict
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    # 4. Metrics (test)
    acc = accuracy_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test, y_score)
    pr_auc = average_precision_score(y_test, y_score)

    # 5. Plot
    plt.figure(figsize=(12, 5))

    # ROC
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}', color='darkorange')
    plt.plot([0, 1], [0, 1], 'k--', label='Baseline')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    # PR
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.3f}', color='darkblue')
    baseline = y_test.mean()
    plt.hlines(baseline, 0, 1, color='gray', linestyle='--', label='Baseline')
    plt.ylim(-0.04, 1.04)
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 6. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    print(f"\nTest Accuracy: {acc:.3f}")
    print(f"Test ROC AUC: {roc_auc:.3f}")
    print(f"Test PR AUC: {pr_auc:.3f}")

    # 7. Feature importance
    coef = pd.Series(model.coef_[0], index=X.columns)
    nonzero_coef = coef[coef != 0].sort_values(key=abs, ascending=False)
    print(f"\n🔍 Selected {len(nonzero_coef)} non-zero features:")
    print(nonzero_coef)

    return model, nonzero_coef
model, important_features = run_l1_logistic(X, y)


















# %%
#####^ HG Model ROC/PRAUC curve #######

pred_proba = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/HGmodel/116_HGmodel_proba.txt', sep='\t')
val_clin['group'] = val_clin['type'].replace({'CR': 'R', 'IR': 'NR'})

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, accuracy_score,
    precision_recall_curve, average_precision_score
)

def plot_auc_curves(pred_proba, val_clin):
    # 정답값 (index: sample ID, value: 0 or 1)
    y_true_series = val_clin['group'].map({'NR': 0, 'R': 1})
    y_true_series.index.name = 'id'  # join을 위해 이름 통일

    # pred_proba와 y_true를 'id' 기준으로 join
    merged = pred_proba.set_index('id').join(y_true_series, how='inner')
    merged = merged.rename(columns={'group': 'y_true'})  # rename for clarity

    # 예측값과 정답값
    y_score = merged['pred_HRD']
    y_true = merged['y_true']

    # ROC Curve
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, accuracy_score

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    # Accuracy at 0.5 cutoff
    y_pred = (y_score >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)

    # Plot
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # ROC
    axs[0].plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}', color='darkorange')
    axs[0].plot([0, 1], [0, 1], 'k--', label='Baseline')
    axs[0].set_title('ROC Curve')
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    axs[0].legend(loc='lower right')

    # PR
    axs[1].plot(recall, precision, label=f'AUPRC = {pr_auc:.3f}', color='darkblue')
    baseline = y_true.mean()
    axs[1].hlines(baseline, 0, 1, color='gray', linestyle='--', label='Baseline')
    axs[1].set_title('Precision-Recall Curve')
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')
    axs[1].legend(loc='upper right')
    axs[1].set_ylim(bottom =-0.02)

    plt.tight_layout()
    #plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/valfigures/HGmodel_ROCPRcurve.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # 출력
    print(f"Number of samples used: {len(y_true)}")
    print(f"Accuracy: {acc:.3f}")
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"PR AUC: {pr_auc:.3f}")

plot_auc_curves(pred_proba, val_clin)

# %%
