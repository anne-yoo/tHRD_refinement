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
####^^^ validation cohort check ########
val_df = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_TU.txt', sep='\t', index_col=0)
val_clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_validation_clinicalinfo_new.txt', sep='\t', index_col=0)
val_df = val_df.apply(pd.to_numeric, errors='coerce')

vallist = list(val_df.columns)
val_clin = val_clin.loc[vallist,:]

##### without ongoing ######
# val_clin = val_clin[~((val_clin['ongoing']==1) & (val_clin['type']=='CR'))]
# val_df =  val_df.iloc[:,val_df.columns.isin(val_clin.index.to_list())]
##### without ongoing ######


##### only AR + IR ######
# val_clin = val_clin[val_clin['type']!='CR']
# val_df =  val_df.iloc[:,val_df.columns.isin(val_clin.index.to_list())]
##### only AR + IR ######

#%%
with open('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/group1_new_translist.txt', 'r') as file:
    repair_transcripts = [line.strip() for line in file]
with open('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/group2_new_translist.txt', 'r') as file:
    immune_transcripts = [line.strip() for line in file]
with open('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/group3_new_translist.txt', 'r') as file:
    devel_transcripts = [line.strip() for line in file]

#%%
from statannotations.Annotator import Annotator
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 샘플별 평균 TU 계산
repair_pre = val_df.loc[repair_transcripts].mean(axis=0)
immune_pre = val_df.loc[immune_transcripts].mean(axis=0)
plas_pre = val_df.loc[devel_transcripts].mean(axis=0)

# 2. Group 정보 가져오기 (index 맞춰서)
group_info = val_clin.loc[repair_pre.index, 'type']  # 또는 'Group', 너가 쓰는 label 컬럼 이름으로!

# 3. 최종 데이터프레임 생성
plot_df = pd.DataFrame({
    'Group': group_info,
    'repair_pre': repair_pre,
    'immune_pre': immune_pre,
    'plas_pre': plas_pre
}).reset_index(drop=True)  # 인덱스는 다시 숫자로 초기화


# 튜플 형식 비교쌍
pairs = [('CR', 'AR'), ('CR', 'IR'), ('AR', 'IR')]
#pairs = [('AR', 'IR')]

# 컬러 팔레트 (선택사항)
palette={"AR": "#FFCC29", "IR": "#81B214", "CR": "#409EDD"}
# --------------------------------------------
# 📊 Immune TU boxplot
# --------------------------------------------
plt.figure(figsize=(5, 5))
ax = sns.boxplot(data=plot_df, x='Group', y='immune_pre', palette=palette)
sns.stripplot(data=plot_df, x='Group', y='immune_pre', color='gray', jitter=True, size=5)

annot = Annotator(ax, pairs, data=plot_df, x='Group', y='immune_pre')
annot.configure(test='Mann-Whitney', text_format='star', loc='inside')
annot.apply_and_annotate()

plt.title("Immune Response: pre-treatment")
plt.ylabel("Mean TU")
plt.tight_layout()
plt.show()

# --------------------------------------------
# 📊 Repair TU boxplot
# --------------------------------------------
plt.figure(figsize=(5, 5))
ax = sns.boxplot(data=plot_df, x='Group', y='repair_pre', palette=palette)
sns.stripplot(data=plot_df, x='Group', y='repair_pre', color='gray', jitter=True, size=5)

annot = Annotator(ax, pairs, data=plot_df, x='Group', y='repair_pre')
annot.configure(test='Mann-Whitney', text_format='star', loc='inside')
annot.apply_and_annotate()

plt.title("DNA Repair: pre-treatment")
plt.ylabel("Mean TU")
plt.tight_layout()
plt.show()

# --------------------------------------------
# 📊 devel TU boxplot
# --------------------------------------------
plt.figure(figsize=(5, 5))
ax = sns.boxplot(data=plot_df, x='Group', y='plas_pre', palette=palette)
sns.stripplot(data=plot_df, x='Group', y='plas_pre', color='gray', jitter=True, size=5)

annot = Annotator(ax, pairs, data=plot_df, x='Group', y='plas_pre')
annot.configure(test='Mann-Whitney', text_format='star', loc='inside')
annot.apply_and_annotate()

plt.title("Cell plasticity: pre-treatment")
plt.ylabel("Mean TU")
plt.tight_layout()
plt.show()

#%%



















#%%
#%%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score

X_repair = val_df.loc[repair_transcripts].T
X_immune = val_df.loc[immune_transcripts].T
X_plas = val_df.loc[devel_transcripts].T

X_combined = val_df.loc[repair_transcripts + immune_transcripts + devel_transcripts].T
y = val_clin['type']  # 'CR', 'AR', 'IR'
y_encoded = LabelEncoder().fit_transform(y)

# 💡 함수 정의
def run_model_auc(X, name):
    clf = LogisticRegression(max_iter=1000)
    
    # cross_val_predict는 각 fold마다 예측 확률을 반환
    y_proba = cross_val_predict(clf, X, y_encoded, cv=5, method='predict_proba')[:, 1]
    
    # AUC 계산
    roc = roc_auc_score(y_encoded, y_proba)
    pr = average_precision_score(y_encoded, y_proba)
    
    print(f"{name} model ROC-AUC: {roc:.3f}")
    print(f"{name} model PR-AUC:  {pr:.3f}\n")

# 🔥 실행
run_model_auc(X_repair, "Repair - matrix")
run_model_auc(X_immune, "Immune - matrix")
run_model_auc(X_plas, "Plasticity - matrix")
run_model_auc(X_combined, "Repair+Immune - matrix")

#%%
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder

# 데이터: AR vs IR만 필터
# SVM 모델 (kernel trick 포함!)
clf = SVC(kernel='rbf', probability=True, random_state=42)

# 예측 확률로 AUC 계산
y_proba = cross_val_predict(clf, X_combined, y_encoded, cv=5, method='predict_proba')[:, 1]

# 평가
roc = roc_auc_score(y_encoded, y_proba)
pr = average_precision_score(y_encoded, y_proba)

print(f"SVM (RBF kernel) ROC-AUC: {roc:.3f}")
print(f"SVM (RBF kernel) PR-AUC:  {pr:.3f}")


#%%

# 🚀 XGBoost 모델 함수
def run_xgb_auc(X, name):
    clf = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=4,
        random_state=42
    )

    # fold마다 예측 확률
    y_proba = cross_val_predict(clf, X, y_encoded, cv=5, method='predict_proba')[:, 1]

    # AUC 계산
    roc = roc_auc_score(y_encoded, y_proba)
    pr = average_precision_score(y_encoded, y_proba)

    print(f"{name} (XGBoost) ROC-AUC: {roc:.3f}")
    print(f"{name} (XGBoost) PR-AUC:  {pr:.3f}\n")

# 🎯 실행
run_xgb_auc(X_repair, "Repair - matrix")
run_xgb_auc(X_immune, "Immune - matrix")
run_xgb_auc(X_plas, "Plasticity - matrix")
run_xgb_auc(X_combined, "Repair+Immune - matrix")










#%%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

X_repair = val_df.loc[repair_transcripts].T
X_immune = val_df.loc[immune_transcripts].T
X_combined = val_df.loc[repair_transcripts + immune_transcripts].T

# label
y = val_clin['type']
y_encoded = LabelEncoder().fit_transform(y)

X = X_combined

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    stratify=y_encoded,
    random_state=42
)

# ============================================
# 🔍 3. Hyperparameter Tuning (GridSearchCV)
# ============================================
param_grid = {
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [100, 150],
    'subsample': [0.8, 1.0]
}

xgb_base = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=4,
    random_state=42
)

grid = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1
)
grid.fit(X_train, y_train)

print("🔧 Best params:", grid.best_params_)
print("📈 Best CV accuracy:", grid.best_score_)

# ============================================
# 🎯 4. Feature Selection (Top 30 by importance)
# ============================================
best_model = grid.best_estimator_
importances = best_model.feature_importances_
top_idx = importances.argsort()[::-1][:30]
top_features = X.columns[top_idx]

# ============================================
# 🚀 5. Final Model 학습 + Test 평가
# ============================================
X_train_sel = X_train[top_features]
X_test_sel = X_test[top_features]

final_model = xgb.XGBClassifier(
    **grid.best_params_,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=4,
    random_state=42
)
final_model.fit(X_train_sel, y_train)
y_pred = final_model.predict(X_test_sel)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Final test accuracy (top 30 features): {acc:.3f}")
# %%
