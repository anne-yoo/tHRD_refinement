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
import os
import re
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import plotly.express as px
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.stats.multitest import multipletests
import gseapy as gp


sns.set(font = 'Arial')
sns.set_style("whitegrid")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# %%
##^ step 1 : use HR or +a GO terms for DUT selection from resistance cohort ####
major = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = list(major[major['type']=='major']['gene_ENST'])

TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0)

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
AR_samples = list(set(sampleinfo.loc[(sampleinfo['response']==1),'sample_id']))
IR_samples = list(set(sampleinfo.loc[(sampleinfo['response']==0),'sample_id']))

AR_dut_df = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t')
IR_dut_df = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUT/nonresponder_stable_DUT_Wilcoxon.txt', sep='\t')

AR_dut = set(AR_dut_df[(AR_dut_df['p_value']<0.05) & (np.abs(AR_dut_df['log2FC'])>1.5)]['gene_ENST'])
IR_dut = set(IR_dut_df[(IR_dut_df['p_value']<0.05) & (np.abs(IR_dut_df['log2FC'])>1.5)]['gene_ENST'])

dutlist = list(AR_dut.union(IR_dut))

AR_dut_gene = set(AR_dut_df[(AR_dut_df['p_value']<0.05) & (np.abs(AR_dut_df['log2FC'])>1.5)]['Gene Symbol'])
IR_dut_gene = set(IR_dut_df[(IR_dut_df['p_value']<0.05) & (np.abs(IR_dut_df['log2FC'])>1.5)]['Gene Symbol'])
gDUTlist = list(AR_dut_gene.union(IR_dut_gene))



enr = gp.enrichr(gene_list=gDUTlist, # or "./tests/data/gene_list.txt",
                    gene_sets=['GO_Biological_Process_2018'], # 'Reactome_2022', 'GO_Biological_Process_2018'
                    organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                    outdir=None, # don't write to disk
    )

enrresult = enr.results.sort_values(by=['Adjusted P-value']) 

#%%
file = enrresult
def string_fraction_to_float(fraction_str):
    numerator, denominator = fraction_str.split('/')
    return float(numerator) / float(denominator)

file = file.sort_values('Adjusted P-value')
file['Term'] = file['Term'].str.rsplit(" ",1).str[0]
file = file[file['Adjusted P-value']<0.1]
file['Adjusted P-value'] = -np.log10(file['Adjusted P-value'])
file = file[(file["Term"].str.contains("homologous recombination", case=False)) | (file["Term"].str.contains("regulation of cell cycle", case=False)) | (file["Term"].str.contains("G2/M transition of mitotic cell cycle", case=False)) | (file["Term"].str.contains("cellular response to DNA damage stimulus", case=False))] ## BP2018

#file = file[(file["Term"].str.contains("homologous recombination", case=False))]

HRlist = list(set([gene for sublist in file['Genes'].str.split(';') for gene in sublist]))

input = TU.loc[dutlist,:]
input['gene'] = input.index.str.split("-",1).str[1]
input = input[input['gene'].isin(HRlist)]
input = input.iloc[:,:-1]
input = input.loc[input.index.isin(majorlist),:]
#%%

data = input


# Initialize a dictionary to store all features
features_dict = {}

# Process each DUT (row in DataFrame)
for transcript, row in data.iterrows():
    for sample in set([col[:-4] for col in data.columns if '-atD' in col or '-bfD' in col]):  # Remove '-atD'/'-bfD'
        pre_col = f"{sample}-bfD"
        post_col = f"{sample}-atD"
        
        if pre_col in data.columns and post_col in data.columns:
            pre_TU = row[pre_col]
            post_TU = row[post_col]
            delta_TU = post_TU - pre_TU
            interaction = delta_TU * pre_TU
            
            # Store features with transcript-feature as index and sample as column
            features_dict[(f"{transcript}-pre", sample)] = pre_TU
            features_dict[(f"{transcript}-delta", sample)] = delta_TU
            features_dict[(f"{transcript}-int", sample)] = interaction

# Convert the dictionary into a DataFrame
features_df = pd.DataFrame.from_dict(features_dict, orient='index', columns=["Value"])
features_df.reset_index(inplace=True)

# Split the multi-index (Feature, Sample)
features_df[['Feature', 'Sample']] = pd.DataFrame(features_df['index'].tolist(), index=features_df.index)
features_df.drop(columns='index', inplace=True)

# Pivot the DataFrame so that samples are columns and features are rows
final_df = features_df.pivot(index='Feature', columns='Sample', values='Value')

# Filter for AR and IR samples only
filtered_sample_ids = AR_samples + IR_samples
filtered_df = final_df[filtered_sample_ids]

# %%

# Create the response variable (y)
response_labels = {sample: 1 for sample in AR_samples}  # Label AR as 1
response_labels.update({sample: 0 for sample in IR_samples})  # Label IR as 0

# Align the response labels with the columns in filtered_df
y = np.array([response_labels[sample] for sample in filtered_df.columns])

# Extract features (X)
X = filtered_df.T  # Transpose so samples are rows and features are columns

######^^ change features: pre / int / delta ########
pre_only_features = [f for f in X.columns if '-pre' in f]
X = X[pre_only_features]

#%%
####*** correlation AR vs. IR ######

# Compute correlation
correlations = [np.corrcoef(X[feature], y)[0, 1] for feature in X.columns]
correlation_df = pd.DataFrame({'Feature': X.columns, 'Correlation': correlations})
selected_features = correlation_df.sort_values(by='Correlation', key=abs, ascending=False).head(20)
print("Top Features by Correlation:")
print(selected_features)
#%%
from sklearn.feature_selection import mutual_info_classif

# Compute mutual information
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual Information': mi_scores})
mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)

# Plot top features
plt.figure(figsize=(10, 6))
plt.barh(mi_df['Feature'].head(30), mi_df['Mutual Information'].head(30), color='orange')
plt.xlabel('Mutual Information')
plt.ylabel('Feature')
plt.title('Top Features by Mutual Information')
plt.gca().invert_yaxis()
plt.show()

# Select top features
selected_features = mi_df[mi_df['Mutual Information'] > 0.05]  # Define a threshold



#%%
#######** PCA ###############################
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

# Subset the data for resistance cohort only
X_resistance = X[(y == 1) | (y == 0)]  # AR = 1, IR = 0
y_resistance = y[(y == 1) | (y == 0)]  # AR = 1, IR = 0

# Perform PCA
pca = PCA(n_components=5, random_state=42)  # Keep the first 5 PCs
X_pca = pca.fit_transform(X_resistance)

# Visualize the first 2 principal components
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[y_resistance == 0, 0], X_pca[y_resistance == 0, 1], label='IR', alpha=0.7, c='orange')
plt.scatter(X_pca[y_resistance == 1, 0], X_pca[y_resistance == 1, 1], label='AR', alpha=0.7, c='blue')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Resistance Cohort (AR vs. IR)')
plt.legend()
plt.show()

# Print explained variance ratios
print("Explained Variance Ratios:", pca.explained_variance_ratio_)

# Extract top features contributing to PC1
pc1_loadings = pd.DataFrame(
    {'Feature': X.columns, 'Loading': pca.components_[0]}
).sort_values(by='Loading', key=abs, ascending=False)

# Top 10 features contributing to PC1
top_features_pc1 = pc1_loadings.head(10)
print("Top Features Contributing to PC1:\n", top_features_pc1)


#%%
########** LDA ########################################
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import pandas as pd

# Apply LDA to the resistance cohort
lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X, y)

# Visualize the LDA projection
plt.figure(figsize=(8, 6))
plt.hist(X_lda[y == 0], bins=20, alpha=0.7, label='IR', color='orange')
plt.hist(X_lda[y == 1], bins=20, alpha=0.7, label='AR', color='blue')
plt.xlabel('LDA Projection')
plt.ylabel('Frequency')
plt.title('LDA Projection of Resistance Cohort (AR vs. IR)')
plt.legend()
plt.show()

# Get the LDA coefficients
lda_coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': lda.coef_[0]})
selected_features = lda_coefficients[lda_coefficients['Coefficient'].abs() > 20]
X_selected = X[selected_features['Feature']]

lda2 = LinearDiscriminantAnalysis()
X_lda2 = lda2.fit_transform(X_selected, y)

# Visualize the LDA projection
plt.figure(figsize=(8, 6))
plt.hist(X_lda2[y == 0], bins=20, alpha=0.7, label='IR', color='orange')
plt.hist(X_lda2[y == 1], bins=20, alpha=0.7, label='AR', color='blue')
plt.xlabel('LDA Projection')
plt.ylabel('Frequency')
plt.title('LDA Projection of Resistance Cohort (AR vs. IR)')
plt.legend()
plt.show()

# Get the LDA coefficients
lda_coefficients2 = pd.DataFrame({'Feature': X_selected.columns, 'Coefficient': lda2.coef_[0]})
selected_features_pos = lda_coefficients2[lda_coefficients2['Coefficient'] > 0]
selected_features_neg = lda_coefficients2[lda_coefficients2['Coefficient'] < 0]



#%%
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression, ElasticNetCV
# from sklearn.model_selection import StratifiedKFold
# from collections import Counter
# import numpy as np
# import pandas as pd

# # Define a cross-validation strategy
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # List to store selected features for each fold
# selected_features_list = []

# # Loop over CV folds
# for train_idx, test_idx in skf.split(X, y):
#     # Split the data
#     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#     y_train, y_test = y[train_idx], y[test_idx]
    
#     # Define the pipeline with ElasticNet for stable feature selection
#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),
#         ('elasticnet', ElasticNetCV(
#             l1_ratio=[0.1, 0.5, 0.9],  # Mix of L1 (LASSO) and L2 (Ridge) penalties
#             cv=3,                      # Internal cross-validation for ElasticNet
#             random_state=42
#         ))
#     ])
    
#     # Fit the pipeline
#     pipeline.fit(X_train, y_train)
    
#     # Get the ElasticNet model
#     elasticnet_model = pipeline.named_steps['elasticnet']
    
#     # Extract selected features (non-zero coefficients)
#     selected_features = X.columns[np.abs(elasticnet_model.coef_) > 1e-4]
#     selected_features_list.append(selected_features)

# # Aggregate feature selection across folds
# feature_counter = Counter([feature for features in selected_features_list for feature in features])
# print("Feature Selection Frequency Across Folds:", feature_counter)

# # Define stable features as those selected in >50% of the folds
# stable_features = [feature for feature, count in feature_counter.items() if count > (skf.get_n_splits() / 2)]
# print("\nStable Features Selected Across Folds:", stable_features)

# # Filter the dataset to keep only stable features
# X_stable = X[stable_features]

# # Print final selected dataset shape
# print("\nFinal Dataset Shape (with Stable Features):", X_stable.shape)


# %%
#selected_transcripts = list(set([feature.rsplit('-',1)[0] for feature in selected_features['Feature']]))
#selected_transcripts = list(selected_features['Feature'])
selected_transcripts = list(set([feature.rsplit('-',1)[0] for feature in pre_only_features]))

mean_tu_data = []

for sample in AR_samples + IR_samples:
    pre_values = []
    delta_values = []
    
    for transcript in selected_transcripts:
        # Define column names
        pre_col = f"{sample}-bfD"
        post_col = f"{sample}-atD"
        
        # Check if both columns exist in the data
        if pre_col in data.columns and post_col in data.columns:
            # Retrieve TU values
            pre_value = data.loc[transcript, pre_col]
            post_value = data.loc[transcript, post_col]
            
            # Exclude NaN or invalid values
            if not np.isnan(pre_value) and not np.isnan(post_value):
                pre_values.append(pre_value)
                delta_values.append(post_value - pre_value)
    
    # Verify number of values matches expected transcripts
    if len(pre_values) != len(selected_transcripts):
        print(f"Warning: Transcript count mismatch for {sample}. Expected {len(selected_transcripts)}, got {len(pre_values)}.")
    
    # Calculate mean TU values (if valid values exist)
    mean_pre_tu = np.mean(pre_values) if pre_values else np.nan
    mean_delta_tu = np.mean(delta_values) if delta_values else np.nan
    
    group = "AR" if sample in AR_samples else "IR"
    
    mean_tu_data.append({"Sample": sample, "Group": group, "Mean Pre TU": mean_pre_tu, "Mean Delta TU": mean_delta_tu})

# Convert to DataFrame
mean_tu_df = pd.DataFrame(mean_tu_data)

# Plot Mean Pre TU
plt.figure(figsize=(4,6))
#sns.set_style("whitegrid")
sns.set_theme(style='ticks',palette='pastel')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 12,    # x축 틱 라벨 글꼴 크기
'ytick.labelsize': 12,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 11,
'legend.title_fontsize': 11, # 범례 글꼴 크기
'figure.titlesize': 14    # figure 제목 글꼴 크기
})

ax = sns.boxplot(data=mean_tu_df, x="Group", y="Mean Pre TU", showfliers=False)
plt.title("Mean Pre TU: IR vs. AR")
plt.ylabel("Mean Pre TU")
plt.xlabel("")

sns.stripplot(y='Mean Pre TU', x='Group', data=mean_tu_df, 
            order=['AR','IR'], 
            color='#7D7C7C',  # Color of the points
            size=4,         # Size of the points
            jitter=True,    # Adds some jitter to avoid overlapping
            alpha=0.8,
            ax=ax)

from statannot import add_stat_annotation

add_stat_annotation(ax, data=mean_tu_df, x='Group', y='Mean Pre TU',
                    box_pairs=[('AR','IR')], 
                    order = ['AR','IR'], 
                    comparisons_correction=None,
                    test='Mann-Whitney', text_format='simple', loc='inside', fontsize=12
                    )
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/int_model/onlyint_preTU_boxplot.pdf', bbox_inches='tight', dpi=300)

plt.show()


plt.figure(figsize=(4,6))
#sns.set_style("whitegrid")
sns.set_theme(style='ticks',palette='pastel')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 12,    # x축 틱 라벨 글꼴 크기
'ytick.labelsize': 12,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 11,
'legend.title_fontsize': 11, # 범례 글꼴 크기
'figure.titlesize': 14    # figure 제목 글꼴 크기
})

ax = sns.boxplot(data=mean_tu_df, x="Group", y="Mean Delta TU", showfliers=False)
plt.title("Mean Delta TU: IR vs. AR")
plt.ylabel("Mean Delta TU")
plt.xlabel("")

sns.stripplot(y='Mean Delta TU', x='Group', data=mean_tu_df, 
            order=['AR','IR'], 
            color='#7D7C7C',  # Color of the points
            size=4,         # Size of the points
            jitter=True,    # Adds some jitter to avoid overlapping
            alpha=0.8,
            ax=ax)

from statannot import add_stat_annotation

add_stat_annotation(ax, data=mean_tu_df, x='Group', y='Mean Delta TU',
                    box_pairs=[('AR','IR')], 
                    order = ['AR','IR'], 
                    comparisons_correction=None,
                    test='Mann-Whitney', text_format='simple', loc='inside', fontsize=12
                    )

#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/int_model/onlyint_deltaTU_boxplot.pdf', bbox_inches='tight', dpi=300)
plt.show()



# %%
####^^ Run XGboost for response cohort ##########
trans = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_TU.txt', sep='\t', index_col=0)
trans = trans.drop(['OM/OS'])
trans.loc['group',:] = trans.loc['group',:].astype('float')
trans.loc['group',:] = trans.loc['group',:].astype('int')
trans['Gene Symbol'] = trans.index.str.split("-",1).str[1]

X = trans.loc[selected_transcripts,:]
X = X.iloc[:,:-1]
X = X.T
X = X.apply(pd.to_numeric, errors='coerce') 
valinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_validation_clinicalinfo.txt', sep='\t', index_col=0)
valinfo = valinfo.loc[list(X.index),:]

valinfo['finalresponse'] = 'x'

valinfo.loc[(valinfo['ongoing']==1) | (valinfo['ongoing']==2) | (valinfo['ongoing']==4),'finalresponse'] = 'CR'
valinfo.loc[(valinfo['ongoing']==0) & (valinfo['response']==1), 'finalresponse'] = 'AR'
valinfo.loc[(valinfo['ongoing']==3) & (valinfo['response']==1), 'finalresponse'] = 'AR'
valinfo.loc[(valinfo['response']==0), 'finalresponse'] = 'IR'

y = pd.DataFrame(valinfo['finalresponse'])
#y = pd.DataFrame(df.iloc[-1,:])
y.columns=['class']
y['class'] = y['class'].map({'CR': 1, 'IR': 0, 'AR': 1})
y = y['class']



#%%
##^^^^ XGBoost ################

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Calculate class imbalance ratio
imbalance_ratio = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# Define parameter grid for XGBoost
param_grid_xgb = {
    'n_estimators': [50, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'scale_pos_weight': [1, imbalance_ratio],
    'gamma': [0, 0.1, 1],
    #'max_delta_step': [1, 2]
}

# param_grid_xgb = {
#     'n_estimators': [50],          # Start with fewer trees
#     'max_depth': [3, 5],           # Keep depth small
#     'learning_rate': [0.1],        # Standard learning rate
#     'subsample': [0.8],            # Subsample for regularization
#     'colsample_bytree': [0.8],     # Feature fraction
#     'scale_pos_weight': [1],       # No balancing initially
#     'gamma': [0]                   # No split constraints
# }
# Create the XGBoost model
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='aucpr', n_jobs=1)

# Perform GridSearchCV
grid_search_xgb = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid_xgb,
    scoring='balanced_accuracy',
    cv=5,          # 5-fold cross-validation
    verbose= 1,
    n_jobs = 1
)

# Fit the grid search
grid_search_xgb.fit(X_train, y_train)

# Get best parameters and evaluate on the test set
best_xgb_model = grid_search_xgb.best_estimator_
print("Best Parameters for XGBoost:", grid_search_xgb.best_params_)

# Evaluate the best model on the test set
y_pred_xgb = best_xgb_model.predict(X_test)
print("Classification Report for XGBoost:")
print(classification_report(y_test, y_pred_xgb))

y_proba = best_xgb_model.predict_proba(X_test)[:, 1]  # Probability for class 1 (R)
from sklearn.metrics import precision_recall_curve, f1_score

# Define thresholds
thresholds = np.arange(0.1, 0.9, 0.05)

# Evaluate precision, recall, and F1-score for each threshold
f1_scores = []
for threshold in thresholds:
    y_pred_threshold = (y_proba > threshold).astype(int)
    f1 = f1_score(y_test, y_pred_threshold)
    f1_scores.append(f1)

# Find the best threshold
best_threshold = thresholds[np.argmax(f1_scores)]

y_pred_optimal = (y_proba > best_threshold).astype(int)

# Evaluate the classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_optimal))

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds_pr = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(thresholds_pr, precision[:-1], label='Precision', color='blue')
plt.plot(thresholds_pr, recall[:-1], label='Recall', color='orange')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision-Recall vs. Threshold')
plt.legend()
plt.grid()
plt.show()

from sklearn.metrics import classification_report, confusion_matrix

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_optimal)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['NR', 'R'], yticklabels=['NR', 'R'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Get feature importance from the best model
importance = best_xgb_model.feature_importances_
features = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
features = features.sort_values(by='Importance', ascending=False)

# Plot top features
plt.figure(figsize=(10, 6))
plt.barh(features['Feature'][:20], features['Importance'][:20], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 20 Feature Importances')
plt.gca().invert_yaxis()
plt.show()

#%%
from sklearn.metrics import precision_recall_curve, average_precision_score

# Compute Precision-Recall curve and PR-AUC
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)

# Plot Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f"PR curve (AUC = {pr_auc:.2f})")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="upper right")
plt.grid()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f"PR curve (AUC = {pr_auc:.2f})")
for i, threshold in enumerate(thresholds):
    if i % 10 == 0:  # Plot every 10th threshold
        plt.annotate(f"{threshold:.2f}", (recall[i], precision[i]), textcoords="offset points", xytext=(-15, 5), ha='center')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve with Thresholds')
plt.legend(loc="upper right")
plt.grid()
plt.show()



from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt

# Predict probabilities for the test set
y_proba = best_xgb_model.predict_proba(X_test)[:, 1]  # Probability for the positive class (R)

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)  # False Positive Rate, True Positive Rate
roc_auc = auc(fpr, tpr)  # Calculate AUC

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid()
plt.show()
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2, label='Random Guess')
for i, threshold in enumerate(thresholds):
    if i % 10 == 0:  # Plot every 10th threshold
        plt.annotate(f"{threshold:.2f}", (fpr[i], tpr[i]), textcoords="offset points", xytext=(-15, 5), ha='center')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Thresholds')
plt.legend(loc="lower right")
plt.grid()
plt.show()



#%%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

from imblearn.combine import SMOTETomek
from collections import Counter

# Apply SMOTE-Tomek
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

print("Class distribution after SMOTE-Tomek:", Counter(y_resampled))

# Define parameter grid for Logistic Regression
param_grid_lr = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'saga'],
}

# Create Logistic Regression model
lr_model = LogisticRegression(max_iter=1000, random_state=42)

# Perform GridSearchCV
grid_search_lr = GridSearchCV(
    estimator=lr_model,
    param_grid=param_grid_lr,
    scoring='f1',  # Focus on F1 score for class 0
    cv=5,
    verbose=1,
)

# Fit on the resampled data
grid_search_lr.fit(X_resampled, y_resampled)

# Evaluate on test set
best_lr_model = grid_search_lr.best_estimator_
y_pred_lr = best_lr_model.predict(X_test)
print("Classification Report for Logistic Regression (SMOTE-Tomek):")
print(classification_report(y_test, y_pred_lr))



#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Define parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced_subsample'],  # Subsample-balanced class weights
}

# Create Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Perform GridSearchCV
grid_search_rf = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid_rf,
    scoring='f1',  # Focus on F1 score for class 0
    cv=5,
    verbose=1,
)

# Fit on the resampled data
grid_search_rf.fit(X_resampled, y_resampled)

# Evaluate on test set
best_rf_model = grid_search_rf.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)
print("Classification Report for Random Forest (SMOTE-Tomek):")
print(classification_report(y_test, y_pred_rf))





#%%
###^^^ 3 class classifier ##########################
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from collections import Counter

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Calculate class weights
class_counts = Counter(y_resampled)
class_weights = {label: len(y_resampled) / count for label, count in class_counts.items()}

# Train XGBoost with class weights
xgb_model = XGBClassifier(
    random_state=42,
    scale_pos_weight=class_weights[0] / class_weights[1]  # Ratio of class imbalance
)
xgb_model.fit(X_resampled, y_resampled)

# Predict and evaluate
y_pred = xgb_model.predict(X_test)
print(classification_report(y_test, y_pred))


#%%
####^^^ binary classifier #############################
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, make_scorer, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np
from imblearn.combine import SMOTETomek

smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

# Compute sample weights for the unbalanced dataset
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

# Define the XGBoost model
xgb_model = XGBClassifier(
    objective='binary:logistic',  # For binary classification
    eval_metric='logloss',       # Evaluation metric
    use_label_encoder=False,     # Suppress label encoding warning
    random_state=42              # Reproducibility
)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Define the grid search with cross-validation
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='accuracy',          # Scoring metric
    cv=5,                        # 5-fold cross-validation
    verbose=1,                   
)

# Fit the grid search with sample weights
grid_search.fit(X_resampled, y_resampled)

# Get the best model and parameters
best_xgb_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Evaluate the model on the test set
y_pred = best_xgb_model.predict(X_test)
classification_report_output = classification_report(y_test, y_pred)
print("\nClassification Report on Test Set:")
print(classification_report_output)

# Optional: Cross-validation performance for each parameter set
cv_results = grid_search.cv_results_
for mean_score, params in zip(cv_results["mean_test_score"], cv_results["params"]):
    print(f"Mean CV Accuracy: {mean_score:.4f} for {params}")

# %%
