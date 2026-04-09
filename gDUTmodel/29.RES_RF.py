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


sns.set(font = 'Arial')
sns.set_style("whitegrid")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# %%
idmatch = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_readcount_symbol.txt', sep='\t', index_col=0)
idmatch = pd.DataFrame(idmatch['Gene Symbol'])

sample = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_TU.txt', sep='\t', index_col=0)
input = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_gene_TPM.txt', sep='\t', index_col=0)

input = input[input.apply(lambda x: (x != 0).sum(), axis=1) >= 70]

input.loc['group',:] = sample.loc['group',:]

input.loc['group',:] = input.loc['group',:].astype('float')
input.loc['group',:] = input.loc['group',:].astype('int')

input['Gene Symbol'] = idmatch

#%%
input.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/116_validation_gene_TPM_symbol_group.txt', sep='\t', index=True)

# %%



#%%
Res = input.loc[:,input.iloc[-1,:]==1]
Res = Res.drop(['group'])
Nonres = input.loc[:,input.iloc[-1,:]==0]
Nonres = Nonres.drop(['group'])

# %%
import pandas as pd
from scipy.stats import mannwhitneyu
import numpy as np
from statsmodels.stats.multitest import multipletests

# Assuming your dataframes are named 'Res' and 'Nonres'

# Calculate p-values and log2 fold changes
p_values = []
log2_fold_changes = []
genes = Res.index

Res = Res.apply(pd.to_numeric, errors='coerce')
Nonres = Nonres.apply(pd.to_numeric, errors='coerce')

for gene in genes:
    # Get the expression values for the transcript in both dataframes
    values1 = Res.loc[gene].values
    values2 = Nonres.loc[gene].values
    
    # Perform Mann-Whitney U test
    stat, p_value = mannwhitneyu(values1, values2, alternative='two-sided')
    p_values.append(p_value)
    
    # Calculate log2 fold change
    mean1 = np.mean(values1)
    mean2 = np.mean(values2)
    log2_fc = np.log2(mean1) - np.log2(mean2)  # Res/Nonres
    log2_fold_changes.append(log2_fc)

# Adjust p-values using Benjamini-Hochberg method (False Discovery Rate correction)
p_adjusted = multipletests(p_values, method='fdr_bh')[1]

# Create a results DataFrame
results = pd.DataFrame({
    'Gene_id': genes,
    'p_value': p_values,
    'p-adjusted': p_adjusted,
    'log2FC': log2_fold_changes,
    'Gene Symbol': list(input['Gene Symbol'])[:-1]
})

results

# Display the results
print(results)
results.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/MW_DEGresult_FC.txt', sep='\t', index=False)


# %%
#########^ DUT #############
trans = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_TU.txt', sep='\t', index_col=0)
trans = trans.drop(['OM/OS'])

last_row = trans.iloc[-1:]  # Selects the last row as a DataFrame
data_to_filter = trans.iloc[:-1]  # Selects all rows except the last one

minimum_samples = int(data_to_filter.shape[1] * 0.4) ########## 40% threshold

# Apply filtering based on non-zero values treated as "1"
filtered_data = data_to_filter[data_to_filter.apply(lambda x: (x != 0).sum(), axis=1) >= minimum_samples]

# Concatenate the filtered data with the last row
trans = pd.concat([filtered_data, last_row])

trans.loc['group',:] = trans.loc['group',:].astype('float')
trans.loc['group',:] = trans.loc['group',:].astype('int')
trans['Gene Symbol'] = trans.index.str.split("-",1).str[1]

#%%
#trans.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/116_validation_TU_group.txt', sep='\t', index=True)

stable = list(results[results['p_value']>=0.05]['Gene Symbol'])
variable = list(results[results['p_value']<0.05]['Gene Symbol'])


types = [stable,variable]
namelist = ['stable','variable']

for i in range(2):
    Res = trans.loc[:,trans.iloc[-1,:]==1]
    Res = Res.drop(['group'])
    Nonres = trans.loc[:,trans.iloc[-1,:]==0]
    Nonres = Nonres.drop(['group'])
    Res['Gene Symbol'] = list(Res.index.str.split("-",1).str[1])
    Nonres['Gene Symbol'] = list(Nonres.index.str.split("-",1).str[1])
    
    Res = Res[Res['Gene Symbol'].isin(types[i])]
    Res = Res.drop(['Gene Symbol'], axis=1)
    Res = Res.apply(pd.to_numeric, errors='coerce')
    
    Nonres = Nonres[Nonres['Gene Symbol'].isin(types[i])]
    finalgenelist = list(Nonres['Gene Symbol'])
    Nonres = Nonres.drop(['Gene Symbol'], axis=1)
    Nonres = Nonres.apply(pd.to_numeric, errors='coerce')
    
    
    transcripts = Res.index
    p_values = []
    log2_fold_changes = []
    
    for t in transcripts:
    # Get the expression values for the transcript in both dataframes
        values1 = Res.loc[t].values
        values2 = Nonres.loc[t].values
        
        # Perform Mann-Whitney U test
        stat, p_value = mannwhitneyu(values1, values2, alternative='two-sided')
        p_values.append(p_value)
        
        # Calculate log2 fold change
        mean1 = np.mean(values1)
        mean2 = np.mean(values2)
        log2_fc = np.log2(mean1) - np.log2(mean2)  # R/NR
        log2_fold_changes.append(log2_fc)

    # Adjust p-values using Benjamini-Hochberg method (False Discovery Rate correction)
    p_adjusted = multipletests(p_values, method='fdr_bh')[1]

    # Create a results DataFrame
    results = pd.DataFrame({
        'transcript_id': transcripts,
        'p_value': p_values,
        'p-adjusted': p_adjusted,
        'log2FC': log2_fold_changes,
        'Gene Symbol': finalgenelist
    })
    
    print(results)
    results.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/MW_DUTresult_FC_'+namelist[i]+'.txt', sep='\t', index=False)

# %%
##^^^^^^^ gDUT vs. DEG ######################
dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/MW_DUTresult_FC_stable.txt', sep='\t', index_col=0)
deg = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/MW_DEGresult_FC.txt', sep='\t', index_col=0)

deglist = set(deg.loc[(deg['p_value']<0.05) & (np.abs(deg['log2FC'])>0.4)]['Gene Symbol'])
gDUTlist = set(dut.loc[(dut['p_value']<0.05) & (np.abs(dut['log2FC'])>1)]['Gene Symbol'])

from matplotlib_venn import venn2

plt.figure(figsize=(6,6))
sns.set_style("white")
vd2 = venn2([deglist, gDUTlist],set_labels=('DEG', 'gDUT'),set_colors=('#F8CF6A','#74B2D4'), alpha=0.8)

###^^^^^^^^^^^^^^^ GO enrichment ####################3

import gseapy as gp

glist = list(gDUTlist)
enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                gene_sets=['GO_Biological_Process_2018',], 
                #gene_sets=['GO_Biological_Process_2021',], 
                organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                outdir=None, # don't write to disk
)

enrresult = enr.results.sort_values(by=['Adjusted P-value']) 

## file save
#enrresult.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/BP2021_gDUT_GOenrichment.txt', index=False, sep='\t')

#%%

#%%
file = enrresult
#file['Term'] = file['Term'].str.rsplit(" ",1).str[0]
file = file[file['Adjusted P-value']<0.1]
file['FDR'] = -np.log10(file['Adjusted P-value'])
file = file[(file["Term"].str.contains("homologous recombination", case=False)) | (file["Term"].str.contains("DNA Damage", case=False)) | (file["Term"].str.contains("cell cycle", case=False)) | (file["Term"].str.contains("Signaling by WNT in cancer", case=False)) | (file["Term"].str.contains("PI3K/AKT Signaling in Cancer", case=False)) | (file["Term"].str.contains("double strand break", case=False))| (file["Term"].str.contains("DNA repair", case=False))]

#%%
#file = enrresult
#fdrcut = file[file['Adjusted P-value']<0.1]
#fdrcut['FDR'] = -np.log10(fdrcut['Adjusted P-value'])
fdrcut = file
terms = fdrcut['Term']
fdr_values = fdrcut['FDR']

plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 13,    # x축 틱 라벨 글꼴 크기

'ytick.labelsize': 13,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 13,
'legend.title_fontsize': 13, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})
sns.set_style("white")
# Create a horizontal bar plot
plt.figure(figsize=(3,7))  # Making the figure tall and narrow
bars = plt.barh(range(len(terms)), fdr_values, color='#859F3D')

# Set y-axis labels and position them on the right
plt.yticks(range(len(terms)), terms, ha='left')
plt.gca().yaxis.tick_right()  # Move y-axis ticks to the right
plt.yticks(fontsize=16)

# Add labels and title
plt.xlabel('-log10(FDR)')
plt.gca().invert_yaxis()  # Invert y-axis to have the first term on top

plt.tight_layout()
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/BP2018_FDR10_barplot.pdf', dpi=300, format='pdf', bbox_inches='tight', pad_inches=0.1)

plt.show()


# %%
###^^^ PARPi-related enriched GO terms ######
# parplist = ['chromatin remodeling (GO:0006338)', 'cell cycle G2/M phase transition (GO:0044839)', 
#             'regulation of transcription, DNA-templated (GO:0006355)', 
#             #'regulation of gene expression (GO:0010468)', 
#             'negative regulation of gene expression (GO:0010629)',
#             'regulation of mRNA stability (GO:0043488)'] ## negative ..?
parplist = file['Term']
parplist = enrresult[enrresult['Term'].isin(parplist)]
parpgene = list(set([gene for sublist in parplist['Genes'].str.split(';') for gene in sublist]))


# %%
#####^^^^^^^^^^^^^^^^^ Random Forest with PARPi-related enriched GO genes ####################################################

dut_filtered = dut.loc[(dut['p_value']<0.05) & (np.abs(dut['log2FC'])>1)]
featurelist = dut_filtered[dut_filtered['Gene Symbol'].isin(parpgene)].index.to_list()

y = np.array(trans.iloc[-1,:-1])
y = y.astype(int)
X = trans.loc[featurelist,:]
X = X.iloc[:,:-1]
X = X.T
#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Parameter grid for Grid Search
# param_grid = {
#     'n_estimators': [50, 100, 150],  # Fewer trees to reduce complexity
#     'max_depth': [5, 10, 15],  # Restrict maximum depth of trees to prevent overfitting
#     'min_samples_split': [5, 10, 15],  # Increase minimum samples required to split a node
#     'min_samples_leaf': [2, 4, 6],  # Increase minimum samples required at each leaf node
#     'max_features': ['sqrt', 'log2']  # Control the number of features considered at each split
# }
param_grid = {
    'n_estimators': [50, 75, 100],  # Reducing the maximum number of trees
    'max_depth': [4, 6, 8],  # Further limit tree depth to prevent complex fits
    'min_samples_split': [10, 20, 30],  # More restrictive splitting
    'min_samples_leaf': [7, 10, 15],  # Increase minimum samples in a leaf
    'max_features': ['sqrt', 'log2']  # Limit features considered at each split
}

# param_grid = {
#     'n_estimators': [50, 100, 150],
#     'max_depth': [5, 10,], #15
#     'min_samples_split': [10, 15, 20], #5
#     'min_samples_leaf': [4, 6], #2
#     'max_features': ['sqrt', 'log2']
# }

# Initialize Random Forest model
rf = RandomForestClassifier(random_state=42,class_weight='balanced')

# Stratified k-fold cross-validation
cv = StratifiedKFold(n_splits=5)

# Grid Search with cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Best parameters from Grid Search
best_rf = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE

rfecv = RFECV(estimator=best_rf, step=1, cv=StratifiedKFold(5), scoring='accuracy')
rfecv.fit(X_train, y_train)

print("Optimal number of features:", rfecv.n_features_)
selected_features = X_train.columns[rfecv.support_]
#print("Selected Features:", selected_features)

# Fit model with selected features
X_train_rfe = X_train[selected_features]
X_test_rfe = X_test[selected_features]
best_rf.fit(X_train_rfe, y_train)


from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc, RocCurveDisplay, PrecisionRecallDisplay

# Predictions and probabilities
y_train_pred = best_rf.predict(X_train_rfe)
y_test_pred = best_rf.predict(X_test_rfe)
y_train_prob = best_rf.predict_proba(X_train_rfe)[:, 1]
y_test_prob = best_rf.predict_proba(X_test_rfe)[:, 1]

# Metrics calculations
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)
train_roc_auc = roc_auc_score(y_train, y_train_prob)
test_roc_auc = roc_auc_score(y_test, y_test_prob)

# Precision-Recall AUC
precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_prob)
precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_prob)
train_pr_auc = auc(recall_train, precision_train)
test_pr_auc = auc(recall_test, precision_test)

# Feature importance (based on RFE-selected features)
feature_importances = pd.DataFrame({
    'Feature': selected_features,  # Column names of X #X.columns
    'Importance': best_rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Print metrics
print("\nTrain Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
print("Train ROC-AUC:", train_roc_auc)
print("Test ROC-AUC:", test_roc_auc)
print("Train PR-AUC:", train_pr_auc)
print("Test PR-AUC:", test_pr_auc)
print("Train F1 Score:", train_f1)
print("Test F1 Score:", test_f1)
print("\nFeature Importances (RFE Selected):\n", feature_importances)

#%%

plt.figure(figsize=(8, 10))
plt.barh(feature_importances['Feature'][:30], feature_importances['Importance'][:30], color='#133E87')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.ylim(-0.5, 30 - 0.5)
plt.gca().invert_yaxis() 
plt.tight_layout()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/Top30_featureimportance.pdf', dpi=300, format='pdf', bbox_inches='tight', transparent=True, pad_inches=0.1)
plt.show()

#%%
# Plot ROC Curve
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, roc_curve, precision_recall_curve
# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, color='#B43F3F', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)  # Diagonal reference line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/testROCAUC_curve.pdf',dpi=300, format='pdf', bbox_inches='tight', transparent=True, pad_inches=0.1)
plt.show()

# Calculate Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_test_prob)
pr_auc = auc(recall, precision)

# Plot Precision-Recall curve
plt.figure(figsize=(7, 5))
plt.plot(recall, precision, color='#B43F3F', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
positive_proportion = sum(y_test) / len(y_test)
plt.ylim([0,1.05])
plt.gca().set_ylim(bottom=0)
plt.axhline(y=positive_proportion, linestyle='--', color='gray', lw=1, label='Baseline')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/testPRAUC_curve.pdf',dpi=300, format='pdf', bbox_inches='tight', transparent=True, pad_inches=0.1)
plt.show()

#%%
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size as needed
disp.plot(ax=ax, cmap=plt.cm.Greens, values_format='d')  # Use 'd' for integer values

# Change font size of text in confusion matrix
for text in disp.text_.ravel():
    text.set_fontsize(14) 
    
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/confusionmatrix.pdf', dpi=300,format='pdf', bbox_inches='tight',transparent=True, pad_inches=0.1)
plt.show()


# %%
###^^ validation with Resistance cohort ####
ext = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_TU.txt', sep='\t', index_col=0)
clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo_new.txt', sep='\t', index_col=0)
# %%
X_external = ext.iloc[:,1::2]
X_external = X_external.loc[selected_features,:]
X_external = X_external.T
y_external = np.array(clin.iloc[1::2,2])

y_external_pred = best_rf.predict(X_external)
y_external_prob = best_rf.predict_proba(X_external)[:, 1]  # Probabilities for the positive class

# Calculate performance metrics
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc

# Calculate accuracy
external_accuracy = accuracy_score(y_external, y_external_pred)

# Calculate ROC-AUC
external_roc_auc = roc_auc_score(y_external, y_external_prob)

# Calculate F1-score
external_f1 = f1_score(y_external, y_external_pred)

# Calculate Precision-Recall AUC
precision, recall, _ = precision_recall_curve(y_external, y_external_prob)
external_pr_auc = auc(recall, precision)

# Print metrics
print("External Accuracy:", external_accuracy)
print("External ROC-AUC:", external_roc_auc)
print("External PR-AUC:", external_pr_auc)
print("External F1 Score:", external_f1)

from scipy.stats import pearsonr
from scipy.stats import spearmanr

print(pearsonr(y_external_prob,clin.iloc[1::2,3]))
print(spearmanr(y_external_prob,clin.iloc[1::2,3]))

#%%






# %%
#####^^^^^^^^^^^^^^^^^ Random Forest with expHRD_356_genes.txt / DNA repair genes ####################################################
with open('/home/jiye/jiye/copycomparison/gDUTresearch/202408_analysis/expHRD_comparison/expHRD_356_genes.txt', 'r') as file:
    explist = file.read()
explist = list(explist.strip("'").split("', '"))
idmatch = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/116_validation_gene_TPM_symbol_group.txt', sep='\t')
idmatch = idmatch[['Geneid','Gene Symbol']]
idmatch['Geneid'] = idmatch['Geneid'].str.split('.').str[0]
idmatch = idmatch.loc[idmatch['Geneid'].isin(explist),:]
finalexp = list(idmatch['Gene Symbol'])

dut_filtered = dut.loc[(dut['p_value']<0.05) & (np.abs(dut['log2FC'])>1)]
featurelist = dut_filtered[dut_filtered['Gene Symbol'].isin(finalexp)].index.to_list()

#%%
####* DNA repair genes###
repair_path = '/home/jiye/jiye/copycomparison/gDUTresearch/data/DNARepairGeneList.txt' #repair gene list as python list
with open(repair_path, 'rb') as lf:
    repairgenes = pickle.load(lf)
dut_filtered = dut.loc[(dut['p_value']<0.05) & (np.abs(dut['log2FC'])>1)]
featurelist = dut_filtered[dut_filtered['Gene Symbol'].isin(repairgenes)].index.to_list()
#%%
y = np.array(trans.iloc[-1,:-1])
y = y.astype(int)
X = trans.loc[featurelist,:]
X = X.iloc[:,:-1]
X = X.T

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=77)


# Parameter grid for Grid Search
param_grid = {
    'n_estimators': [50, 75, 100],  # Reducing the maximum number of trees
    'max_depth': [4, 6, 8],  # Further limit tree depth to prevent complex fits
    'min_samples_split': [10, 20, 30],  # More restrictive splitting
    'min_samples_leaf': [7, 10, 15],  # Increase minimum samples in a leaf
    'max_features': ['sqrt', 'log2']  # Limit features considered at each split
}

# param_grid = {
#     'n_estimators': [50, 100, 150],
#     'max_depth': [5, 10,], #15
#     'min_samples_split': [10, 15, 20], #5
#     'min_samples_leaf': [4, 6], #2
#     'max_features': ['sqrt', 'log2']
# }

# Initialize Random Forest model
rf = RandomForestClassifier(random_state=77,class_weight='balanced')

# Stratified k-fold cross-validation
cv = StratifiedKFold(n_splits=5)

# Grid Search with cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Best parameters from Grid Search
best_rf = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE

rfecv = RFECV(estimator=best_rf, step=1, cv=StratifiedKFold(5), scoring='accuracy')
rfecv.fit(X_train, y_train)

print("Optimal number of features:", rfecv.n_features_)
selected_features = X_train.columns[rfecv.support_]
#print("Selected Features:", selected_features)

# Fit model with selected features
X_train_rfe = X_train[selected_features]
X_test_rfe = X_test[selected_features]
best_rf.fit(X_train_rfe, y_train)


from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc, RocCurveDisplay, PrecisionRecallDisplay

# Predictions and probabilities
y_train_pred = best_rf.predict(X_train_rfe)
y_test_pred = best_rf.predict(X_test_rfe)
y_train_prob = best_rf.predict_proba(X_train_rfe)[:, 1]
y_test_prob = best_rf.predict_proba(X_test_rfe)[:, 1]

# Metrics calculations
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)
train_roc_auc = roc_auc_score(y_train, y_train_prob)
test_roc_auc = roc_auc_score(y_test, y_test_prob)

# Precision-Recall AUC
precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_prob)
precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_prob)
train_pr_auc = auc(recall_train, precision_train)
test_pr_auc = auc(recall_test, precision_test)

# Feature importance (based on RFE-selected features)
feature_importances = pd.DataFrame({
    'Feature': selected_features,  # Column names of X #X.columns
    'Importance': best_rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Print metrics
print("\nTrain Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)
print("Train ROC-AUC:", train_roc_auc)
print("Test ROC-AUC:", test_roc_auc)
print("Train PR-AUC:", train_pr_auc)
print("Test PR-AUC:", test_pr_auc)
print("Train F1 Score:", train_f1)
print("Test F1 Score:", test_f1)
print("\nFeature Importances (RFE Selected):\n", feature_importances)

#%%
plt.figure(figsize=(6, 12))
plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='#133E87')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.ylim(-0.5, len(feature_importances) - 0.5)
plt.gca().invert_yaxis() 
plt.tight_layout()
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/expHRD_featureimportance.pdf', dpi=300, format='pdf', bbox_inches='tight', transparent=True, pad_inches=0.1)
plt.show()

# Plot ROC Curve
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, roc_curve, precision_recall_curve
# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, color='#B43F3F', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)  # Diagonal reference line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/expHRD_testROCAUC_curve.pdf',dpi=300, format='pdf', bbox_inches='tight', transparent=True, pad_inches=0.1)
plt.show()

# Calculate Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_test, y_test_prob)
pr_auc = auc(recall, precision)

# Plot Precision-Recall curve
plt.figure(figsize=(7, 5))
plt.plot(recall, precision, color='#B43F3F', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
positive_proportion = sum(y_test) / len(y_test)
plt.axhline(y=positive_proportion, linestyle='--', color='gray', lw=1, label='Baseline')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0,1.05])
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/expHRD_testPRAUC_curve.pdf',dpi=300, format='pdf', bbox_inches='tight', transparent=True, pad_inches=0.1)
plt.show()

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size as needed
disp.plot(ax=ax, cmap=plt.cm.Greens, values_format='d')  # Use 'd' for integer values

# Change font size of text in confusion matrix
for text in disp.text_.ravel():
    text.set_fontsize(14) 

plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/expHRD_confusionmatrix.pdf', dpi=300,format='pdf', bbox_inches='tight',transparent=True, pad_inches=0.1)
plt.show()
# %%


#%%
###^^ validation with Resistance cohort ####
ext = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_TU.txt', sep='\t', index_col=0)
clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo_new.txt', sep='\t', index_col=0)

X_external = ext.iloc[:,1::2]
X_external = X_external.loc[selected_features,:]
X_external = X_external.T
y_external = np.array(clin.iloc[1::2,2])

y_external_pred = best_rf.predict(X_external)
y_external_prob = best_rf.predict_proba(X_external)[:, 1]  # Probabilities for the positive class

# Calculate performance metrics
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc

# Calculate accuracy
external_accuracy = accuracy_score(y_external, y_external_pred)

# Calculate ROC-AUC
external_roc_auc = roc_auc_score(y_external, y_external_prob)

# Calculate F1-score
external_f1 = f1_score(y_external, y_external_pred)

# Calculate Precision-Recall AUC
precision, recall, _ = precision_recall_curve(y_external, y_external_prob)
external_pr_auc = auc(recall, precision)

# Print metrics
print("External Accuracy:", external_accuracy)
print("External ROC-AUC:", external_roc_auc)
print("External PR-AUC:", external_pr_auc)
print("External F1 Score:", external_f1)

from scipy.stats import pearsonr
from scipy.stats import spearmanr

print(pearsonr(y_external_prob,clin.iloc[1::2,3]))
print(spearmanr(y_external_prob,clin.iloc[1::2,3]))


# %%
#####^^ HGmodel comparison #######################
hgmodel = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/HGmodel/116_HGmodel_proba.txt', sep='\t', )
y = np.array(trans.iloc[-1,:-1])
y_external = y.astype(int)
y_external_prob = list(hgmodel['pred_HRD'])
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc

# Calculate accuracy
#external_accuracy = accuracy_score(y_external, y_external_pred)

# Calculate ROC-AUC
external_roc_auc = roc_auc_score(y_external, y_external_prob)

# Calculate F1-score
#external_f1 = f1_score(y_external, y_external_pred)

# Calculate Precision-Recall AUC
precision, recall, _ = precision_recall_curve(y_external, y_external_prob)
external_pr_auc = auc(recall, precision)

# Print metrics
#print("External Accuracy:", external_accuracy)
print("External ROC-AUC:", external_roc_auc)
print("External PR-AUC:", external_pr_auc)
#print("External F1 Score:", external_f1)

from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, roc_curve, precision_recall_curve


fpr, tpr, _ = roc_curve(y_external, y_external_prob)
roc_auc = auc(fpr, tpr)


# Plot ROC curve
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, color='#B43F3F', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)  # Diagonal reference line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/HGmodel/testROCAUC_curve.pdf',dpi=300, format='pdf', bbox_inches='tight', transparent=True, pad_inches=0.1)
plt.show()

# Calculate Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_external, y_external_prob)
pr_auc = auc(recall, precision)

# Plot Precision-Recall curve

plt.figure(figsize=(7, 5))
plt.plot(recall, precision, color='#B43F3F', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
positive_proportion = sum(y_external) / len(y_external)
plt.axhline(y=positive_proportion, linestyle='--', color='gray', lw=1, label='Baseline')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0,1.05])
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/HGmodel/testPRAUC_curve.pdf',dpi=300, format='pdf', bbox_inches='tight', transparent=True, pad_inches=0.1)
plt.show()

# print(pearsonr(y_external_prob,clin.iloc[1::2,3]))
# print(spearmanr(y_external_prob,clin.iloc[1::2,3]))

# %%
#####^^^^^^ 77 random DUT input * 100 times ##################################
dut_filtered = dut.loc[(dut['p_value']<0.05) & (np.abs(dut['log2FC'])>1)]
featurelist = dut_filtered.index.to_list()

y = np.array(trans.iloc[-1,:-1])
y = y.astype(int)
#X = trans.loc[featurelist,:]
X = trans
X = X.iloc[:,:-1]
X = X.T

from scipy.interpolate import interp1d

n_iterations = 100
n_features = 77
metrics = {
    'accuracy': [],
    'f1_score': [],
    'roc_auc': [],
    'pr_auc': []
}
roc_curves = []
pr_curves = []

# Common x-axis for ROC and PR curve interpolation
mean_fpr = np.linspace(0, 1, 100)
mean_recall = np.linspace(0, 1, 100)

for i in range(n_iterations):
    np.random.seed(i)
    selected_features = X.columns[np.random.choice(X.shape[1], n_features, replace=False)]
    X_selected = X[selected_features]

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, stratify=y, random_state=42)
    
    param_grid = {
        'n_estimators': [50, 75, 100],  # Reducing the maximum number of trees
        'max_depth': [4, 6, 8],  # Further limit tree depth to prevent complex fits
        'min_samples_split': [10, 20, 30],  # More restrictive splitting
        'min_samples_leaf': [7, 10, 15],  # Increase minimum samples in a leaf
        'max_features': ['sqrt', 'log2']  # Limit features considered at each split
    }
    # Stratified k-fold cross-validation
    rf = RandomForestClassifier(random_state=42,class_weight='balanced')
    cv = StratifiedKFold(n_splits=5)

    # Grid Search with cross-validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)

    # Best parameters from Grid Search
    best_rf = grid_search.best_estimator_

    best_rf.fit(X_train, y_train)
    
    y_test_pred = best_rf.predict(X_test)
    y_test_prob = best_rf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_prob)
    precision, recall, _ = precision_recall_curve(y_test, y_test_prob)
    pr_auc = auc(recall, precision)

    metrics['accuracy'].append(acc)
    metrics['f1_score'].append(f1)
    metrics['roc_auc'].append(roc_auc)
    metrics['pr_auc'].append(pr_auc)

    fpr, tpr, _ = roc_curve(y_test, y_test_prob)

    # Interpolate ROC curve
    interp_tpr = interp1d(fpr, tpr, kind='linear', fill_value="extrapolate")(mean_fpr)
    roc_curves.append(interp_tpr)

    # Interpolate PR curve
    interp_precision = interp1d(recall, precision, kind='linear', bounds_error=False, fill_value=(precision[0], precision[-1]))(mean_recall)
    pr_curves.append(interp_precision)

# Mean and std calculation for ROC curves
mean_tpr = np.mean(roc_curves, axis=0)
std_tpr = np.std(roc_curves, axis=0)

# Mean and std calculation for PR curves
mean_precision = np.mean(pr_curves, axis=0)
std_precision = np.std(pr_curves, axis=0)

# Plotting ROC Curve
plt.figure(figsize=(8, 6))
for tpr in roc_curves:
    plt.plot(mean_fpr, tpr, color='#B43F3F', lw=0.5, alpha=0.2)  # Thin gray lines for each iteration
plt.plot(mean_fpr, mean_tpr, color='#B43F3F', lw=2, label=f'Mean ROC curve (AUC = {np.mean(metrics["roc_auc"]):.2f} ± {np.std(metrics["roc_auc"]):.2f})')
plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='#B43F3F', alpha=0.1)  # Confidence interval
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random 100 Iterations')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/randomtrans_testROCAUC_curve.pdf',dpi=300, format='pdf', bbox_inches='tight', transparent=True, pad_inches=0.1)
plt.show()

# Plotting PR Curve
plt.figure(figsize=(8, 6))
for precision in pr_curves:
    plt.plot(mean_recall, precision, color='#B43F3F' ,lw=0.5, alpha=0.2)  # Thin gray lines for each iteration
plt.plot(mean_recall, mean_precision, color='#B43F3F', lw=2, label=f'Mean PR curve (AUC = {np.mean(metrics["pr_auc"]):.2f} ± {np.std(metrics["pr_auc"]):.2f})')
plt.fill_between(mean_recall, mean_precision - std_precision, mean_precision + std_precision, color='#B43F3F', alpha=0.1)  # Confidence interval
positive_proportion = sum(y) / len(y)
plt.axhline(y=positive_proportion, linestyle='--', color='gray', label='Baseline')
plt.ylim([0,1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve - Random 100 Iterations')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/randomtrans_testPRAUC_curve.pdf',dpi=300, format='pdf', bbox_inches='tight', transparent=True, pad_inches=0.1)
plt.show()

# Print mean and std metrics
print(f"Mean Accuracy: {np.mean(metrics['accuracy']):.4f} ± {np.std(metrics['accuracy']):.4f}")
print(f"Mean F1 Score: {np.mean(metrics['f1_score']):.4f} ± {np.std(metrics['f1_score']):.4f}")
print(f"Mean ROC-AUC: {np.mean(metrics['roc_auc']):.4f} ± {np.std(metrics['roc_auc']):.4f}")
print(f"Mean PR-AUC: {np.mean(metrics['pr_auc']):.4f} ± {np.std(metrics['pr_auc']):.4f}")
# %%
