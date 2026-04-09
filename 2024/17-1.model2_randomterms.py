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
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

sns.set(font = 'Arial')
sns.set_style("whitegrid")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# %%
##### ^^^^^^^^^ deltaTU: IR vs. AR #######################
TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_TU.txt', sep='\t', index_col=0)
major = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = major[major['type']=='major']['gene_ENST']
ARdut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t')
DUTlist = ARdut.loc[(ARdut['p_value']<0.05) & (ARdut['log2FC']>1.5)]
majorDUTlist = DUTlist[DUTlist['gene_ENST'].isin(majorlist)]

def read_go_file(file_path):
    go_terms = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into GO term and genes, assuming they are separated by tabs
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                go_term, genes = parts
                genes_list = genes.split()  # Splitting gene symbols which are separated by spaces
                go_terms[go_term] = genes_list
    return go_terms

go_terms = read_go_file('/home/jiye/jiye/copycomparison/OC_transcriptome/GO_Biological_Process_2021.txt')

import pronto
ont = pronto.Ontology( 'http://purl.obolibrary.org/obo/go/go-basic.obo')
root = ont['GO:0008150']

# Function to get children up to a certain depth
def get_upper_terms(term, max_depth, current_depth=0):
    upper_terms = []
    if current_depth < max_depth:
        # Check direct children only
        for child in term.subclasses(distance=1):  # Use subclasses with distance=1 for direct children
            upper_terms.append((child.id, child.name))
            # Recursive call to explore further down to specified depth
            upper_terms.extend(get_upper_terms(child, max_depth, current_depth + 1))
    return set(upper_terms)

# Example: Get terms up to level 2
# Assuming 'GO:0008150' is the biological process root term, modify as necessary for your term
terms = get_upper_terms(root, 5) - get_upper_terms(root, 4)

#%%
import random
random_terms = random.sample(terms, 50)
go_term_ids = [term[0] for term in random_terms]

for term in go_term_ids:
    
    targetgenelist = []
    for key, genes in go_terms.items():
        if term in key:
            targetgenelist.extend(genes)
    
    
    targetfeatures = majorDUTlist[majorDUTlist['Gene Symbol'].isin(targetgenelist)]['gene_ENST'].to_list()
    input = TU[TU.index.isin(targetfeatures)]
    input = input.dropna()
    if input.shape[0]>0:
        sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
        responder = sampleinfo.loc[(sampleinfo['response']==1),'sample_full'].to_list()
        nonresponder = sampleinfo.loc[(sampleinfo['response']==0),'sample_full'].to_list()
        y = sampleinfo['response']
        y = pd.DataFrame(y)
        
        delta_df = pd.DataFrame(index=input.index)
        for i in range(0, len(input.columns), 2):
            post_col = input.columns[i]
            pre_col = input.columns[i + 1]
            
            # Compute the delta and add to the new DataFrame
            # The new column name corresponds to the sample name without 'post' or 'pre'
            sample_name = post_col[:-4]  # Assuming the last 4 characters are 'post' or 'pre'
            delta_df[sample_name + 'delta'] = input[post_col] - input[pre_col]

        X=delta_df.T
        y = y.iloc[0::2]
        y.index = X.index
        
        def plot_roc_curve(y_train_true, y_train_pred_prob, y_test_true, y_test_pred_prob):
        
            # Train ROC
            fpr_train, tpr_train, _ = roc_curve(y_train_true, y_train_pred_prob)
            auc_train = roc_auc_score(y_train_true, y_train_pred_prob)
            plt.plot(fpr_train, tpr_train, label=f'Train AUC = {auc_train:.2f}', color='blue')
            
            # Test ROC
            fpr_test, tpr_test, _ = roc_curve(y_test_true, y_test_pred_prob)
            auc_test = roc_auc_score(y_test_true, y_test_pred_prob)
            plt.plot(fpr_test, tpr_test, label=f'Test AUC = {auc_test:.2f}', color='red')
            
            plt.plot([0, 1], [0, 1], 'k--', label='Chance')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()

        def plot_precision_recall_curve(y_train_true, y_train_pred_prob, y_test_true, y_test_pred_prob):
            # Train Precision-Recall
            precision_train, recall_train, _ = precision_recall_curve(y_train_true, y_train_pred_prob)
            ap_train = average_precision_score(y_train_true, y_train_pred_prob)
            plt.plot(recall_train, precision_train, label=f'Train AP = {ap_train:.2f}', color='blue')
            
            # Test Precision-Recall
            precision_test, recall_test, _ = precision_recall_curve(y_test_true, y_test_pred_prob)
            ap_test = average_precision_score(y_test_true, y_test_pred_prob)
            plt.plot(recall_test, precision_test, label=f'Test AP = {ap_test:.2f}', color='red')
            
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()

            
        def plot_feature_importances(model, feature_names, top_n=30):
            # Extract feature importances
            importance_vals = model.coef_[0]
            feature_importances = pd.Series(importance_vals, index=feature_names)
            
            # Sort features by absolute importance
            feature_importances_sorted = feature_importances.sort_values(ascending=False).head(top_n)
            
            # Plot
            plt.figure(figsize=(10, 6))
            feature_importances_sorted.plot(kind='barh')
            plt.gca().invert_yaxis()  # Invert y-axis to have the most important at the top
            plt.title('Top Feature Importances')
            plt.xlabel('Absolute Importance')
            plt.ylabel('Features')
            plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/model/model2_random/'+term+'_FI.pdf', bbox_inches="tight")
            plt.close()

        def plot_confusion_matrix(y_true, y_pred_prob, threshold=0.5):
            # Convert probabilities to binary predictions based on the threshold
            y_pred = (y_pred_prob >= threshold).astype(int)
            
            # Compute confusion matrix
            conf_matrix = confusion_matrix(y_true, y_pred)
            
            # Plot
            plt.figure(figsize=(6, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', square=True,
                        xticklabels=['Pre', 'Post'], yticklabels=['Pre', 'Post'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/model/model2_random/'+term+'_CM.pdf', bbox_inches="tight")
            plt.close()

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        # Define models and parameters for Grid Search
        model = LogisticRegression(max_iter=1000)
        solvers = ['newton-cg', 'lbfgs', 'liblinear']
        penalty = ['l2']
        c_values = [100, 10, 1.0, 0.1, 0.01]
        grid = dict(solver=solvers, penalty=penalty, C=c_values)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        grid_search = GridSearchCV(estimator=model, param_grid=grid,  cv=cv, scoring='average_precision', error_score=0)
        grid_result = grid_search.fit(X_train, y_train)

        # Evaluate the best model from grid search on the test set
        best_model = grid_result.best_estimator_

        # Standardize features (use full dataset for feature names if X is a DataFrame)
        scaler = StandardScaler().fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Predict probabilities for ROC and PR curves using the best model
        y_train_pred_prob = best_model.predict_proba(X_train_scaled)[:, 1]
        y_test_pred_prob = best_model.predict_proba(X_test_scaled)[:, 1]

        # Calculate AUC and PR-AUC for training and test sets
        train_auc = roc_auc_score(y_train, y_train_pred_prob)
        test_auc = roc_auc_score(y_test, y_test_pred_prob)
        #print(f"Train AUC: {train_auc:.2f}")
        #print(f"Test AUC: {test_auc:.2f}")

        train_ap = average_precision_score(y_train, y_train_pred_prob)
        test_ap = average_precision_score(y_test, y_test_pred_prob)
        #print(f"Train Average Precision: {train_ap:.2f}")
        #print(f"Test Average Precision: {test_ap:.2f}")

        # Plotting ROC and Precision-Recall Curves
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plot_roc_curve(y_train, y_train_pred_prob, y_test, y_test_pred_prob)
        plt.subplot(1, 2, 2)
        plot_precision_recall_curve(y_train, y_train_pred_prob, y_test, y_test_pred_prob)
        plt.tight_layout()
        plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/model/model2_random/'+term+'_curves.pdf', bbox_inches="tight")
        plt.close()
        
        if isinstance(X, pd.DataFrame):
            plot_feature_importances(best_model, X.columns)

        # Plotting Confusion Matrix for the Test Set
        plot_confusion_matrix(y_test, y_test_pred_prob)
        Result1 = [train_auc,test_auc, train_ap, test_ap]
        with open('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/model/model2_random/result/'+term+'_result.txt', 'w') as file:
            for number in Result1:
                # Write each number to a new line
                file.write(f"{number}\n")
        

# %%
####^^^^^ READ THE RESULTS #######
model2_result = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/model/model2_random/result/combined_results.csv', sep=',')

df = model2_result

# Melt the dataframe to make it suitable for seaborn's boxplot
df_melted = df.melt(id_vars='GO_term', var_name='Metric', value_name='Value')

# Adjusts plot to ensure everything fits without overlap
ax = sns.boxplot(x='Metric', y='Value', data=df_melted, palette='pastel')
ax2 = sns.swarmplot(x='Metric', y='Value', data=df_melted, palette='deep')


medians = df_melted.groupby(['Metric'])['Value'].median()
medians = medians[['train_rocauc','test_rocauc','train_prauc','test_prauc']]
medians = medians.round(decimals=5)
vertical_offset = df_melted['Value'].median() * 0.02

for xtick in ax.get_xticks():
    print(xtick)
    ax.text(xtick, medians[xtick] + vertical_offset, medians[xtick], 
            horizontalalignment='center',size='medium',color='w',weight='semibold')

#sns.despine()

topgo = model2_result[(model2_result['train_rocauc']>0.8) & (model2_result['test_rocauc']>0.8) & (model2_result['train_prauc']>0.8) & (model2_result['test_prauc']>0.8)]
go_name_dict = {'GO:' + name.split('(GO:')[1].split(')')[0]: name for name in go_terms}
topgo['GO_term_name'] = topgo['GO_term'].map(go_name_dict)

######^###########
go_term_ids = ['GO:0006974','GO:0006281']
##############^###

for term in topgo['GO_term']:
    
    targetgenelist = []
    for key, genes in go_terms.items():
        if term in key:
            targetgenelist.extend(genes)
    
    
    targetfeatures = majorDUTlist[majorDUTlist['Gene Symbol'].isin(targetgenelist)]['gene_ENST'].to_list()
    input = TU[TU.index.isin(targetfeatures)]
    input = input.dropna()
    print(input.shape)
# %%
