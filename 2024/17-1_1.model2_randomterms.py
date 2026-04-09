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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix
import seaborn as sns
import pronto
import random

def read_go_file(file_path):
    go_terms = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                go_term, genes = parts
                genes_list = genes.split()
                go_terms[go_term] = genes_list
    return go_terms

go_terms = read_go_file('/home/jiye/jiye/copycomparison/OC_transcriptome/GO_Biological_Process_2021.txt')

ont = pronto.Ontology('http://purl.obolibrary.org/obo/go/go-basic.obo')
root = ont['GO:0008150']

def get_upper_terms(term, max_depth, current_depth=0):
    upper_terms = []
    if current_depth < max_depth:
        for child in term.subclasses(distance=1):
            upper_terms.append((child.id, child.name))
            upper_terms.extend(get_upper_terms(child, max_depth, current_depth + 1))
    return set(upper_terms)

terms = get_upper_terms(root, 5) - get_upper_terms(root, 4)
random_terms = random.sample(terms, 600)
go_term_ids = [term[0] for term in random_terms]

TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_TU.txt', sep='\t', index_col=0)
major = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = major[major['type']=='major']['gene_ENST']
ARdut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t')
DUTlist = ARdut.loc[(ARdut['p_value']<0.05) & (ARdut['log2FC']>1.5)]
majorDUTlist = DUTlist[DUTlist['gene_ENST'].isin(majorlist)]

def evaluate_model_with_seed(seed, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

    model = LogisticRegression(max_iter=1000)
    solvers = ['liblinear', 'saga']
    penalty = ['l1', 'l2']
    c_values = [10, 1.0, 0.1]
    grid = dict(solver=solvers, penalty=penalty, C=c_values)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=cv, scoring='average_precision', error_score=0)
    grid_result = grid_search.fit(X_train, y_train)

    best_model = grid_result.best_estimator_

    y_train_pred_prob = best_model.predict_proba(X_train)[:, 1]
    y_test_pred_prob = best_model.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_train, y_train_pred_prob)
    test_auc = roc_auc_score(y_test, y_test_pred_prob)
    train_ap = average_precision_score(y_train, y_train_pred_prob)
    test_ap = average_precision_score(y_test, y_test_pred_prob)

    return train_auc, test_auc, train_ap, test_ap

for term in go_term_ids:
    targetgenelist = []
    for key, genes in go_terms.items():
        if term in key:
            targetgenelist.extend(genes)
    
    targetfeatures = majorDUTlist[majorDUTlist['Gene Symbol'].isin(targetgenelist)]['gene_ENST'].to_list()
    input = TU[TU.index.isin(targetfeatures)]
    input = input.dropna()
    if input.shape[0] > 0:
        sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
        responder = sampleinfo.loc[(sampleinfo['response'] == 1), 'sample_full'].to_list()
        nonresponder = sampleinfo.loc[(sampleinfo['response'] == 0), 'sample_full'].to_list()
        y = sampleinfo['response']
        y = pd.DataFrame(y)
        
        delta_df = pd.DataFrame(index=input.index)
        for i in range(0, len(input.columns), 2):
            post_col = input.columns[i]
            pre_col = input.columns[i + 1]
            sample_name = post_col[:-4]
            delta_df[sample_name + 'delta'] = input[post_col] - input[pre_col]

        X = delta_df.T
        y = y.iloc[0::2]
        y.index = X.index
        y = y.values.ravel()  # Convert y to 1D array

        results = {'train_auc': [], 'test_auc': [], 'train_ap': [], 'test_ap': []}
        n_repeats = 50

        for i in range(n_repeats):
            seed = np.random.randint(0, 10000)
            train_auc, test_auc, train_ap, test_ap = evaluate_model_with_seed(seed, X, y)
            results['train_auc'].append(train_auc)
            results['test_auc'].append(test_auc)
            results['train_ap'].append(train_ap)
            results['test_ap'].append(test_ap)

        metrics = ['train_auc', 'test_auc', 'train_ap', 'test_ap']
        mean_results = {metric: np.mean(results[metric]) for metric in metrics}
        std_results = {metric: np.std(results[metric]) for metric in metrics}

        result_file_path = '/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/model/model2_random/result/' + term + '_result_summary.txt'
        with open(result_file_path, 'w') as file:
            for metric in metrics:
                file.write(f"{metric}_mean: {mean_results[metric]:.4f}\n")
                file.write(f"{metric}_std: {std_results[metric]:.4f}\n")

# %%
