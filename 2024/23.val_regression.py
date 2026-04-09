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
import numpy as np
import pandas as pd
import statsmodels.api as sm


# %%
dis = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_TU.txt', sep='\t', index_col=0)
val = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_TU.txt', sep='\t', index_col=0)
valgene = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_geneTPM.txt', sep='\t', index_col=0)
disinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo_new.txt', sep='\t', index_col=0)
valinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_validation_clinicalinfo.txt', sep='\t', index_col=0)
valinfo = valinfo.loc[list(val.columns),:]

dis_y = pd.DataFrame(disinfo.iloc[0::2,2])
dis_y.replace({0: 'IR', 1: 'AR'}, inplace=True)
valinfo['finalresponse'] = 'x'
valinfo.loc[(valinfo['ongoing']==1) | (valinfo['ongoing']==2) | (valinfo['ongoing']==4),'finalresponse'] = 'R'
valinfo.loc[(valinfo['ongoing']==0) & (valinfo['response']==1), 'finalresponse'] = 'AR'
valinfo.loc[(valinfo['ongoing']==3) & (valinfo['response']==1), 'finalresponse'] = 'AR'
valinfo.loc[(valinfo['ongoing']==0) & (valinfo['response']==0), 'finalresponse'] = 'IR'
valinfo.loc[(valinfo['ongoing']==3) & (valinfo['response']==0), 'finalresponse'] = 'IR'

import pickle

# Reactome2022_featurelist.pkl / BP2018_featurelist.pkl
with open('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/featureselection/Reactome2022_featurelist.pkl', 'rb') as file:
    featurelist_reactome = pickle.load(file)

with open('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/featureselection/BP2018_featurelist.pkl', 'rb') as file:
    featurelist_bp = pickle.load(file)

set1 = set(featurelist_bp).union(set(featurelist_reactome))
set2 = set(featurelist_bp).intersection(set(featurelist_reactome))


###
features = set1
###

df_tu = val.loc[features,:] #### use only 116 samples
df_tu = df_tu.T

pca = PCA(n_components=5)  # Select the number of components you want
pca_components = pca.fit_transform(df_tu)

# Convert PCA components to a DataFrame
df = pd.DataFrame(pca_components, columns=[f'PC{i+1}' for i in range(5)])



#df['response'] = valinfo['finalresponse']
df['BRCAmut'] = valinfo['BRCAmut'].to_list()
#df['interval'] = valinfo['interval'].to_list()
#df['line_binary'] = valinfo['line_binary'].to_list()
#df['OM/OS'] = valinfo['OM/OS'].to_list()


X = pd.get_dummies(df, drop_first=True)
X = sm.add_constant(X)
y = pd.Categorical(valinfo['finalresponse'], categories=['R', 'AR', 'IR'])

model = sm.MNLogit(y, X).fit(maxiter=300) 

print(model.summary())


#%%
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for each feature in X
vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)


#%%
###^^ expHRD ###
valgene = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_geneTPM.txt', sep='\t', index_col=0)


with open('/home/jiye/jiye/copycomparison/gDUTresearch/202408_analysis/expHRD_comparison/expHRD_356_genes.txt', 'r') as file:
    explist = file.read()
explist = explist.strip("'").split("', '")
ensg2gene = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/annotation/ensg2symbol.dict', sep='\t', header=None)
ensg2gene.columns = ['ensg','genesymbol']
ensg2gene['ensg_parsed'] = ensg2gene['ensg'].str.split(".",1).str[0]

explistdf = pd.DataFrame({'ensg_parsed':explist})

merged = pd.merge(explistdf, ensg2gene, how='inner', left_on='ensg_parsed', right_on='ensg_parsed')

exp_X = val.T
exp_X = exp_X.loc[:, exp_X.columns.str.contains('|'.join(merged['genesymbol']))]


majorlist = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majortrans = list(majorlist[majorlist['type']=='major']['gene_ENST'])
major_exp_X = exp_X.loc[:,exp_X.columns.isin(majortrans)]

plotdf = major_exp_X

df_tu = major_exp_X #### use only 116 samples
df_gene = valgene.loc[valgene.index.isin(list(merged['ensg'])),:]
df_gene = np.log2(df_gene+1)
df_gene = df_gene.T


pca = PCA(n_components=3)  # Select the number of components you want
pca_components = pca.fit_transform(df_tu)

# Convert PCA components to a DataFrame
df = pd.DataFrame(pca_components, columns=[f'PC{i+1}' for i in range(3)])


#df['response'] = valinfo['finalresponse']
df['BRCAmut'] = valinfo['BRCAmut'].to_list()
#df['interval'] = valinfo['interval'].to_list()
#df['line_binary'] = valinfo['line_binary'].to_list()
#df['OM/OS'] = valinfo['OM/OS'].to_list()
df['meanTU'] = list(df_tu.mean(axis=1))
df['meanTPM'] = list(df_gene.mean(axis=1))



X = pd.get_dummies(df, drop_first=True)

#X = X[['BRCAmut','meanTU','meanTPM']]
X = sm.add_constant(X)
y_encoded = list(valinfo['finalresponse'].map({'AR': 0, 'R': 1, 'IR': 2})) ### AR vs. R & IR
#y_encoded = list(valinfo['finalresponse'].map({'R': 0, 'AR': 1, 'IR': 2})) ### R vs. AR & IR

model = sm.MNLogit(y_encoded, X).fit(maxiter=300) 

print(model.summary())


# %%
