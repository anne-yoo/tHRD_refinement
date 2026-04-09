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
from lifelines import CoxPHFitter

#%%
### make dataframe of delta TU (pre -> post)
TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0)
TU['Gene Symbol'] = TU.index.str.split("-",2).str[1]
sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')

dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t')
enr = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_Wilcoxon_GOenrichment.txt', sep='\t')

dut = dut[(dut['p_value']<0.05) & (dut['log2FC']>1.5)]
dutlist = list(dut['gene_ENST'])
dut_df = TU.loc[TU.index.isin(dutlist),:]

enr = enr[(enr["Term"].str.contains("DNA repair", case=False))]

dnarepairgenes = [gene for sublist in enr['Genes'].str.split(';') for gene in sublist]
dnarepairgenes = set(dnarepairgenes)

finaldf = dut_df[dut_df['Gene Symbol'].isin(dnarepairgenes)]

###only AR ###
ARlist = list(sampleinfo.loc[sampleinfo['response']==1,'sample_full'])
finaldf = finaldf.loc[:,ARlist]

df_bfD = finaldf[[col for col in finaldf.columns if '-bfD' in col]]
df_atD = finaldf[[col for col in finaldf.columns if '-atD' in col]]

df_bfD.columns = df_bfD.columns.str[:-4]
df_atD.columns = df_atD.columns.str[:-4]

deltaTU = df_atD.subtract(df_bfD)

ARinfo = sampleinfo.loc[sampleinfo['sample_full'].isin(ARlist),:]
ARinfo = ARinfo.loc[0::2,['sample_id','interval','BRCAmut']]
#%%

###*** Cox regression ###

ARinfo['event'] = 1
ARinfo.columns = ['sample_id','PFS','BRCAmut','event']
pfs_data = ARinfo
transcript_usage = deltaTU

# Transpose transcript_usage so columns are transcripts and rows are samples
transcript_usage = transcript_usage.transpose()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
normalized_data = scaler.fit_transform(transcript_usage)
normalized_df = pd.DataFrame(normalized_data, columns=transcript_usage.columns, index=transcript_usage.index)

# Merge the datasets on sample identifier
merged_data = pd.concat([pfs_data.set_index('sample_id'), normalized_df], axis=1)

# Univariate Cox regression for each transcript
results = []
for transcript in normalized_df.columns:
    # Subset the data to include only the PFS, event, and current transcript
    subset = merged_data[['PFS', 'event', transcript]]
    
    # Fit the Cox model
    cph = CoxPHFitter()
    cph.fit(subset, duration_col='PFS', event_col='event')
    
    # Extract the coefficient and p-value
    coef = cph.hazard_ratios_[transcript]
    p_value = cph.summary.loc[transcript, 'p']
    
    results.append({'transcript': transcript, 'coefficient': coef, 'p_value': p_value})

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Display or save the results
print(results_df)


# %%
###*** Linear Regression ####

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

lrdf = deltaTU.transpose()
lrdf['PFS'] = list(ARinfo['PFS'])
lrdf['BRCAmut'] = list(ARinfo['BRCAmut'])

results_df = pd.DataFrame(columns=['Transcript', 'Coefficient', 'P-value', 'R-squared'])

for transcript in lrdf.columns.drop(['PFS', 'BRCAmut']):
    # Define the predictors (transcript and BRCA_status) and outcome (PFS)
    #X = lrdf[['BRCAmut', transcript]]
    X = lrdf[[transcript]]
    y = lrdf['PFS']

    # Add a constant to the model (intercept)
    X = sm.add_constant(X)

    # Fit the OLS model
    model = sm.OLS(y, X).fit()

    # Extract the results for the transcript
    coef = model.params[transcript]
    p_value = model.pvalues[transcript]
    r_squared = model.rsquared

    # Append to results DataFrame
    results_df = results_df.append({'Transcript': transcript, 'Coefficient': coef, 'P-value': p_value, 'R-squared': r_squared}, ignore_index=True)

# Display or save the results DataFrame
print(results_df)



# %%
######** Pearson Correlation #######

results_df = pd.DataFrame(columns=['Transcript', 'Pearson Correlation', 'P-value'])

for transcript in lrdf.columns.drop(['PFS', 'BRCAmut']):
    correlation, p_value = lrdf['PFS'].corr(lrdf[transcript], method='pearson'), 'N/A'  # Pandas corr() doesn't provide p-values
    results_df = results_df.append({'Transcript': transcript, 'Pearson Correlation': correlation, 'P-value': p_value}, ignore_index=True)

print(results_df)



# %%
###*** only major DUT #####

