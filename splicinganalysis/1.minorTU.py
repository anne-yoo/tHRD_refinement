#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.stats import pearsonr
import random
from scipy.stats import ttest_ind

# %%
#^ change of minor transcript usage: pre -> post samples 

features = pd.read_csv("/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/merged_TU/whole_TU/XGBoost/response_group.txt", header=None, index_col=0)
features = features.iloc[1:,:]
features.reset_index(inplace=True)
features.columns = ['gene_ENST']

# %%
tu = pd.read_csv("/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/pre_post_TU/final_data/minor_prop.txt", sep='\t')
# tu = pd.read_csv("/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/pre_post_TU/final_data/major_prop.txt", sep='\t')
# %%
#^ merge selected transcipts with TU data
m_tu = pd.merge(features, tu, on='gene_ENST')
# %%
m_tu.set_index(['gene_ENST'], inplace=True)
samples = m_tu.columns.to_list()
for i in range(int(len(samples)/2)):
    i = i*2
    if "atD" in samples[i]:
        samples[i+1] = samples[i+1][:10]+"-bfD"
    elif "bfD" in samples[i]:
        samples[i+1] = samples[i+1][:10]+"-atD"
    elif "atD" in samples[i+1]:
        samples[i] = samples[i][:10]+"-bfD"
    elif "bfD" in samples[i+1]:
        samples[i] = samples[i][:10]+"-atD"

m_tu.columns = samples

# %%
pre = [x for x in samples if 'bfD' in x ]
post = [x for x in samples if 'atD' in x ]

pre_tu = m_tu.filter(pre)
post_tu = m_tu.filter(post)

pre_tu.columns = pre_tu.columns.str[:-4]
post_tu.columns = post_tu.columns.str[:-4]

pre_tu['pre/post'] = 'pre'
post_tu['pre/post'] = 'post'


final_tu = pd.concat([pre_tu,post_tu])
final_tu['gene_ENST'] = final_tu.index

#%%
#^ mannwhitney
mw_test = pd.DataFrame()
mw_test['gene_ENST'] = final_tu.iloc[:45,-1]
mw_test['p-val'] = 0

for i in range(45):
    list_a = final_tu.iloc[i,:27]
    list_b = final_tu.iloc[i+45,:27]

    mw_test.iloc[i,1] = mannwhitneyu(list_a, list_b, alternative='less')[1]
    # print(mannwhitneyu(list_a, list_b)[1])

print(mw_test[mw_test['p-val']<0.05])

# %%
#^ ttest
t_test = pd.DataFrame()
t_test['gene_ENST'] = final_tu.iloc[:45,-1]
t_test['p-val'] = 0

for i in range(45):
    list_a = final_tu.iloc[i,:27]
    list_b = final_tu.iloc[i+45,:27]

    t_test.iloc[i,1] = ttest_ind(list_a, list_b, alternative='greater',equal_var=False)[1]
    # print(mannwhitneyu(list_a, list_b)[1])

print(t_test[t_test['p-val']<0.05])
# %%
