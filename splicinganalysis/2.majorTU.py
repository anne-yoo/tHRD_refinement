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
#^ feature gene list
features = pd.read_csv("/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/merged_TU/whole_TU/XGBoost/response_group.txt", header=None, index_col=0)
features = features.iloc[1:,:]
features.reset_index(inplace=True)
features.columns = ['gene_ENST']
features['gene'] = features['gene_ENST'].str.split('-',1).str[1]

genelist = features[['gene']].drop_duplicates()

# %%
#^ tu data download
tu = pd.read_csv("/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/pre_post_TU/final_data/major_prop.txt", sep='\t')
# tu = pd.read_csv("/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/pre_post_TU/final_data/minor_prop.txt", sep='\t')

# %%
#^ merge with feature gene list
g_tu = tu
g_tu['gene'] = tu['gene_ENST'].str.split('-',1).str[1]
m_tu = pd.merge(genelist, g_tu, on='gene')

m_tu.set_index(['gene_ENST'], inplace=True)
m_tu = m_tu.drop(['gene'],axis=1)

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

#%%
#^ sample filtering: only responders!
group_info = pd.read_csv('/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/clinical_info_gHRD/processed_info_with_originalID.txt', sep="\t", index_col="GID")

group_info = group_info[["OM/OS", "ORR", "drug", "interval"]]
group_info.columns = ["OM/OS", "group", "drug", "interval"]
group_info["drug"] = group_info["drug"].str.replace("Olapairb","Olaparib")
group_info["drug"] = group_info["drug"].str.replace("Olapairib","Olaparib")
group_info["drug"] = group_info["drug"].str.replace("olaparib","Olaparib")
group_info = group_info.dropna()
group_info = group_info.drop("drug", axis=1)
group_info["GID"] = group_info.index.str.replace("F","T").str[:10]
group_info["GID"] = group_info["GID"].str.replace("T","P")
group_info = group_info.drop_duplicates()
group_info = group_info.set_index("GID")

clinical = group_info

wholesample = pd.DataFrame(m_tu.columns, columns=['sample'])

res_int = clinical[clinical['group']==1]

mtsample = pd.DataFrame(res_int.index)
wholesample['GID'] = wholesample['sample'].str[:-4]

onlymt = pd.merge(wholesample, mtsample, on='GID', how='inner')

mtsamplelist = onlymt['sample'].tolist()

m_tu = m_tu[mtsamplelist]

#%%
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

# %%
#^ mannwhitney
num = pre_tu.shape[0]
mw_test = pd.DataFrame()
mw_test['gene_ENST'] = final_tu.iloc[:num,-1].to_list()
mw_test['p-val'] = 1
num = pre_tu.shape[0]
samplenum = pre_tu.shape[1]-1
for i in range(num):
    list_a = final_tu.iloc[i,:samplenum]
    list_b = final_tu.iloc[i+num,:samplenum]
    if all(item == 0 for item in list_a) or all(item == 0 for item in list_b) :
        mw_test.iloc[i,1] = 1
    else:
        mw_test.iloc[i,1] = mannwhitneyu(list_a, list_b, alternative='less')[1]
    # print(mannwhitneyu(list_a, list_b)[1])

final = mw_test[mw_test['p-val']<0.05]
print("major TU: pre < post - ", final.shape[0], "transcripts")
print(mw_test[mw_test['p-val']<0.05])

# %%
