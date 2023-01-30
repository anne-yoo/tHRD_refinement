#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.stats import pearsonr
import random
from scipy.stats import ttest_ind
import gseapy as gp

# %%
#^ DEG TMM pre-post data and check for splicing genes!

tmm = pd.read_csv("/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/readcount/final_data/TMM_220923_final_pre_post_samples_with_symbol.txt", sep='\t')
# %%

tmm.set_index(['Geneid'], inplace=True)

samples = tmm.columns.to_list()
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

tmm.columns = samples

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

wholesample = pd.DataFrame(tmm.columns, columns=['sample'])

res_int = clinical[clinical['group']==1]

mtsample = pd.DataFrame(res_int.index)
wholesample['GID'] = wholesample['sample'].str[:-4]

onlymt = pd.merge(wholesample, mtsample, on='GID', how='inner')

mtsamplelist = onlymt['sample'].tolist()

tmm = tmm[mtsamplelist]

#%%

pre = [x for x in samples if 'bfD' in x ]
post = [x for x in samples if 'atD' in x ]

pre_tu = tmm.filter(pre)
post_tu = tmm.filter(post)

pre_tu.columns = pre_tu.columns.str[:-4]
post_tu.columns = post_tu.columns.str[:-4]

final_tu = pd.concat([pre_tu,post_tu])
final_tu['gene'] = final_tu.index


# %%
#^ mannwhitney
num = pre_tu.shape[0]
mw_test = pd.DataFrame()
mw_test['gene'] = final_tu.iloc[:num,-1].to_list()
mw_test['p-val'] = 1
num = pre_tu.shape[0]
samplenum = pre_tu.shape[1]
for i in range(num):
    list_a = final_tu.iloc[i,:samplenum]
    list_b = final_tu.iloc[i+num,:samplenum]
    if all(item == 0 for item in list_a) or all(item == 0 for item in list_b) :
        mw_test.iloc[i,1] = 1
    else:
        mw_test.iloc[i,1] = mannwhitneyu(list_a, list_b, alternative='two-sided')[1]
    # print(mannwhitneyu(list_a, list_b)[1])

final = mw_test[mw_test['p-val']<0.05]
print("TMM DEG ", final.shape[0], "transcripts")
# print(mw_test[mw_test['p-val']<0.05])

# %%
#^ GO enrichment
pcut = final['gene']
pcut = pcut.drop_duplicates()
glist = pcut.squeeze().str.strip().to_list()

enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                gene_sets=['KEGG_2016','KEGG_2021_Human','GO_Biological_Process_2021'], #,'GO_Biological_Process_2021'
                organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                outdir=None, # don't write to disk
)

enrresult = enr.results.sort_values(by=['Adjusted P-value']) 

# %%
#^ GO enrichment result save
enrresult.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/res_TMM_DEG_GOenrichment.csv', index=False)
# %%
