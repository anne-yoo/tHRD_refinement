#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.stats import pearsonr
import random
from scipy.stats import ttest_ind
from scipy.stats import ranksums

#%%
####* PART 1 : BRIP psi comparison  ####

events=['A3','A5','AF','AL','MX','RI','SE'] #A3, A5
for i in range(7):
    event = events[i]
    pre_raw = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/pre-suppa_'+events[i]+'_variable_10.ioe.psi', sep='\t', index_col=0)
    post_raw = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/post-suppa_'+events[i]+'_variable_10.ioe.psi', sep='\t', index_col=0)

    pre_raw.columns = pre_raw.columns.str[:10]
    post_raw.columns = post_raw.columns.str[:10]

    pre_sample = pre_raw.columns.to_list()
    post_sample = post_raw.columns.to_list()
    mutual_samples = set(pre_sample).intersection(set(post_sample))
    m_sl = list(mutual_samples)

    ###for loop
    dpsi = pd.DataFrame(m_sl)
    dpsi.columns = ['sample']
    dpsi['MW_pval'] = 5

    for k in range(len(m_sl)):
        sample = m_sl[k]
        pre = pre_raw[[sample]]
        post = post_raw[[sample]]
        df = pd.concat([pre,post], axis=1)
        df.columns = ['pre','post']
        df['gene_id'] = df.index.str.split(";",1).str[0]
        df = df.dropna()

        #^ check BRIP1
        brip = ['ENSG00000136492.4','MSTRG.49834','MSTRG.49888','MSTRG.49911','MSTRG.49912','MSTRG.49913','MSTRG.49915','MSTRG.49921','MSTRG.49922']
        brip_df = df[df['gene_id'].isin(brip)]

        

        if set(brip_df['pre']) != set(brip_df['post']):
            dpsi.iloc[k,1] = mannwhitneyu(brip_df['pre'],brip_df['post'], alternative='two-sided')[1]
        
    f_dpsi = dpsi[dpsi['MW_pval']<0.05]
    print("event:", event, f_dpsi)



#%%
####* PART 2 : BRIP major TU comparison  ####
tu = pd.read_csv("/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/pre_post_TU/final_data/230105_major_prop.txt", sep='\t', index_col=0) #major TU

samples = tu.columns.to_list()
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
tu.columns = samples


tu = tu.drop(["SV-OV-P080-atD","SV-OV-P250-atD","SV-OV-P055-atD",\
            "SV-OV-P143-atD","SV-OV-P137-atD","SV-OV-P134-atD",\
            "SV-OV-P174-bfD","SV-OV-P164-atD"], axis=1)

samples = tu.columns


pre_list = [x for x in samples if 'bfD' in x ]
post_list = [x for x in samples if 'atD' in x ]

pre_tu = tu.filter(pre_list)
post_tu = tu.filter(post_list)

pre_tu.columns = pre_tu.columns.str[:10]
post_tu.columns = post_tu.columns.str[:10]

pre_list = [w[:10] for w in pre_list]
post_list = [w[:10] for w in post_list]

mutual_samplelist = set(pre_list).intersection(set(post_list))
sl = list(mutual_samplelist)


briptu = pd.DataFrame(sl)
briptu.columns = ['sample']
briptu['pre'] = 'pre'
briptu['post'] = 'post'

for k in range(len(sl)):
    sample = sl[k]
    pre = pre_tu[[sample]]
    post = post_tu[[sample]]
    tudf = pd.concat([pre,post], axis=1)
    tudf.columns = ['pre','post']
    tudf['gene_id'] = tudf.index.str.split("-",1).str[1]
    brip_1 = tudf[tudf['gene_id']=='BRIP1']
    briptu.iloc[k,1] = brip_1.iloc[0,0]
    briptu.iloc[k,2] = brip_1.iloc[0,1]


#%%
#^ catplot
precat = briptu[['sample','pre']]
postcat = briptu[['sample','post']]

briptu['delta'] = briptu['post'] - briptu['pre']

precat['pre/post'] = 'pre'
postcat['pre/post'] = 'post'

precat.columns = ['sample','BRIP major TU', 'pre/post']
postcat.columns = ['sample','BRIP major TU', 'pre/post']

finalcat = pd.concat([precat,postcat])
finalcat['delta'] = 0
finalcat.iloc[:20,3] = briptu['delta']
finalcat.iloc[20:,3] = briptu['delta']
finalcat['up/down'] = 'down'
finalcat.loc[finalcat['delta']>0,'up/down'] = 'up'

sns.set(rc = {'figure.figsize':(4,4)})
sns.set_style('whitegrid')
sns.catplot(x="pre/post", y="BRIP major TU",hue="sample", kind="point", data=finalcat[finalcat['up/down']=='down'], palette='tab10', height = 3.6, aspect = 1.2)
plt.suptitle('<pre vs. post BRIP1 major TU>', y=1.03, size=11)
plt.show()


#%%
#^ merge clinical info
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
clinical['sample'] = clinical.index

cl_briptu = pd.merge(briptu, clinical, how='inner', on='sample')


sns.set(rc = {'figure.figsize':(4,4)})
sns.set_style('whitegrid')
predf = cl_briptu[['sample','pre','group']]
postdf = cl_briptu[['sample','post','group']]

predf['pre/post'] = 'pre'
postdf['pre/post'] = 'post'

predf.columns = ['sample','BRIP major TU', 'group', 'pre/post']
postdf.columns = ['sample','BRIP major TU', 'group', 'pre/post']

finaldf = pd.concat([predf,postdf])

finaldf['group'] = finaldf['group'].replace([1], 'Responder')
finaldf['group'] = finaldf['group'].replace([0], 'Non-Responder')
sns.set_style('whitegrid')
sns.boxplot(data=finaldf, x="group", y="BRIP major TU", hue="pre/post")
plt.suptitle('<pre vs. post BRIP1 major TU>', y=1.03, size=11)

plt.show()





# %%
####* PART 3 : BRIP major/minor whole TU comparison  ####

tu = pd.read_csv("/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/pre_post_TU/final_data/minor_prop.txt", sep='\t', index_col=0) #whole TU
samples = tu.columns.to_list()
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
tu.columns = samples


tu = tu.drop(["SV-OV-P080-atD","SV-OV-P250-atD","SV-OV-P055-atD",\
            "SV-OV-P143-atD","SV-OV-P137-atD","SV-OV-P134-atD",\
            "SV-OV-P174-bfD","SV-OV-P164-atD"], axis=1)

samples = tu.columns

tu['gene_id'] = tu.index.str.split("-",1).str[1]
briptu = tu[tu['gene_id']=='BRIP1']
# %%
#^ non-matched RES vs. NR BRIP1 whole TU comparison
pre_list = [x for x in samples if 'bfD' in x ]
post_list = [x for x in samples if 'atD' in x ]

pre_tu = briptu.filter(pre_list)
post_tu = briptu.filter(post_list)

post_stacked = post_tu.stack().reset_index()
pre_stacked = pre_tu.stack().reset_index()

post_stacked.columns = ['gene_ENST','sample','BRIP minor TU']
pre_stacked.columns = ['gene_ENST','sample','BRIP minor TU']

post_stacked['pre/post'] = 'post'
pre_stacked['pre/post'] = 'pre'


final_stacked = pd.concat([pre_stacked, post_stacked])
final_stacked['sample'] = final_stacked['sample'].str[:10]

cl_final = pd.merge(final_stacked, clinical, how='inner', on='sample')
cl_final['group'] = cl_final['group'].replace([1], 'Responder')
cl_final['group'] = cl_final['group'].replace([0], 'Non-Responder')

#%%
#^ 1. mungtaengi box plot
sns.set_style('whitegrid')
sns.boxplot(data=cl_final, x="gene_ENST", y="BRIP minor TU", hue="pre/post", showfliers = False)
plt.suptitle('<pre vs. post BRIP1 minor TU>', y=1.03, size=11)

plt.show()

# %%
#^ 2. subplot with 24 isoforms
isoforms = cl_final['gene_ENST'].unique()
fig, axes = plt.subplots(4, 6, sharex=True, figsize=(35,25))
for i in range(4):
    for j in range(6):
        sns.boxplot(ax=axes[i][j], data=cl_final[cl_final['gene_ENST']==isoforms[i*6+j]], x="group", y="BRIP minor TU", hue="pre/post", showfliers = False)
        axes[i][j].set_title(isoforms[i*6+j])
fig.tight_layout()


# %%
