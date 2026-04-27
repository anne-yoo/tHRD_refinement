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
import re
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.stats.multitest import multipletests
from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42  
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'Helvetica'

plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 13,    # x축 틱 라벨 글꼴 크기

'ytick.labelsize': 13,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 13,
'legend.title_fontsize': 13, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})
sns.set_style("ticks")

# %%
#####^^^ Paired Wilcoxon DEG ####

geneexp = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_80_gene_TPM.txt',sep='\t', index_col=0)
geneexp['gene_name'] = geneexp.index
f_gene = geneexp.iloc[:,:-1]
genesym = geneexp[['gene_name']]

deg_pval = []
for index, row in f_gene.iterrows():
    pre_samples = row[1::2].values  # Even-indexed columns are pre-treatment samples
    post_samples = row[::2].values  # Odd-indexed columns are post-treatment samples
    
    # Perform the paired Wilcoxon test and store the p-value
    #w, p = stats.wilcoxon(post_samples, pre_samples)
    if set([a - b for a, b in zip(pre_samples, post_samples)]) != {0}: 
        w, p = stats.wilcoxon(post_samples, pre_samples)
        deg_pval.append(p)
    else:
        deg_pval.append(1)
    
# Adjust for multiple testing using the Benjamini-Hochberg procedure
p_adjusted = multipletests(deg_pval, method='fdr_bh')[1]

# Create a new DataFrame with geneid and respective p-values
result_df = pd.DataFrame({
    'p_value': deg_pval,
    'adj_p' : p_adjusted
})
result_df.index = f_gene.index
result_df = pd.merge(result_df,genesym, how='inner', left_index=True, right_index=True)

### ^^ add FC ###

avg_pre = f_gene.iloc[:, 1::2].mean(axis=1)
avg_post = f_gene.iloc[:, ::2].mean(axis=1)

# Calculate the fold change as the log2 ratio of average post-treatment to pre-treatment expression
fold_change = np.log2(avg_post / avg_pre)
result_df['log2FC'] = fold_change

#result_df.to_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_analysis/wilcoxon_DEGresult_FC.txt', sep='\t')


#%%
# ####^ whole genes: DUT ####
# #transexp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt',sep='\t', index_col=0)
# transexp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_transcript_exp_symbol.txt',sep='\t', index_col=0)
# filtered_trans = transexp.iloc[:,:-1]
# #filtered_trans = transexp

# dut_pval = []
# for index, row in filtered_trans.iterrows():
#     pre_samples = row[1::2].values  # Even-indexed columns are pre-treatment samples
#     post_samples = row[::2].values  # Odd-indexed columns are post-treatment samples
    
#     # Perform the paired Wilcoxon test and store the p-value
#     if set([a - b for a, b in zip(pre_samples, post_samples)]) != {0}: 
#         #w, p = stats.wilcoxon(post_samples, pre_samples)
#         w, p = stats.wilcoxon(post_samples, pre_samples)
#         dut_pval.append(p)
#     else:
#         dut_pval.append(1)


# dut_p_adjusted = multipletests(dut_pval, method='fdr_bh')[1]

# # Create a new DataFrame with geneid and respective p-values
# dut_result = pd.DataFrame({
#     'p_value': dut_pval,
#     'adj_p'  : dut_p_adjusted
# })

# dut_result.index = filtered_trans.index
# dut_result['Gene Symbol'] = dut_result.index.str.split("-",1).str[1]

# avg_pre = filtered_trans.iloc[:, 1::2].mean(axis=1)
# avg_post = filtered_trans.iloc[:, ::2].mean(axis=1)

# # Calculate the fold change as the log2 ratio of average post-treatment to pre-treatment expression
# fold_change = np.log2(avg_post / avg_pre)
# dut_result['log2FC'] = fold_change

# #%%%

# ###* DEG vs. DET venn diagram ###

# from matplotlib_venn import venn2

# deglist = set(result_df[result_df['p_value'] < 0.05]['Gene Symbol'])
# detlist = set(dut_result[dut_result['p_value'] <0.01]['Gene Symbol'])


# plt.figure(figsize=(6,6))
# sns.set_style("white")
# vd2 = venn2([deglist, detlist],set_labels=('DEG', 'DET'),set_colors=('#F8CF6A','#74B2D4'), alpha=0.8)

# for text in vd2.set_labels:  #change label size
#     text.set_fontsize(13)
# for text in vd2.subset_labels:  #change number size
#     text.set_fontsize(16)

# vd2.get_patch_by_id('11').set_color('#C8C452')
# vd2.get_patch_by_id('11').set_alpha(1)

# plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202312_analysis/Wilcoxon_DEG_DET_Venn.pdf", bbox_inches="tight")
# plt.show()


#%%
####^ variable genes: DUT ####
transexp = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_80_transcript_TPM.txt',sep='\t', index_col=0)
transexp = transexp.iloc[:,:-1]
transexp = transexp.loc[(transexp > 0).sum(axis=1) >= transexp.shape[1]*0.3]
transexp["gene"] = transexp.index.str.split("-", n=1).str[-1]
gene_sum = transexp.groupby("gene").transform("sum")
filtered_trans = transexp.iloc[:, :-1].div(gene_sum.iloc[:, :-1])
# filtered_trans.fillna(0, inplace=True)

#%%
filtered_trans['gene_name'] = filtered_trans.index.str.split("-",n=1).str[1]
DEGlist = result_df[result_df['p_value'] < 0.05]['gene_name']
nonDEGlist = result_df[result_df['p_value'] > 0.05]['gene_name']


variable_trans = filtered_trans[filtered_trans['gene_name'].isin(DEGlist)]
variable_trans = variable_trans.iloc[:,:-1]


variable_dut_pval = []
for index, row in variable_trans.iterrows():
    pre_samples = row[1::2].values  # Even-indexed columns are pre-treatment samples
    post_samples = row[::2].values  # Odd-indexed columns are post-treatment samples
    
    # Perform the paired Wilcoxon test and store the p-value
    if set([a - b for a, b in zip(pre_samples, post_samples)]) != {0}: 
        w, p = stats.wilcoxon(post_samples, pre_samples)
        variable_dut_pval.append(p)
    else:
        variable_dut_pval.append(1)
    

# Create a new DataFrame with geneid and respective p-values
variable_result = pd.DataFrame({
    'p_value':variable_dut_pval,
})
variable_result.index = variable_trans.index
variable_result['gene_name'] = variable_result.index.str.split("-",n=1).str[1]

variable_DUT = variable_result[variable_result['p_value'] < 0.05]


####^ stable genes: DUT ####
stable_trans = filtered_trans[filtered_trans['gene_name'].isin(nonDEGlist)]
stable_trans = stable_trans.iloc[:,:-1]


stable_dut_pval = []
for index, row in stable_trans.iterrows():
    pre_samples = row[1::2].values  # Even-indexed columns are pre-treatment samples
    post_samples = row[::2].values  # Odd-indexed columns are post-treatment samples
    
    # Perform the paired Wilcoxon test and store the p-value
    if set([a - b for a, b in zip(pre_samples, post_samples)]) != {0}: 
        w, p = stats.wilcoxon(post_samples, pre_samples)
        stable_dut_pval.append(p)
    else:
        stable_dut_pval.append(1)
    

# Create a new DataFrame with geneid and respective p-values
stable_result = pd.DataFrame({
    'p_value': stable_dut_pval,
})
stable_result.index = stable_trans.index
stable_result['gene_name'] = stable_result.index.str.split("-",n=1).str[1]

stable_DUT = stable_result[stable_result['p_value'] < 0.05]

variable_result.to_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_analysis/variable_DUT_Wilcoxon.txt', sep='\t')
stable_result.to_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_analysis/stable_DUT_Wilcoxon.txt', sep='\t')

# %%
####** save #####
#degs.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202309_analysis/DEG/whole_Wilcoxon_DEG.txt', sep='\t')
#deseq_degs.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202309_analysis/DEG/whole_DESeq2_DEG.txt', sep='\t')
#variable_DUT.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202309_analysis/DUT/whole_variable_DUT_Wilcoxon.txt', sep='\t')
#stable_DUT.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202309_analysis/DUT/whole_stable_DUT_Wilcoxon.txt', sep='\t')

# result_df.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202312_analysis/DEG/whole_Wilcoxon_DEGresult_FC.txt', sep='\t')
# dut_result.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202312_analysis/DET/whole_Wilcoxon_DETresult_FC.txt', sep='\t')





# %%
################^^#####################################################################################
###############^ Responder / Nonresponder #############################################################
##################^^###################################################################################

#####^^^ Paired Wilcoxon DEG responder / non-responder ####
proteincoding = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM_symbol.txt', sep='\t', index_col=0)
proteincodinglist = proteincoding['Gene Symbol'].to_list()
geneexp = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_80_gene_TPM.txt',sep='\t', index_col=0)
geneexp['gene_name'] = geneexp.index
geneexp = geneexp[geneexp.index.isin(proteincodinglist)]
sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost_80_clinicalinfo.txt', sep='\t', index_col=0)

#%%
#sampleinfo = sampleinfo[sampleinfo['purpose']=='maintenance']

responder = sampleinfo[sampleinfo['response']==1]['sample_full']
nonresponder = sampleinfo[sampleinfo['response']==0]['sample_full']
responder = responder.to_list()
nonresponder = nonresponder.to_list()

list = [responder, nonresponder]
namelist = ['AR', 'IR']

#%%
for i in range(2): 
    f_gene = geneexp[list[i]]
    genesym = geneexp[['gene_name']]
    

    deg_pval = []
    for index, row in f_gene.iterrows():
        pre_samples = row[1::2].values  # Even-indexed columns are pre-treatment samples
        post_samples = row[::2].values  # Odd-indexed columns are post-treatment samples
        
        # Perform the paired Wilcoxon test and store the p-value
        if set([a - b for a, b in zip(pre_samples, post_samples)]) != {0}: 
            w, p = stats.wilcoxon(post_samples, pre_samples,zero_method="pratt")
            deg_pval.append(p)
        else:
            deg_pval.append(1)
        
    # Adjust for multiple testing using the Benjamini-Hochberg procedure
    p_adjusted = multipletests(deg_pval, method='fdr_bh')[1]

    # Create a new DataFrame with geneid and respective p-values
    result_df = pd.DataFrame({
        'p_value': deg_pval,
        'adj_p' : p_adjusted,
    })
    result_df.index = f_gene.index
    result_df = pd.merge(result_df,genesym, how='inner', left_index=True, right_index=True)

    degs = result_df[result_df['p_value'] < 0.05]
    
    ## FC
    avg_pre = f_gene.iloc[:, 1::2].mean(axis=1)
    avg_post = f_gene.iloc[:, ::2].mean(axis=1)

    # Calculate the fold change as the log2 ratio of average post-treatment to pre-treatment expression
    fold_change = np.log2(avg_post / avg_pre)
    result_df['log2FC'] = fold_change
        
    result_df.to_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/whole_'+ namelist[i]+'_Wilcoxon_DEGresult_FC.txt', sep='\t')



# %%
#####^^^ Paired Wilcoxon DUT responder / non-responder ####
transexp = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_80_transcript_TPM.txt',sep='\t', index_col=0)
transexp = transexp[transexp['gene_name'].isin(proteincodinglist)]
##
transexp = transexp[[col for col in transexp.columns if col in responder + nonresponder]]
##

#transexp = transexp.iloc[:,:-1]
###
transexp = transexp.loc[(transexp > 0).sum(axis=1) >= 14] #14 #3
###
transexp["gene"] = transexp.index.str.split("-", n=1).str[-1]
gene_sum = transexp.groupby("gene").transform("sum")
filtered_trans = transexp.iloc[:, :-1].div(gene_sum.iloc[:, :-1])
#filtered_trans.fillna(0, inplace=True)
filtered_trans['gene_name'] = filtered_trans.index.str.split("-",n=1).str[1]

#%%
list = [responder, nonresponder]
namelist = ['AR', 'IR']

for i in range(2):
    degresult = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/whole_'+namelist[i]+'_Wilcoxon_DEGresult_FC.txt', sep='\t')
    
    DEGlist = set(degresult[(degresult['p_value']<0.05) & (np.abs(degresult['log2FC'])>1)]['gene_name'])
    nonDEGlist = set(degresult['gene_name']) - DEGlist
    
    print(namelist[i]," variable: ", len(DEGlist))
    print(namelist[i]," stable: ", len(nonDEGlist))

    variable_trans = filtered_trans[filtered_trans['gene_name'].isin(DEGlist)]
    variable_trans = variable_trans[list[i]]

    variable_dut_pval = []
    
    for index, row in variable_trans.iterrows():
        pre_samples = row[1::2].values
        post_samples = row[::2].values

        # Identify valid pairs: both pre and post must be non-NA and gene detected in both
        valid_mask = ~np.isnan(pre_samples) & ~np.isnan(post_samples)

        valid_pre = pre_samples[valid_mask]
        valid_post = post_samples[valid_mask]

        # If no valid pair exists → no test
        if len(valid_pre) == 0:
            variable_dut_pval.append(1.0)
            continue

        # If all values identical → no signal
        if np.all(valid_post - valid_pre == 0):
            variable_dut_pval.append(1.0)
            continue

        # Wilcoxon paired test
        w, p = stats.wilcoxon(valid_post, valid_pre, zero_method="pratt")
        variable_dut_pval.append(p)

    # Create a new DataFrame with geneid and respective p-values
    variable_result = pd.DataFrame({
        'p_value':variable_dut_pval,
    })
    variable_result.index = variable_trans.index
    variable_result['gene_name'] = variable_result.index.str.split("-",n=1).str[1]

    ##### FC #####
    avg_pre = variable_trans.iloc[:, 1::2].mean(axis=1)
    avg_post = variable_trans.iloc[:, ::2].mean(axis=1)

    fold_change = np.log2(avg_post / avg_pre)
    variable_result['log2FC'] = fold_change
    ##############
    
    ##### ΔTU #####
    delta_TU = avg_post - avg_pre
    variable_result['delta_TU'] = delta_TU
    ###############
    
    

    ####^ stable genes: DUT ####
    stable_trans = filtered_trans[filtered_trans['gene_name'].isin(nonDEGlist)]
    stable_trans = stable_trans[list[i]]

    stable_dut_pval = []
    
    for index, row in stable_trans.iterrows():
            pre_samples = row[1::2].values
            post_samples = row[::2].values

            # Identify valid pairs: both pre and post must be non-NA and gene detected in both
            valid_mask = ~np.isnan(pre_samples) & ~np.isnan(post_samples)

            valid_pre = pre_samples[valid_mask]
            valid_post = post_samples[valid_mask]

            # If no valid pair exists → no test
            if len(valid_pre) == 0:
                stable_dut_pval.append(1.0)
                continue

            # If all values identical → no signal
            if np.all(valid_post - valid_pre == 0):
                stable_dut_pval.append(1.0)
                continue

            # Wilcoxon paired test
            w, p = stats.wilcoxon(valid_post, valid_pre, zero_method="pratt")
            stable_dut_pval.append(p)

    # Create a new DataFrame with geneid and respective p-values
    stable_result = pd.DataFrame({
        'p_value': stable_dut_pval,
    })
    stable_result.index = stable_trans.index
    stable_result['gene_name'] = stable_result.index.str.split("-",n=1).str[1]
    
    ##### FC #####
    avg_pre = stable_trans.iloc[:, 1::2].mean(axis=1)
    avg_post = stable_trans.iloc[:, ::2].mean(axis=1)

    fold_change = np.log2(avg_post / avg_pre)
    stable_result['log2FC'] = fold_change
    ##############
    
    ##### ΔTU #####
    delta_TU = avg_post - avg_pre
    stable_result['delta_TU'] = delta_TU
    ###############
    
    #############
    variable_DUT = variable_result[variable_result['p_value'] < 0.05]['gene_name']
    stable_DUT = stable_result[stable_result['p_value'] < 0.05]['gene_name']
    
    print('variable DUT: ', len(variable_DUT))
    print('stable DUT: ', len(stable_DUT))
    
    variable_result.to_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/whole_'+namelist[i]+'_variable_DUT_Wilcoxon_delta_withna.txt', sep='\t')
    stable_result.to_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/whole_'+namelist[i]+'_stable_DUT_Wilcoxon_delta_withna.txt', sep='\t')

stable_result
#%%
########** Responder vs. Nonresponder DUT venn diagram ##########
from matplotlib_venn import venn2

r_variable = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/maintenance/AR_variable_DUT_Wilcoxon_delta_withna.txt', sep='\t', index_col=0)
nr_variable = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/maintenance/IR_variable_DUT_Wilcoxon_delta_withna.txt', sep='\t', index_col=0)

dut1 = set(r_variable[(r_variable['p_value']<0.05) & (np.abs(r_variable['delta_TU'])>0.01)].index)
dut2 = set(nr_variable[(nr_variable['p_value']<0.05) & (np.abs(nr_variable['delta_TU'])>0.01)].index)

plt.figure(figsize=(6,6))
sns.set_style("white")
vd2 = venn2([dut1, dut2],set_labels=('AR', 'IR'),set_colors=('#F8CF6A','#74B2D4'), alpha=0.8)

for text in vd2.set_labels:  #change label size
    text.set_fontsize(13)
for text in vd2.subset_labels:  #change number size
    text.set_fontsize(16)
    
vd2.get_patch_by_id('11').set_color('#C8C452')
vd2.get_patch_by_id('11').set_alpha(1)

plt.savefig("/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/maintenance/figures/001_variable_AR_IR_DUT_Venn.pdf", bbox_inches="tight")
plt.show()



# %%
# %%
#####^^^ Paired Wilcoxon DUT responder / non-responder WITHOUT GENE STABLE/VARIABLE GROUPING!! ####

list = [responder, nonresponder]
namelist = ['responder', 'nonresponder']

for i in range(2):
    degresult = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202309_analysis/DEG/'+namelist[i]+'_Wilcoxon_DEGresult_FC.txt', sep='\t')
    
    DEGlist = set(degresult[degresult['p_value']<0.05]['Gene Symbol'])
    nonDEGlist = set(degresult[degresult['p_value'] > 0.05]['Gene Symbol'])

    transexp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_transcript_exp.txt',sep='\t', index_col=0)
    filtered_trans = transexp.loc[(transexp > 0).sum(axis=1) >= transexp.shape[1]*0.6]
    filtered_trans['Gene Symbol'] = filtered_trans.index.str.split("-",1).str[1]
    filtered_trans = filtered_trans[filtered_trans['Gene Symbol']!= '-']

    
    ####^ whole genes: DUT ####
    stable_trans = filtered_trans
    stable_trans = stable_trans[list[i]]

    stable_dut_pval = []
    
    for index, row in stable_trans.iterrows():
        pre_samples = row[1::2].values  # Even-indexed columns are pre-treatment samples
        post_samples = row[::2].values  # Odd-indexed columns are post-treatment samples
        
        # Perform the paired Wilcoxon test and store the p-value
        w, p = stats.wilcoxon(post_samples, pre_samples)
        stable_dut_pval.append(p)
        

    # Create a new DataFrame with geneid and respective p-values
    stable_result = pd.DataFrame({
        'p_value': stable_dut_pval,
    })
    stable_result.index = stable_trans.index
    stable_result['Gene Symbol'] = stable_result.index.str.split("-",1).str[1]
    
    ##### FC #####
    avg_pre = stable_trans.iloc[:, 1::2].mean(axis=1)
    avg_post = stable_trans.iloc[:, ::2].mean(axis=1)

    fold_change = np.log2(avg_post / avg_pre)
    stable_result['log2FC'] = fold_change
    ##############
    
    
    #############
    
    stable_DUT = stable_result[stable_result['p_value'] < 0.05]['Gene Symbol']
    
    
    print('stable DUT: ', len(stable_DUT))
    
    
    stable_result.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202310_analysis/DUT/'+namelist[i]+'_wholegene_DUT_Wilcoxon.txt', sep='\t')


# %%
#####^^^ Baseline AR pre vs IR pre DUT with stable / variable grouping ####

baseline_sampleinfo = pd.read_csv(
    '/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost_80_clinicalinfo.txt',
    sep='\t',
    index_col=0
)

baseline_ar_pre = baseline_sampleinfo.loc[
    (baseline_sampleinfo['response'] == 1)
    & baseline_sampleinfo['sample_full'].str.contains(r'-bfD$', case=False, na=False),
    'sample_full'
].tolist()

baseline_ir_pre = baseline_sampleinfo.loc[
    (baseline_sampleinfo['response'] == 0)
    & baseline_sampleinfo['sample_full'].str.contains(r'-bfD$', case=False, na=False),
    'sample_full'
].tolist()

baseline_pre_cols = baseline_ar_pre + baseline_ir_pre


def run_baseline_mwu(expr_df, ar_cols, ir_cols):
    p_values = []
    mean_ar = []
    mean_ir = []

    for _, row in expr_df.iterrows():
        ar_vals = row[ar_cols].astype(float).to_numpy()
        ir_vals = row[ir_cols].astype(float).to_numpy()

        ar_valid = ar_vals[~np.isnan(ar_vals)]
        ir_valid = ir_vals[~np.isnan(ir_vals)]

        mean_ar.append(np.nan if len(ar_valid) == 0 else ar_valid.mean())
        mean_ir.append(np.nan if len(ir_valid) == 0 else ir_valid.mean())

        if len(ar_valid) == 0 or len(ir_valid) == 0:
            p_values.append(1.0)
            continue

        if (
            np.all(ar_valid == ar_valid[0])
            and np.all(ir_valid == ir_valid[0])
            and ar_valid[0] == ir_valid[0]
        ):
            p_values.append(1.0)
            continue

        try:
            _, p = stats.mannwhitneyu(ar_valid, ir_valid, alternative='two-sided')
        except ValueError:
            p = 1.0

        p_values.append(p)

    return p_values, mean_ar, mean_ir


def safe_log2_ratio(numerator, denominator):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log2(numerator / denominator)


baseline_geneexp = pd.read_csv(
    '/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_80_gene_TPM.txt',
    sep='\t',
    index_col=0
)
baseline_geneexp = baseline_geneexp[baseline_geneexp.index.isin(proteincodinglist)]
baseline_gene_cols = [c for c in baseline_pre_cols if c in baseline_geneexp.columns]
baseline_gene_ar_cols = [c for c in baseline_ar_pre if c in baseline_gene_cols]
baseline_gene_ir_cols = [c for c in baseline_ir_pre if c in baseline_gene_cols]
baseline_geneexp = baseline_geneexp[baseline_gene_cols]

baseline_deg_pval, baseline_gene_mean_ar, baseline_gene_mean_ir = run_baseline_mwu(
    baseline_geneexp,
    baseline_gene_ar_cols,
    baseline_gene_ir_cols
)

baseline_deg_adj = multipletests(baseline_deg_pval, method='fdr_bh')[1]
baseline_deg_result = pd.DataFrame({
    'p_value': baseline_deg_pval,
    'adj_p': baseline_deg_adj,
    'gene_name': baseline_geneexp.index,
    'mean_AR_pre': baseline_gene_mean_ar,
    'mean_IR_pre': baseline_gene_mean_ir,
})
baseline_deg_result.index = baseline_geneexp.index
baseline_deg_result['log2FC'] = safe_log2_ratio(
    baseline_deg_result['mean_AR_pre'],
    baseline_deg_result['mean_IR_pre']
)
baseline_deg_result['delta_exp'] = (
    baseline_deg_result['mean_AR_pre'] - baseline_deg_result['mean_IR_pre']
)

# stable / variable split uses p-value only
baseline_variable_genes = set(
    baseline_deg_result.loc[baseline_deg_result['p_value'] < 0.05, 'gene_name']
)
baseline_stable_genes = set(baseline_deg_result['gene_name']) - baseline_variable_genes

print('Baseline variable genes:', len(baseline_variable_genes))
print('Baseline stable genes:', len(baseline_stable_genes))

baseline_deg_result.to_csv(
    '/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/whole_baseline_ARpre_vs_IRpre_MannWhitney_DEGresult_FC.txt',
    sep='\t'
)


baseline_transexp = pd.read_csv(
    '/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_80_transcript_TPM.txt',
    sep='\t',
    index_col=0
)
baseline_transexp = baseline_transexp[baseline_transexp['gene_name'].isin(proteincodinglist)].copy()

baseline_tx_cols = [c for c in baseline_pre_cols if c in baseline_transexp.columns]
baseline_tx_ar_cols = [c for c in baseline_ar_pre if c in baseline_tx_cols]
baseline_tx_ir_cols = [c for c in baseline_ir_pre if c in baseline_tx_cols]

baseline_detect_min = max(1, int(np.ceil(len(baseline_tx_cols) * 0.3)))
baseline_detect_mask = (baseline_transexp[baseline_tx_cols] > 0).sum(axis=1) >= baseline_detect_min
baseline_transexp = baseline_transexp.loc[baseline_detect_mask].copy()

baseline_transexp['gene'] = baseline_transexp.index.str.split("-", n=1).str[-1]
baseline_gene_sum = baseline_transexp.groupby('gene')[baseline_tx_cols].transform('sum')
baseline_filtered_trans = baseline_transexp[baseline_tx_cols].div(baseline_gene_sum)
baseline_filtered_trans['gene_name'] = baseline_transexp['gene_name'].values


def build_baseline_dut_result(trans_df, gene_set, ar_cols, ir_cols):
    sub = trans_df[trans_df['gene_name'].isin(gene_set)].copy()
    expr_only = sub[ar_cols + ir_cols].copy()

    dut_pval, mean_ar, mean_ir = run_baseline_mwu(expr_only, ar_cols, ir_cols)
    dut_adj = multipletests(dut_pval, method='fdr_bh')[1]

    result = pd.DataFrame({
        'p_value': dut_pval,
        'adj_p': dut_adj,
        'gene_name': sub['gene_name'].values,
        'mean_AR_pre': mean_ar,
        'mean_IR_pre': mean_ir,
    }, index=sub.index)

    result['log2FC'] = safe_log2_ratio(result['mean_AR_pre'], result['mean_IR_pre'])
    result['delta_TU'] = result['mean_AR_pre'] - result['mean_IR_pre']
    return result


baseline_variable_result = build_baseline_dut_result(
    baseline_filtered_trans,
    baseline_variable_genes,
    baseline_tx_ar_cols,
    baseline_tx_ir_cols
)

baseline_stable_result = build_baseline_dut_result(
    baseline_filtered_trans,
    baseline_stable_genes,
    baseline_tx_ar_cols,
    baseline_tx_ir_cols
)

print('Baseline variable DUT:', (baseline_variable_result['p_value'] < 0.05).sum())
print('Baseline stable DUT:', (baseline_stable_result['p_value'] < 0.05).sum())

baseline_variable_result.to_csv(
    '/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/whole_baseline_ARpre_vs_IRpre_variable_DUT_MannWhitney_delta_withna.txt',
    sep='\t'
)
baseline_stable_result.to_csv(
    '/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/whole_baseline_ARpre_vs_IRpre_stable_DUT_MannWhitney_delta_withna.txt',
    sep='\t'
)

# %%

