#%%
#! this code is.....
"""
1.major DUT clustermap: stable/variable gene / pre vs. post
2.minor DUT clustermap: stable/variable gene / pre vs. post 
"""
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
import matplotlib.cm as cm
from matplotlib.pyplot import gcf

# %%
##* data preparation for heatmap
degresult = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DEG/whole_Wilcoxon_DEGresult_FC.txt', sep='\t')
DEGlist = set(degresult[degresult['p_value']<0.05]['Gene Symbol'])
nonDEGlist = set(degresult[degresult['p_value'] > 0.05]['Gene Symbol'])
# %%
dutresult = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/whole_Wilcoxon_DUTresult_FC.txt', sep='\t')
dut = dutresult[dutresult['p_value']<0.05]
dutlist = dut['gene_ENST'].tolist()

transexp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt',sep='\t', index_col=0)
transexp['Gene Symbol'] = transexp.index.str.split("-",1).str[1]
transexp = transexp[transexp['Gene Symbol']!='-']

dut_tu = transexp[transexp.index.isin(dutlist)]

major = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/final_202306_discovery_major_TU.txt', sep='\t', index_col=0)
minor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/final_202306_discovery_minor_TU.txt', sep='\t', index_col=0)

majortrans = major.index.tolist()
minortrans = minor.index.tolist()

dut_tu['type'] = dut_tu.index.to_series().apply(lambda x: 'major' if x in majortrans else ('minor' if x in minortrans else None))

#%%
major_dut = dut_tu[dut_tu['type']=='major']
minor_dut = dut_tu[dut_tu['type']=='minor']

major_dut['gene'] = major_dut['Gene Symbol'].apply(lambda x: 'stable' if x in nonDEGlist else ('variable' if x in DEGlist else None))
minor_dut['gene'] = minor_dut['Gene Symbol'].apply(lambda x: 'stable' if x in nonDEGlist else ('variable' if x in DEGlist else None))

major_stable = major_dut[major_dut['gene']=='stable']
major_variable = major_dut[major_dut['gene']=='variable']

minor_stable = minor_dut[minor_dut['gene']=='stable']
minor_variable = minor_dut[minor_dut['gene']=='variable']

major_plot = pd.concat([major_stable,major_variable],axis=0)
minor_plot = pd.concat([minor_stable,minor_variable],axis=0)



# %%
##* major dut heatmap
sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.csv', sep=',')
major_info = major_plot['gene']

major_data = major_plot.iloc[:,:-3]
major_zscore = major_data.apply(lambda x: (x-x.mean())/x.std(), axis = 1)

#^ color mapping
colors_dict = { "pre":"#4998FF", "post":"#FF4949"}
colors_dict2 = { "stable":"#689111", "variable":"#E7961D"}
col_colors = sampleinfo['treatment'].map(colors_dict)
row_colors = major_plot['gene'].map(colors_dict2)

##^ draw seaborn clustermap
# methods = ['single', 'complete', 'average', 'weighted', 'ward'] #5
methods = ['ward']
metrics = ['cityblock', 'minkowski', 'seuclidean', 'cosine', 'correlation', 'hamming', 'jaccard', 'chebyshev', 'canberra', 'dice', 'rogerstanimoto', 'russellrao', 'sokalsneath'] #13

for i in range(1):
    
    method = 'ward'

    g = sns.clustermap(major_zscore, col_colors=[col_colors],row_colors=[row_colors], cmap="RdBu_r" , figsize=(6,9),vmin=-2,vmax=2,center=0,
                    method=method, #single, complete, average, weighted, ward
                    metric='euclidean', #cityblock, minkowski, euclidean, cosine, correlation, hamming, jaccard, chebyshev, canberra, dice, rogerstanimoto, russellrao, sokalsneath
                    #z_score=0,
                    #vmax=1, vmin=0,
                    #standard_scale=0,
                        linewidths=0, xticklabels=False, yticklabels=False, col_cluster=True, row_cluster=True)


    for group in  sampleinfo['treatment'].unique():
        g.ax_col_dendrogram.bar(0, 0, color=colors_dict[group],
                                label=group, linewidth=0)

    l1 = g.ax_col_dendrogram.legend(title='sample type', loc="center", ncol=2, bbox_to_anchor=(0.57, 1.01), bbox_transform=gcf().transFigure)

    # for event in event_info['event'].unique():
    #     g.ax_row_dendrogram.bar(0, 0, color=colors_dict2[event], label=event, linewidth=0);

    # l2 = g.ax_row_dendrogram.legend(title='Dataset', loc="center", ncol=2, bbox_to_anchor=(0.57, 1.06), bbox_transform=gcf().transFigure)

    #g.cax.set_position([.15, .2, .03, .45])
    ax = g.ax_heatmap
    # ax.set_xlabel("Samples")
    ax.set_ylabel("")

plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/fig2b_whole_majordut_heatmap.pdf", bbox_inches="tight")


# %%
##* minor dut heatmap
sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.csv', sep=',')
minor_info = minor_plot['gene']

minor_data = minor_plot.iloc[:,:-3]
minor_zscore = minor_data.apply(lambda x: (x-x.mean())/x.std(), axis = 1)

#^ color mapping
colors_dict = { "pre":"#4998FF", "post":"#FF4949"}
colors_dict2 = { "stable":"#689111", "variable":"#E7961D"}
col_colors = sampleinfo['treatment'].map(colors_dict)
row_colors = minor_plot['gene'].map(colors_dict2)

##^ draw seaborn clustermap
# methods = ['single', 'complete', 'average', 'weighted', 'ward'] #5
methods = ['ward']
metrics = ['cityblock', 'minkowski', 'seuclidean', 'cosine', 'correlation', 'hamming', 'jaccard', 'chebyshev', 'canberra', 'dice', 'rogerstanimoto', 'russellrao', 'sokalsneath'] #13

for i in range(1):
    
    method = 'ward'

    g = sns.clustermap(minor_zscore, col_colors=[col_colors],row_colors=[row_colors], cmap="RdBu_r" , figsize=(6,9),vmin=-2,vmax=2,center=0,
                    method=method, #single, complete, average, weighted, ward
                    metric='euclidean', #cityblock, minkowski, euclidean, cosine, correlation, hamming, jaccard, chebyshev, canberra, dice, rogerstanimoto, russellrao, sokalsneath
                    #z_score=0,
                    #vmax=1, vmin=0,
                    #standard_scale=0,
                        linewidths=0, xticklabels=False, yticklabels=False, col_cluster=True, row_cluster=True)


    for group in  sampleinfo['treatment'].unique():
        g.ax_col_dendrogram.bar(0, 0, color=colors_dict[group],
                                label=group, linewidth=0)

    l1 = g.ax_col_dendrogram.legend(title='sample type', loc="center", ncol=2, bbox_to_anchor=(0.57, 1.01), bbox_transform=gcf().transFigure)

    # for event in event_info['event'].unique():
    #     g.ax_row_dendrogram.bar(0, 0, color=colors_dict2[event], label=event, linewidth=0);

    # l2 = g.ax_row_dendrogram.legend(title='Dataset', loc="center", ncol=2, bbox_to_anchor=(0.57, 1.06), bbox_transform=gcf().transFigure)

    #g.cax.set_position([.15, .2, .03, .45])
    ax = g.ax_heatmap
    # ax.set_xlabel("Samples")
    ax.set_ylabel("")
plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/fig2b_whole_minordut_heatmap.pdf", bbox_inches="tight")




# %%
#######** gene group Venn Diagram #########
ARdeg = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DEG/responder_Wilcoxon_DEGresult_FC.txt', sep='\t')
IRdeg = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DEG/nonresponder_Wilcoxon_DEGresult_FC.txt', sep='\t')

AR_variable = set(ARdeg[ARdeg['p_value']<0.05]['Gene Symbol'])
AR_stable = set(ARdeg[ARdeg['p_value']>0.05]['Gene Symbol'])
IR_variable = set(IRdeg[IRdeg['p_value']<0.05]['Gene Symbol'])
IR_stable = set(IRdeg[IRdeg['p_value']>0.05]['Gene Symbol'])

AR_stable.remove('ZFP41')


from venny4py.venny4py import *

plt.figure(figsize=(4,4))
sns.set_style("white")

#dict of sets
sets = {
    'AR variable': AR_variable,
    'AR stable': AR_stable,
    'IR variable': IR_variable,
    'IR stable': IR_stable}
    
venny4py(sets=sets)
#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/fig2c_genegroup_venn.pdf", bbox_inches="tight")
plt.show()

# %%
#####* SUPPA2 input #######
tpm = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_transcript_exp.txt',sep='\t',index_col=0)
pre_tpm = tpm[tpm.columns[1::2]]
post_tpm = tpm[tpm.columns[0::2]]

pre_tpm.index = pre_tpm.index.str.split("-",1).str[0]
post_tpm.index = post_tpm.index.str.split("-",1).str[0]

pre_tpm.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/suppa2/pre_TPM.txt', sep='\t', index=True)
post_tpm.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/suppa2/post_TPM.txt', sep='\t', index=True)


# %%
###*### DUT group Venn Diagram ######
ARstable = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t')
ARvariable = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_variable_DUT_Wilcoxon.txt', sep='\t')
IRstable = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/nonresponder_stable_DUT_Wilcoxon.txt', sep='\t')
IRvariable = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/nonresponder_variable_DUT_Wilcoxon.txt', sep='\t')


AR_variable = set(ARvariable[ARvariable['p_value']<0.05]['gene_ENST'])
AR_stable = set(ARstable[ARstable['p_value']<0.05]['gene_ENST'])
IR_variable = set(IRvariable[IRvariable['p_value']<0.05]['gene_ENST'])
IR_stable = set(IRstable[IRstable['p_value']<0.05]['gene_ENST'])

#dict of sets
sets = {
    'AR variable': AR_variable,
    'AR stable': AR_stable,
    'IR variable': IR_variable,
    'IR stable': IR_stable}
from venny4py.venny4py import *
venny4py(sets=sets)
#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/fig2c_DUTgroup_venn.pdf", bbox_inches="tight")
plt.show()

# %%
# %%
###*### baseline DUT group Venn Diagram ######
ARstable = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/baseline/1L_stable_DUT_MW.txt', sep='\t')
ARvariable = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/baseline/1L_variable_DUT_MW.txt', sep='\t')
IRstable = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/baseline/non1L_stable_DUT_MW.txt', sep='\t')
IRvariable = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/baseline/non1L_variable_DUT_MW.txt', sep='\t')


AR_variable = set(ARvariable[ARvariable['p_value']<0.05]['gene_ENST'])
AR_stable = set(ARstable[ARstable['p_value']<0.05]['gene_ENST'])
IR_variable = set(IRvariable[IRvariable['p_value']<0.05]['gene_ENST'])
IR_stable = set(IRstable[IRstable['p_value']<0.05]['gene_ENST'])

#dict of sets
sets = {
    '1L variable': AR_variable,
    '1L stable': AR_stable,
    'non-1L variable': IR_variable,
    'non-1L stable': IR_stable}
    
venny4py(sets=sets)
plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/fig2e_baseline_DUTgroup_venn.pdf", bbox_inches="tight")
plt.show()

# %%
#######** baseline gene group Venn Diagram #########
ARdeg = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DEG/baseline/1L_MW_DEGresult.txt', sep='\t')
IRdeg = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DEG/baseline/non1L_MW_DEGresult.txt', sep='\t')

AR_variable = set(ARdeg[ARdeg['p_value']<0.05]['Gene Symbol'])
AR_stable = set(ARdeg[ARdeg['p_value']>0.05]['Gene Symbol'])
IR_variable = set(IRdeg[IRdeg['p_value']<0.05]['Gene Symbol'])
IR_stable = set(IRdeg[IRdeg['p_value']>0.05]['Gene Symbol'])


from venny4py.venny4py import *

plt.figure(figsize=(4,4))
sns.set_style("white")

#dict of sets
sets = {
    '1L variable': AR_variable,
    '1L stable': AR_stable,
    'non-1L variable': IR_variable,
    'non-1L stable': IR_stable}
    
venny4py(sets=sets)
plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/fig2e_baseline_genegroup_venn.pdf", bbox_inches="tight")
plt.show()








# %%
#######^^^^^ HR gene vs. DEG vs. DET vs. DSG #############
degresult = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202312_analysis/DEG/whole_Wilcoxon_DEGresult_FC.txt', sep='\t')
DEGlist = set(degresult[(degresult['p_value']<0.01)]['Gene Symbol']) # (abs(degresult['log2FC'])>0.5)
detresult = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202312_analysis/DET/whole_Wilcoxon_DETresult_FC.txt', sep='\t')
DETlist = set(detresult[(detresult['p_value']<0.01)]['Gene Symbol']) # (abs(detresult['log2FC'])>1)
dsgresult = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/rMATS/dpsi_analysis/MW_dpsi_5events.txt', sep='\t')
dsgresult = dsgresult[(dsgresult['pval']<0.001) & (dsgresult['d_psi']>0.1)]
DSGlist = set(dsgresult['gene symbol'])
HRlist = set(open('/home/jiye/jiye/copycomparison/gDUTresearch/data/HR_genes.txt').read().splitlines())

from venny4py.venny4py import *

plt.figure(figsize=(4,4))
sns.set_style("white")

#dict of sets
sets = {
    'DEG': DEGlist,
    'DET gene': DETlist,
    'DSG': DSGlist,
    'HRD gene': HRlist}
    
venny4py(sets=sets)
#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/fig1d_venn.pdf", bbox_inches="tight")
plt.show()

# %%
######^^^^^ heatmap: DEG vs. DET vs. DSG #############
geneexp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM_symbol.txt', sep='\t', index_col=0)
transexp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_transcript_exp.txt', sep='\t', index_col=0)
psi = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/rMATS/dpsi_analysis/whole_psi.txt',sep='\t')

genedf = geneexp[geneexp['Gene Symbol'].isin(DEGlist)]
detlist = detresult[(detresult['p_value']<0.01)]['Unnamed: 0'] #(abs(detresult['log2FC'])>1)
transdf = transexp[transexp.index.isin(detlist)]
psidf = psi[psi.index.isin(list(dsgresult.index))]

# %%

geneinput = genedf.iloc[:,:-1]
geneinput = np.log2(geneinput+1)
gene_zscore = geneinput.apply(lambda x: (x-x.mean())/x.std(), axis = 1)

gene_pre = gene_zscore.iloc[:,1::2]
gene_post = gene_zscore.iloc[:,0::2]
gene_zscore = pd.concat([gene_pre,gene_post],axis=1)
gene_zscore.dropna(inplace=True)

#^ color mapping
colors_dict = { "pre":"#4998FF", "post":"#FF4949"}
col_colors = [colors_dict["pre"]] * 40 + [colors_dict["post"]] * 40

plt.figure(figsize=(6,6))
sns.set_style("white")

g = sns.clustermap(gene_zscore, col_colors=[col_colors], cmap="RdBu_r" , figsize=(6,9),vmin=-2,vmax=2,center=0,
                method='ward', #single, complete, average, weighted, ward
                metric='euclidean', #cityblock, minkowski, euclidean, cosine, correlation, hamming, jaccard, chebyshev, canberra, dice, rogerstanimoto, russellrao, sokalsneath
                #z_score=0,
                #vmax=1, vmin=0,
                #standard_scale=0,
                    linewidths=0, xticklabels=False, yticklabels=False, col_cluster=True, row_cluster=True,
                    cbar_pos=(-0.01, 0.7, 0.05, 0.18))

for group in ['pre','post']:
    g.ax_col_dendrogram.bar(0, 0, color=colors_dict[group],
                            label=group, linewidth=0)

l1 = g.ax_col_dendrogram.legend(title='sample type', ncol=2, loc="center")

#l1 = g.ax_col_dendrogram.legend(title='sample type', ncol=2)

# for event in event_info['event'].unique():
#     g.ax_row_dendrogram.bar(0, 0, color=colors_dict2[event], label=event, linewidth=0);

# l2 = g.ax_row_dendrogram.legend(title='Dataset', loc="center", ncol=2, bbox_to_anchor=(0.57, 1.06), bbox_transform=gcf().transFigure)

#g.cax.set_position([.15, .2, .03, .45])

ax = g.ax_heatmap
sns.despine()
# ax.set_xlabel("Samples")
ax.set_ylabel("")

plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/fig1e_col_DEG_heatmap.pdf", bbox_inches="tight")


# %%
####^^^ DET heatmap ######
transinput = np.log2(transdf+1)
trans_zscore = transdf.apply(lambda x: (x-x.mean())/x.std(), axis = 1)


trans_pre = trans_zscore.iloc[:,1::2]
trans_post = trans_zscore.iloc[:,0::2]
trans_zscore = pd.concat([trans_pre,trans_post],axis=1)
trans_zscore.dropna(inplace=True)

colors_dict = { "pre":"#4998FF", "post":"#FF4949"}
col_colors = [colors_dict["pre"]] * 40 + [colors_dict["post"]] * 40

for i in range(1):
    
    method = 'ward'

    sns.set_style("white")
    g = sns.clustermap(trans_zscore, col_colors=[col_colors], cmap="RdBu_r" , figsize=(6,9),vmin=-2,vmax=2,center=-0.2,
                    method=method, #single, complete, average, weighted, ward
                    metric='euclidean', #cityblock, minkowski, euclidean, cosine, correlation, hamming, jaccard, chebyshev, canberra, dice, rogerstanimoto, russellrao, sokalsneath
                    #z_score=0,
                    #vmax=1, vmin=0,
                    #standard_scale=0,
                        linewidths=0, xticklabels=False, yticklabels=False, col_cluster=True, row_cluster=True,
                        cbar_pos=(-0.01, 0.7, 0.05, 0.18))


    for group in ['pre','post']:
        g.ax_col_dendrogram.bar(0, 0, color=colors_dict[group],
                                label=group, linewidth=0)

    l1 = g.ax_col_dendrogram.legend(title='sample type', ncol=2, loc="center")

    #l1 = g.ax_col_dendrogram.legend(title='sample type', ncol=2)
    # for event in event_info['event'].unique():
    #     g.ax_row_dendrogram.bar(0, 0, color=colors_dict2[event], label=event, linewidth=0);

    # l2 = g.ax_row_dendrogram.legend(title='Dataset', loc="center", ncol=2, bbox_to_anchor=(0.57, 1.06), bbox_transform=gcf().transFigure)

    #g.cax.set_position([.15, .2, .03, .45])
    ax = g.ax_heatmap
    # ax.set_xlabel("Samples")
    ax.set_ylabel("")
    #plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/fig1e_col_DET_heatmap.pdf", bbox_inches="tight")

    plt.show()


# %%
# %%
####^^^ DSG heatmap ######
psi_input = psi
dsg_zscore = psi_input.iloc[:,:-1].apply(lambda x: (x-x.mean())/x.std(), axis = 1)
dsg_zscore.dropna(inplace=True)
colors_dict = { "pre":"#4998FF", "post":"#FF4949"}
col_colors = [colors_dict["pre"]] * 40 + [colors_dict["post"]] * 40
for i in range(1):
    
    method = 'ward'
    
    sns.set_style("white")
    g = sns.clustermap(dsg_zscore, col_colors=[col_colors], cmap="RdBu_r" , figsize=(6,9),vmin=-2,vmax=2,center=0,
                    method=method, #single, complete, average, weighted, ward
                    metric='euclidean', #cityblock, minkowski, euclidean, cosine, correlation, hamming, jaccard, chebyshev, canberra, dice, rogerstanimoto, russellrao, sokalsneath
                    #z_score=0,
                    #vmax=1, vmin=0,
                    #standard_scale=0,
                        linewidths=0, xticklabels=False, yticklabels=False, col_cluster=True, row_cluster=True,
                        cbar_pos=(-0.01, 0.7, 0.05, 0.18))


    for group in ['pre','post']:
        g.ax_col_dendrogram.bar(0, 0, color=colors_dict[group],
                                label=group, linewidth=0)

    l1 = g.ax_col_dendrogram.legend(title='sample type', ncol=2, loc="center")

    # for event in event_info['event'].unique():
    #     g.ax_row_dendrogram.bar(0, 0, color=colors_dict2[event], label=event, linewidth=0);

    # l2 = g.ax_row_dendrogram.legend(title='Dataset', loc="center", ncol=2, bbox_to_anchor=(0.57, 1.06), bbox_transform=gcf().transFigure)

    #g.cax.set_position([.15, .2, .03, .45])
    ax = g.ax_heatmap
    # ax.set_xlabel("Samples")
    ax.set_ylabel("")

plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/fig1e_col_DSG_heatmap.pdf", bbox_inches="tight")




# %%
#########^^^^^ EnrichR ####################
import gseapy as gp
glist = list(DSGlist)
enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                gene_sets=['GO_Biological_Process_2021'], 
                organism='Human', # don't forget to set organism to the one you desired! e.g. Yeast
                outdir=None, # don't write to disk
)

enrresult = enr.results.sort_values(by=['Adjusted P-value']) 

data = enrresult
data["logq"] = -np.log10(data["Adjusted P-value"])
data["logp"] = -np.log10(data["P-value"])

data = data[(data["Term"].str.contains("repair", case=False))]

# ##################################
df = data[["logp"]]
#df = data[["logq"]]
df = df.astype(float)

#### Term 확인을 위한 scatter plot (FDR < 0.1 ONLY)
if data[data["logp"]>1.0].shape[0] > 0:
    
    #plt.figure(figsize=(9,11))
    sns.set(rc={'figure.figsize':(3,1)})
    sns.set_style("whitegrid")

    ax = sns.scatterplot(
                        x="logp",y="Term",data=data[data["logp"]>=1],
                        s=130, edgecolor="None",
                        color="#FF8A8A"
                        )

    plt.xlim(left=1.0)
    
    if data[data["logp"]>=1.0].shape[0] < 5:
        every_nth = 1
    elif 10 < data[data["logp"]>=1.0].shape[0] < 25:
        # https://stackoverflow.com/questions/6682784/reducing-number-of-plot-ticks
        every_nth = 4
    elif 25 < data[data["logp"]>=1.0].shape[0]:
        every_nth = 4
    for n, label in enumerate(ax.yaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    
    plt.xlabel("$-Log_{10}$FDR", fontsize=13)
    plt.ylabel("")
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(13)
        
    #plt.title("DETs", size=14)


    plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/forDefense/fig1f_DSG_GOenrichment.pdf", bbox_inches="tight")

# %%
