

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

#%%
#^ 0. Go enrichment

import gseapy as gp

###*enrichR GO enrichment analysis


results = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t')
#results = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202310_analysis/DUT/whole_Wilcoxon_DUTresult_FC.txt', sep='\t')
results = results[(results['p_value']<0.05) & (results['log2FC']>1.5)]

######
#results = results[results['log2FC']>0]
######

pcut = results[['Gene Symbol']]
pcut = pcut.drop_duplicates()
glist = pcut.squeeze().str.strip().to_list()

#%%
# with open('/home/jiye/jiye/copycomparison/gDUTresearch/202310_analysis/DEG/upregulated_genelist.txt', 'w') as f:
#     for line in glist:
#         f.write("%s\n" % line)
# with open('/home/jiye/jiye/copycomparison/gDUTresearch/202310_analysis/DUT/downregulated_genelist.txt', 'w') as f:
#     for line in glist:
#         f.write("%s\n" % line)
#%%
enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                gene_sets=['GO_Biological_Process_2021'], 
                organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                outdir=None, # don't write to disk
)

enrresult = enr.results.sort_values(by=['Adjusted P-value']) 

#%%

data = enrresult
data["logq"] = -np.log10(data["Adjusted P-value"])
data["logp"] = -np.log10(data["P-value"])

data = data[data["Gene_set"]=="GO_Biological_Process_2021"]

data = data[(data["Term"].str.contains("splicing", case=False))]

sig = data[data["Adjusted P-value"] <= 0.05]

#%%

# ##################################
df = data[["logp"]]
#df = data[["logq"]]
df = df.astype(float)

#### Term 확인을 위한 scatter plot (FDR < 0.1 ONLY)
if data[data["logp"]>1.0].shape[0] > 0:
    
    #plt.figure(figsize=(9,11))
    sns.set(rc={'figure.figsize':(3,3)})
    sns.set_style("whitegrid")

    ax = sns.scatterplot(
                        x="logp",y="Term",data=data[data["logp"]>=1],
                        s=130, edgecolor="None",
                        color="#FF8A8A"
                        )

    plt.xlim(left=1.0)
    
    if data[data["logp"]>=1.0].shape[0] < 10:
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


    #plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DET_GO.pdf", bbox_inches="tight")
# %%
