#%%
#! this code is.....

"""
20230619

R vs. NR 비교한 whole gene 대상 DEG / DTU PCA 한 것과 whole gene PCA 한거 비교하기

1. elbow point 찾기
2. elbow point PC들에 해당하는 gene list 비교
2. gene list GO enrichment

++++ 3. 이건 그냥 15.volcano 만들기 위해 DTU / DEG merged file 만드는 것임
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
###

##^ 1. elbow point PC들에 해당하는 gene list comparison - whole gene


##** download datasets
##** gene: 2 components, transcript: 3 components

gene = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/DESeq2/response/PCA_whole_topPC.txt', header=None)
trans = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/input/discovery/response/PCA_whole_topPC.txt', header=None)

# gene = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/DESeq2/response/PCA_DEG_topPC.txt', header=None)
# trans = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/satuRn/input/discovery/response/PCA_DTU_topPC.txt', header=None)

gene.columns = ['id']

#%%

ensgtosymbol = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/ensgidtosymbol.dict', sep='\t', header=None)
ensgtosymbol.columns = ['id','symbol']


gene_sym = pd.merge(gene,ensgtosymbol,on='id')

#%%

from matplotlib_venn import venn2

genelist = set(gene_sym['symbol'])
trans['symbol'] = trans[0].str.split("-",1).str[1]

translist = set(trans['symbol'])

#%%
plt.figure(figsize=(6,6))
sns.set_style("white")
vd2 = venn2([genelist,translist],set_labels=('gene', 'transcript'),set_colors=('#F8CF6A','#74B2D4'), alpha=0.8)

a,b,c = vd2.subset_labels

for text in vd2.set_labels:  #change label size
        text.set_fontsize(13)
#for a,b in vd2.subset_labels:  #change number size
a.set_fontsize(13)
b.set_fontsize(13)
        

# vd2.get_patch_by_id('11').set_color('#C8C452')
# vd2.get_patch_by_id('11').set_alpha(1)

#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/figures/topPC_deg_dtu.pdf", bbox_inches="tight")
plt.show()




# %%
##^^ 2. gene list GO enrichment

import gseapy as gp

genelist = list(genelist)
translist = list(translist) 

#%%
enr = gp.enrichr(gene_list=translist, # or "./tests/data/gene_list.txt",
                gene_sets=['GO_Biological_Process_2021'], 
                organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                outdir=None, # don't write to disk
)
data = enr.results
data["logq"] = -np.log10(data["Adjusted P-value"])
data = data[data["Gene_set"]=="GO_Biological_Process_2021"]
data = data[data["Adjusted P-value"] <= 0.05]

df = data[["logq"]]
df = df.astype(float)

### Term 확인을 위한 scatter plot (FDR < 0.1 ONLY)
if data[data["Adjusted P-value"] <= 0.05]:
        sns.set_style("whitegrid")
        plt.figure(figsize=(3,6))
        ax = sns.scatterplot(
                        x="logq",y="Term",data=data[data["Adjusted P-value"] <= 0.05],
                        s=130, edgecolor="None",
                        color="#FF8A8A"
                        )

        plt.xlim(left=1.0)
#        print(data[data["logq"]>=1.0].shape[0], path)
        if data[data["logq"]>=1.0].shape[0] < 10:
                every_nth = 1
        elif 10 < data[data["logq"]>=1.0].shape[0] < 25:
        # https://stackoverflow.com/questions/6682784/reducing-number-of-plot-ticks
                every_nth = 4
        elif 25 < data[data["logq"]>=1.0].shape[0]:
                every_nth = 6
        for n, label in enumerate(ax.yaxis.get_ticklabels()):
                if n % every_nth != 0:
                        label.set_visible(False)
        
        plt.title("transcripts in top PCs", fontsize=14, position=(0.5, 1.0+0.02))
        plt.xlabel("$-Log_{10}$FDR", fontsize=13)
        plt.ylabel("")
        for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(13)
                
        #plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/figures/GOenrichment_wholetranscript_topPC.pdf", bbox_inches="tight")
        
        plt.show()
# %%

##* 3. make a merged dataset for drawing volcano plot for DEG and DTU analysis

ensgtosymbol = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/ensgidtosymbol.dict', sep='\t', header=None)
ensgtosymbol.columns = ['id','gene']

deg = pd.read_csv("/home/jiye/jiye/copycomparison/gDUTresearch/DTUDEGcomp/DESeq2_DEG_result.txt")
dtu = pd.read_csv("/home/jiye/jiye/copycomparison/gDUTresearch/DTUDEGcomp/DEXSeq_DTU_result.txt")

deg.columns = ['id','baseMean','log2fc','lfcSE','stat','pvalue','padj']
dtu.columns = ['transcript','gene','pvalue','log2fc']

deg_sym= pd.merge(ensgtosymbol,deg,on='id', how="inner")


# %%

merged = pd.merge(dtu, deg_sym, on='gene',how="inner")
merged = merged.dropna(axis=0, inplace=False)
merged = merged.drop_duplicates(subset='transcript')
merged.columns=['transcript','gene','dtu_pval','dtu_log2fc','geneid','baseMean','deg_log2fc','lfcSE','stat','deg_pval','padj']
# %%

final_deg = merged[['gene','log2fc_y','pvalue_y']]
final_dtu = merged[['transcript','gene','log2fc_x','pvalue_x']]

final_deg = final_deg.drop_duplicates(subset=['gene'])
final_dtu = final_dtu.drop_duplicates(subset='transcript')
# %%
final_deg.columns = ['gene','log2fc','pval']
final_dtu.columns = ['transcript','gene','log2fc','pval']
final_deg.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/DTUDEGcomp/volcano_deg.txt',index=False)
final_dtu.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/DTUDEGcomp/volcano_dtu.txt',index=False)


# %%
deg_sym = deg_sym.dropna(axis=0, inplace=False)
deg_sym = deg_sym.drop_duplicates(subset=['gene'])
deg_sym.to_csv("/home/jiye/jiye/copycomparison/gDUTresearch/DTUDEGcomp/DESeq2_DEG_result_symbol.txt",index=False)

# %%
#merged.to_csv("/home/jiye/jiye/copycomparison/gDUTresearch/DTUDEGcomp/volcano_DEGDTU.txt",index=False)
# %%
