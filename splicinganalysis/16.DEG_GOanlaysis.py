#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.stats import pearsonr
import gseapy as gp

#%%
###* GO enrichment result check
file = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/DESeq2/KW_deg_genelist_withTPM.csv') #KW
# file = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/DESeq2/DESeq2_deg_genelist.csv') #DESeq2

#%% 
###*enrichR GO enrichment analysis


pcut = file.drop_duplicates()
glist = pcut.squeeze().str.strip().to_list()

enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                gene_sets=['KEGG_2016','KEGG_2021_Human','GO_Biological_Process_2021'], #,'GO_Biological_Process_2021'
                organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                outdir=None, # don't write to disk
)

enrresult = enr.results.sort_values(by=['Adjusted P-value']) 

#%%
##* file saving
enrresult.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/DESeq2/DEG_GOenrichment/KW_DEG_GOenrichment.csv', index=False)


# %%
