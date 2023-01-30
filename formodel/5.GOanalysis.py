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
filelist = ['stable_LT','stable_ST','variable_LT','variable_ST']

file = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/satuRn_results/wholegene_stable_LT_GOenrichment.csv')
file = file[file['Adjusted P-value']<=0.1]


#%% 
###*enrichR GO enrichment analysis

filelist = ['stable_NR','stable_R','variable_NR','variable_R']
i=3
results = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/satuRn_results/new/'+filelist[i]+'_DTUresult.csv')

results.rename(columns = {"Unnamed: 0": "transcript"}, inplace = True)
results['gene'] = results['transcript'].str.split("-",1).str[1]

pcut = results[results['pval']<0.05]['gene']
pcut = pcut.drop_duplicates()
glist = pcut.squeeze().str.strip().to_list()

enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                gene_sets=['KEGG_2016','KEGG_2021_Human','GO_Biological_Process_2021'], #,'GO_Biological_Process_2021'
                organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                outdir=None, # don't write to disk
)

enrresult = enr.results.sort_values(by=['Adjusted P-value']) 

#%%
##* file saving
enrresult.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/satuRn_results/new/'+filelist[i]+'_GOenrichment.csv', index=False)


# %%
