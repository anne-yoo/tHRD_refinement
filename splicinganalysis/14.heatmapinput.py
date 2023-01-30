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
from scipy.stats import kruskal
# %%
#######^ TPM pre vs. post heatmap

tpm = pd.read_csv('/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/pre_post_TU/final_data/230106_new_final_pre_post_samples_TU_input.txt', index_col=0, sep='\t')
# %%
deg_list = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/DESeq2/DESeq2_deg_genelist.csv')
deg_list.columns = ['target_gene']

# %%
deg_tpm = pd.merge(tpm, deg_list, how='inner', on='target_gene')

#%%
tpm = deg_tpm
tpm = tpm[tpm['target_gene']!= '-']

# sample name change
sample = tpm.columns.tolist()# %%
samples = sample
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

tpm.columns = samples
try:
    tpm = tpm.drop(["SV-OV-P080-atD","SV-OV-P250-atD","SV-OV-P055-atD",\
            "SV-OV-P143-atD","SV-OV-P137-atD","SV-OV-P134-atD",\
            "SV-OV-P174-bfD","SV-OV-P164-atD"], axis=1)
except:
    

    pass

deg_tpm = tpm


# %%
deg_tpm = deg_tpm.set_index(['target_gene'])
#%%
deg_tpm = np.log2(deg_tpm+1)

#%%
zscore = (deg_tpm - np.mean(deg_tpm)) / np.std(deg_tpm)
#%%
zscore.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/DESeq2/TPM_deg_DESeq2_forheatmap.csv', index=True)

# %%
dtu = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/suppa2/dtu_psi_forheatmap.csv', index_col=0)
# %%
event = dtu[['event']]
dtu = dtu.drop(['event'], axis=1)

#%%
z_dtu = (dtu - np.mean(dtu)) / np.std(dtu)

#%%
z_dtu.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/suppa2/zscore_dsg_psi_forheatmap.csv', index=True)
event.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/suppa2/dsg_eventdata.csv', index=True)


# %%
