#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.stats import pearsonr


#%%
fi = pd.read_csv('/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/merged_TU/whole_TU/XGBoost/100BOP_tuning_importance.txt', sep='\t', index_col=0)

# %%
mean_fi = fi.mean(axis=0)
sorted_fi = mean_fi.sort_values(ascending=False)
# %%
fidf = pd.DataFrame(sorted_fi)
fidf.columns = ['feature importance']
fidf['gene'] = fidf.index.str.split('-',2).str[1]
# %%

df = fidf.iloc[:28,:]
df['feature'] = df.index
# %%
sns.set(rc = {'figure.figsize':(8,6)})
sns.set(font_scale = 0.9)
sns.set_style('whitegrid')
sns.barplot(y="feature", x="feature importance", data=df, orient='h',color='#2A947A')
plt.rcParams['font.family'] = 'arial'



# %%
##^ gene expression comparison BRIP1/ASCC3
tpm = pd.read_csv('/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/readcount/final_data/add_symbol_TMM_230106_final_pre_post_samples_raw_counts.txt', sep='\t')

# %%
tpm=tpm.rename(columns = {"Unnamed: 0": "Geneid"})
tpm_2genes = tpm[(tpm['Geneid']== "BRIP1") | (tpm['Geneid'] =="ASCC3")]
# %%
tpm_035 = tpm_2genes.loc[:,['Geneid','SV-OV-P048-bfD','SV-OV-P048-atD','SV-OV-P068-bfD','SV-OV-P068-atD']]
# %%
