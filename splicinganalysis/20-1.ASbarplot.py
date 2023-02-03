#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2
# %%
####* stacked barplot
dsg = pd.read_csv('/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/readcount/final_data/DSG_DEG_comp/DSG_p5.txt', sep='\t')

o_dsg = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/suppa2/DSGinfo_beforefiltering.csv',sep='\t') #original psi file
# %%
df = pd.DataFrame({'A3':[12433,361],'A5':[8439,295],'RI':[11519,458],'AF':[8083,367],'SE':[17194,475],'AL':[3608,257],'MX':[542,26]}, index=['whole','significant'])
sns.set(style='white')
ax = df.iloc[1,:].plot.barh(align='center', stacked=True, figsize=(8, 4))

# %%
col_map = plt.get_cmap('Paired')
fig, ax = plt.subplots()
bars = df.iloc[1,:].plot.barh(align='center', stacked=True, figsize=(8, 4),color=col_map.colors)
title = plt.title('Significant AS Events')
plt.show()

fig, ax = plt.subplots()
bars = df.iloc[0,:].plot.barh(align='center', stacked=True, figsize=(8, 4),color=col_map.colors)
title = plt.title('Whole AS Events')

plt.show()
# %%
