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

# %%
dt = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/MW_psi_7events.txt', sep='\t')


# %%
data = {'dpsi': [4056,3324,411,244,14,1386,351],
        'total': [231326,191319,128986,44137,7232,74476,55296],
        'event':['A3','A5','AF','AL','MX','RI','SE']}

df = pd.DataFrame(data)
# %%
df['non-dpsi'] = df['total'] - df['dpsi']
# %%
sns.set(style="whitegrid")
plt.figure(figsize=(14, 14))

bar1 = sns.barplot(x="event",  y="total", data=df, color='lightblue')
bar2 = sns.barplot(x='event', y = 'dpsi', data=df, color='darkblue')

# %%
pie, ax = plt.subplots(figsize=[8,8])
sns.set(style='whitegrid')
plt.pie(df['dpsi'], labels = df['event'], colors = sns.color_palette('tab10'),explode=[0.03]*7, autopct="%.1f%%",pctdistance=0.5)
plt.title("number of dpsi transcripts by event", fontsize=13)


# %%
#^ heatmap
import matplotlib.cm as cm
from matplotlib.pyplot import gcf

events=['A3','A5','AF','AL','MX','RI','SE'] 
for i in range(7):
        pre = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/pre-suppa_'+events[i]+'_variable_10.ioe.psi', sep='\t', index_col=0)
        post = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/post-suppa_'+events[i]+'_variable_10.ioe.psi', sep='\t', index_col=0)



        hmdf = pd.concat([pre,post], axis=1)

        
        hmdf.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/forpsiheatmap_'+events[i]+'.txt', sep='\t')

# %%
sample_info = pd.DataFrame()
sample_info.index = hmdf.columns
sample_info['group'] = 'post'
sample_info.iloc[:26,0] = 'pre'

# %%
sample_info.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/sample_info.txt', sep='\t')
# %%
