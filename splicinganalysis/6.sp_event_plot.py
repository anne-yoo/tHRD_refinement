#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.stats import pearsonr
import random
from scipy.stats import ttest_ind

# %%
##^ number of splicing events per group

events=['A3','A5','AF','AL','MX','RI','SE'] #A3, A5

###file download###
pre = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/pre-suppa_'+events[6]+'_variable_10.ioe.psi', sep='\t', index_col=0)
post = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/post-suppa_'+events[6]+'_variable_10.ioe.psi', sep='\t', index_col=0)

#%%
prenull = pd.DataFrame(pre.isnull().sum(axis=1))
postnull = pd.DataFrame(post.isnull().sum(axis=1))
prenull.columns = ['null count']
postnull.columns = ['null count']

# %%
sns.kdeplot(data=prenull, x="null count", cumulative = True)

# %%

f_df = pd.DataFrame()
for i in range(7):
    pre = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/pre-suppa_'+events[i]+'_variable_10.ioe.psi', sep='\t', index_col=0)
    post = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/post-suppa_'+events[i]+'_variable_10.ioe.psi', sep='\t', index_col=0)
    prenull = pd.DataFrame(pre.isnull().sum(axis=1)/pre.shape[1])
    postnull = pd.DataFrame(post.isnull().sum(axis=1)/post.shape[1])
    prenull.columns = ['null count']
    postnull.columns = ['null count']

    tmp_df = pd.concat([prenull,postnull])
    f_df = pd.concat([f_df,tmp_df])


f_df = f_df.reset_index()
sns.kdeplot(data=f_df, x="null count", cumulative = True)
plt.show()
sns.kdeplot(data=f_df, x="null count")
plt.show()
print(np.quantile(f_df['null count'], 0.7))


# %% #^ pre

f_df = pd.DataFrame()
for i in range(7):
    pre = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/pre-suppa_'+events[i]+'_variable_10.ioe.psi', sep='\t', index_col=0)
    
    prenull = pd.DataFrame(pre.isnull().sum(axis=1)/pre.shape[1])
    prenull.columns = ['null count']

    f_df = pd.concat([f_df,prenull])

sns.kdeplot(data=f_df, x="null count", cumulative = True)
plt.show()
sns.kdeplot(data=f_df, x="null count")
print(np.quantile(f_df['null count'], 0.7))



# %% #^ post
f_df = pd.DataFrame()
for i in range(7):
    post = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/post-suppa_'+events[i]+'_variable_10.ioe.psi', sep='\t', index_col=0)
    
    postnull = pd.DataFrame(post.isnull().sum(axis=1)/post.shape[1])
    postnull.columns = ['null count']

    f_df = pd.concat([f_df,postnull])

sns.kdeplot(data=f_df, x="null count", cumulative = True)
plt.show()
sns.kdeplot(data=f_df, x="null count")
print(np.quantile(f_df['null count'], 0.7))












# %%
###################*   
events=['AF','AL','MX','RI','SE'] #A3, A5

###file download###
for i in range(5):
    dpsi = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/dpsi_suppa_'+events[i]+'_variable_10.ioe.dpsi', sep='\t', index_col=0)

    dpsi.columns = ['delta psi', 'corrected pval']
    filtered_dpsi = dpsi.dropna()
    con = (dpsi['delta psi'].abs() >= 0.1) & (dpsi['corrected pval']<0.1)
    filtered_dpsi = filtered_dpsi.loc[con,:]
    print(events[i], filtered_dpsi.shape[0])
# %%




#%%
dpsi_gc = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/dpsi_test_withgc.dpsi', sep='\t', index_col=0)
dpsi_wogc = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/dpsi_test_withoutgc.dpsi', sep='\t', index_col=0)
dpsi_05 = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/dpsi_test_0.5.dpsi', sep='\t', index_col=0)
dpsi_08 = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/dpsi_test_0.8.dpsi', sep='\t', index_col=0)
dpsi_00 = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/suppa2/dpsi_test_00.dpsi', sep='\t', index_col=0)

dpsi_gc.columns = ['delta psi', 'corrected pval']
dpsi_wogc.columns = ['delta psi', 'pval']
dpsi_05.columns = ['delta psi','corrected pval']
dpsi_08.columns = ['delta psi','corrected pval']
dpsi_00.columns = ['delta psi','corrected pval']

sns.kdeplot(data=dpsi_gc, x="corrected pval")
sns.kdeplot(data=dpsi_wogc, x="pval")
sns.kdeplot(data=dpsi_05,x="corrected pval" )
sns.kdeplot(data=dpsi_08,x="corrected pval" )
sns.kdeplot(data=dpsi_00,x="corrected pval" )


# %%
