#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.stats import pearsonr
# %%
exp = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/proteome/brip1ascc3.txt', sep='\t', index_col=0)
# %%
