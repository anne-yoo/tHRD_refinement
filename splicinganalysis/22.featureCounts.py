#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.stats import pearsonr
import glob
# %%
Dir = '/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/resistance/brip1_R/featurecounts/'
filelist = glob.glob(Dir+'*.txt')

# %%
