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
import re
import matplotlib
from sklearn.preprocessing import MinMaxScaler

# %%
v301 = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/SR_283_transcript_TPM.txt', sep='\t', index_col=0)
v221 = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/mixSRLR_v221/quantification/SR_238_transcript_TPM.txt', sep='\t', index_col=0)

# %%
