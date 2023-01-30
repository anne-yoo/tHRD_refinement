#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu


WES_BRCA1 = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/sample-merged/WES-BRCA1-whole-sample')
WES_BRCA2 = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/sample-merged/WES-BRCA2-whole-sample')
WGS_BRCA1 = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/sample-merged/WGS-BRCA1-whole-sample')
WGS_BRCA2 = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/sample-merged/WGS-BRCA2-whole-sample')

filelist = [WES_BRCA1, WES_BRCA2, WGS_BRCA1, WGS_BRCA2]
filenamelist = ['WES_BRCA1', 'WES_BRCA2', 'WGS_BRCA1', 'WGS_BRCA2']

clinical = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/final_clinicaldata')

WES_matched_samplelist = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/WES_matched_samplelist.csv', header=None)
WGS_matched_samplelist = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/WGS_matched_samplelist.csv', header=None)
WES_matched_samplelist.columns = ['sample_ID','pre/post']
WGS_matched_samplelist.columns = ['sample_ID','pre/post']

matchedlist = [WES_matched_samplelist,WGS_matched_samplelist]

# %%
##* mean RNA AF for each sample in each file
for i in range(4):
    filename = filenamelist[i]
    file = filelist[i]
    vars()[filename] = pd.DataFrame(file.groupby('sample_ID')['RNA_AF']. mean())
    

# %%
