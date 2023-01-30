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
rawcount = pd.read_csv('/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/readcount/final_data/230106_final_pre_post_samples_raw_counts.txt', sep='\t', index_col=0)
# %%
rawcount = rawcount.drop(["SV-OV-P080-atD","SV-OV-P250-atD","SV-OV-P055-atD",\
                "SV-OV-P143-atD","SV-OV-P137-atD","SV-OV-P134-atD",\
                "SV-OV-P174-bfD","SV-OV-P164-atD"], axis=1)

# %%
sampleinfo = pd.DataFrame(rawcount.columns, columns=['samples'])
sampleinfo['group'] = 'pre'

sampleinfo.loc[sampleinfo['samples'].str.contains("atD"),'group'] = "post"

# %%
rawcount.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/DESeq2/countdata.csv', index=True)
sampleinfo.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/DESeq2/metadata.csv',index=False)
# %%

###########* result ensg id to gene symbol

result = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/DESeq2/DEGresult.csv', index_col=0)
annot = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/annotation/ensg2symbol.dict', sep='\t', header=None)
annot.columns = ['ensg', 'gene symbol']
# %%
result['ensg'] = result.index
annot = annot.drop_duplicates()
m_result = pd.merge(result, annot, how='inner', on='ensg')

# %%
final = m_result[['gene symbol','log2FoldChange','pvalue','padj']]
final.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/DESeq2/sortedDEGresult_symbol.csv', index=False)
# %%
