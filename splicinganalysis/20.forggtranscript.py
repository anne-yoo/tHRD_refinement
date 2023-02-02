#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%
brip = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/annotation/BRIP1.gtf', sep='\t', header=None)
brip.columns = ['seqnames','ss','type','start','end','quality','strand','.','etc']
brip2 = brip['etc'].str.split('; ', expand=True)

brip = brip[['seqnames','start','end','strand','type']]
# %%
etc = brip2[[1]]
etc.columns = ['transcript_name']
etc['transcript_name'] = etc['transcript_name'].str.split('"',2).str[1]
# %%
brip['gene_name'] = 'BRIP1'
brip['transcript_name'] = etc['transcript_name']
brip['transcript_biotype'] = 'None'
# %%
brip.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/annotation/ggtranscript_BRIP1.gtf', sep='\t',index=False, header=True)
# %%
