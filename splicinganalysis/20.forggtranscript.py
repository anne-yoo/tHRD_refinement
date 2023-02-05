#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2

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




#################* for Venn Diagram ################
dsg = pd.read_csv('/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/readcount/final_data/DSG_DEG_comp/DSG_p5.txt', sep='\t')
deg = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/DESeq2/DESeq2_deg_genelist.csv', sep='\t')
hrgene = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/otherdata/HR_genes.txt', header=None)
# %%
deglist = set(deg['deg_genelist'])
dsglist = set(dsg['mstrg gene symbol'])
hrgene = set(hrgene[0])

# %%
sns.set_style('whitegrid')

v=venn2([deglist, dsglist], ('DEG', 'DSG'))
v.get_patch_by_id('10').set_color('#4E0E75')
v.get_patch_by_id('01').set_color('#BE2A0F')
v.get_patch_by_id('11').set_color('#620924')
v.get_patch_by_id('11').set_edgecolor('none')
v.get_patch_by_id('10').set_edgecolor('none')
v.get_patch_by_id('01').set_edgecolor('none')
v.get_patch_by_id('10').set_alpha(0.6)
v.get_patch_by_id('01').set_alpha(0.6)
v.get_patch_by_id('11').set_alpha(0.5)

plt.show()

v=venn2([deglist, hrgene], ('DEG', 'HR'))
v.get_patch_by_id('10').set_color('#4E0E75')
v.get_patch_by_id('01').set_color('#0F3EBE')
v.get_patch_by_id('11').set_color('#270962')
v.get_patch_by_id('11').set_edgecolor('none')
v.get_patch_by_id('10').set_edgecolor('none')
v.get_patch_by_id('01').set_edgecolor('none')
v.get_patch_by_id('10').set_alpha(0.6)
v.get_patch_by_id('01').set_alpha(0.6)
plt.show()

v=venn2([dsglist, hrgene], ('DSG', 'HR'))
v.get_patch_by_id('10').set_color('#BE2A0F')
v.get_patch_by_id('01').set_color('#0F3EBE')
v.get_patch_by_id('11').set_color('#560962')
v.get_patch_by_id('11').set_edgecolor('none')
v.get_patch_by_id('10').set_edgecolor('none')
v.get_patch_by_id('01').set_edgecolor('none')
v.get_patch_by_id('10').set_alpha(0.6)
v.get_patch_by_id('01').set_alpha(0.6)
plt.show()


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
ax.bar_label(bars)
plt.show()

fig, ax = plt.subplots()
bars = df.iloc[0,:].plot.barh(align='center', stacked=True, figsize=(8, 4),color=col_map.colors)
title = plt.title('Whole AS Events')
ax.bar_label(bars)

plt.show()

# %%
