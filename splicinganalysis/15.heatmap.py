#%%
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.cm as cm
from matplotlib.pyplot import gcf
# %%

###########################** psi heatmap ###############################

#^ data download + simple processing
psi = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/suppa2/dsg_psi_forheatmap.csv', index_col=0)
pre = psi.loc[:, psi.columns.str.contains('bfD')]
post = psi.loc[:, psi.columns.str.contains('atD')]
data_mat = pd.concat([pre,post],axis=1)

data_info = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/DESeq2/metadata.csv')
d_pre = data_info[data_info['group']=='pre']
d_post = data_info[data_info['group']=='post']
data_info = pd.concat([d_pre,d_post])

#%%
#^ data zscore normalization: one gene <-> across all samples!!
data_zscore =  data_mat.apply(lambda x: (x-x.mean())/x.std(), axis = 1)

# %%
#^ color mapping
colors = cm.rainbow(np.linspace(0, 1, len(data_info['group'].unique())))
colors_dict = { "pre":"#4998FF", "post":"#FF4949"}
col_colors = data_info['group'].map(colors_dict)

# %%
##^ event group
# event_info = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/suppa2/dsg_eventdata.csv')
# colors2 = cm.Set2(np.linspace(0, 1, len(event_info['event'].unique())))
# colors_dict2 = dict(zip(event_info['event'].unique(), colors2))
# col_colors2 = event_info['event'].map(colors_dict2)


# %%
##^ draw seaborn clustermap
# methods = ['single', 'complete', 'average', 'weighted', 'ward'] #5
methods = ['ward']
metrics = ['cityblock', 'minkowski', 'seuclidean', 'cosine', 'correlation', 'hamming', 'jaccard', 'chebyshev', 'canberra', 'dice', 'rogerstanimoto', 'russellrao', 'sokalsneath'] #13

for i in range(1):
    
    method = methods[i]

    g = sns.clustermap(data_zscore, col_colors=[col_colors], cmap="RdBu" , figsize=(6,11),vmin=-2,vmax=2,center=0,
                    method=method, #single, complete, average, weighted, ward
                    metric='euclidean', #cityblock, minkowski, euclidean, cosine, correlation, hamming, jaccard, chebyshev, canberra, dice, rogerstanimoto, russellrao, sokalsneath
                    #z_score=0,
                    #vmax=1, vmin=0,
                    #standard_scale=0,
                        linewidths=0, xticklabels=False, yticklabels=False, col_cluster=True)


    for group in  data_info['group'].unique():
        g.ax_col_dendrogram.bar(0, 0, color=colors_dict[group],
                                label=group, linewidth=0)

    l1 = g.ax_col_dendrogram.legend(title='sample type', loc="center", ncol=2, bbox_to_anchor=(0.57, 1.01), bbox_transform=gcf().transFigure)

    # for event in event_info['event'].unique():
    #     g.ax_row_dendrogram.bar(0, 0, color=colors_dict2[event], label=event, linewidth=0);

    # l2 = g.ax_row_dendrogram.legend(title='Dataset', loc="center", ncol=2, bbox_to_anchor=(0.57, 1.06), bbox_transform=gcf().transFigure)

    #g.cax.set_position([.15, .2, .03, .45])
    ax = g.ax_heatmap
    # ax.set_xlabel("Samples")
    ax.set_ylabel("")
# %%











#%%
# %%

###########################** TPM heatmap ###############################

#^ data download + simple processing
psi = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/DESeq2/KW_TPM_DEG_forheatmap.csv', index_col=0)
pre = psi.loc[:, psi.columns.str.contains('bfD')]
post = psi.loc[:, psi.columns.str.contains('atD')]
data_mat = pd.concat([pre,post],axis=1)

data_info = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/DESeq2/metadata.csv')
d_pre = data_info[data_info['group']=='pre']
d_post = data_info[data_info['group']=='post']
data_info = pd.concat([d_pre,d_post])

#%%
#^ data zscore normalization: one gene <-> across all samples!!
data_mat = np.log2(data_mat+1)
data_zscore =  data_mat.apply(lambda x: (x-x.mean())/x.std(), axis = 1)

# %%
#^ color mapping
colors = cm.rainbow(np.linspace(0, 1, len(data_info['group'].unique())))
colors_dict = { "pre":"#4998FF", "post":"#FF4949"}
col_colors = data_info['group'].map(colors_dict)

# %%
##^ event group
# event_info = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/suppa2/dsg_eventdata.csv')
# colors2 = cm.Set2(np.linspace(0, 1, len(event_info['event'].unique())))
# colors_dict2 = dict(zip(event_info['event'].unique(), colors2))
# col_colors2 = event_info['event'].map(colors_dict2)


# %%
##^ draw seaborn clustermap
# methods = ['single', 'complete', 'average', 'weighted', 'ward'] #5
methods = ['ward']
metrics = ['cityblock', 'minkowski', 'seuclidean', 'cosine', 'correlation', 'hamming', 'jaccard', 'chebyshev', 'canberra', 'dice', 'rogerstanimoto', 'russellrao', 'sokalsneath'] #13

for i in range(1):
    
    method = methods[i]

    g = sns.clustermap(data_zscore, col_colors=[col_colors], cmap="RdBu" ,vmin=-2,vmax=2,center=0,figsize=(6,11),
                    method=method, #single, complete, average, weighted, ward
                    metric='euclidean', #cityblock, minkowski, euclidean, cosine, correlation, hamming, jaccard, chebyshev, canberra, dice, rogerstanimoto, russellrao, sokalsneath
                    #z_score=0,
                    #vmax=1, vmin=0,
                    #standard_scale=0,
                        linewidths=0, xticklabels=False, yticklabels=False, col_cluster=True)


    for group in  data_info['group'].unique():
        g.ax_col_dendrogram.bar(0, 0, color=colors_dict[group],
                                label=group, linewidth=0)

    l1 = g.ax_col_dendrogram.legend(title='sample type', loc="center", ncol=2, bbox_to_anchor=(0.57, 1.01), bbox_transform=gcf().transFigure)

    # for event in event_info['event'].unique():
    #     g.ax_row_dendrogram.bar(0, 0, color=colors_dict2[event], label=event, linewidth=0);

    # l2 = g.ax_row_dendrogram.legend(title='Dataset', loc="center", ncol=2, bbox_to_anchor=(0.57, 1.06), bbox_transform=gcf().transFigure)

    #g.cax.set_position([.15, .2, .03, .45])
    ax = g.ax_heatmap
    # ax.set_xlabel("Samples")
    ax.set_ylabel("")
# %%
