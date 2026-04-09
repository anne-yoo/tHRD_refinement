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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.stats.multitest import multipletests
from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42  
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'Helvetica'

plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 13,    # x축 틱 라벨 글꼴 크기

'ytick.labelsize': 13,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 13,
'legend.title_fontsize': 13, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})
sns.set_style("ticks")
#%%

df = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/DEGDUT/AR_Wilcoxon_DEGresult_FC.txt', sep='\t')
#df = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DEG/nonresponder_Wilcoxon_DEGresult_FC.txt', sep='\t')

# Calculate -log10(p-value)
df['-log10(p_value)'] = -np.log10(df['p_value'])

# Determine significance
df['color'] = 'Stable'
df.loc[(df['p_value'] < 0.05),  'color'] = 'Variable'

# Plotting

plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 12,    # x축 틱 라벨 글꼴 크기
'ytick.labelsize': 12,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 13,
'legend.title_fontsize': 13, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})
plt.figure(figsize=(6,5))
sns.set_style("white")
colors = {'Variable': '#FFC96F', 'Stable': '#ACD793',}

for color in colors:
    subset = df[df['color'] == color]
    plt.scatter(subset['log2FC'], subset['-log10(p_value)'], 
                color=colors[color], alpha=0.7, label=color)

# Adding threshold lines
plt.axhline(y=-np.log10(0.05), color='#543310', linestyle='--', linewidth=1, alpha=0.3)

# Adding labels and titleplt.xlabel('Log2 Fold Change')
#plt.title('AR+IR')
plt.ylabel('-log10(pval)')
plt.xlabel('log2FC')
plt.legend(title='Gene Class')
plt.tight_layout()
plt.title('AR DEG')
sns.despine()

# Save the plot as a high-resolution image
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/figures/AR_DEG_volcano.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
###^^^^^^^^ DEG vs. DSG ###############
dsgresult = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/rMATS/dpsi_analysis/MW_dpsi_5events.txt', sep='\t')
dsgresult = dsgresult[(dsgresult['pval']<0.001) & (dsgresult['d_psi']>0.1)]
DSGlist = set(dsgresult['gene symbol'])


# %%

#########^^^^^^^^^^^^ INPUT FOR SUPPA2 #######################
sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
responder = list(set(sampleinfo.loc[(sampleinfo['response']==1),'sample_id']))
nonresponder = list(set(sampleinfo.loc[(sampleinfo['response']==0),'sample_id']))

# %%
AR = df[df['Sample'].isin(responder)]
IR = df[df['Sample'].isin(nonresponder)]



#%%
ratio = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/majorminor/major2minorratio_mean.txt', sep='\t', index_col=0)
#ratio = ratio.dropna()
ratio_atd = ratio.iloc[:,0::2]
ratio_bfd = ratio.iloc[:,1::2]
# %%
gene = 'SHLD2'
figinput_bfd = pd.DataFrame({'value' : ratio_bfd.loc[gene,:].to_list(),'treatment' : 'pre'})
figinput_atd = pd.DataFrame({'value' : ratio_atd.loc[gene,:].to_list(),'treatment' : 'post'})
figinput = pd.concat([figinput_bfd,figinput_atd], axis=0)

ax = sns.boxplot(x='treatment',y='value', data=figinput)
from statannot import add_stat_annotation
add_stat_annotation(ax, data=figinput, x='treatment', y='value',
                    box_pairs=[("pre", "post")],
                    test='Wilcoxon',  text_format='star', loc='inside', fontsize=14)
# %%

#####^^^^^^^^^^^^^^^^^^^^^^ BRIP1 Regression test ##############
TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0)
brip1_atD = TU.iloc[TU.index=='MSTRG.49834.247-BRIP1',0::2].values[0]
brip1_bfD = TU.iloc[TU.index=='MSTRG.49834.247-BRIP1',1::2].values[0]
x = brip1_atD - brip1_bfD
#
y = sampleinfo.iloc[0::2,3]
#x = np.array(x).reshape(-1, 1)
#y = np.array(y)
from sklearn.linear_model import LogisticRegression
df = pd.DataFrame({'x': x, 'y': y})
import statsmodels.api as sm

# Add a constant to the independent variable matrix
X = sm.add_constant(df['x'])

# Perform linear regression
model = sm.OLS(df['y'], X).fit()

# Get the p-values
p_values = model.pvalues

# Get the correlation matrix
corr_matrix = df.corr()

# Display the results
print("P-values:")
print(p_values)
print("\nCorrelation matrix:")
print(corr_matrix)

# %%
###^^ DEG AR vs. IR ###
deg_ar = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202310_analysis/DEG/responder_Wilcoxon_DEGresult_FC.txt', sep='\t', index_col=0)
deg_ar = set(deg_ar[(deg_ar['p_value']<0.05) & (np.abs(deg_ar['log2FC'])>1)]['Gene Symbol'])
#deg_ar = set(deg_ar[(deg_ar['p_value']<0.05) ]['Gene Symbol'])

deg_ir = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202310_analysis/DEG/nonresponder_Wilcoxon_DEGresult_FC.txt', sep='\t', index_col=0)
deg_ir = set(deg_ir[(deg_ir['p_value']<0.05) & (np.abs(deg_ir['log2FC'])>1)]['Gene Symbol'])
#deg_ir = set(deg_ir[(deg_ir['p_value']<0.05) ]['Gene Symbol'])

# %%
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 12,    # x축 틱 라벨 글꼴 크기
'ytick.labelsize': 12,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 13,
'legend.title_fontsize': 13, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})
plt.figure(figsize=(4,4))
sns.set_style("white")
from matplotlib_venn import venn2
vd2 = venn2([deg_ar, deg_ir],set_labels=('AR', 'IR'),set_colors=('#F8CF6A','#74B2D4'), alpha=0.8)

for text in vd2.set_labels:  #change label size
    text.set_fontsize(13)
for text in vd2.subset_labels:  #change number size
    text.set_fontsize(13)
    
vd2.get_patch_by_id('11').set_color('#C8C452')
vd2.get_patch_by_id('11').set_alpha(1)

plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/fig1/DEG_ARIR_Venn.pdf", bbox_inches="tight")
plt.show()




# %%
###^^^ gene TU stacked barplot ################
TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0)
TU['TU_mean'] = TU.mean(axis=1)
TU['TU_max'] = TU.max(axis=1)
TU['TU_min'] = TU.min(axis=1)
TU['gene'] = TU.index.str.split("-",2).str[-1]

# %%
################
gene = 'CHEK2'
################

newdf = TU[TU['gene']==gene]
newdf = newdf.sort_values(by=['TU_max'], ascending=False)
newdf = newdf.iloc[:15,:]

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
responder = list(sampleinfo.loc[(sampleinfo['response']==1),'sample_full'])
nonresponder = list(sampleinfo.loc[(sampleinfo['response']==0),'sample_full'])

AR = newdf[responder]
IR = newdf[nonresponder]

df = pd.concat([AR,IR],axis=1)

transcripts = df.index

df_melted = df.reset_index().melt(id_vars='gene_ENST', var_name='sample', value_name='usage')
df_melted['individual'] = df_melted['sample'].apply(lambda x: '-'.join(x.split('-')[:-1]))
df_melted['condition'] = df_melted['sample'].apply(lambda x: x.split('-')[-1])
df_melted = df_melted.rename(columns={'gene_ENST': 'transcript'})

# Initialize the plot
# Define a custom color palette
palette = sns.color_palette("Set2", n_colors=len(df_melted['transcript'].unique()))  # You can change "husl" to any palette you prefer

# Create a dictionary to map transcripts to colors
transcript_colors = {transcript: color for transcript, color in zip(df_melted['transcript'].unique(), palette)}

# Initialize the plot
sns.set_style('ticks')

fig, ax = plt.subplots(figsize=(13, 3))
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 11,    # x축 틱 라벨 글꼴 크기
'ytick.labelsize': 11,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 13,
'legend.title_fontsize': 13, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})

# Create a position mapping for individuals and conditions
positions = {}
current_pos = 0
for individual in df_melted['individual'].unique():
    positions[individual + '-bfD'] = current_pos
    positions[individual + '-atD'] = current_pos + 0.5
    current_pos += 1.5

# Plot for each individual and condition
for individual in df_melted['individual'].unique():
    subset = df_melted[df_melted['individual'] == individual]
    bottom_atD = 0
    bottom_bfD = 0
    for transcript in df_melted['transcript'].unique():
        usage_atD = subset[(subset['transcript'] == transcript) & (subset['condition'] == 'atD')]['usage'].values[0] if not subset[(subset['transcript'] == transcript) & (subset['condition'] == 'atD')]['usage'].empty else 0
        usage_bfD = subset[(subset['transcript'] == transcript) & (subset['condition'] == 'bfD')]['usage'].values[0] if not subset[(subset['transcript'] == transcript) & (subset['condition'] == 'bfD')]['usage'].empty else 0
        
        ax.bar(positions[individual + '-bfD'], usage_bfD, width=0.4, bottom=bottom_bfD, color=transcript_colors[transcript], label=transcript if individual == df_melted['individual'].unique()[0] else "")
        ax.bar(positions[individual + '-atD'], usage_atD, width=0.4, bottom=bottom_atD, color=transcript_colors[transcript])
        
        bottom_bfD += usage_bfD
        bottom_atD += usage_atD

# Adjust x-axis to show individual names without overlapping
ax.set_xticks([positions[individual + '-atD'] - 0.2 for individual in df_melted['individual'].unique()])
ax.set_xticklabels(df_melted['individual'].unique(), rotation=45, ha='right')

# Add labels and title
#ax.set_ylabel('Transcript Usage', fontsize=13)
#ax.set_xlabel('Samples', fontsize=13)
#ax.set_title(gene, fontsize=15)
ax.set_xlim(-0.8, current_pos - 0.2)
ax.set(xticklabels=[])
ax.set_ylabel('Transcript Usage', fontsize=13)
ax.set_xlabel('Samples', fontsize=13)
#ax.legend(title='Transcripts')

# Show plot
plt.tight_layout()
sns.despine()
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/fig1/TUbarplot_'+gene+'.png', dpi=300, bbox_inches='tight')
plt.show()









# %%
###&######### 뿅뿅이 plot new ver #############
import gseapy as gp

path_list1 = ['responder_stable','responder_variable','nonresponder_stable','nonresponder_variable']
path_list2 = ['AR_stable','AR_variable','IR_stable','IR_variable']

path = '/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/DEGDUT/'
#path2 = '/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUT/'
TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost_80_transcript_TU.txt', sep='\t', index_col=0)
major = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/majorlist.txt', sep='\t')
majorlist = list(major['Transcript-Gene'])
sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost_80_clinicalinfo.txt', sep='\t')
responder = list(sampleinfo.loc[(sampleinfo['response']==1),'sample_full'])
nonresponder = list(sampleinfo.loc[(sampleinfo['response']==0),'sample_full'])


#### DUT ###########
###*** ARbpdutdf, IRdpdutdf 각각 만들어야 함, i=0, i=2 바꿔서 ###
for i in [0]:    #[0,2]
    count = 0
    results = pd.read_csv(path+path_list2[i]+'_DUT_Wilcoxon.txt', sep='\t')
    pcut = results[(results['p_value']<0.05) & (np.abs(results['log2FC'])>1.5)]['Gene Symbol']
    pcut = pcut.drop_duplicates()
    glist = pcut.squeeze().str.strip().to_list()
    print(path_list2[i], len(glist))

    enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                    gene_sets=['GO_Biological_Process_2021'], # 'Reactome_2022', 'GO_Biological_Process_2018'
                    organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                    outdir=None, # don't write to disk
    )

    enrresult = enr.results.sort_values(by=['Adjusted P-value']) 
    file = enrresult
    def string_fraction_to_float(fraction_str):
        numerator, denominator = fraction_str.split('/')
        return float(numerator) / float(denominator)

    file = file.sort_values('Adjusted P-value')
    file['Term'] = file['Term'].str.rsplit(" ",n=1).str[0]
    file = file[file['Adjusted P-value']<0.1]
    print(file)
    file['Adjusted P-value'] = -np.log10(file['Adjusted P-value'])
    file = file[(file["Term"].str.contains("homologous recombination", case=False)) | (file["Term"].str.contains("DNA Damage", case=False)) | (file["Term"].str.contains("cell cycle", case=False)) | (file["Term"].str.contains("Signaling by WNT in cancer", case=False)) | (file["Term"].str.contains("PI3K/AKT Signaling in Cancer", case=False)) | (file["Term"].str.contains("double strand break", case=False))]

    reactomelist = ['Cell Cycle', 'SUMOylation Of DNA Damage Response And Repair Proteins','PI3K/AKT Signaling In Cancer', 'HDR Thru Homologous Recombination (HRR)', 'Signaling By WNT In Cancer']
    #bplist = ['DNA Damage Response', 'Regulation Of Cell Cycle', 'Double-Strand Break Repair Via Homologous Recombination', 'G2/M Transition Of Mitotic Cell Cycle'] ##2023
    bplist = ['cellular response to DNA damage stimulus', 'regulation of cell cycle', 'double-strand break repair via homologous recombination', 'G2/M transition of mitotic cell cycle'] ##2018
    #bplist=['as']
    
    ARbpdutdf = {}
    ARmajorminordf = {}
    
    for term in bplist: #bplist or reactomelist
        newfile = file[file["Term"]==term]
        #print(file)
        # if file.shape[0]>0:
        #     plt.rcParams["font.family"] = "Arial"
        #     plt.rcParams.update({
        #     'axes.titlesize': 13,     # 제목 글꼴 크기
        #     'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
        #     'xtick.labelsize': 12,    # x축 틱 라벨 글꼴 크기
        #     'ytick.labelsize': 12,    # y축 틱 라벨 글꼴 크기
        #     'legend.fontsize': 13,
        #     'legend.title_fontsize': 13, # 범례 글꼴 크기
        #     'figure.titlesize': 15    # figure 제목 글꼴 크기
        #     })
        #     plt.figure(figsize=(4,12))
        #     sns.set_style("whitegrid")
        #     scatter = sns.barplot(
        #         data=file, x='Adjusted P-value', y='Term', color = '#81A263'
        #     )
        #     plt.xlabel('-log10(FDR)')
        #     plt.ylabel('')
        #     plt.show()
            
        if newfile.shape[0]>0:
            DUTlist = results[(results['p_value']<0.05) & (np.abs(results['log2FC'])>1)]
            genelist = newfile.iloc[0,-1].split(';')
            
            majorDUTlist = DUTlist[DUTlist['Transcript-Gene'].isin(majorlist)]
            finalmajorDUTlist = list(majorDUTlist[majorDUTlist['Gene Symbol'].isin(genelist)]['Transcript-Gene'])
            finalDUTlist = list(DUTlist[DUTlist['Gene Symbol'].isin(genelist)]['Transcript-Gene'])
            ARbpdutdf[term] = finalmajorDUTlist
            ARmajorminordf[term] = finalDUTlist
            figdf = TU.loc[finalmajorDUTlist,:]
            figAR = figdf[responder]
            figIR = figdf[nonresponder]
            
            figlist = [figAR,figIR]
            titlelist = ['AR','IR']
            
            
            fig = figlist[int(i/2)]
            pre = fig.iloc[:,1::2]
            post = fig.iloc[:,0::2]
            premean = list(pre.mean(axis=1))
            postmean = list(post.mean(axis=1))
            predf = pd.DataFrame({'TU_mean':premean, 'Treatment':'Pre', 'Transcript': finalmajorDUTlist})
            postdf = pd.DataFrame({'TU_mean':postmean, 'Treatment':'Post', 'Transcript': finalmajorDUTlist})
            
            finaldf = pd.concat([predf,postdf],axis=0)

            plt.figure(figsize=(3,5))
            #sns.set_style("whitegrid")
            sns.set_theme(style='ticks',palette='pastel')
            matplotlib.rcParams['pdf.fonttype'] = 42
            matplotlib.rcParams['ps.fonttype'] = 42
            plt.rcParams["font.family"] = "Arial"
            plt.rcParams.update({
            'axes.titlesize': 13,     # 제목 글꼴 크기
            'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
            'xtick.labelsize': 12,    # x축 틱 라벨 글꼴 크기
            'ytick.labelsize': 12,    # y축 틱 라벨 글꼴 크기
            'legend.fontsize': 11,
            'legend.title_fontsize': 11, # 범례 글꼴 크기
            'figure.titlesize': 14    # figure 제목 글꼴 크기
            })

            ax = sns.boxplot(y='TU_mean', x='Treatment', data=finaldf, 
                        showfliers=True, order=['Pre','Post'], palette='vlag'
                        #meanprops = {"marker":"o", "color":"blue","markeredgecolor": "blue", "markersize":"10"}
                        )
            #ax.set_ylim([0,0.01])
            ax.set_ylabel('mean TU')
            plt.title(term)
            sns.despine()

            from statannot import add_stat_annotation
            add_stat_annotation(ax, data=finaldf, x='Treatment', y='TU_mean',
                                box_pairs=[("Pre", "Post")], 
                                #comparisons_correction=None,
                                test='Wilcoxon',  text_format='star', loc='inside', fontsize=15)

            for p in finaldf['Transcript'].unique():
                subset = finaldf[finaldf['Transcript']==p]
                x = [list(subset['Treatment'])[0], list(subset['Treatment'])[1]]
                y = [list(subset['TU_mean'])[0], list(subset['TU_mean'])[1]]
                plt.plot(x,y, marker="o", markersize=4, color='grey', linestyle="--", linewidth=0.7)
                
            #plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/fig2/Reactome2022_'+titlelist[int(i/2)] +'_'+str(count)+'.png', bbox_inches='tight', dpi=300)
            count +=1
            plt.show()
                
    
    
    
    
    
    
    
# %%
###^^ Bubble Plot ! ####
# data = { ####BP2023
#     'GO Term': [
#         'DNA Damage Response', 'Regulation of Cell Cycle', 'Double-Strand Break Repair Via Homologous Recombination', 'G2/M Transition of Mitotic Cell Cycle'
#     ],
#     'AR (stable)': [7.091102,3.468124,1.988467,1.911641	],
#     'AR (variable)': [np.nan,np.nan,np.nan,np.nan],
#     'IR (stable)': [2.249205  ,np.nan,np.nan,np.nan],
#     'IR (variable)': [np.nan,np.nan,np.nan,np.nan]
# }
# df = pd.DataFrame(data)
# df_melted = df.melt(id_vars='GO Term', var_name='Group', value_name='log10_FDR')

# data = {
#     'GO Term': [
#         'Cell Cycle', 'SUMOylation of DNA Damage Response And Repair Proteins','PI3K/AKT Signaling in Cancer', 'HDR Thru Homologous Recombination (HRR)', 'Signaling By WNT in Cancer'
#     ],
#     'AR (stable)': [10.573852,4.696117,1.632123,1.430105,1.003869 ],
#     'AR (variable)': [3.305896,np.nan,np.nan,1.058028,np.nan],
#     'IR (stable)': [2.249205,1.314123,1.669282,np.nan,np.nan],
#     'IR (variable)': [2.510031,np.nan,np.nan,np.nan,np.nan]
# }
# df = pd.DataFrame(data)
# df_melted = df.melt(id_vars='GO Term', var_name='Group', value_name='log10_FDR')

data = { ####BP2018
    'GO Term': [
        'cellular response to DNA damage stimulus', 'G2/M transition of mitotic cell cycle', 'regulation of cell cycle', 'double-strand break repair via homologous recombination', 
    ],
    'AR (stable)': [10.640785, 6.537368, 4.528068, 3.401801],
    'AR (variable)': [np.nan,np.nan,np.nan,np.nan],
    'IR (stable)': [3.728948,1.055052,1.133219,np.nan],
    'IR (variable)': [np.nan,np.nan,np.nan,np.nan]
}
df = pd.DataFrame(data)
df_melted = df.melt(id_vars='GO Term', var_name='Group', value_name='log10_FDR')

# Initialize the figure
plt.figure(figsize=(7, 3))
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
    'axes.titlesize': 13,     # 제목 글꼴 크기
    'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
    'xtick.labelsize': 12,    # x축 틱 라벨 글꼴 크기
    'ytick.labelsize': 12,    # y축 틱 라벨 글꼴 크기
    'legend.fontsize': 13,
    'legend.title_fontsize': 13, # 범례 글꼴 크기
    'figure.titlesize': 14    # figure 제목 글꼴 크기
    })
sns.set_style("whitegrid")# Create the scatter plot
colors = sns.color_palette("Set3", len(df_melted['Group'].unique()))
for i, group in enumerate(df_melted['Group'].unique()):
    subset = df_melted[df_melted['Group'] == group]
    plt.scatter(
        [group] * len(subset), subset['GO Term'], s=subset['log10_FDR'] * 70, 
        color=colors[i], label=group, alpha=1, edgecolors="black", linewidth=0.5
    )

# Customize the plot
#plt.xlabel("Sample Groups")
#plt.ylabel("GO Terms")
#plt.legend(title="Sample Groups", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=40, ha='right')

# Adjust x and y limits to ensure all bubbles are fully visible
plt.xlim(-0.5, len(df_melted['Group'].unique()) - 0.5)
plt.ylim(-0.5, len(df_melted['GO Term'].unique()) - 0.5)
plt.gcf().subplots_adjust(left=0.6)

# Show plot
plt.tight_layout()
plt.title("Biological Process 2021")

#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/fig2/BP2018_bubbleplot.pdf', dpi=300, bbox_inches='tight')

plt.show()



# %%
###############^^ Forest Plot ###############
names = list(sampleinfo.iloc[0::2,0])
TU_post = TU.iloc[:,0::2]
TU_pre = TU.iloc[:,1::2]
TU_post.columns = names
TU_pre.columns = names
deltaTU = TU_post.subtract(TU_pre)

AR_samples = sampleinfo[sampleinfo['response']==1]
AR_samples = list(AR_samples.iloc[0::2,0])

IR_samples = sampleinfo[sampleinfo['response']==0]
IR_samples = list(IR_samples.iloc[0::2,0])


summary_stats = []
# Calculate statistics for AR samples
for category, transcripts in ARbpdutdf.items():
    if all(transcript in deltaTU.index for transcript in transcripts):
        AR_values = deltaTU.loc[transcripts, AR_samples]
        AR_mean = AR_values.mean().mean()
        AR_std = AR_values.mean(axis=1).std()
        
        summary_stats.append({
            'Category': category,
            #'Transcript': transcript,
            'Group': 'AR',
            'Mean': AR_mean,
            'Std': AR_std
        })
for category, transcripts in ARbpdutdf.items():
    if all(transcript in deltaTU.index for transcript in transcripts):
        IR_values = deltaTU.loc[transcripts, IR_samples]
        IR_mean = IR_values.mean().mean()
        IR_std =IR_values.mean(axis=1).std()
        
        summary_stats.append({
            'Category': category,
            #'Transcript': transcript,
            'Group': 'IR',
            'Mean': IR_mean,
            'Std': IR_std
        })
        
summary_df = pd.DataFrame(summary_stats)


# Set the style
sns.set_style("ticks")
plt.rcParams.update({
    'axes.titlesize': 13,     # 제목 글꼴 크기
    'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
    'xtick.labelsize': 12,    # x축 틱 라벨 글꼴 크기
    'ytick.labelsize': 12,    # y축 틱 라벨 글꼴 크기
    'legend.fontsize': 13,
    'legend.title_fontsize': 13, # 범례 글꼴 크기
    'figure.titlesize': 15    # figure 제목 글꼴 크기
    })
# Initialize the figure
fig, ax = plt.subplots(figsize=(9, 4))

# Define colors for Pre and Post groups
colors = {'AR': '#A2C579', 'IR': '#61A3BA'}
# Plot each category
y_positions = []
labels = []
current_y = 0
category_labels = list(ARbpdutdf.keys())
num_categories = len(category_labels)

for i, category in enumerate(category_labels):
    pre_row = summary_df[(summary_df['Category'] == category) & (summary_df['Group'] == 'AR')]
    post_row = summary_df[(summary_df['Category'] == category) & (summary_df['Group'] == 'IR')]
    
    y_pos_pre = current_y - 0.1
    y_pos_post = current_y + 0.1
    
    # Pre group
    ax.errorbar(pre_row['Mean'], y_pos_pre, xerr=pre_row['Std'], fmt='s', color=colors['AR'], label='AR' if i == 0 else "", markersize=9)
    #ax.text(pre_row['Mean'].values[0], y_pos_pre+0.1, f'{pre_row["Mean"].values[0]:.2f}', va='center', ha='left', color=colors['AR'])
    
    # Post group
    ax.errorbar(post_row['Mean'], y_pos_post, xerr=post_row['Std'], fmt='s', color=colors['IR'], label='IR' if i == 0 else "", markersize=9)
    #ax.text(post_row['Mean'].values[0], y_pos_post+0.1, f'{post_row["Mean"].values[0]:.2f}', va='center', ha='left', color=colors['IR'])
    
    y_positions.append(current_y)
    labels.append(category)
    current_y += 1

# Customize the plot
ax.set_yticks(y_positions)
ax.set_yticklabels(labels)
#ax.set_xlabel('Delta TU')

#ax.legend(title='Group')

# Adjust layout
plt.tight_layout()
sns.despine(left=True)
plt.xlim(left=-0.1, right=0.15)
ax.tick_params(axis='y', which='both', left=False, right=False)
ax.axvline(x=0.0, color='black', linestyle='--', alpha=0.2)

plt.legend()
plt.xticks

plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/fig2/BP2018_forestplot.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
###########^^^^ DEG vs. DUT ###############
R_deg = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DEG/responder_Wilcoxon_DEGresult_FC.txt', sep='\t', index_col=0)
R_dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t')
merged_df = R_dut.merge(R_deg[['Gene Symbol', 'p_value']], on='Gene Symbol', how='left', suffixes=('_DUT', '_DEG'))

# Rename columns as needed
final_df = merged_df[['gene_ENST', 'p_value_DEG', 'p_value_DUT']].rename(columns={
    'gene_ENST': 'transcript',
    'p_value_DEG': 'pval_DEG',
    'p_value_DUT': 'pval_DUT'
})

# Display the final dataframe
print(final_df)

final_df = final_df[final_df['pval_DUT']<0.05]
# Plotting the scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=final_df, x='pval_DEG', y='pval_DUT')
plt.xlabel('pval_DEG')
plt.ylabel('pval_DUT')
plt.title('Scatter Plot of pval_DEG vs pval_DUT')
plt.grid(True)
plt.axhline(y=0, color='grey', linestyle='--')
plt.axvline(x=0, color='grey', linestyle='--')
plt.show()

# %%
bp_majorTU = []

for items in ARbpdutdf.values():
    bp_majorTU.extend(items)

bp_majorTU = list(set(bp_majorTU))

import pickle

# Save to a pickle file
with open('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/featureselection/Reactome2022_featurelist.pkl', 'wb') as file:
    pickle.dump(bp_majorTU, file)
# %%
# %%
bp_majorTU = []

for items in ARmajorminordf.values():
    bp_majorTU.extend(items)

bp_majorTU = list(set(bp_majorTU))

import pickle

# Save to a pickle file
with open('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/featureselection/BP2018_majorminorfeaturelist.pkl', 'wb') as file:
    pickle.dump(bp_majorTU, file)




# %%
#######^^^^ DUT proportion violinplot ############

df_stable = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/DEGDUT/AR_stable_DUT_Wilcoxon.txt', sep='\t', index_col=0)
df_variable = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/DEGDUT/IR_variable_DUT_Wilcoxon.txt', sep='\t', index_col=0)

# Add stability column
df_stable['Stability'] = 'IR Stable'
df_variable['Stability'] = 'IR Variable'

# Combine the two DataFrames
df_combined = pd.concat([df_stable, df_variable])

# Vectorized Filtering
significant_mask = (df_combined['p_value'] < 0.05) & (df_combined['log2FC'].abs() > 1.5)
df_combined['is_significant'] = significant_mask.astype(int)

# Calculate Total Transcripts and Significant Transcripts
total_transcripts = df_combined.groupby('Gene Symbol').size()
significant_transcripts = df_combined.groupby('Gene Symbol')['is_significant'].sum()

# Combine into a single DataFrame
proportion_df = pd.DataFrame({
    'total_transcripts': total_transcripts,
    'significant_transcripts': significant_transcripts
})
proportion_df['Proportion'] = proportion_df['significant_transcripts'] / proportion_df['total_transcripts']

# Merge stability information
proportion_df = proportion_df.merge(df_combined[['Gene Symbol', 'Stability']].drop_duplicates(), left_index=True, right_on='Gene Symbol')

proportions_IR = proportion_df

###############################################################

df_stable = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/DEGDUT/AR_stable_DUT_Wilcoxon.txt', sep='\t', index_col=0)
df_variable = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/DEGDUT/IR_variable_DUT_Wilcoxon.txt', sep='\t', index_col=0)

# Add stability column
df_stable['Stability'] = 'AR Stable'
df_variable['Stability'] = 'AR Variable'

# Combine the two DataFrames
df_combined = pd.concat([df_stable, df_variable])

# Vectorized Filtering
significant_mask = (df_combined['p_value'] < 0.05) & (df_combined['log2FC'].abs() > 1.5)
df_combined['is_significant'] = significant_mask.astype(int)

# Calculate Total Transcripts and Significant Transcripts
total_transcripts = df_combined.groupby('Gene Symbol').size()
significant_transcripts = df_combined.groupby('Gene Symbol')['is_significant'].sum()

# Combine into a single DataFrame
proportion_df = pd.DataFrame({
    'total_transcripts': total_transcripts,
    'significant_transcripts': significant_transcripts
})
proportion_df['Proportion'] = proportion_df['significant_transcripts'] / proportion_df['total_transcripts']

# Merge stability information
proportion_df = proportion_df.merge(df_combined[['Gene Symbol', 'Stability']].drop_duplicates(), left_index=True, right_on='Gene Symbol')


proportions_AR = proportion_df


finaldf = pd.concat([proportions_AR, proportions_IR], axis=0)
finaldf['group'] = 'IR'
finaldf.loc[(finaldf['Stability']=='AR Stable') | (finaldf['Stability']=='AR Variable'),'group'] = 'AR'
finaldf['gene'] = 'Stable'
finaldf.loc[(finaldf['Stability']=='AR Variable') | (finaldf['Stability']=='IR Variable'),'gene'] = 'Variable'

colors = {'AR':'#F9B572', 'IR':'#748E63'}
colors = {'IR': '#81B214', 'AR': '#FFCC29'}

# Create the Violin Plot
plt.figure(figsize=(3, 5))
sns.boxplot(x='gene', y='Proportion', data=finaldf, hue='group',
            linewidth=1, 
            palette=colors, showfliers=True, fliersize=1)
#sns.swarmplot(x='Stability', y='Proportion', data=finaldf, color='k', alpha=0.6)
# Customize Plot
#plt.title('Proportion of Transcripts with Significant Changes by Gene Stability')
plt.xlabel('Gene Class')
plt.ylabel('Proportion of DUTs')
plt.legend(loc='upper right')
sns.despine()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/fig1/proportion_DUT.pdf', dpi=300, bbox_inches='tight')

plt.show()

# %%
##^ number of gDUTs
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 12,    # x축 틱 라벨 글꼴 크기
'ytick.labelsize': 12,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 13,
'legend.title_fontsize': 13, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})
sns.set_style("white")
data = pd.DataFrame({'group': ['AR', 'IR', 'AR','IR'], 'Gene Class':['Stable','Stable','Variable','Variable'],'Value': [9121,4438,1111,462]})
colors = {'AR': '#F9B572', 'IR': '#748E63'}
colors = {'IR': '#81B214', 'AR': '#FFCC29'}

plt.figure(figsize=(3, 5))
ax = sns.barplot(data=data, x='Gene Class', hue='group', y='Value', palette=colors, linewidth=1)
ax.legend().set_title(None)
plt.ylabel('# of genes with DUTs')
sns.despine()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/fig1/numberofgDUTs.pdf', dpi=300, bbox_inches='tight')
plt.show()
# %%
##^ number of DUTs
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 12,    # x축 틱 라벨 글꼴 크기
'ytick.labelsize': 12,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 13,
'legend.title_fontsize': 13, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})
sns.set_style("white")
data = pd.DataFrame({'group': ['AR', 'IR', 'AR','IR'], 'Gene Class':['Stable','Stable','Variable','Variable'],'Value': [22793,6452,2614,695]})
colors = {'AR': '#F9B572', 'IR': '#748E63'}
colors = {'IR': '#81B214', 'AR': '#FFCC29'}
plt.figure(figsize=(3, 5))
ax = sns.barplot(data=data, x='Gene Class', hue='group', y='Value', palette=colors, linewidth=1)
ax.legend().set_title(None)
plt.ylabel('# of DUTs')
sns.despine()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/fig1/numberofDUTs.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
##^distribution of |deltaTU|

ARdut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/DEGDUT/AR_stable_DUT_Wilcoxon.txt', sep='\t', index_col=0)
IRdut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/DEGDUT/IR_stable_DUT_Wilcoxon.txt', sep='\t', index_col=0)
ARdut = ARdut[(ARdut['p_value'] < 0.05) & (ARdut['log2FC'].abs() > 1.5)]
IRdut = IRdut[(IRdut['p_value'] < 0.05) & (IRdut['log2FC'].abs() > 1.5)]
# plt.rcParams["font.family"] = "Arial"
# plt.rcParams.update({
# 'axes.titlesize': 13,     # 제목 글꼴 크기
# 'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
# 'xtick.labelsize': 12,    # x축 틱 라벨 글꼴 크기
# 'ytick.labelsize': 12,    # y축 틱 라벨 글꼴 크기
# 'legend.fontsize': 13,
# 'legend.title_fontsize': 13, # 범례 글꼴 크기
# 'figure.titlesize': 15    # figure 제목 글꼴 크기
# })

plt.figure(figsize=(6, 5))
sns.kdeplot(ARdut['log2FC'], fill=True, alpha=0.4, label='AR', color='#FFCC29')  # 첫 번째 데이터
sns.kdeplot(IRdut['log2FC'], fill=True, alpha=0.4, label='IR', color='#81B214')  # 두 번째 데이터

plt.xlabel('log2FC')
plt.ylabel('Density')
plt.xlim([-20,20])
#plt.ylim([0,0.26])
plt.legend()
#plt.title('major+minor')
sns.despine()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/fig1/log2FC_density.pdf', dpi=300, bbox_inches='tight')
plt.show()
# %%
ARdut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/DEGDUT/AR_stable_DUT_Wilcoxon.txt', sep='\t', index_col=0)
IRdut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/DEGDUT/IR_stable_DUT_Wilcoxon.txt', sep='\t', index_col=0)
ARdut = ARdut[(ARdut['p_value'] < 0.05) & (ARdut['log2FC'].abs() > 1.5)]
IRdut = IRdut[(IRdut['p_value'] < 0.05) & (IRdut['log2FC'].abs() > 1.5)]
major = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/majorlist.txt', sep='\t')
tu_df =pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost_80_transcript_TU.txt', sep='\t', index_col=0)
majorlist = list(major['Transcript-Gene'])
minorlist = tu_df.index.difference(majorlist)
ARdut = ARdut.loc[ARdut.index.isin(minorlist),:]
IRdut = IRdut.loc[IRdut.index.isin(minorlist),:]

plt.figure(figsize=(6, 5))
sns.kdeplot(ARdut['log2FC'], fill=True, alpha=0.4, label='AR', color='#FFCC29')  # 첫 번째 데이터
sns.kdeplot(IRdut['log2FC'], fill=True, alpha=0.4, label='IR', color='#81B214')  # 두 번째 데이터

plt.xlabel('log2FC')
plt.ylabel('Density')
plt.xlim([-13,13])
plt.ylim([0,0.45])
plt.legend()
sns.despine()
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/figures/log2FC_minorTU_density.pdf', dpi=300, bbox_inches='tight')
plt.show()
# %%
major = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = list(major[major['type']=='major']['gene_ENST'])
minorlist = list(major[major['type']=='minor']['gene_ENST'])
ARdut = ARdut.loc[ARdut.index.isin(minorlist),:]
IRdut = IRdut.loc[IRdut.index.isin(minorlist),:]

plt.figure(figsize=(4, 4))
sns.kdeplot(ARdut['log2FC'], fill=True, alpha=0.4, label='AR', color='#FFCC29')  # 첫 번째 데이터
sns.kdeplot(IRdut['log2FC'], fill=True, alpha=0.4, label='IR', color='#81B214')  # 두 번째 데이터

plt.xlabel('log2FC')
plt.ylabel('Density')
plt.xlim([-20,20])
plt.ylim([0,0.26])
plt.legend()
sns.despine()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/fig1/log2FC_minor_density.pdf', dpi=300, bbox_inches='tight')
plt.show()
# %%
import gseapy as gp

geneset1 = ["double-strand break repair via homologous recombination (GO:0000724)"] 
geneset2 = ["replication fork processing (GO:0031297)"]
geneset3 = ["positive regulation of Wnt signaling pathway (GO:0030177)"]
geneset3_1 = ['PI3K/AKT/mTOR  Signaling']
geneset4 = ["DNA damage checkpoint signaling (GO:0000077)","DNA integrity checkpoint signaling (GO:0031570)","DNA damage response, signal transduction by p53 class mediator (GO:0030330)"]

# Pull gene lists for each GO term
go_results = gp.get_library("GO_Biological_Process_2021", organism="Human")
go_hallmark = gp.get_library(name="MSigDB_Hallmark_2020", organism="Human")

# Extract genes from selected GO terms
geneset1 = {term: go_results[term] for term in geneset1 if term in go_results}
geneset2 = {term: go_results[term] for term in geneset2 if term in go_results}
geneset3 = {term: go_results[term] for term in geneset3 if term in go_results}
geneset3_1 = {term: go_hallmark[term] for term in geneset3_1 if term in go_hallmark}
geneset4 = {term: go_results[term] for term in geneset4 if term in go_results}

geneset1 = sorted(set(gene for genes in geneset1.values() for gene in genes))
geneset2 = sorted(set(gene for genes in geneset2.values() for gene in genes))
geneset3 = sorted(set(gene for genes in geneset3.values() for gene in genes))
geneset3_1 = sorted(set(gene for genes in geneset3_1.values() for gene in genes))
geneset4 = sorted(set(gene for genes in geneset4.values() for gene in genes))

geneset3 = list(set(geneset3).union(set(geneset3_1)))

# %%
# %%
ARdut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t', index_col=0)
IRdut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/nonresponder_stable_DUT_Wilcoxon.txt', sep='\t', index_col=0)
ARdut = ARdut[(ARdut['p_value'] < 0.05) & (ARdut['log2FC'].abs() > 1.5)]
IRdut = IRdut[(IRdut['p_value'] < 0.05) & (IRdut['log2FC'].abs() > 1.5)]
ARdf = ARdut.loc[ARdut.index.isin(minorlist),:]
IRdf = IRdut.loc[IRdut.index.isin(minorlist),:]
ARdf = ARdf.loc[ARdf['Gene Symbol'].isin(geneset4),:]
IRdf = IRdf.loc[IRdf['Gene Symbol'].isin(geneset4),:]


plt.figure(figsize=(4, 4))
sns.kdeplot(ARdf['log2FC'], fill=True, alpha=0.4, label='AR', color='#FFCC29')  # 첫 번째 데이터
sns.kdeplot(IRdf['log2FC'], fill=True, alpha=0.4, label='IR', color='#81B214')  # 두 번째 데이터

plt.xlabel('log2FC')
plt.ylabel('Density')
plt.xlim([-20,20])
plt.ylim([0,0.26])
plt.legend()
sns.despine()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/fig1/log2FC_major_density.pdf', dpi=300, bbox_inches='tight')
plt.show()
# %%
