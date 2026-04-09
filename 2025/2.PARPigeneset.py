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
# import gseapy as gp
# go_results = gp.get_library("GO_Biological_Process_2023", organism="Human")

# with open("/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/GO_Biological_Process_2023.txt", "w", encoding="utf-8") as f:
#     for go_term, genes in go_results.items():
#         f.write(f"{go_term}:\n")
#         f.write(", ".join(genes) + "\n\n")



#%%

AR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t', index_col=0)
IR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUT/nonresponder_stable_DUT_Wilcoxon.txt', sep='\t', index_col=0)

ARdutlist = AR_dut.loc[(AR_dut['p_value']<0.05) & (np.abs(AR_dut['log2FC'])>1.5)].index.to_list()
IRdutlist = IR_dut.loc[(IR_dut['p_value']<0.05) & (np.abs(IR_dut['log2FC'])>1.5)].index.to_list()

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
ARlist = list(set(sampleinfo.loc[(sampleinfo['response']==1),'sample_id']))
IRlist = list(set(sampleinfo.loc[(sampleinfo['response']==0),'sample_id']))

TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0)
TU.columns = TU.columns.str[:-4]
preTU = TU.iloc[:,1::2]
postTU = TU.iloc[:,0::2]

###############################
deltaTU = postTU - preTU
#deltaTU = preTU
###############################
AR_delta = deltaTU.loc[ARdutlist,ARlist]
IR_delta = deltaTU.loc[IRdutlist,IRlist]

majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
majorlist = majorminor[majorminor['type']=='major']['gene_ENST'].to_list()
minorlist = majorminor[majorminor['type']=='minor']['gene_ENST'].to_list()


#%%
# %%
import gseapy as gp
# Pull gene lists for each GO term
go_results = gp.get_library("GO_Biological_Process_2021", organism="Human")
go_hallmark = gp.get_library(name="MSigDB_Hallmark_2020", organism="Human")

#%%
geneset1 = ["double-strand break repair via homologous recombination (GO:0000724)"] #,"double-strand break repair via homologous recombination (GO:0000724)"
geneset2 = ["replication fork processing (GO:0031297)", "replication fork protection (GO:0048478)"]
geneset3 = ["positive regulation of Wnt signaling pathway (GO:0030177)"]
geneset3_1 = ['PI3K/AKT/mTOR  Signaling']
geneset4 = [ "cell cycle G2/M phase transition (GO:0044839)","DNA damage checkpoint signaling (GO:0000077)","DNA integrity checkpoint signaling (GO:0031570)"] #"DNA damage checkpoint signaling (GO:0000077)" #"DNA damage checkpoint signaling (GO:0000077)"] #"DNA damage response, signal transduction by p53 class mediator (GO:0030330)",,"DNA integrity checkpoint signaling (GO:0031570)","signal transduction in response to DNA damage (GO:0042770)"

geneset4 = [ "cell cycle G2/M phase transition (GO:0044839)"]

geneset5 = ["double-strand break repair (GO:0006302)"] #"double-strand break repair via nonhomologous end joining (GO:0006303)", "double-strand break repair (GO:0006302)"

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

#%%

ARdf = pd.DataFrame(AR_delta.mean(axis=1))
IRdf = pd.DataFrame(IR_delta.mean(axis=1))
ARdf['gene'] = ARdf.index.str.split("-",n=1).str[1]
IRdf['gene'] = IRdf.index.str.split("-",n=1).str[1]
ARdf.columns = ['delta TU','gene']
IRdf.columns = ['delta TU','gene']


#%%
genesets = [geneset1, geneset2, geneset3, geneset4]
geneset_names = ["Gene set 1", "Gene set 2", "Gene set 3", "Gene set 4"]
# 2x2 그리드 설정
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

# 서브플롯 순회하며 KDE 플롯 그리기
for i, (geneset, name) in enumerate(zip(genesets, geneset_names)):
    row, col = divmod(i, 2)  # 2x2 배치 결정

    # 데이터 필터링
    ARinput = ARdf.loc[ARdf['gene'].isin(geneset), :]
    IRinput = IRdf.loc[IRdf['gene'].isin(geneset), :]
    
    # ARinput = ARinput.loc[ARinput.index.isin(minorlist), :]
    # IRinput = IRinput.loc[IRinput.index.isin(minorlist), :]

    # KDE 플롯
    sns.kdeplot(ARinput['delta TU'], fill=True, alpha=0.4, label='AR', color='#FFCC29', ax=axes[row, col])
    sns.kdeplot(IRinput['delta TU'], fill=True, alpha=0.4, label='IR', color='#81B214', ax=axes[row, col])
    
    axes[row, col].axvline(x=0, color='grey', linestyle='dashed', linewidth=0.8, alpha=0.4)
    
    # 축 및 제목 설정
    axes[row, col].set_xlabel('delta TU')
    axes[row, col].set_ylabel('Density')
    axes[row, col].set_xlim([-0.4, 0.4])
    axes[row, col].set_title(name)
    axes[row, col].legend()
    sns.despine(ax=axes[row, col])

# 전체 레이아웃 정리
plt.tight_layout()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/fig2/deltaTU_whole_density_grid.pdf', dpi=300, bbox_inches='tight')  # 저장 옵션
plt.show()

# %%

import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator

# 4 gene sets
genesets = [geneset1, geneset2, geneset3, geneset4]
geneset_names = ["Homologous Recombination Restoration", "Replication Fork Stabilization", "Upregulation of Pro-survival Pathway", "Cell Cycle Checkpoints"]  # Titles for each plot

# Create 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

# Iterate over each geneset and plot boxplot with statannotations
for i, (geneset, name) in enumerate(zip(genesets, geneset_names)):
    row, col = divmod(i, 2)  # Determine subplot position

    # Filter data
    ARinput = ARdf.loc[ARdf['gene'].isin(geneset), :]
    IRinput = IRdf.loc[IRdf['gene'].isin(geneset), :]
    
    ARmajor = ARinput.loc[ARinput.index.isin(majorlist), :]
    IRmajor = IRinput.loc[IRinput.index.isin(majorlist), :]
    ARmajor['transcript'] = 'major'
    IRmajor['transcript'] = 'major'
    
    ARminor = ARinput.loc[ARinput.index.isin(minorlist), :]
    IRminor = IRinput.loc[IRinput.index.isin(minorlist), :]
    ARminor['transcript'] = 'non-major'
    IRminor['transcript'] = 'non-major'
    
    
    # Combine data for boxplot
    data = (
        pd.concat([ARmajor.assign(Group="AR"), IRmajor.assign(Group="IR"),ARminor.assign(Group="AR"), IRminor.assign(Group="IR")])
        .reset_index()
    )

    # Boxplot
    ax = axes[row, col]
    sns.boxplot(x="transcript", hue='Group', y="delta TU", data=data, ax=ax,
                palette={"AR": "#FFCC29", "IR": "#81B214"}, width=0.6, showfliers=False)

    # # Add statistical annotation
    pairs = pairs = [(("major", "AR"), ("major", "IR")), (("non-major", "AR"), ("non-major", "IR"))]

    annot = Annotator(ax, pairs, data=data, x="transcript", hue="Group", y="delta TU")
    annot.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=1, correction_format='fdr_bh',)
    annot.apply_and_annotate()

    # Customize labels and title
    ax.set_xlabel('')
    ax.set_ylabel('baseline TU') #delta TU
    ax.set_title(name)
    ax.legend_.remove()

    sns.despine(ax=ax)

# Adjust layout
plt.tight_layout()
# plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/fig1/deltaTU_major_boxplot_grid.pdf', dpi=300, bbox_inches='tight')  # Save option
plt.show()

# %%
###^^^^ plot by sample #######

genesets = [geneset1, geneset2, geneset3, geneset4]
#genesets = [geneset1]
geneset_names = ["Homologous Recombination Restoration", "Replication Fork Stabilization", "Upregulation of Pro-survival Pathway", "Cell Cycle Checkpoint Activation"] 

# Create 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

# Iterate over each geneset and plot boxplot with statannotations
for i, (geneset, name) in enumerate(zip(genesets, geneset_names)):
    row, col = divmod(i, 2)  # Determine subplot position

    AR_delta = deltaTU.loc[ARdutlist,ARlist]
    IR_delta = deltaTU.loc[IRdutlist,IRlist]
    AR_delta['gene'] = AR_delta.index.str.split("-",n=1).str[1]
    IR_delta['gene'] = IR_delta.index.str.split("-",n=1).str[1]

    AR_delta = AR_delta.loc[AR_delta['gene'].isin(geneset),:]
    IR_delta = IR_delta.loc[IR_delta['gene'].isin(geneset),:]

    AR_delta_major = AR_delta.loc[AR_delta.index.isin(majorlist),:]
    AR_delta_minor = AR_delta.loc[AR_delta.index.isin(minorlist),:]

    IR_delta_major = IR_delta.loc[IR_delta.index.isin(majorlist),:]
    IR_delta_minor = IR_delta.loc[IR_delta.index.isin(minorlist),:]

    ARdf_major = pd.DataFrame(AR_delta_major.mean(axis=0), columns=['mean delta TU'])
    ARdf_major['transcript'] = 'major'
    ARdf_minor = pd.DataFrame(AR_delta_minor.mean(axis=0), columns=['mean delta TU'])
    ARdf_minor['transcript'] = 'non-major'
    IRdf_major = pd.DataFrame(IR_delta_major.mean(axis=0), columns=['mean delta TU'])
    IRdf_major['transcript'] = 'major'
    IRdf_minor = pd.DataFrame(IR_delta_minor.mean(axis=0), columns=['mean delta TU'])
    IRdf_minor['transcript'] = 'non-major'
    
    # Combine data for boxplot
    data = (
        pd.concat([ARdf_major.assign(Group="AR"), IRdf_major.assign(Group="IR"),ARdf_minor.assign(Group="AR"), IRdf_minor.assign(Group="IR")])
        .reset_index()
    )

    # Boxplot
    ax = axes[row, col]
    sns.boxplot(x="transcript", hue='Group', y="mean delta TU", data=data, ax=ax,
                palette={"AR": "#FFCC29", "IR": "#81B214"}, width=0.6, showfliers=False)
    sns.swarmplot(x="transcript", hue="Group", y="mean delta TU", data=data, ax=ax,
              palette={"AR": "#977400", "IR": "#4C6F00"}, dodge=True, size=3, alpha=1)

    # # Add statistical annotation
    pairs = pairs = [(("major", "AR"), ("major", "IR")), (("non-major", "AR"), ("non-major", "IR"))]

    annot = Annotator(ax, pairs, data=data, x="transcript", hue="Group", y="mean delta TU")
    annot.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=1, correction_format='fdr_bh',)
    annot.apply_and_annotate()

    # Customize labels and title
    ax.set_xlabel('')
    ax.set_ylabel('baseline TU') #mean Δ TU
    ax.set_title(name)
    ax.legend_.remove()

    sns.despine(ax=ax)

# Adjust layout
plt.tight_layout()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/fig2/preTU_majorminor_boxplot_grid.pdf', dpi=300, bbox_inches='tight')  # Save option
plt.show()



# %%
#^^ comparison group ##

geneset5 = ["double-strand break repair (GO:0006302)"] #"double-strand break repair via nonhomologous end joining (GO:0006303)", "double-strand break repair (GO:0006302)" "DNA repair (GO:0006281)"

geneset5 = {term: go_results[term] for term in geneset5 if term in go_results}
geneset5 = sorted(set(gene for genes in geneset5.values() for gene in genes))

ARdf = pd.DataFrame(AR_delta.mean(axis=1))
IRdf = pd.DataFrame(IR_delta.mean(axis=1))
ARdf['gene'] = ARdf.index.str.split("-",n=1).str[1]
IRdf['gene'] = IRdf.index.str.split("-",n=1).str[1]
ARdf.columns = ['delta TU','gene']
IRdf.columns = ['delta TU','gene']

AR_delta = deltaTU.loc[ARdutlist,ARlist]
IR_delta = deltaTU.loc[IRdutlist,IRlist]
AR_delta['gene'] = AR_delta.index.str.split("-",n=1).str[1]
IR_delta['gene'] = IR_delta.index.str.split("-",n=1).str[1]

AR_delta = AR_delta.loc[AR_delta['gene'].isin(geneset5),:]
IR_delta = IR_delta.loc[IR_delta['gene'].isin(geneset5),:]

AR_delta_major = AR_delta.loc[AR_delta.index.isin(majorlist),:]
AR_delta_minor = AR_delta.loc[AR_delta.index.isin(minorlist),:]

IR_delta_major = IR_delta.loc[IR_delta.index.isin(majorlist),:]
IR_delta_minor = IR_delta.loc[IR_delta.index.isin(minorlist),:]

ARdf_major = pd.DataFrame(AR_delta_major.mean(axis=0), columns=['mean delta TU'])
ARdf_major['transcript'] = 'major'
ARdf_minor = pd.DataFrame(AR_delta_minor.mean(axis=0), columns=['mean delta TU'])
ARdf_minor['transcript'] = 'non-major'
IRdf_major = pd.DataFrame(IR_delta_major.mean(axis=0), columns=['mean delta TU'])
IRdf_major['transcript'] = 'major'
IRdf_minor = pd.DataFrame(IR_delta_minor.mean(axis=0), columns=['mean delta TU'])
IRdf_minor['transcript'] = 'non-major'

# Combine data for boxplot
data = (
    pd.concat([ARdf_major.assign(Group="AR"), IRdf_major.assign(Group="IR"),ARdf_minor.assign(Group="AR"), IRdf_minor.assign(Group="IR")])
    .reset_index()
)

plt.figure(figsize=(4, 5))
ax = sns.boxplot(x="transcript", hue='Group', y="mean delta TU", data=data,
            palette={"AR": "#FFCC29", "IR": "#81B214"}, width=0.6, showfliers=False)
sns.swarmplot(x="transcript", hue="Group", y="mean delta TU", data=data,
            palette={"AR": "#977400", "IR": "#4C6F00"}, dodge=True, size=3, alpha=1)

# # Add statistical annotation
pairs = [(("major", "AR"), ("major", "IR")), (("non-major", "AR"), ("non-major", "IR"))]

annot = Annotator(pairs=pairs, data=data, x="transcript", hue="Group", y="mean delta TU", ax=ax)
annot.configure(test="Mann-Whitney", text_format="star", loc="inside", verbose=1, correction_format='fdr_bh',)
annot.apply_and_annotate()

# Customize labels and title
ax.set_xlabel('')
ax.set_ylabel('mean baseline TU') #mean Δ TU
ax.set_title("Double-strand Break Repair")
ax.legend_.remove()
#plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/fig2/DSBrepair_preTU_majorminor_boxplot.pdf', dpi=300, bbox_inches='tight')
sns.despine(ax=ax)
plt.show()


# %%
########^^^^^^ RANDOM GO SET ##########################

import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# Step 1: Select 300 random GO terms
random_GO_terms = list(go_results.keys())

# Store results
results_major = []
results_minor = []

###############################
deltaTU = postTU - preTU
#deltaTU = preTU
###############################
AR_delta = deltaTU.loc[ARdutlist,ARlist]
IR_delta = deltaTU.loc[IRdutlist,IRlist]
AR_delta['gene'] = AR_delta.index.str.split("-",n=1).str[1]
IR_delta['gene'] = IR_delta.index.str.split("-",n=1).str[1]

#random_GO_terms = ["double-strand break repair via homologous recombination (GO:0000724)"]

# Step 2: Compute log2FC and p-value for each GO term
for go_term in random_GO_terms:
    if go_term not in go_results:
        continue
    
    gene_list = go_results[go_term]  # Genes associated with this GO term
    
    # Filter AR/IR delta values for these genes
    AR_delta_filtered = AR_delta[AR_delta['gene'].isin(gene_list)]
    IR_delta_filtered = IR_delta[IR_delta['gene'].isin(gene_list)]
    
    # Major/Minor filtering
    AR_delta_major = AR_delta_filtered[AR_delta_filtered.index.isin(majorlist)]
    AR_delta_minor = AR_delta_filtered[AR_delta_filtered.index.isin(minorlist)]
    
    IR_delta_major = IR_delta_filtered[IR_delta_filtered.index.isin(majorlist)]
    IR_delta_minor = IR_delta_filtered[IR_delta_filtered.index.isin(minorlist)]
    
    # If major transcripts exist, analyze both major and minor
    if not AR_delta_major.empty and not IR_delta_major.empty:
        # Compute sample-wise mean delta TU
        mean_AR_major = AR_delta_major.mean(axis=0)
        mean_IR_major = IR_delta_major.mean(axis=0)
        
        # log2 Fold Change
        mean_diff_major = mean_AR_major.mean() - mean_IR_major.mean()

        # Statistical test
        p_major = mannwhitneyu(mean_AR_major, mean_IR_major, alternative='two-sided').pvalue #if not mean_AR_major.empty and not mean_IR_major.empty else 1

        # Store results
        results_major.append([go_term, mean_diff_major, p_major])

    # If no major transcripts, only analyze minor
    if not AR_delta_minor.empty and not IR_delta_minor.empty:
        # Compute sample-wise mean delta TU
        mean_AR_minor = AR_delta_minor.mean(axis=0)
        mean_IR_minor = IR_delta_minor.mean(axis=0)
        
        # log2 Fold Change
        mean_diff_minor = mean_AR_minor.mean() - mean_IR_minor.mean()

        # Statistical test
        p_minor = mannwhitneyu(mean_AR_minor, mean_IR_minor, alternative='two-sided').pvalue # if not mean_AR_minor.empty and not mean_IR_minor.empty else 1

        # Store results
        results_minor.append([go_term, mean_diff_minor, p_minor])


#%%
# Convert to DataFrame
df_major = pd.DataFrame(results_major, columns=['GO_term', 'log2FC', 'p_value'])
df_minor = pd.DataFrame(results_minor, columns=['GO_term', 'log2FC', 'p_value'])

df_major["Adjusted P-value"] = multipletests(df_major["p_value"], method="fdr_bh")[1]
df_minor["Adjusted P-value"] = multipletests(df_minor["p_value"], method="fdr_bh")[1]

# Adjust p-values with -log10 transformation
df_major['-log10(p)'] = -np.log10(df_major['p_value'])
df_minor['-log10(p)'] = -np.log10(df_minor['p_value'])


#%%
def plot_volcano(df, title, highlight_terms):
    plt.figure(figsize=(6, 6))
    
    # Scatter plot (log2FC vs. -log10(p))
    sns.scatterplot(data=df, x='log2FC', y='-log10(p)', alpha=0.7, color="#3B6790")
    
    # Highlight specific GO terms
    highlight_df = df[df['GO_term'].isin(highlight_terms)]
    # sns.scatterplot(data=highlight_df, x='log2FC', y='-log10(p)', color='red', s=100, label="Highlighted Terms")

    # Labels and aesthetics
    plt.axhline(-np.log10(0.05), color='gray', linestyle='--', linewidth=1, alpha=0.8)  # p=0.05 threshold
    plt.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.4)  # log2FC=0 threshold
    plt.xlabel("mean ΔTU difference (AR - IR)")
    plt.ylabel("-log10(pval)")
    plt.title(title)
    plt.xlim([-0.6,0.6])
    #plt.legend()
    plt.show()

# Highlight specific GO terms (example: DNA repair related terms)
highlight_terms = ["double-strand break repair via homologous recombination (GO:0000724)",] # "positive regulation of Wnt signaling pathway (GO:0030177)","cell cycle G2/M phase transition (GO:0044839)"
highlight_terms =[]
# Plot volcano for major transcripts
plot_volcano(df_major, "Major Transcripts", highlight_terms)

# Plot volcano for minor transcripts
plot_volcano(df_minor, "Minor Transcripts", highlight_terms)


# %%

geneset1 = ["double-strand break repair via homologous recombination (GO:0000724)"] #,"double-strand break repair via homologous recombination (GO:0000724)"
geneset2 = ["replication fork processing (GO:0031297)", "replication fork protection (GO:0048478)"]
geneset3 = ["positive regulation of Wnt signaling pathway (GO:0030177)"]
geneset3_1 = ['PI3K/AKT/mTOR  Signaling']
geneset4 = [ "cell cycle G2/M phase transition (GO:0044839)","DNA damage checkpoint signaling (GO:0000077)","DNA integrity checkpoint signaling (GO:0031570)"]