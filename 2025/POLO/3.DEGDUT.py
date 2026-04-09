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
# %%
val_clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2025clinical/83_POLO_clinicalinfo.txt', sep='\t', index_col=0)
geneexp = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/merged_83_gene_TPM.txt', sep='\t', index_col=0)
transexp = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/merged_83_transcript_TPM.txt', sep='\t', index_col=0)

# %%
majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/merged_83_cov5_majorminorlist.txt', sep='\t')
sqanti = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/sqantioutput/sqanti_classification.txt', sep='\t')
sqanti.dropna(axis=1, how='all', inplace=True)

coding = sqanti[['isoform','coding','structural_category']]
majorminor = pd.merge(majorminor, coding, left_on='transcriptid', right_on='isoform', how='left')

group1 = majorminor[(majorminor['type']=='major')&(majorminor['coding']=='coding') & (majorminor['structural_category'].isin(['full-splice_match','novel_in_catalog']))]['Transcript-Gene'].to_list()
group2 = majorminor[(majorminor['type']=='minor')&(majorminor['coding']=='coding')]['Transcript-Gene'].to_list()
group3 = majorminor[(majorminor['type']=='minor')&(majorminor['coding']!='coding')]['Transcript-Gene'].to_list()

# %%
### condition 1 ######
res_list = val_clin[(val_clin['recur']==0)&(val_clin['PFS']>730)].index.to_list()
nonres_list = val_clin[(val_clin['recur']==1)&(val_clin['PFS']<365)].index.to_list()
######################

### condition 2 ######
# res_list = val_clin[(val_clin['PFS']>365)].index.to_list()
# nonres_list = val_clin[(val_clin['recur']==1)&(val_clin['PFS']<365)].index.to_list()
######################

# ### condition 3 ######
# res_list = val_clin[(val_clin['recur']==0)&(val_clin['PFS']>365)].index.to_list()
# nonres_list = val_clin[(val_clin['recur']==1)&(val_clin['PFS']<365)].index.to_list()
# ######################

# # ### condition 4 ######
# res_list = val_clin[(val_clin['recur']==0)&(val_clin['PFS']>540)].index.to_list()
# nonres_list = val_clin[(val_clin['PFS']<540)].index.to_list()
# ######################
# %% ######^^ DEG with pyDESeq2 (in sksurv_env) ####################

##^^ DEG ################

import pandas as pd
import numpy as np
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# ==========================================
# PART 1: Data Preparation & Aggregation
# ==========================================

# 1. Load your data
# Rows: Transcript IDs (e.g., ENST000...), Cols: Sample IDs
tx_counts = pd.read_csv("/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/merged_83_transcript_count.csv", index_col=0)
gene_counts = pd.read_csv("/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/merged_83_gene_count.csv", index_col=0)
# Load annotation (You need a map of Transcript -> Gene and Type)
# This assumes you have processed your GTF into a CSV
tx_meta = majorminor.copy() 
# Example columns: ['transcript_id', 'gene_id', 'transcript_type', 'appris_level']

# 2. Filter for "Functional" Transcripts
# Example logic: Keep only Protein Coding AND (APPRIS Principal OR MANE Select)
#############################** major coding vs. just coding ##########################################
functional_tx = majorminor[(majorminor['type']=='major')&(majorminor['coding']=='coding') & (majorminor['structural_category'].isin(['full-splice_match','novel_in_catalog']))]['transcriptid'].to_list()

# functional_tx = majorminor[(majorminor['coding']=='coding') & (majorminor['structural_category'].isin(['full-splice_match','novel_in_catalog']))]['transcriptid'].to_list()
###########################**##########################################################################

# Filter the count matrix
######################################################** whole vs. only major ##########################
#tx_counts_filtered = tx_counts.loc[tx_counts.index.intersection(functional_tx)]
tx_counts_filtered = tx_counts.copy()
#########################**#############################################################################

print(f"Original Transcripts: {tx_counts.shape[0]}")
print(f"Functional Transcripts: {tx_counts_filtered.shape[0]}")

# 3. Aggregate (Sum) to Gene Level
# Map transcript index to gene ID
tx_to_gene_map = tx_meta.set_index('transcriptid')['genename']
tx_counts_filtered['genename'] = tx_counts_filtered.index.map(tx_to_gene_map)

# Group by Gene ID and Sum
gene_counts_functional = tx_counts_filtered.groupby('genename').sum()

# Round to integers (DESeq2 requires integers)
gene_counts_functional = gene_counts_functional.round().astype(int)

print(f"Final Functional Genes: {gene_counts_functional.shape[0]}")

# ==========================================
# PART 2: Differential Expression (PyDESeq2)
# ==========================================

# 1. Prepare Metadata (Clinical Data)
# Ensure the index matches the columns of gene_counts_functional
clinical_df = val_clin.copy()
clinical_df = clinical_df.loc[clinical_df.index.isin(res_list+nonres_list),:]
clinical_df['group'] = 'responder'
clinical_df.loc[clinical_df.index.isin(nonres_list),'group'] = 'nonresponder'

# Filter clinical data to match the samples in counts (N=45)
common_samples = gene_counts_functional.columns.intersection(clinical_df.index)
counts_matrix = gene_counts_functional[common_samples].T # PyDESeq2 expects (Samples x Genes)
metadata = clinical_df.loc[common_samples]

gene_counts_matrix = gene_counts[common_samples].T
gene_counts_matrix = gene_counts_matrix.fillna
#%%
# Ensure your 'condition' column is set correctly
# Let's say: 1 = Resistant (Non-responder), 0 = Sensitive (Responder)
# PyDESeq2 calculates log2FC as (Level 1 vs Level 0). 
# So if you want Positive FC to mean "Upregulated in Resistant", set Resistant as the primary factor.

# 2. Initialize DESeq2 Object
dds = DeseqDataSet(
    counts=gene_counts_matrix,
    metadata=metadata,
    design_factors="group",  # The column name for classification
    refit_cooks=True,
    n_cpus = 4
)

# 3. Run DESeq2 (Normalization & Dispersion)
dds.deseq2()

# 4. Statistical Test (Wald Test)
stat_res = DeseqStats(
    dds, 
    contrast=["group", "responder", "nonresponder"], # Compare Class 1 (Resistant) vs Class 0 (Sensitive)
)
stat_res.summary()

# 5. Extract Results
res_df = stat_res.results_df
res_df = res_df.sort_values("padj")

res_df.to_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/analysis/con1_geneexp_DEG.txt', sep='\t', index=True)

sigs = res_df[(res_df['padj'] < 0.05) & (abs(res_df['log2FoldChange']) > 0.58)] # log2FC 0.58 is approx 1.5x fold
print(f"Number of DEGs found: {len(sigs)}")


# %%
coding_deg = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/analysis/codingexp_DEG.txt', sep='\t', index_col=0)
major_deg = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/analysis/majorexp_DEG.txt', sep='\t', index_col=0)
whole_deg = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/analysis/wholeexp_DEG.txt', sep='\t', index_col=0)

print(len(coding_deg[(coding_deg['padj'] < 0.1) & (abs(coding_deg['log2FoldChange']) > 0.58)]))
print(len(major_deg[(major_deg['padj'] < 0.1) & (abs(major_deg['log2FoldChange']) > 0.58)]))
print(len(whole_deg[(whole_deg['padj'] < 0.1) & (abs(whole_deg['log2FoldChange']) > 0.58)]))

# %%
glist = whole_deg[(whole_deg['pvalue'] < 0.05) & (abs(whole_deg['log2FoldChange']) > 0.58)].index.to_list()
#glist = coding_deg[(coding_deg['padj'] < 0.01) & (abs(coding_deg['log2FoldChange']) > 0.58)].index.to_list()
#glist = major_deg[(major_deg['padj'] < 0.1) & (abs(major_deg['log2FoldChange']) > 0.58)].index.to_list()

print(len(glist))

import gseapy as gp
enr = gp.enrichr(gene_list=glist, # or "./tests/data/gene_list.txt",
                gene_sets=['GO_Biological_Process_2021','Reactome_2022'
                           #'Reactome_2022'
                           #'GO_Biological_Process_2018','GO_Biological_Process_2023','Reactome_2022'
                           ], 
                organism='human', # don't forget to set organism to the one you desired! e.g. Yeast
                outdir=None, # don't write to disk
)

enrresult = enr.results.sort_values(by=['Adjusted P-value']) 
print(enrresult[enrresult['Adjusted P-value']<0.1]['Term'])

# %%
##^^ pre-ranked GSEA ################

def make_prerank_series(whole_deg: pd.DataFrame, score_col: str = "stat") -> pd.Series:
    df = whole_deg.copy()

    # Keep only valid gene names + scores
    df = df[[score_col]].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # If there are duplicated gene symbols, keep the one with largest |score|
    df["abs_score"] = df[score_col].abs()
    df = df.sort_values("abs_score", ascending=False).drop_duplicates(keep="first")
    df = df.drop(columns=["abs_score"])

    # Create ranked series (index=gene, value=score), sorted descending
    rnk = df[score_col].sort_values(ascending=False)
    rnk.index = rnk.index.astype(str)

    return rnk

rnk = make_prerank_series(whole_deg, score_col="stat")
rnk.head()

import gseapy as gp

gmt_path = "/home/jiye/jiye/refdata/c2.all.v2026.1.Hs.symbols.gmt"  # example (Hallmark)

pre_res = gp.prerank(
    rnk=rnk,
    gene_sets='GO_Biological_Process_2021',
    min_size=15,
    max_size=500,
    permutation_num=10000,
    outdir=None,
    seed=0,
    verbose=True,
)

res_df = pre_res.res2d.sort_values("FDR q-val")
res_df.head(20)
# Pick a term name from res_df.index, e.g.:
res_df = pre_res.res2d.copy()

# gseapy 버전에 따라 term 컬럼명이 다를 수 있어서 자동 탐색
term_col_candidates = ["Term", "term", "Pathway", "pathway", "NAME", "Gene_set", "gene_set"]
term_col = next((c for c in term_col_candidates if c in res_df.columns), None)

if term_col is None:
    # 보통 이 경우는 term이 index에 들어있는 형태
    if res_df.index.dtype == "object":
        top_term = res_df.sort_values("FDR q-val").index[0]
    else:
        raise ValueError(
            "Cannot find term column and index is not term strings. "
            "Print pre_res.res2d.head() to inspect columns/index."
        )
else:
    top_term = (
        res_df.sort_values("FDR q-val")
              .iloc[0][term_col]
    )

top_term = str(top_term)
print("Top term:", top_term)

# pre_res.results의 key들 확인 (디버깅용)
# print(list(pre_res.results.keys())[:5])

#%%
#gseaplot
gp.gseaplot(
    rank_metric=rnk,
    term="mRNA splicing, via spliceosome (GO:0000398)", #"double-strand break repair via homologous recombination (GO:0000724)", "mRNA splicing, via spliceosome (GO:0000398)",
    **pre_res.results["mRNA splicing, via spliceosome (GO:0000398)"], 
)

#plt.savefig("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/DEG_HRR_GSEA.pdf", bbox_inches='tight', dpi=300)
#plt.close()
# %%
