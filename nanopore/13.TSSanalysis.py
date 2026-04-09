#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. SUPPA2 AF 이벤트 파일 읽기
# Tab-delimited 파일 로드, NaN 값 처리
suppa_af = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/TSSanalysis/AF_summary.txt', sep='\t', header=None, names=['event', 'dPSI', 'pval', 'event_type'], na_values=['nan'])

suppa_af = suppa_af.dropna()
#%%

def parse_event_info(event_info):
    parts = event_info.split(':')
    chrom = parts[1]  # Chromosome
    strand = parts[-1]  # Strand (+ or -)

    # Initialize fields
    s1, e1, s2, e2, e3, s3 = None, None, None, None, None, None

    if strand == '+':
        # + strand: s1:e1-s3:s2:e2-s3
        s1 = parts[2]  # s1
        e1, s3 = parts[3].split('-')  # e1 and s3
        s2 = parts[4]
        e2 = parts[5].split('-')[0]

    elif strand == '-':
        # - strand: e1-s2:e2:e1-s3:e3
        e1, s2 = parts[2].split('-')  # e1 and s2
        e2 = parts[3]  # e2
        s3 = parts[4].split('-')[1]
        e3 = parts[5]

    return pd.Series([chrom, strand, s1, e1, s2, e2, e3, s3])





# Apply parsing function to extract coordinates
suppa_af[['chrom', 'strand', 's1', 'e1', 's2', 'e2', 'e3', 's3']] = suppa_af['event'].apply(parse_event_info)

# Convert all coordinates to numeric
suppa_af[['s1', 'e1', 's2', 'e2', 'e3', 's3']] = suppa_af[['s1', 'e1', 's2', 'e2', 'e3', 's3']].apply(pd.to_numeric, errors='coerce')

# Check parsed data
print(suppa_af.head())


#%%

# Load reference TSS file
reference_tss = pd.read_csv(
    '/home/jiye/jiye/nanopore/202411_analysis/TSSanalysis/human.refTSS_v3.1.hg38.bed',
    sep='\t',
    header=None,
    names=['chrom', 'start', 'end', 'name', 'score', 'strand', 's','e','rgb']
)
def assign_first_exon(row):
    if row['strand'] == '+':
        return row['s1'], row['e1']  
    elif row['strand'] == '-':
        return row['s3'], row['e3']  
    else:
        return None, None

# Apply the function to determine the first exon
suppa_af[['first_exon_start', 'first_exon_end']] = suppa_af.apply(assign_first_exon, axis=1, result_type='expand')

# def check_tss_overlap(row, tss_df):
#     overlaps = tss_df[
#         (tss_df['chrom'] == row['chrom']) &
#         (tss_df['strand'] == row['strand']) &  # Match strand
#         (row['first_exon_start'] <= tss_df['end']) &  # Start overlaps TSS
#         (row['first_exon_end'] >= tss_df['start'])    # End overlaps TSS
#     ]
#     return 'Overlap' if not overlaps.empty else 'No Overlap'

def check_tss_overlap(row, tss_df):
    overlaps = tss_df[
        (tss_df['chrom'] == row['chrom']) &
        (tss_df['strand'] == row['strand']) &  # Match strand
        (row['first_exon_start'] <= tss_df['start']+70) &  # Start overlaps TSS
        (row['first_exon_start'] >= tss_df['start']-70)    # End overlaps TSS
    ]
    return 'Overlap' if not overlaps.empty else 'No Overlap'

# Apply overlap checking function
suppa_af['tss_overlap'] = suppa_af.apply(lambda row: check_tss_overlap(row, reference_tss), axis=1)

summary = suppa_af.groupby('tss_overlap')['dPSI'].describe()
print(summary)

#%%
suppa_af.to_csv('/home/jiye/jiye/nanopore/202411_analysis/TSSanalysis/TSS_AF_overlap.txt', sep='\t', index=False)
#%%

sig = suppa_af[suppa_af['pval']<0.05]
overlap = suppa_af[suppa_af['tss_overlap']=='Overlap']
nonoverlap = suppa_af[suppa_af['tss_overlap']=='No Overlap']

import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot of dPSI values for overlap vs no overlap
plt.figure(figsize=(5, 7))
ax = sns.boxplot(data=sig, x='tss_overlap', y='dPSI', showfliers=False, palette='Set2')
sns.stripplot(x='tss_overlap', y='dPSI', data=sig, color='grey', alpha=0.7, jitter=True, s=4, order=["Overlap", "No Overlap"])
#plt.axhline(0, color='red', linestyle='--', linewidth=1) 
from statannot import add_stat_annotation
add_stat_annotation(ax, data=sig, x='tss_overlap', y='dPSI',
                    box_pairs=[("Overlap", "No Overlap")], 
                    comparisons_correction=None,
                    test='Mann-Whitney',  text_format='star', loc='inside', fontsize=15) # Reference line at dPSI = 0
plt.title('Alternative First Exon (AF) Events')
plt.xlabel('TSS Overlap')
plt.ylabel('dPSI')
plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/TSSanalysis/AF_dPSI.pdf',bbox_inches='tight', dpi=300)
plt.show()

#%%
plt.figure(figsize=(5, 7))
sns.countplot(data=sig, x='tss_overlap', palette='Pastel1')

#plt.axhline(0, color='red', linestyle='--', linewidth=1)  # Reference line at dPSI = 0
plt.title('Alternative First Exon (AF) Events')
plt.xlabel('TSS ref overlap')
plt.ylabel('Count')
plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/TSSanalysis/AF_tssoverlap_count.pdf',bbox_inches='tight', dpi=300)
plt.show()

#%%
####^^^ TST ######


###* TST
trans = pd.read_csv('/home/jiye/jiye/nanopore/FINALDATA/matched_transcript_TPM.txt', sep='\t', index_col=0)
trans = trans[trans.apply(lambda x: (x != 0).sum(), axis=1) >= trans.shape[1]*0.2]
tumor = trans.iloc[:,1::2]
normal = trans.iloc[:,0::2]
tumor = tumor[tumor.apply(lambda x: (x != 0).sum(), axis=1) >= tumor.shape[1]*0.2]
normal = normal[normal.apply(lambda x: (x != 0).sum(), axis=1) >= normal.shape[1]*0.2]

tumorlist = tumor.index.str.split("-",1).str[0]
normallist = normal.index.str.split("-",1).str[0]

tst = list(set(tumorlist) - set(normallist))
nst = list(set(normallist) - set(tumorlist))

#%%
###* match transcript ####
columns = ["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attributes"]
first_exons = pd.read_csv("/home/jiye/jiye/nanopore/202411_analysis/TSSanalysis/first_exons.gtf", sep="\t", names=columns)

# Extract transcript_id from the attributes field
def extract_transcript_id(attributes):
    for item in attributes.split(';'):
        if "transcript_id" in item:
            return item.split('"')[1]
    return None

first_exons['transcript_id'] = first_exons['attributes'].apply(extract_transcript_id)

# Keep only relevant columns
first_exons = first_exons[["chrom", "start", "end", "strand", "transcript_id"]]

# Rename columns in first_exons to match suppa_af for merge
first_exons = first_exons.rename(columns={"start": "first_exon_start", "end": "first_exon_end"})

#%%
####^^^ TST vs. NST ###########
t_df = first_exons[first_exons['transcript_id'].isin(tst)]
t_df['type'] = 'tumor-specific'
n_df = first_exons[first_exons['transcript_id'].isin(nst)]
n_df['type'] = 'normal-specific'

nt_df = pd.concat([t_df,n_df], axis=0)

def check_tss_overlap(row, tss_df):
    overlaps = tss_df[
        (tss_df['chrom'] == row['chrom']) &
        (tss_df['strand'] == row['strand']) &  # Match strand
        (row['first_exon_start'] <= tss_df['start']+70) &  # Start overlaps TSS
        (row['first_exon_start'] >= tss_df['start']-70)    # End overlaps TSS
    ]
    return 'Overlap' if not overlaps.empty else 'No Overlap'

nt_df['tss_overlap'] = nt_df.apply(lambda row: check_tss_overlap(row, reference_tss), axis=1)


#%%
plt.figure(figsize=(5, 7))
sns.countplot(data=nt_df, hue='tss_overlap', x='type', palette='Pastel1')

#plt.axhline(0, color='red', linestyle='--', linewidth=1)  # Reference line at dPSI = 0
plt.title('human.refTSS Overlap')
plt.xlabel('Isoform Type')
plt.ylabel('Count')
plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/TSSanalysis/TSSoverlap_count.pdf',bbox_inches='tight', dpi=300)
plt.show()


#%%

# Perform the merge
merged = suppa_af.merge(
    first_exons,
    on=["chrom", "strand", "first_exon_start", "first_exon_end"],
    how="inner"  # Use left join to keep all rows in suppa_af
)

merged = merged.drop_duplicates()

# %%
tst_merged = merged[merged['transcript_id'].isin(tst)]
nst_merged = merged[merged['transcript_id'].isin(nst)]
tst_merged['type'] = 'tumor-specific'
nst_merged['type'] = 'normal-specific'

nt_merged = pd.concat([tst_merged,nst_merged], axis=0)
nt_sig = nt_merged[nt_merged['pval']<0.05]

plt.figure(figsize=(6, 8))
sns.boxplot(data=nt_merged, x='type', y='dPSI', showfliers=False)
plt.axhline(0, color='red', linestyle='--', linewidth=1)  # Reference line at dPSI = 0
plt.title('dPSI Distribution by TSS Overlap')
plt.xlabel('TSS Overlap')
plt.ylabel('dPSI')
plt.show()

# %%
##^^^ gDUT ######
results = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/isoformswitchanalyzer/gDUTlist.txt', sep='\t')
gdut = results['gene_name'].to_list()

match = pd.DataFrame(trans.index)
match['gene'] = match['transcript_id'].str.split("-",1).str[1]
match['trans'] = match['transcript_id'].str.split("-",1).str[0]

gdutlist = match[match['gene'].isin(gdut)]
dut = gdutlist['trans'].to_list()

dut_merged = merged[merged['transcript_id'].isin(dut)]
dut_sig = dut_merged[dut_merged['pval']<0.05]

plt.figure(figsize=(6, 8))
sns.boxplot(data=dut_sig, x='tss_overlap', y='dPSI')
plt.axhline(0, color='red', linestyle='--', linewidth=1)  # Reference line at dPSI = 0
plt.title('dPSI Distribution by TSS Overlap')
plt.xlabel('TSS Overlap')
plt.ylabel('dPSI')
plt.show()



# %%
gene = pd.read_csv('/home/jiye/jiye/nanopore/FINALDATA/137_gene_TPM.txt', sep='\t', index_col=0)
gene = gene.iloc[:,1:]
# %%

aa = gene.loc['KRAS',:]
nor = aa[0::2]
tum = aa[1::2]

# %%
###^^^^^ CAGE Peak #########
####^^match: AF gene id, first exon, cage, TPM
cage = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/TSSanalysis/sqt_output_classification.txt', sep='\t', index_col=0)
cage = cage.loc[:,['dist_to_CAGE_peak','within_CAGE_peak','exons','structural_category']]
af = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/TSSanalysis/AF_summary.txt', sep='\t', header=None, names=['event', 'dPSI', 'pval', 'event_type'], na_values=['nan'])
af = af[af['pval']<0.05]
af['gene_id'] = af['event'].str.split(';').str[0]
match = pd.read_csv('/home/jiye/jiye/nanopore/gtfcompare/stringtiesample/all_transcripts_with_gene_info.tsv', sep='\t')

af_merged = pd.merge(af,match,left_on='gene_id',right_on='gene_id',how='left')

# %%
import numpy as np

translist = trans.index.str.split('-',1).str[0]
af_fil = af_merged[af_merged['transcript_id'].isin(translist)]
firstexonmatchlist = merged['transcript_id'].to_list()
af_fil = af_fil[af_fil['transcript_id'].isin(firstexonmatchlist)]
af_fil = pd.merge(af_fil,cage,left_on='transcript_id', right_index=True, how='left')
filterlist = af_fil['transcript_id'].to_list()

tpm = trans
tpm = tpm.iloc[:,1::2]
tpm['mean TPM'] = tpm.mean(axis=1)
tpm['transcriptid'] = tpm.index.str.split('-',1).str[0]
tpm_fil = tpm.loc[tpm['transcriptid'].isin(filterlist),:]
tpm_fil = tpm_fil.loc[:,['transcriptid','mean TPM']]

finaldf = pd.merge(af_fil,tpm_fil,left_on='transcript_id',right_on='transcriptid',how='left')

finaldf['dist_to_CAGE_peak_abs'] = np.abs(finaldf['dist_to_CAGE_peak'])
#%%
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

finaldf = finaldf.dropna()
finaldf['within_CAGE_peak'] = finaldf['within_CAGE_peak'].map({True: 'True', False: 'False'})

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

# Boxplot of dPSI values for overlap vs no overlap
plt.figure(figsize=(5, 7))
ax = sns.boxplot(data=finaldf, x='within_CAGE_peak', y='mean TPM', showfliers=False, palette='Set2', order=['True','False'])
ax.set_ylim(0, 12) 
from statannot import add_stat_annotation
add_stat_annotation(ax, data=finaldf, x='within_CAGE_peak', y='mean TPM',
                    box_pairs=[('True', 'False')], 
                    #comparisons_correction=None,
                    test='Mann-Whitney',  text_format='star',loc='outside', fontsize=15,
                    line_offset=-0.08,  # Adjust the position of the horizontal line
                    text_offset=-0.2 )# Reference line at dPSI = 0
plt.xlabel('TSS within CAGE peak')
plt.title('Transcripts with AF Events', fontsize=13)
#plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/TSSanalysis/CAGEpeak_TPM.pdf',bbox_inches='tight', dpi=300)
plt.show()

#%%
plt.figure(figsize=(5, 7))
sns.countplot(data=finaldf, x='within_CAGE_peak', palette='Set2', )
#plt.axhline(0, color='red', linestyle='--', linewidth=1)  # Reference line at dPSI = 0
plt.title('Transcripts with AF Events', fontsize=13)
plt.ylabel('Count')
plt.xlabel('TSS within CAGE peak')
#plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/TSSanalysis/CAGEpeak_count.pdf',bbox_inches='tight', dpi=300)
plt.show()

# %%
plt.figure(figsize=(5, 7))
ax = sns.boxplot(data=finaldf, x='within_CAGE_peak', y='mean TPM', showfliers =False, palette='Set2', order=['True','False'])
ax.set_ylim(0, 12) 
from statannot import add_stat_annotation
add_stat_annotation(ax, data=finaldf, x='within_CAGE_peak', y='mean TPM',
                    box_pairs=[('True', 'False')], 
                    #comparisons_correction=None,
                    test='Mann-Whitney',  text_format='star',loc='outside', fontsize=15,
                    line_offset=-0.08,  # Adjust the position of the horizontal line
                    text_offset=-0.2
                    )# Reference line at dPSI = 0
plt.xlabel('TSS within CAGE peak')
plt.title('Transcripts with AF Events', fontsize=13)
#plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/TSSanalysis/CAGEpeak_TPM.pdf',bbox_inches='tight', dpi=300)
plt.show()

# %%
#finaldf.to_csv('/home/jiye/jiye/nanopore/202411_analysis/TSSanalysis/AF_CAGE_info.txt', sep='\t', index=False)

# %%
tpm = trans
n = tpm.iloc[:,0::2]
n['mean TPM'] = n.mean(axis=1)
n['transcriptid'] = n.index.str.split('-',1).str[0]
n = n.loc[n['transcriptid'].isin(filterlist),:]
n['sample'] = 'normal'
n = n[['mean TPM','transcriptid','sample']]

t = tpm.iloc[:,1::2]
t['mean TPM'] = t.mean(axis=1)
t['transcriptid'] = t.index.str.split('-',1).str[0]
t = t.loc[t['transcriptid'].isin(filterlist),:]
t['sample'] = 'tumor'
t = t[['mean TPM','transcriptid','sample']]

nt = pd.concat([n,t],axis=0)
#ntcheck= pd.merge(af_fil,nt,left_on='transcript_id',right_on='transcriptid',how='inner')

plt.figure(figsize=(5, 7))
ax = sns.boxplot(data=nt, x='sample', y='mean TPM', showfliers=False, palette='Set2', order=['normal','tumor'])
# from statannot import add_stat_annotation
# add_stat_annotation(ax, data=nt, x='sample', y='mean TPM',
#                     box_pairs=[('normal', 'tumor')], 
#                     #comparisons_correction=None,
#                     test='Mann-Whitney',  text_format='star',loc='inside', fontsize=15,
#                     )# Reference line at dPSI = 0
plt.show()
t# %%

# %%
