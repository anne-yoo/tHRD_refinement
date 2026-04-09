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
import os
import re
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.stats.multitest import multipletests


sns.set(font = 'Arial')
sns.set_style("whitegrid")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# %%
###^^ gencode v38 #################
import re

gtf_path = "/home/jiye/jiye/refdata/gencode.v38.annotation.gtf"

appris_transcripts = []

pattern = re.compile(r'tag "appris_principal_([1-5])"')

with open(gtf_path, 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        
        # 원하는 패턴이 라인에 있는지 체크
        if pattern.search(line):
            # transcript_id 추출
            match = re.search(r'transcript_id "([^"]+)"', line)
            if match:
                transcript_id = match.group(1)
                appris_transcripts.append(transcript_id)

# 중복 제거 (CDS/exon 여러 줄로 반복되므로)
appris_transcripts = list(set(appris_transcripts))

print(len(appris_transcripts))
print(appris_transcripts[:20])

#%%
transinfo = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/merged_83_filtered_transcripts_with_gene_info.tsv', sep='\t', index_col=0)
major = transinfo[transinfo['transcript_id'].isin(appris_transcripts)]
major['Transcript-Gene'] = major['transcript_id'] + '-' + major['gene_name']



#%%
mergedorf = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/merged_83_cov5_ORF.txt', sep='\t', index_col=0)
#major = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/majorlist.txt', sep='\t', index_col=0)
tpmdf = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/merged_83_filtered_transcripts_with_gene_info.tsv', sep='\t', index_col=0) #원래는 tpm 
tpmdf = tpmdf.dropna()
tpmdf['Transcript-Gene'] = tpmdf['transcript_id'] + '-' + tpmdf['gene_name']
majortranslist = set(major['Transcript-Gene']) 
minortranslist =  set(tpmdf['Transcript-Gene']) - majortranslist #tpmdf.index.difference(majortranslist).to_list()

majortranslist = list(majortranslist)
minortranslist = list(minortranslist)

transcript_names = [item.split('-',1)[0] for item in majortranslist]
gene_names = [item.split('-',1)[1] for item in majortranslist]
majorlist = pd.DataFrame({'transcriptid':transcript_names, 'genename':gene_names, 'Transcript-Gene':majortranslist})


transcript_names = [item.split('-')[0] for item in minortranslist]
gene_names = [item.split('-')[1] for item in minortranslist]
minorlist = pd.DataFrame({'transcriptid':transcript_names, 'genename':gene_names, 'Transcript-Gene':minortranslist})
majorlist['type'] = 'major'
minorlist['type'] = 'minor'

#%%
####^^^ 202512: new quant with gencode -> major/minor ##################

# tu = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/80_transcript_TPM.txt', sep='\t', index_col=0)
# translist = tu.index.to_list()
# finalmajorlist = majorlist[majorlist['Transcript-Gene'].isin(translist)]
# finalmajorlist.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/majorlist.txt', sep='\t', index=False)

# %%
major_with_seq = majorlist.join(mergedorf['putative_peptide'], on='transcriptid')
major_with_seq = major_with_seq.rename(columns={'putative_peptide': 'major_peptide'})

minor_with_seq = minorlist.join(mergedorf['putative_peptide'], on='transcriptid')
minor_with_seq = minor_with_seq.rename(columns={'putative_peptide': 'minor_peptide'})

merged = minor_with_seq.merge(
    major_with_seq[['genename', 'transcriptid', 'major_peptide']],
    on='genename',
    suffixes=('_minor', '_major')
)

merged['is_same'] = merged['minor_peptide'] == merged['major_peptide']

same_peptide = (
    merged.groupby('transcriptid_minor')['is_same']
    .any()
    .rename('should_be_major')
)

minorlist = minorlist.join(same_peptide, on='transcriptid')

minorlist.loc[minorlist['should_be_major'] == True, 'type'] = 'major'
minorlist = minorlist.drop(columns=['should_be_major'])


# %%
# for i in range(minorlist.shape[0]):
#     trans = minorlist.iloc[i,0]
#     targetgene = minorlist.iloc[i,1]
    
#     targetmajor = list(majorlist[majorlist['genename']== targetgene]['transcriptid'])
    
#     trans_sequence = mergedorf.loc[trans,'putative_peptide']
    
#     for t in targetmajor:
#         if t in mergedorf.index.to_list() == True:
#             if mergedorf.loc[t,'putative_peptide'] == trans_sequence:
#                 minorlist.iloc[i,3] = 'major'
#             else:
#                 pass

# %%
finaldf = pd.concat([majorlist,minorlist])
finaldf.to_csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/merged_83_cov5_majorminorlist.txt', sep='\t', index=False)
#%%
TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_TU.txt', sep='\t')

# %%
translist = TU['gene_ENST'].to_list()
transcript_names = [item.split('-')[0] for item in translist]
savedf = pd.DataFrame({'gene_ENST':translist, 'transcriptid':transcript_names})

# %%
mergedf = pd.merge(savedf,finaldf, how='inner', left_on='transcriptid', right_on='transcriptid')
# %%
#mergedf.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t', index=False)


# %%
TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_TU.txt', sep='\t')
mergedf = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
TU['type'] = mergedf['type']
# %%
TU['gene'] = TU['gene_ENST'].str.split('-',1).str[1]

# %%
#^***** calculate the TU ratio major/minor
grouped_df = TU.groupby(['gene', 'type']).sum()
new_df = pd.DataFrame(columns=TU.columns[1:-2], index=TU['gene'].unique())
for sample in TU.columns[1:-2]:
    for gene in TU['gene'].unique():
        major_sum = grouped_df.loc[(gene, 'major'), sample] if (gene, 'major') in grouped_df.index else 0
        minor_sum = grouped_df.loc[(gene, 'minor'), sample] if (gene, 'minor') in grouped_df.index else 0
        if minor_sum != 0:
            new_df.at[gene, sample] = major_sum / minor_sum
        else:
            new_df.at[gene, sample] = float('nan')  # Set to NaN if denominator is zero



# %%
new_df.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/majorminor/major2minorratio_mean.txt', index=True, sep='\t')
# %%
