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
'legend.fontsize': 12,
'legend.title_fontsize': 12, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})
sns.set_style("ticks")
# %%
mergedorf = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/majorminor/merged_ORF.txt', sep='\t', index_col=0)
majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t', index_col=0)
#TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0)
originalmajorlist = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/final_202306_discovery_major_TU.txt', sep='\t', index_col=0)
originalmajorlist = originalmajorlist.index.to_list()
TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/116_validation_TU_group.txt', sep='\t', index_col=0)
TU = TU.iloc[:-1,:-1]

#%%

import pandas as pd

# Step 1. Prepare: clean transcript id
majorminor['transcriptid_clean'] = majorminor['transcriptid'].apply(lambda x: x.split('-')[0])
majorminor['putative_peptide'] = majorminor['transcriptid_clean'].map(mergedorf['putative_peptide'])

# Step 2. Group transcripts by ORF sequence
grouped = majorminor.groupby('putative_peptide')

# Step 3. Precompute global mean TU for speed
global_mean_tu = TU.mean(axis=1)

# Step 4. Make transcript_map: maps each transcript -> representative transcript
transcript_map = {}

for peptide, group in grouped:
    if pd.isna(peptide):
        continue  # Skip transcripts without ORF info

    candidates = group['transcriptid'].tolist()

    # Split ENST / MSTRG
    enst_candidates = [c for c in candidates if c.startswith('ENST')]
    mstrg_candidates = [c for c in candidates if c.startswith('MSTRG')]

    if enst_candidates:
        pool = enst_candidates
    else:
        pool = mstrg_candidates

    if len(pool) == 1:
        representative = pool[0]
    else:
        mean_tu_values = global_mean_tu.reindex(pool)

        if mean_tu_values.dropna().empty:
            representative = pool[0]
        else:
            representative = mean_tu_values.idxmax()

    for c in candidates:
        transcript_map[c] = representative

# Step 5. Update majorminor_combined
majorminor_combined = majorminor.copy()
majorminor_combined['representative_transcript'] = majorminor_combined['transcriptid'].map(transcript_map)
majorminor_combined['representative_transcript'] = majorminor_combined['representative_transcript'].combine_first(majorminor_combined['transcriptid'])
import pandas as pd

# 1. Make transcript -> representative_transcript mapping
transcript_to_representative = majorminor_combined['representative_transcript'].to_dict()

# 2. TU에 mapping 적용
TU_mapped = TU.copy()
TU_mapped['representative_transcript'] = TU_mapped.index.map(transcript_to_representative)

# 3. 매칭 실패한 transcript는 없다 (확인)
assert TU_mapped['representative_transcript'].isna().sum() == 0, "Some transcripts not mapped!"

# 4. 대표 transcript 기준으로 합치기
TU_mapped.index = TU_mapped['representative_transcript']
TU_mapped = TU_mapped.drop(columns=['representative_transcript'])
TU_combined = TU_mapped.groupby(level=0).sum()

# 5. TU_combined index에 '-GENENAME' 붙이기
# 준비: representative_transcript -> genename mapping 만들기
transcript_to_genename = majorminor_combined.set_index('transcriptid')['genename'].to_dict()

# index를 수정
new_index = [
    f"{transcript}-{transcript_to_genename.get(transcript, 'Unknown')}"
    for transcript in TU_combined.index
]
TU_combined.index = new_index
TU_combined.index.name = None

# 6. majorminor_final 만들기
# majorminor_combined의 transcriptid가 TU_combined index에 매칭된 것만 남기기
# (주의: TU_combined index는 '-GENENAME' 달린 상태)

# 먼저, transcriptid + genename 형태로 majorminor_combined에서 만들기
majorminor_combined['transcriptid_with_gene'] = majorminor_combined['representative_transcript'] + '-' + majorminor_combined['genename']

# 이제 TU_combined index에 매칭되는 것만 추린다
majorminor_final = majorminor_combined[
    majorminor_combined['transcriptid_with_gene'].isin(TU_combined.index)
].copy()

# 깔끔하게 포맷팅
majorminor_final = majorminor_final.set_index('transcriptid_with_gene')
majorminor_final = majorminor_final[['genename', 'type', 'putative_peptide']]
majorminor_final = majorminor_final[~majorminor_final.index.duplicated(keep='first')]

# %%
majorminor_final.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/ORFcombined/116_combined_majorminorinfo.txt', sep='\t', index=True)
TU_combined.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/ORFcombined/116_combined_TU.txt', sep='\t', index=True)

# %%
# %%
sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.csv', sep=',')

responder = sampleinfo[sampleinfo['response']==1]['sample_full']
nonresponder = sampleinfo[sampleinfo['response']==0]['sample_full']
responder = responder.to_list()
nonresponder = nonresponder.to_list()

list = [responder, nonresponder]
namelist = ['responder', 'nonresponder']

for i in range(2):
    degresult = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DEG/'+namelist[i]+'_Wilcoxon_DEGresult_FC.txt', sep='\t')
    
    DEGlist = set(degresult[degresult['p_value']<0.05]['Gene Symbol'])
    nonDEGlist = set(degresult[degresult['p_value'] > 0.05]['Gene Symbol'])
    
    print(namelist[i]," variable: ", len(DEGlist))
    print(namelist[i]," stable: ", len(nonDEGlist))

    #######*** TU file #########################################################
    filtered_trans = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/ORFcombined/80_combined_TU.txt', sep='\t', index_col=0)
    ##**#################################################################################
    
    filtered_trans['Gene Symbol'] = filtered_trans.index.str.split("-",n=1).str[-1]
    
    # ####^ filter only major transcripts ####
    # major = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t')
    # majortrans = major[major['type']=='major']['gene_ENST'].to_list()
    # filtered_trans = filtered_trans.loc[filtered_trans.index.isin(majortrans)]

    ####^ variable genes: DUT ####
    variable_trans = filtered_trans[filtered_trans['Gene Symbol'].isin(DEGlist)]
    variable_trans = variable_trans[list[i]]


    variable_dut_pval = []
    for index, row in variable_trans.iterrows():
        pre_samples = row[1::2].values  # Even-indexed columns are pre-treatment samples
        post_samples = row[::2].values  # Odd-indexed columns are post-treatment samples
        
        # Perform the paired Wilcoxon test and store the p-value        
        if set([a - b for a, b in zip(pre_samples, post_samples)]) != {0}: 
            w, p = stats.wilcoxon(post_samples, pre_samples)
            variable_dut_pval.append(p)
        else:
            variable_dut_pval.append(1)

    # Create a new DataFrame with geneid and respective p-values
    variable_result = pd.DataFrame({
        'p_value':variable_dut_pval,
    })
    variable_result.index = variable_trans.index
    variable_result['Gene Symbol'] = variable_result.index.str.split("-",n=1).str[-1]

    ##### FC #####
    avg_pre = variable_trans.iloc[:, 1::2].mean(axis=1)
    avg_post = variable_trans.iloc[:, ::2].mean(axis=1)

    fold_change = np.log2(avg_post / avg_pre)
    variable_result['log2FC'] = fold_change
    ##############

    ####^ stable genes: DUT ####
    stable_trans = filtered_trans[filtered_trans['Gene Symbol'].isin(nonDEGlist)]
    stable_trans = stable_trans[list[i]]

    stable_dut_pval = []
    
    for index, row in stable_trans.iterrows():
        pre_samples = row[1::2].values  # Even-indexed columns are pre-treatment samples
        post_samples = row[::2].values  # Odd-indexed columns are post-treatment samples
        
        # Perform the paired Wilcoxon test and store the p-value
        if set([a - b for a, b in zip(pre_samples, post_samples)]) != {0}: 
            w, p = stats.wilcoxon(post_samples, pre_samples)
            stable_dut_pval.append(p)
        else:
            stable_dut_pval.append(1)

    # Create a new DataFrame with geneid and respective p-values
    stable_result = pd.DataFrame({
        'p_value': stable_dut_pval,
    })
    stable_result.index = stable_trans.index
    stable_result['Gene Symbol'] = stable_result.index.str.split("-",n=1).str[-1]
    
    ##### FC #####
    avg_pre = stable_trans.iloc[:, 1::2].mean(axis=1)
    avg_post = stable_trans.iloc[:, ::2].mean(axis=1)

    fold_change = np.log2(avg_post / avg_pre)
    stable_result['log2FC'] = fold_change
    ##############
    
    
    #############
    variable_DUT = variable_result[variable_result['p_value'] < 0.05]['Gene Symbol']
    stable_DUT = stable_result[stable_result['p_value'] < 0.05]['Gene Symbol']
    
    print('variable DUT: ', len(variable_DUT))
    print('stable DUT: ', len(stable_DUT))
    
    variable_result.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/orfcombined/'+namelist[i]+'_variable_DUT_Wilcoxon.txt', sep='\t')
    stable_result.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/orfcombined/'+namelist[i]+'_stable_DUT_Wilcoxon.txt', sep='\t')
# %%
