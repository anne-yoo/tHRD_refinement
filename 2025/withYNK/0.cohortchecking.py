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
###^^^ POLO 임상정보 정리 ###################

polo = pd.read_excel('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/POLO/POLO.xlsx')
tpm = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/POLO/83_POLO_transcript_exp.txt', sep='\t',  index_col=0)
# %%
samplelist = tpm.columns.to_list()
polo_filtered = polo[polo['sampleid'].isin(samplelist)]
polo_filtered = polo_filtered.set_index('sampleid')
polo_filtered = polo_filtered[['HRD 결과','첫 투약 시작일','재발유무','재발일자','지우기']]
polo_filtered.columns = ['gHRD','startdate','recur','recurdate','lastOPD']
polo_filtered = polo_filtered.reindex(samplelist)
polo_filtered = polo_filtered.iloc[:-1,:]

# %%
df = polo_filtered.copy()
# 1. Convert recurYN from '유'/'무' to 'Yes'/'No'
df['recur'] = df['recur'].map({'유': 1, '무': 0})

# 2. Make sure all date columns are proper datetime
for col in ['startdate', 'recurdate', 'lastOPD']:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# 3. Compute PFS column
def calc_pfs(row):
    if row['recur'] == 1:  # recurrence happened
        if pd.notna(row['recurdate']):
            return (row['recurdate'] - row['startdate']).days + 1
        else:
            return pd.NA
    else:  # No recurrence
        if pd.notna(row['lastOPD']):
            return (row['lastOPD'] - row['startdate']).days + 1
        else:
            return pd.NA

df['PFS'] = df.apply(calc_pfs, axis=1)

df = df[['gHRD','recur','PFS']]
df['OM/OS'] = 'maintenance'
df['BRCAmut'] = 0
df['gHRD'] = df['gHRD'].map({'Yes': 1, 'No': 0})
df['line'] = '1L'
df['line_binary'] = 'FL'
df['response'] = (df['PFS'] >= 360).astype(int)

# %%
df.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/POLO/83_POLO_clinicalinfo.txt', sep='\t', index=True)


# %%
####^^ 116 dataset ######################
tpm_116 = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_TPM.txt', sep='\t', index_col=0)
clin = pd.read_excel('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2025clinical/nsj_dat_250819.xls', header=1)
clin_2 = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2025clinical/2508_116_clinical_response_tmp.txt', sep='\t', index_col=0)

# %%
samplelist_116 = tpm_116.columns.to_list()
clin['GCgenome'] = clin['GCgenome'].str.replace('SV-OV-F', 'SV-OV-P', regex=False)
clin['GCgenome'] = clin['GCgenome'].astype('string')

# bfD만 남기고, atD는 제거
mask_bfD = clin['GCgenome'].str.endswith('-bfD')
mask_atD = clin['GCgenome'].str.endswith('-atD')

# 1. atD로 끝나는 행 제거
clin_filtered = clin[~mask_atD].copy()

# 2. bfD로 끝나는 행에서 '-bfD' 제거
clin_filtered.loc[mask_bfD, 'GCgenome'] = clin_filtered.loc[mask_bfD, 'GCgenome'].str.replace('-bfD', '', regex=False)
clin_filtered = clin_filtered[clin_filtered['GCgenome'].isin(samplelist_116)]
clin_filtered = clin_filtered[['GCgenome','BRCA_stat','line','setting','drug','parp_start','cause ','recur','recur date','last_opd','GSS','tHRD score']]
clin_filtered = clin_filtered.drop_duplicates(subset=['GCgenome'])
clin_filtered = clin_filtered.set_index('GCgenome')

# %%
df2 = clin_filtered.copy()
df2['setting'] = df2['setting'].map({'Maintenance':'maintenance','salvage':'salvage'})
df2['line'] = df2['line'].map({'1L-m':'1L','2L-m':'2L','3L-m':'3L','4L-m':'4L','5L-m':'5L','6L-m':'6L'})

# 2. Make sure all date columns are proper datetime
for col in ['parp_start', 'recur date', 'last_opd']:
    df2[col] = pd.to_datetime(df2[col], errors='coerce')



# 3. Compute PFS column
def calc_pfs(row):
    if row['recur'] == 1:  # recurrence happened
        if pd.notna(row['recur date']):
            return (row['recur date'] - row['parp_start']).days + 1
        else:
            return pd.NA
    else:  # No recurrence
        if pd.notna(row['last_opd']):
            return (row['last_opd'] - row['parp_start']).days + 1
        else:
            return pd.NA

df2 = df2[['recur','cause ','parp_start','recur date','last_opd','setting']]

# %%
#0: ongoing
#1: PD
#2: AE
#3: 투약완료
#4. 사망

new_clin = pd.merge(clin_2, df2, left_index=True, right_index=True, how='left')

bestORR = ['PR','SD','PD','PR','CR','PR','PD','PR','PD','SD','SD','PR','PD','SD','SD','PR','SD','NaN','NaN','NaN','PD','NaN']
# 1. 조건 정의
mask = (new_clin['setting'] == 'salvage') | (new_clin['OM/OS'] == 'salvage')

# 2. 기본값은 'Maintenance'
new_clin['bestORR'] = 'maintenance'

# 3. 조건에 맞는 행의 인덱스 뽑기
idx = new_clin[mask].index

# 4. 리스트 길이와 행 개수가 맞는지 확인
assert len(idx) == len(bestORR), "리스트 길이와 조건에 맞는 행 개수가 다릅니다!"

new_clin.loc[idx, 'bestORR'] = bestORR

add = ['4L',0,'salvage',1,0.4332492311,19,107,0,'Olaparib',1,1,'2018-05-29','2018-09-19','NaN','salvage','PD']
new_clin.loc['SV-OV-P078',:] = add
new_clin['PFS'] = new_clin.apply(calc_pfs, axis=1)
new_clin = new_clin.rename(columns={'cause ':'cause'})
new_clin = new_clin[new_clin['cause'].isin([0, 1, 2, 3, 4])]

##### * 김유나 교수님 상의해서 수정 #############

new_clin.loc['SV-OV-P169', 'setting'] = 'salvage'
new_clin.loc['SV-OV-P169', 'bestORR'] = 'PR'
new_clin.loc['SV-OV-P170', 'setting'] = 'maintenance'
new_clin.loc['SV-OV-P187', 'setting'] = 'maintenance'
new_clin.loc['SV-OV-P189', 'setting'] = 'maintenance'


new_clin.loc['SV-OV-P188', 'setting'] = 'salvage'
new_clin.loc['SV-OV-P188', 'bestORR'] = 'PD'


new_clin = new_clin.drop(axis=0,index=['SV-OV-P179'])

# %%
final_clin = new_clin[['line','setting','BRCAmt','tHRDscore','gHRDscore','drug','recur','cause','bestORR','PFS',]]

# 원본 보존
final = final_clin.copy()

# --- 전처리 ---
# setting: 소문자/공백정리
final['setting'] = final['setting'].astype('string').str.lower().str.strip()

# BRCAmt: 숫자화(1/0로 가정, 비정상값은 0 처리)
final['BRCAmt'] = pd.to_numeric(final['BRCAmt'], errors='coerce').fillna(0).astype(int)

# line: '1L','2L','3L'... 에서 숫자만 추출
final['line_num'] = (
    final['line'].astype('string').str.extract(r'(\d+)')[0].astype('Int64')
)

# PFS: 숫자화 (결측은 NaN)
final['PFS'] = pd.to_numeric(final['PFS'], errors='coerce')

# bestORR: 대문자 통일
final['bestORR'] = final['bestORR'].astype('string').str.upper().str.strip()

# --- 마스크 정의 ---
maint = final['setting'].eq('maintenance')
salv  = final['setting'].eq('salvage')

# Maintenance 그룹 규칙
m_brca1_1L = maint & (final['BRCAmt'].eq(1)) & (final['line_num'].eq(1)) & (final['PFS'] >= 540)
m_brca1_2p = maint & (final['BRCAmt'].eq(1)) & (final['line_num'] >= 2)   & (final['PFS'] >= 360)

m_brca0_1L = maint & (final['BRCAmt'].eq(0)) & (final['line_num'].eq(1)) & (final['PFS'] >= 360)
m_brca0_2p = maint & (final['BRCAmt'].eq(0)) & (final['line_num'] >= 2)   & (final['PFS'] >= 180)

maint_ok = m_brca1_1L | m_brca1_2p | m_brca0_1L | m_brca0_2p

# Salvage 그룹 규칙: bestORR ∈ {PR, CR, NED}
salv_ok = salv & final['bestORR'].isin(['PR','CR','NED'])

# --- response 산출 ---
final['response'] = 0
final.loc[maint_ok | salv_ok, 'response'] = 1

final_clin = final[['line','setting','BRCAmt','tHRDscore','gHRDscore','drug','recur','cause','bestORR','PFS','response']]

# %%
final_clin.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2025clinical/112_validation_clinicalinfo_new.txt', sep='\t', index=True)




# %%
################^ to YNK ######################

val_tu = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_TU.txt', sep='\t', index_col=0)
val_tpm = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_TPM.txt', sep='\t', index_col=0)
val_gene = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/116_validation_gene_TPM.txt', sep='\t', index_col=0)
val_clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2025clinical/112_validation_clinicalinfo_new.txt', sep='\t', index_col=0)

polo_tu = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/POLO/83_POLO_TU.txt', sep='\t', index_col=0)
polo_tpm = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/POLO/83_POLO_transcript_exp.txt', sep='\t', index_col=0)
polo_gene = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/POLO/83_POLO_gene_exp_TPM.txt', sep='\t', index_col=0)
polo_clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2025clinical/83_POLO_clinicalinfo.txt', sep='\t', index_col=0)

# %%
new_tu = val_tu.loc[val_tu.index[:-2], val_clin.index]
new_tpm = val_tpm.loc[new_tu.index, val_clin.index]
new_gene = val_gene.loc[:, val_clin.index]

new_tu.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/112_PARPi_transcript_usage.txt', sep='\t', index=True)
new_tpm.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/112_PARPi_transcript_exp.txt', sep='\t', index=True)
new_gene.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/112_PARPi_gene_exp.txt', sep='\t', index=True)

# %%
new_tu = polo_tu.copy()
new_tpm = polo_tpm.loc[new_tu.index, :]
new_gene = polo_gene.loc[:, :]

new_tu.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/83_POLO_transcript_usage.txt', sep='\t', index=True)
new_tpm.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/83_POLO_transcript_exp.txt', sep='\t', index=True)
new_gene.to_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/withYNK/83_POLO_gene_exp.txt', sep='\t', index=True)


# %%
