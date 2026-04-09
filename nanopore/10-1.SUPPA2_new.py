#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# %%
event = ['SE','MX','A5','A3','RI','AF','AL']
df = pd.DataFrame()
for e in event:
    input = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/SUPPA2/stringtie_'+e+'_variable_10.ioe', sep='\t')
    input['event'] = e
    input = input[['event_id','alternative_transcripts','event']]
    df = pd.concat([df,input])

AS_summary = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/SUPPA2/AS_summary.txt', sep='\t')
AS_summary.columns = ['event_id','dPSI','pval','event']

trans_class = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/TSSanalysis/sqt_output_classification.txt', sep='\t')
trans_class = trans_class[['isoform','structural_category']]

AS_trans = df

# %%
significant_events = AS_summary[AS_summary['pval'] < 0.05]['event_id']
#significant_events = AS_summary[AS_summary['dPSI'] !=0]['event_id']

# Filter AS_trans for significant events
filtered_trans = AS_trans[AS_trans['event_id'].isin(significant_events)]

# Expand `alternative_transcripts` (comma-delimited) into individual rows
filtered_trans['alternative_transcripts'] = filtered_trans['alternative_transcripts'].str.split(',')
filtered_trans = filtered_trans.explode('alternative_transcripts')

# Merge with transcript classification
merged = filtered_trans.merge(trans_class, left_on='alternative_transcripts', right_on='isoform', how='left')

# Group and count transcripts by event type and structural category
grouped = merged.groupby(['event', 'structural_category']).size().reset_index(name='count')
s_list = ['full-splice_match','novel_in_catalog','novel_not_in_catalog']
grouped = grouped[grouped['structural_category'].isin(s_list)]
grouped['proportion'] = grouped.groupby('structural_category')['count'].transform(lambda x: x / x.sum())

# Plot
plt.figure(figsize=(8, 7))
sns.set_style("ticks")
plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 13,    # x축 틱 라벨 글꼴 크기
'ytick.labelsize': 12,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 12,
'legend.title_fontsize': 12, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})
#sns.barplot(data=grouped, x='event', y='count', hue='structural_category', palette='Set2')
sns.barplot(data=grouped, x='structural_category', y='proportion', hue='event', palette='Set2')

#plt.title('Transcript Counts by AS Event Type and Classification')
plt.xlabel('Structural Category', fontsize=13)
plt.ylabel('proportion', fontsize=13)
#plt.ylabel('Transcript Count', fontsize=13)
#plt.legend(loc='upper right')
plt.legend([],[], frameon=False)
plt.tight_layout()
sns.despine()
plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/SUPPA2/AS_sig_transcript_category.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
novellist = ['novel_in_catalog','novel_not_in_catalog']
check = merged[(merged['event']=='AF') & (merged['structural_category'].isin(novellist))]
checklist = check['isoform'].to_list()

#%%
trans = pd.read_csv('/home/jiye/jiye/nanopore/FINALDATA/matched_transcript_TPM.txt', sep='\t', index_col=0)
det = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/DESeq2/deseq2_det.txt', sep='\t', index_col=0)
det = det.loc[(np.abs(det['log2FoldChange'])<-1.5) & (det['padj']<0.05),:]
det['genename'] = det.index.str.split("-",1).str[1]
det['trans'] = det.index.str.split("-",1).str[0]
detlist = det['trans'].to_list()

# %%
trans = pd.read_csv('/home/jiye/jiye/nanopore/FINALDATA/137_transcript_TPM.txt', sep='\t', index_col=0)

#%%
novellist = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/TSSanalysis/sqt_output_classification.txt', sep='\t')
novellist = novellist[(novellist['structural_category']=='novel_in_catalog') |(novellist['structural_category']=='novel_not_in_catalog') ]['isoform'].to_list()

det = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/DESeq2/deseq2_det.txt', sep='\t', index_col=0)
#det = det.loc[(np.abs(det['log2FoldChange'])>1.5) & (det['padj']<0.05),:]
det = det.loc[(det['log2FoldChange']<-1.5) & (det['padj']<0.05),:]

det['genename'] = det.index.str.split("-",1).str[1]
det['trans'] = det.index.str.split("-",1).str[0]
det = det[det['trans'].isin(novellist)]
detlist = det.index.to_list()

metadata = pd.read_csv("/home/jiye/jiye/nanopore/FINALDATA/137_clinicaldata_final.txt", sep='\t', index_col=0)

# Ensure appropriate data types
check = trans[trans.index.isin(detlist)]
check = check.iloc[:,1::2]
metadata['mean TPM'] = check.mean().to_list()
metadata['CMS'] = metadata['CMS'].str.strip()  # Remove extra spaces
metadata['CMS'] = metadata['CMS'].replace(np.nan, 'undefined') 
order = ['CMS1', 'CMS2', 'CMS3', 'CMS4', 'undefined']

# Convert CMS column to a categorical type with the specified order
metadata['CMS'] = pd.Categorical(metadata['CMS'], categories=order, ordered=True)

# %%
plt.figure(figsize=(5, 7))
ax = sns.boxplot(data=metadata, x='CMS', y='mean TPM', showfliers=False, palette='Set2',order=['CMS1','CMS2','CMS3','CMS4','undefined']
                )
plt.ylim([0,7])
#plt.xticks(rotation=45)


from statannot import add_stat_annotation
add_stat_annotation(ax, data=metadata, x='CMS', y='mean TPM',
                    box_pairs=[('CMS3', 'CMS4'),('CMS2', 'CMS3'),('CMS1', 'CMS3')], 
                    comparisons_correction=None,
                    test='Mann-Whitney',  text_format='star', loc='inside', fontsize=13,
                    #line_offset=-0.1, 
                    #text_offset=-0.1
                    ) 

plt.title('Tumor-downregulated DET expression', pad=0)  
sns.despine()
plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/figures/novel_downDET_CMS_TPM.pdf',bbox_inches='tight', dpi=300)
plt.show()

# %%
