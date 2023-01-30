#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

#%%

WES_BRCA1 = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/sample-merged/WES-BRCA1-whole-sample')
WES_BRCA2 = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/sample-merged/WES-BRCA2-whole-sample')
WGS_BRCA1 = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/sample-merged/WGS-BRCA1-whole-sample')
WGS_BRCA2 = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/sample-merged/WGS-BRCA2-whole-sample')

filelist = [WES_BRCA1, WES_BRCA2, WGS_BRCA1, WGS_BRCA2]
filenamelist = ['WES_BRCA1', 'WES_BRCA2', 'WGS_BRCA1', 'WGS_BRCA2']

clinical = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/final_clinicaldata')

WES_matched_samplelist = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/WES_matched_samplelist.csv', header=None)
WGS_matched_samplelist = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/WGS_matched_samplelist.csv', header=None)
WES_matched_samplelist.columns = ['sample_ID','pre/post']
WGS_matched_samplelist.columns = ['sample_ID','pre/post']

matchedlist = [WES_matched_samplelist,WGS_matched_samplelist]

# %%
#* 1. atD/bfD 뺀 sample name column 추가해서 clinical과 merge
#* 2. duration으로 sorting한 sample name list 만들기
#* 3. durationd으로 sorting + NR/R 나눈 sample name list 만들기

unique_clinical = clinical.drop(['Seq_time'],axis=1)
unique_clinical['sample_ID']=unique_clinical['sample_ID'].str.replace('-atD','')
unique_clinical['sample_ID']=unique_clinical['sample_ID'].str.replace('-bfD','')
unique_clinical.drop_duplicates(['sample_ID'],keep='first',ignore_index=True, inplace=True)
unique_clinical.rename(columns = {'sample_ID':'sample'}, inplace=True)

for i in range(4):
    file = filelist[i]
    file['sample'] = file['sample_ID']
    file['sample'] = file['sample'].str.replace('-atD','')
    file['sample'] = file['sample'].str.replace('-bfD','')

    if i <2:
        j=0
        matchedfile = pd.merge(matchedlist[j], file, on='sample_ID')
        mergedfile = pd.merge(matchedfile, unique_clinical, on='sample')
        duration_sorted = mergedfile.sort_values('Duration')
        duration_sorted['deltaAF'] = duration_sorted['RNA_AF'] - duration_sorted['genome_AF']


        sns.set(rc = {'figure.figsize':(15,5)})
        sns.set_style('whitegrid')
        plt.suptitle('<'+filenamelist[i]+': RNA_AF - genome_AF>', fontsize=13)
        orders = ['pre','post']
        sns.violinplot(data=duration_sorted, y='deltaAF', x="sample", hue='pre/post', hue_order=orders, split=True, palette='hls')
        # plt.legend(loc='upper left')
        # plt.savefig('/home/omics/DATA6/jiye/copycomparison/variantinfo/modifiedAF/modifiedfigures/violinplot/'+filenamelist[i]+'.png', bbox_inches='tight', dpi=150)
        plt.show()

    else:
        j=1
        matchedfile = pd.merge(matchedlist[j], file, on='sample_ID')
        mergedfile = pd.merge(matchedfile, unique_clinical, on='sample')
        duration_sorted = mergedfile.sort_values('Duration')
        duration_sorted['deltaAF'] = duration_sorted['RNA_AF'] - duration_sorted['genome_AF']


        sns.set(rc = {'figure.figsize':(15,5)})
        sns.set_style('whitegrid')
        plt.suptitle('<'+filenamelist[i]+': RNA_AF - genome_AF>', fontsize=13)
        orders = ['pre','post']
        sns.violinplot(data=duration_sorted, y='deltaAF', x="sample", hue='pre/post', hue_order=orders, split=True, palette='hls')
        plt.xticks(rotation=45)

        # plt.legend(loc='upper left')
        # plt.savefig('/home/omics/DATA6/jiye/copycomparison/variantinfo/modifiedAF/modifiedfigures/violinplot/'+filenamelist[i]+'.png', bbox_inches='tight', dpi=150)
        plt.show()









# %%
##& file list 수정 후 (genome post = genome pre)
WES_BRCA1 = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/pre-post/sample-merged/WES-BRCA1-whole-sample')
WES_BRCA2 = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/pre-post/sample-merged/WES-BRCA2-whole-sample')
WGS_BRCA1 = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/pre-post/sample-merged/WGS-BRCA1-whole-sample')
WGS_BRCA2 = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/pre-post/sample-merged/WGS-BRCA2-whole-sample')

filelist = [WES_BRCA1, WES_BRCA2, WGS_BRCA1, WGS_BRCA2]
filenamelist = ['WES_BRCA1', 'WES_BRCA2', 'WGS_BRCA1', 'WGS_BRCA2']

clinical = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/final_clinicaldata')

onlypre_WES_matched_samplelist = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/only-pre_WES_matched_samplelist', sep="\t", header=None)
onlypre_WGS_matched_samplelist = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/only-pre_WGS_matched_samplelist', sep="\t", header=None)
onlypre_WES_matched_samplelist.columns = ['sample_ID','pre/post','wes','rna']
onlypre_WGS_matched_samplelist.columns = ['sample_ID','pre/post','wgs','rna']
matchedlist = [onlypre_WES_matched_samplelist,onlypre_WGS_matched_samplelist]

unique_clinical = clinical.drop(['Seq_time'],axis=1)
unique_clinical['sample_ID']=unique_clinical['sample_ID'].str.replace('-atD','')
unique_clinical['sample_ID']=unique_clinical['sample_ID'].str.replace('-bfD','')
unique_clinical.drop_duplicates(['sample_ID'],keep='first',ignore_index=True, inplace=True)
unique_clinical.rename(columns = {'sample_ID':'sample'}, inplace=True)

for i in range(2,3):
    file = filelist[i]
    file['sample'] = file['sample_ID']
    file['sample'] = file['sample'].str.replace('-atD','')
    file['sample'] = file['sample'].str.replace('-bfD','')

    if i <2:
        j=0
        matchedfile = pd.merge(matchedlist[j], file, on='sample_ID')
        mergedfile = pd.merge(matchedfile, unique_clinical, on='sample')
        duration_sorted = mergedfile.sort_values('Duration')
        duration_sorted['deltaAF'] = duration_sorted['RNA_AF'] - duration_sorted['genome_AF']


        sns.set(rc = {'figure.figsize':(15,5)})
        sns.set_style('whitegrid')
        plt.suptitle('<'+filenamelist[i]+': RNA_AF - genome_AF>', fontsize=13)
        orders = ['pre','post']
        sns.boxplot(data=duration_sorted, y='deltaAF', x="sample", hue='pre/post', hue_order=orders, palette='hls',showfliers=False)
        # plt.legend(loc='upper left')
        # plt.savefig('/home/omics/DATA6/jiye/copycomparison/variantinfo/modifiedAF/modifiedfigures/violinplot/'+filenamelist[i]+'.png', bbox_inches='tight', dpi=150)
        plt.show()

    else:
        j=1
        matchedfile = pd.merge(matchedlist[j], file, on='sample_ID')
        mergedfile = pd.merge(matchedfile, unique_clinical, on='sample')
        duration_sorted = mergedfile.sort_values('Duration')
        duration_sorted['deltaAF'] = duration_sorted['RNA_AF'] - duration_sorted['genome_AF']


        sns.set(rc = {'figure.figsize':(15,5)})
        sns.set_style('whitegrid')
        plt.suptitle('<'+filenamelist[i]+': RNA_AF - genome_AF>', fontsize=13)
        orders = ['pre','post']
        sns.boxplot(data=duration_sorted, y='deltaAF', x="sample", hue='pre/post', hue_order=orders,palette='hls',showfliers=False)
        plt.xticks(rotation=45)

        # plt.legend(loc='upper left')
        # plt.savefig('/home/omics/DATA6/jiye/copycomparison/variantinfo/modifiedAF/modifiedfigures/violinplot/'+filenamelist[i]+'.png', bbox_inches='tight', dpi=150)
        plt.show()








# %%
##* duration: short(<=188; 0.3) | medium(<=422; 0.6) | long(else)

WES_BRCA1 = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/pre-post/sample-merged/WES-BRCA1-whole-sample')
WES_BRCA2 = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/pre-post/sample-merged/WES-BRCA2-whole-sample')
WGS_BRCA1 = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/pre-post/sample-merged/WGS-BRCA1-whole-sample')
WGS_BRCA2 = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/pre-post/sample-merged/WGS-BRCA2-whole-sample')

filelist = [WES_BRCA1, WES_BRCA2, WGS_BRCA1, WGS_BRCA2]
filenamelist = ['WES_BRCA1', 'WES_BRCA2', 'WGS_BRCA1', 'WGS_BRCA2']

clinical = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/final_clinicaldata')

onlypre_WES_matched_samplelist = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/only-pre_WES_matched_samplelist', sep="\t", header=None)
onlypre_WGS_matched_samplelist = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/only-pre_WGS_matched_samplelist', sep="\t", header=None)
onlypre_WES_matched_samplelist.columns = ['sample_ID','pre/post','wes','rna']
onlypre_WGS_matched_samplelist.columns = ['sample_ID','pre/post','wgs','rna']
matchedlist = [onlypre_WES_matched_samplelist,onlypre_WGS_matched_samplelist]

unique_clinical = clinical.drop(['Seq_time'],axis=1)
unique_clinical['sample_ID']=unique_clinical['sample_ID'].str.replace('-atD','')
unique_clinical['sample_ID']=unique_clinical['sample_ID'].str.replace('-bfD','')
unique_clinical.drop_duplicates(['sample_ID'],keep='first',ignore_index=True, inplace=True)
unique_clinical.rename(columns = {'sample_ID':'sample'}, inplace=True)

# #%%
# save unique clinical data
# unique_clinical.to_csv('/home/jiye/jiye/copycomparison/control_PD/unique_clinicaldata.csv', index=False)




#%%
##& only "non-synonymous" variants
for i in range(4):
    file = filelist[i]
    file['sample'] = file['sample_ID']
    file['sample'] = file['sample'].str.replace('-atD','')
    file['sample'] = file['sample'].str.replace('-bfD','')
    

    if i <2:
        j=0
        matchedfile = pd.merge(matchedlist[j], file, on='sample_ID')
        mergedfile = pd.merge(matchedfile, unique_clinical, on='sample')
        duration_sorted = mergedfile.sort_values('Duration')
        duration_sorted['deltaAF'] = duration_sorted['RNA_AF'] - duration_sorted['genome_AF']
        
        duration_sorted['therapy_duration'] = 'tmp'
        # duration_sorted['Duration'] = duration_sorted['Duration'].astype(int)
        duration_sorted.loc[duration_sorted['Duration']<=180,'therapy_duration'] = 'short'
        duration_sorted.loc[(duration_sorted['Duration']>180) & (duration_sorted['Duration']<=365),'therapy_duration'] ='medium'
        duration_sorted.loc[duration_sorted['Duration']>365,'therapy_duration'] = 'long'

        duration_sorted = duration_sorted[duration_sorted['impact']!='LOW']
        sns.set(rc = {'figure.figsize':(6,5)})
        sns.set_style('whitegrid')
        plt.suptitle('<'+filenamelist[i]+': RNA_AF - genome_AF>', fontsize=13)
        orders = ['pre','post']
        sns.boxplot(data=duration_sorted, y='deltaAF', x="therapy_duration", hue='pre/post', hue_order=orders, palette='hls', showfliers=False)
        # plt.legend(loc='upper left')
        # plt.savefig('/home/omics/DATA6/jiye/copycomparison/variantinfo/modifiedAF/modifiedfigures/violinplot/'+filenamelist[i]+'.png', bbox_inches='tight', dpi=150)
        plt.show()
        #*pearson correlation
        pre = duration_sorted[duration_sorted['pre/post']=='pre']
        post = duration_sorted[duration_sorted['pre/post']=='post']
        print('pre: duration~deltaAF:',pre['Duration'].corr(pre['deltaAF']))
        print('post: duration~deltaAF:',post['Duration'].corr(post['deltaAF']))

    else:
        j=1
        matchedfile = pd.merge(matchedlist[j], file, on='sample_ID')
        mergedfile = pd.merge(matchedfile, unique_clinical, on='sample')
        duration_sorted = mergedfile.sort_values('Duration')
        duration_sorted['deltaAF'] = duration_sorted['RNA_AF'] - duration_sorted['genome_AF']

        duration_sorted['therapy_duration'] = 'tmp'
        # duration_sorted['Duration'] = duration_sorted['Duration'].astype(int)
        duration_sorted.loc[duration_sorted['Duration']<=180,'therapy_duration'] = 'short'
        duration_sorted.loc[(duration_sorted['Duration']>180) & (duration_sorted['Duration']<=365),'therapy_duration'] ='medium'
        duration_sorted.loc[duration_sorted['Duration']>365,'therapy_duration'] = 'long'

        duration_sorted = duration_sorted[duration_sorted['impact']!='LOW']
        sns.set(rc = {'figure.figsize':(6,5)})
        sns.set_style('whitegrid')
        plt.suptitle('<'+filenamelist[i]+': RNA_AF - genome_AF>', fontsize=13)
        orders = ['pre','post']
        sns.boxplot(data=duration_sorted, y='deltaAF', x="therapy_duration", hue='pre/post', hue_order=orders,  palette='hls',showfliers=False)
        plt.xticks(rotation=45)

        # plt.legend(loc='upper left')
        # plt.savefig('/home/omics/DATA6/jiye/copycomparison/variantinfo/modifiedAF/modifiedfigures/violinplot/'+filenamelist[i]+'.png', bbox_inches='tight', dpi=150)
        plt.show()

        #*pearson correlation
        pre = duration_sorted[duration_sorted['pre/post']=='pre']
        post = duration_sorted[duration_sorted['pre/post']=='post']
        print('pre: duration~deltaAF:',pre['Duration'].corr(pre['deltaAF']))
        print('post: duration~deltaAF:',post['Duration'].corr(post['deltaAF']))

# %%
#%%
##& only "LOW" variants
for i in range(4):
    file = filelist[i]
    file['sample'] = file['sample_ID']
    file['sample'] = file['sample'].str.replace('-atD','')
    file['sample'] = file['sample'].str.replace('-bfD','')
    

    if i <2:
        j=0
        matchedfile = pd.merge(matchedlist[j], file, on='sample_ID')
        mergedfile = pd.merge(matchedfile, unique_clinical, on='sample')
        duration_sorted = mergedfile.sort_values('Duration')
        duration_sorted['deltaAF'] = duration_sorted['RNA_AF'] - duration_sorted['genome_AF']
        
        duration_sorted['therapy_duration'] = 'tmp'
        # duration_sorted['Duration'] = duration_sorted['Duration'].astype(int)
        duration_sorted.loc[duration_sorted['Duration']<=180,'therapy_duration'] = 'short'
        duration_sorted.loc[(duration_sorted['Duration']>180) & (duration_sorted['Duration']<=365),'therapy_duration'] ='medium'
        duration_sorted.loc[duration_sorted['Duration']>365,'therapy_duration'] = 'long'

        duration_sorted = duration_sorted[duration_sorted['impact']=='LOW']
        sns.set(rc = {'figure.figsize':(6,5)})
        sns.set_style('whitegrid')
        plt.suptitle('<'+filenamelist[i]+': RNA_AF - genome_AF>', fontsize=13)
        orders = ['pre','post']
        sns.boxplot(data=duration_sorted, y='deltaAF', x="therapy_duration", hue='pre/post', hue_order=orders, palette='hls', showfliers=False)
        # plt.legend(loc='upper left')
        # plt.savefig('/home/omics/DATA6/jiye/copycomparison/variantinfo/modifiedAF/modifiedfigures/violinplot/'+filenamelist[i]+'.png', bbox_inches='tight', dpi=150)
        plt.show()
        #*pearson correlation
        pre = duration_sorted[duration_sorted['pre/post']=='pre']
        post = duration_sorted[duration_sorted['pre/post']=='post']
        print('pre: duration~deltaAF:',pre['Duration'].corr(pre['deltaAF']))
        print('post: duration~deltaAF:',post['Duration'].corr(post['deltaAF']))

    else:
        j=1
        matchedfile = pd.merge(matchedlist[j], file, on='sample_ID')
        mergedfile = pd.merge(matchedfile, unique_clinical, on='sample')
        duration_sorted = mergedfile.sort_values('Duration')
        duration_sorted['deltaAF'] = duration_sorted['RNA_AF'] - duration_sorted['genome_AF']

        duration_sorted['therapy_duration'] = 'tmp'
        # duration_sorted['Duration'] = duration_sorted['Duration'].astype(int)
        duration_sorted.loc[duration_sorted['Duration']<=180,'therapy_duration'] = 'short'
        duration_sorted.loc[(duration_sorted['Duration']>180) & (duration_sorted['Duration']<=365),'therapy_duration'] ='medium'
        duration_sorted.loc[duration_sorted['Duration']>365,'therapy_duration'] = 'long'

        duration_sorted = duration_sorted[duration_sorted['impact']=='LOW']
        sns.set(rc = {'figure.figsize':(6,5)})
        sns.set_style('whitegrid')
        plt.suptitle('<'+filenamelist[i]+': RNA_AF - genome_AF>', fontsize=13)
        orders = ['pre','post']
        sns.boxplot(data=duration_sorted, y='deltaAF', x="therapy_duration", hue='pre/post', hue_order=orders,  palette='hls',showfliers=False)
        plt.xticks(rotation=45)

        # plt.legend(loc='upper left')
        # plt.savefig('/home/omics/DATA6/jiye/copycomparison/variantinfo/modifiedAF/modifiedfigures/violinplot/'+filenamelist[i]+'.png', bbox_inches='tight', dpi=150)
        plt.show()

        #*pearson correlation
        pre = duration_sorted[duration_sorted['pre/post']=='pre']
        post = duration_sorted[duration_sorted['pre/post']=='post']
        print('pre: duration~deltaAF:',pre['Duration'].corr(pre['deltaAF']))
        print('post: duration~deltaAF:',post['Duration'].corr(post['deltaAF']))






# %%
##* 
for i in range(4):
    filename = filenamelist[i]
    file = filelist[i]
    vars()[filename] = pd.DataFrame(file.groupby('sample_ID')['RNA_AF']. mean())