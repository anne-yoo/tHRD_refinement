#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.stats import pearsonr
import random

# %%
##* only pre/post samples
##* duration: short(<=180) | medium(<=365) | long(else)
def makeplotdf():
    WES_BRCA1 = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/pre-post/sample-merged/WES-BRCA1-whole-sample')
    WES_BRCA2 = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/pre-post/sample-merged/WES-BRCA2-whole-sample')
    WGS_BRCA1 = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/pre-post/sample-merged/WGS-BRCA1-whole-sample')
    WGS_BRCA2 = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/AFdata/pre-post/sample-merged/WGS-BRCA2-whole-sample')

    filelist = [WES_BRCA1, WES_BRCA2, WGS_BRCA1, WGS_BRCA2]
    
    plotnamelist = ['WES_BRCA1_plot', 'WES_BRCA2_plot', 'WGS_BRCA1_plot', 'WGS_BRCA2_plot']

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

    plotlist =[]
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

            forplot = plotnamelist[i]
            vars()[forplot] = duration_sorted
            plotlist.append(vars()[forplot])


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
            
            forplot = plotnamelist[i]
            vars()[forplot] = duration_sorted
            plotlist.append(vars()[forplot])

    
    return plotlist

#%%
#** plot!
plotlist = makeplotdf()
for i in range(4):     
    filenamelist = ['WES_BRCA1', 'WES_BRCA2', 'WGS_BRCA1', 'WGS_BRCA2']
    forplot = plotlist[i]   
    sns.set(rc = {'figure.figsize':(6,5)})
    sns.set_style('whitegrid')
    plt.suptitle('<'+filenamelist[i]+': RNA_AF - genome_AF>', fontsize=13)
    orders = ['pre','post']
    sns.boxplot(data=forplot, y='deltaAF', x="therapy_duration", hue='pre/post', hue_order=orders, palette='hls', showfliers=False)
    # plt.legend(loc='upper left')
    # plt.savefig('/home/omics/DATA6/jiye/copycomparison/variantinfo/modifiedAF/modifiedfigures/violinplot/'+filenamelist[i]+'.png', bbox_inches='tight', dpi=150)
    plt.show()
    #*pearson correlation
    pre = forplot[forplot['pre/post']=='pre']
    post = forplot[forplot['pre/post']=='post']
    print(filenamelist[i]+' pre: duration~deltaAF:',pre['Duration'].corr(pre['deltaAF']))
    print(filenamelist[i]+' post: duration~deltaAF:',post['Duration'].corr(post['deltaAF']))









# %%
#** hypothesis 1-1: therapy duration (continuous) ~ mean deltaAF per sample + plot
dflist = makeplotdf()
filenamelist = ['WES_BRCA1', 'WES_BRCA2', 'WGS_BRCA1', 'WGS_BRCA2']
for i in range(4):
    df = dflist[i]
    df = df[df['pre/post']=='pre']
    df = df[df['Purpose']=='Maintenance'] #only maintenance samples

    condition1 = (0.4<=df['genome_AF']) & (df['genome_AF'] <= 0.6)
    df = df.loc[condition1,:]
    syn = df[df['consequence']=='synonymous_variant']
    non = df[df['consequence']!='synonymous_variant']

    meansyn = pd.DataFrame(syn.groupby('sample_ID')['deltaAF'].mean())
    durationsyn = syn[['sample_ID','Duration']]
    durationsyn =durationsyn.drop_duplicates(subset=['sample_ID'])
    durationsyn['variant type'] = 'syn'
    corrsyn = pd.merge(meansyn,durationsyn, on='sample_ID')
    # print(filenamelist[i],corrdf['Duration'].corr(corrdf['deltaAF']))
    # print(pearsonr(corrsyn['Duration'], corrsyn['deltaAF']))

    meannon = pd.DataFrame(non.groupby('sample_ID')['deltaAF'].mean())
    durationnon = non[['sample_ID','Duration']]
    durationnon = durationnon.drop_duplicates(subset=['sample_ID'])
    durationnon['variant type'] = 'non-syn'
    corrnon = pd.merge(meannon,durationnon, on='sample_ID')
    # print(filenamelist[i],corrdf['Duration'].corr(corrdf['deltaAF']))
    # print(pearsonr(corrnon['Duration'], corrnon['deltaAF']))

    finaldf = pd.concat([corrsyn,corrnon])
    sns.set(rc = {'figure.figsize':(4,4)})
    sns.set_style('whitegrid')
    # plt.suptitle('<'+filenamelist[i]+': duration ~ delta AF>', fontsize=11)
    # sns.scatterplot(data=corrdf, x="Duration", y="deltaAF", color='red')
    g=sns.lmplot(data=df, x="Duration", y="deltaAF", fit_reg=True)
    g.set(ylim=(-0.7, 0.7))
    plt.title('<'+filenamelist[i]+': duration ~ delta AF>')
    plt.show()

    # print(filenamelist[i],"whole variant: ", pearsonr(df['Duration'], df['deltaAF']))
    # print(filenamelist[i],"mean per sample: ", pearsonr(finaldf['Duration'], finaldf['deltaAF']))





# %%
#** hypothesis 1-2: therapy duration (continuous) ~ positive d.AF ratio per sample + plot (d.AF >= 0)
dflist = makeplotdf()
for i in range(1):
    df = dflist[i]
    df = df[df['pre/post']=='pre']

    df = df[df['Purpose']=='Maintenance'] #only maintenance samples
    condition1 = (0.4<=df['genome_AF']) & (df['genome_AF'] <= 0.6)
    df = df.loc[condition1,:]

    tmpdf = pd.DataFrame(df.groupby('sample_ID')['deltaAF'])
    countdf = pd.DataFrame(df.groupby('sample_ID')['deltaAF'].apply(lambda x: pd.Series([(x < 0).sum(), (x >= 0).sum()])).unstack())
    countdf.columns = ['others','positive']
    countdf['pos_ratio'] = countdf['positive']/(countdf['others']+countdf['positive'])
    countdf = countdf.rename_axis('sample_ID').reset_index()

    durationdf = df[['sample_ID','Duration']]
    durationdf = durationdf.drop_duplicates(subset=['sample_ID'])

    corrdf = pd.merge(countdf,durationdf, on='sample_ID')

    # syn = df[df['consequence']=='synonymous_variant']
    # non = df[df['consequence']!='synonymous_variant']

    # tmpsyn = pd.DataFrame(syn.groupby('sample_ID')['deltaAF'])
    # countsyn = pd.DataFrame(syn.groupby('sample_ID')['deltaAF'].apply(lambda x: pd.Series([(x < 0).sum(), (x >= 0).sum()])).unstack())
    # countsyn.columns = ['others','positive']
    # countsyn['pos_ratio'] = countsyn['positive']/(countsyn['others']+countsyn['positive'])
    # countsyn = countsyn.rename_axis('sample_ID').reset_index()

    # durationsyn = syn[['sample_ID','Duration']]
    # durationsyn = durationsyn.drop_duplicates(subset=['sample_ID'])
    # durationsyn['variant type'] = 'syn'

    # corrsyn = pd.merge(countsyn,durationsyn, on='sample_ID')

    # tmpnon = pd.DataFrame(non.groupby('sample_ID')['deltaAF'])
    # countnon = pd.DataFrame(non.groupby('sample_ID')['deltaAF'].apply(lambda x: pd.Series([(x < 0).sum(), (x >= 0).sum()])).unstack())
    # countnon.columns = ['others','positive']
    # countnon['pos_ratio'] = countnon['positive']/(countnon['others']+countnon['positive'])
    # countnon = countnon.rename_axis('sample_ID').reset_index()

    # durationnon = non[['sample_ID','Duration']]
    # durationnon = durationnon.drop_duplicates(subset=['sample_ID'])
    # durationnon['variant type'] = 'non-syn'

    # corrnon = pd.merge(countnon,durationnon, on='sample_ID')
    # # print(filenamelist[i],corrdf['Duration'].corr(corrdf['pos_ratio']))
    # # print(pearsonr(corrdf['Duration'],corrdf['pos_ratio']))

    # finaldf = pd.concat([corrsyn,corrnon])


    sns.set(rc = {'figure.figsize':(4,4)})
    sns.set_style('whitegrid')
    # plt.suptitle('<'+filenamelist[i]+': duration ~ positive ratio>', fontsize=11)
    # sns.scatterplot(data=corrdf, x="Duration", y="pos_ratio", color='red')
    g = sns.lmplot(data=corrdf, x="Duration", y="pos_ratio")
    g.set(ylim=(-0.5,1.05))
    plt.title('<'+filenamelist[i]+': duration ~ positive ratio>')
    plt.show()


# %%
#^^ hypothesis 1-3: therapy duration (continuous) ~ mean AI per sample + plot
dflist = makeplotdf()
filenamelist = ['WES_BRCA1', 'WES_BRCA2', 'WGS_BRCA1', 'WGS_BRCA2']
for i in range(4):
    df = dflist[i]
    df = df[df['pre/post']=='pre']

    df = df[df['Purpose']=='Maintenance'] #only maintenance samples
    df['AI'] = df['RNA_AF'] - 0.5

    condition1 = (0.4<=df['genome_AF']) & (df['genome_AF'] <= 0.6)
    df = df.loc[condition1,:]

    syn = df[df['consequence']=='synonymous_variant']
    non = df[df['consequence']!='synonymous_variant']

    meansyn = pd.DataFrame(syn.groupby('sample_ID')['AI'].mean())
    durationsyn = syn[['sample_ID','Duration']]
    durationsyn =durationsyn.drop_duplicates(subset=['sample_ID'])
    durationsyn['variant type'] = 'syn'
    corrsyn = pd.merge(meansyn,durationsyn, on='sample_ID')
    # print(filenamelist[i],corrdf['Duration'].corr(corrdf['deltaAF']))
    # print(pearsonr(corrsyn['Duration'], corrsyn['deltaAF']))

    meannon = pd.DataFrame(non.groupby('sample_ID')['AI'].mean())
    durationnon = non[['sample_ID','Duration']]
    durationnon =durationnon.drop_duplicates(subset=['sample_ID'])
    durationnon['variant type'] = 'non-syn'
    corrnon = pd.merge(meannon,durationnon, on='sample_ID')
    # print(filenamelist[i],corrdf['Duration'].corr(corrdf['deltaAF']))
    # print(pearsonr(corrnon['Duration'], corrnon['deltaAF']))

    finaldf = pd.concat([corrsyn,corrnon])
    sns.set(rc = {'figure.figsize':(4,4)})
    sns.set_style('whitegrid')
    # plt.suptitle('<'+filenamelist[i]+': duration ~ delta AF>', fontsize=11)
    # sns.scatterplot(data=corrdf, x="Duration", y="deltaAF", color='red')
    g = sns.lmplot(data=df, x="Duration", y="AI")
    g.set(ylim=(-0.6,0.6))
    plt.title('<'+filenamelist[i]+': duration ~ AI>')
    plt.show()

    # print(filenamelist[i],"whole variant: ", pearsonr(df['Duration'], df['AI']))
    # print(filenamelist[i],"mean per sample: ", pearsonr(finaldf['Duration'], finaldf['AI']))


# %%
#^^ hypothesis 1-4: therapy duration (continuous) ~ positive AI ratio per sample + plot (d.AF >= 0)
dflist = makeplotdf()
for i in range(1):
    df = dflist[i]
    df = df[df['pre/post']=='pre']

    df = df[df['Purpose']=='Maintenance'] #only maintenance samples
    df['AI'] = df['RNA_AF'] - 0.5

    condition1 = (0.4<=df['genome_AF']) & (df['genome_AF'] <= 0.6)
    df = df.loc[condition1,:]

    tmpdf = pd.DataFrame(df.groupby('sample_ID')['AI'])
    countdf = pd.DataFrame(df.groupby('sample_ID')['AI'].apply(lambda x: pd.Series([(x < 0).sum(), (x >= 0).sum()])).unstack())
    countdf.columns = ['others','positive']
    countdf['pos_ratio'] = countdf['positive']/(countdf['others']+countdf['positive'])
    countdf = countdf.rename_axis('sample_ID').reset_index()

    durationdf = df[['sample_ID','Duration']]
    durationdf = durationdf.drop_duplicates(subset=['sample_ID'])

    corrdf = pd.merge(countdf,durationdf, on='sample_ID')

    # syn = df[df['consequence']=='synonymous_variant']
    # non = df[df['consequence']!='synonymous_variant']

    # tmpsyn = pd.DataFrame(syn.groupby('sample_ID')['AI'])
    # countsyn = pd.DataFrame(syn.groupby('sample_ID')['AI'].apply(lambda x: pd.Series([(x < 0).sum(), (x >= 0).sum()])).unstack())
    # countsyn.columns = ['others','positive']
    # countsyn['pos_ratio'] = countsyn['positive']/(countsyn['others']+countsyn['positive'])
    # countsyn = countsyn.rename_axis('sample_ID').reset_index()

    # durationsyn = syn[['sample_ID','Duration']]
    # durationsyn = durationsyn.drop_duplicates(subset=['sample_ID'])
    # durationsyn['variant type'] = 'syn'

    # corrsyn = pd.merge(countsyn,durationsyn, on='sample_ID')

    # tmpnon = pd.DataFrame(non.groupby('sample_ID')['AI'])
    # countnon = pd.DataFrame(non.groupby('sample_ID')['AI'].apply(lambda x: pd.Series([(x < 0).sum(), (x >= 0).sum()])).unstack())
    # countnon.columns = ['others','positive']
    # countnon['pos_ratio'] = countnon['positive']/(countnon['others']+countnon['positive'])
    # countnon = countnon.rename_axis('sample_ID').reset_index()

    # durationnon = non[['sample_ID','Duration']]
    # durationnon = durationnon.drop_duplicates(subset=['sample_ID'])
    # durationnon['variant type'] = 'non-syn'

    # corrnon = pd.merge(countnon,durationnon, on='sample_ID')
    # # print(filenamelist[i],corrdf['Duration'].corr(corrdf['pos_ratio']))
    # # print(pearsonr(corrdf['Duration'],corrdf['pos_ratio']))

    # finaldf = pd.concat([corrsyn,corrnon])
    sns.set(rc = {'figure.figsize':(4,4)})
    sns.set_style('whitegrid')
    # plt.suptitle('<'+filenamelist[i]+': duration ~ positive ratio>', fontsize=11)
    # sns.scatterplot(data=corrdf, x="Duration", y="pos_ratio", color='red')
    g = sns.lmplot(data=corrdf, x="Duration", y="pos_ratio")
    g.set(ylim=(-0.5,1.05))
    plt.title('<'+filenamelist[i]+': duration ~ positive ratio>')
    plt.show()














# %%
#** therapy duration (short, medium, long) ~ mean deltaAF per sample
dflist = makeplotdf()
for i in range(4):
    df = dflist[i]
    df = df[df['pre/post']=='pre']

    df = df[df['Purpose']=='Maintenance'] #only maintenance samples

    meandf = pd.DataFrame(df.groupby('sample_ID')['deltaAF'].mean())
    duration = df[['sample_ID','therapy_duration']]
    duration = duration.drop_duplicates(subset=['sample_ID'])
    corrdf = pd.merge(meandf,duration, on='sample_ID')
    corrdf['therapy_duration'].replace(['short','medium','long'],[0,1,2], inplace=True)
    # print(filenamelist[i],corrdf['therapy_duration'].corr(corrdf['deltaAF']))
    print(pearsonr(corrdf['therapy_duration'],corrdf['deltaAF']))


# %%
#** therapy duration (short/medium/long) ~ positive d.AF ratio per sample (d.AF >=0 )
dflist = makeplotdf()
for i in range(4):
    df = dflist[i]
    df = df[df['pre/post']=='pre']

    df = df[df['Purpose']=='Maintenance'] #only maintenance samples

    tmpdf = pd.DataFrame(df.groupby('sample_ID')['deltaAF'])
    countdf = pd.DataFrame(df.groupby('sample_ID')['deltaAF'].apply(lambda x: pd.Series([(x < 0).sum(), (x >= 0).sum()])).unstack())
    countdf.columns = ['others','positive']
    countdf['pos_ratio'] = countdf['positive']/(countdf['others']+countdf['positive'])
    countdf = countdf.rename_axis('sample_ID').reset_index()

    duration = df[['sample_ID','therapy_duration']]
    duration = duration.drop_duplicates(subset=['sample_ID'])

    corrdf = pd.merge(countdf,duration, on='sample_ID')
    corrdf['therapy_duration'].replace(['short','medium','long'],[0,1,2], inplace=True)
    print(filenamelist[i],corrdf['therapy_duration'].corr(corrdf['pos_ratio']))
    

#%%
##**count variants per sample

def count(i):
    dflist = makeplotdf()
    clinical = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/final_clinicaldata')
    df = dflist[i]
    df = df[df['pre/post']=='pre']
    df = df[df['Purpose']=='Maintenance'] #only maintenance samples
    clinical = clinical[['Duration', 'sample_ID']]
    countdf = pd.DataFrame(df.groupby('sample_ID')['position'].count())
    final = pd.merge(countdf, clinical, on='sample_ID')
    return final

print(count(2))


# %%
aa = dflist[0]
# %%
