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
# %%
#** hypothesis 2-1: pre → post: change of delta AF per sample
dflist = makeplotdf()
filenamelist = ['WES_BRCA1', 'WES_BRCA2', 'WGS_BRCA1', 'WGS_BRCA2']
for i in range(4):
    df = dflist[i]
    df = df[df['Purpose']=='Maintenance']
    condition1 = (0.4<=df['genome_AF']) & (df['genome_AF'] <= 0.6)
    df = df.loc[condition1,:]
    df = df[['sample','pre/post','deltaAF','consequence']]

    df['variant type'] = 'non-syn'
    df.loc[df['consequence']=='synonymous_variant','variant type'] = 'syn'

    sns.set(rc = {'figure.figsize':(4,4)})
    sns.set_style('whitegrid')
    sns.catplot(x ='pre/post', y ='deltaAF', hue = 'sample', data = df, kind='point', palette='tab10', order=['pre','post'], height = 5, aspect = 0.9)
    
    plt.suptitle('<'+filenamelist[i]+': pre vs. post delta AF>', y=1.03)
    plt.show()

    print(df['sample'].nunique())

# %%
#** hypothesis 2-2: pre → post: change of AI per sample
dflist = makeplotdf()
filenamelist = ['WES_BRCA1', 'WES_BRCA2', 'WGS_BRCA1', 'WGS_BRCA2']
for i in range(4):
    df = dflist[i]
    df = df[df['Purpose']=='Maintenance']
    df['AI'] = df['RNA_AF'] - 0.5
    # df['AI'] = df['AI'].abs()
    condition1 = (0.4<=df['genome_AF']) & (df['genome_AF'] <= 0.6)
    df = df.loc[condition1,:]
    df = df[['sample','pre/post','AI','consequence']]

    df['variant type'] = 'non-syn'
    df.loc[df['consequence']=='synonymous_variant','variant type'] = 'syn'

    sns.set(rc = {'figure.figsize':(6,6)})
    sns.set_style('whitegrid')
    sns.catplot(x ='pre/post', y ='AI', hue = 'sample', data = df, kind='point', palette='tab10', order=['pre','post'], height = 5, aspect = 0.9)
    plt.suptitle('<'+filenamelist[i]+': pre vs. post AI>', y=1.03)
    plt.show()
    print(df['sample'].nunique())
    

#%%
#** hypothesis 2-3: pre -> post: matched loci change of AI per sample 
dflist = makeplotdf()
filenamelist = ['WES_BRCA1', 'WES_BRCA2', 'WGS_BRCA1', 'WGS_BRCA2']
for i in range(4):
    df = dflist[i]
    df = df[df['Purpose']=='Maintenance']
    condition1 = (0.4<=df['genome_AF']) & (df['genome_AF'] <= 0.6)
    df = df.loc[condition1,:]
    df['AI'] = df['RNA_AF'] - 0.5
    pre = df[df['pre/post']=='pre']
    post = df[df['pre/post']=='post']

    pre = pre[['sample','position','AI']]
    post = post[['sample','position','AI']]
    
    merged = pd.merge(pre,post,how='inner',on=['position','sample'])
    m_pre = merged[['sample','position','AI_x']]
    m_post = merged[['sample','position','AI_y']]
    m_pre.columns = ['sample','position','AI']
    m_post.columns = ['sample','position','AI']
    m_pre['pre/post'] = 'pre'
    m_post['pre/post'] ='post'
    concatdf = pd.concat([m_pre,m_post])

    sns.set(rc = {'figure.figsize':(6,6)})
    sns.set_style('whitegrid')
    sns.catplot(x ='pre/post', y ='AI', hue = 'sample',  data = concatdf, kind='point', palette='tab10', order=['pre','post'], height = 5, aspect = 0.9)
    # sns.barplot(x='sample',y='AI',data=concatdf,hue='pre/post',showfliers=False,palette='hls')
    plt.suptitle('<'+filenamelist[i]+': pre vs. post matched AI>', y=1.03)
    plt.show()


#%%
#** hypothesis 2-4: pre ~ post AI regression plot
dflist = makeplotdf()
filenamelist = ['WES_BRCA1', 'WES_BRCA2', 'WGS_BRCA1', 'WGS_BRCA2']
for i in range(4):
    df = dflist[i]
    df = df[df['Purpose']=='Maintenance']
    df['AI'] = df['RNA_AF'] - 0.5
    condition1 = (0.4<=df['genome_AF']) & (df['genome_AF'] <= 0.6)
    df = df.loc[condition1,:]
    pre = df[df['pre/post']=='pre']
    post = df[df['pre/post']=='post']

    pre = pre[['sample','position','AI']]
    post = post[['sample','position','AI']]
    
    merged = pd.merge(pre,post,how='inner',on=['position','sample'])
    merged.columns  = ['sample','position','pre AI','post AI']

    sns.set(rc = {'figure.figsize':(8,4)})
    sns.set_style('whitegrid')
    g = sns.lmplot(x="pre AI", y="post AI", hue="sample", data=merged, fit_reg=False)
    g.set(ylim=(-0.6,0.6))
    # sns.regplot(x="pre AI", y="post AI", data=merged, scatter=False, ax=g.axes[0, 0])
    plt.plot([-0.6,0.6],[-0.6,0.6], linestyle='--', color='black', linewidth=0.7)
    plt.suptitle('<'+filenamelist[i]+': pre vs. post matched AI>', y=1.03, size=12)
    plt.show()



# %%
#^^ hypothesis 3-1: heatmap: loci ~ delta AF (RNA AF - genome AF), pre vs. post
dflist = makeplotdf()
filenamelist = ['WES_BRCA1', 'WES_BRCA2', 'WGS_BRCA1', 'WGS_BRCA2']

for i in range(4):
    df = dflist[i]
    pre = df[df["pre/post"]=='pre']
    post = df[df["pre/post"]=='post']

    mean_pre = pd.DataFrame(pre.groupby('position')['deltaAF'].mean())
    mean_post = pd.DataFrame(post.groupby('position')['deltaAF'].mean())
    merged = pd.merge(mean_pre, mean_post, how='outer', left_index=True, right_index=True)
    merged.columns = ['pre','post']

    sns.set(rc = {'figure.figsize':(6,6)})
    sns.set_style('whitegrid')
    plt.suptitle('<'+filenamelist[i]+': pre/post deltaAF heatmap>', fontsize=11)
    sns.heatmap(merged)
    plt.show()
    

#%%
#^^ hypothesis 3-2: heatmap: loci ~ AI (RNA AF - 0.5), pre vs. post
dflist = makeplotdf()
filenamelist = ['WES_BRCA1', 'WES_BRCA2', 'WGS_BRCA1', 'WGS_BRCA2']

for i in range(4):
    df = dflist[i]
    df['AI'] = df['RNA_AF'] - 0.5
    pre = df[df["pre/post"]=='pre']
    post = df[df["pre/post"]=='post']

    mean_pre = pd.DataFrame(pre.groupby('position')['AI'].mean())
    mean_post = pd.DataFrame(post.groupby('position')['AI'].mean())
    merged = pd.merge(mean_pre, mean_post, how='outer', left_index=True, right_index=True)
    merged.columns = ['pre','post']

    sns.set(rc = {'figure.figsize':(6,6)})
    sns.set_style('whitegrid')
    plt.suptitle('<'+filenamelist[i]+': pre/post AI heatmap>', fontsize=11)
    sns.heatmap(merged,vmin=-0.5,vmax=0.5)
    plt.show()
#%%
#^^ hypothesis 3-3: heatmap: loci ~ post RNA AF - pre RNA AF per sample
dflist = makeplotdf()
filenamelist = ['WES_BRCA1', 'WES_BRCA2', 'WGS_BRCA1', 'WGS_BRCA2']
onlypre_WES_matched_samplelist = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/only-pre_WES_matched_samplelist', sep="\t", header=None)
onlypre_WGS_matched_samplelist = pd.read_csv('/home/jiye/jiye/copycomparison/control_PD/only-pre_WGS_matched_samplelist', sep="\t", header=None)    
onlypre_WES_matched_samplelist.columns = ['sample_ID','pre/post','wes','rna']
onlypre_WGS_matched_samplelist.columns = ['sample_ID','pre/post','wgs','rna']

WES_samplelist = pd.DataFrame()
WES_samplelist['pre'] = onlypre_WES_matched_samplelist.iloc[0:10,0]
WES_samplelist['post'] = np.array(onlypre_WES_matched_samplelist.iloc[10:20,0])


WGS_samplelist = pd.DataFrame()
WGS_samplelist['pre'] = onlypre_WGS_matched_samplelist.iloc[0:14,0]
WGS_samplelist['post'] = np.array(onlypre_WGS_matched_samplelist.iloc[14:28,0])

WES_samplelist = WES_samplelist.drop([2,3,5]) #remove salvage samples
WGS_samplelist = WGS_samplelist.drop([2,5,8,10]) #remove salvage samples

##& WES ##
for i in range(4):
    if i <2:
        df = dflist[i]
        pre = df[df['sample_ID']== WES_samplelist.iloc[0,0]]
        pre = pre[['sample','RNA_AF', 'position']]
        post = df[df['sample_ID']==WES_samplelist.iloc[0,1]]
        post = post[['sample','RNA_AF', 'position']]
        merged = pd.merge(pre,post,on='position')
        merged['post-pre'] = merged['RNA_AF_y'] - merged['RNA_AF_x']
        sample = merged[['position','post-pre']]
        name = merged['sample_x'][0]
        sample.columns = ['position',name]
        final = sample
    
        for j in range(1,7):
            df = dflist[i]
            pre = df[df['sample_ID']== WES_samplelist.iloc[j,0]]
            pre = pre[['sample','RNA_AF', 'position']]
            post = df[df['sample_ID']==WES_samplelist.iloc[j,1]]
            post = post[['sample','RNA_AF', 'position']]
            merged = pd.merge(pre,post,on='position')
            merged['post-pre'] = merged['RNA_AF_y'] - merged['RNA_AF_x']
            sample = merged[['position','post-pre']]
            if sample.shape[0] >0: 
                name = merged['sample_x'][0]
                sample.columns = ['position',name]
                final = pd.merge(final,sample, on='position', how='outer')
        
        final = final.set_index('position')
        sns.set(rc = {'figure.figsize':(6,6)})
        sns.set_style('whitegrid')
        plt.suptitle('<'+filenamelist[i]+': post RNA_AF - pre RNA_AF>', fontsize=11)
        sns.heatmap(final)
        plt.show()
    
    elif i==2:
        df = dflist[i]
        pre = df[df['sample_ID']== WGS_samplelist.iloc[0,0]]
        pre = pre[['sample','RNA_AF', 'position']]
        post = df[df['sample_ID']==WGS_samplelist.iloc[0,1]]
        post = post[['sample','RNA_AF', 'position']]
        merged = pd.merge(pre,post,on='position')
        merged['post-pre'] = merged['RNA_AF_y'] - merged['RNA_AF_x']
        sample = merged[['position','post-pre']]
        name = merged['sample_x'][0]
        sample.columns = ['position',name]
        final = sample
    
        for j in range(1,10):
            df = dflist[i]
            pre = df[df['sample_ID']== WGS_samplelist.iloc[j,0]]
            pre = pre[['sample','RNA_AF', 'position']]
            post = df[df['sample_ID']==WGS_samplelist.iloc[j,1]]
            post = post[['sample','RNA_AF', 'position']]
            merged = pd.merge(pre,post,on='position')
            merged['post-pre'] = merged['RNA_AF_y'] - merged['RNA_AF_x']
            sample = merged[['position','post-pre']]
            if sample.shape[0] >0: 
                name = merged['sample_x'][0]
                sample.columns = ['position',name]
                final = pd.merge(final,sample, on='position', how='outer')

        final = final.set_index('position')
        sns.set(rc = {'figure.figsize':(6,6)})
        sns.set_style('whitegrid')
        plt.suptitle('<'+filenamelist[i]+': post RNA_AF - pre RNA_AF>', fontsize=11)
        sns.heatmap(final)
        plt.show()

    else:
        df = dflist[i]
        pre = df[df['sample_ID']== WGS_samplelist.iloc[1,0]]
        pre = pre[['sample','RNA_AF', 'position']]
        post = df[df['sample_ID']==WGS_samplelist.iloc[1,1]]
        post = post[['sample','RNA_AF', 'position']]
        merged = pd.merge(pre,post,on='position')
        merged['post-pre'] = merged['RNA_AF_y'] - merged['RNA_AF_x']
        sample = merged[['position','post-pre']]
        name = merged['sample_x'][0]
        sample.columns = ['position',name]
        final = sample
    
        for j in range(2,10):
            df = dflist[i]
            pre = df[df['sample_ID']== WGS_samplelist.iloc[j,0]]
            pre = pre[['sample','RNA_AF', 'position']]
            post = df[df['sample_ID']==WGS_samplelist.iloc[j,1]]
            post = post[['sample','RNA_AF', 'position']]
            merged = pd.merge(pre,post,on='position')
            merged['post-pre'] = merged['RNA_AF_y'] - merged['RNA_AF_x']
            sample = merged[['position','post-pre']]
            if sample.shape[0] >0: 
                name = merged['sample_x'][0]
                sample.columns = ['position',name]
                final = pd.merge(final,sample, on='position', how='outer')

        final = final.set_index('position')
        sns.set(rc = {'figure.figsize':(6,6)})
        sns.set_style('whitegrid')
        plt.suptitle('<'+filenamelist[i]+': post RNA_AF - pre RNA_AF>', fontsize=11)
        sns.heatmap(final)
        plt.show()

# %%
