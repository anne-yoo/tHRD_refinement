#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from scipy.stats import pearsonr

#%%
#%%
TUdata = pd.read_csv('/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/merged_TU/whole_TU/whole_TU.txt', sep='\t', index_col=0)


#%%
filelist = ['stable_mt','stable_resint']
file = filelist[1]

tmp = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/satuRn_results/'+file+'_DTUresult.csv')

# GOresult = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/satuRn_results/'+file+'_GOenrichment.csv')

repair = pd.read_csv("/home/hyeongu/DATA5/hyeongu/GSEA/gmt/merged_GO_NER.txt", sep="\t", header=None)
vegf = pd.read_csv("/home/hyeongu/DATA5/hyeongu/GSEA/gmt/vegf_repair-related_fgf.txt", sep="\t", header=None)

repair.columns=['gene']
vegf.columns=['gene']
geneterms = pd.merge(repair, vegf, on='gene', how='outer')

gene_tu = pd.merge(tmp,geneterms,on='gene')
# pval_gene_tu = gene_tu.sort_values(by='pval')
# pval_gene_tu = pval_gene_tu.iloc[:50,:]
pval_gene_tu = gene_tu[gene_tu['pval']<0.05]
features = pval_gene_tu[['transcript']]
features = features.set_index(['transcript'])

#** input file processing

# TPM = pd.read_csv('//home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/pre_post_TU/final_data/new_final_pre_post_samples_TU_input.txt', sep='\t', index_col=0)
# TUdata = TPM[TPM['target_gene']!= '-']
# TUdata = TUdata.drop(['target_gene'],axis=1)
# TUdata = rawdata[rawdata['target_gene']!= '-']
# TUdata = TUdata.drop(['target_gene'],axis=1)

#%%%
tu = TUdata
tu.columns = tu.columns.str.replace("SMC_OV_OVLB_","")
tu.columns = tu.columns.str.replace("SV_OV_HRD_","").str[:10]
tu.columns = tu.columns.str.replace("T","P")

sample = tu.columns.tolist()# %%

#%%
samples = sample.copy()
for i in range(int(len(samples)/2)):
    i = i*2
    if "atD" in samples[i]:
        samples[i+1] = samples[i+1][:10]+"-bfD"
    elif "bfD" in samples[i]:
        samples[i+1] = samples[i+1][:10]+"-atD"
    elif "atD" in samples[i+1]:
        samples[i] = samples[i][:10]+"-bfD"
    elif "bfD" in samples[i+1]:
        samples[i] = samples[i][:10]+"-atD"

TUdata.columns = samples
try:
    TUdata = TUdata.drop(["SV-OV-P080-atD","SV-OV-P250-atD","SV-OV-P055-atD",\
            "SV-OV-P143-atD","SV-OV-P137-atD","SV-OV-P134-atD",\
            "SV-OV-P174-bfD","SV-OV-P164-atD"], axis=1)
except:
    

    pass



TUdata = TUdata[(TUdata>0).astype(int).sum(axis=1) > 30]

#** final input
input = pd.merge(TUdata,features, how='inner', left_index=True, right_index=True)
input = input.T

#%%
input.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/XGboost/repairvegf_whole_'+file+'.csv')










# %%
filelist = ['stable_mt_DTUresult.csv','stable_resint_DTUresult.csv']
for file in filelist:
    #** selected features (p<0.05)
    tmp = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/satuRn_results/'+file)
    filtered = tmp[tmp['pval']<0.05]
    features = filtered[['transcript']]
    features = features.set_index(['transcript'])

    #** input file processing
    TUdata = pd.read_csv('//home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/pre_post_TU/final_data/whole_TU.txt', sep='\t', index_col=0)
    # TUdata = rawdata[rawdata['target_gene']!= '-']
    # TUdata = TUdata.drop(['target_gene'],axis=1)

    sample = TUdata.columns.tolist()# %%
    samples = sample.copy()
    for i in range(int(len(samples)/2)):
        i = i*2
        if "atD" in samples[i]:
            samples[i+1] = samples[i+1][:10]+"-bfD"
        elif "bfD" in samples[i]:
            samples[i+1] = samples[i+1][:10]+"-atD"
        elif "atD" in samples[i+1]:
            samples[i] = samples[i][:10]+"-bfD"
        elif "bfD" in samples[i+1]:
            samples[i] = samples[i][:10]+"-atD"

    TUdata.columns = samples
    try:
        TUdata = TUdata.drop(["SV-OV-P080-atD","SV-OV-P250-atD","SV-OV-P055-atD",\
                "SV-OV-P143-atD","SV-OV-P137-atD","SV-OV-P134-atD",\
                "SV-OV-P174-bfD","SV-OV-P164-atD"], axis=1)
    except:
        

        pass

    
    
    TUdata = TUdata[(TUdata>0).astype(int).sum(axis=1) > 30]

    #** final input
    input = pd.merge(TUdata,features, how='inner', left_index=True, right_index=True)
    input = input.T
    # input.to_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/XGboost/input_'+file)

# %%
ff = pd.read_csv('/home/jiye/jiye/copycomparison/OC_transcriptome/model_input/input_stable_mt_DTUresult.csv', index_col=0)
# %%
