#%%
import pandas as pd
import numpy as np
#%%
trans = pd.read_csv('/home/jiye/jiye/nanopore/FINALDATA/matched_transcript_TPM.txt', sep='\t', index_col=0)
trans = trans[trans.apply(lambda x: (x != 0).sum(), axis=1) >= trans.shape[1]*0.2]
translist = trans.index.to_list()
det = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/DESeq2/deseq2_det.txt', sep='\t', index_col=0)
det = det.loc[det.index.isin(translist),:]
det = det.loc[(np.abs(det['log2FoldChange'])>1.5) & (det['padj']<0.05),:]
det['genename'] = det.index.str.split("-",1).str[1]
det['trans'] = det.index.str.split("-",1).str[0]
gdetlist = set(det['genename'])

t = det.loc[det['log2FoldChange']>1.5,:]
n = det.loc[det['log2FoldChange']<-1.5,:]
#%%
pfam_data = pd.read_csv("/home/jiye/jiye/nanopore/202411_analysis/PfamScan/tab_pfam_result.txt", sep="\t",)

# Filter significant matches (e.g., E-value < 1e-5)
significant_domains = pfam_data
#print(significant_domains)

#%%
tumor_transcripts = t['trans'].to_list()
normal_transcripts = n['trans'].to_list()

domains_per_transcript = significant_domains.groupby("seq id")["hmm name"].apply(set)

# Compare tumor vs normal transcripts
t_list = set(significant_domains['seq id']).intersection(set(tumor_transcripts))
n_list = set(significant_domains['seq id']).intersection(set(normal_transcripts))

tumor_domains = set(pfam_data[pfam_data["seq id"].isin(t_list)]["hmm name"])

# Normal 그룹의 전체 domain 집합
normal_domains = set(pfam_data[pfam_data["seq id"].isin(n_list)]["hmm name"])

# Lost (Normal에서만 있는 domain)
lost_domains = normal_domains - tumor_domains

# Gained (Tumor에서만 있는 domain)
gained_domains = tumor_domains - normal_domains


print("Lost Domains in Tumor:", lost_domains)
print("Gained Domains in Tumor:", gained_domains)

# %%
pfam_data_with_clans = pfam_data[pfam_data["hmm name"].isin(lost_domains | gained_domains)]
domains_by_clan = pfam_data_with_clans.groupby("clan")["hmm name"].apply(set)

print("Domains Grouped by Clan:\n", domains_by_clan)
# %%
# Frequency of lost domains
lost_domain_counts = pfam_data[pfam_data["hmm name"].isin(lost_domains)]["hmm name"].value_counts()

# Frequency of gained domains
gained_domain_counts = pfam_data[pfam_data["hmm name"].isin(gained_domains)]["hmm name"].value_counts()

# Display top domains
print("downregulated DETs:\n", lost_domain_counts.head(15))
print("upregulated DETs:\n", gained_domain_counts.head(15))



# %%
import matplotlib.pyplot as plt

lost_domain_counts = {
    "DEAD": 411,
    "SCAN": 394,
    "Glyco_transf_29": 276,
    "zf-C4": 242,
    "NPIP": 211,
    "PID": 193,
    "Adaptin_N": 178,
    "Ndr": 174,
    "Sema": 158,
    "Sec7": 156,
    "DHHC": 148,
    "Proteasome": 146,
    "C2-set_3": 144,
    "WW": 141,
    "Laminin_G_2": 140,
}

gained_domain_counts = {
    "7tm_4": 1002,
    "SNF2-rel_dom": 358,
    "Pro_isomerase": 264,
    "RVT_1": 256,
    "SET": 211,
    "ubiquitin": 211,
    "Aldedh": 202,
    "DnaJ": 201,
    "zf-H2C2_2": 194,
    "Hist_deacetyl": 187,
    "zf-RING_2": 180,
    "Cpn60_TCP1": 159,
    "TPR_12": 147,
    "Helicase_C": 146,
    "Sec1": 145,
}


import seaborn as sns
import matplotlib

sns.set(font = 'Arial')
sns.set_style("ticks")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 16})  # Increase this value to make fonts larger

# Plot Lost Domains
plt.figure(figsize=(6, 6))
plt.bar(lost_domain_counts.keys(), lost_domain_counts.values(), color="#3E6D9C")
plt.title("Top 15 unique domains in tumor-downregulated DETs", fontsize=13)
plt.xlabel("Domain")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right",fontsize=12)
plt.tight_layout()
#plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/figures/lostdomains_barplot.pdf', dpi=300, bbox_inches='tight')
sns.despine()
plt.show()

# Plot Gained Domains
plt.figure(figsize=(6, 6))
plt.bar(gained_domain_counts.keys(), gained_domain_counts.values(), color="#DB3951")
plt.title("Top 15 unique domains in tumor-upregulated DETs", fontsize=13)
plt.xlabel("Domain")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right",fontsize=12)
plt.tight_layout()
#plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/figures/gaineddomains_barplot.pdf', dpi=300, bbox_inches='tight')
sns.despine()
plt.show()
# %%
