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
import pandas as pd

# 파일 경로
tpm_file = {
    "SR": "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/SR_283_transcript_TPM.txt",
    "LR": "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/LR_283_transcript_TPM.txt"
}
count_file = {
    "SR": "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/SR_transcript_count_matrix.csv",
    "LR": "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/LR_transcript_count_matrix.csv"
}
gtf_file = {
    "SR": "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/SR_238_merged.gtf",
    "LR": "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/LR_238_merged.gtf"
}

#%%
import matplotlib.pyplot as plt

def load_transcript_gene_map(tpm_path):
    df = pd.read_csv(tpm_path, sep="\t", index_col=0)

    # Step 1: transcript ID만 남기기
    transcript_ids = df.index.str.split("-", n=1).str[0]
    gene_names = df.index.str.split("-", n=1).str[-1]

    # Step 2: gene_map: transcript_id → gene_name
    gene_map = pd.Series(gene_names.values, index=transcript_ids)

    # Step 3: TPM index도 transcript_id로
    df.index = transcript_ids

    return gene_map, df

def parse_strand_info(gtf_path):
    strand_map = {}
    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"): continue
            fields = line.strip().split("\t")
            if fields[2] != "transcript": continue
            attr = fields[8]
            tid = attr.split('transcript_id "')[1].split('"')[0]
            strand = fields[6]
            strand_map[tid] = strand
    return pd.Series(strand_map)

def apply_filters(tpm_path, count_path, gtf_path):
    # Load gene name and TPM
    gene_map, tpm = load_transcript_gene_map(tpm_path)

    # Load read count
    count_df = pd.read_csv(count_path, index_col=0)

    # Load strand info
    strand_map = parse_strand_info(gtf_path)

    all_transcripts = count_df.index
    result = []

    # Step 1: Before filtering
    current = set(all_transcripts)
    result.append(len(current))

    # Step 2: Strand filter
    strand_ok = strand_map[strand_map.isin(["+", "-"])].index
    current = current.intersection(strand_ok)
    result.append(len(current))

    # Step 3: Gene type filter (exclude Ig and TCR)
    gene_filtered = gene_map[~gene_map.str.contains("IG[HKL]|TR[ABDG]")].index
    current = current.intersection(gene_filtered)
    result.append(len(current))

    # Step 4: Read count filter (95% 샘플에서 count < 10 → 제거)
    filtered_counts = count_df.loc[list(current)]
    mask = (filtered_counts < 10).sum(axis=1)>= int(filtered_counts.shape[1] * 0.95)
    current = filtered_counts[~mask].index
    result.append(len(current))

    # Step 5: Gene annotation filter (GENCODE-known → ENST로 시작하는 애들만)
    annotated = gene_map[~gene_map.str.startswith("MSTRG")].index
    current = current.intersection(annotated)
    result.append(len(current))

    return result, current
#%%

labels = [
    "Before filtering",
    "Strand filter",
    "Gene type filter",
    "Read count filter",
    "Gene annotation filter"
]

sr_values, sr_filtered = apply_filters(tpm_file["SR"], count_file["SR"], gtf_file["SR"])
lr_values, lr_filtered = apply_filters(tpm_file["LR"], count_file["LR"], gtf_file["LR"])


#%%
plt.figure(figsize=(6, 6))
sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.plot(labels, sr_values, marker='o', label='Short-read')
plt.plot(labels, lr_values, marker='o', label='Long-read')
plt.xticks(rotation=45, ha="right")
plt.ylabel("Number of remaining transcripts")
plt.xlabel("Filtering step")
plt.grid(True, axis='y')
plt.legend()
plt.tight_layout()
sns.despine()
plt.savefig('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/figures/filtering.pdf', dpi=300, bbox_inches='tight')
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt

# 경로 설정
tmap_path = "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/gffcompare/gffcompare.SR_238_merged.gtf.tmap"
sr_tpm_path = "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/SR_283_transcript_TPM.txt"
lr_tpm_path = "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/LR_283_transcript_TPM.txt"
sr_count_path = "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/SR_transcript_count_matrix.csv"
lr_count_path = "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/LR_transcript_count_matrix.csv"

# Step 1: load tmap and filter "=" transcripts
tmap = pd.read_csv(tmap_path, sep="\t")
matched = tmap[tmap['class_code'] == '='][['qry_id', 'ref_id']]
matched.columns = ['sr_id', 'lr_id']  # for clarity

# Step 2: TPM and read count data (align index by transcript ID)
def load_tpm(path):
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.index = df.index.str.split("-", n=1).str[0]
    return df

sr_tpm = load_tpm(sr_tpm_path)
lr_tpm = load_tpm(lr_tpm_path)
sr_count = pd.read_csv(sr_count_path, index_col=0)
lr_count = pd.read_csv(lr_count_path, index_col=0)

# 최종 filtering을 통과한 transcript ID만 고려
sr_final_matched = matched[matched['sr_id'].isin(sr_filtered)].copy()
lr_final_ids = sr_final_matched.set_index('sr_id')['lr_id']

# TPM / Count에서 filtering된 transcript만 남김
sr_tpm_matched = sr_tpm.loc[sr_final_matched['sr_id']]
lr_tpm_matched = lr_tpm.loc[lr_final_ids.values]
lr_tpm_matched.index = sr_final_matched['sr_id']  # index 통일

sr_count_matched = sr_count.loc[sr_final_matched['sr_id']]
lr_count_matched = lr_count.loc[lr_final_ids.values]
lr_count_matched.index = sr_final_matched['sr_id']

print(f"Number of '=' matched transcripts in tmap: {matched.shape[0]}")
print(f"After final filtering: {sr_final_matched.shape[0]} transcripts remain")

#%%
gene_map, _ = load_transcript_gene_map(sr_tpm_path)

#%%
####^^ genelist ###########################

genelist = pd.read_csv('/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/cancergenelist.txt', sep="\t", index_col=0)
# gene_map 생성
# gene_map 로딩

genelist = genelist[(genelist['Oncogene']=='Yes') & (genelist['TSG']=='Yes')]
# gene list 기반 transcript 추출
transcripts_in_genelist = gene_map[gene_map.isin(genelist.index)].index

# 필터링 적용
sr_tpm_subset = sr_tpm_matched.loc[sr_tpm_matched.index.intersection(transcripts_in_genelist)]
lr_tpm_subset = lr_tpm_matched.loc[lr_tpm_matched.index.intersection(transcripts_in_genelist)]

sr_count_subset = sr_count_matched.loc[sr_count_matched.index.intersection(transcripts_in_genelist)]
lr_count_subset = lr_count_matched.loc[lr_count_matched.index.intersection(transcripts_in_genelist)]


#%%

from scipy.stats import pearsonr

def plot_dot(x, y, title, xlabel, ylabel, log=False):
    import numpy as np

    # 정리
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)

    # 상관계수
    r, p = pearsonr(x, y)

    # Plot
    plt.figure(figsize=(6,6))
    if log:
        x = np.log10(x + 1)
        y = np.log10(y + 1)
        xlabel += " (log10)"
        ylabel += " (log10)"

    sns.regplot(x=x, y=y, scatter_kws={'alpha': 0.4, 's': 15}, line_kws={"color": "red", "linestyle": "--"})

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title}\nPearson r = {r:.3f}, p = {p:.1e}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ③ TPM per transcript
plot_dot(
    sr_tpm_subset.mean(axis=1),
    lr_tpm_subset.mean(axis=1),
    title="Mean TPM per transcript (SR vs LR)",
    xlabel="Short-read mean TPM",
    ylabel="Long-read mean TPM",
    log=True
)

# ④ Count per transcript
plot_dot(
    sr_count_subset.mean(axis=1),
    lr_count_subset.mean(axis=1),
    title="Mean read count per transcript (SR vs LR)",
    xlabel="Short-read mean read count",
    ylabel="Long-read mean read count",
    log=True
)

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

def plot_density(x, y, title, xlabel, ylabel, log=False):
    import pandas as pd
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    
    r, p = pearsonr(x, y)
    if log:
        x = np.log10(x + 1)
        y = np.log10(y + 1)
        xlabel += " (log10)"
        ylabel += " (log10)"

    df = pd.DataFrame({'x': x, 'y': y})

    plt.figure(figsize=(6,6))
    sns.kdeplot(data=df, x='x', y='y', fill=True, cmap='Reds', thresh=0.05, alpha=0.6)
    #sns.scatterplot(data=df, x='x', y='y', s=10, alpha=0.3, edgecolor=None)
    sns.regplot(x=x, y=y, scatter=False, color='blue', line_kws={"linestyle": "--"}, linewidth=1)

    plt.title(f"{title}\nPearson r = {r:.3f}, p = {p:.1e}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
plot_density(
    sr_tpm_matched.mean(axis=1),
    lr_tpm_matched.mean(axis=1),
    title="Mean TPM per transcript (SR vs LR)",
    xlabel="Short-read mean TPM",
    ylabel="Long-read mean TPM",
    log=True
)

# ④ Count per transcript
plot_density(
    sr_count_matched.mean(axis=1),
    lr_count_matched.mean(axis=1),
    title="Mean read count per transcript (SR vs LR)",
    xlabel="Short-read mean read count",
    ylabel="Long-read mean read count",
    log=True
)

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_2d_density(x, y, title="", xlabel="", ylabel="", log=True, bins=100):
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)

    if log:
        x = np.log10(x + 1)
        y = np.log10(y + 1)
        xlabel += " (log10)"
        ylabel += " (log10)"

    df = pd.DataFrame({'x': x, 'y': y})

    plt.figure(figsize=(6,6))
    sns.kdeplot(data=df, x='x', y='y', fill=True, cmap="viridis", thresh=0.01)
    plt.plot([x.min(), x.max()], [x.min(), x.max()], 'w--', linewidth=1.5)  # y = x
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()

plot_2d_density(
    sr_tpm_matched.mean(axis=1),
    lr_tpm_matched.mean(axis=1),
    title="Mean TPM per transcript (SR vs LR)",
    xlabel="Short-read mean TPM",
    ylabel="Long-read mean TPM",
    log=True
)

# ④ Count per transcript
plot_2d_density(
    sr_count_matched.mean(axis=1),
    lr_count_matched.mean(axis=1),
    title="Mean read count per transcript (SR vs LR)",
    xlabel="Short-read mean read count",
    ylabel="Long-read mean read count",
    log=True
)
# %%
