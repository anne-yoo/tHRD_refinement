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
from statannotations.Annotator import Annotator
from statannot import add_stat_annotation
rcParams['pdf.fonttype'] = 42  
rcParams['ps.fonttype'] = 42
rcParams['font.family'] = 'Helvetica'

plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 13,    # x축 틱 라벨 글꼴 크기

'ytick.labelsize': 13,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 13,
'legend.title_fontsize': 13, # 범례 글꼴 크기|
'figure.titlesize': 15    # figure 제목 글꼴 크기
})
sns.set_style("ticks")

#%%
from Bio import SeqIO
import pandas as pd

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Bio import SeqIO

# 1. 데이터 로드
fasta_path = "/home/jiye/jiye/darkproteome/data/nuORFdb/DA_nuORFdb_v1.2_dna.fasta"
anno_path = "/home/jiye/jiye/darkproteome/data/nuORFdb/nuORFdb_v1.2_annotations.csv" # csv로 변환하여 로드 권장

# FASTA 서열 로드 및 길이 계산
fasta_seqs = {record.id.split('|')[0]: str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")}
fasta_df = pd.DataFrame.from_dict(fasta_seqs, orient='index', columns=['seq']).reset_index().rename(columns={'index': 'ORF_ID_hg38'})
fasta_df['length'] = fasta_df['seq'].apply(len)
fasta_df['start_codon'] = fasta_df['seq'].apply(lambda x: x[:3])
fasta_df['stop_codon'] = fasta_df['seq'].apply(lambda x: x[-3:])

# 2. Annotation 파일과 병합
anno_df = pd.read_csv(anno_path) # Annotation 파일
df = pd.merge(fasta_df, anno_df, on='ORF_ID_hg38')

codon_counts = df['start_codon'].value_counts().reset_index()
codon_counts.columns = ['start_codon', 'count']

# 2. 정렬된 순서대로 막대 그래프를 그립니다.
plt.figure(figsize=(10, 6))
# order 인자에 정렬된 리스트를 넘겨주어 순서를 강제합니다.
sns.barplot(data=codon_counts, x='start_codon', y='count', 
            order=codon_counts['start_codon'], palette='husl')

plt.title("Start Codon Frequency (Sorted by Count)")
plt.ylabel("Frequency")
plt.xlabel("Start Codon")
plt.xticks(rotation=45) # 라벨이 겹치지 않게 회전
plt.tight_layout()
plt.show()
# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 필터링 (아까 확인한 유효한 코돈들만 선택)
valid_codons = ['CTG', 'ATG', 'GTG', 'TTG', 'AGG', 'AAG', 'ATT', 'ATC', 'ACG', 'ATA']
df_filtered = df[df['start_codon'].isin(valid_codons)]

# 2. 개수 막대 그래프 (Barplot) - 아까 확인하신 것과 동일한 기준
plt.figure(figsize=(10, 5))
codon_order = valid_codons # 순서 지정
sns.countplot(data=df_filtered, x='start_codon',order=codon_order )#order=codon_order)
plt.title("Frequency of Start Codons")
plt.ylabel("Frequency")
sns.despine()
plt.savefig("/home/jiye/jiye/darkproteome/figures/startcodon_count.pdf", dpi=300, bbox_inches='tight')
plt.show()

# 3. Start Codon 종류별 길이 분포 (Boxplot)
codon_order = df_filtered['start_codon'].value_counts().index.tolist()
plt.figure(figsize=(10, 5))
sns.boxplot(data=df_filtered, x='start_codon', y='length', order=codon_order, showfliers=False)
plt.title("ORF Length Distribution")
plt.xlabel("Start Codon")
plt.ylabel("ORF Length (bp)")
sns.despine()
plt.savefig("/home/jiye/jiye/darkproteome/figures/startcodon_ORFlength.pdf", dpi=300, bbox_inches='tight')
plt.show()

codon_order = df_filtered['start_codon'].value_counts().index.tolist()

# 2. crosstab 생성 시 columns 인자를 사용하여 순서 고정
codon_frame_ct = pd.crosstab(df_filtered['plotType'], df_filtered['start_codon'])
# 여기서 columns를 지정하여 정렬합니다ㅛ7ㅠㅎ
codon_frame_ct = codon_frame_ct[codon_order]

# 3. 히트맵 그리기
plt.figure(figsize=(13, 7))
sns.heatmap(codon_frame_ct, annot=True, fmt='d', cmap='Greens', cbar_kws={'label': 'Frequency'})
plt.title("Start Codon Distribution")
plt.xlabel("Start Codon")
plt.ylabel("ORF type")
plt.tight_layout()
plt.savefig("/home/jiye/jiye/darkproteome/figures/startcodon_count_ORFtype.pdf", dpi=300, bbox_inches='tight')
plt.show()


# %%
###^^ check stop codons #########
from Bio import SeqIO
import pandas as pd

dna_fasta = "/home/jiye/jiye/darkproteome/data/nuORFdb/DA_nuORFdb_v1.2_dna.fasta"
protein_fasta = "/home/jiye/jiye/darkproteome/data/nuORFdb/PA_nuORFdb_v1.2_protein.fasta"


def extract_key(record):
    # e.g. ENST00000488147.1_1_1:17367-18318:-
    return record.description.split("|")[0].strip()


dna_dict = {}
for rec in SeqIO.parse(dna_fasta, "fasta"):
    key = extract_key(rec)
    dna_dict[key] = {
        "dna_header": rec.description,
        "dna_seq": str(rec.seq)
    }

protein_dict = {}
for rec in SeqIO.parse(protein_fasta, "fasta"):
    key = extract_key(rec)
    protein_dict[key] = {
        "protein_header": rec.description,
        "protein_seq": str(rec.seq)
    }

all_keys = sorted(set(dna_dict) | set(protein_dict))

rows = []
for key in all_keys:
    rows.append({
        "orf_id": key,
        "dna_header": dna_dict.get(key, {}).get("dna_header"),
        "dna_seq": dna_dict.get(key, {}).get("dna_seq"),
        "protein_header": protein_dict.get(key, {}).get("protein_header"),
        "protein_seq": protein_dict.get(key, {}).get("protein_seq"),
        "in_dna": key in dna_dict,
        "in_protein": key in protein_dict,
    })

df = pd.DataFrame(rows)

# common entries first if you want to inspect matched ones
df_common = df[df["in_dna"] & df["in_protein"]].copy()
df_dna_only = df[df["in_dna"] & ~df["in_protein"]].copy()
df_protein_only = df[~df["in_dna"] & df["in_protein"]].copy()

# %%
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

flow = (
    df.groupby(['transcriptType','mergeType'])
    .size()
    .reset_index(name='count')
)

G = nx.DiGraph()

for _, r in flow.iterrows():
    G.add_edge(
        r['transcriptType'],
        r['mergeType'],
        weight=r['count']
    )

pos = nx.spring_layout(G, k=1)

plt.figure(figsize=(12,10))

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=3000,
    node_color='lightblue',
    font_size=9
)

edge_labels = {
    (u,v): f"n={d['weight']}"
    for u,v,d in G.edges(data=True)
}

nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.title("transcriptType → mergeType")
plt.show()
# %%
####^^ check my data from PRICE #######

import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# Input files
# ---------------------------
faa_path = Path("/home/jiye/jiye/darkproteome/data/HG_JH/simil_check/HGJH_clademergedPRICE_orfs_over21nt.aa.fa")
tsv_path = Path("//home/jiye/jiye/darkproteome/data/HG_JH/simil_check/HGJH_clademergedPRICE_orfs.tsv")
out_dir = Path("/home/jiye/jiye/darkproteome/data/HG_JH/simil_check/figures")
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------
# 1. Parse FASTA headers and extract ORF IDs
# Example header:
# >MSTRG.7795.19_ncRNA_0::chr1:207995955-207996003(-)
# We want: MSTRG.7795.19_ncRNA_0
# ---------------------------
orf_ids = []
with open(faa_path) as f:
    for line in f:
        if line.startswith(">"):
            header = line[1:].strip()
            orf_id = header.split("::")[0]
            orf_ids.append(orf_id)

orf_ids = list(dict.fromkeys(orf_ids))  # preserve order, remove duplicates
orf_id_set = set(orf_ids)

print(f"Number of ORFs in FASTA: {len(orf_ids):,}")

# ---------------------------
# 2. Load PRICE TSV
# ---------------------------
df = pd.read_csv(tsv_path, sep="\t")

print("Columns:", df.columns.tolist())
print(f"Number of rows in TSV: {len(df):,}")

# ---------------------------
# 3. Filter by ORF ID
# Assumes the TSV column name is exactly 'Id'
# ---------------------------
df_filtered = df[df["Id"].astype(str).isin(orf_id_set)].copy()
#df_filtered = df.copy()
print(f"Number of matched rows after filtering: {len(df_filtered):,}")

# ---------------------------
# 4. Add ORF length in nt if not already present
# Use Candidate Location if available, otherwise Location
# Supports spliced coordinates like:
# 2+:39107355-39107370|39107740-39107791
# ---------------------------
def calc_length_from_location(loc: str) -> int:
    if pd.isna(loc):
        return pd.NA
    loc = str(loc)

    # Remove leading chromosome/frame info, e.g. "2+:" or "1-:"
    if ":" in loc:
        loc = loc.split(":", 1)[1]

    total = 0
    for block in loc.split("|"):
        m = re.match(r"(\d+)-(\d+)$", block)
        if m:
            start = int(m.group(1))
            end = int(m.group(2))
            total += end - start + 1
    return total if total > 0 else pd.NA

location_col = "Candidate Location" if "Candidate Location" in df_filtered.columns else "Location"
df_filtered["length"] = df_filtered[location_col].apply(calc_length_from_location)

# ---------------------------
# 5. Plot 1: Start codon frequency
# Same style as your previous plot
# ---------------------------
sns.set_style("white")

valid_codons = df_filtered["Codon"].value_counts().index.tolist()

plt.figure(figsize=(10, 5))
sns.countplot(data=df_filtered, x="Codon", order=valid_codons, palette='Set1')
plt.ylabel("Frequency")
plt.xlabel("Start Codon")
sns.despine()
plt.savefig(out_dir / "startcodon_count_Clademerged.pdf", dpi=300, bbox_inches="tight")
plt.show()

# ---------------------------
# 6. Plot 2: ORF length distribution by start codon
# Same style as your previous plot
# ---------------------------
codon_order = df_filtered["Codon"].value_counts().index.tolist()

plt.figure(figsize=(10, 5))
sns.boxplot(
    data=df_filtered,
    x="Codon",
    y="length",
    order=codon_order,
    showfliers=False,
    palette='Set1'
)
plt.xlabel("Start Codon")
plt.ylabel("ORF Length (bp)")
sns.despine()
plt.savefig(out_dir / "startcodon_ORFlength_Clademerged.pdf", dpi=300, bbox_inches="tight")
plt.show()
# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")

# 데이터
labels = [
    "All ORFs",
    "After canonical\nfiltering",
    "Original nuORFdb",
    "Overlap with\noriginal nuORFdb"
]

values = [
    308833,
    227966,
    229251,
    29229
]

plt.figure(figsize=(5, 5))
ax = sns.barplot(x=labels, y=values,color='green')

# 값 표시
for i, v in enumerate(values):
    ax.text(i, v + 5000, f"{v:,}", ha='center', fontsize=11, color='black')

plt.ylabel("Number of ORFs")
sns.despine()

plt.tight_layout()
plt.show()
# %%
