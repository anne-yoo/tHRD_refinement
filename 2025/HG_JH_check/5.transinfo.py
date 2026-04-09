#%%
import pandas as pd
from collections import defaultdict

def merge_intervals(intervals):
    """겹치는 구간을 병합"""
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = []
    for start, end in sorted_intervals:
        if not merged or merged[-1][1] < start - 1:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged

def extract_transcript_structure(gtf_path):
    exon_dict = defaultdict(list)

    with open(gtf_path, 'r') as f:
        for line in f:
            if line.startswith("#") or '\t' not in line:
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            feature = parts[2]
            start, end = int(parts[3]), int(parts[4])
            attributes = parts[8]

            if feature != 'exon':
                continue

            # attribute 파싱
            attr_dict = {}
            for attr in attributes.strip().split(';'):
                if attr.strip():
                    key, value = attr.strip().split(' ', 1)
                    attr_dict[key] = value.strip('"')

            tid = attr_dict.get('transcript_id')
            if tid:
                exon_dict[tid].append((start, end))

    # 결과 정리
    records = []
    for tid, exons in exon_dict.items():
        merged_exons = merge_intervals(exons)
        merged_length = sum(end - start + 1 for start, end in merged_exons)
        merged_length_kb = merged_length / 1000
        exon_count = len(exons)
        records.append({
            'transcript_id': tid,
            'exon_count': exon_count,
            'length_kb': merged_length_kb
        })

    return pd.DataFrame(records)


#%%
# 사용 예시
gtf_file = "/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/LR_merged_238.gtf"
output = extract_transcript_structure(gtf_file)
output.to_csv("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/LR_transcriptinfo.txt", sep="\t", index=False)


# %%
###^^ make GeTMM input ############

readcount_df = pd.read_csv("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/quantification/SR_transcript_count_matrix.csv", index_col=0)
readcount_df = readcount_df.fillna(0)
info_df = pd.read_csv("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/SR_transcriptinfo.txt", sep="\t", index_col=0)

length_kb = info_df["length_kb"]
length_kb = length_kb.reindex(readcount_df.index)

# 4. 길이 정보를 새로운 DataFrame으로 맨 앞에 추가
readcount_with_length = pd.concat([length_kb.rename("length_kb"), readcount_df], axis=1)
readcount_with_length.to_csv("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR_new/quantification/SR_GeTMM_input.txt", sep='\t', index=True)


# %%
####^^ CPAT vs. CPC2 ############
cpat_df = pd.read_csv("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/CPAT/SR/SR_cpat.ORF_prob.best.tsv", sep="\t", index_col=0)
cpc_df = pd.read_csv("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/CPC2/SR/cpc2output.txt",   sep="\t", index_col=0)

cpat_nc = cpat_df[cpat_df['Coding_prob']<0.364].index.to_list()
cpc_nc = cpc_df[cpc_df['label']=='noncoding'].index.to_list()

nc_trans = set(cpat_nc).union(set(cpc_nc))



# %%
###^^ make CPM ############
import pandas as pd

def counts_to_cpm(count_df):
    total_counts = count_df.sum(axis=0)  # 각 sample의 총 read 수
    cpm_df = count_df.divide(total_counts, axis=1) * 1e6
    return cpm_df

readcount_df = pd.read_csv("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/SR_transcript_count_matrix.csv", index_col=0)
readcount_df = readcount_df.fillna(0)

cpm_df = counts_to_cpm(readcount_df)
cpm_df.to_csv("/home/jiye/jiye/nanopore/HG_JH_check/SR_LR/quantification/SR_transcript_CPM.txt", sep='\t', index=True)

# %%
import pandas as pd
tmp = pd.read_csv('/home/sujie/sujie/nanopore/datafromhrs/2025data/prepDE/289_gene_readcount_all.txt', sep='\t', index_col=0)
# %%
