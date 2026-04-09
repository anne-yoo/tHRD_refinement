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
'legend.fontsize': 13,
'legend.title_fontsize': 13, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})
sns.set_style("ticks")
# %%
import pandas as pd
import re

# Input paths
major_file = "/home/jiye/jiye/copycomparison/GENCODEquant/gencode_majorminorlist.txt"
gtf_file   = "/home/jiye/jiye/copycomparison/GENCODEquant/gencode.v19.annotation.gtf"

# ---------------------------
# 1. Load your major/minor list
# ---------------------------
df = pd.read_csv(major_file, sep="\t", dtype=str)

# transcriptid column must exist
df['transcriptid'] = df['transcriptid'].str.strip()

# ---------------------------
# 2. Parse GTF to extract gene_type / transcript_type
# ---------------------------
gtf_records = []

with open(gtf_file, "r") as f:
    for line in f:
        if line.startswith("#"):
            continue
        fields = line.strip().split("\t")
        
        # We only care about transcript entries
        if fields[2] != "transcript":
            continue
        
        attr_field = fields[8]

        # helper: extract key "value"
        def extract(key):
            m = re.search(r'%s "([^"]+)"' % key, attr_field)
            return m.group(1) if m else None

        gtf_records.append({
            "transcriptid": extract("transcript_id"),
            "gene_type": extract("gene_type"),
            "transcript_type": extract("transcript_type"),
        })

gtf_df = pd.DataFrame(gtf_records)
gtf_df = gtf_df.dropna(subset=["transcriptid"])

# ---------------------------
# 3. Merge to attach types
# ---------------------------
merged = df.merge(gtf_df, on="transcriptid", how="left")

# ---------------------------
# 4. Save result
# ---------------------------
# out_path = "/home/jiye/jiye/copycomparison/GENCODEquant/gencode_majorminorlist_with_types.txt"
# merged.to_csv(out_path, sep="\t", index=False)

# print("Done! Saved to:", out_path)

# %%
trans = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost_80_transcript_TPM.txt', sep='\t', index_col=0)
gene = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost_80_gene_TPM.txt', sep='\t', index_col=0)
clin = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost_80_clinicalinfo.txt', sep='\t', index_col=1)

DDRlist = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/DDR_genelist_whole.txt', sep='\t')
DDRcorelist = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/DDR_genelist_core.txt', sep='\t')

from collections import OrderedDict

rename_map_1 = {'Nucleotide Excision Repair (NER) - includes TC-NER and GC-NER': 'Nucleotide Excision Repair (NER)'}
rename_map_2 = {'Nucleotide Excision Repair (NER, including TC-NER and GC-NER))': 'Nucleotide Excision Repair (NER)', 
                'Homologous Recomination (HR)': 'Homologous Recombination (HR)',}

ddr_gene = {}

for col in DDRlist.columns:
    # NaN 제거하고 리스트로 변환
    genes = DDRlist[col].dropna().tolist()
    ddr_gene[col] = genes

ddr_genelist = OrderedDict()
for k, v in ddr_gene.items(): 
    new_k = rename_map_1.get(k, k)
    ddr_genelist[new_k] = v

ddr_coregene = {}

for col in DDRcorelist.columns:
    # NaN 제거하고 리스트로 변환
    genes = DDRcorelist[col].dropna().tolist()
    ddr_coregene[col] = genes

ddr_coregenelist = OrderedDict()
for k, v in ddr_coregene.items():
    new_k = rename_map_2.get(k, k)
    ddr_coregenelist[new_k] = v
# %%
#####^ transcript type 별 ################

conditions = [ ("AR", 1), ("IR", 0) ] 
times = ["pre", "post"]
stack_order = [ "non_stop_decay", "nonsense_mediated_decay", "retained_intron", "processed_transcript", "protein_coding", ] 
order = stack_order
color_map = { "protein_coding": "#d62728", "processed_transcript": "#2ca02c", "retained_intron": "#9467bd", "nonsense_mediated_decay": "#ff7f0e", "non_stop_decay": "#1f77b4" }

def summarize_expression(expr_df, group_samples, categories):
    expr = expr_df[group_samples] 
    means = expr.groupby(categories).mean().mean(axis=1) 
    sds = expr.groupby(categories).mean().std(axis=1) 
    return means, sds

def run_analysis(filter_ratio=0.3):

    print(f"=== Running analysis with filter_ratio={filter_ratio} ===")

    # ========================================
    # 1) meta index 정렬
    # ========================================
    meta = merged.copy().set_index("Transcript-Gene")

    # ========================================
    # 2) aligned trans only
    # ========================================
    common = meta.index.intersection(trans.index)
    meta = meta.loc[common]
    trans_filt = trans.loc[common]

    # ========================================
    # 3) Ensure sample ordering
    # ========================================
    trans_filt = trans_filt.loc[:, clin.index]

    # ========================================
    # 4) transcript filtering by ratio
    # ========================================
    nonzero_fraction = (trans_filt > 0).sum(axis=1) / trans_filt.shape[1]
    keep_idx = nonzero_fraction >= filter_ratio

    print(f"Filtering kept {keep_idx.sum()} / {len(keep_idx)} transcripts")

    trans_filt = trans_filt.loc[keep_idx]
    meta = meta.loc[keep_idx]

    # ========================================
    # 5) TU 계산
    # ========================================
    gene_of_trans = meta["genename"]
    gene_sum = trans_filt.groupby(gene_of_trans).transform("sum")
    gene_sum = gene_sum.replace(0, np.nan)
    TU = trans_filt / gene_sum

    # ========================================
    # 6) protein-coding subset
    # ========================================
    ######***##############################
    pc_idx = (meta["gene_type"] == "protein_coding") & (meta['type']=='minor')
    ###############**######################

    pc_trans = trans_filt[pc_idx]
    pc_TU = TU[pc_idx]
    pc_types = meta.loc[pc_idx, "transcript_type"]

    # ========================================
    # 7) plotting function (내부에서 사용)
    # ========================================
    def plot_stacked(expr_df, category_series, ylabel, title, prefix):

        fig, ax = plt.subplots(figsize=(10, 6))

        bar_index = 0
        bar_positions = []

        for resp_label, resp_val in conditions:
            for time in times:

                group_samples = clin[
                    (clin["response"] == resp_val) &
                    (clin["treatment"] == time)
                ].index.tolist()

                means, sds = summarize_expression(expr_df, group_samples, category_series)

                means = means.reindex(stack_order).fillna(0)
                sds   = sds.reindex(stack_order).fillna(0)

                bottom = 0
                for cat in reversed(stack_order):
                    m = means[cat]
                    sd = sds[cat]

                    ax.bar(
                        bar_index, m, bottom=bottom,
                        color=color_map[cat], edgecolor="black",
                        label=cat if bar_index == 0 else None
                    )

                    ax.errorbar(
                        bar_index, bottom + m/2, yerr=sd/2,
                        color="black", capsize=3, linewidth=1
                    )

                    bottom += m

                bar_positions.append(f"{resp_label}-{time}")
                bar_index += 1

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_xticks(range(len(bar_positions)))
        ax.set_xticklabels(bar_positions)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        handles, labels = ax.get_legend_handles_labels()
        label_to_handle = dict(zip(labels, handles))
        ordered_handles = [label_to_handle[l] for l in stack_order]
        ordered_labels = stack_order

        ax.legend(
            ordered_handles, ordered_labels,
            bbox_to_anchor=(1.02, 1), loc="upper left"
        )

        plt.tight_layout()

        out_path = f"/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/figures/minor_transtype_{prefix}_filtered{filter_ratio}.pdf"
        plt.savefig(out_path)
        plt.show()
        print("Saved:", out_path)

    # ========================================
    # 8) Run plots
    # ========================================
    plot_stacked(
        pc_trans, pc_types,
        ylabel="Mean expression",
        title=f"Minor transcript expression - protein coding genes (ratio={filter_ratio})",
        prefix="exp"
    )

    plot_stacked(
        pc_TU, pc_types,
        ylabel="Mean TU",
        title=f"Minor transcript usage (TU) - protein coding genes (ratio={filter_ratio})",
        prefix="TU"
    )

    print("=== Done ===")


# ========================
# Example usage:
# ========================
run_analysis(filter_ratio=0.30)
# run_analysis(filter_ratio=0.10)
# run_analysis(filter_ratio=0.50)


#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal

# =========================================
# Global config
# =========================================

conditions = [("AR", 1), ("IR", 0)]
times = ["pre", "post"]

# 위에서부터: protein_coding, processed_transcript, retained_intron, NMD, non_stop_decay
stack_order = [
    "non_stop_decay",
    "nonsense_mediated_decay",
    "retained_intron",
    "processed_transcript",
    "protein_coding",
]
order = stack_order

color_map = {
    "protein_coding": "#d62728",
    "processed_transcript": "#2ca02c",
    "retained_intron": "#9467bd",
    "nonsense_mediated_decay": "#ff7f0e",
    "non_stop_decay": "#1f77b4",
}

out_dir = "/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost/figures"


# =========================================
# Helper: summarize expression/TU
# =========================================

def summarize_expression(expr_df, group_samples, categories):
    """
    expr_df: transcript x samples
    group_samples: list of sample IDs
    categories: transcript_type Series (index aligned with expr_df.index)
    """
    expr = expr_df[group_samples]
    grouped = expr.groupby(categories)
    means = grouped.mean().mean(axis=1)
    sds = grouped.mean().std(axis=1)
    return means, sds


# =========================================
# Helper: subset transcripts by gene list
# =========================================

def subset_by_gene(expr_df, gene_list):
    """
    expr_df index = 'Transcript-Gene' (예: ENST...-BRCA1)
    gene_list = ['BRCA1', 'RAD51', ...]
    """
    gene_part = expr_df.index.str.split("-").str[-1]
    idx = gene_part.isin(gene_list)
    return expr_df.loc[idx]


# =========================================
# Helper: DDR pathway test (Kruskal-Wallis)
# =========================================

def ddr_pathway_test(expr_df, clin, gene_list):
    """
    expr_df: transcript x samples
    clin: sample metadata
    gene_list: pathway에 속한 gene 리스트
    """
    subset = subset_by_gene(expr_df, gene_list)
    if subset.shape[0] == 0:
        return None

    # transcript → gene 평균으로 collapse
    gene_name = subset.index.str.split("-").str[-1]
    collapsed = subset.groupby(gene_name).mean()

    groups = []
    for resp_label, resp_val in conditions:
        for t in times:
            group_samples = clin[
                (clin["response"] == resp_val) &
                (clin["treatment"] == t)
            ].index.tolist()
            if len(group_samples) == 0:
                groups.append(np.array([]))
            else:
                arr = collapsed[group_samples].values.flatten()
                groups.append(arr)

    # 빈 그룹만 있으면 테스트 불가
    if all(len(g) == 0 for g in groups):
        return None

    try:
        
        groups_nonempty = [g for g in groups if len(g) > 0]
        if len(groups_nonempty) < 2:
            return None
        _, p = kruskal(*groups_nonempty)
        return p
    except Exception:
        return None


# =========================================
# Helper: stacked bar plot (공용)
# =========================================

def plot_stacked(expr_df, category_series, clin, ylabel, title, out_path):
    """
    expr_df: transcript x samples (expression or TU)
    category_series: transcript_type Series (index aligned with expr_df.index)
    clin: clinical dataframe
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_index = 0
    bar_positions = []

    for resp_label, resp_val in conditions:
        for time in times:
            group_samples = clin[
                (clin["response"] == resp_val) &
                (clin["treatment"] == time)
            ].index.tolist()

            if len(group_samples) == 0:
                bar_positions.append(f"{resp_label}-{time}")
                bar_index += 1
                continue

            means, sds = summarize_expression(expr_df, group_samples, category_series)

            means = means.reindex(stack_order).fillna(0)
            sds = sds.reindex(stack_order).fillna(0)

            bottom = 0.0
            for cat in reversed(stack_order):
                m = means[cat]
                sd = sds[cat]

                ax.bar(
                    bar_index,
                    m,
                    bottom=bottom,
                    color=color_map[cat],
                    edgecolor="black",
                    label=cat if bar_index == 0 else None,
                )

                ax.errorbar(
                    bar_index,
                    bottom + m / 2.0,
                    yerr=sd / 2.0,
                    color="black",
                    capsize=3,
                    linewidth=1,
                )

                bottom += m

            bar_positions.append(f"{resp_label}-{time}")
            bar_index += 1

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks(range(len(bar_positions)))
    ax.set_xticklabels(bar_positions)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ordered_handles = [label_to_handle[l] for l in order if l in label_to_handle]
    ordered_labels = [l for l in order if l in label_to_handle]

    ax.legend(
        ordered_handles,
        ordered_labels,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.show()
    print("Saved:", out_path)


# =========================================
# DDR pathway loop + 조건부 plotting
# =========================================

def analyze_ddr_and_plot(pc_trans, pc_TU, pc_types, clin, ddr_dict, out_dir, p_cut=0.05):
    significant_terms = []

    for term, genes in ddr_dict.items():
        print(f"\n[DDR] Testing pathway: {term}")

        # expression
        p_exp = ddr_pathway_test(pc_trans, clin, genes)
        print("  Expression p =", p_exp)

        # TU
        p_tu = ddr_pathway_test(pc_TU, clin, genes)
        print("  TU p         =", p_tu)

        # expression plot
        if p_exp is not None and p_exp < p_cut:
            print(f"  → Significant EXP (p={p_exp:.3e}) → Plotting...")
            expr_subset = subset_by_gene(pc_trans, genes)
            types_subset = pc_types.loc[expr_subset.index]

            out_path = os.path.join(
                out_dir,
                f"major_DDR_{term.replace(' ', '_')}_EXP_p{p_exp:.2e}.pdf",
            )
            plot_stacked(
                expr_subset,
                types_subset,
                clin,
                ylabel="Mean expression",
                title=f"{term} (Expression)  p={p_exp:.3e}",
                out_path=out_path,
            )
            significant_terms.append((term, "expression", p_exp))

        # TU plot
        if p_tu is not None and p_tu < p_cut:
            print(f"  → Significant TU (p={p_tu:.3e}) → Plotting...")
            tu_subset = subset_by_gene(pc_TU, genes)
            types_subset = pc_types.loc[tu_subset.index]

            out_path = os.path.join(
                out_dir,
                f"DDR_{term.replace(' ', '_')}_TU_p{p_tu:.2e}.pdf",
            )
            plot_stacked(
                tu_subset,
                types_subset,
                clin,
                ylabel="Mean TU",
                title=f"{term} (TU)  p={p_tu:.3e}",
                out_path=out_path,
            )
            significant_terms.append((term, "TU", p_tu))

    print("\n=== Significant DDR pathways ===")
    for term, mode, p in significant_terms:
        print(f"{term:40s} {mode:10s} p={p:.3e}")

    return significant_terms


# =========================================
# Main: minor-only pipeline + DDR 분석
# =========================================

def run_analysis(filter_ratio=0.3, p_cut=0.05):
    print(f"=== Running analysis with filter_ratio={filter_ratio} ===")

    # 1) meta index = Transcript-Gene
    meta = merged.copy().set_index("Transcript-Gene")

    # 2) align trans to meta index
    common = meta.index.intersection(trans.index)
    meta = meta.loc[common]
    trans_filt = trans.loc[common]

    # 3) ensure sample ordering
    trans_filt = trans_filt.loc[:, clin.index]

    # 4) transcript filtering by ratio
    nonzero_fraction = (trans_filt > 0).sum(axis=1) / trans_filt.shape[1]
    keep_idx = nonzero_fraction >= filter_ratio

    print(f"Filtering kept {keep_idx.sum()} / {len(keep_idx)} transcripts")

    trans_filt = trans_filt.loc[keep_idx]
    meta = meta.loc[keep_idx]

    # 5) TU 계산
    gene_of_trans = meta["genename"]
    gene_sum = trans_filt.groupby(gene_of_trans).transform("sum")
    gene_sum = gene_sum.replace(0, np.nan)
    TU = trans_filt / gene_sum
    TU = TU.dropna(how='all')
    meta = meta.loc[TU.index]
    trans_filt = trans_filt.loc[TU.index]

    # 6) protein-coding & minor subset
    pc_idx = (meta["gene_type"] == "protein_coding") & (meta["type"] == "major")

    pc_trans = trans_filt.loc[pc_idx]
    pc_TU = TU.loc[pc_idx]
    pc_types = meta.loc[pc_idx, "transcript_type"]

    print(f"  - Protein-coding major transcripts: {pc_trans.shape[0]}")

    # 7) whole (minor) plots
    os.makedirs(out_dir, exist_ok=True)

    exp_path = os.path.join(
        out_dir,
        f"major_transtype_exp_filtered{filter_ratio}.pdf",
    )
    plot_stacked(
        pc_trans,
        pc_types,
        clin,
        ylabel="Mean expression",
        title=f"Major transcript expression - protein coding genes (ratio={filter_ratio})",
        out_path=exp_path,
    )

    tu_path = os.path.join(
        out_dir,
        f"major_transtype_TU_filtered{filter_ratio}.pdf",
    )
    plot_stacked(
        pc_TU,
        pc_types,
        clin,
        ylabel="Mean TU",
        title=f"Major transcript usage (TU) - protein coding genes (ratio={filter_ratio})",
        out_path=tu_path,
    )

    # 8) DDR pathway별 유의성 테스트 + 유의한 것만 plot
    ddr_out_dir = os.path.join(out_dir, f"DDR_filtered{filter_ratio}")
    os.makedirs(ddr_out_dir, exist_ok=True)

    signif = analyze_ddr_and_plot(
        pc_trans=pc_trans,
        pc_TU=pc_TU,
        pc_types=pc_types,
        clin=clin,
        ddr_dict=ddr_coregene,  # <- 너가 이미 가진 dict
        out_dir=ddr_out_dir,
        p_cut=p_cut,
    )

    print("=== Done ===")
    return signif


# =========================================
# 실제 실행
# =========================================
run_analysis(filter_ratio=0.30, p_cut=0.05)

# %%
