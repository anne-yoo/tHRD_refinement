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
import os
import matplotlib
import gseapy as gp
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats
from statsmodels.stats.multitest import multipletests
from matplotlib import rcParams
from statannotations.Annotator import Annotator
from statannot import add_stat_annotation
from matplotlib_venn import venn2
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

# %%
####^^ (1-1) group 1 증가 GO enrichment######
sqanti = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/sqantioutput/sqanti_hg19_classification.txt', sep='\t')
sqanti.dropna(axis=1, how='all', inplace=True)

AR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/whole_AR_stable_DUT_Wilcoxon_delta_withna.txt', sep='\t', index_col=0)
IR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/whole_IR_stable_DUT_Wilcoxon_delta_withna.txt', sep='\t', index_col=0)
baseline_dut = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/whole_baseline_ARpre_vs_IRpre_stable_DUT_MannWhitney_delta_withna.txt', sep='\t', index_col=0)
ARdeg = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/whole_AR_Wilcoxon_DEGresult_FC.txt', sep='\t')
IRdeg = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/whole_IR_Wilcoxon_DEGresult_FC.txt', sep='\t')
# AR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/maintenance/AR_stable_DUT_Wilcoxon_delta_withna.txt', sep='\t', index_col=0) #^Only maintenance
# IR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/maintenance/IR_stable_DUT_Wilcoxon_delta_withna.txt', sep='\t', index_col=0) #^Only maintenance

ARdutlist = AR_dut.loc[(AR_dut['p_value']<0.05) & (np.abs(AR_dut['delta_TU'])>0.05)].index.to_list()
IRdutlist = IR_dut.loc[(IR_dut['p_value']<0.05) & (np.abs(IR_dut['delta_TU'])>0.05)].index.to_list()
baseline_dutlist = baseline_dut.loc[(baseline_dut['p_value']<0.05) & (np.abs(baseline_dut['delta_TU'])>0.05)].index.to_list()

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost_80_clinicalinfo.txt', sep='\t', index_col=0)
sampleinfo_full = sampleinfo.copy()
#sampleinfo = sampleinfo[sampleinfo['purpose']=='maintenance'] #^ Only maintenance

ARlist = list(set(sampleinfo.loc[(sampleinfo['response']==1),'sample_full']))
IRlist = list(set(sampleinfo.loc[(sampleinfo['response']==0),'sample_full']))

transexp = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_80_transcript_TPM.txt',sep='\t', index_col=0)
transexp = transexp.iloc[:,:-1]
transexp = transexp.loc[(transexp > 0).sum(axis=1) >= 8] #20% 이상에서는 나오긴 해야됨 ...

#transexp = transexp.loc[(transexp > 0).sum(axis=1) >= transexp.shape[1]*0.3]
transexp["gene"] = transexp.index.str.split("-", n=1).str[-1]
gene_sum = transexp.groupby("gene").transform("sum")
filtered_trans = transexp.iloc[:, :-1].div(gene_sum)
TU = filtered_trans.copy()

mainlist =(list(set(sampleinfo[sampleinfo['purpose']=='maintenance']['sample_full'])))
sallist = (list(set(sampleinfo_full[sampleinfo_full['purpose']=='salvage']['sample_full'])))

main_TU = TU[mainlist]
sal_TU = TU[sallist]

main_TU = main_TU.sort_index(axis=1)
main_TU.columns = main_TU.columns.str[:-4]

sal_TU = sal_TU.sort_index(axis=1)
sal_TU.columns = sal_TU.columns.str[:-4]

majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_80_majorminorlist.txt',sep='\t')
majorlist = majorminor[majorminor['type']=='major']['Transcript-Gene'].to_list()
minorlist = majorminor[majorminor['type']=='minor']['Transcript-Gene'].to_list()

sampleinfo = sampleinfo.iloc[::2,:]

proteincoding = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/80_discovery_gene_exp_TPM_symbol.txt', sep='\t', index_col=0)
proteincodinglist = proteincoding['Gene Symbol'].to_list()

TU = filtered_trans.copy()
#TU = TU.loc[:,TU.columns.isin(mainlist)] #^Only maintenance
TU = TU[TU.index.str.split("-", n=1).str[-1].isin(proteincodinglist)]
TU.columns = TU.columns.str[:-4] 

# 인덱스 정리 (Transcript ID만 남김)
TU.index = TU.index.str.split("-", n=1).str[0]

# Pre/Post 분리
preTU = TU.iloc[:, 1::2] 
postTU = TU.iloc[:, 0::2]

preTU = preTU.fillna(0)
postTU = postTU.fillna(0)

# ---------------------------------------------------------
# 2. 샘플 분리 및 리스트 전처리
# ---------------------------------------------------------
# (1) 샘플 분리
ar_samples = sampleinfo[sampleinfo['response'] == 1].index.intersection(preTU.columns)
ir_samples = sampleinfo[sampleinfo['response'] == 0].index.intersection(preTU.columns)

# (2) DUT List 전처리 (ID만 추출)
ar_isoforms_clean = [x.split('-', 1)[0] for x in ARdutlist]
ir_isoforms_clean = [x.split('-', 1)[0] for x in IRdutlist]

# ---------------------------------------------------------
# 3. 데이터 집계 (Aggregation): 샘플별 평균 계산
# ---------------------------------------------------------
# Coding 여부(True/False)와 Response(AR/IR)에 따라 반복문을 돌며 평균을 계산합니다.

# df_cat의 isoform 컬럼도 ID만 남도록 정리 필요 (안전장치)
df_cat = sqanti[['isoform','structural_category','subcategory','within_CAGE_peak','coding']].copy()
df_cat['isoform_clean'] = df_cat['isoform'].astype(str).str.split('-', n=1).str[0]
majorlist_set = set([x.split('-', 1)[0] for x in majorlist])
coding_set = set(sqanti[sqanti['coding']=='coding']['isoform'])
majorlist_set = majorlist_set.intersection(coding_set)
df_cat['major'] = df_cat['isoform'].isin(majorlist_set)
####filter#####
valid_cat = {"full-splice_match", "novel_in_catalog"}
df_cat['major'] = df_cat['major'] & df_cat['structural_category'].isin(valid_cat)
df_cat.set_index('isoform', inplace=True)
###############

group1 = df_cat[df_cat['major']==True]['isoform_clean'].to_list()
group2 = df_cat[(df_cat['major']==False)&(df_cat['coding']=='coding')]['isoform_clean'].to_list()
group3 = df_cat[(df_cat['major']==False)&(df_cat['coding']=='non_coding')]['isoform_clean'].to_list()

class1 = majorminor[majorminor['transcriptid'].isin(group1)]['Transcript-Gene'].to_list()
class2 = majorminor[majorminor['transcriptid'].isin(group2)]['Transcript-Gene'].to_list()
class3 = majorminor[majorminor['transcriptid'].isin(group3)]['Transcript-Gene'].to_list()

geneexp = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_80_gene_TPM.txt', sep='\t', index_col=0)
# %%
####^^ BRCA1 example: gene-level non-significant but DUT significant in AR ####
gene_pre = geneexp.iloc[:, 1::2].copy()
gene_post = geneexp.iloc[:, 0::2].copy()
gene_pre.columns = gene_pre.columns.str[:-4]
gene_post.columns = gene_post.columns.str[:-4]

brca1_ar_samples = sampleinfo[sampleinfo['response'] == 1].index.tolist()


def build_paired_long_df(pre_mat, post_mat, feature_id, sample_ids):
    valid_samples = [s for s in sample_ids if s in pre_mat.columns and s in post_mat.columns]
    paired_df = pd.DataFrame({
        'Sample': valid_samples,
        'pre': pre_mat.loc[feature_id, valid_samples].astype(float).values,
        'post': post_mat.loc[feature_id, valid_samples].astype(float).values
    }).dropna()

    long_df = paired_df.melt(
        id_vars='Sample',
        value_vars=['pre', 'post'],
        var_name='Time',
        value_name='Value'
    )
    long_df['Time'] = pd.Categorical(long_df['Time'], categories=['pre', 'post'], ordered=True)
    return paired_df, long_df


def add_wilcoxon_annotation(ax, long_df, order=('pre', 'post')):
    annotator = Annotator(
        ax,
        [('pre', 'post')],
        data=long_df,
        x='Time',
        y='Value',
        order=list(order)
    )
    annotator.configure(
        test='Wilcoxon',
        text_format='star',
        pvalue_format_string='{:.3g}',
        show_test_name=False,
        loc='inside',
        line_height=0.02,
        verbose=0
    )
    annotator.apply_and_annotate()
    return annotator


def plot_paired_boxplot(ax, paired_df, long_df, title, ylabel):
    palette = {"pre": "#FFEDA0","post": "#FEB24C"}

    sns.boxplot(
        data=long_df,
        x='Time',
        y='Value',
        hue='Time',
        order=['pre', 'post'],
        hue_order=['pre', 'post'],
        palette=palette,
        dodge=False,
        legend=False,
        showfliers=False,
        width=0.55,
        linewidth=1.3,
        ax=ax
    )

    for _, row in paired_df.iterrows():
        ax.plot([0, 1], [row['pre'], row['post']], color='grey', alpha=0.45, lw=1, zorder=1)

    sns.stripplot(
        data=long_df,
        x='Time',
        y='Value',
        order=['pre', 'post'],
        color='black',
        size=4,
        alpha=0.7,
        jitter=0.08,
        ax=ax,
        zorder=2
    )

    add_wilcoxon_annotation(ax, long_df)
    ax.set_title(title, pad=10)
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    sns.despine(ax=ax)


brca1_gene_paired, brca1_gene_long = build_paired_long_df(
    gene_pre, gene_post, 'BRCA1', brca1_ar_samples
)
brca1_tx1_paired, brca1_tx1_long = build_paired_long_df(
    preTU, postTU, 'ENST00000357654.3', brca1_ar_samples
)
brca1_tx2_paired, brca1_tx2_long = build_paired_long_df(
    preTU, postTU, 'ENST00000467274.1', brca1_ar_samples
)

fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))

plot_paired_boxplot(
    axes[0],
    brca1_gene_paired,
    brca1_gene_long,
    'BRCA1',
    'TPM'
)
plot_paired_boxplot(
    axes[1],
    brca1_tx1_paired,
    brca1_tx1_long,
    'ENST00000357654.3-BRCA1',
    'Transcript usage'
)
plot_paired_boxplot(
    axes[2],
    brca1_tx2_paired,
    brca1_tx2_long,
    'ENST00000467274.1-BRCA1',
    'Transcript usage'
)

plt.tight_layout()

save_path = '/home/jiye/jiye/copycomparison/GENCODEquant/figures/BRCA1_AR_prepost_gene_transcript_boxplots.pdf'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig)

print(f'Saved BRCA1 AR boxplots to: {save_path}')

# %%
####^^ Class 1/2/3 AR DUT vs IR DUT venn diagrams ####

from matplotlib_venn import venn2

def get_parent_genes(tx_set):
    return {tx.split('-', 1)[-1] for tx in tx_set}


def style_venn(ax, venn_obj, title):
    ax.set_title(title)

    patch_colors = {
        '10': '#FEB24C',
        '01': '#5AAE61',
        '11': '#ACB056'
    }
    for patch_id, color in patch_colors.items():
        patch = venn_obj.get_patch_by_id(patch_id)
        if patch is not None:
            patch.set_color(color)
            patch.set_alpha(0.75)

    if venn_obj.set_labels is not None:
        for text in venn_obj.set_labels:
            if text is not None:
                text.set_fontsize(12)

    if venn_obj.subset_labels is not None:
        for text in venn_obj.subset_labels:
            if text is not None:
                text.set_fontsize(14)
                text.set_weight('bold')


def plot_class_dut_venn(class_label, class_members, ar_dut, ir_dut, outdir):
    ar_tx = set(ar_dut).intersection(set(class_members))
    ir_tx = set(ir_dut).intersection(set(class_members))
    ar_genes = get_parent_genes(ar_tx)
    ir_genes = get_parent_genes(ir_tx)

    print(f"\n===== {class_label} =====")
    print(f"AR DUT transcripts: {len(ar_tx)}")
    print(f"IR DUT transcripts: {len(ir_tx)}")
    print(f"Shared DUT transcripts: {len(ar_tx & ir_tx)}")
    print(f"AR DUT parent genes: {len(ar_genes)}")
    print(f"IR DUT parent genes: {len(ir_genes)}")
    print(f"Shared DUT parent genes: {len(ar_genes & ir_genes)}")

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))

    if len(ar_tx) == 0 and len(ir_tx) == 0:
        axes[0].axis('off')
        axes[0].text(0.5, 0.5, 'No DUT transcripts', ha='center', va='center', fontsize=13)
        axes[0].set_title(f'{class_label} transcript-level DUT')
    else:
        venn_tx = venn2(
            [ar_tx, ir_tx],
            set_labels=('AR DUT', 'IR DUT'),
            ax=axes[0]
        )
        style_venn(axes[0], venn_tx, f'{class_label} transcript-level DUT')

    if len(ar_genes) == 0 and len(ir_genes) == 0:
        axes[1].axis('off')
        axes[1].text(0.5, 0.5, 'No DUT parent genes', ha='center', va='center', fontsize=13)
        axes[1].set_title(f'{class_label} parent gene')
    else:
        venn_gene = venn2(
            [ar_genes, ir_genes],
            set_labels=('AR DUT gene', 'IR DUT gene'),
            ax=axes[1]
        )
        style_venn(axes[1], venn_gene, f'{class_label} parent gene')

    plt.tight_layout()

    save_path = f'{outdir}/{class_label}_AR_IR_DUT_venn.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f'Saved {class_label} venn figure to: {save_path}')


venn_outdir = '/home/jiye/jiye/copycomparison/GENCODEquant/figures'
os.makedirs(venn_outdir, exist_ok=True)

class_membership = {
    'Class1': class1,
    'Class2': class2,
    'Class3': class3
}

for class_label, class_members in class_membership.items():
    plot_class_dut_venn(
        class_label=class_label,
        class_members=class_members,
        ar_dut=ARdutlist,
        ir_dut=IRdutlist,
        outdir=venn_outdir
    )

# %%
####^^ Overall AR DUT vs IR DUT venn diagram ####
def plot_overall_dut_venn(ar_dut, ir_dut, outdir):
    ar_tx = set(ar_dut)
    ir_tx = set(ir_dut)
    ar_genes = get_parent_genes(ar_tx)
    ir_genes = get_parent_genes(ir_tx)

    print("\n===== Overall DUT =====")
    print(f"AR DUT transcripts: {len(ar_tx)}")
    print(f"IR DUT transcripts: {len(ir_tx)}")
    print(f"Shared DUT transcripts: {len(ar_tx & ir_tx)}")
    print(f"AR DUT parent genes: {len(ar_genes)}")
    print(f"IR DUT parent genes: {len(ir_genes)}")
    print(f"Shared DUT parent genes: {len(ar_genes & ir_genes)}")

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))

    venn_tx = venn2(
        [ar_tx, ir_tx],
        set_labels=('AR DUT', 'IR DUT'),
        ax=axes[0]
    )
    style_venn(axes[0], venn_tx, 'Overall transcript-level DUT')

    venn_gene = venn2(
        [ar_genes, ir_genes],
        set_labels=('AR DUT gene', 'IR DUT gene'),
        ax=axes[1]
    )
    style_venn(axes[1], venn_gene, 'Overall parent gene')

    plt.tight_layout()

    save_path = f'{outdir}/Overall_AR_IR_DUT_venn.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f'Saved overall venn figure to: {save_path}')


plot_overall_dut_venn(
    ar_dut=ARdutlist,
    ir_dut=IRdutlist,
    outdir=venn_outdir
)

# %%
####^^ Cell cycle (cyclone) dynamics: AR/IR pre vs post ####
cyclone = pd.read_csv(
    '/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/80_cyclone_result.tsv',
    sep='\t'
)

# Use the phase label already assigned in the cyclone output.
sample_to_response = (
    sampleinfo_full[['sample_full', 'response']]
    .drop_duplicates()
    .set_index('sample_full')['response']
    .map({1: 'AR', 0: 'IR'})
)

cyclone['Group'] = cyclone['Sample'].map(sample_to_response)
cyclone['Patient'] = cyclone['Sample'].str.replace(r'-(atD|bfD)$', '', regex=True)
cyclone['Time'] = cyclone['Sample'].str.extract(r'-(atD|bfD)$')[0].map({
    'bfD': 'pre',
    'atD': 'post'
})

cyclone = cyclone.dropna(subset=['Group', 'Time']).copy()
phase_order = ['G1', 'S', 'G2M']
time_order = ['pre', 'post']
phase_palette = {
    'G1': '#F4A259',
    'S': '#4C9F70',
    'G2M': '#5B8EDE'
}


def plot_phase_composition(ax, cyclone_df, group_label):
    sub = cyclone_df[cyclone_df['Group'] == group_label].copy()
    comp = pd.crosstab(sub['Time'], sub['Phase'], normalize='index')
    comp = comp.reindex(index=time_order, columns=phase_order, fill_value=0)

    x = np.arange(len(time_order))
    bottom = np.zeros(len(time_order))

    for phase in phase_order:
        vals = comp[phase].values
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=phase_palette[phase],
            edgecolor='white',
            linewidth=1.2,
            width=0.6,
            label=phase
        )
        for i, val in enumerate(vals):
            if val >= 0.08:
                ax.text(
                    x[i],
                    bottom[i] + val / 2,
                    f'{val:.0%}',
                    ha='center',
                    va='center',
                    fontsize=11,
                    color='white',
                    weight='bold'
                )
        bottom += vals

    n_patients = sub['Patient'].nunique()
    ax.set_xticks(x)
    ax.set_xticklabels(time_order)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_xlabel('')
    ax.set_ylabel('Sample proportion')
    ax.set_title(f'{group_label}: dominant phase composition (n={n_patients})')
    sns.despine(ax=ax)


def plot_phase_transition_heatmap(ax, cyclone_df, group_label):
    sub = cyclone_df[cyclone_df['Group'] == group_label].copy()
    paired_phase = (
        sub.pivot_table(index='Patient', columns='Time', values='Phase', aggfunc='first')
        .reindex(columns=time_order)
        .dropna()
    )

    transition = pd.crosstab(
        pd.Categorical(paired_phase['pre'], categories=phase_order, ordered=True),
        pd.Categorical(paired_phase['post'], categories=phase_order, ordered=True),
        dropna=False
    ).reindex(index=phase_order, columns=phase_order, fill_value=0)

    sns.heatmap(
        transition,
        annot=True,
        fmt='d',
        cmap=sns.light_palette('#5B8EDE', as_cmap=True),
        cbar=False,
        linewidths=1,
        linecolor='white',
        square=True,
        ax=ax
    )
    ax.set_xlabel('Post dominant phase')
    ax.set_ylabel('Pre dominant phase')
    ax.set_title(f'{group_label}: pre → post phase transition')


print("\n===== Cyclone cell cycle summary =====")
for group_label in ['AR', 'IR']:
    sub = cyclone[cyclone['Group'] == group_label]
    paired_n = (
        sub.pivot_table(index='Patient', columns='Time', values='Phase', aggfunc='first')
        .dropna()
        .shape[0]
    )
    print(f"{group_label} paired patients: {paired_n}")
    print(pd.crosstab(sub['Time'], sub['Phase']).reindex(index=time_order, columns=phase_order, fill_value=0))

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

plot_phase_composition(axes[0, 0], cyclone, 'AR')
plot_phase_composition(axes[0, 1], cyclone, 'IR')
plot_phase_transition_heatmap(axes[1, 0], cyclone, 'AR')
plot_phase_transition_heatmap(axes[1, 1], cyclone, 'IR')

legend_handles = [
    matplotlib.patches.Patch(facecolor=phase_palette[phase], edgecolor='white', label=phase)
    for phase in phase_order
]
fig.legend(
    handles=legend_handles,
    labels=phase_order,
    loc='upper center',
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, 1.02)
)

plt.tight_layout(rect=[0, 0, 1, 0.96])

save_path = '/home/jiye/jiye/copycomparison/GENCODEquant/figures/cyclone_AR_IR_prepost_summary.pdf'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig)

print(f'Saved cyclone summary figure to: {save_path}')

# %%
####^^ GSVA-based cell-cycle scoring (S / G2M) ####

import gseapy as gp

# Standard human S and G2M phase markers used in Seurat/Scanpy-style cell-cycle scoring.
s_genes = [
    'MCM5', 'PCNA', 'TYMS', 'FEN1', 'MCM2', 'MCM4', 'RRM1', 'UNG', 'GINS2',
    'MCM6', 'CDCA7', 'DTL', 'PRIM1', 'UHRF1', 'HELLS', 'RFC2', 'RPA2', 'NASP',
    'RAD51AP1', 'GMNN', 'WDR76', 'SLBP', 'CCNE2', 'UBR7', 'POLD3', 'MSH2',
    'ATAD2', 'RAD51', 'RRM2', 'CDC45', 'CDC6', 'EXO1', 'TIPIN', 'DSCC1', 'BLM',
    'CASP8AP2', 'USP1', 'CLSPN', 'POLA1', 'CHAF1B', 'BRIP1', 'E2F8'
]
g2m_genes = [
    'HMGB2', 'CDK1', 'NUSAP1', 'UBE2C', 'BIRC5', 'TPX2', 'TOP2A', 'NDC80',
    'CKS2', 'NUF2', 'CKS1B', 'MKI67', 'TMPO', 'CENPF', 'TACC3', 'FAM64A',
    'SMC4', 'CCNB2', 'CKAP2L', 'CKAP2', 'AURKB', 'BUB1', 'KIF11', 'ANP32E',
    'TUBB4B', 'GTSE1', 'KIF20B', 'HJURP', 'CDCA3', 'HN1', 'CDC20', 'TTK',
    'CDC25C', 'KIF2C', 'RANGAP1', 'NCAPD2', 'DLGAP5', 'CDCA2', 'CDCA8', 'ECT2',
    'KIF23', 'HMMR', 'AURKA', 'PSRC1', 'ANLN', 'LBR', 'CKAP5', 'CENPE',
    'CTCF', 'NEK2', 'G2E3', 'GAS2L3', 'CBX5', 'CENPA'
]

gsva_gene_sets = {
    'S_phase': s_genes,
    'G2M_phase': g2m_genes
}

gsva_expr = geneexp.copy()
gsva_expr.index = gsva_expr.index.astype(str).str.upper()
gsva_expr = gsva_expr.groupby(gsva_expr.index).mean()
gsva_expr = gsva_expr.loc[~gsva_expr.index.duplicated(), :]
gsva_expr = gsva_expr.fillna(0)

print("\n===== GSVA cell-cycle setup =====")
for term, genes in gsva_gene_sets.items():
    overlap = sorted(set(genes).intersection(gsva_expr.index))
    print(f"{term}: {len(overlap)}/{len(genes)} genes found in gene TPM matrix")

gsva_res = gp.gsva(
    data=gsva_expr,
    gene_sets=gsva_gene_sets,
    outdir=None,
    kcdf='Gaussian',
    min_size=10,
    max_size=500,
    threads=1,
    verbose=False
)

gsva_scores = (
    gsva_res.res2d
    .pivot(index='Name', columns='Term', values='ES')
    .rename_axis(index='Sample')
    .reset_index()
)

sample_to_response_gsva = (
    sampleinfo_full[['sample_full', 'response']]
    .drop_duplicates()
    .set_index('sample_full')['response']
    .map({1: 'AR', 0: 'IR'})
)

gsva_scores['Group'] = gsva_scores['Sample'].map(sample_to_response_gsva)
gsva_scores['Patient'] = gsva_scores['Sample'].str.replace(r'-(atD|bfD)$', '', regex=True)
gsva_scores['Time'] = gsva_scores['Sample'].str.extract(r'-(atD|bfD)$')[0].map({
    'bfD': 'pre',
    'atD': 'post'
})
gsva_scores = gsva_scores.dropna(subset=['Group', 'Time']).copy()
gsva_scores['S_phase'] = pd.to_numeric(gsva_scores['S_phase'], errors='coerce')
gsva_scores['G2M_phase'] = pd.to_numeric(gsva_scores['G2M_phase'], errors='coerce')
gsva_phase_order = ['G1', 'S', 'G2M']
gsva_phase_palette = {
    'G1': '#F4A259',
    'S': '#4C9F70',
    'G2M': '#5B8EDE'
}

gsva_scores['DominantPhase'] = np.where(
    (gsva_scores['S_phase'] <= 0) & (gsva_scores['G2M_phase'] <= 0),
    'G1',
    np.where(gsva_scores['S_phase'] >= gsva_scores['G2M_phase'], 'S', 'G2M')
)

gsva_scores.to_csv(
    '/home/jiye/jiye/copycomparison/GENCODEquant/figures/gsva_cell_cycle_scores.tsv',
    sep='\t',
    index=False
)


def plot_gsva_score_boxplot(ax, score_df, score_col, title, palette):
    order = ['AR', 'IR']
    hue_order = ['pre', 'post']
    plot_df = score_df.copy()
    plot_df[score_col] = pd.to_numeric(plot_df[score_col], errors='coerce')
    plot_df = plot_df.dropna(subset=[score_col, 'Group', 'Time'])

    sns.boxplot(
        data=plot_df,
        x='Group',
        y=score_col,
        hue='Time',
        order=order,
        hue_order=hue_order,
        palette=palette,
        showfliers=False,
        width=0.65,
        linewidth=1.3,
        ax=ax
    )
    sns.stripplot(
        data=plot_df,
        x='Group',
        y=score_col,
        hue='Time',
        order=order,
        hue_order=hue_order,
        dodge=True,
        palette={'pre': 'black', 'post': 'black'},
        size=3.5,
        alpha=0.65,
        ax=ax
    )

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    for group_idx, group_label in enumerate(order):
        sub = plot_df[plot_df['Group'] == group_label]
        x_pre = group_idx - 0.2
        x_post = group_idx + 0.2
        for patient in sub['Patient'].unique():
            patient_df = sub[sub['Patient'] == patient]
            if set(patient_df['Time']) == {'pre', 'post'}:
                pre_val = patient_df.loc[patient_df['Time'] == 'pre', score_col].iloc[0]
                post_val = patient_df.loc[patient_df['Time'] == 'post', score_col].iloc[0]
                ax.plot([x_pre, x_post], [pre_val, post_val], color='grey', alpha=0.4, lw=1, zorder=1)

    pairs = [(('AR', 'pre'), ('AR', 'post')), (('IR', 'pre'), ('IR', 'post'))]
    annotator = Annotator(
        ax,
        pairs,
        data=plot_df,
        x='Group',
        y=score_col,
        hue='Time',
        order=order,
        hue_order=hue_order
    )
    annotator.configure(
        test='Wilcoxon',
        text_format='full',
        pvalue_format_string='{:.3g}',
        show_test_name=False,
        loc='inside',
        line_height=0.02,
        verbose=0
    )
    annotator.apply_test(zero_method='pratt')
    annotator.annotate()

    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('GSVA score')
    sns.despine(ax=ax)


def plot_gsva_phase_composition(ax, score_df):
    plot_df = score_df.copy()
    plot_df['Condition'] = plot_df['Group'] + '_' + plot_df['Time']
    condition_order = ['AR_pre', 'AR_post', 'IR_pre', 'IR_post']
    comp = pd.crosstab(plot_df['Condition'], plot_df['DominantPhase'], normalize='index')
    comp = comp.reindex(index=condition_order, columns=gsva_phase_order, fill_value=0)

    x = np.arange(len(condition_order))
    bottom = np.zeros(len(condition_order))
    for phase in gsva_phase_order:
        vals = comp[phase].values
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=gsva_phase_palette[phase],
            edgecolor='white',
            linewidth=1.2,
            width=0.65,
            label=phase
        )
        for i, val in enumerate(vals):
            if val >= 0.08:
                ax.text(
                    x[i],
                    bottom[i] + val / 2,
                    f'{val:.0%}',
                    ha='center',
                    va='center',
                    fontsize=10,
                    color='white',
                    weight='bold'
                )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(['AR\npre', 'AR\npost', 'IR\npre', 'IR\npost'])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.set_xlabel('')
    ax.set_ylabel('Sample proportion')
    ax.set_title('dominant phase composition')
    sns.despine(ax=ax)


print("\n===== GSVA cell-cycle summary =====")
for group_label in ['AR', 'IR']:
    sub = gsva_scores[gsva_scores['Group'] == group_label]
    print(f"\n{group_label} dominant phase counts")
    print(pd.crosstab(sub['Time'], sub['DominantPhase']).reindex(index=time_order, columns=gsva_phase_order, fill_value=0))

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))

plot_gsva_score_boxplot(
    axes[0],
    gsva_scores,
    'S_phase',
    'S-phase score',
    {'pre': '#BFE3CF', 'post': '#4C9F70'}
)
plot_gsva_score_boxplot(
    axes[1],
    gsva_scores,
    'G2M_phase',
    'G2/M-phase score',
    {'pre': '#C9DAF2', 'post': '#5B8EDE'}
)
plot_gsva_phase_composition(axes[2], gsva_scores)

phase_handles = [
    matplotlib.patches.Patch(facecolor=gsva_phase_palette[phase], edgecolor='white', label=phase)
    for phase in gsva_phase_order
]
axes[2].legend(handles=phase_handles, frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1))

plt.tight_layout()

save_path = '/home/jiye/jiye/copycomparison/GENCODEquant/figures/gsva_cell_cycle_AR_IR_prepost_summary.pdf'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig)

print(f'Saved GSVA cell-cycle summary figure to: {save_path}')

# %%

# %%
###^^ SF gene #########

SF_genes = [
    'ACIN1', 'AGGF1', 'ALYREF', 'AQR', 'ARGLU1', 'BAG2', 'BCAS1', 'BCAS2', 'BUB3', 'BUD13',
    'BUD31', 'C17orf85', 'C19orf43', 'C1orf55', 'C1QBP', 'C9orf78', 'CACTIN', 'CCAR1', 'CCDC12', 'CCDC130',
    'CCDC75', 'CCDC94', 'CD2BP2', 'CDC40', 'CDC5L', 'CDK10', 'CDK11A', 'CDK12', 'CELF1', 'CELF2',
    'CELF3', 'CELF4', 'CELF5', 'CELF6', 'CFAP20', 'CHERP', 'CIRBP', 'CLASRP', 'CLK1', 'CLK2',
    'CLK3', 'CLK4', 'CLNS1A', 'CPSF6', 'CRNKL1', 'CSN3', 'CTNNBL1', 'CWC15', 'CWC22', 'CWC25',
    'CWC27', 'CXorf56', 'DDX1', 'DDX17', 'DDX18', 'DDX19A', 'DDX19B', 'DDX20', 'DDX21', 'DDX23',
    'DDX26B', 'DDX27', 'DDX39A', 'DDX39B', 'DDX3X', 'DDX3Y', 'DDX41', 'DDX42', 'DDX46', 'DDX5',
    'DDX50', 'DDX6', 'DGCR14', 'DHX15', 'DHX16', 'DHX30', 'DHX34', 'DHX35', 'DHX36', 'DHX38',
    'DHX40', 'DHX57', 'DHX8', 'DHX9', 'DNAJC6', 'DNAJC8', 'EEF1A1', 'EFTUD2', 'EIF2S2', 'EIF3A',
    'EIF4A3', 'ELAVL1', 'ELAVL2', 'ELAVL3', 'ELAVL4', 'FAM32A', 'FAM50A', 'FAM50B', 'FAM58A', 'FMR1',
    'FRA10AC1', 'FRG1', 'FUBP1', 'FUBP3', 'FUS', 'GEMIN2', 'GEMIN5', 'GNB2L1', 'GPATCH1', 'GPATCH3',
    'GPATCH8', 'GPKOW', 'GRSF1', 'HNRNPA0', 'HNRNPA1', 'HNRNPA2B1', 'HNRNPA3', 'HNRNPAB', 'HNRNPC', 'HNRNPCL1',
    'HNRNPD', 'HNRNPDL', 'HNRNPF', 'HNRNPH1', 'HNRNPH2', 'HNRNPH3', 'HNRNPK', 'HNRNPL', 'HNRNPLL', 'HNRNPM',
    'HNRNPR', 'HNRNPU', 'HNRNPUL1', 'HNRNPUL2', 'HSPA1A', 'HSPA1B', 'HSPA5', 'HSPA8', 'HSPB1', 'HTATSF1',
    'IGF2BP3', 'IK', 'ILF2', 'ILF3', 'INTS1', 'INTS3', 'INTS4', 'INTS5', 'INTS6', 'INTS7',
    'ISY1', 'JUP', 'KHDRBS1', 'KHDRBS3', 'KHSRP', 'KIAA1429', 'KIAA1967', 'KIN', 'LENG1', 'LOC649330',
    'LSM1', 'LSM10', 'LSM2', 'LSM3', 'LSM4', 'LSM5', 'LSM6', 'LSM7', 'NAA38', 'LSMD1',
    'LUC7L', 'LUC7L2', 'LUC7L3', 'MAGOH', 'MATR3', 'MBNL1', 'MBNL2', 'MBNL3', 'MFAP1', 'MFSD11',
    'MOV10', 'MSI1', 'MSI2', 'MYEF2', 'NCBP1', 'NCBP2', 'NELFE', 'NKAP', 'NONO', 'NOSIP',
    'NOVA1', 'NOVA2', 'NRIP2', 'NSRP1', 'NUDT21', 'NUMA1', 'PABPC1', 'PAXBP1', 'PCBP1', 'PCBP2',
    'PCBP3', 'PCBP4', 'PDCD7', 'PHF5A', 'PLRG1', 'PNN', 'PPIE', 'PPIG', 'PPIH', 'PPIL1',
    'PPIL2', 'PPIL3', 'PPIL4', 'PPM1G', 'PPP1CA', 'PPP1R8', 'PPWD1', 'PQBP1', 'PRCC', 'PRMT5',
    'PRPF18', 'PRPF19', 'PRPF3', 'PRPF31', 'PRPF38A', 'PRPF38B', 'PRPF39', 'PRPF4', 'PRPF40A', 'PRPF40B',
    'PRPF4B', 'PRPF6', 'PRPF8', 'PSEN1', 'PSIP1', 'PTBP1', 'PTBP2', 'PTBP3', 'PUF60', 'QKI',
    'RALY', 'RALYL', 'RAVER1', 'RAVER2', 'RBBP6', 'RBFOX2', 'RBM10', 'RBM14', 'RBM15', 'RBM15B',
    'RBM17', 'RBM22', 'RBM23', 'RBM25', 'RBM26', 'RBM27', 'RBM3', 'RBM39', 'RBM4', 'RBM42',
    'RBM45', 'RBM47', 'RBM4B', 'RBM5', 'RBM7', 'RBM8A', 'RBMS1', 'RBMX', 'RBMX2', 'RBMXL1',
    'RBMXL2', 'RNF113A', 'RNF20', 'RNF213', 'RNF34', 'RNF40', 'RNPC3', 'RNPS1', 'RNU1-1', 'RNU2-1',
    'RNU4-1', 'RNU5A-1', 'RNU6-1', 'SAP18', 'SAP30BP', 'SART1', 'SEC31B', 'SF1', 'SF3A1', 'SF3A2',
    'SF3A3', 'SF3B1', 'SF3B2', 'SF3B3', 'SF3B4', 'SF3B5', 'SF3B6', 'SFPQ', 'SKIV2L2', 'SLU7',
    'SMN1', 'SMNDC1', 'SMU1', 'SNIP1', 'SNRNP200', 'SNRNP25', 'SNRNP27', 'SNRNP35', 'SNRNP40', 'SNRNP48',
    'SNRNP70', 'SNRPA', 'SNRPA1', 'SNRPB', 'SNRPB2', 'SNRPC', 'SNRPD1', 'SNRPD2', 'SNRPD3', 'SNRPE',
    'SNRPF', 'SNRPG', 'SNRPN', 'NHP2L1', 'SNURF', 'SNW1', 'SPEN', 'SREK1', 'SRPK1', 'SRPK2',
    'SRPK3', 'SRRM1', 'SRRM2', 'SRRT', 'SRSF1', 'SRSF10', 'SRSF11', 'SRSF12', 'SRSF2', 'SRSF3',
    'SRSF4', 'SRSF5', 'SRSF6', 'SRSF7', 'SRSF8', 'SRSF9', 'SSB', 'SUGP1', 'SYF2', 'SYNCRIP',
    'TAF15', 'TCERG1', 'TFIP11', 'THOC1', 'THOC2', 'THOC3', 'THOC5', 'THOC6', 'THOC7', 'THRAP3',
    'TIA1', 'TIAL1', 'TNPO1', 'TOE1', 'TOP1MT', 'TOPORS', 'TRA2A', 'TRA2B', 'TRIM24', 'TTC14',
    'TXNL4A', 'U2AF1', 'U2AF1L4', 'U2AF2', 'U2SURP', 'UBL5', 'USP39', 'WBP11', 'WBP4', 'WDR77',
    'WDR83', 'WTAP', 'XAB2', 'YBX1', 'YBX3', 'ZC3H11A', 'ZC3H13', 'ZC3H18', 'ZC3H4', 'ZC3HAV1',
    'ZCCHC10', 'ZCCHC8', 'ZCRB1', 'ZFR', 'ZMAT2', 'ZMAT5', 'ZMYM3', 'ZNF131', 'ZNF207', 'ZNF326',
    'ZNF346', 'ZNF830', 'ZRSR1', 'ZRSR2'
]

# %%
####^^ Protein-coding gene-wise pre/post Spearman correlation ranking: all genes vs SF genes ####
def compute_gene_prepost_spearman(pre_df, post_df, sample_ids, sf_gene_list):
    valid_samples = [s for s in sample_ids if s in pre_df.columns and s in post_df.columns]
    common_genes = pre_df.index.intersection(post_df.index)
    pre_mat = pre_df.loc[common_genes, valid_samples].apply(pd.to_numeric, errors='coerce')
    post_mat = post_df.loc[common_genes, valid_samples].apply(pd.to_numeric, errors='coerce')

    valid_mask = (
        pre_mat.notna().all(axis=1)
        & post_mat.notna().all(axis=1)
        & (pre_mat.nunique(axis=1) > 1)
        & (post_mat.nunique(axis=1) > 1)
    )

    pre_mat = pre_mat.loc[valid_mask]
    post_mat = post_mat.loc[valid_mask]

    pre_rank = pre_mat.rank(axis=1, method='average')
    post_rank = post_mat.rank(axis=1, method='average')

    pre_center = pre_rank.sub(pre_rank.mean(axis=1), axis=0)
    post_center = post_rank.sub(post_rank.mean(axis=1), axis=0)
    numerator = (pre_center * post_center).sum(axis=1)
    denominator = np.sqrt(pre_center.pow(2).sum(axis=1) * post_center.pow(2).sum(axis=1))
    rho = (numerator / denominator).replace([np.inf, -np.inf], np.nan).dropna()

    corr_df = pd.DataFrame({
        'Gene': rho.index,
        'rho': rho.values,
        'n_pairs': len(valid_samples)
    })
    corr_df = corr_df.sort_values('rho', ascending=False).reset_index(drop=True)
    corr_df['rank'] = np.arange(1, corr_df.shape[0] + 1)
    corr_df['is_sf'] = corr_df['Gene'].isin(set(sf_gene_list))
    return corr_df


def plot_spearman_rank_scatter(ax, corr_df, group_label, base_color):
    sf_df = corr_df[corr_df['is_sf']].copy()
    top_5pct_cutoff = max(1, int(np.ceil(corr_df.shape[0] * 0.05)))
    top_10pct_cutoff = max(1, int(np.ceil(corr_df.shape[0] * 0.10)))

    ax.scatter(
        corr_df['rank'],
        corr_df['rho'],
        s=10,
        color=base_color,
        alpha=0.55,
        linewidth=0,
        rasterized=True,
        label='All genes'
    )
    ax.scatter(
        sf_df['rank'],
        sf_df['rho'],
        s=18,
        color='#C94137',
        alpha=0.95,
        linewidth=0,
        rasterized=True,
        label='SF genes'
    )

    ax.axhline(0, color='grey', linestyle='--', linewidth=1)
    ax.axvline(top_5pct_cutoff, color='#C94137', linestyle=':', linewidth=1)
    ax.set_xlim(0, corr_df.shape[0] + 1)
    ax.set_ylim(-1.02, 1.02)
    ax.set_xlabel('Gene rank (higher rho to the left)')
    ax.set_ylabel('Spearman rho')
    ax.set_title(f'{group_label}: pre vs post protein-coding gene correlation')
    ax.text(
        0.98,
        0.04,
        f'SF in top 5%: {(sf_df["rank"] <= top_5pct_cutoff).sum()}\n'
        f'SF in top 10%: {(sf_df["rank"] <= top_10pct_cutoff).sum()}',
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=10,
        color='#A3221A'
    )
    sns.despine(ax=ax)

protein_coding_gene_pre = gene_pre.loc[gene_pre.index.intersection(proteincodinglist)].copy()
protein_coding_gene_post = gene_post.loc[gene_post.index.intersection(proteincodinglist)].copy()

sf_ar_corr = compute_gene_prepost_spearman(
    protein_coding_gene_pre,
    protein_coding_gene_post,
    list(ar_samples),
    SF_genes
)
sf_ir_corr = compute_gene_prepost_spearman(
    protein_coding_gene_pre,
    protein_coding_gene_post,
    list(ir_samples),
    SF_genes
)

print("\n===== Protein-coding gene-wise pre/post Spearman correlation summary =====")
print(f"Genes tested: {protein_coding_gene_pre.shape[0]}")
for group_label, corr_df in [('AR', sf_ar_corr), ('IR', sf_ir_corr)]:
    sf_df = corr_df[corr_df['is_sf']]
    top_5pct_cutoff = max(1, int(np.ceil(corr_df.shape[0] * 0.05)))
    top_10pct_cutoff = max(1, int(np.ceil(corr_df.shape[0] * 0.10)))
    print(f"\n{group_label}")
    print(f"All genes with valid rho: {corr_df.shape[0]}")
    print(f"SF genes with valid rho: {sf_df.shape[0]}")
    print(f"SF genes in top 5% rho: {(sf_df['rank'] <= top_5pct_cutoff).sum()}")
    print(f"SF genes in top 10% rho: {(sf_df['rank'] <= top_10pct_cutoff).sum()}")
    print(f"Median rho (all genes): {corr_df['rho'].median():.3f}")
    print(f"Median rho (SF genes): {sf_df['rho'].median():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(8, 4.4), sharey=True)
plot_spearman_rank_scatter(axes[0], sf_ar_corr, 'AR', '#FEB24C')
plot_spearman_rank_scatter(axes[1], sf_ir_corr, 'IR', '#5AAE61')

handles = [
    matplotlib.lines.Line2D([0], [0], marker='o', linestyle='', color='#FEB24C', markersize=6, alpha=0.7, label='All genes'),
    matplotlib.lines.Line2D([0], [0], marker='o', linestyle='', color='#C94137', markersize=6, label='SF genes')
]
fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))

plt.tight_layout(rect=[0, 0, 1, 0.95])

save_path = '/home/jiye/jiye/copycomparison/GENCODEquant/figures/prepost_spearman_rank_SF_scatter_AR_IR.pdf'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig)

print(f'Saved SF correlation rank scatter to: {save_path}')

# %%
####^^ Strong SF genes from AR class1/class3 DUT delta correlation ####
def compute_sf_gene_vs_class_delta_corr(
    pre_gene_df,
    post_gene_df,
    pre_tu_df,
    post_tu_df,
    sample_ids,
    sf_gene_list,
    dut_tx_gene_list,
    rho_threshold=0.7
):
    valid_samples = [
        s for s in sample_ids
        if s in pre_gene_df.columns
        and s in post_gene_df.columns
        and s in pre_tu_df.columns
        and s in post_tu_df.columns
    ]

    dut_tx_ids = sorted({
        tx.split('-', 1)[0]
        for tx in dut_tx_gene_list
        if tx.split('-', 1)[0] in pre_tu_df.index and tx.split('-', 1)[0] in post_tu_df.index
    })

    delta_tu = post_tu_df.loc[dut_tx_ids, valid_samples] - pre_tu_df.loc[dut_tx_ids, valid_samples]
    mean_delta_tu = delta_tu.mean(axis=0)

    sf_genes_present = [
        gene for gene in sf_gene_list
        if gene in pre_gene_df.index and gene in post_gene_df.index
    ]
    delta_sf_exp = post_gene_df.loc[sf_genes_present, valid_samples] - pre_gene_df.loc[sf_genes_present, valid_samples]

    corr_rows = []
    for gene in sf_genes_present:
        x_vals = delta_sf_exp.loc[gene]
        valid_mask = x_vals.notna() & mean_delta_tu.notna()

        if valid_mask.sum() < 3:
            continue
        if x_vals.loc[valid_mask].nunique() < 2 or mean_delta_tu.loc[valid_mask].nunique() < 2:
            continue

        rho, pval = stats.spearmanr(
            x_vals.loc[valid_mask].values,
            mean_delta_tu.loc[valid_mask].values
        )
        corr_rows.append({
            'Gene': gene,
            'rho': rho,
            'p_value': pval,
            'n_samples': int(valid_mask.sum())
        })

    corr_df = pd.DataFrame(corr_rows)
    if corr_df.empty:
        corr_df = pd.DataFrame(columns=['Gene', 'rho', 'p_value', 'n_samples', 'selected_rho_gt_0.7'])
        selected_genes = []
        return corr_df, selected_genes, mean_delta_tu, dut_tx_ids

    corr_df = corr_df.sort_values('rho', ascending=False).reset_index(drop=True)
    corr_df['selected_rho_gt_0.7'] = corr_df['rho'] > rho_threshold
    selected_genes = corr_df.loc[corr_df['selected_rho_gt_0.7'], 'Gene'].tolist()

    return corr_df, selected_genes, mean_delta_tu, dut_tx_ids


def build_selected_sf_mean_expression_df(selected_genes, pre_gene_df, post_gene_df, ar_ids, ir_ids):
    valid_genes = [gene for gene in selected_genes if gene in pre_gene_df.index and gene in post_gene_df.index]
    if len(valid_genes) == 0:
        return pd.DataFrame(columns=['Sample', 'Group', 'Mean_TPM'])

    group_specs = [
        ('AR pre', pre_gene_df, list(ar_ids)),
        ('AR post', post_gene_df, list(ar_ids)),
        ('IR pre', pre_gene_df, list(ir_ids)),
        ('IR post', post_gene_df, list(ir_ids))
    ]

    plot_frames = []
    for group_label, source_df, sample_list in group_specs:
        valid_samples = [sample for sample in sample_list if sample in source_df.columns]
        mean_exp = source_df.loc[valid_genes, valid_samples].mean(axis=0)
        plot_frames.append(pd.DataFrame({
            'Sample': mean_exp.index,
            'Group': group_label,
            'Mean_TPM': mean_exp.values
        }))

    return pd.concat(plot_frames, ignore_index=True)


def plot_selected_sf_mean_expression_boxplot(ax, plot_df, title):
    order = ['AR pre', 'AR post', 'IR pre', 'IR post']
    palette = {
        'AR pre': '#FDD49E',
        'AR post': '#F28E2B',
        'IR pre': '#C7E9C0',
        'IR post': '#5AAE61'
    }

    if plot_df.empty:
        ax.text(0.5, 0.5, 'No SF genes with rho > 0.7', ha='center', va='center', fontsize=12)
        ax.set_title(title)
        ax.set_xlabel('')
        ax.set_ylabel('Mean TPM')
        ax.set_xticks([])
        sns.despine(ax=ax, left=False, bottom=True)
        return

    sns.boxplot(
        data=plot_df,
        x='Group',
        y='Mean_TPM',
        hue='Group',
        order=order,
        hue_order=order,
        palette=palette,
        dodge=False,
        width=0.6,
        showfliers=False,
        ax=ax
    )
    if ax.legend_ is not None:
        ax.legend_.remove()
    sns.stripplot(
        data=plot_df,
        x='Group',
        y='Mean_TPM',
        order=order,
        color='black',
        alpha=0.55,
        size=3,
        jitter=0.18,
        ax=ax
    )

    pairs = [
        
        ('AR pre', 'AR post'),
        ('IR pre', 'IR post')
    ]
    annotator = Annotator(
        ax,
        pairs,
        data=plot_df,
        x='Group',
        y='Mean_TPM',
        order=order
    )
    annotator.configure(
        test='Wilcoxon',
        text_format='star',
        loc='outside',
        show_test_name=False,
        verbose=0
    )
    annotator.apply_and_annotate()

    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('Mean TPM across selected SF genes')
    sns.despine(ax=ax)


def build_selected_sf_delta_df(selected_genes, pre_gene_df, post_gene_df, ar_ids, ir_ids):
    valid_genes = [gene for gene in selected_genes if gene in pre_gene_df.index and gene in post_gene_df.index]
    if len(valid_genes) == 0:
        return pd.DataFrame(columns=['Sample', 'Group', 'Delta_Mean_TPM'])

    group_specs = [
        ('AR', list(ar_ids)),
        ('IR', list(ir_ids))
    ]

    plot_frames = []
    for group_label, sample_list in group_specs:
        valid_samples = [
            sample for sample in sample_list
            if sample in pre_gene_df.columns and sample in post_gene_df.columns
        ]
        delta_mean_exp = (
            post_gene_df.loc[valid_genes, valid_samples].mean(axis=0)
            - pre_gene_df.loc[valid_genes, valid_samples].mean(axis=0)
        )
        plot_frames.append(pd.DataFrame({
            'Sample': delta_mean_exp.index,
            'Group': group_label,
            'Delta_Mean_TPM': delta_mean_exp.values
        }))

    return pd.concat(plot_frames, ignore_index=True)


def plot_selected_sf_delta_boxplot(ax, plot_df, title):
    order = ['AR', 'IR']
    palette = {
        'AR': '#F28E2B',
        'IR': '#5AAE61'
    }

    if plot_df.empty:
        ax.text(0.5, 0.5, 'No SF genes with rho threshold', ha='center', va='center', fontsize=12)
        ax.set_title(title)
        ax.set_xlabel('')
        ax.set_ylabel('Delta mean TPM')
        ax.set_xticks([])
        sns.despine(ax=ax, left=False, bottom=True)
        return

    sns.boxplot(
        data=plot_df,
        x='Group',
        y='Delta_Mean_TPM',
        hue='Group',
        order=order,
        hue_order=order,
        palette=palette,
        dodge=False,
        width=0.6,
        showfliers=False,
        ax=ax
    )
    if ax.legend_ is not None:
        ax.legend_.remove()
    sns.stripplot(
        data=plot_df,
        x='Group',
        y='Delta_Mean_TPM',
        order=order,
        color='black',
        alpha=0.6,
        size=3.5,
        jitter=0.16,
        ax=ax
    )

    annotator = Annotator(
        ax,
        [('AR', 'IR')],
        data=plot_df,
        x='Group',
        y='Delta_Mean_TPM',
        order=order
    )
    annotator.configure(
        test='Mann-Whitney',
        text_format='star',
        loc='outside',
        show_test_name=False,
        verbose=0
    )
    annotator.apply_and_annotate()

    ax.axhline(0, color='grey', linestyle='--', linewidth=1)
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('Delta mean TPM (post - pre)\nacross selected SF genes')
    sns.despine(ax=ax)


strong_sf_rho_threshold = 0.8
ar_class1_dut = sorted(
    set(class1).intersection(
        set(AR_dut.loc[(AR_dut['p_value'] < 0.05) & (AR_dut['delta_TU'] > 0.05)].index)
    )
)
ar_class3_dut = sorted(
    set(class3).intersection(
        set(AR_dut.loc[(AR_dut['p_value'] < 0.05) & (AR_dut['delta_TU'] < -0.05)].index)
    )
)

class1_sf_corr, class1_strong_sf_genes, class1_mean_delta_tu, class1_dut_tx_ids = compute_sf_gene_vs_class_delta_corr(
    gene_pre,
    gene_post,
    preTU,
    postTU,
    list(ar_samples),
    SF_genes,
    ar_class1_dut,
    rho_threshold=strong_sf_rho_threshold
)
class3_sf_corr, class3_strong_sf_genes, class3_mean_delta_tu, class3_dut_tx_ids = compute_sf_gene_vs_class_delta_corr(
    gene_pre,
    gene_post,
    preTU,
    postTU,
    list(ar_samples),
    SF_genes,
    ar_class3_dut,
    rho_threshold=strong_sf_rho_threshold
)

class1_sf_corr['Class'] = 'Class1'
class3_sf_corr['Class'] = 'Class3'
strong_sf_corr_df = pd.concat([class1_sf_corr, class3_sf_corr], ignore_index=True)

strong_sf_table_path = '/home/jiye/jiye/copycomparison/GENCODEquant/figures/AR_class1_class3_SF_delta_correlation.tsv'
strong_sf_corr_df.to_csv(strong_sf_table_path, sep='\t', index=False)

print("\n===== Strong SF genes from AR class-specific DUT delta correlation =====")
print(f"Class1 AR DUT transcripts used: {len(class1_dut_tx_ids)}")
print(f"Class3 AR DUT transcripts used: {len(class3_dut_tx_ids)}")
print(f"Class1 strong SF genes (rho > {strong_sf_rho_threshold}): {len(class1_strong_sf_genes)}")
print(class1_strong_sf_genes)
print(f"Class3 strong SF genes (rho > {strong_sf_rho_threshold}): {len(class3_strong_sf_genes)}")
print(class3_strong_sf_genes)
print(f"Saved SF correlation table to: {strong_sf_table_path}")

class1_meanexp_df = build_selected_sf_mean_expression_df(
    class1_strong_sf_genes,
    gene_pre,
    gene_post,
    ar_samples,
    ir_samples
)
class3_meanexp_df = build_selected_sf_mean_expression_df(
    class3_strong_sf_genes,
    gene_pre,
    gene_post,
    ar_samples,
    ir_samples
)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
plot_selected_sf_mean_expression_boxplot(
    axes[0],
    class1_meanexp_df,
    f'Class1-correlated SF genes\n(rho > {strong_sf_rho_threshold}, n={len(class1_strong_sf_genes)})'
)
plot_selected_sf_mean_expression_boxplot(
    axes[1],
    class3_meanexp_df,
    f'Class3-correlated SF genes\n(rho > {strong_sf_rho_threshold}, n={len(class3_strong_sf_genes)})'
)

plt.tight_layout()

save_path = '/home/jiye/jiye/copycomparison/GENCODEquant/figures/AR_class1_class3_strong_SF_mean_expression_boxplots.pdf'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig)

print(f'Saved strong SF mean-expression boxplots to: {save_path}')

# %%
class1_delta_df = build_selected_sf_delta_df(
    class1_strong_sf_genes,
    gene_pre,
    gene_post,
    ar_samples,
    ir_samples
)
class3_delta_df = build_selected_sf_delta_df(
    class3_strong_sf_genes,
    gene_pre,
    gene_post,
    ar_samples,
    ir_samples
)

strong_sf_delta_table_path = '/home/jiye/jiye/copycomparison/GENCODEquant/figures/AR_class1_class3_strong_SF_sample_delta.tsv'
delta_table_frames = []
if not class1_delta_df.empty:
    delta_table_frames.append(class1_delta_df.assign(Class='Class1'))
if not class3_delta_df.empty:
    delta_table_frames.append(class3_delta_df.assign(Class='Class3'))

if delta_table_frames:
    delta_table_df = pd.concat(delta_table_frames, ignore_index=True)
else:
    delta_table_df = pd.DataFrame(columns=['Sample', 'Group', 'Delta_Mean_TPM', 'Class'])

delta_table_df.to_csv(strong_sf_delta_table_path, sep='\t', index=False)

print("\n===== Sample-level delta comparison for selected SF genes =====")
if not class1_delta_df.empty:
    class1_ar_delta = class1_delta_df.loc[class1_delta_df['Group'] == 'AR', 'Delta_Mean_TPM']
    class1_ir_delta = class1_delta_df.loc[class1_delta_df['Group'] == 'IR', 'Delta_Mean_TPM']
    class1_mwu = stats.mannwhitneyu(class1_ar_delta, class1_ir_delta, alternative='two-sided')
    print(f"Class1 delta AR vs IR Mann-Whitney p = {class1_mwu.pvalue:.4g}")
else:
    print("Class1 delta AR vs IR: no selected SF genes")

if not class3_delta_df.empty:
    class3_ar_delta = class3_delta_df.loc[class3_delta_df['Group'] == 'AR', 'Delta_Mean_TPM']
    class3_ir_delta = class3_delta_df.loc[class3_delta_df['Group'] == 'IR', 'Delta_Mean_TPM']
    class3_mwu = stats.mannwhitneyu(class3_ar_delta, class3_ir_delta, alternative='two-sided')
    print(f"Class3 delta AR vs IR Mann-Whitney p = {class3_mwu.pvalue:.4g}")
else:
    print("Class3 delta AR vs IR: no selected SF genes")

print(f"Saved sample-level delta table to: {strong_sf_delta_table_path}")

fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.6), sharey=True)
plot_selected_sf_delta_boxplot(
    axes[0],
    class1_delta_df,
    f'Class1 selected SF genes delta\n(rho > {strong_sf_rho_threshold}, n={len(class1_strong_sf_genes)})'
)
plot_selected_sf_delta_boxplot(
    axes[1],
    class3_delta_df,
    f'Class3 selected SF genes delta\n(rho > {strong_sf_rho_threshold}, n={len(class3_strong_sf_genes)})'
)

plt.tight_layout()

save_path = '/home/jiye/jiye/copycomparison/GENCODEquant/figures/AR_class1_class3_strong_SF_delta_boxplots.pdf'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig)

print(f'Saved strong SF delta boxplots to: {save_path}')

# %%
