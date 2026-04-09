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
##############* py39 env for hdbscan #################

# %%
####^^ data prep ######
sqanti = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/sqantioutput/sqanti_hg19_classification.txt', sep='\t')
sqanti.dropna(axis=1, how='all', inplace=True)

AR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/whole_AR_stable_DUT_Wilcoxon_delta_withna.txt', sep='\t', index_col=0)
IR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/whole_IR_stable_DUT_Wilcoxon_delta_withna.txt', sep='\t', index_col=0)
# AR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/maintenance/AR_stable_DUT_Wilcoxon_delta_withna.txt', sep='\t', index_col=0) #^Only maintenance
# IR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/maintenance/IR_stable_DUT_Wilcoxon_delta_withna.txt', sep='\t', index_col=0) #^Only maintenance

ARdutlist = AR_dut.loc[(AR_dut['p_value']<0.05) & (np.abs(AR_dut['delta_TU'])>0.05)].index.to_list()
IRdutlist = IR_dut.loc[(IR_dut['p_value']<0.05) & (np.abs(IR_dut['delta_TU'])>0.05)].index.to_list()

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/GEN_FINALDATA/SEV_prepost_80_clinicalinfo.txt', sep='\t', index_col=0)
sampleinfo_full = sampleinfo.copy()
# sampleinfo = sampleinfo[sampleinfo['purpose']=='maintenance'] #^ Only maintenance

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

preTU = preTU.dropna()
postTU = postTU.dropna()

# ---------------------------------------------------------
# 2. 샘플 분리 및 리스트 전처리
# ---------------------------------------------------------
# (1) 샘플 분리
ar_samples = sampleinfo[sampleinfo['response'] == 1].index.intersection(preTU.columns)
ir_samples = sampleinfo[sampleinfo['response'] == 0].index.intersection(preTU.columns)

# (2) DUT List 전처리 (ID만 추출)
ar_isoforms_clean = [x.split('-', 1)[0] for x in ARdutlist]
ir_isoforms_clean = [x.split('-', 1)[0] for x in IRdutlist]

# df_cat의 isoform 컬럼도 ID만 남도록 정리 필요 (안전장치)
df_cat = sqanti[['isoform','structural_category','subcategory','within_CAGE_peak','coding']]
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

ARuplist = AR_dut.loc[(AR_dut['p_value']<0.05) & (AR_dut['delta_TU']>0.05)].index.to_list()
ARdownlist = AR_dut.loc[(AR_dut['p_value']<0.05) & (AR_dut['delta_TU']<-0.05)].index.to_list()
IRuplist = IR_dut.loc[(IR_dut['p_value']<0.05) & (IR_dut['delta_TU']>0.05)].index.to_list()
IRdownlist = IR_dut.loc[(IR_dut['p_value']<0.05) & (IR_dut['delta_TU']<-0.05)].index.to_list()

ar_psi = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/suppaoutput/AR/MW_psi_7events.txt',sep='\t')
ir_psi = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/suppaoutput/IR/MW_psi_7events.txt',sep='\t')
transinfo = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/cov5_filtered_transcripts_with_gene_info.tsv', sep='\t')
geneinfo = transinfo[['mstrg_gene_id','gene_name']].drop_duplicates()

ar_psi = pd.merge(ar_psi, geneinfo, left_on='gene_id', right_on='mstrg_gene_id', how='inner')
ir_psi = pd.merge(ir_psi, geneinfo, left_on='gene_id', right_on='mstrg_gene_id', how='inner')

#%%
####^^^^ ARprepost IRprepost 경향성 보기 #################33

from scipy.stats import mannwhitneyu, wilcoxon
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statannotations.Annotator import Annotator

def plot_responsive_dut_boxplot_final(
    pre_TU_gene, post_TU_gene, ar_dut_list, ir_dut_list, sampleinfo,
    response_col="response", ar_value=1, ir_value=0, title="Class1 + AR/IR DUTs"
):
    # 1. 샘플 및 유전자 준비
    ar_samples = sampleinfo[sampleinfo[response_col] == ar_value].index.intersection(pre_TU_gene.columns)
    ir_samples = sampleinfo[sampleinfo[response_col] == ir_value].index.intersection(pre_TU_gene.columns)

    tx_ar = set(ar_dut_list) & set(pre_TU_gene.index)
    tx_ir = set(ir_dut_list) & set(pre_TU_gene.index)

    # 데이터 구성 (결측치 제외)
    df_long = pd.concat([
        pd.DataFrame({'TU': pre_TU_gene.loc[list(tx_ar.union(tx_ir)), ar_samples].mean(), 'group': 'AR_pre'}), #*union
        pd.DataFrame({'TU': post_TU_gene.loc[list(tx_ar.union(tx_ir)), ar_samples].mean(), 'group': 'AR_post'}),
        pd.DataFrame({'TU': pre_TU_gene.loc[list(tx_ar.union(tx_ir)), ir_samples].mean(), 'group': 'IR_pre'}),
        pd.DataFrame({'TU': post_TU_gene.loc[list(tx_ar.union(tx_ir)), ir_samples].mean(), 'group': 'IR_post'})
    ]).dropna().reset_index()

    # 2. 시각화
    order = ["AR_pre", "AR_post", "IR_pre", "IR_post"]
    custom_palette = { "AR_pre": "#FFEDA0","AR_post": "#FEB24C","IR_pre": "#D9F0D3","IR_post": "#5AAE61"}
    #{'AR_pre': '#F1B08F', 'AR_post': '#EE7824', 'IR_pre': '#B2D085', 'IR_post': '#588513'}
    #{ "AR_Pre": "#FFEDA0","AR_Post": "#FEB24C","IR_Pre": "#D9F0D3","IR_Post": "#5AAE61"}

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(data=df_long, x='group', y='TU', order=order, palette=custom_palette, 
                showfliers=False, ax=ax, width=0.6, boxprops=dict(alpha=0.8))
    
    # 3. 통계 어노테이션 (Mann-Whitney로 변경하여 에러 방지)
    pairs = [("AR_pre", "AR_post"), ("IR_pre", "IR_post"), ("AR_pre", "IR_pre"),("AR_post", "IR_pre")]
    
    annotator = Annotator(ax, pairs, data=df_long, x='group', y='TU', order=order)
    # 샘플 수가 달라도 작동하는 Mann-Whitney 테스트 사용
    annotator.configure(test='Mann-Whitney', text_format='simple', loc='inside')
    annotator.apply_and_annotate()

    #ax.set_title(title, pad=20, fontweight='bold')
    ax.set_ylabel("Mean TU")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    sns.despine()
    plt.tight_layout()
    return df_long, fig, ax

def plot_all_class1_boxplot_final(
    pre_TU_gene, post_TU_gene, all_class1_list, sampleinfo,
    response_col="response", ar_value=1, ir_value=0, title="Class1 Transcripts"
):
    tx_all = list(set(all_class1_list) & set(pre_TU_gene.index) & set(post_TU_gene.index))
    ar_samples = sampleinfo[sampleinfo[response_col] == ar_value].index.intersection(pre_TU_gene.columns)
    ir_samples = sampleinfo[sampleinfo[response_col] == ir_value].index.intersection(pre_TU_gene.columns)

    df_long = pd.concat([
        pd.DataFrame({'TU': pre_TU_gene.loc[tx_all, ar_samples].mean(), 'group': 'AR_pre'}),
        pd.DataFrame({'TU': post_TU_gene.loc[tx_all, ar_samples].mean(), 'group': 'AR_post'}),
        pd.DataFrame({'TU': pre_TU_gene.loc[tx_all, ir_samples].mean(), 'group': 'IR_pre'}),
        pd.DataFrame({'TU': post_TU_gene.loc[tx_all, ir_samples].mean(), 'group': 'IR_post'})
    ]).dropna().reset_index()

    order = ["AR_pre", "AR_post", "IR_pre", "IR_post"]
    custom_palette = { "AR_pre": "#FFEDA0","AR_post": "#FEB24C","IR_pre": "#D9F0D3","IR_post": "#5AAE61"}

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(data=df_long, x='group', y='TU', order=order, palette=custom_palette, 
                showfliers=False, ax=ax, width=0.6, boxprops=dict(alpha=0.8))

    pairs = [("AR_pre", "AR_post"), ("IR_pre", "IR_post"), ("AR_pre", "IR_pre"), ("AR_post", "IR_pre"), ("AR_pre", "IR_post")]
    annotator = Annotator(ax, pairs, data=df_long, x='group', y='TU', order=order)
    annotator.configure(test='Mann-Whitney', text_format='simple', loc='inside')
    annotator.apply_and_annotate()

    ax.set_title(title, pad=20, fontweight='bold')
    ax.set_ylabel("Mean TU")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    sns.despine()
    plt.tight_layout()
    return df_long, fig, ax

delta_TU_gene = filtered_trans.copy()
delta_TU_gene = delta_TU_gene[delta_TU_gene.index.str.split("-", n=1).str[-1].isin(proteincodinglist)]

pre_TU_gene = delta_TU_gene.iloc[:, 1::2]
post_TU_gene = delta_TU_gene.iloc[:, 0::2]

pre_TU_gene.columns = pre_TU_gene.columns.str[:-4] 
post_TU_gene.columns = post_TU_gene.columns.str[:-4] 

ar_tx_class1 = list(set(ARdutlist) & set(class1))
ir_tx_class1 = list(set(IRdutlist) & set(class1))
ar_tx_class3 = list(set(ARdutlist) & set(class3))
ir_tx_class3 = list(set(IRdutlist) & set(class3))

# --- 실행 부분 ---
# (AR/IR dutlist 및 class1 준비 코드는 그대로 유지)
df_long_dut, fig_dut, ax_dut = plot_responsive_dut_boxplot_final(pre_TU_gene, post_TU_gene, ar_tx_class1, ir_tx_class1, sampleinfo)
plt.savefig("/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/class1_ARIRdutunion_boxplot.pdf", bbox_inches='tight', dpi=300)
plt.show()

df_long_all, fig_all, ax_all = plot_all_class1_boxplot_final(pre_TU_gene, post_TU_gene, class1, sampleinfo)
plt.show()

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logit
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse


def _prep_logit_z(preX, postX, ir_samples, eps=1e-3, use_zscore=True):
    # logit
    preX = preX.clip(eps, 1 - eps)
    postX = postX.clip(eps, 1 - eps)

    preT = pd.DataFrame(logit(preX.values), index=preX.index, columns=preX.columns)
    postT = pd.DataFrame(logit(postX.values), index=postX.index, columns=postX.columns)

    if use_zscore:
        mu = preT[ir_samples].mean(axis=1)
        sd = preT[ir_samples].std(axis=1).replace(0, np.nan)
        preT = preT.sub(mu, axis=0).div(sd, axis=0).fillna(0.0)
        postT = postT.sub(mu, axis=0).div(sd, axis=0).fillna(0.0)

    return preT, postT


def _add_cov_ellipse(ax, pts2d, nsig=1.0, face_alpha=0.08, edge_lw=2.5, zorder=50):
    """
    pts2d: (n,2) array
    """
    if pts2d.shape[0] < 5:
        return None

    center = pts2d.mean(axis=0)
    cov = np.cov(pts2d.T)

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * nsig * np.sqrt(np.maximum(vals, 1e-12))

    e = Ellipse(
        xy=center, width=width, height=height, angle=angle,
        facecolor="black", alpha=face_alpha,
        edgecolor="black", linewidth=edge_lw,
        zorder=zorder
    )
    ax.add_patch(e)
    ax.scatter([center[0]], [center[1]], marker="X", s=120, linewidths=2, zorder=zorder + 1)
    return e


def plot_ir_axis_main_figure(
    pre_TU,
    post_TU,
    sampleinfo,
    class1_list,
    ARdutlist,
    IRdutlist,
    response_col="response",
    ar_value=1,
    ir_value=0,
    eps=1e-3,
    use_zscore=True,
    drop_nan_features=True,   # 추천 True
    ellipse_nsig=1.0,         # 1.0이 "cluster 느낌" 잘 남
    ellipse_use_all_ir=True,  # True면 IR_pre+IR_post로 ellipse
    title="Class1 ∩ (AR/IR DUT union): shift along IR axis"
):
    # ---------------------------------------------------------
    # 1) Feature set = Class1 ∩ (ARdut ∪ IRdut)
    # ---------------------------------------------------------
    dut_union = set(ARdutlist) | set(IRdutlist)
    feat = list(set(class1_list) & dut_union &
                set(pre_TU.index) & set(post_TU.index))
    if len(feat) < 5:
        raise ValueError("Too few features after intersection: {}".format(len(feat)))

    preX = pre_TU.loc[feat].copy()
    postX = post_TU.loc[feat].copy()

    # ---------------------------------------------------------
    # 2) Align samples
    # ---------------------------------------------------------
    common_samples = preX.columns.intersection(postX.columns)
    preX = preX[common_samples]
    postX = postX[common_samples]

    resp = sampleinfo[response_col]
    ar_samples = resp[resp == ar_value].index.intersection(common_samples)
    ir_samples = resp[resp == ir_value].index.intersection(common_samples)

    if len(ar_samples) < 3 or len(ir_samples) < 3:
        raise ValueError("Not enough samples: AR={}, IR={}".format(len(ar_samples), len(ir_samples)))

    # ---------------------------------------------------------
    # 3) NaN handling
    # ---------------------------------------------------------
    if drop_nan_features:
        ok = (~preX.isna().any(axis=1)) & (~postX.isna().any(axis=1))
        preX = preX.loc[ok]
        postX = postX.loc[ok]
    else:
        preX = preX.fillna(0.0)
        postX = postX.fillna(0.0)

    n_features = preX.shape[0]
    if n_features < 5:
        raise ValueError("Too few features after NaN filtering: {}".format(n_features))

    # ---------------------------------------------------------
    # 4) logit + zscore
    # ---------------------------------------------------------
    preT, postT = _prep_logit_z(preX, postX, ir_samples, eps=eps, use_zscore=use_zscore)

    # ---------------------------------------------------------
    # 5) Direction vector v = mean(IR_pre) - mean(AR_pre)
    # ---------------------------------------------------------
    v = preT[ir_samples].mean(axis=1) - preT[ar_samples].mean(axis=1)
    v = v / (np.linalg.norm(v.values) + 1e-12)

    # scores
    score_pre = preT.T.dot(v)
    score_post = postT.T.dot(v)

    delta_ar = (score_post.loc[ar_samples] - score_pre.loc[ar_samples])
    _, p_delta = wilcoxon(delta_ar.values, alternative="greater")
    frac_pos = float((delta_ar > 0).mean())

    # ---------------------------------------------------------
    # 6) PCA coordinates (fit on IR_pre, IR_post, AR_pre, AR_post)
    # ---------------------------------------------------------
    cols_for_pca = list(ir_samples) + list(ar_samples)

    X_pre = preT[cols_for_pca].T
    X_post = postT[cols_for_pca].T
    X_all = pd.concat([X_pre, X_post], axis=0)

    pca = PCA(n_components=2, random_state=0)
    Z_all = pca.fit_transform(X_all.values)

    n = X_pre.shape[0]
    coord_pre = pd.DataFrame(Z_all[:n, :], index=X_pre.index, columns=["PC1", "PC2"])
    coord_post = pd.DataFrame(Z_all[n:, :], index=X_post.index, columns=["PC1", "PC2"])

    # centroids for arrow
    ar_pre_ctr = coord_pre.loc[ar_samples].mean(axis=0).values
    ir_pre_ctr = coord_pre.loc[ir_samples].mean(axis=0).values

    # ---------------------------------------------------------
    # 7) Plot (two panels)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7, 5))

    # --- Panel A: PCA + ellipse + arrow + AR flow
    ax = axes[0]

    # Points
    ax.scatter(coord_pre.loc[ir_samples, "PC1"], coord_pre.loc[ir_samples, "PC2"], alpha=0.9, label="IR_pre")
    ax.scatter(coord_pre.loc[ar_samples, "PC1"], coord_pre.loc[ar_samples, "PC2"], alpha=0.9, label="AR_pre")
    ax.scatter(coord_post.loc[ar_samples, "PC1"], coord_post.loc[ar_samples, "PC2"], alpha=0.9, label="AR_post")
    ax.scatter(coord_post.loc[ir_samples, "PC1"], coord_post.loc[ir_samples, "PC2"], alpha=0.6, label="IR_post")

    # AR pre->post lines
    for s in ar_samples:
        x0, y0 = coord_pre.loc[s, ["PC1", "PC2"]]
        x1, y1 = coord_post.loc[s, ["PC1", "PC2"]]
        ax.plot([x0, x1], [y0, y1], alpha=0.25, linewidth=1)

    # IR ellipse (use IR_pre only or IR_all)
    ir_pts = []
    for s in ir_samples:
        ir_pts.append(coord_pre.loc[s, ["PC1","PC2"]].values)
        if ellipse_use_all_ir:
            ir_pts.append(coord_post.loc[s, ["PC1","PC2"]].values)
    ir_pts = np.array(ir_pts, dtype=float)

    _add_cov_ellipse(ax, ir_pts, nsig=ellipse_nsig, face_alpha=0.08, edge_lw=2.8, zorder=30)

    # IR axis arrow (AR_pre -> IR_pre)
    ax.arrow(
        ar_pre_ctr[0], ar_pre_ctr[1],
        (ir_pre_ctr[0] - ar_pre_ctr[0]) * 0.9,
        (ir_pre_ctr[1] - ar_pre_ctr[1]) * 0.9,
        head_width=0.2, length_includes_head=True, linewidth=2.5, zorder=40
    )
    ax.text(ir_pre_ctr[0], ir_pre_ctr[1], "IR axis", fontsize=10, ha="left", va="bottom", zorder=41)

    # Text box
    txt = (
        f"n_features={n_features}\n"
        f"AR n={len(ar_samples)}, IR n={len(ir_samples)}\n"
        f"Δ>0 fraction={frac_pos:.2f}\n"
        f"Wilcoxon Δ>0 p={p_delta:.2e}"
    )
    ax.text(
        0.02, 0.98, txt, transform=ax.transAxes,
        ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    ax.set_title("PCA view + IR cluster + axis")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    # --- Panel B: Projection scores (paired AR)
    ax = axes[1]
    s_pre_ar = score_pre.loc[ar_samples]
    s_post_ar = score_post.loc[ar_samples]

    ax.scatter(np.zeros(len(ar_samples)), s_pre_ar.values)
    ax.scatter(np.ones(len(ar_samples)), s_post_ar.values)
    for s in ar_samples:
        ax.plot([0, 1], [s_pre_ar.loc[s], s_post_ar.loc[s]], alpha=0.35, linewidth=1)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["AR_pre", "AR_post"])
    ax.set_ylabel("IR-axis score")
    ax.set_title("Projection to IR direction")

    ax.text(
        0.02, 0.98,
        f"Δ>0 fraction={frac_pos:.2f}\nWilcoxon Δ>0 p={p_delta:.2e}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=10, bbox=dict(boxstyle="round", alpha=0.15)
    )
    ax.grid(alpha=0.2)

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()

    out = {
        "fig": fig,
        "coord_pre": coord_pre,
        "coord_post": coord_post,
        "score_pre": score_pre,
        "score_post": score_post,
        "delta_ar": delta_ar,
        "p_delta": float(p_delta),
        "frac_pos": frac_pos,
        "n_features": n_features,
        "v": v
    }
    return out


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse


def _ellipse_from_points(ax, pts2d, nsig=1.3, alpha=0.10, lw=2.5, zorder=5):
    # pts2d: (n,2)
    if pts2d.shape[0] < 5:
        return None
    center = pts2d.mean(axis=0)
    cov = np.cov(pts2d.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * nsig * np.sqrt(np.maximum(vals, 1e-12))
    e = Ellipse(
        xy=center, width=width, height=height, angle=angle,
        facecolor="black", alpha=alpha, edgecolor="black", linewidth=lw, zorder=zorder
    )
    ax.add_patch(e)
    return e


def plot_unsupervised_clusters_on_pca(
    coord_pre, coord_post,
    ar_samples, ir_samples,
    k=3,
    title="Unsupervised clusters on PCA (k=2)",
    draw_cluster_ellipses=True,
    ellipse_nsig=1.3
):
    """
    coord_pre/coord_post: DataFrames with columns ["PC1","PC2"], index=samples
    ar_samples, ir_samples: sample IDs (must match coord indices)

    Clustering is run on all 4 groups:
      AR_pre (from coord_pre), AR_post (from coord_post),
      IR_pre (from coord_pre), IR_post (from coord_post)

    Visualization:
      - fill color = biological group (AR_pre/AR_post/IR_pre/IR_post)
      - marker shape = cluster label (cluster 0 vs 1)
      - optional: ellipse per cluster
      - prints cluster composition table
    """

    # ---- build a single table of points
    rows = []

    for s in ir_samples:
        if s in coord_pre.index:
            rows.append((s, "IR_pre", coord_pre.loc[s, "PC1"], coord_pre.loc[s, "PC2"]))
        if s in coord_post.index:
            rows.append((s, "IR_post", coord_post.loc[s, "PC1"], coord_post.loc[s, "PC2"]))

    for s in ar_samples:
        if s in coord_pre.index:
            rows.append((s, "AR_pre", coord_pre.loc[s, "PC1"], coord_pre.loc[s, "PC2"]))
        if s in coord_post.index:
            rows.append((s, "AR_post", coord_post.loc[s, "PC1"], coord_post.loc[s, "PC2"]))

    df = pd.DataFrame(rows, columns=["sample", "group", "PC1", "PC2"])
    if df.shape[0] < 10:
        raise ValueError("Too few points for clustering/plot.")

    X = df[["PC1", "PC2"]].values

    # ---- kmeans
    km = KMeans(n_clusters=k, random_state=0, n_init=30).fit(X)
    df["cluster"] = km.labels_

    # ---- choose which cluster is "IR-like" by majority vote on IR groups
    # (so labeling is stable across runs)
    ir_mask = df["group"].isin(["IR_pre", "IR_post"])
    ir_counts = df[ir_mask].groupby("cluster").size()
    ir_like_cluster = int(ir_counts.idxmax()) if len(ir_counts) > 0 else 0

    # ---- composition table
    comp = pd.crosstab(df["cluster"], df["group"])
    # reorder columns
    for col in ["AR_pre", "AR_post", "IR_pre", "IR_post"]:
        if col not in comp.columns:
            comp[col] = 0
    comp = comp[["AR_pre", "AR_post", "IR_pre", "IR_post"]]
    comp["total"] = comp.sum(axis=1)

    # ---- AR cluster-shift summary
    # match AR_pre and AR_post per sample
    ar_pre_cluster = df[df["group"] == "AR_pre"].set_index("sample")["cluster"]
    ar_post_cluster = df[df["group"] == "AR_post"].set_index("sample")["cluster"]
    common_ar = ar_pre_cluster.index.intersection(ar_post_cluster.index)
    moved_to_ir_like = ((ar_pre_cluster.loc[common_ar] != ir_like_cluster) &
                        (ar_post_cluster.loc[common_ar] == ir_like_cluster)).mean()

    # ---- plot
    fig, ax = plt.subplots(figsize=(6, 5))

    palette = { "AR_pre": "#FFEDA0","AR_post": "#FEB24C","IR_pre": "#D9F0D3","IR_post": "#5AAE61"}

    # marker shapes by cluster (k=2)
    markers = {0: "o", 1: "s", 2: "^", 3: "D"}

    # optional cluster ellipses (draw behind points)
    if draw_cluster_ellipses:
        for c in sorted(df["cluster"].unique()):
            pts = df[df["cluster"] == c][["PC1", "PC2"]].values
            _ellipse_from_points(ax, pts, nsig=ellipse_nsig, alpha=0.07, lw=2.0, zorder=1)

    # scatter points: color by group, shape by cluster
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        for g in ["IR_pre", "IR_post", "AR_pre", "AR_post"]:
            sub2 = sub[sub["group"] == g]
            if len(sub2) == 0:
                continue
            ax.scatter(
                sub2["PC1"], sub2["PC2"],
                s=65,
                marker=markers.get(int(c), "o"),
                alpha=0.9,
                label=f"{g} | C{c}" if (g == "IR_pre") else None,  # avoid huge legend
                color=palette[g],
                edgecolor="black",
                linewidth=0.6,
                zorder=3
            )

    # draw AR pre->post lines
    # (thin grey) to emphasize movement toward cluster
    pre_xy = df[df["group"] == "AR_pre"].set_index("sample")[["PC1", "PC2"]]
    post_xy = df[df["group"] == "AR_post"].set_index("sample")[["PC1", "PC2"]]
    common = pre_xy.index.intersection(post_xy.index)
    for s in common:
        x0, y0 = pre_xy.loc[s]
        x1, y1 = post_xy.loc[s]
        ax.plot([x0, x1], [y0, y1], color="grey", alpha=0.25, linewidth=1, zorder=2)

    #ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)

    # concise legend (manual)
    # show group colors
    handles = []
    labels = []
    for g in ["IR_pre", "IR_post", "AR_pre", "AR_post"]:
        h = ax.scatter([], [], color=palette[g], s=60, marker="o", edgecolor="black", linewidth=0.6)
        handles.append(h); labels.append(g)
    # show cluster shapes
    for c in sorted(df["cluster"].unique()):
        h = ax.scatter([], [], color="white", s=60, marker=markers.get(int(c), "o"),
                       edgecolor="black", linewidth=0.9)
        handles.append(h); labels.append(f"Cluster C{c}" + (" (IR-like)" if int(c)==ir_like_cluster else ""))
    ax.legend(handles, labels, frameon=False, loc="upper right", fontsize=9)

    # add composition text box
    # comp_lines = []
    # for c in comp.index:
    #     row = comp.loc[c]
    #     comp_lines.append(
    #         f"C{c}: AR_pre {int(row['AR_pre'])}, AR_post {int(row['AR_post'])}, "
    #         f"IR_pre {int(row['IR_pre'])}, IR_post {int(row['IR_post'])} (n={int(row['total'])})"
    #     )
    # txt = "Cluster composition\n" + "\n".join(comp_lines) + f"\nAR moved→IR-like: {moved_to_ir_like:.2f}"
    # ax.text(0.02, 0.98, txt, transform=ax.transAxes,
    #         ha="left", va="top", fontsize=9,
    #         bbox=dict(boxstyle="round", alpha=0.15))

    plt.tight_layout()
    sns.despine()
    plt.savefig("/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/class1_pca_kmeans_clusters.pdf", bbox_inches='tight', dpi=300)
    plt.show()

    return df, comp, ir_like_cluster, moved_to_ir_like


out = plot_ir_axis_main_figure(
    pre_TU=pre_TU_gene,
    post_TU=post_TU_gene,
    sampleinfo=sampleinfo,
    class1_list=class1,
    ARdutlist=ARdutlist,
    IRdutlist=IRdutlist,
    drop_nan_features=True,   # main figure 추천
    ellipse_nsig=2.0,         # 1.0이면 cluster 느낌 확
    ellipse_use_all_ir=True,  # IR_pre+post로 ellipse
)


df_pts, comp, ir_like_cluster, moved_frac = plot_unsupervised_clusters_on_pca(
    coord_pre=out["coord_pre"],
    coord_post=out["coord_post"],
    ar_samples=sampleinfo[sampleinfo["response"]==1].index,
    ir_samples=sampleinfo[sampleinfo["response"]==0].index,
    k=3,
    draw_cluster_ellipses=True,
    ellipse_nsig=1.2,
    title="PCA + k-means clusters (k=3)"
)

print(comp)
print("IR-like cluster:", ir_like_cluster, "AR moved fraction:", moved_frac)
# %%
cluster_counts = comp.copy()

cluster_counts.index.name = "cluster"
cluster_counts["total"] = cluster_counts.sum(axis=1)

# sns.set_style("ticks")

# def plot_cluster_composition_stacked(cluster_counts, save_path=None):
#     df = cluster_counts.copy()

#     # total column 있으면 제외
#     cols = ["AR_pre", "AR_post", "IR_pre", "IR_post"]
#     prop = df[cols].div(df[cols].sum(axis=1), axis=0)

#     colors = { "AR_pre": "#FFEDA0","AR_post": "#FEB24C","IR_pre": "#D9F0D3","IR_post": "#5AAE61"}

#     fig, ax = plt.subplots(figsize=(8, 3))  # 납작하게
#     left = np.zeros(len(prop))

#     ylabels = [f"C{idx}" if str(idx).isdigit() else str(idx) for idx in prop.index]

#     for col in cols:
#         ax.barh(
#             y=np.arange(len(prop)),
#             width=prop[col].values,
#             left=left,
#             color=colors[col],
#             edgecolor="white",
#             height=0.6,
#             label=col
#         )
#         left += prop[col].values

#     # bar 내부에 count 같이 쓰고 싶으면
#     for i, idx in enumerate(df.index):
#         cum = 0
#         for col in cols:
#             val = df.loc[idx, col]
#             frac = prop.loc[idx, col]
#             if frac > 0.08:  # 너무 작으면 글씨 생략
#                 ax.text(
#                     cum + frac / 2, i, str(val),
#                     ha="center", va="center",
#                     fontsize=8, color="black"
#                 )
#             cum += frac

#     ax.set_yticks(np.arange(len(prop)))
#     ax.set_yticklabels(ylabels)
#     ax.set_xlim(0, 1)
#     ax.set_xlabel("proportion")
#     ax.set_ylabel("")
#     ax.legend(
#         frameon=False, ncol=4, loc="upper center",
#         bbox_to_anchor=(0.5, 1.16), fontsize=10,
#         handlelength=1.2, columnspacing=1.0
#     )

#     sns.despine()
#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.show()

# plot_cluster_composition_stacked(cluster_counts, save_path="/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/class3_pca_kmeans_cluster_composition.pdf")

sns.set_style("ticks")

def plot_cluster_composition_stacked_vertical(cluster_counts, save_path=None):
    df = cluster_counts.copy()

    cols = ["AR_pre", "AR_post", "IR_pre", "IR_post"]
    prop = df[cols].div(df[cols].sum(axis=1), axis=0)
    order = [0, 2, 1]
    prop = prop.reindex(order)
    df = df.reindex(order)

    colors = {
        "AR_pre": "#FFEDA0",
        "AR_post": "#FEB24C",
        "IR_pre": "#D9F0D3",
        "IR_post": "#5AAE61"
    }

    fig, ax = plt.subplots(figsize=(3.5, 6))
    bottom = np.zeros(len(prop))

    x = np.arange(len(prop))
    xlabels = [f"C{idx}" if str(idx).isdigit() else str(idx) for idx in prop.index]

    for col in cols:
        ax.bar(
            x=x,
            height=prop[col].values,
            bottom=bottom,
            color=colors[col],
            edgecolor="white",
            width=0.65,
            label=col
        )
        bottom += prop[col].values

    for i, idx in enumerate(df.index):
        cum = 0
        for col in cols:
            val = df.loc[idx, col]
            frac = prop.loc[idx, col]
            if frac > 0.08:
                ax.text(
                    i, cum + frac / 2, str(val),
                    ha="center", va="center",
                    fontsize=8, color="black"
                )
            cum += frac

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Cluster")

    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10
    )

    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    
plot_cluster_composition_stacked_vertical(cluster_counts, save_path="/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/class1_pca_kmeans_cluster_composition.pdf")


#%%
#%%
####^^ PC1 boxplot  ###########
delta_TU_gene = filtered_trans.copy()
delta_TU_gene = delta_TU_gene[delta_TU_gene.index.str.split("-", n=1).str[-1].isin(proteincodinglist)]

pre_TU_gene = delta_TU_gene.iloc[:, 1::2]
post_TU_gene = delta_TU_gene.iloc[:, 0::2]

pre_TU_gene.columns = pre_TU_gene.columns.str[:-4] 
post_TU_gene.columns = post_TU_gene.columns.str[:-4] 

delta_TU_gene = post_TU_gene.copy()
delta_TU_gene.index = pre_TU_gene.index 
delta_TU_gene = post_TU_gene.values - pre_TU_gene.values
delta_TU_gene = pd.DataFrame(delta_TU_gene, index=post_TU_gene.index, columns=pre_TU_gene.columns)
delta_TU_gene = delta_TU_gene.dropna(axis=0) #dropna(), fillna(0)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

coord_pre = out["coord_pre"]
coord_post = out["coord_post"]

rows = []

for s in ar_samples:
    rows.append((s, "AR_pre", coord_pre.loc[s, "PC1"]))
    rows.append((s, "AR_post", coord_post.loc[s, "PC1"]))

for s in ir_samples:
    rows.append((s, "IR_pre", coord_pre.loc[s, "PC1"]))
    rows.append((s, "IR_post", coord_post.loc[s, "PC1"]))

pc1_df = pd.DataFrame(rows, columns=["sample", "Group", "PC1"])

order = ["AR_pre", "AR_post", "IR_pre", "IR_post"]
palette = {
    "AR_pre": "#FFEDA0",
    "AR_post": "#FEB24C",
    "IR_pre": "#D9F0D3",
    "IR_post": "#5AAE61"
}

plt.figure(figsize=(4,5))

ax = sns.boxplot(
    data=pc1_df,
    x="Group",
    y="PC1",
    order=order,
    palette=palette,
    boxprops=dict(alpha=0.9),
    fliersize=0,
    width=0.6
)

sns.stripplot(
    data=pc1_df,
    x="Group",
    y="PC1",
    order=order,
    color="black",
    size=3.5,
    alpha=0.65,
    jitter=0.15
)

plt.xticks(rotation=45)
plt.xlabel("")
plt.ylabel("PC1")
sns.despine()
plt.tight_layout()
plt.savefig("/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/class1_pca_PC1_boxplot.pdf", bbox_inches='tight', dpi=300)
plt.show()

#%%
##^^ PCA loading
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def get_pc1_loading_and_topgenes(
    pre_TU,
    post_TU,
    sampleinfo,
    class_list,
    ARdutlist,
    IRdutlist,
    response_col="response",
    ar_value=1,
    ir_value=0,
    eps=1e-3,
    use_zscore=True,
    top_n=100
):
    # ---------------------------------------------------------
    # same feature set logic as plot_ir_axis_main_figure
    # ---------------------------------------------------------
    dut_union = set(ARdutlist) | set(IRdutlist)
    feat = list(set(class_list) & dut_union & set(pre_TU.index) & set(post_TU.index))

    preX = pre_TU.loc[feat].copy()
    postX = post_TU.loc[feat].copy()

    common_samples = preX.columns.intersection(postX.columns)
    preX = preX[common_samples]
    postX = postX[common_samples]

    resp = sampleinfo[response_col]
    ar_samples = resp[resp == ar_value].index.intersection(common_samples)
    ir_samples = resp[resp == ir_value].index.intersection(common_samples)

    # NaN filtering
    ok = (~preX.isna().any(axis=1)) & (~postX.isna().any(axis=1))
    preX = preX.loc[ok]
    postX = postX.loc[ok]

    # same transform
    preT, postT = _prep_logit_z(preX, postX, ir_samples, eps=eps, use_zscore=use_zscore)

    # same PCA input
    cols_for_pca = list(ir_samples) + list(ar_samples)
    X_pre = preT[cols_for_pca].T
    X_post = postT[cols_for_pca].T
    X_all = pd.concat([X_pre, X_post], axis=0)

    pca = PCA(n_components=2, random_state=0)
    pca.fit(X_all.values)

    pc1_loading = pd.Series(
        pca.components_[0],
        index=preT.index,
        name="PC1_loading"
    ).sort_values(ascending=False)

    pc2_loading = pd.Series(
        pca.components_[1],
        index=preT.index,
        name="PC2_loading"
    ).sort_values(ascending=False)

    # transcript-gene format: ENST.....-GENE
    top_transcripts = pc1_loading.head(top_n).index.tolist()
    top_genes = pd.Index(top_transcripts).str.split("-").str[-1].unique().tolist()

    return {
        "pc1_loading": pc1_loading,
        "pc2_loading": pc2_loading,
        "top_transcripts": top_transcripts,
        "top_genes": top_genes,
        "n_features": len(preT.index)
    }
import gseapy as gp

def get_top_enrichment(gene_list, label, gene_sets=('GO_Biological_Process_2021', 'Reactome_2022')):
    enr = gp.enrichr(
        gene_list=gene_list,
        gene_sets=list(gene_sets),
        organism='Human',
        outdir=None
    )

    res = enr.results.copy()
    res = res.sort_values(by='Adjusted P-value')
    res['Term'] = res['Term'].astype(str).str.rsplit(" ", n=1).str[0]
    res['-log10(FDR)'] = -np.log10(res['Adjusted P-value'] + 1e-300)

    top5 = res.head(5).copy()
    top5['Group'] = label
    return top5
# class1
res_class1 = get_pc1_loading_and_topgenes(
    pre_TU=pre_TU_gene,
    post_TU=post_TU_gene,
    sampleinfo=sampleinfo,
    class_list=class1,
    ARdutlist=ARdutlist,
    IRdutlist=IRdutlist,
    top_n=100
)

# class3
res_class3 = get_pc1_loading_and_topgenes(
    pre_TU=pre_TU_gene,
    post_TU=post_TU_gene,
    sampleinfo=sampleinfo,
    class_list=class3,
    ARdutlist=ARdutlist,
    IRdutlist=IRdutlist,
    top_n=100
)

top_genes_class1 = res_class1["top_genes"]
top_genes_class3 = res_class3["top_genes"]

from matplotlib.patches import Patch

print("Class1 top genes:", len(top_genes_class1))
print("Class3 top genes:", len(top_genes_class3))
top5_class1 = get_top_enrichment(top_genes_class1, 'PC1 loading top100 (Class 1)')
top5_class3 = get_top_enrichment(top_genes_class3, 'PC1 loading top100 (Class 3)')

df_plot = pd.concat([top5_class1, top5_class3], ignore_index=True)

# 순서 정리: class1 top5 먼저, class3 top5 아래
term_order = top5_class1["Term"].tolist() + top5_class3["Term"].tolist()

plt.rcParams["font.family"] = "Arial"
fig = plt.figure(figsize=(8, 5.5))
ax = fig.add_axes([0.12, 0.12, 0.42, 0.78])
palette = ['#FF9616'] * len(top5_class1) + ['#1E9652'] * len(top5_class3)

sns.barplot(
    data=df_plot,
    x='-log10(FDR)',
    y='Term',
    order=term_order,
    palette=palette,
    ax=ax
)
legend_elements = [
    Patch(facecolor='#FF9616', label='PC1 loading top100 genes (Class 1)'),
    Patch(facecolor='#1E9652', label='PC1 loading top100 genes (Class 3)')
]

ax.yaxis.tick_right()
ax.yaxis.set_label_position("right")
for label in ax.get_yticklabels():
    label.set_horizontalalignment("left")

ax.set_xlabel('-log10(FDR)', fontsize=13)
ax.set_ylabel('')
ax.grid(axis='x', linestyle='--', alpha=0.5)
ax.set_xlim(0, df_plot['-log10(FDR)'].max() * 1.1)

# ax.legend(
#     handles=legend_elements,
#     #loc='upper left',
#     #bbox_to_anchor=(1.02, 1.0),
#     frameon=False,
#     fontsize=10,
#     handlelength=1.5
# )

sns.despine(right=True)
ax.spines["left"].set_visible(True)

plt.show()
#%%
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

score_pre = out["score_pre"]
score_post = out["score_post"]

# AR sample에서 IR-axis shift
delta_score_ar = (score_post - score_pre).loc[ar_samples]

# delta_TU_gene에서 AR sample만
common_ar = delta_TU_gene.columns.intersection(delta_score_ar.index)
delta_score_ar = delta_score_ar.loc[common_ar]

delta_tu_ar = delta_TU_gene.loc[:, common_ar].copy()

results = []
for gene in delta_tu_ar.index:
    x = delta_tu_ar.loc[gene].astype(float)
    y = delta_score_ar.astype(float)

    ok = x.notna() & y.notna()
    if ok.sum() < 5:
        continue

    rho, pval = spearmanr(x[ok], y[ok])
    results.append((gene, rho, pval))

corr_df = pd.DataFrame(results, columns=["feature", "rho", "pval"])
corr_df["neglog10p"] = -np.log10(corr_df["pval"] + 1e-300)
corr_df["sig"] = (corr_df["pval"] < 0.05) & (corr_df["rho"].abs() > 0.3)

plt.figure(figsize=(5, 4))

plt.scatter(
    corr_df["rho"],
    corr_df["neglog10p"],
    s=18,
    alpha=0.6,
    color="gray"
)

sig_df = corr_df[corr_df["sig"]]
plt.scatter(
    sig_df["rho"],
    sig_df["neglog10p"],
    s=22,
    alpha=0.9,
    color="#E66C5C"
)

plt.axvline(0, linestyle="--", color="gray", linewidth=1)
plt.axhline(-np.log10(0.05), linestyle="--", color="gray", linewidth=1)

plt.xlabel("Spearman rho\n(ΔTU vs IR-axis shift)")
plt.ylabel("-log10(p)")
plt.title("Gene-wise association with IR-direction shift")
sns.despine()
plt.tight_layout()
plt.show()

#%% ##^^ hierarchical clustering ######

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def make_union_feature_matrix_for_clustering(
    pre_TU,
    post_TU,
    sampleinfo,
    class1,
    class3,
    ARdutlist,
    IRdutlist,
    response_col="response",
    ar_value=1,
    ir_value=0,
    eps=1e-3,
    use_zscore=True,
    drop_nan_features=True
):
    # ---------------------------------------------------------
    # 1) feature = (class1 ∪ class3) ∩ (ARdut ∪ IRdut)
    # ---------------------------------------------------------
    feature_union = (set(class1) | set(class3)) & (set(ARdutlist) | set(IRdutlist))
    feat = list(feature_union & set(pre_TU.index) & set(post_TU.index))
    if len(feat) < 5:
        raise ValueError(f"Too few features after intersection: {len(feat)}")

    preX = pre_TU.loc[feat].copy()
    postX = post_TU.loc[feat].copy()

    # ---------------------------------------------------------
    # 2) align samples
    # ---------------------------------------------------------
    common_samples = preX.columns.intersection(postX.columns)
    preX = preX[common_samples]
    postX = postX[common_samples]

    resp = sampleinfo[response_col]
    ar_samples = resp[resp == ar_value].index.intersection(common_samples)
    ir_samples = resp[resp == ir_value].index.intersection(common_samples)

    # ---------------------------------------------------------
    # 3) NaN handling
    # ---------------------------------------------------------
    if drop_nan_features:
        ok = (~preX.isna().any(axis=1)) & (~postX.isna().any(axis=1))
        preX = preX.loc[ok]
        postX = postX.loc[ok]
    else:
        preX = preX.fillna(0.0)
        postX = postX.fillna(0.0)

    if preX.shape[0] < 5:
        raise ValueError(f"Too few features after NaN filtering: {preX.shape[0]}")

    # ---------------------------------------------------------
    # 4) PCA 때와 같은 transform
    # ---------------------------------------------------------
    preT, postT = _prep_logit_z(preX, postX, ir_samples, eps=eps, use_zscore=use_zscore)

    # ---------------------------------------------------------
    # 5) sample x feature matrix로 합치기
    #    row = sample_state, col = transcript-gene feature
    # ---------------------------------------------------------
    X_pre = preT.T.copy()
    X_post = postT.T.copy()

    X_pre.index = [f"{s}|pre" for s in X_pre.index]
    X_post.index = [f"{s}|post" for s in X_post.index]

    X_all = pd.concat([X_pre, X_post], axis=0)

    # ---------------------------------------------------------
    # 6) sample annotation
    # ---------------------------------------------------------
    group_labels = []
    for idx in X_all.index:
        s, tp = idx.rsplit("|", 1)
        r = sampleinfo.loc[s, response_col]
        if r == ar_value and tp == "pre":
            group_labels.append("AR_pre")
        elif r == ar_value and tp == "post":
            group_labels.append("AR_post")
        elif r == ir_value and tp == "pre":
            group_labels.append("IR_pre")
        elif r == ir_value and tp == "post":
            group_labels.append("IR_post")
        else:
            group_labels.append("Unknown")

    annot = pd.DataFrame(index=X_all.index)
    annot["Group"] = group_labels

    palette = {
        "AR_pre": "#FFEDA0",
        "AR_post": "#FEB24C",
        "IR_pre": "#D9F0D3",
        "IR_post": "#5AAE61"
    }
    col_colors = annot["Group"].map(palette)

    return X_all, annot, col_colors

def plot_hclust_union_heatmap(
    pre_TU,
    post_TU,
    sampleinfo,
    class1,
    class3,
    ARdutlist,
    IRdutlist,
    figsize=(10, 12),
    cmap="vlag",
    z_score_rows=False,
    standard_scale_rows=False,
    save_path=None
):
    X_all, annot, col_colors = make_union_feature_matrix_for_clustering(
        pre_TU=pre_TU,
        post_TU=post_TU,
        sampleinfo=sampleinfo,
        class1=class1,
        class3=class3,
        ARdutlist=ARdutlist,
        IRdutlist=IRdutlist
    )
    print(X_all.shape)

    # heatmap matrix: rows=features, cols=samples
    mat = X_all.T.copy()

    # optional row scaling
    if z_score_rows:
        mat = mat.sub(mat.mean(axis=1), axis=0).div(mat.std(axis=1).replace(0, np.nan), axis=0).fillna(0)
    elif standard_scale_rows:
        mat = mat.sub(mat.min(axis=1), axis=0).div((mat.max(axis=1) - mat.min(axis=1)).replace(0, np.nan), axis=0).fillna(0)

    sns.set_style("white")

    g = sns.clustermap(
        mat,
        cmap=cmap,
        col_colors=col_colors,
        figsize=figsize,
        xticklabels=False,
        yticklabels=False,
        method="ward",
        #metric="euclidean",
        dendrogram_ratio=(0.12, 0.12),
        cbar_pos=(0.02, 0.82, 0.03, 0.12)
    )

    # legend
    palette = {
        "AR_pre": "#FFEDA0",
        "AR_post": "#FEB24C",
        "IR_pre": "#D9F0D3",
        "IR_post": "#5AAE61"
    }
    for label, color in palette.items():
        g.ax_col_dendrogram.bar(0, 0, color=color, label=label, linewidth=0)

    g.ax_col_dendrogram.legend(
        loc="center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.15),
        frameon=False
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return g, X_all, annot

g, X_all, annot = plot_hclust_union_heatmap(
    pre_TU=pre_TU_gene,
    post_TU=post_TU_gene,
    sampleinfo=sampleinfo,
    class1=class1,
    class3=class1,
    ARdutlist=ARdutlist,
    IRdutlist=ARdutlist,
    figsize=(10, 12),
    cmap="vlag",
    z_score_rows=False,   # PCA와 비슷하게 가려면 False 추천
    save_path="/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/class13_union_ARIR_hclust.pdf"
)

#%% ##^^ AR interval analysis #############

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

out = plot_ir_axis_main_figure(
    pre_TU=pre_TU_gene,
    post_TU=post_TU_gene,
    sampleinfo=sampleinfo,
    class1_list=class1,
    ARdutlist=ARdutlist,
    IRdutlist=IRdutlist,
    drop_nan_features=True,   # main figure 추천
    ellipse_nsig=2.0,         # 1.0이면 cluster 느낌 확
    ellipse_use_all_ir=True,  # IR_pre+post로 ellipse
)

# AR samples
ar_samples = sampleinfo[sampleinfo["response"] == 1].index

# interval
interval_ar = sampleinfo.loc[ar_samples, "interval"].astype(float)

# IR-axis shift
delta_ar = out["delta_ar"].copy()
common_ar = delta_ar.index.intersection(interval_ar.index)
delta_ar = delta_ar.loc[common_ar]
interval_ar = interval_ar.loc[common_ar]

# PCA distance
coord_pre = out["coord_pre"]
coord_post = out["coord_post"]

common_ar2 = (
    ar_samples
    .intersection(coord_pre.index)
    .intersection(coord_post.index)
    .intersection(sampleinfo.index)
)

pca_dist = pd.Series(index=common_ar2, dtype=float)
pc1_shift = pd.Series(index=common_ar2, dtype=float)
pc2_shift = pd.Series(index=common_ar2, dtype=float)

for s in common_ar2:
    dx = coord_post.loc[s, "PC1"] - coord_pre.loc[s, "PC1"]
    dy = coord_post.loc[s, "PC2"] - coord_pre.loc[s, "PC2"]
    pca_dist.loc[s] = np.sqrt(dx**2 + dy**2)
    pc1_shift.loc[s] = dx
    pc2_shift.loc[s] = dy

plot_df = pd.DataFrame({
    "interval": sampleinfo.loc[common_ar2, "interval"].astype(float),
    "delta_ir_score": delta_ar.reindex(common_ar2),
    "pca_distance": pca_dist,
    "pc1_shift": pc1_shift,
    "pc2_shift": pc2_shift
}).dropna()

plot_df.head()
rho1, p1 = spearmanr(plot_df["interval"], plot_df["delta_ir_score"])

plt.figure(figsize=(3, 4))
ax = sns.regplot(
    data=plot_df,
    x="interval",
    y="delta_ir_score",
    scatter_kws={"s": 45, "alpha": 0.8},
    line_kws={"linewidth": 1.5}, color="#077010"
)

ax.text(
    0.03, 0.97,
    f"r={rho1:.2f}\np={p1:.2e}",
    transform=ax.transAxes,
    ha="left", va="top",
    #bbox=dict(boxstyle="round", alpha=0.12)
)

plt.xlabel("Treatment interval")
plt.ylabel("Δ IR-axis score")
sns.despine()
plt.tight_layout()
plt.show()

rho2, p2 = spearmanr(plot_df["interval"], plot_df["pc1_shift"])

plt.figure(figsize=(3, 4))
ax = sns.regplot(
    data=plot_df,
    x="interval",
    y="pc1_shift",
    scatter_kws={"s": 45, "alpha": 0.8},
    line_kws={"linewidth": 1.5}, color="#077010"
)

ax.text(
    0.03, 0.97,
    f"r={rho2:.2f}\np={p2:.2e}",
    transform=ax.transAxes,
    ha="left", va="top",
    #bbox=dict(boxstyle="round", alpha=0.12)
)

plt.xlabel("Treatment interval")
plt.ylabel("PCA shift distance")
sns.despine()
plt.tight_layout()
plt.show()

ar_post_cluster = df_pts[df_pts["group"] == "AR_post"].set_index("sample")["cluster"]
ar_pre_cluster = df_pts[df_pts["group"] == "AR_pre"].set_index("sample")["cluster"]

common_ar3 = ar_post_cluster.index.intersection(ar_pre_cluster.index).intersection(sampleinfo.index)

cluster_df = pd.DataFrame({
    "interval": sampleinfo.loc[common_ar3, "interval"].astype(float),
    "AR_pre_cluster": ar_pre_cluster.loc[common_ar3],
    "AR_post_cluster": ar_post_cluster.loc[common_ar3]
})

cluster_df["Moved_to_IR_like"] = (
    (cluster_df["AR_pre_cluster"] != ir_like_cluster) &
    (cluster_df["AR_post_cluster"] == ir_like_cluster)
)

x = cluster_df.loc[~cluster_df["Moved_to_IR_like"], "interval"]
y = cluster_df.loc[cluster_df["Moved_to_IR_like"], "interval"]

if len(x) > 0 and len(y) > 0:
    stat, p3 = mannwhitneyu(x, y, alternative="two-sided")
else:
    p3 = np.nan

plt.figure(figsize=(4, 4))
ax = sns.boxplot(
    data=cluster_df,
    x="Moved_to_IR_like",
    y="interval",
    order=[False, True],
    boxprops=dict(facecolor="none")
)
sns.stripplot(
    data=cluster_df,
    x="Moved_to_IR_like",
    y="interval",
    order=[False, True],
    color="black",
    alpha=0.7,
    size=4,
    jitter=0.15
)

ax.set_xticklabels(["No", "Yes"])
ax.set_xlabel("AR_post moved to IR-like cluster")
ax.set_ylabel("Treatment interval")
ax.set_title(f"M.W.W. p = {p3:.2e}" if pd.notna(p3) else "Interval by IR-like transition")
sns.despine()
plt.tight_layout()
plt.show()


import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# AR samples
ar_samples = sampleinfo[sampleinfo["response"] == 1].index

common = (
    ar_samples
    .intersection(pre_TU_gene.columns)
    .intersection(post_TU_gene.columns)
)

# delta TU
delta_TU = (post_TU_gene - pre_TU_gene)[common]

# class gene sets
class1_genes = list(set(class1) & set(delta_TU.index))
class3_genes = list(set(class3) & set(delta_TU.index))

# class-level mean change
class1_delta = delta_TU.loc[class1_genes].mean(axis=0)
class3_delta = delta_TU.loc[class3_genes].mean(axis=0)

# interval
interval = sampleinfo.loc[common, "interval"].astype(float)
rho1, p1 = spearmanr(interval, class1_delta)
rho3, p3 = spearmanr(interval, class3_delta)

print("Class1 ΔTU vs interval:", rho1, p1)
print("Class3 ΔTU vs interval:", rho3, p3)

import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(7, 4))

# class1
sns.regplot(
    x=interval,
    y=class1_delta,
    ax=axes[0],
    scatter_kws={"s": 45},
    line_kws={"linewidth": 1.5}, color="#077010"
)
axes[0].set_xlabel("Treatment interval")
axes[0].set_ylabel("ΔTU (Class1)")

# class3
sns.regplot(
    x=interval,
    y=class3_delta,
    ax=axes[1],
    scatter_kws={"s": 45},
    line_kws={"linewidth": 1.5}, color="#077010"
)
axes[1].set_xlabel("Treatment interval")
axes[1].set_ylabel("ΔTU (Class3)")

sns.despine()
plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# -----------------------------
# 1. AR sample 정의
# -----------------------------
ar_samples = sampleinfo[sampleinfo["response"] == 1].index

# -----------------------------
# 2. 공통 샘플 맞추기
# -----------------------------
common = (
    ar_samples
    .intersection(pre_TU_gene.columns)
    .intersection(post_TU_gene.columns)
    .intersection(out["coord_pre"].index)
    .intersection(out["coord_post"].index)
)

# -----------------------------
# 3. Class1 TU (AR_pre)
# -----------------------------
class1_genes = list(set(class1) & set(pre_TU_gene.index))
class1_pre = pre_TU_gene.loc[class1_genes, common].mean(axis=0)

# -----------------------------
# 4. PC1 이동량
# -----------------------------
coord_pre = out["coord_pre"]
coord_post = out["coord_post"]

pc1_shift = pd.Series(index=common, dtype=float)

for s in common:
    pc1_shift.loc[s] = coord_post.loc[s, "PC1"] - coord_pre.loc[s, "PC1"]

# -----------------------------
# 5. DataFrame 정리
# -----------------------------
df = pd.DataFrame({
    "class1_pre": class1_pre,
    "pc1_shift": pc1_shift
}).dropna()

# -----------------------------
# 6. correlation
# -----------------------------
rho, p = spearmanr(df["class1_pre"], df["pc1_shift"])

print(f"Spearman rho={rho:.2f}, p={p:.2e}")

plt.figure(figsize=(4.2, 4.2))

ax = sns.regplot(
    data=df,
    x="class1_pre",
    y="pc1_shift",
    scatter_kws={"s": 50, "alpha": 0.85},
    line_kws={"linewidth": 1.6}, color="#077010"
)

ax.text(
    0.63, 0.9,
    f"r={rho:.2f}\np={p:.2e}",
    transform=ax.transAxes,
    ha="left", va="top",
    #bbox=dict(boxstyle="round", alpha=0.12)
)

plt.xlabel("Class1 DUT (AR pre)")
plt.ylabel("PC1 shift")
sns.despine()
plt.tight_layout()
plt.show()
#%%
##^^ psi PCA ######

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

diff_SF_genes = ['ALYREF','DDX23','DDX1','PPM1G','MYEF2','RBM47','EIF3A','SNRPB','ZNF207', ] # 'INTS4','CELF5'
geneexp = pd.read_csv('/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_80_gene_TPM.txt', sep='\t', index_col=0)

raw_geneexp = pd.read_csv(
    '/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_80_gene_TPM.txt',
    sep='\t',
    index_col=0
)

# pre / post 분리
pre_gene = raw_geneexp.loc[:, raw_geneexp.columns.str.endswith('-bfD')].copy()
post_gene = raw_geneexp.loc[:, raw_geneexp.columns.str.endswith('-atD')].copy()

# sample 이름 통일
pre_gene.columns = pre_gene.columns.str.replace('-bfD', '', regex=False)
post_gene.columns = post_gene.columns.str.replace('-atD', '', regex=False)

# 공통 샘플만 유지 + 정렬
common_samples = pre_gene.columns.intersection(post_gene.columns)
pre_gene = pre_gene[common_samples].sort_index(axis=1)
post_gene = post_gene[common_samples].sort_index(axis=1)

# sampleinfo와 맞추기
sampleinfo2 = sampleinfo.copy()
sampleinfo2 = sampleinfo2.loc[sampleinfo2.index.intersection(common_samples)].copy()

ar_samples = sampleinfo2[sampleinfo2["response"] == 1].index
ir_samples = sampleinfo2[sampleinfo2["response"] == 0].index

print("AR n =", len(ar_samples), "IR n =", len(ir_samples))
print(pre_gene.shape, post_gene.shape)

from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse

sns.set_style("ticks")

PALETTE_4GROUP = {
    "AR_pre": "#FFEDA0",
    "AR_post": "#FEB24C",
    "IR_pre": "#D9F0D3",
    "IR_post": "#5AAE61",
}

CLUSTER_MARKERS = {
    0: "o",
    1: "s",
    2: "^",
    3: "D",
    4: "P"
}


def _ellipse_from_points(ax, pts2d, nsig=1.3, alpha=0.10, lw=2.5, zorder=5):
    if pts2d.shape[0] < 5:
        return None

    center = pts2d.mean(axis=0)
    cov = np.cov(pts2d.T)

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * nsig * np.sqrt(np.maximum(vals, 1e-12))

    e = Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        facecolor="black",
        alpha=alpha,
        edgecolor="black",
        linewidth=lw,
        zorder=zorder
    )
    ax.add_patch(e)
    return e


def run_sf_pca_kmeans(
    pre_gene,
    post_gene,
    sampleinfo,
    feature_genes,
    k=3,
    log_transform=True,
    zscore_by_ir_pre=True,
    title="SF gene expression PCA + kmeans",
    save_prefix=None,
    ellipse_nsig=1.2
):
    # --------------------------------------------------
    # 1) feature selection
    # --------------------------------------------------
    feat = list(set(feature_genes) & set(pre_gene.index) & set(post_gene.index))
    feat = sorted(feat)

    if len(feat) < 3:
        raise ValueError(f"Too few genes after filtering: {len(feat)}")

    preX = pre_gene.loc[feat].copy()
    postX = post_gene.loc[feat].copy()

    # --------------------------------------------------
    # 2) align samples
    # --------------------------------------------------
    common_samples = preX.columns.intersection(postX.columns)
    common_samples = common_samples.intersection(sampleinfo.index)

    preX = preX[common_samples]
    postX = postX[common_samples]
    sampleinfo = sampleinfo.loc[common_samples].copy()

    ar_samples = sampleinfo[sampleinfo["response"] == 1].index
    ir_samples = sampleinfo[sampleinfo["response"] == 0].index

    if len(ar_samples) < 3 or len(ir_samples) < 3:
        raise ValueError(f"Not enough samples: AR={len(ar_samples)}, IR={len(ir_samples)}")

    # --------------------------------------------------
    # 3) transform
    # --------------------------------------------------
    if log_transform:
        preT = np.log1p(preX)
        postT = np.log1p(postX)
    else:
        preT = preX.copy()
        postT = postX.copy()

    if zscore_by_ir_pre:
        mu = preT[ir_samples].mean(axis=1)
        sd = preT[ir_samples].std(axis=1).replace(0, np.nan)

        preT = preT.sub(mu, axis=0).div(sd, axis=0).fillna(0.0)
        postT = postT.sub(mu, axis=0).div(sd, axis=0).fillna(0.0)

    # --------------------------------------------------
    # 4) IR-axis score
    # --------------------------------------------------
    v = preT[ir_samples].mean(axis=1) - preT[ar_samples].mean(axis=1)
    v = v / (np.linalg.norm(v.values) + 1e-12)

    score_pre = preT.T.dot(v)
    score_post = postT.T.dot(v)

    delta_ar = score_post.loc[ar_samples] - score_pre.loc[ar_samples]
    try:
        _, p_delta = wilcoxon(delta_ar.values, alternative="greater")
    except ValueError:
        p_delta = np.nan

    frac_pos = float((delta_ar > 0).mean())

    # --------------------------------------------------
    # 5) PCA
    # --------------------------------------------------
    cols_for_pca = list(ir_samples) + list(ar_samples)

    X_pre = preT[cols_for_pca].T
    X_post = postT[cols_for_pca].T
    X_all = pd.concat([X_pre, X_post], axis=0)

    pca = PCA(n_components=2, random_state=0)
    Z_all = pca.fit_transform(X_all.values)

    n_pre = X_pre.shape[0]
    coord_pre = pd.DataFrame(Z_all[:n_pre, :], index=X_pre.index, columns=["PC1", "PC2"])
    coord_post = pd.DataFrame(Z_all[n_pre:, :], index=X_post.index, columns=["PC1", "PC2"])

    # --------------------------------------------------
    # 6) build point table for clustering
    # --------------------------------------------------
    rows = []

    for s in ir_samples:
        if s in coord_pre.index:
            rows.append((s, "IR_pre", coord_pre.loc[s, "PC1"], coord_pre.loc[s, "PC2"]))
        if s in coord_post.index:
            rows.append((s, "IR_post", coord_post.loc[s, "PC1"], coord_post.loc[s, "PC2"]))

    for s in ar_samples:
        if s in coord_pre.index:
            rows.append((s, "AR_pre", coord_pre.loc[s, "PC1"], coord_pre.loc[s, "PC2"]))
        if s in coord_post.index:
            rows.append((s, "AR_post", coord_post.loc[s, "PC1"], coord_post.loc[s, "PC2"]))

    df_pts = pd.DataFrame(rows, columns=["sample", "group", "PC1", "PC2"])

    # --------------------------------------------------
    # 7) kmeans
    # --------------------------------------------------
    km = KMeans(n_clusters=k, random_state=0, n_init=30)
    df_pts["cluster"] = km.fit_predict(df_pts[["PC1", "PC2"]].values)

    # IR-like cluster
    ir_mask = df_pts["group"].isin(["IR_pre", "IR_post"])
    ir_counts = df_pts[ir_mask].groupby("cluster").size()
    ir_like_cluster = int(ir_counts.idxmax()) if len(ir_counts) > 0 else 0

    # composition
    comp = pd.crosstab(df_pts["cluster"], df_pts["group"])
    for col in ["AR_pre", "AR_post", "IR_pre", "IR_post"]:
        if col not in comp.columns:
            comp[col] = 0
    comp = comp[["AR_pre", "AR_post", "IR_pre", "IR_post"]]
    comp["total"] = comp.sum(axis=1)

    # AR pre -> post, moved to IR-like cluster?
    ar_pre_cluster = df_pts[df_pts["group"] == "AR_pre"].set_index("sample")["cluster"]
    ar_post_cluster = df_pts[df_pts["group"] == "AR_post"].set_index("sample")["cluster"]
    common_ar = ar_pre_cluster.index.intersection(ar_post_cluster.index)

    moved_to_ir_like = (
        (ar_pre_cluster.loc[common_ar] != ir_like_cluster) &
        (ar_post_cluster.loc[common_ar] == ir_like_cluster)
    ).mean()

    # --------------------------------------------------
    # 8) Plot 1: PCA + kmeans clusters
    # --------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))

    for c in sorted(df_pts["cluster"].unique()):
        pts = df_pts[df_pts["cluster"] == c][["PC1", "PC2"]].values
        _ellipse_from_points(ax, pts, nsig=ellipse_nsig, alpha=0.07, lw=2.0, zorder=1)

    # scatter with same color code
    for c in sorted(df_pts["cluster"].unique()):
        sub = df_pts[df_pts["cluster"] == c]
        for g in ["IR_pre", "IR_post", "AR_pre", "AR_post"]:
            sub2 = sub[sub["group"] == g]
            if len(sub2) == 0:
                continue
            ax.scatter(
                sub2["PC1"],
                sub2["PC2"],
                s=65,
                marker=CLUSTER_MARKERS.get(int(c), "o"),
                alpha=0.9,
                color=PALETTE_4GROUP[g],
                edgecolor="black",
                linewidth=0.6,
                zorder=3
            )

    # AR pre->post lines
    pre_xy = df_pts[df_pts["group"] == "AR_pre"].set_index("sample")[["PC1", "PC2"]]
    post_xy = df_pts[df_pts["group"] == "AR_post"].set_index("sample")[["PC1", "PC2"]]
    common = pre_xy.index.intersection(post_xy.index)
    for s in common:
        x0, y0 = pre_xy.loc[s]
        x1, y1 = post_xy.loc[s]
        ax.plot([x0, x1], [y0, y1], color="grey", alpha=0.25, linewidth=1, zorder=2)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)

    handles = []
    labels = []

    for g in ["IR_pre", "IR_post", "AR_pre", "AR_post"]:
        h = ax.scatter([], [], color=PALETTE_4GROUP[g], s=60, marker="o",
                       edgecolor="black", linewidth=0.6)
        handles.append(h)
        labels.append(g)

    for c in sorted(df_pts["cluster"].unique()):
        h = ax.scatter([], [], color="white", s=60,
                       marker=CLUSTER_MARKERS.get(int(c), "o"),
                       edgecolor="black", linewidth=0.9)
        handles.append(h)
        labels.append(f"Cluster C{c}" + (" (IR-like)" if int(c) == ir_like_cluster else ""))

    ax.legend(handles, labels, frameon=False, loc="upper right", fontsize=9)
    ax.set_title(title)

    txt = (
        f"n_genes={len(feat)}\n"
        f"AR n={len(ar_samples)}, IR n={len(ir_samples)}\n"
        f"Δ>0 fraction={frac_pos:.2f}\n"
        f"Wilcoxon Δ>0 p={p_delta:.2e}\n"
        f"AR moved→IR-like={moved_to_ir_like:.2f}"
    )
    # ax.text(
    #     0.02, 0.98, txt,
    #     transform=ax.transAxes,
    #     ha="left", va="top",
    #     fontsize=9,
    #     bbox=dict(boxstyle="round", alpha=0.15)
    # )

    sns.despine()
    plt.tight_layout()

    if save_prefix is not None:
        plt.savefig(f"{save_prefix}_pca_kmeans.pdf", dpi=300, bbox_inches="tight")
    plt.show()

    # --------------------------------------------------
    # 9) Plot 2: cluster composition stacked bar
    # --------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(6, 2))

    cols = ["AR_pre", "AR_post", "IR_pre", "IR_post"]
    prop = comp[cols].div(comp[cols].sum(axis=1), axis=0)

    left = np.zeros(len(prop))
    ylabels = [f"C{idx}" for idx in prop.index]

    for col in cols:
        ax2.barh(
            y=np.arange(len(prop)),
            width=prop[col].values,
            left=left,
            color=PALETTE_4GROUP[col],
            edgecolor="white",
            height=0.6,
            label=col
        )
        left += prop[col].values

    for i, idx in enumerate(comp.index):
        cum = 0
        for col in cols:
            val = comp.loc[idx, col]
            frac = prop.loc[idx, col]
            if frac > 0.08:
                ax2.text(
                    cum + frac / 2,
                    i,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black"
                )
            cum += frac

    ax2.set_yticks(np.arange(len(prop)))
    ax2.set_yticklabels(ylabels)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("proportion")
    ax2.set_ylabel("")
    ax2.legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.3),
        fontsize=10,
        handlelength=1.2,
        columnspacing=1.0
    )

    sns.despine()
    plt.tight_layout()

    if save_prefix is not None:
        plt.savefig(f"{save_prefix}_cluster_composition.pdf", dpi=300, bbox_inches="tight")
    plt.show()

    return {
        "coord_pre": coord_pre,
        "coord_post": coord_post,
        "df_pts": df_pts,
        "comp": comp,
        "ir_like_cluster": ir_like_cluster,
        "moved_to_ir_like": moved_to_ir_like,
        "score_pre": score_pre,
        "score_post": score_post,
        "delta_ar": delta_ar,
        "p_delta": p_delta,
        "frac_pos": frac_pos,
        "features_used": feat,
        "pca_model": pca,
    }

def filter_genes_by_expression(
    pre_gene,
    post_gene,
    feature_genes,
    min_expr=1.0,
    min_prop=0.2,
    use_both_pre_post=True
):
    """
    Keep genes that are expressed (>= min_expr) in at least min_prop fraction
    of all samples.

    Parameters
    ----------
    pre_gene, post_gene : pd.DataFrame
        rows = genes, cols = samples
    feature_genes : list-like
        candidate genes to filter
    min_expr : float
        expression threshold
    min_prop : float
        minimum fraction of samples passing min_expr
        e.g. 0.2 means >=20% of all samples
    use_both_pre_post : bool
        if True, use all pre+post samples together
        if False, use only pre samples

    Returns
    -------
    kept_genes : list
    expr_summary : pd.DataFrame
        summary table for each candidate gene
    """
    feat = sorted(set(feature_genes) & set(pre_gene.index) & set(post_gene.index))

    if len(feat) == 0:
        return [], pd.DataFrame(columns=["n_pass", "n_total", "prop_pass"])

    if use_both_pre_post:
        expr_mat = pd.concat(
            [pre_gene.loc[feat], post_gene.loc[feat]],
            axis=1
        )
    else:
        expr_mat = pre_gene.loc[feat].copy()

    pass_mat = expr_mat >= min_expr
    n_pass = pass_mat.sum(axis=1)
    n_total = expr_mat.shape[1]
    prop_pass = n_pass / n_total

    expr_summary = pd.DataFrame({
        "n_pass": n_pass,
        "n_total": n_total,
        "prop_pass": prop_pass
    }).sort_values("prop_pass", ascending=False)

    kept_genes = expr_summary.index[expr_summary["prop_pass"] >= min_prop].tolist()
    return kept_genes, expr_summary

def filter_genes_by_variance(pre_gene, post_gene, genes, top_k=500):
    expr = pd.concat([pre_gene.loc[genes], post_gene.loc[genes]], axis=1)
    
    var = expr.var(axis=1)
    var = var.sort_values(ascending=False)
    
    return var.index[:top_k].tolist(), var

# sf_all_res = run_sf_pca_kmeans(
#     pre_gene=pre_gene,
#     post_gene=post_gene,
#     sampleinfo=sampleinfo2,
#     feature_genes=SF_genes,
#     k=3,
#     log_transform=True,
#     zscore_by_ir_pre=True,
#     title="SF genes (all) PCA + k-means clusters (k=3)",
#     #save_prefix="/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/SF_all"
# )

# print(sf_all_res["comp"])
# print("IR-like cluster:", sf_all_res["ir_like_cluster"])
# print("AR moved fraction:", sf_all_res["moved_to_ir_like"])

# 예시: 전체 pre+post 샘플 중 30% 이상에서 expression >= 1 인 유전자만 사용

# pre 상태에서 AR vs IR 차이 큰 gene
from scipy.stats import ttest_ind

pvals = []
for g in SF_genes:
    if g in pre_gene.index.to_list():
        x = pre_gene.loc[g, ar_samples]
        y = pre_gene.loc[g, ir_samples]
        _, p = ttest_ind(x, y, equal_var=False)
        pvals.append((g, p))

genes_sorted = sorted(pvals, key=lambda x: x[1])
top_genes = [g for g, _ in genes_sorted[:50]]


SF_genes_filtered_expr, _ = filter_genes_by_expression(
    pre_gene, post_gene, SF_genes,
    min_expr=1.0,
    min_prop=0.4
)

SF_genes_final, var_series = filter_genes_by_variance(
    pre_gene, post_gene,
    SF_genes_filtered_expr,
    top_k=50   # 200~1000 사이에서 튜닝
)

print("Original SF genes:", len(set(SF_genes)))
print("Filtered SF genes:", len(SF_genes_final))

sf_all_res = run_sf_pca_kmeans(
    pre_gene=pre_gene,
    post_gene=post_gene,
    sampleinfo=sampleinfo2,
    feature_genes=top_genes,
    k=2,
    log_transform=True,
    zscore_by_ir_pre=True,
    title="",
    # save_prefix="/home/jiye/..."
)

print(sf_all_res["comp"])
print("IR-like cluster:", sf_all_res["ir_like_cluster"])
print("AR moved fraction:", sf_all_res["moved_to_ir_like"])

# diff_SF_genes_uniq = sorted(set(diff_SF_genes))

# sf_diff_res = run_sf_pca_kmeans(
#     pre_gene=pre_gene,
#     post_gene=post_gene,
#     sampleinfo=sampleinfo2,
#     feature_genes=diff_SF_genes_uniq,
#     k=3,
#     log_transform=True,
#     zscore_by_ir_pre=True,
#     title="Differential SF genes PCA + k-means clusters (k=3)",
#     #save_prefix="/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/SF_diff"
# )

# print(sf_diff_res["comp"])
# print("IR-like cluster:", sf_diff_res["ir_like_cluster"])
# print("AR moved fraction:", sf_diff_res["moved_to_ir_like"])

#%%
####^^^^ Class 1 /3 ~ SF gene #################

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("ticks")


##################################################
# 1. TU rewiring score: class1 DUT / class3 DUT
##################################################

def compute_class_dut_tu_scores(
    pre_TU_gene,
    post_TU_gene,
    sampleinfo,
    class1_list,
    class3_list,
    dut_list=None,
    response_col="response"
):
    """
    Compute per-sample TU change scores using ALL DUT features in each class.

    Parameters
    ----------
    pre_TU_gene, post_TU_gene : DataFrame
        rows = transcript IDs, cols = sample IDs

    class1_list, class3_list : list-like
        transcript IDs for class1 / class3

    dut_list : list-like or None
        if provided, restrict to DUT transcripts only
        if None, use all transcripts in each class

    Returns
    -------
    out_df : DataFrame
        index = sample
        columns:
            response
            class1_delta
            class3_delta
            total_tu_score   (= class1_delta - class3_delta)

    feat_info : dict
        actual features used
    """

    common_samples = (
        pre_TU_gene.columns
        .intersection(post_TU_gene.columns)
        .intersection(sampleinfo.index)
    )

    preX = pre_TU_gene.loc[:, common_samples].copy()
    postX = post_TU_gene.loc[:, common_samples].copy()
    sampleinfo_sub = sampleinfo.loc[common_samples].copy()

    available_feats = set(preX.index) & set(postX.index)

    class1_set = set(class1_list)
    class3_set = set(class3_list)

    if dut_list is not None:
        dut_set = set(dut_list)
        class1_feats = sorted(class1_set & dut_set & available_feats)
        class3_feats = sorted(class3_set & dut_set & available_feats)
    else:
        class1_feats = sorted(class1_set & available_feats)
        class3_feats = sorted(class3_set & available_feats)

    if len(class1_feats) < 3:
        raise ValueError(f"Too few class1 DUT features: {len(class1_feats)}")
    if len(class3_feats) < 3:
        raise ValueError(f"Too few class3 DUT features: {len(class3_feats)}")

    dX = postX - preX

    class1_delta = dX.loc[class1_feats].mean(axis=0)
    class3_delta = dX.loc[class3_feats].mean(axis=0)

    # desired direction:
    # class1 increase = positive
    # class3 decrease = positive contribution after subtraction
    total_tu_score = class1_delta - class3_delta

    out_df = pd.DataFrame(index=common_samples)
    out_df["response"] = sampleinfo_sub.loc[common_samples, response_col].values
    out_df["class1_delta"] = class1_delta.loc[common_samples].values
    out_df["class3_delta"] = class3_delta.loc[common_samples].values
    out_df["total_tu_score"] = total_tu_score.loc[common_samples].values

    feat_info = {
        "class1_feats": class1_feats,
        "class3_feats": class3_feats
    }

    return out_df, feat_info


##################################################
# 2. Merge with SF score
##################################################

def merge_sf_and_tu_scores(sf_score_df, tu_score_df):
    """
    sf_score_df should contain:
      - response
      - SF_pre
      - SF_post
      - delta_SF

    tu_score_df should contain:
      - response
      - class1_delta
      - class3_delta
      - total_tu_score
    """
    tu_only = tu_score_df.drop(columns=["response"], errors="ignore")
    merged = sf_score_df.join(tu_only, how="inner")
    return merged


##################################################
# 3. Generic correlation scatter
##################################################

from scipy.stats import spearmanr

def plot_one_correlation(
    df,
    x_col,
    y_col,
    only_ar=True,
    save_path=None,
    title=None,
    add_regline=True
):
    sub = df.copy()

    fig, ax = plt.subplots(figsize=(5, 4.5))

    if only_ar:
        sub = sub[sub["response"] == 1].copy()
        point_color = "#D97706"

        ax.scatter(sub[x_col], sub[y_col], s=44, alpha=0.85, color=point_color)

        if add_regline:
            sns.regplot(
                data=sub, x=x_col, y=y_col,
                scatter=False, ci=None,
                line_kws={"linewidth": 1.2, "alpha": 0.8},
                ax=ax
            )

        rho, p = spearmanr(sub[x_col], sub[y_col])

        txt = f"AR: ρ={rho:.2f}, p={p:.2e}"

    else:
        palette = {1: "#FEB24C", 0: "#5AAE61"}
        texts = []

        for resp_val, g in sub.groupby("response"):
            label = {1: "AR", 0: "IR"}[resp_val]

            ax.scatter(
                g[x_col], g[y_col],
                s=44, alpha=0.85,
                color=palette[resp_val],
                label=label
            )

            if add_regline:
                sns.regplot(
                    data=g, x=x_col, y=y_col,
                    scatter=False, ci=None,
                    line_kws={"linewidth": 1.0, "alpha": 0.7},
                    ax=ax
                )

            rho, p = spearmanr(g[x_col], g[y_col])
            texts.append(f"{label}: ρ={rho:.2f}, p={p:.2e}")

        ax.legend(frameon=False)

        txt = "\n".join(texts)   # 🔥 핵심: 줄바꿈

    # 축선
    ax.axhline(0, linestyle="--", linewidth=1, color="grey", alpha=0.45)
    ax.axvline(0, linestyle="--", linewidth=1, color="grey", alpha=0.45)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    # 🔥 title 대신 text로 넣기
    ax.text(
        0.02, 0.98,
        txt,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10
    )

    ax.grid(alpha=0.25)
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


##################################################
# 4. 2 x 3 panel plot
##################################################

def plot_sf_vs_tu_correlation_panel(
    merged_df,
    only_ar=True,
    save_path=None,
    add_regline=True
):
    """
    2 rows x 3 cols panel:
      rows:
        1) x = SF_pre
        2) x = delta_SF

      cols:
        1) y = total_tu_score
        2) y = class1_delta
        3) y = class3_delta
    """
    df = merged_df.copy()

    if only_ar:
        df = df[df["response"] == 1].copy()

    x_cols = ["SF_pre", "delta_SF"]
    y_cols = ["total_tu_score", "class1_delta", "class3_delta"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = np.array(axes)

    for i, x_col in enumerate(x_cols):
        for j, y_col in enumerate(y_cols):
            ax = axes[i, j]

            if only_ar:
                ax.scatter(
                    df[x_col], df[y_col],
                    s=42, alpha=0.85, color="#D97706"
                )
                if add_regline:
                    sns.regplot(
                        data=df, x=x_col, y=y_col,
                        scatter=False, ci=None,
                        line_kws={"linewidth": 1.2, "alpha": 0.8},
                        ax=ax
                    )
            else:
                palette = {1: "#FEB24C", 0: "#5AAE61"}
                for resp_val, g in df.groupby("response"):
                    ax.scatter(
                        g[x_col], g[y_col],
                        s=42, alpha=0.85,
                        color=palette[resp_val],
                        label={1: "AR", 0: "IR"}[resp_val]
                    )
                    if add_regline:
                        sns.regplot(
                            data=g, x=x_col, y=y_col,
                            scatter=False, ci=None,
                            line_kws={"linewidth": 1.0, "alpha": 0.7},
                            ax=ax
                        )

            rho, p = spearmanr(df[x_col], df[y_col])

            ax.axhline(0, linestyle="--", linewidth=1, color="grey", alpha=0.45)
            ax.axvline(0, linestyle="--", linewidth=1, color="grey", alpha=0.45)

            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            rho_ar, p_ar = spearmanr(df[df["response"]==1][x_col],
                        df[df["response"]==1][y_col])

            rho_ir, p_ir = spearmanr(df[df["response"]==0][x_col],
                                    df[df["response"]==0][y_col])

            txt = f"AR: ρ={rho_ar:.2f}, p={p_ar:.1e}\nIR: ρ={rho_ir:.2f}, p={p_ir:.1e}"

            ax.text(
                0.02, 0.98,
                txt,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9
            )
            ax.grid(alpha=0.25)

            if (not only_ar) and i == 0 and j == 2:
                ax.legend(frameon=False)

    prefix = "AR only" if only_ar else "All samples"
    #fig.suptitle(f"{prefix}: SF metrics vs TU rewiring metrics", fontsize=13, y=1.02)

    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    
import numpy as np
import pandas as pd

def compute_sf_module_score(
    pre_gene,
    post_gene,
    feature_genes,
    sampleinfo,
    response_col="response",
    normalize=True
):
    """
    Compute SF module score per sample.

    Parameters
    ----------
    pre_gene, post_gene : DataFrame
        rows = genes, cols = samples

    feature_genes : list-like
        SF genes to use

    sampleinfo : DataFrame
        must contain response_col, index = sample IDs

    normalize : bool
        If True:
            gene-wise z-score using all samples provided in input
        If False:
            raw mean expression across feature genes

    Returns
    -------
    sample_score_df:
        index = sample
        columns = response, SF_pre, SF_post, delta_SF

    pre_used, post_used:
        matrices actually used for scoring
        (z-scored matrices if normalize=True,
         raw matrices if normalize=False)

    genes_used:
        actual genes used
    """

    # 1) gene filtering
    genes_used = sorted(
        set(feature_genes)
        & set(pre_gene.index)
        & set(post_gene.index)
    )

    if len(genes_used) < 2:
        raise ValueError(f"Too few SF genes used: {len(genes_used)}")

    # 2) sample alignment
    common_samples = (
        pre_gene.columns
        .intersection(post_gene.columns)
        .intersection(sampleinfo.index)
    )

    preX = pre_gene.loc[genes_used, common_samples].copy()
    postX = post_gene.loc[genes_used, common_samples].copy()
    sampleinfo_sub = sampleinfo.loc[common_samples].copy()

    # 3) normalize or not
    if normalize:
        # normalize using ALL samples in current input
        mu = preX.mean(axis=1)
        sd = preX.std(axis=1).replace(0, np.nan)

        pre_used = preX.sub(mu, axis=0).div(sd, axis=0).fillna(0.0)
        post_used = postX.sub(mu, axis=0).div(sd, axis=0).fillna(0.0)
    else:
        pre_used = preX.copy()
        post_used = postX.copy()

    # 4) module score
    sf_pre = pre_used.mean(axis=0)
    sf_post = post_used.mean(axis=0)

    # 5) output
    sample_score_df = pd.DataFrame(index=common_samples)
    sample_score_df["response"] = sampleinfo_sub.loc[common_samples, response_col].values
    sample_score_df["SF_pre"] = sf_pre.loc[common_samples].values
    sample_score_df["SF_post"] = sf_post.loc[common_samples].values
    sample_score_df["delta_SF"] = sample_score_df["SF_post"] - sample_score_df["SF_pre"]

    return sample_score_df, pre_used, post_used, genes_used

diff_sf_ar_only = [
    'ALYREF','DDX23','DDX1','PPM1G','MYEF2',
    'RBM47','EIF3A','SNRPB','ZNF207'
]

ar_samples = sampleinfo.index[sampleinfo["response"] == 1]

pre_gene_ar = pre_gene.loc[:, pre_gene.columns.intersection(ar_samples)]
post_gene_ar = post_gene.loc[:, post_gene.columns.intersection(ar_samples)]
sampleinfo_ar = sampleinfo.loc[ar_samples].copy()

sf_score_df_ar, pre_used_ar, post_used_ar, sf_used_ar = compute_sf_module_score(
    pre_gene=pre_gene_ar,
    post_gene=post_gene_ar,
    feature_genes=diff_sf_ar_only,
    sampleinfo=sampleinfo_ar,
    response_col="response",
    normalize=False
)

sf_score_df, pre_used_sf, post_used_sf, sf_used = compute_sf_module_score(
    pre_gene=pre_gene,
    post_gene=post_gene,
    feature_genes=diff_sf_ar_only,
    sampleinfo=sampleinfo,
    response_col="response",
    normalize=False
)

all_dut_list = list(set(ARdutlist) | set(IRdutlist))
# 1) TU score 계산
tu_score_df, feat_info = compute_class_dut_tu_scores(
    pre_TU_gene=pre_TU_gene,
    post_TU_gene=post_TU_gene,
    sampleinfo=sampleinfo,
    class1_list=class1,
    class3_list=class3,
    dut_list=all_dut_list,   # pre_TU_gene.index.to_list()
    response_col="response"
)

print("class1 DUT used:", len(feat_info["class1_feats"]))
print("class3 DUT used:", len(feat_info["class3_feats"]))

# sf_score_df는 이미 위에서 만든 것 사용
# columns: response, SF_pre, SF_post, delta_SF
merged_sf_tu = merge_sf_and_tu_scores(sf_score_df, tu_score_df)
# AR only 2x3 panel
plot_sf_vs_tu_correlation_panel(
    merged_sf_tu,
    only_ar=True,
    # save_path=".../SF_vs_TU_panel_AR_only.pdf"
)

# all samples 2x3 panel
plot_sf_vs_tu_correlation_panel(
    merged_sf_tu,
    only_ar=False,
    # save_path=".../SF_vs_TU_panel_all.pdf"
)



#%%
#####^^^^ RANDOM FEATURE CHECK ################

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("ticks")


def random_sf_rho_distribution(
    pre_gene,
    post_gene,
    sampleinfo,
    sf_gene_pool,
    pre_TU_gene,
    post_TU_gene,
    class1_list,
    class3_list,
    dut_list,
    sf_n=10,
    n_iter=100,
    response_col="response",
    normalize=True,
    random_state=0
):
    """
    Randomly sample SF genes and compute rho distributions in AR / IR.

    Returns
    -------
    result_df : DataFrame
        each row = one random gene set
        columns include:
            iter
            genes
            rho_AR_SFpre_total
            rho_IR_SFpre_total
            rho_AR_deltaSF_total
            rho_IR_deltaSF_total
            rho_AR_SFpre_class1
            rho_IR_SFpre_class1
            rho_AR_deltaSF_class1
            rho_IR_deltaSF_class1
            rho_AR_SFpre_class3
            rho_IR_SFpre_class3
            rho_AR_deltaSF_class3
            rho_IR_deltaSF_class3
    """

    rng = np.random.default_rng(random_state)

    # usable SF pool
    sf_pool = sorted(set(sf_gene_pool) & set(pre_gene.index) & set(post_gene.index))
    if len(sf_pool) < sf_n:
        raise ValueError(f"Usable SF pool too small: {len(sf_pool)} < {sf_n}")

    # TU score is fixed for a given dut_list
    tu_score_df, feat_info = compute_class_dut_tu_scores(
        pre_TU_gene=pre_TU_gene,
        post_TU_gene=post_TU_gene,
        sampleinfo=sampleinfo,
        class1_list=class1_list,
        class3_list=class3_list,
        dut_list=dut_list,
        response_col=response_col
    )

    rows = []

    for i in range(n_iter):
        genes = sorted(rng.choice(sf_pool, size=sf_n, replace=False).tolist())

        sf_score_df, _, _, genes_used = compute_sf_module_score(
            pre_gene=pre_gene,
            post_gene=post_gene,
            feature_genes=genes,
            sampleinfo=sampleinfo,
            response_col=response_col,
            normalize=normalize
        )

        merged = merge_sf_and_tu_scores(sf_score_df, tu_score_df)

        ar = merged[merged["response"] == 1].copy()
        ir = merged[merged["response"] == 0].copy()

        def safe_rho(df, x, y):
            if df.shape[0] < 3:
                return np.nan
            if df[x].nunique() < 2 or df[y].nunique() < 2:
                return np.nan
            return spearmanr(df[x], df[y]).statistic

        row = {
            "iter": i,
            "genes": ",".join(genes_used),

            "rho_AR_SFpre_total": safe_rho(ar, "SF_pre", "total_tu_score"),
            "rho_IR_SFpre_total": safe_rho(ir, "SF_pre", "total_tu_score"),
            "rho_AR_deltaSF_total": safe_rho(ar, "delta_SF", "total_tu_score"),
            "rho_IR_deltaSF_total": safe_rho(ir, "delta_SF", "total_tu_score"),

            "rho_AR_SFpre_class1": safe_rho(ar, "SF_pre", "class1_delta"),
            "rho_IR_SFpre_class1": safe_rho(ir, "SF_pre", "class1_delta"),
            "rho_AR_deltaSF_class1": safe_rho(ar, "delta_SF", "class1_delta"),
            "rho_IR_deltaSF_class1": safe_rho(ir, "delta_SF", "class1_delta"),

            "rho_AR_SFpre_class3": safe_rho(ar, "SF_pre", "class3_delta"),
            "rho_IR_SFpre_class3": safe_rho(ir, "SF_pre", "class3_delta"),
            "rho_AR_deltaSF_class3": safe_rho(ar, "delta_SF", "class3_delta"),
            "rho_IR_deltaSF_class3": safe_rho(ir, "delta_SF", "class3_delta"),
        }

        rows.append(row)

    result_df = pd.DataFrame(rows)
    return result_df

def plot_random_rho_distribution(
    result_df,
    ar_col,
    ir_col,
    title=None,
    bins=40,
    save_path=None
):
    """
    Plot rho distributions for AR vs IR from random SF sets.
    """
    plot_df = pd.DataFrame({
        "rho": pd.concat([
            result_df[ar_col].rename("rho"),
            result_df[ir_col].rename("rho")
        ], axis=0).values,
        "group": (["AR"] * len(result_df)) + (["IR"] * len(result_df))
    }).dropna()

    fig, ax = plt.subplots(figsize=(6, 4.5))

    palette = {"AR": "#FEB24C", "IR": "#5AAE61"}

    for grp in ["AR", "IR"]:
        sub = plot_df[plot_df["group"] == grp]
        sns.histplot(
            sub["rho"],
            bins=bins,
            stat="density",
            kde=True,
            element="step",
            fill=False,
            ax=ax,
            color=palette[grp],
            label=grp
        )

    ar_med = result_df[ar_col].median()
    ir_med = result_df[ir_col].median()

    ax.axvline(ar_med, color=palette["AR"], linestyle="--", alpha=0.8)
    ax.axvline(ir_med, color=palette["IR"], linestyle="--", alpha=0.8)

    ax.set_xlabel("Spearman ρ")
    ax.set_ylabel("Density")
    if title is not None:
        ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_random_rho_distribution_panel(
    result_df,
    save_path=None,
    bins=35
):
    """
    2x3 panel of rho distributions:
      rows = SF_pre / delta_SF
      cols = total / class1 / class3
    """
    pairs = [
        ("rho_AR_SFpre_total",   "rho_IR_SFpre_total",   "SF_pre vs total_tu_score"),
        ("rho_AR_SFpre_class1",  "rho_IR_SFpre_class1",  "SF_pre vs class1_delta"),
        ("rho_AR_SFpre_class3",  "rho_IR_SFpre_class3",  "SF_pre vs class3_delta"),
        ("rho_AR_deltaSF_total", "rho_IR_deltaSF_total", "delta_SF vs total_tu_score"),
        ("rho_AR_deltaSF_class1","rho_IR_deltaSF_class1","delta_SF vs class1_delta"),
        ("rho_AR_deltaSF_class3","rho_IR_deltaSF_class3","delta_SF vs class3_delta"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = np.array(axes)

    palette = {"AR": "#FEB24C", "IR": "#5AAE61"}

    for ax, (ar_col, ir_col, ttl) in zip(axes.flat, pairs):
        for grp, col in [("AR", ar_col), ("IR", ir_col)]:
            sub = result_df[col].dropna()
            sns.histplot(
                sub,
                bins=bins,
                stat="density",
                kde=True,
                element="step",
                fill=False,
                ax=ax,
                color=palette[grp],
                label=grp if ttl == pairs[0][2] else None
            )

            med = sub.median()
            ax.axvline(med, color=palette[grp], linestyle="--", alpha=0.8)

        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("Spearman ρ")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, frameon=False)

    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# 예시: 랜덤 후보군
sf_gene_pool = SF_genes   # 네가 가진 전체 SF gene 리스트라고 가정

# DUT list 선택 가능
# dut_list = ARdutlist
# dut_list = IRdutlist
dut_list = all_dut_list

rand_rho_df = random_sf_rho_distribution(
    pre_gene=pre_gene,
    post_gene=post_gene,
    sampleinfo=sampleinfo,
    sf_gene_pool=sf_gene_pool,
    pre_TU_gene=pre_TU_gene,
    post_TU_gene=post_TU_gene,
    class1_list=class1,
    class3_list=class3,
    dut_list=ARdutlist,
    sf_n=10,
    n_iter=100,
    response_col="response",
    normalize=True,
    random_state=42
)

rand_rho_df.head()

plot_random_rho_distribution_panel(rand_rho_df)



#%%
####^^ corr volcano ###################
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


def compute_sf_gene_volcano_table(
    pre_gene,
    post_gene,
    sampleinfo,
    sf_gene_pool,
    pre_TU_gene,
    post_TU_gene,
    class1_list,
    class3_list,
    dut_list,
    response_col="response",
    log_transform=True,
    add_pseudocount=True
):
    """
    For each SF gene, compute Spearman correlation between:
      - pre expression / delta expression
    and
      - total_tu_score / class1_delta / class3_delta

    separately in AR and IR.

    Returns
    -------
    result_df : DataFrame
        one row per gene x expr_type x score_type x response_group
    """
    # --------------------------------------------
    # 1) usable genes
    # --------------------------------------------
    genes = sorted(set(sf_gene_pool) & set(pre_gene.index) & set(post_gene.index))
    if len(genes) == 0:
        raise ValueError("No usable genes found in sf_gene_pool.")

    # --------------------------------------------
    # 2) TU score table
    # --------------------------------------------
    tu_score_df, feat_info = compute_class_dut_tu_scores(
        pre_TU_gene=pre_TU_gene,
        post_TU_gene=post_TU_gene,
        sampleinfo=sampleinfo,
        class1_list=class1_list,
        class3_list=class3_list,
        dut_list=dut_list,
        response_col=response_col
    )

    # --------------------------------------------
    # 3) sample alignment
    # --------------------------------------------
    common_samples = (
        pre_gene.columns
        .intersection(post_gene.columns)
        .intersection(sampleinfo.index)
        .intersection(tu_score_df.index)
    )

    preX = pre_gene.loc[genes, common_samples].copy()
    postX = post_gene.loc[genes, common_samples].copy()
    sampleinfo2 = sampleinfo.loc[common_samples].copy()
    tu_score_df = tu_score_df.loc[common_samples].copy()

    # --------------------------------------------
    # 4) transform expression
    # --------------------------------------------
    if log_transform:
        if add_pseudocount:
            preX = np.log1p(preX)
            postX = np.log1p(postX)
        else:
            preX = np.log(preX)
            postX = np.log(postX)

    deltaX = postX - preX

    # --------------------------------------------
    # 5) build per-sample table
    # --------------------------------------------
    rows = []

    score_cols = ["total_tu_score", "class1_delta", "class3_delta"]

    for resp_value, resp_label in [(1, "AR"), (0, "IR")]:
        samples = sampleinfo2.index[sampleinfo2[response_col] == resp_value]

        if len(samples) < 3:
            continue

        for gene in genes:
            x_pre = preX.loc[gene, samples]
            x_delta = deltaX.loc[gene, samples]

            for score_col in score_cols:
                y = tu_score_df.loc[samples, score_col]

                for expr_type, x in [("pre_expr", x_pre), ("delta_expr", x_delta)]:
                    valid = pd.concat(
                        [x.rename("x"), y.rename("y")], axis=1
                    ).dropna()

                    n = valid.shape[0]

                    if n < 3 or valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
                        rho = np.nan
                        pval = np.nan
                    else:
                        res = spearmanr(valid["x"], valid["y"])
                        rho = res.statistic
                        pval = res.pvalue

                    rows.append({
                        "gene": gene,
                        "group": resp_label,              # AR / IR
                        "expr_type": expr_type,          # pre_expr / delta_expr
                        "score_type": score_col,         # total_tu_score / class1_delta / class3_delta
                        "n": n,
                        "rho": rho,
                        "pval": pval,
                    })

    result_df = pd.DataFrame(rows)

    # --------------------------------------------
    # 6) FDR correction within each panel
    # --------------------------------------------
    result_df["fdr"] = np.nan

    for (grp, expr_type, score_type), idx in result_df.groupby(
        ["group", "expr_type", "score_type"]
    ).groups.items():
        sub = result_df.loc[idx]
        mask = sub["pval"].notna()
        if mask.sum() > 0:
            fdr = multipletests(sub.loc[mask, "pval"], method="fdr_bh")[1]
            result_df.loc[sub.loc[mask].index, "fdr"] = fdr

    result_df["neglog10p"] = -np.log10(result_df["pval"])
    result_df["neglog10fdr"] = -np.log10(result_df["fdr"])

    return result_df

import matplotlib.pyplot as plt
import seaborn as sns


def plot_sf_gene_volcano_panel(
    volcano_df,
    use_fdr=False,
    alpha=0.05,
    label_top_n=5,
    min_abs_rho_for_label=0.3,
    figsize=(7, 14),
    save_path=None
):
    """
    2x3 volcano panel:
      rows = pre_expr / delta_expr
      cols = total_tu_score / class1_delta / class3_delta

    AR and IR are overlaid in each panel.

    Parameters
    ----------
    use_fdr : bool
        if True, y-axis = -log10(FDR), threshold = alpha on FDR
        if False, y-axis = -log10(pval), threshold = alpha on pval
    """
    df = volcano_df.copy()

    ycol = "neglog10fdr" if use_fdr else "neglog10p"
    sigcol = "fdr" if use_fdr else "pval"

    expr_order = ["pre_expr", "delta_expr"]
    score_order = ["total_tu_score", "class1_delta", "class3_delta"]

    title_map = {
        ("pre_expr", "total_tu_score"): "Pre SF expression vs total TU score",
        ("pre_expr", "class1_delta"):   "Pre SF expression vs class1 delta",
        ("pre_expr", "class3_delta"):   "Pre SF expression vs class3 delta",
        ("delta_expr", "total_tu_score"): "Delta SF expression vs total TU score",
        ("delta_expr", "class1_delta"):   "Delta SF expression vs class1 delta",
        ("delta_expr", "class3_delta"):   "Delta SF expression vs class3 delta",
    }

    palette = {"AR": "#F0B44D", "IR": "#63B96A"}

    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex=True, sharey=True)
    axes = np.array(axes)

    for j, expr_type in enumerate(expr_order):
        for i, score_type in enumerate(score_order):
            ax = axes[i, j]

            sub = df[
                (df["expr_type"] == expr_type) &
                (df["score_type"] == score_type)
            ].copy()

            for grp in ["AR", "IR"]:
                ss = sub[sub["group"] == grp].copy()

                ax.scatter(
                    ss["rho"],
                    ss[ycol],
                    s=28,
                    alpha=0.75,
                    color='gray',
                    edgecolor="none",
                    label=grp if (i == 0 and j == 0) else None
                )

                # significant points
                sig = ss[ss[sigcol] < alpha].copy()
                if len(sig) > 0:
                    ax.scatter(
                        sig["rho"],
                        sig[ycol],
                        s=36,
                        alpha=0.95,
                        color=palette[grp],
                        edgecolor="none",
                        linewidth=0.4
                    )

                # label top genes
                cand = ss[
                    (ss[sigcol] < alpha) &
                    (ss["rho"].abs() >= min_abs_rho_for_label)
                ].copy()

                if len(cand) > 0:
                    cand = cand.sort_values([ycol, "rho"], ascending=[False, False])
                    cand = cand.head(label_top_n)

                    for _, r in cand.iterrows():
                        ax.text(
                            r["rho"], r[ycol], r["gene"],
                            fontsize=8, alpha=0.9
                        )

            ax.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
            ax.axhline(-np.log10(alpha), color="gray", linestyle=":", linewidth=1, alpha=0.7)

            ax.set_title(title_map[(expr_type, score_type)], fontsize=11)
            ax.grid(alpha=0.2)

    for ax in axes[1, :]:
        ax.set_xlabel("Spearman rho")

    for ax in axes[:, 0]:
        ax.set_ylabel("-log10(FDR)" if use_fdr else "-log10(p-value)")

    # handles, labels = axes[0, 0].get_legend_handles_labels()
    # if handles:
    #     axes[0, 0].legend(handles, labels, frameon=False, loc="upper right")

    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


sf_gene_pool = SF_genes

volcano_df = compute_sf_gene_volcano_table(
    pre_gene=pre_gene,
    post_gene=post_gene,
    sampleinfo=sampleinfo,
    sf_gene_pool=sf_gene_pool,
    pre_TU_gene=pre_TU_gene,
    post_TU_gene=post_TU_gene,
    class1_list=class1,
    class3_list=class3,
    dut_list=ARdutlist,   # 또는 all_dut_list
    response_col="response",
    log_transform=True
)

volcano_df.head()

plot_sf_gene_volcano_panel(
    volcano_df,
    use_fdr=False,      # 먼저 p-value로 보고
    alpha=0.05,
    label_top_n=3,
    min_abs_rho_for_label=0.4,
    figsize=(7, 8),
    save_path=None
)

#%%
def get_sig_genes(df, group, expr_type, score_type, alpha=0.05):
    sub = df[
        (df["group"] == group) &
        (df["expr_type"] == expr_type) &
        (df["score_type"] == score_type)
    ]
    return set(sub[sub["pval"] < alpha]["gene"])
from matplotlib_venn import venn2

def plot_pre_delta_venn(
    pre_set,
    delta_set,
    title="",
    colors=("#F4A3A3", "#8BC48A"),   # left, right
    alpha=0.75,
    number_fontsize=14,
    figsize=(3, 3),
    save_path=None
):
    fig, ax = plt.subplots(figsize=figsize)

    v = venn2(
        [pre_set, delta_set],
        set_labels=("", ""),   # label 제거
        set_colors=colors,
        alpha=alpha,
        ax=ax
    )
    
    if v.get_patch_by_id('11'):
        v.get_patch_by_id('11').set_color("#E1AC2E")
    
    # 숫자(font size) 키우기
    for sid in ["10", "01", "11"]:
        txt = v.get_label_by_id(sid)
        if txt is not None:
            txt.set_fontsize(number_fontsize)
            txt.set_fontweight("bold")

    # 혹시 남아 있으면 set label 완전히 숨김
    for sid in ["A", "B"]:
        txt = v.get_label_by_id(sid)
        if txt is not None:
            txt.set_text("")

    ax.set_title(title, fontsize=13)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    
score_name_map = {
    "total_tu_score": "Total TU score",
    "class1_delta": "Class 1 delta",
    "class3_delta": "Class 3 delta"
}

for score_type in ["total_tu_score", "class1_delta", "class3_delta"]:
    pre_set = get_sig_genes(volcano_df, "AR", "pre_expr", score_type)
    delta_set = get_sig_genes(volcano_df, "AR", "delta_expr", score_type)

    plot_pre_delta_venn(
        pre_set,
        delta_set,
        colors=("#F6A6A6", "#8FCA8E"),   # 원하는 색으로 조절
        alpha=0.7,
        number_fontsize=15
    )

#%%
####^^ coexp network #######################
score_types = ["total_tu_score", "class1_delta", "class3_delta"]

intersection_dict = {}
for score_type in score_types:
    pre_set = get_sig_genes(volcano_df, "AR", "pre_expr", score_type)
    delta_set = get_sig_genes(volcano_df, "AR", "delta_expr", score_type)

    intersection_dict[score_type] = pre_set & delta_set

# 각 score별 intersection 확인
for k, v in intersection_dict.items():
    print(k, len(v))

# 최종 union
final_gene_set = set().union(*intersection_dict.values())

print("Final union size:", len(final_gene_set))
print(sorted(final_gene_set))

core_gene_set = (
    intersection_dict["total_tu_score"]
    & intersection_dict["class1_delta"]
    & intersection_dict["class3_delta"]
)

print("Core intersection size:", len(core_gene_set))
print(sorted(core_gene_set))

def get_ar_pre_corr_matrix(
    pre_gene,
    sampleinfo,
    gene_set,
    response_col="response",
    log_transform=True,
    method="spearman"
):
    genes = sorted(set(gene_set) & set(pre_gene.index))
    if len(genes) < 2:
        raise ValueError(f"Need at least 2 genes, got {len(genes)}")

    ar_samples = sampleinfo.index[sampleinfo[response_col] == 1]
    ar_samples = pre_gene.columns.intersection(ar_samples)

    X = pre_gene.loc[genes, ar_samples].copy()

    if log_transform:
        X = np.log1p(X)

    # rows=genes, cols=samples
    corr_mat = X.T.corr(method=method)

    return corr_mat, X

corr_mat, ar_pre_expr_mat = get_ar_pre_corr_matrix(
    pre_gene=pre_gene,
    sampleinfo=sampleinfo,
    gene_set=final_gene_set,
    response_col="response",
    log_transform=True,
    method="spearman"
)

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def plot_corr_network(
    corr_mat,
    threshold=0.6,
    figsize=(8, 8),
    pos_edge_color="#D95F5F",
    neg_edge_color="#5F8DD9",
    node_color="#D9D9D9",
    node_size=900,
    font_size=9,
    layout_seed=0,
    save_path=None
):
    genes = corr_mat.index.tolist()
    G = nx.Graph()
    G.add_nodes_from(genes)

    # edge 추가
    for i in range(len(genes)):
        for j in range(i + 1, len(genes)):
            g1, g2 = genes[i], genes[j]
            r = corr_mat.loc[g1, g2]

            if pd.isna(r):
                continue
            if abs(r) >= threshold:
                G.add_edge(g1, g2, weight=abs(r), corr=r)

    if G.number_of_edges() == 0:
        print(f"No edges found with |corr| >= {threshold}")
        return G

    pos = nx.spring_layout(G, seed=layout_seed, k=0.6)

    edge_colors = [
        pos_edge_color if G[u][v]["corr"] > 0 else neg_edge_color
        for u, v in G.edges()
    ]
    edge_widths = [
        1.5 + 4.0 * (G[u][v]["weight"] - threshold) / (1 - threshold)
        for u, v in G.edges()
    ]

    plt.figure(figsize=figsize)

    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_color,
        node_size=node_size,
        edgecolors="black",
        linewidths=0.8
    )

    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.8
    )

    nx.draw_networkx_labels(
        G, pos,
        font_size=font_size,
        font_family="sans-serif"
    )

    plt.axis("off")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return G

import seaborn as sns
import matplotlib.pyplot as plt
def summarize_corr_network(G):
    deg = pd.Series(dict(G.degree())).sort_values(ascending=False)
    print("Top hubs:")
    print(deg.head(20))
    return deg

def plot_corr_clustermap(
    corr_mat,
    figsize=(8, 8),
    cmap="vlag",
    center=0,
    linewidths=0.3,
    save_path=None
):
    g = sns.clustermap(
        corr_mat,
        cmap=cmap,
        center=center,
        linewidths=linewidths,
        figsize=figsize,
        xticklabels=True,
        yticklabels=True,
        method="ward",
    )

    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)

    if save_path:
        g.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

G = plot_corr_network(
    corr_mat,
    threshold=0.6,   # 0.5~0.7 사이 조절
    figsize=(8, 8),
    pos_edge_color="#E07A7A",
    neg_edge_color="#6D8FD5",
    node_color="#EAEAEA",
    node_size=1000,
    font_size=8
)
# --------------------------------------------------
# 1. score별 intersection
# --------------------------------------------------
score_types = ["total_tu_score", "class1_delta", "class3_delta"]

intersection_dict = {}
for score_type in score_types:
    pre_set = get_sig_genes(volcano_df, "AR", "pre_expr", score_type, alpha=0.05)
    delta_set = get_sig_genes(volcano_df, "AR", "delta_expr", score_type, alpha=0.05)

    intersection_dict[score_type] = pre_set & delta_set

    plot_pre_delta_venn(
        pre_set,
        delta_set,
        title=score_type,
        colors=("#F4A6A6", "#95C98F"),
        alpha=0.72,
        number_fontsize=16
    )

# --------------------------------------------------
# 2. final union
# --------------------------------------------------
final_gene_set = set().union(*intersection_dict.values())
print("Final gene set size:", len(final_gene_set))
print(sorted(final_gene_set))

# optional core
core_gene_set = (
    intersection_dict["total_tu_score"]
    & intersection_dict["class1_delta"]
    & intersection_dict["class3_delta"]
)
print("Core gene set size:", len(core_gene_set))
print(sorted(core_gene_set))

# --------------------------------------------------
# 3. AR pre corr matrix
# --------------------------------------------------
corr_mat, ar_pre_expr_mat = get_ar_pre_corr_matrix(
    pre_gene=pre_gene,
    sampleinfo=sampleinfo,
    gene_set=final_gene_set,
    response_col="response",
    log_transform=True,
    method="spearman"
)

# --------------------------------------------------
# 4. clustermap
# --------------------------------------------------
plot_corr_clustermap(
    corr_mat,
    figsize=(10, 10),
    cmap="coolwarm",
    center=0
)

# --------------------------------------------------
# 5. network
# --------------------------------------------------
G = plot_corr_network(
    corr_mat,
    threshold=0.6,
    figsize=(10, 10),
    pos_edge_color="#E07A7A",
    neg_edge_color="#6A8FD7",
    node_color="#EFEFEF",
    node_size=1100,
    font_size=10
)

hub_degree = summarize_corr_network(G)
print(hub_degree.head(15))

#%%
score_types = ["total_tu_score", "class1_delta", "class3_delta"]

intersection_dict = {}
for score_type in score_types:
    pre_set = get_sig_genes(volcano_df, "AR", "pre_expr", score_type, alpha=0.05)
    delta_set = get_sig_genes(volcano_df, "AR", "delta_expr", score_type, alpha=0.05)
    intersection_dict[score_type] = pre_set & delta_set

core_gene_set = (
    intersection_dict["total_tu_score"]
    & intersection_dict["class1_delta"]
    & intersection_dict["class3_delta"]
)

print("core size:", len(core_gene_set))
print(sorted(core_gene_set))

corr_core, ar_pre_expr_core = get_ar_pre_corr_matrix(
    pre_gene=pre_gene,
    sampleinfo=sampleinfo,
    gene_set=core_gene_set,
    response_col="response",
    log_transform=True,
    method="spearman"
)

plot_corr_clustermap(
    corr_core,
    figsize=(8, 8),
    cmap="coolwarm",
    center=0
)

G_core = plot_corr_network(
    corr_core,
    threshold=0.75,   # 0.6 말고 더 높게
    figsize=(8, 8),
    pos_edge_color="#953809",
    neg_edge_color="#6A8FD7",
    node_color="#FFFFFF",
    node_size=300,
    font_size=8
)

#%%
##^^ PSI ############# 

AR_PRE_RI  = "/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/suppaoutput/AR/AR_pre-suppaevent_RI_variable_10.ioe.psi"
AR_POST_RI = "/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/suppaoutput/AR/AR_post-suppaevent_RI_variable_10.ioe.psi"
IR_PRE_RI  = "/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/suppaoutput/IR/IR_pre-suppaevent_RI_variable_10.ioe.psi"
IR_POST_RI = "/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/suppaoutput/IR/IR_post-suppaevent_RI_variable_10.ioe.psi"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse
from scipy.stats import mannwhitneyu, wilcoxon, spearmanr

sns.set_style("ticks")


def read_suppa_psi_matrix(path):
    # 첫 줄 header 직접 읽기
    with open(path, "r") as f:
        header = f.readline().rstrip("\n").split("\t")

    # 첫 칸이 비어 있는 SUPPA 형식이라고 가정하고 event_id 추가
    colnames = ["event_id"] + header

    # 실제 데이터 읽기
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        skiprows=1,
        names=colnames
    )

    df = df.set_index("event_id")
    df.index.name = "event_id"

    # SUPPA missing value가 'NA', 'nan', 'na' 등일 수 있으니 숫자화
    df = df.replace(["NA", "na", "NaN", "nan"], np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")

    # 중복 event_id 있으면 평균
    if not df.index.is_unique:
        print(f"[WARN] duplicated event_id found in {path}: {df.index.duplicated().sum()}")
        df = df.groupby(df.index).mean()

    return df

def get_sig_ri_events_from_sig_tables(ar_psi, ir_psi, use_union=True):
    """
    ar_psi, ir_psi are already significant-event tables.

    Returns
    -------
    sig_ri_df : DataFrame
        metadata table for RI events
    sig_ri_events : list
        RI event_id list
    """

    ar_ri = ar_psi.loc[ar_psi["event"] == "RI"].copy()
    ir_ri = ir_psi.loc[ir_psi["event"] == "RI"].copy()

    ar_ri["group"] = "AR"
    ir_ri["group"] = "IR"

    if use_union:
        sig_ri_events = sorted(set(ar_ri["event_id"]) | set(ir_ri["event_id"]))
    else:
        sig_ri_events = sorted(set(ar_ri["event_id"]) & set(ir_ri["event_id"]))

    sig_ri_df = pd.concat([ar_ri, ir_ri], axis=0, ignore_index=True)
    sig_ri_df = sig_ri_df.drop_duplicates()

    return sig_ri_df, sig_ri_events

sig_ri_df, sig_ri_events = get_sig_ri_events_from_sig_tables(
    ar_psi=ar_psi,
    ir_psi=ir_psi,
    use_union=True   # 또는 False
)

def build_sig_ri_pre_post_matrices(
    ri_ar_pre,
    ri_ar_post,
    ri_ir_pre,
    ri_ir_post,
    sig_event_ids
):
    # 공통 event 중 significant RI event만 유지
    common_events = (
        ri_ar_pre.index
        .intersection(ri_ar_post.index)
        .intersection(ri_ir_pre.index)
        .intersection(ri_ir_post.index)
        .intersection(pd.Index(sig_event_ids))
    )
    print("common_events:", len(common_events))

    ri_ar_pre  = ri_ar_pre.loc[common_events].copy()
    ri_ar_post = ri_ar_post.loc[common_events].copy()
    ri_ir_pre  = ri_ir_pre.loc[common_events].copy()
    ri_ir_post = ri_ir_post.loc[common_events].copy()

    # pre / post 합치기
    psi_pre = pd.concat([ri_ar_pre, ri_ir_pre], axis=1)
    psi_post = pd.concat([ri_ar_post, ri_ir_post], axis=1)

    psi_pre.columns = psi_pre.columns.str[:-4]
    psi_post.columns = psi_post.columns.str[:-4]
    # 공통 sample 유지
    common_samples = psi_pre.columns.intersection(psi_post.columns)
    psi_pre = psi_pre.loc[:, common_samples].copy()
    psi_post = psi_post.loc[:, common_samples].copy()
    return psi_pre, psi_post

# 1) sig table에서 RI event만 추출
sig_ri_df, sig_ri_events = get_sig_ri_events_from_sig_tables(
    ar_psi=ar_psi,
    ir_psi=ir_psi,
    use_union=True
)

print("n sig RI events:", len(sig_ri_events))

# 2) SUPPA PSI matrix 읽기
ri_ar_pre  = read_suppa_psi_matrix(AR_PRE_RI)
ri_ar_post = read_suppa_psi_matrix(AR_POST_RI)
ri_ir_pre  = read_suppa_psi_matrix(IR_PRE_RI)
ri_ir_post = read_suppa_psi_matrix(IR_POST_RI)

print("AR_pre :", ri_ar_pre.shape)
print("AR_post:", ri_ar_post.shape)
print("IR_pre :", ri_ir_pre.shape)
print("IR_post:", ri_ir_post.shape)

# 3) sig RI event만 유지해서 pre/post matrix 만들기
psi_pre_ri, psi_post_ri = build_sig_ri_pre_post_matrices(
    ri_ar_pre=ri_ar_pre,
    ri_ar_post=ri_ar_post,
    ri_ir_pre=ri_ir_pre,
    ri_ir_post=ri_ir_post,
    sig_event_ids=sig_ri_events
)

print("psi_pre_ri :", psi_pre_ri.shape)
print("psi_post_ri:", psi_post_ri.shape)
print("n common samples:", len(psi_pre_ri.columns))

def filter_and_impute_psi(psi_pre, psi_post, min_valid_frac=0.8):
    keep = (
        (psi_pre.notna().mean(axis=1) >= min_valid_frac) &
        (psi_post.notna().mean(axis=1) >= min_valid_frac)
    )

    psi_pre = psi_pre.loc[keep].copy()
    psi_post = psi_post.loc[keep].copy()

    psi_pre = psi_pre.apply(lambda x: x.fillna(x.mean()), axis=1)
    psi_post = psi_post.apply(lambda x: x.fillna(x.mean()), axis=1)

    return psi_pre, psi_post

psi_pre_ri, psi_post_ri = filter_and_impute_psi(
    psi_pre_ri,
    psi_post_ri,
    min_valid_frac=0.8
)

print("after filtering")
print("psi_pre_ri :", psi_pre_ri.shape)
print("psi_post_ri:", psi_post_ri.shape)

from scipy.special import logit
from sklearn.decomposition import PCA
from scipy.stats import wilcoxon
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _prep_logit_z(preX, postX, ir_samples, eps=1e-3, use_zscore=True):
    preX = preX.clip(eps, 1 - eps)
    postX = postX.clip(eps, 1 - eps)

    preT = pd.DataFrame(logit(preX.values), index=preX.index, columns=preX.columns)
    postT = pd.DataFrame(logit(postX.values), index=postX.index, columns=postX.columns)

    if use_zscore:
        mu = preT[ir_samples].mean(axis=1)
        sd = preT[ir_samples].std(axis=1).replace(0, np.nan)
        preT = preT.sub(mu, axis=0).div(sd, axis=0).fillna(0.0)
        postT = postT.sub(mu, axis=0).div(sd, axis=0).fillna(0.0)

    return preT, postT


def _add_cov_ellipse(ax, pts2d, nsig=1.0, face_alpha=0.08, edge_lw=2.5, zorder=50):
    if pts2d.shape[0] < 5:
        return None

    center = pts2d.mean(axis=0)
    cov = np.cov(pts2d.T)

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * nsig * np.sqrt(np.maximum(vals, 1e-12))

    e = Ellipse(
        xy=center, width=width, height=height, angle=angle,
        facecolor="black", alpha=face_alpha,
        edgecolor="black", linewidth=edge_lw,
        zorder=zorder
    )
    ax.add_patch(e)
    ax.scatter([center[0]], [center[1]], marker="X", s=120, linewidths=2, zorder=zorder + 1)
    return e


def plot_ir_axis_main_figure_events(
    pre_mat,
    post_mat,
    sampleinfo,
    feature_list=None,
    ar_event_list=None,
    ir_event_list=None,
    feature_mode="union",   # "union", "intersection", "feature_list"
    response_col="response",
    ar_value=1,
    ir_value=0,
    eps=1e-3,
    use_zscore=True,
    drop_nan_features=True,
    ellipse_nsig=1.0,
    ellipse_use_all_ir=True,
    title="RI event shift along IR axis"
):
    """
    pre_mat, post_mat:
        rows = event_id
        cols = samples
    """

    # ---------------------------------------------------------
    # 1) feature selection
    # ---------------------------------------------------------
    available = set(pre_mat.index) & set(post_mat.index)

    if feature_mode == "feature_list":
        if feature_list is None:
            raise ValueError("feature_list is required when feature_mode='feature_list'")
        feat = sorted(set(feature_list) & available)

    elif feature_mode == "union":
        if ar_event_list is None or ir_event_list is None:
            raise ValueError("ar_event_list and ir_event_list are required when feature_mode='union'")
        feat = sorted((set(ar_event_list) | set(ir_event_list)) & available)

    elif feature_mode == "intersection":
        if ar_event_list is None or ir_event_list is None:
            raise ValueError("ar_event_list and ir_event_list are required when feature_mode='intersection'")
        feat = sorted((set(ar_event_list) & set(ir_event_list)) & available)

    else:
        raise ValueError("feature_mode must be one of: 'union', 'intersection', 'feature_list'")

    if len(feat) < 5:
        raise ValueError(f"Too few features after filtering: {len(feat)}")

    preX = pre_mat.loc[feat].copy()
    postX = post_mat.loc[feat].copy()

    # ---------------------------------------------------------
    # 2) align samples
    # ---------------------------------------------------------
    common_samples = preX.columns.intersection(postX.columns).intersection(sampleinfo.index)
    preX = preX[common_samples]
    postX = postX[common_samples]

    resp = sampleinfo.loc[common_samples, response_col]
    ar_samples = resp[resp == ar_value].index
    ir_samples = resp[resp == ir_value].index

    if len(ar_samples) < 3 or len(ir_samples) < 3:
        raise ValueError(f"Not enough samples: AR={len(ar_samples)}, IR={len(ir_samples)}")

    # ---------------------------------------------------------
    # 3) NaN handling
    # ---------------------------------------------------------
    if drop_nan_features:
        ok = (~preX.isna().any(axis=1)) & (~postX.isna().any(axis=1))
        preX = preX.loc[ok]
        postX = postX.loc[ok]
    else:
        preX = preX.apply(lambda x: x.fillna(x.mean()), axis=1)
        postX = postX.apply(lambda x: x.fillna(x.mean()), axis=1)

    n_features = preX.shape[0]
    if n_features < 5:
        raise ValueError(f"Too few features after NaN filtering: {n_features}")

    # ---------------------------------------------------------
    # 4) logit + zscore
    # ---------------------------------------------------------
    preT, postT = _prep_logit_z(preX, postX, ir_samples, eps=eps, use_zscore=use_zscore)

    # ---------------------------------------------------------
    # 5) direction vector = mean(IR_pre) - mean(AR_pre)
    # ---------------------------------------------------------
    v = preT[ir_samples].mean(axis=1) - preT[ar_samples].mean(axis=1)
    v = v / (np.linalg.norm(v.values) + 1e-12)

    score_pre = preT.T.dot(v)
    score_post = postT.T.dot(v)

    delta_ar = score_post.loc[ar_samples] - score_pre.loc[ar_samples]
    try:
        _, p_delta = wilcoxon(delta_ar.values, alternative="greater")
    except:
        p_delta = np.nan
    frac_pos = float((delta_ar > 0).mean())

    # ---------------------------------------------------------
    # 6) PCA
    # ---------------------------------------------------------
    cols_for_pca = list(ir_samples) + list(ar_samples)

    X_pre = preT[cols_for_pca].T
    X_post = postT[cols_for_pca].T
    X_all = pd.concat([X_pre, X_post], axis=0)

    pca = PCA(n_components=2, random_state=0)
    Z_all = pca.fit_transform(X_all.values)

    n = X_pre.shape[0]
    coord_pre = pd.DataFrame(Z_all[:n, :], index=X_pre.index, columns=["PC1", "PC2"])
    coord_post = pd.DataFrame(Z_all[n:, :], index=X_post.index, columns=["PC1", "PC2"])

    ar_pre_ctr = coord_pre.loc[ar_samples].mean(axis=0).values
    ir_pre_ctr = coord_pre.loc[ir_samples].mean(axis=0).values

    # ---------------------------------------------------------
    # 7) plot
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A
    ax = axes[0]
    ax.scatter(coord_pre.loc[ir_samples, "PC1"], coord_pre.loc[ir_samples, "PC2"], alpha=0.9, label="IR_pre")
    ax.scatter(coord_pre.loc[ar_samples, "PC1"], coord_pre.loc[ar_samples, "PC2"], alpha=0.9, label="AR_pre")
    ax.scatter(coord_post.loc[ar_samples, "PC1"], coord_post.loc[ar_samples, "PC2"], alpha=0.9, label="AR_post")
    ax.scatter(coord_post.loc[ir_samples, "PC1"], coord_post.loc[ir_samples, "PC2"], alpha=0.6, label="IR_post")

    for s in ar_samples:
        x0, y0 = coord_pre.loc[s, ["PC1", "PC2"]]
        x1, y1 = coord_post.loc[s, ["PC1", "PC2"]]
        ax.plot([x0, x1], [y0, y1], alpha=0.25, linewidth=1)

    ir_pts = []
    for s in ir_samples:
        ir_pts.append(coord_pre.loc[s, ["PC1", "PC2"]].values)
        if ellipse_use_all_ir:
            ir_pts.append(coord_post.loc[s, ["PC1", "PC2"]].values)
    ir_pts = np.array(ir_pts, dtype=float)

    _add_cov_ellipse(ax, ir_pts, nsig=ellipse_nsig, face_alpha=0.08, edge_lw=2.8, zorder=30)

    ax.arrow(
        ar_pre_ctr[0], ar_pre_ctr[1],
        (ir_pre_ctr[0] - ar_pre_ctr[0]) * 0.9,
        (ir_pre_ctr[1] - ar_pre_ctr[1]) * 0.9,
        head_width=0.2, length_includes_head=True, linewidth=2.5, zorder=40
    )
    ax.text(ir_pre_ctr[0], ir_pre_ctr[1], "IR axis", fontsize=10, ha="left", va="bottom", zorder=41)

    txt = (
        f"n_features={n_features}\n"
        f"AR n={len(ar_samples)}, IR n={len(ir_samples)}\n"
        f"Δ>0 fraction={frac_pos:.2f}\n"
        f"Wilcoxon Δ>0 p={p_delta:.2e}" if pd.notna(p_delta) else
        f"n_features={n_features}\nAR n={len(ar_samples)}, IR n={len(ir_samples)}\nΔ>0 fraction={frac_pos:.2f}\nWilcoxon p=NA"
    )
    ax.text(
        0.02, 0.98, txt, transform=ax.transAxes,
        ha="left", va="top", fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    ax.set_title("PCA view + IR cluster + axis")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    # Panel B
    ax = axes[1]
    s_pre_ar = score_pre.loc[ar_samples]
    s_post_ar = score_post.loc[ar_samples]

    ax.scatter(np.zeros(len(ar_samples)), s_pre_ar.values)
    ax.scatter(np.ones(len(ar_samples)), s_post_ar.values)
    for s in ar_samples:
        ax.plot([0, 1], [s_pre_ar.loc[s], s_post_ar.loc[s]], alpha=0.35, linewidth=1)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["AR_pre", "AR_post"])
    ax.set_ylabel("IR-axis score")
    ax.set_title("Projection to IR direction")
    ax.text(
        0.02, 0.98,
        f"Δ>0 fraction={frac_pos:.2f}\nWilcoxon Δ>0 p={p_delta:.2e}" if pd.notna(p_delta)
        else f"Δ>0 fraction={frac_pos:.2f}\nWilcoxon p=NA",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=10, bbox=dict(boxstyle="round", alpha=0.15)
    )
    ax.grid(alpha=0.2)

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()

    out = {
        "fig": fig,
        "coord_pre": coord_pre,
        "coord_post": coord_post,
        "score_pre": score_pre,
        "score_post": score_post,
        "delta_ar": delta_ar,
        "p_delta": float(p_delta) if pd.notna(p_delta) else np.nan,
        "frac_pos": frac_pos,
        "n_features": n_features,
        "v": v,
        "features_used": preX.index.tolist()
    }
    return out

ar_isoforms_clean = [x.split('-', 1)[0] for x in ARdutlist]
ir_isoforms_clean = [x.split('-', 1)[0] for x in IRdutlist]
ar_DUT_glist = set(transinfo[transinfo['transcript_id'].isin(ar_isoforms_clean)]['gene_name']) - {np.nan}
ir_DUT_glist = set(transinfo[transinfo['transcript_id'].isin(ir_isoforms_clean)]['gene_name']) - {np.nan}

##^^^ DUT gene or whole RI event## 
ar_ri_events = ar_psi.loc[(ar_psi["event"] == "RI"), "event_id"].dropna().unique().tolist() #&(ar_psi['gene_name'].isin(ar_DUT_glist))
ir_ri_events = ir_psi.loc[(ir_psi["event"] == "RI"), "event_id"].dropna().unique().tolist() #&(ir_psi['gene_name'].isin(ir_DUT_glist))
###^^##

out_ri = plot_ir_axis_main_figure_events(
    pre_mat=psi_pre_ri,
    post_mat=psi_post_ri,
    sampleinfo=sampleinfo,
    ar_event_list=ar_ri_events,
    ir_event_list=ir_ri_events,
    feature_mode="union",   # union / intersection / feature_list
    drop_nan_features=True,
    ellipse_nsig=2.0,
    ellipse_use_all_ir=True,
    use_zscore=True,
    title="Significant RI events: shift along IR axis"
)
df_pts_ri, comp_ri, ir_like_cluster_ri, moved_frac_ri = plot_unsupervised_clusters_on_pca(
    coord_pre=out_ri["coord_pre"],
    coord_post=out_ri["coord_post"],
    ar_samples=sampleinfo[sampleinfo["response"] == 1].index,
    ir_samples=sampleinfo[sampleinfo["response"] == 0].index,
    k=2,
    draw_cluster_ellipses=True,
    ellipse_nsig=1.2,
    title="RI PCA + k-means clusters (k=3)"
)

print(comp_ri)
print("IR-like cluster:", ir_like_cluster_ri, "AR moved fraction:", moved_frac_ri)

cluster_counts = pd.DataFrame(comp_ri)

cluster_counts.index.name = "cluster"
cluster_counts["total"] = cluster_counts.sum(axis=1)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("ticks")

def plot_cluster_composition_stacked(cluster_counts, save_path=None):
    df = cluster_counts.copy()

    # total column 있으면 제외
    cols = ["AR_pre", "AR_post", "IR_pre", "IR_post"]
    prop = df[cols].div(df[cols].sum(axis=1), axis=0)

    colors = { "AR_pre": "#FFEDA0","AR_post": "#FEB24C","IR_pre": "#D9F0D3","IR_post": "#5AAE61"}

    fig, ax = plt.subplots(figsize=(8, 3))  # 납작하게
    left = np.zeros(len(prop))

    ylabels = [f"C{idx}" if str(idx).isdigit() else str(idx) for idx in prop.index]

    for col in cols:
        ax.barh(
            y=np.arange(len(prop)),
            width=prop[col].values,
            left=left,
            color=colors[col],
            edgecolor="white",
            height=0.6,
            label=col
        )
        left += prop[col].values

    # bar 내부에 count 같이 쓰고 싶으면
    for i, idx in enumerate(df.index):
        cum = 0
        for col in cols:
            val = df.loc[idx, col]
            frac = prop.loc[idx, col]
            if frac > 0.08:  # 너무 작으면 글씨 생략
                ax.text(
                    cum + frac / 2, i, str(val),
                    ha="center", va="center",
                    fontsize=8, color="black"
                )
            cum += frac

    ax.set_yticks(np.arange(len(prop)))
    ax.set_yticklabels(ylabels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("proportion")
    ax.set_ylabel("")
    ax.legend(
        frameon=False, ncol=4, loc="upper center",
        bbox_to_anchor=(0.5, 1.16), fontsize=10,
        handlelength=1.2, columnspacing=1.0
    )

    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

plot_cluster_composition_stacked(cluster_counts, save_path="/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/gDUT_RIpsi_pca_kmeans_cluster_composition.pdf")
#%%































# %%
###^^ pca clustering (1) with AR DUT for both AR/IR ####
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_pca_on_transcripts(
    pre_TU_gene,
    post_TU_gene,
    ar_samples,
    ir_samples,
    transcript_list,
    title=""
):
    # --- subset transcripts
    tx = list(set(transcript_list) & set(pre_TU_gene.index))
    
    pre = pre_TU_gene.loc[tx,  list(ar_samples) + list(ir_samples)]
    post = post_TU_gene.loc[tx, list(ar_samples) + list(ir_samples)]

    # --- stack pre & post as separate samples
    pre_df = pre.copy()
    post_df = post.copy()

    pre_df.columns = [c + "_pre" for c in pre_df.columns]
    post_df.columns = [c + "_post" for c in post_df.columns]

    combined = pd.concat([pre_df, post_df], axis=1)

    # --- transpose: samples x transcripts
    X = combined.T.dropna(axis=1) #dropna(), fillna(0)

    # --- scale
    X_scaled = StandardScaler().fit_transform(X)
    
    print(X_scaled.shape)

    # --- PCA
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"], index=X.index)

    # --- metadata
    
    all_samples = list(ar_samples) + list(ir_samples)

    # sample 이름만 추출 (_pre/_post 제거)
    base_sample = pca_df.index.str.replace("_pre", "").str.replace("_post", "")

    pca_df["Group"] = np.where(
        base_sample.isin(ar_samples),
        "AR",
        "IR"
    )

    pca_df["Time"] = np.where(
        pca_df.index.str.contains("_post"),
        "Post",
        "Pre"
    )


    # --- plot
    plt.figure(figsize=(6,5))
    sns.scatterplot(
        data=pca_df,
        x="PC1", y="PC2",
        hue="Group",
        style="Time",
        palette={"AR":"#FF8D29", "IR":"#8AC509"},
        s=80
    )
    plt.title(title)
    plt.axhline(0, color="grey", alpha=0.3)
    plt.axvline(0, color="grey", alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("Explained variance:", pca.explained_variance_ratio_)
    return pca_df

# transcript groups
tx_class1 = list(set(ARdutlist) & set(class1))
tx_class3 = list(set(ARdutlist) & set(class3))


##########^ transcript filtering #############
with open("/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/ranksig_class1_geneset.txt", "r") as f:
    ranksig_c1_geneset = [line.strip() for line in f]
with open("/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/ranksig_class3_geneset.txt", "r") as f:
    ranksig_c3_geneset = [line.strip() for line in f]

with open("/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/ranksig_class1_top300.txt", "r") as f:
    ranksig_c1_top300 = [line.strip() for line in f]
with open("/home/jiye/jiye/copycomparison/GENCODEquant/SEV_prepost/merged_cov5_analysis/0210figures/ranksig_class3_top300.txt", "r") as f:
    ranksig_c3_top300 = [line.strip() for line in f]

def get_top_variable_within_group(
    pre_TU_gene,
    post_TU_gene,
    ar_samples,
    ir_samples,
    transcript_group,   # e.g. ARdut ∩ class1
    top_n
):
    all_samples = list(ar_samples) + list(ir_samples)

    tx = list(set(transcript_group) & set(pre_TU_gene.index))

    pre = pre_TU_gene.loc[tx, all_samples]
    post = post_TU_gene.loc[tx, all_samples]

    combined = pd.concat([pre, post], axis=1)

    var = combined.var(axis=1, skipna=True)

    top_tx = var.sort_values(ascending=False).head(top_n).index.tolist()

    print(f"Selected {len(top_tx)} transcripts from group (variance filtered)")
    return top_tx

cc_tx_class1 = [
    tx for tx in tx_class1
    if tx.split("-", 1)[1] in ranksig_c1_geneset
]

hrr_tx_class3 = [
    tx for tx in tx_class3
    if tx.split("-", 1)[1] in ranksig_c3_geneset
]

delta_TU_gene = filtered_trans.copy()
delta_TU_gene = delta_TU_gene[delta_TU_gene.index.str.split("-", n=1).str[-1].isin(proteincodinglist)]

pre_TU_gene = delta_TU_gene.iloc[:, 1::2]
post_TU_gene = delta_TU_gene.iloc[:, 0::2]

pre_TU_gene.columns = pre_TU_gene.columns.str[:-4] 
post_TU_gene.columns = post_TU_gene.columns.str[:-4] 


top_tx_class1 = get_top_variable_within_group(
    pre_TU_gene,
    post_TU_gene,
    ar_samples,
    ir_samples,
    tx_class1,
    top_n=500
)

top_tx_class3 = get_top_variable_within_group(
    pre_TU_gene,
    post_TU_gene,
    ar_samples,
    ir_samples,
    tx_class3,
    top_n=500
)

# Class 1 PCA
pca_c1 = run_pca_on_transcripts(
    pre_TU_gene,
    post_TU_gene,
    ar_samples,
    ir_samples,
    tx_class1, #^ ranksig_c1_top300, cc_tx_class1
    title="Class1 AR DUT top300"
)

# Class 3 PCA
pca_c3 = run_pca_on_transcripts(
    pre_TU_gene,
    post_TU_gene,
    ar_samples,
    ir_samples,
    tx_class3, #^ ranksig_c3_top300, hrr_tx_class3
    title="Class3 AR DUT top300"
)

def plot_pc1_distribution(pca_df, title):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(5,5))
    sns.boxplot(
        data=pca_df,
        x="Time",
        y="PC1",
        hue="Group",
        palette={"AR":"#FF8D29", "IR":"#8AC509"}
    )
    plt.title(title)
    plt.axhline(0, linestyle="--", color="grey", alpha=0.4)
    plt.tight_layout()
    plt.show()
    
plot_pc1_distribution(pca_c1, "PC1 distribution – Class1")
plot_pc1_distribution(pca_c3, "PC1 distribution – Class3")

# %%
###^^ umap + clustering ####

import umap

def run_umap_group_usage(
    pre_TU_gene,
    post_TU_gene,
    ar_samples,
    ir_samples,
    tx_list,
    title=""
):
    all_samples = list(ar_samples) + list(ir_samples)

    pre = pre_TU_gene.loc[tx_list, all_samples]
    post = post_TU_gene.loc[tx_list, all_samples]

    pre.columns = [c+"_pre" for c in pre.columns]
    post.columns = [c+"_post" for c in post.columns]

    combined = pd.concat([pre, post], axis=1)

    X = combined.T.dropna(axis=1) #fillna(0), dropna()

    X_scaled = StandardScaler().fit_transform(X)

    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(X_scaled)

    df = pd.DataFrame(embedding, columns=["UMAP1","UMAP2"], index=X.index)

    base_sample = df.index.str.replace("_pre","").str.replace("_post","")

    df["Group"] = np.where(base_sample.isin(ar_samples),"AR","IR")
    df["Time"] = np.where(df.index.str.contains("_post"),"Post","Pre")

    plt.figure(figsize=(6,5))
    sns.scatterplot(
        data=df,
        x="UMAP1", y="UMAP2",
        hue="Group",
        style="Time",
        palette={"AR":"#FFCC29","IR":"#81B214"},
        s=90
    )

    # trajectory lines
    for sample in all_samples:
        pre_name = sample + "_pre"
        post_name = sample + "_post"
        if pre_name in df.index and post_name in df.index:
            plt.plot(
                [df.loc[pre_name,"UMAP1"], df.loc[post_name,"UMAP1"]],
                [df.loc[pre_name,"UMAP2"], df.loc[post_name,"UMAP2"]],
                color="grey", alpha=0.3
            )

    plt.title(title)
    plt.tight_layout()
    plt.show()

    return df

df_class1 = run_umap_group_usage(
    pre_TU_gene,
    post_TU_gene,
    ar_samples,
    ir_samples,
    tx_class1, #^ ranksig_c1_top300, cc_tx_class1, tx_class1
    title="UMAP – ARdut ∩ Class1 (variance filtered)"
)

df_class3 = run_umap_group_usage(
    pre_TU_gene,
    post_TU_gene,
    ar_samples,
    ir_samples,
    tx_class3, #^ ranksig_c3_top300, hrr_tx_class3, tx_class3
    title="UMAP – ARdut ∩ Class1 (variance filtered)"
)

def add_condition_cols(df):
    df = df.copy()
    df["Condition"] = df["Group"] + "_" + df["Time"]  # AR_Pre, AR_Post, IR_Pre, IR_Post
    return df

from sklearn.cluster import KMeans

def cluster_umap(df, method="hdbscan", min_cluster_size=5, k=3, random_state=42):
    """
    df must have columns: UMAP1, UMAP2
    Returns df with a new column 'Cluster' (int), noise = -1 for hdbscan.
    """
    df = df.copy()
    X = df[["UMAP1", "UMAP2"]].values

    if method.lower() == "hdbscan":
        try:
            import hdbscan
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
            labels = clusterer.fit_predict(X)
        except ImportError:
            print("hdbscan not installed -> fallback to kmeans")
            labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(X)
    else:
        labels = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(X)

    df["Cluster"] = labels
    return df

def cluster_composition(df, cluster_col="Cluster", cond_col="Condition"):
    """
    Returns:
      counts: cluster x condition
      props : cluster x condition (row-normalized)
    """
    counts = pd.crosstab(df[cluster_col], df[cond_col])

    # ensure consistent column order if present
    desired = ["AR_Pre", "AR_Post", "IR_Pre", "IR_Post"]
    cols = [c for c in desired if c in counts.columns] + [c for c in counts.columns if c not in desired]
    counts = counts[cols]

    props = counts.div(counts.sum(axis=1), axis=0)
    return counts, props

def add_progress_projection(df):
    """
    Adds df['Progress'] = projection onto vector (mean of all Post) - (mean of all Pre).
    This is a simple pseudotime-like coordinate.
    """
    df = df.copy()

    pre_center  = df[df["Time"] == "Pre"][["UMAP1","UMAP2"]].mean().values
    post_center = df[df["Time"] == "Post"][["UMAP1","UMAP2"]].mean().values

    v = post_center - pre_center
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-12:
        # degenerate case
        df["Progress"] = df["UMAP1"].values
        return df

    v = v / v_norm
    X = df[["UMAP1","UMAP2"]].values
    df["Progress"] = (X - pre_center) @ v
    return df

def order_clusters_by_progress(df, cluster_col="Cluster"):
    """
    Returns a DataFrame with cluster mean Progress (sorted) and size.
    """
    summary = (
        df.groupby(cluster_col)
          .agg(n=("Progress","size"), mean_progress=("Progress","mean"))
          .sort_values("mean_progress")
          .reset_index()
    )
    return summary

import matplotlib.pyplot as plt
import seaborn as sns

def plot_umap_clusters(df, title="UMAP clusters"):
    plt.figure(figsize=(6,5))
    sns.scatterplot(data=df, x="UMAP1", y="UMAP2", hue="Cluster", palette="tab10", s=80)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_cluster_composition_heatmap(props, title="Cluster composition (proportions)"):
    plt.figure(figsize=(6,4))
    sns.heatmap(props, annot=True, fmt=".2f", cmap="Blues")
    plt.title(title)
    plt.ylabel("Cluster")
    plt.xlabel("Condition")
    plt.tight_layout()
    plt.show()

import hdbscan

df = add_condition_cols(df_class1)

# 1) cluster
df = cluster_umap(df, method="hdbscan", min_cluster_size=5, k=3)

# 2) add pseudotime-like progress coordinate
df = add_progress_projection(df)

# 3) composition
counts, props = cluster_composition(df)

print(counts)
print(props)

# 4) order clusters by progress
cluster_order = order_clusters_by_progress(df)
print(cluster_order)

# 5) plots
plot_umap_clusters(df, title="UMAP + Clusters")
plot_cluster_composition_heatmap(props, title="Cluster composition (AR/IR × Pre/Post)")

# reorder props by cluster order
ordered_clusters = cluster_order["Cluster"].tolist()
props_ordered = props.loc[ordered_clusters]

print(props_ordered)
plot_cluster_composition_heatmap(props_ordered, title="Composition ordered by Progress")

props_ordered = props_ordered.copy()
props_ordered["Post_frac"] = props_ordered.get("AR_Post",0) + props_ordered.get("IR_Post",0)
props_ordered["Pre_frac"]  = props_ordered.get("AR_Pre",0)  + props_ordered.get("IR_Pre",0)
print(props_ordered[["Post_frac","Pre_frac"]])

interval_df = sampleinfo.copy()
interval_df['Sample'] = interval_df.index
interval_df['Interval'] = interval_df['interval']

interval_df = interval_df[['Sample','Interval']]

def compute_pc1_shift(pca_df):
    shifts = []
    
    for sample in pca_df.index.str.replace("_pre","").str.replace("_post","").unique():
        pre_name = sample + "_pre"
        post_name = sample + "_post"
        
        if pre_name in pca_df.index and post_name in pca_df.index:
            shift = pca_df.loc[post_name,"PC1"] - pca_df.loc[pre_name,"PC1"]
            shifts.append((sample, shift))
    
    shift_df = pd.DataFrame(shifts, columns=["Sample","PC1_shift"])
    return shift_df

shift_df = compute_pc1_shift(pca_c1)
shift_df = shift_df.merge(interval_df, on="Sample")

shift_df['Group'] = shift_df['Sample'].apply(lambda x: "AR" if x in ar_samples else ("IR" if x in ir_samples else "Unknown"))

import statsmodels.formula.api as smf

# shift_df must contain:
# Sample, Shift, Interval, Group

model = smf.ols("PC1_shift ~ Interval * Group", data=shift_df).fit()
print(model.summary())

sns.lmplot(
    data=shift_df,
    x="Interval",
    y="PC1_shift",
    hue="Group",
    height=5,
    aspect=1.2
)

from scipy.stats import spearmanr

plt.figure(figsize=(5,4))
sns.regplot(data=shift_df, x="Interval", y="PC1_shift")
plt.title("Interval vs PC1 shift")
plt.show()

print(spearmanr(shift_df["Interval"], shift_df["PC1_shift"]))


def compute_umap_projection_shift(df_umap):
    df = df_umap.copy()

    # global pre/post 중심 계산
    pre_center  = df[df["Time"]=="Pre"][["UMAP1","UMAP2"]].mean().values
    post_center = df[df["Time"]=="Post"][["UMAP1","UMAP2"]].mean().values

    direction = post_center - pre_center
    direction = direction / np.linalg.norm(direction)

    df["Sample"] = df.index.str.replace("_pre","").str.replace("_post","")

    shifts = []

    for sample in df["Sample"].unique():
        pre_name  = sample + "_pre"
        post_name = sample + "_post"

        if pre_name in df.index and post_name in df.index:
            pre_vec  = df.loc[pre_name,  ["UMAP1","UMAP2"]].values
            post_vec = df.loc[post_name, ["UMAP1","UMAP2"]].values

            shift_vec = post_vec - pre_vec

            # projection onto global direction
            proj_shift = np.dot(shift_vec, direction)

            shifts.append((sample, proj_shift))

    shift_df = pd.DataFrame(shifts, columns=["Sample","UMAP_proj_shift"])
    return shift_df

from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt

shift_df = compute_umap_projection_shift(df_class1)

shift_df = shift_df.merge(interval_df, on="Sample")

plt.figure(figsize=(5,4))
sns.regplot(data=shift_df, x="Interval", y="UMAP_proj_shift")
plt.title("Interval vs UMAP projection shift")
plt.show()

print(spearmanr(shift_df["Interval"], shift_df["UMAP_proj_shift"]))



# %%
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logit
from scipy.stats import wilcoxon


def _logit_zscore(preX, postX, ir_samples, eps=1e-3, use_zscore=True):
    preX = preX.clip(eps, 1 - eps)
    postX = postX.clip(eps, 1 - eps)

    preT = pd.DataFrame(logit(preX.values), index=preX.index, columns=preX.columns)
    postT = pd.DataFrame(logit(postX.values), index=postX.index, columns=postX.columns)

    if use_zscore:
        mu = preT[ir_samples].mean(axis=1)
        sd = preT[ir_samples].std(axis=1).replace(0, np.nan)
        preT = preT.sub(mu, axis=0).div(sd, axis=0).fillna(0.0)
        postT = postT.sub(mu, axis=0).div(sd, axis=0).fillna(0.0)

    return preT, postT


def plot_main_ir_centered_decomposition(
    pre_TU,
    post_TU,
    sampleinfo,
    class1_list,
    ARdutlist,
    IRdutlist,
    response_col="response",
    ar_value=1,
    ir_value=0,
    eps=1e-3,
    use_zscore=True,
    drop_nan_features=True,
    title="Shift toward IR axis with concurrent orthogonal remodeling"
):
    # -----------------------
    # 1) feature set
    # -----------------------
    dut_union = set(ARdutlist) | set(IRdutlist)
    feat = sorted(set(class1_list) & dut_union & set(pre_TU.index) & set(post_TU.index))
    if len(feat) < 5:
        raise ValueError(f"Too few features after intersection: {len(feat)}")

    preX = pre_TU.loc[feat].copy()
    postX = post_TU.loc[feat].copy()

    # -----------------------
    # 2) align samples
    # -----------------------
    common_samples = preX.columns.intersection(postX.columns)
    preX = preX[common_samples]
    postX = postX[common_samples]

    resp = sampleinfo[response_col]
    ar_samples = resp[resp == ar_value].index.intersection(common_samples)
    ir_samples = resp[resp == ir_value].index.intersection(common_samples)

    if len(ar_samples) < 3 or len(ir_samples) < 3:
        raise ValueError(f"Not enough samples: AR={len(ar_samples)}, IR={len(ir_samples)}")

    # -----------------------
    # 3) NaN handling
    # -----------------------
    if drop_nan_features:
        ok = (~preX.isna().any(axis=1)) & (~postX.isna().any(axis=1))
        preX = preX.loc[ok]
        postX = postX.loc[ok]
    else:
        preX = preX.fillna(0.0)
        postX = postX.fillna(0.0)

    n_features = int(preX.shape[0])
    if n_features < 5:
        raise ValueError(f"Too few features after NaN filtering: {n_features}")

    # -----------------------
    # 4) logit + zscore (IR_pre reference)
    # -----------------------
    preT, postT = _logit_zscore(preX, postX, ir_samples, eps=eps, use_zscore=use_zscore)

    # -----------------------
    # 5) define IR axis vhat
    #    v = mean(IR_pre) - mean(AR_pre)
    # -----------------------
    mu_AR = preT[ar_samples].mean(axis=1)
    mu_IR = preT[ir_samples].mean(axis=1)

    v = mu_IR - mu_AR
    vhat = v / (np.linalg.norm(v.values) + 1e-12)

    # -----------------------
    # 6) coordinates: IR-centered t_IR and orthogonal residual r
    # -----------------------
    # IR-centered coordinate (overshoot interpretable)
    def t_ir(X):
        Y = X.sub(mu_IR, axis=0)          # centered at IR
        return Y.T.dot(vhat)              # samples

    # residual distance to the axis through IR centroid
    def r_ir(X, tvals):
        Y = X.sub(mu_IR, axis=0)
        R = Y - np.outer(vhat.values, tvals.values)
        return np.sqrt((R ** 2).sum(axis=0))

    tIR_pre = t_ir(preT)
    tIR_post = t_ir(postT)
    r_pre = r_ir(preT, tIR_pre)
    r_post = r_ir(postT, tIR_post)

    # AR-only paired deltas
    tpre_ar = tIR_pre.loc[ar_samples]
    tpost_ar = tIR_post.loc[ar_samples]
    rpre_ar = r_pre.loc[ar_samples]
    rpost_ar = r_post.loc[ar_samples]

    delta_t = tpost_ar - tpre_ar
    delta_r = rpost_ar - rpre_ar

    # tests
    p_dt = wilcoxon(delta_t.values, alternative="less").pvalue
    # NOTE: because t is IR-centered with v pointing AR->IR, AR samples should move "toward 0",
    # which means t should increase toward 0 (become less negative). Depending on sign, "less" vs "greater" flips.
    # We'll make it robust by explicitly testing "distance to 0" improvement below.

    # Better: test improvement in absolute IR-centered distance along axis:
    abs_pre = np.abs(tpre_ar.values)
    abs_post = np.abs(tpost_ar.values)
    p_abs = wilcoxon(abs_post - abs_pre, alternative="less").pvalue  # want |t| to decrease

    p_dr_2s = wilcoxon(delta_r.values, alternative="two-sided").pvalue
    p_dr_gt = wilcoxon(delta_r.values, alternative="greater").pvalue

    # -----------------------
    # 7) overshoot and "% progress to IR"
    # -----------------------
    overshoot_pct = float((tpost_ar > 0).mean())  # AR_post beyond IR centroid along axis

    # % progress to IR along axis, anchored at AR_pre mean position:
    # Let baseline = mean tIR over AR_pre (usually negative).
    # Define progress fraction per sample:
    #   frac = (t_post - t_pre) / (0 - t_pre)
    # so reaching IR centroid along axis gives frac=1; overshoot gives >1.
    denom = (0.0 - tpre_ar).replace(0, np.nan)
    frac_to_ir = (tpost_ar - tpre_ar) / denom
    frac_to_ir = frac_to_ir.replace([np.inf, -np.inf], np.nan).dropna()
    mean_frac_to_ir = float(frac_to_ir.mean()) if len(frac_to_ir) else np.nan
    med_frac_to_ir = float(frac_to_ir.median()) if len(frac_to_ir) else np.nan

    # -----------------------
    # 8) main figure: 3 panels
    # -----------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # Panel A: IR-centered axis coordinate (t_IR) with IR=0 line
    ax = axes[0]
    ax.scatter(np.zeros(len(tpre_ar)), tpre_ar.values, label="AR_pre")
    ax.scatter(np.ones(len(tpost_ar)), tpost_ar.values, label="AR_post")
    for s in ar_samples:
        ax.plot([0, 1], [tpre_ar.loc[s], tpost_ar.loc[s]], alpha=0.35, linewidth=1)
    ax.axhline(0, linestyle="--", linewidth=1)  # IR centroid
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["AR_pre", "AR_post"])
    ax.set_ylabel("t_IR (IR-centered axis coordinate)")
    ax.set_title("IR-centered axis coordinate\n(0 = IR centroid; >0 overshoot)")
    ax.grid(alpha=0.2)

    # Panel B: improvement along axis (|t| decreases)
    ax = axes[1]
    ax.boxplot((abs_post - abs_pre), showfliers=False)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xticks([1])
    ax.set_xticklabels(["Δ|t_IR| (post - pre)"])
    ax.set_ylabel("Change in axis distance to IR")
    ax.set_title(f"Axis convergence (|t|↓)\nWilcoxon p={p_abs:.2e}")
    ax.grid(alpha=0.2)

    # Panel C: orthogonal residual r
    ax = axes[2]
    ax.scatter(np.zeros(len(rpre_ar)), rpre_ar.values, label="AR_pre")
    ax.scatter(np.ones(len(rpost_ar)), rpost_ar.values, label="AR_post")
    for s in ar_samples:
        ax.plot([0, 1], [rpre_ar.loc[s], rpost_ar.loc[s]], alpha=0.35, linewidth=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["AR_pre", "AR_post"])
    ax.set_ylabel("r (orthogonal residual)")
    ax.set_title(f"Off-axis remodeling\nΔr two-sided p={p_dr_2s:.2e}\n(Δr>0 p={p_dr_gt:.2e})")
    ax.grid(alpha=0.2)

    # global annotation
    txt = (
        f"n_features={n_features}\n"
        f"AR n={len(ar_samples)}, IR n={len(ir_samples)}\n"
        f"overshoot (t_IR>0) in AR_post: {overshoot_pct:.2f}\n"
        f"%progress to IR (mean/median): {mean_frac_to_ir:.2f}/{med_frac_to_ir:.2f}"
    )
    fig.text(0.01, 0.99, txt, ha="left", va="top", fontsize=10)

    fig.suptitle(title, y=1.06)
    plt.tight_layout()
    plt.show()

    stats = {
        "n_features": n_features,
        "n_AR": int(len(ar_samples)),
        "n_IR": int(len(ir_samples)),
        "p_axis_convergence_abs_less": float(p_abs),
        "p_delta_r_two_sided": float(p_dr_2s),
        "p_delta_r_greater": float(p_dr_gt),
        "overshoot_pct_AR_post": float(overshoot_pct),
        "mean_frac_to_ir": float(mean_frac_to_ir),
        "median_frac_to_ir": float(med_frac_to_ir),
    }

    out = {
        "fig": fig,
        "tIR_pre": tIR_pre,
        "tIR_post": tIR_post,
        "r_pre": r_pre,
        "r_post": r_post,
        "vhat": vhat,
        "mu_IR": mu_IR,
        "mu_AR": mu_AR,
        "stats": stats,
        "frac_to_ir_AR": frac_to_ir
    }
    return out

out_main = plot_main_ir_centered_decomposition(
    pre_TU=pre_TU_gene,
    post_TU=post_TU_gene,
    sampleinfo=sampleinfo,
    class1_list=class1,
    ARdutlist=ARdutlist,
    IRdutlist=IRdutlist,
    drop_nan_features=True,
    title="Class1 ∩ (AR/IR DUT union): partial IR-axis convergence with off-axis remodeling"
)

print(out_main["stats"])
# %%
