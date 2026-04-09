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
#^^ AR vs. IR TU distribution jointplot ###############

AR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t', index_col=0)
IR_dut = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202403_analysis/DUT/nonresponder_stable_DUT_Wilcoxon.txt', sep='\t', index_col=0)

ARdutlist = AR_dut.loc[(AR_dut['p_value']<0.05) & (np.abs(AR_dut['log2FC'])>1.5)].index.to_list()
IRdutlist = IR_dut.loc[(IR_dut['p_value']<0.05) & (np.abs(IR_dut['log2FC'])>1.5)].index.to_list()

sampleinfo = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/sorted_discovery_fileinfo.txt', sep='\t')
ARlist = list(set(sampleinfo.loc[(sampleinfo['response']==1),'sample_id']))
IRlist = list(set(sampleinfo.loc[(sampleinfo['response']==0),'sample_id']))

TU = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/filtered/80_discovery_TU.txt', sep='\t', index_col=0)
TU.columns = TU.columns.str[:-4]
preTU = TU.iloc[:,1::2]
postTU = TU.iloc[:,0::2]

ARpre = preTU.loc[ARdutlist,ARlist]
ARpost = postTU.loc[ARdutlist,ARlist]
IRpre = preTU.loc[IRdutlist,IRlist]
IRpost = postTU.loc[IRdutlist,IRlist]

#%%

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_combined_dut_scatter(preTU, postTU, 
                               ARdutlist, ARlist, AR_palette, 
                               IRdutlist, IRlist, IR_palette):
    # AR 평균 TU 계산
    pre_AR_all = preTU.loc[:, ARlist].mean(axis=1)
    post_AR_all = postTU.loc[:, ARlist].mean(axis=1)
    pre_AR_dut = preTU.loc[ARdutlist, ARlist].mean(axis=1)
    post_AR_dut = postTU.loc[ARdutlist, ARlist].mean(axis=1)

    # IR 평균 TU 계산
    pre_IR_all = preTU.loc[:, IRlist].mean(axis=1)
    post_IR_all = postTU.loc[:, IRlist].mean(axis=1)
    pre_IR_dut = preTU.loc[IRdutlist, IRlist].mean(axis=1)
    post_IR_dut = postTU.loc[IRdutlist, IRlist].mean(axis=1)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    # AR plot
    ax = axes[0]
    ax.scatter(pre_AR_all.values, post_AR_all.values, s=2, color='black', alpha=0.2)
    ax.scatter(pre_AR_dut.values, post_AR_dut.values, s=4, color=AR_palette["Pre"], alpha=0.4)
    ax.scatter(pre_AR_dut.values, post_AR_dut.values, s=4, color=AR_palette["Post"], alpha=0.4)
    x = np.linspace(0, 1, 100)
    ax.plot(x, x, color='gray', linewidth=1)
    ax.set_title("AR")
    ax.set_xlabel("Before Treatment (TU)")
    ax.set_ylabel("After Treatment (TU)")

    # IR plot
    ax = axes[1]
    ax.scatter(pre_IR_all.values, post_IR_all.values, s=2, color='black', alpha=0.2)
    ax.scatter(pre_IR_dut.values, post_IR_dut.values, s=4, color=IR_palette["Pre"], alpha=0.4)
    ax.scatter(pre_IR_dut.values, post_IR_dut.values, s=4, color=IR_palette["Post"], alpha=0.4)
    ax.plot(x, x, color='gray', linewidth=1)
    ax.set_title("IR")
    ax.set_xlabel("Before Treatment (TU)")

    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.tight_layout()
    plt.savefig(f'/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/scatterplot_ARIR.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
AR_palette = {"Pre": "#FFCC29", "Post": "#FFAE4B"}
IR_palette = {"Pre": "#81B214", "Post": "#5F8805"}

plot_combined_dut_scatter(preTU, postTU, 
                          ARdutlist, ARlist, AR_palette, 
                          IRdutlist, IRlist, IR_palette)


from scipy.stats import fisher_exact

def test_dut_enrichment(dut_AR, total_AR, dut_IR, total_IR):
    contingency = [
        [dut_AR, total_AR - dut_AR],
        [dut_IR, total_IR - dut_IR]
    ]
    oddsratio, pval = fisher_exact(contingency, alternative='two-sided')
    print(f"Fisher's exact test p = {pval:.4g} (odds ratio = {oddsratio:.2f})")
    return pval, oddsratio

# 전체 transcript 수 (AR / IR에서의 대상 transcript 개수 기준)
total_AR = preTU.loc[:, ARlist].shape[0]
total_IR = preTU.loc[:, IRlist].shape[0]

test_dut_enrichment(
    dut_AR=19167,
    total_AR=total_AR,
    dut_IR=2715,
    total_IR=total_IR
)

# %%
#####^^ AR vs. IR network ##############

plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 13,    # x축 틱 라벨 글꼴 크기

'ytick.labelsize': 13,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 13,
'legend.title_fontsize': 13, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})


with open('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/hrr_translist.txt', 'r') as file:
    hrr_transcripts = [line.strip() for line in file]
with open('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/cycle_translist.txt', 'r') as file:
    cycle_transcripts = [line.strip() for line in file]
with open('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/pathway_translist.txt', 'r') as file:
    pathway_transcripts = [line.strip() for line in file]

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr

def build_and_plot_spearman_network(preTU, postTU, trans_list, sample_list, title="Network", threshold=0.3):
    # ΔTU 계산
    delta_TU = postTU.loc[trans_list, sample_list] - preTU.loc[trans_list, sample_list]

    # Spearman correlation 계산
    corr_matrix, _ = spearmanr(delta_TU.T)
    corr_df = pd.DataFrame(corr_matrix, index=trans_list, columns=trans_list)

    # threshold 이상인 상관관계만 간선으로 추가
    edges = []
    for i in range(len(trans_list)):
        for j in range(i + 1, len(trans_list)):
            corr_val = corr_df.iloc[i, j]
            if abs(corr_val) >= threshold:
                edges.append((trans_list[i], trans_list[j], corr_val))

    # NetworkX 그래프 생성
    G = nx.Graph()
    G.add_nodes_from(trans_list)
    for a, b, weight in edges:
        G.add_edge(a, b, weight=weight)

    # ✅ degree centrality로 노드 크기 계산
    centrality = nx.degree_centrality(G)
    node_sizes = [200 + centrality[node]*1200 for node in G.nodes()]

    # 시각화
    plt.figure(figsize=(9, 8))
    pos = nx.spring_layout(G, seed=42)
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    edge_colors = ['red' if w > 0 else 'blue' for w in edge_weights]

    nx.draw_networkx_nodes(G, pos, node_color='gray', node_size=node_sizes, alpha=0.85)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=[abs(w)*2 for w in edge_weights])
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title(title)
    plt.axis('off')
    plt.show()


# ✅ 실행 예시
# AR 그룹
build_and_plot_spearman_network(preTU, postTU, hrr_transcripts, ARlist, "AR - HRR")
# build_and_plot_spearman_network(preTU, postTU, cycle_transcripts, ARlist, "AR - Cell Cycle")
# build_and_plot_spearman_network(preTU, postTU, pathway_transcripts, ARlist, "AR - Pro-survival Pathway")

# IR 그룹
# build_and_plot_spearman_network(preTU, postTU, hrr_transcripts, IRlist, "IR - HRR")
# build_and_plot_spearman_network(preTU, postTU, cycle_transcripts, IRlist, "IR - Cell Cycle")
# build_and_plot_spearman_network(preTU, postTU, pathway_transcripts, IRlist, "IR - Pro-survival Pathway")

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr

plt.rcParams.update({
'axes.titlesize': 13,     # 제목 글꼴 크기
'axes.labelsize': 13,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 13,    # x축 틱 라벨 글꼴 크기

'ytick.labelsize': 13,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 9,
'legend.title_fontsize': 9, # 범례 글꼴 크기
'figure.titlesize': 15    # figure 제목 글꼴 크기
})

def build_colored_network(preTU, postTU, axis_dict, sample_list, title="Combined Network", threshold=0.):
    # 모든 transcript 리스트 (중복 제거)
    all_transcripts = list(set(sum(axis_dict.values(), [])))

    # ΔTU 계산
    delta_TU = postTU.loc[all_transcripts, sample_list] - preTU.loc[all_transcripts, sample_list]

    # Spearman 상관계수 계산
    corr_matrix, _ = spearmanr(delta_TU.T)
    corr_df = pd.DataFrame(corr_matrix, index=all_transcripts, columns=all_transcripts)

    # 상관계수 기반 edge 추출
    edges = []
    for i in range(len(all_transcripts)):
        for j in range(i + 1, len(all_transcripts)):
            corr_val = corr_df.iloc[i, j]
            if abs(corr_val) >= threshold:
                edges.append((all_transcripts[i], all_transcripts[j], corr_val))

    # 그래프 생성
    G = nx.Graph()
    G.add_nodes_from(all_transcripts)
    for a, b, weight in edges:
        G.add_edge(a, b, weight=weight)

    # 노드 색상 설정 (axis별 구분)
    node_colors = []
    color_map = {
        "Homologous Recombination Restoration": "#F94144",
        "Cell Cycle Checkpoint Activation": "#F9C74F",
        "Pro-Survival Pathway Upregulation": "#90BE6D",
    }
    node_color_dict = {}
    for node in G.nodes():
        label = "Other"
        for axis, tlist in axis_dict.items():
            if node in tlist:
                label = axis
                break
        node_colors.append(color_map[label])
        node_color_dict[node] = label

    # 중심성 기반 노드 크기
    centrality = nx.degree_centrality(G)
    node_sizes = [200 + centrality[n]*2500 for n in G.nodes()]

    # 네트워크 시각화
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42, k=0.7)
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    edge_colors = ['#ED4242' if w > 0 else '#243BEA' for w in edge_weights]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=[abs(w)*1.5 for w in edge_weights])
    nx.draw_networkx_labels(G, pos, font_size=7)

    plt.title(title)
    plt.axis('off')

    # 범례
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=axis) for axis, color in color_map.items()]
    plt.legend(handles=legend_elements, loc='lower left')

    plt.tight_layout()
    plt.savefig(f'/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/network_{title}.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# transcript 리스트들을 dictionary로 모음
axis_dict = {
    "Homologous Recombination Restoration": hrr_transcripts,
    "Cell Cycle Checkpoint Activation": cycle_transcripts,
    "Pro-Survival Pathway Upregulation": pathway_transcripts,
}

# 예: AR에서 combined 네트워크 보기
build_colored_network(preTU, postTU, axis_dict, ARlist, title="AR", threshold=0.5)

# 예: IR에서도
build_colored_network(preTU, postTU, axis_dict, IRlist, title="IR", threshold=0.5)


# %%
###^^^ Transcript info: coding potential & NMD ##############
cp = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/transinfo/merged_output.txt', sep='\t', index_col=0)
majorminor = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/2024majorminorinfo.txt', sep='\t', index_col=1)
nmd = pd.read_csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/transinfo/whole_transfeat_NMD_features.csv', sep=',', index_col=0)

majorminor = majorminor.merge(cp, left_index=True, right_index=True, how='left')

#%%
from statannotations.Annotator import Annotator

plt.figure(figsize=(4, 5))
ax = sns.boxplot(data=majorminor, x='type', y='coding_probability',showfliers=False, palette={'major':'#AAAAAA', 'minor':'#C9C9C9'})
plt.ylabel("coding potential")
plt.xlabel("")
plt.tight_layout()
sns.despine()

pairs = [('major', 'minor')]

annotator = Annotator(
    ax, pairs,
    data=majorminor,
    x='type', y='coding_probability',
    order=['major', 'minor'],
)
annotator.configure(test='Mann-Whitney', text_format='simple', loc='inside')
annotator.apply_and_annotate()
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/majorminor_codingpotential.pdf', dpi=300, bbox_inches='tight')
plt.show()

# %%
nmdlist = list(set(nmd.index))
majorminor['NMD'] = majorminor.index.isin(nmdlist)

plt.figure(figsize=(4, 5))
ax = sns.barplot(data=majorminor, x='type', y='NMD', palette={'major':'#AAAAAA', 'minor':'#AAAAAA'}, errorbar=None) #palette={'major':'#AAAAAA', 'minor':'#C9C9C9'}
plt.ylabel("NMD proportion")
plt.xlabel("")
plt.tight_layout()
sns.despine()
plt.savefig('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/ARsubtype/majorminor_NMD.pdf', dpi=300, bbox_inches='tight')
plt.show()
# %%
