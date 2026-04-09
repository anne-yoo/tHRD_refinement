# %%
import pandas as pd
import sys
prop = sys.argv[1]
#%%
# def parse_gtf(gtf_file):
#     """
#     Extract exon information
#     """
#     columns = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
#     # gtf_data = pd.read_csv(gtf_file, sep='\t', comment='#', names=columns)
#     gtf_data = pd.read_csv(gtf_file, sep='\t')
#     exons = gtf_data[gtf_data['feature'] == 'exon']
#     return exons


def parse_gtf(gtf_file):
    
    columns = ['seqname', 'source', 'feature', 'start', 'end', 
            'score', 'strand', 'frame', 'attribute']
    
    # 파일을 한 줄씩 읽어서 탭으로 분리하여 DataFrame 생성
    data = []
    with open(gtf_file, 'r') as f:
        for line in f:
            if not line.startswith('#'):  # 주석 무시
                fields = line.strip().split('\t')
                if len(fields) == 9:  # 9개의 열로 구성된 경우에만 처리
                    data.append(fields)
    
    # DataFrame 생성
    gtf_data = pd.DataFrame(data, columns=columns)
    
    # exon 행만 필터링
    exons = gtf_data[gtf_data['feature'] == 'exon']
    
    return exons


def extract_transcript_junctions(exons):
    """
    Extract junction chain
    """
    ## Extract transcript ID from attribute tab
    exons['start'] = exons['start'].astype(int)
    exons['end'] = exons['end'].astype(int)
    exons['transcript_id'] = exons['attribute'].str.extract('transcript_id "([^"]+)')
    ## Sort exons per each tranbscript
    exons_sorted = exons.sort_values(by=['transcript_id', 'start'])
    transcript_junctions = {}
    ## Make junction chain
    for transcript_id, group in exons_sorted.groupby('transcript_id'):
        if group.shape[0] > 1:  # Skip mono-exon
            junction_chain = []
            previous_end = None
            for _, exon in group.iterrows():
                if previous_end is not None:
                    junction_chain.append((previous_end, exon['start'] - 1))  # -1 to get the junction between exons
                previous_end = exon['end']
            transcript_junctions[transcript_id] = junction_chain
    return transcript_junctions

def save_junctions_to_file(transcript_junctions, output_file):
    """
    Save junction data
    """
    with open(output_file, 'w') as f:
        f.write("Transcript_ID,Junction_Chain\n")
        for transcript_id, junction_chain in transcript_junctions.items():
            junction_chain_str = ";".join([f"{start}-{end}" for start, end in junction_chain])
            f.write(f"{transcript_id},{junction_chain_str}\n")
            
def main():
    """
    DIR = path of GTF
    gtf_file = GTF file
    output_file = Prefix name of output
    """
    # met = "Std"

    #DIR = "/home/jiye/jiye/refdata/"
    DIR = "/home/jiye/jiye/nanopore/gtfcompare/"
    
    gtf_file = "isoquant_merged.gtf"
    output_file = 'isoquant_junction_chain.txt'
    exons = parse_gtf(DIR+gtf_file)
    transcript_junctions = extract_transcript_junctions(exons)
    save_junctions_to_file(transcript_junctions, DIR+output_file)
if __name__ == "__main__":
    main()


###
#input: GTF
#output: unique_junction_chain.txt
###
# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from upsetplot import UpSet
# %%
def load_junction(met):
    if met == "Gen":
        ## Reference annotation
        # gen_DIR = "/flashscratch/kangh/OPT_visium/ONT/inputs/"
        gen_DIR = "/home/jiye/jiye/refdata/"
        data = pd.read_csv(gen_DIR+"unique_junction_chain.txt",
                            sep=",")
    else:
        ## Custom annotation from Long-read
        # DIR="/home/kangh/lab-server/Opt_visium/ONT/gtf/flair/{}_1st_SQANTI3/".format(met)
        DIR="/home/jiye/jiye/nanopore/gtfcompare/"
        data = pd.read_csv(DIR+met+"_unique_junction_chain.txt",
                            sep=",")
    val_list = [i for i in data["Junction_Chain"].tolist()]
    return val_list,data
# std,std_df = load_junction("Std")
# opt1,op1_df = load_junction("Opt1")
# opt2,op2_df = load_junction("Opt2")

#std,std_df = load_junction("isoqant")  
opt1,op1_df = load_junction("291_stringtie")
opt2,op2_df = load_junction("bambu_whole")
opt3,op3_df = load_junction("isoquant") 
gen,gen_df = load_junction("Gen")


#%%
#
def Over(query,query2):
    """
    Check Overlaps using the junction chain info
    """
    shared = list(set(query) & set(query2))
    print(f"{len(shared)} same junction chains")
    return len(shared)

#std_gen = Over(std,gen)
op1_gen = Over(opt1,gen)
op2_gen = Over(opt2,gen)
op3_gen = Over(opt3,gen)

print(f"Genecode chains: {len(set(gen))}")
#print(f"Std chains: {len(set(std))}")
print(f"Opt1 chains: {len(set(opt1))}")
print(f"Opt2 chains: {len(set(opt2))}")
print(f"Opt3 chains: {len(set(opt3))}")

genlen = len(gen)
df = [op1_gen, op2_gen, op3_gen]
df = pd.DataFrame({
    'group': ['Stringtie','Bambu','Isoquant'],
    'shared': [op1_gen, op2_gen, op3_gen]  # 각 그룹의 접합부 일치 개수
})

plt.figure(figsize=(7, 1.5))
sns.barplot(data=df, x='shared', y='group', palette={
    #"reference gtf": "#E8DE1B",
    "30_stringtie": "#FF7F0E", 
    "100_stringtie": "#1FB426FF", 
    "150_stringtie": "#1F77B4",
    "291_stringtie": "#5E0787",
})
sns.despine()

plt.xlim([200000,210000])
plt.ylabel("Methods")
plt.xlabel("Same junction chains \nwith Gencode")
plt.legend(frameon=False, bbox_to_anchor=(1.01, 1.01), loc="upper left")
plt.show()

# %%

from venny4py.venny4py import *
sets = {
    '30 samples': set(std),
    '100 samples': set(opt1),
    '150 samples': set(opt2),
    '291 samples': set(opt3)}
    
aa = venny4py(sets=sets)
plt.savefig('/home/jiye/jiye/nanopore/gtfcompare/stringtiesample/4sets.pdf', dpi=300,format='pdf', bbox_inches='tight',transparent=True, pad_inches=0.1)

# %%
#%%%
############^^^ only novel transcripts ############################
op1_novel = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/sqantioutput/stringtie_novel_list.txt', sep='\t')
op2_novel = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/sqantioutput/bambu_novel_list.txt', sep='\t')
op3_novel = pd.read_csv('/home/jiye/jiye/nanopore/202411_analysis/sqantioutput/isoquant_novel_list.txt', sep='\t')
op1_novel = list(op1_novel['transcript_id'])
op2_novel = list(op2_novel['transcript_id'])
op3_novel = list(op3_novel['transcript_id'])

op1_novel_df = op1_df[op1_df['Transcript_ID'].isin(op1_novel)]
op2_novel_df = op2_df[op2_df['Transcript_ID'].isin(op2_novel)]
op3_novel_df = op3_df[op3_df['Transcript_ID'].isin(op3_novel)]

opt1_novel = set(op1_novel_df['Junction_Chain'])
opt2_novel = set(op2_novel_df['Junction_Chain'])
opt3_novel = set(op3_novel_df['Junction_Chain'])

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3
from matplotlib_venn import venn2


    
plt.rcParams["font.family"] = "Arial"
plt.rcParams.update({
'axes.titlesize': 20,     # 제목 글꼴 크기
'axes.labelsize': 20,     # x, y축 라벨 글꼴 크기
'xtick.labelsize': 20,    # x축 틱 라벨 글꼴 크기

'ytick.labelsize': 20,    # y축 틱 라벨 글꼴 크기
'legend.fontsize': 20,
'legend.title_fontsize': 20, # 범례 글꼴 크기
'figure.titlesize': 20    # figure 제목 글꼴 크기
})
sns.set_style("white")
# Create a horizontal bar plot
plt.figure(figsize=(8,8))
#aa = venn3([opt1_novel, opt2_novel, opt3_novel], ('Stringtie', 'Bambu', 'Isoquant'))
aaa = venn2([opt1_novel, opt2_novel], ('Stringtie', 'Bambu'), set_colors=('skyblue', 'lightgreen'))
#aaa = venn2([opt1_novel, opt3_novel], ('Stringtie', 'Isoquant'), set_colors=('skyblue', 'lightgreen'))

# Set font sizes for the Venn diagram labelsㄴ
for text in aaa.set_labels:
    if text:  # Check if the text is not None
        text.set_fontsize(18)
for text in aaa.subset_labels:
    if text:  # Check if the text is not None
        text.set_fontsize(18)

plt.savefig('/home/jiye/jiye/nanopore/202411_analysis/sqantioutput/noveltranscript_comparison_stringtiebambu.pdf', dpi=300,format='pdf', bbox_inches='tight',ad_inches=0.1, transparent=True)
plt.show()

# %%
