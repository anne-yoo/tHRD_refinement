#%%
import os
import re
import pandas as pd

# =========================
# Config
# =========================
BASE_PATH = "/home/jiye/jiye/darkproteome/data/HG_JH/PRICE/bam4price/chrfilter_mergeCIT/PRICE_output/"
FILES = {
    "B_cell": "B_cell_lineage.orfs.tsv",
    "Colon": "Colon.orfs.tsv",
    "Melan_Lin": "Melanocyte_lineage.orfs.tsv",
    "All_merged": "All_merged.orfs.tsv",
}

OUT_DIR = os.path.join(BASE_PATH, "merged_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# PRICE 핵심 컬럼
CORE_COLS = [
    "Gene", "Id", "Location", "Candidate Location", "Codon",
    "Type", "Start", "Range", "p value"
]

# =========================
# Helpers
# =========================
def clean_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def parse_location_full(loc: str):
    """
    Parse PRICE Location string like:
      1+:236306192-236306336|236332005-236332055|...
      12-:9094413-9094536|9095011-9095138|...
    Returns:
      chrom, strand, blocks(list of (start,end)), span_start, span_end
    """
    loc = clean_str(loc)
    m = re.match(r"^(.+?)([+-]):(.+)$", loc)
    if not m:
        raise ValueError(f"Could not parse Location: {loc}")

    chrom, strand, coord_str = m.groups()
    blocks = []
    for block in coord_str.split("|"):
        s, e = block.split("-")
        blocks.append((int(s), int(e)))

    span_start = min(s for s, _ in blocks)
    span_end = max(e for _, e in blocks)
    return chrom, strand, blocks, span_start, span_end

def make_display_id(row):
    """
    nuORFdb-like display label:
      >ENST00000454784.8_2_12:40421230-40485543:+|GN=MUC19 Type|myDB_v1
    여기서는 PRICE Id + condensed span 사용
    """
    chrom, strand, blocks, span_start, span_end = parse_location_full(row["Location"])

    gene = clean_str(row["Gene"])
    orf_type = clean_str(row["Type"])
    price_id = clean_str(row["Id"])

    # gene symbol이 아니라 Gene id가 들어있으므로 일단 그대로 사용
    return f"{price_id}_{chrom}:{span_start}-{span_end}:{strand}|GN={gene} {orf_type}|custom_hier_v1"

def make_orf_key(row):
    # exact ORF key
    return f"{clean_str(row['Location'])}|{clean_str(row['Codon']).upper()}"

def choose_best_row(group: pd.DataFrame) -> pd.Series:
    """
    exact same orf_key 내에서 대표 row 선택.
    p value 가장 작은 행 우선.
    """
    g = group.sort_values(["p value", "Start", "Range"], ascending=[True, False, False]).copy()
    best = g.iloc[0].copy()

    best["Source_Clade"] = ",".join(sorted(set(g["Source_Clade"].astype(str))))
    best["n_support_rows"] = len(g)

    # 참고용 summary
    best["all_p_values"] = ",".join(map(str, g["p value"].tolist()))
    best["all_codons"] = ",".join(sorted(set(g["Codon"].astype(str))))
    best["all_ids"] = "||".join(sorted(set(g["Id"].astype(str))))
    return best

# =========================
# Load + merge
# =========================
all_dfs = []

for source_name, fname in FILES.items():
    path = os.path.join(BASE_PATH, fname)
    df = pd.read_csv(path, sep="\t")

    missing = [c for c in CORE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{fname} missing columns: {missing}")

    df = df[CORE_COLS].copy()
    df["Source_Clade"] = source_name

    # normalize
    df["Gene"] = df["Gene"].map(clean_str)
    df["Id"] = df["Id"].map(clean_str)
    df["Location"] = df["Location"].map(clean_str)
    df["Candidate Location"] = df["Candidate Location"].map(clean_str)
    df["Codon"] = df["Codon"].map(lambda x: clean_str(x).upper())
    df["Type"] = df["Type"].map(clean_str)

    # numeric normalize
    for col in ["Start", "Range", "p value"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["display_id"] = df.apply(make_display_id, axis=1)
    df["orf_key"] = df.apply(make_orf_key, axis=1)

    all_dfs.append(df)

combined_df = pd.concat(all_dfs, ignore_index=True)
#combined_df.to_csv(os.path.join(OUT_DIR, "01_combined_all_rows.tsv"), sep="\t", index=False)

print("Combined rows:", len(combined_df))
print("Unique orf_key:", combined_df["orf_key"].nunique())

# =========================
# Exact duplicate collapse by Location|Codon
# =========================
collapsed_df = (
    combined_df
    .groupby("orf_key", as_index=False, sort=False)
    .apply(choose_best_row)
    .reset_index(drop=True)
)

#collapsed_df.to_csv(os.path.join(OUT_DIR, "02_collapsed_exact_orf.tsv"), sep="\t", index=False)

print("Collapsed exact ORFs:", len(collapsed_df))

# =========================
# Check 1:
# Same full Location but multiple Codons?
# =========================
def check_same_location_diff_codon(df: pd.DataFrame):
    loc_summary = (
        df.groupby("Location")
          .agg(
              n_rows=("Location", "size"),
              n_codon=("Codon", "nunique"),
              codons=("Codon", lambda x: "||".join(sorted(set(x)))),
              ids=("Id", lambda x: "||".join(sorted(set(x)))),
              sources=("Source_Clade", lambda x: "||".join(sorted(set(x)))),
          )
          .reset_index()
    )

    weird = loc_summary[loc_summary["n_codon"] > 1].copy()

    if len(weird) == 0:
        detail = pd.DataFrame()
    else:
        weird_locs = set(weird["Location"])
        detail = (
            df[df["Location"].isin(weird_locs)]
            .sort_values(["Location", "Codon", "Source_Clade", "p value"])
            [["Gene", "Id", "Location", "Candidate Location", "Codon",
              "Type", "Start", "Range", "p value", "Source_Clade", "display_id", "orf_key"]]
            .copy()
        )

    return weird, detail

weird_loc_summary, weird_loc_detail = check_same_location_diff_codon(combined_df)
#weird_loc_summary.to_csv(os.path.join(OUT_DIR, "03_same_location_multi_codon_summary.tsv"), sep="\t", index=False)
#weird_loc_detail.to_csv(os.path.join(OUT_DIR, "04_same_location_multi_codon_detail.tsv"), sep="\t", index=False)

print("Locations with >1 codon:", len(weird_loc_summary))

# =========================
# Optional sanity checks
# =========================
# Same orf_key but different metadata?
meta_cols = ["Gene", "Id", "Location", "Candidate Location", "Codon", "Type"]
conflict_summary = (
    combined_df.groupby("orf_key")[meta_cols]
    .nunique(dropna=False)
    .reset_index()
)

conflict_rows = conflict_summary[
    (conflict_summary[meta_cols] > 1).any(axis=1)
].copy()

#conflict_rows.to_csv(os.path.join(OUT_DIR, "05_same_orfkey_metadata_conflicts.tsv"), sep="\t", index=False)
print("orf_key groups with metadata conflicts:", len(conflict_rows))

print("\nSaved files:")
for fn in [
    "01_combined_all_rows.tsv",
    "02_collapsed_exact_orf.tsv",
    "03_same_location_multi_codon_summary.tsv",
    "04_same_location_multi_codon_detail.tsv",
    "05_same_orfkey_metadata_conflicts.tsv",
]:
    print(" -", os.path.join(OUT_DIR, fn))
    
# %%
weird_loc_summary, weird_loc_detail = check_same_location_diff_codon(combined_df)

print(weird_loc_summary.head(20))
print(weird_loc_detail.head(50).to_string())

meta_cols = ["Gene", "Id", "Candidate Location", "Type"]

conflict_breakdown = {
    col: int((combined_df.groupby("orf_key")[col].nunique(dropna=False) > 1).sum())
    for col in meta_cols
}

print(conflict_breakdown)
# %%
collapsed_df.to_csv('/home/jiye/jiye/darkproteome/data/HG_JH/simil_check/clademergedPRICE_orfs.tsv', sep='\t', index=False)

# %%
import os
import re
import pandas as pd

# ---------------------------------
# helpers
# ---------------------------------
def clean_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def parse_location_full(loc: str):
    """
    Parse PRICE Location string like:
      1+:236306192-236306336|236332005-236332055|...
      12-:9094413-9094536|9095011-9095138|...
    Returns:
      chrom, strand, blocks(list of (start,end)), span_start, span_end
    """
    loc = clean_str(loc)
    m = re.match(r"^(.+?)([+-]):(.+)$", loc)
    if not m:
        raise ValueError(f"Could not parse Location: {loc}")

    chrom, strand, coord_str = m.groups()

    blocks = []
    for block in coord_str.split("|"):
        s, e = block.split("-")
        s, e = int(s), int(e)
        if e <= s:
            raise ValueError(f"Invalid block with end <= start: {loc}")
        blocks.append((s, e))

    span_start = min(s for s, _ in blocks)
    span_end = max(e for _, e in blocks)
    return chrom, strand, blocks, span_start, span_end

def make_display_id_from_row(row):
    """
    nuORFdb-like display label, but keep it readable.
    Example:
      ENST00000454784.8_2_12:40421230-40485543:+|GN=GENE Type|custom_hier_v1
    """
    chrom, strand, blocks, span_start, span_end = parse_location_full(row["Location"])
    gene = clean_str(row["Gene"])
    orf_type = clean_str(row["Type"])
    price_id = clean_str(row["Id"])
    return f"{price_id}_{chrom}:{span_start}-{span_end}:{strand}|GN={gene} {orf_type}|custom_hier_v1"

def location_to_bed12(loc: str, name: str, add_chr_prefix=False):
    """
    Convert PRICE Location -> BED12 fields
    """
    chrom, strand, blocks, span_start, span_end = parse_location_full(loc)

    if add_chr_prefix and not chrom.startswith("chr"):
        chrom = "chr" + chrom

    # sort blocks by genomic start for BED12
    blocks = sorted(blocks, key=lambda x: x[0])

    block_sizes = [e - s for s, e in blocks]
    block_starts = [s - span_start for s, e in blocks]

    bed12 = {
        "chrom": chrom,
        "chromStart": span_start,
        "chromEnd": span_end,
        "name": name,
        "score": 0,
        "strand": strand,
        "thickStart": span_start,
        "thickEnd": span_end,
        "itemRgb": 0,
        "blockCount": len(blocks),
        "blockSizes": ",".join(map(str, block_sizes)) + ",",
        "blockStarts": ",".join(map(str, block_starts)) + ",",
    }
    return bed12

def collapsed_df_to_bed12(
    collapsed_df: pd.DataFrame,
    out_bed_path: str,
    add_chr_prefix=False,
    use_orfkey_in_name=True
):
    """
    Create BED12 from collapsed_df.
    Required columns in collapsed_df:
      - Location
      - Gene
      - Id
      - Type
      - orf_key (recommended)
    """
    rows = []

    for _, row in collapsed_df.iterrows():
        if use_orfkey_in_name and "orf_key" in collapsed_df.columns:
            # preserve exact recoverable key in name
            name = row["orf_key"]
        else:
            name = make_display_id_from_row(row)

        bed12 = location_to_bed12(
            loc=row["Location"],
            name=name,
            add_chr_prefix=add_chr_prefix
        )
        rows.append(bed12)

    bed12_df = pd.DataFrame(rows, columns=[
        "chrom", "chromStart", "chromEnd", "name", "score", "strand",
        "thickStart", "thickEnd", "itemRgb", "blockCount",
        "blockSizes", "blockStarts"
    ])

    bed12_df.to_csv(out_bed_path, sep="\t", header=False, index=False)
    return bed12_df

OUT_BED = "/home/jiye/jiye/darkproteome/data/HG_JH/simil_check/HGJH_clademergedPRICE_orfs.bed12"

bed12_df = collapsed_df_to_bed12(
    collapsed_df=collapsed_df,
    out_bed_path=OUT_BED,
    add_chr_prefix=True,   # genome.fa 헤더가 chr1, chr2... 면 True
    use_orfkey_in_name=True
)

print("BED12 rows:", len(bed12_df))
print("Saved:", OUT_BED)
print(bed12_df.head())
# %%
import pandas as pd

input_tsv = "/home/jiye/jiye/darkproteome/data/HG_JH/simil_check/All_merged.orfs.tsv"
output_bed = "/home/jiye/jiye/darkproteome/data/HG_JH/simil_check/All_merged.orfs.bed12"

df = pd.read_csv(input_tsv, sep="\t", dtype=str)

required_cols = ["Id", "Location"]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

def parse_location(loc: str):
    # example:
    # 1+:182305443-182305458
    # 1-:101701426-101701437|101701880-101701914

    chromstrand, coords = loc.split(":", 1)
    strand = chromstrand[-1]
    chrom = chromstrand[:-1]

    # add chr prefix if needed
    if not chrom.startswith("chr"):
        chrom = "chr" + chrom

    blocks = []
    for part in coords.split("|"):
        s, e = part.split("-")
        s = int(s)
        e = int(e)
        if e < s:
            raise ValueError(f"Invalid interval in {loc}")
        blocks.append((s, e))

    chrom_start = min(s for s, e in blocks)
    chrom_end = max(e for s, e in blocks)

    block_sizes = [e - s for s, e in blocks]
    block_starts = [s - chrom_start for s, e in blocks]

    return chrom, chrom_start, chrom_end, strand, block_sizes, block_starts

with open(output_bed, "w") as out:
    for _, row in df.iterrows():
        loc = row["Location"]
        name = row["Id"]

        chrom, chrom_start, chrom_end, strand, block_sizes, block_starts = parse_location(loc)

        bed_fields = [
            chrom,
            str(chrom_start),
            str(chrom_end),
            name,
            "0",
            strand,
            str(chrom_start),   # thickStart
            str(chrom_end),     # thickEnd
            "0",
            str(len(block_sizes)),
            ",".join(map(str, block_sizes)) + ",",
            ",".join(map(str, block_starts)) + ",",
        ]
        out.write("\t".join(bed_fields) + "\n")

print(f"Written: {output_bed}")
# %%
