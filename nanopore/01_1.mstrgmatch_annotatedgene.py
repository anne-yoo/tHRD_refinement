#%%
import pandas as pd


def extract_gene_mappings(stringtie_gtf):
    # Dictionary to map MSTRG gene IDs to gene_name and ensembl_gene_id
    mstrg_gene_info = {}

    with open(stringtie_gtf, 'r') as infile:
        for line in infile:
            if line.startswith("#"):
                continue  # Skip comment lines

            fields = line.strip().split("\t")
            if len(fields) > 8:
                attributes = fields[8].split(";")
                gene_id = ""
                gene_name = "NA"
                ensembl_gene_id = "NA"

                for attr in attributes:
                    attr = attr.strip().replace('"', '')
                    if attr.startswith("gene_id"):
                        gene_id = attr.split(" ")[1].strip()
                    elif attr.startswith("gene_name"):
                        gene_name = attr.split(" ")[1].strip()
                    elif attr.startswith("ref_gene_id"):  # Assuming this holds Ensembl gene IDs
                        ensembl_gene_id = attr.split(" ")[1].strip()

                # Only add MSTRG genes that map to Ensembl gene IDs or have a gene name
                if gene_id.startswith("MSTRG") and (gene_name != "NA" or ensembl_gene_id != "NA"):
                    mstrg_gene_info[gene_id] = {
                        "gene_name": gene_name,
                        "ensembl_gene_id": ensembl_gene_id
                    }

    return mstrg_gene_info

def extract_transcripts_info(stringtie_gtf, mstrg_gene_info):
    # Lists to store ENST and MSTRG transcript information
    enst_transcripts = []
    mstrg_transcripts = []

    with open(stringtie_gtf, 'r') as infile:
        for line in infile:
            if line.startswith("#"):
                continue  # Skip comment lines

            fields = line.strip().split("\t")
            if len(fields) > 8 and fields[2] == "transcript":  # Only process transcript entries
                attributes = fields[8].split(";")
                gene_id = ""
                transcript_id = ""
                gene_name = "NA"
                ensembl_gene_id = "NA"

                for attr in attributes:
                    attr = attr.strip().replace('"', '')
                    if attr.startswith("gene_id"):
                        gene_id = attr.split(" ")[1].strip()
                    elif attr.startswith("transcript_id"):
                        transcript_id = attr.split(" ")[1].strip()
                    elif attr.startswith("gene_name"):
                        gene_name = attr.split(" ")[1].strip()
                    elif attr.startswith("ref_gene_id"):  # Assuming this holds Ensembl gene IDs
                        ensembl_gene_id = attr.split(" ")[1].strip()

                # Process ENST transcripts and map using gene_name
                if transcript_id.startswith("ENST"):
                    enst_transcripts.append({
                        "mstrg_gene_id": gene_id,
                        "ensembl_gene_id": ensembl_gene_id,
                        "transcript_id": transcript_id,
                        "gene_name": gene_name
                    })

                # Process MSTRG transcripts and map using gene_id from the dictionary
                elif transcript_id.startswith("MSTRG"):
                    mapped_info = mstrg_gene_info.get(gene_id, {})
                    mstrg_transcripts.append({
                        "mstrg_gene_id": gene_id,
                        "ensembl_gene_id": mapped_info.get("ensembl_gene_id", "NA"),
                        "transcript_id": transcript_id,
                        "gene_name": mapped_info.get("gene_name", "NA")
                    })

    # Convert lists to DataFrames
    enst_df = pd.DataFrame(enst_transcripts).drop_duplicates()
    mstrg_df = pd.DataFrame(mstrg_transcripts).drop_duplicates()

    # Combine ENST and MSTRG DataFrames
    combined_df = pd.concat([enst_df, mstrg_df]).drop_duplicates().reset_index(drop=True)
    return combined_df

# Step 1: Extract MSTRG to Ensembl gene mappings from 291_merge.gtf
stringtie_gtf_path = "/home/jiye/jiye/nanopore/gtfcompare/stringtiesample/291_merge.gtf"
mstrg_gene_info = extract_gene_mappings(stringtie_gtf_path)

# Step 2: Extract transcripts (both ENST and MSTRG) and map information
combined_df = extract_transcripts_info(stringtie_gtf_path, mstrg_gene_info)

filtered_df = combined_df

# Save the final filtered DataFrame to a file
filtered_output_path = "/home/jiye/jiye/nanopore/gtfcompare/stringtiesample/everyannotatedgene_filtered_transcripts_with_gene_info.tsv"
#filtered_df.to_csv(filtered_output_path, sep='\t', index=False)



# %%
###########^^^ make quantification matrix ##################
import os
import pandas as pd

def extract_tpm_for_target_transcripts(gtf_file, target_transcripts_set):
    # Extract TPM values for target transcripts using line-by-line parsing
    tpm_values = {transcript: 0.0 for transcript in target_transcripts_set}

    with open(gtf_file, 'r') as file:
        for line in file:
            if line.startswith("#"):
                continue  # Skip comments

            # Split the line into fields and ensure it's a 'transcript' line
            fields = line.strip().split("\t")
            if len(fields) < 9 or fields[2] != "transcript":  # Check if the feature is 'transcript'
                continue

            attributes = fields[8]

            # Parse the attributes to create a dictionary
            attr_dict = {}
            for attr in attributes.split(";"):
                attr = attr.strip()
                if " " in attr:
                    key, value = attr.split(" ", 1)
                    attr_dict[key] = value.replace('"', '').strip()

            transcript_id = attr_dict.get("transcript_id")
            tpm_value = attr_dict.get("TPM")

            # Check if both transcript_id and TPM were found
            if transcript_id is None or tpm_value is None:
                print(f"Warning: Missing transcript_id or TPM in line: {line.strip()}")
                continue

            # Convert TPM to float and update if transcript is in target set
            try:
                tpm_value = float(tpm_value)
            except ValueError:
                print(f"Warning: Unable to convert TPM value to float: {tpm_value}")
                continue

            if transcript_id in target_transcripts_set:
                tpm_values[transcript_id] = tpm_value

    return tpm_values

def build_tpm_matrix_fast(base_directory, matrix_file_path):
    # Load the matrix file containing target transcripts
    matrix_df = pd.read_csv(matrix_file_path, sep="\t")
    target_transcripts_set = set(matrix_df['transcript_id'])
    transcript_gene_map = {row['transcript_id']: row['gene_name'] for _, row in matrix_df.iterrows()}

    # Initialize a dictionary to store TPM values for all transcripts across samples
    tpm_data = {}

    # Iterate over all subdirectories (samples) in the base directory
    for sample_dir in os.listdir(base_directory):
        sample_path = os.path.join(base_directory, sample_dir)
        gtf_file = os.path.join(sample_path, f"{sample_dir}.gtf")  # Assumes GTF file is named as the sample_dir

        if os.path.exists(gtf_file):
            print(f"Processing {gtf_file}...")
            sample_tpm_values = extract_tpm_for_target_transcripts(gtf_file, target_transcripts_set)
            # Update the main TPM data dictionary
            for transcript_id, tpm_value in sample_tpm_values.items():
                transcript_key = f"{transcript_id}-{transcript_gene_map.get(transcript_id, 'NA')}"
                if transcript_key not in tpm_data:
                    tpm_data[transcript_key] = {}
                tpm_data[transcript_key][sample_dir] = tpm_value

    # Convert the TPM data dictionary to a DataFrame
    tpm_df = pd.DataFrame.from_dict(tpm_data, orient='index').fillna(0)
    tpm_df.index.name = "Transcript-Gene"
    return tpm_df

# Example usage
base_directory = "/home/omics/DATA6/sujie/nanopore/bulk_stringtie/afterQC_quantif/"  # Update with the correct path
matrix_file_path = "/home/jiye/jiye/nanopore/gtfcompare/stringtiesample/everyannotatedgene_filtered_transcripts_with_gene_info.tsv"  # Update with the correct path
tpm_df = build_tpm_matrix_fast(base_directory, matrix_file_path)

# Save the DataFrame to a file
output_file_path = "/home/jiye/jiye/nanopore/FINALDATA/wholeannot/291_transcript_TPM.txt"  # Update with the desired output path
#tpm_df.to_csv(output_file_path, sep="\t")
print(f"TPM matrix saved to {output_file_path}")


# %%
##^^^^^^^^^^^ gene level ##############
import os
import pandas as pd

def create_gene_abundance_matrix(transcript_tpm_matrix_path, matrix_file_path, output_file_path):
    # Load the transcript TPM matrix
    tpm_df = pd.read_csv(transcript_tpm_matrix_path, sep="\t", index_col=0)

    # Extract gene names from the index (assuming the format is 'transcript_id-gene_name')
    tpm_df['Gene'] = tpm_df.index.map(lambda x: '-'.join(x.split('-', 1)[1:]))  # Correctly handle gene names with '-'

    # Load the matrix file to map gene names to ENSG IDs
    matrix_df = pd.read_csv(matrix_file_path, sep="\t")
    gene_to_ensg = dict(zip(matrix_df['gene_name'], matrix_df['ensembl_gene_id']))

    # Group by 'Gene' and sum the TPM values for all transcripts of each gene
    gene_tpm_df = tpm_df.groupby('Gene').sum()

    # Add an ENSG column based on the mapping
    gene_tpm_df['ENSG'] = gene_tpm_df.index.map(lambda x: gene_to_ensg.get(x, 'NA'))

    # Reorder columns to have ENSG first
    columns = ['ENSG'] + [col for col in gene_tpm_df.columns if col != 'ENSG']
    gene_tpm_df = gene_tpm_df[columns]

    # Save the resulting gene abundance matrix
    gene_tpm_df = gene_tpm_df.dropna()
    #gene_tpm_df.to_csv(output_file_path, sep="\t")
    print(f"Gene abundance matrix with ENSG IDs saved to {output_file_path}")


# Example usage
transcript_tpm_matrix_path = "/home/jiye/jiye/nanopore/FINALDATA/wholeannot/291_transcript_TPM.txt"  # Update with the correct path
matrix_file_path = "/home/jiye/jiye/nanopore/gtfcompare/stringtiesample/everyannotatedgene_filtered_transcripts_with_gene_info.tsv"  # Update with the correct path
output_file_path = "/home/jiye/jiye/nanopore/FINALDATA/wholeannot/291_gene_TPM.txt"  # Update with the desired output path
create_gene_abundance_matrix(transcript_tpm_matrix_path, matrix_file_path, output_file_path)


# %%
#^^^ gene exp leave only paired #####
import re

def filter_and_reorder_columns(input_file, output_file):
    # Load the data
    df = pd.read_csv(input_file, sep="\t")
    print("Initial columns:", df.columns.tolist())  # Debugging statement

    # Identify columns with '-N' and '-T' suffixes (excluding 'Gene' and 'ENSG' columns)
    columns = df.columns.tolist()
    sample_columns = [col for col in columns if col not in ['Gene', 'ENSG']]
    normal_samples = [col for col in sample_columns if '-N' in col]
    tumor_samples = [col for col in sample_columns if '-T' in col]
    print("Normal samples found:", normal_samples)  # Debugging statement
    print("Tumor samples found:", tumor_samples)  # Debugging statement

    # Create a mapping for base sample names by removing the '-N'/'-T' suffixes and the trailing 'A1'/'A2'
    def strip_suffix(sample_name):
        # Remove '-N' or '-T' and the trailing '-A1', '-A2', etc.
        return re.sub(r'(-N|-T)-A\d+$', '', sample_name)

    # Omit specific samples
    samples_to_omit = {'PM-PM-1072-N-A1', 'PM-PM-1072-T-A1'}
    normal_samples = [sample for sample in normal_samples if sample not in samples_to_omit]
    tumor_samples = [sample for sample in tumor_samples if sample not in samples_to_omit]

    paired_samples = {}
    unpaired_normals = []  # List to store unpaired normal samples
    unpaired_tumors = list(tumor_samples)  # Copy of tumor samples to track unpaired ones

    for normal in normal_samples:
        base_name = strip_suffix(normal)
        matching_tumor = next((tumor for tumor in tumor_samples if strip_suffix(tumor) == base_name), None)
        if matching_tumor:
            paired_samples[base_name] = (normal, matching_tumor)
            unpaired_tumors.remove(matching_tumor)  # Remove paired tumor from unpaired list
        else:
            unpaired_normals.append(normal)  # Add to unpaired normals if no match found

    print("Paired samples:", paired_samples)  # Debugging statement
    print(f"Number of pairs identified: {len(paired_samples)}")  # Display the number of pairs
    print("Unpaired normal samples:", unpaired_normals)  # Display unpaired normal samples
    print("Unpaired tumor samples:", unpaired_tumors)  # Display unpaired tumor samples

    # Flatten the list of paired columns in the desired order: sample1-N, sample1-T, sample2-N, sample2-T, ...
    ordered_columns = ['Gene', 'ENSG'] + [col for pair in paired_samples.values() for col in pair]
    print("Ordered columns to keep:", ordered_columns)  # Debugging statement

    # Check if there are any columns left to reorder
    if len(ordered_columns) <= 2:
        print("No paired columns found, exiting...")
        return

    # Reorder the DataFrame and keep only the relevant columns
    df_filtered = df[ordered_columns]

    # Remove '-A1'/'-A2' suffixes from column names
    df_filtered.columns = [re.sub(r'-A\d+$', '', col) for col in df_filtered.columns]

    # Save the reordered DataFrame
    df_filtered = df_filtered.dropna()
    #df_filtered.to_csv(output_file, sep="\t", index=False)
    print(f"Filtered and reordered DataFrame saved to {output_file}")
# Example usage

# Example usage
input_file = "/home/jiye/jiye/nanopore/FINALDATA/wholeannot/291_gene_TPM.txt"  # Update with the correct path
output_file = "/home/jiye/jiye/nanopore/FINALDATA/wholeannot/matched_gene_TPM.txt"  # Update with the desired output path
filter_and_reorder_columns(input_file, output_file)

# %% ##^^ matched transcript

def filter_transcript_tpm_matrix(transcript_tpm_file, paired_gene_exp_file, output_file):
    # Load the paired gene expression matrix to extract the columns (samples) to keep
    paired_gene_df = pd.read_csv(paired_gene_exp_file, sep="\t")

    # Extract columns (samples) to keep from the paired gene expression matrix
    columns_to_keep = paired_gene_df.columns.tolist()
    columns_to_keep = [col for col in columns_to_keep if col not in ['Gene', 'ENSG']]  # Exclude 'Gene' and 'ENSG'
    print("Columns to keep from paired gene expression matrix:", columns_to_keep)

    # Load the transcript TPM matrix
    transcript_df = pd.read_csv(transcript_tpm_file, sep="\t", index_col=0)
    print("Initial transcript TPM matrix columns:", transcript_df.columns.tolist())  # Debugging statement

    # Remove '-A1'/'-A2' suffixes from transcript TPM matrix column names
    transcript_df.columns = [re.sub(r'-A\d+$', '', col) for col in transcript_df.columns]
    print("Transcript TPM matrix columns after removing suffixes:", transcript_df.columns.tolist())  # Debugging statement

    # Drop specific columns if present
    columns_to_drop = {'PM-PM-1072-N', 'PM-PM-1072-T'}
    transcript_df = transcript_df.drop(columns=[col for col in columns_to_drop if col in transcript_df.columns], errors='ignore')
    print("Transcript TPM matrix columns after dropping specified samples:", transcript_df.columns.tolist())  # Debugging statement

    # Filter the transcript TPM matrix to keep only the relevant columns from the paired gene expression matrix
    df_filtered = transcript_df[columns_to_keep]

    # Save the filtered and reordered transcript TPM matrix
    df_filtered.to_csv(output_file, sep="\t")
    print(f"Filtered transcript TPM matrix saved to {output_file}")

# Example usage
transcript_tpm_file = "/home/jiye/jiye/nanopore/FINALDATA/wholeannot/291_transcript_TPM.txt"  # Update with the correct path
paired_gene_exp_file = "/home/jiye/jiye/nanopore/FINALDATA/wholeannot/matched_gene_TPM.txt"  # Use the output from previous step
output_file = "/home/jiye/jiye/nanopore/FINALDATA/wholeannot/matched_transcript_TPM.txt"  # Update with the desired output path
filter_transcript_tpm_matrix(transcript_tpm_file, paired_gene_exp_file, output_file)

# %%
aa = pd.read_csv("/home/jiye/jiye/nanopore/FINALDATA/wholeannot/matched_transcript_TPM.txt", sep='\t')
bb = pd.read_csv("/home/jiye/jiye/nanopore/FINALDATA/wholeannot/matched_gene_TPM.txt", sep='\t')

# %%
#####^^^ readcount format: transcript #######
import pandas as pd

def transform_readcount_to_tpm_format(readcount_file, tpm_matrix_file, transcript_info_file, output_file):
    # Load the readcount file
    readcount_df = pd.read_csv(readcount_file, sep=",")
    
    # Load the transcript TPM matrix to get the columns order and list of transcripts
    tpm_df = pd.read_csv(tpm_matrix_file, sep="\t", index_col=0)
    tpm_columns = tpm_df.columns.tolist()
    tpm_transcripts = tpm_df.index.tolist()

    # Load transcript information to get gene names
    transcript_info_df = pd.read_csv(transcript_info_file, sep="\t")  # Assume tab-separated, adjust if different
    transcript_info_df = transcript_info_df.set_index('transcript_id')  # Ensure transcript_id is the index for mapping

    # Add '-genename' to each transcript ID if a corresponding gene name exists
    readcount_df['transcript_id'] = readcount_df['transcript_id'].apply(
        lambda x: f"{x}-{transcript_info_df.loc[x, 'gene_name']}" if x in transcript_info_df.index else x
    )

    # Filter the readcount DataFrame to include only transcripts present in the TPM matrix
    readcount_df = readcount_df[readcount_df['transcript_id'].isin(tpm_transcripts)]

    # Set the transcript_id as index
    readcount_df = readcount_df.set_index('transcript_id')

    # Remove '-A1'/'-A2' suffixes from column names in the readcount DataFrame
    readcount_df.columns = [re.sub(r'-A\d+$', '', col) for col in readcount_df.columns]
    print("Readcount matrix columns after removing suffixes:", readcount_df.columns.tolist())  # Debugging statement

    # Drop specific columns if present
    columns_to_drop = {'PM-PM-1072-N', 'PM-PM-1072-T'}
    readcount_df = readcount_df.drop(columns=[col for col in columns_to_drop if col in readcount_df.columns], errors='ignore')
    print("Readcount matrix columns after dropping specified samples:", readcount_df.columns.tolist())  # Debugging statement

    # Match and filter columns to the TPM matrix order using a string match
    filtered_columns = [col for col in tpm_columns if any(re.sub(r'-A\d+$', '', col) == rc_col for rc_col in readcount_df.columns)]
    print("Filtered columns to keep:", filtered_columns)  # Debugging statement
    readcount_df = readcount_df[filtered_columns]

    # Save the transformed data to a new file
    readcount_df.to_csv(output_file, sep="\t")
    print(f"Transformed readcount matrix saved to {output_file}")

# Example usage
readcount_file = "/home/jiye/jiye/nanopore/gtfcompare/makereadcount/transcript_count_matrix.csv"  # Replace with actual file path
tpm_matrix_file = "/home/jiye/jiye/nanopore/FINALDATA/wholeannot/matched_transcript_TPM.txt"  # Replace with actual file path
transcript_info_file = "/home/jiye/jiye/nanopore/gtfcompare/stringtiesample/everyannotatedgene_filtered_transcripts_with_gene_info.tsv"  # File containing transcript_id and gene_name mapping
output_file ="/home/jiye/jiye/nanopore/FINALDATA/wholeannot/matched_transcript_readcount.txt" # Output file
transform_readcount_to_tpm_format(readcount_file, tpm_matrix_file, transcript_info_file, output_file)

# %%
#####^^ gene level readcount data ###########
import pandas as pd
import re

def transform_gene_readcount_to_tpm_format(readcount_file, tpm_matrix_file, output_file):
    # Load the gene-level readcount file
    readcount_df = pd.read_csv(readcount_file, sep=",", header=0)
    readcount_df = readcount_df.drop_duplicates()
    
    # Extract and split the 'gene_id|gene_name' column to get only the gene_name
    readcount_df['gene_name'] = readcount_df.iloc[:, 0].apply(lambda x: x.split('|')[-1])
    
    # Set the 'gene_name' as the index
    readcount_df = readcount_df.set_index('gene_name')
    
    # Drop the original 'gene_id|gene_name' column
    readcount_df = readcount_df.drop(columns=readcount_df.columns[0])

    # Load the gene TPM matrix to get the column order and gene list
    tpm_df = pd.read_csv(tpm_matrix_file, sep="\t", index_col=0)
    tpm_columns = tpm_df.columns.tolist()
    tpm_genes = tpm_df.index.tolist()

    # Remove '-A1'/'-A2' suffixes from column names in the readcount DataFrame
    readcount_df.columns = [re.sub(r'-A\d+$', '', col) for col in readcount_df.columns]
    print("Readcount matrix columns after removing suffixes:", readcount_df.columns.tolist())  # Debugging statement

    # Drop specific samples if present
    columns_to_drop = {'PM-PM-1072-N', 'PM-PM-1072-T'}
    readcount_df = readcount_df.drop(columns=[col for col in columns_to_drop if col in readcount_df.columns], errors='ignore')
    print("Readcount matrix columns after dropping specified samples:", readcount_df.columns.tolist())  # Debugging statement

    # Match and filter columns to the TPM matrix order
    filtered_columns = [col for col in tpm_columns if any(re.sub(r'-A\d+$', '', col) == rc_col for rc_col in readcount_df.columns)]
    print("Filtered columns to keep:", filtered_columns)  # Debugging statement
    readcount_df = readcount_df[filtered_columns]

    # Filter the readcount DataFrame to include only genes present in the TPM matrix
    readcount_df = readcount_df.loc[tpm_genes]

    # Save the transformed data to a new file
    readcount_df.to_csv(output_file, sep="\t")
    print(f"Transformed readcount matrix saved to {output_file}")

# Example usage
readcount_file = "/home/jiye/jiye/nanopore/gtfcompare/makereadcount/gene_count_matrix.csv"
tpm_matrix_file = "/home/jiye/jiye/nanopore/FINALDATA/wholeannot/matched_gene_TPM.txt"  # Replace with actual file path
output_file = "/home/jiye/jiye/nanopore/FINALDATA/wholeannot/matched_gene_readcount.txt"  # Output file
transform_gene_readcount_to_tpm_format(readcount_file, tpm_matrix_file, output_file)

tpm_df = pd.read_csv(tpm_matrix_file, sep='\t', index_col=0)
readcount_df = pd.read_csv(output_file, sep='\t', index_col=0)
tpm_genes = set(tpm_df.index)
readcount_genes = set(readcount_df.index)
missing_genes = tpm_genes - readcount_genes
# %%
duplicates = readcount_df[readcount_df.index.duplicated(keep=False)]

# %%
def drop_smaller_mean_duplicates(df):
    # Calculate the mean readcount for each row
    df['mean_readcount'] = df.mean(axis=1)

    # Sort the DataFrame by index (gene name) and mean readcount
    df = df.sort_values(by=['gene_name', 'mean_readcount'], ascending=[True, False])

    # Drop duplicates while keeping the row with the higher mean readcount
    df = df.drop_duplicates(subset='gene_name', keep='first')

    # Drop the 'mean_readcount' column as it was only needed for sorting
    df = df.drop(columns=['mean_readcount'])

    return df

# Assuming 'readcount_df' is your DataFrame
readcount_df = pd.read_csv("/home/jiye/jiye/nanopore/FINALDATA/wholeannot/matched_gene_readcount.txt", sep='\t')  # Replace with your actual file path
readcount_df = drop_smaller_mean_duplicates(readcount_df)

# Save the cleaned DataFrame if needed
readcount_df.to_csv("/home/jiye/jiye/nanopore/FINALDATA/wholeannot/cleaned_matched_gene_readcount.txt", index=False, sep='\t')

# %%
#####^^ clinical data + only matched data ##############
import pandas as pd
import numpy as np  

clin = pd.read_csv('/home/jiye/jiye/nanopore/FINALDATA/clinical_data_cleaned.txt', sep='\t')
clin = clin.replace('PM-PM-1072-N', 'PM-PU-1072-N')
clin = clin.replace('PM-PM-1072-T', 'PM-PU-1072-T')
list = ['PM-PU-1005','PM-PU-1044','PM-PU-1087','PM-PA-1001']
clin['sample'] = clin['sample_normal'].str[:-2]
clin.index = clin['sample']
clin = clin.drop(list)
clin = clin.drop(['stage_roman'], axis=1)

clin['age'] = clin['age'].astype('Int64').fillna('NA')
clin['DFS'] = clin['DFS'].apply(lambda x: int(x) if pd.notnull(x) else 'NA')
clin['OS'] = clin['OS'].apply(lambda x: int(x) if pd.notnull(x) else 'NA')
clin['stage'] = clin['stage'].apply(lambda x: int(x) if pd.notnull(x) else 'NA')
clin['KRAS_mut_detail'] = clin['KRAS_mut']
clin['KRAS_mut'] = clin['KRAS_mut'].str[:2]

# Handle categorical/binary columns as before
clin['lymphatic_invasion'] = clin['lymphatic_invasion'].replace({'Yes': 1, 'No': 0}).fillna('NA')
clin['venous_invasion'] = clin['venous_invasion'].replace({'Yes': 1, 'No': 0}).fillna('NA')
clin['perineural_invasion'] = clin['perineural_invasion'].replace({'Yes': 1, 'No': 0}).fillna('NA')


#%%

generead = pd.read_csv("/home/jiye/jiye/nanopore/FINALDATA/wholeannot/matched_gene_readcount.txt", sep='\t', index_col=0)
clinlist = set(clin.index.to_list())
readlist = set(generead.columns.str[:-2].to_list())

#%%
gene_readcount = pd.read_csv("/home/jiye/jiye/nanopore/FINALDATA/wholeannot/matched_gene_readcount.txt", sep='\t', index_col=0)
gene_tpm = pd.read_csv("/home/jiye/jiye/nanopore/FINALDATA/wholeannot/matched_gene_TPM.txt", sep='\t', index_col=0)
trans_readcount = pd.read_csv("/home/jiye/jiye/nanopore/FINALDATA/wholeannot/matched_transcript_readcount.txt", sep='\t', index_col=0)
trans_tpm = pd.read_csv("/home/jiye/jiye/nanopore/FINALDATA/wholeannot/matched_transcript_TPM.txt", sep='\t', index_col=0)

list = ['PM-PU-1005-N','PM-PU-1005-T','PM-PU-1044-N','PM-PU-1044-T','PM-PU-1087-N','PM-PU-1087-T','PM-PA-1001-N','PM-PA-1001-T']

gene_readcount = gene_readcount.drop(list,axis=1)
gene_tpm = gene_tpm.drop(list,axis=1)
trans_readcount = trans_readcount.drop(list,axis=1)
trans_tpm = trans_tpm.drop(list,axis=1)

clin = clin.sort_index(axis=0)
gene_readcount = gene_readcount.sort_index(axis=1)
gene_tpm = gene_tpm.sort_index(axis=1)
trans_readcount = trans_readcount.sort_index(axis=1)
trans_tpm = trans_tpm.sort_index(axis=1)

#%%
samplelist = gene_readcount.columns.to_list()[0::2]
samplelist = [item[:-2] for item in samplelist]
finalclin = clin.loc[samplelist,:]

# %%
#finalclin.to_csv("/home/jiye/jiye/nanopore/FINALDATA/137_clinicaldata.txt", sep='\t', index=True)
gene_readcount.to_csv("/home/jiye/jiye/nanopore/FINALDATA/wholeannot/137_gene_readcount.txt", sep='\t', index=True)
gene_tpm.to_csv("/home/jiye/jiye/nanopore/FINALDATA/wholeannot/137_gene_TPM.txt", sep='\t', index=True)
trans_readcount.to_csv("/home/jiye/jiye/nanopore/FINALDATA/wholeannot/137_transcript_readcount.txt", sep='\t', index=True)
trans_tpm.to_csv("/home/jiye/jiye/nanopore/FINALDATA/wholeannot/137_transcript_TPM.txt", sep='\t', index=True)

# %%
####^^ add CMS subtype #######
cms = pd.read_csv('/home/sujie/sujie/nanopore/bulk_processing/241112_processing/matrix/05_Subtyping/CMSsubtype.txt', sep='\t', index_col=0)
cms['prediction'] = cms['prediction'].astype(str)

finalclin['CMS'] = cms['prediction'].to_list()
finalclin.to_csv("/home/jiye/jiye/nanopore/FINALDATA/137_clinicaldata_new.txt", sep='\t', index=True)


# %%
