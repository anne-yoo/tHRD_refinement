#%%
""" 
Whole samples 중 중복 (replicates) 을 제거하기 위한 코드
"""
import numpy as np
import pandas as pd
import math
import os


def Replicates_filtering(d_type, direction):
    """ Replicates 를 제거하기 위함
        >동일 샘플에 대해 평균값을 취하는 것은 어떨지?
    Args:
        d_type (str): readcounts or TU
        direction (str): pos, abs 으로 PFS~aTU correlation 방향을 의미
    """

    if d_type == "TU":
        DIR="/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/merged_TU/whole_TU/"
        data=pd.read_csv(
            DIR+"230105_whole_TU.txt", sep="\t", header=None
            # DIR+"XGBoost/"+direction+"_sig_response_230105_TU.txt", sep="\t", header=None
            # DIR+"XGBoost/old_230105_TU.txt", sep="\t", header=None
            )
        data = data.drop([i for i in range(132,148)], axis=1)   # 21DNB23-A
        data = data.drop([i for i in range(257,267)], axis=1)   # 22RNA01-woRegenome
        data.columns = data.iloc[0,:]
        data = data.iloc[1:,:]
        data = data.drop(["SV_OV_HRD_SV-OV-P055-atD","SV_OV_HRD_SV-OV-P080-atD_RP",\
        "SV_OV_HRD_SV-OV-P134-atD_RP","SV_OV_HRD_SV-OV-P137-atD_RP",\
        "SV_OV_HRD_SV-OV-P143-atD_RP","SV_OV_HRD_SV-OV-P164-atD",\
        "SV_OV_HRD_SV-OV-P174-bfD","SV_OV_HRD_SV-OV-P250-atD_RP"], axis=1)
        ## QC fail samples
        data = data.set_index("gene_ENST")
        
    else:
        DIR="/home/hyeongu/DATA5/hyeongu/YUHS_transfer_data/data/readcount/"
        data=pd.read_csv(
            DIR+"220923_final_whole_raw_counts_coding_only.txt", 
            sep="\t", header=None)
        data.columns = data.iloc[0,:]
        data = data.iloc[1:,:]
        data = data.set_index("Geneid")
        data = data.drop(["SV-OV-P055-atD","SV-OV-P080-atD",\
        "SV-OV-P134-atD","SV-OV-P137-atD",\
        "SV-OV-P143-atD","SV-OV-P164-atD",\
        "SV-OV-P174-bfD","SV-OV-P250-atD"], axis=1)
        ## QC fail samples
        print(data.head())

    filter_data = pd.read_csv(
                            "/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/merged_TU/whole_TU/230105_remove_samples_list.txt",
                            sep="\t", header=None
                            )
    filtered_sample = list(set(data.columns) - set(filter_data[0].tolist()))
    f_data = data[filtered_sample]
    f_data = f_data.astype(float)
    
    # ## Gene feature 를 사용했을 경우 filetering 을 위함
    # if "GENE" in direction:
    #     f_data = f_data[(f_data==0).astype(int).sum(axis=1).astype(int) <= f_data.shape[1]*.2]
    # f_data["mean"] = f_data.mean(axis=1)
    # f_data = f_data[f_data["mean"] >= 0.05]
    # f_data.drop("mean", axis=1)
    #############
    f_data = f_data.T
    f_data = f_data[f_data.index.str.contains("SV")]
    f_data = f_data[~f_data.index.str.contains("-F-")]
    f_data = f_data[~f_data.index.str.contains("dep")]
    f_data["10ID"] = f_data.index.str.replace("SV_OV_HRD_", "").str\
                            .replace("SMC_OV_OVLB_", "").str.replace("P", "T").\
                            str[:10]
    f_data = f_data[~f_data.index.str.contains("SV-OV-T021_cap_M")]
    f_data = f_data[~f_data.index.str.contains("SV-OV-T037-RNA_FFPE_M")]
    f_data = f_data[~(f_data["10ID"].str.contains("F"))]
    # SV-OV-T037-RNA_FFPE_M, SV-OV-T021-RNA_M
    for i in ["SV-OV-T021","SV-OV-T022","SV-OV-T023",
            "SV-OV-T024","SV-OV-T025","SV-OV-T026"]:
        f_data = f_data[f_data["10ID"]!=i]
    #f_data.to_csv(DIR+"check.txt", sep="\t")

    pre_post_list = pd.read_csv(
                                "/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/merged_TU/whole_TU/230105_pre_post_samples.txt", sep="\t",
                                header=None
                                )
    ### Pre-post 제거
    for i in pre_post_list[0]:
        f_data = f_data[f_data["10ID"]!=i]
    f_data = f_data[~(f_data["10ID"].str.contains("F"))]

    ## Combine clinical information
    group_info = pd.read_csv(
        "/home/hyeongu/DATA5/hyeongu/FFPE_align_files/GC_server/GC_TU/clinical_info_gHRD/processed_info_with_originalID.txt", 
        sep="\t", index_col="GID")
    group_info = group_info[["OM/OS", "ORR", "drug"]]
    group_info.columns = ["OM/OS", "group", "drug"]
    group_info["drug"] = group_info["drug"].str.replace("Olapairb","Olaparib")
    group_info["drug"] = group_info["drug"].str.replace("Olapairib","Olaparib")
    group_info["drug"] = group_info["drug"].str.replace("olaparib","Olaparib")
    group_info = group_info.dropna()

    ## 약제별 구분
    group_info = group_info.drop("drug", axis=1)
    group_info["10ID"] = group_info.index.str.replace("F","T").str[:10]
    group_info = group_info.drop_duplicates()
    group_info = group_info.set_index("10ID")
    merged = pd.merge(f_data, group_info, left_on="10ID",
                    right_index=True, how="inner").set_index("10ID")
    
    ###
    if d_type == "TU":
        merged.T.to_csv(
                    "/home/jiye/jiye/copycomparison/gDUTresearch/FINALDATA/modelinput/validation_TU.txt",
                    # DIR+"XGBoost/230109_ex_pre-post_old_input.txt",
                    sep="\t"
                    )
    else:
        merged.T.to_csv(
                        DIR+"230109_ex_pre-post_"+direction+"_sig_response_input.txt",
                        sep="\t"
                        )
    
# Replicates_filtering("TU","abs")
# Replicates_filtering("TU","pos")
# Replicates_filtering("TU","relaxed_pos")
# Replicates_filtering("TU","From_term_pos")
# Replicates_filtering("TU","non_zero_From_term_pos")
# Replicates_filtering("TU","From_term_FDR5_pos")
# Replicates_filtering("TU","From_term_GENE_pos")
# Replicates_filtering("readcounts","From_term_GENE_pos")
# Replicates_filtering("TU","add105_From_term_pos")   # 230409 INPUT

#%%
Replicates_filtering("TU","231102_val")
#%%