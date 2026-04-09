#%%
from SigProfilerMatrixGenerator import install as genInstall

genInstall.install('GRCh38', rsync=False, bash=True)

#%%
###* tumor sample만 남김: ex bcftools view -s OVA-01_tumor -Oz -o OVA-01.tumor.only.vcf.gz OVA-01.vcf.gz ##
###* 폴더 안에 압축 풀기 ###

from SigProfilerAssignment import Analyzer as Analyze

Analyze.cosmic_fit(
    samples="/home/jiye/jiye/SYdata/MutationSignature/vcffiles", # input_data -> samples로 변경 (아까 만든 깨끗한 폴더 추천!)
    output="/home/jiye/jiye/SYdata/MutationSignature/SBS3",
    input_type="vcf",
    genome_build="GRCh38",
    exome=True,           # WES인 경우 True
    cosmic_version="3.3", # 최신 COSMIC 사용
    context_type="96",    # SBS96 (기본값)
    collapse_to_SBS96=True 
)

# %%
