# 1. devtools 설치 (없다면)
if (!require("devtools")) install.packages("devtools")
devtools::install_github("buschlab/copynumber", build_vignettes = FALSE)
devtools::install_github("buschlab/sequenza", build_vignettes = FALSE)
#install_local("/home/jiye/Desktop/jiye/SYdata/sequenza-WortJohn-patch-1.zip", force=TRUE, repos=NULL, type="source", dependencies=FALSE)
library(sequenza)

# 2. ScarHRD 설치 (GitHub에서 직접 설치)
library(devtools)
install_github("sztup/scarHRD")
library(scarHRD)
library(tools) # 파일명 처리를 위해
Sys.setenv(VROOM_CONNECTION_SIZE = "500000000")
scar_score("/home/jiye/jiye/SYdata/ScarHRD/Sequenza/OVA-03.seqz.gz",reference = "grch38", seqz=TRUE)
