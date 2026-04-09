###### DEXSEq!!!!!! #####
library("DEXSeq")
library("pasilla")
library("randomcoloR")

pythonScriptsDir = system.file( "python_scripts", package="DEXSeq" )
list.files(pythonScriptsDir)
system.file( "python_scripts", package="DEXSeq", mustWork=TRUE )

## find .gff file
inDir = '/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/DEXSeq/BRCA1_2_gencode'
flattenedFile = list.files(inDir, pattern="gff", full.names=TRUE)
basename(flattenedFile)

## make df containing sample info (#'PL-OV-P038',)
res_list = c('PL-OV-P013', 'PL-OV-P014', 'PL-OV-P016', 'PL-OV-P018', 'PL-OV-P020', 'PL-OV-P022', 'PL-OV-P023', 'PL-OV-P024', 'PL-OV-P029', 'PL-OV-P036','PL-OV-P038', 'PL-OV-P039', 'PL-OV-P040', 'PL-OV-P042', 'PL-OV-P048', 'PL-OV-P049', 'PL-OV-P056', 'PL-OV-P058', 'PL-OV-P071', 'PL-OV-P094', 'PL-OV-P095', 'PL-OV-P096')
nonres_list = c('PL-OV-P004', 'PL-OV-P008', 'PL-OV-P010', 'PL-OV-P025', 'PL-OV-P026', 'PL-OV-P028', 'PL-OV-P030', 'PL-OV-P031', 'PL-OV-P033', 'PL-OV-P041', 'PL-OV-P047', 'PL-OV-P057', 'PL-OV-P065', 'PL-OV-P067', 'PL-OV-P069', 'PL-OV-P074', 'PL-OV-P078', 'PL-OV-P081', 'PL-OV-P086', 'PL-OV-P088', 'PL-OV-P090', 'PL-OV-P092', 'PL-OV-P097')

library(stringr)

countFiles = list.files(
  inDir,
  pattern = "edited-count-.*\\.txt$",
  full.names = TRUE
)

sample_names = str_replace_all(
  basename(countFiles),
  c("edited-count-" = "", "\\.txt$" = "") 
)

head(sample_names)
# "PL-OV-P013" "PL-OV-P014" ...

group = ifelse(
  sample_names %in% res_list, "responder",
  ifelse(sample_names %in% nonres_list, "nonresponder", NA)
)

keep = !is.na(group)

countFiles_filt = countFiles[keep]
sample_names_filt = sample_names[keep]
group_filt = group[keep]

sampleTable = data.frame(
  row.names = sample_names_filt,
  condition = group_filt
)

stopifnot(
  length(countFiles_filt) == nrow(sampleTable),
  all(sample_names_filt == rownames(sampleTable))
)


## run DEXSeq
dxd = DEXSeqDataSetFromHTSeq(
  countFiles_filt,
  sampleData=sampleTable,
  design= ~ sample + exon + condition:exon,
  flattenedfile=flattenedFile )

cnt <- counts(dxd)
keep <- rowSums(cnt > 2) >= 2
dxd <- dxd[keep, ]


#dxd = dxd[geneIDs( dxd ) %in% genesForSubset,]

## check the result!
#colData(dxd)
#counts(dxd)
#split( seq_len(ncol(dxd)), colData(dxd)$exon )
#head( featureCounts(dxd), 5 )
#head( rowRanges(dxd), 3 )
#sampleAnnotation( dxd )

############test############
#geoMeans <- apply(counts(dxd), 1, function(row) if (all(row == 0)) 0 else exp(mean(log(row[row != 0]))))

## normalize
geoMeans <- apply(counts(dxd), 1, function(row) {
  non_zero <- row[row > 0] # 0보다 큰 값만 추출
  if (length(non_zero) == 0) {
    return(0) # 전부 0이면 0 반환
  } else {
    return(exp(mean(log(non_zero)))) # 기하평균 계산
  }
})

dxdn = estimateSizeFactors( dxd )
## estimate dispersion + visualize
dxdn = estimateDispersions( dxdn )
plotDispEsts(dxdn)

## !!!!!!!!!! Differential Exon Usage test !!!!!!!!!!
dxdn = testForDEU( dxdn )
dxdn = estimateExonFoldChanges( dxdn, fitExpToVar="condition")
dxr1 = DEXSeqResults( dxdn )
dxr1
plotMA( dxr1, cex=0.8 )

##dev.off before drawing new plot of ggplot2
dev.off()


RandomCol <- distinctColorPalette(k = 14)

## visualize:BRCA1 = MSTRG.130765 / BRCA2 = MSTRG.86372
## ENSG00000012048.23 / ENSG00000139618.17
plotDEXSeq(dxr1, "ENSG00000012048.23", legend=TRUE,  displayTranscripts=FALSE, cex.axis=1.2, cex=1.3, lwd=2 )
## per sample
plotDEXSeq( dxr1, "ENSG00000012048.23", legend=TRUE,  expression=FALSE, norCounts=TRUE, displayTranscripts=FALSE, cex.axis=1.2, cex=1.3, lwd=2, color.samples = RandomCol )

## splicing
pdf("/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/BRCA2_DEXSeq.pdf", width=12, height=8)
plotDEXSeq( dxr1, FDR=0.10, "ENSG00000012048.23", legend=TRUE, splicing=TRUE, expression=FALSE, norCounts=FALSE, displayTranscripts=FALSE, names=FALSE, cex.axis=1.2, cex=1.2, lwd=2, color=c("#DA4343", "#3396D3"))
dev.off()

res <- as.data.frame(dxr1)
res_brca1 <- res[res$groupID == "MSTRG.130765", ]
res_brca1 <- res_brca1[res_brca1$featureID == "E016",]


## plot candidates ##

### post:"#Ea3b2a" | pre:"#0C5DE6" ###

default = c("#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2",
            "#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2") 

p068 = c("#E2E2E2","#Ea3b2a","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2",
         "#0C5DE6","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2") 
p070 = c("#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#Ea3b2a","#E2E2E2",
         "#E2E2E2","#E2E2E2","#E2E2E2","#0C5DE6","#E2E2E2","#E2E2E2","#E2E2E2") 
p107 = c("#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#Ea3b2a",
         "#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#0C5DE6","#E2E2E2","#E2E2E2") 
p035 = c("#Ea3b2a","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2",
         "#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#0C5DE6","#E2E2E2")
p142 = c("#E2E2E2","#E2E2E2","#Ea3b2a","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2",
         "#E2E2E2","#0C5DE6","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2") 
p048 = c("#E2E2E2","#E2E2E2","#E2E2E2","#Ea3b2a","#E2E2E2","#E2E2E2","#E2E2E2",
         "#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#0C5DE6") 
p069 = c("#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#Ea3b2a","#E2E2E2","#E2E2E2",
         "#E2E2E2","#E2E2E2","#0C5DE6","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2") 


plotDEXSeq( dxr1, "ENSG00000136492.4", legend=TRUE,  expression=FALSE, norCounts=TRUE, displayTranscripts=FALSE, cex.axis=1.4, cex=1, lwd=2.3, color.samples = p068)
ooo


