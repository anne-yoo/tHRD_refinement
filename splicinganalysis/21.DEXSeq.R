###### DEXSEq!!!!!! #####
library("DEXSeq")
library("pasilla")
library("randomcoloR")

pythonScriptsDir = system.file( "python_scripts", package="DEXSeq" )
list.files(pythonScriptsDir)
system.file( "python_scripts", package="DEXSeq", mustWork=TRUE )

## find .txt files processed by HTSeq
inDir = '/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/resistance/brip1_R/DEXSeq'
countFiles = list.files(inDir, pattern="edited", full.names=TRUE)
basename(countFiles)

## find .gff file
inDir2 = '/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/resistance/brip1_R/'
flattenedFile = list.files(inDir2, pattern="gff", full.names=TRUE)
basename(flattenedFile)

## make df containing sample info
sampleTable = data.frame(
  row.names = c( "SV-OV-P035-atD", "SV-OV-P068-atD", "SV-OV-P142-atD","SV-OV-P048-atD", "SV-OV-P069-atD", "SV-OV-P070-atD", "SV-OV-P107-atD",
                 "SV-OV-P068-bfD","SV-OV-P142-bfD", "SV-OV-P069-bfD", "SV-OV-P070-bfD", "SV-OV-P107-bfD", "SV-OV-P035-bfD", "SV-OV-P048-bfD"),
  condition = c("post",  "post", "post", "post", "post", "post", "post",
                "pre", "pre", "pre", "pre", "pre", "pre", "pre" ))

#sampleTable = data.frame(
#  row.names = c("SV-OV-P107-atD","SV-OV-P107-bfD"),
#  condition = c("post","pre"))

## run DEXSeq
dxd = DEXSeqDataSetFromHTSeq(
  countFiles,
  sampleData=sampleTable,
  design= ~ sample + exon + condition:exon,
  flattenedfile=flattenedFile )

#dxd = dxd[geneIDs( dxd ) %in% genesForSubset,]

## check the result!
#colData(dxd)
#counts(dxd)
#split( seq_len(ncol(dxd)), colData(dxd)$exon )
#head( featureCounts(dxd), 5 )
#head( rowRanges(dxd), 3 )
#sampleAnnotation( dxd )

############test############
geoMeans <- apply(counts(dxd), 1, function(row) if (all(row == 0)) 0 else exp(mean(log(row[row != 0]))))

## normalize
dxdn = estimateSizeFactors( dxd , geoMeans = geoMeans)

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

## visualize
plotDEXSeq(dxr1, "ENSG00000136492.4", legend=TRUE,  displayTranscripts=FALSE, cex.axis=1.2, cex=1.3, lwd=2 )
## per sample
plotDEXSeq( dxr1[0:3,], "ENSG00000136492.4", legend=TRUE,  expression=FALSE, norCounts=TRUE, displayTranscripts=FALSE, cex.axis=1.2, cex=1.3, lwd=2, color.samples = RandomCol )
## splicing
plotDEXSeq( dxr1, "ENSG00000136492.4", legend=TRUE,  expression=FALSE, splicing=TRUE, displayTranscripts=FALSE, names=FALSE, cex.axis=1.3, cex=1.1, lwd=2 )


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


