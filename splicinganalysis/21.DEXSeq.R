###### DEXSEq!!!!!! #####
library("DEXSeq")
library("pasilla")
pythonScriptsDir = system.file( "python_scripts", package="DEXSeq" )
list.files(pythonScriptsDir)
system.file( "python_scripts", package="DEXSeq", mustWork=TRUE )

## find .txt files processed by HTSeq
inDir = '/home/jiye/jiye/copycomparison/OC_transcriptome/splicing/resistance/brip1_R/DEXSeq'
countFiles = list.files(inDir, pattern="edited", full.names=TRUE)
basename(countFiles)

## find .gff file
flattenedFile = list.files(inDir, pattern="gff", full.names=TRUE)
basename(flattenedFile)

## make df containing sample info
sampleTable = data.frame(
  row.names = c( "SV-OV-P035-atD", "SV-OV-P068-atD", "SV-OV-P142-atD","SV-OV-P048-atD", "SV-OV-P069-atD", "SV-OV-P070-atD", "SV-OV-P107-atD",
                 "SV-OV-P068-bfD", "SV-OV-P142-bfD", "SV-OV-P069-bfD", "SV-OV-P070-bfD", "SV-OV-P107-bfD", "SV-OV-P035-bfD", "SV-OV-P048-bfD"),
  condition = c("post", "post", "post", "post", "post", "post", "post",  
                "pre", "pre", "pre", "pre", "pre", "pre", "pre" ))

## run DEXSeq
dxd = DEXSeqDataSetFromHTSeq(
  countFiles,
  sampleData=sampleTable,
  design= ~ sample + exon + condition:exon,
  flattenedfile=flattenedFile )

dxd = dxd[geneIDs( dxd ) %in% genesForSubset,]

## check the result!
head( counts(dxd), 10 )
split( seq_len(ncol(dxd)), colData(dxd)$exon )
head( featureCounts(dxd), 5 )
head( rowRanges(dxd), 3 )
sampleAnnotation( dxd )

## normalize
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

## visualize
plotDEXSeq( dxr1, "MSTRG.49834", legend=TRUE,  displayTranscripts=FALSE, cex.axis=1.2, cex=1.3, lwd=2 )
## per sample
plotDEXSeq( dxr1, "MSTRG.49834", legend=TRUE,  expression=FALSE, norCounts=TRUE, displayTranscripts=TRUE, cex.axis=1.2, cex=1.3, lwd=2 )
## splicing
plotDEXSeq( dxr1, "MSTRG.49834", legend=TRUE,  expression=FALSE, splicing=TRUE, displayTranscripts=FALSE, cex.axis=1.2, cex=1.3, lwd=2 )







