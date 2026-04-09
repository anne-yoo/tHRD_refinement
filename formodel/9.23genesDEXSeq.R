################################################################
## !!!!!! DEXSeq for the 23 final feature genes !!!!!!       ###   
### only responders: pre vs. post -> for resistance analysis ###
################################################################

library("DEXSeq")
library("pasilla")
library("randomcoloR")


## find .txt files processed by HTSeq
inDir = '/home/jiye/jiye/copycomparison/OC_transcriptome/featureanalysis/23genes_bamfiles/onlyR_4DEXSeq'
countFiles = list.files(inDir, pattern="edited", full.names=TRUE)
basename(countFiles)

## find .gff file
flattenedFile = list.files(inDir, pattern="gff", full.names=TRUE)
basename(flattenedFile)

## make df containing sample info
sampleTable = data.frame(
  row.names = c( "SV-OV-P035-atD", "SV-OV-P068-bfD", "SV-OV-P068-atD", "SV-OV-P142-bfD","SV-OV-P142-atD", "SV-OV-P048-atD", "SV-OV-P069-atD", 
                 "SV-OV-P069-bfD", "SV-OV-P070-atD","SV-OV-P070-bfD", "SV-OV-P107-bfD", "SV-OV-P035-bfD", "SV-OV-P048-bfD","SV-OV-P107-atD"),
  condition = c("post","pre","post","pre","post","post","post",
                "pre","post","pre","pre","pre","pre","post"))



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


#RandomCol <- distinctColorPalette(k = 14)

## visualize
##ASCC3: ENSG00000112249.9 / BRIP1: ENSG00000136492.4
plotDEXSeq(dxr1, "ENSG00000136492.4", legend=TRUE,  displayTranscripts=FALSE, cex.axis=1.2, cex=1.3, lwd=2 )
## per sample
plotDEXSeq( dxr1, "ENSG00000136492.4", legend=TRUE,  expression=FALSE, norCounts=TRUE, displayTranscripts=FALSE, cex.axis=1.2, cex=1.3, lwd=2 )
## splicing
plotDEXSeq( dxr1, "ENSG00000136492.4", legend=TRUE,  expression=FALSE, splicing=TRUE, displayTranscripts=FALSE, names=FALSE, cex.axis=1.3, cex=1.1, lwd=2 )


## plot candidates ##

### post:"#Ea3b2a" | pre:"#0C5DE6" ###

default = c("#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2",
            "#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2","#E2E2E2") 


p035 = c("#Ea3b2a", "#E2E2E2", "#E2E2E2", "#E2E2E2","#E2E2E2", "#E2E2E2", "#E2E2E2", 
         "#E2E2E2", "#E2E2E2","#E2E2E2", "#E2E2E2", "#0C5DE6", "#E2E2E2","#E2E2E2")

p048 = c("#E2E2E2","#E2E2E2", "#E2E2E2", "#E2E2E2","#E2E2E2", "#Ea3b2a", "#E2E2E2", 
         "#E2E2E2", "#E2E2E2","#E2E2E2", "#E2E2E2", "#E2E2E2", "#0C5DE6","#E2E2E2") 


#ASCC3 per sample
plotDEXSeq( dxr1, "ENSG00000136492.4", legend=TRUE,  expression=FALSE, norCounts=TRUE, displayTranscripts=TRUE, cex.axis=1.4, cex=1, lwd=2.3, color.samples = p048)

######################################################################################################

drawPlot <- function(matr, ylimn, ecs, intervals, rango, fitExpToVar, numexons, textAxis, rt, color, colorlines, ...)
{
  plot.new()
  plot.window(xlim=c(0, 1), ylim=c(0, max(matr)))
  makevstaxis(1/ncol(matr), ylimn, ecs, ...)
  intervals<-(0:nrow(matr))/nrow(matr)
  middle <- apply(cbind(intervals[rango], (intervals[rango+1]-((intervals[rango+1])-intervals[rango])*0.2)), 1, median)
  matr<-rbind(matr, NA)
  j <- seq_len(ncol(matr))
  segments(intervals[rango], matr[rango,j], intervals[rango+1]-((intervals[rango+1]-intervals[rango])*0.2), matr[rango,j], col=color, ...)  #### line with the y level
  segments(intervals[rango+1]-((intervals[rango+1]-intervals[rango])*0.2), matr[rango,j], intervals[rango+1], matr[rango+1,j], col=color, lty="dotted", ...)  #### line joining the y levels
  abline(v=middle[rango], lty="dotted", col=colorlines)
  mtext(textAxis, side=2, adj=0.5, line=1.5, outer=FALSE, ...)
  axis(1, at=middle[seq(along=rt)], labels=featureIDs(ecs)[rt], ...)
}

