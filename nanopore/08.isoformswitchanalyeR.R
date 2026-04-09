##isoformswitchanalyzeR

library("IsoformSwitchAnalyzeR")

#############only coding genes vs. all genes##################
tpm <- read.csv('/home/jiye/jiye/nanopore/FINALDATA/matched_transcript_TPM.txt', sep='\t', header=TRUE, check.names = F, row.names=1) ## only coding genes
count <- read.csv('/home/jiye/jiye/nanopore/FINALDATA/matched_transcript_readcount.txt', sep='\t', header=TRUE, check.names = F, row.names=1) ## only coding genes
#tpm <- read.csv('/home/jiye/jiye/nanopore/FINALDATA/wholeannot/matched_transcript_TPM.txt', sep='\t', header=TRUE, check.names  = F) ## all genes
#count <- read.csv('/home/jiye/jiye/nanopore/FINALDATA/wholeannot/matched_transcript_readcount.txt', sep='\t', header=TRUE, check.names  = F) ## all genes

rownames(tpm) <- sub("-.*", "", rownames(tpm))
rownames(count) <- sub("-.*", "", rownames(count))

#############################################################

myDesign <- data.frame(
  sampleID = colnames(tpm),
  condition = rep(c("normal", "tumor"), times = 141)
)

aSwitchList <- importRdata(
  isoformCountMatrix   = count,
  isoformRepExpression = tpm,
  designMatrix         = myDesign,
  isoformExonAnnoation =  "/home/jiye/jiye/nanopore/FINALDATA/291_merged.gtf",
  #isoformNtFasta       = system.file("extdata/example_isoform_nt.fasta.gz", package="IsoformSwitchAnalyzeR"),
  fixStringTieAnnotationProblem = TRUE,
  showProgress = FALSE
)

aSwitchList <- preFilter(
  aSwitchList,
  geneExpressionCutoff = 1,
  isoformExpressionCutoff = 0,
  removeSingleIsoformGenes = TRUE
)

aSwitchListAnalyzed <- isoformSwitchTestDRIMSeq(
  switchAnalyzeRlist = aSwitchList,
  reduceToSwitchingGenes = TRUE
)

exampleSwitchListAnalyzed <- analyzeCPAT(
    switchAnalyzeRlist   = aSwitchListAnalyzed,
    pathToCPATresultFile = "/home/jiye/jiye/nanopore/202411_analysis/isoformswitchanalyzer/cpat_results.txt",
    codingCutoff         = 0.725, # the coding potential cutoff we suggested for human
    removeNoncodinORFs   = TRUE   # because ORF was predicted de novo
)

### Add CPC2 analysis
exampleSwitchListAnalyzed <- analyzeCPC2(
    switchAnalyzeRlist   = exampleSwitchListAnalyzed,
    pathToCPC2resultFile = "/home/jiye/jiye/nanopore/202411_analysis/isoformswitchanalyzer/cpc2_result.txt",
    removeNoncodinORFs   = TRUE   # because ORF was predicted de novo
)

### Add PFAM analysis
exampleSwitchListAnalyzed <- analyzePFAM(
    switchAnalyzeRlist   = exampleSwitchListAnalyzed,
    pathToPFAMresultFile = "/home/jiye/jiye/nanopore/202411_analysis/isoformswitchanalyzer/pfam_results.txt",
    showProgress=FALSE
)

### Add SignalP analysis
exampleSwitchListAnalyzed <- analyzeSignalP(
    switchAnalyzeRlist       = exampleSwitchListAnalyzed,
    pathToSignalPresultFile  = "/home/jiye/jiye/nanopore/202411_analysis/isoformswitchanalyzer/signalP_results.txt"
)

### Add IUPred2A analysis
exampleSwitchListAnalyzed <- analyzeIUPred2A(
    switchAnalyzeRlist        = exampleSwitchListAnalyzed,
    pathToIUPred2AresultFile = "/home/jiye/jiye/nanopore/202411_analysis/isoformswitchanalyzer/iupred2a_result.txt.gz",
    showProgress = FALSE
)


### Add DeepLoc2 analysis
exampleSwitchListAnalyzed <- analyzeDeepLoc2(
    switchAnalyzeRlist = exampleSwitchListAnalyzed,
    pathToDeepLoc2resultFile = "/home/jiye/jiye/nanopore/202411_analysis/isoformswitchanalyzer/deeploc2.csv",
    quiet = FALSE
)

### Add DeepTMHMM analysis
exampleSwitchListAnalyzed <- analyzeDeepTMHMM(
    switchAnalyzeRlist   = exampleSwitchListAnalyzed,
    pathToDeepTMHMMresultFile = "/home/jiye/jiye/nanopore/202411_analysis/isoformswitchanalyzer/DeepTMHMM.gff3",
    showProgress=FALSE
)


# the consequences highlighted in the text above
consequencesOfInterest <- c('intron_retention','coding_potential','NMD_status','domains_identified','ORF_seq_similarity')

exampleSwitchListAnalyzed <- analyzeSwitchConsequences(
    exampleSwitchListAnalyzed,
    consequencesToAnalyze = consequencesOfInterest, 
    #dIFcutoff = 0.1, # very high cutoff for fast runtimes - you should use the default (0.1)
    showProgress=FALSE
)
