########## Volcano plot ##############

library(EnhancedVolcano)
library(DEXSeq)
library(stageR)
library(ggplot2)
library(DRIMSeq)
library(DESeq2)

splicinggenes <- readLines("/home/jiye/jiye/copycomparison/gDUTresearch/data/Splicing_genes.txt")

det <- read.csv('/home/jiye/jiye/nanopore/202411_analysis/DESeq2/deseq2_det.txt', sep='\t', header=TRUE, check.names = F)
deg <- read.csv('/home/jiye/jiye/copycomparison/gDUTresearch/202310_analysis/DEG/responder_Wilcoxon_DEGresult_FC.txt',  sep="\t", header=TRUE, check.names = F, row.names = 1)
dsg <- read.csv('/home/jiye/jiye/copycomparison/gDUTresearch/suppa2/dpsi_analysis/MW_dpsi_5events.txt', sep='\t', header=TRUE, check.names = F)
deg <- read.csv('/home/jiye/jiye/nanopore/202411_analysis/DESeq2/deseq2_deg.txt', sep='\t', header = TRUE, check.names = F)
deg <- read.csv('/home/jiye/jiye/copycomparison/GENCODEquant/POLO_hg38/analysis/majorexp_DEG.txt', sep='\t', header = TRUE, check.names = F)

dsg <- read.csv('/home/jiye/jiye/nanopore/202411_analysis/SUPPA2/AS_summary.txt', sep="\t", header=TRUE, check.names = F, row.names = 1)
dsg <- dsg[complete.cases(dsg), ]
dsg$MSTRG <- sub(";.*", "", rownames(dsg))
idmatch <- read.table("/home/jiye/jiye/nanopore/gtfcompare/stringtiesample/filtered_transcripts_with_gene_info.tsv", header = TRUE, sep = "\t", stringsAsFactors = FALSE)
dsg$genename <- idmatch$gene_name[match(dsg$MSTRG, idmatch$mstrg_gene_id)]
write.table(dsg, file = '/home/jiye/jiye/nanopore/202411_analysis/SUPPA2/AS_event_genename.txt',sep = '\t', row.names = TRUE, col.names = TRUE, quote = FALSE)

#idtosym <- read.csv('/home/jiye/jiye/copycomparison/gDUTresearch/data/ensgidtosymbol.dict',  sep="\t", header=FALSE, check.names = F, row.names = 1)
#colnames(idtosym) <- c('Gene')
#sym_deg <- merge(deg,idtosym,by='row.names')

#dtu <- read.csv('/home/jiye/jiye/copycomparison/gDUTresearch/DTUDEGcomp/DEXSeq_DTU_result.txt', sep=",", header=TRUE, check.names = F, row.names = 1)
dtu <- read.csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/responder_stable_DUT_Wilcoxon.txt', sep='\t', header = TRUE, check.names = F, row.names=1)

deg <- read.csv('/home/jiye/jiye/copycomparison/gDUTresearch/202411_analysis/whole_MW_DUTresult_FC.txt', sep='\t', header = TRUE, check.names = F, row.names=1)

#both <- read.csv('/home/jiye/jiye/copycomparison/gDUTresearch/DTUDEGcomp/volcano_DEGDTU.txt', sep=",", header=TRUE, check.names = F, row.names = 1)

############## ONLY SPLICING GENES ##################
#splicinggenes <- c('SF3B1', 'PTBP1', 'CUGBP2', 'YBX1', 'PCBP1', 'DAZAP1', 'A2BP1', 'KHSRP', 'SFRS13A', 'HNRNPC', 'ELAVL1', 'HNRNPM', 'RBFOX2', 'SFRS5', 'FMR1', 'ESRP2', 'HNRNPA3', 'HNRNPL', 'NOVA2', 'ELAVL2', 'SFRS9', 'SFRS3', 'KHDRBS2', 'PTBP2', 'ELAVL4', 'SFRS7', 'HNRNPA2B1', 'SFRS4', 'RBMX', 'HNRNPK', 'RBM25', 'HNRPDL', 'CUGBP1', 'KHDRBS1', 'HNRNPA0', 'SFRS6', 'HNRNPH2', 'KHDRBS3', 'HNRNPH3', 'SRRM1', 'NOVA1', 'SFPQ', 'SFRS1', 'TRA2B', 'HNRNPD', 'SYNCRIP', 'HNRPLL', 'SFRS2', 'RBM5', 'TIAL1', 'MBNL1', 'SF1', 'HNRNPU', 'ZRANB2', 'QKI', 'PCBP2', 'HNRNPH1', 'ELAVL3', 'SFRS11', 'TARDBP', 'HNRNPF', 'ESRP1', 'RBM4', 'TIA1', 'HNRNPA1', 'TRA2A', 'FUS'
)
deg_tmp <- deg[deg$genename %in% splicinggenes, ]
DE_splicegenes <- deg_tmp[(deg_tmp$padj < 0.1) & (abs(deg_tmp$log2FoldChange) > 0.75),]$genename
#####################################################

dsgplot <- EnhancedVolcano(dsg,
                           lab = dsg$genename,
                           title = 'AS',
                           subtitle = 'cutoff: pval=0.05, dPSI=0.1',
                           titleLabSize = 0.1,
                           subtitleLabSize = 0.1,
                           legendLabSize = 10,
                           legendIconSize = 3.5,
                           x = 'dPSI', #log2FC
                           y = 'pval',
                           ylab = expression(-~Log[10]~pval),
                           xlab = 'dPSI',
                           #selectLab = DE_splicegenes,
                           ylim = c(0,4),
                           xlim = c(-0.6,0.6),
                           captionLabSize = 0,
                           pCutoff = 0.05,
                           FCcutoff = 0.1,
                           drawConnectors = TRUE,
                           boxedLabels = TRUE,
                           max.overlaps = 30,
                           labSize = 2.5,#genelabel
                           labCol = 'black',
                           labFace = 'bold',
                           axisLabSize = 12,
                           pointSize=2.5,
                           col = c('#B4B4B8', '#B4B4B8', '#86A3B8', '#F48484'),
                           colAlpha = 4/5,
                           widthConnectors = 0.7,
                           legendPosition = 'bottom',
                           #colConnectors = '#FFFF80',
                           #colConnectors = 'white',
                           #selectLab = c('BRCA1','BRCA2','RAD51', 'ATM', 'PALB2',
                           #              'EXO1','BRIP1','BARD1','ATR','CHEK2','CDK12','RIF1','REV7',
                           #              'PTEN','MRE11','TP53BP1','BRIP1'),
                           lengthConnectors = unit(0.03, 'npc'),
                           arrowheads = FALSE,
                           legendLabels = c('NS', expression(dPSI),
                                            'pval', expression(pval~and~dPSI)))


dsgplot

ggsave(filename = "/home/jiye/jiye/nanopore/202411_analysis/SUPPA2/AS_volcano.pdf", plot = dsgplot, dpi = 300, width = 6, height = 7)

deg <- read.csv('/home/jiye/jiye/copycomparison/gDUTresearch/202501_analysis/VALDUT/noAR_maintenance_DEGresult_FC.txt', sep='\t', header = TRUE, check.names = F)
deg <- deg[is.finite(deg$log2FC) & is.finite(deg$p_value), ]
degplot <- EnhancedVolcano(deg,
                lab = deg$genename,
                title = 'DEG',
                subtitle = 'cutoff: pval=0.05, log2FC=1',
                titleLabSize = 0.1,
                subtitleLabSize = 0.1,
                legendLabSize = 12,
                legendIconSize = 3.5,
                x = 'log2FoldChange', #log2FC
                y = 'padj',
                ylab = expression(-~Log[10]~adjp),
                xlab = 'log2FC',
                #selectLab = DE_splicegenes,
                ylim = c(0,4),
                xlim = c(-7,7),
                captionLabSize = 0,
                pCutoff = 0.1,
                FCcutoff = 0.75,
                drawConnectors = TRUE,
                boxedLabels = TRUE,
                max.overlaps = 25,
                labSize = 3,#genelabel
                labCol = 'black',
                labFace = 'bold',
                axisLabSize = 12,
                pointSize=3,
                col = c('#B4B4B8', '#B4B4B8', '#86A3B8', '#F48484'),
                colAlpha = 4/5,
                widthConnectors = 0.7,
                legendPosition = 'bottom',
                colConnectors = '#FFFF80',
                #colConnectors = 'white',
                selectLab = c('BRCA1','BRCA2','RAD51', 'ATM', 'PALB2',
                              'EXO1','BRIP1','CHEK2','CDK12', #'RIF1','REV7',
                              'PTEN','MRE11','TP53BP1','BRIP1','UBL5','WDR83','CELF6','RBFOX3'),
                #selectLab = DE_splicegenes,
                lengthConnectors = unit(0.08, 'npc'),
                arrowheads = FALSE,
                legendLabels = c('NS', expression(Log[2]~FC),
                                 'adjp', expression(adjp~and~log[2]~FC)))
                


degplot

ggsave(filename = "/home/jiye/jiye/copycomparison/gDUTresearch/withYNK/POLO_paper/con1_DEG_volcano_parp.pdf", plot = degplot, dpi = 300, width = 5, height = 6)

detplot <- EnhancedVolcano(det,
                           lab = det$transcript_id,
                           title = 'DEG',
                           subtitle = 'cutoff: pval=0.05, log2FC=1',
                           titleLabSize = 0.1,
                           subtitleLabSize = 0.1,
                           legendLabSize = 10,
                           legendIconSize = 3.5,
                           x = 'log2FoldChange', #log2FC
                           y = 'padj',
                           ylab = expression(-~Log[10]~pval),
                           xlab = 'log2FC',
                           #selectLab = DE_splicegenes,
                           ylim = c(0,100),
                           xlim = c(-10,10),
                           captionLabSize = 0,
                           pCutoff = 0.01,
                           FCcutoff = 2,
                           drawConnectors = FALSE,
                           boxedLabels = FALSE,
                           max.overlaps = 15,
                           labSize = 0,#genelabel
                           labCol = 'black',
                           labFace = 'bold',
                           axisLabSize = 12,
                           pointSize=3,
                           col = c('#B4B4B8', '#B4B4B8', '#86A3B8', '#F48484'),
                           colAlpha = 4/5,
                           widthConnectors = 0.7,
                           legendPosition = 'bottom',
                           #colConnectors = '#FFFF80',
                           #colConnectors = 'white',
                           #selectLab = c('BRCA1','BRCA2','RAD51', 'ATM', 'PALB2',
                           #              'EXO1','BRIP1','BARD1','ATR','CHEK2','CDK12','RIF1','REV7',
                           #              'PTEN','MRE11','TP53BP1','BRIP1'),
                           lengthConnectors = unit(0.03, 'npc'),
                           arrowheads = FALSE,
                           legendLabels = c('NS', expression(Log[2]~FC),
                                            'adjp', expression(adjp~and~log[2]~FC)))



detplot

ggsave(filename = "/home/jiye/jiye/nanopore/202411_analysis/figures/DET_volcano.pdf", plot = detplot, dpi = 300, width = 5, height = 6)

dtu <- read.csv('/home/jiye/jiye/copycomparison/gDUTresearch/202311_analysis/DUT/nonresponder_stable_DUT_Wilcoxon.txt', sep='\t', header = TRUE, check.names = F, row.names=1)
dtu <- dtu[!is.infinite(dtu$log2FC),]
#dtu <- dtu[1:100,]
select_trans = c('ENST00000584322.1-BRIP1','ENST00000497488.1-BRCA1','ENST00000530893.2-BRCA2','ENST00000531277.2-RAD51',
                 'ENST00000532765.1-ATM','ENST00000567003.1-PALB2','ENST00000450748.1-EXO1','ENST00000456369.1-CHEK2',
                 'ENST00000434595.1-TP53BP1','ENST00000462694.1-PTEN','MSTRG.56952.27-RIF1','MSTRG.67243.46-CHEK2','ENST00000558240.1-CDK12')

select_trans_2 = c('ENST00000494123.1-BRCA1','ENST00000542785.1-RAD52','ENST00000584322.1-BRIP1','MSTRG.67258.19-CHEK2','MSTRG.44983.145-CDK12','MSTRG.56952.17-RIF1','MSTRG.44983.124-CDK12','ENST00000487939.1-PTEN','ENST00000571145.1-TP53BP1')
dtuplot <- EnhancedVolcano(dtu,
                lab = rownames(dtu),
                title = 'Differential Transcript Usage',
                subtitle = 'cutoff: pval=0.05, log2FC=1.5',
                x = 'log2FC',
                y = 'p_value',
                ylab = expression(-~Log[10]~pval),
                xlab = 'log2FC',
                pCutoff = 0.05,
                FCcutoff = 1.5,
                #xlim = c(-10,10),
                ylim = c(0,5),
                selectLab = select_trans_2,
                titleLabSize = 0.1,
                subtitleLabSize = 0.1,
                legendLabSize = 12,
                legendIconSize = 3.5,
                captionLabSize = 0,
                #drawConnectors = TRUE,
                boxedLabels = TRUE,
                max.overlaps = 17,
                labSize = 3,#genelabel
                labCol = 'black',
                labFace = 'bold',
                axisLabSize = 16,
                pointSize = 2,
                col = c('#B4B4B8', '#B4B4B8', '#86A3B8', '#F48484'),
                colAlpha = 4/5,
                #widthConnectors = 0.7,
                legendPosition = 'bottom',
                #colConnectors = '#FFFF80',
                #lengthConnectors = unit(0.03, 'npc'),
                #arrowheads = FALSE,
                legendLabels = c('NS', expression(Log[2]~FC),
                                 'adjp', expression(adjp~and~log[2]~FC)),
               drawConnectors = TRUE,
               widthConnectors = 0.7,
               colConnectors = '#FFFF80'
)

dtuplot

#ggsave(filename = "/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/IR_DUT_volcano_translabel.pdf", plot = dtuplot, dpi = 300, width = 6, height = 7)



  
filtered_dtu = dtu[dtu$p_value<0.05,]
#filtered_dtu = filtered_dtu[filtered_dtu$log2FC<5,]
#filtered_dtu = filtered_dtu[filtered_dtu$log2FC>-5,]
filtered_dtu =  filtered_dtu[!is.infinite(filtered_dtu$log2FC),]
filtered_deg = deg[!is.infinite(deg$log2FC),]
merged <- merge(filtered_dtu, filtered_deg, by="Gene Symbol", all.x = TRUE)

######## dtu: log2fc.x / deg: log2fc.y ###########
######## dtu: pval / deg: pvalue ###########

source("/home/jiye/jiye/copycomparison/copycomparisoncode/gDUTmodel/modifiedEnhancedVolcano.R")


fcplot <- modifiedEnhancedVolcano(merged,
                lab = merged$`Gene Symbol`,
                title = 'transcript vs. gene unit expression',
                subtitle = 'cutoff: log2FC=1.5',
                titleLabSize = 1,
                subtitleLabSize = 1,
                legendLabSize = 12,
                legendIconSize = 4,
                x = 'log2FC.x',
                y = 'log2FC.y',
                xlab = 'transcript expression log2fc',
                ylab = 'gene expression log2fc',
                pCutoff = 1,
                FCcutoff = 1,
                ylim = c(-3.5,3.5),
                xlim = c(-3.5,3.5),
                drawConnectors = FALSE,
                widthConnectors = 0.35,
                col = c("grey60",  "orange", "skyblue", "grey60"),
                max.overlaps = 1,
                labSize = 0,
                axisLabSize = 12,
                captionLabSize = 0,
                pointSize=2,
                legendLabels=c('NS', 'gene & transcript exp.', 'gene exp.','transcript exp.'))
fcplot


ggsave(filename = "/home/jiye/jiye/copycomparison/gDUTresearch/finalanalysis/finalfigures/fig1/FCcompare_gene_trans.pdf", plot = fcplot, dpi = 300, width = 8, height = 8)

threshold <- 1
library(dplyr)
# Create a new column to classify each point into one of the nine areas
data <- merged %>%
  mutate(area = case_when(
    log2FC.x > threshold & log2FC.y > threshold ~ "Top Right",
    log2FC.x > threshold & abs(log2FC.y) <= threshold ~ "Middle Right",
    log2FC.x > threshold & log2FC.y < -threshold ~ "Bottom Right",
    abs(log2FC.x) <= threshold & log2FC.y > threshold ~ "Top Center",
    abs(log2FC.x) <= threshold & abs(log2FC.y) <= threshold ~ "Center",
    abs(log2FC.x) <= threshold & log2FC.y < -threshold ~ "Bottom Center",
    log2FC.x < -threshold & log2FC.y > threshold ~ "Top Left",
    log2FC.x < -threshold & abs(log2FC.y) <= threshold ~ "Middle Left",
    log2FC.x < -threshold & log2FC.y < -threshold ~ "Bottom Left"
  ))

area_counts <- data %>%
  group_by(area) %>%
  summarise(count = n()) %>%
  mutate(percentage = (count / sum(count)) * 100)

# Print the counts and percentages
print(area_counts)

