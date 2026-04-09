library(EnhancedVolcano)
library(ggplot2)

dut <- read.csv('/home/jiye/jiye/nanopore/HG_JH_check/cDUT/cDUT_volcano_data.csv', sep = '\t', header=TRUE)
highlight <- dut[dut$highlight == "True", "label"]
highlight2 <- dut[dut$log2FC > 1 & -log10(dut$pval) > 15, "label"]

# 초기화: 모두 회색
keyvals <- rep('#B4B4B8', nrow(dut))      # NS 색
names(keyvals) <- rep('NS', nrow(dut))

# Cancer-preferred: log2FC > 0 & pval < 0.05
keyvals[dut$log2FC > 0 & dut$pval < 0.05] <- '#E33C89'
names(keyvals)[dut$log2FC > 0 & dut$pval < 0.05] <- 'Cancer-preferred'

# Normal-preferred: log2FC < 0 & pval < 0.05
keyvals[dut$log2FC < 0 & dut$pval < 0.05] <- '#4B74B6'
names(keyvals)[dut$log2FC < 0 & dut$pval < 0.05] <- 'Normal-preferred'


dutplot <- EnhancedVolcano(dut,
                           lab = dut$label,
                           title = '',
                           subtitle = '',
                           titleLabSize = 0,
                           subtitleLabSize = 0,
                           legendLabSize = 10,
                           legendIconSize = 3.5,
                           x = 'log2FC',
                           y = 'pval',
                           ylab = expression(-~Log[10]~pval),
                           xlab = 'log2FC',
                           xlim = c(-4,4),
                           captionLabSize = 0,
                           pCutoff = 0.05,
                           FCcutoff = 0,
                           drawConnectors = TRUE,
                           boxedLabels = TRUE,
                           max.overlaps = 30,
                           labSize = 2, #genelabel
                           labCol = 'black',
                           labFace = 'bold',
                           axisLabSize = 12,
                           pointSize=3,
                           colAlpha = 0.6,
                           widthConnectors = 0.7,
                           legendPosition = 'bottom',
                           colConnectors = 'black',
                           
                           #selectLab = union(highlight, highlight2),
                           selectLab = highlight,
                           lengthConnectors = unit(0.05, 'npc'),
                           arrowheads = FALSE,
                           #legendLabels = c('NS', expression(dPSI),
                           #                  'pval', expression(pval~and~dPSI)),
                           colCustom = keyvals,
                           legendLabels = c('NS', 'Normal-preferred', 'Cancer-preferred'),
                           
                           )


dutplot

ggsave(filename = "/home/jiye/jiye/nanopore/HG_JH_check/cDUT/cDUT_volccano_CAGlabel.pdf", plot = dutplot, dpi = 300, width = 6, height = 7)
