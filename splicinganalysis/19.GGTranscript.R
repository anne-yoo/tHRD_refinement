##ggtranscript!!!
#install.packages("devtools", dependencies=TRUE, INSTALL_opts = c('--no-lock'))
#devtools::install_github("dzhang32/ggtranscript")
library("magrittr")
library("dplyr")
library("ggtranscript")
library("ggplot2")
library("rtracklayer")

brip1 = read.csv('/home/jiye/jiye/copycomparison/OC_transcriptome/annotation/brip1only4_ggtranscriptinput.txt', sep='\t',header=TRUE)

brip1 <- brip1 %>% dplyr::filter(strand == "-")
brip1_exons <- brip1 %>% dplyr::filter(type == "exon")
brip1_exons %>%
    ggplot(aes(
        xstart = start,
        xend = end,
        y = transcript_name
    )) + theme_bw()+
    geom_range(
        aes(fill = transcript_biotype)
    ) +
    geom_intron(
        data = to_intron(brip1_exons, "transcript_name"),
        aes(strand = strand)
    )

################

# extract exons
brip1_exons <- brip1 %>% dplyr::filter(type == "exon")

brip1_rescaled <- shorten_gaps(
  exons = brip1_exons, 
  introns = to_intron(brip1_exons, "transcript_name"), 
  group_var = "transcript_name"
)

# shorten_gaps() returns exons and introns all in one data.frame()
# let's split these for plotting 
brip1_rescaled_exons <- brip1_rescaled %>% dplyr::filter(type == "exon") 
brip1_rescaled_introns <- brip1_rescaled %>% dplyr::filter(type == "intron") 

brip1_rescaled_exons %>% 
  ggplot(aes(
    xstart = start,
    xend = end,
    y = transcript_name
  )) + theme_bw()+
  geom_range(
    aes(fill = transcript_biotype)
  ) +
  geom_intron(
    data = brip1_rescaled_introns,
    aes(strand = strand), 
    arrow.min.intron.length = 50
  )

################




#####
mane <- brip1_rescaled_exons %>% 
  dplyr::filter(transcript_name == "ENST00000259008.2")

not_mane <- brip1_rescaled_exons %>% 
  dplyr::filter(transcript_name != "ENST00000259008.2")

brip1_rescaled_diffs <- to_diff(
  exons = not_mane,
  ref_exons = mane,
  group_var = "transcript_name"
)

brip1_rescaled_diffs <- brip1_rescaled_diffs %>%
  dplyr::filter(diff_type != "in_ref")

brip1_rescaled_exons %>%
  ggplot(aes(
    xstart = start,
    xend = end,
    y = transcript_name
  )) +
  geom_range() +
  geom_intron(
    data = brip1_rescaled_introns,
    arrow.min.intron.length = 300
  ) +
  geom_range(
    data = brip1_rescaled_diffs,
    aes(fill = diff_type),
    alpha = 0.2
  )

