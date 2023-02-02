##ggtranscript!!!
#install.packages("devtools", dependencies=TRUE, INSTALL_opts = c('--no-lock'))
#devtools::install_github("dzhang32/ggtranscript")
library("magrittr")
library("dplyr")
library("ggtranscript")
library("ggplot2")
library("rtracklayer")

brip1 = read.csv('/home/jiye/jiye/copycomparison/OC_transcriptome/annotation/ggtranscript_BRIP1.gtf', sep='\t',header=TRUE)

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
    arrow.min.intron.length = 300
  )

################


# extract the two transcripts to be compared
brip1_rescaled_288_exons <- brip1_rescaled_exons %>% 
  dplyr::filter(transcript_name == "ENST00000577598.1")
brip1_rescaled_canon_exons <- brip1_rescaled_exons %>% 
  dplyr::filter(transcript_name == "ENST00000259008.2")

brip1_rescaled_288_exons %>%
  ggplot(aes(
    xstart = start,
    xend = end,
    y = "BRIP1-577598/259008"
  )) + theme_bw()+
  geom_half_range(fill='red') +
  geom_intron(
    data = to_intron(brip1_rescaled_288_exons, "transcript_name"), 
    arrow.min.intron.length = 200
  ) +
  geom_half_range(
    data = brip1_rescaled_canon_exons,
    range.orientation = "top", 
    fill = "purple"
  ) +
  geom_intron(
    data = to_intron(brip1_rescaled_canon_exons, "transcript_name"), 
    arrow.min.intron.length = 200
  )



#####


mane <- brip1_rescaled_exons %>% 
  dplyr::filter(transcript_name == "ENST00000259008.2")

target = c("ENST00000577598.1","MSTRG.49384.288","ENST00000577598.1")
not_mane <- brip1_rescaled_exons %>% 
  dplyr::filter(transcript_name =="ENST00000577598.1"|transcript_name =="MSTRG.49834.288"|transcript_name =="MSTRG.49834.293")

brip1_rescaled_diffs <- to_diff(
  exons = not_mane,
  ref_exons = mane,
  group_var = "transcript_name"
)

brip1_4_exons <- brip1_rescaled_exons %>% 
  dplyr::filter(transcript_name == "ENST00000259008.2"|transcript_name =="ENST00000577598.1"|transcript_name =="MSTRG.49834.288"|transcript_name =="MSTRG.49834.293")
  
brip1_4_exons %>%
  ggplot(aes(
    xstart = start,
    xend = end,
    y = transcript_name
  )) + theme_bw()+
  geom_range() +
  geom_intron(
    data = to_intron(brip1_4_exons),
    arrow.min.intron.length = 300
  ) 
