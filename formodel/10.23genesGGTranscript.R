#### GGTranscript for 23 genes !!!!! ####

library("magrittr")
library("dplyr")
library("ggtranscript")
library("ggplot2")
library("rtracklayer")

path = '/home/jiye/jiye/copycomparison/OC_transcriptome/annotation/gencode.v19.annotation.gtf'
gtf <- rtracklayer::import(path)
gtf <- gtf %>% dplyr::as_tibble()

class(gtf)

################ filter your gtf for the gene of interest ##################
gene_of_interest <- "ASCC3"
############################################################################

gene_annotation_from_gtf <- gtf %>% 
  dplyr::filter(
    !is.na(gene_name), 
    gene_name == gene_of_interest
  ) 

# extract the required annotation columns
gene_annotation_from_gtf <- gene_annotation_from_gtf %>% 
  dplyr::select(
    seqnames,
    start,
    end,
    strand,
    type,
    gene_name,
    transcript_name,
    transcript_type
  )

gene_annotation_from_gtf %>% head()


# to illustrate the package's functionality
# ggtranscript includes example transcript annotation
gene_annotation <- gene_annotation_from_gtf

# extract exons
gene_exons <- gene_annotation %>% dplyr::filter(type == "exon")

gene_exons %>%
  ggplot(aes(
    xstart = start,
    xend = end,
    y = transcript_name
  )) + theme_bw()+
  geom_range(
    aes(fill = transcript_type)
  ) +
  geom_intron(
    data = to_intron(gene_exons, "transcript_name"),
    aes(strand = strand)
  )


# extract exons
gene_exons <- gene_annotation %>% dplyr::filter(type == "exon")

gene_rescaled <- shorten_gaps(
  exons = gene_exons, 
  introns = to_intron(gene_exons, "transcript_name"), 
  group_var = "transcript_name"
)

# shorten_gaps() returns exons and introns all in one data.frame()
# let's split these for plotting 
gene_rescaled_exons <- gene_rescaled %>% dplyr::filter(type == "exon") 
gene_rescaled_introns <- gene_rescaled %>% dplyr::filter(type == "intron") 

gene_rescaled_exons %>% 
  ggplot(aes(
    xstart = start,
    xend = end,
    y = transcript_name
  )) +theme_bw()+
  geom_range(
    aes(fill = transcript_type)
  ) +
  geom_intron(
    data = gene_rescaled_introns,
    aes(strand = strand), 
    arrow.min.intron.length = 300
  )

####### compare 2 transcripts!! ###########
gene_rescaled_001_exons <- gene_rescaled_exons %>% 
dplyr::filter(transcript_name == "ASCC3-001")
gene_rescaled_007_exons <- gene_rescaled_exons %>% 
  dplyr::filter(transcript_name == "ASCC3-007")

gene_rescaled_001_exons %>%
  ggplot(aes(
    xstart = start,
    xend = end,
    y = "ASCC3-001 / ASCC3-007"
  )) +theme_bw()+
  geom_half_range() +
  geom_intron(
    data = to_intron(gene_rescaled_001_exons, "transcript_name"), 
    arrow.min.intron.length = 300
  ) +
  geom_half_range(
    data = gene_rescaled_007_exons,
    range.orientation = "top", 
    fill = "purple"
  ) +
  geom_intron(
    data = to_intron(gene_rescaled_007_exons, "transcript_name"), 
    arrow.min.intron.length = 300
  )

