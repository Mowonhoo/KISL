# 加载包
library(DOSE)
library(org.Hs.eg.db)
library(topGO)
library(clusterProfiler)
library(pathview)

rm(list=ls())

workdir <- "./KISL"
filename = "BLCA.tcga_gtex.tpm.updown.feature_selection.csv"
indir <- "./outdir/feature selection"
outdir <- "./outdir/enrichment analysis"

setwd(workdir)

prefix = unlist(strsplit(filename, split="[.]"), recursive = FALSE)[[1]]
outdir <- paste(outdir, prefix, sep='/')

if(!dir.exists(outdir)){dir.create(outdir, recursive = TRUE)}

exp.matrix <- read.csv(file = paste(indir, prefix, filename, sep="/"), sep=",", row.names = 1, check.names = F)

geneslist <-  rownames(exp.matrix)

# 转换基因名
transID = bitr(geneslist,
               fromType="SYMBOL",
               toType=c("ENSEMBL", "ENTREZID"),
               OrgDb="org.Hs.eg.db"
)

# GO富集
if(!dir.exists(paste(outdir, "GO", sep='/'))){dir.create(paste(outdir, "GO", sep='/'), recursive = TRUE)}

## GO_BP注释
BP <- enrichGO(transID$ENTREZID, "org.Hs.eg.db", keyType="ENTREZID", ont="BP", pvalueCutoff=0.05, pAdjustMethod="BH", qvalueCutoff=0.1)
BP <- setReadable(BP, OrgDb=org.Hs.eg.db)

pdf(file=paste(outdir,"GO", paste0(prefix, ".GO_BP.pdf"), sep="/"), bg="transparent")
dotplot(BP, showCategory=12, font.size=8, title="GO_BP") # + theme(axis.text.y = element_text(angle = 45))
barplot(BP, showCategory=12, title="GO_BP", font.size=8)
plotGOgraph(BP)
dev.off()

write.table(as.data.frame(BP@result), file=paste(outdir, "GO", paste0(prefix, ".GO_BP.txt"), sep='/'), quote = F, sep="\t", row.names=F)

sprintf("--- DONE! ---")

