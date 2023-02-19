library("optparse")

option_list = list(
  make_option(c("-j", "--ADJ_file"), type = "character", default = NULL, help = "input files [default = %default]", metavar = "character"),
  make_option(c("-j", "--TOM_file"), type = "character", default = NULL, help = "input files [default = %default]", metavar = "character"),
  make_option(c("-j", "--Gene_Expression_Profile_File"), type = "character", default = NULL, help = "input files [default = %default]", metavar = "character"),
  make_option(c("-j", "--cluster_label_file"), type = "character", default = NULL, help = "input files [default = %default]", metavar = "character"),
  make_option(c("-o", "--outdir"), type = "character", default = ".", help = "result file directory [default = %default]"),
  make_option(c("-x", "--prefix"), type = "character", default = ".", help = "output prefix [default = %default]")
)


opt_parser = OptionParser(option_list = option_list)
opt = parse_args(opt_parser)

outdir <- opt$outdir
prefix <- opt$prefix
TOMdir = opt$TOM_file
ADJdir <- opt$ADJ_file
datExprDir <- opt$Gene_Expression_Profile_File
labelsDir <- opt$cluster_label_file


library(WGCNA)
library(dplyr)
library(stringr)
# options(stringsAsFactors = FALSE)



if(!dir.exists(outdir)){dir.create(outdir, recursive = TRUE)}


# prefix = unlist(strsplit(basename(datExprDir), split="[.]"), recursive = FALSE)[[1]] #unlist(strsplit(filename, split = "\\."))[2]


ADJ <- read.delim(ADJdir, sep=",")

k = as.vector(apply(ADJ, 1, sum, na.rm=T))
 
dir1=paste(outdir, paste0(prefix, ".Check Scale free topology.png"), sep="/")
png(file=dir1,width = 2000, height = 1800, res = 300)
par(mfrow = c(1, 2))
hist(k)
scaleFreePlot(k, main = 'Check Scale free topology\n')
dev.off()

pdf(file=paste(outdir, paste0(prefix, ".Check Scale free topology.pdf"), sep="/"))
par(mfrow = c(1, 2))
hist(k)
scaleFreePlot(k, main = 'Check Scale free topology\n')
dev.off()

df <- read.delim(datExprDir, sep=',', row.names = 1, check.names = FALSE)#
df <- df[complete.cases(df), ]

cat(sum(is.na(df)))
symbolID <- rownames(df)
sampleID <- colnames(df)
df <- as.data.frame(lapply(df, as.numeric), col.names = sampleID, row.names = symbolID, check.names = FALSE)

datExpr <- t(df)#subset(df, rowSums(df)/ncol(df) >= 1)
df <- data.frame(df, stringsAsFactors = F, check.names = FALSE)


datTraits <- data.frame(sample_type=as.factor(as.numeric(grepl('TCGA', sampleID))), row.names = sampleID, check.names = FALSE)

nGenes = ncol(datExpr)
nSamples = nrow(datExpr)

cluster_labels <-  read.delim(labelsDir, sep=",")

MEList <- moduleEigengenes(datExpr, colors = cluster_labels$labels)
MEs <- MEList$eigengenes
head(MEs)
MEs <- orderMEs(MEs)

dir_moduleEigengenes = paste(outdir, paste0(prefix, ".moduleEigengenes.csv"),sep="/")
write.csv(MEs, dir_moduleEigengenes, row.names = TRUE, quote = FALSE)

ME_cor <- cor(MEs)
head(ME_cor)

METree <- hclust(as.dist(1-ME_cor), method = "average")
dir_eigengenes=paste(outdir, paste0(prefix, ".eigengenes.png"), sep="/")
png(file=dir_eigengenes, bg="transparent",width = 2000,height = 1800,res = 300)

plot(METree, main = "Clustering of module eigengenes", xlab = "", sub = "")
rect(1, 5, 3, 7, col="white")
dev.off()


dir_plotEigengeneNetworks=paste(outdir, paste0(prefix, ".plotEigengeneNetworks.png"), sep="/")
png(file=dir_plotEigengeneNetworks, bg="transparent",width = 2000,height = 1800,res = 300)

plotEigengeneNetworks(MEs, "", cex.lab = 0.8, xLabelsAngle= 90,
                      marDendro = c(0, 4, 1, 2), marHeatmap = c(3, 4, 1, 2))
dev.off()

moduleTraitCor <- data.frame(cor(MEs, datTraits, use="p"))
moduleTraitPvalue = corPvalueStudent(moduleTraitCor$sample_type, nSamples)

#sizeGrWindow(10, 6)
textMatrix <-  paste(signif(moduleTraitCor$sample_type, 2), "\n(",
                     signif(moduleTraitPvalue, 1), ")", sep="")
dim(textMatrix) <- dim(moduleTraitCor)

pdf(file=paste(outdir, paste0(prefix, ".modle_and_Traits_heatmap.pdf"),sep="/"), width=4, height=10)
# sizeGrWindow(3, 10)
par(mar = c(10, 8.5, 3, 3));
labeledHeatmap(Matrix=moduleTraitCor,
               xLabels=names(datTraits),
               #xLabels="OS",
               yLabels=rownames(moduleTraitCor),
               ySymbols=rownames(moduleTraitCor),
               colorLabels=FALSE,
               colors=greenWhiteRed(50),
               textMatrix=textMatrix,
               setStdMargins=FALSE,
               cex.text=0.5,
               zlim=c(-1,1),
               main=paste("Module-trait relationships"))
dev.off()


Selected.module <- substring(names(MEs)[which(abs(moduleTraitCor)==max(abs(moduleTraitCor)))], first=3)
module_color <- labels2colors(as.integer(Selected.module)+1)
sprintf("Selected module: %s", Selected.module)

probes <- colnames(datExpr)
length(probes)
inModule <- is.finite(match(cluster_labels$labels, Selected.module))
modProbes <- probes[inModule]

tom_sim <- read.delim(TOMdir, sep=",")
rownames(tom_sim) <- colnames(tom_sim)

modTOM <- tom_sim[inModule, inModule]

dimnames(modTOM) <- list(modProbes, modProbes)

dir_edge=paste(outdir, paste0(prefix, "edge_color.txt"), sep="/")
dir_node=paste(outdir, paste0(prefix, "node_color.txt"), sep="/")
cyt <- exportNetworkToCytoscape(modTOM,
                                dir_edge,
                                dir_node,
                                weighted=T, threshold = 0.06,
                                nodeNames=modProbes,
                                nodeAttr=colnames(modTOM))



color <- labels2colors(cluster_labels$labels+1)#labels2colors(c(0,1,2,3,4,5,6)) "grey" "turquoise" "blue"      "brown"     "yellow"    "green"     "red"
colorlevels=unique(color)
print(colorlevels)

datME <- moduleEigengenes(datExpr,color)$eigengenes
signif(cor(datME, use="p"), 2)

dissimME=(1-cor(datME, method="p"))/2
hclustdatME=hclust(as.dist(dissimME), method="average" )
# Plot the eigengene dendrogram
# dir1=paste(outdir, paste0(prefix, ".eigengene dendrogram.png"), sep="/")
# png(file=dir1,width = 3000,height = 2800,res = 300)
pdf(file=paste(outdir,paste0(prefix, ".eigengene dendrogram.pdf"),sep="/"), width=9, height=12)
par(mfrow=c(1,1))
plot(hclustdatME, main="Clustering tree based of the module eigengenes")
dev.off()

# dir1=paste(outdir, paste0(prefix, "Relationship between module eigengenes.png"),sep="/")
# png(file=dir1,width = 3000,height = 2800,res = 300)
pdf(file = paste(outdir, paste0(prefix, "Relationship between module eigengenes.pdf"), sep="/"), width=9, height=12)
#sizeGrWindow(8,9)
plotMEpairs(datME, y=datTraits$sample_type)
dev.off()

# if (length(unique(color))>6){
#   colorlevels=unique(color)[1:6]
# }else{colorlevels=unique(color)}

dir1=paste(outdir, paste0(prefix, "heatmap plots of module expression.png"), sep="/")
png(file=dir1, width = 3000, height = 700*length(colorlevels), res = 300)
# pdf(paste(outdir, paste0(prefix, "heatmap plots of module expression.pdf"),sep="/"), width=9, height=12)
#sizeGrWindow(18,16)
par(mfrow=c(6, 1), mar=c(1, 2, 4, 1))
for (which.module in colorlevels) {
  plotMat(t(scale(datExpr[1:100,color==which.module])), nrgcols=30, rlabels=T,
          clabels=T, rcols=which.module,
          title=which.module )
}
dev.off()

which.module <- module_color
ME=datME[, paste("ME", which.module, sep="")]
# dir1=paste(outdir, paste0(prefix, "displaying module heatmap and the eigengene.png"),sep="/")
# png(file=dir1, width = 3000, height = 2800, res = 300)
pdf(file=paste(outdir, paste0(prefix, "displaying module heatmap and the eigengene.pdf"), sep="/"), width=9, height=12)
#sizeGrWindow(8,7);
par(mfrow=c(2,1), mar=c(0.3, 5.5, 3, 2))
plotMat(t(scale(datExpr[1:100, color==which.module]) ),
        nrgcols=30, rlabels=F, rcols=which.module,
        main=which.module, cex.main=2)
par(mar=c(5, 4.2, 0, 0.7))
barplot(ME[1:100], col=which.module, main="", cex.main=2,
        ylab="eigengene expression", xlab="array sample")
dev.off()


# signif(cor(datTraits,datME, use="p"), 2)
#
# p.values = corPvalueStudent(cor(datTraits,datME, use="p"), nSamples = length(datTraits$sample_type))


GS1 = as.numeric(cor(datTraits, datExpr, use="p"))
GeneSignificance = abs(GS1)
# Next module significance is defined as average gene significance.
ModuleSignificance = tapply(GeneSignificance, color, mean, na.rm=T)
ModuleSignificance = as.data.frame(ModuleSignificance)
write.csv(ModuleSignificance, file = paste(outdir, paste0(prefix, ".Gene significance across modules.csv"), sep="/"), row.names = TRUE, quote = FALSE)


# dir1=paste(outdir, paste0(prefix, "Gene significance across modules.png"),sep="/")
# png(file=dir1,width = 3000,height = 2800,res = 300)
pdf(file=paste(outdir, paste0(prefix, "Gene significance across modules.pdf"), sep="/"), width=9, height=6)
#sizeGrWindow(8,7)
par(mfrow = c(1,1))
plotModuleSignificance(GeneSignificance, colors=color)
dev.off()


Alldegrees <- intramodularConnectivity(ADJ, color)
head(Alldegrees)


# if (length(unique(color))>12){
#   colorlevels=unique(color)[1:12]
# }else{colorlevels=unique(color)}

#sizeGrWindow(9,6)
# dir1=paste(outdir, paste0(prefix, "Gene Significance vs Connectivity.png"),sep="/")
# png(file=dir1,width = 2400,height = 3000,res = 300)
pdf(paste(outdir, paste0(prefix, "Gene Significance vs Connectivity.pdf"), sep="/"), width=9, height=12)
par(mfrow=c(as.integer(0.5+length(colorlevels)/3), 3), mar = c(4,5,3,1)) #
for (i in c(1:length(colorlevels)))
{
  whichmodule=colorlevels[[i]];
  restrict = (color==whichmodule);
  verboseScatterplot(Alldegrees$kWithin[restrict],
                     GeneSignificance[restrict], col=color[restrict],
                     main=whichmodule,
                     xlab = "Connectivity", ylab = "Gene Significance", abline = TRUE)
}
dev.off()

datKME <- signedKME(datExpr, datME, outputColumnName="MM.")
dir_datKME <- paste(outdir, paste0(prefix, ".datKME.csv"), sep="/")
write.csv(datKME, dir_datKME, row.names = TRUE, quote = FALSE)
# Display the first few rows of the data frame
head(datKME)


# if (length(unique(color))>12){
#   colorlevels=unique(color)[1:12]
# }else{colorlevels=unique(color)}

# dir1=paste(outdir, paste0(prefix, "Module Membership vs Intramodular Connectivity.png"),sep="/")
# png(file=dir1, width = 2400, height = 3000, res = 300)
pdf(paste(outdir, paste0(prefix, "Module Membership vs Intramodular Connectivity.pdf"), sep="/"), width=9, height=12)
par(mfrow=c(as.integer(0.5+length(colorlevels)/3), 3), mar = c(4,5,3,1)) #
for (i in c(1:length(colorlevels)))
{
  whichmodule=colorlevels[[i]];
  restrict = (color==whichmodule);
  verboseScatterplot(Alldegrees$kWithin[restrict],
                     (datKME[restrict, paste("MM.", whichmodule, sep="")])^6,
                     col=whichmodule,
                     xlab="Intramodular Connectivity",
                     ylab="(Module Membership)^6", abline = TRUE)
}
dev.off()


FilterGenes <- abs(GS1)> .2 & abs(datKME[, paste("MM.", module_color, sep="")])>.8
table(FilterGenes)
FilterGenes <- data.frame(dimnames(data.frame(datExpr))[[2]][FilterGenes])
names(FilterGenes) <- module_color
print(length(FilterGenes[, 1]))
dir_FilterGenes <- paste(outdir, paste0(prefix, ".FilterGenes.txt"), sep="/")
write.table(FilterGenes, dir_FilterGenes, sep="/", row.names = FALSE, quote = FALSE)



