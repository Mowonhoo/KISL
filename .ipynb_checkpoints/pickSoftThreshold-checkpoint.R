
#if (!requireNamespace("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")
#BiocManager::install("doSNOW")

rm(list = ls())

.networkTypes = c("unsigned", "signed", "signed hybrid") #from internalConstants.R
.threadAllowVar = "ALLOW_WGCNA_THREADS" #from internalConstants.R

# Function to calculate an appropriate blocksize
blockSize = function(matrixSize, rectangularBlocks = TRUE, maxMemoryAllocation = NULL, overheadFactor = 3) # from useNThreads.R
{
  if (is.null(maxMemoryAllocation))
  {
    maxAlloc = .checkAvailableMemory();
  } else {
    maxAlloc = maxMemoryAllocation/8;
  }
  maxAlloc = maxAlloc/overheadFactor;
  
  if (rectangularBlocks)
  {
    blockSz = floor(maxAlloc/matrixSize);
  } else
    blockSz = floor(sqrt(maxAlloc));
  
  return( min (matrixSize, blockSz) )
}


#========================================================================================================
#
# allocateJobs # from useNThreads.R
#
#========================================================================================================

# Facilitates multi-threading by producing an even allocation of jobs 
# Works even when number of jobs is less than number of threads in which case some components of the
# returned allocation will have length 0.

allocateJobs = function(nTasks, nWorkers)
{
  if (is.na(nWorkers))
  {
    warning("In function allocateJobs: 'nWorkers' is NA. Will use 1 worker.");
    nWorkers = 1;
  }
  n1 = floor(nTasks/nWorkers);
  n2 = nTasks - nWorkers*n1;
  allocation = list();
  start = 1;
  for (t in 1:nWorkers)
  {
    end = start + n1 - 1 + as.numeric(t<=n2);
    if (start > end)
    {
      allocation[[t]] = numeric(0);
    } else allocation[[t]] = c(start:end);
    start = end+1;
  }
  
  allocation;
}


WGCNAnThreads = function() # from useNThreads.R
{
  n = suppressWarnings(as.numeric(as.character(Sys.getenv(.threadAllowVar, unset = 1))));
  if (is.na(n)) n = 1;
  if (length(n)==0) n = 1;
  n;
}

#printFlush: Print Arguments and Flush the Console
#Description：Passes all its arguments unchaged to the standard print function; after the execution of print it flushes the console, if possible.
printFlush = function(...)  #from dynamicTreeCut (version 1.63-1)
{
  #  x = print(...)
  cat(...); cat("\n");
  if (exists("flush.console")) flush.console();
}

indentSpaces = function(indent = 0)
{
  if (indent>0) 
  {
    spaces = paste(rep("  ", times=indent), collapse="");
  } else
  {
    spaces = "";
  }
  spaces;
}

#================================================================================
#
# spaste  #from Functions.R
#
#================================================================================

spaste = function(...) { paste(..., sep = "") }  #from Functions.R

#========================================================================================================

scaleFreeFitIndex=function(k,nBreaks=10, removeFirst = FALSE)  #from Functions.R
{
  discretized.k = cut(k, nBreaks)
  dk = tapply(k, discretized.k, mean)
  p.dk = as.vector(tapply(k, discretized.k, length)/length(k))
  breaks1 = seq(from = min(k), to = max(k), 
                length = nBreaks + 1)
  hist1 = hist(k, breaks = breaks1, plot = FALSE, right = TRUE)
  dk2 = hist1$mids
  dk = ifelse(is.na(dk), dk2, dk)
  dk = ifelse(dk == 0, dk2, dk)
  p.dk = ifelse(is.na(p.dk), 0, p.dk)
  log.dk = as.vector(log10(dk))
  if (removeFirst) {
    p.dk = p.dk[-1]
    log.dk = log.dk[-1]
  }
  log.p.dk= as.numeric(log10(p.dk + 1e-09))
  lm1 = try(lm(log.p.dk ~ log.dk));
  if (inherits(lm1, "try-error")) browser();
  lm2 = lm(log.p.dk ~ log.dk + I(10^log.dk))
  datout=data.frame(Rsquared.SFT=summary(lm1)$r.squared,
                    slope.SFT=summary(lm1)$coefficients[2, 1], 
                    truncatedExponentialAdjRsquared= summary(lm2)$adj.r.squared)
  datout
} # end of function scaleFreeFitIndex


checkSimilarity = function(similarity, min=-1, max=1) # from Functions-fromSimilarity.R
{
  checkAdjMat(similarity, min, max);
}

checkAdjMat = function(adjMat, min = 0, max = 1) # from Functions.R
{
  dim = dim(adjMat)
  if (is.null(dim) || length(dim)!=2 )
    stop("adjacency is not two-dimensional");
  if (!is.numeric(adjMat))
    stop("adjacency is not numeric");
  if (dim[1]!=dim[2])
    stop("adjacency is not square");
  if (max(abs(adjMat - t(adjMat)), na.rm = TRUE) > 1e-12)
    stop("adjacency is not symmetric");
  if (min(adjMat, na.rm = TRUE) < min || max(adjMat, na.rm = TRUE) > max)
    stop("some entries are not between", min, "and", max)
}


#========================================================================================================

scaleFreeFitIndex=function(k,nBreaks=10, removeFirst = FALSE)
{
  discretized.k = cut(k, nBreaks)
  dk = tapply(k, discretized.k, mean)
  p.dk = as.vector(tapply(k, discretized.k, length)/length(k))
  breaks1 = seq(from = min(k), to = max(k), 
                length = nBreaks + 1)
  hist1 = hist(k, breaks = breaks1, plot = FALSE, right = TRUE)
  dk2 = hist1$mids
  dk = ifelse(is.na(dk), dk2, dk)
  dk = ifelse(dk == 0, dk2, dk)
  p.dk = ifelse(is.na(p.dk), 0, p.dk)
  log.dk = as.vector(log10(dk))
  if (removeFirst) {
    p.dk = p.dk[-1]
    log.dk = log.dk[-1]
  }
  log.p.dk= as.numeric(log10(p.dk + 1e-09))
  lm1 = try(lm(log.p.dk ~ log.dk));
  if (inherits(lm1, "try-error")) browser();
  lm2 = lm(log.p.dk ~ log.dk + I(10^log.dk))
  datout=data.frame(Rsquared.SFT=summary(lm1)$r.squared,
                    slope.SFT=summary(lm1)$coefficients[2, 1], 
                    truncatedExponentialAdjRsquared= summary(lm2)$adj.r.squared)
  datout
} # end of function scaleFreeFitIndex




#==============================================================================================
#
# pickSoftThreshold  #from Functions.R
#
#===============================================================================================
# The function pickSoftThreshold allows one to estimate the power parameter when using
# a soft thresholding approach with the use of the power function AF(s)=s^Power
# The removeFirst option removes the first point (k=1, P(k=1)) from the regression fit.
# PL: a rewrite that splits the data into a few blocks.
# SH: more netowkr concepts added.
# PL: re-written for parallel processing
# Alexey Sergushichev: speed up by pre-calculating correlation powers
library("doSNOW") # foreach()

pickSoftThreshold = function (
  data,
  dataIsExpr = FALSE,
  weights = NULL,
  RsquaredCut = 0.85,
  powerVector = c(seq(1, 10, by = 1), seq(12, 20, by = 2)),
  prefix="01",
  outdir="./",
  removeFirst = FALSE, nBreaks = 10, blockSize = NULL,
  corFnc = cor, corOptions = list(use = 'p'),
  networkType = "unsigned",
  moreNetworkConcepts = FALSE,
  gcInterval = NULL,
  verbose = 0, indent = 0, savefile=TRUE)
{
  powerVector = sort(powerVector)
  intType = charmatch(networkType, .networkTypes)
  if (is.na(intType)) 
    stop(paste("Unrecognized 'networkType'. Recognized values are", 
               paste(.networkTypes, collapse = ", ")))
  nGenes = ncol(data);
  if (nGenes<3) 
  { 
    stop("The input data data contain fewer than 3 rows (nodes).", 
         "\nThis would result in a trivial correlation network." )
  }
  if (!dataIsExpr) 
  {
    checkSimilarity(data);
    if (any(diag(data)!=1)) diag(data) = 1;
  }
  
  if (is.null(blockSize))
  {
    blockSize = blockSize(nGenes, rectangularBlocks = TRUE, maxMemoryAllocation = 2^30);
    if (verbose > 0) 
      printFlush(spaste("pickSoftThreshold: will use block size ", blockSize, "."))
  }
  if (length(gcInterval)==0) gcInterval = 4*blockSize;
  
  colname1 = c("Power", "SFT.R.sq", "slope", "truncated R.sq", 
               "mean(k)", "median(k)", "max(k)")
  if(moreNetworkConcepts) 
  {
    colname1=c(colname1,"Density", "Centralization", "Heterogeneity")
  }
  datout = data.frame(matrix(666, nrow = length(powerVector), ncol = length(colname1)))
  names(datout) = colname1
  datout[, 1] = powerVector
  spaces = indentSpaces(indent)
  if (verbose > 0) {
    cat(paste(spaces, "pickSoftThreshold: calculating connectivity for given powers..."))
    if (verbose == 1) pind = initProgInd()
    else cat("\n")
  }
  
  # if we're using one of WGNCA's own correlation functions, set the number of threads to 1.
  corFnc = match.fun(corFnc); #The match.fun() function looks for a function whose name matches "corFnc"
  corFormals = formals(corFnc); #If we want to manipulate a function's argument list in our R code, the formals function is a good tool. It returns a pair list object (corresponding to the parameter name and the default value of the parameter).
  if ("nThreads" %in% names(corFormals)) corOptions$nThreads = 1;
  
  # Resulting connectivities
  datk = matrix(0, nrow = nGenes, ncol = length(powerVector))
  
  # Number of threads. In this case I need this explicitly.
  nThreads = WGCNAnThreads();
  
  nPowers = length(powerVector);
  
  # Main loop
  startG = 1
  lastGC = 0;
  corOptions$x = data;
  if (!is.null(weights))
  {
    if (!dataIsExpr) 
      stop("Weights can only be used when 'data' represents expression data ('dataIsExpr' must be TRUE).");
    if (!isTRUE(all.equal(dim(data), dim(weights))))
      stop("When 'weights' are given, dimensions of 'data' and 'weights' must be the same.");
    corOptions$weights.x = weights;
  }
  while (startG <= nGenes) 
  {
    endG = min (startG + blockSize - 1, nGenes)
    
    if (verbose > 1) 
      printFlush(paste(spaces, "  ..working on genes", startG, "through", endG, "of", nGenes))
    
    nBlockGenes = endG - startG + 1;
    jobs = allocateJobs(nBlockGenes, nThreads);
    # This assumes that the non-zero length allocations
    # precede the zero-length ones
    actualThreads = which(sapply(jobs, length) > 0); 
    
    datk[ c(startG:endG), ] = foreach(t = actualThreads, .combine = rbind) %dopar% 
      {
        useGenes = c(startG:endG)[ jobs[[t]] ]
        nGenes1 = length(useGenes);
        if (dataIsExpr)
        {
          corOptions$y = data[ , useGenes];
          if (!is.null(weights))
            corOptions$weights.y = weights[ , useGenes];
          corx = do.call(corFnc, corOptions);
          if (intType == 1) {
            corx = abs(corx)
          } else if (intType == 2) {
            corx = (1 + corx)/2
          } else if (intType == 3) {
            corx[corx < 0] = 0
          }
          if (sum(is.na(corx)) != 0) 
            warning(paste("Some correlations are NA in block", 
                          startG, ":", endG, "."));
        } else {
          corx = data[, useGenes];
          if (intType == 1) {
            corx = abs(corx)
          } else if (intType == 2) {
            corx = (1 + corx)/2
          } else if (intType == 3) {
            corx[corx < 0] = 0
          }
        }
        # Set the diagonal elements of corx to exactly 1. Possible small numeric errors can in extreme cases lead to
        # negative connectivities.
        ind = cbind(useGenes, 1:length(useGenes));
        corx[ind] = 1;
        datk.local = matrix(NA, nGenes1, nPowers);
        corxPrev = matrix(1, nrow=nrow(corx), ncol=ncol(corx))
        powerVector1 <- c(0, head(powerVector, -1))
        powerSteps <- powerVector - powerVector1
        uniquePowerSteps <- unique(powerSteps)
        corxPowers <- lapply(uniquePowerSteps, function(p) corx^p)
        names(corxPowers) <- uniquePowerSteps
        for (j in 1:nPowers) {
          corxCur <- corxPrev * corxPowers[[as.character(powerSteps[j])]]
          datk.local[, j] = colSums(corxCur, na.rm = TRUE) - 1
          corxPrev <- corxCur
        };
        datk.local
      } # End of %dopar% evaluation
    # Move to the next block of genes.
    startG = endG + 1
    if ((gcInterval > 0) && (startG - lastGC > gcInterval)) { gc(); lastGC = startG; }
    if (verbose == 1) pind = updateProgInd(endG/nGenes, pind)
  }
  if (verbose == 1) printFlush("");
  #View(datk)
  for (i in c(1:length(powerVector))) 
  {
    khelp= datk[, i] 
    if (any(khelp < 0)) browser();
    SFT1=scaleFreeFitIndex(k=khelp,nBreaks=nBreaks,removeFirst=removeFirst)
    datout[i, 2] = SFT1$Rsquared.SFT  
    datout[i, 3] = SFT1$slope.SFT 
    datout[i, 4] = SFT1$truncatedExponentialAdjRsquared
    datout[i, 5] = mean(khelp,na.rm = TRUE)
    datout[i, 6] = median(khelp,na.rm = TRUE)
    datout[i, 7] = max(khelp,na.rm = TRUE)
    if(moreNetworkConcepts) 
    { 
      Density = sum(khelp)/(nGenes * (nGenes - 1))
      datout[i, 8] =Density
      Centralization = nGenes*(max(khelp)-mean(khelp))/((nGenes-1)*(nGenes-2))
      datout[i, 9] = Centralization
      Heterogeneity = sqrt(nGenes * sum(khelp^2)/sum(khelp)^2 - 1)
      datout[i, 10] = Heterogeneity
    }
  }
  print(signif(data.frame(datout),3))
  ind1 = datout[, 2] > RsquaredCut
  indcut = 6 #NA
  indcut = if (sum(ind1) > 0) min(c(1:length(ind1))[ind1]) else indcut;
  powerEstimate = powerVector[indcut][[1]]
  gc();
  
  sft <- list(powerEstimate = powerEstimate, fitIndices = data.frame(datout))

  if (savefile){
      dir1 <- paste(outdir, paste0(prefix, ".sft_fitIndices.csv"), sep="/")
      write.table(sft$fitIndices, dir1, sep = ',', row.names =FALSE, col.names =TRUE, quote = FALSE)#, col.names = NA


      dir1 <- paste(outdir, paste0(prefix, ".Scale_independence.png"), sep="/")
      png(file=dir1,width = 2000,height = 1800,res = 300)
      pdf(file=paste(outdir, paste0(prefix, ".Scale_independence.pdf"), sep="/"))
      plot(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2], type = 'n',
           xlab = 'Soft Threshold (power)', ylab = 'Scale Free Topology Model Fit,signed R^2',
           main = paste('Scale independence'))
      text(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],
           labels = powerVector, col = 'red');
      hline <- if (sum(ind1) > 0) RsquaredCut else datout[datout[, 1]==indcut, 2]
      abline(h = hline, col = 'red')
      dev.off()

      pdf(file=paste(outdir, paste0(prefix, ".Scale_independence.pdf"), sep="/"))
      plot(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2], type = 'n',
           xlab = 'Soft Threshold (power)', ylab = 'Scale Free Topology Model Fit,signed R^2',
           main = paste('Scale independence'))
      text(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],
           labels = powerVector, col = 'red');
      hline <- if (sum(ind1) > 0) RsquaredCut else datout[datout[, 1]==indcut, 2]
      abline(h = hline, col = 'red')
      dev.off()

      dir1 <- paste(outdir, paste0(prefix, ".Mean_connectivity.png"), sep="/")
      png(file=dir1,width = 2000,height = 1800,res = 300)
      plot(sft$fitIndices[,1], sft$fitIndices[,5],
                    xlab = 'Soft Threshold (power)', ylab = 'Mean Connectivity', type = 'n',
                    main = paste('Mean connectivity'))
      text(sft$fitIndices[,1], sft$fitIndices[,5], labels = powerVector, col = 'red')
      dev.off()

      pdf(file=paste(outdir, paste0(prefix, ".Mean_connectivity.pdf"), sep="/"))
      plot(sft$fitIndices[,1], sft$fitIndices[,5],
                    xlab = 'Soft Threshold (power)', ylab = 'Mean Connectivity', type = 'n',
                    main = paste('Mean connectivity'))
      text(sft$fitIndices[,1], sft$fitIndices[,5], labels = powerVector, col = 'red')
      dev.off()
    }
  
  return(sft)
}


