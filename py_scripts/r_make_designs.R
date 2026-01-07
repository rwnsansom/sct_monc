library(lhs) 

meshgridn = function(L){
  out = list()
  n = sapply(L, length)
  w = 1:length(L)
  for(i in w){
    out[[i]] = rep(rep(L[[i]], rep(prod(n[w<i]), n[i])), prod(n[w>i])) # prod(NULL) == 1, so this works.
  }
  return(out)
}


## Load data
## LH design
args <- commandArgs(TRUE)
thresholds <- args[1]
design=read.csv(sprintf("data/input_data/sc_beginning_inputs%s.csv",thresholds),header=TRUE,stringsAsFactors=FALSE)

## Transform the autoconversion
design[,6]=10^(design[,6])
n=6
m=length(design[,1])

## Get mins and maxs and transform to unit design
Min<-c(min(design[,1],na.rm=TRUE),min(design[,2],na.rm=TRUE),min(design[,3],na.rm=TRUE),min(design[,4],na.rm=TRUE),min(design[,5],na.rm=TRUE),min(design[,6],na.rm=TRUE))
Max<-c(max(design[,1],na.rm=TRUE),max(design[,2],na.rm=TRUE),max(design[,3],na.rm=TRUE),max(design[,4],na.rm=TRUE),max(design[,5],na.rm=TRUE),max(design[,6],na.rm=TRUE))

## Make LH to sample param space
lh_predicted<-maximinLHS(n=1000,k=6,dup=5)
lh_ranges<-lh_predicted
lh_ranges[,1]<-qunif(lh_predicted[,1],min=Min[1],max=Max[1])
lh_ranges[,2]<-qunif(lh_predicted[,2],min=Min[2],max=Max[2])
lh_ranges[,3]<-qunif(lh_predicted[,3],min=Min[3],max=Max[3])
lh_ranges[,4]<-qunif(lh_predicted[,4],min=Min[4],max=Max[4])
lh_ranges[,5]<-qunif(lh_predicted[,5],min=Min[5],max=Max[5])
lh_ranges[,6]<-log10(qunif(lh_predicted[,6],min=Min[6],max=Max[6]))
lh_ranges <- data.frame(lh_ranges)
header <- c("bl_qv", "bl_z", "delta_theta", "delta_qv", "n_a", "baut")
colnames(lh_ranges) <- header
lh_design_file<-sprintf("predictions/lh1000_design%s.csv", thresholds)
write.csv(lh_ranges, lh_design_file, row.names=FALSE, quote=FALSE)

input = list()
for (i in 1:n) {
  input[[i]] <- seq(0, 1, length.out=10)
}

meshgrid = meshgridn(input)

for (i in 1:n) {
  meshgrid[[i]]<-qunif(meshgrid[[i]],min=Min[i],max=Max[i])
}

meshgrid[[6]]<-log10(meshgrid[[6]])
meshgrid <- data.frame(meshgrid)
colnames(meshgrid) <- header
grid_design<-sprintf("predictions/grid10_design%s.csv", thresholds)
write.csv(meshgrid, grid_design, row.names=FALSE, quote=FALSE)