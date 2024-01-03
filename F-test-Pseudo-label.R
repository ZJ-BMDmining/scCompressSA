library(FEAST)

myData = read.csv("./data/PBMC-A/data.csv",header=T,row.names=1,sep=",",check.names=F)
datExpr <- as.matrix(t(myData))
Y=datExpr
Y = process_Y(Y)
#Consensus clustering
con_res = Consensus(Y, k=8)

write.csv(con_res, file = "con_res_PBMC-A.csv")
