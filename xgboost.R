

#xgboost
library(xgboost)
library(Matrix)
library(methods)


datMy <- read.csv("train.csv", header = TRUE)
datMyTest <- read.csv("test.csv", header = TRUE)

#in_train = createDataPartition(datMy$target, p=0.80, list=FALSE)
trIdx = datMy[, -1]
tsIdx = datMyTest[, -1]

dtrIdx = as(as.matrix(trIdx[,names(trIdx) != 'target']), "dgCMatrix")
dtsIdx = as(as.matrix(tsIdx[,names(tsIdx) != 'target']), "dgCMatrix")

outcomes = as.integer(gsub('Class_','',trIdx[,ncol(trIdx)])) - 1
xdtrain <- xgb.DMatrix(data = dtrIdx, label = outcomes)

cvIn <- list("num_class" = 9, "nthread" = 4, "eval_metric" = "mlogloss")
res.cvIn = xgb.cv(cvIn, xdtrain, nrounds=150, nfold = 4, objective='multi:softprob')

# Train the model
#bst <- xgb.train(params = cvIn, data=xdtrain, nround=150, objective = 'multi:softprob')
bst <- xgboost(params = cvIn, data=xdtrain, nround=150, objective = 'multi:softprob')
pred = predict(bst, dtsIdx)
pred = t(matrix(pred,9,length(pred)/9))
pred <- formatC(pred, digits=3)
pred <- format(pred, scientific=FALSE)
pred = data.frame(seq(1,nrow(pred)),pred)

classList <- unlist(lapply(1:9, function(x) paste0("Class_", x)))
names(pred) <- c('id', classList)

write.csv(pred, file = "submit.csv", row.names = FALSE, quote=FALSE)



#Lets get top 30 vars
varImp <- xgb.importance(names(tsIdx), model=bst)
varList2 <- tail(varImp, 25)
eqnXGB <- as.formula(paste("target ~ .+", paste("(", paste(varList2$Feature, sep = '', collapse = '+'), ")^2")))
eqnXGB2 <- as.formula(paste("~ .+", paste("(", paste(varList2$Feature, sep = '', collapse = '+'), ")^2")))

trIdx <- as.data.frame(model.matrix(eqnXGB,trIdx))
tsIdx <- as.data.frame(model.matrix(eqnXGB2,tsIdx))

#in_train = createDataPartition(datMy$target, p=0.80, list=FALSE)
trIdx = trIdx[, -1]
tsIdx = tsIdx[, -1]

dtrIdx = as(as.matrix(trIdx[,names(trIdx) != 'target']), "dgCMatrix")
dtsIdx = as(as.matrix(tsIdx[,names(tsIdx) != 'target']), "dgCMatrix")

xdtrain <- xgb.DMatrix(data = dtrIdx, label = outcomes)

cvIn <- list("num_class" = 9, "nthread" = 4, "eval_metric" = "mlogloss")
res.cvIn = xgb.cv(cvIn, xdtrain, nrounds=200, nfold = 4, objective='multi:softprob')

# Train the model
#bst <- xgb.train(params = cvIn, data=xdtrain, nround=150, objective = 'multi:softprob')
bst <- xgboost(params = cvIn, data=xdtrain, nround=200, objective = 'multi:softprob', verbose=FALSE)

pred = predict(bst, dtsIdx)
pred = t(matrix(pred,9,length(pred)/9))
pred <- formatC(pred, digits=3)
pred <- format(pred, scientific=FALSE)
pred = data.frame(seq(1,nrow(pred)),pred)

classList <- unlist(lapply(1:9, function(x) paste0("Class_", x)))
names(pred) <- c('id', classList)

write.csv(pred, file = "submit.csv", row.names = FALSE, quote=FALSE)


