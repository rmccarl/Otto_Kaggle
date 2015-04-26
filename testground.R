###Variable Mining
options(scipen=100)


library(ggplot2)
library(caret)
library(gbm)
library(randomForest)
library(plyr)
library(dplyr)
library(reshape2)
library(corrplot)
library(scales)

library(doParallel) 
cl <- makeCluster(4)
registerDoParallel(cl)

datMy <- read.csv("train.csv", header = TRUE)
datMyTest <- read.csv("test.csv", header = TRUE)

##Check correlation matrix 
datMy.scale <- scale(datMy[,2:ncol(datMy)-1],center=TRUE,scale=TRUE)[,-1]
corMatMy <- cor(datMy.scale)
corrplot(corMatMy, order = "hclust", tl.pos = "n")
highlyCor <- findCorrelation(corMatMy, 0.4)
datMyFiltered.scale <- datMy.scale[,-highlyCor]
corMatMy <- cor(datMyFiltered.scale)
corrplot(corMatMy, order = "hclust", tl.pos = "n")
rm(datMy.scale)
rm(datMyFiltered.scale)

##Output graphs
classnum = 5
classname = paste0("Class_", classnum)
classList <- unlist(lapply(1:9, function(x) paste0("Class_", x)))


in_train = createDataPartition(datMy$target, p=0.80, list=FALSE)
trIdx = datMy[in_train, ]
tsIdx = datMy[-in_train, ]



##

mcLogLoss <- function (data,
                       lev = NULL,
                       model = NULL) {
  
  if (!all(levels(data[, "pred"]) == levels(data[, "obs"]))) 
    stop("levels of observed and predicted data do not match")
  
  LogLoss <- function(actual, predicted, eps=1e-15) {
    predicted[predicted < eps] <- eps
    predicted[predicted > 1 - eps] <- 1 - eps
    -1/nrow(actual)*(sum(actual*log(predicted)))
  }
  
  dtest <- dummyVars(~obs, data=data, levelsOnly=TRUE)
  actualClasses <- predict(dtest, data[,-1])
  
  out <- LogLoss(actualClasses, data[,-c(1:2)])  
  names(out) <- "mcLogLoss"
  out
}

#Test2 gbmGrid <- expand.grid(.shrinkage = 0.1, .n.trees = c(1, 50, 100), .interaction.depth = c(1,3,5))
#Test3 gbmGrid <- expand.grid(.shrinkage = 0.1, .n.trees = c(1, 50, 100), .interaction.depth = c(1,3,5))

gbmGrid <- expand.grid(.shrinkage = 0.1, .n.trees = 100, .interaction.depth = 5)

fitControl <- trainControl(method="repeatedcv",
                           number=5,
                           repeats=1,
                           verboseIter=TRUE,
                           summaryFunction = mcLogLoss,
                           classProbs=TRUE)

set.seed(9)
gbmTest <- train(as.factor(target) ~ ., data=trIdx[,-c(1)], method="gbm",
                 trControl=fitControl,
                 distribution="multinomial",
                 tuneGrid=gbmGrid,
                 metric="mcLogLoss",
                 maximize=FALSE,
                 verbose=FALSE)


pred <- predict(gbmTest4,tsIdx[,-c(1,ncol(tsIdx))],type="raw")
confusionMatrix(tsIdx[['target']],pred)

#Class 1, 3 and 4 hav every low pos pred value

###


stopCluster(cl)


gbmGrid <- expand.grid(.shrinkage = 0.1, .n.trees = 100, .interaction.depth = 5)

fitControl <- trainControl(method="repeatedcv",
                           number=5,
                           repeats=1,
                           verboseIter=TRUE,
                           classProbs=TRUE)

set.seed(5)
gbmTest <- train(as.factor(target) ~ ., data=trIdx[,-c(1)], method="gbm",
                 trControl=fitControl,
                 distribution="multinomial",
                 tuneGrid=gbmGrid,
                 maximize=FALSE,
                 verbose=TRUE)

print(gbmTest)


submit <- data.frame(id = tsIdx$id, pred)

##Random Forest Test
datMy <- mutate(datMy, feat_94 = feat_9 + feat_64)
datMy <- select(datMy, id:feat_93, feat_94, target)

in_train = createDataPartition(datMy$target, p=0.80, list=FALSE)
trIdx = datMy[in_train, ]
tsIdx = datMy[-in_train, ]

rf = randomForest(target ~., data = trIdx[,-1], ntree=200, mtry = 8,do.trace=5, importance = TRUE)
test = data.frame(varImpPlot(rf, scale=FALSE, type=2))
test = mutate(test, featname = row.names(test)) 
test = arrange(test, desc(MeanDecreaseGini))

rf.eqn <- as.formula(paste("target~ ", paste(test$featname[1:40], sep = '', collapse = '+')))
rf = randomForest(rf.eqn, data = trIdx[,-c(1)], ntree=200, mtry = 7,do.trace=5)

predictClasses = predict(rf, tsIdx[,-c(1)], type = "prob")
importance(rf, type=1, scale=FALSE)

pred <- predict(rf,datMyTest,type="prob")
submit <- data.frame(id = datMyTest$id, pred)
write.csv(submit, file = "firstsubmit.csv", row.names = FALSE)

##


##Otto Group Classification Kaggle Competition   DRAFT

**GOAL: The purpose of the draft is to describe the current approach taken with respect to the Otto Group competition. The intent is to allow a possible team member to understand the thought process and replicate the results provided. **
  
  **At http://www.kaggle.com/c/otto-group-product-classification-challenge can be found the PROBLEM STATEMENT below:**
  
  **"For this competition, we have provided a dataset with 93 features for more than 200,000 products. The objective   is to build a predictive model which is able to distinguish between our main product categories. The winning models will be open sourced." (added: There are nine Classes with which to bucket the 93 features).**
  
  We now have a **question** to answer. 


###**Current Results**

Three models have been utilized to benchmark and iteratively improve the Kaggle multi-class logloss score. Gradient boosting (GBM) was used initially due to publicized success in past data science competitions and also to test the model performance under the R ‘caret’ package. The model proved to be a heavy consumer of computing resources and there was some instability with respect to runtime behavior.  Repeated cross-validation of the training data often resulted in runtime issues leading to the related execution being halted. This may have been due to the addition of a custom cost function to extend supported error reporting.  

A random forest model (RF) was tested after the initial GBM results were obtained. The confusion matrix produced by the RF model shows likely clustering issues with some of the classes provided in the data. This is consistent with univariate procedures performed prior to model execution. The multi-class logloss value was close to the GBM model and runtime behavior is stable. The last model tested was an extreme gradient boosting model (XGB) wrapped in an R package ‘xgboost’. This model produced the best results of the three tested and was used as a candidate for further study. 

**Current score is top 25% with more than a month left before the competition concludes.**
  
  However, a point to consider is at what level of difference are the scores statistically significant? There appears to be a lot of clustering between .40 - .45. Unclear if some of this is due to randomness at the moment.  

Next Steps : Feature Selection, Check XGBoost cross validation stratified selection

####Document Structure

First, the basic univariate examination of the data is discussed. A description of the various models used and their results are presented afterwards. Specific comments related to each step are interspersed throughout the text. 


```{r initBlock, echo=FALSE, warning=FALSE, message=FALSE}
###Variable Mining 
options(scipen=100)
library(devtools)
library(ggplot2)
library(Rtsne)
library(caret)
library(gbm)
library(randomForest)
library(plyr)
library(dplyr)
library(reshape2)
library(corrplot)
library(scales)
library(xgboost)
library(Matrix)
library(methods)

library(doParallel) 
library(ape)
library(xtable)




classname = "Class_2"

  eventCnt <- subset(group_by(datMy, target) %>% summarise_each(funs(sum(.))),,-2)
  eventPer <- sweep(eventCnt[,-c(1)], 1, rowSums(eventCnt[,-c(1)]), "/")
  eventPer = cbind(eventCnt[,1], eventPer)
  dataAll = melt(eventPer, id = 1)
  datapts <- dataAll[dataAll$target==classname,]

  minval = min(datapts[datapts$value > quantile(datapts$value,prob=0.95),]$value)
  datapts <- mutate(datapts, top5 = ifelse(value > minval, TRUE, FALSE ))
  datapts$showRow <- ifelse(datapts$top5 == TRUE, rownames(datapts), " ")




  p = ggplot(data=datapts, aes(x = seq(1,nrow(datapts)), y = value, fill=factor(top5)))
  p = p + geom_bar(stat = "identity") +  geom_text(aes(label=datapts$showRow, vjust=0)) 
  p = p + labs(title = paste0(classname, " - % of All Events by Feature"), x = "Feature Number", y = "% of Class Events")
  p = p + scale_x_continuous(breaks=seq(0,93,5)) + scale_y_continuous(labels=percent)
  p = p + scale_fill_discrete(name="Top 5%")
  p 

          
head(data[order(data$V2,decreasing=T),],.05*nrow(data))
