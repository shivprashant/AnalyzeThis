WKDir="C:/Users/shivsood/Documents/GitHub/AnalyzeThis/LifeInsuranceAssessment"
setwd(WKDir)

df<-read.csv("data/train.csv",header=TRUE)
validationSet<-read.csv("data/test.csv",header=TRUE)

#Sampler Test. Comment this line to get the realstuff.
df<-df[1:1000,]

dataShuffler<-function(df,predictionVar,p,list) {
  library(caret)
  #Generated a random seed between 0 and 29122015
  #mySeed=as.character(as.integer(runif(1,0,29122015)))
  mySeed="12292015"
  set.seed(mySeed)
  index=createDataPartition(df$Response,p=0.4,list=FALSE)
  return(index)    
}

removeSparsePredictors<-function(df,sparseTolerance){
  #Remove columns that are complete NAs.
  dfSparseness = sapply(df,function(z) round(100*sum(is.na(z))/nrow(df),2))
  
  dfSparseness[dfSparseness!=sparseTolerance]
  colsToDrop=dfSparseness[dfSparseness!=sparseTolerance]
  
  return(colsToDrop)
}

removeZeroVariancePredictors<-function(df){
  varOfCols=sapply(df,function(z) round(var(z),2))
  varOfCols[varOfCols==0]
  varOfCols[is.na(varOfCols)]
  
  #Remove columns that have a variance of 0
  colsToDrop=varOfCols[varOfCols==0]
    
  return(colsToDrop)
}

createModels<-function(trainSetNoFactors,predictionVar){
  library(caret)
  trainingSetNoFactors$Response=as.factor(trainingSetNoFactors$Response)
  preProc=preProcess(trainingSetNoFactors[,-c(ncol(trainingSetNoFactors))],method="pca", thresh=0.8)
  trainPC=predict(preProc,trainingSetNoFactors[,-c(ncol(trainingSetNoFactors))])
  
  #Create a Randomforest based model
  if(file.exists("models/LICRFPCA.rds")) {
    modelFitRPartsPCA<-readRDS("models/LICRFPCA.rds")
  } else{
    modelFitRPartsPCA<-train(trainingSetNoFactors$Response~.,method="rf",data=trainPC)
    saveRDS(modelFitRPartsPCA,"models/LICRFPCA.rds")
  }
  
  #Create a RParts based model
  if(file.exists("models/LICRPartsPCA.rds")) {
    modelFitRPartsPCA<-readRDS("models/LICRPartsPCA.rds")
  } else{
    modelFitRPartsPCA<-train(trainingSetNoFactors$Response~.,method="rpart",data=trainPC)
    saveRDS(modelFitRPartsPCA,"models/LICRPartsPCA.rds")
  }
  return(preProc)
}


compareModels<-function(testSetNoFactors,predictionVar,preProc){
  
  if(file.exists("models/LICRPartsPCA.rds")) {
    modelFitRPartsPCA<-readRDS("models/LICRPartsPCA.rds")
    
    testSetNoFactors$Response=as.factor(testSetNoFactors$Response)
    testPC=predict(preProc,testSetNoFactors[,-c(ncol(testSetNoFactors))]) #Apply the same preprocessing to testset.
    cf<-confusionMatrix(testSetNoFactors$Response, predict(modelFitRPartsPCA, testPC))
    print(cf)      
  } 
  
  if(file.exists("models/LICRFPCA.rds")) {
    modelFitRFPCA<-readRDS("models/LICRFPCA.rds")
    
    testSetNoFactors$Response=as.factor(testSetNoFactors$Response)
    testPC=predict(preProc,testSetNoFactors[,-c(ncol(testSetNoFactors))]) #Apply the same preprocessing to testset.
    cf<-confusionMatrix(testSetNoFactors$Response, predict(modelFitRFPCA, testPC))
    print(cf)      
  }
  
}

## Model Creation. Create multiple models. Avoid overfitting by resampling training data for each model.

for(count in 1:1) {
  #Create Training and Test sets.
  trainIndex = dataShuffler(df,df$Response,p=0.4,list=FALSE)
  trainingSet=df[trainIndex,]
  testSet=df[-trainIndex,]
  
  nrow(trainingSet)
  nrow(testSet)
  
  #Remove factor variables from the Training and Test sets
  dummies=dummyVars(Response~.,data=trainingSet, levelsOnly=FALSE)
  trainingSetNoFactors=as.data.frame(predict(dummies,newdata=trainingSet))
  trainingSetNoFactors$Response<-trainingSet$Response
  
  #Remove Factors in TestDataSet
  dummies=dummyVars(Response~.,data=testSet, levelsOnly=FALSE)
  testSetNoFactors=as.data.frame(predict(dummies,newdata=testSet))
  testSetNoFactors$Response<-testSet$Response
  
  #Data Cleaning.
  colsToDropSparse=removeSparsePredictors(trainingSetNoFactors,0)
  trainingSetNoFactors=trainingSetNoFactors[,-which(names(trainingSetNoFactors) %in% names(colsToDropSparse))]
  testSetNoFactors=testSetNoFactors[,-which(names(testSetNoFactors) %in% names(colsToDropSparse))]
  colsToDropVar=removeZeroVariancePredictors(trainingSetNoFactors)
  trainingSetNoFactors=trainingSetNoFactors[,-which(names(trainingSetNoFactors) %in% names(colsToDropVar))]
  testSetNoFactors=testSetNoFactors[,-which(names(testSetNoFactors) %in% names(colsToDropVar))]
  
  
  
  preProc=createModels(trainingSetNoFactors,trainingSetNoFactors$Response)
  compareModels(testSetNoFactors,testSetNoFactors$Response,preProc)
  
  #print(modelFitRPartsPCA$finalModel)
  
  
}