# https://machinelearningmastery.com/feature-selection-with-the-caret-r-package/
# https://www.r-bloggers.com/2010/11/feature-selection-using-the-caret-package/
# https://www.machinelearningplus.com/machine-learning/feature-selection/
# https://towardsdatascience.com/effective-feature-selection-recursive-feature-elimination-using-r-148ff998e4f7
# https://topepo.github.io/caret/recursive-feature-elimination.html
# https://topepo.github.io/caret/feature-selection-overview.html
  
# Activate the R packages
library(dplyr)
library(faux)
library(DataExplorer)
library(randomForest)
library(party)
library(caret)
library(skimr)
library(RANN)
library(fastAdaboost)
library(gbm)
library(xgboost)
library(C50)
library(earth)
library(FSinR)
library(mlbench)
library(ipred)
library(Boruta)

# Import dataset
file = "/home/lk/Templates/Feature_Selection/data/MT/DB6Tr.csv"
data <- read.csv(file, header = TRUE)

### Remove Redundant Features

correlationMatrix <- cor(data[,1:13])

# summarize the correlation matrix
print(correlationMatrix)

# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)

# print indexes of highly correlated attributes
print(highlyCorrelated)

### Rank Features By Importance

control <- trainControl(method="repeatedcv", number=3, repeats=3)

# train the model
model <- train(Viscosity~., data=data, method="lvq", preProcess="scale", trControl=control)

# estimate variable importance
importance <- varImp(model, scale=FALSE)

# summarize importance
print(importance)

# plot importance
plot(importance)

### Feature Selection 

set.seed(666)

# # define the control using a random forest selection function
# control <- rfeControl(functions=rfFuncs, method="cv", number=3)
# 
# # run the RFE algorithm
# results <- rfe(data[,1:12], data[,13], sizes=c(1:12), rfeControl=control)
# 
# # summarize the results
# print(results)
# # list the chosen features
# predictors(results)
# 
# # plot the results
# plot(results, type=c("g", "o"))


# Perform Boruta search
boruta_output <- Boruta(Viscosity ~ ., data=na.omit(data), doTrace=0)  
names(boruta_output)
