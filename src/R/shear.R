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

# Import dataset
file = "/home/lk/Public/Feature_Selection/data/STS/transport/boltzmann/shear_viscosity.csv"
data <- read.csv(file, header = TRUE)

# Structure of the dataframe
str(data)

# See top 6 rows and 10 columns
head(data[, 1:10])

# Exploratory data analysis
plot_intro(data)
plot_bar(data)
plot_correlation(data)

# Define the control using a random forest selection function
control <- rfeControl(functions = rfFuncs, # random forest
                      method = "repeatedcv", # or just cv
                      repeats = 5, # number of repeats
                      number = 10) # the number of folds

# Create the training and test datasets
set.seed(23)

#x <- data %>%
#  select(-target, -catvar, -contvar1, -contvar2, -contvar3) %>%
#  as.data.frame()

#y <- data$Viscosity

# Step 1: Get row numbers for the training data
trainRowNumbers <- createDataPartition(data$Viscosity, p=0.8, list=FALSE)
#inTrain <- createDataPartition(y, p = .80, list = FALSE)[,1]

# Step 2: Create the training dataset
trainData <- data[trainRowNumbers,]
#x_train <- x[ inTrain, ]
#x_test  <- x[-inTrain, ]

# Step 3: Create the test dataset
testData <- data[-trainRowNumbers,]
#y_train <- y[ inTrain]
#y_test  <- y[-inTrain]

# Store X and Y for later use.
x = trainData[, 1:51]
y = trainData$Viscosity

#skimmed <- skim_to_wide(trainData)
skimmed <- skim(trainData)
skimmed[, c(1:12)]

### How to preprocess to transform the data?
# - range: Normalize values so it ranges between 0 and 1
# - center: Subtract Mean
# - scale: Divide by standard deviation
# - BoxCox: Remove skewness leading to normality. Values must be > 0
# - YeoJohnson: Like BoxCox, but works for negative values.
# - expoTrans: Exponential transformation, works for negative values.
# - pca: Replace with principal components
# - ica: Replace with independent components
# - spatialSign: Project the data to a unit circle

preProcess_range_model <- preProcess(trainData, method='range')
trainData <- predict(preProcess_range_model, newdata = trainData)

# Append the Y variable
trainData$Viscosity <- y

apply(trainData[, 1:10], 2, FUN=function(x){c('min'=min(x), 'max'=max(x))})

### How to visualize the importance of variables using featurePlot()?
featurePlot(x = trainData[, 1:51], 
            y = trainData$Viscosity, 
            plot = "box",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))

featurePlot(x = trainData[, 1:51],
            y = trainData$Viscosity,
            plot = "density",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"),
                          y = list(relation="free")))

### How to do feature selection using recursive feature elimination (rfe)?
options(warn=-1)

subsets <- c(1:51)

ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

lmProfile <- rfe(x=trainData[, 1:51], y=trainData$Viscosity,
                 sizes = subsets,
                 rfeControl = ctrl)

lmProfile

### Training and Tuning the model
## How to train() the model and interpret the results?
# See available algorithms in caret
modelnames <- paste(names(getModelInfo()), collapse=',  ')
modelnames

modelLookup('earth')

# Train the model using randomForest and predict on the training data itself.
model_mars = train(Viscosity ~ ., data=trainData, method='earth')
fitted <- predict(model_mars)

model_mars
plot(model_mars, main="Model Accuracies with MARS")

## How to compute variable importance?
varimp_mars <- varImp(model_mars)
plot(varimp_mars, main="Variable Importance with MARS")

## Prepare the test dataset and predict
# Step 1: Impute missing values
#testData2 <- predict(preProcess_missingdata_model, testData)

# Step 2: Create one-hot encodings (dummy variables)
#testData3 <- predict(dummies_model, testData2)

# Step 3: Transform the features to range between 0 and 1
#testData4 <- predict(preProcess_range_model, testData3)
testData4 <- predict(preProcess_range_model, testData)

# View
head(testData4[, 1:51])

# Predict on testData
predicted <- predict(model_mars, testData4)
head(predicted)

# Compute the confusion matrix
confusionMatrix(reference = testData$Viscosity, data = predicted, mode='everything', positive='MM')

### How to do hyperparameter tuning to optimize the model for better performance?
## Setting up the trainControl()
# Define the training control
fitControl <- trainControl(
    method = 'cv',                   # k-fold cross validation
    number = 5,                      # number of folds
    savePredictions = 'final',       # saves predictions for optimal tuning parameter
    classProbs = T,                  # should class probabilities be returned
    summaryFunction=twoClassSummary  # results summary function
)

# Hyper Parameter Tuning using tuneLength
# Step 1: Tune hyper parameters by setting tuneLength
model_mars2 = train(Viscosity ~ ., data=trainData, method='earth', tuneLength = 5, metric='ROC', trControl = fitControl)
model_mars2

# Step 2: Predict on testData and Compute the confusion matrix
predicted2 <- predict(model_mars2, testData4)
confusionMatrix(reference = testData$Viscosity, data = predicted2, mode='everything', positive='MM')

# Hyper Parameter Tuning using tuneGrid
# Step 1: Define the tuneGrid
marsGrid <-  expand.grid(nprune = c(2, 4, 6, 8, 10), degree = c(1, 2, 3))

# Step 2: Tune hyper parameters by setting tuneGrid
model_mars3 = train(Viscosity ~ ., data=trainData, method='earth', metric='ROC', tuneGrid = marsGrid, trControl = fitControl)
model_mars3

# Step 3: Predict on testData and Compute the confusion matrix
predicted3 <- predict(model_mars3, testData4)
confusionMatrix(reference = testData$Viscosity, data = predicted3, mode='everything', positive='MM')

### How to evaluate performance of multiple machine learning algorithms?
## Training Adaboost
# Train the model using adaboost
model_adaboost = train(Viscosity ~ ., data=trainData, method='adaboost', tuneLength=2, trControl = fitControl)
model_adaboost

## Training Random Forest
# Train the model using rf
model_rf = train(Viscosity ~ ., data=trainData, method='rf', tuneLength=5, trControl = fitControl)
model_rf

## Training xgBoost Dart
# Train the model using MARS
model_xgbDART = train(Viscosity ~ ., data=trainData, method='xgbDART', tuneLength=5, trControl = fitControl, verbose=F)
model_xgbDART

## Training SVM
# Train the model using MARS
model_svmRadial = train(Viscosity ~ ., data=trainData, method='svmRadial', tuneLength=15, trControl = fitControl)
model_svmRadial

## Run resamples() to compare the models
# Compare model performances using resample()
models_compare <- resamples(list(ADABOOST=model_adaboost, RF=model_rf, XGBDART=model_xgbDART, MARS=model_mars3, SVM=model_svmRadial))

# Summary of the models performances
summary(models_compare)

# Draw box plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(models_compare, scales=scales)

### Ensembling the predictions
## How to ensemble predictions from multiple models using caretEnsemble?
# Stacking Algorithms - Run multiple algos in one call.
trainControl <- trainControl(method="repeatedcv",
                             number=10,
                             repeats=3,
                             savePredictions=TRUE,
                             classProbs=TRUE)

algorithmList <- c('rf', 'adaboost', 'earth', 'xgbDART', 'svmRadial')

models <- caretList(Viscosity ~ ., data=trainData, trControl=trainControl, methodList=algorithmList)
results <- resamples(models)
summary(results)

# Box plots to compare models
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(results, scales=scales)

## How to combine the predictions of multiple models to form a final prediction?
# Create the trainControl
stackControl <- trainControl(method="repeatedcv", 
                             number=10, 
                             repeats=3,
                             savePredictions=TRUE, 
                             classProbs=TRUE)

# Ensemble the predictions of `models` to form a new combined prediction based on glm
stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
print(stack.glm)

# Predict on testData
stack_predicteds <- predict(stack.glm, newdata=testData4)
head(stack_predicteds)
