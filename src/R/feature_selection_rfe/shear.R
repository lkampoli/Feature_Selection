# Activate the R packages
library("dplyr")
library("faux")
library("DataExplorer")
library("caret")
library("randomForest")

#file = "/home/lk/Public/Feature_Selection/data/STS/transport/boltzmann/shear_viscosity.csv"
file = "/home/lk/Public/Feature_Selection/data/STS/kinetic/dataset_N2N_only_RDm1_raw.csv"
data <- read.csv(file, header = TRUE)

head(data)

str(data)

# Add pseudo variables into the data
set.seed(2021)

#data <- mutate(data,
#               # random categorical variable
#               catvar = as.factor(sample(sample(letters[1:3], nrow(data), replace = TRUE))),
#               
#               # random continuous variable (mean = 10, sd = 2, r = 0)
#               contvar1 = rnorm(nrow(data), mean = 10, sd = 2),
#               
#               # continuous variable with low correlation (mean = 10, sd = 2, r = 0.2)
#               contvar2 = rnorm_pre(data$target, mu = 10, sd = 2, r = 0.2, empirical = TRUE),
#               
#               # continuous variable with high correlation (mean = 10, sd = 2, r = 0.8)
#               contvar3 = rnorm_pre(data$target, mu = 10, sd = 2, r = 0.5, empirical = TRUE))
#
## Save the updated data
#write.csv(data, "heart_final.csv", row.names = FALSE, quote = FALSE)
#
## Recode variables
#data <- data %>%
#  mutate_at(c("sex", "cp", "fbs", "restecg", "exang", "slope", "thal", "target", "catvar"), as.factor) %>%
#  mutate_if(is.numeric, scale)

# Exploratory data analysis
plot_intro(data)
#plot_bar(data)
plot_correlation(data)

#-------------------------------------------------------------------------------#

# Define the control using a random forest selection function
control <- rfeControl(functions = rfFuncs, # random forest
                      method = "repeatedcv", # or just cv
                      repeats = 5, # number of repeats
                      number = 10) # the number of folds

x <- data %>%
  select(-RDm1) %>%
  as.data.frame()

y <- data$RDm1

inTrain <- createDataPartition(y, p = .80, list = FALSE)[,1]

x_train <- x[ inTrain, ]
x_test  <- x[-inTrain, ]

y_train <- y[ inTrain]
y_test  <- y[-inTrain]

# Run RFE
result_rfe1 <- rfe(x = x_train, 
                   y = y_train, 
                   sizes = c(1:10), 
                   rfeControl = control)

# Print the results
result_rfe1

# Predictors
predictors(result_rfe1)

# Variable importance
varImp(result_rfe1)

varimp_data <- data.frame(feature = row.names(varImp(result_rfe1))[1:10],
                          importance = varImp(result_rfe1)[1:10, 1])

ggplot(data = varimp_data, 
       aes(x = reorder(feature, -importance), y = importance, fill = feature)) +
  geom_bar(stat="identity") + labs(x = "Features", y = "Variable Importance") + 
  geom_text(aes(label = round(importance, 2)), vjust=1.6, color="white", size=4) + 
  theme_bw() + theme(legend.position = "none")

# Visualize the results
ggplot(data = result_rfe1, metric = "Accuracy") + theme_bw()
ggplot(data = result_rfe1, metric = "Kappa") + theme_bw()

# Post prediction
postResample(predict(result_rfe1, x_test), y_test)

#-----------------------------------------------------------------------#

x <- data %>%
  select(-RDm1) %>%
  as.data.frame()

x_train <- x[ inTrain, ]
x_test  <- x[-inTrain, ]

# Run RFE
result_rfe2 <- rfe(x = x_train, 
                   y = y_train, 
                   sizes = c(1:54), 
                   rfeControl = control)

# Print the results
result_rfe2

# Variable importance
varimp_data <- data.frame(feature = row.names(varImp(result_rfe2))[1:51],
                          importance = varImp(result_rfe2)[1:51, 1])

ggplot(data = varimp_data, 
       aes(x = reorder(feature, -importance), y = importance, fill = feature)) +
  geom_bar(stat="identity") + labs(x = "Features", y = "Variable Importance") + 
  geom_text(aes(label = round(importance, 2)), vjust=1.6, color="white", size=4) + 
  theme_bw() + theme(legend.position = "none")
