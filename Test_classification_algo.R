### Test file to try and understand different classification algorithms

#Load important R libraries

library(ggplot2)
library(knitr)
library(dplyr)
library(randomForest) #RF Model
library(rpart)        #Decision Tree Model
library(rpart.plot)   #Decision Tree Plot
library(RColorBrewer)
#library(rattle)
library(caret)
library(e1071)
library(ROCR)


#Setting Seed for reproducibility of results

set.seed(666)

#Check the current working directory

getwd()

#Check files in working directory

dir()

#load test data (iris dataset)

### Random Forest ######################################################################################

##Ref: https://www.blopig.com/blog/2017/04/a-very-basic-introduction-to-random-forests-using-r/




# Calculate the size of each of the data sets:
data_set_size <- floor(nrow(iris)/2)
# Generate a random sample of "data_set_size" indexes
indexes <- sample(1:nrow(iris), size = data_set_size)

# Assign the data to the correct sets
training <- iris[indexes,]
validation1 <- iris[-indexes,]


# Perform training:
rf_classifier = randomForest(Species ~ ., data=training, ntree=100, mtry=2, importance=TRUE)

rf_classifier


varImpPlot(rf_classifier)

# Validation set assessment #1: looking at confusion matrix
prediction_for_table <- predict(rf_classifier,validation1[,-5])
table(observed=validation1[,5],predicted=prediction_for_table)


# Validation set assessment #2: ROC curves and AUC

# Needs to import ROCR package for ROC curve plotting:
library(ROCR)

# Calculate the probability of new observations belonging to each class
# prediction_for_roc_curve will be a matrix with dimensions data_set_size x number_of_classes
prediction_for_roc_curve <- predict(rf_classifier,validation1[,-5],type="prob")

# Use pretty colours:
pretty_colours <- c("#F8766D","#00BA38","#619CFF")
# Specify the different classes 
classes <- levels(validation1$Species)
# For each class
for (i in 1:3)
{
  # Define which observations belong to class[i]
  true_values <- ifelse(validation1[,5]==classes[i],1,0)
  # Assess the performance of classifier for class[i]
  pred <- prediction(prediction_for_roc_curve[,i],true_values)
  perf <- performance(pred, "tpr", "fpr")
  if (i==1)
  {
    plot(perf,main="ROC Curve",col=pretty_colours[i]) 
  }
  else
  {
    plot(perf,main="ROC Curve",col=pretty_colours[i],add=TRUE) 
  }
  # Calculate the AUC and print it to screen
  auc.perf <- performance(pred, measure = "auc")
  print(auc.perf@y.values)
}

### Testing ============================ RandomForest documentation

data(mtcars)
mtcars.rf <- randomForest(mpg ~ ., data=mtcars, ntree=1000, keep.forest=FALSE, importance=TRUE)
                          
importance(mtcars.rf)
importance(mtcars.rf, type=1)

confusionMatrix(mtcars.rf)


### RF on Titanic dataset ------------------------------------------------------

#Load important R libraries

library(ggplot2)
library(knitr)
library(dplyr)
library(randomForest) #RF Model
library(rpart)        #Decision Tree Model
library(rpart.plot)   #Decision Tree Plot
library(RColorBrewer)
library(rattle)
library(caret)
library(e1071)

#Setting Seed for reproducibility of results

set.seed(666)

#Read the input data file

train <- read.csv2("input/train.csv", header = TRUE, sep = ",", na.strings = "", stringsAsFactors = FALSE)
test <- read.csv2("input/test.csv", header = TRUE, sep = ",", na.strings = "", stringsAsFactors = FALSE)

#combine train and test data set

#create identifier column if we need to separate train and test data later, we'll add a column "IsTrain" with values (train=TRUE, test=FALSE) to both datasets

train$IsTrain <- TRUE
test$IsTrain <- FALSE


#Add "Survived" column to test and fill NA before merging test and train

test$Survived <- NA

# Merge train and test

merge <- rbind(train, test)

#Missing value overview in Merge dataset

missmap(merge, col=c("red", "green"), main = "Missing Values in combined dataset")
#Result: Missing values mostly in "Cabin" and "Age" columns, Survived column is not available for test records - Which is normal. For other columns also, we'll check if any missing values are there and treat them if possible

#Check missing values in all relevant columns (Exc. PassengerID, Survived, IsTrain)


table(is.na(merge$Age)) #263 missing values
table(merge$Age)

table(is.na(merge$Pclass)) #NO missing values, Identified as categorical variable
table(merge$Pclass) #1: 323 + 2: 277 + 3: 709 (Most Passengers are traveling in 3rd Class)

table(is.na(merge$Name)) #NO missing values

table(is.na(merge$Sex)) #NO missing values, Identified as categorical variable
table(merge$Sex) #F: 466 + M: 843 (Two-third passengers are Male)

table(is.na(merge$SibSp)) #NO missing values

table(is.na(merge$Parch)) #NO missing values

table(is.na(merge$Ticket)) #NO missing values

table(is.na(merge$Fare)) #1 missing values

table(is.na(merge$Cabin)) #1014 missing values

table(is.na(merge$Embarked)) #2 missing values, Identified as categorical variable
table(merge$Embarked) # C: 270 + Q: 123 + S: 914 + 2 NA (missing values)

# We need to treat Cabin, Age, fare and Embarked for missing values. For a simplistic approach, we can fill median values for Age and Fare. Also, we can take mode (most common) class for  Embarked. Cabin? It could be bit tricky, so we will not use this column for our prediction for now.

merge[is.na(merge$Embarked), "Embarked"] <- 'S'  #Replace NA with mode (Most frequent) value

merge$Age <- as.numeric(as.character(merge$Age))   #convert Age to numeric if imported as chr
merge[is.na(merge$Age), "Age"] <- median(merge$Age, na.rm = TRUE) #Replace NA with median value


merge$Fare <- as.numeric(as.character(merge$Fare))   #convert Fare to numeric if imported as chr
merge[is.na(merge$Fare), "Fare"] <- median(merge$Fare, na.rm = TRUE) #Replace NA with median value

# We need to treat categorical columns too, Pclass, Sex and Embarked (as identified earlier)

merge$Pclass <- as.factor(merge$Pclass)
merge$Sex <- as.factor(merge$Sex)
merge$Embarked <- as.factor(merge$Embarked)

# Also  change the predictor var "Survived" as factor (0, 1)

merge$Survived <- as.factor(merge$Survived)

train <- merge[merge$IsTrain==TRUE,]
test <- merge[!merge$IsTrain==TRUE,]

# Base Model with Random Forest

#explicitly select the columns for prediction and strore in formulae

form <- as.formula("Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked")
rf_model <- randomForest(formula = form, data = train, ntree = 500, mtry = 3, nodesize = 0.01 * nrow(train),  importance=TRUE)

#Make Prediction on Test set

Survived <- predict(rf_model, newdata = test)

#Check Prediction

Survived

##-------- Improvements -----------------

## https://uc-r.github.io/random_forests
## https://www.analyticsvidhya.com/blog/2016/08/practicing-machine-learning-techniques-in-r-with-mlr-package/
## https://machinelearningmastery.com/tune-machine-learning-algorithms-in-r/
## https://www.hackerearth.com/practice/machine-learning/machine-learning-algorithms/tutorial-random-forest-parameter-tuning-r/tutorial/



library(rsample)      # data splitting 
#library(randomForest) # basic implementation
library(ranger)       # a faster implementation of randomForest
#library(caret)        # an aggregator package for performing many machine learning models
#library(h2o)          # an extremely fast java-based platform

rf_model$confusion

rf_model$importance

rf_model$importanceSD

rf_model$forest

plot(rf_model)

# number of trees with lowest MSE
which.min(rf_model$ms)
## [1] 344

# RMSE of this optimal random forest
sqrt(m1$mse[which.min(m1$mse)])
## [1] 25673.5

# Tuning RF model ---- Finding optimal mtry values based on OOB

set.seed(666)

train_index <- names(train)

y <- as.factor(train$Survived) #Just "Survived":: factor for classification, numeric for regression

x <- train_index[-c(1,2,4, 11, 13)] # Dropping id, survived, name, cabin, istrain

form <- as.formula("Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked")

tuneRF(x=train[x], y, mtryStart=2, ntreeTry=500, stepFactor=2, improve=0.02, trace=TRUE, plot=TRUE, doBest=FALSE)

# Optimal mtry found to be 4




