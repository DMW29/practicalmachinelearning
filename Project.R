## LOAD DATA
## Get training data
trainURL <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
training <- read.csv(url(trainURL), na.strings=c("NA","#DIV/0!",""), 
                     stringsAsFactors = FALSE)

## Get the test data
testURL <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
testing <- read.csv(url(testURL), na.strings=c("NA","#DIV/0!",""), 
                    stringsAsFactors = FALSE)

## CREATE TRAINING AND TESTING SETS
## All work is done on the training dataset. The testing dataset is used only 
## exactly one time to assess the model. But cross-validation can be used to 
## break the training set into a train and test set. But first we need to look
## at the data.

library(ISLR); library(ggplot2); library(caret)
data <- data.frame(training)
myinTrain <- createDataPartition(y=data$classe, p=0.8, list=FALSE)
mytrain <- data[myinTrain,]
mytest <- data[-myinTrain,]
dim(mytrain); dim(mytest)

myinval <- createDataPartition(y=mytrain$classe, p=0.6, list=FALSE)
mytrain <- mytrain[myinval,]
myval <- mytrain[-myinval,]
dim(mytrain); dim(mytest); dim(myval)

## CHOOSE VARIABLES TO USE IN MODEL CREATING NEW TRAIN AND TEST SETS
## The original measurements are extracted to create a new training set. This
## is done in the spirit of having a larger number of observations as opposed
## to the summarized information and to use the information most related to the 
## outcome variable. It is assumed the data are not time dependent. Therefore,
## the time variables are also left out.
names(mytrain)
orig <- c("classe", "roll_belt", "roll_arm", "roll_dumbbell", "roll_forearm",
          "pitch_belt", "pitch_arm", "pitch_dumbbell", "pitch_forearm", 
          "yaw_belt", "yaw_arm", "yaw_dumbbell", "yaw_forearm", 
          "amplitude_roll_belt", "amplitude_roll_arm", "amplitude_roll_dumbbell", "amplitude_roll_forearm",
          "amplitude_pitch_belt", "amplitude_pitch_arm", "amplitude_pitch_dumbbell", "amplitude_pitch_forearm",
          "amplitude_yaw_belt", "amplitude_yaw_arm", "amplitude_yaw_dumbbell", "amplitude_yaw_forearm",
          "gyros_belt_x", "gyros_arm_x", "gyros_dumbbell_x", "gyros_forearm_x",
          "gyros_belt_y", "gyros_arm_y", "gyros_dumbbell_y", "gyros_forearm_y",
          "gyros_belt_z", "gyros_arm_z", "gyros_dumbbell_z", "gyros_forearm_z",
          "accel_belt_x", "accel_arm_x", "accel_dumbbell_x", "accel_forearm_x",
          "accel_belt_y", "accel_arm_y", "accel_dumbbell_y", "accel_forearm_y",
          "accel_belt_z", "accel_arm_z", "accel_dumbbell_z", "accel_forearm_z",
          "magnet_belt_x", "magnet_arm_x", "magnet_dumbbell_x", "magnet_forearm_x",
          "magnet_belt_y", "magnet_arm_y", "magnet_dumbbell_y", "magnet_forearm_y",
          "magnet_belt_z", "magnet_arm_z", "magnet_dumbbell_z", "magnet_forearm_z")
mytrainraw <- mytrain[,orig]
summary(mytrainraw)

## REMOVE FEATURES THAT ARE MOSTLY NA
## The amplitude measurements are almost 100% NA and are removed from the dataset
amplitude <- names(mytrainraw) %in% c("amplitude_roll_belt", "amplitude_roll_arm", "amplitude_roll_dumbbell", "amplitude_roll_forearm",
           "amplitude_pitch_belt", "amplitude_pitch_arm", "amplitude_pitch_dumbbell", "amplitude_pitch_forearm",
           "amplitude_yaw_belt", "amplitude_yaw_arm", "amplitude_yaw_dumbbell", "amplitude_yaw_forearm")
mytrainraw <- mytrainraw[!amplitude]
summary(mytrainraw)

## Do the same for mytest dataset
mytestraw <- mytest[,orig]
mytestraw <- mytestraw[!amplitude]

## Do the same for myVal dataset
myvalraw <- myval[,orig]
myvalraw <- myvalraw[!amplitude]

## Do the same to the 20 samples used to test the prediction model
names(testing)
testing <- testing[,orig[-1]]
testing <- testing[, c(-13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24)]
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## DATA EXPLORATION to determine how to set options
## SKEWNeSS
## Look for skewed features. Skewness will be close to zero when the distribution
## is close to symmetric. Large positive or negative skewness values indicate
## right and left skewness, respectively. Since all the predictors are numeric 
## columns, the apply function can be used to calculate the skewness across columns.
library(e1071)
skewValues <- apply(mytrainraw[,-1], 2, skewness)
skewValues

## Some features are highly skewed. gyros_dumbbell_x, gyros_forearm_x, 
## gyros_dumbbell_y, gyros_forearm_y, gyros_dumbbell_z, gyros_forearm_z, and 
## magnet_belt_y all have skewness values greater than 2.
par(mfrow=c(4,4))
hist(mytrainraw$gyros_dumbbell_x, main = "gyros_dumbbell_x")
hist(mytrainraw$gyros_forearm_x, main = "gyros_forearm_x")
hist(mytrainraw$gyros_dumbbell_y, main = "gyros_dumbbell_y")
hist(mytrainraw$gyros_forearm_y, main = "gyros_forearm_y")
hist(mytrainraw$gyros_dumbbell_z, main = "gyros_dumbbell_z")
hist(mytrainraw$gyros_forearm_z, main = "gyros_forearm_z")
hist(mytrainraw$magnet_belt_y, main = "magnet_belt_y")

## To transform it is necessary to ensure there are not a large amount of 0 or
## negative values. These skewed variables have the same proportion of 0 values.
## This implies something sytemic in the data. Because the values of gyros are
## generally very small, I don't this bias will be an issue and believe is due
## to missing values on one person in one class.
pzerogdx<- length(mytrainraw[mytrainraw$gyros_dumbbell_x == 0.00,])/length(mytrainraw$gyros_dumbbell_x)
pzerogfx<- length(mytrainraw[mytrainraw$gyros_forearm_x == 0.00,])/length(mytrainraw$gyros_forearm_x)
pzerogdy<- length(mytrainraw[mytrainraw$gyros_dumbbell_y == 0.00,])/length(mytrainraw$gyros_dumbbell_y)
pzerogfy<- length(mytrainraw[mytrainraw$gyros_forearm_y == 0.00,])/length(mytrainraw$gyros_forearm_y)
pzerogdz<- length(mytrainraw[mytrainraw$gyros_dumbbell_z == 0.00,])/length(mytrainraw$gyros_dumbbell_z)
pzerogdz<- length(mytrainraw[mytrainraw$gyros_forearm_z == 0.00,])/length(mytrainraw$gyros_forearm_z)

## In fact, due to the features having both large values and small values, I will
## standarize the variables to have 0 mean and variance 1. This will also take
## care of the skewed distributed features.

## NEAR ZERO VARIANCE
## There are no variables with near zero variance.
nearZeroVar(mytrainraw[,-1])

## COLLINEARITY
## Correlated features look the same to the outcome variable and add more complexity
## to the model than information they provide to the model. A parsimoneous model
## is best. When obtaining predictor data is costly, less is better. Also, linear
## regression is sensitive to collinearity and causes inflated parameter variance.
## Principle Components Analysis (PCA) can tease out the correlated variables and
## is useful when there are many covariates.
correlations <- cor(mytrainraw[,-1])
dim(correlations)
library(corrplot)
corrplot(correlations, order = "hclust")
correlations <- cor(mytrainraw[,-1])
highCorr <- findCorrelation(correlations, cutoff = 0.75)
highCorr
vars <- mytrainraw[,-1]
names(vars)
highCorrVar <- names(vars[,highCorr])
highCorrVar
names(mytrainraw)
## Conclusion of data exporation is to use pca to manage skewed distributions
## and the various ranges of data.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

## Train the model using k-nearest neighbors
set.seed(1720)
fitControl <- trainControl(method = "cv", number = 20)
modFit <- train(classe ~ ., data = mytrainraw, 
                method = "knn", 
                preProcess = "pca",
                trControl = trainControl("cv"),
                tuneLength = 3)
modFit
plot(modFit)
modFit$finalModel

## Predict using the test set
set.seed(1720)
predknn <- predict(modFit, newdata = mytestraw[,-1])
confusionMatrix(mytestraw$classe, predknn)

## Now apply all of this learning to the testing dataset for the project
set.seed(1790)
predTestknn <- predict(modFit, newdata = testing)
predTestknn

## Out of sample error estimation for knn
missClass = function(values, predicted) {
        sum(predicted != values) / length(values)
}
OOS_errRateknn = missClass(mytestraw$classe, predknn)
OOS_errRateknn

## In general, with a multistep modeling procedure, cross-validation must be 
## applied to the entire sequence of modeling steps. In particular, samples must
## be "left out" before any selection or filtering steps are applied. The caveat
## being initial unsupervised screening steps can be done before samples are left
## out. For example, selecting features that have highest variance across all
## samples.


## The true error rate of a binomial predictor is 50%. The outcome variable here
## has 5 levels. So, the true error is estimated to be 20% when the predictors
## are independent of the outcome.

## Random Forest
## Train the model using random forests. Two-fold cv is used to speed processing
## but may increase estimation bias.
set.seed(1820)
modFitRF <- train(classe ~ ., data = mytrainraw, 
                method = "rf",
                trControl = trainControl("cv", number = 2),
                prox=TRUE,
                verbose=TRUE,
                allowParallel=TRUE)
modFitRF
plot(modFitRF)
modFitRF$finalModel

## Predict classes on test data
set.seed(1820)
predRF <- predict(modFitRF, mytestraw[,-1])
confusionMatrix(mytestraw$classe, predRF)


set.seed(1820)
predTest <- predict(modFitRF, testing)
predTest

## Out of sample error estimation for knn
missClass = function(values, predicted) {
        sum(predicted != values) / length(values)
}
OOS_errRateRF = missClass(mytestraw$classe, predRF)
OOS_errRateRF

## Compare knn to rf on the 20 samples
table(predTest, predTestknn)

## Plotting a few variables.
## qplot(roll_belt, magnet_dumbbell_y, colour = classe, data=mytrainraw)

## Conclusion
## Predicting the 20 test cases provided by the instructor using k-nearest 
## neighbors resulted in predknn: [1] B A C A A B D B A A D C B A E E A B B B and
## out of sample error predknn [1] 0.04703033, which means there will be at least
## 0.94 errors in the 20 samples.

## Predicting the 20 test cases provded by the instructor using random forests
## resulted in predRF: [1] B A B A A E D B A A B C B A E E A B B B and out of 
## sample error predRF  [1] 0.005990314, which means there will be at least 0.12
## error in the 20 samples.

## This means random forests has an 87.3% decrease in out of sample error. 
## Although, random forests takes significanly more time in the order of ~10 
## minutes, when accuracy is most important, random forests may be preferable to
## k-nearest neighbors, which is more interpretable.










