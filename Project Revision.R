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
myinTrain <- createDataPartition(y=data$classe, p=0.6, list=FALSE)
mytrain <- data[myinTrain,]
mytest <- data[-myinTrain,]
dim(mytrain); dim(mytest)

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

## Do the same to the 20 samples used to test the prediction model
names(testing)
testing <- testing[,orig[-1]]
testing <- testing[, c(-13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24)]

## Random Forest
## Train the model using random forests.
library(randomForest)
library(rattle)
set.seed(17)
modRF <- randomForest(factor(classe) ~ ., 
                      data = mytrainraw, 
                      importance = TRUE,
                      do.trace = 100)

modRF
plot(modRF)
treeset.randomForest(modRF, n=500, root=1, format = "R")[1:20]
varImpPlot(modRF)

## Test the model using the test dataset to predict and capturing the out of 
## sample error.
predRF <- predict(modRF, newdata = mytestraw[,-1])
confusionMatrix(predRF, mytestraw$classe)

## Calculate OOS Error
missClass = function(values, predicted) {
        sum(predicted != values) / length(values)
}
OOS_errRateRF = missClass(mytestraw$classe, predRF)
OOS_errRateRF

## Predict values of blind sample
predBlind <- predict(modRF, testing)
predBlind

## To understand the relative importance of variables by group. Just pass in the
## names of the variables in the group as the members parameter. The result of 
## this matches the order in which the variables are listed in the 
## "MeanDecreaseAccuracy" table form varImpPlot(rf.obj) in number format.
var.share <- function(rf.obj, members) {
        count <- table(rf.obj$forest$bestvar)[-1]
        names(count) <- names(rf.obj$forest$ncat)
        share <- count[members] / sum(count[members])
        return(share)
}

members <- c("yaw_belt", "roll_belt", "magnet_dumbbell_z", "pitch_belt", 
             "magnet_dumbbell_y", "pitch_forearm")
var.share(modRF, members)


## Answer files as input to project 20 questions.
write_files = function(x){
        n = length(x)
        for(i in 1:n){
                filename = paste0("problem_",i,".txt")
                write.table(x[i], file=filename, quote=FALSE, row.names=FALSE, col.names=FALSE)
        }
}
write_files(predBlind)


