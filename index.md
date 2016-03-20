# Practical Machine Learning Course Project
DMWaterson  
March 19, 2016  

### Introduction
One thing that people regularly do is quantify how much of a particular activity
they do, but they rarely quantify how well they do it. In this project, data 
from accelerometers on the belt, forearm, arm, and dumbell of six participants
will be used create a model to predict whether the exercise is being done 
correctly or incorrectly in one of four possible ways.

### Natural Error Rate
The probability of perfect classification is approximately: (1/5)^sample size. 
So, when n = 1, the probability of 100% accuracy is 20%. When n = 2, the 
probability of 100% accuracy is 4%. When n = 10, then the probability of 100% 
accuracy is $10^{-7}$. So, with a larger sample there is a smaller probability 
of high accuracy. Therefore, there is a higher probability the model is indeed 
accurate and not just random. This is the prediction goal.

### Data Processing
The data for this assignment come from [this source](http://groupware.les.inf.puc-rio.br/har). The training dataset is downloaded from [here](http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv). The
blind test set is downloaded from [here](http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv). The
training dataset is split into a training set and a test set for building and testing the prediction model. The split percentages are 60% and 40%, respectively. The choice of variables to include in the model are those that are most closely related to the outcome. This is done because the relationship of indirectly related variables to the outcome can change over time.  With this dataset the most closely related also provide for the largest sample size to help with variance reduction. Therefore, variables that are summary statistics are excluded. Because of the additional complexity of model building with time factors, time variables are also excluded with the intent of reintroducing them if an acceptable model cannot be found. Amplitude variables such as amplitude_roll_belt, amplitude_roll_arm, etc., include 98% to 100% missing observations and also are excluded for model building. The resulting datasets include 49 predictor variables with the number of observations as described below.

Variable Count|Training Set|Testing Set
--------------|------------|-----------
   49         |   11776    |   7846           

### Data Exploration
The range of values for the 49 variables varies widely. For example, while roll,
pitch and yaw values range roughly between -100 and 100, most gyros variables 
range between roughly -3 and 3. Gyros distributions are also highly skewed. Further still, magnet variables can range from -1000 to 1000 and 
can have the effect of overwhelming the analysis to choose in favor of highest variance variables. No variables have near zero variance. There are 15 highly correlated variables that will have the effect of variance inflation. However,
based on [this article](http://blog.explainmydata.com/2012/07/should-you-apply-pca-to-your-data.html), no variable transformations are done. However, transformation options will be revisited if an acceptable model is not found.

### Train the model
The Random Forests algorithm is used to pick variables and the prediction 
function. It is assumed data collection costs are low and interpretability is not a requirement. When data collection costs are high, the number of variables in the final model should be at a minumum. Interpetability can influence algorithm choice. For example, rpart() will result in an interpretable tree, however, randomForests() will not. 

### Test the model
The Random Forests algorithm was run 5 times on a different set of training data each time as a type of cross-validation to obtain an estimate of the out-of-sample (OOS) error. This is accomplished by splitting the dataset into training and testing sets for each run and then setting the seed for the algorithm. The code for data partitioning, model training and calculating OOS follows.

```r
## Split the original training dataset to create new training and testing sets for model building
library(caret)
data <- data.frame(training)
myinTrain <- createDataPartition(y=data$classe, p=0.6, list=FALSE)
mytrain <- data[myinTrain,]
mytest <- data[-myinTrain,]
dim(mytrain); dim(mytest)

## Build the model
library(randomForest)
set.seed(17)
modRF <- randomForest(factor(classe) ~ ., 
                      data = mytrainraw, 
                      importance = TRUE,
                      do.trace = 100)
modRF

## Calculate OOS Error
missClass = function(values, predicted) {
        sum(predicted != values) / length(values)
}
OOS_errRateRF = missClass(mytestraw$classe, predRF)
OOS_errRateRF
```

### Estimate Out-of-Sample error
randomForests() provides an out-of-bag (OOB) estimate of the OOS error. The following table compares OOB and OOS for each run. On average, the OOB estimate of the OOS error and the calculated OOS estimate are very similar. The OOB error estimate seems only slightly more conservative. I think this estimate means that out of 1,000,000 predictions 6,900 of them will be in error, or more conservatively, 7,020 will be in error.

Training Run|Out-Of-Bag Estimate|Out-Of-Sample Estimate
------------|-------------------|----------------------
1           |0.01               |0.0092
2           |0.0063             |0.0054
3           |0.0069             |0.0064
4           |0.0055             |0.0068
5           |0.0064             |0.0065
------------|-------------------|----------------------
Average     |0.00702            |0.0069

### Consider the most important predictors
The importance option is set during model training and for each of the five training runs. The top six important variables were captured based on MeanDecreaseAccuracy that results from running modRF after model fit. The variables ranked beyond the top six were inconsistent. It is only the top six found to be consistent. The relative importance of the top six is captured and averaged for all five runs. The following table presents these data.

Variable          |Averaged Relative Importance
------------------|----------------------------
yaw_belt          |0.1982
pitch_belt        |0.1760
roll_belt         |0.1734
magnet_dumbbell_z |0.1706
magnet_dumbbell_y |0.1498
pitch_forearm     |0.1316

This is the code to calculate the relative importance of the top six variables consistantly reported for each of five training runs. Please reference [this website](http://stats.stackexchange.com/questions/92419/relative-importance-of-a-set-of-predictors-in-a-random-forests-classification-in/92843) for the code and a nice explanation of Gini importance.

```r
## Calculate group relative importance
var.share <- function(rf.obj, members) {
        count <- table(rf.obj$forest$bestvar)[-1]
        names(count) <- names(rf.obj$forest$ncat)
        share <- count[members] / sum(count[members])
        return(share)
}

members <- c("yaw_belt", "roll_belt", "magnet_dumbbell_z", "pitch_belt", 
             "magnet_dumbbell_y", "pitch_forearm")
var.share(modRF, members)
```

### Blind testing
Each of the five test runs resulted in the same predictions for the blinded test of twenty observations. They are as follows: 
 B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 


### Conclusions and Further Study
The OOS error rate is estimated to be 0.69% and means that 6,900 errors will be made for every 1,000,000 predictions. To decrease this amount of error further, it is possible to tune the model to better capture classes C and D as these categories have the highest class error as reported by the confusion matrix. For example, the classe C average class error is 1.23% over the five training runs. Stacking is one option for decreasing this error. Including the time variables into the analysis is also an option for reducing OOS error. Principle components analysis can also be explored to reduce the number of predictors and possibly improve the model's predictive capability. 

Incorrectly performed dumbbell bicept curls can be predicted with a decent accuracy rate. Not only can dumbbell curls be determined as incorrect, they can be classified into the type of incorrectness. Further study can include other types of exercise feedback. In this case, machine learning must be able to identify the exercise being performed and then classify it into a correctness category to provide correct/incorrect feedback. Also, understanding the most important features by classification category could then be used to provide feedback to the user on how the exercise is being performed so they can correct their exercise form. It is true that injury is more prone to novice exercisers who perform exercise repeatedly and incorrectly. This is also true with heavy weights as referenced by [Gov.UK Health and Social Care Information Centre (HSCIC)](http://www.nhs.uk/Livewell/fitness/Pages/Top-10-gym-exercises-done-incorrectly.aspx)

