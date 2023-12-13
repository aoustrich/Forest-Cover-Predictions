library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(ranger)
library(parallel)

# library(poissonreg)
# library(rpart)
# library(stacks)
# library(discrim)
# library(naivebayes)
# library(kknn)
# library(kernlab)
# library(themis)
# library(keras)
# library(bonsai)
# library(lightgbm)
# library(dbarts)


# Reading Data ------------------------------------------------------------

train <- vroom("data/train.csv")
test <- vroom("data/test.csv")
train$Cover_Type <- as.factor(train$Cover_Type)
test$Id <- as.integer(test$Id)

# create smaller sample to test on before running full model
# s.train <- train %>% sample_n(1000)

# Output Function ---------------------------------------------------------

predict_export <- function(workflowName, fileName){
  # make predictions and prep data for Kaggle format
  x <- predict(workflowName,
               new_data=test,
               type="class")
  if (typeof(test$Id != "integer")){
    test$Id <- as.integer(test$Id)
  }
  
  output <- as.data.frame(cbind(test$Id, as.character(x$.pred_class)))
  colnames(output) <- c("Id", "Cover_Type")
  
  if (typeof(output$Id != "integer")){
    output$Id <- as.integer(output$Id)
  }
  
  path <- paste0("submissions/",fileName,".csv")
  vroom_write(output, file=path,delim=',')
}