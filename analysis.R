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

# create smaller sample to test on before running full model
# s.train <- train %>% sample_n(1000)

# Output Function ---------------------------------------------------------

predict_export <- function(workflowName, fileName){
  # make predictions and prep data for Kaggle format
  x <- predict(workflowName,
                      new_data=test,
                      type="class")
  
  output <- as.data.frame(cbind(test$Id, as.character(x$.pred_class)))
  colnames(output) <- c("Id", "type")
  
  path <- paste0("/submissions/",fileName,".csv")
  vroom_write(output, file=path,delim=',')
}


# Random Forest -----------------------------------------------------------
RFmodel <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

RFrecipe <- recipe(Cover_Type ~ ., data = train) %>% 
# RFrecipe <- recipe(Cover_Type ~ ., data = s.train) %>% 
            update_role(Id, new_role = "ID") %>% 
            step_lencode_mixed(all_nominal_predictors(), outcome = vars(Cover_Type)) #%>%  #target encoding
            # step_mutate_at(all_of("Hillshade_9am","Hillshade_Noon","Hillshade_3pm"), fn = ~as.factor(.))

RFbaked <- bake(prep(RFrecipe), new_data = train)
# RFbaked <- bake(prep(RFrecipe), new_data = s.train)

RFworkflow <- workflow() %>%
  add_recipe(RFrecipe) %>%
  add_model(RFmodel)

RF_tuning_grid <- grid_regular(mtry(c(1,ncol(RFbaked))), min_n(), levels=10)

folds <- vfold_cv(train, v = 10, repeats = 1)

tune_control <- control_grid(verbose = TRUE)

cluster <- makePSOCKcluster(10)
doParallel::registerDoParallel(cluster)
start <- proc.time()
RF_CVresults <- RFworkflow %>% 
  tune_grid(resamples = folds,
            grid = RF_tuning_grid,
            metrics = metric_set(accuracy),
            control = tune_control)

RF_bestTune <- RF_CVresults %>% 
  select_best("accuracy")

RF_Final_wf <- RFworkflow %>% 
  finalize_workflow(RF_bestTune) %>% 
  fit(data=train)
  # fit(data=s.train)

proc.time()-start
stopCluster(cluster)

predict_export(RF_Final_wf, "RF1")


