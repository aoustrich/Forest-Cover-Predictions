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

# Recipe ------------------------------------------------------------------
  # select all the variables that are not pre-dummy encoded
nonDummyVars <- colnames(train)[2:11]
nonDummy <- c("Elevation","Aspect", "Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Horizontal_Distance_To_Fire_Points")
dummyVars <- colnames(train)[12:55]

  # set up recipe
myRecipe <- recipe(Cover_Type ~ ., data = train) %>% 
  update_role(Id, new_role = "ID") %>% 
  # step_integer(all_of(dummyVars)) %>%
  # step_num2factor(all_of(dummyVars), levels=c("0","1"), transform = function(x) x+1) %>% prep()
  # step_num2factor(Wilderness_Area1, levels=c("0","1"), transform = function(x) x+1) %>% prep()
  # step_normalize(all_of(colnames(train)[2:11])) %>%
  # step_pca(all_of(colnames(train)[2:11]), threshold = .85) %>%  prep()

baked <- bake(myRecipe, new_data = train)



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

RF_tuning_grid <- grid_regular(mtry(c(1,ncol(RFbaked))), min_n(), levels=2)

folds <- vfold_cv(train, v = 2, repeats = 1)

tune_control <- control_grid(verbose = TRUE)

cluster <- makePSOCKcluster(3)
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



# Naive Bayes -------------------------------------------------------------
# 
  #   model
  naiveModel <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
    set_mode("classification") %>%
    # set_engine("klaR")
    set_engine("naivebayes")

  #   workflow
  naiveWF <- workflow() %>%
    # add_recipe(hauntedRecipeNoID) %>% 	##### switch recipe to klaR_recipe
    add_recipe(myRecipe) %>%
    add_model(naiveModel)



  #   tuning
  naiveGrid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5)

  #   folds for cross validation
  naiveFolds <- vfold_cv(train, v=5, repeats=1)

  cl <- makePSOCKcluster(3)
  doParallel::registerDoParallel(cl)

  #   fit model with cross validation
  naiveResultsCV <- naiveWF %>%
    tune_grid(resamples=naiveFolds,
              grid=naiveGrid,
              metric_set("accuracy"))

  #   find best tune
  naiveBestTune <- naiveResultsCV %>%
    select_best("accuracy")

  #   finalize the Workflow & fit it
  naiveFinalWF <-
    naiveWF %>%
    finalize_workflow(naiveBestTune) %>%
    fit(data=train)

  #   predict and export
  outputCSV <-  predict_export(naiveFinalWF,"naiveBayes_klaR")
  stopCluster(cl)
# 
# 
#   
# 
