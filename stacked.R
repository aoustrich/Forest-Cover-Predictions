library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(ranger)
library(parallel)
library(bonsai)
library(lightgbm)
library(stacks) 
library(keras)
library(parsnip)
library(baguette)

# library(tidyverse)
# library(tidymodels)
# library(vroom)
# library(parsnip)
# library(keras)
# library(baguette)
# library(bonsai)

train <- vroom("data/train.csv")
test <- vroom("data/test.csv")
train$Cover_Type = as.factor(train$Cover_Type)

# train %>% 
#   count(Cover_Type)

# Recipes -----------------------------------------------------------------

boosted_recipe <- recipe(Cover_Type ~ ., data = train) %>%
  update_role(Id, new_role = "Id") %>%
  step_mutate(Id = factor(Id)) %>%
  step_mutate_at(all_outcomes(), fn = factor, skip = TRUE) %>% 
  step_mutate_at("Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology",
                 "Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways",
                 "Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Horizontal_Distance_To_Fire_Points",
                 fn=as.numeric) %>%
  step_normalize("Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology",
                 "Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways",
                 "Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Horizontal_Distance_To_Fire_Points") %>%
  step_zv() 

rf_recipe <- recipe(Cover_Type ~ ., data = train) %>% 
  update_role(Id, new_role = "Id") %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_predictors()) 

nn_recipe <- recipe(Cover_Type~., data = train) %>%
  update_role(Id, new_role = "Id") %>%
  step_rm(Id) %>%
  step_zv(all_predictors()) %>% 
  step_range(all_numeric_predictors(), min=0, max=1)

# Stacking Settings -------------------------------------------------------

# Control settings for Stacking
untunedModel <- control_stack_grid()
tunedModel <- control_stack_resamples()

# Split data for Cross Validation
folds <- vfold_cv(train, v = 5, repeats=1)

# Boosting ----------------------------------------------------------------

boosted_model <- boost_tree(tree_depth=8,
                            trees=1000,
                            learn_rate=.2) %>%
            set_engine("lightgbm") %>%
            set_mode("classification")

boost_wf <- workflow() %>%
  add_recipe(boosted_recipe) %>%
  add_model(boosted_model)

cluster <- makePSOCKcluster(2)

doParallel::registerDoParallel(cluster)
startBoostFit <- proc.time()
# fit 
boosted_results_stack <- fit_resamples(
                            boost_wf,
                            resamples = folds,
                            metrics = metric_set(accuracy,roc_auc),
                            control = tunedModel)
proc.time() - startBoostFit

stopCluster(cluster)

  # XG BOOST VERSION

XGboosted_model <- boost_tree(tree_depth=2,
                            trees=500,
                            learn_rate=.2) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

XGboost_wf <- workflow() %>%
  add_recipe(boosted_recipe) %>%
  add_model(XGboosted_model)

XGcluster <- makePSOCKcluster(4)

doParallel::registerDoParallel(XGcluster)
XGstartBoostFit <- proc.time()
# fit 
XGboosted_results_stack <- fit_resamples(
  XGboost_wf,
  resamples = folds,
  metrics = metric_set(accuracy,roc_auc),
  control = tunedModel)
proc.time() - XGstartBoostFit

stopCluster(XGcluster)

# Random Forest -----------------------------------------------------------
doParallel::registerDoParallel(cluster)
rf_mod <- rand_forest(mtry = 10,
                      min_n = 5,
                      trees=750) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_wf <- workflow() %>%
  add_recipe(rf_recipe) %>%
  add_model(rf_mod)

# rf_tuning_grid <- grid_regular(mtry(c(1,ncol(baked))), min_n(), levels=10)

# rf_results_stack <- rf_wf %>% 
#                       tune_grid(resamples = folds,
#                             metrics = metric_set(accuracy),
#                             control = untunedModel)
startRFFit <- proc.time()
rf_results_stack <- fit_resamples(rf_wf,
                                  resamples = folds,
                                  metrics = metric_set(roc_auc),
                                  control = tunedModel)
proc.time() - startRFFit
stopCluster(cluster)

# Neural Network ----------------------------------------------------------
# NN model
nn_model <- mlp(hidden_units = 10,
                epochs = 100) %>%
  set_engine("keras") %>%
  set_mode("classification")

nn_wf <- workflow() %>%
        add_model(nn_model) %>%
        add_recipe(nn_recipe)

startNNFit <- proc.time()
nn_results_stack <- fit_resamples(nn_wf,
                                  resamples = folds,
                                  metrics = metric_set(roc_auc),
                                  control = tunedModel)
proc.time() - startNNFit



# Stacking Models ----------------------------------------------------------

# set up the stacked model

my_stack <- stacks() %>%
  add_candidates(XGboosted_results_stack) %>% 
  add_candidates(rf_results_stack) #%>% 
  add_candidates(nn_results_stack) 

cluster <- makePSOCKcluster(2)
doParallel::registerDoParallel(cluster)
startStackFit <- proc.time()
# fit the stacked model
stacked_model <- my_stack %>%
  blend_predictions() %>%
  fit_members()

proc.time() - startStackFit
stopCluster(cluster)

# use the stacked model to generate predictions

startStatPred <- proc.time()
stack_preds <- stacked_model %>%
  predict(new_data = test, type = "class")
proc.time() - startStatPred

# prepare the output for submission to kaggle

stack_output <- tibble(Id = test$Id, Cover_Type = stack_preds$.pred_class)

vroom_write(stack_output, "submissions/stack_output.csv", delim = ",")
