library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(ranger)
library(parallel)


train <- vroom("data/train.csv")
test <- vroom("data/test.csv")
train$Cover_Type <- as.factor(train$Cover_Type)

train$Cover_Type <- as.factor(train$Cover_Type)

my_recipe <- recipe(Cover_Type ~ ., data = train) %>%
  update_role(Id, new_role = "ID") %>%
  #step_mutate_at(all_numeric_predictors(), fn = factor) %>% # turn all numeric features into factors
  #step_other(all_factor_predictors(), threshold = .005) %>% # combines categorical values that occur <5% into an "other" value
  #step_dummy(all_nominal_predictors()) # dummy variable encoding
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(Cover_Type))  #target encoding


bake(prep(my_recipe), new_data = train)


# Random Forest -----------------------------------------------------------

rf_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_mod)

rf_tuning_grid <- grid_regular(mtry(c(1,ncol(train))), min_n(), levels=10)

folds <- vfold_cv(train, v = 3, repeats = 1)

tune_control <- control_grid(verbose = TRUE)

rf_results <- rf_wf %>% 
  tune_grid(resamples = folds,
            grid = rf_tuning_grid,
            metrics = metric_set(accuracy),
            control = tune_control)

rf_bestTune <- rf_results %>% 
  select_best("accuracy")

rf_final_wf <- rf_wf %>% 
  finalize_workflow(rf_bestTune) %>% 
  fit(data=train)

rf_preds <- predict(rf_final_wf,
                    new_data=test,
                    type="class")

rf_submit <- as.data.frame(cbind(test$id, as.character(rf_preds$.pred_class)))
colnames(rf_submit) <- c("id", "type")
write_csv(rf_submit, "rf_submit.csv")
