library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(ranger)
library(parallel)


train <- vroom("data/train.csv")
test <- vroom("data/test.csv")
train$Cover_Type <- as.factor(train$Cover_Type)

my_recipe <- recipe(Cover_Type ~ ., data = train) %>%
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


baked <- bake(prep(my_recipe), new_data = train)


# Random Forest -----------------------------------------------------------

rf_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=750) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_mod)

rf_tuning_grid <- grid_regular(mtry(c(1,ncol(baked))), min_n(), levels=10)

folds <- vfold_cv(train, v = 3, repeats = 1)


cluster <- makePSOCKcluster(4)
doParallel::registerDoParallel(cluster)
startCV <- proc.time()

rf_results <- rf_wf %>% 
  tune_grid(resamples = folds,
            grid = rf_tuning_grid,
            metrics = metric_set(accuracy))
proc.time() - startCV

rf_bestTune <- rf_results %>% 
  select_best("accuracy")

rf_final_wf <- rf_wf %>% 
  finalize_workflow(rf_bestTune) %>% 
  fit(data=train)

startPred <- proc.time()
preds <- predict(rf_final_wf,
                 new_data=test,
                 type="class")

stopCluster(cluster)
proc.time() - startPred

output <- as.data.frame(cbind(as.integer(test$Id), as.character(preds$.pred_class)))
colnames(output) <- c("Id", "Cover_Type")

vroom_write(output, file="submissions/boosted10.csv",delim=',')
