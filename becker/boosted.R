library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(ranger)
library(parallel)
library(bonsai)
library(lightgbm)

train <- vroom("data/train.csv")
test <- vroom("data/test.csv")

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
  step_zv() %>%
  prep()



boosted_model <- boost_tree(tree_depth=tune(), 
                            trees=1000, 
                            learn_rate=tune()) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

boost_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boosted_model)

tuning_grid <- grid_regular(tree_depth(), 
                            learn_rate(), 
                            levels = 10)

folds <- vfold_cv(train, v = 5, repeats = 1)


cluster <- makePSOCKcluster(4)
doParallel::registerDoParallel(cluster)
startCV <- proc.time()

boosted_cv_results <- boost_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))
proc.time() - startCV

boosted_best_tune <- boosted_cv_results %>% 
  select_best("accuracy")

final_boosted_wf <- boost_wf %>% 
  finalize_workflow(boosted_best_tune) %>% 
  fit(data=train)



startPred <- proc.time()
preds <- predict(fitted_wf, new_data=test) 

stopCluster(cluster)
proc.time() - startPred

output <- as.data.frame(cbind(as.integer(test$Id), as.character(preds$.pred_class)))
colnames(output) <- c("Id", "Cover_Type")

vroom_write(output, file="submissions/boosted10.csv",delim=',')
