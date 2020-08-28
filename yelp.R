# This script builds a ConvNCF model for yelp reviews.

# Load libraries ----------------------------------------------------------

library(tidyverse)
library(magrittr)
library(pins)

# Read data ---------------------------------------------------------------

base_name <- "https://github.com/duxy-me/ConvNCF/raw/master/Data/yelp."
test_negative <- read_tsv(gzfile(pins::pin(str_c(base_name, "test.negative.gz"))), col_names = FALSE)
test_rating <- read_tsv(gzfile(pins::pin(str_c(base_name, "test.rating.gz"))), col_names = c("user","item", "rating", "timestamp"))
train_rating <- read_tsv(gzfile(pins::pin(str_c(base_name, "train.rating.gz"))), col_names = c("user","item", "rating", "timestamp"))

# Variable definitions ----------------------------------------------------

num_users <- max(train_rating$user) + 1
num_items <- max(train_rating$item) + 1
neg_pos_ratio_train <- 1 # Ratio of negative training samples to positive to select for training. 

# Wrangle data ------------------------------------------------------------

# Test set
test <- test_negative %>% 
  tidyr::extract(X1, into = c("user", "pos_item"), "([[:alnum:]]+),([[:alnum:]]+)", convert = TRUE) %>%
  pivot_longer(cols = pos_item:X1000, names_to = "label", values_to = "item") %>%
  mutate(label = as.integer(!str_detect(label,"X")))

# Get number of training ratings per user (will need for sampling negatives)
num_ratings_per_user <- train_rating %>% group_by(user) %>% count()

# Define negatives (those movies that were not rated for each user) to use for training 
train_negative <- 
  data.frame(user = rep(0:(num_users-1), each=num_items), 
             item = 0:(num_items-1)) %>% # start by listing all user/item pairs 
  anti_join(train_rating) %>% # remove user/item pairs that are in positive training set 
  anti_join(test) %>%       # remove user/item pairs that are in test set
  group_by(user) %>%  # Remaining operations used for sampling some of the negatives based on chosen negative to positive ratio
  nest() %>%
  inner_join(num_ratings_per_user) %>%
  mutate(subsamp = map2(data, n*neg_pos_ratio_train, ~slice_sample(.x, n=.y))) %>% 
  select(user, subsamp) %>%
  unnest(cols = c(subsamp))
#TODO: Paper randomly generates negative samples on the fly. See page 12

# Define validation data by picking the most recent rating for each user from training
validation <- train_rating %>% 
  group_by(user) %>% 
  slice_max(timestamp) %>% 
  slice_sample(1) %>% #some user/item pairs have same timestamp, so randomly pick one
  select(user, item) %>%
  add_column(label = 1)  # Only positive class was sampled for validation. See section 4.1 of NCF paper.

# Define training as data not used for validation
train <- anti_join(train_rating, validation) %>% 
  select(user, item) %>%
  bind_rows("pos" = ., "neg" = train_negative, .id = "label") %>%
  mutate(label = as.integer(str_detect(label,"pos")))


# Build model -------------------------------------------------------------

model <- build_convNCF()
summary(model)

# Train model -------------------------------------------------------------

# First define callbacks to stop model early when validation loss increases and to save best model
callback_list <- list(
  callback_early_stopping(patience = 2),
  callback_model_checkpoint(filepath = "model.h5", monitor = "val_loss", save_best_only = TRUE)
)

# Train model
history <- 
  model %>% 
  fit(
    x = list(user_input = as.array(train$user), 
             item_input = as.array(train$item)),
    y = as.array(train$label),
    epochs = 10,
    batch_size = 4096, 
    validation_data = list(list(user_input = as.array(validation$user), 
                                item_input = as.array(validation$item)), 
                           as.array(validation$label)),
    shuffle = TRUE, 
    callbacks = callback_list
  ) 

# Load best model:
model <- load_model_hdf5("my_model.h5")

# Evaluate results --------------------------------------------------------

history
#plot(history)

# Evaluate returns same metrics that were defined in the compile (accuracy in this case)
(results <- model %>% evaluate(list(test$user, test$item), test$label))

# Get predictions for test set:
test_pred <- model %>% 
  predict(x = list(test$user, test$item)) %>%
  bind_cols(pred = ., test)

# Compute hit rate and ndcg
source("evaluation.R")
compute_hr(test_pred, 10)
compute_ndcg(test_pred, 10)




