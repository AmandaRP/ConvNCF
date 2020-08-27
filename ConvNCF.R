

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
neg_pos_ratio_train <- 4 # Ratio of negative training samples to positive to select for training. NCF authors used 4.

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

library(keras)

build_convNCF <- function(K=64, lambda_1 = 10^-6, lambda_2 = 10^-6, lambda_3 = 10^-1, lambda_4 = 10^-1){
  
  user_input <- layer_input(shape=1, name = "user_input") 
  item_input <- layer_input(shape=1, name = "item_input")
  
  user_embedding <- user_input %>%  
    layer_embedding(input_dim = num_users, # "dictionary" size
                    output_dim = K,
                    #embeddings_initializer = initializer_random_normal(0, sigma), # Use N(0,sigma) initialization  
                    embeddings_regularizer = regularizer_l2(lambda_1), 
                    input_length = 1,  # the length of the sequence that is being fed in (one integer)
                    name = "user_embedding") 
  
  item_embedding <- item_input %>%  
    layer_embedding(input_dim = num_items, # "dictionary" size
                    output_dim = K,
                    #embeddings_initializer = initializer_random_normal(0, sigma), # Use N(0,sigma) initialization  
                    embeddings_regularizer = regularizer_l2(lambda_2), 
                    input_length = 1,  # the length of the sequence that is being fed in (one integer)
                    name = "item_embedding") 
  
  outer_product <- layer_dot(list(k_permute_dimensions(user_embedding, c(1, 3, 2)), 
                                  item_embedding), 
                             axes=c(2,1)) %>%    #TODO: Why is axes not c(3,2 here)? Not 1-based???
    layer_reshape(c(64,64,1))
    
  
  output <-  
    outer_product %>% 
    layer_conv_2d(filters = 32, kernel_size = c(2,2), strides = 2, activation = "relu", kernel_regularizer = regularizer_l2(lambda_3)) %>%
    layer_conv_2d(filters = 32, kernel_size = c(2,2), strides = 2, activation = "relu", kernel_regularizer = regularizer_l2(lambda_3)) %>%
    layer_conv_2d(filters = 32, kernel_size = c(2,2), strides = 2, activation = "relu", kernel_regularizer = regularizer_l2(lambda_3)) %>%
    layer_conv_2d(filters = 32, kernel_size = c(2,2), strides = 2, activation = "relu", kernel_regularizer = regularizer_l2(lambda_3)) %>%
    layer_conv_2d(filters = 32, kernel_size = c(2,2), strides = 2, activation = "relu", kernel_regularizer = regularizer_l2(lambda_3)) %>%
    layer_conv_2d(filters = 32, kernel_size = c(2,2), strides = 2, activation = "relu", kernel_regularizer = regularizer_l2(lambda_3)) %>% #TODO: All layers have a regularizer? What type of regularizer? kernel, bias, or activity?
    layer_dense(units = 1, activation = "linear", kernel_regularizer = regularizer_l2(lambda_4), use_bias = FALSE, name = "output")
  
  model <- keras_model(list(user_input, item_input), output)

  # Compile model 
  
  model %>% compile(
    optimizer = "adagrad",
    loss = "binary_crossentropy", 
    metrics = c("accuracy")
  )
  
  summary(model)
  
}



