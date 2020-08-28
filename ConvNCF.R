# This script loads the keras library and creates a function that can be used to define the ConvNCF model

library(keras)

build_convNCF <- function(K=64, lambda_1 = 10^-6, lambda_2 = 10^-6, lambda_3 = 10^-1, lambda_4 = 10^-1){
  
  user_input <- layer_input(shape=1, name = "user_input") 
  item_input <- layer_input(shape=1, name = "item_input")
  
  user_embedding <- user_input %>%  
    layer_embedding(input_dim = num_users, # "dictionary" size
                    output_dim = K,
                    embeddings_regularizer = regularizer_l2(lambda_1), 
                    input_length = 1,  # the length of the sequence that is being fed in (one integer)
                    name = "user_embedding") 
  
  item_embedding <- item_input %>%  
    layer_embedding(input_dim = num_items, # "dictionary" size
                    output_dim = K,
                    embeddings_regularizer = regularizer_l2(lambda_2), 
                    input_length = 1,  
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
    loss = "binary_crossentropy", #TODO: Replace with BPR loss
    metrics = c("accuracy")
  )
  
}


