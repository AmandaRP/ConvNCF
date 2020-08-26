

# Load libraries ----------------------------------------------------------

library(tidyverse)
library(magrittr)
library(pins)
library(jsonlite)


# Read data ---------------------------------------------------------------

base_name <- "https://github.com/duxy-me/ConvNCF/raw/master/Data/yelp."
test_neg <- read_tsv(gzfile(pins::pin(str_c(base_name, "test.negative.gz"))), col_names = FALSE)
test_rating <- read_tsv(gzfile(pins::pin(str_c(base_name, "test.rating.gz"))), col_names = c("userID","itemID", "rating", "timestamp"))
train_rating <- read_tsv(gzfile(pins::pin(str_c(base_name, "train.rating.gz"))), col_names = c("userID","itemID", "rating", "timestamp"))


