# This script contains evaluation functions

# Compute hit rate at k:
compute_hr <- function(test_pred, k){
  test_pred %>%
    group_by(user) %>%
    slice_max(order_by = pred, n = k) %>%
    summarize(hits = sum(label)) %>%
    summarize(hr = mean(hits)) 
  
}


# Compute ndcg:
# Note that the current implementation assumes one 1 for each user. If this is not the case, need to divide by an idcg.
compute_ndcg <- function(test_pred, k){
  test_pred %>% 
    group_by(user) %>% 
    slice_max(order_by = pred, n = k) %>%
    mutate(rank = rank(desc(pred), ties.method = "random")) %>% 
    mutate(dcg = (2^label-1)/log(rank+1, base = 2)) %>%
    summarize(ndcg_user = sum(dcg)) %>%
    ungroup() %>%
    summarize(ndcg = mean(ndcg_user))
}
