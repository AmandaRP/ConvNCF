# This script contains starter code to define a BPR loss. 
# It is not finished and has not been tested.

# BPR triplet loss is the output (from https://github.com/nanxstats/deep-learning-recipes)
#loss <- list(embed_user, embed_item_positive, embed_item_negative) %>%
#  layer_lambda(loss_bpr_triplet, output_shape = c(1))

#model <- keras_model(
#  inputs = c(input_user, input_item_positive, input_item_negative),
#  outputs = loss
#)

# BPR loss:
#loss_bpr_triplet <- function(x) {
#  embed_user <- x[[1]]
#  embed_item_positive <- x[[2]]
#  embed_item_negative <- x[[3]]
#  
#  loss <- 1.0 - k_sigmoid(
#    k_sum(embed_user * embed_item_positive, axis = -1, keepdims = TRUE) -
#    k_sum(embed_user * embed_item_negative, axis = -1, keepdims = TRUE)
#  )
#}

# ---

#python pytorch version for convncf:
#class BPRLoss(nn.Module):

#  def __init__(self):
#  super(BPRLoss, self).__init__()
#  self.sigmoid = nn.Sigmoid()

#  def forward(self, pos_preds, neg_preds):
#  distance = pos_preds - neg_preds
#  loss = torch.sum(torch.log((1 + torch.exp(-distance))))

#  return loss

# ---

my_bpr <- function(x){
  pos_preds <- x[[1]] 
  neg_preds <- x[[2]]
  distance <- pos_preds - neg_preds
  loss <- 1 - k_sigmoid(k_sum(distance))
}
bpr_loss <- list(pos_preds, neg_preds) %>% layer_lambda(my_bpr, output_shape = c(1))

