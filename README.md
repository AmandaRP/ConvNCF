Convolutional Neural Collaborative Filtering
================

This is an implementation of Convolutional Neural Collaborative
Filtering (ConvNCF) using [R
Keras](https://keras.rstudio.com/index.html). The model is described in
the following paper and implemented by its authors using Tensorflow (see
[this GitHub repo](https://github.com/duxy-me/ConvNCF)).

> Du, Xiaoyu, Xiangnan He, Fajie Yuan, Jinhui Tang, Zhiguang Qin, and
> Tat-Seng Chua. [Modeling Embedding Dimension Correlations via
> Convolutional Neural Collaborative
> Filtering.](https://dl.acm.org/doi/abs/10.1145/3357154) ACM
> Transactions on Information Systems (TOIS) 37, no. 4 (2019): 1-22.

## Code

The model implementation is in `ConvNCF.R`. This implementation
currently assumes binary feedback (1 = user liked movie, 0 otherwise).
An example using Yelp ratings is provided in `yelp.R`.

## Disclaimers

The bulk of the code in this repo has been completed, however, before it
is used for a recommender system application the following items need to
be completed:

  - Perform grid search to select regularization hyperparameters:
    `lambda_1`, `lambda_2`, `lambda_3`, and `lambda_4`
  - Pretrain the weights of the model using traditional (non deep
    learning) recsys methods (see section 5.4 of the paper and authors’
    [python
    code](https://github.com/duxy-me/ConvNCF/blob/master/MF_BPR.py)).
  - Use BPR objective function (starter code in `bpr_loss.R`).
  - Select negative examples on the fly for each epoch. See example R
    code
    [here](https://github.com/nanxstats/deep-learning-recipes/blob/a0aa18c23d0bb458d8bae084a46db47279e610f9/triplet-loss-keras/triplet-loss-bpr-keras.R#L118).
