import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  num_train = X.shape[0];
  num_classes = W.shape[1];

  for i in range(num_train):
    sum_exp = 0.0
    for j in range(num_classes):
      e_f = np.exp(X[i, :].dot(W)[j])
      sum_exp += e_f
    for j in range(num_classes): 
      e_f = np.exp(X[i, :].dot(W)[j])
      if j == y[i]:
        dW[:, y[i]] += (-1.0 + e_f / sum_exp) * (X[i, :])
      else:
        dW[:, j] += e_f / sum_exp * X[i, :]
    loss += np.log(sum_exp) - X[i, :].dot(W)[y[i]]

  loss /= num_train
  dW /= num_train

  # add regularization
  loss += reg * np.sum(W * W)
  dW += 2.0 * reg * W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0];
  num_classes = W.shape[1];

  scores = np.matmul(X, W)
  exp_scores = np.exp(scores)
  sums = exp_scores.sum(axis=1)

  rows = range(X.shape[0])
  L = -scores[rows, y[rows]] + np.log(sums)
  loss = L.sum()
  loss /= num_train 

  exp_scores /= sums[:, np.newaxis]
  exp_scores[rows, y[rows]] -= 1.0
  dW = np.matmul(X.T, exp_scores)
  dW /= num_train

  loss += reg * np.sum(W * W)
  dW += 2.0 * reg * W

  return loss, dW














