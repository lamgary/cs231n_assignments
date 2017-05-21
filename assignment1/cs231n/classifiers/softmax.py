import numpy as np
from random import shuffle
from past.builtins import xrange

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
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores) # shift values so highest score is 0
    probabilities = np.exp(scores) / np.sum(np.exp(scores))
    correct_class_p = probabilities[y[i]]
    loss += - np.log(correct_class_p)
    probabilities[y[i]] -= 1 # subtract 1 for correct class
    dW += np.tile(X[i], [num_classes, 1]).T * probabilities
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  dW /= num_train
  dW += reg * W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

   Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0

  d = W.shape[0]
  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = X.dot(W) # (N, C)
  scores -= np.tile(np.max(scores, axis=1), [num_classes, 1]).T #(N, C)
  exp_scores = np.exp(scores)
  probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) #(N, C)

  loss = - np.sum(np.log(probabilities[np.arange(num_train),y]))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  probabilities[np.arange(num_train),y] -= 1 #subtract 1 for correct class
  dW = X.T.dot(probabilities)
  dW /= num_train
  dW += reg * W

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

