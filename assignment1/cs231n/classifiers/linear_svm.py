import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    classes_meeting_margin = 0
    for j in xrange(num_classes):
      margin = scores[j] - correct_class_score + 1  # note delta = 1
      if j == y[i]:
        continue
      if margin > 0:
        loss += margin
        classes_meeting_margin += 1
        dW[:, j] += X[i]
    dW[:, y[i]] -= (classes_meeting_margin * X[i])

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  dW /= num_train
  dW += reg * 2 * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  # X: num_train x num_features
  # W: num_features x num_classes
  num_train = X.shape[0]
  num_classes = W.shape[1]

  dW = np.zeros(W.shape)

  scores = X.dot(W)  # num_train x num_classes
  correct_class_score = scores[np.arange(num_train), y]  # num_train x 1
  margins = np.maximum(0, scores.T - correct_class_score + 1).T # num_train x num_classes
  margins[np.arange(num_train),y] = 0 # clear out margins of correct label

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss = np.sum(margins)
  loss /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  margins = (margins > 0) * np.ones(margins.shape) # clear out negative margins; num_train x
  number_classes_meeting_margin = np.sum(margins, axis=1) #num_train x 1
  margins[np.arange(num_train), y] = -number_classes_meeting_margin

  # update losses for correct class
  dW += X.T.dot(margins) #num_features x num_classes
  dW /= num_train
  dW += reg * 2 * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
