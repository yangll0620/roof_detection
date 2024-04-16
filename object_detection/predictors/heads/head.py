"""Base head class.

All the different kinds of prediction heads in different models will inherit
from this class. What is in common between all head classes is that they have a
`predict` function that receives `features` as its first argument.

How to add a new prediction head to an existing meta architecture?
For example, how can we add a `3d shape` prediction head to Mask RCNN?

We have to take the following steps to add a new prediction head to an
existing meta arch:
(a) Add a class for predicting the head. This class should inherit from the
`Head` class below and have a `predict` function that receives the features
and predicts the output. The output is always a tf.float32 tensor.
(b) Add the head to the meta architecture. For example in case of Mask RCNN,
go to box_predictor_builder and put in the logic for adding the new head to the
Mask RCNN box predictor.
(c) Add the logic for computing the loss for the new head.
(d) Add the necessary metrics for the new head.
(e) (optional) Add visualization for the new head.
"""


from abc import abstractmethod

import tensorflow.compat.v1 as tf


class KerasHead(tf.keras.layers.Layer):
  """Keras head base class."""

  def call(self, features):
    """The Keras model call will delegate to the `_predict` method."""
    return self._predict(features)

  @abstractmethod
  def _predict(self, features):
    """Returns the head's predictions.

    Args:
      features: A float tensor of features.

    Returns:
      A tf.float32 tensor.
    """
    pass
