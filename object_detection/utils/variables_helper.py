"""Helper functions for manipulating collections of variables during training.
"""

import tensorflow.compat.v1 as tf

def get_global_variables_safely():
  """If not executing eagerly, returns tf.global_variables().

  Raises a ValueError if eager execution is enabled,
  because the variables are not tracked when executing eagerly.

  If executing eagerly, use a Keras model's .variables property instead.

  Returns:
    The result of tf.global_variables()
  """
  with tf.init_scope():
    if tf.executing_eagerly():
      raise ValueError("Global variables collection is not tracked when "
                       "executing eagerly. Use a Keras model's `.variables` "
                       "attribute instead.")
  return tf.global_variables()
