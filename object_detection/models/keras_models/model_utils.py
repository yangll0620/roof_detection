import tensorflow.compat.v1 as tf


def input_layer(shape, placeholder_with_default):
  if tf.executing_eagerly():
    return tf.keras.layers.Input(shape=shape)
  else:
    return tf.keras.layers.Input(tensor=placeholder_with_default)