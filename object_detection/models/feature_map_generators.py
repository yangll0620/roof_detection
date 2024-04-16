import tensorflow.compat.v1 as tf
import collections

from object_detection.utils import shape_utils
from object_detection.utils import ops



ACTIVATION_BOUND = 6.0


def create_conv_block(
    use_depthwise, kernel_size, padding, stride, layer_name, conv_hyperparams,
    is_training, freeze_batchnorm, depth):
  """Create Keras layers for depthwise & non-depthwise convolutions.

  Args:
    use_depthwise: Whether to use depthwise separable conv instead of regular
      conv.
    kernel_size: A list of length 2: [kernel_height, kernel_width] of the
      filters. Can be an int if both values are the same.
    padding: One of 'VALID' or 'SAME'.
    stride: A list of length 2: [stride_height, stride_width], specifying the
      convolution stride. Can be an int if both strides are the same.
    layer_name: String. The name of the layer.
    conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
      containing hyperparameters for convolution ops.
    is_training: Indicates whether the feature generator is in training mode.
    freeze_batchnorm: Bool. Whether to freeze batch norm parameters during
      training or not. When training with a small batch size (e.g. 1), it is
      desirable to freeze batch norm update and use pretrained batch norm
      params.
    depth: Depth of output feature maps.

  Returns:
    A list of conv layers.
  """
  layers = []
  if use_depthwise:
    kwargs = conv_hyperparams.params()
    # Both the regularizer and initializer apply to the depthwise layer,
    # so we remap the kernel_* to depthwise_* here.
    kwargs['depthwise_regularizer'] = kwargs['kernel_regularizer']
    kwargs['depthwise_initializer'] = kwargs['kernel_initializer']
    layers.append(
        tf.keras.layers.SeparableConv2D(
            depth, [kernel_size, kernel_size],
            depth_multiplier=1,
            padding=padding,
            strides=stride,
            name=layer_name + '_depthwise_conv',
            **kwargs))
  else:
    layers.append(tf.keras.layers.Conv2D(
        depth,
        [kernel_size, kernel_size],
        padding=padding,
        strides=stride,
        name=layer_name + '_conv',
        **conv_hyperparams.params()))
  layers.append(
      conv_hyperparams.build_batch_norm(
          training=(is_training and not freeze_batchnorm),
          name=layer_name + '_batchnorm'))
  layers.append(
      conv_hyperparams.build_activation_layer(
          name=layer_name))
  return layers




class KerasFpnTopDownFeatureMaps(tf.keras.Model):
  """Generates Keras based `top-down` feature maps for Feature Pyramid Networks.

  See https://arxiv.org/abs/1612.03144 for details.
  """

  def __init__(self,
               num_levels,
               depth,
               is_training,
               conv_hyperparams,
               freeze_batchnorm,
               use_depthwise=False,
               use_explicit_padding=False,
               use_bounded_activations=False,
               use_native_resize_op=False,
               scope=None,
               name=None):
    """Constructor.

    Args:
      num_levels: the number of image features.
      depth: depth of output feature maps.
      is_training: Indicates whether the feature generator is in training mode.
      conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
        containing hyperparameters for convolution ops.
      freeze_batchnorm: Bool. Whether to freeze batch norm parameters during
        training or not. When training with a small batch size (e.g. 1), it is
        desirable to freeze batch norm update and use pretrained batch norm
        params.
      use_depthwise: whether to use depthwise separable conv instead of regular
        conv.
      use_explicit_padding: whether to use explicit padding.
      use_bounded_activations: Whether or not to clip activations to range
        [-ACTIVATION_BOUND, ACTIVATION_BOUND]. Bounded activations better lend
        themselves to quantized inference.
      use_native_resize_op: If True, uses tf.image.resize_nearest_neighbor op
        for the upsampling process instead of reshape and broadcasting
        implementation.
      scope: A scope name to wrap this op under.
      name: A string name scope to assign to the model. If 'None', Keras
        will auto-generate one from the class name.
    """
    super(KerasFpnTopDownFeatureMaps, self).__init__(name=name)

    self.scope = scope if scope else 'top_down'
    self.top_layers = []
    self.residual_blocks = []
    self.top_down_blocks = []
    self.reshape_blocks = []
    self.conv_layers = []

    padding = 'VALID' if use_explicit_padding else 'SAME'
    stride = 1
    kernel_size = 3
    def clip_by_value(features):
      return tf.clip_by_value(features, -ACTIVATION_BOUND, ACTIVATION_BOUND)

    # top layers
    self.top_layers.append(tf.keras.layers.Conv2D(
        depth, [1, 1], strides=stride, padding=padding,
        name='projection_%d' % num_levels,
        **conv_hyperparams.params(use_bias=True)))
    if use_bounded_activations:
      self.top_layers.append(tf.keras.layers.Lambda(
          clip_by_value, name='clip_by_value'))

    for level in reversed(list(range(num_levels - 1))):
      # to generate residual from image features
      residual_net = []
      # to preprocess top_down (the image feature map from last layer)
      top_down_net = []
      # to reshape top_down according to residual if necessary
      reshaped_residual = []
      # to apply convolution layers to feature map
      conv_net = []

      # residual block
      residual_net.append(tf.keras.layers.Conv2D(
          depth, [1, 1], padding=padding, strides=1,
          name='projection_%d' % (level + 1),
          **conv_hyperparams.params(use_bias=True)))
      if use_bounded_activations:
        residual_net.append(tf.keras.layers.Lambda(
            clip_by_value, name='clip_by_value'))

      # top-down block
      # TODO (b/128922690): clean-up of ops.nearest_neighbor_upsampling
      if use_native_resize_op:
        def resize_nearest_neighbor(image):
          image_shape = shape_utils.combined_static_and_dynamic_shape(image)
          return tf.image.resize_nearest_neighbor(
              image, [image_shape[1] * 2, image_shape[2] * 2])
        top_down_net.append(tf.keras.layers.Lambda(
            resize_nearest_neighbor, name='nearest_neighbor_upsampling'))
      else:
        def nearest_neighbor_upsampling(image):
          return ops.nearest_neighbor_upsampling(image, scale=2)
        top_down_net.append(tf.keras.layers.Lambda(
            nearest_neighbor_upsampling, name='nearest_neighbor_upsampling'))

      # reshape block
      if use_explicit_padding:
        def reshape(inputs):
          residual_shape = tf.shape(inputs[0])
          return inputs[1][:, :residual_shape[1], :residual_shape[2], :]
        reshaped_residual.append(
            tf.keras.layers.Lambda(reshape, name='reshape'))

      # down layers
      if use_bounded_activations:
        conv_net.append(tf.keras.layers.Lambda(
            clip_by_value, name='clip_by_value'))

      if use_explicit_padding:
        def fixed_padding(features, kernel_size=kernel_size):
          return ops.fixed_padding(features, kernel_size)
        conv_net.append(tf.keras.layers.Lambda(
            fixed_padding, name='fixed_padding'))

      layer_name = 'smoothing_%d' % (level + 1)
      conv_block = create_conv_block(
          use_depthwise, kernel_size, padding, stride, layer_name,
          conv_hyperparams, is_training, freeze_batchnorm, depth)
      conv_net.extend(conv_block)

      self.residual_blocks.append(residual_net)
      self.top_down_blocks.append(top_down_net)
      self.reshape_blocks.append(reshaped_residual)
      self.conv_layers.append(conv_net)

  def call(self, image_features):
    """Generate the multi-resolution feature maps.

    Executed when calling the `.__call__` method on input.

    Args:
      image_features: list of tuples of (tensor_name, image_feature_tensor).
        Spatial resolutions of succesive tensors must reduce exactly by a factor
        of 2.

    Returns:
      feature_maps: an OrderedDict mapping keys (feature map names) to
        tensors where each tensor has shape [batch, height_i, width_i, depth_i].
    """
    output_feature_maps_list = []
    output_feature_map_keys = []

    with tf.name_scope(self.scope):
      top_down = image_features[-1][1]
      for layer in self.top_layers:
        top_down = layer(top_down)
      output_feature_maps_list.append(top_down)
      output_feature_map_keys.append('top_down_%s' % image_features[-1][0])

      num_levels = len(image_features)
      for index, level in enumerate(reversed(list(range(num_levels - 1)))):
        residual = image_features[level][1]
        top_down = output_feature_maps_list[-1]
        for layer in self.residual_blocks[index]:
          residual = layer(residual)
        for layer in self.top_down_blocks[index]:
          top_down = layer(top_down)
        for layer in self.reshape_blocks[index]:
          top_down = layer([residual, top_down])
        top_down += residual
        for layer in self.conv_layers[index]:
          top_down = layer(top_down)
        output_feature_maps_list.append(top_down)
        output_feature_map_keys.append('top_down_%s' % image_features[level][0])
    return collections.OrderedDict(reversed(
        list(zip(output_feature_map_keys, output_feature_maps_list))))

