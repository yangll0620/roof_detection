"""Preprocess images and bounding boxes for detection.

We perform two sets of operations in preprocessing stage:
(a) operations that are applied to both training and testing data,
(b) operations that are applied only to training data for the purpose of
    data augmentation.

A preprocessing function receives a set of inputs,
e.g. an image and bounding boxes,
performs an operation on them, and returns them.
Some examples are: randomly cropping the image, randomly mirroring the image,
                   randomly changing the brightness, contrast, hue and
                   randomly jittering the bounding boxes.

The preprocess function receives a tensor_dict which is a dictionary that maps
different field names to their tensors. For example,
tensor_dict[fields.InputDataFields.image] holds the image tensor.
The image is a rank 4 tensor: [1, height, width, channels] with
dtype=tf.float32. The groundtruth_boxes is a rank 2 tensor: [N, 4] where
in each row there is a box with [ymin xmin ymax xmax].
Boxes are in normalized coordinates meaning
their coordinate values range in [0, 1]

To preprocess multiple images with the same operations in cases where
nondeterministic operations are used, a preprocessor_cache.PreprocessorCache
object can be passed into the preprocess function or individual operations.
All nondeterministic operations except random_jitter_boxes support caching.
E.g.
Let tensor_dict{1,2,3,4,5} be copies of the same inputs.
Let preprocess_options contain nondeterministic operation(s) excluding
random_jitter_boxes.

cache1 = preprocessor_cache.PreprocessorCache()
cache2 = preprocessor_cache.PreprocessorCache()
a = preprocess(tensor_dict1, preprocess_options, preprocess_vars_cache=cache1)
b = preprocess(tensor_dict2, preprocess_options, preprocess_vars_cache=cache1)
c = preprocess(tensor_dict3, preprocess_options, preprocess_vars_cache=cache2)
d = preprocess(tensor_dict4, preprocess_options, preprocess_vars_cache=cache2)
e = preprocess(tensor_dict5, preprocess_options)

Then correspondings tensors of object pairs (a,b) and (c,d)
are guaranteed to be equal element-wise, but the equality of any other object
pair cannot be determined.

Important Note: In tensor_dict, images is a rank 4 tensor, but preprocessing
functions receive a rank 3 tensor for processing the image. Thus, inside the
preprocess function we squeeze the image to become a rank 3 tensor and then
we pass it to the functions. At the end of the preprocess we expand the image
back to rank 4.
"""


import tensorflow.compat.v1 as tf


from object_detection.utils import shape_utils

def resize_to_range(image,
                    masks=None,
                    min_dimension=None,
                    max_dimension=None,
                    method=tf.image.ResizeMethod.BILINEAR,
                    align_corners=False,
                    pad_to_max_dimension=False,
                    per_channel_pad_value=(0, 0, 0)):
  """Resizes an image so its dimensions are within the provided value.

  The output size can be described by two cases:
  1. If the image can be rescaled so its minimum dimension is equal to the
     provided value without the other dimension exceeding max_dimension,
     then do so.
  2. Otherwise, resize so the largest dimension is equal to max_dimension.

  Args:
    image: A 3D tensor of shape [height, width, channels]
    masks: (optional) rank 3 float32 tensor with shape
           [num_instances, height, width] containing instance masks.
    min_dimension: (optional) (scalar) desired size of the smaller image
                   dimension.
    max_dimension: (optional) (scalar) maximum allowed size
                   of the larger image dimension.
    method: (optional) interpolation method used in resizing. Defaults to
            BILINEAR.
    align_corners: bool. If true, exactly align all 4 corners of the input
                   and output. Defaults to False.
    pad_to_max_dimension: Whether to resize the image and pad it with zeros
      so the resulting image is of the spatial size
      [max_dimension, max_dimension]. If masks are included they are padded
      similarly.
    per_channel_pad_value: A tuple of per-channel scalar value to use for
      padding. By default pads zeros.

  Returns:
    Note that the position of the resized_image_shape changes based on whether
    masks are present.
    resized_image: A 3D tensor of shape [new_height, new_width, channels],
      where the image has been resized (with bilinear interpolation) so that
      min(new_height, new_width) == min_dimension or
      max(new_height, new_width) == max_dimension.
    resized_masks: If masks is not None, also outputs masks. A 3D tensor of
      shape [num_instances, new_height, new_width].
    resized_image_shape: A 1D tensor of shape [3] containing shape of the
      resized image.

  Raises:
    ValueError: if the image is not a 3D tensor.
  """
  if len(image.get_shape()) != 3:
    raise ValueError('Image should be 3D tensor')

  def _resize_landscape_image(image):
    # resize a landscape image
    return tf.image.resize_images(
        image, tf.stack([min_dimension, max_dimension]), method=method,
        align_corners=align_corners, preserve_aspect_ratio=True)

  def _resize_portrait_image(image):
    # resize a portrait image
    return tf.image.resize_images(
        image, tf.stack([max_dimension, min_dimension]), method=method,
        align_corners=align_corners, preserve_aspect_ratio=True)

  with tf.name_scope('ResizeToRange', values=[image, min_dimension]):
    if image.get_shape().is_fully_defined():
      if image.get_shape()[0] < image.get_shape()[1]:
        new_image = _resize_landscape_image(image)
      else:
        new_image = _resize_portrait_image(image)
      new_size = tf.constant(new_image.get_shape().as_list())
    else:
      new_image = tf.cond(
          tf.less(tf.shape(image)[0], tf.shape(image)[1]),
          lambda: _resize_landscape_image(image),
          lambda: _resize_portrait_image(image))
      new_size = tf.shape(new_image)

    if pad_to_max_dimension:
      channels = tf.unstack(new_image, axis=2)
      if len(channels) != len(per_channel_pad_value):
        raise ValueError('Number of channels must be equal to the length of '
                         'per-channel pad value.')
      new_image = tf.stack(
          [
              tf.pad(  # pylint: disable=g-complex-comprehension
                  channels[i], [[0, max_dimension - new_size[0]],
                                [0, max_dimension - new_size[1]]],
                  constant_values=per_channel_pad_value[i])
              for i in range(len(channels))
          ],
          axis=2)
      new_image.set_shape([max_dimension, max_dimension, len(channels)])

    result = [new_image]
    if masks is not None:
      new_masks = tf.expand_dims(masks, 3)
      new_masks = tf.image.resize_images(
          new_masks,
          new_size[:-1],
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
          align_corners=align_corners)
      if pad_to_max_dimension:
        new_masks = tf.image.pad_to_bounding_box(
            new_masks, 0, 0, max_dimension, max_dimension)
      new_masks = tf.squeeze(new_masks, 3)
      result.append(new_masks)

    result.append(new_size)
    return result





# TODO(alirezafathi): Investigate if instead the function should return None if
# masks is None.
# pylint: disable=g-doc-return-or-yield
def resize_image(image,
                 masks=None,
                 new_height=600,
                 new_width=1024,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False):
  """Resizes images to the given height and width.

  Args:
    image: A 3D tensor of shape [height, width, channels]
    masks: (optional) rank 3 float32 tensor with shape
           [num_instances, height, width] containing instance masks.
    new_height: (optional) (scalar) desired height of the image.
    new_width: (optional) (scalar) desired width of the image.
    method: (optional) interpolation method used in resizing. Defaults to
            BILINEAR.
    align_corners: bool. If true, exactly align all 4 corners of the input
                   and output. Defaults to False.

  Returns:
    Note that the position of the resized_image_shape changes based on whether
    masks are present.
    resized_image: A tensor of size [new_height, new_width, channels].
    resized_masks: If masks is not None, also outputs masks. A 3D tensor of
      shape [num_instances, new_height, new_width]
    resized_image_shape: A 1D tensor of shape [3] containing the shape of the
      resized image.
  """
  with tf.name_scope(
      'ResizeImage',
      values=[image, new_height, new_width, method, align_corners]):
    new_image = tf.image.resize_images(
        image, tf.stack([new_height, new_width]),
        method=method,
        align_corners=align_corners)
    image_shape = shape_utils.combined_static_and_dynamic_shape(image)
    result = [new_image]
    if masks is not None:
      num_instances = tf.shape(masks)[0]
      new_size = tf.stack([new_height, new_width])
      def resize_masks_branch():
        new_masks = tf.expand_dims(masks, 3)
        new_masks = tf.image.resize_nearest_neighbor(
            new_masks, new_size, align_corners=align_corners)
        new_masks = tf.squeeze(new_masks, axis=3)
        return new_masks

      def reshape_masks_branch():
        # The shape function will be computed for both branches of the
        # condition, regardless of which branch is actually taken. Make sure
        # that we don't trigger an assertion in the shape function when trying
        # to reshape a non empty tensor into an empty one.
        new_masks = tf.reshape(masks, [-1, new_size[0], new_size[1]])
        return new_masks

      masks = tf.cond(num_instances > 0, resize_masks_branch,
                      reshape_masks_branch)
      result.append(masks)

    result.append(tf.stack([new_height, new_width, image_shape[2]]))
    return result
  


def _get_image_info(image):
  """Returns the height, width and number of channels in the image."""
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]
  num_channels = tf.shape(image)[2]
  return (image_height, image_width, num_channels)



def resize_to_max_dimension(image, masks=None, max_dimension=600,
                            method=tf.image.ResizeMethod.BILINEAR):
  """Resizes image and masks given the max size maintaining the aspect ratio.

  If one of the image dimensions is greater than max_dimension, it will scale
  the image such that its largest dimension is equal to max_dimension.
  Otherwise, will keep the image size as is.

  Args:
    image: a tensor of size [height, width, channels].
    masks: (optional) a tensors of size [num_instances, height, width].
    max_dimension: maximum image dimension.
    method: (optional) interpolation method used in resizing. Defaults to
    BILINEAR.

  Returns:
    An array containing resized_image, resized_masks, and resized_image_shape.
    Note that the position of the resized_image_shape changes based on whether
    masks are present.
    resized_image: A tensor of size [new_height, new_width, channels].
    resized_masks: If masks is not None, also outputs masks. A 3D tensor of
      shape [num_instances, new_height, new_width]
    resized_image_shape: A 1D tensor of shape [3] containing the shape of the
      resized image.

  Raises:
    ValueError: if the image is not a 3D tensor.
  """
  if len(image.get_shape()) != 3:
    raise ValueError('Image should be 3D tensor')

  with tf.name_scope('ResizeGivenMaxDimension', values=[image, max_dimension]):
    (image_height, image_width, num_channels) = _get_image_info(image)
    max_image_dimension = tf.maximum(image_height, image_width)
    max_target_dimension = tf.minimum(max_image_dimension, max_dimension)
    target_ratio = tf.cast(max_target_dimension, dtype=tf.float32) / tf.cast(
        max_image_dimension, dtype=tf.float32)
    target_height = tf.cast(
        tf.cast(image_height, dtype=tf.float32) * target_ratio, dtype=tf.int32)
    target_width = tf.cast(
        tf.cast(image_width, dtype=tf.float32) * target_ratio, dtype=tf.int32)
    image = tf.image.resize_images(
        tf.expand_dims(image, axis=0), size=[target_height, target_width],
        method=method,
        align_corners=True)
    result = [tf.squeeze(image, axis=0)]

    if masks is not None:
      masks = tf.image.resize_nearest_neighbor(
          tf.expand_dims(masks, axis=3),
          size=[target_height, target_width],
          align_corners=True)
      result.append(tf.squeeze(masks, axis=3))

    result.append(tf.stack([target_height, target_width, num_channels]))
    return result
