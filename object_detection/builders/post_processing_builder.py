"""Builder function for post processing operations."""

import functools

import tensorflow.compat.v1 as tf
from object_detection.protos import post_processing_pb2
from object_detection.core import post_processing

def _build_non_max_suppressor(nms_config):
  """Builds non-max suppresson based on the nms config.

  Args:
    nms_config: post_processing_pb2.PostProcessing.BatchNonMaxSuppression proto.

  Returns:
    non_max_suppressor_fn: Callable non-max suppressor.

  Raises:
    ValueError: On incorrect iou_threshold or on incompatible values of
      max_total_detections and max_detections_per_class or on negative
      soft_nms_sigma.
  """
  if nms_config.iou_threshold < 0 or nms_config.iou_threshold > 1.0:
    raise ValueError('iou_threshold not in [0, 1.0].')
  if nms_config.max_detections_per_class > nms_config.max_total_detections:
    raise ValueError('max_detections_per_class should be no greater than '
                     'max_total_detections.')
  if nms_config.soft_nms_sigma < 0.0:
    raise ValueError('soft_nms_sigma should be non-negative.')
  if nms_config.use_combined_nms and nms_config.use_class_agnostic_nms:
    raise ValueError('combined_nms does not support class_agnostic_nms.')
  non_max_suppressor_fn = functools.partial(
      post_processing.batch_multiclass_non_max_suppression,
      score_thresh=nms_config.score_threshold,
      iou_thresh=nms_config.iou_threshold,
      max_size_per_class=nms_config.max_detections_per_class,
      max_total_size=nms_config.max_total_detections,
      use_static_shapes=nms_config.use_static_shapes,
      use_class_agnostic_nms=nms_config.use_class_agnostic_nms,
      max_classes_per_detection=nms_config.max_classes_per_detection,
      soft_nms_sigma=nms_config.soft_nms_sigma,
      use_partitioned_nms=nms_config.use_partitioned_nms,
      use_combined_nms=nms_config.use_combined_nms,
      change_coordinate_frame=nms_config.change_coordinate_frame,
      use_hard_nms=nms_config.use_hard_nms,
      use_cpu_nms=nms_config.use_cpu_nms)

  return non_max_suppressor_fn



def _score_converter_fn_with_logit_scale(tf_score_converter_fn, logit_scale):
  """Create a function to scale logits then apply a Tensorflow function."""
  def score_converter_fn(logits):
    scaled_logits = tf.multiply(logits, 1.0 / logit_scale, name='scale_logits')
    return tf_score_converter_fn(scaled_logits, name='convert_scores')
  score_converter_fn.__name__ = '%s_with_logit_scale' % (
      tf_score_converter_fn.__name__)
  return score_converter_fn



def _build_score_converter(score_converter_config, logit_scale):
  """Builds score converter based on the config.

  Builds one of [tf.identity, tf.sigmoid, tf.softmax] score converters based on
  the config.

  Args:
    score_converter_config: post_processing_pb2.PostProcessing.score_converter.
    logit_scale: temperature to use for SOFTMAX score_converter.

  Returns:
    Callable score converter op.

  Raises:
    ValueError: On unknown score converter.
  """
  if score_converter_config == post_processing_pb2.PostProcessing.IDENTITY:
    return _score_converter_fn_with_logit_scale(tf.identity, logit_scale)
  if score_converter_config == post_processing_pb2.PostProcessing.SIGMOID:
    return _score_converter_fn_with_logit_scale(tf.sigmoid, logit_scale)
  if score_converter_config == post_processing_pb2.PostProcessing.SOFTMAX:
    return _score_converter_fn_with_logit_scale(tf.nn.softmax, logit_scale)
  raise ValueError('Unknown score converter.')




def build(post_processing_config):
  """Builds callables for post-processing operations.

  Builds callables for non-max suppression, score conversion, and (optionally)
  calibration based on the configuration.

  Non-max suppression callable takes `boxes`, `scores`, and optionally
  `clip_window`, `parallel_iterations` `masks, and `scope` as inputs. It returns
  `nms_boxes`, `nms_scores`, `nms_classes` `nms_masks` and `num_detections`. See
  post_processing.batch_multiclass_non_max_suppression for the type and shape
  of these tensors.

  Score converter callable should be called with `input` tensor. The callable
  returns the output from one of 3 tf operations based on the configuration -
  tf.identity, tf.sigmoid or tf.nn.softmax. If a calibration config is provided,
  score_converter also applies calibration transformations, as defined in
  calibration_builder.py. See tensorflow documentation for argument and return
  value descriptions.

  Args:
    post_processing_config: post_processing.proto object containing the
      parameters for the post-processing operations.

  Returns:
    non_max_suppressor_fn: Callable for non-max suppression.
    score_converter_fn: Callable for score conversion.

  Raises:
    ValueError: if the post_processing_config is of incorrect type.
  """
  if not isinstance(post_processing_config, post_processing_pb2.PostProcessing):
    raise ValueError('post_processing_config not of type '
                     'post_processing_pb2.Postprocessing.')
  
  non_max_suppressor_fn = _build_non_max_suppressor(
      post_processing_config.batch_non_max_suppression)
  
  score_converter_fn = _build_score_converter(
      post_processing_config.score_converter,
      post_processing_config.logit_scale)
  

  return non_max_suppressor_fn, score_converter_fn
