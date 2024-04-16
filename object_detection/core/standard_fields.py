"""Contains classes specifying naming conventions used for object detection.


Specifies:
  InputDataFields: standard fields used by reader/preprocessor/batcher.
  DetectionResultFields: standard fields returned by object detector.
  BoxListFields: standard field used by BoxList
  TfExampleFields: standard fields for tf-example data format (go/tf-example).
"""

class BoxListFields(object):
  """Naming conventions for BoxLists.

  Attributes:
    boxes: bounding box coordinates.
    classes: classes per bounding box.
    scores: scores per bounding box.
    weights: sample weights per bounding box.
    objectness: objectness score per bounding box.
    masks: masks per bounding box.
    mask_weights: mask weights for each bounding box.
    boundaries: boundaries per bounding box.
    keypoints: keypoints per bounding box.
    keypoint_visibilities: keypoint visibilities per bounding box.
    keypoint_heatmaps: keypoint heatmaps per bounding box.
    keypoint_depths: keypoint depths per bounding box.
    keypoint_depth_weights: keypoint depth weights per bounding box.
    densepose_num_points: number of DensePose points per bounding box.
    densepose_part_ids: DensePose part ids per bounding box.
    densepose_surface_coords: DensePose surface coordinates per bounding box.
    is_crowd: is_crowd annotation per bounding box.
    temporal_offsets: temporal center offsets per bounding box.
    track_match_flags: match flags per bounding box.
  """
  boxes = 'boxes'
  classes = 'classes'
  scores = 'scores'
  weights = 'weights'
  confidences = 'confidences'
  objectness = 'objectness'
  masks = 'masks'
  mask_weights = 'mask_weights'
  boundaries = 'boundaries'
  keypoints = 'keypoints'
  keypoint_visibilities = 'keypoint_visibilities'
  keypoint_heatmaps = 'keypoint_heatmaps'
  keypoint_depths = 'keypoint_depths'
  keypoint_depth_weights = 'keypoint_depth_weights'
  densepose_num_points = 'densepose_num_points'
  densepose_part_ids = 'densepose_part_ids'
  densepose_surface_coords = 'densepose_surface_coords'
  is_crowd = 'is_crowd'
  group_of = 'group_of'
  track_ids = 'track_ids'
  temporal_offsets = 'temporal_offsets'
  track_match_flags = 'track_match_flags'


class PredictionFields(object):
  """Naming conventions for standardized prediction outputs.

  Attributes:
    feature_maps: List of feature maps for prediction.
    anchors: Generated anchors.
    raw_detection_boxes: Decoded detection boxes without NMS.
    raw_detection_feature_map_indices: Feature map indices from which each raw
      detection box was produced.
  """
  feature_maps = 'feature_maps'
  anchors = 'anchors'
  raw_detection_boxes = 'raw_detection_boxes'
  raw_detection_feature_map_indices = 'raw_detection_feature_map_indices'


class DetectionResultFields(object):
  """Naming conventions for storing the output of the detector.

  Attributes:
    source_id: source of the original image.
    key: unique key corresponding to image.
    detection_boxes: coordinates of the detection boxes in the image.
    detection_scores: detection scores for the detection boxes in the image.
    detection_multiclass_scores: class score distribution (including background)
      for detection boxes in the image including background class.
    detection_classes: detection-level class labels.
    detection_masks: contains a segmentation mask for each detection box.
    detection_surface_coords: contains DensePose surface coordinates for each
      box.
    detection_boundaries: contains an object boundary for each detection box.
    detection_keypoints: contains detection keypoints for each detection box.
    detection_keypoint_scores: contains detection keypoint scores.
    detection_keypoint_depths: contains detection keypoint depths.
    num_detections: number of detections in the batch.
    raw_detection_boxes: contains decoded detection boxes without Non-Max
      suppression.
    raw_detection_scores: contains class score logits for raw detection boxes.
    detection_anchor_indices: The anchor indices of the detections after NMS.
    detection_features: contains extracted features for each detected box
      after NMS.
  """

  source_id = 'source_id'
  key = 'key'
  detection_boxes = 'detection_boxes'
  detection_scores = 'detection_scores'
  detection_multiclass_scores = 'detection_multiclass_scores'
  detection_features = 'detection_features'
  detection_classes = 'detection_classes'
  detection_masks = 'detection_masks'
  detection_surface_coords = 'detection_surface_coords'
  detection_boundaries = 'detection_boundaries'
  detection_keypoints = 'detection_keypoints'
  detection_keypoint_scores = 'detection_keypoint_scores'
  detection_keypoint_depths = 'detection_keypoint_depths'
  detection_embeddings = 'detection_embeddings'
  detection_offsets = 'detection_temporal_offsets'
  num_detections = 'num_detections'
  raw_detection_boxes = 'raw_detection_boxes'
  raw_detection_scores = 'raw_detection_scores'
  detection_anchor_indices = 'detection_anchor_indices'


class InputDataFields(object):
  """Names for the input tensors.

  Holds the standard data field names to use for identifying input tensors. This
  should be used by the decoder to identify keys for the returned tensor_dict
  containing input tensors. And it should be used by the model to identify the
  tensors it needs.

  Attributes:
    image: image.
    image_additional_channels: additional channels.
    original_image: image in the original input size.
    original_image_spatial_shape: image in the original input size.
    key: unique key corresponding to image.
    source_id: source of the original image.
    filename: original filename of the dataset (without common path).
    groundtruth_image_classes: image-level class labels.
    groundtruth_image_confidences: image-level class confidences.
    groundtruth_labeled_classes: image-level annotation that indicates the
      classes for which an image has been labeled.
    groundtruth_boxes: coordinates of the ground truth boxes in the image.
    groundtruth_classes: box-level class labels.
    groundtruth_track_ids: box-level track ID labels.
    groundtruth_temporal_offset: box-level temporal offsets, i.e.,
      movement of the box center in adjacent frames.
    groundtruth_track_match_flags: box-level flags indicating if objects
      exist in the previous frame.
    groundtruth_confidences: box-level class confidences. The shape should be
      the same as the shape of groundtruth_classes.
    groundtruth_label_types: box-level label types (e.g. explicit negative).
    groundtruth_is_crowd: [DEPRECATED, use groundtruth_group_of instead]
      is the groundtruth a single object or a crowd.
    groundtruth_area: area of a groundtruth segment.
    groundtruth_difficult: is a `difficult` object
    groundtruth_group_of: is a `group_of` objects, e.g. multiple objects of the
      same class, forming a connected group, where instances are heavily
      occluding each other.
    proposal_boxes: coordinates of object proposal boxes.
    proposal_objectness: objectness score of each proposal.
    groundtruth_instance_masks: ground truth instance masks.
    groundtruth_instance_mask_weights: ground truth instance masks weights.
    groundtruth_instance_boundaries: ground truth instance boundaries.
    groundtruth_instance_classes: instance mask-level class labels.
    groundtruth_keypoints: ground truth keypoints.
    groundtruth_keypoint_depths: Relative depth of the keypoints.
    groundtruth_keypoint_depth_weights: Weights of the relative depth of the
      keypoints.
    groundtruth_keypoint_visibilities: ground truth keypoint visibilities.
    groundtruth_keypoint_weights: groundtruth weight factor for keypoints.
    groundtruth_label_weights: groundtruth label weights.
    groundtruth_verified_negative_classes: groundtruth verified negative classes
    groundtruth_not_exhaustive_classes: groundtruth not-exhaustively labeled
      classes.
    groundtruth_weights: groundtruth weight factor for bounding boxes.
    groundtruth_dp_num_points: The number of DensePose sampled points for each
      instance.
    groundtruth_dp_part_ids: Part indices for DensePose points.
    groundtruth_dp_surface_coords: Image locations and UV coordinates for
      DensePose points.
    num_groundtruth_boxes: number of groundtruth boxes.
    is_annotated: whether an image has been labeled or not.
    true_image_shapes: true shapes of images in the resized images, as resized
      images can be padded with zeros.
    multiclass_scores: the label score per class for each box.
    context_features: a flattened list of contextual features.
    context_feature_length: the fixed length of each feature in
      context_features, used for reshaping.
    valid_context_size: the valid context size, used in filtering the padded
      context features.
    context_features_image_id_list: the list of image source ids corresponding
      to the features in context_features
    image_format: format for the images, used to decode
    image_height: height of images, used to decode
    image_width: width of images, used to decode
  """
  image = 'image'
  image_additional_channels = 'image_additional_channels'
  original_image = 'original_image'
  original_image_spatial_shape = 'original_image_spatial_shape'
  key = 'key'
  source_id = 'source_id'
  filename = 'filename'
  groundtruth_image_classes = 'groundtruth_image_classes'
  groundtruth_image_confidences = 'groundtruth_image_confidences'
  groundtruth_labeled_classes = 'groundtruth_labeled_classes'
  groundtruth_boxes = 'groundtruth_boxes'
  groundtruth_classes = 'groundtruth_classes'
  groundtruth_track_ids = 'groundtruth_track_ids'
  groundtruth_temporal_offset = 'groundtruth_temporal_offset'
  groundtruth_track_match_flags = 'groundtruth_track_match_flags'
  groundtruth_confidences = 'groundtruth_confidences'
  groundtruth_label_types = 'groundtruth_label_types'
  groundtruth_is_crowd = 'groundtruth_is_crowd'
  groundtruth_area = 'groundtruth_area'
  groundtruth_difficult = 'groundtruth_difficult'
  groundtruth_group_of = 'groundtruth_group_of'
  proposal_boxes = 'proposal_boxes'
  proposal_objectness = 'proposal_objectness'
  groundtruth_instance_masks = 'groundtruth_instance_masks'
  groundtruth_instance_mask_weights = 'groundtruth_instance_mask_weights'
  groundtruth_instance_boundaries = 'groundtruth_instance_boundaries'
  groundtruth_instance_classes = 'groundtruth_instance_classes'
  groundtruth_keypoints = 'groundtruth_keypoints'
  groundtruth_keypoint_depths = 'groundtruth_keypoint_depths'
  groundtruth_keypoint_depth_weights = 'groundtruth_keypoint_depth_weights'
  groundtruth_keypoint_visibilities = 'groundtruth_keypoint_visibilities'
  groundtruth_keypoint_weights = 'groundtruth_keypoint_weights'
  groundtruth_label_weights = 'groundtruth_label_weights'
  groundtruth_verified_neg_classes = 'groundtruth_verified_neg_classes'
  groundtruth_not_exhaustive_classes = 'groundtruth_not_exhaustive_classes'
  groundtruth_weights = 'groundtruth_weights'
  groundtruth_dp_num_points = 'groundtruth_dp_num_points'
  groundtruth_dp_part_ids = 'groundtruth_dp_part_ids'
  groundtruth_dp_surface_coords = 'groundtruth_dp_surface_coords'
  num_groundtruth_boxes = 'num_groundtruth_boxes'
  is_annotated = 'is_annotated'
  true_image_shape = 'true_image_shape'
  multiclass_scores = 'multiclass_scores'
  context_features = 'context_features'
  context_feature_length = 'context_feature_length'
  valid_context_size = 'valid_context_size'
  context_features_image_id_list = 'context_features_image_id_list'
  image_timestamps = 'image_timestamps'
  image_format = 'image_format'
  image_height = 'image_height'
  image_width = 'image_width'

