import tensorflow.compat.v1 as tf


from object_detection.utils import ops
from object_detection.core import box_list
from object_detection.utils import shape_utils


def area(boxlist, scope=None):
  """Computes area of boxes.

  Args:
    boxlist: BoxList holding N boxes
    scope: name scope.

  Returns:
    a tensor with shape [N] representing box areas.
  """
  with tf.name_scope(scope, 'Area'):
    y_min, x_min, y_max, x_max = tf.split(
        value=boxlist.get(), num_or_size_splits=4, axis=1)
    return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])


def _copy_extra_fields(boxlist_to_copy_to, boxlist_to_copy_from):
  """Copies the extra fields of boxlist_to_copy_from to boxlist_to_copy_to.

  Args:
    boxlist_to_copy_to: BoxList to which extra fields are copied.
    boxlist_to_copy_from: BoxList from which fields are copied.

  Returns:
    boxlist_to_copy_to with extra fields.
  """
  for field in boxlist_to_copy_from.get_extra_fields():
    boxlist_to_copy_to.add_field(field, boxlist_to_copy_from.get_field(field))
  return boxlist_to_copy_to

def scale(boxlist, y_scale, x_scale, scope=None):
  """scale box coordinates in x and y dimensions.

  Args:
    boxlist: BoxList holding N boxes
    y_scale: (float) scalar tensor
    x_scale: (float) scalar tensor
    scope: name scope.

  Returns:
    boxlist: BoxList holding N boxes
  """
  with tf.name_scope(scope, 'Scale'):
    y_scale = tf.cast(y_scale, tf.float32)
    x_scale = tf.cast(x_scale, tf.float32)
    y_min, x_min, y_max, x_max = tf.split(
        value=boxlist.get(), num_or_size_splits=4, axis=1)
    y_min = y_scale * y_min
    y_max = y_scale * y_max
    x_min = x_scale * x_min
    x_max = x_scale * x_max
    scaled_boxlist = box_list.BoxList(
        tf.concat([y_min, x_min, y_max, x_max], 1))
    return _copy_extra_fields(scaled_boxlist, boxlist)




def to_normalized_coordinates(boxlist, height, width,
                              check_range=True, scope=None):
  """Converts absolute box coordinates to normalized coordinates in [0, 1].

  Usually one uses the dynamic shape of the image or conv-layer tensor:
    boxlist = box_list_ops.to_normalized_coordinates(boxlist,
                                                     tf.shape(images)[1],
                                                     tf.shape(images)[2]),

  This function raises an assertion failed error at graph execution time when
  the maximum coordinate is smaller than 1.01 (which means that coordinates are
  already normalized). The value 1.01 is to deal with small rounding errors.

  Args:
    boxlist: BoxList with coordinates in terms of pixel-locations.
    height: Maximum value for height of absolute box coordinates.
    width: Maximum value for width of absolute box coordinates.
    check_range: If True, checks if the coordinates are normalized or not.
    scope: name scope.

  Returns:
    boxlist with normalized coordinates in [0, 1].
  """
  with tf.name_scope(scope, 'ToNormalizedCoordinates'):
    height = tf.cast(height, tf.float32)
    width = tf.cast(width, tf.float32)

    if check_range:
      max_val = tf.reduce_max(boxlist.get())
      max_assert = tf.Assert(tf.greater(max_val, 1.01),
                             ['max value is lower than 1.01: ', max_val])
      with tf.control_dependencies([max_assert]):
        width = tf.identity(width)

    return scale(boxlist, 1 / height, 1 / width)

class SortOrder(object):
  """Enum class for sort order.

  Attributes:
    ascend: ascend order.
    descend: descend order.
  """
  ascend = 1
  descend = 2


def sort_by_field(boxlist, field, order=SortOrder.descend, scope=None):
  """Sort boxes and associated fields according to a scalar field.

  A common use case is reordering the boxes according to descending scores.

  Args:
    boxlist: BoxList holding N boxes.
    field: A BoxList field for sorting and reordering the BoxList.
    order: (Optional) descend or ascend. Default is descend.
    scope: name scope.

  Returns:
    sorted_boxlist: A sorted BoxList with the field in the specified order.

  Raises:
    ValueError: if specified field does not exist
    ValueError: if the order is not either descend or ascend
  """
  with tf.name_scope(scope, 'SortByField'):
    if order != SortOrder.descend and order != SortOrder.ascend:
      raise ValueError('Invalid sort order')

    field_to_sort = boxlist.get_field(field)
    if len(field_to_sort.shape.as_list()) != 1:
      raise ValueError('Field should have rank 1')

    num_boxes = boxlist.num_boxes()
    num_entries = tf.size(field_to_sort)
    length_assert = tf.Assert(
        tf.equal(num_boxes, num_entries),
        ['Incorrect field size: actual vs expected.', num_entries, num_boxes])

    with tf.control_dependencies([length_assert]):
      _, sorted_indices = tf.nn.top_k(field_to_sort, num_boxes, sorted=True)

    if order == SortOrder.ascend:
      sorted_indices = tf.reverse_v2(sorted_indices, [0])

    return gather(boxlist, sorted_indices)




def intersection(boxlist1, boxlist2, scope=None):
  """Compute pairwise intersection areas between boxes.

  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes
    scope: name scope.

  Returns:
    a tensor with shape [N, M] representing pairwise intersections
  """
  with tf.name_scope(scope, 'Intersection'):
    y_min1, x_min1, y_max1, x_max1 = tf.split(
        value=boxlist1.get(), num_or_size_splits=4, axis=1)
    y_min2, x_min2, y_max2, x_max2 = tf.split(
        value=boxlist2.get(), num_or_size_splits=4, axis=1)
    all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
    all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
    all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths




def iou(boxlist1, boxlist2, scope=None):
  """Computes pairwise intersection-over-union between box collections.

  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes
    scope: name scope.

  Returns:
    a tensor with shape [N, M] representing pairwise iou scores.
  """
  with tf.name_scope(scope, 'IOU'):
    intersections = intersection(boxlist1, boxlist2)
    areas1 = area(boxlist1)
    areas2 = area(boxlist2)
    unions = (
        tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
    return tf.where(
        tf.equal(intersections, 0.0),
        tf.zeros_like(intersections), tf.truediv(intersections, unions))


def gather(boxlist, indices, fields=None, scope=None, use_static_shapes=False):
  """Gather boxes from BoxList according to indices and return new BoxList.

  By default, `gather` returns boxes corresponding to the input index list, as
  well as all additional fields stored in the boxlist (indexing into the
  first dimension).  However one can optionally only gather from a
  subset of fields.

  Args:
    boxlist: BoxList holding N boxes
    indices: a rank-1 tensor of type int32 / int64
    fields: (optional) list of fields to also gather from.  If None (default),
      all fields are gathered from.  Pass an empty fields list to only gather
      the box coordinates.
    scope: name scope.
    use_static_shapes: Whether to use an implementation with static shape
      gurantees.

  Returns:
    subboxlist: a BoxList corresponding to the subset of the input BoxList
    specified by indices
  Raises:
    ValueError: if specified field is not contained in boxlist or if the
      indices are not of type int32
  """
  with tf.name_scope(scope, 'Gather'):
    if len(indices.shape.as_list()) != 1:
      raise ValueError('indices should have rank 1')
    if indices.dtype != tf.int32 and indices.dtype != tf.int64:
      raise ValueError('indices should be an int32 / int64 tensor')
    gather_op = tf.gather
    if use_static_shapes:
      gather_op = ops.matmul_gather_on_zeroth_axis
    subboxlist = box_list.BoxList(gather_op(boxlist.get(), indices))
    if fields is None:
      fields = boxlist.get_extra_fields()
    fields += ['boxes']
    for field in fields:
      if not boxlist.has_field(field):
        raise ValueError('boxlist must contain all specified fields')
      subfieldlist = gather_op(boxlist.get_field(field), indices)
      subboxlist.add_field(field, subfieldlist)
    return subboxlist
  

def matched_intersection(boxlist1, boxlist2, scope=None):
  """Compute intersection areas between corresponding boxes in two boxlists.

  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding N boxes
    scope: name scope.

  Returns:
    a tensor with shape [N] representing pairwise intersections
  """
  with tf.name_scope(scope, 'MatchedIntersection'):
    y_min1, x_min1, y_max1, x_max1 = tf.split(
        value=boxlist1.get(), num_or_size_splits=4, axis=1)
    y_min2, x_min2, y_max2, x_max2 = tf.split(
        value=boxlist2.get(), num_or_size_splits=4, axis=1)
    min_ymax = tf.minimum(y_max1, y_max2)
    max_ymin = tf.maximum(y_min1, y_min2)
    intersect_heights = tf.maximum(0.0, min_ymax - max_ymin)
    min_xmax = tf.minimum(x_max1, x_max2)
    max_xmin = tf.maximum(x_min1, x_min2)
    intersect_widths = tf.maximum(0.0, min_xmax - max_xmin)
    return tf.reshape(intersect_heights * intersect_widths, [-1])



def matched_iou(boxlist1, boxlist2, scope=None):
  """Compute intersection-over-union between corresponding boxes in boxlists.

  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding N boxes
    scope: name scope.

  Returns:
    a tensor with shape [N] representing pairwise iou scores.
  """
  with tf.name_scope(scope, 'MatchedIOU'):
    intersections = matched_intersection(boxlist1, boxlist2)
    areas1 = area(boxlist1)
    areas2 = area(boxlist2)
    unions = areas1 + areas2 - intersections
    return tf.where(
        tf.equal(intersections, 0.0),
        tf.zeros_like(intersections), tf.truediv(intersections, unions))


def concatenate(boxlists, fields=None, scope=None):
  """Concatenate list of BoxLists.

  This op concatenates a list of input BoxLists into a larger BoxList.  It also
  handles concatenation of BoxList fields as long as the field tensor shapes
  are equal except for the first dimension.

  Args:
    boxlists: list of BoxList objects
    fields: optional list of fields to also concatenate.  By default, all
      fields from the first BoxList in the list are included in the
      concatenation.
    scope: name scope.

  Returns:
    a BoxList with number of boxes equal to
      sum([boxlist.num_boxes() for boxlist in BoxList])
  Raises:
    ValueError: if boxlists is invalid (i.e., is not a list, is empty, or
      contains non BoxList objects), or if requested fields are not contained in
      all boxlists
  """
  with tf.name_scope(scope, 'Concatenate'):
    if not isinstance(boxlists, list):
      raise ValueError('boxlists should be a list')
    if not boxlists:
      raise ValueError('boxlists should have nonzero length')
    for boxlist in boxlists:
      if not isinstance(boxlist, box_list.BoxList):
        raise ValueError('all elements of boxlists should be BoxList objects')
    concatenated = box_list.BoxList(
        tf.concat([boxlist.get() for boxlist in boxlists], 0))
    if fields is None:
      fields = boxlists[0].get_extra_fields()
    for field in fields:
      first_field_shape = boxlists[0].get_field(field).get_shape().as_list()
      first_field_shape[0] = -1
      if None in first_field_shape:
        raise ValueError('field %s must have fully defined shape except for the'
                         ' 0th dimension.' % field)
      for boxlist in boxlists:
        if not boxlist.has_field(field):
          raise ValueError('boxlist must contain all requested fields')
        field_shape = boxlist.get_field(field).get_shape().as_list()
        field_shape[0] = -1
        if field_shape != first_field_shape:
          raise ValueError('field %s must have same shape for all boxlists '
                           'except for the 0th dimension.' % field)
      concatenated_field = tf.concat(
          [boxlist.get_field(field) for boxlist in boxlists], 0)
      concatenated.add_field(field, concatenated_field)
    return concatenated


def clip_to_window(boxlist, window, filter_nonoverlapping=True, scope=None):
  """Clip bounding boxes to a window.

  This op clips any input bounding boxes (represented by bounding box
  corners) to a window, optionally filtering out boxes that do not
  overlap at all with the window.

  Args:
    boxlist: BoxList holding M_in boxes
    window: a tensor of shape [4] representing the [y_min, x_min, y_max, x_max]
      window to which the op should clip boxes.
    filter_nonoverlapping: whether to filter out boxes that do not overlap at
      all with the window.
    scope: name scope.

  Returns:
    a BoxList holding M_out boxes where M_out <= M_in
  """
  with tf.name_scope(scope, 'ClipToWindow'):
    y_min, x_min, y_max, x_max = tf.split(
        value=boxlist.get(), num_or_size_splits=4, axis=1)
    win_y_min = window[0]
    win_x_min = window[1]
    win_y_max = window[2]
    win_x_max = window[3]
    y_min_clipped = tf.maximum(tf.minimum(y_min, win_y_max), win_y_min)
    y_max_clipped = tf.maximum(tf.minimum(y_max, win_y_max), win_y_min)
    x_min_clipped = tf.maximum(tf.minimum(x_min, win_x_max), win_x_min)
    x_max_clipped = tf.maximum(tf.minimum(x_max, win_x_max), win_x_min)
    clipped = box_list.BoxList(
        tf.concat([y_min_clipped, x_min_clipped, y_max_clipped, x_max_clipped],
                  1))
    clipped = _copy_extra_fields(clipped, boxlist)
    if filter_nonoverlapping:
      areas = area(clipped)
      nonzero_area_indices = tf.cast(
          tf.reshape(tf.where(tf.greater(areas, 0.0)), [-1]), tf.int32)
      clipped = gather(clipped, nonzero_area_indices)
    return clipped


def change_coordinate_frame(boxlist, window, scope=None):
  """Change coordinate frame of the boxlist to be relative to window's frame.

  Given a window of the form [ymin, xmin, ymax, xmax],
  changes bounding box coordinates from boxlist to be relative to this window
  (e.g., the min corner maps to (0,0) and the max corner maps to (1,1)).

  An example use case is data augmentation: where we are given groundtruth
  boxes (boxlist) and would like to randomly crop the image to some
  window (window). In this case we need to change the coordinate frame of
  each groundtruth box to be relative to this new window.

  Args:
    boxlist: A BoxList object holding N boxes.
    window: A rank 1 tensor [4].
    scope: name scope.

  Returns:
    Returns a BoxList object with N boxes.
  """
  with tf.name_scope(scope, 'ChangeCoordinateFrame'):
    win_height = window[2] - window[0]
    win_width = window[3] - window[1]
    boxlist_new = scale(box_list.BoxList(
        boxlist.get() - [window[0], window[1], window[0], window[1]]),
                        1.0 / win_height, 1.0 / win_width)
    boxlist_new = _copy_extra_fields(boxlist_new, boxlist)
    return boxlist_new


def pad_or_clip_box_list(boxlist, num_boxes, scope=None):
  """Pads or clips all fields of a BoxList.

  Args:
    boxlist: A BoxList with arbitrary of number of boxes.
    num_boxes: First num_boxes in boxlist are kept.
      The fields are zero-padded if num_boxes is bigger than the
      actual number of boxes.
    scope: name scope.

  Returns:
    BoxList with all fields padded or clipped.
  """
  with tf.name_scope(scope, 'PadOrClipBoxList'):
    subboxlist = box_list.BoxList(shape_utils.pad_or_clip_tensor(
        boxlist.get(), num_boxes))
    for field in boxlist.get_extra_fields():
      subfield = shape_utils.pad_or_clip_tensor(
          boxlist.get_field(field), num_boxes)
      subboxlist.add_field(field, subfield)
    return subboxlist
