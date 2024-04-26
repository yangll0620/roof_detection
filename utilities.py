import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np


from object_detection.utils import visualization_utils as viz_utils


def read_image_to_numpy(image_file):
    """ Read image from file and convert to NumPy array
            convert to RGB format if RGBA
    """
    image = Image.open(image_file)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    return np.array(image)

def plot_detection(image_np, boxes, classes, scores, category_index, min_score_thresh = 0.8, figsize=(12, 16), image_name = None):
    """ 
        Args:
            image_np
            boxes
            classes
            scores
    """
    image_np_with_annotations = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_annotations,
        boxes,
        classes,
        scores,
        category_index,
        use_normalized_coordinates = True,
        min_score_thresh = min_score_thresh
    )
    
    if image_name:
        plt.imsave(image_name, image_np_with_annotations)
    else:
        plt.imshow(image_np_with_annotations)