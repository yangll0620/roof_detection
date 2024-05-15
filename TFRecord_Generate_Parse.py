import os
import numpy as np
import sys
import glob
import xml.etree.ElementTree as ET

import tensorflow as tf


from object_detection.utils import dataset_util
from utilities import read_image_to_numpy


# Define your class labels here (if applicable)
class_labels = {
    'roof': 1
}


def encode_image(image):
    """Encode image as byte string 
    """
    return tf.io.encode_jpeg(image).numpy()


def create_tf_example(xml_file, image_file, class_name2labels):
    """ create an TF example using xml_file and image_file

        Args:
            class_labels: a dictionary with key is the class name and the value is the class label, 
                i.e class_labels = {'helmet': 1, 'head': 2, 'person': 3}
        
        Returns:
            tf_example: an TF Example
    """
    
    # Read XML file
    with tf.io.gfile.GFile(xml_file, 'r') as f:
        xml_str = f.read()
    xml = ET.fromstring(xml_str)

    # Extract image filename and size
    image_filename = os.path.basename(image_file)
    image_format = image_filename.split('.')[-1]

    # image width and height
    image_width = int(xml.find('size/width').text)
    image_height = int(xml.find('size/height').text)

    # read image
    image_np = read_image_to_numpy(image_file)
    encoded_image_data = encode_image(image_np)

    # Extract bounding box coordinates and class labels
    xmins, xmaxs, ymins, ymaxs, classes_text, classes = [], [], [], [], [], []
    for obj in xml.findall('object'):
        class_name = obj.find('name').text
        classes_text.append(class_name.encode('utf8'))
        classes.append(class_name2labels[class_name])
        bbox = obj.find('bndbox')
        xmins.append(float(bbox.find('xmin').text) / image_width)
        xmaxs.append(float(bbox.find('xmax').text) / image_width)
        ymins.append(float(bbox.find('ymin').text) / image_height)
        ymaxs.append(float(bbox.find('ymax').text) / image_height)

    # Create TF example
    tf_features = tf.train.Features(feature={
        'image/data': dataset_util.bytes_feature(encoded_image_data),
        'image/height': dataset_util.int64_feature(image_height),
        'image/width': dataset_util.int64_feature(image_width),
        'image/filename': dataset_util.bytes_feature(image_filename.encode('utf-8')),
        'image/format': dataset_util.bytes_feature(image_format.encode('utf-8')),
        
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    })
    tf_example = tf.train.Example(features=tf_features)

    return tf_example


def create_tf_record(output_file, xml_dir, image_dir, image_suffix = 'jpg'):
    """ create TFRecord
        Args:
            image_suffix: the suffix for image (e.g 'jpg', 'tif'), default = 'tif'
    """
    with tf.io.TFRecordWriter(output_file) as writer:
        # Find all XML files in the directory
        xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
        for xml_file in xml_files:
            
            # Extract image filename
            image_filename = os.path.splitext(os.path.basename(xml_file))[0] + '.' + image_suffix
            image_file = os.path.join(image_dir, image_filename)
            
            # Create TF example
            tf_example = create_tf_example(xml_file, image_file, class_name2labels = class_labels)
            writer.write(tf_example.SerializeToString())
    
    print(f"Generated TFRecord File: {output_file}")



def parse_tfexample(example):
    """ Parse tf.train.Example

        Returns:
            decoded_image_np
            filename
            height
            width
            boxes
            class_name
            class_label

    """
    feature_description = {
    'image/data': tf.io.FixedLenFeature([], tf.string),
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.string),

    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    'image/object/class/label': tf.io.VarLenFeature(tf.int64)
    }
    
    example = tf.io.parse_single_example(example, feature_description)
    
    # Extract image data, height, width, filename
    image = tf.io.decode_jpeg(example['image/data'], channels = 3)
    height = example['image/height']
    width = example['image/width']
    filename = example['image/filename']

    # Extract bounding box coordinates
    xmin = tf.sparse.to_dense(example['image/object/bbox/xmin'])
    xmax = tf.sparse.to_dense(example['image/object/bbox/xmax'])
    ymin = tf.sparse.to_dense(example['image/object/bbox/ymin'])
    ymax = tf.sparse.to_dense(example['image/object/bbox/ymax'])
    boxes = [ymin, xmin, ymax, xmax]

    # Convert class text and labels to dense tensors
    class_name = tf.sparse.to_dense(example['image/object/class/text'])
    class_label = tf.sparse.to_dense(example['image/object/class/label'])

    
    return image, height, width, filename, boxes, class_name, class_label