import os
import numpy as np
import sys
import glob
import xml.etree.ElementTree as ET
from PIL import Image

import tensorflow as tf


from object_detection.utils import dataset_util


# Define your class labels here (if applicable)
class_labels = {
    'roof': 1
}



def read_image_to_numpy(image_file):
    """ Read image from file and convert to NumPy array
            convert to RGB format if RGBA
    """
    image = Image.open(image_file)
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    return np.array(image)


def encode_image(image):
    """Encode image as byte string 
    """
    return tf.io.encode_jpeg(image).numpy()



# 
def create_tf_example(xml_file, image_file, class_name2labels):
    """ create an TF example using xml_file and image_file

        Args:
            class_labels: a dictionary with key is the class name and the value is the class label, 
                i.e class_labels = {'helmet': 1, 'head': 2, 'person': 3}
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
        'image/filename': dataset_util.bytes_feature(image_filename.encode('utf8')),
        'image/format': dataset_util.bytes_feature(image_format.encode('utf8')),
        
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    })
    tf_example = tf.train.Example(features=tf_features)

    return tf_example


# create TFRecord
def create_tf_record(output_file, xml_dir, image_dir, image_suffix = 'jpg'):
    """
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

def main(xml_dir, image_dir, output_file, image_suffix = 'jpg'):
    create_tf_record(output_file, xml_dir, image_dir, image_suffix = image_suffix)

if __name__ == '__main__':
    """
        example usage:
            python generate_tfrecord.py  data/sample_data/roof/rawdata/train/ data/sample_data/roof/rawdata/train/  data/sample_data/roof/train.tfrecord tif
    """

    if len(sys.argv) != 4 and len(sys.argv) != 5:
        print("Usage: python generate_tfrecord.py <xml_dir> <image_dir> <output_file> [image_suffix]")

    else:
        xml_dir, image_dir, output_file = sys.argv[1], sys.argv[2], sys.argv[3]
        image_suffix = 'jpg'
        if len(sys.argv) == 5:
            image_suffix = sys.argv[4]

        main(xml_dir = xml_dir, image_dir = image_dir, output_file = output_file, image_suffix = image_suffix)