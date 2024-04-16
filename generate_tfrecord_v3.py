import os
import io
import glob
import xml.etree.ElementTree as ET
import tensorflow as tf
from object_detection.utils import dataset_util



# Define your class labels here (if applicable)
class_labels = {
    'helmet': 1,
    'head': 2,
    'person': 3,
    # Add more classes as needed
}



def encode_image(image_path):
    with tf.io.gfile.GFile(image_path, 'rb') as f:
        return f.read()


# Function to create TF example
def create_tf_example(xml_path, image_path):
    # Read XML file
    with tf.io.gfile.GFile(xml_path, 'r') as f:
        xml_str = f.read()
    xml = ET.fromstring(xml_str)

    # Extract image filename and size
    image_filename = os.path.basename(image_path)
    image_width = int(xml.find('size/width').text)
    image_height = int(xml.find('size/height').text)

    # Encode image
    encoded_image_data = encode_image(image_path)

    # Extract bounding box coordinates and class labels
    xmins, xmaxs, ymins, ymaxs, classes_text, classes = [], [], [], [], [], []
    for obj in xml.findall('object'):
        class_name = obj.find('name').text
        classes_text.append(class_name.encode('utf8'))
        classes.append(class_labels[class_name])
        bbox = obj.find('bndbox')
        xmins.append(float(bbox.find('xmin').text) / image_width)
        xmaxs.append(float(bbox.find('xmax').text) / image_width)
        ymins.append(float(bbox.find('ymin').text) / image_height)
        ymaxs.append(float(bbox.find('ymax').text) / image_height)

    # Create TF example
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(image_height),
        'image/width': dataset_util.int64_feature(image_width),
        'image/filename': dataset_util.bytes_feature(image_filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(image_filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example

# Function to create TFRecord
def create_tf_record(output_file, xml_dir, image_dir):
    with tf.io.TFRecordWriter(output_file) as writer:
        # Find all XML files in the directory
        xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
        for xml_file in xml_files:
            # Extract image filename
            image_filename = os.path.splitext(os.path.basename(xml_file))[0] + '.jpg'
            image_path = os.path.join(image_dir, image_filename)
            
            # Create TF example
            tf_example = create_tf_example(xml_file, image_path)
            writer.write(tf_example.SerializeToString())

# Main function
def main():
    
    # Define input directories and output TFRecord file
    data_dir = 'data/sample_data/hard_hat/'
    xml_dir = os.path.join(data_dir, 'rawdata/train')
    image_dir = xml_dir
    output_file = os.path.join(data_dir, 'annotations/train.tfrecord')

    # Create TFRecord
    create_tf_record(output_file, xml_dir, image_dir)
    print(f'TFRecord created: {output_file}')

if __name__ == '__main__':
    main()