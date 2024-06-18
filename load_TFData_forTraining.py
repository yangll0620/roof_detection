import tensorflow as tf
from TFRecord_Generate_Parse import parse_tfexample

def load_from_tfrecordFile(tfrecord_file):
    """ load data from tfrecord_file
        Args:
            tfrecord_file:
        
        Return:
            images_np
            gt_boxes
            class_names
            class_labels
            filenames

    """
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    parsed_dataset = dataset.map(parse_tfexample)


    # extract from parsed_dataset  
    gt_boxes = [] 
    images_np = [] 
    class_names = []
    class_labels = []
    filenames = []
    for image, height, width, filename, boxes, class_name, class_label in parsed_dataset:
        filenames.append(filename.numpy().decode('utf-8'))

        images_np.append(image.numpy()) 

        gt_boxes.append(boxes.numpy().T)
        class_names.append([name.decode() for name in class_name.numpy()])
        class_labels.append(class_label.numpy())

    return images_np, gt_boxes, class_names, class_labels, filenames

def convert_2_tensor(images_np, gt_boxes, gt_class_labels, label_id_offset, num_classes): 
    """ convert to dataset used for training from tfrecord_file
        Args:
            tfrecord_file:
        
        Return:
            image_tensors
            gt_box_tensors
            gt_classes_one_hot_tensors
    """

    # convert everythin to tensors, convert class labels to one-hot
    image_tensors = []
    gt_box_tensors = []
    gt_classes_one_hot_tensors = []
    for image_np, gt_box, gt_class_label in zip(images_np, gt_boxes, gt_class_labels):
        image_tensors.append(tf.convert_to_tensor(image_np, dtype=tf.float32))
        gt_box_tensors.append(tf.convert_to_tensor(gt_box, dtype=tf.float32))

        zero_indexed_groundtruth_classes = tf.convert_to_tensor(gt_class_label - label_id_offset)
        gt_classes_one_hot_tensors.append(tf.one_hot(zero_indexed_groundtruth_classes, num_classes))
    

    return image_tensors, gt_box_tensors, gt_classes_one_hot_tensors


def load_sampledata():
    label_id_offset = 1
    num_classes = 1
    sample_file = 'data/sample_data/roof/train.tfrecord'

    train_images_np, train_gt_boxes, train_class_names, train_class_labels, train_filenames = load_from_tfrecordFile(sample_file)
    train_image_tensors, train_gt_box_tensors, train_gt_classes_one_hot_tensors = convert_2_tensor(train_images_np, train_gt_boxes, train_class_labels, label_id_offset, num_classes)

    print("sample data in " + sample_file)

    return train_image_tensors, train_gt_box_tensors, train_gt_classes_one_hot_tensors