import tensorflow as tf

def restore_weights_from_pretrained(detection_model, pretrained_checkpoint_path):
    """ Restore weights from pretrained model

        Args:
            detection_model: detection model
            pretrain_checkpoint_path: pretrained model checkpoint path

        Returns:
            detection_model: detection_model with weights restored from pretrained model specified by checkpoint_path
    """
    
    # Set up object-based checkpoint restore --- RetinaNet has two prediction
    # `heads` --- one for classification, the other for box regression.  
    # restore the box regression head but initialize the classification head from scratch 
    fake_box_predictor = tf.compat.v2.train.Checkpoint(
        _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
        _box_prediction_head=detection_model._box_predictor._box_prediction_head)
    
    fake_model = tf.compat.v2.train.Checkpoint(_feature_extractor=detection_model._feature_extractor,
                                           _box_predictor=fake_box_predictor)
    ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
    
    ckpt.restore(pretrained_checkpoint_path).expect_partial()


    # Run model through a dummy image so that variables are created
    image, shapes = detection_model.preprocess(tf.zeros([1, 640, 640, 3]))
    prediction_dict = detection_model.predict(image, shapes)
    _ = detection_model.postprocess(prediction_dict, shapes)

    print('Weights of detection_model restored from \n{}'.format(pretrained_checkpoint_path))

    return detection_model


def get_model_train_step_function(model, optimizer, vars_to_fine_tune):
    """Get a tf.function for training step."""
    
    def train_step_fn(image_tensors, gt_boxes_list, gt_classes_list, batch_size = 4):
        """ A single training iteration

            Args:
                image_tensors:  a list of [height_in, width_in, channels] float tensor, len = batch_size 
                gt_boxes_list:
                gt_classes_list:
        """
    
        shapes = tf.constant(batch_size * [[640, 640, 3]], dtype=tf.int32)
        model.provide_groundtruth(groundtruth_boxes_list = gt_boxes_list, groundtruth_classes_list =gt_classes_list)
    
        with tf.GradientTape() as tape:
            preprocessed_images = tf.concat([model.preprocess(tf.expand_dims(image_tensor, axis = 0))[0] for image_tensor in image_tensors],axis = 0)
            prediction_dict = model.predict(preprocessed_images, shapes)
            losses_dict = model.loss(prediction_dict, shapes)
            total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
    
            gradients = tape.gradient(total_loss, vars_to_fine_tune)
            optimizer.apply_gradients(zip(gradients, vars_to_fine_tune))
            
        return total_loss

    
    def valid_step_fn(image_tensors, gt_boxes_list, gt_classes_list, batch_size):
        """ validate model 

            Args:
                image_tensors:  a list of [height_in, width_in, channels] float tensor, len = batch_size 
                gt_boxes_list:
                gt_classes_list:
        """
        shapes = tf.constant(batch_size * [[640, 640, 3]], dtype=tf.int32)
        model.provide_groundtruth(groundtruth_boxes_list = gt_boxes_list, groundtruth_classes_list =gt_classes_list)


        preprocessed_images = tf.concat([model.preprocess(tf.expand_dims(image_tensor, axis = 0))[0] for image_tensor in image_tensors],axis = 0)
        prediction_dict = model.predict(preprocessed_images, shapes)
        losses_dict = model.loss(prediction_dict, shapes)
        total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']


        return total_loss
        

    return train_step_fn, valid_step_fn