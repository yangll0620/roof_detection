

from object_detection.builders import hyperparams_builder
from object_detection.models import ssd_resnet_v1_fpn_keras_feature_extractor
from object_detection.builders import box_coder_builder
from object_detection.builders import matcher_builder
from object_detection.builders import region_similarity_calculator_builder
from object_detection.builders import anchor_generator_builder
from object_detection.builders import box_predictor_builder
from object_detection.builders import image_resizer_builder
from object_detection.builders import post_processing_builder
from object_detection.builders import losses_builder
from object_detection.utils import ops
from object_detection.core import target_assigner
from object_detection.meta_architectures import ssd_meta_arch

def _build_ssd_feature_extractor(feature_extractor_config,
                                 is_training,
                                 freeze_batchnorm,
                                 reuse_weights=None):
    
    feature_type = feature_extractor_config.type
    depth_multiplier = feature_extractor_config.depth_multiplier
    min_depth = feature_extractor_config.min_depth
    pad_to_multiple = feature_extractor_config.pad_to_multiple
    use_explicit_padding = feature_extractor_config.use_explicit_padding
    use_depthwise = feature_extractor_config.use_depthwise


    conv_hyperparams = hyperparams_builder.KerasLayerHyperparams(feature_extractor_config.conv_hyperparams)
    override_base_feature_extractor_hyperparams = (feature_extractor_config.override_base_feature_extractor_hyperparams)
    
    kwargs = {
      'is_training':
          is_training,
      'depth_multiplier':
          depth_multiplier,
      'min_depth':
          min_depth,
      'pad_to_multiple':
          pad_to_multiple,
      'use_explicit_padding':
          use_explicit_padding,
      'use_depthwise':
          use_depthwise,
      'override_base_feature_extractor_hyperparams':
          override_base_feature_extractor_hyperparams
    }

    kwargs.update({
        'conv_hyperparams': conv_hyperparams,
        'inplace_batchnorm_update': False,
        'freeze_batchnorm': freeze_batchnorm
    })

    kwargs.update({
        'fpn_min_level':
            feature_extractor_config.fpn.min_level,
        'fpn_max_level':
            feature_extractor_config.fpn.max_level,
        'additional_layer_depth':
            feature_extractor_config.fpn.additional_layer_depth,
    })

    return ssd_resnet_v1_fpn_keras_feature_extractor.SSDResNet50V1FpnKerasFeatureExtractor(**kwargs)



def _build_ssd_model(ssd_config, is_training, add_summaries):
    num_classes = ssd_config.num_classes
    # Feature extractor
    
    feature_extractor = _build_ssd_feature_extractor(
      feature_extractor_config=ssd_config.feature_extractor,
      freeze_batchnorm=ssd_config.freeze_batchnorm,
      is_training=is_training)
    
    box_coder = box_coder_builder.build(ssd_config.box_coder)
    matcher = matcher_builder.build(ssd_config.matcher)
    region_similarity_calculator = region_similarity_calculator_builder.build(
      ssd_config.similarity_calculator)
    anchor_generator = anchor_generator_builder.build(ssd_config.anchor_generator)
    
    encode_background_as_zeros = ssd_config.encode_background_as_zeros
    negative_class_weight = ssd_config.negative_class_weight

    ## yll just remain feature_extractor.is_keras_model:
    ssd_box_predictor = box_predictor_builder.build_keras(
        hyperparams_fn=hyperparams_builder.KerasLayerHyperparams,
        freeze_batchnorm=ssd_config.freeze_batchnorm,
        inplace_batchnorm_update=False,
        num_predictions_per_location_list=anchor_generator.num_anchors_per_location(),
        box_predictor_config=ssd_config.box_predictor,
        is_training=is_training,
        num_classes=num_classes,
        add_background_class=ssd_config.add_background_class)
    



    image_resizer_fn = image_resizer_builder.build(ssd_config.image_resizer)
    non_max_suppression_fn, score_conversion_fn = post_processing_builder.build(ssd_config.post_processing)
    (classification_loss, localization_loss, classification_weight,
    localization_weight, hard_example_miner, random_example_sampler,
    expected_loss_weights_fn) = losses_builder.build(ssd_config.loss)
    normalize_loss_by_num_matches = ssd_config.normalize_loss_by_num_matches
    normalize_loc_loss_by_codesize = ssd_config.normalize_loc_loss_by_codesize


    equalization_loss_config = ops.EqualizationLossConfig(
      weight=ssd_config.loss.equalization_loss.weight,
      exclude_prefixes=ssd_config.loss.equalization_loss.exclude_prefixes)
    

    target_assigner_instance = target_assigner.TargetAssigner(
      region_similarity_calculator,
      matcher,
      box_coder,
      negative_class_weight=negative_class_weight)
    

    ssd_meta_arch_fn = ssd_meta_arch.SSDMetaArch
    kwargs = {}

    return ssd_meta_arch_fn(
      is_training=is_training,
      anchor_generator=anchor_generator,
      box_predictor=ssd_box_predictor,
      box_coder=box_coder,
      feature_extractor=feature_extractor,
      encode_background_as_zeros=encode_background_as_zeros,
      image_resizer_fn=image_resizer_fn,
      non_max_suppression_fn=non_max_suppression_fn,
      score_conversion_fn=score_conversion_fn,
      classification_loss=classification_loss,
      localization_loss=localization_loss,
      classification_loss_weight=classification_weight,
      localization_loss_weight=localization_weight,
      normalize_loss_by_num_matches=normalize_loss_by_num_matches,
      hard_example_miner=hard_example_miner,
      target_assigner_instance=target_assigner_instance,
      add_summaries=add_summaries,
      normalize_loc_loss_by_codesize=normalize_loc_loss_by_codesize,
      freeze_batchnorm=ssd_config.freeze_batchnorm,
      inplace_batchnorm_update=ssd_config.inplace_batchnorm_update,
      add_background_class=ssd_config.add_background_class,
      explicit_background_class=ssd_config.explicit_background_class,
      random_example_sampler=random_example_sampler,
      expected_loss_weights_fn=expected_loss_weights_fn,
      use_confidences_as_targets=ssd_config.use_confidences_as_targets,
      implicit_example_weight=ssd_config.implicit_example_weight,
      equalization_loss_config=equalization_loss_config,
      return_raw_detections_during_predict=(
          ssd_config.return_raw_detections_during_predict),
      **kwargs)
    



def build(model_config, is_training, add_summaries=True):
    ssd_config = getattr(model_config, 'ssd')
    return _build_ssd_model(ssd_config, is_training = is_training, add_summaries = add_summaries)