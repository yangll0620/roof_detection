from object_detection.utils import config_util
from object_detection.builders import model_builder

def create_detection_model(pipeline_config_path, num_classes = 1):
    """ Create a detection model based on pipeline_config.

        Args:
            pipeline_config_path: Path to pipeline_pb2.TrainEvalPipelineConfig text proto.
    """

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    model_config = configs['model']
    model_config.ssd.num_classes = num_classes
    model_config.ssd.freeze_batchnorm = True
    detection_model = model_builder.build(model_config=model_config, is_training=True)

    return detection_model