# Roof Detection 

## Preparison

### Setting up Conda Environment

Create Conda environment from the YAML file
```terminal
conda env create -f environment.yml
```

Generate a YAML file in Conda
``` Terminal
conda env export > environment.yml
```

### Pretrained_model
Down the SSD ResNet50 V1 FPN 640x640  model from [TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md), unzip and put inside models/pretrain_models

### Workspace Director

<img width="405" alt="Screen Shot 2024-04-16 at 11 55 44 AM" src="https://github.com/yangll0620/roof_detection/assets/15356284/8b9d19bc-2c30-4040-8963-61e5e747927d">


## Run

1. Prepare TFRecord 
2. Roof_Detect.ipynb
   
