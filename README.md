# Real-time Crowd Segmentation from 2D/3D lidar person detections

# Simulating crowds of people in Isaac Sim

This repository...


If you find this code relevant for your work, please consider citing one or both of these papers. A bibtex entry is provided below:
```
@article{hughes2023foundations,
         title={TODO},
         author={TODO},
         year={2024},
         eprint={xxxx.xxxxx},
         archivePrefix={arXiv},
         primaryClass={cs.RO}
}
```

## Installation

### Requirements
- GPU for machine learning
    - When not using CUDA install pytorch manually!

### Setup

1. Create a virtual python environment with `python3 -m venv env`
2. Enter into the environment using `source env/bin/activate`
3. Install required packages `pip install -r requirements.txt`

## Configuration
Most basic configurations for both input/label generation and the scene segmentation itself can be found in the `config` folder.



## Run

### Input and Label Generator
#### Requirements
Have detection data in either CrowdBot or JRDB (with additional velocities of all people) format. See [this](https://github.com/SCAI-Lab/isaac-crowd-sim) repository for a simulator to create data in JRDB format.

#### Usage with JRDB based data

Run `python3 generator/generator.py --dataformat "JRDB"` with the following additional arguments. 
- `--input` with the path to your detection data
- `--output` with the path where the generated images should be stored.

#### Usage with CrowdBot data

*This requires a few more arguments*

Run `python3 generator/generator.py --dataformat "CrowdBot"` with the following additional arguments.
- `--input` with the path to your detection data
- `--output` with the path where the generated images should be stored.
- `--input_tf` with the path to the file with all tf frames.
- `--input_vel` with the path to the file with all detection velocities.

Additionally the argument `--label "False"` can be added for both cases to only generate the input without generating all the label.


### Scene segmentation

#### Train

1. Create a text file at the location given by `data_root` in the `config/config.yml` file. This file should have all relative paths to the training data folder.


Optionally: The same can be done for the validation data using `validation_dir` as the config variable

2. Run `python3 engine/train.py` to start training of the model.

#### Predict

Run `python3 engine/predict.py` with the following arguments.
- `--input` with the location to the input images you want to predict
- `--output` with the location where the predicted segmentations should be stored
- `--checkpoint` with the path to the checkpoint file to restart at. If empty it will retrain with the given data.