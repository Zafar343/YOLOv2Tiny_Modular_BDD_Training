# YOLOv2Tiny_Modular Code

This repository contains a modular implementation of YOLOv2-Tiny for object detection, supporting custom dataset training, self-training, and distributed training setups. The code is based on the original YOLOv2 implementation but with enhanced modularity and additional features.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Validation](#validation)
- [Inference](#inference)
- [Configuration](#configuration)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Upcoming Features](#upcoming-features)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Prerequisites

- Python 3.6+
- CUDA-capable GPU (recommended)
- PyTorch 1.7+
- OpenCV
- Other dependencies listed in requirements.txt

## Installation

1. Create a new conda environment:
```bash
conda create -n yolov2tiny python=3.6
conda activate yolov2tiny
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

### Custom Dataset Format
The dataset must be in YOLO format with the following structure:
```
dataset/
├── train/
│   ├── image1.jpg
│   ├── image1.txt
│   ├── image2.jpg
│   ├── image2.txt
│   └── ...
└── val/
    ├── image1.jpg
    ├── image1.txt
    ├── image2.jpg
    ├── image2.txt
    └── ...
```

Each label file (`.txt`) should have the same name as its corresponding image file and contain annotations in the format:
```
class_id x_center y_center width height
```
where:
- `class_id`: Integer representing the class (0-based)
- `x_center`, `y_center`: Normalized coordinates of the bounding box center (0-1)
- `width`, `height`: Normalized dimensions of the bounding box (0-1)

### Configuration File
Create a `data.yaml` file with the following structure:
```yaml
train: train.txt     # Path to training image list
val: val.txt         # Path to validation image list
val_dir: /val        # Path to validation images directory
nc: 1                # Number of classes
names: ["Vehicles"]  # Class names
```

Note: The `train.txt` and `val.txt` files should contain the absolute paths to the images in their respective splits.

## Training

### Training from Scratch
```bash
python train_torch_epoch.py --dataset custom --data path/to/data.yaml
```

### Training with Pretrained Weights
```bash
python train_torch_epoch.py --dataset custom --data path/to/data.yaml --resume --weights path/to/weights.pt
```

### Training Parameters
- `--dataset`: Dataset type (default: 'custom')
- `--data`: Path to data configuration file
- `--resume`: Resume training from checkpoint
- `--weights`: Path to pretrained weights
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size (default: 64)
- `--img-size`: Input image size (default: 416)
- `--device`: GPU device number (default: 0)

## Validation

Run validation with:
```bash
python validate.py --model_name path/to/model.pt --data path/to/data.yaml
```

### Validation Parameters
- `--model_name`: Path to model weights
- `--data`: Path to data configuration file
- `--conf-thres`: Confidence threshold (default: 0.2)
- `--nms-thres`: NMS threshold (default: 0.4)
- `--vis`: Enable visualization (default: False)
- `--device`: GPU device number (default: 0)

## Inference

For single image inference:
```bash
python demo.py --model_name path/to/model.pt --data path/to/dir or path/to/data text file
```

### Inference Parameters
- `--model_name`: Path to model weights
- `--source`: Path to input image/video
- `--conf-thres`: Confidence threshold
- `--nms-thres`: NMS threshold
- `--device`: GPU device number

## Configuration

The model configuration can be modified in `config/config_bdd.py`:
- Anchor boxes
- Learning rate schedule
- Input size
- Training parameters
- Loss function weights

## Performance Tuning

1. **Batch Size**: Adjust based on GPU memory
2. **Input Size**: Modify in config file (default: 416x416)
3. **Anchor Boxes**: Optimize for your dataset
4. **Learning Rate**: Adjust based on training progress
5. **Data Augmentation**: Modify in config file

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**
   - Reduce batch size
   - Reduce input image size
   - Use gradient checkpointing

2. **Training Instability**
   - Adjust learning rate
   - Check data format
   - Verify anchor box sizes

3. **Low Validation Performance**
   - Check data quality
   - Adjust confidence threshold
   - Verify class balance

### Support
For additional support, please open an issue in the repository.

## Upcoming Features

🚀 **Self-Training Module** (Coming Soon)
- Semi-supervised learning support
- Automatic pseudo-label generation
- Confidence-based filtering
- Iterative training pipeline

Stay tuned for updates! ⏳

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is based on the following works:

1. **Original YOLOv2 Paper**:
   - Title: YOLO9000: Better, Faster, Stronger
   - Authors: Joseph Redmon, Ali Farhadi
   - Conference: CVPR 2017
   - [Paper Link](https://arxiv.org/abs/1612.08242)

   Citation:
   ```bibtex
   @article{redmon2016yolo9000,
     title={YOLO9000: Better, Faster, Stronger},
     author={Redmon, Joseph and Farhadi, Ali},
     journal={arXiv preprint arXiv:1612.08242},
     year={2016}
   }
   ```

2. **Original PyTorch Implementation**:
   - Repository: [yolov2.pytorch](https://github.com/tztztztztz/yolov2.pytorch)
   - This implementation served as the foundation for our modular version
   - Special thanks to the original authors for their work

3. **BDD100K Dataset**:
   - Used for training and evaluation
   - [Dataset Link](https://www.bdd100k.com/)
   - Citation:
   ```bibtex
   @article{yu2018bdd100k,
     title={BDD100K: A Diverse Driving Video Database with Scalable Annotation Tooling},
     author={Yu, Fisher and Chen, Haofeng and Wang, Xin and Xian, Wenqi and Chen, Yingying and Liu, Fangchen and Madhavan, Vashisht and Darrell, Trevor},
     journal={arXiv preprint arXiv:1805.04687},
     year={2018}
   }
   ```