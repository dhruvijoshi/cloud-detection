# 🌤️ Cloud Detection in Sentinel-2 Satellite Imagery

A deep learning solution for automated cloud detection in multi-spectral Sentinel-2 satellite imagery using PyTorch and advanced segmentation techniques.


##  Overview

This project implements a state-of-the-art semantic segmentation model to accurately identify clouds in Sentinel-2 satellite imagery. Built with a U-Net architecture and pre-trained ResNet34 encoder, the model processes 4-band multi-spectral data (RGB + Near-Infrared) to generate precise cloud masks.

## Features

- **Multi-spectral Processing**: Handles 4-channel Sentinel-2 imagery (RGB + NIR)
- **Deep Learning Architecture**: U-Net with ResNet34 backbone for superior performance
- **Interactive Web App**: Gradio-powered interface for real-time inference
- **Comprehensive Metrics**: Precision, recall, F1-score, and IoU evaluation
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## Model Architecture

### Technical Specifications
- **Backbone**: ResNet34 (pre-trained on ImageNet)
- **Input Channels**: 4 (Red, Green, Blue, Near-Infrared)
- **Output**: Binary segmentation mask (cloud vs. no cloud)
- **Activation**: Sigmoid for binary classification
- **Framework**: PyTorch with `segmentation_models.pytorch`

### Training Configuration
- **Loss Function**:  Dice Loss
- **Optimizer**: Adam with learning rate scheduling


## Dataset

Trained on a curated subset of the [Sentinel-2 Cloud Cover Detection dataset](https://www.kaggle.com/datasets/willkoehrsen/sentinel2-drivendata-cloud-cover) containing:
- **1,000 training chips** for model development
- **200 test chips** for validation
- Multi-spectral 4-band imagery (RGB + NIR)

## Performance

| Metric    | Training | Validation |
|-----------|----------|------------|
| Accuracy  | 0.95     | 0.93       |
| Precision | 0.92     | 0.89       |
| Recall    | 0.88     | 0.85       |
| F1-Score  | 0.90     | 0.87       |
| IoU       | 0.82     | 0.78       |

## Cloud Detection Web Application

Interactive Gradio-based web interface featuring:

### Core Functionality
- **Easy Upload**: Drag-and-drop satellite image interface
- **Real-time Processing**: Instant cloud detection predictions
- **Export Results**: Download predictions for further analysis
- **Visual Comparison**: Side-by-side input and prediction views


## Acknowledgments

- **[Sentinel-2 Cloud Cover Detection Dataset](https://www.kaggle.com/datasets/willkoehrsen/sentinel2-drivendata-cloud-cover)** - Training data
- **[segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)** - Model architecture framework
- **European Space Agency - ESA** - Sentinel-2 satellite imagery