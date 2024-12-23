# Multi-Modal Lunch Calorie Prediction

A deep learning system that predicts lunch calorie intake using multiple data modalities including CGM (Continuous Glucose Monitoring) data, demographic information, and food images.

## Features

- Multi-modal fusion of different data types:
  - CGM time series data
  - Demographic and medical information
  - Food images (breakfast and lunch)
- Attention-based architecture for optimal feature combination
- Comprehensive data preprocessing pipeline
- Custom loss function for calorie prediction

## Architecture

### Data Preprocessing
- CGM data: Feature extraction from time series including mean, std, min/max glucose levels
- Demographic data: One-hot encoding for categorical variables, standardization for numerical features
- Image data: Resizing, color space conversion, and normalization
- PCA reduction for high-dimensional features

### Model Components
- CGM Encoder: LSTM-based network for processing glucose data
- Image Encoder: CNN architecture for food image processing
- Demographic Encoder: Multi-layer perceptron for patient data
- Multi-head Attention Fusion: Combines features from different modalities

## Requirements

```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
```

## Usage

1. Prepare your data files:
   - CGM data (csv)
   - Demographic data (csv)
   - Food images
   - Calorie labels (csv)

2. Initialize the preprocessor:
```python
preprocessor = DataPreprocessor()
```

3. Create datasets:
```python
train_dataset = MultimodalDataset(
    cgm_file='cgm_train.csv',
    demo_file='demo_viome_train.csv',
    img_file='img_train.csv',
    label_file='label_train.csv',
    preprocessor=preprocessor,
    is_train=True
)
```

4. Train the model:
```python
optimizer = optim.Adam(parameters, lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
```

## Performance

The model achieves competitive performance in calorie prediction using Root Mean Squared Relative Error (RMSRE) as the evaluation metric. Early stopping and learning rate scheduling are implemented for optimal training.

## License

This project is available under standard open-source licensing terms.
