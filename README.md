# Age and Gender Prediction Using CNNs

## Overview

This project implements a model for predicting age and gender from facial images using Convolutional Neural Networks (CNNs). The model is trained on the UTKFace dataset, which contains a diverse set of images labeled with age and gender. Two CNN models are developed: CNN-1, which represents the baseline model, and CNN-2, which is trained after hyperparameter tuning to improve performance.

### Project Contributors

This project was developed collaboratively by:
- @fatimacoding
- @AseelIsCoding
- @hdla22

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Additional Models](#additional-models)

## Requirements
- Python 3.x
- TensorFlow
- Keras
- OpenCV
- Matplotlib
- NumPy
- Pandas
- Scikit-learn

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/fatimacoding/CNN-Python.git
   cd CNN-Python
   pip install -r requirements.txt
2. Download the UTKFace dataset and place it in the data directory.
3. Run the Jupyter Notebook or Python script to start the training and evaluation process.

## Model Architecture

The main model is built using a Convolutional Neural Network (CNN) architecture designed for multi-task learning. The architecture consists of:

- **Convolutional Layers:** To extract features from images.
- **Max Pooling Layers:** To reduce the spatial dimensions.
- **Fully Connected Layers:** To perform final predictions for age and gender.
- **Dropout Layers:** To prevent overfitting.

The model has two outputs:

- **Gender Classification:** Using binary crossentropy loss.
- **Age Regression:** Using mean absolute error (MAE) loss.

### Summary of Architecture:
- **Input Shape:** (128, 128, 1)
- **Convolutional Layers:** 4 layers with increasing filters (32, 64, 128, 256)
- **Fully Connected Layers:** 2 layers with 128 neurons each
- **Output Layers:**
  - **Gender:** Sigmoid activation for binary classification
  - **Age:** ReLU activation for regression

## Results

The CNN model achieved impressive results in both gender classification and age regression tasks. Below are visualizations depicting the model's performance.

### Sample Predictions

Here are some sample predictions made by the model:

![cnn-2 sample](https://github.com/user-attachments/assets/0e29de30-0f6f-43ee-8407-90570175aa0d)


### Gender Prediction Accuracy
![Gender Prediction Accuracy](https://github.com/user-attachments/assets/d1ddc9d3-511b-479f-9fe3-40db6007f942)

This graph shows the training and validation accuracy for the gender classification task over the epochs. The model demonstrates a strong learning curve, indicating effective training.

### Age Prediction Mean Absolute Error
![Age Prediction Mean Absolute Error](https://github.com/user-attachments/assets/7f2c5d4e-fd3d-443f-94d5-b38d1129aa30)

The plot illustrates the training and validation mean absolute error (MAE) for age prediction. The decreasing trend in error indicates that the model is learning to predict age more accurately over time.

### CNN Models Performance

| Model   | Accuracy | Mean Squared Error (MSE) |
|---------|----------|----------------------------|
| CNN-1   | 85.13%   | 46.09                      |
| CNN-2   | 95.43%   | 44.17                      |

## Additional Models

In addition to the CNN, the project explores other models for age and gender prediction:

- **k-Nearest Neighbors (k-NN):**
  - Used for both gender classification and age regression.

- **Logistic Regression:**
  - Implemented for gender classification with a simple linear model approach.

- **Linear Regression:**
  - Used for age prediction.

Each model's performance is evaluated, and results are compared to the CNN model, providing insights into their effectiveness for this task.

## Additional Models Performance

| Model       | Accuracy | Mean Absolute Error (MAE) |
|-------------|----------|---------------------------|
| k-Nearest Neighbors (k-NN) | 74.67%   | 11.97                     |
| Logistic & Linear Regression (L&L) | 84.54%   | 34.21                     |
