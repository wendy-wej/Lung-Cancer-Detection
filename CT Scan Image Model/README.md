---
license: mit
---

# Lung Cancer Detection Model

This repository contains a deep learning model for detecting lung cancer from CT scan images. The model is trained to classify CT images into three categories: **Benign**, **Malignant**, and **Normal**.

## Model Overview

The model architecture is a Convolutional Neural Network (CNN) built with TensorFlow and Keras. It uses multiple convolutional layers, pooling layers, batch normalization, and dropout for regularization. The final layer uses softmax activation to output probabilities for the three classes.

### Model Architecture:
- Input shape: (256, 256, 3) - Images resized to 256x256 pixels with 3 color channels (RGB).
- Convolutional layers with ReLU activation functions.
- Batch Normalization layers to stabilize learning.
- Pooling layers (Average and Max pooling).
- Dropout layers to reduce overfitting.
- Dense layers with ReLU and softmax activations.

## Dataset

The model was trained on the [IQ-OTHNCCD lung cancer dataset](https://www.kaggle.com/datasets/sfikas/iqothnccd-lung-cancer-dataset), which contains images classified into three categories: Benign, Malignant, and Normal cases.

### Data Augmentation:
- Rescaling of images (1./255).
- Augmentations like rotation, shift, and zoom can be optionally applied.

## Training

The model was trained with:
- Optimizer: SGD with a learning rate of 0.0001.
- Loss function: Categorical Crossentropy.
- Metrics: Accuracy.
- Early stopping and model checkpoints to prevent overfitting.

### Training Callbacks:
- Early Stopping: Monitors validation accuracy with patience of 15 epochs.
- ReduceLROnPlateau: Reduces learning rate by a factor of 0.5 if validation accuracy does not improve for 5 epochs.
- Model Checkpoint: Saves the best model based on validation accuracy.

## Evaluation

The model was evaluated on a separate test set with the following results:
- **Test Accuracy**: [insert test accuracy]
- **Test Loss**: [insert test loss]

### Confusion Matrix:
![Confusion Matrix](confusion_matrix.png)

## How to Use

### Load the Model
To load the model from the Hugging Face Hub, use the following code:

```python
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('https://huggingface.co/your_username/your_model_name/resolve/main/saved_model.pb')

# Use the model for predictions
predictions = model.predict(your_data)

