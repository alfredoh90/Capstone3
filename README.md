# Safecheck AI: Using Convolutional Neural Networks to Improve Safety at Warehouses

### Project Overview
**Safecheck AI** is a proof-of-concept designed to minimize the number of employees entering a warehouse floor without the appropriate safety gear, such as helmets and vests. By detecting when employees forget to wear this equipment, the system alerts the safety team, helping prevent accidents and ensuring compliance with safety standards.

### Target Users
This solution is intended for safety officers and warehouse management who are responsible for ensuring employee safety on the warehouse floor.

### System Requirements
To run Safecheck AI locally, the following Python libraries are required:
- numpy
- keras
- tensorflow
- PIL
Ensure you have these installed before proceeding.

### Data
The training data for Safecheck AI consists of images sourced from three open Kaggle datasets:

- [Employee Wearing Safety Gear](https://www.kaggle.com/datasets/khananikrahman/is-an-employee-wearing-safety-gear)
- [Safety Helmet and Reflective Jacket](https://www.kaggle.com/datasets/niravnaik/safety-helmet-and-reflective-jacket}
- [Negative Dataset](https://www.kaggle.com/datasets/ahmadahmadzada/images2000)

### Pre Processing
The images were preprocessed by resizing and rescaling, and data augmentation techniques such as random rotation, brightness, and contrast adjustments were applied to enhance model performance and minimize memorization of the training dataset.

### Model Architecture
The model used in Safecheck AI is a custom Convolutional Neural Network (CNN) with five convolutional layers. Below is a summary of the architecture:

| Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ sequential (Sequential)              │ (None, 400, 400, 3)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_5 (Conv2D)                    │ (None, 400, 400, 32)        │             896 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation_6 (Activation)            │ (None, 400, 400, 32)        │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_5                │ (None, 400, 400, 32)        │             128 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_5 (MaxPooling2D)       │ (None, 200, 200, 32)        │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_6 (Dropout)                  │ (None, 200, 200, 32)        │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_6 (Conv2D)                    │ (None, 200, 200, 64)        │          51,264 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation_7 (Activation)            │ (None, 200, 200, 64)        │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_6                │ (None, 200, 200, 64)        │             256 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_6 (MaxPooling2D)       │ (None, 100, 100, 64)        │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_7 (Dropout)                  │ (None, 100, 100, 64)        │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_7 (Conv2D)                    │ (None, 100, 100, 128)       │          73,856 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation_8 (Activation)            │ (None, 100, 100, 128)       │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_7                │ (None, 100, 100, 128)       │             512 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_7 (MaxPooling2D)       │ (None, 50, 50, 128)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_8 (Dropout)                  │ (None, 50, 50, 128)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_8 (Conv2D)                    │ (None, 50, 50, 256)         │         819,456 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation_9 (Activation)            │ (None, 50, 50, 256)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_8                │ (None, 50, 50, 256)         │           1,024 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_8 (MaxPooling2D)       │ (None, 25, 25, 256)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_9 (Dropout)                  │ (None, 25, 25, 256)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_9 (Conv2D)                    │ (None, 25, 25, 512)         │       1,180,160 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation_10 (Activation)           │ (None, 25, 25, 512)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ batch_normalization_9                │ (None, 25, 25, 512)         │           2,048 │
│ (BatchNormalization)                 │                             │                 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_9 (MaxPooling2D)       │ (None, 13, 13, 512)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_10 (Dropout)                 │ (None, 13, 13, 512)         │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten_1 (Flatten)                  │ (None, 86528)               │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 2048)                │     177,211,392 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ activation_11 (Activation)           │ (None, 2048)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout_11 (Dropout)                 │ (None, 2048)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_3 (Dense)                      │ (None, 2)                   │           4,098 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘

The model does not use any pre-trained architectures. Instead, it was built from scratch using five convolutional layers, batch normalization, and max pooling, followed by dense layers for classification.

### Model Training
The model was trained on a dataset containing 12,000 images (positive and negative samples). The following settings were used:

Loss function: binary_crossentropy
Optimizer: Adam
Training duration: 70 epochs
Performance metric: Accuracy
Data augmentation techniques were applied to increase model robustness, and the ** final training accuracy was 93.79%, with a validation accuracy of over 90% **.

### Usage
Safecheck AI runs as a web-based application using Streamlit. The app accesses the webcam to capture images every second, analyzes them in real-time, and provides continuous feedback on whether the person is wearing the appropriate safety gear.

If the person is detected without a hard hat or vest, the app alerts the user with a notification.
Users can also manually capture images from the webcam, which are saved with a timestamp and the model’s prediction.

### Performance
The model achieved an overall accuracy of 93.79% on the training set and over 90% on the validation set. This shows strong performance in detecting whether a person is wearing the required safety gear.

### Heatmaps and Visualizations
The following image shows a side-by-side comparison of the original image, the heatmap showing areas of focus, and the model’s prediction:

### Deployment
At this stage, Safecheck AI is a proof of concept and has not yet been deployed. Future plans include deploying the model in a live warehouse environment and implementing a re-training schedule to continuously improve the model’s accuracy based on locally acquired data.

### Future Improvements
One potential next step for Safecheck AI is to deploy the model in a warehouse, where it can continuously monitor safety compliance. Over time, the model can be re-trained with new data collected from the local environment to improve its accuracy and adapt to specific warehouse conditions. Re-training could occur on a schedule of 1, 3, and 6 months.
