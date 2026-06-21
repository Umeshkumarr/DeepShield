# DeepShield – Deepfake Detection System

## Overview

DeepShield is an AI-powered Deepfake Detection System designed to identify manipulated images and videos using deep learning and computer vision techniques. The system leverages a MobileNetV2-based model trained on real and fake media data to provide accurate predictions through a user-friendly web interface.

## Features

* Detects deepfakes in both images and videos.
* Deep learning model based on MobileNetV2 with transfer learning.
* Flask-powered web application.
* Real-time image and video analysis.
* Simple and intuitive user interface.

## Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* Flask
* NumPy

## Model Architecture

* MobileNetV2 (Pretrained on ImageNet)
* Global Average Pooling Layer
* Dense Layer (ReLU Activation)
* Output Layer (Sigmoid Activation)

## Project Structure

```
DeepShield/
│
├── model/
│   └── deepfake_model.h5
├── static/
├── templates/
├── dataset/
│   ├── real/
│   └── fake/
├── app.py
├── train.py
└── README.md
```

## How It Works
1. Upload an image or video.
2. The system preprocesses the media and extracts frames (for videos).
3. The trained MobileNetV2 model analyzes the content.
4. The application displays whether the media is Real or Fake.

## Future Enhancements
* Improved model accuracy using larger datasets.
* Face localization and attention visualization.
* Support for additional deepfake generation techniques.
* Cloud deployment for public access.

## Author
Umesh Kumar

