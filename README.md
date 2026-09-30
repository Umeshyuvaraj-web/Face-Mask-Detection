# 😷 Face Mask Detection Using CNN, OpenCV & Streamlit

## 📌 Project Overview

Face Mask Detection is a Deep Learning and Computer Vision project that detects human faces and classifies whether a person is wearing a face mask or not.

The project uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras for mask classification and OpenCV for face detection.

A Streamlit web application is also included, allowing users to upload an image and receive a mask detection result with confidence percentage.

The project also supports real-time webcam-based face mask detection.

---

## 🎯 Objectives

The main objectives of this project are:

- Detect human faces from images and webcam frames.
- Classify detected faces into MASK and NO MASK categories.
- Build and train a CNN-based image classification model.
- Perform real-time face mask detection using OpenCV.
- Develop a simple web interface using Streamlit.
- Display prediction confidence for detected faces.
- Demonstrate the practical application of Deep Learning and Computer Vision.

---

## 🚀 Features

- 😷 MASK / NO MASK classification
- 📷 Real-time webcam detection
- 🖼️ Image upload detection
- 🤖 CNN-based Deep Learning model
- 👤 Face detection using OpenCV
- 📊 Prediction confidence percentage
- 👥 Multiple face detection
- ⚡ Real-time processing
- 🌐 Streamlit web interface
- 📦 Trained Keras model
- 🎯 Bounding boxes around detected faces

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow | Deep Learning framework |
| Keras | CNN model development |
| OpenCV | Face detection and image processing |
| Streamlit | Web application interface |
| NumPy | Numerical operations |
| Pandas | Dataset and annotation processing |
| Matplotlib | Data visualization |
| CNN | Image classification |

---

## 🧠 Machine Learning Approach

The project uses a Convolutional Neural Network (CNN) for binary image classification.

The input face image is resized to `128 × 128` pixels and normalized before being passed to the CNN model.

### CNN Architecture

```text
Input Image
     ↓
Resize to 128 × 128
     ↓
Convolutional Layer - 32 Filters
     ↓
Max Pooling
     ↓
Convolutional Layer - 64 Filters
     ↓
Max Pooling
     ↓
Convolutional Layer - 128 Filters
     ↓
Max Pooling
     ↓
Dropout
     ↓
Flatten
     ↓
Dense Layer - 128 Neurons
     ↓
Dropout
     ↓
Sigmoid Output
     ↓
MASK / NO MASK
