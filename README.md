# 😷 Face Mask Detection Using CNN & OpenCV

## 📌 Project Overview

Face Mask Detection is a Deep Learning and Computer Vision project that detects human faces in real time and classifies whether a person is wearing a face mask or not.

The system uses a Convolutional Neural Network (CNN) trained with face mask images and OpenCV for real-time face detection through a webcam.

The application displays a bounding box around detected faces along with the predicted class and confidence percentage.

---

## 🎯 Objectives

- Detect human faces in real time using a webcam.
- Classify faces into **MASK** and **NO MASK** categories.
- Use a CNN model for image classification.
- Display prediction confidence.
- Provide a real-time computer vision application.
- Demonstrate the practical use of Deep Learning with OpenCV.

---

## 🚀 Features

- ✅ Real-time face detection
- ✅ Mask / No Mask classification
- ✅ CNN-based Deep Learning model
- ✅ OpenCV webcam integration
- ✅ Confidence percentage
- ✅ Bounding boxes around detected faces
- ✅ Multiple face detection
- ✅ FPS display
- ✅ Real-time prediction

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow | Deep Learning framework |
| Keras | CNN model development |
| OpenCV | Computer Vision and webcam processing |
| NumPy | Numerical operations |
| Pandas | Dataset and annotation processing |
| Matplotlib | Training visualization |
| CNN | Image classification |

---

## 🧠 Machine Learning Approach

The project uses a **Convolutional Neural Network (CNN)** for binary image classification.

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
