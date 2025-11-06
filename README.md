# 🔥 Facial Emotion Recognition (FER-2013) with PyTorch ✨

Recognize emotions from facial expressions with a powerful PyTorch CNN! This project demonstrates an end-to-end computer vision pipeline, from data preparation and augmentation to building, training, and evaluating a deep learning model for real-world emotion classification.



---

## 🗂 Dataset: FER-2013 Emotions 😔😄😠

This project utilizes the **Facial Emotion Recognition (FER-2013) Dataset** from Kaggle, a collection of $48 \times 48$ pixel grayscale face images.

**Source:** [Kaggle: Facial Emotion Recognition (FER-2013) Dataset](https://www.kaggle.com/datasets/msambare/fer2013)

### 📊 Emotion Categories & Mapping

The model is trained to classify images into one of the 7 fundamental emotions:

| Index | Emotion | Emoji |
| :---: | :---: | :---: |
| 0 | Anger | 😡 |
| 1 | Disgust | 🤢 |
| 2 | Fear | 😨 |
| 3 | Happy | 😄 |
| 4 | Sadness | 😢 |
| 5 | Surprise | 😮 |
| 6 | Neutral | 😐 |

---

## 🧹 Data Magic: Preprocessing & Augmentation ✨

The data pipeline uses `torchvision` transforms for robust training:

* **📏 Resizing:** Images are scaled up to $128 \times 128$ pixels.
* **⚫ Grayscale:** Ensuring single-channel input.
* **📈 Normalization:** Standardizing pixel values $(\text{mean}=0.5, \text{std}=0.5)$.
* **🔄 Augmentation:** Includes **Random Horizontal Flips**, **Rotations**, **Color Jitter**, and **Affine Transforms** applied *only* to the training set to boost generalization.

---

## 🧠 The Brain: `FER_Model0` CNN Architecture 🚀

The model is a custom-built, three-block **Convolutional Neural Network (CNN)**.

* **Architecture:** It consists of three sequential blocks, each featuring two convolutional layers, **Batch Normalization**, **ReLU** activation, and a **Max Pooling** layer.
* **Classifier Head:** The extracted features are flattened and passed to a **Linear Layer** to output scores for the 7 emotion classes.
* **Hardware:** Optimized for **GPU (CUDA)** acceleration when available.

---

## 🏋️ Training Journey: Learning Emotions 📈

The model was trained for 20 epochs using standard deep learning practices:

* **Loss Function:** `CrossEntropyLoss` (ideal for multi-class tasks).
* **Optimizer:** `Adam` with a learning rate of $0.001$.
* **Batch Size:** $32$.

### 📊 Performance Milestones

| Epoch | Train Loss | Train Accuracy | Test Loss | Test Accuracy |
| :---: | :---: | :---: | :---: | :---: |
| **0** | `2.0459` | `24.48%` | `1.6876` | `33.83%` |
| **5** | `1.2914` | `51.17%` | `1.1750` | `55.42%` |
| **10** | `1.1482` | `56.55%` | `1.1138` | `57.55%` |
| **15** | `1.0866` | `58.96%` | `1.0498` | `60.66%` |
| **19** | `1.0422` | **`61.09%`** | `1.0379` | **`61.38%`** |

---

## 🎯 Evaluation & Visual Predictions 🖼️

* **Metrics:** Accuracy and Loss are tracked at every step to monitor learning.
* **Visual Confidence:** Predictions on a sample of test images are plotted, clearly showing the **predicted emotion** versus the **true emotion**. Correct predictions are highlighted in **💚 GREEN** and incorrect ones in **💔 RED**, providing an immediate, intuitive view of model performance.

---

## 🌟 Key Takeaways & Highlights

* **🚀 End-to-End CV Pipeline:** Fully working PyTorch computer vision project.
* **🏗️ Custom CNN:** Implementation of a simple, effective deep learning architecture.
* **🎯 Data-Driven:** Robust use of data augmentation for better generalization and stability.
* **✨ Clarity & Reproducibility:** Modular code design for easy understanding and future adaptation.
