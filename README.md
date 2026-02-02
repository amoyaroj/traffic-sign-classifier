# 🤖 Traffic Sign Classifier
### Image Processing & Machine Learning Pipeline

This project features an end-to-end computer vision pipeline developed in Python to preprocess raw image data, extract geometric features, and classify traffic signs.

---

## 🚀 Project Overview
I developed a system capable of classifying images—specifically traffic signs—by extracting shape and color-based features and applying machine learning algorithms. 

### **Key Technical Workflow**
* **Preprocessing**: Normalization, 100x100 resizing, and linearization to standardize input data.
* **Feature Engineering**: Extracted HSV color profiles and implemented Sobel filters for high-precision edge detection.
* **Geometry Tracking**: Leveraged Hough Transforms to count lines and detect circular patterns.

[Image of machine learning pipeline flowchart]

---

## 📊 Model Performance
I evaluated three distinct models, with **Logistic Regression** proving the most effective for this specific dataset.

| Model | Accuracy | Methodology |
| :--- | :--- | :--- |
| **Logistic Regression** | **91.84%** | Probability-based classification via Sigmoid functions. |
| **KNN** | **91.00%** | Feature similarity and Euclidean distance. |
| **Decision Tree** | **90.21%** | Iterative splitting based on information gain. |

---

## 🛠️ Technical Stack
* **Language**: Python ![Python](https://img.shields.io/badge/python-3670A0?style=flat&logo=python&logoColor=ffdd54)
* **Libraries**: NumPy, Pandas, Matplotlib, Pillow (PIL), OpenCV.
* **Concepts**: Gradient Descent, Computer Vision, Signal Processing.

---

## 📂 Repository Structure
* `traffic_sign_classifier.py`: Main execution script containing preprocessing and ML models.

---

### 🎓 Academic Context
This project was completed as part of the **EN
