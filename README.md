# 🩺 Multi-Modal Chest X-ray Classification

This project explores a multi-modal deep learning approach to classify thoracic diseases using **chest X-ray images** and their associated **radiology reports**. It combines visual and textual data streams to enhance diagnostic performance. The textual data is processed using custom tokenization and embedded via a feedforward neural network , while images are processed using a ResNet model.

---
## 📁 Dataset
- **Contents**:
  - Chest X-ray images
  - Corresponding textual clinical reports
- **Text Reports**:
  - Extracted, cleaned, and structured into CSV format
  - Each row includes: image path, tokenized text, and corresponding labels
---

## ⚙️ Preprocessing

- **Images**:
  - Loaded in grayscale
  - Resized to 224×224
  - Normalized using standard values

- **Text**:
  - Lowercased and tokenized using `Keras Tokenizer`
  - Converted to sequences and padded to a fixed length (100 tokens)
  - Final input: fixed-length numeric vector (not word embeddings or contextualized vectors)

- **Labels**:
  - Extracted from the CSV for binary classification tasks (e.g., normal vs. abnormal)

---

## 🧠 Model Architecture & Training
![Screenshot 2025-05-09 221731](https://github.com/user-attachments/assets/dd9f32ef-3e3c-4bb8-9faf-e263c380daef)

### Image Encoder
- **ResNet-18** (pretrained)
- Final output: 512-dimensional feature vector

### Text Encoder
- Simple **Feedforward Neural Network**:
  - Input: 100-dimensional token sequence
  - Layers: Linear → ReLU → Linear (128 → 64 output)

### Fusion & Classification
- Concatenation of image and text features
- Final classifier: Linear → Sigmoid (for binary classification)

## 📊 Evaluation Metrics
Classification Report:
              precision    recall  f1-score   support

           0       0.87      0.85      0.86        61
           1       0.88      0.89      0.88        72
           
    accuracy                           0.87       133


-Precision: 0.8721
-Recall: 0.8722
-F1-Score: 0.8721
-Test Accuracy: 0.8722

# confusion matrix
![image](https://github.com/user-attachments/assets/dd8d66ac-da3f-41ae-9c41-34fd5e1e827e)

# ROC curve

![image](https://github.com/user-attachments/assets/bcb3fb72-5bed-41ac-8e58-ef4254d3d295)

---
## 🔍 XAI Usage

- **Grad-CAM** is applied to visualize class-specific activation regions from the image encoder (ResNet-18).
---

## 🚀 App screenshots

![image](https://github.com/user-attachments/assets/d701aa89-39eb-4672-a99d-74966eee65c1)

![Screenshot 2025-05-04 193708](https://github.com/user-attachments/assets/b86b02e5-187a-4ab6-9565-7cfa7efd5d3d)


