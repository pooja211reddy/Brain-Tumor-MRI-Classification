# 🧠 Brain Tumor MRI Classification  
### 🚀 Deep Learning + Transfer Learning + Streamlit Deployment

---

## 🌟 Project Summary

This project builds an **end-to-end AI system** to classify brain MRI scans into tumor categories using deep learning.  

It combines:
- 🧠 Custom CNN architecture
- 🚀 Transfer Learning (ResNet50)
- 🌐 Interactive Streamlit Web App

👉 Goal: Assist in **early detection of brain tumors** through automated image classification.

---

## 🎯 Problem Statement

Manual MRI diagnosis is:
- Time-consuming ⏳  
- Prone to human error ⚠️  
- Requires expert radiologists 👨‍⚕️  

This project aims to:
> **Automate tumor classification with high accuracy and real-time usability**

---

## 🧬 Classes Predicted

| Label | Description |
|------|------------|
| 🧠 Glioma | Malignant tumor |
| 🧠 Meningioma | Typically benign |
| 🧠 Pituitary | Hormonal tumor |
| ✅ No Tumor | Healthy brain |

---

## 🏗️ System Architecture

```
MRI Image
   ↓
Preprocessing (Resize + Normalize)
   ↓
Deep Learning Model (ResNet / CNN)
   ↓
Softmax Probabilities
   ↓
Prediction + Confidence
   ↓
Streamlit UI
```

---

## 🧪 Models Implemented

### 🔹 Custom CNN
- 3 Conv Blocks (Conv + BN + ReLU + Pool)
- Dropout Regularization
- Fully Connected Layers

👉 Pros:
- Lightweight
- Fully controlled architecture

👉 Cons:
- Lower generalization vs pretrained models

---

### 🔹 ResNet50 (Transfer Learning) ⭐
- Pretrained on ImageNet
- Fine-tuned top layers
- Deep residual connections

👉 Pros:
- Faster convergence
- Superior feature extraction
- Higher accuracy

---

## 📊 Results & Performance

### ✅ Final Model: **ResNet50**

| Metric        | Score |
|--------------|------|
| Accuracy      | ~96–97% |
| Precision     | High across all classes |
| Recall        | Strong detection capability |
| F1-score      | Balanced performance |

---

## 📉 Confusion Matrix Insights

- Strong diagonal dominance → accurate predictions  
- Minor confusion:
  - Glioma ↔ Meningioma  
- Near-perfect detection:
  - Pituitary tumors  

---

## 📈 Training Behavior

- Smooth loss convergence  
- No severe overfitting  
- Stable validation accuracy  

👉 Indicates **good generalization**

---

## 🧠 Key Learnings

- Transfer learning drastically improves performance  
- Data preprocessing consistency is critical  
- Class imbalance affects recall  
- Deep models capture subtle MRI features  

---

## 🌐 Streamlit Web App

### 🔥 Features

- 📤 Upload MRI image  
- 🎯 Predict tumor type  
- 📊 Show confidence score  
- 📉 Display confusion matrix  
- 📋 Show classification report  

---

## 🚀 How to Run

### 1️⃣ Clone repo
```bash
git clone https://github.com/your-username/brain-tumor-classifier.git
cd brain-tumor-classifier
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Launch app
```bash
streamlit run tumor_app.py
```

---

## 📦 Tech Stack

- 🧠 PyTorch
- 🖼️ Torchvision
- 📊 Scikit-learn
- 🌐 Streamlit
- 📈 Matplotlib & Seaborn

---

## ⚙️ Model Pipeline

```
Image → Transform → Tensor → Model → Softmax → Prediction
```

---

## ⚠️ Limitations

- Sensitive to image quality  
- Slight confusion between similar tumor types  
- Requires curated dataset  

---

## 🔮 Future Enhancements

- 🔥 Grad-CAM (tumor localization)
- 📱 Mobile deployment
- ☁️ Cloud hosting (AWS / Streamlit Cloud)
- 🧠 Model ensemble (CNN + ResNet)
- 📊 Explainable AI integration

---

## 💡 Real-World Impact

This system can:
- Assist radiologists 👨‍⚕️  
- Speed up diagnosis ⏱️  
- Reduce human error ❌  
- Enable early detection 🧠  

---

## 👩‍💻 Author

**Pooja Reddy Nedhunuri**  
Capstone Project — Deep Learning  

---

## ⭐ If you like this project

Give it a ⭐ and share!
