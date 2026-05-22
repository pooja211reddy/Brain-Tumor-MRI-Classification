import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torchvision import transforms, models
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# -------------------------------
# 1. Define classes
# -------------------------------
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 2. Load Model (ResNet)
# -------------------------------
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 4)

    model.load_state_dict(torch.load("/Users/Pooja/Documents/capstone_projects/projects/Tumor/resnet_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# -------------------------------
# 3. Image Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------------
# 4. Prediction Function
# -------------------------------
def predict(image):
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)

    return class_names[pred.item()]

# -------------------------------
# 5. UI
# -------------------------------
st.title("🧠 Brain Tumor MRI Classifier")

st.write("Upload an MRI image to predict tumor type")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

cm = np.load("/Users/Pooja/Documents/capstone_projects/projects/Tumor/cm.npy")
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

report_df = pd.read_csv("/Users/Pooja/Documents/capstone_projects/projects/Tumor/classification_report.csv", index_col=0)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        result = predict(image)
        st.success(f"Prediction: **{result.upper()}**")
        with st.expander("📊 View Confusion Matrix"):
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d",
                        xticklabels=class_names,
                        yticklabels=class_names)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
        
        with st.expander("📊 Classification Report"):
            st.dataframe(
                report_df.style
                .background_gradient(cmap="Blues")
                .format("{:.2f}")
            )






