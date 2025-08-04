import streamlit as st
import numpy as np
import cv2

import os
from tensorflow.keras.models import load_model
import kagglehub

# -------------------------------------
# 📌 Load model
model = load_model("covid_vgg16_finetuned.h5")
class_names = ['Normal', 'Viral Pneumonia', 'COVID-19']
IMG_SIZE = 128

# -------------------------------------
# 📦 Download dataset using kagglehub
@st.cache_resource
def get_dataset_path():
    path = kagglehub.dataset_download("pranavraikokte/covid19-image-dataset")
    return os.path.join(path, "Covid19-dataset", "train")

train_folder = get_dataset_path()

# -------------------------------------
# 📂 Load image file paths
image_paths = {
    "COVID-19": os.listdir(os.path.join(train_folder, "Covid")),
    "Normal": os.listdir(os.path.join(train_folder, "Normal")),
    "Viral Pneumonia": os.listdir(os.path.join(train_folder, "Viral Pneumonia"))
}

# -------------------------------------
# 🎯 Streamlit UI
st.title("🩺 COVID-19 Chest X-ray Classifier")
st.markdown("Model: **Fine-tuned VGG16** on 3 classes: Normal, Viral Pneumonia, COVID-19.")

# Choose class and image
selected_class = st.selectbox("Choose a Class", list(image_paths.keys()))
selected_image = st.selectbox("Choose an Image", image_paths[selected_class])

# Load and preprocess the image
img_path = os.path.join(train_folder, selected_class, selected_image)
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
img_input = img_resized / 255.0
img_input = np.expand_dims(img_input, axis=0)

# Predict
if st.button("Predict"):
    pred = model.predict(img_input)[0]
    pred_class = class_names[np.argmax(pred)]
    confidence = np.max(pred) * 100

    # Show image and prediction
    st.image(img_rgb, caption=f"Selected: {selected_class}", use_column_width=True)
    st.markdown(f"### 🧠 Predicted: **{pred_class}**")
    st.markdown(f"Confidence: **{confidence:.2f}%**")
