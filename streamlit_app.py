import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import os

st.set_page_config(page_title="Bug Bite Classifier", layout="centered")

# =====================================================
# Load Model from HuggingFace
# =====================================================
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="AdityaHK/BugBiteStack_v2",
        filename="stacked_model_4_backbone_multigpu.keras"
    )
    model = tf.keras.models.load_model(model_path)
    return model


# =====================================================
# Prediction Function
# =====================================================
def predict_class(image, model):

    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, [331, 331])   # IMPORTANT (NASNet uses 331)
    image = image / 255.0                       # Normalize
    image = tf.expand_dims(image, axis=0)

    prediction = model.predict(image)
    return prediction


# =====================================================
# UI
# =====================================================
st.title("🦟 Bug Bite Classifier")
st.write("Upload an image to classify the type of bug bite.")

model = load_model()

file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

class_names = [
    "ants",
    "bed bugs",
    "chiggers",
    "fleas",
    "mosquitos",
    "no bites",
    "spiders",
    "ticks"
]

if file is None:
    st.info("Waiting for image upload...")
else:
    image = Image.open(file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Running inference..."):
        prediction = predict_class(np.array(image), model)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100

    st.success(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: {confidence:.2f}%")
