import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

MODEL_PATH = 'person_x_classifier.h5'
model = tf.keras.models.load_model(MODEL_PATH)
IMG_SIZE = (128, 128)

st.title("👤 Person X Image Classifier")
st.write("Upload an image and find out if it contains **Person X**.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)


    st.write("🧠 Analyzing...")
    image_array = preprocess_image(image)
    prediction = model.predict(image_array)[0][0]

    if prediction > 0.5:
        st.success("✅ This image contains Vishwanath.")
    else:
        st.error("❌ This image does NOT contain Vishwanath.")
