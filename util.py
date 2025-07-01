from PIL import Image
import numpy as np
import streamlit as st
import base64

def classify(image, model, class_names):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction[0])
    confidence = prediction[0][class_idx]
    if confidence < 0.5:
        class_idx = 1
        confidence = 1 - confidence
    return class_names[class_idx], confidence

def set_background(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{b64}");
        background-size: cover;
    }}
    </style>
    """, unsafe_allow_html=True)
