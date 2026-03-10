import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import requests

st.set_page_config(page_title="AI Image Restoration", layout="wide")

st.title("AI Image Restoration Platform")
st.write("Enhance blurry and low-resolution images using AI")

# ---------- MODEL SETUP ----------

os.makedirs("models", exist_ok=True)
model_path = "models/FSRCNN_x4.pb"

def download_model():
    if os.path.exists(model_path):
        return
    
    st.write("Downloading enhancement model...")
    url = "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/FSRCNN_x4.pb"
    
    r = requests.get(url, stream=True)
    with open(model_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

download_model()

# ---------- LOAD SUPER RESOLUTION MODEL ----------

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(model_path)
sr.setModel("fsrcnn", 4)

# ---------- UI ----------

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(img)

    if st.button("Enhance Image"):
        with st.spinner("Enhancing image..."):

            result = sr.upsample(img)

            # Optional sharpening for better clarity
            kernel = np.array([[0,-1,0],
                               [-1,5,-1],
                               [0,-1,0]])
            result = cv2.filter2D(result, -1, kernel)

            with col2:
                st.subheader("Enhanced Image")
                st.image(result)

            _, buffer = cv2.imencode(".png", result)

            st.download_button(
                "Download Enhanced Image",
                buffer.tobytes(),
                "enhanced_image.png"
            )
