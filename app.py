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
    
    st.write("Downloading AI enhancement model...")

    url = "https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/master/models/FSRCNN_x4.pb"

    r = requests.get(url)

    with open(model_path, "wb") as f:
        f.write(r.content)

download_model()

# ---------- LOAD MODEL ----------

sr = cv2.dnn_superres.DnnSuperResImpl_create()

if os.path.exists(model_path):

    sr.readModel(model_path)
    sr.setModel("fsrcnn", 4)

else:
    st.error("Model download failed")
    st.stop()

# ---------- UI ----------

uploaded_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

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

            # small sharpening for clarity
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
