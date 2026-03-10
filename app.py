import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import requests

st.set_page_config(page_title="AI Image Restoration", layout="wide")

st.title("AI Image Restoration Platform")
st.write("Enhance blurry and low-resolution images using AI")

# ---------------- MODEL SETUP ----------------

os.makedirs("models", exist_ok=True)
model_path = "models/FSRCNN_x4.pb"

MODEL_URL = "https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/master/models/FSRCNN_x4.pb"
EXPECTED_MIN_SIZE = 1_000_000  # ~1MB sanity check

def download_model():
    need_download = True
    if os.path.exists(model_path):
        size = os.path.getsize(model_path)
        if size > EXPECTED_MIN_SIZE:
            need_download = False
        else:
            os.remove(model_path)

    if need_download:
        st.write("Downloading enhancement model...")
        with requests.get(MODEL_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

download_model()

# ---------------- LOAD MODEL ----------------

try:
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("fsrcnn", 4)
except Exception as e:
    st.error("Model failed to load. The file may be corrupted.")
    st.stop()

# ---------------- UI ----------------

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

            # mild sharpening
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
