import streamlit as st
import cv2
import numpy as np
import torch
import os
import requests
from PIL import Image

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

st.set_page_config(page_title="AI Image Restoration", layout="wide")

st.title("AI Image Restoration Platform")
st.write("Enhance blurry and low-resolution images using AI")

# -----------------------
# MODEL SETUP
# -----------------------

os.makedirs("models", exist_ok=True)

model_path = "models/RealESRGAN_x4plus.pth"

def download_model():

    if os.path.exists(model_path):
        return

    st.info("Downloading AI model... (first run only)")

    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

    r = requests.get(url, stream=True)

    with open(model_path, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)

download_model()

# -----------------------
# LOAD MODEL
# -----------------------

@st.cache_resource
def load_model():

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4
    )

    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=128,         # memory safe
        tile_pad=10,
        pre_pad=0,
        half=False,
        device="cpu"
    )

    return upsampler

model = load_model()

# -----------------------
# IMAGE UPLOAD
# -----------------------

uploaded = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

if uploaded:

    image = Image.open(uploaded)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(image)

    if st.button("Enhance Image"):

        with st.spinner("Enhancing image..."):

            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            output, _ = model.enhance(img, outscale=4)

            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

            with col2:
                st.subheader("Enhanced Image")
                st.image(output)

            _, buffer = cv2.imencode(".png", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

            st.download_button(
                "Download Enhanced Image",
                buffer.tobytes(),
                "enhanced.png"
            )
