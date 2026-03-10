import os
import cv2
import numpy as np
import torch
import requests
import gradio as gr
from PIL import Image

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# -----------------------
# Model download
# -----------------------

os.makedirs("models", exist_ok=True)
model_path = "models/RealESRGAN_x4plus.pth"

def download_model():
    if os.path.exists(model_path):
        return

    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    print("Downloading RealESRGAN model...")

    r = requests.get(url, stream=True)

    with open(model_path, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)

download_model()

# -----------------------
# Load AI model
# -----------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    tile=128,
    tile_pad=10,
    pre_pad=0,
    half=False,
    device=device
)

# -----------------------
# Image enhancement
# -----------------------

def enhance_image(image):

    if image is None:
        return None

    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    output, _ = upsampler.enhance(img, outscale=4)

    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

    return Image.fromarray(output)

# -----------------------
# Gradio UI
# -----------------------

title = "AI Image Restoration Platform"

description = """
Upload a blurry or low-resolution image and enhance it using RealESRGAN AI.
"""

interface = gr.Interface(
    fn=enhance_image,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Image(type="pil", label="Enhanced Image"),
    title=title,
    description=description
)

interface.launch()
