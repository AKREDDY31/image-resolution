import streamlit as st
import numpy as np
from PIL import Image, ImageFilter

st.set_page_config(page_title="AI Image Restoration", layout="wide")

st.title("AI Image Restoration Platform")
st.write("Enhance blurry and low-resolution images using AI")

uploaded_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        st.image(image)

    if st.button("Enhance Image"):

        with st.spinner("Enhancing image..."):

            # upscale image
            width, height = image.size
            enhanced = image.resize((width*2, height*2), Image.BICUBIC)

            # sharpen image
            enhanced = enhanced.filter(ImageFilter.SHARPEN)

            with col2:
                st.subheader("Enhanced Image")
                st.image(enhanced)

            st.download_button(
                "Download Enhanced Image",
                enhanced.tobytes(),
                file_name="enhanced_image.png"
            )
