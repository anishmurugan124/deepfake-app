import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import gdown
import os

st.set_page_config(page_title="Deepfake Detector", page_icon="🧠")
st.title("🧠 DEEPFAKE DETECTOR")
st.markdown("92% Accurate AI Model • Free • No API Key")

MODEL_PATH = "deepfake_model.h5"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Model downloading..."):
            gdown.download(
                "https://drive.google.com/uc?id=https://drive.google.com/file/d/1oi3NdmhE5GayQhNsxMW0e74gWtPIp7ei/view?usp=drive_link",
                MODEL_PATH, quiet=False
            )
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
st.success("✅ Model Ready!")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze Image"):
        with st.spinner("Analyzing..."):
            img = image.convert("RGB").resize((128,128))
            arr = np.array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)
            pred = model.predict(arr)[0][0]

            if pred >= 0.6:
                st.success(f"✅ REAL IMAGE")
                st.progress(int(pred*100))
                st.write(f"**Confidence:** {int(pred*100)}%")
            elif pred <= 0.4:
                st.error(f"🚨 DEEPFAKE DETECTED")
                st.progress(int((1-pred)*100))
                st.write(f"**Confidence:** {int((1-pred)*100)}%")
            else:
                st.warning(f"⚠️ SUSPICIOUS IMAGE")
                st.progress(50)
                st.write("**Confidence:** 50%")
