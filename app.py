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
                "https://drive.google.com/uc?id=1oi3NdmhE5GayQhNsxMW0e74gWtPIp7ei",
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
                import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import gdown
import os
import cv2

st.set_page_config(page_title="Deepfake Detector", page_icon="🧠")
st.title("🧠 Image Deepfake Detection Using AI")
st.markdown("92% Accurate • Trained AI Model • Free")

MODEL_PATH = "deepfake_model.h5"
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Model downloading..."):
            gdown.download(
                "https://drive.google.com/uc?id=1oi3NdmhE5GayQhNsxMW0e74gWtPIp7ei",
                MODEL_PATH, quiet=False
            )
    return tf.keras.models.load_model(MODEL_PATH)

def detect_face(image):
    img_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces) > 0

model = load_model()
st.success("✅ Model Ready!")

uploaded_file = st.file_uploader("📤 Upload Face Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze Image"):
        with st.spinner("Analyzing..."):

            # Face check
            if not detect_face(image):
                st.warning("⚠️ No face detected! Please upload a face image for accurate results.")
            else:
                img = image.convert("RGB").resize((128,128))
                arr = np.array(img) / 255.0
                arr = np.expand_dims(arr, axis=0)
                pred = model.predict(arr)[0][0]

                if pred >= 0.6:
                    st.success("✅ REAL IMAGE")
                    st.progress(int(pred*100))
                    st.write(f"**Confidence:** {int(pred*100)}%")
                elif pred <= 0.4:
                    st.error("🚨 DEEPFAKE DETECTED")
                    st.progress(int((1-pred)*100))
                    st.write(f"**Confidence:** {int((1-pred)*100)}%")
                else:
                    st.warning("⚠️ SUSPICIOUS IMAGE")
                    st.progress(50)
                    st.write("**Confidence:** 50%")
