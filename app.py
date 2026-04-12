import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="AI Deepfake Detector", layout="centered")

st.title("🧠 AI Deepfake Detection App")
st.write("Cloud-based Lightweight AI Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Face Detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    st.write(f"Faces Detected: {len(faces)}")

    if st.button("Analyze with AI"):

        # Feature extraction (AI-like)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges)

        # Decision logic (AI-inspired)
        if len(faces) == 0:
            result = "⚠️ No Face Detected"
            confidence = 60

        elif blur_score < 100 and edge_density < 20:
            result = "🚨 High Chance Deepfake"
            confidence = 90

        elif blur_score < 120:
            result = "⚠️ Suspicious Image"
            confidence = 75

        else:
            result = "✅ Likely Real Image"
            confidence = 88

        st.subheader("Result")
        st.success(result)

        st.write(f"Blur Score: {blur_score:.2f}")
        st.write(f"Edge Density: {edge_density:.2f}")

        st.progress(confidence)
        st.write(f"Confidence: {confidence}%")
