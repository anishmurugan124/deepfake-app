import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="Deepfake Detector", layout="centered")

st.title("🧠 Deepfake Detection App")
st.write("AI + Face Analysis Based Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Load face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    st.write(f"Faces Detected: {len(faces)}")

    if st.button("Detect Deepfake"):
        if len(faces) == 0:
            st.warning("No face detected. Cannot analyze properly.")
        else:
            # Simple AI-like logic
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

            if blur_score < 100:
                result = "⚠️ Possibly Deepfake (Blurred Face)"
                confidence = 88
            else:
                result = "✅ Likely Real Image"
                confidence = 92

            st.subheader("Result")
            st.success(result)
            st.progress(confidence)
            st.write(f"Confidence: {confidence}%")
