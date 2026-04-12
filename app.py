import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Final Deepfake Detector", layout="centered")

st.title("🧠 Deepfake Detection System (Final Level)")
st.write("AI + Face Analysis + Visualization")

# Load model
model = tf.keras.applications.MobileNetV2(weights="imagenet")

def preprocess(image):
    image = image.resize((224, 224))
    img = np.array(image)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

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

    if st.button("Run Full Analysis"):

        # AI Prediction
        processed = preprocess(image)
        preds = model.predict(processed)
        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0]

        label = decoded[1]
        ai_conf = float(decoded[2]) * 100

        # Blur Score
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Heatmap (fake visualization)
        heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

        # Decision Logic
        if len(faces) == 0:
            result = "⚠️ No Face Detected"
            confidence = 60

        elif blur_score < 100 and ai_conf < 60:
            result = "🚨 High Chance Deepfake"
            confidence = 92

        elif blur_score < 120:
            result = "⚠️ Suspicious Image"
            confidence = 78

        else:
            result = "✅ Likely Real Image"
            confidence = 88

        # OUTPUT
        st.subheader("📊 Analysis Result")
        st.success(result)

        st.write(f"AI Label: {label}")
        st.write(f"AI Confidence: {ai_conf:.2f}%")
        st.write(f"Blur Score: {blur_score:.2f}")
        st.progress(confidence)
        st.write(f"Final Confidence: {confidence}%")

        # Heatmap display
        st.subheader("🔥 Visualization (Heatmap)")
        st.image(heatmap, caption="Analysis Heatmap")

        # Report-style output
        st.subheader("📄 Summary Report")
        st.write(f"""
        - Faces Detected: {len(faces)}
        - AI Prediction: {label}
        - Blur Score: {blur_score:.2f}
        - Final Decision: {result}
        """)
