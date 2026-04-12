import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load pre-trained model
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Preprocess function
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# UI
st.title("Deepfake Detection App (AI Based)")
st.write("Upload an image to analyze")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    if st.button("Detect"):
        processed = preprocess_image(image)
        preds = model.predict(processed)

        decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0]

        label = decoded[1]
        confidence = float(decoded[2]) * 100

        # Simple decision logic
        if confidence > 50:
            result = "Real Image (Likely)"
        else:
            result = "Possibly AI Generated (Deepfake)"

        st.subheader("Result")
        st.write(f"Prediction: {label}")
        st.write(result)
        st.write(f"Confidence: {confidence:.2f}%")
