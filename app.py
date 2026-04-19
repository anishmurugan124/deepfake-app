import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

st.title("🧠 Real Deepfake Detection AI")

# Load trained model
model = tf.keras.models.load_model("deepfake_model.h5")

def preprocess(image):
    image = image.resize((224, 224))
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image)

    if st.button("Detect"):

        processed = preprocess(image)
        prediction = model.predict(processed)[0][0]

        if prediction > 0.5:
            result = "🚨 Deepfake"
            confidence = prediction * 100
        else:
            result = "✅ Real"
            confidence = (1 - prediction) * 100

        st.subheader("Result")
        st.write(result)
        st.write(f"Confidence: {confidence:.2f}%")
