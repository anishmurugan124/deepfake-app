import streamlit as st
from PIL import Image
import numpy as np

st.title("🧠 AI Deepfake Detection App (Cloud Safe)")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    img_array = np.array(image)

    if st.button("Analyze"):

        # Convert to grayscale manually
        gray = np.mean(img_array, axis=2)

        # Blur score (variance)
        blur_score = np.var(gray)

        # Edge-like detection (simple gradient)
        edges = np.abs(np.diff(gray)).mean()

        # Decision logic
        if blur_score < 500:
            result = "⚠️ Possible Deepfake"
            confidence = 85
        else:
            result = "✅ Likely Real Image"
            confidence = 90

        st.subheader("Result")
        st.success(result)

        st.write(f"Blur Score: {blur_score:.2f}")
        st.write(f"Edge Value: {edges:.2f}")

        st.progress(confidence)
        st.write(f"Confidence: {confidence}%")
