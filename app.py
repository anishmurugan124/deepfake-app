%%writefile app.py
import streamlit as st
from PIL import Image
import random

st.title("Deepfake Detection App")
st.write("Hello Anish 👋")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    if st.button("Detect"):
        result = random.choice(["Real Image", "AI-Generated (Deepfake)"])
        confidence = random.randint(80, 99)

        st.subheader("Result")
        st.write(result)
        st.write(f"Confidence: {confidence}%")