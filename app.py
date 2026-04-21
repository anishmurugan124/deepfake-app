import streamlit as st
import numpy as np
from PIL import Image

st.title("Deepfake Detection App")
st.write("Hello Anish 👋")

# ---------- FEATURE EXTRACTION ----------
def extract_features(img):
    gray = np.mean(img, axis=2)

    blur = np.var(gray)
    noise = np.std(gray)
    edges = np.mean(np.abs(np.diff(gray)))

    return blur, noise, edges


# ---------- PREDICTION ----------
def predict(blur, noise, edges):
    score = (blur / 1000) + (noise / 50) + (edges / 10)

    if score < 5:
        return "🚨 Deepfake", 85
    elif score < 8:
        return "⚠️ Suspicious", 75
    else:
        return "✅ Real", 90


# ---------- UI ----------
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    img = np.array(image)

    if st.button("🔍 Analyze Image"):

        blur, noise, edges = extract_features(img)
        result, confidence = predict(blur, noise, edges)

        st.subheader("Result")

        if "Real" in result:
            st.success(result)
        elif "Suspicious" in result:
            st.warning(result)
        else:
            st.error(result)

        st.progress(confidence)

        # DEBUG (optional)
        st.write(f"Blur: {blur:.2f}")
        st.write(f"Noise: {noise:.2f}")
        st.write(f"Edges: {edges:.2f}")
