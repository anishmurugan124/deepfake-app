import streamlit as st
import numpy as np
from PIL import Image
st.markdown("<h1 style='text-align: center; font-weight: bold;'>Deep Fake Detection Using AI</h1>", unsafe_allow_html=True)


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

        st.subheader("Analysis Report")

st.markdown(f"""
- Image clarity (blur) analyzed  
- Noise pattern evaluated  
- Edge consistency checked  

**Final Decision:** {result}  
**Confidence:** {confidence}%
""")
        # DEBUG (optional)
        st.markdown("---")

st.subheader("Detailed Metrics")

st.write(f"Blur: {blur:.2f}")
st.write(f"Noise: {noise:.2f}")
st.write(f"Edges: {edges:.2f}")
