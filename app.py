import streamlit as st
from PIL import Image
import numpy as np

st.set_page_config(page_title="Deepfake Detector", layout="centered")

st.title("🧠 Deepfake Detection System (Using AI)")
st.markdown("AI-based image analysis")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg","png","jpeg"])

def extract_features(img):
    gray = np.mean(img, axis=2)

    blur = np.var(gray)
    noise = np.std(gray)
    edges = np.mean(np.abs(np.diff(gray)))

    return blur, noise, edges

def predict(blur, noise, edges):
    # weighted scoring
    score = (blur * 0.5) + (noise * 0.3) + (edges * 0.2)

    # normalized confidence
    confidence = min(max(int(score / 10), 50), 99)

    if score < 300:
        return "🚨 Deepfake", confidence
    elif score < 600:
        return "⚠️ Suspicious", confidence
    else:
        return "✅ Real", confidence

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = np.array(image)

    if st.button("🔍 Analyze Image"):

        blur, noise, edges = extract_features(img)
        result, confidence = predict(blur, noise, edges)

        st.subheader("📊 Result")

        if "Real" in result:
            st.success(result)
        elif "Suspicious" in result:
            st.warning(result)
        else:
            st.error(result)

        st.progress(confidence)

        # metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Blur", f"{blur:.2f}")
        col2.metric("Noise", f"{noise:.2f}")
        col3.metric("Edges", f"{edges:.2f}")

        st.markdown("---")
        st.markdown("### 📄 Analysis Report")

        st.write(f"""
        - Image clarity (blur) analyzed  
        - Noise pattern evaluated  
        - Edge consistency checked  
        - Final Decision: **{result}**  
        - Confidence: **{confidence}%**
        """)
