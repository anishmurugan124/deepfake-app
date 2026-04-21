import streamlit as st
import google.generativeai as genai
import base64
import json
from PIL import Image
import io

st.set_page_config(page_title="Deepfake Detector", page_icon="🧠", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #020408; color: #00ff88; }
    h1 { color: #00ff88 !important; font-family: monospace; text-align: center; }
    .subtitle { text-align: center; color: #00ff8877; font-size: 0.8rem; letter-spacing: 3px; margin-bottom: 30px; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 DEEPFAKE.SCAN")
st.markdown('<p class="subtitle">AI VISION ANALYSIS • GEMINI POWERED • FREE</p>', unsafe_allow_html=True)

api_key = st.text_input("🔑 Gemini API Key", type="password", placeholder="AIza...")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze Image", disabled=not api_key):
        with st.spinner("🤖 AI Analysis in progress..."):
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel("gemini-1.5-flash-latest")

                img_buffer = io.BytesIO()
                image.convert("RGB").save(img_buffer, format="JPEG")
                img_bytes = img_buffer.getvalue()

                prompt = """You are a deepfake detection expert. Analyze this image carefully.
Return ONLY valid JSON, no extra text:
{
  "verdict": "REAL" or "FAKE" or "SUSPICIOUS",
  "confidence": <number 50-99>,
  "reasons": ["reason 1", "reason 2", "reason 3"],
  "summary": "2-3 sentence analysis"
}
Check for: unnatural skin, face blurring, lighting issues, GAN artifacts, hair artifacts, eye reflections."""

                response = model.generate_content([
                    prompt,
                    {"mime_type": "image/jpeg", "data": img_bytes}
                ])

                clean = response.text.replace("```json", "").replace("```", "").strip()
                result = json.loads(clean)

                verdict = result.get("verdict", "UNKNOWN")
                confidence = result.get("confidence", 0)
                reasons = result.get("reasons", [])
                summary = result.get("summary", "")

                st.subheader("📊 Analysis Result")

                if verdict == "REAL":
                    st.success("✅ AUTHENTIC IMAGE")
                elif verdict == "FAKE":
                    st.error("🚨 DEEPFAKE DETECTED")
                else:
                    st.warning("⚠️ SUSPICIOUS IMAGE")

                st.progress(confidence)
                st.write(f"**Confidence:** {confidence}%")

                st.markdown("**🔎 Detection Reasons:**")
                for reason in reasons:
                    st.write(f"› {reason}")

                st.markdown("**📝 Summary:**")
                st.write(summary)

            except json.JSONDecodeError:
                st.error("Response parsing error. Please try again.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    elif not api_key:
        st.info("👆 Please enter your Gemini API Key above")
