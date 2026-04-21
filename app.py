import streamlit as st
import json
from PIL import Image
import io
import requests
import base64

st.set_page_config(page_title="Deepfake Detector", page_icon="🧠", layout="centered")

st.markdown("""
<style>
    .stApp { background-color: #020408; color: #00ff88; }
    h1 { color: #00ff88 !important; font-family: monospace; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 DEEPFAKE.SCAN")
st.markdown("AI VISION ANALYSIS • GEMINI POWERED • FREE")

api_key = st.text_input("🔑 Gemini API Key", type="password", placeholder="AIza...")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze Image", disabled=not api_key):
        with st.spinner("🤖 AI Analysis in progress..."):
            try:
                # Image to base64
                img_buffer = io.BytesIO()
                image.convert("RGB").save(img_buffer, format="JPEG")
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

                # Direct REST API call - v1 version
                url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"

                payload = {
                    "contents": [{
                        "parts": [
                            {
                                "text": """You are a deepfake detection expert. Analyze this image.
Return ONLY valid JSON, no extra text:
{
  "verdict": "REAL" or "FAKE" or "SUSPICIOUS",
  "confidence": <number 50-99>,
  "reasons": ["reason 1", "reason 2", "reason 3"],
  "summary": "2-3 sentence analysis"
}"""
                            },
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": img_base64
                                }
                            }
                        ]
                    }]
                }

                response = requests.post(url, json=payload)
                data = response.json()

                if "error" in data:
                    st.error(f"API Error: {data['error']['message']}")
                else:
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    clean = text.replace("```json", "").replace("```", "").strip()
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
