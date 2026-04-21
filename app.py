import streamlit as st
import anthropic
import base64
import json
from PIL import Image
import io

st.set_page_config(page_title="Deepfake Detector", page_icon="🧠", layout="centered")

st.markdown("""
<style>
    .main { background-color: #020408; }
    .stApp { background-color: #020408; color: #00ff88; }
    h1 { color: #00ff88 !important; font-family: monospace; text-align: center; }
    .subtitle { text-align: center; color: #00ff8877; font-size: 0.8rem; letter-spacing: 3px; margin-bottom: 30px; }
    .result-real { background: #00ff8808; border: 1px solid #00ff8866; border-radius: 8px; padding: 20px; }
    .result-fake { background: #ff000808; border: 1px solid #ff003366; border-radius: 8px; padding: 20px; }
    .result-suspicious { background: #ffaa0008; border: 1px solid #ffaa0066; border-radius: 8px; padding: 20px; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 DEEPFAKE.SCAN")
st.markdown('<p class="subtitle">AI VISION ANALYSIS SYSTEM • CLAUDE POWERED</p>', unsafe_allow_html=True)

# API Key input
api_key = st.text_input("🔑 Anthropic API Key", type="password", placeholder="sk-ant-...")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze Image", disabled=not api_key):
        if not api_key:
            st.error("Please enter your Anthropic API Key")
        else:
            with st.spinner("🤖 AI Vision Analysis in progress..."):
                try:
                    # Convert image to base64
                    img_buffer = io.BytesIO()
                    image.convert("RGB").save(img_buffer, format="JPEG")
                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

                    # Claude API call
                    client = anthropic.Anthropic(api_key=api_key)

                    message = client.messages.create(
                        model="claude-opus-4-5",
                        max_tokens=1024,
                        system="""You are an expert deepfake and AI-generated image detection system.
Analyze the image and return ONLY valid JSON (no markdown, no extra text).

Return this exact format:
{
  "verdict": "REAL" or "FAKE" or "SUSPICIOUS",
  "confidence": <number between 50-99>,
  "reasons": ["reason 1", "reason 2", "reason 3"],
  "summary": "2-3 sentence analysis"
}

Check for:
- Unnatural skin texture or blurring around face edges
- Inconsistent lighting or shadows
- Asymmetrical facial features
- Hair artifacts or blurry hairlines
- GAN artifacts like checkerboard patterns
- Over-smooth or waxy skin appearance
- Unnatural eye reflections

Be accurate and critical.""",
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/jpeg",
                                            "data": img_base64,
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": "Analyze this image for deepfake or AI-generation artifacts. Return only JSON."
                                    }
                                ],
                            }
                        ],
                    )

                    import json
                    response_text = message.content[0].text
                    clean = response_text.replace("```json", "").replace("```", "").strip()
                    result = json.loads(clean)

                    # Display Result
                    st.subheader("📊 Analysis Result")

                    verdict = result.get("verdict", "UNKNOWN")
                    confidence = result.get("confidence", 0)
                    reasons = result.get("reasons", [])
                    summary = result.get("summary", "")

                    if verdict == "REAL":
                        st.success(f"✅ AUTHENTIC IMAGE")
                        css_class = "result-real"
                    elif verdict == "FAKE":
                        st.error(f"🚨 DEEPFAKE DETECTED")
                        css_class = "result-fake"
                    else:
                        st.warning(f"⚠️ SUSPICIOUS IMAGE")
                        css_class = "result-suspicious"

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
        st.info("👆 Please enter your Anthropic API Key above to analyze")
