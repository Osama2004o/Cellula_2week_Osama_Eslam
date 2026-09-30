import os
import io
import requests
from PIL import Image
import pandas as pd
import streamlit as st

from models.pipeline import (
    LSTMPipeline,
    save_prediction_record,
    load_prediction_history,
    clear_prediction_history
)

st.set_page_config(
    page_title="Multimodal Classifier (Colab BLIP-1 + LSTM)",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
VOCAB_PATH = os.path.join(MODELS_DIR, "vocab.json")
CLASSES_PATH = os.path.join(MODELS_DIR, "classes.json")
WEIGHTS_PATH = os.path.join(BASE_DIR, "best_lstm.pt")
CSV_PATH = os.path.join(BASE_DIR, "predictions_history.csv")

# you add the ngrok url you get from colab here
DEFAULT_API_URL = "https://unclustering-irresistible-camille.ngrok-free.dev"

# ---------------------------------------------------------
# Model Pipeline Initialization (Cached)
# ---------------------------------------------------------
@st.cache_resource
def get_pipeline():
    return LSTMPipeline(
        weights_path=WEIGHTS_PATH,
        vocab_path=VOCAB_PATH,
        classes_path=CLASSES_PATH,
        max_len=64,
        embed_dim=300,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    )


try:
    pipeline = get_pipeline()
    pipeline_ready = True
except Exception as e:
    st.error(f"Failed to load LSTM Pipeline: {e}")
    pipeline_ready = False


# ---------------------------------------------------------
# Remote BLIP Captioning Function (Ngrok / FastAPI)
# ---------------------------------------------------------
def fetch_caption_from_api(image: Image.Image, api_base_url: str) -> str:
    """
    Sends the image as multipart/form-data to the remote Colab FastAPI endpoint:
    POST {api_base_url}/caption
    """
    base = api_base_url.strip().rstrip("/")
    if not base:
        raise ValueError("API URL is empty. Please configure a valid ngrok URL in the sidebar.")

    endpoint = f"{base}/caption"

    # Convert PIL Image to JPEG bytes
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)

    # ngrok-skip-browser-warning bypasses the free ngrok browser confirmation page
    headers = {
        "ngrok-skip-browser-warning": "true"
    }
    files = {
        "file": ("image.jpg", buffer, "image/jpeg")
    }

    response = requests.post(endpoint, files=files, headers=headers, timeout=45)
    if response.status_code == 200:
        data = response.json()
        if "caption" in data:
            return data["caption"]
        elif "error" in data:
            raise RuntimeError(f"Colab Server Error: {data['error']}")
        else:
            raise ValueError(f"Unexpected response format: {data}")
    else:
        raise RuntimeError(f"HTTP {response.status_code} Error: {response.text}")


def check_api_health(api_base_url: str) -> bool:
    """Checks the root GET / health check endpoint."""
    try:
        base = api_base_url.strip().rstrip("/")
        headers = {"ngrok-skip-browser-warning": "true"}
        res = requests.get(base, headers=headers, timeout=10)
        return res.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .title-text {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .subtitle-text {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .result-container {
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        background-color: #F8FAFC;
        border-left: 6px solid #2563EB;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .label-badge {
        display: inline-block;
        font-weight: bold;
        padding: 0.35rem 0.85rem;
        border-radius: 8px;
        color: white;
    }
    .badge-safe {
        background-color: #16A34A;
    }
    .badge-unsafe {
        background-color: #DC2626;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.header("🌐 Colab GPU Connection")
    api_url = st.text_input(
        "Colab Ngrok API URL:",
        value=DEFAULT_API_URL,
        help="Update this URL whenever your Colab ngrok tunnel restarts."
    )

    if st.button("🔌 Test API Connection", width="stretch"):
        with st.spinner("Pinging Colab endpoint..."):
            if check_api_health(api_url):
                st.success("✅ Colab BLIP API is active and reachable!")
            else:
                st.error("❌ Could not connect. Please ensure your Colab notebook is running.")

    st.markdown("---")
    st.header("⚙️ Local Model Info")
    if pipeline_ready:
        st.write(f"**LSTM Device:** `{pipeline.device}`")
        st.write(f"**Vocab Size:** `{len(pipeline.vocab)}`")
        st.write(f"**Classes ({len(pipeline.classes)}):**")
        st.caption(", ".join(pipeline.classes))

    st.markdown("---")
    st.subheader("📁 History Options")
    if st.button("📋 Toggle History View", width="stretch"):
        st.session_state["show_history"] = not st.session_state.get("show_history", False)

    history_data = load_prediction_history(CSV_PATH)
    if not history_data.empty:
        st.download_button(
            label="📥 Download History CSV",
            data=history_data.to_csv(index=False).encode("utf-8"),
            file_name="predictions_history.csv",
            mime="text/csv",
            width="stretch"
        )


# ---------------------------------------------------------
# Presentation Helper
# ---------------------------------------------------------
def display_prediction_result(res: dict):
    # Logs only the 2 required columns: Text, Predicted Label
    save_prediction_record(CSV_PATH, res)
    st.success("Prediction complete and logged to CSV!")

    pred_label = res["predicted_label"]
    is_safe = ("safe" in pred_label.lower()) and ("unsafe" not in pred_label.lower())
    badge_class = "badge-safe" if is_safe else "badge-unsafe"

    st.markdown(f"""
    <div class="result-container">
        <h4 style="margin-top:0;">Prediction Result</h4>
        <p><strong>Predicted Label:</strong> <span class="label-badge {badge_class}">{pred_label}</span></p>
        <p><strong>Confidence:</strong> {res["confidence"] * 100:.2f}%</p>
        <p><strong>Input Stored in CSV:</strong> <code>{res["combined_input"]}</code></p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📊 View All Class Probabilities"):
        prob_series = pd.DataFrame(
            list(res["probabilities"].items()),
            columns=["Category", "Probability"]
        ).sort_values(by="Probability", ascending=False)
        st.bar_chart(prob_series.set_index("Category"))


# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------
st.markdown('<div class="title-text">Multimodal Text & Image Caption Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Fast GPU captioning via Colab Ngrok API with local Bidirectional LSTM classification.</div>', unsafe_allow_html=True)

tab_text, tab_image = st.tabs(["✍️ Direct Text", "🖼️ Image Caption (Remote BLIP-1)"])

# 1. Direct Text Tab
with tab_text:
    st.subheader("Direct Text Input")
    user_query = st.text_area(
        "Enter text to classify:",
        placeholder="Type or paste text here...",
        height=120,
        key="text_input_area"
    )

    if st.button("🚀 Classify Text", key="btn_classify_text"):
        if not user_query.strip():
            st.warning("Please enter text to classify.")
        elif not pipeline_ready:
            st.error("LSTM pipeline is not loaded.")
        else:
            with st.spinner("Classifying text with LSTM..."):
                result = pipeline.predict(text=user_query)
                display_prediction_result(result)


# 2. Image Caption Tab (Using Ngrok Colab API)
with tab_image:
    st.subheader("Image Input via Colab BLIP-1 API")
    uploaded_file = st.file_uploader(
        "Upload an image:",
        type=["png", "jpg", "jpeg", "webp"],
        key="img_uploader"
    )

    if uploaded_file is not None:
        col_img, col_actions = st.columns([1, 1])
        with col_img:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Uploaded Image", width="stretch")

        with col_actions:
            st.markdown("#### Captioning & Classification")

            if "remote_caption" not in st.session_state:
                st.session_state["remote_caption"] = ""

            if st.button("✨ Generate Caption (via Colab GPU)", key="btn_colab_caption"):
                with st.spinner("Sending image to Colab BLIP-1 endpoint..."):
                    try:
                        caption = fetch_caption_from_api(img, api_url)
                        st.session_state["remote_caption"] = caption
                        st.session_state["caption_area"] = caption
                        st.success(f"Caption generated: \"{caption}\"")
                    except Exception as err:
                        st.error(f"Failed to fetch caption: {err}")

            caption_val = st.text_area(
                "Generated Image Caption (editable):",
                value=st.session_state.get("remote_caption", ""),
                height=90,
                key="caption_area"
            )

            optional_query = st.text_input(
                "Optional Query / Text to combine with image (query sep caption):",
                key="optional_query_text"
            )

            if st.button("🚀 Classify Image Input", key="btn_classify_image"):
                target_caption = caption_val.strip()
                if not target_caption and not optional_query.strip():
                    st.warning("Please generate or provide a caption first.")
                elif not pipeline_ready:
                    st.error("LSTM pipeline is not loaded.")
                else:
                    with st.spinner("Classifying input with LSTM..."):
                        result = pipeline.predict(text=optional_query, image_caption=target_caption)
                        display_prediction_result(result)


# ---------------------------------------------------------
# History Section
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📜 Predictions History")

btn_col1, btn_col2 = st.columns([2, 1])

with btn_col1:
    if st.button("👁️ View Past Predictions from CSV", key="btn_view_csv"):
        st.session_state["show_history"] = True

with btn_col2:
    if st.button("🗑️ Clear History CSV", key="btn_clear_csv"):
        clear_prediction_history(CSV_PATH)
        st.success("History CSV cleared!")
        st.session_state["show_history"] = False

if st.session_state.get("show_history", False):
    history_df = load_prediction_history(CSV_PATH)
    if history_df.empty:
        st.info("No records in history CSV yet.")
    else:
        st.markdown(f"**Total Predictions Logged:** `{len(history_df)}`")
        st.dataframe(history_df, width="stretch", hide_index=True)
