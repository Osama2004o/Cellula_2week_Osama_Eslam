import os
from PIL import Image
import pandas as pd
import streamlit as st

from models.pipeline import (
    LSTMPipeline,
    save_prediction_record,
    load_prediction_history,
    clear_prediction_history
)

# ---------------------------------------------------------
# Page Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Multimodal LSTM Classifier",
    page_icon="🛡️",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
VOCAB_PATH = os.path.join(MODELS_DIR, "vocab.json")
CLASSES_PATH = os.path.join(MODELS_DIR, "classes.json")
WEIGHTS_PATH = os.path.join(BASE_DIR, "best_lstm.pt")
CSV_PATH = os.path.join(BASE_DIR, "predictions_history.csv")

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
    st.error(f"Failed to initialize LSTM Pipeline: {e}")
    pipeline_ready = False


def generate_caption(image: Image.Image) -> str:
    """Invokes BLIP captioning locally if available, or returns empty string."""
    try:
        from models.imagecaption import caption_image
        return caption_image(image)
    except Exception as ex:
        st.warning(f"Local BLIP captioning not ready ({ex}). You can enter caption manually or connect to your GPU Colab.")
        return ""


# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .title-text {
        font-size: 2.2rem;
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
        background-color: #F1F5F9;
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
# Header & Sidebar
# ---------------------------------------------------------
st.markdown('<div class="title-text">Multimodal Text & Image Caption Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Classify direct text or BLIP-1 generated image descriptions using the trained bidirectional LSTM model.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Model Details")
    if pipeline_ready:
        st.success("✅ LSTM Model Loaded")
        st.write(f"**Device:** `{pipeline.device}`")
        st.write(f"**Vocab Size:** `{len(pipeline.vocab)}`")
        st.write(f"**Classes ({len(pipeline.classes)}):**")
        st.caption(", ".join(pipeline.classes))
    else:
        st.error("❌ Model not loaded")

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
# Prediction Presentation Helper
# ---------------------------------------------------------
def display_prediction_result(res: dict):
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
        <p><strong>Source:</strong> <em>{res["source"]}</em></p>
        <p><strong>Processed Input:</strong> <code>{res["cleaned_input"]}</code></p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📊 View All Class Probabilities"):
        prob_series = pd.DataFrame(
            list(res["probabilities"].items()),
            columns=["Category", "Probability"]
        ).sort_values(by="Probability", ascending=False)
        st.bar_chart(prob_series.set_index("Category"))


# ---------------------------------------------------------
# Input Tabs
# ---------------------------------------------------------
tab_text, tab_image = st.tabs(["✍️ Direct Text", "🖼️ Image Caption (BLIP-1)"])

# 1. Direct Text Tab
with tab_text:
    st.subheader("Direct Text Input")
    user_query = st.text_area(
        "Enter text to classify:",
        placeholder="Type or paste any text query here...",
        height=120,
        key="text_input_area"
    )

    if st.button("🚀 Classify Text", key="btn_classify_text"):
        if not user_query.strip():
            st.warning("Please enter text to classify.")
        elif not pipeline_ready:
            st.error("Pipeline is not available.")
        else:
            with st.spinner("Classifying text with LSTM..."):
                result = pipeline.predict(text=user_query)
                display_prediction_result(result)

# 2. Image Caption Tab
with tab_image:
    st.subheader("Image Input & BLIP-1 Captioning")
    uploaded_file = st.file_uploader(
        "Upload an image:",
        type=["png", "jpg", "jpeg", "webp"],
        key="img_file_uploader"
    )

    if uploaded_file is not None:
        col_img, col_act = st.columns([1, 1])
        with col_img:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Uploaded Image", width="stretch")

        with col_act:
            st.markdown("#### Captioning & Classification")

            # Initialize caption in session state
            if "current_caption" not in st.session_state:
                st.session_state["current_caption"] = ""

            if st.button("✨ Generate Caption (BLIP-1)", key="btn_gen_caption"):
                with st.spinner("Generating image description..."):
                    gen_cap = generate_caption(img)
                    st.session_state["current_caption"] = gen_cap
                    st.session_state["caption_text_area"] = gen_cap

            caption_val = st.text_area(
                "Image Description (Caption):",
                value=st.session_state.get("current_caption", ""),
                height=90,
                key="caption_text_area"
            )

            optional_query = st.text_input(
                "Optional Query / Text to combine with image (query sep caption):",
                key="optional_query_input"
            )

            if st.button("🚀 Classify Image Input", key="btn_classify_image"):
                target_caption = caption_val.strip()
                if not target_caption and not optional_query.strip():
                    st.warning("Please provide or generate an image caption first.")
                elif not pipeline_ready:
                    st.error("Pipeline is not available.")
                else:
                    with st.spinner("Classifying multimodal input with LSTM..."):
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
