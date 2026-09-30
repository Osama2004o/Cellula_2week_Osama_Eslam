# Multimodal Text & Image Caption Safety Classifier

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

An end-to-end multimodal content moderation and safety classification pipeline developed for the **Cellula NLP** program. The system identifies hazardous, criminal, or sensitive content across both pure text prompts and multimodal inputs (images paired with text queries) using **BLIP-1** image captioning and a custom **Bidirectional LSTM** neural network.

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Model Architecture & Pipeline](#model-architecture--pipeline)
- [Target Classes](#target-classes)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
  - [Option 1: Streamlit with Remote Colab GPU (Recommended)](#option-1-streamlit-with-remote-colab-gpu-recommended)
  - [Option 2: Streamlit with Local BLIP Execution](#option-2-streamlit-with-local-blip-execution)
  - [Option 3: Python Pipeline API](#option-3-python-pipeline-api)
- [Inference History & Logging](#inference-history--logging)
- [Technologies Used](#technologies-used)
- [License](#license)

---

## Overview

Modern safety guardrails must evaluate both textual and visual context. This project solves multimodal safety classification by converting visual inputs into semantic textual descriptions via **Salesforce BLIP** (`Salesforce/blip-image-captioning-base`), unifying them with optional user text queries through a multimodal delimiter (`<query> sep <caption>`), and classifying the sequence with an optimized **Bidirectional LSTM** classifier.

### Core Workflow
```
[User Text] ────────────────────────┐
                                    ▼
[User Image] ──► [BLIP Captioner] ──► [Fusion: Text + 'sep' + Caption] ──► [Text Cleaning & Vocab Encoding] ──► [Bi-LSTM Classifier] ──► [Safety Category & Probabilities]
```

---

## Key Features

- **Dual-Modality Support**:
  - **Direct Text**: Instant evaluation of text queries, messages, or prompts.
  - **Image + Text**: Generates dense descriptive captions from uploaded images, which can be evaluated alone or fused with an accompanying user query.
- **Flexible Execution Modes**:
  - **Remote GPU Acceleration (`app.py`)**: Connects to an external Google Colab GPU instance via an Ngrok tunnel to run BLIP without demanding local GPU memory.
  - **Local Execution (`main.py`)**: Runs the complete pipeline locally when GPU or local PyTorch capabilities are available.
- **Interactive Streamlit Web Dashboard**:
  - Live prediction badges (Safe vs. Unsafe categories).
  - Confidence scoring and full class probability bar charts.
  - Editable caption interface enabling human-in-the-loop adjustments prior to classification.
- **Automated Prediction Logging**:
  - Automatically records input queries and predicted labels to `predictions_history.csv`.
  - Built-in UI to view, download, or clear past inferences.

---

## Model Architecture & Pipeline

### 1. Vision-to-Language: BLIP-1
- **Model**: `Salesforce/blip-image-captioning-base`
- **Function**: Translates raw visual input images into descriptive natural language sentences capturing entities, interactions, and scene context.

### 2. Text Preprocessing & Tokenization
- Implemented in `models/pipeline.py`:
  - Lowercasing and whitespace normalization.
  - URL and special character removal (preserving apostrophes).
  - Vocabulary lookup based on `models/vocab.json` with `<pad>: 0` and `<unk>: 1`.
  - Fixed sequence padding/truncation (`max_len = 64`).

### 3. Classifier: Bidirectional LSTM
- Implemented in `models/LSTM.py` (`LSTMClassifier`):
  - **Embedding Layer**: Vocabulary size mapped to dense embedding vectors.
  - **Recurrent Core**: 2-layer Bidirectional LSTM (`hidden_dim = 128`, dropout = 0.3–0.4).
  - **Masked Mean Pooling**: Aggregates token representations while masking out padding indices.
  - **Classification Head**: Fully connected layer outputting unnormalized logits over 9 target classes.
  - **Weights**: Checkpoint stored in `best_lstm.pt`.

---

## Target Classes

The classifier categorizes content into 9 predefined safety classes defined in `models/classes.json`:

| # | Class Label | Description |
|---|---|---|
| 1 | `safe` | Benign, safe, and appropriate content |
| 2 | `unsafe` | General safety policy violation |
| 3 | `violent crimes` | Physical violence, threats, weapons, or violent acts |
| 4 | `non-violent crimes` | Theft, fraud, property damage, or illicit non-violent activity |
| 5 | `sex-related crimes` | Sexually explicit violations or non-consensual content |
| 6 | `child sexual exploitation` | Severe harm and child safety violations |
| 7 | `suicide & self-harm` | Self-harm ideation, promotion, or instruction |
| 8 | `elections` | Election integrity and political influence content |
| 9 | `unknown s-type` | Ambiguous or unclassified safety category |

---

## Repository Structure

```
Cellula_2week_Osama_Eslam/
├── app.py                      # Streamlit application with Colab Ngrok GPU integration
├── main.py                     # Streamlit application with local BLIP model execution
├── best_lstm.pt                # Trained PyTorch Bidirectional LSTM weights
├── predictions_history.csv     # Persistent CSV log of predictions (Text, Predicted Label)
├── requirements.txt            # Project Python dependencies
├── LICENSE                     # Apache 2.0 License
├── README.md                   # Project documentation
└── models/
    ├── LSTM.py                 # PyTorch Bidirectional LSTM model architecture
    ├── imagecaption.py         # BLIP image captioning pipeline (Hugging Face Transformers)
    ├── pipeline.py             # Preprocessing, vocabulary encoder, predictor, and CSV logger
    ├── classes.json            # Target classification labels (9 classes)
    └── vocab.json              # Trained vocabulary token-to-ID mapping
```

---

## Installation

### 1. Prerequisites
- Python 3.9, 3.10, or 3.11
- CUDA-enabled GPU (optional, for accelerated local inference)

### 2. Clone the Repository
```bash
git clone https://github.com/Osama2004o/Cellula_2week_Osama_Eslam.git
cd Cellula_2week_Osama_Eslam
```

### 3. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## How to Run

### Option 1: Streamlit with Remote Colab GPU (Recommended)
This mode allows running the web UI locally on low-spec hardware while delegating heavy BLIP image captioning to a free Google Colab GPU via Ngrok:

1. Launch your Colab notebook hosting the FastAPI BLIP server and copy the public `ngrok` URL (e.g., `https://xxxx-xx-xx.ngrok-free.dev`).
2. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
3. Open the sidebar in the browser, paste your Colab Ngrok URL into the **Colab Ngrok API URL** field, and click **Test API Connection**.
4. Upload an image or enter text to classify!

---

### Option 2: Streamlit with Local BLIP Execution
If your local machine has sufficient GPU/CPU resources to run both BLIP and the LSTM classifier locally:

```bash
streamlit run main.py
```

---

### Option 3: Python Pipeline API
You can also integrate the classification pipeline directly into custom scripts:

```python
from models.pipeline import LSTMPipeline

# Initialize the pipeline
pipeline = LSTMPipeline(
    weights_path="best_lstm.pt",
    vocab_path="models/vocab.json",
    classes_path="models/classes.json",
    max_len=64,
    embed_dim=300,
    hidden_dim=128,
    num_layers=2,
    dropout=0.3
)

# 1. Direct text prediction
res = pipeline.predict(text="A family enjoying a sunny picnic in the park.")
print(f"Predicted: {res['predicted_label']} (Confidence: {res['confidence']:.2%})")

# 2. Multimodal prediction (query + image caption)
res_multimodal = pipeline.predict(
    text="person near bank entrance",
    image_caption="a man wearing a black mask holding a crowbar"
)
print(f"Predicted: {res_multimodal['predicted_label']} (Confidence: {res_multimodal['confidence']:.2%})")
```

---

## Inference History & Logging

All predictions executed via the Streamlit interfaces are automatically appended to `predictions_history.csv` with the following schema:
- **`Text`**: The raw text, image caption, or combined `query sep caption` string.
- **`Predicted Label`**: The assigned classification label.

In the Streamlit interface, you can:
- View previous prediction records in real time.
- Download the updated history file directly as a CSV.
- Clear the history log with one click.

---

## Technologies Used

- **Deep Learning**: [PyTorch](https://pytorch.org/) (Bidirectional LSTM), [Hugging Face Transformers](https://huggingface.co/transformers/) (Salesforce BLIP)
- **Web Interface**: [Streamlit](https://streamlit.io/)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Image Processing**: [Pillow (PIL)](https://python-pillow.org/)
- **Remote Bridge**: [FastAPI](https://fastapi.tiangolo.com/), [Requests](https://requests.readthedocs.io/), [Ngrok](https://ngrok.com/)

---

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for complete terms.