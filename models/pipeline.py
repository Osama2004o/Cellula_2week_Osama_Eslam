import os
import re
import json
from datetime import datetime
import pandas as pd
import torch

from models.LSTM import LSTMClassifier

# -------------------------------------------------------------------
# Preprocessing functions (Matching the original training notebook)
# -------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Cleans raw text using the exact rules from lstm_task.ipynb:
    - Lowercase
    - Remove URLs
    - Remove non-alphanumeric characters except whitespace and apostrophes
    - Normalize whitespace
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def encode_text(text: str, vocab: dict, max_len: int = 64) -> list:
    """
    Encodes cleaned text into a sequence of token IDs padded/truncated to max_len.
    0 = <pad>, 1 = <unk>
    """
    tokens = text.split()
    ids = [vocab.get(w, 1) for w in tokens][:max_len]
    return ids + [0] * (max_len - len(ids))


# -------------------------------------------------------------------
# Model Loader & Predictor
# -------------------------------------------------------------------
class LSTMPipeline:
    """Encapsulates vocabulary, class names, and the trained PyTorch LSTM model."""

    def __init__(
        self,
        weights_path: str,
        vocab_path: str,
        classes_path: str,
        max_len: int = 64,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.4,
        device: torch.device = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len

        # Load vocab
        with open(vocab_path, "r", encoding="utf-8") as vf:
            self.vocab = json.load(vf)

        # Load class names
        with open(classes_path, "r", encoding="utf-8") as cf:
            self.classes = json.load(cf)

        # Build LSTM architecture
        self.model = LSTMClassifier(
            vocab_size=len(self.vocab),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=len(self.classes),
            dropout=dropout
        )

        # Load weights
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str = "", image_caption: str = ""):
        """
        Takes input text, image caption, or both.
        If both are given, combines them with ' sep ' exactly as trained in lstm_task.ipynb:
        clean = (query + " sep " + image descriptions)
        """
        text = text.strip() if text else ""
        image_caption = image_caption.strip() if image_caption else ""

        if text and image_caption:
            source = "Multimodal (Text + Image Caption)"
            raw_combined = f"{text} sep {image_caption}"
        elif image_caption:
            source = "Image Caption (BLIP-1)"
            raw_combined = image_caption
        else:
            source = "Direct Text"
            raw_combined = text

        cleaned = clean_text(raw_combined)
        token_ids = encode_text(cleaned, self.vocab, max_len=self.max_len)
        tensor_input = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model(tensor_input)
            probs = torch.softmax(logits, dim=-1)[0]
            top_idx = torch.argmax(probs).item()
            predicted_label = self.classes[top_idx]
            confidence = float(probs[top_idx].item())

        prob_breakdown = {self.classes[i]: float(probs[i].item()) for i in range(len(self.classes))}

        return {
            "source": source,
            "raw_text": text,
            "raw_caption": image_caption,
            "combined_input": raw_combined,
            "cleaned_input": cleaned,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "probabilities": prob_breakdown
        }


# -------------------------------------------------------------------
# CSV History Helpers
# -------------------------------------------------------------------
def save_prediction_record(csv_path: str, record_data: dict):
    """Appends an inference record with 2 columns: Text and Predicted Label."""
    row = {
        "Text": record_data.get("combined_input", ""),
        "Predicted Label": record_data.get("predicted_label", "")
    }
    df = pd.DataFrame([row])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode="a", header=False, index=False)


def load_prediction_history(csv_path: str) -> pd.DataFrame:
    """Reads all saved records from the CSV file."""
    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path)
        except Exception:
            return pd.DataFrame(columns=["Text", "Predicted Label"])
    return pd.DataFrame(columns=["Text", "Predicted Label"])


def clear_prediction_history(csv_path: str):
    """Resets the CSV file with empty rows and the 2 headers."""
    headers = ["Text", "Predicted Label"]
    pd.DataFrame(columns=headers).to_csv(csv_path, index=False)

