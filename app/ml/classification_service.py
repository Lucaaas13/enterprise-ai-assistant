from functools import lru_cache
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from app.ml.dataset import ID_TO_LABEL

MODEL_PATH = Path(__file__).parent.parent.parent / "artifacts" / "document_classifier"

MAX_LENGTH = 128


@lru_cache
def get_classifier():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Le modèle de classification est introuvable. "
            "Lance d'abord : "
            "python -m scripts.train_transformer_classifier"
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    model.eval()

    return tokenizer, model


def predict_category(
    text: str,
) -> dict:
    if not text.strip():
        raise ValueError("Le texte ne peut pas être vide.")

    tokenizer, model = get_classifier()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1,
        )

        predicted_id = int(
            torch.argmax(
                probabilities,
                dim=1,
            ).item()
        )

        confidence = float(
            probabilities[
                0,
                predicted_id,
            ].item()
        )

    return {
        "category": ID_TO_LABEL[predicted_id],
        "confidence": confidence,
    }
