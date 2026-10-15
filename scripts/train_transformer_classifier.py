from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from app.ml.dataset import LABEL_TO_ID, load_dataset

MODEL_OUTPUT_PATH = Path(__file__).parent.parent / "artifacts" / "document_classifier"

DATASET_PATH = Path(__file__).parent.parent / "data" / "document_categories.json"

MODEL_NAME = "distilbert-base-multilingual-cased"

NUM_CLASSES = 4
BATCH_SIZE = 4
EPOCHS = 3
MAX_LENGTH = 128
LEARNING_RATE = 2e-5


class TransformerDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[str],
        tokenizer,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.texts[index],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        item = {key: value.squeeze(0) for key, value in encoding.items()}

        item["labels"] = torch.tensor(
            LABEL_TO_ID[self.labels[index]],
            dtype=torch.long,
        )

        return item


def main():
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device :", device)
    print("Modèle :", MODEL_NAME)

    texts, labels = load_dataset(DATASET_PATH)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
    )

    model.to(device)

    train_dataset = TransformerDataset(
        train_texts,
        train_labels,
        tokenizer,
    )

    test_dataset = TransformerDataset(
        test_texts,
        test_labels,
        tokenizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print()
    print("Début du fine-tuning...")

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            batch = {key: value.to(device) for key, value in batch.items()}

            outputs = model(**batch)

            loss = outputs.loss

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {average_loss:.4f}")

    model.eval()

    predictions = []
    expected = []

    with torch.no_grad():
        for batch in test_loader:
            labels_batch = batch.pop("labels")

            batch = {key: value.to(device) for key, value in batch.items()}

            outputs = model(**batch)

            predicted = torch.argmax(
                outputs.logits,
                dim=1,
            )

            predictions.extend(predicted.cpu().tolist())

            expected.extend(labels_batch.tolist())

    accuracy = accuracy_score(
        expected,
        predictions,
    )

    print()
    print("=" * 60)
    print("RESULTATS TRANSFORMER HUGGING FACE")
    print("=" * 60)

    print(
        "Accuracy :",
        f"{accuracy:.2%}",
    )

    print()

    print(
        classification_report(
            expected,
            predictions,
            target_names=[
                "securite",
                "infrastructure",
                "developpement",
                "rh",
            ],
            zero_division=0,
        )
    )

    MODEL_OUTPUT_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(MODEL_OUTPUT_PATH)

    tokenizer.save_pretrained(MODEL_OUTPUT_PATH)

    print()
    print(
        "Modèle sauvegardé :",
        MODEL_OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
