from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from app.ml.dataset import (
    DocumentCategoryDataset,
    build_vocabulary,
    load_dataset,
)
from app.ml.document_classifier import (
    DocumentClassifier,
)

DATASET_PATH = Path(__file__).parent.parent / "data" / "document_categories.json"


def main():
    texts, labels = load_dataset(DATASET_PATH)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    vocabulary = build_vocabulary(train_texts)

    train_dataset = DocumentCategoryDataset(
        train_texts,
        train_labels,
        vocabulary,
    )

    test_dataset = DocumentCategoryDataset(
        test_texts,
        test_labels,
        vocabulary,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
    )

    model = DocumentClassifier(
        input_size=len(vocabulary),
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    epochs = 30

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0

        for inputs, labels_batch in train_loader:
            optimizer.zero_grad()

            logits = model(inputs)

            loss = criterion(
                logits,
                labels_batch,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1:02d} - Loss: {total_loss / len(train_loader):.4f}")

    model.eval()

    predictions = []
    expected = []

    with torch.no_grad():
        for inputs, labels_batch in test_loader:
            logits = model(inputs)

            predicted = torch.argmax(
                logits,
                dim=1,
            )

            predictions.extend(predicted.tolist())

            expected.extend(labels_batch.tolist())

    accuracy = accuracy_score(
        expected,
        predictions,
    )

    print()
    print("=" * 60)
    print("RESULTATS BASELINE PYTORCH")
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


if __name__ == "__main__":
    main()
