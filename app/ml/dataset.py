import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

LABEL_TO_ID = {
    "securite": 0,
    "infrastructure": 1,
    "developpement": 2,
    "rh": 3,
}

ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


class DocumentCategoryDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[str],
        vocabulary: dict[str, int],
    ):
        self.texts = texts
        self.labels = labels
        self.vocabulary = vocabulary

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index].lower()

        vector = torch.zeros(
            len(self.vocabulary),
            dtype=torch.float32,
        )

        for word in text.split():
            word = word.strip(".,;:!?()[]'\"")

            if word in self.vocabulary:
                vector[self.vocabulary[word]] += 1.0

        label = LABEL_TO_ID[self.labels[index]]

        return vector, torch.tensor(
            label,
            dtype=torch.long,
        )


def load_dataset(path: str | Path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    texts = [item["text"] for item in data]

    labels = [item["label"] for item in data]

    return texts, labels


def build_vocabulary(
    texts: list[str],
) -> dict[str, int]:
    words = set()

    for text in texts:
        for word in text.lower().split():
            word = word.strip(".,;:!?()[]'\"")

            if word:
                words.add(word)

    return {word: index for index, word in enumerate(sorted(words))}
