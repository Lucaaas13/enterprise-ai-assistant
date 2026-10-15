import torch
from torch import nn


class DocumentClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int = 4,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_size,
                64,
            ),
            nn.ReLU(),
            nn.Linear(
                64,
                num_classes,
            ),
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(inputs)
