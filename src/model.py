"""
model.py — нейросеть LeakNet для предсказания координаты утечки.

Архитектура: полносвязная сеть (MLP) на PyTorch.
Вход: 9 признаков (геометрия трубы + параметры газа + давления + расходы).
Выход: 1 число — координата утечки в метрах.
"""

import torch
import torch.nn as nn


class LeakNet(nn.Module):
    """
    Полносвязная нейросеть для предсказания координаты утечки.

    Архитектура:
        Linear(9, 64) -> ReLU
        Linear(64, 64) -> ReLU
        Linear(64, 32) -> ReLU
        Linear(32, 1) -> Sigmoid
    """

    def __init__(self, input_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)