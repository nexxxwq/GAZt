"""
inference.py — предсказание координаты утечки с помощью обученной модели.

Загружает сохранённую модель LeakNet и делает предсказание
для новых входных данных.
"""

import numpy as np
import torch

from model import LeakNet


def predict(features, model_path="results/leaknet.pt"):
    """
    Предсказывает координату утечки по 9 признакам.

    features — список из 9 значений:
        [d, L, k_e, rho, P_start, P_end, Q_start, Q_end, k]
    """
    model = LeakNet(input_dim=9)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    x = torch.tensor([features], dtype=torch.float32)
    with torch.no_grad():
        y_norm = model(x).item()

    return y_norm


if __name__ == "__main__":
    # Пример: координата утечки для случайного набора параметров
    example = [0.4, 300, 0.00005, 0.8, 5e6, 4.2e6, 2.0, 1.2, 1.5e-3]
    print("Входные признаки:", example)
    print("Предсказанная координата (нормированная):", predict(example))