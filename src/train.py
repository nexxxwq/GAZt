"""
train.py — обучение нейросети LeakNet.

Загружает dataset.csv, обучает нейросеть предсказывать координату утечки
по 9 входным признакам, считает метрики (MAE, RMSE, R²),
сохраняет модель и графики в папку results/.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from model import LeakNet


def load_data(path=None):
    """Загружает датасет и делит на признаки и целевую переменную."""
    if path is None:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(BASE_DIR, "data", "dataset.csv")
    df = pd.read_csv(path)
    X = df[["d", "L", "k_e", "rho", "P_start", "P_end", "Q_start", "Q_end", "k"]].values
    y = df["x"].values
    return X, y


def train_model():
    """Основной цикл обучения."""
    # Загрузка данных
    X, y = load_data()
    print(f"Загружено {len(X)} наблюдений.")

    # Нормализация признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Нормализация целевой переменной (для Sigmoid на выходе)
    y_min, y_max = y.min(), y.max()
    y_norm = (y - y_min) / (y_max - y_min)

    # Разделение на train / val
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_norm, test_size=0.2, random_state=42
    )

    # Тензоры
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

    # Модель
    model = LeakNet(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Обучение
    epochs = 200
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(X_train_t)
        train_losses.append(epoch_loss)

        # Валидация
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()
            val_losses.append(val_loss)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} | train_loss={epoch_loss:.5f} | val_loss={val_loss:.5f}")

    # Финальные предсказания
    model.eval()
    with torch.no_grad():
        y_pred_norm = model(X_val_t).numpy().flatten()

    # Обратная нормализация
    y_pred = y_pred_norm * (y_max - y_min) + y_min
    y_true = y_val * (y_max - y_min) + y_min

    # Метрики
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("\n=== Метрики гибридной модели ===")
    print(f"MAE  = {mae:.2f} м")
    print(f"RMSE = {rmse:.2f} м")
    print(f"R²   = {r2:.3f}")

    # Сохранение результатов
    os.makedirs("results", exist_ok=True)
    np.save("results/y_true.npy", y_true)
    np.save("results/y_pred.npy", y_pred)
    np.save("results/train_losses.npy", np.array(train_losses))
    np.save("results/val_losses.npy", np.array(val_losses))
    torch.save(model.state_dict(), "results/leaknet.pt")

    print("\nМодель сохранена: results/leaknet.pt")


if __name__ == "__main__":
    train_model()