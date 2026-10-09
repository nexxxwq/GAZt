"""
plot_results.py — построение графиков по результатам обучения.

Создаёт три графика:
1. Scatter plot: истинные vs предсказанные координаты.
2. Гистограмма ошибок.
3. Кривая обучения (train/val loss).

Сохраняет PNG в папку results/.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    y_true = np.load(os.path.join(BASE_DIR, "results", "y_true.npy"))
    y_pred = np.load(os.path.join(BASE_DIR, "results", "y_pred.npy"))
    train_losses = np.load(os.path.join(BASE_DIR, "results", "train_losses.npy"))
    val_losses = np.load(os.path.join(BASE_DIR, "results", "val_losses.npy"))

    # === 1. Scatter plot ===
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.4, s=10, color="#1f4e79")
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             "r--", linewidth=1.5, label="Идеал")
    plt.xlabel("Истинная координата утечки, м")
    plt.ylabel("Предсказанная координата утечки, м")
    plt.title("Гибридная модель: истинные и предсказанные координаты")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "results", "scatter.png"), dpi=150)
    plt.close()
    print("scatter.png — сохранён")

    # === 2. Гистограмма ошибок ===
    errors = y_pred - y_true
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=50, color="#2e7d32", alpha=0.75, edgecolor="black")
    plt.axvline(0, color="red", linestyle="--", linewidth=1.5)
    plt.xlabel("Ошибка предсказания, м")
    plt.ylabel("Количество наблюдений")
    plt.title("Гистограмма ошибок гибридной модели")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "results", "histogram.png"), dpi=150)
    plt.close()
    print("histogram.png — сохранён")

    # === 3. Кривая обучения ===
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train", color="#1f4e79")
    plt.plot(val_losses, label="val", color="#e67e22")
    plt.xlabel("Эпоха обучения")
    plt.ylabel("Значение функции потерь")
    plt.title("Обучение")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "results", "learning_curve.png"), dpi=150)
    plt.close()
    print("learning_curve.png — сохранён")

    # === Метрики в консоль ===
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print("\n=== Финальные метрики ===")
    print(f"MAE  = {mae:.2f} м")
    print(f"RMSE = {rmse:.2f} м")
    print(f"R²   = {r2:.3f}")


if __name__ == "__main__":
    main()