"""
dataset.py — генератор синтетического датасета.

Создаёт 12 000 наблюдений для обучения нейросети.
Каждое наблюдение содержит 9 входных признаков и координату утечки
(целевую переменную), рассчитанную аналитическим методом.

Сохраняет результат в data/dataset.csv
"""

import os
import sys
import random
import csv
import math

from physics import (
    velocity,
    reynolds_number,
    friction_coefficient,
    resistance_coefficient,
    leak_coordinate,
)


def generate_one():
    """
    Генерирует одно наблюдение.
    """
    # Геометрия трубы
    d = random.uniform(0.3, 0.6)
    L = random.uniform(300, 500)
    k_e = random.uniform(0.00001, 0.0001)

    # Параметры газа
    rho = random.uniform(0.7, 0.9)
    eta = 1.1e-5

    # Расходы
    Q_start = random.uniform(2.0, 3.0)
    Q_end = Q_start * random.uniform(0.3, 0.6)

    # Расчёт коэффициентов
    v = velocity(Q_start, d)
    Re = reynolds_number(rho, v, d, eta)
    lam = friction_coefficient(Re, k_e, d)
    k = resistance_coefficient(lam, d, rho)

    # Потери давления на весь участок при Q_start и Q_end
    dP_start = k * L * Q_start ** 2
    dP_end = k * L * Q_end ** 2

    # Разница давлений = среднее потерь (чтобы x попал в середину)
    # Плюс случайное отклонение ±20% для разнообразия
    P_diff = (dP_start + dP_end) / 2 * random.uniform(0.8, 1.2)

    # Абсолютные давления (для реалистичности)
    P_start = random.uniform(4e6, 6e6)
    P_end = P_start - P_diff

    # Координата утечки
    x = leak_coordinate(P_start, P_end, Q_start, Q_end, k, L)

    if x is None:
        return None

    return {
        "d": d, "L": L, "k_e": k_e, "rho": rho,
        "P_start": P_start, "P_end": P_end,
        "Q_start": Q_start, "Q_end": Q_end,
        "k": k, "x": x,
    }
def generate_dataset(n=12000, save_path="data/dataset.csv"):
    """
    Генерирует датасет из n наблюдений и сохраняет в CSV.
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(BASE_DIR, "data", "dataset.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    rows = []
    attempts = 0
    max_attempts = n * 100  # было n * 10

    while len(rows) < n and attempts < max_attempts:
        attempts += 1
        sample = generate_one()
        if sample is not None:
            rows.append(sample)

    if len(rows) == 0:
        print("ОШИБКА: не удалось сгенерировать ни одного наблюдения.")
        print("Проверьте параметры в generate_one().")
        return

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Сгенерировано {len(rows)} наблюдений.")
    print(f"Сохранено в {save_path}")

if __name__ == "__main__":
    generate_dataset()