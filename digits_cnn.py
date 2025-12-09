#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Абсолютно детерминированная версия решения.
Гарантии:
- все операции TensorFlow детерминированы,
- веса модели фиксированы,
- результат классификации digits.zip всегда один и тот же.

Модель обучается ТОЛЬКО первый раз и сохраняется в mnist_model.h5.
Все последующие запуски используют уже сохранённую модель.
"""

import os
import random
import numpy as np

# ============================================================
# 1. Полная фиксация детерминизма
# ============================================================
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

# Отключаем XLA, чтобы гарантировать детерминированные CPU kernels
tf.config.optimizer.set_jit(False)

# ============================================================
# 2. Импорты
# ============================================================
from pathlib import Path
from PIL import Image
import zipfile
from tensorflow.keras import layers, models


# ============================================================
# Настройки
# ============================================================
ZIP_PATH = Path("digits.zip")
MODEL_PATH = Path("mnist_model.h5")

IMG_SIZE = 28
BATCH_SIZE = 128
EPOCHS = 3


# ============================================================
# Создание CNN модели
# ============================================================
def build_cnn():
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))

    x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPool2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# Обучение модели MNIST (детерминированное)
# ============================================================
def train_and_save_model():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    model = build_cnn()

    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,   # Критично: фиксирован порядок батчей
        validation_data=(x_test, y_test),
        verbose=2
    )

    model.save(MODEL_PATH)
    print(f"\nМодель сохранена в {MODEL_PATH}")
    return model


# ============================================================
# Загрузка и предобработка изображений
# ============================================================
def preprocess(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, -1)
    return arr


def load_images_from_zip(zip_path):
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = sorted([
            n for n in zf.namelist()
            if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ])

        images = []
        for n in names:
            with zf.open(n) as f:
                img = Image.open(f).convert("L")
                images.append(preprocess(img))

    X = np.stack(images)
    return names, X


# ============================================================
# Основной метод
# ============================================================
def main():
    print("=== Проверка наличия сохранённой модели ===")

    if MODEL_PATH.exists():
        print("Найдена сохранённая модель. Загружаем...")
        model = tf.keras.models.load_model(MODEL_PATH)
    else:
        print("Сохранённой модели нет. Обучаем заново...")
        model = train_and_save_model()

    print("\n=== Загрузка и обработка изображений ===")
    names, X = load_images_from_zip(ZIP_PATH)
    print(f"Файлов найдено: {len(names)}")

    print("\n=== Классификация ===")
    preds = model.predict(X, batch_size=BATCH_SIZE, verbose=1)
    labels = preds.argmax(axis=1)

    counts = np.bincount(labels, minlength=10)

    print("\n=== Итоговый ответ ===")
    print(counts.tolist())
    print(f"sum = {counts.sum()}, files = {len(names)}")


if __name__ == "__main__":
    main()
