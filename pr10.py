# main.py
import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


MODEL_PATH = Path("realty_model.pkl")
DATA_PATH = Path("realty_data.csv")

FEATURE_COLUMNS = ["total_square", "rooms", "floor"]
TARGET_COLUMN = "price"


def train_model():
    """
    Обучаем простую модель линейной регрессии
    по признакам total_square, rooms, floor и сохраняем её в файл.
    """

    df = pd.read_csv(DATA_PATH)

    # Берём только нужные признаки и целевую переменную, убираем пропуски
    data = df[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna()

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    r2 = model.score(X_test, y_test)
    print(f"R^2 на тесте: {r2:.4f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Модель сохранена в {MODEL_PATH.resolve()}")
    return model


def load_or_train_model():
    """
    Если модель уже сохранена — грузим её.
    Если файла нет — обучаем заново.
    """
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print("Модель загружена из файла.")
        return model
    else:
        print("Файл модели не найден, обучаем модель...")
        return train_model()


# Загружаем/обучаем модель при старте приложения
model = load_or_train_model()

app = FastAPI(title="Realty Price Prediction API")


class RealtyFeatures(BaseModel):
    total_square: float
    rooms: float
    floor: float


def make_prediction(total_square: float, rooms: float, floor: float) -> float:
    """
    Общая функция для получения предсказания моделью.
    Используется и в GET, и в POST эндпоинтах.
    """
    X = pd.DataFrame(
        [[total_square, rooms, floor]],
        columns=FEATURE_COLUMNS,
    )
    pred = model.predict(X)[0]
    return float(pred)


@app.get("/health")
def health():
    """
    Liveness-проба (health-check).
    """
    return {"status": "ok"}


@app.get("/predict_get")
def predict_get(
    total_square: float,
    rooms: float,
    floor: float,
):
    """
    Получение предсказания через GET-запрос.
    Пример запроса:
    /predict_get?total_square=50&rooms=2&floor=5
    """
    price = make_prediction(total_square, rooms, floor)
    return {
        "total_square": total_square,
        "rooms": rooms,
        "floor": floor,
        "predicted_price": price,
    }


@app.post("/predict_post")
def predict_post(features: RealtyFeatures):
    """
    Получение предсказания через POST-запрос.
    Тело запроса (JSON):
    {
      "total_square": 50.0,
      "rooms": 2,
      "floor": 5
    }
    """
    price = make_prediction(
        features.total_square,
        features.rooms,
        features.floor,
    )
    return {
        "total_square": features.total_square,
        "rooms": features.rooms,
        "floor": features.floor,
        "predicted_price": price,
    }


if __name__ == "__main__":
    # Дополнительно: единичное предсказание в консоли,
    # чтобы явно показать "код для получения предсказания обученной моделью"
    example_price = make_prediction(total_square=50.0, rooms=2, floor=5)
    print(f"Пример предсказания для 50 м², 2 комнаты, 5 этаж: {example_price:.2f}")
