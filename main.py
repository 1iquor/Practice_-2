import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

#Загружаем данные
df = pd.read_csv(
    "D:/ITMO/Python_analiz_dannyx/realty_data.csv",
    engine="python",
    on_bad_lines="skip"
)


y = df["price"]


feature_cols = ["total_square", "rooms", "floor", "lat", "lon"]

# Убираем строки с пропусками в этих колонках
df_model = df.dropna(subset=feature_cols)
X = df_model[feature_cols]
y = df_model["price"]

#Делим на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Обучить модель прогнозирования стоимости недвижимости
# (модель может быть любой сложности, даже линейная регрессия на двух признаках)
model = LinearRegression()
model.fit(X_train, y_train)

print("R^2 на тесте:", model.score(X_test, y_test))


joblib.dump(model, "realty_model.pkl")
print("Модель сохранена в realty_model.pkl")

#Реализуйте код для получения предсказания обученной моделью
import numpy as np
import joblib

model = joblib.load("realty_model.pkl")

def predict_price(total_square: float,
                  rooms: float,
                  floor: float,
                  lat: float,
                  lon: float) -> float:

    X_new = np.array([[total_square, rooms, floor, lat, lon]])
    y_pred = model.predict(X_new)[0]
    return float(y_pred)

# Пример вызова:
if __name__ == "__main__":
    price = predict_price(
        total_square=40,
        rooms=1,
        floor=5,
        lat=55.75,
        lon=37.6,
    )
    print("Прогнозируемая цена:", round(price))

#Реализуйте интерфейс с помощью streamlit для введения значений признаков для прогнозирования
import streamlit as st
import numpy as np
import joblib

# Загрузка обученной модели
@st.cache_resource
def load_model():
    return joblib.load("realty_model.pkl")

model = load_model()

def predict_price(total_square: float,
                  rooms: float,
                  floor: float,
                  lat: float,
                  lon: float) -> float:
    X_new = np.array([[total_square, rooms, floor, lat, lon]])
    y_pred = model.predict(X_new)[0]
    return float(y_pred)



st.title("Прогноз стоимости недвижимости")

st.write("Введите параметры объекта, и модель оценит его стоимость.")


col1, col2 = st.columns(2)

with col1:
    total_square = st.number_input(
        "Общая площадь, м²",
        min_value=5.0,
        max_value=500.0,
        value=40.0,
        step=1.0,
    )
    rooms = st.number_input(
        "Количество комнат",
        min_value=1.0,
        max_value=10.0,
        value=1.0,
        step=1.0,
    )
    floor = st.number_input(
        "Этаж",
        min_value=1.0,
        max_value=100.0,
        value=5.0,
        step=1.0,
    )

with col2:
    lat = st.number_input(
        "Широта (lat)",
        min_value=40.0,
        max_value=80.0,
        value=55.75,
        step=0.001,
        format="%.6f",
    )
    lon = st.number_input(
        "Долгота (lon)",
        min_value=20.0,
        max_value=60.0,
        value=37.60,
        step=0.001,
        format="%.6f",
    )


if st.button("Рассчитать стоимость"):
    pred_price = predict_price(
        total_square=total_square,
        rooms=rooms,
        floor=floor,
        lat=lat,
        lon=lon,
    )

    st.subheader("Результат прогнозирования")
    st.success(f"Прогнозируемая стоимость: {pred_price:,.0f} ₽".replace(",", " "))


    st.write(f"≈ {pred_price / 1_000_000:.2f} млн ₽")