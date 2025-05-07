import streamlit as st
import requests
import pandas as pd
import numpy as np
import pytz
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
API_KEY = "6098d0cb0a6e777a63bd8f0c8a20f70e"
BASE_URL = "https://api.openweathermap.org/data/2.5/"

# --- Fetch current weather ---
def get_curr_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url).json()
    if "main" not in response:
        return None
    return {
        "city": response["name"],
        "lat": response["coord"]["lat"],
        "lon": response["coord"]["lon"],
        "curr_temp": round(response["main"]["temp"]),
        "feels_like": round(response["main"]["feels_like"]),
        "temp_min": round(response["main"]["temp_min"]),
        "temp_max": round(response["main"]["temp_max"]),
        "pressure": round(response["main"]["pressure"]),
        "humidity": round(response["main"]["humidity"]),
        "description": response["weather"][0]["description"].title(),
        "icon": response["weather"][0]["icon"],
        "country": response["sys"]["country"],
        "wind_deg": response["wind"]["deg"],
        "wind_speed": response["wind"]["speed"]
    }

# --- Read and prepare data ---
@st.cache_data
def load_data():
    df = pd.read_csv("weather.csv").dropna().drop_duplicates()
    le = LabelEncoder()
    df["WindGustDir"] = le.fit_transform(df["WindGustDir"])
    df["RainTomorrow"] = le.fit_transform(df["RainTomorrow"])
    X = df[['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Humidity', 'Pressure', 'Temp']]
    y = df['RainTomorrow']
    return df, X, y, le

def prepare_regression(df, feature):
    X, y = [], []
    for i in range(len(df) - 1):
        X.append(df[feature].iloc[i])
        y.append(df[feature].iloc[i + 1])
    return np.array(X).reshape(-1, 1), np.array(y)

def predict_future(model, current_val):
    preds = []
    val = current_val
    for _ in range(5):
        val = model.predict(np.array(val).reshape(1, -1))
        preds.append(val[0])
    return preds

# --- Dashboard UI ---
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    body {
        background: linear-gradient(-45deg, #56CCF2, #2F80ED, #56CCF2, #2F80ED);
        background-size: 400% 400%;
        animation: gradient 10s ease infinite;
    }
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 12px;
        padding: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌦️ Real-Time Weather & Rain Prediction Dashboard")

cities = ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Chennai', 'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur']
city = st.sidebar.selectbox("Choose a City", options=cities, index=cities.index("Delhi"))

forecast_hours = st.sidebar.slider("Forecast Next Hours", 1, 5, 5)
show_map = st.sidebar.checkbox("Show Location on Map", True)

if city:
    weather = get_curr_weather(city)
    if weather:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader(f"{weather['city']}, {weather['country']} 🌍")
            st.metric("Temperature", f"{weather['curr_temp']} °C", f"Feels like {weather['feels_like']} °C")
            st.metric("Humidity", f"{weather['humidity']} %")
            st.metric("Pressure", f"{weather['pressure']} hPa")
            st.metric("Wind Speed", f"{weather['wind_speed']} m/s")
        with col2:
            st.image(f"http://openweathermap.org/img/wn/{weather['icon']}@2x.png", width=100)
            st.markdown(f"**Condition:** {weather['description']}")

        if show_map:
            m = folium.Map(location=[weather["lat"], weather["lon"]], zoom_start=8)
            folium.Marker([weather["lat"], weather["lon"]], tooltip=city).add_to(m)
            folium_static(m)

        df, X, y, le = load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
        reg_temp = RandomForestRegressor().fit(*prepare_regression(df, "Temp"))
        reg_hum = RandomForestRegressor().fit(*prepare_regression(df, "Humidity"))

        compass_points = [
            ("N", 348.75, 360), ("N", 0, 11.25),
            ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
            ("ENE", 56.25, 78.75), ("E", 78.75, 101.25),
            ("ESE", 101.25, 123.75), ("SE", 123.75, 146.25),
            ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
            ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25),
            ("WSW", 236.25, 258.75), ("W", 258.75, 281.25),
            ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
            ("NNW", 326.25, 348.75)
        ]
        direction = next((name for name, start, end in compass_points if start <= weather["wind_deg"] < end), "N")
        dir_encoded = le.transform([direction])[0] if direction in le.classes_ else 0
        sample = pd.DataFrame([{
            'MinTemp': weather['temp_min'],
            'MaxTemp': weather['temp_max'],
            'WindGustDir': dir_encoded,
            'WindGustSpeed': weather['wind_speed'],
            'Humidity': weather['humidity'],
            'Pressure': weather['pressure'],
            'Temp': weather['curr_temp']
        }])
        rain_pred = clf.predict(sample)[0]
        st.subheader("🌧️ Rain Prediction:")
        st.success("Yes, it will rain 🌧️") if rain_pred else st.info("No rain expected ☀️")

        # --- Feature Importance ---
        st.subheader("📊 Feature Importance:")
        imp = clf.feature_importances_
        fig_imp, ax = plt.subplots()
        sns.barplot(x=imp, y=X.columns, ax=ax)
        st.pyplot(fig_imp)

        # --- Future Predictions Charts ---
        temp_preds = predict_future(reg_temp, weather['temp_min'])
        hum_preds = predict_future(reg_hum, weather['humidity'])
        now = datetime.now(pytz.timezone("Asia/Kolkata"))
        future_times = [(now + timedelta(hours=i+1)).strftime("%H:%M") for i in range(forecast_hours)]

        st.subheader("📈 Temperature & Humidity Forecast:")
        fig_temp, ax = plt.subplots()
        ax.plot(future_times[:forecast_hours], temp_preds[:forecast_hours], label="Temperature", marker='o')
        ax.plot(future_times[:forecast_hours], hum_preds[:forecast_hours], label="Humidity", marker='s')
        ax.set_ylabel("Value")
        ax.set_xlabel("Time")
        ax.set_title("Predicted Temperature & Humidity")
        ax.legend()
        st.pyplot(fig_temp)

        # --- Donut Chart for Weather Breakdown ---
        st.subheader("⛅ Weather Condition Breakdown (Sample Data)")
        cond_counts = df["RainTomorrow"].value_counts()
        fig_pie, ax = plt.subplots()
        ax.pie(cond_counts, labels=["No Rain", "Rain"], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig_pie)

    else:
        st.error("City not found. Please try again.")
