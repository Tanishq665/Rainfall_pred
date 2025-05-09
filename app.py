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
    /* Animate background gradient */
    body {
        background: linear-gradient(-45deg, #56CCF2, #2F80ED, #56CCF2, #2F80ED);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }

    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }

    /* Container */
    .stApp {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0px 0px 20px rgba(0,0,0,0.1);
    }

    /* Title and Headers */
    h1, h2, h3, h4 {
        color: #2F80ED;
        font-weight: 700;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f2f6fc !important;
        border-right: 2px solid #2F80ED;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #edf3fc;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }

    /* Image styling (Weather Icon) */
    img {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }

    /* Plot area adjustments */
    .element-container:has(canvas), .element-container:has(svg) {
        background: #ffffff;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    /* Buttons, sliders, and inputs */
    .stButton>button, .stSlider, .stTextInput>div>div>input {
        border-radius: 8px;
    }

    /* Folium Map Styling */
    iframe {
        border-radius: 12px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
    }

    /* Success and Info boxes */
    .stAlert {
        border-radius: 10px;
        font-weight: 600;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background-color: #2F80ED;
        border-radius: 6px;
    }
    ::-webkit-scrollbar-track {
        background-color: #f2f6fc;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌦️ Real-Time Weather & Rain Prediction Dashboard")

cities = ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Chennai', 'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur']

city_choice = st.sidebar.selectbox("Choose a City", options=["Select from list"] + cities)

custom_city = st.sidebar.text_input("Or type a city name")

# Determine final city input
if custom_city:
    city = custom_city.title()
elif city_choice != "Select from list":
    city = city_choice
else:
    st.warning("Please select or enter a city.")
    st.stop()

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
        if rain_pred:
            st.success("Yes, it will rain 🌧️")
        else:
            st.info("No rain expected ☀️")

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
        
        # --- Error Distribution for Temperature Prediction ---
        st.subheader("📉 Error Distribution - Temperature Prediction")
        X_temp, y_temp = prepare_regression(df, "Temp")
        temp_errors = y_temp - reg_temp.predict(X_temp)
        fig_errors, ax_errors = plt.subplots()
        sns.histplot(temp_errors, bins=30, kde=True, ax=ax_errors)
        ax_errors.set_title("Error Distribution - Temperature Regression")
        ax_errors.set_xlabel("Prediction Error")
        st.pyplot(fig_errors)
        
        # --- Future Temperature Predictions ---
        st.subheader("🌡️ Future Temperature Predictions")
        fig_temp_future, ax_temp = plt.subplots(figsize=(8, 4))
        ax_temp.plot(future_times[:forecast_hours], temp_preds[:forecast_hours], marker='o', color='orange', label='Predicted Temp')
        ax_temp.set_title("Future Temperature Predictions")
        ax_temp.set_xlabel("Time")
        ax_temp.set_ylabel("Temperature (°C)")
        ax_temp.grid(True)
        ax_temp.legend()
        st.pyplot(fig_temp_future)
        
        # --- Future Humidity Predictions ---
        st.subheader("💧 Future Humidity Predictions")
        fig_hum_future, ax_hum = plt.subplots(figsize=(8, 4))
        ax_hum.plot(future_times[:forecast_hours], hum_preds[:forecast_hours], marker='o', color='blue', label='Predicted Humidity')
        ax_hum.set_title("Future Humidity Predictions")
        ax_hum.set_xlabel("Time")
        ax_hum.set_ylabel("Humidity (%)")
        ax_hum.grid(True)
        ax_hum.legend()
        st.pyplot(fig_hum_future)
        

    else:
        st.error("City not found. Please try again.")
