import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(page_title="Crop Recommendation System", page_icon="🌾", layout="centered")
st.title("🌾 Crop Recommendation System")

# -----------------------------------
# SAMPLE DATASET
# -----------------------------------
data = {
    "N": [90,85,60,74,78,69,55,88]*10,
    "P": [42,58,55,35,40,37,50,48]*10,
    "K": [43,41,44,40,42,38,45,41]*10,
    "temperature": [20,22,26,18,21,23,25,24]*10,
    "humidity": [82,80,75,88,85,83,78,79]*10,
    "ph": [6.5,6.8,6.2,6.0,6.3,6.7,6.1,6.4]*10,
    "rainfall": [200,180,220,150,170,210,190,205]*10,
    "label": ["rice","rice","maize","wheat","rice","maize","wheat","rice"]*10
}

df = pd.DataFrame(data)

# -----------------------------------
# PREPARE DATA
# -----------------------------------
X = df.drop("label", axis=1)
y = df["label"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------------
# TRAIN MODEL
# -----------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write(f"✅ Model Accuracy: {round(acc,3)}")

# -----------------------------------
# USER INPUT UI
# -----------------------------------
st.subheader("Enter Soil & Weather Details")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", 0, 150, 90)
    P = st.number_input("Phosphorus (P)", 0, 150, 40)
    K = st.number_input("Potassium (K)", 0, 150, 40)
    temp = st.slider("Temperature (°C)", 0, 50, 25)

with col2:
    humidity = st.slider("Humidity (%)", 0, 100, 80)
    ph = st.slider("pH Level", 0.0, 14.0, 6.5)
    rainfall = st.slider("Rainfall (mm)", 0, 300, 200)

# -----------------------------------
# PREDICTION
# -----------------------------------
if st.button("🌱 Recommend Crop"):

    input_data = [[N, P, K, temp, humidity, ph, rainfall]]
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    st.success(f"🌾 Recommended Crop: {prediction.upper()}")