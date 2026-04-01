# =========================================
# 🌾 AI Smart Farming Decision System
# =========================================

import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Load Model Files
# -------------------------------
model = pickle.load(open("model/final_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
encoders = pickle.load(open("model/encoders.pkl", "rb"))
features = pickle.load(open("model/feature_columns.pkl", "rb"))

# -------------------------------
# Feature Importance (for explanation)
# -------------------------------
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({
    "feature": features,
    "importance": feature_importance
}).sort_values(by="importance", ascending=False)

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Smart Farming System", layout="wide")

st.markdown("<h1 style='text-align:center; color:green;'>🌾 AI Smart Farming Decision System</h1>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------
# Input Section
# -------------------------------
st.header("📥 Enter Farm Details")

col1, col2, col3 = st.columns(3)

state_list = list(encoders["state_name"].classes_)
district_list = list(encoders["dist_name"].classes_)
crop_list = list(encoders["crop"].classes_)
soil_list = list(encoders["soil_type"].classes_)

with col1:
    state = st.selectbox("State", state_list)
    district = st.selectbox("District", district_list)
    crop = st.selectbox("Current Crop", crop_list)
    soil = st.selectbox("Soil Type", soil_list)

with col2:
    temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 70.0)
    rainfall = st.slider("Rainfall (mm)", 0.0, 2000.0, 1000.0)
    wind_speed = st.slider("Wind Speed (m/s)", 0.0, 10.0, 2.0)
    solar = st.slider("Solar Radiation", 0.0, 30.0, 15.0)

with col3:
    ph = st.slider("Soil pH", 0.0, 14.0, 6.5)

    unknown_npk = st.checkbox("I don't know N, P, K (Auto-fill)")

    if unknown_npk:
        n = 10.0
        p = 5.0
        k = 10.0
        st.info("Using default nutrient values")
    else:
        n = st.number_input("Nitrogen (N)", value=10.0)
        p = st.number_input("Phosphorus (P)", value=5.0)
        k = st.number_input("Potassium (K)", value=10.0)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🌾 Predict Yield"):

    input_data = {
        "state_name": state,
        "dist_name": district,
        "crop": crop,
        "soil_type": soil,
        "year": 2020,
        "area_ha": 1.0,
        "n_req_kg_per_ha": n,
        "p_req_kg_per_ha": p,
        "k_req_kg_per_ha": k,
        "total_n_kg": n * 1000,
        "total_p_kg": p * 1000,
        "total_k_kg": k * 1000,
        "temperature_c": temperature,
        "humidity_%": humidity,
        "ph": ph,
        "rainfall_mm": rainfall,
        "wind_speed_m_s": wind_speed,
        "solar_radiation_mj_m2_day": solar
    }

    df = pd.DataFrame([input_data])

    # Encode categorical
    for col in encoders:
        df[col] = encoders[col].transform(df[col])

    # Feature Engineering
    df["temp_rain_interaction"] = df["temperature_c"] * df["rainfall_mm"]
    df["nutrient_index"] = (
        df["n_req_kg_per_ha"] +
        df["p_req_kg_per_ha"] +
        df["k_req_kg_per_ha"]
    )

    # Arrange columns
    df = df[features]

    # Scale
    df_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(df_scaled)[0]

    # Recommendation
    if prediction < 200:
        recommendation = "⚠️ Low Yield → Improve irrigation & fertilizer"
    elif prediction < 500:
        recommendation = "⚡ Moderate Yield → Optimize nutrients"
    else:
        recommendation = "✅ High Yield → Maintain current practices"

    # -------------------------------
    # Output
    # -------------------------------
    st.success(f"🌾 Predicted Yield: {prediction:.2f} kg/ha")
    st.info(f"📊 Recommendation: {recommendation}")

    # -------------------------------
    # Explanation
    # -------------------------------
    st.subheader("🧠 Why this prediction?")

    top_features = importance_df.head(5)

    for _, row in top_features.iterrows():
        st.write(f"🔹 {row['feature']} influenced the prediction")

    # -------------------------------
    # Smart Insights
    # -------------------------------
    st.subheader("📊 Smart Insights")

    if rainfall < 500:
        st.warning("Low rainfall detected → Irrigation needed")

    if n < 5:
        st.warning("Low nitrogen → Add fertilizer")

    if ph < 5.5 or ph > 7.5:
        st.warning("Soil pH is not optimal")

    # -------------------------------
    # Crop Recommendation
    # -------------------------------
    st.subheader("🌱 Recommended Crops")

    crop_scores = []

    for crop_option in crop_list:
        temp_df = df.copy()
        temp_df["crop"] = encoders["crop"].transform([crop_option])[0]

        temp_scaled = scaler.transform(temp_df)
        pred = model.predict(temp_scaled)[0]

        crop_scores.append((crop_option, pred))

    crop_scores = sorted(crop_scores, key=lambda x: x[1], reverse=True)

    for crop_name, score in crop_scores[:3]:
        st.write(f"🌾 {crop_name} → {score:.2f} kg/ha")

    # -------------------------------
    # Visualization
    # -------------------------------
    st.subheader("📊 Environmental Factors")

    fig, ax = plt.subplots(figsize=(3, 2))
    labels = ["Temperature", "Rainfall", "Humidity"]
    values = [temperature, rainfall, humidity]
    ax.bar(labels, values)
    ax.set_title("Environmental Conditions")
    st.pyplot(fig)

    # # -------------------------------
    # # Report
    # # -------------------------------
    # report = f"""
    # SMART FARMING REPORT

    # Yield: {prediction:.2f} kg/ha
    # State: {state}
    # District: {district}
    # Crop: {crop}

    # Recommendation:
    # {recommendation}
    # """

    # st.download_button("📄 Download Report", report, file_name="report.txt")