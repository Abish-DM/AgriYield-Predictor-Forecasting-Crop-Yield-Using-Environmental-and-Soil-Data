# =========================================
# 🌾 Crop Yield Prediction - FINAL TRAINING
# =========================================

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_excel("data/dataset.xlsx")

print("✅ Dataset Loaded Successfully!")
print(df.head())

# -------------------------------
# 2. Clean Column Names
# -------------------------------
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print("\n📌 Columns after cleaning:")
print(df.columns)

# -------------------------------
# 3. Drop Unnecessary Columns
# -------------------------------
drop_cols = ["dist_code", "state_code"]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# -------------------------------
# 4. Handle Missing Values
# -------------------------------
df = df.dropna()

# -------------------------------
# 5. Feature Engineering
# -------------------------------

# Interaction feature
df["temp_rain_interaction"] = df["temperature_c"] * df["rainfall_mm"]

# Nutrient index (FIXED)
df["nutrient_index"] = (
    df["n_req_kg_per_ha"] +
    df["p_req_kg_per_ha"] +
    df["k_req_kg_per_ha"]
)

# -------------------------------
# 6. Encoding Categorical Data
# -------------------------------
encoders = {}

categorical_cols = [
    "state_name",
    "dist_name",
    "crop",
    "soil_type"
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# -------------------------------
# 7. Define Features & Target
# -------------------------------

target = "yield_kg_per_ha"   # IMPORTANT

X = df.drop(columns=[target])
y = df[target]

# -------------------------------
# 8. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 9. Feature Scaling
# -------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 10. Model Training
# -------------------------------
model = GradientBoostingRegressor()

model.fit(X_train_scaled, y_train)

# -------------------------------
# 11. Evaluation
# -------------------------------
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Performance:")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.2f}")

# -------------------------------
# 12. Save Model Files
# -------------------------------
pickle.dump(model, open("model/final_model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))
pickle.dump(encoders, open("model/encoders.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("model/feature_columns.pkl", "wb"))

print("\n✅ Model & files saved successfully in /model folder!")