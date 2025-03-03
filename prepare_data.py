import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

df = pd.read_csv("fuel_consumption_dataset.csv")

df["fuel_type"] = np.random.choice([0, 1, 2], size=len(df))  # Бенз, дизель, гибрид
df["tire_pressure"] = np.random.uniform(2.0, 3.0, size=len(df))  # Давление в шинах (бар)
df["road_type"] = np.random.choice([0, 1, 2], size=len(df))  # Асфальт, грунтовка, мокрый асфальт

features = [
    "speed_kmh", "rpm", "temp_outside_C", "mass_kg", "aero_cd",
    "engine_temp_C", "road_slope_deg", "passengers",
    "fuel_type", "tire_pressure", "road_type"
]
target = "fuel_consumption_l_100km"

X = df[features].values
y = df[target].values.reshape(-1, 1)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

X_train_lstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_lstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

np.save("X_train.npy", X_train_lstm)
np.save("X_test.npy", X_test_lstm)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")

print("✅ Данные успешно подготовлены с новыми параметрами!🐻")
