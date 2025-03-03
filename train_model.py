import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import joblib

X_train_lstm = np.load("X_train.npy")
X_test_lstm = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

model = Sequential([
    Input(shape=(X_train_lstm.shape[1], 1)),
    LSTM(64, activation="tanh", return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation="tanh"),
    Dense(1, activation="linear")
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

history = model.fit(X_train_lstm, y_train, epochs=50, batch_size=64, validation_data=(X_test_lstm, y_test), verbose=1)

test_loss, test_mae = model.evaluate(X_test_lstm, y_test)
print(f"✅ Тестовый MAE: {test_mae:.4f}")

model.save("fuel_lstm_model.keras")
print("✅ Успешно сохранена!!!!")
