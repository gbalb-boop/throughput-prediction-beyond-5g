import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input


# Load dataset
df = pd.read_csv("data/ue-lte-network-traffic-stats.csv")

print(df.head())


# Rename throughput column
df["UE1 Throughput"] = df["UE1: web-rtc"]


# Select features
df = df[
    [
        "UE1 Throughput",
        "UE1-CQI",
        "UE1-Jitter"
    ]
]

print(df.head())


# Normalize data
scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(df)

print(scaled_data[:5])


# Create sequences for LSTM
def create_sequences(data, sequence_length):

    X = []
    y = []

    for i in range(len(data) - sequence_length):

        X.append(data[i:i + sequence_length])

        # Predict future throughput
        y.append(data[i + sequence_length, 0])

    return np.array(X), np.array(y)


# Number of previous time steps used for prediction
sequence_length = 3


X, y = create_sequences(
    scaled_data,
    sequence_length
)


print("X shape:", X.shape)
print("y shape:", y.shape)


# Train/test split (chronological)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)


print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# Build LSTM model
model = Sequential()

model.add(
    Input(
        shape=(sequence_length, 3)
    )
)

model.add(
    LSTM(50)
)

model.add(
    Dense(1)
)


# Compile model
model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)


# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)


# Make predictions
predictions = model.predict(X_test)


# Evaluate model
rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)


print("MAE:", mean_absolute_error(y_test, predictions))
print("MSE:", mean_squared_error(y_test, predictions))
print("RMSE:", rmse)
print("R²:", r2_score(y_test, predictions))
