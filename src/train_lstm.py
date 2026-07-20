import random
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Concatenate
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)


# ============================================================
# FINAL LSTM HYPERPARAMETERS
#
# Selected from hyperparameter tuning
# ============================================================

LSTM_UNITS = 50
DENSE_UNITS = 32
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 50


# ============================================================
# LOAD SAME PROCESSED DATA USED BY OTHER MODELS
# ============================================================

X_train_df = pd.read_csv("data/X_train_ue1.csv")
X_val_df = pd.read_csv("data/X_val_ue1.csv")
X_test_df = pd.read_csv("data/X_test_ue1.csv")

y_train_raw = pd.read_csv("data/y_train_ue1.csv").squeeze()
y_val_raw = pd.read_csv("data/y_val_ue1.csv").squeeze()
y_test_raw = pd.read_csv("data/y_test_ue1.csv").squeeze()


# ============================================================
# PREPARE INPUT FEATURES
#
# LSTM sequence:
# Lag3 -> Lag2 -> Lag1
#
# Additional network features:
# Jitter + CQI
# ============================================================

lag_features = [
    "UE1_throughput_lag3",
    "UE1_throughput_lag2",
    "UE1_throughput_lag1",
]

network_features = [
    "UE1-Jitter",
    "UE1-CQI",
]


# LSTM lag sequence
# Shape: (samples, 3 timesteps, 1 feature)

X_train_lags = (
    X_train_df[lag_features]
    .to_numpy()
    .reshape(-1, 3, 1)
)

X_val_lags = (
    X_val_df[lag_features]
    .to_numpy()
    .reshape(-1, 3, 1)
)

X_test_lags = (
    X_test_df[lag_features]
    .to_numpy()
    .reshape(-1, 3, 1)
)


# Current network features
# Shape: (samples, 2)

X_train_network = (
    X_train_df[network_features]
    .to_numpy()
)

X_val_network = (
    X_val_df[network_features]
    .to_numpy()
)

X_test_network = (
    X_test_df[network_features]
    .to_numpy()
)


print("\n========== LSTM Input Shapes ==========")
print("Training lag sequence:", X_train_lags.shape)
print("Validation lag sequence:", X_val_lags.shape)
print("Testing lag sequence:", X_test_lags.shape)

print("Training network features:", X_train_network.shape)
print("Validation network features:", X_val_network.shape)
print("Testing network features:", X_test_network.shape)


# ============================================================
# NORMALIZE TARGET
#
# X features are already normalized from preprocessing.
# Fit target scaler using training target only.
# ============================================================

target_scaler = MinMaxScaler()

y_train = target_scaler.fit_transform(
    y_train_raw
    .to_numpy()
    .reshape(-1, 1)
).flatten()

y_val = target_scaler.transform(
    y_val_raw
    .to_numpy()
    .reshape(-1, 1)
).flatten()

y_test = target_scaler.transform(
    y_test_raw
    .to_numpy()
    .reshape(-1, 1)
).flatten()


# ============================================================
# BUILD FINAL LSTM MODEL
# ============================================================

# Lag sequence input
lag_input = Input(
    shape=(3, 1),
    name="lag_input"
)

lstm_output = LSTM(
    LSTM_UNITS,
    name="lstm_layer"
)(
    lag_input
)


# Jitter and CQI input
network_input = Input(
    shape=(2,),
    name="network_input"
)


# Combine temporal output with current network features
combined = Concatenate(
    name="combined_features"
)(
    [
        lstm_output,
        network_input
    ]
)


# Dense layer
dense_output = Dense(
    DENSE_UNITS,
    activation="relu",
    name="dense_layer"
)(
    combined
)


# Final total UE1 throughput prediction
output = Dense(
    1,
    name="throughput_output"
)(
    dense_output
)


model = Model(
    inputs=[
        lag_input,
        network_input
    ],
    outputs=output
)


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="mean_squared_error"
)


# ============================================================
# EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# ============================================================
# TRAIN MODEL
# ============================================================

history = model.fit(
    [
        X_train_lags,
        X_train_network
    ],
    y_train,

    validation_data=(
        [
            X_val_lags,
            X_val_network
        ],
        y_val
    ),

    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


print(
    "\nEpochs Trained:",
    len(history.history["loss"])
)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

val_predictions_scaled = model.predict(
    [
        X_val_lags,
        X_val_network
    ],
    verbose=0
).flatten()


val_predictions = (
    target_scaler
    .inverse_transform(
        val_predictions_scaled
        .reshape(-1, 1)
    )
    .flatten()
)


val_actual = (
    y_val_raw
    .to_numpy()
)


# Validation metrics
val_mae = mean_absolute_error(
    val_actual,
    val_predictions
)

val_mse = mean_squared_error(
    val_actual,
    val_predictions
)

val_rmse = val_mse ** 0.5

val_r2 = r2_score(
    val_actual,
    val_predictions
)


print(
    "\n========== LSTM Validation Results =========="
)

print(f"Validation MAE  : {val_mae:.2f}")
print(f"Validation MSE  : {val_mse:.2f}")
print(f"Validation RMSE : {val_rmse:.2f}")
print(f"Validation R²   : {val_r2:.4f}")


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

test_predictions_scaled = model.predict(
    [
        X_test_lags,
        X_test_network
    ],
    verbose=0
).flatten()


test_predictions = (
    target_scaler
    .inverse_transform(
        test_predictions_scaled
        .reshape(-1, 1)
    )
    .flatten()
)


test_actual = (
    y_test_raw
    .to_numpy()
)


# Test metrics
test_mae = mean_absolute_error(
    test_actual,
    test_predictions
)

test_mse = mean_squared_error(
    test_actual,
    test_predictions
)

test_rmse = test_mse ** 0.5

test_r2 = r2_score(
    test_actual,
    test_predictions
)


print(
    "\n========== LSTM Test Results =========="
)

print(f"Test MAE  : {test_mae:.2f}")
print(f"Test MSE  : {test_mse:.2f}")
print(f"Test RMSE : {test_rmse:.2f}")
print(f"Test R²   : {test_r2:.4f}")


# ============================================================
# SHOW FIRST 10 TEST PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "Actual Total UE1 Throughput":
        test_actual,

    "Predicted Total UE1 Throughput":
        test_predictions
})


print(
    "\nFirst 10 Test Predictions"
)

print(
    results.head(10)
)


# ============================================================
# ACTUAL VS PREDICTED PLOT
# ============================================================

plt.rcParams[
    "font.family"
] = "DejaVu Serif"


num_samples = 200


fig, ax = plt.subplots(
    figsize=(6.8, 3.8)
)


ax.plot(
    results.index[:num_samples],
    results[
        "Actual Total UE1 Throughput"
    ].iloc[:num_samples],
    label="Actual Throughput",
    linewidth=1.2
)


ax.plot(
    results.index[:num_samples],
    results[
        "Predicted Total UE1 Throughput"
    ].iloc[:num_samples],
    label="Predicted Throughput",
    linewidth=0.9,
    linestyle="--"
)


ax.set_xlabel(
    "Sample Index",
    fontsize=11
)

ax.set_ylabel(
    "Total UE1 Throughput (Bytes/s)",
    fontsize=11
)


ax.tick_params(
    axis="both",
    labelsize=10
)


ax.yaxis.set_major_formatter(
    StrMethodFormatter(
        "{x:,.0f}"
    )
)


ax.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    alpha=0.25
)


ax.legend(
    loc="upper center",
    fontsize=9,
    frameon=True
)


plt.tight_layout()


plt.savefig(
    "plots/lstm_actual_vs_predicted.png",
    dpi=600,
    bbox_inches="tight"
)


plt.close()