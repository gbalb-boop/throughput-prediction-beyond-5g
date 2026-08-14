from pathlib import Path
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    Concatenate,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


DATA_PATH = Path("data")

# Change this to 1, 2, 3, 4, 5, or 6
RUN_PART = 6

SEED = 42
MAX_EPOCHS = 50

LAG_FEATURES = [
    "UE1_throughput_lag3",
    "UE1_throughput_lag2",
    "UE1_throughput_lag1",
]

NETWORK_FEATURES = [
    "UE1-Jitter",
    "UE1-CQI",
]


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)


# --------------------------------------------------
# Load training and validation data
# --------------------------------------------------

X_train_df = pd.read_csv(DATA_PATH / "X_train_ue1.csv")
X_val_df = pd.read_csv(DATA_PATH / "X_val_ue1.csv")

y_train_raw = pd.read_csv(
    DATA_PATH / "y_train_ue1.csv"
).squeeze()

y_val_raw = pd.read_csv(
    DATA_PATH / "y_val_ue1.csv"
).squeeze()


# --------------------------------------------------
# Prepare LSTM inputs
#
# Lag sequence is ordered oldest -> newest:
# Lag3 -> Lag2 -> Lag1
# --------------------------------------------------

X_train_lags = (
    X_train_df[LAG_FEATURES]
    .to_numpy()
    .reshape(-1, 3, 1)
)

X_val_lags = (
    X_val_df[LAG_FEATURES]
    .to_numpy()
    .reshape(-1, 3, 1)
)

X_train_network = (
    X_train_df[NETWORK_FEATURES]
    .to_numpy()
)

X_val_network = (
    X_val_df[NETWORK_FEATURES]
    .to_numpy()
)

print("\n========== LSTM Input Shapes ==========")
print("Training lag sequence:", X_train_lags.shape)
print("Validation lag sequence:", X_val_lags.shape)
print("Training network features:", X_train_network.shape)
print("Validation network features:", X_val_network.shape)


# --------------------------------------------------
# Scale target using training data only
#
# Input features were already normalized during
# preprocessing. Target scaling is performed here
# specifically for neural-network training.
# --------------------------------------------------

target_scaler = MinMaxScaler()

y_train = target_scaler.fit_transform(
    y_train_raw.to_numpy().reshape(-1, 1)
).flatten()

y_val = target_scaler.transform(
    y_val_raw.to_numpy().reshape(-1, 1)
).flatten()


# --------------------------------------------------
# Hyperparameter configurations
#
# 6 parts x 5 configurations = 30 total
# --------------------------------------------------

if RUN_PART == 1:
    # Test number of LSTM units
    param_grid = [
        {
            "lstm_units": 16,
            "dense_units": 16,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 32,
        },
        {
            "lstm_units": 32,
            "dense_units": 16,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 32,
        },
        {
            "lstm_units": 50,
            "dense_units": 16,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 32,
        },
        {
            "lstm_units": 64,
            "dense_units": 16,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 32,
        },
        {
            "lstm_units": 100,
            "dense_units": 16,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 32,
        },
    ]

elif RUN_PART == 2:
    # Test dense layer size
    param_grid = [
        {
            "lstm_units": 50,
            "dense_units": 8,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 32,
        },
        {
            "lstm_units": 50,
            "dense_units": 16,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 32,
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 32,
        },
        {
            "lstm_units": 50,
            "dense_units": 50,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 32,
        },
        {
            "lstm_units": 50,
            "dense_units": 64,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 32,
        },
    ]

elif RUN_PART == 3:
    # Test dropout
    param_grid = [
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 32,
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "batch_size": 32,
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.3,
            "learning_rate": 0.001,
            "batch_size": 32,
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.4,
            "learning_rate": 0.001,
            "batch_size": 32,
        },
    ]

elif RUN_PART == 4:
    # Test learning rate
    param_grid = [
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.0,
            "learning_rate": 0.0001,
            "batch_size": 32,
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.0,
            "learning_rate": 0.0005,
            "batch_size": 32,
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 32,
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.0,
            "learning_rate": 0.002,
            "batch_size": 32,
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.0,
            "learning_rate": 0.005,
            "batch_size": 32,
        },
    ]

elif RUN_PART == 5:
    # Test batch size
    param_grid = [
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 16,
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 32,
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 64,
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 128,
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 256,
        },
    ]

elif RUN_PART == 6:
    # Additional combinations around promising settings
    param_grid = [
        {
            "lstm_units": 32,
            "dense_units": 32,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "batch_size": 32,
        },
        {
            "lstm_units": 64,
            "dense_units": 32,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "batch_size": 32,
        },
        {
            "lstm_units": 64,
            "dense_units": 64,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "batch_size": 32,
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.1,
            "learning_rate": 0.0005,
            "batch_size": 32,
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "batch_size": 64,
        },
    ]

else:
    raise ValueError(
        "RUN_PART must be 1, 2, 3, 4, 5, or 6."
    )


# --------------------------------------------------
# Tune using validation RMSE
# --------------------------------------------------

best_params = None
best_val_rmse = float("inf")
best_val_metrics = None

all_results = []

print(
    f"\n========== LSTM Hyperparameter Tuning: "
    f"Part {RUN_PART} =========="
)


for i, params in enumerate(param_grid, start=1):
    print(f"\nCombination {i}/{len(param_grid)}")
    print("Parameters:", params)

    # Clear the previous TensorFlow model
    tf.keras.backend.clear_session()

    # Reset random state for each configuration
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    # --------------------------------------------------
    # Build LSTM branch
    # --------------------------------------------------

    lag_input = Input(
        shape=(3, 1),
        name="lag_input",
    )

    lstm_output = LSTM(
        params["lstm_units"],
        name="lstm_layer",
    )(lag_input)

    if params["dropout"] > 0:
        lstm_output = Dropout(
            params["dropout"],
            name="lstm_dropout",
        )(lstm_output)

    # --------------------------------------------------
    # Network-feature branch
    # --------------------------------------------------

    network_input = Input(
        shape=(2,),
        name="network_input",
    )

    # --------------------------------------------------
    # Combine inputs
    # --------------------------------------------------

    combined = Concatenate(
        name="combined_features",
    )([
        lstm_output,
        network_input,
    ])

    dense_output = Dense(
        params["dense_units"],
        activation="relu",
        name="dense_layer",
    )(combined)

    if params["dropout"] > 0:
        dense_output = Dropout(
            params["dropout"],
            name="dense_dropout",
        )(dense_output)

    output = Dense(
        1,
        name="throughput_output",
    )(dense_output)

    model = Model(
        inputs=[
            lag_input,
            network_input,
        ],
        outputs=output,
    )

    # --------------------------------------------------
    # Compile model
    # --------------------------------------------------

    model.compile(
        optimizer=Adam(
            learning_rate=params["learning_rate"],
        ),
        loss="mean_squared_error",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    # --------------------------------------------------
    # Train model
    # --------------------------------------------------

    history = model.fit(
        [
            X_train_lags,
            X_train_network,
        ],
        y_train,
        validation_data=(
            [
                X_val_lags,
                X_val_network,
            ],
            y_val,
        ),
        epochs=MAX_EPOCHS,
        batch_size=params["batch_size"],
        callbacks=[early_stopping],
        verbose=0,
    )

    # --------------------------------------------------
    # Validation predictions
    # --------------------------------------------------

    val_predictions_scaled = model.predict(
        [
            X_val_lags,
            X_val_network,
        ],
        verbose=0,
    ).flatten()

    val_predictions = target_scaler.inverse_transform(
        val_predictions_scaled.reshape(-1, 1)
    ).flatten()

    val_actual = y_val_raw.to_numpy()

    # --------------------------------------------------
    # Validation metrics
    # --------------------------------------------------

    val_mae = mean_absolute_error(
        val_actual,
        val_predictions,
    )

    val_mse = mean_squared_error(
        val_actual,
        val_predictions,
    )

    val_rmse = val_mse ** 0.5

    val_r2 = r2_score(
        val_actual,
        val_predictions,
    )

    epochs_trained = len(
        history.history["loss"]
    )

    all_results.append({
        "Parameters": params,
        "Validation MAE": val_mae,
        "Validation MSE": val_mse,
        "Validation RMSE": val_rmse,
        "Validation R2": val_r2,
        "Epochs Trained": epochs_trained,
    })

    print(f"Epochs Trained  : {epochs_trained}")
    print(f"Validation MAE  : {val_mae:.2f}")
    print(f"Validation MSE  : {val_mse:.2f}")
    print(f"Validation RMSE : {val_rmse:.2f}")
    print(f"Validation R²   : {val_r2:.4f}")

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_params = params

        best_val_metrics = {
            "MAE": val_mae,
            "MSE": val_mse,
            "RMSE": val_rmse,
            "R2": val_r2,
            "Epochs": epochs_trained,
        }


# --------------------------------------------------
# Display results
# --------------------------------------------------

summary = pd.DataFrame(all_results)
summary = summary.sort_values("Validation RMSE")

summary["Parameters"] = summary["Parameters"].astype(str)

print("\n========== Summary For This Part ==========")
print(summary.to_string(index=False))

print("\n========== Best LSTM Model For This Part ==========")
print("Best Parameters:", best_params)
print(f"Best Validation MAE  : {best_val_metrics['MAE']:.2f}")
print(f"Best Validation MSE  : {best_val_metrics['MSE']:.2f}")
print(f"Best Validation RMSE : {best_val_metrics['RMSE']:.2f}")
print(f"Best Validation R²   : {best_val_metrics['R2']:.4f}")
print(f"Epochs Trained       : {best_val_metrics['Epochs']}")

print(
    "\nTest-set evaluation is intentionally excluded from tuning. "
    "The final selected configuration is evaluated in "
    "train_lstm.py."
)
