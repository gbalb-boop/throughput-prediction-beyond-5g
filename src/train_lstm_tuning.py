import pandas as pd
import numpy as np
import tensorflow as tf

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
    Dropout,
    Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# Change this to 1, 2, 3, 4, 5, or 6
RUN_PART = 6


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# LOAD THE SAME PROCESSED DATA USED BY THE OTHER MODELS
# ============================================================

X_train_df = pd.read_csv("data/X_train_ue1.csv")
X_val_df = pd.read_csv("data/X_val_ue1.csv")
X_test_df = pd.read_csv("data/X_test_ue1.csv")

y_train_raw = pd.read_csv(
    "data/y_train_ue1.csv"
).squeeze()

y_val_raw = pd.read_csv(
    "data/y_val_ue1.csv"
).squeeze()

y_test_raw = pd.read_csv(
    "data/y_test_ue1.csv"
).squeeze()


# ============================================================
# PREPARE LSTM INPUTS
#
# Same information as the other models:
#
# Lag3 -> Lag2 -> Lag1 = 3-step temporal sequence
# Jitter + CQI       = current network features
#
# The sequence is ordered from oldest to newest.
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


# LSTM sequence inputs
# Shape:
# (samples, 3 timesteps, 1 feature)

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


# Jitter and CQI inputs
# Shape:
# (samples, 2 features)

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

print(
    "Training lag sequence:",
    X_train_lags.shape
)

print(
    "Validation lag sequence:",
    X_val_lags.shape
)

print(
    "Testing lag sequence:",
    X_test_lags.shape
)

print(
    "Training network features:",
    X_train_network.shape
)

print(
    "Validation network features:",
    X_val_network.shape
)

print(
    "Testing network features:",
    X_test_network.shape
)


# ============================================================
# NORMALIZE TARGET USING TRAINING DATA ONLY
#
# X features were already normalized during preprocessing.
#
# Scaling y helps neural-network training.
# Metrics are converted back to original throughput units.
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
# HYPERPARAMETER CONFIGURATIONS
#
# 6 parts x 5 models = 30 total configurations
# ============================================================


# ============================================================
# PART 1
# Test number of LSTM units
# ============================================================

if RUN_PART == 1:

    param_grid = [
        {
            "lstm_units": 16,
            "dense_units": 16,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 32
        },
        {
            "lstm_units": 32,
            "dense_units": 16,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 32
        },
        {
            "lstm_units": 50,
            "dense_units": 16,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 32
        },
        {
            "lstm_units": 64,
            "dense_units": 16,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 32
        },
        {
            "lstm_units": 100,
            "dense_units": 16,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 32
        },
    ]


# ============================================================
# PART 2
# Test Dense layer size after combining LSTM + Jitter/CQI
# ============================================================

elif RUN_PART == 2:

    param_grid = [
        {
            "lstm_units": 50,
            "dense_units": 8,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 32
        },
        {
            "lstm_units": 50,
            "dense_units": 16,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 32
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 32
        },
        {
            "lstm_units": 50,
            "dense_units": 50,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 32
        },
        {
            "lstm_units": 50,
            "dense_units": 64,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 32
        },
    ]


# ============================================================
# PART 3
# Test dropout
# ============================================================

elif RUN_PART == 3:

    param_grid = [
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 32
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "batch_size": 32
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.3,
            "learning_rate": 0.001,
            "batch_size": 32
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.4,
            "learning_rate": 0.001,
            "batch_size": 32
        },
    ]


# ============================================================
# PART 4
# Test learning rate
# ============================================================

elif RUN_PART == 4:

    param_grid = [
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.0,
            "learning_rate": 0.0001,
            "batch_size": 32
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.0,
            "learning_rate": 0.0005,
            "batch_size": 32
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 32
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.0,
            "learning_rate": 0.002,
            "batch_size": 32
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.0,
            "learning_rate": 0.005,
            "batch_size": 32
        },
    ]


# ============================================================
# PART 5
# Test batch size
# ============================================================

elif RUN_PART == 5:

    param_grid = [
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 16
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 32
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 64
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 128
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.0,
            "learning_rate": 0.001,
            "batch_size": 256
        },
    ]


# ============================================================
# PART 6
# Additional combinations around promising settings
# ============================================================

elif RUN_PART == 6:

    param_grid = [
        {
            "lstm_units": 32,
            "dense_units": 32,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "batch_size": 32
        },
        {
            "lstm_units": 64,
            "dense_units": 32,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "batch_size": 32
        },
        {
            "lstm_units": 64,
            "dense_units": 64,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "batch_size": 32
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.1,
            "learning_rate": 0.0005,
            "batch_size": 32
        },
        {
            "lstm_units": 50,
            "dense_units": 32,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "batch_size": 64
        },
    ]


else:

    raise ValueError(
        "RUN_PART must be 1, 2, 3, 4, 5, or 6."
    )


# ============================================================
# TUNING SETUP
# ============================================================

best_model = None
best_params = None

best_val_rmse = float("inf")
best_val_metrics = None

all_results = []


print(
    f"\n========== LSTM Hyperparameter Tuning: "
    f"Part {RUN_PART} =========="
)


# ============================================================
# TRAIN EACH CONFIGURATION
# ============================================================

for i, params in enumerate(
    param_grid,
    start=1
):

    print(
        f"\nCombination {i}/5"
    )

    print(
        "Parameters:",
        params
    )


    # Clear previous model
    tf.keras.backend.clear_session()

    np.random.seed(42)
    tf.random.set_seed(42)


    # ========================================================
    # LSTM BRANCH
    #
    # Input:
    # Lag3 -> Lag2 -> Lag1
    # ========================================================

    lag_input = Input(
        shape=(3, 1),
        name="lag_input"
    )

    lstm_output = LSTM(
        params["lstm_units"],
        name="lstm_layer"
    )(
        lag_input
    )


    # Optional dropout after LSTM
    if params["dropout"] > 0:

        lstm_output = Dropout(
            params["dropout"],
            name="lstm_dropout"
        )(
            lstm_output
        )


    # ========================================================
    # NETWORK FEATURE BRANCH
    #
    # Input:
    # Jitter + CQI
    # ========================================================

    network_input = Input(
        shape=(2,),
        name="network_input"
    )


    # ========================================================
    # COMBINE BOTH INPUT BRANCHES
    # ========================================================

    combined = Concatenate(
        name="combined_features"
    )(
        [
            lstm_output,
            network_input
        ]
    )


    # Dense layer after combining all information
    dense_output = Dense(
        params["dense_units"],
        activation="relu",
        name="dense_layer"
    )(
        combined
    )


    # Optional dropout after Dense layer
    if params["dropout"] > 0:

        dense_output = Dropout(
            params["dropout"],
            name="dense_dropout"
        )(
            dense_output
        )


    # Final throughput prediction
    output = Dense(
        1,
        name="throughput_output"
    )(
        dense_output
    )


    # Build model
    model = Model(
        inputs=[
            lag_input,
            network_input
        ],
        outputs=output
    )


    # Compile model
    model.compile(

        optimizer=Adam(
            learning_rate=
            params["learning_rate"]
        ),

        loss="mean_squared_error"
    )


    # Early stopping
    early_stopping = EarlyStopping(

        monitor="val_loss",

        patience=5,

        restore_best_weights=True
    )


    # ========================================================
    # TRAIN MODEL
    # ========================================================

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

        epochs=50,

        batch_size=
        params["batch_size"],

        callbacks=[
            early_stopping
        ],

        verbose=0
    )


    # ========================================================
    # VALIDATION PREDICTIONS
    # ========================================================

    val_predictions_scaled = model.predict(

        [
            X_val_lags,
            X_val_network
        ],

        verbose=0

    ).flatten()


    # Convert predictions back to original throughput units
    val_predictions = (
        target_scaler
        .inverse_transform(
            val_predictions_scaled
            .reshape(-1, 1)
        )
        .flatten()
    )


    # Actual validation throughput
    val_actual = (
        y_val_raw
        .to_numpy()
    )


    # ========================================================
    # VALIDATION METRICS
    # ========================================================

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


    epochs_trained = len(
        history.history["loss"]
    )


    all_results.append({

        "Parameters":
            params,

        "Validation MAE":
            val_mae,

        "Validation MSE":
            val_mse,

        "Validation RMSE":
            val_rmse,

        "Validation R2":
            val_r2,

        "Epochs Trained":
            epochs_trained
    })


    print(
        f"Epochs Trained  : "
        f"{epochs_trained}"
    )

    print(
        f"Validation MAE  : "
        f"{val_mae:.2f}"
    )

    print(
        f"Validation MSE  : "
        f"{val_mse:.2f}"
    )

    print(
        f"Validation RMSE : "
        f"{val_rmse:.2f}"
    )

    print(
        f"Validation R²   : "
        f"{val_r2:.4f}"
    )


    # ========================================================
    # SAVE BEST MODEL FOR THIS PART
    #
    # Selection is based ONLY on validation RMSE
    # ========================================================

    if val_rmse < best_val_rmse:

        best_val_rmse = val_rmse

        best_model = model

        best_params = params

        best_val_metrics = {

            "MAE":
                val_mae,

            "MSE":
                val_mse,

            "RMSE":
                val_rmse,

            "R2":
                val_r2,

            "Epochs":
                epochs_trained
        }


# ============================================================
# SUMMARY FOR THIS PART
# ============================================================

print(
    "\n========== Summary For This Part =========="
)

summary = pd.DataFrame(
    all_results
)

summary = summary.sort_values(
    "Validation RMSE"
)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# BEST MODEL FOR THIS PART
# ============================================================

print(
    "\n========== Best LSTM Model For This Part =========="
)

print(
    "Best Parameters:",
    best_params
)

print(
    f"Best Validation MAE  : "
    f"{best_val_metrics['MAE']:.2f}"
)

print(
    f"Best Validation MSE  : "
    f"{best_val_metrics['MSE']:.2f}"
)

print(
    f"Best Validation RMSE : "
    f"{best_val_metrics['RMSE']:.2f}"
)

print(
    f"Best Validation R²   : "
    f"{best_val_metrics['R2']:.4f}"
)

print(
    f"Epochs Trained       : "
    f"{best_val_metrics['Epochs']}"
)