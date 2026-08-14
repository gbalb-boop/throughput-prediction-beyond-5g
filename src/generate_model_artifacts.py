from pathlib import Path
import random

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Concatenate,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path("data")
MODELS_PATH = Path("models")

DATA_FILE = DATA_PATH / "ue-lte-network-traffic-stats.csv"

MODELS_PATH.mkdir(exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)


# ============================================================
# LOAD RAW DATASET
# ============================================================

df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully")
print("Shape:", df.shape)


# ============================================================
# CREATE TOTAL UE1 THROUGHPUT
# ============================================================

df["UE1_total_throughput"] = (
    df["UE1: web-rtc"]
    + df["UE1: sipp"]
    + df["UE1: web-server"]
)

TARGET_COLUMN = "UE1_total_throughput"


# ============================================================
# CREATE UE1 DATAFRAME
# ============================================================

ue1_df = df[
    [
        TARGET_COLUMN,
        "UE1-Jitter",
        "UE1-CQI",
    ]
].copy()


# ============================================================
# CREATE LAG FEATURES
# ============================================================

ue1_df["UE1_throughput_lag1"] = (
    ue1_df[TARGET_COLUMN].shift(1)
)

ue1_df["UE1_throughput_lag2"] = (
    ue1_df[TARGET_COLUMN].shift(2)
)

ue1_df["UE1_throughput_lag3"] = (
    ue1_df[TARGET_COLUMN].shift(3)
)

ue1_df = ue1_df.dropna()


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

SELECTED_FEATURES = [
    "UE1_throughput_lag1",
    "UE1_throughput_lag2",
    "UE1_throughput_lag3",
    "UE1-Jitter",
    "UE1-CQI",
]

features = ue1_df[SELECTED_FEATURES]
target = ue1_df[TARGET_COLUMN]


# ============================================================
# CHRONOLOGICAL 70 / 15 / 15 SPLIT
# ============================================================

n = len(ue1_df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train_raw = features.iloc[:train_end].copy()
X_val_raw = features.iloc[train_end:val_end].copy()
X_test_raw = features.iloc[val_end:].copy()

y_train_raw = target.iloc[:train_end].copy()
y_val_raw = target.iloc[train_end:val_end].copy()
y_test_raw = target.iloc[val_end:].copy()

print("\nDataset split:")
print("Train:", X_train_raw.shape)
print("Validation:", X_val_raw.shape)
print("Test:", X_test_raw.shape)


# ============================================================
# FEATURE SCALER
#
# Fit ONLY on training data.
# ============================================================

feature_scaler = MinMaxScaler()

X_train = pd.DataFrame(
    feature_scaler.fit_transform(X_train_raw),
    columns=SELECTED_FEATURES,
    index=X_train_raw.index,
)

X_val = pd.DataFrame(
    feature_scaler.transform(X_val_raw),
    columns=SELECTED_FEATURES,
    index=X_val_raw.index,
)

X_test = pd.DataFrame(
    feature_scaler.transform(X_test_raw),
    columns=SELECTED_FEATURES,
    index=X_test_raw.index,
)


# Save feature scaler
joblib.dump(
    feature_scaler,
    MODELS_PATH / "feature_scaler.joblib",
)

print(
    "\nSaved:",
    MODELS_PATH / "feature_scaler.joblib",
)


# ============================================================
# LINEAR REGRESSION
# ============================================================

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train_raw,
)

joblib.dump(
    linear_model,
    MODELS_PATH / "linear_regression.joblib",
)

print(
    "Saved:",
    MODELS_PATH / "linear_regression.joblib",
)


# ============================================================
# RANDOM FOREST
#
# Final configuration from hyperparameter tuning
# ============================================================

random_forest = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=2,
    random_state=SEED,
    n_jobs=1,
)

random_forest.fit(
    X_train,
    y_train_raw,
)

joblib.dump(
    random_forest,
    MODELS_PATH / "random_forest.joblib",
)

print(
    "Saved:",
    MODELS_PATH / "random_forest.joblib",
)


# ============================================================
# XGBOOST
#
# Final configuration from hyperparameter tuning
# ============================================================

xgboost_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=1.0,
    objective="reg:squarederror",
    random_state=SEED,
)

xgboost_model.fit(
    X_train,
    y_train_raw,
)


# Native XGBoost model
xgboost_model.save_model(
    MODELS_PATH / "xgboost.json"
)

print(
    "Saved:",
    MODELS_PATH / "xgboost.json",
)


# Joblib copy for convenient Python loading
joblib.dump(
    xgboost_model,
    MODELS_PATH / "xgboost.joblib",
)

print(
    "Saved:",
    MODELS_PATH / "xgboost.joblib",
)


# ============================================================
# LSTM TARGET SCALER
#
# The final LSTM scales y separately.
# ============================================================

target_scaler = MinMaxScaler()

y_train_scaled = target_scaler.fit_transform(
    y_train_raw
    .to_numpy()
    .reshape(-1, 1)
).flatten()

y_val_scaled = target_scaler.transform(
    y_val_raw
    .to_numpy()
    .reshape(-1, 1)
).flatten()

joblib.dump(
    target_scaler,
    MODELS_PATH / "lstm_target_scaler.joblib",
)

print(
    "Saved:",
    MODELS_PATH / "lstm_target_scaler.joblib",
)


# ============================================================
# PREPARE LSTM INPUTS
#
# Sequence order:
# Lag3 -> Lag2 -> Lag1
# ============================================================

LAG_FEATURES = [
    "UE1_throughput_lag3",
    "UE1_throughput_lag2",
    "UE1_throughput_lag1",
]

NETWORK_FEATURES = [
    "UE1-Jitter",
    "UE1-CQI",
]


X_train_lags = (
    X_train[LAG_FEATURES]
    .to_numpy()
    .reshape(-1, 3, 1)
)

X_val_lags = (
    X_val[LAG_FEATURES]
    .to_numpy()
    .reshape(-1, 3, 1)
)

X_train_network = (
    X_train[NETWORK_FEATURES]
    .to_numpy()
)

X_val_network = (
    X_val[NETWORK_FEATURES]
    .to_numpy()
)


# ============================================================
# BUILD FINAL LSTM
# ============================================================

LSTM_UNITS = 50
DENSE_UNITS = 32
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 50


lag_input = Input(
    shape=(3, 1),
    name="lag_input",
)

lstm_output = LSTM(
    LSTM_UNITS,
    name="lstm_layer",
)(lag_input)


network_input = Input(
    shape=(2,),
    name="network_input",
)


combined = Concatenate(
    name="combined_features",
)([
    lstm_output,
    network_input,
])


dense_output = Dense(
    DENSE_UNITS,
    activation="relu",
    name="dense_layer",
)(combined)


output = Dense(
    1,
    name="throughput_output",
)(dense_output)


lstm_model = Model(
    inputs=[
        lag_input,
        network_input,
    ],
    outputs=output,
)


lstm_model.compile(
    optimizer=Adam(
        learning_rate=LEARNING_RATE,
    ),
    loss="mean_squared_error",
)


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)


# ============================================================
# TRAIN FINAL LSTM
# ============================================================

history = lstm_model.fit(
    [
        X_train_lags,
        X_train_network,
    ],
    y_train_scaled,

    validation_data=(
        [
            X_val_lags,
            X_val_network,
        ],
        y_val_scaled,
    ),

    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping,
    ],

    verbose=1,
)

print(
    "\nLSTM epochs trained:",
    len(history.history["loss"]),
)


# ============================================================
# SAVE FINAL LSTM
# ============================================================

lstm_model.save(
    MODELS_PATH / "lstm.keras"
)

print(
    "Saved:",
    MODELS_PATH / "lstm.keras",
)


# ============================================================
# FINISHED
# ============================================================

print(
    "\n=========================================="
)

print(
    "Final model artifacts generated successfully"
)

print(
    "=========================================="
)

print("\nmodels/")

for file in sorted(MODELS_PATH.iterdir()):
    if file.is_file():
        print("  ", file.name)