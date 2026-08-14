from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor


DATA_PATH = Path("data")
MODELS_PATH = Path("models")


# --------------------------------------------------
# Expected final metrics from the project
# --------------------------------------------------

EXPECTED_RESULTS = {
    "Linear Regression": {
        "mae": 6098.99,
        "rmse": 8575.28,
        "r2": 0.5658,
    },
    "Random Forest": {
        "mae": 4956.63,
        "rmse": 7624.63,
        "r2": 0.6567,
    },
    "XGBoost": {
        "mae": 4987.59,
        "rmse": 7621.46,
        "r2": 0.6570,
    },
    "LSTM": {
        "mae": 5367.59,
        "rmse": 7867.20,
        "r2": 0.6345,
    },
}


# --------------------------------------------------
# Load processed test data
# --------------------------------------------------

X_test = pd.read_csv(
    DATA_PATH / "X_test_ue1.csv"
)

y_test = pd.read_csv(
    DATA_PATH / "y_test_ue1.csv"
).squeeze()


SELECTED_FEATURES = [
    "UE1_throughput_lag1",
    "UE1_throughput_lag2",
    "UE1_throughput_lag3",
    "UE1-Jitter",
    "UE1-CQI",
]

X_test = X_test[SELECTED_FEATURES]


# --------------------------------------------------
# Metric helper
# --------------------------------------------------

def evaluate_model(
    name,
    y_true,
    predictions,
):
    mae = mean_absolute_error(
        y_true,
        predictions,
    )

    mse = mean_squared_error(
        y_true,
        predictions,
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_true,
        predictions,
    )

    expected = EXPECTED_RESULTS[name]

    print(
        f"\n========== {name} =========="
    )

    print(f"MAE  : {mae:.2f}")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")

    print("\nExpected:")
    print(
        f"MAE  : {expected['mae']:.2f}"
    )
    print(
        f"RMSE : {expected['rmse']:.2f}"
    )
    print(
        f"R²   : {expected['r2']:.4f}"
    )

    mae_match = np.isclose(
        mae,
        expected["mae"],
        atol=0.05,
    )

    rmse_match = np.isclose(
        rmse,
        expected["rmse"],
        atol=0.05,
    )

    r2_match = np.isclose(
        r2,
        expected["r2"],
        atol=0.0001,
    )

    if (
        mae_match
        and rmse_match
        and r2_match
    ):
        print(
            "\nSTATUS: PASS"
        )
    else:
        print(
            "\nSTATUS: CHECK"
        )

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Status": (
            "PASS"
            if (
                mae_match
                and rmse_match
                and r2_match
            )
            else "CHECK"
        ),
    }


all_results = []


# --------------------------------------------------
# Linear Regression
# --------------------------------------------------

linear_model = joblib.load(
    MODELS_PATH
    / "linear_regression.joblib"
)

linear_predictions = (
    linear_model.predict(X_test)
)

all_results.append(
    evaluate_model(
        "Linear Regression",
        y_test,
        linear_predictions,
    )
)


# --------------------------------------------------
# Random Forest
# --------------------------------------------------

random_forest = joblib.load(
    MODELS_PATH
    / "random_forest.joblib"
)

rf_predictions = (
    random_forest.predict(X_test)
)

all_results.append(
    evaluate_model(
        "Random Forest",
        y_test,
        rf_predictions,
    )
)


# --------------------------------------------------
# XGBoost joblib artifact
# --------------------------------------------------

xgboost_joblib = joblib.load(
    MODELS_PATH
    / "xgboost.joblib"
)

xgb_predictions = (
    xgboost_joblib.predict(X_test)
)

all_results.append(
    evaluate_model(
        "XGBoost",
        y_test,
        xgb_predictions,
    )
)


# --------------------------------------------------
# Verify native XGBoost JSON artifact
# --------------------------------------------------

xgboost_json = XGBRegressor()

xgboost_json.load_model(
    MODELS_PATH / "xgboost.json"
)

json_predictions = (
    xgboost_json.predict(X_test)
)

same_xgboost_predictions = (
    np.allclose(
        xgb_predictions,
        json_predictions,
        atol=1e-6,
    )
)

print(
    "\n========== XGBoost Artifact Comparison =========="
)

print(
    "joblib and JSON predictions match:",
    same_xgboost_predictions,
)


# --------------------------------------------------
# LSTM
# --------------------------------------------------

lstm_model = tf.keras.models.load_model(
    MODELS_PATH / "lstm.keras"
)

target_scaler = joblib.load(
    MODELS_PATH
    / "lstm_target_scaler.joblib"
)


LAG_FEATURES = [
    "UE1_throughput_lag3",
    "UE1_throughput_lag2",
    "UE1_throughput_lag1",
]

NETWORK_FEATURES = [
    "UE1-Jitter",
    "UE1-CQI",
]


X_test_lags = (
    X_test[LAG_FEATURES]
    .to_numpy()
    .reshape(-1, 3, 1)
)

X_test_network = (
    X_test[NETWORK_FEATURES]
    .to_numpy()
)


lstm_predictions_scaled = (
    lstm_model.predict(
        [
            X_test_lags,
            X_test_network,
        ],
        verbose=0,
    )
    .flatten()
)

lstm_predictions = (
    target_scaler
    .inverse_transform(
        lstm_predictions_scaled
        .reshape(-1, 1)
    )
    .flatten()
)

all_results.append(
    evaluate_model(
        "LSTM",
        y_test,
        lstm_predictions,
    )
)


# --------------------------------------------------
# Final summary
# --------------------------------------------------

summary = pd.DataFrame(
    all_results
)

print(
    "\n=========================================="
)

print(
    "Saved Model Verification Summary"
)

print(
    "=========================================="
)

print(
    summary.to_string(
        index=False
    )
)

if (
    summary["Status"] == "PASS"
).all() and same_xgboost_predictions:

    print(
        "\nAll saved final model artifacts verified successfully."
    )

else:

    print(
        "\nOne or more saved artifacts need to be checked."
    )