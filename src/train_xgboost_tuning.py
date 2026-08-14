from pathlib import Path

import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


DATA_PATH = Path("data")

# Change this to 1, 2, 3, 4, 5, or 6
RUN_PART = 6

SELECTED_FEATURES = [
    "UE1_throughput_lag1",
    "UE1_throughput_lag2",
    "UE1_throughput_lag3",
    "UE1-Jitter",
    "UE1-CQI",
]


# --------------------------------------------------
# Load training and validation data
# --------------------------------------------------

X_train = pd.read_csv(DATA_PATH / "X_train_ue1.csv")
X_val = pd.read_csv(DATA_PATH / "X_val_ue1.csv")

y_train = pd.read_csv(DATA_PATH / "y_train_ue1.csv").squeeze()
y_val = pd.read_csv(DATA_PATH / "y_val_ue1.csv").squeeze()

X_train = X_train[SELECTED_FEATURES]
X_val = X_val[SELECTED_FEATURES]


# --------------------------------------------------
# Hyperparameter search configurations
# --------------------------------------------------

if RUN_PART == 1:
    param_grid = [
        {
            "n_estimators": 100,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    ]

elif RUN_PART == 2:
    param_grid = [
        {
            "n_estimators": 150,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 250,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 7,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    ]

elif RUN_PART == 3:
    param_grid = [
        {
            "n_estimators": 200,
            "learning_rate": 0.01,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 300,
            "learning_rate": 0.01,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 400,
            "learning_rate": 0.01,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 150,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    ]

elif RUN_PART == 4:
    param_grid = [
        {
            "n_estimators": 150,
            "learning_rate": 0.03,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.03,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 250,
            "learning_rate": 0.03,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 7,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 250,
            "learning_rate": 0.05,
            "max_depth": 7,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    ]

elif RUN_PART == 5:
    param_grid = [
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.6,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.7,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 1.0,
            "colsample_bytree": 0.8,
        },
    ]

elif RUN_PART == 6:
    param_grid = [
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.6,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
        },
        {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 1.0,
        },
    ]

else:
    raise ValueError("RUN_PART must be 1, 2, 3, 4, 5, or 6.")


# --------------------------------------------------
# Tune using validation RMSE
# --------------------------------------------------

best_params = None
best_val_rmse = float("inf")
best_val_metrics = None

all_results = []

print(
    f"\n========== XGBoost Hyperparameter Tuning: "
    f"Part {RUN_PART} =========="
)

for i, params in enumerate(param_grid, start=1):
    print(f"\nCombination {i}/{len(param_grid)}")

    model = XGBRegressor(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
    )

    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)

    val_mae = mean_absolute_error(y_val, val_predictions)
    val_mse = mean_squared_error(y_val, val_predictions)
    val_rmse = val_mse ** 0.5
    val_r2 = r2_score(y_val, val_predictions)

    all_results.append({
        "Parameters": params,
        "Validation MAE": val_mae,
        "Validation MSE": val_mse,
        "Validation RMSE": val_rmse,
        "Validation R2": val_r2,
    })

    print("Parameters:", params)
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
        }


# --------------------------------------------------
# Display results
# --------------------------------------------------

summary = pd.DataFrame(all_results)
summary = summary.sort_values("Validation RMSE")

summary["Parameters"] = summary["Parameters"].astype(str)

print("\n========== Summary For This Part ==========")
print(summary.to_string(index=False))

print("\n========== Best XGBoost Model For This Part ==========")
print("Best Parameters:", best_params)
print(f"Best Validation MAE  : {best_val_metrics['MAE']:.2f}")
print(f"Best Validation MSE  : {best_val_metrics['MSE']:.2f}")
print(f"Best Validation RMSE : {best_val_metrics['RMSE']:.2f}")
print(f"Best Validation R²   : {best_val_metrics['R2']:.4f}")

print(
    "\nTest-set evaluation is intentionally excluded from tuning. "
    "The final selected configuration is evaluated in "
    "train_xgboost.py."
)
