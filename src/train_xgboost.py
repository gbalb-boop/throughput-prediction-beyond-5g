from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


DATA_PATH = Path("data")
PLOTS_PATH = Path("plots")

SELECTED_FEATURES = [
    "UE1_throughput_lag1",
    "UE1_throughput_lag2",
    "UE1_throughput_lag3",
    "UE1-Jitter",
    "UE1-CQI",
]

BEST_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
}


# --------------------------------------------------
# Load processed datasets
# --------------------------------------------------

X_train = pd.read_csv(DATA_PATH / "X_train_ue1.csv")
X_val = pd.read_csv(DATA_PATH / "X_val_ue1.csv")
X_test = pd.read_csv(DATA_PATH / "X_test_ue1.csv")

y_train = pd.read_csv(DATA_PATH / "y_train_ue1.csv").squeeze()
y_val = pd.read_csv(DATA_PATH / "y_val_ue1.csv").squeeze()
y_test = pd.read_csv(DATA_PATH / "y_test_ue1.csv").squeeze()

X_train = X_train[SELECTED_FEATURES]
X_val = X_val[SELECTED_FEATURES]
X_test = X_test[SELECTED_FEATURES]


# --------------------------------------------------
# Train final XGBoost model
# --------------------------------------------------

model = XGBRegressor(
    n_estimators=BEST_PARAMS["n_estimators"],
    learning_rate=BEST_PARAMS["learning_rate"],
    max_depth=BEST_PARAMS["max_depth"],
    subsample=BEST_PARAMS["subsample"],
    colsample_bytree=BEST_PARAMS["colsample_bytree"],
    objective="reg:squarederror",
    random_state=42,
)

model.fit(X_train, y_train)


# --------------------------------------------------
# Validation evaluation
# --------------------------------------------------

val_predictions = model.predict(X_val)

val_mae = mean_absolute_error(y_val, val_predictions)
val_mse = mean_squared_error(y_val, val_predictions)
val_rmse = val_mse ** 0.5
val_r2 = r2_score(y_val, val_predictions)

print("\n========== XGBoost Validation Results ==========")
print("Parameters:", BEST_PARAMS)
print(f"Validation MAE  : {val_mae:.2f}")
print(f"Validation MSE  : {val_mse:.2f}")
print(f"Validation RMSE : {val_rmse:.2f}")
print(f"Validation R²   : {val_r2:.4f}")


# --------------------------------------------------
# Test evaluation
# --------------------------------------------------

test_predictions = model.predict(X_test)

test_mae = mean_absolute_error(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)
test_rmse = test_mse ** 0.5
test_r2 = r2_score(y_test, test_predictions)

print("\n========== XGBoost Test Results ==========")
print(f"Test MAE  : {test_mae:.2f}")
print(f"Test MSE  : {test_mse:.2f}")
print(f"Test RMSE : {test_rmse:.2f}")
print(f"Test R²   : {test_r2:.4f}")


# --------------------------------------------------
# Sample predictions
# --------------------------------------------------

results = pd.DataFrame({
    "Actual Total UE1 Throughput": y_test,
    "Predicted Total UE1 Throughput": test_predictions,
})

print("\nFirst 10 Test Predictions")
print(results.head(10))


# Hyperparameter tuning is performed separately in
# train_xgboost_tuning.py.


# --------------------------------------------------
# Plot actual vs. predicted throughput
# --------------------------------------------------

plt.rcParams["font.family"] = "DejaVu Serif"

num_samples = 300

fig, ax = plt.subplots(figsize=(6.8, 3.8))

ax.plot(
    results.index[:num_samples],
    results["Actual Total UE1 Throughput"].iloc[:num_samples],
    label="Actual Throughput",
    linewidth=1.2,
)

ax.plot(
    results.index[:num_samples],
    results["Predicted Total UE1 Throughput"].iloc[:num_samples],
    label="Predicted Throughput",
    linewidth=0.9,
    linestyle="--",
)

ax.set_xlabel("Sample Index", fontsize=11)
ax.set_ylabel("Total UE1 Throughput (Bytes/s)", fontsize=11)

ax.tick_params(axis="both", labelsize=10)
ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

ax.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    alpha=0.25,
)

ax.legend(
    loc="upper center",
    fontsize=9,
    frameon=True,
)

plt.tight_layout()

plt.savefig(
    PLOTS_PATH / "xgboost_actual_vs_predicted.png",
    dpi=600,
    bbox_inches="tight",
)

plt.close()
