from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from sklearn.ensemble import RandomForestRegressor
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
    "max_depth": 10,
    "min_samples_leaf": 2,
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
# Train final Random Forest model
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=BEST_PARAMS["n_estimators"],
    max_depth=BEST_PARAMS["max_depth"],
    min_samples_leaf=BEST_PARAMS["min_samples_leaf"],
    random_state=42,
    n_jobs=1,
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

print("\n========== Random Forest Validation Results ==========")
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

print("\n========== Random Forest Test Results ==========")
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


# --------------------------------------------------
# Feature importance
# --------------------------------------------------

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_,
}).sort_values("Importance", ascending=False)

print("\nFeature Importance")
print(importance)


# Hyperparameter tuning is performed separately in
# train_random_forest_tuning.py.
#
# The final configuration was selected because it achieved
# the lowest validation MAE during tuning.


# --------------------------------------------------
# Plot actual vs. predicted throughput
# --------------------------------------------------

plt.rcParams["font.family"] = "DejaVu Serif"

num_samples = 200

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
    PLOTS_PATH / "random_forest_actual_vs_predicted.png",
    dpi=600,
    bbox_inches="tight",
)

plt.close()


# --------------------------------------------------
# Plot Random Forest feature importance
# --------------------------------------------------

importance_plot = importance.copy()

importance_plot["Feature"] = importance_plot["Feature"].replace({
    "UE1_throughput_lag1": "Lag 1",
    "UE1_throughput_lag2": "Lag 2",
    "UE1_throughput_lag3": "Lag 3",
    "UE1-CQI": "CQI",
    "UE1-Jitter": "Jitter",
})

importance_plot = importance_plot.sort_values(
    "Importance",
    ascending=True,
)

fig, ax = plt.subplots(figsize=(6.8, 3.8))

ax.barh(
    importance_plot["Feature"],
    importance_plot["Importance"],
    height=0.55,
)

ax.set_xlabel("Feature Importance", fontsize=11)
ax.set_ylabel("Feature", fontsize=11)

ax.tick_params(axis="both", labelsize=10)

ax.grid(
    axis="x",
    linestyle="--",
    linewidth=0.5,
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    PLOTS_PATH / "random_forest_feature_importance.png",
    dpi=600,
    bbox_inches="tight",
)

plt.close()
