import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt


# Load training, validation, and testing data
X_train = pd.read_csv("data/X_train_ue1.csv")
X_val = pd.read_csv("data/X_val_ue1.csv")
X_test = pd.read_csv("data/X_test_ue1.csv")

y_train = pd.read_csv("data/y_train_ue1.csv").squeeze()
y_val = pd.read_csv("data/y_val_ue1.csv").squeeze()
y_test = pd.read_csv("data/y_test_ue1.csv").squeeze()


# Select input features
selected_features = [
    "UE1_throughput_lag1",
    "UE1_throughput_lag2",
    "UE1_throughput_lag3",
    "UE1-Jitter",
    "UE1-CQI",
]

X_train = X_train[selected_features]
X_val = X_val[selected_features]
X_test = X_test[selected_features]


# Best Random Forest model found during hyperparameter tuning
best_params = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_leaf": 2,
}


# Create and train Random Forest model
model = RandomForestRegressor(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    min_samples_leaf=best_params["min_samples_leaf"],
    random_state=42,
    n_jobs=1,
)

model.fit(X_train, y_train)


# Evaluate on validation set
val_predictions = model.predict(X_val)

val_mae = mean_absolute_error(y_val, val_predictions)
val_mse = mean_squared_error(y_val, val_predictions)
val_rmse = val_mse ** 0.5
val_r2 = r2_score(y_val, val_predictions)

print("\n========== Random Forest Validation Results ==========")
print("Parameters:", best_params)
print(f"Validation MAE  : {val_mae:.2f}")
print(f"Validation MSE  : {val_mse:.2f}")
print(f"Validation RMSE : {val_rmse:.2f}")
print(f"Validation R²   : {val_r2:.4f}")


# Final evaluation on test set
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


# Show first 10 test predictions
results = pd.DataFrame({
    "Actual Total UE1 Throughput": y_test,
    "Predicted Total UE1 Throughput": test_predictions
})

print("\nFirst 10 Test Predictions")
print(results.head(10))


# Feature importance
importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nFeature Importance")
print(importance)


# -------------------------------------------------------------------
# The hyperparameter tuning code used to find the best model has been
# removed. The selected model was:
#
# n_estimators = 200
# max_depth = 10
# min_samples_leaf = 2
#
# This configuration achieved the lowest validation RMSE during tuning
# and is used for all final evaluations reported in the paper.
# -------------------------------------------------------------------


# Paper-style font
plt.rcParams["font.family"] = "DejaVu Serif"

importance_plot = importance.copy()

importance_plot["Feature"] = importance_plot["Feature"].replace({
    "UE1_throughput_lag1": "Lag 1",
    "UE1_throughput_lag2": "Lag 2",
    "UE1_throughput_lag3": "Lag 3",
    "UE1-CQI": "CQI",
    "UE1-Jitter": "Jitter",
})

# Sort so the most important feature appears at the top
importance_plot = importance_plot.sort_values(
    "Importance",
    ascending=True
)

fig, ax = plt.subplots(figsize=(6.8, 3.8))

ax.barh(
    importance_plot["Feature"],
    importance_plot["Importance"],
    height=0.55
)

ax.set_xlabel("Feature Importance", fontsize=11)
ax.set_ylabel("Feature", fontsize=11)

ax.tick_params(axis="both", labelsize=10)

ax.grid(
    axis="x",
    linestyle="--",
    linewidth=0.5,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "plots/random_forest_feature_importance.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()