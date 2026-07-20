import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# Load training, validation, and testing data
X_train = pd.read_csv("data/X_train_ue1.csv")
X_val = pd.read_csv("data/X_val_ue1.csv")
X_test = pd.read_csv("data/X_test_ue1.csv")

y_train = pd.read_csv("data/y_train_ue1.csv").squeeze()
y_val = pd.read_csv("data/y_val_ue1.csv").squeeze()
y_test = pd.read_csv("data/y_test_ue1.csv").squeeze()


# Best XGBoost model found during hyperparameter tuning
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=1.0,
    objective="reg:squarederror",
    random_state=42
)


# Train model
model.fit(X_train, y_train)


# Evaluate on validation set
val_predictions = model.predict(X_val)

val_mae = mean_absolute_error(y_val, val_predictions)
val_mse = mean_squared_error(y_val, val_predictions)
val_rmse = val_mse ** 0.5
val_r2 = r2_score(y_val, val_predictions)

print("\n========== XGBoost Validation Results ==========")
print("Parameters:")
print("n_estimators      : 200")
print("learning_rate     : 0.05")
print("max_depth         : 6")
print("subsample         : 0.8")
print("colsample_bytree  : 1.0")

print(f"\nValidation MAE  : {val_mae:.2f}")
print(f"Validation MSE  : {val_mse:.2f}")
print(f"Validation RMSE : {val_rmse:.2f}")
print(f"Validation R²   : {val_r2:.4f}")


# Final evaluation on test set
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


# Feature importance
importance = model.feature_importances_

plt.figure(figsize=(8, 5))
plt.bar(X_train.columns, importance)

plt.title("XGBoost Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=30)

plt.tight_layout()

plt.savefig(
    "plots/xgboost_feature_importance.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()


# Actual vs predicted throughput
plt.figure(figsize=(12, 6))

plt.plot(
    y_test.values[:300],
    label="Actual Total UE1 Throughput"
)

plt.plot(
    test_predictions[:300],
    label="Predicted Total UE1 Throughput"
)

plt.title("Actual vs Predicted Total UE1 Throughput")
plt.xlabel("Sample Index")
plt.ylabel("Total UE1 Throughput")

plt.legend()
plt.tight_layout()

plt.savefig(
    "plots/xgboost_actual_vs_predicted.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()