import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Change this to 1 or 2
RUN_PART = 2


X_train = pd.read_csv("data/X_train_ue1.csv")
X_val = pd.read_csv("data/X_val_ue1.csv")
X_test = pd.read_csv("data/X_test_ue1.csv")

y_train = pd.read_csv("data/y_train_ue1.csv").squeeze()
y_val = pd.read_csv("data/y_val_ue1.csv").squeeze()
y_test = pd.read_csv("data/y_test_ue1.csv").squeeze()


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


if RUN_PART == 1:
    param_grid = [
        {"n_estimators": 10, "max_depth": None},
        {"n_estimators": 25, "max_depth": None},
        {"n_estimators": 50, "max_depth": None},
        {"n_estimators": 100, "max_depth": None},
        {"n_estimators": 200, "max_depth": None},
    ]
else:
    param_grid = [
        {"n_estimators": 50, "max_depth": 10},
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 50, "max_depth": 20},
        {"n_estimators": 100, "max_depth": 20},
        {"n_estimators": 200, "max_depth": 20},
    ]


best_model = None
best_params = None
best_val_rmse = float("inf")

print(f"\n========== Random Forest Hyperparameter Tuning: Part {RUN_PART} ==========")

for i, params in enumerate(param_grid, start=1):
    print(f"\nCombination {i}/5")
    
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=42,
        n_jobs=1
    )

    model.fit(X_train, y_train)
    val_predictions = model.predict(X_val)

    val_mae = mean_absolute_error(y_val, val_predictions)
    val_mse = mean_squared_error(y_val, val_predictions)
    val_rmse = val_mse ** 0.5
    val_r2 = r2_score(y_val, val_predictions)

    print("Parameters:", params)
    print(f"Validation MAE  : {val_mae:.2f}")
    print(f"Validation MSE  : {val_mse:.2f}")
    print(f"Validation RMSE : {val_rmse:.2f}")
    print(f"Validation R²   : {val_r2:.4f}")

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_model = model
        best_params = params


print("\n========== Best Random Forest Model For This Part ==========")
print("Best Parameters:", best_params)
print(f"Best Validation RMSE: {best_val_rmse:.2f}")


test_predictions = best_model.predict(X_test)

test_mae = mean_absolute_error(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)
test_rmse = test_mse ** 0.5
test_r2 = r2_score(y_test, test_predictions)

print("\n========== Random Forest Test Results For Best Model In This Part ==========")
print(f"Test MAE  : {test_mae:.2f}")
print(f"Test MSE  : {test_mse:.2f}")
print(f"Test RMSE : {test_rmse:.2f}")
print(f"Test R²   : {test_r2:.4f}")


results = pd.DataFrame({
    "Actual Throughput": y_test,
    "Predicted Throughput": test_predictions
})

print("\nFirst 10 Test Predictions")
print(results.head(10))


importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": best_model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nFeature Importance")
print(importance)


plt.figure(figsize=(8, 5))
plt.bar(importance["Feature"], importance["Importance"])
plt.title("Random Forest Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(f"plots/random_forest_feature_importance_part{RUN_PART}.png")

print("\nFeature importance plot saved to:")
print(f"plots/random_forest_feature_importance_part{RUN_PART}.png")


plt.figure(figsize=(8, 6))
plt.scatter(y_test, test_predictions, alpha=0.3)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)
plt.title("Random Forest: Actual vs Predicted Throughput")
plt.xlabel("Actual Throughput (Bytes)")
plt.ylabel("Predicted Throughput (Bytes)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"plots/random_forest_actual_vs_predicted_part{RUN_PART}.png", dpi=300)

print("\nActual vs predicted plot saved to:")
print(f"plots/random_forest_actual_vs_predicted_part{RUN_PART}.png")