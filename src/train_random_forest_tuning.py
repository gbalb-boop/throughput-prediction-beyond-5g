import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Change this to 1, 2, 3, 4, 5, or 6
RUN_PART = 6


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

elif RUN_PART == 2:
    param_grid = [
        {"n_estimators": 50, "max_depth": 10},
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 50, "max_depth": 20},
        {"n_estimators": 100, "max_depth": 20},
        {"n_estimators": 200, "max_depth": 20},
    ]

elif RUN_PART == 3:
    param_grid = [
        {"n_estimators": 75, "max_depth": 8},
        {"n_estimators": 100, "max_depth": 8},
        {"n_estimators": 150, "max_depth": 8},
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 150, "max_depth": 10},
    ]

elif RUN_PART == 4:
    param_grid = [
        {"n_estimators": 200, "max_depth": 10},
        {"n_estimators": 100, "max_depth": 12},
        {"n_estimators": 150, "max_depth": 12},
        {"n_estimators": 200, "max_depth": 12},
        {"n_estimators": 150, "max_depth": 14},
    ]

elif RUN_PART == 5:
    param_grid = [
        {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 1},
        {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 2},
        {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 4},
        {"n_estimators": 150, "max_depth": 10, "min_samples_leaf": 2},
        {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 2},
    ]

elif RUN_PART == 6:
    param_grid = [
        {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 2},
        {"n_estimators": 150, "max_depth": 8, "min_samples_leaf": 2},
        {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 2},
        {"n_estimators": 150, "max_depth": 12, "min_samples_leaf": 2},
        {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 2},
    ]

else:
    raise ValueError("RUN_PART must be 1, 2, 3, 4, 5, or 6.")


best_model = None
best_params = None
best_val_rmse = float("inf")
best_val_metrics = None

all_results = []

print(f"\n========== Random Forest Hyperparameter Tuning: Part {RUN_PART} ==========")

for i, params in enumerate(param_grid, start=1):
    print(f"\nCombination {i}/5")

    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params.get("min_samples_leaf", 1),
        random_state=42,
        n_jobs=1
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
        "Validation R2": val_r2
    })

    print("Parameters:", params)
    print(f"Validation MAE  : {val_mae:.2f}")
    print(f"Validation MSE  : {val_mse:.2f}")
    print(f"Validation RMSE : {val_rmse:.2f}")
    print(f"Validation R²   : {val_r2:.4f}")

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_model = model
        best_params = params
        best_val_metrics = {
            "MAE": val_mae,
            "MSE": val_mse,
            "RMSE": val_rmse,
            "R2": val_r2
        }


print("\n========== Summary For This Part ==========")

summary = pd.DataFrame(all_results)
summary = summary.sort_values("Validation RMSE")

print(summary.to_string(index=False))


print("\n========== Best Random Forest Model For This Part ==========")
print("Best Parameters:", best_params)
print(f"Best Validation MAE  : {best_val_metrics['MAE']:.2f}")
print(f"Best Validation MSE  : {best_val_metrics['MSE']:.2f}")
print(f"Best Validation RMSE : {best_val_metrics['RMSE']:.2f}")
print(f"Best Validation R²   : {best_val_metrics['R2']:.4f}")


test_predictions = best_model.predict(X_test)

test_mae = mean_absolute_error(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)
test_rmse = test_mse ** 0.5
test_r2 = r2_score(y_test, test_predictions)

print("\n========== Test Results For Best Model In This Part ==========")
print(f"Test MAE  : {test_mae:.2f}")
print(f"Test MSE  : {test_mse:.2f}")
print(f"Test RMSE : {test_rmse:.2f}")
print(f"Test R²   : {test_r2:.4f}")