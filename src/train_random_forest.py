import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# -----------------------------
# Load training/testing data
# -----------------------------

X_train = pd.read_csv("data/X_train_ue1.csv")
X_test = pd.read_csv("data/X_test_ue1.csv")

y_train = pd.read_csv("data/y_train_ue1.csv").squeeze()
y_test = pd.read_csv("data/y_test_ue1.csv").squeeze()

# -----------------------------
# Use only CQI and Jitter
# -----------------------------

X_train = X_train[
    [
        "UE1_throughput_lag1",
        "UE1_throughput_lag2",
        "UE1_throughput_lag3",
        "UE1-Jitter",
        "UE1-CQI",
    ]
]

X_test = X_test[
    [
        "UE1_throughput_lag1",
        "UE1_throughput_lag2",
        "UE1_throughput_lag3",
        "UE1-Jitter",
        "UE1-CQI",
    ]
]

# -----------------------------
# Create Random Forest
# -----------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# -----------------------------
# Train
# -----------------------------

model.fit(X_train, y_train)

# -----------------------------
# Predict
# -----------------------------

predictions = model.predict(X_test)

# -----------------------------
# Evaluate
# -----------------------------

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("\n========== Random Forest Results ==========")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# -----------------------------
# Show first 10 predictions
# -----------------------------

results = pd.DataFrame({
    "Actual Throughput": y_test,
    "Predicted Throughput": predictions
})

print("\nFirst 10 Predictions")
print(results.head(10))


# -----------------------------
# Feature Importance
# -----------------------------

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nFeature Importance")
print(importance)

# -----------------------------
# Plot Feature Importance
# -----------------------------

plt.figure(figsize=(8,5))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Importance")

plt.xticks(rotation=25)

plt.tight_layout()

plt.savefig("plots/random_forest_feature_importance.png")

plt.show()

print("\nFeature importance plot saved to:")
print("plots/random_forest_feature_importance.png")


# -----------------------------
# Actual vs Predicted Plot
# -----------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.3
)

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

plt.savefig("plots/random_forest_actual_vs_predicted.png", dpi=300)
plt.show()

print("\nActual vs predicted plot saved to:")
print("plots/random_forest_actual_vs_predicted.png")