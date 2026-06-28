import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
# Think about normalizing the variables and redoing it 
# -----------------------------
# Load training and testing data

X_train = pd.read_csv("data/X_train_ue1.csv")
X_test = pd.read_csv("data/X_test_ue1.csv")

y_train = pd.read_csv("data/y_train_ue1.csv")
y_test = pd.read_csv("data/y_test_ue1.csv")

# ---------------------------------------
# Use only Jitter and CQI as input features

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

# Convert DataFrame to Series
y_train = y_train.squeeze()
y_test = y_test.squeeze()

# -----------------------------
# Create Linear Regression model

model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# -----------------------------
# Evaluate model performance

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("\n========== Linear Regression Results ==========")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# -----------------------------
# Show first 10 predictions

results = pd.DataFrame({
    "Actual Throughput": y_test,
    "Predicted Throughput": predictions
})

print("\nFirst 10 Predictions")
print(results.head(10))