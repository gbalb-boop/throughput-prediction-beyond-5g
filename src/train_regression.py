import pandas as pd

#for the graph
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


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


#Create and train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)


#Evaluate on validation set
val_predictions = model.predict(X_val)

val_mae = mean_absolute_error(y_val, val_predictions)
val_mse = mean_squared_error(y_val, val_predictions)
val_rmse = val_mse ** 0.5
val_r2 = r2_score(y_val, val_predictions)

print("\n========== Linear Regression Validation Results ==========")
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

print("\n========== Linear Regression Test Results ==========")
print(f"Test MAE  : {test_mae:.2f}")
print(f"Test MSE  : {test_mse:.2f}")
print(f"Test RMSE : {test_rmse:.2f}")
print(f"Test R²   : {test_r2:.4f}")


# Show first 10 test predictions
results = pd.DataFrame({
    "Actual Throughput": y_test,
    "Predicted Throughput": test_predictions
})

print("\nFirst 10 Test Predictions")
print(results.head(10))

# note to self even though its testing on the validation
# set its not realy doing anything because it doesnt
# have any real parameters to fine tune it  


# Paper-style font
plt.rcParams["font.family"] = "DejaVu Serif"

# Number of test samples to display
num_samples = 200

# Create figure
fig, ax = plt.subplots(figsize=(6.8, 3.8))

# Actual throughput
ax.plot(
    results.index[:num_samples],
    results["Actual Throughput"].iloc[:num_samples],
    label="Actual Throughput",
    linewidth=1.2
)

# Predicted throughput
ax.plot(
    results.index[:num_samples],
    results["Predicted Throughput"].iloc[:num_samples],
    label="Predicted Throughput",
    linewidth=0.9,
    linestyle="--"
)

# Axis labels
ax.set_xlabel("Sample Index", fontsize=11)
ax.set_ylabel("Throughput (Bytes/s)", fontsize=11)

# Tick formatting
ax.tick_params(axis="both", labelsize=10)
ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

# Grid
ax.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    alpha=0.25
)

# Legend
ax.legend(
    loc="upper center",
    fontsize=9,
    frameon=True
)

# Remove unnecessary whitespace
plt.tight_layout()

# Save figure
plt.savefig(
    "plots/linear_regression_actual_vs_predicted.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()