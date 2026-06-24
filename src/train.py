import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = Path("data")
FILE_NAME = "6G_network_slicing_qos_dataset_2345.csv"
TARGET_COLUMN = "QoS Metric (Throughput)"

df = pd.read_csv(DATA_PATH / FILE_NAME)

X = df.drop(columns=[TARGET_COLUMN, "Timestamp"])
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

baseline_pred = [y_train.mean()] * len(y_test)

baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_mse = mean_squared_error(y_test, baseline_pred)
baseline_r2 = r2_score(y_test, baseline_pred)

print("\nBaseline Results")
print("Baseline MAE:", baseline_mae)
print("Baseline MSE:", baseline_mse)
print("Baseline R2 Score:", baseline_r2)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Random Forest Results")
print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(feature_importance)