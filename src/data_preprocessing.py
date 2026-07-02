import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# Testing the new dataset from GitHub:
# https://github.com/teo-tsou/app_aware_5g/tree/master/dataset

DATA_PATH = Path("data")
FILE_NAME = "ue-lte-network-traffic-stats.csv"

file_path = DATA_PATH / FILE_NAME
df = pd.read_csv(file_path)

print("Dataset loaded successfully")
print("File:", file_path)
print("Shape:", df.shape)

# """
# UE1 throughput prediction setup
# Creates lag features, performs train/test split,
# normalizes the input features, and saves the data.
# """

TARGET_COLUMN = "UE1: web-rtc"

ue1_df = df[["UE1: web-rtc", "UE1-Jitter", "UE1-CQI"]].copy()

# Creates the lag features
ue1_df["UE1_throughput_lag1"] = ue1_df[TARGET_COLUMN].shift(1)
ue1_df["UE1_throughput_lag2"] = ue1_df[TARGET_COLUMN].shift(2)
ue1_df["UE1_throughput_lag3"] = ue1_df[TARGET_COLUMN].shift(3)

# Remove rows that contain NaN from the lag features
ue1_df = ue1_df.dropna()

# features inputs
features = ue1_df[
    [
        "UE1_throughput_lag1",
        "UE1_throughput_lag2",
        "UE1_throughput_lag3",
        "UE1-Jitter",
        "UE1-CQI",
    ]
]

# Target outputs
target = ue1_df[TARGET_COLUMN]

# chronological 80/20 split
split_index = int(len(ue1_df) * 0.8)

X_train = features.iloc[:split_index].copy()
X_test = features.iloc[split_index:].copy()

y_train = target.iloc[:split_index]
y_test = target.iloc[split_index:]

print("\nUE1 Dataset Shape:", ue1_df.shape)
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

# """
# Normalize FEATURES ONLY
# """

scaler = MinMaxScaler()

X_train = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index,
)

X_test = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index,
)

print("\nFeatures normalized using MinMaxScaler.")

# """
# Save processed data
# """

X_train.to_csv("data/X_train_ue1.csv", index=False)
X_test.to_csv("data/X_test_ue1.csv", index=False)

y_train.to_csv("data/y_train_ue1.csv", index=False)
y_test.to_csv("data/y_test_ue1.csv", index=False)

print("\nProcessed dataset saved successfully.")
print("Lag features: lag1, lag2, lag3")
print("Normalization: MinMaxScaler")
print("Train/Test Split: 80/20")