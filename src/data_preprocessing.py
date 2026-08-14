import pandas as pd

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


# Dataset source:
# https://github.com/teo-tsou/app_aware_5g/tree/master/dataset

DATA_PATH = Path("data")
FILE_NAME = "ue-lte-network-traffic-stats.csv"

file_path = DATA_PATH / FILE_NAME

df = pd.read_csv(file_path)

print("Dataset loaded successfully")
print("File:", file_path)
print("Shape:", df.shape)


# --------------------------------------------------
# Create UE1 total throughput
# --------------------------------------------------

df["UE1_total_throughput"] = (
    df["UE1: web-rtc"]
    + df["UE1: sipp"]
    + df["UE1: web-server"]
)

TARGET_COLUMN = "UE1_total_throughput"

ue1_df = df[
    [
        TARGET_COLUMN,
        "UE1-Jitter",
        "UE1-CQI",
    ]
].copy()


# --------------------------------------------------
# Create lag features
# --------------------------------------------------

ue1_df["UE1_throughput_lag1"] = ue1_df[TARGET_COLUMN].shift(1)
ue1_df["UE1_throughput_lag2"] = ue1_df[TARGET_COLUMN].shift(2)
ue1_df["UE1_throughput_lag3"] = ue1_df[TARGET_COLUMN].shift(3)

ue1_df = ue1_df.dropna()


# --------------------------------------------------
# Define features and target
# --------------------------------------------------

features = ue1_df[
    [
        "UE1_throughput_lag1",
        "UE1_throughput_lag2",
        "UE1_throughput_lag3",
        "UE1-Jitter",
        "UE1-CQI",
    ]
]

target = ue1_df[TARGET_COLUMN]


# --------------------------------------------------
# Chronological 70/15/15 split
# --------------------------------------------------

n = len(ue1_df)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = features.iloc[:train_end].copy()
X_val = features.iloc[train_end:val_end].copy()
X_test = features.iloc[val_end:].copy()

y_train = target.iloc[:train_end].copy()
y_val = target.iloc[train_end:val_end].copy()
y_test = target.iloc[val_end:].copy()

print("\nUE1 Dataset Shape:", ue1_df.shape)
print("Training features:", X_train.shape)
print("Validation features:", X_val.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Validation target:", y_val.shape)
print("Testing target:", y_test.shape)


# --------------------------------------------------
# Normalize input features
# --------------------------------------------------

scaler = MinMaxScaler()

X_train = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index,
)

X_val = pd.DataFrame(
    scaler.transform(X_val),
    columns=X_val.columns,
    index=X_val.index,
)

X_test = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index,
)

print("\nFeatures normalized using MinMaxScaler.")


# --------------------------------------------------
# Save processed datasets
# --------------------------------------------------

X_train.to_csv(DATA_PATH / "X_train_ue1.csv", index=False)
X_val.to_csv(DATA_PATH / "X_val_ue1.csv", index=False)
X_test.to_csv(DATA_PATH / "X_test_ue1.csv", index=False)

y_train.to_csv(DATA_PATH / "y_train_ue1.csv", index=False)
y_val.to_csv(DATA_PATH / "y_val_ue1.csv", index=False)
y_test.to_csv(DATA_PATH / "y_test_ue1.csv", index=False)

print("\nProcessed dataset saved successfully.")
print("Target: Total UE1 Throughput")
print("Total throughput = WebRTC + SIPp + Web Server")
print("Lag features: lag1, lag2, lag3")
print("Normalization: MinMaxScaler")
print("Train/Validation/Test Split: 70/15/15")
