import pandas as pd
from pathlib import Path

# Testing the new dataset from GitHub:
# https://github.com/teo-tsou/app_aware_5g/tree/master/dataset
DATA_PATH = Path("data")
FILE_NAME = "ue-lte-network-traffic-stats.csv"

file_path = DATA_PATH / FILE_NAME
df = pd.read_csv(file_path)

print("Dataset loaded successfully")
print("File:", file_path)
print("Shape:", df.shape)

# print("\nColumns:")
# print(df.columns.tolist())

# print("\nFirst 5 rows:")
# print(df.head())

# print("\nDataset info:")
# df.info()

# print("\nMissing values:")
# print(df.isnull().sum())

# print("\nSummary statistics:")
# print(df.describe())

# print("\nColumns with only one unique value:")
# for col in df.columns:
#     if df[col].nunique() == 1:
#         print(col)

# print("\nCorrelation matrix:")
# corr = df.corr(numeric_only=True)
# print(corr)

# for col in df.columns:
#     print(f"{col}: {df[col].nunique()} unique values")

# print(df.iloc[:20])


# UE1 throughput prediction setup
# Creates 4 smaller csv for input/output and testing/training

TARGET_COLUMN = "UE1: web-rtc"

ue1_df = df[["UE1: web-rtc", "UE1-Jitter", "UE1-CQI"]].copy()

features = ue1_df[["UE1: web-rtc", "UE1-Jitter", "UE1-CQI"]]
target = ue1_df[TARGET_COLUMN]

split_index = int(len(ue1_df) * 0.8)

X_train = features.iloc[:split_index]
X_test = features.iloc[split_index:]

y_train = target.iloc[:split_index]
y_test = target.iloc[split_index:]

print("\nUE1 Dataset Shape:", ue1_df.shape)
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

# Save split data
X_train.to_csv("data/X_train_ue1.csv", index=False)
X_test.to_csv("data/X_test_ue1.csv", index=False)
y_train.to_csv("data/y_train_ue1.csv", index=False)
y_test.to_csv("data/y_test_ue1.csv", index=False)

print("\n80/20 split saved successfully.")
