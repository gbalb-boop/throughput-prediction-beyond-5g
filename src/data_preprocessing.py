import pandas as pd
from pathlib import Path

DATA_PATH = Path("data")
FILE_NAME = "6G_network_slicing_qos_dataset_2345.csv"
TARGET_COLUMN = "QoS Metric (Throughput)"

file_path = DATA_PATH / FILE_NAME

if not file_path.exists():
    raise FileNotFoundError(f"Could not find file: {file_path}")

df = pd.read_csv(file_path)

print("Dataset loaded successfully")
print("File:", file_path)
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column not found: {TARGET_COLUMN}")

print("\nTarget column found:")
print(TARGET_COLUMN)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset info:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nSummary statistics:")
print(df.describe())

print("\nTarget summary:")
print(df[TARGET_COLUMN].describe())

print("\nCorrelation with QoS Metric (Throughput):")
correlations = df.corr(numeric_only=True)[TARGET_COLUMN].sort_values(ascending=False)
print(correlations)

#newish
print("\nUnique values:")

for col in df.columns:
    print(f"\n{col}")
    print(df[col].nunique())

print("\nCorrelation Matrix")

corr = df.corr(numeric_only=True)

print(
    corr["QoS Metric (Throughput)"]
    .sort_values(key=abs, ascending=False)
)