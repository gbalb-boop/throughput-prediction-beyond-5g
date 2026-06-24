import pandas as pd #pd short notation for tables
from pathlib import Path #for library or folders

DATA_PATH = Path("data") #convention

csv_files = list(DATA_PATH.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV file found in the data folder.")

df = pd.read_csv(csv_files[0])

print("Dataset loaded successfully")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nSummary statistics:")
print(df.describe())