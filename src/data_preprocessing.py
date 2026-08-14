import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# Dataset from GitHub:
# https://github.com/teo-tsou/app_aware_5g/tree/master/dataset

DATA_PATH = Path("data")
FILE_NAME = "ue-lte-network-traffic-stats.csv"

file_path = DATA_PATH / FILE_NAME
df = pd.read_csv(file_path)

print("Dataset loaded successfully")
print("File:", file_path)
print("Shape:", df.shape)

# """
# UE1 total throughput prediction setup
# Creates total UE1 throughput, lag features,
# performs chronological train/validation/test split,
# normalizes the input features, and saves the data.
# """

# Create total UE1 throughput by summing throughput
# from all three applications for each row
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

# Create lag features from total UE1 throughput
ue1_df["UE1_throughput_lag1"] = ue1_df[TARGET_COLUMN].shift(1)
ue1_df["UE1_throughput_lag2"] = ue1_df[TARGET_COLUMN].shift(2)
ue1_df["UE1_throughput_lag3"] = ue1_df[TARGET_COLUMN].shift(3)

# Remove rows that contain NaN from the lag features
ue1_df = ue1_df.dropna()

# Feature inputs
features = ue1_df[
    [
        "UE1_throughput_lag1",
        "UE1_throughput_lag2",
        "UE1_throughput_lag3",
        "UE1-Jitter",
        "UE1-CQI",
    ]
]

# Target output
target = ue1_df[TARGET_COLUMN]

# Chronological 70/15/15 split
n = len(ue1_df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = features.iloc[:train_end].copy()
X_val = features.iloc[train_end:val_end].copy()
X_test = features.iloc[val_end:].copy()

y_train = target.iloc[:train_end]
y_val = target.iloc[train_end:val_end]
y_test = target.iloc[val_end:]

print("\nUE1 Dataset Shape:", ue1_df.shape)
print("Training features:", X_train.shape)
print("Validation features:", X_val.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Validation target:", y_val.shape)
print("Testing target:", y_test.shape)

# """
# Normalize FEATURES ONLY
# Fit the scaler only on the training data
# """

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

# """
# Save processed data
# """

X_train.to_csv("data/X_train_ue1.csv", index=False)
X_val.to_csv("data/X_val_ue1.csv", index=False)
X_test.to_csv("data/X_test_ue1.csv", index=False)

y_train.to_csv("data/y_train_ue1.csv", index=False)
y_val.to_csv("data/y_val_ue1.csv", index=False)
y_test.to_csv("data/y_test_ue1.csv", index=False)

print("\nProcessed dataset saved successfully.")
print("Target: Total UE1 Throughput")
print("Total throughput = WebRTC + SIPp + Web Server")
print("Lag features: lag1, lag2, lag3 of total UE1 throughput")
print("Normalization: MinMaxScaler")
print("Train/Validation/Test Split: 70/15/15")

# """
# Figure 1: Total UE1 Throughput Over Sample Index
# """

# Create the plots directory if it does not already exist
PLOTS_PATH = Path("plots")
PLOTS_PATH.mkdir(parents=True, exist_ok=True)

# Number of samples to display
NUM_SAMPLES = 1000

# Select the first 1,000 total-throughput values
throughput_plot = df[TARGET_COLUMN].iloc[:NUM_SAMPLES].reset_index(drop=True)

# Create a clean sample index from 0 to 999
sample_index = range(len(throughput_plot))

# Use a readable serif font throughout the figure
plt.rcParams["font.family"] = "DejaVu Serif"

fig, ax = plt.subplots(figsize=(7.2, 4.0))

ax.plot(
    sample_index,
    throughput_plot,
    linewidth=0.9,
)

# Research-paper figures normally use the caption instead of a title
ax.set_xlabel(
    "Sample Index",
    fontsize=11,
)

ax.set_ylabel(
    "Total UE1 Throughput (Bytes/s)",
    fontsize=11,
)

# Set readable tick-label sizes
ax.tick_params(
    axis="both",
    labelsize=10,
)

# Format throughput values with commas
ax.yaxis.set_major_formatter(
    StrMethodFormatter("{x:,.0f}")
)

# Keep the horizontal axis aligned with the displayed samples
ax.set_xlim(
    0,
    len(throughput_plot) - 1,
)

# Add a light grid without distracting from the data
ax.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    alpha=0.4,
)

plt.tight_layout()

figure_path = PLOTS_PATH / "figure1_ue1_total_throughput.png"

plt.savefig(
    figure_path,
    dpi=600,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)

print("\nFigure 1 saved successfully.")
print("File:", figure_path)