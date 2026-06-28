import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load dataset
DATA_PATH = Path("data")
FILE_NAME = "ue-lte-network-traffic-stats.csv"

df = pd.read_csv(DATA_PATH / FILE_NAME)

# Create plots folder if it doesn't exist
PLOTS_PATH = Path("plots")
PLOTS_PATH.mkdir(exist_ok=True)

# 1. Throughput Over Time
# -----------------------------
plt.figure(figsize=(12, 5))
plt.plot(df["UE1: web-rtc"])
plt.title("UE1 WebRTC Throughput Over Time")
plt.xlabel("Sample")
plt.ylabel("Throughput (Bytes)")
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_PATH / "01_throughput_over_time.png", dpi=300)
plt.close()

# 2. Throughput Distribution
# -----------------------------
plt.figure(figsize=(8, 5))
plt.hist(df["UE1: web-rtc"], bins=50)
plt.title("Distribution of UE1 WebRTC Throughput")
plt.xlabel("Throughput (Bytes)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(PLOTS_PATH / "02_throughput_histogram.png", dpi=300)
plt.close()

# 3. CQI Distribution
# -----------------------------
plt.figure(figsize=(8, 5))
plt.hist(df["UE1-CQI"], bins=16)
plt.title("Distribution of UE1 CQI")
plt.xlabel("CQI")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(PLOTS_PATH / "03_cqi_histogram.png", dpi=300)
plt.close()

# 4. Jitter Distribution
# -----------------------------
plt.figure(figsize=(8, 5))
plt.hist(df["UE1-Jitter"], bins=30)
plt.title("Distribution of UE1 Jitter")
plt.xlabel("Jitter (Seconds)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(PLOTS_PATH / "04_jitter_histogram.png", dpi=300)
plt.close()

# 5. CQI vs Throughput
# -----------------------------
plt.figure(figsize=(8, 5))
plt.scatter(
    df["UE1-CQI"],
    df["UE1: web-rtc"],
    alpha=0.3
)
plt.title("UE1 CQI vs Throughput")
plt.xlabel("CQI")
plt.ylabel("Throughput (Bytes)")
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_PATH / "05_cqi_vs_throughput.png", dpi=300)
plt.close()

# 6. Jitter vs Throughput
# -----------------------------
plt.figure(figsize=(8, 5))
plt.scatter(
    df["UE1-Jitter"],
    df["UE1: web-rtc"],
    alpha=0.3
)
plt.title("UE1 Jitter vs Throughput")
plt.xlabel("Jitter (Seconds)")
plt.ylabel("Throughput (Bytes)")
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOTS_PATH / "06_jitter_vs_throughput.png", dpi=300)
plt.close()

# 7. Correlation Heatmap
# -----------------------------
corr = df[["UE1: web-rtc", "UE1-Jitter", "UE1-CQI"]].corr()

plt.figure(figsize=(6, 5))
plt.imshow(corr, cmap="coolwarm", interpolation="nearest")
plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(
            j,
            i,
            f"{corr.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            color="black"
        )

plt.title("UE1 Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(PLOTS_PATH / "07_correlation_heatmap.png", dpi=300)
plt.close()

print("All plots successfully saved to the 'plots' folder.")