from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


DATA_PATH = Path("data")
PLOTS_PATH = Path("plots")

FILE_NAME = "ue-lte-network-traffic-stats.csv"

PLOTS_PATH.mkdir(exist_ok=True)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH / FILE_NAME)


# --------------------------------------------------
# Create total UE1 throughput
#
# This is the same target used by the ML models.
# --------------------------------------------------

df["UE1_total_throughput"] = (
    df["UE1: web-rtc"]
    + df["UE1: sipp"]
    + df["UE1: web-server"]
)

TARGET = "UE1_total_throughput"

plt.rcParams["font.family"] = "DejaVu Serif"


# --------------------------------------------------
# 1. Total UE1 throughput over time
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(
    df.index,
    df[TARGET],
    linewidth=0.7,
)

ax.set_title("Total UE1 Throughput Over Time")
ax.set_xlabel("Sample Index")
ax.set_ylabel("Total UE1 Throughput (Bytes/s)")

ax.yaxis.set_major_formatter(
    StrMethodFormatter("{x:,.0f}")
)

ax.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    alpha=0.25,
)

plt.tight_layout()

plt.savefig(
    PLOTS_PATH / "01_total_throughput_over_time.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# --------------------------------------------------
# 2. Total throughput distribution
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 4))

ax.hist(
    df[TARGET],
    bins=50,
)

ax.set_title("Distribution of Total UE1 Throughput")
ax.set_xlabel("Total UE1 Throughput (Bytes/s)")
ax.set_ylabel("Frequency")

ax.xaxis.set_major_formatter(
    StrMethodFormatter("{x:,.0f}")
)

plt.tight_layout()

plt.savefig(
    PLOTS_PATH / "02_total_throughput_histogram.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# --------------------------------------------------
# 3. CQI distribution
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 4))

ax.hist(
    df["UE1-CQI"],
    bins=16,
)

ax.set_title("Distribution of UE1 CQI")
ax.set_xlabel("Channel Quality Indicator (CQI)")
ax.set_ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    PLOTS_PATH / "03_cqi_histogram.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# --------------------------------------------------
# 4. Jitter distribution
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 4))

ax.hist(
    df["UE1-Jitter"],
    bins=30,
)

ax.set_title("Distribution of UE1 Jitter")
ax.set_xlabel("Jitter")
ax.set_ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    PLOTS_PATH / "04_jitter_histogram.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# --------------------------------------------------
# 5. CQI vs total throughput
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 4))

ax.scatter(
    df["UE1-CQI"],
    df[TARGET],
    alpha=0.25,
    s=8,
)

ax.set_title("UE1 CQI vs Total Throughput")
ax.set_xlabel("Channel Quality Indicator (CQI)")
ax.set_ylabel("Total UE1 Throughput (Bytes/s)")

ax.yaxis.set_major_formatter(
    StrMethodFormatter("{x:,.0f}")
)

ax.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    alpha=0.25,
)

plt.tight_layout()

plt.savefig(
    PLOTS_PATH / "05_cqi_vs_total_throughput.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# --------------------------------------------------
# 6. Jitter vs total throughput
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 4))

ax.scatter(
    df["UE1-Jitter"],
    df[TARGET],
    alpha=0.25,
    s=8,
)

ax.set_title("UE1 Jitter vs Total Throughput")
ax.set_xlabel("Jitter")
ax.set_ylabel("Total UE1 Throughput (Bytes/s)")

ax.yaxis.set_major_formatter(
    StrMethodFormatter("{x:,.0f}")
)

ax.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    alpha=0.25,
)

plt.tight_layout()

plt.savefig(
    PLOTS_PATH / "06_jitter_vs_total_throughput.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# --------------------------------------------------
# 7. Correlation matrix
# --------------------------------------------------

correlation_columns = [
    TARGET,
    "UE1-Jitter",
    "UE1-CQI",
]

corr = df[correlation_columns].corr()

display_names = [
    "Total Throughput",
    "Jitter",
    "CQI",
]

fig, ax = plt.subplots(figsize=(6, 5))

image = ax.imshow(
    corr,
    interpolation="nearest",
)

fig.colorbar(
    image,
    ax=ax,
    label="Correlation",
)

ax.set_xticks(
    range(len(display_names)),
    labels=display_names,
    rotation=30,
    ha="right",
)

ax.set_yticks(
    range(len(display_names)),
    labels=display_names,
)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        ax.text(
            j,
            i,
            f"{corr.iloc[i, j]:.2f}",
            ha="center",
            va="center",
        )

ax.set_title("UE1 Feature Correlation Matrix")

plt.tight_layout()

plt.savefig(
    PLOTS_PATH / "07_correlation_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


print(
    "Exploratory visualizations successfully saved "
    "to the 'plots' directory."
)
