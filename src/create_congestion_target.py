import pandas as pd

df = pd.read_csv("data/6G_network_slicing_qos_dataset_2345.csv")

df["Congestion Score"] = (
    0.25 * df["Traffic Load (bps)"]
    + 0.25 * df["Network Utilization (%)"]
    + 0.20 * df["Latency (ms)"]
    + 0.15 * df["Packet Loss Rate (%)"]
    + 0.15 * (1 - df["Signal Strength (dBm)"])
)

df["Congestion Class"] = pd.qcut(
    df["Congestion Score"],
    q=3,
    labels=["Low", "Medium", "High"]
)

print(df["Congestion Score"].describe())

print("\nClass Counts:")
print(df["Congestion Class"].value_counts())

df.to_csv(
    "data/6G_network_slicing_congestion_dataset.csv",
    index=False
)

print("\nNew dataset saved.")