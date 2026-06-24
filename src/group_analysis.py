import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/6G_network_slicing_qos_dataset_2345.csv")

print(df.groupby("Overload Status")["QoS Metric (Throughput)"].mean())

print(df.groupby("Network Slice Failure")["QoS Metric (Throughput)"].mean())

print(df.groupby("Weather Conditions")["QoS Metric (Throughput)"].mean())

plt.hist(df["QoS Metric (Throughput)"], bins=30)
plt.xlabel("QoS Throughput")
plt.ylabel("Count")
plt.title("Throughput Distribution")
plt.savefig("plots/throughput_histogram.png")
plt.close()

plt.scatter(
    df["Network Utilization (%)"],
    df["QoS Metric (Throughput)"]
)
plt.xlabel("Network Utilization (%)")
plt.ylabel("QoS Metric (Throughput)")
plt.title("Network Utilization vs Throughput")
plt.savefig("plots/utilization_vs_throughput.png")
plt.close()

plt.scatter(
    df["Packet Loss Rate (%)"],
    df["QoS Metric (Throughput)"]
)
plt.xlabel("Packet Loss Rate (%)")
plt.ylabel("QoS Metric (Throughput)")
plt.title("Packet Loss vs Throughput")
plt.savefig("plots/packetloss_vs_throughput.png")
plt.close()

print("\nPlots saved successfully.")