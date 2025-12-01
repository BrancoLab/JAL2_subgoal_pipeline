from pathlib import Path
import pandas as pd
LEARNING_METRICS_PATH = Path(r"Z:\Laurence\thesis\efizz_chapter\outputs\learning_metrics_per_condition.csv")
df = pd.read_csv(LEARNING_METRICS_PATH)
print(df.columns)
print(df["escapes_median_latency"])
df["escapes_median_latency"] = df["escapes_median_latency"] / 40  # scale to seconds

# plot the distribution
import matplotlib.pyplot as plt
import seaborn as sns
sns.histplot(df["escapes_median_latency"].dropna(), bins=30, kde=True)
plt.xlabel("Escapes Median Latency (seconds)")
plt.title("Distribution of Escapes Median Latency")
plt.show()