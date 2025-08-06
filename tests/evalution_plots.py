import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
df = pd.read_csv("evaluation_results.csv")

# Aggregate results
summary = df.groupby(["chunking", "embedding_model"]).agg(
    avg_precision=("precision", "mean"),
    avg_latency=("latency_sec", "mean")
).reset_index()

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.set_style("whitegrid")

# --- Precision plot ---
sns.barplot(
    data=summary,
    x="chunking",
    y="avg_precision",
    hue="embedding_model",
    ax=axes[0],
    palette="Set2"
)
axes[0].set_title("Average Precision by Chunking & Embedding")
axes[0].set_ylim(0, 1.1)
axes[0].set_ylabel("Precision")
axes[0].set_xlabel("Chunking Method")

# --- Latency plot ---
sns.barplot(
    data=summary,
    x="chunking",
    y="avg_latency",
    hue="embedding_model",
    ax=axes[1],
    palette="Set2"
)
axes[1].set_title("Average Latency by Chunking & Embedding")
axes[1].set_ylabel("Latency (seconds)")
axes[1].set_xlabel("Chunking Method")

plt.tight_layout()
plt.savefig("evaluation_graphs.png", dpi=300)
plt.show()
