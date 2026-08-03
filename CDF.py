import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Global Font and Academic Layout Settings
# ==========================================
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'axes.unicode_minus': False
})

# ==========================================
# 2. Read Excel File and Clean Columns/Data
# ==========================================
file_path = 'CDF.xlsx'
df = pd.read_excel(file_path)

# [Key Fix 1]: Strip leading and trailing whitespace from column names automatically
df.columns = df.columns.astype(str).str.strip()

print("--- Data Loading Diagnostics ---")
print("Loaded column names:", list(df.columns))

# Algorithm column names to plot
algorithms = [
    'Traditional Quantum Q-Learning',
    'Traditional Quantum-DQN',
    'Traditional Quantum-PPO',
    'LightQDRL'
]

styles = {
    'Traditional Quantum Q-Learning': {'color': '#1f77b4', 'linestyle': '--', 'linewidth': 1.8},
    'Traditional Quantum-DQN': {'color': '#ff7f0e', 'linestyle': '-.', 'linewidth': 1.8},
    'Traditional Quantum-PPO': {'color': '#2ca02c', 'linestyle': ':', 'linewidth': 1.8},
    'LightQDRL': {'color': '#d62728', 'linestyle': '-', 'linewidth': 2.5}  # Bold red solid line
}

# ==========================================
# 3. Compute CDF and Plot Curves
# ==========================================
fig, ax = plt.subplots(figsize=(8, 5))

for alg in algorithms:
    if alg in df.columns:
        # [Key Fix 2]: Force conversion to numeric type, replace invalid values with NaN, and drop NaNs
        clean_series = pd.to_numeric(df[alg], errors='coerce').dropna()
        latency_data = clean_series.values

        # Print diagnostic info for troubleshooting
        print(
            f"[{alg}] Valid data count: {len(latency_data)} | Min: {np.min(latency_data) if len(latency_data) > 0 else 'N/A'} | Max: {np.max(latency_data) if len(latency_data) > 0 else 'N/A'}")

        if len(latency_data) == 0:
            print(f"⚠️ Warning: Column [{alg}] contains no valid numeric data, skipping plot!")
            continue

        # 1. Sort in ascending order
        sorted_latency = np.sort(latency_data)

        # 2. Compute Cumulative Distribution Function (CDF) values
        cdf = np.arange(1, len(sorted_latency) + 1) / len(sorted_latency)

        # 3. Plot CDF curve
        style = styles.get(alg, {'linewidth': 1.8})
        plt.plot(sorted_latency, cdf, label=alg, **style)
    else:
        print(f"❌ Error: Column [{alg}] not found in Excel! Please check spelling.")

print("--------------------")

# ==========================================
# 4. Chart Styling and Export
# ==========================================
plt.xlabel('End-to-End Latency (ms)')
plt.ylabel('Cumulative Probability (CDF)')
plt.title('CDF of End-to-End Latency')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right', frameon=True)
plt.ylim(0, 1.02)

plt.tight_layout()

save_path = 'CDF_End_to_End_Latency.png'
plt.savefig(save_path, dpi=300)
plt.show()