import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. Global Font and Academic Layout Settings (Enlarged Font Size)
# ==========================================
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.unicode_minus': False
})

# ==========================================
# 2. Prepare Data (Drawn from bottom to top, LightQDRL at the bottom)
# ==========================================
algorithms = [
    'LightQDRL\n(Ours)',
    'Quantum \nDQN',
    'Quantum \nQ-Learning',
    'Quantum \nPPO'
]

# Latency data for each stage (ms) [LightQDRL, Quantum DQN, Quantum Q-Learning, Quantum PPO]
task_encoding = [0.573, 0.310,  0.573,  0.354]  # Stage 1: Task encoding (Blue)
transmission  =  [9.472, 12.252, 19.108, 12.935]  # Stage 2: 5G Transmission (Green)
#inference     = [1.000, 1.000,  1.000,  1.000]   # Stage 3: Edge Inference (Orange)
#decoding      = [9.472, 12.252, 19.108, 12.935]  # Stage 4: Decoding / Decision (Purple)
#execution     = [1.000, 1.000,  1.000,  1.000]   # Stage 5: Post-processing / Execution (Gray)

data = np.array([task_encoding, transmission])
totals = data.sum(axis=0)  # Compute total latency (~21.52 ms ~ 40.79 ms)

# Legend labels and standard colors for each stage
categories = [
    'Task Encoding',
    'Transmission (5G link)',
    'Inference\n(Edge)',
    'Decoding\n/ Decision',
    'Execution\n/ Post'
]
colors = ['#5b84e3', '#ff922b', '#bebebe', '#78ab63', '#ff922b', '#b07cc6']

# ==========================================
# 3. Create Canvas and Plot Horizontal Stacked Bar Chart
# ==========================================
fig, ax = plt.subplots(figsize=(8.5, 5.5))

left_offset = np.zeros(len(algorithms))

for i in range(len(data)):
    bars = ax.barh(
        algorithms,
        data[i],
        left=left_offset,
        color=colors[i],
        edgecolor='#555555',
        linewidth=0.8,
        height=0.52,
        label=categories[i]
    )

    # Write specific latency values inside each bar
    for j, bar in enumerate(bars):
        val = data[i][j]
        # Skip writing when the bar segment is too small (e.g., < 0.4 ms) to avoid text overlap
        if val >= 0.4:
            # Format numbers: float with 1 decimal place (integer without decimal)
            val_text = f'{val:.1f}' if val % 1 != 0 else f'{int(val)}'
            ax.text(
                left_offset[j] + val / 2.0,
                bar.get_y() + bar.get_height() / 2.0,
                val_text,
                ha='center',
                va='center',
                color='black',
                fontsize=10,
                fontweight='bold' if j == 0 else 'normal'  # Highlight our algorithm in bold
            )

    left_offset += data[i]

# Annotate total latency on the right side of each bar (rounded to 2 decimal places)
for j, total in enumerate(totals):
    ax.text(
        total + 0.8,  # Offset to +0.8 on the right side under the x-axis scale
        j,
        f'{total:.2f} s',
        ha='left',
        va='center',
        fontsize=11,
        fontweight='bold'
    )

# ==========================================
# 4. Chart Detail Adjustments
# ==========================================
ax.set_xlabel('Time (s)', fontsize=13, labelpad=8)
ax.set_title('Real 5G video streaming example\n(task timeline)',
             fontsize=15, fontweight='bold', pad=12)

# Specifically configure Y-axis labels: set LightQDRL (Ours) to bold orange
yticklabels = ax.get_yticklabels()
for label in yticklabels:
    if 'LightQDRL' in label.get_text():
        label.set_color('#ff922b')
        label.set_fontweight('bold')

# [Key Modification]: Set X-axis range from 0 to 22
ax.set_xlim(0, 22)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Bottom horizontal legend
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.22),
    ncol=5,
    frameon=True,
    edgecolor='#888888',
    handlelength=1.2,
    handleheight=1.2,
    columnspacing=1.2
)

plt.tight_layout()

# Save high-resolution paper figure
plt.savefig('Fig3_d_Real5G_Timeline.png', dpi=300, bbox_inches='tight')
plt.show()