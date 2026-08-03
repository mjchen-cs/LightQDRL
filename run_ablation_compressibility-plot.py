import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. Set plotting parameters conforming to Nature Communications specifications
# ==========================================
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.major.size'] = 3.5
plt.rcParams['ytick.major.size'] = 3.5
plt.rcParams['font.size'] = 8
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['axes.labelsize'] = 8.5
plt.rcParams['xtick.labelsize'] = 7.5
plt.rcParams['ytick.labelsize'] = 7.5

# ==========================================
# 2. Prepare ablation experiment data
# ==========================================
configs = [
    'Baseline\n(n=6, FP32)',
    'n = 4',
    'n = 8',
    'Prune 30%',
    'Prune 50%',
    'INT8',
    'INT4'
]

# Experimental metric data
final_loss = [0.0532, 0.0832, 0.0484, 0.0674, 0.0459, 0.0666, 0.0541]
norm_rate = [0.9944, 0.9917, 0.9964, 0.9940, 0.9965, 0.9958, 0.9941]
training_time = [553.98, 493.41, 735.69, 577.84, 562.35, 570.23, 580.43]

# Group and color bar charts by experimental module (NPG scientific color palette)
# Baseline (grey) | Qubit scaling (blue) | Gate pruning (green) | Bit quantization (orange-red)
colors = [
    '#4C566A',  # Baseline
    '#5E81AC', '#81A1C1',  # Qubit scaling
    '#A3BE8C', '#8FBCBB',  # Pruning
    '#D08770', '#E5E9F0'   # Quantization (use bordered palette to enhance contrast)
]
colors = ['#3B4252', '#0072B2', '#56B4E9', '#009E73', '#72C2A2', '#D55E00', '#E69F00']

x = np.arange(len(configs))
bar_width = 0.55

# ==========================================
# 3. Create Fig. 6 (1 x 3 subplots layout)
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), dpi=300)

# ------------------------------------------
# (a) Final Training Loss
# ------------------------------------------
ax1 = axes[0]
bars1 = ax1.bar(x, final_loss, width=bar_width, color=colors, edgecolor='#2E3440', linewidth=0.6)
ax1.set_ylabel('Final Training Loss')
ax1.set_xticks(x)
ax1.set_xticklabels(configs, rotation=35, ha='right')
ax1.set_ylim(0, 0.10)
ax1.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.5)

# Add numerical value labels for bars
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.002, f'{yval:.4f}',
             ha='center', va='bottom', fontsize=6.5, color='#2E3440')

# Subplot label a
ax1.text(-0.22, 1.06, 'a', transform=ax1.transAxes, fontsize=11, fontweight='bold', va='top', ha='right')

# ------------------------------------------
# (b) Normalized Computation Rate
# ------------------------------------------
ax2 = axes[1]
bars2 = ax2.bar(x, norm_rate, width=bar_width, color=colors, edgecolor='#2E3440', linewidth=0.6)
ax2.set_ylabel('Normalized Computation Rate')
ax2.set_xticks(x)
ax2.set_xticklabels(configs, rotation=35, ha='right')
# Truncate Y-axis to magnify minor throughput differences
ax2.set_ylim(0.988, 1.000)
ax2.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.5)

for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.0003, f'{yval:.4f}',
             ha='center', va='bottom', fontsize=6.5, color='#2E3440')

# Subplot label b
ax2.text(-0.22, 1.06, 'b', transform=ax2.transAxes, fontsize=11, fontweight='bold', va='top', ha='right')

# ------------------------------------------
# (c) Total Training Time (s)
# ------------------------------------------
ax3 = axes[2]
bars3 = ax3.bar(x, training_time, width=bar_width, color=colors, edgecolor='#2E3440', linewidth=0.6)
ax3.set_ylabel('Total Training Time (s)')
ax3.set_xticks(x)
ax3.set_xticklabels(configs, rotation=35, ha='right')
ax3.set_ylim(0, 850)
ax3.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.5)

for bar in bars3:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 15, f'{yval:.1f}',
             ha='center', va='bottom', fontsize=6.5, color='#2E3440')

# Subplot label c
ax3.text(-0.22, 1.06, 'c', transform=ax3.transAxes, fontsize=11, fontweight='bold', va='top', ha='right')

# ==========================================
# 4. Adjust layout and export high-resolution figure
# ==========================================
plt.tight_layout()
plt.subplots_adjust(wspace=0.32)

# Export as 300 DPI image or PDF vector graphic as required by Nature
plt.savefig('./result-Compressibility_Ablation/Fig6_Compressibility_Ablation.png', dpi=300, bbox_inches='tight')
plt.savefig('./result-Compressibility_Ablation/Fig6_Compressibility_Ablation.pdf', format='pdf', bbox_inches='tight')
plt.show()