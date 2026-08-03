import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. Set Nature Communications Paper Style Parameters
# ==========================================
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 8.5
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 7.5
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['axes.linewidth'] = 0.6
plt.rcParams['xtick.major.width'] = 0.6
plt.rcParams['ytick.major.width'] = 0.6
plt.rcParams['xtick.minor.width'] = 0.4
plt.rcParams['ytick.minor.width'] = 0.4
plt.rcParams['figure.dpi'] = 300

# ==========================================
# 2. Data Preparation
# ==========================================
algorithms = ['LightQDRL', 'Quantum\nQ-learning', 'Quantum\nQDQN', 'Quantum\nQPPO']
x = np.arange(len(algorithms))

# (a) Inference latency (ms)
avg_latency = [8.3941, 41.0810, 8.2434, 8.3109]
p95_latency = [8.4946, 167.3346, 8.3361, 8.4044]

# (b) Computational complexity (MFLOPs)
flops = [0.0023, 1073.7420, 0.0034, 0.0028]

# (c) Static model/table memory footprint (MB)
model_size = [0.0095, 8192.0005, 0.0275, 0.0110]

# (d) Online fine-tuning energy consumption per step (mJ & ms)
energy_per_step_mj = [0.1131 * 1000, 1.1489 * 1000, 0.1106 * 1000, 0.1027 * 1000] # J -> mJ
step_time_ms = [22.63, 229.78, 22.12, 20.55]

# ==========================================
# 3. Color Palette (Nature Palette: Colorblind-friendly)
# ==========================================
alg_colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
bar_width = 0.35

# ==========================================
# 4. Create 2x2 Canvas with 4 Subplots (Double-column width ~183 mm = 7.2 inches)
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5))
fig.subplots_adjust(wspace=0.28, hspace=0.38)

# ------------------------------------------
# Subplot (a): Inference Latency and P95 Tail Latency (Value annotations included)
# ------------------------------------------
ax1 = axes[0, 0]
rects1 = ax1.bar(x - bar_width/2, avg_latency, bar_width, label='Avg Latency', color='#56B4E9', edgecolor='black', linewidth=0.5)
rects2 = ax1.bar(x + bar_width/2, p95_latency, bar_width, label='P95 Latency', color='#E69F00', edgecolor='black', linewidth=0.5)

ax1.set_ylabel('Inference latency (ms)')
ax1.set_yscale('log')  # Logarithmic scale to accommodate the severe 167ms fluctuation of Q-learning
ax1.set_ylim(1, 400)
ax1.set_xticks(x)
ax1.set_xticklabels(algorithms)
ax1.legend(frameon=False, loc='upper left')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.5)
ax1.text(-0.2, 1.08, 'a', transform=ax1.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')

# Add numerical labels to Avg Latency bars in Subplot (a)
for rect in rects1:
    h = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2.0, h * 1.25, f'{h:.2f}',
             ha='center', va='bottom', fontsize=5.5, rotation=25)

# Add numerical labels to P95 Latency bars in Subplot (a)
for rect in rects2:
    h = rect.get_height()
    ax1.text(rect.get_x() + rect.get_width()/2.0, h * 1.25, f'{h:.2f}',
             ha='center', va='bottom', fontsize=5.5, rotation=25)

# ------------------------------------------
# Subplot (b): Computational Complexity
# ------------------------------------------
ax2 = axes[0, 1]
bars2 = ax2.bar(x, flops, width=0.5, color=alg_colors, edgecolor='black', linewidth=0.5)

ax2.set_ylabel('FLOPs per inference (MFLOPs)')
ax2.set_yscale('log')
ax2.set_ylim(1e-4, 5e3)
ax2.set_xticks(x)
ax2.set_xticklabels(algorithms)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.5)
ax2.text(-0.2, 1.08, 'b', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')

for bar, val in zip(bars2, flops):
    y_val = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, y_val * 1.8, f'{val:.4f}' if val < 1 else f'{val:.1f}',
             ha='center', va='bottom', fontsize=6.5)

# ------------------------------------------
# Subplot (c): Static Memory Footprint
# ------------------------------------------
ax3 = axes[1, 0]
bars3 = ax3.bar(x, model_size, width=0.5, color=alg_colors, edgecolor='black', linewidth=0.5)

ax3.set_ylabel('Static memory size (MB)')
ax3.set_yscale('log')
ax3.set_ylim(1e-3, 5e4)
ax3.set_xticks(x)
ax3.set_xticklabels(algorithms)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.5)
ax3.text(-0.2, 1.08, 'c', transform=ax3.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')

for bar, val in zip(bars3, model_size):
    y_val = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2.0, y_val * 1.8, f'{val:.4f}' if val < 1 else f'{val:.0f}',
             ha='center', va='bottom', fontsize=6.5)

# ------------------------------------------
# Subplot (d): Fine-Tuning Energy Consumption and Step Time (Fine-Tuning Efficiency)
# ------------------------------------------
ax4 = axes[1, 1]
bars4 = ax4.bar(x, energy_per_step_mj, width=0.5, color=alg_colors, alpha=0.85, edgecolor='black', linewidth=0.5)

ax4.set_ylabel('Energy per step (mJ)')
ax4.set_yscale('log')
ax4.set_ylim(10, 5000)
ax4.set_xticks(x)
ax4.set_xticklabels(algorithms)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.5)
ax4.text(-0.2, 1.08, 'd', transform=ax4.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')

for bar, val_e, val_t in zip(bars4, energy_per_step_mj, step_time_ms):
    y_val = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2.0, y_val * 1.4, f'{val_e:.1f} mJ\n({val_t:.1f} ms)',
             ha='center', va='bottom', fontsize=6.0)

# ==========================================
# 5. Save Figures
# ==========================================
plt.savefig('Nature_Comms_Benchmark_4Subplots.pdf', format='pdf', bbox_inches='tight')
plt.savefig('Nature_Comms_Benchmark_4Subplots.png', format='png', dpi=600, bbox_inches='tight')
plt.show()