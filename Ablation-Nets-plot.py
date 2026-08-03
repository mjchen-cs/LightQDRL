import matplotlib.pyplot as plt
import numpy as np

# Set Nature Communications style parameters
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 8.5
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['lines.linewidth'] = 0.8
plt.rcParams['axes.linewidth'] = 0.6
plt.rcParams['xtick.major.width'] = 0.6
plt.rcParams['ytick.major.width'] = 0.6
plt.rcParams['xtick.minor.width'] = 0.4
plt.rcParams['ytick.minor.width'] = 0.4
plt.rcParams['figure.dpi'] = 300

models = ['FC', 'LSTM', 'ResNet', 'ResNeXt', 'Transformer', 'MobileNet']
y = np.arange(len(models))

norm_rate = [
    0.9936676509975257,
    0.9610830832333223,
    0.9952600822111806,
    0.9939282386501899,
    0.9522576739784383,
    0.9943097828728690
]

total_time = [
    593.3792476654053,
    809.1060669422150,
    578.1969642639160,
    603.0539939403534,
    843.3704121112823,
    656.5893306732178
]

avg_time = [
    0.01977930825551351,
    0.026970202231407166,
    0.019273232142130534,
    0.020101799798011778,
    0.02811234707037608,
    0.021886311022440593
]

model_colors = ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7', '#56B4E9']

# Option A: 1x3 with horizontal bars (barh) - very clear, no label overlap!
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
fig.subplots_adjust(wspace=0.45)

# Subplot (a)
ax1 = axes[0]
bars1 = ax1.barh(y, norm_rate, height=0.55, color=model_colors, edgecolor='black', linewidth=0.5)
ax1.set_xlabel('Normalized computation rate')
ax1.set_xlim(0.92, 1.01)
ax1.set_yticks(y)
ax1.set_yticklabels(models)
ax1.invert_yaxis()  # top-down
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='x', linestyle='--', alpha=0.4, linewidth=0.5)
ax1.text(-0.35, 1.08, 'a', transform=ax1.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')

for bar, val in zip(bars1, norm_rate):
    ax1.text(val + 0.001, bar.get_y() + bar.get_height()/2.0, f'{val:.4f}',
             ha='left', va='center', fontsize=6)

# Subplot (b)
ax2 = axes[1]
bars2 = ax2.barh(y, total_time, height=0.55, color=model_colors, edgecolor='black', linewidth=0.5)
ax2.set_xlabel('Total training time (s)')
ax2.set_xlim(0, 1050)
ax2.set_yticks(y)
ax2.set_yticklabels(models)
ax2.invert_yaxis()
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='x', linestyle='--', alpha=0.4, linewidth=0.5)
ax2.text(-0.35, 1.08, 'b', transform=ax2.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')

for bar, val in zip(bars2, total_time):
    ax2.text(val + 15, bar.get_y() + bar.get_height()/2.0, f'{val:.1f}',
             ha='left', va='center', fontsize=6)

# Subplot (c)
ax3 = axes[2]
ms_times = [t * 1000 for t in avg_time]
bars3 = ax3.barh(y, ms_times, height=0.55, color=model_colors, edgecolor='black', linewidth=0.5)
ax3.set_xlabel('Average time / channel (ms)')
ax3.set_xlim(0, 36)
ax3.set_yticks(y)
ax3.set_yticklabels(models)
ax3.invert_yaxis()
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.grid(axis='x', linestyle='--', alpha=0.4, linewidth=0.5)
ax3.text(-0.35, 1.08, 'c', transform=ax3.transAxes, fontsize=10, fontweight='bold', va='top', ha='right')

for bar, val in zip(bars3, ms_times):
    ax3.text(val + 0.5, bar.get_y() + bar.get_height()/2.0, f'{val:.2f}',
             ha='left', va='center', fontsize=6)

plt.tight_layout()
plt.savefig('Nature_Comms_Ablation_Study_v2.png', dpi=300, bbox_inches='tight')
print("Option A (barh) generated successfully.")