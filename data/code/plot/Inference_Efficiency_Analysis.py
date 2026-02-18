import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 14,
    'font.family': 'serif',
    'svg.fonttype': 'none',
    'axes.labelsize': 15,
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'legend.fontsize': 16,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'axes.unicode_minus': False,
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.linestyle': '--'
})

color_serial = '#1976D2'
color_parallel = '#009688'
color_accent = '#FFB366'

datasets = ['MedBullets(op4)', 'MedQA', 'MedXpertQA', 'MedBullets(op5)', 'HLE(Biomed)']
serial_latency = np.array([11.75, 9.75, 9.80, 9.46, 9.97])
parallel_latency = np.array([8.81, 7.57, 7.55, 7.54, 7.66])
speedup_data = serial_latency / parallel_latency

test_stages = ['128', '256', '512', '1024', '2048']
serial_tp = [10.14, 10.22, 9.97, 9.98, 10.09]
parallel_tp = [11.03, 11.98, 12.62, 13.74, 17.08]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

x = np.arange(len(datasets))
width = 0.38

rects1 = ax1.bar(x - width/2, serial_latency, width, label='Autoregressive Baseline (Left axis)',
                color=color_serial, alpha=0.7, edgecolor='white', linewidth=1.2)
rects2 = ax1.bar(x + width/2, parallel_latency, width, label='Multiverse-Med (Left axis)',
                color=color_parallel, alpha=0.9, edgecolor='white', linewidth=1.2)

ax1.set_ylabel('Avg. Latency (s)', fontweight='bold')
ax1.set_title('(a) End-to-End Latency & Speedup', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(datasets, rotation=15)
ax1.set_ylim(0, 15.7)
ax1.grid(axis='y')

ax1_twin = ax1.twinx()

line_speedup = ax1_twin.plot(x, speedup_data, color=color_accent, marker='s', markersize=8,
                             linewidth=3, label='Speedup Rate (Right axis)', zorder=10)

ax1_twin.set_ylabel('Speedup Rate (x Baseline)', color=color_accent, fontweight='bold')
ax1_twin.tick_params(axis='y', labelcolor=color_accent)
ax1_twin.set_ylim(1.0, 1.48)
ax1_twin.grid(False)

for i, val in enumerate(speedup_data):
    ax1_twin.text(x[i], val + 0.015, f'{val:.2f}x', ha='center', va='bottom',
                  fontweight='bold', color=color_accent, fontsize=13)

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(handles2 + handles1, labels2 + labels1, loc='upper right', frameon=True, fancybox=True, framealpha=0.9)


ax2.plot(test_stages, serial_tp, linestyle='--', linewidth=2.5, color=color_serial,
         marker='o', markersize=9, markerfacecolor='white', markeredgewidth=2,
         label='Autoregressive Baseline')
ax2.plot(test_stages, parallel_tp, linestyle='-', linewidth=3, color=color_parallel,
         marker='D', markersize=9, markeredgecolor='white', markeredgewidth=1.5,
         label='Multiverse-Med')
ax2.fill_between(test_stages, serial_tp, parallel_tp, color=color_parallel, alpha=0.15)

ax2.set_xlabel('Token Limit (Sequence Length)', fontweight='bold')
ax2.set_ylabel('Throughput (Tokens/Sec)', fontweight='bold')
ax2.set_title('(b) Iso-Length Throughput Scaling', pad=15)
ax2.set_ylim(8, 20)
ax2.legend(frameon=True, fancybox=True, framealpha=0.9)
ax2.grid(True)

# max_gain = (parallel_tp[-1] / serial_tp[-1] - 1) * 100
# ax2.annotate(f'Max +{max_gain:.1f}% Gain',
#              xy=(4, parallel_tp[-1]), xytext=(1.8, 17.5),
#              arrowprops=dict(facecolor=color_accent, edgecolor=color_accent, arrowstyle='-|>', lw=2.5),
#              fontweight='bold', color=color_accent, fontsize=11)

plt.tight_layout(pad=2)

output_svg = 'multiverse_efficiency_dual_axis.svg'
output_png = 'multiverse_efficiency_dual_axis.png'
plt.savefig(output_svg, format='svg', bbox_inches='tight')
plt.savefig(output_png, dpi=300, bbox_inches='tight')
plt.show()

print(f"已生成双Y轴图表: {output_svg} 和 {output_png}")