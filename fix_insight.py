import matplotlib.pyplot as plt
import os

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 12

fig, ax = plt.subplots(figsize=(12, 6))

ax.text(0.5, 0.9, 'Why CwC-FF Achieves Best Transfer?', fontsize=20, fontweight='bold', 
        ha='center', transform=ax.transAxes)

# Three boxes without emojis
ax.text(0.15, 0.55, 'Standard FF\n\n[X] Label in input\n[X] Task-specific\n[X] Poor transfer', 
        fontsize=12, ha='center', va='center', transform=ax.transAxes, 
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffcccc', edgecolor='#e74c3c', linewidth=2))

ax.text(0.5, 0.55, 'Backprop\n\n[!] Global error\n[!] Classifier focused\n[!] Moderate transfer', 
        fontsize=12, ha='center', va='center', transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffcc', edgecolor='#f39c12', linewidth=2))

ax.text(0.85, 0.55, 'CwC-FF (Ours)\n\n[v] No labels needed\n[v] Universal features\n[v] Best transfer!', 
        fontsize=12, ha='center', va='center', transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#ccffcc', edgecolor='#27ae60', linewidth=2))

ax.text(0.5, 0.15, 'KEY INSIGHT: Channel-wise competition learns task-agnostic features\n'
        'that generalize better across domains', 
        fontsize=14, ha='center', va='center', transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#e8f4f8', edgecolor='#2980b9', linewidth=2))

ax.axis('off')
plt.tight_layout()
plt.savefig('figures/key_insight.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Fixed: key_insight.png")
