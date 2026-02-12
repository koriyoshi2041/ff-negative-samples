#!/usr/bin/env python3
"""Create insight-driven visualizations for FF research."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.facecolor'] = 'white'

# ============================================================================
# Figure 1: The Label Embedding Problem (Core Insight)
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: Standard FF - Label Embedding Visualization
ax1 = axes[0]
ax1.set_xlim(0, 28)
ax1.set_ylim(0, 28)

# Draw MNIST-like grid
for i in range(29):
    ax1.axhline(y=i, color='lightgray', linewidth=0.5)
    ax1.axvline(x=i, color='lightgray', linewidth=0.5)

# Highlight first 10 pixels (label embedding area)
label_area = mpatches.Rectangle((0, 27), 10, 1, linewidth=2,
                                   edgecolor='red', facecolor='#ffcccc', alpha=0.8)
ax1.add_patch(label_area)

# Draw some "digit" pixels
digit_pixels = [(14, 20), (13, 19), (14, 19), (15, 19), (14, 18), (14, 17),
                (14, 16), (13, 15), (14, 15), (15, 15), (14, 14), (14, 13)]
for x, y in digit_pixels:
    rect = mpatches.Rectangle((x, y), 1, 1, facecolor='black')
    ax1.add_patch(rect)

# Annotations
ax1.annotate('Label Embedding\n(first 10 pixels)', xy=(5, 27.5),
             fontsize=10, ha='center', va='center', color='red', fontweight='bold')
ax1.annotate('digit "1"', xy=(14.5, 10), fontsize=10, ha='center')

ax1.set_title('Standard FF: Label in Input\n(Task-specific features)', fontweight='bold')
ax1.set_aspect('equal')
ax1.axis('off')

# Panel 2: Transfer Problem Visualization
ax2 = axes[1]

# Source task (MNIST)
source_box = mpatches.FancyBboxPatch((0.1, 0.6), 0.35, 0.3,
                                      boxstyle="round,pad=0.02",
                                      facecolor='#3498db', alpha=0.3)
ax2.add_patch(source_box)
ax2.text(0.275, 0.75, 'MNIST\nLabels: 0-9', ha='center', va='center', fontsize=11)

# Target task (Fashion-MNIST)
target_box = mpatches.FancyBboxPatch((0.55, 0.6), 0.35, 0.3,
                                      boxstyle="round,pad=0.02",
                                      facecolor='#e74c3c', alpha=0.3)
ax2.add_patch(target_box)
ax2.text(0.725, 0.75, 'Fashion-MNIST\nLabels: 0-9', ha='center', va='center', fontsize=11)

# Arrow with X (transfer fails)
ax2.annotate('', xy=(0.55, 0.75), xytext=(0.45, 0.75),
             arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax2.text(0.5, 0.82, 'Transfer\nFails!', ha='center', va='center',
         fontsize=12, color='red', fontweight='bold')

# Explanation box
explain_box = mpatches.FancyBboxPatch((0.1, 0.1), 0.8, 0.35,
                                       boxstyle="round,pad=0.02",
                                       facecolor='#fff3cd', edgecolor='#ffc107')
ax2.add_patch(explain_box)
ax2.text(0.5, 0.275,
         'Problem: "0" in MNIST = digit zero\n'
         '"0" in Fashion = T-shirt\n'
         'Same label slot, different meaning!\n'
         '→ Features are coupled to source labels',
         ha='center', va='center', fontsize=10)

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_title('Why Transfer Fails\n(Label collision)', fontweight='bold')
ax2.axis('off')

# Panel 3: CwC-FF Solution
ax3 = axes[2]
ax3.set_xlim(0, 28)
ax3.set_ylim(0, 28)

# Draw grid
for i in range(29):
    ax3.axhline(y=i, color='lightgray', linewidth=0.5)
    ax3.axvline(x=i, color='lightgray', linewidth=0.5)

# No label embedding - all pixels are image
for x, y in digit_pixels:
    rect = mpatches.Rectangle((x, y), 1, 1, facecolor='black')
    ax3.add_patch(rect)

# Show channel competition concept
ax3.annotate('No label embedding!\nAll 784 pixels = image',
             xy=(14, 24), fontsize=10, ha='center', color='green', fontweight='bold')
ax3.annotate('Channel competition\nlearns visual features',
             xy=(14, 5), fontsize=10, ha='center', color='green')

ax3.set_title('CwC-FF: No Label Embedding\n(Task-agnostic features)', fontweight='bold')
ax3.set_aspect('equal')
ax3.axis('off')

plt.suptitle('THE CORE INSIGHT: Label Embedding Breaks Transfer Learning',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/insight_label_embedding.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved figures/insight_label_embedding.png")

# ============================================================================
# Figure 2: Transfer Learning Gap (Waterfall Chart)
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 6))

models = ['Random\nInit', 'Standard\nFF', 'Backprop', 'CwC-FF\n[Hinton]']
transfers = [71.89, 54.19, 75.49, 89.05]
colors = ['#95a5a6', '#e74c3c', '#3498db', '#2ecc71']

# Create bars
bars = ax.bar(models, transfers, color=colors, edgecolor='black', linewidth=1.5, width=0.6)

# Add horizontal line for random init baseline
ax.axhline(y=71.89, color='gray', linestyle='--', alpha=0.7, linewidth=2)
ax.text(3.5, 73, 'Random Init Baseline (71.89%)', fontsize=10, color='gray')

# Highlight the paradox: Standard FF below random init
ax.annotate('', xy=(1, 54.19), xytext=(1, 71.89),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(1.15, 63, '-17.7%\nWorse than\nrandom!', fontsize=10, color='red', fontweight='bold')

# Highlight CwC-FF success
ax.annotate('', xy=(3, 89.05), xytext=(3, 71.89),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax.text(3.15, 80, '+17.2%\nBest\nresult!', fontsize=10, color='green', fontweight='bold')

# Labels on bars
for bar, val in zip(bars, transfers):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.2f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Fashion-MNIST Transfer Accuracy (%)', fontsize=12)
ax.set_title('THE PARADOX: Standard FF Transfers Worse Than Random Initialization',
             fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)

# Add insight box
insight = ("Key Insight: Standard FF's label embedding creates features\n"
           "tied to source task labels. These features are HARMFUL\n"
           "when transferred to a new task with different label meanings.")
ax.text(0.02, 0.02, insight, transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'),
        verticalalignment='bottom')

plt.tight_layout()
plt.savefig('figures/insight_transfer_paradox.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved figures/insight_transfer_paradox.png")

# ============================================================================
# Figure 3: Bio-Inspired Attempts - Success vs Failure
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 7))

# Data
approaches = [
    'CwC-FF\n[Hinton §6.3]',
    'Three-Factor\n(top-down)',
    'Layer Collab\n(γ=0.7)',
    'Prospective FF\n(500 iter)',
    'Three-Factor\n(layer_agree)',
    'Prospective FF\n(transfer)',
    'PCL-FF',
    'Three-Factor\n(reward)'
]

# Improvement over baseline (in percentage points)
improvements = [35.0, 1.5, 1.2, 1.8, -3.0, -13.2, -72.5, -79.4]

# Colors based on outcome
colors = ['#2ecc71' if x > 5 else '#f39c12' if x > 0 else '#e74c3c' for x in improvements]

# Create horizontal bar chart
y_pos = np.arange(len(approaches))
bars = ax.barh(y_pos, improvements, color=colors, edgecolor='black', linewidth=0.5)

# Add vertical line at 0
ax.axvline(x=0, color='black', linewidth=2)

# Add labels
for i, (bar, imp) in enumerate(zip(bars, improvements)):
    if imp >= 0:
        ax.text(imp + 1, i, f'+{imp:.1f}%', va='center', fontsize=10, fontweight='bold')
    else:
        ax.text(imp - 1, i, f'{imp:.1f}%', va='center', ha='right', fontsize=10, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(approaches)
ax.set_xlabel('Improvement over Baseline (%)', fontsize=12)
ax.set_title('Bio-Inspired FF Variants: Most Attempts Failed or Showed Marginal Gains',
             fontsize=14, fontweight='bold')
ax.set_xlim(-90, 45)

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#2ecc71', label='Success (>5%)'),
    mpatches.Patch(facecolor='#f39c12', label='Marginal (0-5%)'),
    mpatches.Patch(facecolor='#e74c3c', label='Failed (<0%)')
]
ax.legend(handles=legend_elements, loc='lower right')

# Insight annotation
ax.annotate('Only CwC-FF shows\nsubstantial improvement\n(by removing label embedding)',
            xy=(35, 0), xytext=(20, 2),
            arrowprops=dict(arrowstyle='->', color='green'),
            fontsize=10, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('figures/insight_bio_attempts.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved figures/insight_bio_attempts.png")

# ============================================================================
# Figure 4: Efficiency vs Performance Trade-off
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 8))

# Data: (training_time_relative, accuracy, transfer, name)
models = {
    'Backprop': (1, 99.2, 75.5, '#3498db'),
    'Standard FF': (30, 94.5, 54.2, '#e74c3c'),
    'CwC-FF': (45, 98.7, 89.1, '#2ecc71'),
    'Prospective FF': (240, 91.3, 31.3, '#9b59b6'),
    'Three-Factor FF': (30, 91.1, 64.3, '#f39c12'),
}

for name, (time, acc, transfer, color) in models.items():
    # Size proportional to transfer accuracy
    size = transfer * 5
    ax.scatter(time, acc, s=size, c=color, alpha=0.7, edgecolors='black', linewidth=1.5)

    # Label
    offset = (10, 10) if name != 'Backprop' else (10, -15)
    ax.annotate(f'{name}\n({transfer:.1f}% transfer)',
                xy=(time, acc), xytext=offset, textcoords='offset points',
                fontsize=9, ha='left')

ax.set_xlabel('Training Time (relative to Backprop)', fontsize=12)
ax.set_ylabel('MNIST Accuracy (%)', fontsize=12)
ax.set_title('Efficiency vs Performance: FF Variants Are Slower AND Worse',
             fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.set_xlim(0.5, 500)
ax.set_ylim(85, 100)

# Add reference lines
ax.axhline(y=99.2, color='#3498db', linestyle='--', alpha=0.5)
ax.axvline(x=1, color='#3498db', linestyle='--', alpha=0.5)

# Quadrant labels
ax.text(0.7, 99.5, 'IDEAL\n(fast & accurate)', fontsize=10, color='green', ha='right')
ax.text(300, 99.5, 'Accurate but slow', fontsize=10, color='orange', ha='left')
ax.text(300, 86, 'Slow & inaccurate', fontsize=10, color='red', ha='left')

# Size legend
ax.text(0.98, 0.02, 'Bubble size = Transfer accuracy', transform=ax.transAxes,
        fontsize=9, ha='right', va='bottom', style='italic')

plt.tight_layout()
plt.savefig('figures/insight_efficiency_tradeoff.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved figures/insight_efficiency_tradeoff.png")

# ============================================================================
# Figure 5: The Root Cause Diagram (Flow Chart Style)
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Title
ax.text(7, 7.5, 'ROOT CAUSE: Why Standard FF Fails at Transfer Learning',
        fontsize=16, fontweight='bold', ha='center')

# Box 1: Standard FF Design
box1 = mpatches.FancyBboxPatch((0.5, 5), 3, 1.5, boxstyle="round,pad=0.1",
                                facecolor='#e8f4f8', edgecolor='#3498db', linewidth=2)
ax.add_patch(box1)
ax.text(2, 5.75, 'Standard FF Design\nEmbed label in first 10 pixels',
        ha='center', va='center', fontsize=11)

# Arrow 1
ax.annotate('', xy=(5, 5.75), xytext=(3.5, 5.75),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Box 2: Consequence
box2 = mpatches.FancyBboxPatch((5, 5), 4, 1.5, boxstyle="round,pad=0.1",
                                facecolor='#fff3cd', edgecolor='#ffc107', linewidth=2)
ax.add_patch(box2)
ax.text(7, 5.75, 'Features = f(image, SOURCE_LABEL)\nFeatures are task-specific',
        ha='center', va='center', fontsize=11)

# Arrow 2
ax.annotate('', xy=(10.5, 5.75), xytext=(9, 5.75),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Box 3: Result
box3 = mpatches.FancyBboxPatch((10.5, 5), 3, 1.5, boxstyle="round,pad=0.1",
                                facecolor='#f8d7da', edgecolor='#e74c3c', linewidth=2)
ax.add_patch(box3)
ax.text(12, 5.75, 'Transfer FAILS\n54% < Random 72%',
        ha='center', va='center', fontsize=11, color='#c0392b', fontweight='bold')

# Box 4: CwC-FF Solution (below)
box4 = mpatches.FancyBboxPatch((0.5, 2), 3, 1.5, boxstyle="round,pad=0.1",
                                facecolor='#d4edda', edgecolor='#28a745', linewidth=2)
ax.add_patch(box4)
ax.text(2, 2.75, 'CwC-FF Design\nNo label embedding',
        ha='center', va='center', fontsize=11)

# Arrow 4
ax.annotate('', xy=(5, 2.75), xytext=(3.5, 2.75),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Box 5: CwC Consequence
box5 = mpatches.FancyBboxPatch((5, 2), 4, 1.5, boxstyle="round,pad=0.1",
                                facecolor='#d4edda', edgecolor='#28a745', linewidth=2)
ax.add_patch(box5)
ax.text(7, 2.75, 'Features = f(image)\nFeatures are task-agnostic',
        ha='center', va='center', fontsize=11)

# Arrow 5
ax.annotate('', xy=(10.5, 2.75), xytext=(9, 2.75),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# Box 6: CwC Result
box6 = mpatches.FancyBboxPatch((10.5, 2), 3, 1.5, boxstyle="round,pad=0.1",
                                facecolor='#d4edda', edgecolor='#28a745', linewidth=2)
ax.add_patch(box6)
ax.text(12, 2.75, 'Transfer WORKS\n89% > Random 72%',
        ha='center', va='center', fontsize=11, color='#155724', fontweight='bold')

# Conclusion box
conclusion = mpatches.FancyBboxPatch((3, 0.2), 8, 1.2, boxstyle="round,pad=0.1",
                                      facecolor='#f8f9fa', edgecolor='#343a40', linewidth=2)
ax.add_patch(conclusion)
ax.text(7, 0.8, 'CONCLUSION: The label embedding design is the root cause.\n'
        'Removing it (CwC-FF) enables transfer, but fundamentally changes FF.',
        ha='center', va='center', fontsize=11, fontweight='bold')

plt.savefig('figures/insight_root_cause.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved figures/insight_root_cause.png")

# ============================================================================
# Figure 6: What We Learned (Summary Infographic)
# ============================================================================

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(7, 9.5, 'WHAT WE LEARNED: A Systematic Study of Forward-Forward',
        fontsize=16, fontweight='bold', ha='center')

# Section 1: What Works
ax.text(2.5, 8.5, 'WHAT WORKS', fontsize=14, fontweight='bold', color='#27ae60', ha='center')
works_box = mpatches.FancyBboxPatch((0.3, 6.5), 4.4, 1.8, boxstyle="round,pad=0.1",
                                     facecolor='#d4edda', edgecolor='#28a745', linewidth=2)
ax.add_patch(works_box)
ax.text(2.5, 7.8, '✓ Hinton\'s wrong-label (94.5%)', fontsize=10, ha='center')
ax.text(2.5, 7.4, '✓ CwC-FF transfer (89%)', fontsize=10, ha='center')
ax.text(2.5, 7.0, '✓ Layer Collab γ=0.7 (+1.2%)', fontsize=10, ha='center')

# Section 2: Marginal
ax.text(7, 8.5, 'MARGINAL GAINS', fontsize=14, fontweight='bold', color='#f39c12', ha='center')
marginal_box = mpatches.FancyBboxPatch((4.8, 6.5), 4.4, 1.8, boxstyle="round,pad=0.1",
                                        facecolor='#fff3cd', edgecolor='#ffc107', linewidth=2)
ax.add_patch(marginal_box)
ax.text(7, 7.8, '~ Three-Factor top-down (+1.5%)', fontsize=10, ha='center')
ax.text(7, 7.4, '~ Prospective FF (+1.8% MNIST)', fontsize=10, ha='center')
ax.text(7, 7.0, '~ Complex negative strategies', fontsize=10, ha='center')

# Section 3: Failed
ax.text(11.5, 8.5, 'FAILED', fontsize=14, fontweight='bold', color='#e74c3c', ha='center')
failed_box = mpatches.FancyBboxPatch((9.3, 6.5), 4.4, 1.8, boxstyle="round,pad=0.1",
                                      facecolor='#f8d7da', edgecolor='#dc3545', linewidth=2)
ax.add_patch(failed_box)
ax.text(11.5, 7.8, '✗ PCL-FF (100% neuron death)', fontsize=10, ha='center')
ax.text(11.5, 7.4, '✗ Reward prediction (collapse)', fontsize=10, ha='center')
ax.text(11.5, 7.0, '✗ Standard FF transfer (54%)', fontsize=10, ha='center')

# Key Numbers
ax.text(7, 5.8, 'KEY NUMBERS', fontsize=14, fontweight='bold', ha='center')

numbers = [
    ('6', 'negative strategies tested'),
    ('5', 'bio-inspired variants'),
    ('94.5%', 'best standard FF (MNIST)'),
    ('54%', 'FF transfer (worse than random!)'),
    ('89%', 'CwC-FF transfer (best)'),
    ('30-240×', 'slower than backprop'),
]

for i, (num, desc) in enumerate(numbers):
    col = i % 3
    row = i // 3
    x = 2.5 + col * 4.5
    y = 5.0 - row * 0.8
    ax.text(x, y, num, fontsize=16, fontweight='bold', ha='center', color='#2c3e50')
    ax.text(x, y - 0.3, desc, fontsize=9, ha='center', color='#7f8c8d')

# Core Insight
insight_box = mpatches.FancyBboxPatch((1, 2.2), 12, 1.3, boxstyle="round,pad=0.1",
                                       facecolor='#e8f4f8', edgecolor='#3498db', linewidth=3)
ax.add_patch(insight_box)
ax.text(7, 3.1, 'CORE INSIGHT', fontsize=12, fontweight='bold', ha='center', color='#2980b9')
ax.text(7, 2.6, 'FF\'s label embedding design creates task-specific features that cannot transfer.\n'
        'CwC-FF solves this by removing label embedding—but this fundamentally changes FF.',
        fontsize=11, ha='center')

# Conclusion
ax.text(7, 1.5, 'CONCLUSION', fontsize=12, fontweight='bold', ha='center')
ax.text(7, 0.8, 'FF is a beautiful idea, but it remains a research curiosity—not a practical\n'
        'alternative to backpropagation. The path forward: label-free, efficient variants.',
        fontsize=11, ha='center', style='italic')

plt.savefig('figures/insight_summary.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved figures/insight_summary.png")

print("\n" + "="*60)
print("All insight visualizations created!")
print("="*60)
