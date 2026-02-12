"""
Create a clearer figure explaining why label embedding breaks transfer learning.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Create figure with better layout
fig = plt.figure(figsize=(16, 10))

# ============================================================
# Panel 1: What is Label Embedding?
# ============================================================
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_title("1. What is Label Embedding?", fontsize=14, fontweight='bold')

# Draw a 28x28 grid representing MNIST image
img_data = np.zeros((28, 28))
# Draw a simple "1" digit
img_data[5:23, 13:16] = 0.8
img_data[5:8, 10:14] = 0.8

ax1.imshow(img_data, cmap='gray_r', extent=[0, 28, 0, 28])

# Highlight first 10 pixels (label embedding area)
rect = patches.Rectangle((0, 27), 10, 1, linewidth=3, edgecolor='red', facecolor='red', alpha=0.5)
ax1.add_patch(rect)

# Add labels
ax1.annotate('Label embedding\n(first 10 pixels)', xy=(5, 27.5), xytext=(12, 29),
            fontsize=11, color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

ax1.annotate('Image pixels\n(remaining 774)', xy=(14, 14), xytext=(20, 8),
            fontsize=10, color='blue',
            arrowprops=dict(arrowstyle='->', color='blue'))

ax1.set_xlim(-1, 32)
ax1.set_ylim(-1, 32)
ax1.set_aspect('equal')
ax1.axis('off')

# Add code snippet
code_text = '''# Standard FF input construction:
x[:, 0:10] = 0          # Clear first 10 pixels
x[:, label] = x.max()   # Set pixel[label] to max

# For digit "1":
# pixel[1] = bright, others = 0
# Input = [0, ■, 0, 0, 0, 0, 0, 0, 0, 0, image...]'''

ax1.text(0, -3, code_text, fontsize=9, family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# ============================================================
# Panel 2: The Collision Problem
# ============================================================
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title("2. Why This Breaks Transfer", fontsize=14, fontweight='bold')
ax2.axis('off')

# MNIST side
ax2.text(0.15, 0.85, "MNIST", fontsize=14, fontweight='bold', ha='center',
        transform=ax2.transAxes)
ax2.text(0.15, 0.75, "Label 0 = digit zero", fontsize=11, ha='center',
        transform=ax2.transAxes)
ax2.text(0.15, 0.68, "Label 1 = digit one", fontsize=11, ha='center',
        transform=ax2.transAxes)
ax2.text(0.15, 0.61, "...", fontsize=11, ha='center',
        transform=ax2.transAxes)

# Draw MNIST digit 0
mnist_ax = fig.add_axes([0.42, 0.72, 0.06, 0.08])
digit0 = np.zeros((28, 28))
# Simple circle for 0
for i in range(28):
    for j in range(28):
        dist = np.sqrt((i-14)**2 + (j-14)**2)
        if 6 < dist < 10:
            digit0[i, j] = 0.8
mnist_ax.imshow(digit0, cmap='gray_r')
mnist_ax.axis('off')
mnist_ax.set_title('0', fontsize=10)

# Arrow
ax2.annotate('', xy=(0.65, 0.72), xytext=(0.35, 0.72),
            transform=ax2.transAxes,
            arrowprops=dict(arrowstyle='->', color='red', lw=3))
ax2.text(0.5, 0.76, "Transfer", fontsize=11, ha='center', transform=ax2.transAxes, color='red')
ax2.text(0.5, 0.68, "FAILS!", fontsize=12, ha='center', transform=ax2.transAxes,
        color='red', fontweight='bold')

# Fashion-MNIST side
ax2.text(0.85, 0.85, "Fashion-MNIST", fontsize=14, fontweight='bold', ha='center',
        transform=ax2.transAxes)
ax2.text(0.85, 0.75, "Label 0 = T-shirt", fontsize=11, ha='center',
        transform=ax2.transAxes)
ax2.text(0.85, 0.68, "Label 1 = Trouser", fontsize=11, ha='center',
        transform=ax2.transAxes)
ax2.text(0.85, 0.61, "...", fontsize=11, ha='center',
        transform=ax2.transAxes)

# Draw T-shirt
fashion_ax = fig.add_axes([0.88, 0.72, 0.06, 0.08])
tshirt = np.zeros((28, 28))
# Simple T-shirt shape
tshirt[4:8, 8:20] = 0.8  # collar
tshirt[8:22, 10:18] = 0.8  # body
tshirt[8:12, 4:10] = 0.8  # left sleeve
tshirt[8:12, 18:24] = 0.8  # right sleeve
fashion_ax.imshow(tshirt, cmap='gray_r')
fashion_ax.axis('off')
fashion_ax.set_title('0', fontsize=10)

# The problem explanation
problem_box = '''THE PROBLEM:

Network learned: "When pixel[0] is bright → activate pattern for digit zero"

But in Fashion-MNIST:
  pixel[0] bright → should mean T-shirt, not zero!

The learned features are COUPLED to source task labels.
They don't represent visual features - they represent "what label was embedded"
'''
ax2.text(0.5, 0.25, problem_box, fontsize=10, ha='center', va='top',
        transform=ax2.transAxes,
        bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.9))

# ============================================================
# Panel 3: What Features Actually Learn
# ============================================================
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_title("3. What Features Actually Learn", fontsize=14, fontweight='bold')
ax3.axis('off')

# Standard FF features
ax3.text(0.25, 0.9, "Standard FF Features", fontsize=12, fontweight='bold',
        ha='center', transform=ax3.transAxes, color='red')

ff_features = '''Layer 1 weights learn:
  • "If pixel[0] bright → this might be class 0"
  • "If pixel[1] bright → this might be class 1"
  • ...mostly detecting WHICH LABEL was embedded

Layer 2 weights learn:
  • Patterns that work WITH those label detectors
  • NOT general visual features (edges, curves)

Result: Features = f(image, LABEL)
        Useless when labels change meaning!'''

ax3.text(0.25, 0.7, ff_features, fontsize=9, ha='center', va='top',
        transform=ax3.transAxes, family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# CwC-FF features
ax3.text(0.75, 0.9, "CwC-FF Features", fontsize=12, fontweight='bold',
        ha='center', transform=ax3.transAxes, color='green')

cwc_features = '''NO label in input at all!

Layer 1 weights learn:
  • Edge detectors
  • Corner detectors
  • Texture patterns

Layer 2 weights learn:
  • Combinations of edges → shapes
  • General visual features

Result: Features = f(image)
        Transfer beautifully!'''

ax3.text(0.75, 0.7, cwc_features, fontsize=9, ha='center', va='top',
        transform=ax3.transAxes, family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# ============================================================
# Panel 4: The Numbers
# ============================================================
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_title("4. Transfer Learning Results", fontsize=14, fontweight='bold')
ax4.axis('off')

# Create a simple bar chart comparison
methods = ['Random Init\n(baseline)', 'Standard FF\n(label in input)', 'CwC-FF\n(no label)']
accuracies = [71.89, 54.19, 89.05]
colors = ['gray', 'red', 'green']

bar_positions = [0.2, 0.5, 0.8]
bar_width = 0.15

for i, (pos, acc, color, method) in enumerate(zip(bar_positions, accuracies, colors, methods)):
    # Draw bar
    rect = patches.FancyBboxPatch((pos - bar_width/2, 0.15), bar_width, acc/100 * 0.6,
                                   boxstyle="round,pad=0.01", facecolor=color, alpha=0.7,
                                   transform=ax4.transAxes)
    ax4.add_patch(rect)

    # Label
    ax4.text(pos, 0.08, method, fontsize=10, ha='center', transform=ax4.transAxes)
    ax4.text(pos, 0.15 + acc/100 * 0.6 + 0.02, f'{acc}%', fontsize=11, ha='center',
            fontweight='bold', transform=ax4.transAxes)

# Add insight
insight = '''KEY INSIGHT:
Standard FF (54%) is WORSE than random (72%)!
The label embedding actively HURTS transfer.

CwC-FF (89%) proves: remove label embedding → problem solved.'''

ax4.text(0.5, 0.92, insight, fontsize=10, ha='center', va='top',
        transform=ax4.transAxes,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('/Users/parafee41/Desktop/Rios/ff-research/figures/label_embedding_explained.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Created: figures/label_embedding_explained.png")
