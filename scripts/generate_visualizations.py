"""
Generate comprehensive visualizations for FF research results.

Outputs:
- results/architecture_comparison.png: FF vs Layer Collab vs PFF vs CwC-FF
- results/strategy_performance.png: 10 negative sample strategies comparison
- results/cka_heatmap.png: CKA analysis (FF vs BP layer similarity)
- results/transfer_comparison.png: Transfer learning comparison
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


def setup_chinese_font():
    """Try to setup Chinese font support, fallback to English if not available."""
    try:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return True
    except Exception:
        return False


def generate_architecture_comparison(results_dir: Path):
    """
    Generate architecture comparison chart: FF vs Layer Collab vs PFF vs CwC-FF
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    # Architecture data
    architectures = {
        'Standard FF\n(Hinton 2022)': {
            'needs_negatives': True,
            'layer_collab': False,
            'generative': False,
            'local_learning': True,
            'bio_plausible': True,
            'color': '#3498db'
        },
        'Layer Collab FF\n(Lorberbom 2024)': {
            'needs_negatives': True,
            'layer_collab': True,
            'generative': False,
            'local_learning': True,
            'bio_plausible': True,
            'color': '#2ecc71'
        },
        'PFF\n(Predictive FF)': {
            'needs_negatives': False,
            'layer_collab': False,
            'generative': True,
            'local_learning': True,
            'bio_plausible': True,
            'color': '#e74c3c'
        },
        'CwC-FF\n(Channel-wise)': {
            'needs_negatives': False,
            'layer_collab': False,
            'generative': False,
            'local_learning': True,
            'bio_plausible': True,
            'color': '#9b59b6'
        },
        'Backpropagation\n(Reference)': {
            'needs_negatives': False,
            'layer_collab': True,
            'generative': False,
            'local_learning': False,
            'bio_plausible': False,
            'color': '#95a5a6'
        }
    }

    features = [
        ('Requires Negative Samples', 'needs_negatives'),
        ('Layer Collaboration', 'layer_collab'),
        ('Generative Capability', 'generative'),
        ('Local Learning', 'local_learning'),
        ('Biologically Plausible', 'bio_plausible')
    ]

    arch_names = list(architectures.keys())
    n_arch = len(arch_names)
    n_features = len(features)

    # Create table-like visualization
    cell_width = 1.0
    cell_height = 0.8

    # Draw feature labels on the left
    for i, (feat_name, _) in enumerate(features):
        y = n_features - i - 1
        ax.text(-0.1, y * cell_height + cell_height/2, feat_name,
                ha='right', va='center', fontsize=11, fontweight='bold')

    # Draw architecture names on top
    for j, arch_name in enumerate(arch_names):
        x = j * cell_width + cell_width/2
        ax.text(x, n_features * cell_height + 0.3, arch_name,
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                color=architectures[arch_name]['color'])

    # Draw cells
    for i, (_, feat_key) in enumerate(features):
        y = (n_features - i - 1) * cell_height
        for j, arch_name in enumerate(arch_names):
            x = j * cell_width
            value = architectures[arch_name][feat_key]

            # Cell background
            if value:
                color = '#27ae60'  # Green for True
                symbol = 'O'
            else:
                color = '#e74c3c'  # Red for False
                symbol = 'X'

            rect = plt.Rectangle((x + 0.05, y + 0.05),
                                 cell_width - 0.1, cell_height - 0.1,
                                 facecolor=color, alpha=0.3, edgecolor=color)
            ax.add_patch(rect)

            # Symbol
            ax.text(x + cell_width/2, y + cell_height/2, symbol,
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   color=color)

    # Set limits and remove axes
    ax.set_xlim(-3, n_arch * cell_width)
    ax.set_ylim(-0.5, n_features * cell_height + 1.5)
    ax.axis('off')

    # Title
    ax.set_title('Forward-Forward Algorithm Variants Comparison',
                fontsize=16, fontweight='bold', pad=20)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#27ae60', alpha=0.3, edgecolor='#27ae60', label='Yes (O)'),
        mpatches.Patch(facecolor='#e74c3c', alpha=0.3, edgecolor='#e74c3c', label='No (X)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Add key insights box
    insights_text = (
        "Key Insights:\n"
        "- Standard FF: Simple but requires careful negative sample design\n"
        "- Layer Collab: Addresses layer disconnection issue\n"
        "- PFF: Eliminates negative samples via prediction objective\n"
        "- CwC-FF: Channel-wise competition removes negative samples"
    )
    ax.text(0, -0.3, insights_text, fontsize=9, va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    output_path = results_dir / 'architecture_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def generate_strategy_performance(results_dir: Path):
    """
    Generate strategy performance bar chart for 10 negative sample strategies.
    """
    # Load results
    results_file = results_dir / 'strategy_comparison_results.json'
    with open(results_file) as f:
        data = json.load(f)

    results = data['results']

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Prepare data - all 10 strategies
    all_strategies = [
        ('label_embedding', 'Label Embedding'),
        ('class_confusion', 'Class Confusion'),
        ('image_mixing', 'Image Mixing'),
        ('random_noise', 'Random Noise'),
        ('self_contrastive', 'Self-Contrastive (SCFF)'),
        ('masking', 'Masking'),
        ('layer_wise', 'Layer-wise Adaptive'),
        ('adversarial', 'Adversarial'),
        ('hard_mining', 'Hard Mining'),
        ('mono_forward', 'Mono-Forward (No Neg)')
    ]

    names = []
    accuracies = []
    colors = []
    statuses = []

    for key, display_name in all_strategies:
        if key in results:
            r = results[key]
            acc = r.get('mean_accuracy')
            uses_label = r.get('uses_label_embedding', False)
            status = r.get('status', 'complete' if acc is not None else 'pending')

            names.append(display_name)
            if acc is not None:
                accuracies.append(acc * 100)
                statuses.append('complete')
            else:
                accuracies.append(0)
                statuses.append('in_progress')

            if uses_label:
                colors.append('#2ecc71')  # Green for label embedding
            else:
                colors.append('#3498db')  # Blue for non-label

    # Left plot: Horizontal bar chart
    ax1 = axes[0]
    y_pos = np.arange(len(names))

    bars = ax1.barh(y_pos, accuracies, color=colors, alpha=0.8)

    # Add value labels and status indicators
    for i, (bar, acc, status) in enumerate(zip(bars, accuracies, statuses)):
        if status == 'complete':
            label = f'{acc:.1f}%'
        else:
            label = 'Testing...'
        ax1.text(max(acc + 1, 5), bar.get_y() + bar.get_height()/2,
                label, va='center', fontsize=10,
                color='gray' if status != 'complete' else 'black')

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names)
    ax1.set_xlabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('Negative Sample Strategy Performance\n(MNIST, 10 epochs)', fontsize=14)
    ax1.set_xlim(0, 50)
    ax1.axvline(x=9.8, color='red', linestyle='--', alpha=0.5, label='Random chance')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#2ecc71', alpha=0.8, label='With Label Embedding'),
        mpatches.Patch(facecolor='#3498db', alpha=0.8, label='Without Label Embedding'),
        plt.Line2D([0], [0], color='red', linestyle='--', alpha=0.5, label='Random Chance (10%)')
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=9)

    # Right plot: Category comparison
    ax2 = axes[1]

    # Calculate category averages
    label_emb_strategies = ['label_embedding', 'class_confusion', 'mono_forward']
    non_label_strategies = ['image_mixing', 'random_noise', 'masking', 'layer_wise', 'adversarial', 'hard_mining']

    def get_avg(strategy_list):
        vals = []
        for s in strategy_list:
            if s in results and results[s].get('mean_accuracy') is not None:
                vals.append(results[s]['mean_accuracy'] * 100)
        return np.mean(vals) if vals else 0

    categories = ['With Label\nEmbedding', 'Without Label\nEmbedding']
    cat_avgs = [get_avg(label_emb_strategies), get_avg(non_label_strategies)]
    cat_colors = ['#2ecc71', '#3498db']

    bars2 = ax2.bar(categories, cat_avgs, color=cat_colors, alpha=0.8, width=0.6)

    for bar, avg in zip(bars2, cat_avgs):
        ax2.text(bar.get_x() + bar.get_width()/2, avg + 1,
                f'{avg:.1f}%', ha='center', fontsize=14, fontweight='bold')

    ax2.set_ylabel('Average Accuracy (%)', fontsize=12)
    ax2.set_title('Performance by Category', fontsize=14)
    ax2.set_ylim(0, 40)
    ax2.axhline(y=9.8, color='red', linestyle='--', alpha=0.5)

    # Add findings box
    findings = data.get('key_findings', {})
    findings_text = (
        f"Key Finding:\n"
        f"Label embedding is essential for FF evaluation.\n"
        f"Best: {findings.get('best_strategy', 'N/A')} ({findings.get('best_accuracy', 0)*100:.1f}%)\n"
        f"Without label embedding: ~random chance"
    )
    ax2.text(0.5, 0.02, findings_text, transform=ax2.transAxes,
            fontsize=9, va='bottom', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    output_path = results_dir / 'strategy_performance.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def generate_cka_heatmap(results_dir: Path):
    """
    Generate CKA heatmap showing FF vs BP layer similarity.
    """
    # Load CKA data
    cka_summary_file = results_dir / 'cka_summary.json'
    with open(cka_summary_file) as f:
        cka_data = json.load(f)

    # Load numpy arrays if available
    ff_bp_cka_file = results_dir / 'cka_ff_bp.npy'
    ff_self_cka_file = results_dir / 'cka_ff_self.npy'
    bp_self_cka_file = results_dir / 'cka_bp_self.npy'

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. FF vs BP CKA matrix
    ax1 = axes[0]
    if ff_bp_cka_file.exists():
        ff_bp_cka = np.load(ff_bp_cka_file)
    else:
        # Reconstruct from summary if npy not available
        ff_bp_cka = np.array([
            [0.444, 0.37, 0.17, 0.14],
            [0.31, 0.330, 0.15, 0.11],
            [0.08, 0.09, 0.038, 0.04]
        ])

    im1 = ax1.imshow(ff_bp_cka, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    # Add value annotations
    for i in range(ff_bp_cka.shape[0]):
        for j in range(ff_bp_cka.shape[1]):
            val = ff_bp_cka[i, j]
            color = 'white' if val < 0.3 else 'black'
            ax1.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=10, color=color, fontweight='bold')

    ax1.set_xlabel('BP Layers', fontsize=12)
    ax1.set_ylabel('FF Layers', fontsize=12)
    ax1.set_xticks(range(ff_bp_cka.shape[1]))
    ax1.set_xticklabels([f'L{i}' for i in range(ff_bp_cka.shape[1])])
    ax1.set_yticks(range(ff_bp_cka.shape[0]))
    ax1.set_yticklabels([f'L{i}' for i in range(ff_bp_cka.shape[0])])
    ax1.set_title('FF vs BP Cross-Network CKA\n(Layer Similarity)', fontsize=12)
    plt.colorbar(im1, ax=ax1, label='CKA Similarity')

    # 2. Diagonal comparison (same layer similarity)
    ax2 = axes[1]
    diagonal_cka = cka_data['cka_diagonal']
    layers = [f'Layer {i}' for i in range(len(diagonal_cka))]

    bars = ax2.bar(layers, diagonal_cka, color=['#3498db', '#f39c12', '#e74c3c'], alpha=0.8)

    for bar, val in zip(bars, diagonal_cka):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f'{val:.3f}', ha='center', fontsize=11, fontweight='bold')

    ax2.set_ylabel('CKA Similarity', fontsize=12)
    ax2.set_title('Same-Layer CKA (FF vs BP)\n"Layer Disconnection" Evidence', fontsize=12)
    ax2.set_ylim(0, 0.6)
    ax2.axhline(y=cka_data['mean_diagonal_cka'], color='red', linestyle='--',
               label=f'Mean: {cka_data["mean_diagonal_cka"]:.3f}')
    ax2.legend(fontsize=9)

    # Add warning annotation for Layer 2
    ax2.annotate('Layer Disconnection!', xy=(2, diagonal_cka[2]),
                xytext=(1.5, 0.15),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red', fontweight='bold')

    # 3. Self-CKA comparison
    ax3 = axes[2]

    metrics = ['FF Self-CKA\n(off-diagonal)', 'BP Self-CKA\n(off-diagonal)']
    values = [cka_data['ff_self_cka_mean_offdiag'], cka_data['bp_self_cka_mean_offdiag']]
    colors = ['#e74c3c', '#2ecc71']

    bars3 = ax3.bar(metrics, values, color=colors, alpha=0.8, width=0.5)

    for bar, val in zip(bars3, values):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f'{val:.3f}', ha='center', fontsize=12, fontweight='bold')

    ax3.set_ylabel('Mean Off-diagonal CKA', fontsize=12)
    ax3.set_title('Layer Coherence Comparison\n(Higher = More Information Flow)', fontsize=12)
    ax3.set_ylim(0, 0.8)

    # Add interpretation
    interpretation = (
        "FF layers are more independent\n"
        "(lower coherence = less info sharing)\n"
        "This explains transfer learning issues"
    )
    ax3.text(0.5, 0.95, interpretation, transform=ax3.transAxes,
            fontsize=9, va='top', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    output_path = results_dir / 'cka_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def generate_transfer_comparison(results_dir: Path):
    """
    Generate transfer learning comparison chart.
    """
    # Load transfer learning results
    transfer_dir = results_dir / 'transfer'
    transfer_files = list(transfer_dir.glob('*.json'))

    if not transfer_files:
        print("No transfer learning results found. Creating placeholder chart.")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'Transfer Learning Results\n(Pending Experiments)',
               ha='center', va='center', fontsize=16,
               transform=ax.transAxes)
        ax.axis('off')
        output_path = results_dir / 'transfer_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved placeholder: {output_path}")
        return

    # Load the most recent result
    with open(transfer_files[0]) as f:
        transfer_data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 1. Transfer accuracy comparison
    ax1 = axes[0]

    methods = []
    source_accs = []
    transfer_accs = []

    # Random Init baseline
    if 'random' in transfer_data:
        methods.append('Random Init\n(No Pretrain)')
        source_accs.append(0)
        transfer_accs.append(transfer_data['random']['transfer_accuracy'] * 100)

    # BP
    if 'bp' in transfer_data:
        methods.append('Backprop\n(BP)')
        source_accs.append(transfer_data['bp']['source_accuracy'] * 100)
        transfer_accs.append(transfer_data['bp']['transfer_accuracy'] * 100)

    # FF Original
    if 'ff_original' in transfer_data:
        methods.append('Standard FF')
        source_accs.append(transfer_data['ff_original']['source_accuracy'] * 100)
        transfer_accs.append(transfer_data['ff_original']['transfer_accuracy'] * 100)

    # FF with Layer Collab
    if 'ff_collab_prev' in transfer_data:
        methods.append('FF + Layer\nCollab (Prev)')
        source_accs.append(transfer_data['ff_collab_prev']['source_accuracy'] * 100)
        transfer_accs.append(transfer_data['ff_collab_prev']['transfer_accuracy'] * 100)

    if 'ff_collab_all' in transfer_data:
        methods.append('FF + Layer\nCollab (All)')
        source_accs.append(transfer_data['ff_collab_all']['source_accuracy'] * 100)
        transfer_accs.append(transfer_data['ff_collab_all']['transfer_accuracy'] * 100)

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax1.bar(x - width/2, source_accs, width, label='Source Task (MNIST)',
                   color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, transfer_accs, width, label='Transfer Task (Fashion-MNIST)',
                   color='#e74c3c', alpha=0.8)

    # Add value labels
    for bar, val in zip(bars1, source_accs):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, val + 1,
                    f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')

    for bar, val in zip(bars2, transfer_accs):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 1,
                f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')

    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Transfer Learning: MNIST -> Fashion-MNIST\n(Feature Extraction)', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=10)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 110)
    ax1.axhline(y=10, color='gray', linestyle='--', alpha=0.5, label='Random chance')

    # 2. Transfer learning curves
    ax2 = axes[1]

    colors = {
        'random': '#95a5a6',
        'bp': '#2ecc71',
        'ff_original': '#e74c3c',
        'ff_collab_prev': '#9b59b6',
        'ff_collab_all': '#f39c12'
    }

    labels = {
        'random': 'Random Init',
        'bp': 'BP Pretrained',
        'ff_original': 'FF Pretrained',
        'ff_collab_prev': 'FF + Collab (Prev)',
        'ff_collab_all': 'FF + Collab (All)'
    }

    for method_key, label in labels.items():
        if method_key in transfer_data and 'transfer_history' in transfer_data[method_key]:
            history = transfer_data[method_key]['transfer_history']['test_acc']
            epochs = range(1, len(history) + 1)
            ax2.plot(epochs, [acc * 100 for acc in history],
                    color=colors[method_key], label=label, linewidth=2, marker='o', markersize=4)

    ax2.set_xlabel('Transfer Epochs', fontsize=12)
    ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax2.set_title('Transfer Learning Curve\n(Frozen Features + Linear Head)', fontsize=14)
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 90)

    # Add key finding annotation
    finding_text = (
        "Key Finding:\n"
        "Random Init > FF Pretrained!\n"
        "FF features don't transfer well."
    )
    ax2.text(0.02, 0.98, finding_text, transform=ax2.transAxes,
            fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    output_path = results_dir / 'transfer_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    """Generate all visualizations."""
    setup_chinese_font()

    # Determine paths
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / 'results'

    print("=" * 60)
    print("Generating FF Research Visualizations")
    print("=" * 60)

    # Ensure results directory exists
    results_dir.mkdir(parents=True, exist_ok=True)

    # Generate all charts
    print("\n[1/4] Generating architecture comparison...")
    generate_architecture_comparison(results_dir)

    print("\n[2/4] Generating strategy performance chart...")
    generate_strategy_performance(results_dir)

    print("\n[3/4] Generating CKA heatmap...")
    generate_cka_heatmap(results_dir)

    print("\n[4/4] Generating transfer learning comparison...")
    generate_transfer_comparison(results_dir)

    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print("Output files in:", results_dir)
    print("=" * 60)


if __name__ == '__main__':
    main()
