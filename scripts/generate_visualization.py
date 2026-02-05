"""
Generate strategy comparison visualization
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    results_dir = Path(__file__).parent.parent / 'results'
    vis_dir = results_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    with open(results_dir / 'strategy_comparison_results.json') as f:
        data = json.load(f)
    
    results = data['results']
    
    # Filter completed strategies
    completed = {k: v for k, v in results.items() 
                 if v.get('mean_accuracy') is not None}
    
    # Sort by accuracy
    sorted_items = sorted(completed.items(), 
                         key=lambda x: x[1]['mean_accuracy'], 
                         reverse=True)
    
    names = [item[0] for item in sorted_items]
    accs = [item[1]['mean_accuracy'] * 100 for item in sorted_items]
    uses_label = [item[1].get('uses_label_embedding', False) for item in sorted_items]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Accuracy bar chart
    ax1 = axes[0]
    colors = ['#2ecc71' if ul else '#e74c3c' for ul in uses_label]
    bars = ax1.barh(names, accs, color=colors)
    ax1.set_xlabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Negative Sample Strategy Comparison', fontsize=14)
    ax1.set_xlim(0, 50)
    
    # Add value labels
    for bar, acc in zip(bars, accs):
        ax1.text(acc + 0.5, bar.get_y() + bar.get_height()/2,
                f'{acc:.1f}%', va='center', fontsize=10)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='With Label Embedding'),
        Patch(facecolor='#e74c3c', label='Without Label Embedding'),
    ]
    ax1.legend(handles=legend_elements, loc='lower right')
    
    # 2. Category analysis
    ax2 = axes[1]
    
    # Group by category
    categories = {
        'Label Embedding': [k for k, v in completed.items() 
                           if v.get('uses_label_embedding')],
        'Non-Label': [k for k, v in completed.items() 
                      if not v.get('uses_label_embedding')],
    }
    
    cat_accs = {}
    for cat, strats in categories.items():
        accs_list = [completed[s]['mean_accuracy'] * 100 for s in strats]
        cat_accs[cat] = {
            'mean': np.mean(accs_list) if accs_list else 0,
            'strategies': strats,
            'count': len(strats),
        }
    
    cat_names = list(cat_accs.keys())
    cat_means = [cat_accs[c]['mean'] for c in cat_names]
    cat_colors = ['#2ecc71', '#e74c3c']
    
    bars2 = ax2.bar(cat_names, cat_means, color=cat_colors)
    ax2.set_ylabel('Average Accuracy (%)', fontsize=12)
    ax2.set_title('Average Accuracy by Category', fontsize=14)
    ax2.set_ylim(0, 50)
    
    for bar, mean in zip(bars2, cat_means):
        ax2.text(bar.get_x() + bar.get_width()/2, mean + 1,
                f'{mean:.1f}%', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path = vis_dir / 'strategy_comparison_bar.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    # Also save as the main comparison image
    main_path = results_dir / 'strategy_comparison_final.png'
    plt.savefig(main_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {main_path}")
    
    plt.close()

if __name__ == '__main__':
    main()
