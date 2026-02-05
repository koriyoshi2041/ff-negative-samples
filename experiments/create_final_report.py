"""
生成最终的策略对比实验报告（英文版，避免字体问题）
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def main():
    # 加载现有结果
    results_path = Path(__file__).parent.parent / 'results' / 'strategy_comparison.json'
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    output_dir = Path(__file__).parent.parent / 'results'
    
    # 过滤已完成的策略
    completed = {k: v for k, v in results.items() 
                 if v.get('mean_accuracy') is not None}
    
    print(f"Completed strategies: {len(completed)}/10")
    
    # ============ 生成可视化 ============
    sorted_names = sorted(completed.keys(), 
                         key=lambda x: completed[x]['mean_accuracy'], 
                         reverse=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Negative Sample Strategy Comparison Experiment', 
                 fontsize=14, fontweight='bold')
    
    # 1. 准确率对比
    ax1 = axes[0, 0]
    accuracies = [completed[n]['mean_accuracy'] * 100 for n in sorted_names]
    colors = ['#2ecc71' if acc > 30 else '#e74c3c' for acc in accuracies]
    
    bars = ax1.barh(sorted_names, accuracies, color=colors)
    ax1.set_xlabel('Accuracy (%)')
    ax1.set_title('Final Accuracy Comparison')
    ax1.set_xlim(0, 50)
    ax1.axvline(x=10, color='gray', linestyle='--', alpha=0.5, label='Random (10%)')
    
    for bar, acc in zip(bars, accuracies):
        ax1.text(acc + 1, bar.get_y() + bar.get_height()/2, 
                f'{acc:.1f}%', va='center', fontsize=10)
    
    # 2. 训练曲线
    ax2 = axes[0, 1]
    for name in sorted_names:
        if 'accuracies' in completed[name]:
            accs = completed[name]['accuracies']
            ax2.plot(range(1, len(accs)+1), [a*100 for a in accs], 
                    label=name, marker='o', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Curves')
    ax2.legend(fontsize=8, loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # 3. 标签嵌入 vs 非标签嵌入
    ax3 = axes[1, 0]
    label_emb = [n for n in completed if completed[n].get('uses_label_embedding', False)]
    non_label = [n for n in completed if not completed[n].get('uses_label_embedding', True)]
    
    label_emb_accs = [completed[n]['mean_accuracy']*100 for n in label_emb]
    non_label_accs = [completed[n]['mean_accuracy']*100 for n in non_label]
    
    x = np.arange(2)
    width = 0.6
    
    bars3 = ax3.bar(x, [np.mean(label_emb_accs) if label_emb_accs else 0, 
                        np.mean(non_label_accs) if non_label_accs else 0], 
                   width, color=['#2ecc71', '#e74c3c'])
    ax3.set_xticks(x)
    ax3.set_xticklabels(['With Label Embedding', 'Without Label Embedding'])
    ax3.set_ylabel('Average Accuracy (%)')
    ax3.set_title('Label Embedding Impact')
    ax3.set_ylim(0, 50)
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=12)
    
    # 4. 关键发现文字
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    findings = """
    KEY FINDINGS
    ============
    
    1. Label embedding is CRITICAL
       - With label embedding: ~38.8% accuracy
       - Without: ~9.8% (random chance)
       - Gap: 29 percentage points!
    
    2. Why non-label strategies fail:
       - FF evaluation requires label embedding
       - Network cannot associate images with classes
       - Without labels, all classes look equally "good"
    
    3. Best strategies:
       - LabelEmbedding (Hinton's original)
       - ClassConfusion (same mechanism)
       Both achieve 38.81% on MNIST
    
    4. Remaining strategies (in progress):
       - SelfContrastive (SCFF)
       - Masking, LayerWise, Adversarial
       - HardMining, MonoForward
    """
    ax4.text(0.1, 0.9, findings, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    output_path = output_dir / 'strategy_comparison_final_en.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    # ============ 生成 JSON 摘要 ============
    summary = {
        "experiment": "Negative Sample Strategy Comparison",
        "date": datetime.now().strftime('%Y-%m-%d'),
        "status": "partial_results",
        "completed": len(completed),
        "total": 10,
        "config": {
            "dataset": "MNIST",
            "architecture": "784 -> 500 -> 500",
            "optimizer": "Adam (lr=0.03)",
            "epochs": 10,
            "batch_size": 64
        },
        "results": {
            name: {
                "accuracy": f"{d['mean_accuracy']*100:.2f}%",
                "uses_label_embedding": d.get('uses_label_embedding', False),
                "training_time": f"{d.get('mean_time', 0):.1f}s"
            }
            for name, d in sorted(completed.items(), 
                                 key=lambda x: x[1]['mean_accuracy'], 
                                 reverse=True)
        },
        "key_findings": {
            "label_embedding_avg_accuracy": f"{np.mean(label_emb_accs):.1f}%" if label_emb_accs else "N/A",
            "non_label_avg_accuracy": f"{np.mean(non_label_accs):.1f}%" if non_label_accs else "N/A",
            "accuracy_gap": f"{np.mean(label_emb_accs) - np.mean(non_label_accs):.1f}pp" if label_emb_accs and non_label_accs else "N/A",
            "conclusion": "Label embedding is essential for FF classification. Strategies without it achieve only random-level accuracy."
        }
    }
    
    summary_path = output_dir / 'strategy_comparison_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")
    
    # ============ 打印摘要 ============
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Status: {len(completed)}/10 strategies completed")
    print("\nRanking:")
    for i, (name, d) in enumerate(sorted(completed.items(), 
                                         key=lambda x: x[1]['mean_accuracy'], 
                                         reverse=True), 1):
        label = "✓" if d.get('uses_label_embedding', False) else "✗"
        print(f"  {i}. {name}: {d['mean_accuracy']*100:.2f}% [Label: {label}]")
    
    print(f"\nKey Finding: Label embedding strategies average {np.mean(label_emb_accs):.1f}%")
    print(f"Non-label strategies average only {np.mean(non_label_accs):.1f}% (random chance)")
    print("="*60)

if __name__ == '__main__':
    main()
