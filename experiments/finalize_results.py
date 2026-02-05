"""
整理负样本策略对比实验结果
生成最终的 JSON、可视化和文档
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def load_results():
    """加载实验结果"""
    results_path = Path(__file__).parent.parent / 'results' / 'strategy_comparison.json'
    with open(results_path, 'r') as f:
        return json.load(f)

def generate_visualization(data, output_dir):
    """生成可视化图表"""
    results = data['results']
    
    # 过滤已完成的策略
    completed = {k: v for k, v in results.items() 
                 if v.get('mean_accuracy') is not None}
    
    if len(completed) < 2:
        print("Not enough completed experiments for visualization")
        return
    
    # 按准确率排序
    sorted_names = sorted(completed.keys(), 
                         key=lambda x: completed[x]['mean_accuracy'], 
                         reverse=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('负样本策略对比实验结果', fontsize=14, fontweight='bold')
    
    # 1. 准确率对比
    ax1 = axes[0, 0]
    accuracies = [completed[n]['mean_accuracy'] * 100 for n in sorted_names]
    stds = [completed[n].get('std_accuracy', 0) * 100 for n in sorted_names]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_names)))[::-1]
    
    bars = ax1.barh(sorted_names, accuracies, xerr=stds, color=colors, capsize=3)
    ax1.set_xlabel('Accuracy (%)')
    ax1.set_title('最终准确率对比')
    ax1.set_xlim(0, 100)
    
    for bar, acc in zip(bars, accuracies):
        ax1.text(acc + 1, bar.get_y() + bar.get_height()/2, 
                f'{acc:.1f}%', va='center', fontsize=9)
    
    # 2. 训练曲线
    ax2 = axes[0, 1]
    for name in sorted_names[:5]:
        if 'accuracies' in completed[name]:
            accs = completed[name]['accuracies']
            ax2.plot(range(1, len(accs)+1), [a*100 for a in accs], 
                    label=name, marker='o', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('训练曲线')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. 训练时间对比
    ax3 = axes[1, 0]
    times = [completed[n].get('mean_time', 0) for n in sorted_names]
    ax3.barh(sorted_names, times, color='steelblue')
    ax3.set_xlabel('Training Time (s)')
    ax3.set_title('训练时间对比')
    
    # 4. 策略分类
    ax4 = axes[1, 1]
    categories = {
        'Label Embedding': sum(1 for n in sorted_names if completed[n].get('uses_label_embedding', False)),
        'Non-Label': sum(1 for n in sorted_names if not completed[n].get('uses_label_embedding', True)),
    }
    ax4.pie(categories.values(), labels=categories.keys(), autopct='%1.1f%%', 
            colors=['#66b3ff', '#ff9999'])
    ax4.set_title('策略类型分布')
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / 'strategy_comparison_final.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {output_path}")

def generate_report(data, output_dir):
    """生成 Markdown 报告"""
    results = data['results']
    config = data.get('experiment_config', {})
    
    # 过滤已完成的策略
    completed = {k: v for k, v in results.items() 
                 if v.get('mean_accuracy') is not None}
    pending = {k: v for k, v in results.items() 
               if v.get('mean_accuracy') is None}
    
    sorted_strategies = sorted(
        completed.items(), 
        key=lambda x: x[1]['mean_accuracy'], 
        reverse=True
    )
    
    report = []
    report.append("# 负样本策略对比实验报告")
    report.append("")
    report.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("")
    
    report.append("## 实验配置")
    report.append("")
    report.append(f"- **数据集**: {config.get('dataset', 'MNIST')}")
    report.append(f"- **网络架构**: {config.get('architecture', '784 → 500 → 500')}")
    report.append(f"- **优化器**: {config.get('optimizer', 'Adam')} (lr={config.get('learning_rate', 0.03)})")
    report.append(f"- **Batch Size**: {config.get('batch_size', 64)}")
    report.append(f"- **Epochs**: {config.get('epochs', 10)}")
    report.append(f"- **设备**: {config.get('device', 'mps')}")
    report.append("")
    
    report.append("## 实验进度")
    report.append("")
    report.append(f"- **已完成**: {len(completed)}/10 策略")
    report.append(f"- **待完成**: {len(pending)}/10 策略")
    report.append("")
    
    report.append("## 已完成策略排名")
    report.append("")
    report.append("| 排名 | 策略 | 准确率 | 训练时间 | 使用标签嵌入 |")
    report.append("|------|------|--------|----------|-------------|")
    
    for rank, (name, d) in enumerate(sorted_strategies, 1):
        acc = f"{d['mean_accuracy']*100:.2f}%"
        time_str = f"{d.get('mean_time', 0):.1f}s"
        label_emb = "✅" if d.get('uses_label_embedding', False) else "❌"
        report.append(f"| {rank} | {name} | {acc} | {time_str} | {label_emb} |")
    
    report.append("")
    
    if pending:
        report.append("## 待完成策略")
        report.append("")
        for name, d in pending.items():
            status = d.get('status', 'pending')
            desc = d.get('description', '')
            report.append(f"- **{name}** ({status}): {desc}")
        report.append("")
    
    report.append("## 关键发现")
    report.append("")
    
    if sorted_strategies:
        top_name, top_data = sorted_strategies[0]
        report.append(f"### 🥇 最佳策略: {top_name}")
        report.append("")
        report.append(f"- **准确率**: {top_data['mean_accuracy']*100:.2f}%")
        report.append(f"- **训练时间**: {top_data.get('mean_time', 0):.1f}s")
        if 'description' in top_data:
            report.append(f"- **描述**: {top_data['description']}")
        report.append("")
    
    # 分析发现
    label_emb_strategies = [n for n, d in completed.items() if d.get('uses_label_embedding', False)]
    non_label_strategies = [n for n, d in completed.items() if not d.get('uses_label_embedding', True)]
    
    report.append("### 标签嵌入的重要性")
    report.append("")
    if label_emb_strategies and non_label_strategies:
        label_emb_avg = np.mean([completed[n]['mean_accuracy'] for n in label_emb_strategies]) * 100
        non_label_avg = np.mean([completed[n]['mean_accuracy'] for n in non_label_strategies]) * 100
        report.append(f"- 使用标签嵌入的策略平均准确率: **{label_emb_avg:.1f}%**")
        report.append(f"- 不使用标签嵌入的策略平均准确率: **{non_label_avg:.1f}%**")
        report.append(f"- 差距: **{label_emb_avg - non_label_avg:.1f}** 个百分点")
        report.append("")
        report.append("> **结论**: 标签嵌入对于 Forward-Forward 算法的分类性能至关重要。")
        report.append("> 不使用标签嵌入的策略（如 image_mixing, random_noise）达到接近随机水平（~10%），")
        report.append("> 因为网络无法学习将图像与类别关联。")
    report.append("")
    
    report.append("---")
    report.append("*由 Forward-Forward Research 自动生成*")
    
    output_path = Path(output_dir) / 'strategy_comparison_report.md'
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    print(f"Saved report: {output_path}")

def main():
    print("Loading results...")
    data = load_results()
    
    output_dir = Path(__file__).parent.parent / 'results'
    
    print("\nGenerating visualization...")
    generate_visualization(data, output_dir)
    
    print("\nGenerating report...")
    generate_report(data, output_dir)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
