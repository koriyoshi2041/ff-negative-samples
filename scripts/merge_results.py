"""
合并所有策略实验结果，生成最终报告
"""
import json
from pathlib import Path

def main():
    results_dir = Path(__file__).parent.parent / 'results'
    
    # Load existing results
    with open(results_dir / 'strategy_comparison.json') as f:
        existing = json.load(f)
    
    # Load remaining results if exists
    remaining_path = results_dir / 'remaining_strategies.json'
    if remaining_path.exists():
        with open(remaining_path) as f:
            remaining = json.load(f)
    else:
        remaining = {}
    
    # Merge results
    final_results = existing.get('results', {})
    
    for name, data in remaining.items():
        final_results[name] = {
            'mean_accuracy': data['mean_accuracy'],
            'std_accuracy': data['std_accuracy'],
            'mean_time': data['mean_time'],
            'mean_convergence_epoch': data['mean_convergence_epoch'],
            'accuracies': data['accuracies'],
            'uses_label_embedding': name in ['mono_forward'],
            'description': get_description(name),
            'status': 'complete'
        }
    
    # Create final output
    output = {
        'experiment_config': existing.get('experiment_config', {}),
        'results': final_results,
        'summary': generate_summary(final_results),
        'key_findings': generate_findings(final_results),
    }
    
    # Save
    output_path = results_dir / 'strategy_comparison_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {output_path}")
    
    # Print summary
    print("\n=== Strategy Comparison Results ===\n")
    sorted_results = sorted(
        [(k, v) for k, v in final_results.items() if v.get('mean_accuracy') is not None],
        key=lambda x: x[1]['mean_accuracy'],
        reverse=True
    )
    
    for rank, (name, data) in enumerate(sorted_results, 1):
        acc = data['mean_accuracy'] * 100
        uses_label = '✓' if data.get('uses_label_embedding') else '✗'
        print(f"{rank:2}. {name:20} | {acc:5.2f}% | Label Embed: {uses_label}")

def get_description(name):
    descriptions = {
        'masking': 'Random pixel masking as negative',
        'layer_wise': 'Layer-specific adaptive negatives',
        'adversarial': 'Gradient-based perturbation',
        'hard_mining': 'Select hardest negatives from pool',
        'mono_forward': 'No negatives variant (positive only)',
    }
    return descriptions.get(name, '')

def generate_summary(results):
    complete = [k for k, v in results.items() if v.get('mean_accuracy') is not None]
    label_embed = [k for k in complete if results[k].get('uses_label_embedding')]
    non_label = [k for k in complete if not results[k].get('uses_label_embedding')]
    
    return {
        'total_strategies': len(results),
        'completed_strategies': len(complete),
        'label_embedding_strategies': label_embed,
        'non_label_strategies': non_label,
    }

def generate_findings(results):
    complete = [(k, v) for k, v in results.items() if v.get('mean_accuracy') is not None]
    
    if not complete:
        return {}
    
    sorted_by_acc = sorted(complete, key=lambda x: x[1]['mean_accuracy'], reverse=True)
    best_name, best_data = sorted_by_acc[0]
    worst_name, worst_data = sorted_by_acc[-1]
    
    # Categorize
    label_embed_accs = [v['mean_accuracy'] for k, v in complete if v.get('uses_label_embedding')]
    non_label_accs = [v['mean_accuracy'] for k, v in complete if not v.get('uses_label_embedding')]
    
    return {
        'best_strategy': best_name,
        'best_accuracy': best_data['mean_accuracy'],
        'worst_strategy': worst_name,
        'worst_accuracy': worst_data['mean_accuracy'],
        'label_embedding_mean': sum(label_embed_accs) / len(label_embed_accs) if label_embed_accs else 0,
        'non_label_mean': sum(non_label_accs) / len(non_label_accs) if non_label_accs else 0,
        'critical_finding': 'Label embedding is essential for FF evaluation method',
    }

if __name__ == '__main__':
    main()
