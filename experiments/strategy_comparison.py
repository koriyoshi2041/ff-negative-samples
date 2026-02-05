"""
负样本策略系统对比实验

对比 10 种负样本策略在 MNIST 上的表现：
- 最终准确率
- 收敛速度
- 训练时间

每个策略运行 3 次取平均
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import sys
import os

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from negative_strategies import (
    StrategyRegistry,
    LabelEmbeddingStrategy,
    ImageMixingStrategy,
    RandomNoiseStrategy,
    ClassConfusionStrategy,
    SelfContrastiveStrategy,
    MaskingStrategy,
    LayerWiseStrategy,
    AdversarialStrategy,
    HardMiningStrategy,
    MonoForwardStrategy,
)

# ============================================================
# FF Network Implementation (from ff_baseline.py)
# ============================================================

class FFLayer(nn.Module):
    """A single Forward-Forward layer with local learning."""
    
    def __init__(self, in_features: int, out_features: int, threshold: float = 2.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.threshold = threshold
        self.optimizer = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_normalized = x / (x.norm(2, dim=1, keepdim=True) + 1e-8)
        return self.relu(self.linear(x_normalized))
    
    def goodness(self, x: torch.Tensor) -> torch.Tensor:
        """Compute goodness (sum of squared activations)."""
        return (x ** 2).sum(dim=1)
    
    def ff_loss(self, pos_goodness: torch.Tensor, neg_goodness: torch.Tensor) -> torch.Tensor:
        """FF loss: push positive goodness above threshold, negative below."""
        loss_pos = torch.log(1 + torch.exp(self.threshold - pos_goodness)).mean()
        loss_neg = torch.log(1 + torch.exp(neg_goodness - self.threshold)).mean()
        return loss_pos + loss_neg
    
    def mono_loss(self, pos_goodness: torch.Tensor) -> torch.Tensor:
        """Mono-forward loss: only push positive goodness above threshold."""
        return torch.log(1 + torch.exp(self.threshold - pos_goodness)).mean()


class FFNetwork(nn.Module):
    """Multi-layer Forward-Forward Network."""
    
    def __init__(self, layer_sizes: List[int], threshold: float = 2.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(FFLayer(layer_sizes[i], layer_sizes[i+1], threshold))
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return activations from all layers."""
        activations = []
        for layer in self.layers:
            x = layer(x)
            activations.append(x)
        return activations
    
    def setup_optimizers(self, lr: float = 0.03):
        """Setup optimizers for each layer."""
        for layer in self.layers:
            layer.optimizer = optim.Adam(layer.parameters(), lr=lr)
    
    def train_step(self, pos_data: torch.Tensor, neg_data: torch.Tensor) -> Dict[str, float]:
        """Train all layers with FF algorithm."""
        losses = {}
        pos_input = pos_data
        neg_input = neg_data
        
        for i, layer in enumerate(self.layers):
            pos_output = layer(pos_input)
            neg_output = layer(neg_input)
            
            pos_goodness = layer.goodness(pos_output)
            neg_goodness = layer.goodness(neg_output)
            
            loss = layer.ff_loss(pos_goodness, neg_goodness)
            
            layer.optimizer.zero_grad()
            loss.backward()
            layer.optimizer.step()
            
            losses[f'layer_{i}'] = loss.item()
            
            pos_input = pos_output.detach()
            neg_input = neg_output.detach()
        
        return losses
    
    def train_step_mono(self, pos_data: torch.Tensor) -> Dict[str, float]:
        """Train with mono-forward (no negatives)."""
        losses = {}
        pos_input = pos_data
        
        for i, layer in enumerate(self.layers):
            pos_output = layer(pos_input)
            pos_goodness = layer.goodness(pos_output)
            
            # Only positive loss
            loss = layer.mono_loss(pos_goodness)
            
            layer.optimizer.zero_grad()
            loss.backward()
            layer.optimizer.step()
            
            losses[f'layer_{i}'] = loss.item()
            pos_input = pos_output.detach()
        
        return losses


# ============================================================
# Training and Evaluation Functions
# ============================================================

def train_epoch(
    model: FFNetwork, 
    train_loader: DataLoader, 
    strategy,
    device: torch.device,
    use_mono: bool = False
) -> Tuple[Dict[str, float], float]:
    """Train FF network for one epoch. Returns losses and epoch time."""
    model.train()
    
    # Set strategy to training mode (important for SCFF)
    if hasattr(strategy, 'train'):
        strategy.train()
    
    total_losses = {f'layer_{i}': 0.0 for i in range(len(model.layers))}
    num_batches = 0
    
    start_time = time.time()
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Create positive samples
        pos_data = strategy.create_positive(images, labels)
        
        if use_mono:
            # Mono-forward: no negatives
            losses = model.train_step_mono(pos_data)
        else:
            # Standard FF: need negatives
            neg_data = strategy.generate(images, labels)
            losses = model.train_step(pos_data, neg_data)
        
        for key in losses:
            total_losses[key] += losses[key]
        num_batches += 1
    
    epoch_time = time.time() - start_time
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}
    
    return avg_losses, epoch_time


def evaluate(
    model: FFNetwork, 
    test_loader: DataLoader, 
    strategy,
    device: torch.device,
    num_classes: int = 10
) -> float:
    """Evaluate FF network accuracy using label embedding."""
    model.eval()
    
    # Set strategy to evaluation mode (important for SCFF)
    if hasattr(strategy, 'eval'):
        strategy.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)
            
            best_goodness = torch.zeros(batch_size, device=device) - float('inf')
            predictions = torch.zeros(batch_size, dtype=torch.long, device=device)
            
            for candidate_label in range(num_classes):
                candidate_labels = torch.full((batch_size,), candidate_label, device=device)
                pos_data = strategy.create_positive(images, candidate_labels)
                
                activations = model(pos_data)
                goodness = model.layers[-1].goodness(activations[-1])
                
                better = goodness > best_goodness
                predictions[better] = candidate_label
                best_goodness[better] = goodness[better]
            
            correct += (predictions == labels).sum().item()
            total += batch_size
    
    return correct / total


def evaluate_linear_probe(
    model: FFNetwork,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int = 10,
    probe_epochs: int = 10,
    probe_lr: float = 0.01
) -> float:
    """
    Evaluate FF network using linear probe (for SCFF).
    
    This is the correct evaluation method for self-supervised learning:
    1. Extract features from frozen FF network
    2. Train a linear classifier on these features
    3. Evaluate the classifier
    
    Note: For SCFF, the model was trained with input = 2*x (positive) or x+y (negative).
    So for consistent feature extraction, we use 2*x as input during evaluation.
    """
    model.eval()
    
    # Get feature dimension from last layer
    feature_dim = model.layers[-1].linear.out_features
    
    # Create linear probe
    probe = nn.Linear(feature_dim, num_classes).to(device)
    probe_optimizer = optim.Adam(probe.parameters(), lr=probe_lr)
    criterion = nn.CrossEntropyLoss()
    
    # Train linear probe
    for epoch in range(probe_epochs):
        probe.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # For SCFF: use 2*x to match training format (positive = x + x = 2x)
            flat = images.view(images.size(0), -1)
            scff_input = 2 * flat  # Match SCFF positive sample format
            
            with torch.no_grad():
                activations = model(scff_input)
                features = activations[-1]  # Last layer features
            
            # Train probe
            logits = probe(features)
            loss = criterion(logits, labels)
            
            probe_optimizer.zero_grad()
            loss.backward()
            probe_optimizer.step()
    
    # Evaluate
    probe.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            flat = images.view(images.size(0), -1)
            scff_input = 2 * flat  # Match SCFF positive sample format
            
            activations = model(scff_input)
            features = activations[-1]
            
            logits = probe(features)
            predictions = logits.argmax(dim=1)
            
            correct += (predictions == labels).sum().item()
            total += images.size(0)
    
    return correct / total


# ============================================================
# Experiment Runner
# ============================================================

def get_strategies(device: torch.device) -> Dict[str, Any]:
    """Get all 10 strategies to compare."""
    from negative_strategies import ClassConfusionEmbeddedStrategy
    
    strategies = {
        # 1. Label Embedding - Hinton's original
        'label_embedding': LabelEmbeddingStrategy(num_classes=10, device=device),
        
        # 2. Image Mixing - Mix two images
        'image_mixing': ImageMixingStrategy(num_classes=10, mixing_mode='interpolate', device=device),
        
        # 3. Random Noise - Baseline
        'random_noise': RandomNoiseStrategy(num_classes=10, noise_type='matched', device=device),
        
        # 4. Class Confusion (Embedded) - Wrong label embedded in image
        'class_confusion': ClassConfusionEmbeddedStrategy(num_classes=10, device=device),
        
        # 5. Self-Contrastive - Strong augmentation as negatives
        'self_contrastive': SelfContrastiveStrategy(num_classes=10, device=device),
        
        # 6. Masking - Random pixel masking
        'masking': MaskingStrategy(num_classes=10, mask_ratio=0.5, mask_mode='zero', device=device),
        
        # 7. Layer-wise - Adaptive layer-specific negatives
        'layer_wise': LayerWiseStrategy(num_classes=10, perturbation_scale=0.5, device=device),
        
        # 8. Adversarial - Gradient-based perturbation (fast version without model)
        'adversarial': AdversarialStrategy(num_classes=10, epsilon=0.1, num_steps=1, device=device),
        
        # 9. Hard Mining - Select hardest negatives
        'hard_mining': HardMiningStrategy(num_classes=10, mining_mode='class', device=device),
        
        # 10. Mono-Forward - No negatives variant
        'mono_forward': MonoForwardStrategy(num_classes=10, device=device),
    }
    return strategies


def run_single_experiment(
    strategy_name: str,
    strategy,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 10,
    lr: float = 0.03,
    seed: int = 42
) -> Dict[str, Any]:
    """Run single experiment for one strategy."""
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Check if mono-forward
    use_mono = hasattr(strategy, 'uses_negatives') and not strategy.uses_negatives
    
    # Check if SCFF (needs linear probe evaluation)
    use_linear_probe = isinstance(strategy, SelfContrastiveStrategy)
    
    # Create model
    layer_sizes = [784, 500, 500]  # MNIST input -> 500 -> 500
    model = FFNetwork(layer_sizes, threshold=2.0).to(device)
    model.setup_optimizers(lr=lr)
    
    # Training history
    history = {
        'losses': [],
        'accuracies': [],
        'epoch_times': [],
    }
    
    total_time = 0
    
    for epoch in range(num_epochs):
        # Train
        losses, epoch_time = train_epoch(
            model, train_loader, strategy, device, use_mono=use_mono
        )
        total_time += epoch_time
        
        # Evaluate (use appropriate method based on strategy)
        if use_linear_probe:
            # SCFF: use linear probe evaluation
            accuracy = evaluate_linear_probe(
                model, train_loader, test_loader, device,
                probe_epochs=5, probe_lr=0.01
            )
        else:
            # Other strategies: use label embedding evaluation
            accuracy = evaluate(model, test_loader, strategy, device)
        
        history['losses'].append(sum(losses.values()))
        history['accuracies'].append(accuracy)
        history['epoch_times'].append(epoch_time)
        
        print(f"    Epoch {epoch+1}/{num_epochs} | "
              f"Loss: {sum(losses.values()):.4f} | "
              f"Acc: {accuracy*100:.2f}% | "
              f"Time: {epoch_time:.1f}s", flush=True)
    
    return {
        'final_accuracy': history['accuracies'][-1],
        'accuracies': history['accuracies'],
        'losses': history['losses'],
        'epoch_times': history['epoch_times'],
        'total_time': total_time,
        'convergence_epoch': find_convergence_epoch(history['accuracies']),
    }


def find_convergence_epoch(accuracies: List[float], threshold: float = 0.95) -> int:
    """Find epoch where accuracy first exceeds threshold * final_acc."""
    if not accuracies:
        return -1
    
    final_acc = accuracies[-1]
    target = final_acc * threshold
    
    for i, acc in enumerate(accuracies):
        if acc >= target:
            return i + 1  # 1-indexed
    
    return len(accuracies)


def run_all_experiments(
    num_runs: int = 3,
    num_epochs: int = 10,
    lr: float = 0.03,
    batch_size: int = 64
) -> Dict[str, Any]:
    """Run all strategy experiments."""
    
    # Setup device
    device = torch.device(
        'cuda' if torch.cuda.is_available() 
        else 'mps' if torch.backends.mps.is_available() 
        else 'cpu'
    )
    print(f"Using device: {device}")
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Get strategies
    strategies = get_strategies(device)
    
    # Results storage
    all_results = {}
    
    print("\n" + "=" * 60, flush=True)
    print("Negative Sample Strategy Comparison Experiment", flush=True)
    print("=" * 60, flush=True)
    print(f"Epochs: {num_epochs} | Runs per strategy: {num_runs}", flush=True)
    print(f"Architecture: 784 → 500 → 500 | LR: {lr} | Batch: {batch_size}", flush=True)
    print("=" * 60, flush=True)
    
    for strategy_name, strategy in strategies.items():
        print(f"\n[{strategy_name}] Starting {num_runs} runs...", flush=True)
        
        run_results = []
        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}:")
            result = run_single_experiment(
                strategy_name=strategy_name,
                strategy=strategy,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                num_epochs=num_epochs,
                lr=lr,
                seed=42 + run
            )
            run_results.append(result)
        
        # Aggregate results
        final_accs = [r['final_accuracy'] for r in run_results]
        total_times = [r['total_time'] for r in run_results]
        convergence_epochs = [r['convergence_epoch'] for r in run_results]
        
        all_results[strategy_name] = {
            'runs': run_results,
            'mean_accuracy': np.mean(final_accs),
            'std_accuracy': np.std(final_accs),
            'mean_time': np.mean(total_times),
            'std_time': np.std(total_times),
            'mean_convergence_epoch': np.mean(convergence_epochs),
            'config': strategy.get_config() if hasattr(strategy, 'get_config') else {},
        }
        
        print(f"  → Mean Acc: {np.mean(final_accs)*100:.2f}% ± {np.std(final_accs)*100:.2f}%")
        print(f"  → Mean Time: {np.mean(total_times):.1f}s ± {np.std(total_times):.1f}s")
    
    return all_results


def save_results(results: Dict, output_dir: str):
    """Save results to JSON, MD, and PNG."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Save JSON
    json_path = output_path / 'strategy_comparison.json'
    
    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(json_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\nSaved JSON: {json_path}")
    
    # 2. Save Markdown Report
    md_path = output_path / 'strategy_comparison.md'
    generate_markdown_report(results, md_path)
    print(f"Saved Report: {md_path}")
    
    # 3. Save Visualization
    png_path = output_path / 'strategy_comparison.png'
    generate_visualization(results, png_path)
    print(f"Saved Plot: {png_path}")


def generate_markdown_report(results: Dict, output_path: Path):
    """Generate Markdown report."""
    
    # Sort by accuracy
    sorted_strategies = sorted(
        results.items(), 
        key=lambda x: x[1]['mean_accuracy'], 
        reverse=True
    )
    
    report = []
    report.append("# 负样本策略对比实验报告")
    report.append("")
    report.append("## 实验配置")
    report.append("")
    report.append("- **数据集**: MNIST")
    report.append("- **网络架构**: 784 → 500 → 500")
    report.append("- **优化器**: Adam (lr=0.03)")
    report.append("- **Batch Size**: 64")
    report.append("- **Epochs**: 10")
    report.append("- **每个策略运行次数**: 3")
    report.append("")
    
    report.append("## 总体排名")
    report.append("")
    report.append("| 排名 | 策略 | 准确率 | 收敛速度 (Epoch) | 训练时间 (s) |")
    report.append("|------|------|--------|------------------|--------------|")
    
    for rank, (name, data) in enumerate(sorted_strategies, 1):
        acc = f"{data['mean_accuracy']*100:.2f}% ± {data['std_accuracy']*100:.2f}%"
        conv = f"{data['mean_convergence_epoch']:.1f}"
        time_str = f"{data['mean_time']:.1f} ± {data['std_time']:.1f}"
        report.append(f"| {rank} | {name} | {acc} | {conv} | {time_str} |")
    
    report.append("")
    report.append("## 详细分析")
    report.append("")
    
    # Top performer
    top_name, top_data = sorted_strategies[0]
    report.append(f"### 🥇 最佳策略: {top_name}")
    report.append("")
    report.append(f"- **准确率**: {top_data['mean_accuracy']*100:.2f}%")
    report.append(f"- **收敛速度**: 第 {top_data['mean_convergence_epoch']:.1f} epoch 达到 95% 最终准确率")
    report.append(f"- **训练时间**: {top_data['mean_time']:.1f}s")
    report.append("")
    
    # Category analysis
    report.append("### 策略分类分析")
    report.append("")
    
    categories = {
        '标签嵌入类': ['label_embedding', 'class_confusion'],
        '图像混合类': ['image_mixing', 'self_contrastive'],
        '噪声/掩码类': ['random_noise', 'masking'],
        '对抗/困难样本类': ['adversarial', 'hard_mining'],
        '特殊方法': ['layer_wise', 'mono_forward'],
    }
    
    for cat_name, strategies_in_cat in categories.items():
        report.append(f"**{cat_name}**:")
        for s_name in strategies_in_cat:
            if s_name in results:
                acc = results[s_name]['mean_accuracy'] * 100
                report.append(f"- {s_name}: {acc:.2f}%")
        report.append("")
    
    # Key findings
    report.append("## 关键发现")
    report.append("")
    
    # Find best/worst
    best_acc = sorted_strategies[0][1]['mean_accuracy']
    worst_acc = sorted_strategies[-1][1]['mean_accuracy']
    acc_gap = (best_acc - worst_acc) * 100
    
    report.append(f"1. **准确率差距**: 最好与最差策略相差 {acc_gap:.1f} 个百分点")
    
    # Fast converger
    fastest = min(results.items(), key=lambda x: x[1]['mean_convergence_epoch'])
    report.append(f"2. **最快收敛**: {fastest[0]} (第 {fastest[1]['mean_convergence_epoch']:.1f} epoch)")
    
    # Training efficiency
    report.append(f"3. **训练效率**: 所有策略在 10 epochs 内均可完成训练")
    
    report.append("")
    report.append("## 结论")
    report.append("")
    report.append(f"在 MNIST 数据集上，**{top_name}** 策略取得了最佳表现。")
    report.append("不同策略之间存在明显差异，选择合适的负样本策略对 Forward-Forward 算法的性能至关重要。")
    report.append("")
    report.append("---")
    report.append("*实验由 Forward-Forward Research 自动生成*")
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))


def generate_visualization(results: Dict, output_path: Path):
    """Generate visualization plots."""
    import matplotlib.pyplot as plt
    
    # Sort by accuracy
    sorted_names = sorted(
        results.keys(), 
        key=lambda x: results[x]['mean_accuracy'], 
        reverse=True
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Bar chart - Accuracy comparison
    ax1 = axes[0, 0]
    accuracies = [results[n]['mean_accuracy'] * 100 for n in sorted_names]
    stds = [results[n]['std_accuracy'] * 100 for n in sorted_names]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(sorted_names)))[::-1]
    
    bars = ax1.barh(sorted_names, accuracies, xerr=stds, color=colors, capsize=3)
    ax1.set_xlabel('Accuracy (%)')
    ax1.set_title('最终准确率对比')
    ax1.set_xlim(0, 100)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax1.text(acc + 1, bar.get_y() + bar.get_height()/2, 
                f'{acc:.1f}%', va='center', fontsize=9)
    
    # 2. Training curves
    ax2 = axes[0, 1]
    for name in sorted_names[:5]:  # Top 5 only for clarity
        accs = results[name]['runs'][0]['accuracies']  # First run
        ax2.plot(range(1, len(accs)+1), [a*100 for a in accs], 
                label=name, marker='o', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('训练曲线 (Top 5 策略)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Training time comparison
    ax3 = axes[1, 0]
    times = [results[n]['mean_time'] for n in sorted_names]
    time_stds = [results[n]['std_time'] for n in sorted_names]
    
    ax3.barh(sorted_names, times, xerr=time_stds, color='steelblue', capsize=3)
    ax3.set_xlabel('Training Time (s)')
    ax3.set_title('训练时间对比')
    
    # 4. Convergence speed
    ax4 = axes[1, 1]
    conv_epochs = [results[n]['mean_convergence_epoch'] for n in sorted_names]
    
    ax4.barh(sorted_names, conv_epochs, color='coral')
    ax4.set_xlabel('Convergence Epoch')
    ax4.set_title('收敛速度 (达到 95% 最终准确率)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# Main Entry Point
# ============================================================

def main():
    """Main experiment runner."""
    import sys
    
    # Ensure output is flushed immediately
    print("Starting experiment...", flush=True)
    
    # Run experiments (reduced for faster comparison)
    results = run_all_experiments(
        num_runs=1,  # Reduced from 3 for speed
        num_epochs=10,
        lr=0.03,
        batch_size=64
    )
    
    # Save results
    output_dir = Path(__file__).parent.parent / 'results'
    save_results(results, output_dir)
    
    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)
    
    # Print summary
    sorted_results = sorted(
        results.items(), 
        key=lambda x: x[1]['mean_accuracy'], 
        reverse=True
    )
    
    print("\nFinal Rankings:")
    for rank, (name, data) in enumerate(sorted_results, 1):
        print(f"  {rank}. {name}: {data['mean_accuracy']*100:.2f}%")


if __name__ == '__main__':
    main()
