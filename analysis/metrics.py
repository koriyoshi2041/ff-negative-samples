"""
Metrics for analyzing negative sample properties in Forward-Forward.

Key metrics:
- Hardness: How difficult are negatives to distinguish from positives
- Diversity: How well do negatives cover the feature space
- Distribution Gap: How different are negatives from real data
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Callable
from scipy.stats import wasserstein_distance, pearsonr, spearmanr


class HardnessMetrics:
    """Compute hardness metrics for negative samples."""
    
    @staticmethod
    def goodness_hardness(
        pos_goodness: torch.Tensor, 
        neg_goodness: torch.Tensor
    ) -> Dict[str, float]:
        """
        基于 goodness 分数的难度度量
        难度 = 正负样本 goodness 差距的倒数
        """
        gap = pos_goodness - neg_goodness
        hardness = 1.0 / (gap.abs() + 1e-6)
        
        return {
            'hardness_mean': hardness.mean().item(),
            'hardness_std': hardness.std().item(),
            'goodness_gap_mean': gap.mean().item(),
            'goodness_gap_min': gap.min().item(),  # 最难的样本
            'goodness_gap_std': gap.std().item(),
        }
    
    @staticmethod
    def feature_hardness(
        pos_features: torch.Tensor, 
        neg_features: torch.Tensor
    ) -> Dict[str, float]:
        """
        基于特征距离的难度度量
        难度 = 正负样本特征距离的倒数
        """
        # L2 距离
        l2_dist = torch.norm(pos_features - neg_features, dim=1)
        
        # 余弦相似度
        cos_sim = F.cosine_similarity(pos_features, neg_features)
        
        return {
            'l2_dist_mean': l2_dist.mean().item(),
            'l2_dist_std': l2_dist.std().item(),
            'cosine_sim_mean': cos_sim.mean().item(),
            'cosine_sim_std': cos_sim.std().item(),
            'feature_hardness': (1.0 / (l2_dist + 1e-6)).mean().item(),
        }
    
    @staticmethod
    def classification_hardness(
        neg_features: torch.Tensor,
        pos_labels: torch.Tensor,
        classifier: torch.nn.Module
    ) -> Dict[str, float]:
        """
        使用预训练分类器评估负样本的"可信度"
        """
        with torch.no_grad():
            logits = classifier(neg_features)
            probs = F.softmax(logits, dim=1)
            
            # 负样本被分类为对应正样本标签的概率
            confusion_prob = probs.gather(1, pos_labels.unsqueeze(1)).squeeze()
        
        return {
            'confusion_prob_mean': confusion_prob.mean().item(),
            'confusion_prob_max': confusion_prob.max().item(),
            'confusion_prob_std': confusion_prob.std().item(),
        }


class DiversityMetrics:
    """Compute diversity metrics for negative samples."""
    
    @staticmethod
    def feature_coverage(features: torch.Tensor) -> Dict[str, float]:
        """
        负样本在特征空间的覆盖范围
        """
        # 类内方差（同一 batch 内）
        intra_variance = features.var(dim=0).mean()
        
        # 特征维度激活分布
        activated_dims = (features.abs() > 0.1).float().mean(dim=0)
        coverage = activated_dims.mean()
        
        # 有效维度数量
        effective_dims = (activated_dims > 0.5).sum()
        
        return {
            'intra_variance': intra_variance.item(),
            'feature_coverage': coverage.item(),
            'effective_dimensions': effective_dims.item(),
            'dim_activation_std': activated_dims.std().item(),
        }
    
    @staticmethod
    def pairwise_diversity(
        features: torch.Tensor, 
        sample_size: int = 1000
    ) -> Dict[str, float]:
        """
        负样本之间的两两距离分布
        """
        if features.size(0) > sample_size:
            idx = torch.randperm(features.size(0))[:sample_size]
            features = features[idx]
        
        # 计算两两距离矩阵
        dist_matrix = torch.cdist(features, features)
        
        # 取上三角（排除对角线和重复）
        triu_indices = torch.triu_indices(
            dist_matrix.size(0), dist_matrix.size(1), offset=1
        )
        pairwise_dists = dist_matrix[triu_indices[0], triu_indices[1]]
        
        return {
            'pairwise_dist_mean': pairwise_dists.mean().item(),
            'pairwise_dist_std': pairwise_dists.std().item(),
            'pairwise_dist_min': pairwise_dists.min().item(),
            'pairwise_dist_max': pairwise_dists.max().item(),
        }
    
    @staticmethod
    def entropy_diversity(features: torch.Tensor, n_bins: int = 50) -> Dict[str, float]:
        """
        基于熵的多样性度量
        """
        # 将特征值离散化
        flat = features.view(-1).cpu().numpy()
        hist, _ = np.histogram(flat, bins=n_bins, density=True)
        hist = hist + 1e-10  # 避免 log(0)
        
        # 计算熵（归一化）
        entropy = -np.sum(hist * np.log(hist)) / np.log(n_bins)
        
        # 每个维度的熵
        dim_entropies = []
        for d in range(min(features.size(1), 100)):  # 限制维度数
            dim_flat = features[:, d].cpu().numpy()
            hist, _ = np.histogram(dim_flat, bins=n_bins, density=True)
            hist = hist + 1e-10
            dim_entropy = -np.sum(hist * np.log(hist)) / np.log(n_bins)
            dim_entropies.append(dim_entropy)
        
        return {
            'feature_entropy': entropy,
            'dim_entropy_mean': np.mean(dim_entropies),
            'dim_entropy_std': np.std(dim_entropies),
        }


class DistributionMetrics:
    """Compute distribution gap metrics."""
    
    @staticmethod
    def goodness_distribution_gap(
        real_goodness: torch.Tensor, 
        neg_goodness: torch.Tensor
    ) -> Dict[str, float]:
        """
        比较负样本与真实数据的 goodness 分布
        """
        # Wasserstein 距离
        w_dist = wasserstein_distance(
            real_goodness.cpu().numpy(), 
            neg_goodness.cpu().numpy()
        )
        
        # 基础统计量差距
        mean_gap = (real_goodness.mean() - neg_goodness.mean()).abs().item()
        std_gap = (real_goodness.std() - neg_goodness.std()).abs().item()
        
        return {
            'goodness_wasserstein': w_dist,
            'goodness_mean_gap': mean_gap,
            'goodness_std_gap': std_gap,
        }
    
    @staticmethod
    def feature_distribution_gap(
        real_features: torch.Tensor, 
        neg_features: torch.Tensor
    ) -> Dict[str, float]:
        """
        类似 FID 的特征分布距离
        """
        # 计算均值和协方差
        real_mean = real_features.mean(dim=0)
        neg_mean = neg_features.mean(dim=0)
        
        # 均值差距
        mean_diff = torch.norm(real_mean - neg_mean).item()
        
        # 协方差差距（简化版，只比较对角线）
        real_var = real_features.var(dim=0)
        neg_var = neg_features.var(dim=0)
        var_diff = torch.norm(real_var - neg_var).item()
        
        return {
            'feature_mean_diff': mean_diff,
            'feature_var_diff': var_diff,
            'pseudo_fid': mean_diff ** 2 + var_diff,
        }


class NegativePropertyAnalyzer:
    """
    Complete analyzer for negative sample properties.
    """
    
    def __init__(
        self, 
        goodness_fn: Callable = None,
        device: torch.device = None
    ):
        """
        Args:
            goodness_fn: Function to compute goodness from activations
            device: Device for computations
        """
        self.goodness_fn = goodness_fn or (lambda x: (x ** 2).sum(dim=1))
        self.device = device or torch.device('cpu')
    
    def analyze_samples(
        self,
        pos_samples: torch.Tensor,
        neg_samples: torch.Tensor,
        pos_activations: torch.Tensor = None,
        neg_activations: torch.Tensor = None,
    ) -> Dict[str, float]:
        """
        Analyze a batch of positive and negative samples.
        
        Args:
            pos_samples: Positive samples (B, D)
            neg_samples: Negative samples (B, D)
            pos_activations: Model activations for positives (optional)
            neg_activations: Model activations for negatives (optional)
            
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        # Use raw samples if activations not provided
        pos_features = pos_activations if pos_activations is not None else pos_samples
        neg_features = neg_activations if neg_activations is not None else neg_samples
        
        # Compute goodness if activations provided
        if pos_activations is not None and neg_activations is not None:
            pos_goodness = self.goodness_fn(pos_activations)
            neg_goodness = self.goodness_fn(neg_activations)
            
            metrics.update(HardnessMetrics.goodness_hardness(pos_goodness, neg_goodness))
            metrics.update(DistributionMetrics.goodness_distribution_gap(
                pos_goodness, neg_goodness
            ))
        
        # Feature-based metrics
        metrics.update(HardnessMetrics.feature_hardness(pos_features, neg_features))
        metrics.update(DiversityMetrics.feature_coverage(neg_features))
        metrics.update(DiversityMetrics.pairwise_diversity(neg_features))
        metrics.update(DiversityMetrics.entropy_diversity(neg_features))
        metrics.update(DistributionMetrics.feature_distribution_gap(pos_features, neg_features))
        
        return metrics
    
    def compare_strategies(
        self,
        strategies: Dict[str, 'NegativeStrategy'],
        test_data: torch.Tensor,
        test_labels: torch.Tensor,
        model: torch.nn.Module = None,
    ) -> 'pd.DataFrame':
        """
        Compare multiple negative strategies.
        
        Args:
            strategies: Dict of strategy_name -> strategy_instance
            test_data: Test images
            test_labels: Test labels
            model: Optional model for computing activations
            
        Returns:
            DataFrame with metrics for each strategy
        """
        import pandas as pd
        
        results = []
        
        for name, strategy in strategies.items():
            print(f"Analyzing {name}...")
            
            # Generate samples
            pos_samples = strategy.create_positive(test_data, test_labels)
            neg_samples = strategy.generate(test_data, test_labels)
            
            # Get activations if model provided
            pos_activations = None
            neg_activations = None
            if model is not None:
                with torch.no_grad():
                    pos_activations = model(pos_samples)
                    neg_activations = model(neg_samples)
            
            # Analyze
            metrics = self.analyze_samples(
                pos_samples, neg_samples,
                pos_activations, neg_activations
            )
            metrics['strategy'] = name
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def analyze_training_dynamics(
        self,
        strategy: 'NegativeStrategy',
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        epochs: int = 50,
        checkpoint_interval: int = 5,
    ) -> 'pd.DataFrame':
        """
        Analyze how metrics evolve during training.
        
        Returns:
            DataFrame with metrics at each checkpoint
        """
        import pandas as pd
        
        dynamics = []
        
        for epoch in range(epochs):
            # Train one epoch (placeholder - implement actual training)
            # train_one_epoch(model, train_loader, strategy)
            
            if epoch % checkpoint_interval == 0:
                # Sample a batch for analysis
                images, labels = next(iter(train_loader))
                
                pos_samples = strategy.create_positive(images, labels)
                neg_samples = strategy.generate(images, labels)
                
                with torch.no_grad():
                    pos_activations = model(pos_samples)
                    neg_activations = model(neg_samples)
                
                metrics = self.analyze_samples(
                    pos_samples, neg_samples,
                    pos_activations, neg_activations
                )
                metrics['epoch'] = epoch
                dynamics.append(metrics)
        
        return pd.DataFrame(dynamics)


def compute_correlations(
    df: 'pd.DataFrame', 
    target_col: str = 'accuracy',
    metric_cols: list = None
) -> Dict[str, Tuple[float, float]]:
    """
    Compute correlation between metrics and target (e.g., accuracy).
    
    Returns:
        Dict of metric_name -> (pearson_r, spearman_r)
    """
    if metric_cols is None:
        metric_cols = [c for c in df.columns if c not in [target_col, 'strategy', 'epoch']]
    
    correlations = {}
    
    for col in metric_cols:
        if col in df.columns:
            # Remove NaN values
            valid = df[[col, target_col]].dropna()
            if len(valid) >= 3:
                pearson, _ = pearsonr(valid[col], valid[target_col])
                spearman, _ = spearmanr(valid[col], valid[target_col])
                correlations[col] = (pearson, spearman)
    
    return correlations


# Quick test
if __name__ == "__main__":
    # Test with random data
    pos = torch.randn(100, 784)
    neg = torch.randn(100, 784) * 0.5  # Different distribution
    
    analyzer = NegativePropertyAnalyzer()
    metrics = analyzer.analyze_samples(pos, neg)
    
    print("Sample Metrics:")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")
