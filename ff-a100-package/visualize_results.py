#!/usr/bin/env python3
"""
Visualization Script for FF Experiment Results
===============================================
Run this locally after copying results from A100.

Usage:
    python visualize_results.py results/
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np


def load_all_results(results_dir: str) -> Dict[str, Dict]:
    """Load all JSON result files from directory."""
    results = {}
    for filepath in Path(results_dir).glob("*.json"):
        with open(filepath) as f:
            results[filepath.stem] = json.load(f)
        print(f"Loaded: {filepath.name}")
    return results


def plot_strategy_comparison(data: Dict, save_path: str = None):
    """Plot negative strategy comparison results."""
    strategies = []
    accuracies = []
    colors = []

    for name, result in data.items():
        if name == 'metadata':
            continue
        strategies.append(name)
        if result.get('uses_labels', True):
            acc = result.get('test_acc', 0)
        else:
            acc = result.get('linear_probe_test_acc', 0)
        accuracies.append(acc * 100)
        colors.append('steelblue' if result.get('uses_labels', True) else 'coral')

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(strategies, accuracies, color=colors)

    ax.set_xlabel('Test Accuracy (%)')
    ax.set_title('Negative Sample Strategy Comparison')
    ax.set_xlim(0, 100)

    for bar, acc in zip(bars, accuracies):
        ax.text(acc + 1, bar.get_y() + bar.get_height()/2,
                f'{acc:.1f}%', va='center')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='Uses Labels'),
        Patch(facecolor='coral', label='Label-Free (Linear Probe)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def plot_transfer_comparison(data: Dict, save_path: str = None):
    """Plot transfer learning comparison."""
    models = []
    source_acc = []
    transfer_acc = []

    for name, result in data.items():
        if name == 'metadata':
            continue
        models.append(name.replace('_', ' ').title())
        source_acc.append(result.get('source_acc', 0) * 100)
        transfer_acc.append(result.get('transfer_acc', 0) * 100)

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, source_acc, width, label='Source (MNIST)', color='steelblue')
    bars2 = ax.bar(x + width/2, transfer_acc, width, label='Transfer (FMNIST)', color='coral')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Transfer Learning: MNIST → Fashion-MNIST')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 100)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def plot_bio_inspired_comparison(results: Dict, save_path: str = None):
    """Compare all bio-inspired models."""
    models = []
    source_accs = []
    transfer_accs = []

    bio_files = ['dendritic_ff', 'three_factor_ff', 'pcl_ff', 'prospective_ff']

    for name in bio_files:
        if name in results:
            data = results[name]
            # Extract accuracy - structure varies by experiment
            if 'results' in data:
                if 'standard_ff' in data['results']:
                    models.append('Standard FF')
                    source_accs.append(data['results']['standard_ff'].get('test_accuracy', 0) * 100)
                    if 'transfer' in data['results']:
                        transfer_accs.append(data['results']['transfer'].get('standard_ff', {}).get('transfer_accuracy', 0) * 100)
                    else:
                        transfer_accs.append(0)

            # Get the bio-inspired model accuracy
            nice_name = name.replace('_ff', '').replace('_', ' ').title()
            models.append(nice_name)

            if 'summary' in data:
                source_accs.append(data['summary'].get('source_accuracy', 0) * 100)
                transfer_accs.append(data['summary'].get('transfer_accuracy', 0) * 100)
            elif 'test_accuracy' in data:
                source_accs.append(data.get('test_accuracy', 0) * 100)
                transfer_accs.append(data.get('transfer_accuracy', 0) * 100)

    if not models:
        print("No bio-inspired results found")
        return

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, source_accs, width, label='Source (MNIST)', color='steelblue')
    bars2 = ax.bar(x + width/2, transfer_accs, width, label='Transfer (FMNIST)', color='coral')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Bio-Inspired FF Models Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 100)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def plot_training_history(data: Dict, save_path: str = None):
    """Plot training history if available."""
    if 'training_history' not in data:
        print("No training history found")
        return

    history = data['training_history']
    n_layers = len(history)

    fig, axes = plt.subplots(1, n_layers, figsize=(5*n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    for idx, (layer_name, layer_data) in enumerate(history.items()):
        ax = axes[idx]
        if 'loss' in layer_data:
            ax.plot(layer_data['epochs'], layer_data['loss'], label='Loss')
        if 'pos_goodness' in layer_data and layer_data['pos_goodness']:
            ax.plot(layer_data['epochs'], layer_data['pos_goodness'], label='Pos Goodness', linestyle='--')
        if 'neg_goodness' in layer_data and layer_data['neg_goodness']:
            ax.plot(layer_data['epochs'], layer_data['neg_goodness'], label='Neg Goodness', linestyle=':')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.set_title(layer_name)
        ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def generate_summary_table(results: Dict) -> str:
    """Generate markdown summary table."""
    lines = [
        "# FF Experiment Results Summary\n",
        "| Experiment | Source Acc | Transfer Acc | Notes |",
        "|------------|------------|--------------|-------|"
    ]

    for name, data in results.items():
        if isinstance(data, dict):
            source = data.get('test_accuracy', data.get('source_acc', 'N/A'))
            transfer = data.get('transfer_accuracy', data.get('transfer_acc', 'N/A'))
            if isinstance(source, float):
                source = f"{source*100:.2f}%"
            if isinstance(transfer, float):
                transfer = f"{transfer*100:.2f}%"
            lines.append(f"| {name} | {source} | {transfer} | |")

    return '\n'.join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_results.py <results_dir>")
        print("Example: python visualize_results.py results/")
        sys.exit(1)

    results_dir = sys.argv[1]
    results = load_all_results(results_dir)

    if not results:
        print(f"No JSON files found in {results_dir}")
        sys.exit(1)

    print(f"\nLoaded {len(results)} result files")

    # Create figures directory
    os.makedirs("figures", exist_ok=True)

    # Generate visualizations
    if 'strategy_comparison_1000ep' in results:
        print("\nPlotting strategy comparison...")
        plot_strategy_comparison(results['strategy_comparison_1000ep'],
                                 'figures/strategy_comparison.png')

    if 'transfer_comparison' in results:
        print("\nPlotting transfer comparison...")
        plot_transfer_comparison(results['transfer_comparison'],
                                 'figures/transfer_comparison.png')

    # Bio-inspired comparison
    print("\nPlotting bio-inspired comparison...")
    plot_bio_inspired_comparison(results, 'figures/bio_inspired_comparison.png')

    # Summary
    print("\n" + "="*60)
    print(generate_summary_table(results))

    print("\nFigures saved in figures/")


if __name__ == '__main__':
    main()
