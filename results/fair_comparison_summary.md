# Fair Comparison Results

## Experiment Configuration

- **Epochs**: 10
- **Batch Size**: 128
- **Learning Rate**: 0.03
- **Architecture**: [784, 500, 500]
- **Linear Probe Epochs**: 20

## Results Table

| Rank | Strategy | Accuracy | Assessment Method | Uses Labels | Uses Negatives |
|------|----------|----------|-------------------|-------------|----------------|
| 1 | cwc_ff | 98.54% | channel_wise_ga | Yes | No |
| 2 | label_embedding | 93.93% | label_embedding | Yes | Yes |
| 3 | class_confusion | 93.93% | label_embedding | Yes | Yes |
| 4 | adversarial | 92.66% | linear_probe | No | Yes |
| 5 | random_noise | 78.09% | linear_probe | No | Yes |
| 6 | self_contrastive | 71.26% | linear_probe | No | Yes |
| 7 | image_mixing | 17.79% | linear_probe | No | Yes |
| 8 | mono_forward | 10.32% | label_embedding | Yes | No |
| 9 | masking | 10.30% | linear_probe | No | Yes |
| 10 | hard_mining | 9.80% | label_embedding | Yes | Yes |
| 11 | layer_wise | ERROR | - | - | - |

## Key Insights

- **Best Label-Based Strategy**: label_embedding (93.93%)
- **Best Label-Free Strategy**: adversarial (92.66%)
- **CwC-FF (No Negatives)**: 98.54%