import torch
import sys
sys.path.insert(0, '/Users/parafee41/Desktop/Rios/ff-research')

from negative_strategies import StrategyRegistry, STRATEGY_INFO
from torchvision import datasets, transforms

# 加载少量MNIST数据
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
images, labels = next(iter(loader))
images = images.view(32, -1)  # Flatten

print("=" * 60)
print("策略验证测试")
print("=" * 60)

results = {}
for name in STRATEGY_INFO.keys():
    try:
        strategy = StrategyRegistry.create(name, num_classes=10)

        # 测试生成
        if hasattr(strategy, 'uses_negatives') and not strategy.uses_negatives:
            # mono_forward等无负样本策略
            pos = strategy.create_positive(images, labels)
            results[name] = f"✅ positive shape: {pos.shape}"
        else:
            neg = strategy.generate(images, labels)
            results[name] = f"✅ negative shape: {neg.shape}"
    except Exception as e:
        results[name] = f"❌ Error: {str(e)[:50]}"

print("\n结果:")
for name, result in results.items():
    print(f"  {name:20s}: {result}")
