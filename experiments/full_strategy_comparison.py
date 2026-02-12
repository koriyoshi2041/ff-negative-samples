#!/usr/bin/env python3
"""全部10种负样本策略对比 - 1000 epochs"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import json, time, sys
sys.path.insert(0, '.')

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

EPOCHS = 1000
LR = 0.03
THRESHOLD = 2.0
torch.manual_seed(42)

# Load data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, transform=transform)

x_train = train_data.data.float().view(-1, 784) / 255.0
y_train = train_data.targets
x_test = test_data.data.float().view(-1, 784) / 255.0  
y_test = test_data.targets

mean, std = x_train.mean(), x_train.std()
x_train = (x_train - mean) / (std + 1e-8)
x_test = (x_test - mean) / (std + 1e-8)
x_train, y_train = x_train.to(device), y_train.to(device)
x_test, y_test = x_test.to(device), y_test.to(device)

class FFLayer(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_f, out_f), nn.ReLU(), nn.LayerNorm(out_f))
    def forward(self, x): return self.net(x)
    def goodness(self, x): return (x**2).mean(dim=1)

class FFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FFLayer(784,500), FFLayer(500,500)])
    def forward(self, x, upto=None):
        for i, l in enumerate(self.layers):
            x = l(x)
            if upto is not None and i == upto: break
        return x

def overlay(x, y):
    x = x.clone()
    x[:,:10] = 0
    x[range(len(y)), y] = x.max().item()
    return x

def train(model, x_pos, x_neg):
    for li, layer in enumerate(model.layers):
        opt = optim.Adam(layer.parameters(), lr=LR)
        with torch.no_grad():
            hp = x_pos if li==0 else model.forward(x_pos, li-1)
            hn = x_neg if li==0 else model.forward(x_neg, li-1)
        for ep in range(EPOCHS):
            op, on = layer(hp), layer(hn)
            gp, gn = layer.goodness(op), layer.goodness(on)
            loss = torch.log(1+torch.exp(-(gp-THRESHOLD))).mean() + torch.log(1+torch.exp(gn-THRESHOLD)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if (ep+1) % 200 == 0:
                print(f"  L{li} E{ep+1}: loss={loss.item():.4f}")
        with torch.no_grad():
            hp, hn = layer(hp), layer(hn)

def accuracy(model, x, y):
    model.eval()
    with torch.no_grad():
        gs = [model(overlay(x, torch.full((len(x),),l,device=device))).pow(2).mean(1) for l in range(10)]
        return (torch.stack(gs,1).argmax(1)==y).float().mean().item()

# All strategies
strategies = {
    "label_embedding": lambda x,y: overlay(x, (y+torch.randint(1,10,y.shape,device=device))%10),
    "class_confusion": lambda x,y: overlay(x[torch.randperm(len(x))], y),
    "image_mixing": lambda x,y: 0.5*x + 0.5*x[torch.randperm(len(x))],
    "random_noise": lambda x,y: torch.randn_like(x)*x.std()+x.mean(),
    "masking": lambda x,y: x * (torch.rand_like(x)>0.5).float(),
}

results = {}
for name, neg_fn in strategies.items():
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    model = FFNet().to(device)
    x_pos = overlay(x_train, y_train)
    x_neg = neg_fn(x_train, y_train)
    
    t0 = time.time()
    train(model, x_pos, x_neg)
    t = time.time() - t0
    
    tr_acc = accuracy(model, x_train, y_train)
    te_acc = accuracy(model, x_test, y_test)
    print(f"{name}: Train={tr_acc*100:.2f}%, Test={te_acc*100:.2f}%, Time={t:.0f}s")
    
    results[name] = {"train": tr_acc, "test": te_acc, "time": t}
    with open("results/full_strategy_1000ep.json","w") as f:
        json.dump(results, f, indent=2)

print("\n\nFINAL:")
for k,v in results.items():
    print(f"  {k}: {v['test']*100:.2f}%")
