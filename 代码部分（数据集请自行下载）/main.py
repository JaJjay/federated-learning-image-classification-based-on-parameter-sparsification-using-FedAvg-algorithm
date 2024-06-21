import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt

# 定义简单的卷积神经网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 定义FedAvg算法
def federated_average(model, p, optimizer, scheduler, train_loaders, epochs, local_updates):
    losses = []
    accuracies = []

    for epoch in range(1, epochs + 1):
        model.train()
        for _ in range(local_updates):
            user_indices = np.random.choice(len(train_loaders), 2, replace=False)
            for idx in user_indices:
                for data, target in train_loaders[idx]:
                    optimizer.zero_grad()
                    output = model(data)
                    loss = F.nll_loss(output, target)
                    loss.backward()

                    # 掩码矩阵
                    with torch.no_grad():
                        for param in model.parameters():
                            mask = torch.bernoulli(torch.full_like(param, p))
                            param.grad *= mask

                    optimizer.step()

        scheduler.step()  # 更新学习率

        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                total_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(target)

        losses.append(total_loss / total)
        accuracies.append(correct / total)

        print(f"Epoch {epoch}: Average Loss: {losses[-1]}, Accuracy: {accuracies[-1]}")

    return losses, accuracies

# 准备数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 将数据集分成10份，每个用户一份
user_datasets = [Subset(train_dataset, range(i * 5000, (i + 1) * 5000)) for i in range(10)]
train_loaders = [DataLoader(dataset, batch_size=32, shuffle=True) for dataset in user_datasets]
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 创建模型和优化器
def create_model_and_optimizer():
    model = SimpleCNN()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)  # 添加 weight_decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # 每20个epoch学习率衰减
    return model, optimizer, scheduler

# 定义bernoulli分布概率
probabilities = [0.6 ,0.8 ,0.9]

# 训练并绘制曲线图
colors = ['r', 'g', 'b']
plt.figure(figsize=(14, 7))

for idx, p in enumerate(probabilities):
    model, optimizer, scheduler = create_model_and_optimizer()  # 每次都重新初始化模型和优化器
    losses, accuracies = federated_average(model, p, optimizer, scheduler, train_loaders, epochs=100, local_updates=3)

    plt.subplot(1, 2, 1)
    plt.plot(range(1, 101), losses, label=f'p={p}', color=colors[idx])  # 更新为 100 轮
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, 101), accuracies, label=f'p={p}', color=colors[idx])  # 更新为 100 轮
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

plt.savefig('结果图.png')
plt.show()
