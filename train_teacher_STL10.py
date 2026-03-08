"""
训练 STL-10 教师模型 (ResNet-18)
STL-10 图像尺寸: 96×96
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import os

def train_teacher_stl10():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("="*60)
    print("训练 STL-10 教师模型")
    print("="*60)

    # ==========================================
    # 1. STL-10 数据预处理 (96×96)
    # ==========================================
    # STL-10 的均值和标准差 (需要计算，这里是参考值)
    mean = [0.4467, 0.4398, 0.4066]
    std = [0.2603, 0.2566, 0.2713]
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(96, padding=12),  # STL-10 是 96×96
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # STL-10 数据集 (注意：训练集有 5000 张，测试集 8000 张)
    trainset = torchvision.datasets.STL10(
        root='./data', 
        split='train', 
        download=True, 
        transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=64,  # 96×96 图片更大，batch_size 适当减小
        shuffle=True, 
        num_workers=4
    )

    testset = torchvision.datasets.STL10(
        root='./data', 
        split='test', 
        download=True, 
        transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=64, 
        shuffle=False, 
        num_workers=4
    )

    print(f"训练集: {len(trainset)} 张图片")
    print(f"测试集: {len(testset)} 张图片")
    print(f"图像尺寸: 96×96, 类别数: 10")

    # ==========================================
    # 2. 网络构建 (适配 96×96)
    # ==========================================
    net = resnet18(num_classes=10)
    # 针对 96×96 的修改 (与 CIFAR-10 相同)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()  # 去掉 maxpool，96×96 不算太小
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(), 
        lr=0.1, 
        momentum=0.9, 
        weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_acc = 0
    print("\n开始训练 (200 Epochs)...")
    print("-"*60)

    for epoch in range(200):
        # --- 训练阶段 ---
        net.train()
        train_loss = 0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- 验证阶段 ---
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        scheduler.step()

        # 每20轮打印一次日志
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f'Epoch {epoch+1:3d}/200 | Train Loss: {train_loss/len(trainloader):.3f} | Test Acc: {acc:.2f}%')

        # --- 保存最佳模型 ---
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), 'teacher_resnet18_stl10.pth')
            # print(f'  👑 Epoch {epoch+1}: 新最佳准确率 {acc:.2f}%，已保存')

    print("-"*60)
    print(f"\n训练完成！")
    print(f"最佳准确率: {best_acc:.2f}%")
    print("模型已保存到: teacher_resnet18_stl10.pth")

if __name__ == '__main__':
    train_teacher_stl10()