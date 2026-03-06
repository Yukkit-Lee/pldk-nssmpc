import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import os

def train_teacher():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", torch.cuda.get_device_name(0))

    # ==========================================
    # 1. 数据预处理 (与论文对齐)
    # ==========================================
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    # ==========================================
    # 2. 网络构建 (适配 CIFAR-10)
    # ==========================================
    net = resnet18(num_classes=10)
    # 针对 32x32 分辨率的修改
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity() 
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # 论文中常用的 CosineAnnealing 调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_acc = 0  # 用于追踪最高精度

    # ==========================================
    # 3. 训练与验证循环
    # ==========================================
    print("Starting Teacher Training (200 Epochs)...")
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
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        scheduler.step()

        # 打印日志
        print(f'Epoch {epoch:3d} | Train Loss: {train_loss/len(trainloader):.3f} | Test Acc: {acc:.2f}%')

        # --- 保存最佳模型 (Best Model) ---
        if acc > best_acc:
            print(f'Saving best model (Acc: {acc:.2f}%)...')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(net.state_dict(), 'teacher_resnet18_cifar10.pth')
            best_acc = acc

    print(f"Training Finished. Best Accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    train_teacher()