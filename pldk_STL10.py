"""
PLDK Baseline for STL-10 (修复版)
基于 pldk_train_v2.py 修改，针对 STL-10 优化
主要改动：
1. 数据集从 CIFAR-10 → STL-10
2. 图像尺寸从 32×32 → 96×96（STL-10原图尺寸）
3. 教师模型需要适应 96×96 输入
4. 归一化参数改为 STL-10 的均值/标准差
5. ⭐ 修复显存问题：数据保持在 CPU，batch 时再加载到 GPU
6. ⭐ 修复 num_workers 问题：显式设为 0 避免 CUDA fork
7. ⭐ 修复重复 to(device) 问题：避免不必要的操作
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import os
import time
import glob

# ==========================================
# 1. 数据增强 (适配 96×96)
# ==========================================
class DiffAugment(nn.Module):
    def __init__(self):
        super().__init__()
        self.aug = nn.Sequential(
            transforms.RandomCrop(96, padding=12),  # STL-10 是 96×96
            transforms.RandomHorizontalFlip(),
        )
    
    def forward(self, x):
        return self.aug(x)

def get_network_stl10(model_name, num_classes=10):
    """获取适配 STL-10 的网络"""
    if model_name == 'ConvNet':
        # STL-10 需要调整 ConvNet 的输入尺寸
        from utils import get_network
        return get_network('ConvNet', channel=3, num_classes=num_classes, im_size=(96, 96))
    elif model_name == 'resnet18':
        # ResNet-18 需要修改第一层以适应 96×96
        net = resnet18(num_classes=num_classes)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()  # 去掉 maxpool，因为 96×96 不算太小
        return net
    else:
        raise ValueError(f"Unknown model: {model_name}")

def get_stl10_test_loader(batch_size=256, data_path='./data', num_workers=0):
    """
    获取 STL-10 测试集
    ⭐ num_workers=0 避免 CUDA fork 问题
    """
    # STL-10 的均值和标准差
    mean = [0.4467, 0.4398, 0.4066]
    std = [0.2603, 0.2566, 0.2713]
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    testset = torchvision.datasets.STL10(
        root=data_path, split='test', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers  # ⭐ 显式设为 0
    )
    
    return testloader

class TensorDataset(torch.utils.data.Dataset):
    """⭐ 自定义 Dataset，数据保持在 CPU"""
    def __init__(self, images, labels):
        self.images = images  # 保持在 CPU
        self.labels = labels  # 保持在 CPU
    
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    
    def __len__(self):
        return len(self.images)

def pldk_train_stl10():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print("="*60)
    print("STL-10 Baseline Training (修复版)")
    print("="*60)

    # ==========================================
    # 2. 加载教师模型 (ResNet-18 适配 96×96)
    # ==========================================
    teacher = get_network_stl10('resnet18', num_classes=10)
    
    teacher_path = 'teacher_resnet18_stl10.pth'  # STL-10 的教师模型
    if not os.path.exists(teacher_path):
        print(f"错误: 找不到 {teacher_path}")
        print("请先训练 STL-10 的教师模型")
        return
        
    teacher.load_state_dict(torch.load(teacher_path, map_location='cpu'))  # ⭐ 先加载到 CPU
    teacher = teacher.to(device)
    teacher.eval()
    
    # 验证教师模型
    print("\n正在验证教师模型性能...")
    test_loader = get_stl10_test_loader(batch_size=128, data_path='./data', num_workers=0)
    
    correct_t = 0
    total_t = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = teacher(inputs)
            _, predicted = outputs.max(1)
            total_t += targets.size(0)
            correct_t += predicted.eq(targets).sum().item()
    teacher_acc = 100. * correct_t / total_t
    print(f"教师模型测试准确率: {teacher_acc:.2f}%")
    
    if teacher_acc < 50:
        print("警告: 教师模型准确率过低！")

    # ==========================================
    # 3. 加载 STL-10 蒸馏数据
    # ==========================================
    print("\n加载蒸馏数据...")
    data_path = "images_best_stl10.pt"  # STL-10 的蒸馏数据
    if not os.path.exists(data_path):
        # 尝试搜索
        files = glob.glob("**/images_best_stl10.pt", recursive=True)
        if files:
            data_path = files[0]
            print(f"自动找到: {data_path}")
        else:
            print("错误: 找不到 STL-10 蒸馏数据")
            return

    # ⭐ 修复①：数据加载到 CPU，不直接放 GPU
    images_train = torch.load(data_path, map_location='cpu')
    labels_path = data_path.replace('images_best_stl10.pt', 'labels_best_stl10.pt')
    labels_train = torch.load(labels_path, map_location='cpu')
    
    print(f"蒸馏数据形状: {images_train.shape}")
    print(f"总图片数: {len(images_train)}")
    print(f"数据设备: CPU (训练时才会移到 GPU)")

    # 数据归一化检查 (STL-10 的归一化参数)
    if images_train.max() <= 1.0 and images_train.min() >= 0:
        print("应用 STL-10 归一化...")
        transform_norm = transforms.Normalize(
            (0.4467, 0.4398, 0.4066),  # STL-10 均值
            (0.2603, 0.2566, 0.2713)   # STL-10 标准差
        )
        # ⭐ 归一化操作在 CPU 上完成
        images_train = transform_norm(images_train)

    # ==========================================
    # 4. 准备学生模型 (ConvNet 适配 96×96)
    # ==========================================
    print("\n初始化学生模型...")
    student = get_network_stl10('ConvNet', num_classes=10).to(device)
    
    optimizer = torch.optim.SGD(
        student.parameters(), 
        lr=0.01, 
        momentum=0.9, 
        weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    augmentor = DiffAugment().to(device)

    # PLDK 参数
    T_temp = 4.0
    alpha = 0.5
    
    # ⭐ 修复②：创建 Dataset，数据在 CPU 上
    train_dataset = TensorDataset(images_train, labels_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=64,  # STL-10 图片更大，batch_size 适当减小
        shuffle=True,
        num_workers=0  # ⭐ 显式设为 0，避免 CUDA fork 问题
    )

    print("\n开始训练...")
    print("="*70)
    print(f"{'Epoch':<8} {'Loss':<12} {'KD Loss':<12} {'CE Loss':<12} {'Test Acc':<10}")
    print("="*70)
    
    startTime = time.time()
    epochs = 1000
    best_acc = 0.0
    
    for epoch in range(epochs):
        student.train()
        total_loss = 0
        total_kd = 0
        total_ce = 0
        
        for imgs, labs in train_loader:
            # ⭐ 修复③：只在此时将数据移到 GPU，避免重复操作
            imgs = imgs.to(device)
            labs = labs.to(device)
            
            # 数据增强（在 GPU 上）
            imgs = augmentor(imgs)
            
            # 教师推理
            with torch.no_grad():
                teacher_logits = teacher(imgs)
            
            # 学生推理
            student_logits = student(imgs)
            
            # 知识蒸馏损失
            kd_loss = F.kl_div(
                F.log_softmax(student_logits / T_temp, dim=1),
                F.softmax(teacher_logits / T_temp, dim=1),
                reduction='batchmean'
            ) * (T_temp * T_temp)
            
            # 交叉熵损失
            ce_loss = F.cross_entropy(student_logits, labs)
            loss = alpha * kd_loss + (1 - alpha) * ce_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_kd += kd_loss.item()
            total_ce += ce_loss.item()
        
        scheduler.step()
        
        # 每100轮评估一次
        if (epoch + 1) % 100 == 0 or epoch == 0:
            student.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = student(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            acc = 100. * correct / total
            
            avg_loss = total_loss / len(train_loader)
            avg_kd = total_kd / len(train_loader)
            avg_ce = total_ce / len(train_loader)
            
            print(f"{epoch+1:<8} {avg_loss:<12.4f} {avg_kd:<12.4f} {avg_ce:<12.4f} {acc:<10.2f}")
            
            if acc > best_acc:
                best_acc = acc
                torch.save(student.state_dict(), 'best_student_stl10.pth')
                # print(f"    👑 新最佳模型！准确率: {acc:.2f}%")

    endTime = time.time()
    print("="*70)
    print(f"\n训练完成！")
    print(f"总用时: {endTime - startTime:.2f}秒")
    print(f"最佳准确率: {best_acc:.2f}%")
    print("模型已保存到: best_student_stl10.pth")

if __name__ == '__main__':
    pldk_train_stl10()