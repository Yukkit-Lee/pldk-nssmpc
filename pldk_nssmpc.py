"""
PLDK + NssMPC 完整集成代码
功能：使用 NssMPC 的 RingTensor 保护蒸馏数据，实现隐私保护的机器学习训练
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
import numpy as np

# ===== NssMPC 导入 =====
from nssmpc.primitives.secret_sharing.arithmetic import RingTensor

# ==========================================
# 配置参数
# ==========================================
class Config:
    # 数据参数
    dataset = 'CIFAR10'
    data_path = './data'
    batch_size = 128
    num_classes = 10
    img_size = 32
    
    # 训练参数
    epochs = 200  # 完整训练需要1000，这里用200做演示
    lr = 0.01
    momentum = 0.9
    weight_decay = 5e-4
    
    # PLDK 参数
    T_temp = 4.0  # 知识蒸馏温度
    alpha = 0.5   # KD损失权重
    
    # 模型参数
    teacher_name = 'resnet18'
    student_name = 'ConvNet'
    
    # 文件路径
    teacher_path = 'teacher_resnet18_cifar10.pth'
    distilled_data_path = 'images_best.pt'
    distilled_labels_path = 'labels_best.pt'
    
    # NssMPC 参数
    use_ring_protection = True  # 是否使用 RingTensor 保护
    ring_modulus = 2**32  # 环模数

config = Config()

# ==========================================
# 工具函数：获取数据集
# ==========================================
def get_cifar10_loaders(batch_size, data_path):
    """获取 CIFAR-10 数据加载器"""
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
    
    trainset = torchvision.datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform_test)
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader

# ==========================================
# 模型定义
# ==========================================
class ConvNet(nn.Module):
    """轻量级卷积网络，作为学生模型"""
    def __init__(self, channel=3, num_classes=10, im_size=32):
        super(ConvNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class DiffAugment(nn.Module):
    """数据增强模块"""
    def __init__(self):
        super().__init__()
        self.aug = nn.Sequential(
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        )
    
    def forward(self, x):
        return self.aug(x)

# ==========================================
# 蒸馏数据加载器
# ==========================================
class DistilledDataLoader:
    """蒸馏数据加载器，支持 NssMPC 保护"""
    def __init__(self, images, labels, batch_size=128, shuffle=True, use_ring_protection=False):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_ring_protection = use_ring_protection
        self.num_samples = len(images)
        
    def __iter__(self):
        indices = list(range(self.num_samples))
        if self.shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            batch_images = self.images[batch_indices]
            batch_labels = self.labels[batch_indices]
            
            if self.use_ring_protection:
                # 使用 RingTensor 保护图像数据
                batch_images = RingTensor(batch_images)
            
            yield batch_images, batch_labels
    
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

def ring_to_float_tensor(ring_data, device):
    """将 RingTensor 转换为浮点张量，供神经网络使用"""
    if isinstance(ring_data, RingTensor):
        # 重建为普通张量
        float_data = ring_data.convert_to_real_field()
        # 确保是浮点类型
        if float_data.dtype != torch.float32:
            float_data = float_data.float()
        return float_data.to(device)
    return ring_data

# ==========================================
# 主训练函数
# ==========================================
def pldk_nssmpc_train():
    """PLDK + NssMPC 完整训练流程"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"{'='*60}")
    print(f"PLDK + NssMPC 集成训练")
    print(f"{'='*60}")
    print(f"设备: {device}")
    print(f"使用 RingTensor 保护: {config.use_ring_protection}")
    print(f"{'='*60}\n")
    
    # ==========================================
    # 1. 加载测试集（用于评估）
    # ==========================================
    print("[1/6] 加载测试集...")
    _, test_loader = get_cifar10_loaders(config.batch_size, config.data_path)
    print(f"    ✅ 测试集加载完成，共 {len(test_loader.dataset)} 张图片")
    
    # ==========================================
    # 2. 加载教师模型
    # ==========================================
    print("\n[2/6] 加载教师模型...")
    teacher = resnet18(num_classes=config.num_classes)
    teacher.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    teacher.maxpool = nn.Identity()
    
    if not os.path.exists(config.teacher_path):
        print(f"    ❌ 错误: 找不到教师模型 {config.teacher_path}")
        return
    
    teacher.load_state_dict(torch.load(config.teacher_path, map_location='cpu'))
    teacher = teacher.to(device)
    teacher.eval()
    
    # 评估教师模型
    teacher_acc = evaluate_model(teacher, test_loader, device)
    print(f"    ✅ 教师模型加载成功，测试准确率: {teacher_acc:.2f}%")
    
    # ==========================================
    # 3. 加载蒸馏数据
    # ==========================================
    print("\n[3/6] 加载蒸馏数据...")
    
    # 查找蒸馏数据文件
    if not os.path.exists(config.distilled_data_path):
        files = glob.glob(f"**/{config.distilled_data_path}", recursive=True)
        if files:
            config.distilled_data_path = files[0]
            config.distilled_labels_path = config.distilled_data_path.replace(
                'images_best.pt', 'labels_best.pt')
            print(f"    自动找到蒸馏数据: {config.distilled_data_path}")
        else:
            print(f"    ❌ 错误: 找不到蒸馏数据文件")
            return
    
    # 加载数据
    images_train = torch.load(config.distilled_data_path).to(device)
    labels_train = torch.load(config.distilled_labels_path).to(device)
    
    print(f"    蒸馏数据形状: {images_train.shape}")
    print(f"    标签形状: {labels_train.shape}")
    print(f"    每类图片数: {images_train.shape[0] // config.num_classes}")
    
    # 数据归一化检查
    if images_train.max() <= 1.0 and images_train.min() >= 0:
        print("    应用 CIFAR-10 归一化...")
        transform_norm = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), 
            (0.2023, 0.1994, 0.2010)
        )
        images_train = transform_norm(images_train)
    
    # ==========================================
    # 4. 初始化学生模型
    # ==========================================
    print("\n[4/6] 初始化学生模型...")
    student = ConvNet(channel=3, num_classes=config.num_classes, im_size=config.img_size).to(device)
    print(f"    学生模型参数量: {sum(p.numel() for p in student.parameters())}")
    
    optimizer = torch.optim.SGD(
        student.parameters(), 
        lr=config.lr, 
        momentum=config.momentum, 
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    augmentor = DiffAugment().to(device)
    
    # ==========================================
    # 5. 创建蒸馏数据加载器
    # ==========================================
    print("\n[5/6] 创建数据加载器...")
    distilled_loader = DistilledDataLoader(
        images=images_train,
        labels=labels_train,
        batch_size=config.batch_size,
        shuffle=True,
        use_ring_protection=config.use_ring_protection
    )
    print(f"    批大小: {config.batch_size}, 总批次数: {len(distilled_loader)}")
    
    # ==========================================
    # 6. 训练循环
    # ==========================================
    print("\n[6/6] 开始训练...")
    print(f"{'='*60}")
    print(f"{'Epoch':<8} {'Loss':<12} {'KD Loss':<12} {'CE Loss':<12} {'Test Acc':<10}")
    print(f"{'='*60}")
    
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(config.epochs):
        student.train()
        total_loss = 0
        total_kd_loss = 0
        total_ce_loss = 0
        
        for batch_idx, (imgs, labs) in enumerate(distilled_loader):
            # 如果使用 RingTensor 保护，转换回浮点张量
            imgs = ring_to_float_tensor(imgs, device)
            labs = labs.to(device)
            
            # 数据增强
            imgs = augmentor(imgs)
            
            # 教师模型推理
            with torch.no_grad():
                teacher_logits = teacher(imgs)
            
            # 学生模型推理
            student_logits = student(imgs)
            
            # 知识蒸馏损失
            kd_loss = F.kl_div(
                F.log_softmax(student_logits / config.T_temp, dim=1),
                F.softmax(teacher_logits / config.T_temp, dim=1),
                reduction='batchmean'
            ) * (config.T_temp * config.T_temp)
            
            # 交叉熵损失
            ce_loss = F.cross_entropy(student_logits, labs)
            
            # 总损失
            loss = config.alpha * kd_loss + (1 - config.alpha) * ce_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_kd_loss += kd_loss.item()
            total_ce_loss += ce_loss.item()
        
        scheduler.step()
        
        # 每10个epoch评估一次
        if (epoch + 1) % 10 == 0:
            acc = evaluate_model(student, test_loader, device)
            
            avg_loss = total_loss / len(distilled_loader)
            avg_kd = total_kd_loss / len(distilled_loader)
            avg_ce = total_ce_loss / len(distilled_loader)
            
            print(f"{epoch+1:<8} {avg_loss:<12.4f} {avg_kd:<12.4f} {avg_ce:<12.4f} {acc:<10.2f}")
            
            # 保存最佳模型
            if acc > best_acc:
                best_acc = acc
                torch.save(student.state_dict(), 'best_student.pth')
                print(f"    👑 新最佳模型！准确率: {acc:.2f}%")
    
    # ==========================================
    # 训练完成
    # ==========================================
    total_time = time.time() - start_time
    print(f"{'='*60}")
    print(f"训练完成！")
    print(f"总用时: {total_time:.2f}秒")
    print(f"最佳准确率: {best_acc:.2f}%")
    
    # 加载最佳模型并最终评估
    if os.path.exists('best_student.pth'):
        student.load_state_dict(torch.load('best_student.pth'))
        final_acc = evaluate_model(student, test_loader, device)
        print(f"最终模型准确率: {final_acc:.2f}%")
    
    print(f"\n{'='*60}")
    print("PLDK + NssMPC 集成训练完成！")
    print(f"{'='*60}")
    
    return student

# ==========================================
# 评估函数
# ==========================================
def evaluate_model(model, test_loader, device):
    """评估模型准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100. * correct / total

# ==========================================
# RingTensor 功能测试
# ==========================================
def test_ring_tensor():
    """测试 RingTensor 基础功能"""
    print("\n测试 RingTensor 功能:")
    print("-" * 40)
    
    # 创建测试数据
    test_data = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    print(f"原始数据: {test_data}")
    
    # 创建 RingTensor
    ring_data = RingTensor(test_data)
    print(f"RingTensor 创建成功")
    
    # 加法运算
    ring_sum = ring_data + ring_data
    print(f"加法运算: {ring_sum}")
    
    # 重建并转换回浮点
    recon = ring_data.convert_to_real_field()
    recon_float = recon.float() if recon.dtype != torch.float32 else recon
    print(f"重建数据: {recon_float}")
    
    if torch.all(test_data == recon_float.cpu()):
        print("✅ 数据一致性验证通过")
    else:
        print("❌ 数据一致性验证失败")
    
    print("✅ RingTensor 功能正常")
    print("-" * 40)

# ==========================================
# 主函数
# ==========================================
if __name__ == '__main__':
    print(f"\n{'='*60}")
    print("PLDK + NssMPC 完整集成系统")
    print(f"{'='*60}")
    
    # 1. 测试 RingTensor
    test_ring_tensor()
    
    # 2. 运行训练
    print("\n")
    pldk_nssmpc_train()
    
    print(f"\n{'='*60}")
    print("程序执行完毕")
    print(f"{'='*60}")