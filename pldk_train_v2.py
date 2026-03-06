import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from utils import get_network, TensorDataset, get_eval_pool, get_dataset
import os
import time
# ==========================================
# 1. 定义数据增强 (关键修复)
# ==========================================
# 即使是蒸馏数据，也必须使用 Augmentation 才能让模型学到通用特征
class DiffAugment(nn.Module):
    def __init__(self):
        super().__init__()
        self.aug = nn.Sequential(
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        )
    
    def forward(self, x):
        return self.aug(x)

def pldk_train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ==========================================
    # 2. 检查 Teacher 模型 (关键修复)
    # ==========================================
    from torchvision.models import resnet18
    teacher = resnet18(num_classes=10)
    teacher.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    teacher.maxpool = nn.Identity()
    
    teacher_path = 'teacher_resnet18_cifar10.pth'
    if not os.path.exists(teacher_path):
        print(f"错误: 找不到 {teacher_path}，请先运行 train_teacher.py")
        return
        
    teacher.load_state_dict(torch.load(teacher_path))
    teacher = teacher.to(device)
    teacher.eval()

    # 验证 Teacher 在测试集上的准确率，确保它是一个合格的老师
    print("正在验证 Teacher 模型性能...")
    # 创建一个简单的 args 对象
    class Args:
        def __init__(self):
            self.zca = False
    
    args = Args()
    # 使用 get_dataset 获取测试集
    _, _, _, _, _, _, _, _, test_loader, _, _, _ = get_dataset('CIFAR10', './data', 256, args=args)
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
    print(f"Teacher Test Accuracy: {teacher_acc:.2f}%")
    
    if teacher_acc < 50:
        print("警告: Teacher 准确率过低！请检查 train_teacher.py 是否训练成功。")

    # ==========================================
    # 3. 加载蒸馏数据 & 归一化处理
    # ==========================================
    data_path = "images_best.pt"  # 请确保路径正确，比如 logged_files/CIFAR10/...
    if not os.path.exists(data_path):
         # 尝试搜索当前目录下的 images_best.pt
        import glob
        files = glob.glob("**/images_best.pt", recursive=True)
        if files:
            data_path = files[0]
            print(f"自动找到蒸馏数据: {data_path}")
        else:
            print("错误: 找不到 images_best.pt")
            return

    # 直接加载 images_best.pt 文件
    images_train = torch.load(data_path).to(device)
    # 加载对应的 labels_best.pt 文件
    labels_path = data_path.replace('images_best.pt', 'labels_best.pt')
    labels_train = torch.load(labels_path).to(device)
    
    print(f"Loaded Distilled Data: {images_train.shape}")

    # [关键检查] 数据是否已经归一化？
    # CIFAR-10 标准化后的数据通常在 -2 到 2 之间
    # 如果数据在 0-1 之间，我们需要手动归一化
    if images_train.max() > 1.0 and images_train.min() < 0:
        print("检测到数据已归一化，跳过手动归一化步骤。")
        pass
    else:
        print("检测到数据可能未归一化 (0-1 range)，正在应用 CIFAR-10 Normalization...")
        transform_norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        images_train = transform_norm(images_train)

    # 准备 Student
    student = get_network('ConvNet', channel=3, num_classes=10, im_size=(32, 32)).to(device)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    # 学习率调度器：防止 Loss 卡死
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    augmentor = DiffAugment().to(device)

    # PLDK 参数
    T_temp = 4.0 
    alpha = 0.5         #alpha
    
    print("Starting PLDK Student Training (Fixed Version)...")
    startTime=time.time()
    train_dataset = TensorDataset(images_train, labels_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    epochs = 1000
    
    for epoch in range(epochs):
        student.train()
        total_loss = 0
        
        for imgs, labs in train_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            
            # [关键修复] 应用数据增强
            imgs = augmentor(imgs)
            
            with torch.no_grad():
                teacher_logits = teacher(imgs)
            
            student_logits = student(imgs)
            
            # Loss Calculation
            kd_loss = nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(student_logits / T_temp, dim=1),
                F.softmax(teacher_logits / T_temp, dim=1)
            ) * (T_temp * T_temp)
            
            ce_loss = F.cross_entropy(student_logits, labs)
            loss = alpha * kd_loss + (1 - alpha) * ce_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        
        if epoch % 100 == 0:
            # 实时验证测试集准确率，而不仅仅看 Loss
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
            
            print(f"Epoch {epoch:4d} | Loss: {total_loss:.4f} | Test Acc: {acc:.2f}%")

    print(f"Final PLDK Student Accuracy: {acc:.2f}%")
    endTime = time.time()
    print(f"总用时:{endTime - startTime:.4f}s")
if __name__ == '__main__':
    pldk_train()


    #CUDA_VISIBLE_DEVICES=0 python train_full.py

    
#     文件名	                    阶段	        是否加密	    作用定位
# pldk_train_v2.py	            明文 KD 阶段	        ❌	    Baseline：验证蒸馏有效性
# pldk_student_mpc.py	        MPC 实验初版	        ✅	    密态 Student 训练（工程过渡版）
# train_mpc_distillation.py	    MPC 正式版	            ✅	    论文级加密蒸馏方案