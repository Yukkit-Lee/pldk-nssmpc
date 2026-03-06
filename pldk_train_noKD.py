import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from utils import get_network, TensorDataset, get_dataset
import os
import glob
import time
# ==========================================
# 1. 定义数据增强 (保持与原代码完全一致)
# ==========================================
class DiffAugment(nn.Module):
    def __init__(self):
        super().__init__()
        self.aug = nn.Sequential(
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        )
    
    def forward(self, x):
        return self.aug(x)

def pldk_train_no_kd():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ==========================================
    # 2. 准备测试集 (用于实时验证精度)
    # ==========================================
    class Args:
        def __init__(self):
            self.zca = False
    args = Args()
    
    print("正在加载测试集以进行公平验证...")
    _, _, _, _, _, _, _, _, test_loader, _, _, _ = get_dataset('CIFAR10', './data', 256, args=args)

    # ==========================================
    # 3. 加载蒸馏数据
    # ==========================================
    data_path = "images_best.pt"
    import glob
    files = glob.glob("**/images_best.pt", recursive=True)
    if files:
        data_path = files[0]
        print(f"自动找到蒸馏数据: {data_path}")
    else:
        print("错误: 找不到 images_best.pt")
        return

    images_train = torch.load(data_path).to(device)
    labels_path = data_path.replace('images_best.pt', 'labels_best.pt')
    labels_train = torch.load(labels_path).to(device)
    
    print(f"Loaded Distilled Data: {images_train.shape}")

    # 数据归一化检查
    if not (images_train.max() > 1.0 and images_train.min() < 0):
        print("应用 CIFAR-10 Normalization...")
        transform_norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        images_train = transform_norm(images_train)

    # ==========================================
    # 4. 初始化 Student (无 KD 模式)
    # ==========================================
    # 保持与 Acc_A 相同的 ConvNet 配置
    student = get_network('ConvNet', channel=3, num_classes=10, im_size=(32, 32)).to(device)
    
    # 保持优化器参数一致
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    # 保持调度器一致 (T_max 对应 Epoch 数)
    epochs = 1000
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    augmentor = DiffAugment().to(device)

    train_dataset = TensorDataset(images_train, labels_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    print(f"Starting Acc_B Training (Plain Distilled Data, No KD)...")
    startTime=time.time()
    # print(f"")
    for epoch in range(epochs):
        student.train()
        total_loss = 0
        
        for imgs, labs in train_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            
            # 同样的数据增强
            imgs = augmentor(imgs)
            
            # 只计算明文交叉熵损失，不使用 Teacher Logits
            student_logits = student(imgs)
            loss = F.cross_entropy(student_logits, labs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        
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
            print(f"Epoch {epoch+1:4d} | Loss: {total_loss:.4f} | Test Acc (Acc_B): {acc:.2f}%")

    print(f"\n对比实验完成！")
    print(f"Final Acc_B (Plain Distilled): {acc:.2f}%")
    endTime = time.time()
    print(f"总用时:{endTime - startTime:.4f}s")

if __name__ == '__main__':
    pldk_train_no_kd()