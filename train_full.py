import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_network, get_dataset # 调用你项目原有的工具类
import os
import time
def train_acc_a():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 1. 获取全量 CIFAR-10 数据集 (Acc_A 必须使用全部 50,000 张原图)
    # 这里的 256 是 batch_size
    class Args:
        def __init__(self):
            self.zca = False # 保持与你蒸馏脚本一致，不使用 ZCA
    
    args = Args()
    print("正在加载全量原始数据集...")
    # get_dataset 会自动处理标准预处理（Crop, Flip, Normalize）
    channel, im_size, num_classes, class_names, mean, std, \
    dst_train, dst_test, test_loader, loader_train_dict, \
    class_map, class_map_inv = get_dataset('CIFAR10', './data', 256, args=args)
    
    train_loader = torch.utils.data.DataLoader(dst_train, batch_size=256, shuffle=True, num_workers=4)

    # 2. 初始化 Student 模型 (必须与 Acc_B 使用相同的架构)
    # 使用你代码中的 get_network('ConvNet', ...) 确保参数完全一致
    print("初始化基准模型 (ConvNet)...")
    model = get_network('ConvNet', channel=3, num_classes=10, im_size=(32, 32)).to(device)

    # 3. 设置优化器 (参考 PLDK 论文和你的脚本)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    criterion = nn.CrossEntropyLoss()

    # 4. 开始训练
    epochs = 100 # 全量数据 100 轮通常就收敛到 Acc_A 上限了
    print("Starting Full Data Training for Acc_A...")
    startTime=time.time()
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()

        # 每 10 轮验证一次
        if (epoch + 1) % 10 == 0 or epoch == 0:
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
            
            acc_a = 100. * correct / total
            print(f'Epoch {epoch+1:3d} | Loss: {train_loss:.4f} | Test Acc (Acc_A): {acc_a:.2f}%')

    print(f"\n实验达成！最终 Acc_A (Full Data): {acc_a:.2f}%")
    endTime = time.time()
    print(f"总用时:{endTime - startTime:.4f}s")
    # print(f"请对比你之前的 Acc_B: 63.41%")

if __name__ == '__main__':



    
    train_acc_a()