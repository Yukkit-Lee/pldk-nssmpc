import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_network, get_daparam, TensorDataset, get_eval_pool
import copy

def pldk_train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", torch.cuda.get_device_name(0))
    
    # 1. 加载第一步生成的合成数据
    # 假设路径是 logged_files/CIFAR10/xxx/images_best.pt
    # 你需要根据实际路径修改
    images_path = "./logged_files/CIFAR10/efficient-cosmos-2/images_best.pt" 
    labels_path = "./logged_files/CIFAR10/efficient-cosmos-2/labels_best.pt" 
    images_train = torch.load(images_path).to(device) # [IPC*10, 3, 32, 32]
    labels_train = torch.load(labels_path).to(device) # [IPC*10]
    
    # 2. 加载第二步训练的 Teacher 模型
    # 注意：Teacher 架构要和你训练时定义的一致
    from torchvision.models import resnet18
    teacher = resnet18(num_classes=10)
    teacher.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    teacher.maxpool = nn.Identity()
    teacher.load_state_dict(torch.load('teacher_resnet18_cifar10.pth')) #teacher_resnet18_cifar10.pth
    teacher = teacher.to(device)
    teacher.eval() # 冻结 Teacher，只用于推理

    # 3. 初始化 Student 模型
    # 论文使用的是 ConvNet (3 depth) [cite: 271]
    # 使用 MTT utils 中的 get_network 直接调用
    student = get_network('ConvNet', channel=3, num_classes=10, im_size=(32, 32)).to(device)
    
    # 4. 定义优化器
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    # 5. 定义 PLDK Hyperparameters
    # 论文中提到使用了较高的 temperature [cite: 181, 243]
    T_temp = 4.0 
    # alpha 用于控制权衡 
    alpha = 0.5 
    
    print("Starting PLDK Student Training...")
    
    # 将数据包装成 DataLoader
    train_dataset = TensorDataset(images_train, labels_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

    epochs = 1000 # 因为数据量小，通常需要较多 epoch 或重复迭代
    
    for epoch in range(epochs):
        student.train()
        total_loss = 0
        
        for imgs, labs in train_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            
            # === PLDK 核心逻辑 ===
            
            # A. 获取 Teacher 的 Soft Labels
            with torch.no_grad():
                teacher_logits = teacher(imgs)
            
            # B. 获取 Student 的预测
            student_logits = student(imgs)
            
            # C. 计算 Loss (公式 12) 
            # Part 1: KL Divergence (Knowledge Distillation)
            # PyTorch KLDivLoss 期望输入是 log_softmax，目标是 softmax (概率分布)
            kd_loss = nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(student_logits / T_temp, dim=1),
                F.softmax(teacher_logits / T_temp, dim=1)
            ) * (T_temp * T_temp)
            
            # Part 2: Cross Entropy (Data Distillation Label Matching)
            ce_loss = F.cross_entropy(student_logits, labs)
            
            # Part 3: Combined Loss
            loss = alpha * kd_loss + (1 - alpha) * ce_loss
            
            # ===================
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

    # 6. 评估 Student 性能 (在真实测试集上)
    # 使用 MTT utils 获取测试集
    from utils import get_dataset
    
    # 创建一个简单的 args 对象
    class Args:
        def __init__(self):
            self.zca = False
    
    args = Args()
    _, _, _, _, _, _, _, _, test_loader, _, _, _ = get_dataset('CIFAR10', './data', 256, args=args)
    
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
    print(f"Final PLDK Student Accuracy: {acc:.2f}%")

if __name__ == '__main__':
    pldk_train()