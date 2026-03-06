import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import crypten
from torchvision.models import resnet18
# 假设 utils.py 在同级目录下，如果没有，请确保你有相应的 Dataset 代码
from utils import TensorDataset, get_dataset 
import time
from tqdm import tqdm

# ============================================================
# 0. 初始化 & 配置
# ============================================================
crypten.init()
device = torch.device("cpu")
# 建议根据你的 CPU 核数调整线程数，避免争抢资源
torch.set_num_threads(4) 
print(f"Using device: {device} (Optimized for MPC)")

# ============================================================
# 1. 模型定义
# ============================================================
class ConvNet_MPC(nn.Module):
    def __init__(self, channel=3, num_classes=10):
        super().__init__()
        # 这是一个标准网络。如果想更快，可以参考之前的建议减小通道数
        self.features = nn.Sequential(
            nn.Conv2d(channel, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ============================================================
# 2. 数据集类
# ============================================================
class TensorDatasetTriple(torch.utils.data.Dataset):
    def __init__(self, images, labels, teacher_logits):
        self.images = images
        self.labels = labels
        self.teacher_logits = teacher_logits

    def __getitem__(self, index):
        return self.images[index], self.labels[index], self.teacher_logits[index]

    def __len__(self):
        return self.images.shape[0]

# ============================================================
# 3. 辅助函数 (加载数据 & 验证)
# ============================================================
def load_teacher():
    teacher = resnet18(num_classes=10)
    teacher.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    teacher.maxpool = nn.Identity()
    
    path = "teacher_resnet18_cifar10.pth"
    if os.path.exists(path):
        teacher.load_state_dict(torch.load(path, map_location=device))
    else:
        print(">> Warning: Teacher checkpoint not found! Using Random Teacher.")
    
    teacher.to(device).eval()
    return teacher

def load_distilled_data():
    path = "images_best.pt"
    if not os.path.exists(path):
        import glob
        files = glob.glob("**/images_best.pt", recursive=True)
        if files:
            path = files[0]
        else:
            print(">> Error: images_best.pt not found! Using DUMMY data for demo.")
            return torch.randn(500, 3, 32, 32).to(device), torch.randint(0, 10, (500,)).to(device)
            
    images = torch.load(path).to(device)
    # 尝试加载对应的 labels
    label_path = path.replace("images_best.pt", "labels_best.pt")
    if os.path.exists(label_path):
        labels = torch.load(label_path).to(device)
    else:
        labels = torch.randint(0, 10, (images.shape[0],)).to(device)

    # 简单检查数据归一化
    if images.max() > 5.0 or images.min() < -5.0:
        print(f"!! Warning: Data range large (Min:{images.min():.2f}, Max:{images.max():.2f}).")
        
    return images, labels

def evaluate_fast(student, test_loader):
    """验证函数：只跑一个 Batch 以节省时间"""
    student.eval()
    correct = 0
    total = 0
    
    try:
        imgs, labs = next(iter(test_loader))
    except StopIteration:
        return 0.0

    with torch.no_grad():
        imgs = imgs.to(device)
        labs = labs.to(device)
        
        # 加密推理
        x_enc = crypten.cryptensor(imgs)
        out_enc = student(x_enc)
        out = out_enc.get_plain_text()
        
        _, pred = out.max(1)
        correct += pred.eq(labs).sum().item()
        total += labs.size(0)
        
    student.train()
    return 100. * correct / total

# ============================================================
# 4. 主训练流程 (减少验证频率版)
# ============================================================
def train_mpc_reduced_validation():
    # --- A. 准备阶段 ---
    teacher = load_teacher()
    images, labels = load_distilled_data()
    
    print(">> Pre-computing Teacher Logits...")
    with torch.no_grad():
        teacher_logits = teacher(images)
    del teacher # 释放内存
    
    # 这里的 Batch Size 可以根据内存情况调整，250 比较快但吃内存
    BATCH_SIZE = 64
    train_dataset = TensorDatasetTriple(images, labels, teacher_logits)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 准备测试集
    try:
        class Args: zca = False
        _, _, _, _, _, _, _, _, test_loader, _, _, _ = get_dataset('CIFAR10', './data', 100, args=Args())
    except:
        # 如果没有真实数据，造假数据防止报错
        dummy_test_data = torch.randn(100, 3, 32, 32), torch.randint(0, 10, (100,))
        test_loader = [(dummy_test_data[0], dummy_test_data[1])]

    # --- B. 模型加密 ---
    student_plain = ConvNet_MPC().to(device)
    dummy_input = torch.empty(1, 3, 32, 32).to(device)
    
    print(">> Encrypting model (this may take a moment)...")
    try:
        student = crypten.nn.from_pytorch(student_plain, dummy_input, opset_version=11)
    except:
        student = crypten.nn.from_pytorch(student_plain, dummy_input)
        
    student.encrypt()
    student.train()
    
    # --- C. 训练配置 ---
    lr = 0.01
    epochs = 20
    VAL_FREQ = 5  # <--- 关键修改：每 5 个 Epoch 才验证一次
    
    criterion = crypten.nn.MSELoss()
    
    print(f"\nStart MPC Training (Validation every {VAL_FREQ} epochs)...")
    global_start = time.time()
    
    for epoch in range(epochs):
        loss_accum = 0.0
        
        # 进度条设置
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs}", leave=True)
        
        for batch_idx, (img_batch, _, tea_logits_batch) in enumerate(loop):
            # 加密
            img_enc = crypten.cryptensor(img_batch)
            tea_enc = crypten.cryptensor(tea_logits_batch)
            
            # 前向
            out_enc = student(img_enc)
            loss = criterion(out_enc, tea_enc)
            
            # 反向 & 更新
            student.zero_grad()
            loss.backward()
            student.update_parameters(lr)
            
            # 记录 Loss (解密回明文)
            loss_val = loss.get_plain_text().item()
            loss_accum += loss_val
            
            # 实时更新进度条上的 Loss
            loop.set_postfix(loss=f"{loss_val:.4f}")
            
            if torch.isnan(torch.tensor(loss_val)):
                tqdm.write(f"\n[Error] Loss is NaN! Stopping.")
                return

        # --- 验证逻辑 (关键修改) ---
        avg_loss = loss_accum / len(train_loader)
        
        # 只有在特定 Epoch 才进行验证
        if (epoch + 1) % VAL_FREQ == 0 or (epoch + 1) == epochs:
            tqdm.write(f">> Validating at Epoch {epoch+1}...")
            val_acc = evaluate_fast(student, test_loader)
            tqdm.write(f"Done Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")
        else:
            # 跳过验证，只打印 Loss
            tqdm.write(f"Done Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} | (Validation Skipped)")

    total_time = (time.time() - global_start) / 60
    print(f"\nTraining Finished in {total_time:.2f} minutes.")

if __name__ == "__main__":
    train_mpc_reduced_validation()