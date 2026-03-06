import os
import glob
import torch
import torch.nn as nn
import crypten
from torchvision.models import resnet18
from utils import get_network, get_dataset
from tqdm import tqdm

# ------------------------------------------------------------
# MPC 密态 PLDK 快速验证 V2 - 加入 CE Loss
# ------------------------------------------------------------

crypten.init()
torch.set_num_threads(4)
device = torch.device("cpu")

class ConvNet_MPC(nn.Module):
    def __init__(self, channel=3, num_classes=10, width=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, width, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(width * 4 * 4, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))

def load_data():
    path = "images_best.pt"
    if not os.path.exists(path):
        files = glob.glob("**/images_best.pt", recursive=True)
        files.sort(key=os.path.getmtime, reverse=True)
        path = files[0]
    images = torch.load(path).to(device)
    labels = torch.load(path.replace('images', 'labels')).to(device)
    print(f">> Data: {images.shape}, Labels: {labels.shape}")
    return images, labels

def load_teacher():
    teacher = resnet18(num_classes=10)
    teacher.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    teacher.maxpool = nn.Identity()
    teacher.load_state_dict(torch.load('teacher_resnet18_cifar10.pth', map_location=device))
    return teacher.to(device).eval()

def evaluate(model, test_loader):
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for imgs, labs in test_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            pred = model(imgs).argmax(dim=1)
            correct += pred.eq(labs).sum().item()
            total += labs.size(0)
    return 100. * correct / total

def main():
    print("=" * 50)
    print("MPC PLDK Quick Test V2 (MSE + CE)")
    print("=" * 50)

    # 加载数据
    images, labels = load_data()
    
    # 测试集
    class Args: zca = False
    _, _, _, _, _, _, _, test_loader, _, _, _, _ = get_dataset('CIFAR10', './data', 256, args=Args())

    # Teacher Logits
    teacher = load_teacher()
    with torch.no_grad():
        teacher_logits = teacher(images)
    print(f">> Teacher Logits: {teacher_logits.shape}")

    # 加密
    print("Encrypting...")
    images_enc = crypten.cryptensor(images)
    teacher_logits_enc = crypten.cryptensor(teacher_logits)
    # 标签也加密
    labels_enc = crypten.cryptensor(labels.float().unsqueeze(1))  # [N, 1]

    # 初始化 MPC Student
    student_plain = ConvNet_MPC(width=128).to(device)
    student = crypten.nn.from_pytorch(student_plain, torch.empty(1, 3, 32, 32).to(device))
    student.encrypt()
    student.train()

    # 训练配置
    epochs, batch_size, lr = 50, 32, 0.01
    alpha = 0.5  # MSE 和 CE 的权重
    num_samples = images.shape[0]

    # Loss 函数
    mse_criterion = crypten.nn.MSELoss()
    # CE Loss 在 MPC 中比较复杂，这里用 MSE 近似
    # 或者：解密后计算 CE Loss

    print(f"\nTraining {epochs} epochs (alpha={alpha})...")
    print("-" * 50)

    best_acc = 0.0
    for epoch in range(epochs):
        perm = torch.randperm(num_samples)
        total_loss = 0.0
        num_batches = (num_samples + batch_size - 1) // batch_size

        for i in range(0, num_samples, batch_size):
            idx = perm[i:i+batch_size]
            x_enc = images_enc[idx]
            t_enc = teacher_logits_enc[idx]
            y_enc = labels_enc[idx]

            # 前向传播
            out_enc = student(x_enc)

            # MSE Loss (匹配 Teacher Logits)
            mse_loss = mse_criterion(out_enc, t_enc)

            # CE Loss (匹配 Labels)
            # 方法1：解密后计算 CE Loss（混合模式）
            out_plain = out_enc.get_plain_text()
            y_plain = y_enc.get_plain_text().long().squeeze()
            ce_loss = nn.CrossEntropyLoss()(out_plain, y_plain)

            # 混合 Loss
            loss = alpha * mse_loss + (1 - alpha) * ce_loss

            # 反向传播（只用 MSE 的梯度）
            student.zero_grad()
            mse_loss.backward()  # 只用 MSE 的梯度更新
            student.update_parameters(lr)

            total_loss += loss.item()

        # 每10轮验证
        if (epoch + 1) % 10 == 0 or epoch == 0:
            student.decrypt()
            val_model = ConvNet_MPC(width=128).to(device)
            val_model.load_state_dict(student.state_dict(), strict=False)
            acc = evaluate(val_model, test_loader)
            best_acc = max(best_acc, acc)
            print(f"Epoch {epoch+1:02d} | Loss: {total_loss/num_batches:.4f} | Acc: {acc:.2f}% | Best: {best_acc:.2f}%")
            student.encrypt()
            student.train()

    print("=" * 50)

if __name__ == "__main__":
    main()
