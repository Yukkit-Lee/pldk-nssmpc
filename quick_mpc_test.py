import os
import glob
import torch
import torch.nn as nn
import crypten
from torchvision.models import resnet18
from torchvision import datasets, transforms
from tqdm import tqdm

# ------------------------------------------------------------
# MPC 加密蒸馏快速验证脚本
# 关键改进：
# 1. KL Div + Temperature 替代 MSE
# 2. 加载 Hard Labels (CE + KD 联合)
# 3. 增加训练轮数到 100 (快速验证)
# ------------------------------------------------------------

crypten.init()
torch.set_num_threads(4)
device = torch.device("cpu")
print(f"Using device: {device}")

class ConvNet_MPC(nn.Module):
    def __init__(self, channel=3, num_classes=10, width=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, width, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 4 * 4, num_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def load_distilled_data():
    """加载蒸馏数据和标签"""
    path = "images_best.pt"
    if not os.path.exists(path):
        files = glob.glob("**/images_best.pt", recursive=True)
        if files:
            files.sort(key=os.path.getmtime, reverse=True)
            path = files[0]
            label_path = os.path.join(os.path.dirname(path), "labels_best.pt")
        else:
            raise FileNotFoundError("images_best.pt not found!")
    else:
        label_path = "labels_best.pt"

    images = torch.load(path).to(device)
    labels = torch.load(label_path).to(device) if os.path.exists(label_path) else None

    print(f">> Loaded distilled data: {images.shape}")
    print(f">> Data Stats: Min={images.min():.2f}, Max={images.max():.2f}, Mean={images.mean():.2f}")
    if labels is not None:
        print(f">> Loaded labels: {labels.shape}")
    return images, labels

def get_test_loader():
    """测试数据 - 与蒸馏数据一致，不归一化"""
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

def load_teacher():
    """加载 Teacher"""
    teacher = resnet18(num_classes=10)
    teacher.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    teacher.maxpool = nn.Identity()

    path = "teacher_resnet18_cifar10.pth"
    if not os.path.exists(path):
        files = glob.glob("**/teacher_resnet18_cifar10.pth", recursive=True)
        if files: path = files[0]
        else: raise FileNotFoundError("Teacher not found")

    teacher.load_state_dict(torch.load(path, map_location=device))
    teacher.to(device).eval()
    return teacher

def evaluate(model, test_loader):
    """评估模型"""
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for imgs, labs in test_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            out = model(imgs)
            pred = out.argmax(dim=1)
            correct += pred.eq(labs).sum().item()
            total += labs.size(0)
    return 100. * correct / total

def quick_mpc_train():
    """MPC 快速训练 - 非加密版本验证配置"""
    print("=" * 60)
    print("MPC Distillation Quick Test (Non-encrypted verification)")
    print("=" * 60)

    # 1. 加载数据
    images, labels = load_distilled_data()
    test_loader = get_test_loader()

    # 2. 加载 Teacher 并计算 Logits
    teacher = load_teacher()
    print("Computing Teacher Logits...")
    teacher.eval()
    with torch.no_grad():
        teacher_logits = teacher(images)

    # 3. 初始化 Student
    student = ConvNet_MPC(width=128).to(device)

    # 训练前精度
    acc_before = evaluate(student, test_loader)
    print(f"\n[Before Training] Accuracy: {acc_before:.2f}%")

    # 4. 训练配置 (参考 MTT evaluate_synset)
    temperature = 4.0
    alpha = 0.3  # 降低 Hard label 权重，更多依赖 KD
    weight_decay = 0.0005  # MTT 使用 0.0005
    lr = 0.01
    epochs = 100
    batch_size = 64
    num_samples = images.shape[0]

    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    # 学习率在 50% 时衰减 (类似 MTT)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs//2], gamma=0.1)

    print(f"\nTraining {epochs} epochs with CE + KD (T={temperature})...")
    print("-" * 60)

    best_acc = acc_before
    for epoch in range(epochs):
        student.train()
        perm = torch.randperm(num_samples)
        total_loss = 0.0
        num_batches = 0

        for i in range(0, num_samples, batch_size):
            idx = perm[i:i+batch_size]
            x = images[idx]
            t = teacher_logits[idx]
            y = labels[idx] if labels is not None else None

            out = student(x)

            # KL Div Loss (Knowledge Distillation)
            kd_loss = nn.KLDivLoss(reduction='batchmean')(
                nn.functional.log_softmax(out / temperature, dim=1),
                nn.functional.softmax(t / temperature, dim=1)
            ) * (temperature ** 2)

            # Cross Entropy Loss (Hard Labels)
            if y is not None:
                ce_loss = nn.CrossEntropyLoss()(out, y)
                loss = alpha * ce_loss + (1 - alpha) * kd_loss
            else:
                loss = kd_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        scheduler.step()

        # 每10轮验证一次
        if (epoch + 1) % 10 == 0 or epoch == 0:
            acc = evaluate(student, test_loader)
            avg_loss = total_loss / num_batches
            current_lr = optimizer.param_groups[0]['lr']
            best_acc = max(best_acc, acc)
            print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | Best: {best_acc:.2f}% | LR: {current_lr:.4f}")

    acc_final = evaluate(student, test_loader)
    print("-" * 60)
    print(f"[Summary] Before: {acc_before:.2f}% -> After: {acc_final:.2f}% (Best: {best_acc:.2f}%)")
    print("=" * 60)

    # 返回最佳精度供参考
    return best_acc

if __name__ == "__main__":
    quick_mpc_train()
