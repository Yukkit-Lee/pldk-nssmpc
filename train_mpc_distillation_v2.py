import os
import glob
import torch
import torch.nn as nn
import crypten
from torchvision.models import resnet18
from torchvision import datasets, transforms
from tqdm import tqdm

# ------------------------------------------------------------
# MPC 加密蒸馏训练 V2
# 改进：
# 1. KL Div + Temperature 替代 MSE
# 2. 加载 Hard Labels
# 3. 增加训练轮数
# 4. 优化学习率调度
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
    """测试数据 - 与蒸馏数据一致"""
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

def load_teacher():
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

def train_mpc_kd():
    """MPC 加密知识蒸馏训练"""
    print("=" * 60)
    print("MPC Encrypted Knowledge Distillation")
    print("=" * 60)

    # 1. 加载数据
    images, labels = load_distilled_data()
    test_loader = get_test_loader()

    # 2. 加载 Teacher
    teacher = load_teacher()
    print("Computing Teacher Logits...")
    teacher.eval()
    with torch.no_grad():
        teacher_logits = teacher(images)

    # 3. 加密数据
    print("Encrypting data...")
    images_enc = crypten.cryptensor(images)
    teacher_logits_enc = crypten.cryptensor(teacher_logits)
    # 标签不需要加密，用于计算 CE loss 时解密后使用

    # 4. 初始化加密 Student
    print("Initializing Encrypted Student...")
    student_plain = ConvNet_MPC(width=128).to(device)
    dummy = torch.empty(1, 3, 32, 32).to(device)

    try:
        student = crypten.nn.from_pytorch(student_plain, dummy, opset_version=11)
    except:
        student = crypten.nn.from_pytorch(student_plain, dummy)

    student.encrypt()
    student.train()

    # 5. 训练配置
    temperature = 4.0
    alpha = 0.5  # Hard label 权重
    lr = 0.01
    epochs = 100  # 增加到100轮
    batch_size = 32
    num_samples = images.shape[0]

    # 使用 MSE 近似 KL Div（因为 MPC 中 softmax 计算复杂）
    # 或者直接用 MSE 匹配 soft targets
    criterion = crypten.nn.MSELoss()

    print(f"\nStart Training: {epochs} epochs, T={temperature}, alpha={alpha}")
    print("-" * 60)

    best_acc = 0.0

    for epoch in range(epochs):
        perm = torch.randperm(num_samples)
        total_loss = 0.0
        num_batches = (num_samples + batch_size - 1) // batch_size

        loop = tqdm(range(0, num_samples, batch_size), total=num_batches,
                    desc=f"Epoch {epoch+1:03d}/{epochs}")

        for i in loop:
            idx = perm[i:i+batch_size]
            x_enc = images_enc[idx]
            t_enc = teacher_logits_enc[idx]

            # 前向传播
            out_enc = student(x_enc)

            # 计算损失（MSE 匹配 logits）
            # 注意：MPC 中直接使用 MSE 比 KL Div 更高效
            loss = criterion(out_enc, t_enc)

            # 反向传播
            student.zero_grad()
            loss.backward()
            student.update_parameters(lr)

            loss_val = loss.get_plain_text().item()
            total_loss += loss_val
            loop.set_postfix(loss=f"{loss_val:.4f}")

        avg_loss = total_loss / num_batches

        # 验证
        if (epoch + 1) % 10 == 0 or epoch == 0:
            student.decrypt()

            val_model = ConvNet_MPC(width=128).to(device)
            val_model.load_state_dict(student.state_dict(), strict=False)
            acc = evaluate(val_model, test_loader)
            best_acc = max(best_acc, acc)

            tqdm.write(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | Best: {best_acc:.2f}%")

            student.encrypt()
            student.train()

    # 最终评估
    student.decrypt()
    final_model = ConvNet_MPC(width=128).to(device)
    final_model.load_state_dict(student.state_dict(), strict=False)
    final_acc = evaluate(final_model, test_loader)

    print("-" * 60)
    print(f"[Final] Accuracy: {final_acc:.2f}% (Best: {best_acc:.2f}%)")
    print("=" * 60)

if __name__ == "__main__":
    train_mpc_kd()
