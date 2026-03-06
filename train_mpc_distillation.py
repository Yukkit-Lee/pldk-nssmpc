import os
import glob
import torch
import torch.nn as nn
import crypten
from torchvision.models import resnet18
from tqdm import tqdm

# ------------------------------------------------------------
# 0. Init
# ------------------------------------------------------------
crypten.init()
torch.set_num_threads(4)
device = torch.device("cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------
# 1. Model (修正 Width=128 以匹配 MTT 标准)
# ------------------------------------------------------------
class ConvNet_MPC(nn.Module):
    # [修改] 默认 width 改为 128，这是 MTT 论文的标准配置
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
        # [新增] 显式初始化，防止运气不好
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

# ------------------------------------------------------------
# 2. Load Data (移除有害的归一化)
# ------------------------------------------------------------
def load_distilled_data():
    path = "images_best.pt"
    if not os.path.exists(path):
        files = glob.glob("**/images_best.pt", recursive=True)
        if files: path = files[0]
        else: raise FileNotFoundError("images_best.pt not found!")

    images = torch.load(path).to(device)
    print(f">> Loaded distilled data: {images.shape}")
    
    # [重要] 打印统计数据，确认数据是否正常 (Min/Max 应该在 -2 到 +2 左右)
    print(f">> Data Stats: Min={images.min():.2f}, Max={images.max():.2f}, Mean={images.mean():.2f}")
    
    # [删除] 绝对不要在这里做 images / max()，这会破坏 MTT 的数据分布！
    
    return images

def get_test_loader():
    from torchvision import datasets, transforms
    # [修复] MTT 蒸馏数据已经是归一化后的，测试时只需要 ToTensor，不要重复归一化！
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # 验证时使用完整测试集
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

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

# ------------------------------------------------------------
# 3. Diagnostic (Teacher 体检)
# ------------------------------------------------------------
def check_teacher_performance(teacher, test_loader):
    """确保 Teacher 在当前环境下是正常的，防止环境或归一化问题"""
    print("\n[Diagnostics] Checking Teacher Health...")
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labs in test_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            out = teacher(imgs)
            pred = out.argmax(dim=1)
            correct += pred.eq(labs).sum().item()
            total += labs.size(0)
    
    acc = 100. * correct / total
    print(f"[Diagnostics] Teacher Accuracy: {acc:.2f}%")
    if acc < 15.0:
        raise RuntimeError("Teacher model is performing randomly! Check your normalization or model weights.")

# ------------------------------------------------------------
# 4. Main Training
# ------------------------------------------------------------
def train_logits_kd_mpc():
    teacher = load_teacher()
    
    # [修复] Teacher 是用标准 CIFAR-10 归一化训练的，诊断时需要使用相同的归一化
    from torchvision import datasets, transforms
    teacher_test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])),
        batch_size=100, shuffle=False
    )
    
    # 1. 现场确认 Teacher 是好的 (使用标准归一化)
    check_teacher_performance(teacher, teacher_test_loader)
    
    # Student 测试数据使用与蒸馏数据相同的归一化（无额外归一化）
    test_loader = get_test_loader()
    
    # 2. 加载数据
    images = load_distilled_data()
    
    # 3. 计算 Teacher Logits
    print("Pre-computing Teacher Logits...")
    with torch.no_grad():
        # MTT 是针对 CrossEntropy 优化的，直接用 Logits 匹配效果最好
        teacher_logits = teacher(images)

    # 4. 加密
    print("Encrypting data...")
    images_enc = crypten.cryptensor(images)
    teacher_logits_enc = crypten.cryptensor(teacher_logits)

    # 5. Student (Width=128)
    print("Initializing Student (Width=128)...")
    student_plain = ConvNet_MPC(width=128).to(device)
    dummy = torch.empty(1, 3, 32, 32).to(device)
    
    try:
        student = crypten.nn.from_pytorch(student_plain, dummy, opset_version=11)
    except:
        student = crypten.nn.from_pytorch(student_plain, dummy)
        
    student.encrypt()
    student.train()

    criterion = crypten.nn.MSELoss()
    # 尝试使用稍大的学习率，因为 MSE Logits 的数值范围较大
    lr = 0.05 
    epochs = 50
    batch_size = 32
    num_samples = images.shape[0]

    print(f"\nStart MPC Logits Distillation (T=1.0, No Norm Hack)")
    
    for epoch in range(epochs):
        perm = torch.randperm(num_samples)
        total_loss = 0.0
        
        num_batches = (num_samples + batch_size - 1) // batch_size
        loop = tqdm(range(0, num_samples, batch_size), total=num_batches, desc=f"Epoch {epoch+1:02d}/{epochs}")

        for i in loop:
            idx = perm[i:i+batch_size]
            x_enc = images_enc[idx]
            t_enc = teacher_logits_enc[idx]

            out_enc = student(x_enc)
            loss = criterion(out_enc, t_enc)

            student.zero_grad()
            loss.backward()
            student.update_parameters(lr)

            loss_val = loss.get_plain_text().item()
            total_loss += loss_val
            loop.set_postfix(loss=f"{loss_val:.4f}")

        avg_loss = total_loss / num_batches

        # 验证
        if (epoch + 1) % 5 == 0 or (epoch + 1) == 1:
            student.decrypt()
            
            val_model = ConvNet_MPC(width=128).to(device)
            val_model.load_state_dict(student.state_dict(), strict=False)
            val_model.eval()

            correct, total = 0, 0
            with torch.no_grad():
                for imgs, labs in test_loader:
                    imgs, labs = imgs.to(device), labs.to(device)
                    out = val_model(imgs)
                    pred = out.argmax(dim=1)
                    correct += pred.eq(labs).sum().item()
                    total += labs.size(0)

            acc = 100. * correct / total
            tqdm.write(f"Done Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} |  Val Acc: {acc:.2f}%")

            student.encrypt()
            student.train()
        else:
            tqdm.write(f"Done Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    train_logits_kd_mpc()