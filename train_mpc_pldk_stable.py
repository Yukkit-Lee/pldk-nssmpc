import os
import glob
import torch
import torch.nn as nn
import crypten
from torchvision.models import resnet18
from torchvision import datasets, transforms
from tqdm import tqdm

# ------------------------------------------------------------
# MPC 密态 PLDK 稳定版本 - 快速验证
# ------------------------------------------------------------

crypten.init()
torch.set_num_threads(4)
device = torch.device("cpu")
print(f"Using device: {device}")

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

def get_test_loader():
    """测试数据 - 标准 CIFAR-10 归一化"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

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
    print("MPC PLDK Stable (50 epochs)")
    print("=" * 50)

    # 加载数据
    images, labels = load_data()
    test_loader = get_test_loader()

    # Teacher Logits
    teacher = load_teacher()
    with torch.no_grad():
        teacher_logits = teacher(images)
    print(f">> Teacher Logits: {teacher_logits.shape}")

    # 加密
    print("Encrypting...")
    images_enc = crypten.cryptensor(images)
    teacher_logits_enc = crypten.cryptensor(teacher_logits)

    # 初始化 MPC Student
    student_plain = ConvNet_MPC(width=128).to(device)
    student = crypten.nn.from_pytorch(student_plain, torch.empty(1, 3, 32, 32).to(device))
    student.encrypt()
    student.train()

    # 训练配置
    epochs, batch_size, lr = 50, 32, 0.01
    num_samples = images.shape[0]
    criterion = crypten.nn.MSELoss()

    print(f"\nTraining {epochs} epochs...")
    print("-" * 50)

    best_acc = 0.0
    for epoch in range(epochs):
        perm = torch.randperm(num_samples)
        total_loss = 0.0
        num_batches = (num_samples + batch_size - 1) // batch_size

        for i in range(0, num_samples, batch_size):
            idx = perm[i:i+batch_size]
            out_enc = student(images_enc[idx])
            loss = criterion(out_enc, teacher_logits_enc[idx])

            student.zero_grad()
            loss.backward()
            student.update_parameters(lr)

            total_loss += loss.get_plain_text().item()

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
