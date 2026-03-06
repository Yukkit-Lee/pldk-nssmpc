import os
import glob
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import datasets, transforms

# ------------------------------------------------------------
# 快速验证脚本 - 训练10轮看精度是否上升
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    # 找到最新的 images_best.pt
    path = "images_best.pt"
    if not os.path.exists(path):
        files = glob.glob("**/images_best.pt", recursive=True)
        if files:
            # 按修改时间排序，取最新的
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
    # [关键修复] 不使用 Normalize，与蒸馏数据保持一致
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

def quick_train():
    teacher = load_teacher()
    images, labels = load_distilled_data()
    test_loader = get_test_loader()

    # 计算 Teacher Logits
    print("Computing Teacher Logits...")
    teacher.eval()
    with torch.no_grad():
        teacher_logits = teacher(images)

    # 初始化 Student
    student = ConvNet_MPC(width=128).to(device)

    # 训练前精度
    acc_before = evaluate(student, test_loader)
    print(f"\n[Before Training] Accuracy: {acc_before:.2f}%")

    # 快速训练 - 使用 Hard Label + KL Div (联合训练)
    # [关键] MTT 使用 1000 epoch 训练，我们至少用 300 epoch
    temperature = 4.0
    alpha = 0.5  # Hard label 权重
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    epochs = 300
    batch_size = 64
    num_samples = images.shape[0]

    print(f"\nQuick Training ({epochs} epochs) with Hard Labels + KD...")
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

            # KL Div with Temperature (知识蒸馏)
            kd_loss = nn.KLDivLoss(reduction='batchmean')(
                nn.functional.log_softmax(out / temperature, dim=1),
                nn.functional.softmax(t / temperature, dim=1)
            ) * (temperature ** 2)

            # Hard Label Loss (如果有标签)
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

        scheduler.step()  # 更新学习率
        avg_loss = total_loss / num_batches
        acc = evaluate(student, test_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | LR: {current_lr:.4f}")

    print(f"\n[Summary] Before: {acc_before:.2f}% -> After: {acc:.2f}% (Δ {acc - acc_before:+.2f}%)")

if __name__ == "__main__":
    quick_train()
