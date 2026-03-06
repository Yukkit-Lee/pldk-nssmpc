import os
import glob
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import datasets, transforms
from tqdm import tqdm

# ------------------------------------------------------------
# MPC 快速验证 V2 - 完全按照 MTT 的 evaluate_synset 方式训练
# 关键改进：
# 1. 使用 CrossEntropyLoss (不是 KL Div)
# 2. 添加 weight_decay=0.0005
# 3. 学习率衰减策略
# 4. 训练 1000 轮 (与 MTT 一致)
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
    """测试数据 - 与 MTT 一致"""
    transform = transforms.Compose([transforms.ToTensor()])
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

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

def train_epoch(model, train_loader, optimizer, criterion):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for imgs, labs in train_loader:
        imgs, labs = imgs.to(device), labs.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labs)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += pred.eq(labs).sum().item()
        total += labs.size(0)
    
    return total_loss / len(train_loader), 100. * correct / total

def quick_mpc_test_v2():
    """按照 MTT 的 evaluate_synset 方式训练"""
    print("=" * 60)
    print("MPC Quick Test V2 - MTT Style Training")
    print("=" * 60)

    # 1. 加载数据
    images, labels = load_distilled_data()
    if labels is None:
        raise ValueError("Labels are required!")
    
    test_loader = get_test_loader()

    # 2. 创建训练集 (与 MTT 一致)
    from torch.utils.data import TensorDataset
    train_dataset = TensorDataset(images, labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

    # 3. 初始化模型
    model = ConvNet_MPC(width=128).to(device)
    
    # 训练前精度
    acc_before = evaluate(model, test_loader)
    print(f"\n[Before Training] Accuracy: {acc_before:.2f}%")

    # 4. 训练配置 (与 MTT evaluate_synset 完全一致)
    epochs = 1000  # MTT 使用 1000 轮
    lr = 0.01  # MTT 默认学习率
    lr_schedule = [epochs // 2 + 1]  # 一半时衰减
    weight_decay = 0.0005  # MTT 使用 0.0005
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    print(f"\nTraining {epochs} epochs (MTT style)...")
    print(f"LR: {lr}, Weight Decay: {weight_decay}, LR decay at epoch: {lr_schedule[0]}")
    print("-" * 60)

    best_acc = acc_before
    for epoch in range(epochs + 1):
        loss_train, acc_train = train_epoch(model, train_loader, optimizer, criterion)
        
        # 学习率衰减
        if epoch in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
            print(f"[LR Decay] New LR: {lr}")
        
        # 每100轮验证一次
        if epoch % 100 == 0 or epoch == epochs:
            acc_test = evaluate(model, test_loader)
            best_acc = max(best_acc, acc_test)
            print(f"Epoch {epoch:04d} | Train Loss: {loss_train:.4f} | Train Acc: {acc_train:.2f}% | Test Acc: {acc_test:.2f}% | Best: {best_acc:.2f}%")

    print("-" * 60)
    print(f"[Summary] Before: {acc_before:.2f}% -> After: {acc_test:.2f}% (Best: {best_acc:.2f}%)")
    print("=" * 60)

if __name__ == "__main__":
    quick_mpc_test_v2()
