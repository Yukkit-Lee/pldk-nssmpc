import torch
import torch.nn as nn
from torchvision import datasets, transforms

# ------------------------------------------------------------
# Baseline: 用真实 CIFAR-10 训练，看 Student 能达到多少精度
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

def get_loaders():
    # 训练数据：标准 CIFAR-10
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    # 测试数据：同样归一化
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)
    
    return train_loader, test_loader

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

def train_baseline():
    train_loader, test_loader = get_loaders()
    
    student = ConvNet_MPC(width=128).to(device)
    
    # 训练前
    acc_before = evaluate(student, test_loader)
    print(f"[Before Training] Accuracy: {acc_before:.2f}%")
    
    # 标准训练
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    print(f"\nTraining on full CIFAR-10 (50 epochs)...")
    for epoch in range(50):
        student.train()
        total_loss = 0
        for imgs, labs in train_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            
            out = student(imgs)
            loss = criterion(out, labs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            acc = evaluate(student, test_loader)
            print(f"Epoch {epoch+1:02d} | Acc: {acc:.2f}%")
    
    acc_final = evaluate(student, test_loader)
    print(f"\n[Summary] Before: {acc_before:.2f}% -> After: {acc_final:.2f}%")

if __name__ == "__main__":
    train_baseline()
