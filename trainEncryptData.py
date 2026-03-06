import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
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
# 1. Model
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# 2. Load Data
# ------------------------------------------------------------
def load_distilled_data():
    path = "images_best.pt"
    if not os.path.exists(path):
        files = glob.glob("**/images_best.pt", recursive=True)
        if files:
            path = files[0]
        else:
            raise FileNotFoundError("images_best.pt not found!")

    images = torch.load(path).to(device)
    print(f">> Loaded distilled data: {images.shape}")
    print(f">> Data Stats: Min={images.min():.2f}, Max={images.max():.2f}")
    return images

# ------------------------------------------------------------
# 3. Test Loader
# ------------------------------------------------------------
def get_test_loader():
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    testset = datasets.CIFAR10(root="./data",
                               train=False,
                               download=True,
                               transform=transform)
    return torch.utils.data.DataLoader(testset,
                                       batch_size=256,
                                       shuffle=False)

# ------------------------------------------------------------
# 4. Load Teacher
# ------------------------------------------------------------
def load_teacher():
    teacher = resnet18(num_classes=10)
    teacher.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    teacher.maxpool = nn.Identity()

    path = "teacher_resnet18_cifar10.pth"
    if not os.path.exists(path):
        files = glob.glob("**/teacher_resnet18_cifar10.pth", recursive=True)
        if files:
            path = files[0]
        else:
            raise FileNotFoundError("Teacher not found")

    teacher.load_state_dict(torch.load(path, map_location=device))
    teacher.to(device).eval()
    return teacher

# ------------------------------------------------------------
# 5. Main Training (FAST KD VERSION)
# ------------------------------------------------------------
def train_logits_kd_mpc():

    teacher = load_teacher()
    test_loader = get_test_loader()

    images = load_distilled_data()

    # ====== 温度 ======
    T = 4.0

    # ====== 计算 Teacher soft targets ======
    print("Pre-computing Teacher Soft Targets...")
    with torch.no_grad():
        teacher_logits = teacher(images)
        teacher_probs = F.softmax(teacher_logits / T, dim=1)

    # ====== 加密 ======
    print("Encrypting data...")
    images_enc = crypten.cryptensor(images)
    teacher_probs_enc = crypten.cryptensor(teacher_probs)

    # ====== Student ======
    print("Initializing Student...")
    student_plain = ConvNet_MPC(width=128).to(device)
    dummy = torch.empty(1, 3, 32, 32).to(device)

    student = crypten.nn.from_pytorch(student_plain, dummy)
    student.encrypt()
    student.train()

    criterion = crypten.nn.MSELoss()

    # 🔥 快速验证设置
    lr = 0.02
    epochs = 10
    batch_size = 128
    num_samples = images.shape[0]

    print("\nStart FAST Distribution KD (10 Epochs)\n")

    for epoch in range(epochs):

        perm = torch.randperm(num_samples)
        total_loss = 0.0

        for i in range(0, num_samples, batch_size):
            idx = perm[i:i+batch_size]

            x_enc = images_enc[idx]
            t_enc = teacher_probs_enc[idx]

            out_enc = student(x_enc)

            # Student softmax + temperature
            out_enc = (out_enc / T).softmax(dim=1)

            loss = criterion(out_enc, t_enc)

            student.zero_grad()
            loss.backward()
            student.update_parameters(lr)

            total_loss += loss.get_plain_text().item()

        avg_loss = total_loss / (num_samples // batch_size)

        # ===== 每个 epoch 都验证 =====
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
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}%")

        student.encrypt()
        student.train()


if __name__ == "__main__":
    train_logits_kd_mpc()