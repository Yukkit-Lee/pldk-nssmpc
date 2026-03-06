import os
import glob
import torch
import torch.nn as nn
import crypten
from torchvision.models import resnet18
from torchvision import datasets, transforms
from tqdm import tqdm

# ------------------------------------------------------------
# Init
# ------------------------------------------------------------
crypten.init()
torch.set_num_threads(4)

device = torch.device("cpu")
print("Device:", device)

# ------------------------------------------------------------
# Student Model
# ------------------------------------------------------------
class ConvNet_MPC(nn.Module):

    def __init__(self, channel=3, num_classes=10, width=32):

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
            nn.AvgPool2d(2)

        )

        self.classifier = nn.Sequential(

            nn.Flatten(),
            nn.Linear(width * 4 * 4, num_classes)

        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


# ------------------------------------------------------------
# Load distilled data
# ------------------------------------------------------------
def load_distilled_data():

    path = "images_best.pt"

    if not os.path.exists(path):

        files = glob.glob("**/images_best.pt", recursive=True)

        if len(files) == 0:
            raise FileNotFoundError("images_best.pt not found")

        path = files[0]

    images = torch.load(path).to(device)

    labels_path = path.replace("images_best.pt", "labels_best.pt")
    labels = torch.load(labels_path).to(device)

    print("Distilled images:", images.shape)
    print("Labels:", labels.shape)

    print("Data stats:",
          images.min().item(),
          images.max().item(),
          images.mean().item())

    return images, labels


# ------------------------------------------------------------
# Teacher
# ------------------------------------------------------------
def load_teacher():

    teacher = resnet18(num_classes=10)

    teacher.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    teacher.maxpool = nn.Identity()

    path = "teacher_resnet18_cifar10.pth"

    if not os.path.exists(path):

        files = glob.glob("**/teacher_resnet18_cifar10.pth", recursive=True)

        if len(files) == 0:
            raise FileNotFoundError("teacher not found")

        path = files[0]

    teacher.load_state_dict(torch.load(path, map_location=device))

    teacher = teacher.to(device)

    teacher.eval()

    return teacher


# ------------------------------------------------------------
# Test loader
# ------------------------------------------------------------
def get_test_loader():

    transform = transforms.Compose([

        transforms.ToTensor(),

        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )

    ])

    testset = datasets.CIFAR10(

        root="./data",
        train=False,
        download=True,
        transform=transform

    )

    loader = torch.utils.data.DataLoader(

        testset,
        batch_size=256,
        shuffle=False

    )

    return loader


# ------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------
def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for imgs, labs in loader:

            imgs = imgs.to(device)
            labs = labs.to(device)

            out = model(imgs)

            pred = out.argmax(1)

            correct += pred.eq(labs).sum().item()
            total += labs.size(0)

    return 100 * correct / total


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
def train_mpc():

    print("="*60)
    print("MPC KD Debug Version")
    print("="*60)

    images, labels = load_distilled_data()

    teacher = load_teacher()

    test_loader = get_test_loader()

    # ------------------------------------------------------------
    # Teacher soft labels
    # ------------------------------------------------------------

    T = 4.0

    print("Computing teacher logits...")

    with torch.no_grad():

        teacher_logits = teacher(images)

        teacher_probs = torch.softmax(
            teacher_logits / T,
            dim=1
        )

    # ------------------------------------------------------------
    # Encrypt data
    # ------------------------------------------------------------

    print("Encrypting data...")

    images_enc = crypten.cryptensor(images)

    teacher_probs_enc = crypten.cryptensor(teacher_probs)

    # ------------------------------------------------------------
    # Student
    # ------------------------------------------------------------

    student_plain = ConvNet_MPC(width=128).to(device)

    dummy = torch.empty(1, 3, 32, 32).to(device)

    student = crypten.nn.from_pytorch(student_plain, dummy)

    student.encrypt()

    student.train()

    # ------------------------------------------------------------
    # Train config
    # ------------------------------------------------------------

    epochs = 50
    batch_size = 32
    lr = 0.1

    num_samples = images.shape[0]

    criterion = crypten.nn.MSELoss()

    best_acc = 0

    print("\nStart Training")

    for epoch in range(epochs):

        perm = torch.randperm(num_samples)

        total_loss = 0

        loop = tqdm(range(0, num_samples, batch_size))

        for i in loop:

            idx = perm[i:i+batch_size]

            x_enc = images_enc[idx]

            t_enc = teacher_probs_enc[idx]

            out = student(x_enc)

            # Soft KD
            student_prob = out.softmax(dim=1)

            loss = criterion(student_prob, t_enc)

            student.zero_grad()

            loss.backward()

            student.update_parameters(lr)

            loss_val = loss.get_plain_text().item()

            total_loss += loss_val

            loop.set_postfix(loss=loss_val)

        avg_loss = total_loss / (num_samples // batch_size)

       # ------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------

        student.decrypt()

        val_model = ConvNet_MPC(width=32).to(device)

        val_model.load_state_dict(student.state_dict(), strict=False)

        acc = evaluate(val_model, test_loader)

        best_acc = max(best_acc, acc)

        print(
            f"Epoch {epoch+1} | "
            f"Loss {avg_loss:.4f} | "
            f"Acc {acc:.2f}% | "
            f"Best {best_acc:.2f}%"
        )

        student.encrypt()
        student.train()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":

    train_mpc()