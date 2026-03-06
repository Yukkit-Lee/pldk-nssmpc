import os
import glob
import torch
import torch.nn as nn
from torchvision.models import resnet18

device = torch.device("cpu")
print("Device:", device)

# ------------------------------------------------------------
# Student
# ------------------------------------------------------------

class ConvNet(nn.Module):

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
            raise FileNotFoundError("teacher model not found")

        path = files[0]

    teacher.load_state_dict(torch.load(path, map_location=device))

    teacher = teacher.to(device)

    teacher.eval()

    return teacher


# ------------------------------------------------------------
# Main Diagnostic
# ------------------------------------------------------------

def diagnose():

    print("\n==============================")
    print("PLDK Distillation Diagnostics")
    print("==============================\n")

    images, labels = load_distilled_data()

    teacher = load_teacher()

    # ------------------------------------------------------------
    # 1️⃣ Teacher accuracy on distilled data
    # ------------------------------------------------------------

    print("\n[1] Teacher accuracy on distilled data")

    with torch.no_grad():

        pred = teacher(images).argmax(1)

        acc = (pred == labels).float().mean()

    print("Teacher Acc:", acc.item())


    # ------------------------------------------------------------
    # 2️⃣ Teacher logits distribution
    # ------------------------------------------------------------

    print("\n[2] Teacher logits distribution")

    with torch.no_grad():

        logits = teacher(images)

        probs = torch.softmax(logits, dim=1)

    print("Logits mean:", logits.mean().item())
    print("Logits max :", logits.max().item())
    print("Logits min :", logits.min().item())

    print("Prob mean :", probs.mean().item())
    print("Prob max  :", probs.max().item())
    print("Prob min  :", probs.min().item())


    # ------------------------------------------------------------
    # 3️⃣ Student overfit distilled data
    # ------------------------------------------------------------

    print("\n[3] Student overfit test")

    student = ConvNet(width=32).to(device)

    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)

    criterion = nn.CrossEntropyLoss()

    for step in range(200):

        out = student(images)

        loss = criterion(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print("step", step, "loss", loss.item())

    with torch.no_grad():

        pred = student(images).argmax(1)

        train_acc = (pred == labels).float().mean()

    print("Student train acc:", train_acc.item())


    # ------------------------------------------------------------
    # 4️⃣ KD signal strength
    # ------------------------------------------------------------

    print("\n[4] KD signal test")

    student = ConvNet(width=32).to(device)

    with torch.no_grad():

        s = student(images)

        t = teacher(images)

    kd_loss = ((s - t) ** 2).mean()

    print("KD Loss:", kd_loss.item())


    print("\n==============================")
    print("Diagnosis finished")
    print("==============================\n")


# ------------------------------------------------------------

if __name__ == "__main__":

    diagnose()