# cryptGPU 集成方案

## 1. 项目现状分析

### 当前使用的技术
- **CrypTen**: 用于 MPC 加密训练
- **模型结构**: `ConvNet_MPC` (卷积神经网络)
- **训练流程**: 加载蒸馏数据 → 计算 Teacher logits → 加密数据 → 密态训练 → 解密评估

### 性能瓶颈
- CrypTen 基于 CPU 计算，速度较慢
- 密态训练过程耗时较长

## 2. cryptGPU 集成方案

### 2.1 环境搭建

#### 2.1.1 快速安装（通过 pip）

```bash
# 克隆 cryptGPU 仓库
git clone https://github.com/jeffreysijuntan/CryptGPU
cd CryptGPU

# 安装 cryptGPU
python3 setup.py install
```

#### 2.1.2 完整安装（从源码构建，推荐用于生产环境）

1. **构建 PyTorch 从源码**
   ```bash
   git clone --recursive https://github.com/pytorch/pytorch
   cd pytorch
   git checkout 1.6
   git submodule sync
   git submodule update --init --recursive
   export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
   python setup.py install
   ```

2. **构建 torchcsprng 从源码**
   ```bash
   git clone https://github.com/pytorch/csprng
   cd csprng
   git checkout 64fedf7e0ab93188cc06b39308ad2ec0e3771bb2
   python setup.py install
   ```

3. **安装 cryptGPU**
   ```bash
   git clone https://github.com/jeffreysijuntan/CryptGPU
   cd CryptGPU
   pip3 install -r requirements_source.txt
   python3 setup.py install
   ```

#### 2.1.3 环境要求
- Python 3.8+
- PyTorch 1.6+
- CUDA 10.2+
- NVIDIA GPU (支持 CUDA)
- torchcsprng (用于加密安全的随机数生成)

### 2.2 代码修改

#### 2.2.1 运行配置设置

1. **配置 launcher.py**
   ```python
   # CryptGPU/launcher.py
   use_csprng = True  # 使用 torchcsprng 作为随机数生成器
   sync_key = False   # 是否同步 AES 密钥（从源码构建时可设为 True）
   ```

2. **核心文件修改 (`mpc_pldk_demo_cryptgpu.py`)**

```python
import os
import glob
import torch
import torch.nn as nn
import crypten # cryptGPU 兼容 CrypTen API
from torchvision.models import resnet18
from torchvision import datasets, transforms
from tqdm import tqdm

# ------------------------------------------------------------
# cryptGPU 加速的 MPC 加密 PLDK 演示
# 基于 pldk_train_v2.py，使用 cryptGPU 加速
# ------------------------------------------------------------

# 初始化 cryptGPU
crypten.init()
torch.set_num_threads(4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ConvNet_MPC(nn.Module):
    """cryptGPU 兼容的 ConvNet"""
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

def load_distilled_data():
    """加载蒸馏数据（与 pldk_train_v2 一致）"""
    data_path = "images_best.pt"
    if not os.path.exists(data_path):
        files = glob.glob("**/images_best.pt", recursive=True)
        if files:
            files.sort(key=os.path.getmtime, reverse=True)
            data_path = files[0]
            print(f"Found distilled data: {data_path}")
        else:
            raise FileNotFoundError("images_best.pt not found")

    images = torch.load(data_path).to(device)
    labels_path = data_path.replace('images_best.pt', 'labels_best.pt')
    labels = torch.load(labels_path).to(device)
    
    print(f"Loaded: {images.shape}, Labels: {labels.shape}")
    return images, labels

def load_teacher():
    """加载 Teacher 模型（与 pldk_train_v2 一致）"""
    teacher = resnet18(num_classes=10)
    teacher.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    teacher.maxpool = nn.Identity()
    
    teacher_path = 'teacher_resnet18_cifar10.pth'
    if not os.path.exists(teacher_path):
        files = glob.glob("**/teacher_resnet18_cifar10.pth", recursive=True)
        if files: teacher_path = files[0]
        else: raise FileNotFoundError("Teacher model not found")
    
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    return teacher.to(device).eval()

def get_test_loader():
    """测试集（与 pldk_train_v2 一致）"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    return torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False)

def evaluate(model, test_loader):
    """评估模型"""
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for imgs, labs in test_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            pred = model(imgs).argmax(dim=1)
            correct += pred.eq(labs).sum().item()
            total += labs.size(0)
    return 100. * correct / total

def mpc_pldk_demo_cryptgpu():
    """cryptGPU 加速的 MPC 加密 PLDK 演示"""
    print("=" * 60)
    print("cryptGPU Accelerated MPC Encrypted PLDK Demo")
    print("=" * 60)

    # 1. 加载数据（与 pldk_train_v2 一致）
    images, labels = load_distilled_data()
    test_loader = get_test_loader()

    # 2. 加载 Teacher 并计算 Logits（与 pldk_train_v2 一致）
    teacher = load_teacher()
    print("Computing Teacher Logits...")
    with torch.no_grad():
        teacher_logits = teacher(images)
    print(f"Teacher Logits: {teacher_logits.shape}")

    # 3. 加密数据（使用 cryptGPU）
    print("Encrypting data with cryptGPU...")
    images_enc = crypten.cryptensor(images)
    teacher_logits_enc = crypten.cryptensor(teacher_logits)
    print(f"Encrypted images: {images_enc.shape}")
    print(f"Encrypted logits: {teacher_logits_enc.shape}")

    # 4. 初始化加密 Student（使用 cryptGPU）
    print("Initializing encrypted student...")
    student_plain = ConvNet_MPC(width=128).to(device)
    dummy_input = torch.empty(1, 3, 32, 32).to(device)
    student = crypten.nn.from_pytorch(student_plain, dummy_input)
    student.encrypt()
    student.train()

    # 5. 训练配置（使用 cryptGPU 支持的操作）
    epochs = 50  # 快速验证
    batch_size = 32
    lr = 0.01
    num_samples = images.shape[0]
    
    # cryptGPU 支持的损失函数：MSE 或 CrossEntropy
    criterion = crypten.nn.MSELoss()

    print(f"\nTraining {epochs} epochs...")
    print("-" * 60)

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

    print("=" * 60)

if __name__ == "__main__":
    mpc_pldk_demo_cryptgpu()
```

#### 2.2.2 依赖配置更新 (`requirements.txt`)

```
pytorch
torchvision
kornia
scipy
tqdm
wandb
crypten  # cryptGPU 会替换原生 CrypTen
```

### 2.3 性能优化建议

#### 2.3.1 批处理大小优化
- **增大 batch_size**: cryptGPU 在 GPU 上处理更大的 batch 时效率更高
- **建议值**: 64-128（根据 GPU 内存调整）

#### 2.3.2 模型结构调整
- **简化模型**: 减少通道数或层数以加快加密计算
- **示例**: 将 `width` 从 128 减少到 64

#### 2.3.3 训练策略优化
- **学习率调度**: 使用 `CosineAnnealingLR` 进一步加速收敛
- **混合精度训练**: 如果 cryptGPU 支持，启用混合精度

## 3. 运行步骤

### 3.1 配置 cryptGPU

1. **修改 launcher.py**
   ```bash
   # 进入 CryptGPU 目录
   cd CryptGPU
   
   # 编辑 launcher.py
   nano launcher.py
   ```

   在文件中设置：
   ```python
   use_csprng = True  # 使用 torchcsprng 作为随机数生成器
   sync_key = False   # 从 pip 安装时设为 False，从源码构建时设为 True
   ```

### 3.2 运行演示脚本

1. **基本运行**
   ```bash
   # 在 mtt-distillation 目录下运行
   python mpc_pldk_demo_cryptgpu.py
   ```

2. **性能基准测试**
   ```bash
   # 进入 CryptGPU 目录
   cd CryptGPU
   
   # 运行基准测试
   python3 scripts/benchmark.py --exp train_all
   ```

## 4. 验证步骤

### 4.1 基础验证
1. **环境测试**: 运行 `python -c "import crypten; print('cryptGPU loaded successfully')"`
2. **小型测试**: 使用少量数据（100-200 样本）运行完整流程
3. **性能对比**: 比较原生 CrypTen 和 cryptGPU 的训练速度

### 4.2 完整验证
1. **完整数据集**: 使用全部蒸馏数据运行训练
2. **精度验证**: 确保加密训练的精度与明文训练相当
3. **性能评估**: 记录训练时间和 GPU 利用率

## 5. 故障排除

### 5.1 常见问题
1. **CUDA 版本不匹配**: 确保 CUDA 版本与 cryptGPU 要求一致
2. **内存不足**: 减小 batch_size 或模型大小
3. **操作不支持**: 确保只使用 cryptGPU 支持的操作（矩阵乘法、卷积、ReLU、平均池化、批量归一化、交叉熵损失）
4. **torchcsprng 错误**: 如果遇到 torchcsprng 相关错误，尝试在 launcher.py 中设置 `use_csprng = False`

### 5.2 解决方案
- **操作替换**: 将不支持的操作替换为支持的等效操作
- **模型简化**: 调整模型结构以仅使用支持的操作
- **分批处理**: 对于大型数据集，实现分批加密和训练
- **降级到非 CSPRNG**: 如果 torchcsprng 安装失败，使用 PyTorch 内置的随机数生成器

## 6. 预期性能提升

| 配置 | 原生 CrypTen (CPU) | cryptGPU (GPU) | 提升倍数 |
|------|-------------------|----------------|----------|
| 小批量 (32) | ~10-15 分钟/轮 | ~1-2 分钟/轮 | 5-15x |
| 大批量 (128) | ~8-12 分钟/轮 | ~30-45 秒/轮 | 10-24x |

## 7. 注意事项

1. **安全性**: cryptGPU 保持与 CrypTen 相同的安全保证，但请注意这是学术原型，不建议用于生产环境
2. **兼容性**: 仅支持三方参与 (3PC) 设置
3. **操作限制**: 仅支持有限的神经网络操作
4. **GPU 内存**: 需要足够的 GPU 内存来存储加密数据
5. **随机数生成**: 
   - 从源码构建时：使用 torchcsprng 并同步密钥，提供完全安全的随机数
   - 从 pip 安装时：可以选择使用 torchcsprng（输出可能不正确）或 PyTorch 内置 RNG（不完全安全）

## 8. 结论

cryptGPU 是加速 MPC 加密训练的理想选择，特别是对于基于卷积神经网络的知识蒸馏任务。通过本方案的集成，您可以显著减少训练时间，同时保持与原生 CrypTen 相同的安全级别。

集成步骤简单明了，主要是：
1. 安装 cryptGPU（快速安装或从源码构建）
2. 配置 launcher.py 中的运行参数
3. 使用提供的演示脚本运行加密训练

预期的性能提升可以达到 5-24 倍，具体取决于硬件配置和模型复杂度。对于您的知识蒸馏任务，cryptGPU 应该能够将训练时间从数小时缩短到数十分钟，大大加快实验迭代速度。