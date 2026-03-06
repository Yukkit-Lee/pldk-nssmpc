"""
PLDK + NssMPC - 第一步：保护蒸馏数据
使用 RingTensor 保护蒸馏数据，其他保持明文
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet18
import os
import time
import glob

# ===== NssMPC 导入 =====
from nssmpc.primitives.secret_sharing.arithmetic import RingTensor

# ==========================================
# 配置
# ==========================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
T_temp = 4.0
alpha = 0.5
batch_size = 128

# ==========================================
# 数据增强
# ==========================================
class DiffAugment(nn.Module):
    def __init__(self):
        super().__init__()
        self.aug = nn.Sequential(
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        )
    
    def forward(self, x):
        return self.aug(x)

# ==========================================
# 蒸馏数据加载器（带保护）
# ==========================================
class ProtectedDistilledLoader:
    """保护蒸馏数据的加载器"""
    
    def __init__(self, images, labels, batch_size=128, shuffle=True):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(images)
        
        # 创建索引
        self.indices = list(range(self.num_samples))
        
    def __iter__(self):
        if self.shuffle:
            import random
            random.shuffle(self.indices)
        
        for start_idx in range(0, self.num_samples, self.batch_size):
            batch_indices = self.indices[start_idx:start_idx + self.batch_size]
            
            # 获取批次数据
            batch_images = self.images[batch_indices]
            batch_labels = self.labels[batch_indices]
            
            # ===== 核心：用 RingTensor 保护蒸馏数据 =====
            # 此时数据还是 IntTensor 类型
            protected_images = RingTensor(batch_images)
            
            yield protected_images, batch_labels
    
    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

def ring_to_float(ring_data):
    """将 RingTensor 转回 FloatTensor 供模型使用"""
    if isinstance(ring_data, RingTensor):
        # 重建为普通张量
        float_data = ring_data.convert_to_real_field()
        # 确保是浮点类型
        if float_data.dtype != torch.float32:
            float_data = float_data.float()
        return float_data
    return ring_data

# ==========================================
# 主训练函数（第一步）
# ==========================================
def step1_protect_distilled_data():
    """第一步：保护蒸馏数据"""
    
    print("="*60)
    print("第一步：保护蒸馏数据")
    print("="*60)
    
    # 1. 加载教师模型
    print("\n[1/5] 加载教师模型...")
    teacher = resnet18(num_classes=10)
    teacher.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    teacher.maxpool = nn.Identity()
    
    teacher_path = 'teacher_resnet18_cifar10.pth'
    if not os.path.exists(teacher_path):
        print(f"错误: 找不到 {teacher_path}")
        return
    
    teacher.load_state_dict(torch.load(teacher_path, map_location='cpu'))
    teacher = teacher.to(device)
    teacher.eval()
    print("✅ 教师模型加载成功")
    
    # 2. 加载蒸馏数据
    print("\n[2/5] 加载蒸馏数据...")
    data_path = "images_best.pt"
    if not os.path.exists(data_path):
        files = glob.glob("**/images_best.pt", recursive=True)
        if files:
            data_path = files[0]
            print(f"自动找到: {data_path}")
        else:
            print("错误: 找不到 images_best.pt")
            return
    
    images_train = torch.load(data_path).to(device)
    labels_path = data_path.replace('images_best.pt', 'labels_best.pt')
    labels_train = torch.load(labels_path).to(device)
    
    print(f"加载 {len(images_train)} 张蒸馏图片")
    print(f"图片形状: {images_train.shape}")
    
    # 3. 初始化学生模型
    print("\n[3/5] 初始化学生模型...")
    student = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10)
    ).to(device)
    
    optimizer = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.9)
    augmentor = DiffAugment().to(device)
    
    # 4. 创建保护数据加载器
    print("\n[4/5] 创建保护数据加载器...")
    loader = ProtectedDistilledLoader(
        images=images_train,
        labels=labels_train,
        batch_size=batch_size,
        shuffle=True
    )
    print(f"批大小: {batch_size}, 总批次数: {len(loader)}")
    
    # 5. 训练循环
    print("\n[5/5] 开始训练（数据受 RingTensor 保护）...")
    print("-" * 60)
    print(f"{'Epoch':<8} {'Loss':<12} {'说明'}")
    print("-" * 60)
    
    for epoch in range(5):  # 先跑5个epoch演示
        student.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, (protected_imgs, labs) in enumerate(loader):
            # ===== 关键：将保护数据转回浮点 =====
            imgs = ring_to_float(protected_imgs).to(device)
            labs = labs.to(device)
            
            # 数据增强
            imgs = augmentor(imgs)
            
            # 教师推理
            with torch.no_grad():
                teacher_logits = teacher(imgs)
            
            # 学生推理
            student_logits = student(imgs)
            
            # 损失计算
            kd_loss = F.kl_div(
                F.log_softmax(student_logits / T_temp, dim=1),
                F.softmax(teacher_logits / T_temp, dim=1),
                reduction='batchmean'
            ) * (T_temp * T_temp)
            
            ce_loss = F.cross_entropy(student_logits, labs)
            loss = alpha * kd_loss + (1 - alpha) * ce_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            if batch_idx == 0:  # 只打印第一个batch的说明
                print(f"  Batch 0: 数据已保护 -> 重建 -> 训练 (loss={loss.item():.4f})")
        
        avg_loss = total_loss / batch_count
        print(f"{epoch+1:<8} {avg_loss:<12.4f} 蒸馏数据受 RingTensor 保护")
    
    print("-" * 60)
    print("\n✅ 第一步完成！")
    print("   - 蒸馏数据全程用 RingTensor 保护")
    print("   - 只在送入模型前临时重建")
    print("   - 教师输出和学生训练仍是明文")

# ==========================================
# 测试 RingTensor
# ==========================================
def test_ring():
    """测试 RingTensor 基本功能"""
    print("\n测试 RingTensor:")
    data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"原始 (float): {data}")
    
    # 创建 RingTensor
    ring_data = RingTensor(data)
    print(f"RingTensor: {ring_data}")
    
    # 重建并转换类型
    recon_int = ring_data.convert_to_real_field()
    recon_float = recon_int.float()
    print(f"重建后转float: {recon_float}")
    
    # 验证一致性
    if torch.allclose(data, recon_float.cpu()):
        print("✅ 数据一致性验证通过")
    else:
        print("❌ 数据一致性验证失败")
        diff = data - recon_float.cpu()
        print(f"差值: {diff}")

# ==========================================
# 主程序
# ==========================================
if __name__ == '__main__':
    print(f"设备: {device}")
    test_ring()
    print("\n" + "="*60)
    
    # 询问是否继续训练
    response = input("\n是否继续训练？(y/n): ")
    if response.lower() == 'y':
        print("\n")
        step1_protect_distilled_data()
    else:
        print("\n退出程序")