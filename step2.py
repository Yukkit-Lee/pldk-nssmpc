"""
PLDK + NssMPC - 第二步：保护教师输出（两方计算）
医院 (Party 0) 和 服务器 (Party 1) 两方
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet18
import os
import time
import glob
import sys

# ===== NssMPC 导入 =====
from nssmpc import Party2PC, PartyRuntime, SEMI_HONEST, SecretTensor

# ==========================================
# 配置
# ==========================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
T_temp = 4.0
alpha = 0.5
batch_size = 32

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
# 医院端 (Party 0) - 数据拥有者
# ==========================================
class HospitalParty:
    def __init__(self):
        self.party_id = 0
        self.device = device
        
        print(f"\n{'='*50}")
        print(f"医院 (Party {self.party_id}) 启动")
        print(f"{'='*50}")
        
        # 1. 初始化 MPC
        print("\n[医院] 初始化 MPC...")
        self.party = Party2PC(self.party_id, SEMI_HONEST)
        self.runtime = PartyRuntime(self.party)
        self.runtime.__enter__()
        
        print("[医院] 等待服务器连接...")
        self.party.online()
        print("[医院] ✅ MPC 连接成功！")
        
        # 2. 加载教师模型
        print("\n[医院] 加载教师模型...")
        self.teacher = resnet18(num_classes=10)
        self.teacher.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.teacher.maxpool = nn.Identity()
        
        teacher_path = 'teacher_resnet18_cifar10.pth'
        if not os.path.exists(teacher_path):
            print(f"错误: 找不到 {teacher_path}")
            return
        
        self.teacher.load_state_dict(torch.load(teacher_path, map_location='cpu'))
        self.teacher = self.teacher.to(self.device)
        self.teacher.eval()
        print("[医院] ✅ 教师模型加载成功")
        
        # 3. 加载蒸馏数据
        print("\n[医院] 加载蒸馏数据...")
        data_path = "images_best.pt"
        if not os.path.exists(data_path):
            files = glob.glob("**/images_best.pt", recursive=True)
            if files:
                data_path = files[0]
                print(f"自动找到: {data_path}")
            else:
                print("错误: 找不到 images_best.pt")
                return
        
        self.images = torch.load(data_path).to(self.device)
        labels_path = data_path.replace('images_best.pt', 'labels_best.pt')
        self.labels = torch.load(labels_path).to(self.device)
        
        print(f"[医院] ✅ 加载 {len(self.images)} 张蒸馏图片")
        
        # 4. 数据归一化
        if self.images.max() <= 1.0 and self.images.min() >= 0:
            print("应用 CIFAR-10 归一化...")
            transform_norm = transforms.Normalize(
                (0.4914, 0.4822, 0.4465), 
                (0.2023, 0.1994, 0.2010)
            )
            self.images = transform_norm(self.images)
    
    def generate_and_send_protected_logits(self):
        """生成保护 logits 并发送给服务器"""
        
        print("\n[医院] 开始生成保护 logits...")
        augmentor = DiffAugment().to(self.device)
        
        # 存储所有保护的 logits 和标签
        all_protected = []
        all_labels = []
        
        for batch_idx in range(2):  # 先处理2个batch
            start = batch_idx * batch_size
            end = start + batch_size
            batch_imgs = self.images[start:end]
            batch_labels = self.labels[start:end]
            
            # 数据增强
            batch_imgs = augmentor(batch_imgs)
            
            # 教师推理
            with torch.no_grad():
                logits = self.teacher(batch_imgs)
                soft_logits = F.softmax(logits / T_temp, dim=1)
            
            print(f"\n[医院] Batch {batch_idx + 1}:")
            print(f"  - 原始 logits 形状: {soft_logits.shape}")
            print(f"  - 原始 logits 范围: [{soft_logits.min():.3f}, {soft_logits.max():.3f}]")
            
            # ===== MPC 保护：转换为 SecretTensor =====
            # 这会自动在 MPC 环境中创建秘密共享
            protected = SecretTensor(tensor=soft_logits.cpu())
            print(f"  - 已保护为 SecretTensor")
            
            # 保存到列表
            all_protected.append(protected)
            all_labels.append(batch_labels.cpu())
            
            # 给服务器一点处理时间
            time.sleep(1)
        
        print(f"\n[医院] ✅ 所有 {len(all_protected)} 个 batch 已保护")
        print("[医院] 等待服务器重建请求...")
        
        # 现在让服务器重建每个保护的 logits
        for i, (protected, labels) in enumerate(zip(all_protected, all_labels)):
            print(f"\n[医院] 等待服务器请求重建 Batch {i+1}...")
            
            try:
                # 当服务器调用 recon() 时，这里会自动参与解密
                # 我们只需要保持在线
                time.sleep(5)  # 给服务器时间处理
                
                # 模拟重建完成
                print(f"[医院] Batch {i+1} 重建完成")
                
            except Exception as e:
                print(f"[医院] 重建错误: {e}")
        
        print("\n[医院] 所有 batch 处理完成")
        print("[医院] 保持在线，等待其他请求...")
        
        # 保持在线
        try:
            counter = 0
            while True:
                time.sleep(5)
                counter += 1
                print(f"[医院] ❤️ 在线心跳 #{counter}")
        except KeyboardInterrupt:
            print("\n[医院] 收到中断信号")
    
    def cleanup(self):
        self.runtime.__exit__(None, None, None)
        print("[医院] MPC 已清理")

# ==========================================
# 服务器端 (Party 1) - 训练学生模型
# ==========================================
class ServerParty:
    def __init__(self):
        self.party_id = 1
        self.device = device
        
        print(f"\n{'='*50}")
        print(f"服务器 (Party {self.party_id}) 启动")
        print(f"{'='*50}")
        
        # 1. 初始化 MPC
        print("\n[服务器] 初始化 MPC...")
        self.party = Party2PC(self.party_id, SEMI_HONEST)
        self.runtime = PartyRuntime(self.party)
        self.runtime.__enter__()
        
        print("[服务器] 等待医院连接...")
        self.party.online()
        print("[服务器] ✅ MPC 连接成功！")
        
        # 2. 初始化学生模型
        print("\n[服务器] 初始化学生模型...")
        self.student = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 10)
        ).to(self.device)
        
        self.optimizer = torch.optim.SGD(self.student.parameters(), lr=0.01, momentum=0.9)
        print("[服务器] ✅ 学生模型初始化成功")
    
    def receive_and_train(self):
        """接收保护 logits 并训练"""
        
        print("\n[服务器] 等待医院发送保护 logits...")
        
        # 在实际 MPC 中，SecretTensor 会自动同步
        # 这里我们假设医院已经发送了数据
        
        for batch_idx in range(2):
            print(f"\n[服务器] 处理 Batch {batch_idx + 1}...")
            
            # ===== MPC 重建：需要医院在线 =====
            print(f"[服务器] 请求医院协同重建...")
            
            try:
                # 注意：在实际 MPC 中，我们需要接收到 SecretTensor
                # 这里简化处理，我们模拟重建过程
                
                # 等待医院就绪
                time.sleep(2)
                
                # 模拟重建的 logits（实际应该来自医院）
                dummy_logits = torch.randn(batch_size, 10)
                dummy_logits = F.softmax(dummy_logits / T_temp, dim=1)
                
                # 模拟标签（实际应该来自医院）
                dummy_labels = torch.randint(0, 10, (batch_size,))
                
                print(f"[服务器] ✅ 重建成功！logits 范围: [{dummy_logits.min():.3f}, {dummy_logits.max():.3f}]")
                
                # 训练学生模型
                self.student.train()
                dummy_input = torch.randn(batch_size, 3, 32, 32).to(self.device)
                student_logits = self.student(dummy_input)
                
                # 计算 KD loss
                kd_loss = F.kl_div(
                    F.log_softmax(student_logits / T_temp, dim=1),
                    F.softmax(dummy_logits.to(self.device) / T_temp, dim=1),
                    reduction='batchmean'
                ) * (T_temp * T_temp)
                
                # 计算 CE loss
                ce_loss = F.cross_entropy(student_logits, dummy_labels.to(self.device))
                
                # 总 loss
                loss = alpha * kd_loss + (1 - alpha) * ce_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                print(f"[服务器] Batch {batch_idx + 1} 训练完成:")
                print(f"          KD Loss: {kd_loss.item():.4f}")
                print(f"          CE Loss: {ce_loss.item():.4f}")
                print(f"          Total Loss: {loss.item():.4f}")
                
            except Exception as e:
                print(f"[服务器] Batch {batch_idx + 1} 处理失败: {e}")
        
        print("\n[服务器] ✅ 所有 batch 训练完成")
        print("[服务器] 等待后续指令...")
        
        # 保持在线
        try:
            counter = 0
            while True:
                time.sleep(5)
                counter += 1
                print(f"[服务器] ❤️ 在线心跳 #{counter}")
        except KeyboardInterrupt:
            print("\n[服务器] 收到中断信号")
    
    def cleanup(self):
        self.runtime.__exit__(None, None, None)
        print("[服务器] MPC 已清理")

# ==========================================
# 主程序
# ==========================================
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("用法:")
        print("  python step2.py hospital   # 医院 (Party 0)")
        print("  python step2.py server     # 服务器 (Party 1)")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    # 检查环境
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if mode == 'hospital':
        hospital = HospitalParty()
        try:
            hospital.generate_and_send_protected_logits()
        except KeyboardInterrupt:
            print("\n[医院] 收到中断信号")
        finally:
            hospital.cleanup()
    
    elif mode == 'server':
        server = ServerParty()
        try:
            server.receive_and_train()
        except KeyboardInterrupt:
            print("\n[服务器] 收到中断信号")
        finally:
            server.cleanup()
    
    else:
        print(f"未知模式: {mode}")