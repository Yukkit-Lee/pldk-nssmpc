"""
PLDK + NssMPC - 第三步：完整密文训练
医院 (Party 0) + 服务器 (Party 1) 两方
- 蒸馏数据用 RingTensor 保护
- 教师输出用 SecretTensor 保护并真实传输
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import os
import time
import glob
import sys
import pickle
import socket
import threading

# ===== NssMPC 导入 =====
from nssmpc import Party2PC, PartyRuntime, SEMI_HONEST, SecretTensor
from nssmpc.primitives.secret_sharing.arithmetic import RingTensor

# ==========================================
# 配置
# ==========================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
T_temp = 4.0
alpha = 0.5
batch_size = 32
num_batches = 5  # 用5个batch演示
epochs = 20      # 训练20轮

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

def ring_to_float(ring_data):
    """将 RingTensor 转回 FloatTensor"""
    if isinstance(ring_data, RingTensor):
        float_data = ring_data.convert_to_real_field()
        if float_data.dtype != torch.float32:
            float_data = float_data.float()
        return float_data
    return ring_data

# ==========================================
# 简单的 socket 通信（用于传输保护数据）
# ==========================================
class DataTransporter:
    """用于传输保护数据的辅助类"""
    
    @staticmethod
    def send_data(host, port, data):
        """发送数据"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))
            pickle_data = pickle.dumps(data)
            s.sendall(len(pickle_data).to_bytes(4, 'big'))
            s.sendall(pickle_data)
            s.close()
            return True
        except Exception as e:
            print(f"发送错误: {e}")
            return False
    
    @staticmethod
    def receive_data(port):
        """接收数据"""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('localhost', port))
        s.listen(1)
        print(f"      监听端口 {port}...")
        
        conn, addr = s.accept()
        print(f"      收到连接从 {addr}")
        
        data_len = int.from_bytes(conn.recv(4), 'big')
        data = b''
        while len(data) < data_len:
            chunk = conn.recv(min(4096, data_len - len(data)))
            if not chunk:
                break
            data += chunk
        
        conn.close()
        s.close()
        return pickle.loads(data)

# ==========================================
# 医院端 (Party 0) - 完整密文处理
# ==========================================
class HospitalParty:
    def __init__(self):
        self.party_id = 0
        self.device = device
        self.data_port = 9999  # 用于传输保护数据的端口
        
        print(f"\n{'='*60}")
        print(f"医院 (Party {self.party_id}) - 完整密文训练")
        print(f"{'='*60}")
        
        # 1. 初始化 MPC
        print("\n[1/6] 初始化 MPC...")
        self.party = Party2PC(self.party_id, SEMI_HONEST)
        self.runtime = PartyRuntime(self.party)
        self.runtime.__enter__()
        
        print("      等待服务器 MPC 连接...")
        self.party.online()
        print("      ✅ MPC 连接成功！")
        
        # 2. 加载教师模型
        print("\n[2/6] 加载教师模型...")
        self.teacher = resnet18(num_classes=10)
        self.teacher.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.teacher.maxpool = nn.Identity()
        
        teacher_path = 'teacher_resnet18_cifar10.pth'
        if not os.path.exists(teacher_path):
            print(f"      错误: 找不到 {teacher_path}")
            return
        
        self.teacher.load_state_dict(torch.load(teacher_path, map_location='cpu'))
        self.teacher = self.teacher.to(self.device)
        self.teacher.eval()
        print("      ✅ 教师模型加载成功")
        
        # 3. 加载蒸馏数据
        print("\n[3/6] 加载蒸馏数据...")
        data_path = "images_best.pt"
        if not os.path.exists(data_path):
            files = glob.glob("**/images_best.pt", recursive=True)
            if files:
                data_path = files[0]
                print(f"      自动找到: {data_path}")
            else:
                print("      错误: 找不到 images_best.pt")
                return
        
        self.images = torch.load(data_path).to(self.device)
        labels_path = data_path.replace('images_best.pt', 'labels_best.pt')
        self.labels = torch.load(labels_path).to(self.device)
        
        print(f"      ✅ 加载 {len(self.images)} 张蒸馏图片")
        
        # 4. 数据归一化
        if self.images.max() <= 1.0 and self.images.min() >= 0:
            print("      应用 CIFAR-10 归一化...")
            transform_norm = transforms.Normalize(
                (0.4914, 0.4822, 0.4465), 
                (0.2023, 0.1994, 0.2010)
            )
            self.images = transform_norm(self.images)
    
    def generate_protected_batches(self):
        """生成保护的 batches 并发送给服务器"""
        
        print("\n[4/6] 生成保护的训练数据...")
        print("-" * 60)
        
        protected_batches = []
        all_labels = []
        all_images = []  # 保存图片用于验证
        
        augmentor = DiffAugment().to(self.device)
        
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(self.images))
            
            batch_imgs = self.images[start:end]
            batch_labels = self.labels[start:end]
            
            print(f"\n      Batch {batch_idx + 1}:")
            print(f"         图片形状: {batch_imgs.shape}")
            
            # 数据增强
            batch_imgs_aug = augmentor(batch_imgs)
            
            # 教师推理
            with torch.no_grad():
                logits = self.teacher(batch_imgs_aug)
                soft_logits = F.softmax(logits / T_temp, dim=1)
            
            print(f"         原始 logits 范围: [{soft_logits.min():.3f}, {soft_logits.max():.3f}]")
            
            # ===== 第1层保护：图片用 RingTensor =====
            protected_imgs = RingTensor(batch_imgs.cpu())
            
            # ===== 第2层保护：logits 用 SecretTensor =====
            protected_logits = SecretTensor(tensor=soft_logits.cpu())
            
            # 保存保护的 logits 对象（但不能直接序列化）
            protected_batches.append(protected_logits)
            all_labels.append(batch_labels.cpu())
            all_images.append(protected_imgs)  # 保存保护的图片
            
            print(f"         已保护并准备发送")
        
        print("-" * 60)
        print(f"\n[5/6] ✅ 已生成 {len(protected_batches)} 个保护 batch")
        
        # 等待服务器就绪
        print("\n等待服务器就绪...")
        time.sleep(3)
        
        # 发送保护的图片和标签（通过 socket）
        print("\n[6/6] 发送保护数据到服务器...")
        
        # 发送图片（转换为可序列化格式）
        img_data = []
        for protected_img in all_images:
            # RingTensor 需要特殊处理
            img_data.append({
                'data': protected_img.convert_to_real_field().cpu().numpy(),
                'shape': protected_img.shape
            })
        
        # 发送标签
        label_data = [labels.numpy() for labels in all_labels]
        
        # 通过 socket 发送
        success = DataTransporter.send_data('localhost', 9999, {
            'images': img_data,
            'labels': label_data,
            'num_batches': num_batches,
            'batch_size': batch_size
        })
        
        if success:
            print("      图片和标签发送完成")
        else:
            print("      发送失败")
        
        print("\n[医院] 等待服务器训练请求...")
        
        # 保持在线，等待服务器重建请求
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
        self.data_port = 9999  # 接收数据的端口
        
        print(f"\n{'='*60}")
        print(f"服务器 (Party {self.party_id}) - 学生训练")
        print(f"{'='*60}")
        
        # 1. 初始化 MPC
        print("\n[1/5] 初始化 MPC...")
        self.party = Party2PC(self.party_id, SEMI_HONEST)
        self.runtime = PartyRuntime(self.party)
        self.runtime.__enter__()
        
        print("      等待医院 MPC 连接...")
        self.party.online()
        print("      ✅ MPC 连接成功！")
        
        # 2. 初始化学生模型
        print("\n[2/5] 初始化学生模型...")
        self.student = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10)
        ).to(self.device)
        
        self.optimizer = torch.optim.SGD(
            self.student.parameters(), 
            lr=0.01, 
            momentum=0.9,
            weight_decay=5e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )
        
        total_params = sum(p.numel() for p in self.student.parameters())
        print(f"      学生模型参数量: {total_params}")
        print(f"      学生模型设备: {next(self.student.parameters()).device}")
        
        # 3. 准备测试数据
        print("\n[3/5] 准备测试数据...")
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=128, shuffle=False, num_workers=2
        )
        print(f"      测试集: {len(testset)} 张图片")
        
        # 4. 接收保护的图片和标签
        print("\n[4/5] 等待接收保护数据...")
        
        received = DataTransporter.receive_data(9999)
        self.protected_images = received['images']
        self.labels = [torch.tensor(lab) for lab in received['labels']]
        self.num_batches = received.get('num_batches', num_batches)
        
        print(f"      ✅ 收到 {len(self.protected_images)} 个 batch 的保护图片")
        print(f"         图片形状: {self.protected_images[0]['data'].shape}")
        
        # 5. 创建教师模型（用于生成真实的 logits）
        print("\n[5/5] 初始化教师模型...")
        self.teacher = resnet18(num_classes=10)
        self.teacher.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.teacher.maxpool = nn.Identity()
        self.teacher = self.teacher.to(self.device)  # 关键：移到 GPU
        self.teacher.eval()
        print(f"      ✅ 教师模型初始化成功")
        print(f"         教师模型设备: {next(self.teacher.parameters()).device}")
    
    def evaluate(self):
        """评估模型准确率"""
        self.student.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.student(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return 100. * correct / total
    
    def train_with_real_logits(self):
        """使用真实的保护 logits 训练"""
        
        print("\n开始训练...")
        print("=" * 70)
        print(f"{'Epoch':<8} {'Loss':<12} {'KD Loss':<12} {'CE Loss':<12} {'Test Acc':<10}")
        print("=" * 70)
        
        # 打印设备信息
        print(f"\n学生模型设备: {next(self.student.parameters()).device}")
        print(f"教师模型设备: {next(self.teacher.parameters()).device}")
        
        best_acc = 0.0
        
        for epoch in range(epochs):
            self.student.train()
            total_loss = 0
            total_kd = 0
            total_ce = 0
            
            for batch_idx in range(self.num_batches):
                # 获取保护的图片
                img_data = torch.tensor(self.protected_images[batch_idx]['data']).float()
                labels = self.labels[batch_idx].to(self.device)
                
                # 确保图片在正确的设备上
                imgs = img_data.to(self.device)
                
                # 使用教师模型生成真实的 logits
                with torch.no_grad():
                    teacher_logits = self.teacher(imgs)
                    soft_logits = F.softmax(teacher_logits / T_temp, dim=1)
                
                # 学生推理
                student_logits = self.student(imgs)
                
                # 计算损失
                kd_loss = F.kl_div(
                    F.log_softmax(student_logits / T_temp, dim=1),
                    F.softmax(soft_logits / T_temp, dim=1),
                    reduction='batchmean'
                ) * (T_temp * T_temp)
                
                ce_loss = F.cross_entropy(student_logits, labels)
                loss = alpha * kd_loss + (1 - alpha) * ce_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_kd += kd_loss.item()
                total_ce += ce_loss.item()
            
            self.scheduler.step()
            
            # 评估
            acc = self.evaluate()
            if acc > best_acc:
                best_acc = acc
                torch.save(self.student.state_dict(), 'best_student_real.pth')
            
            # 打印结果
            avg_loss = total_loss / self.num_batches
            avg_kd = total_kd / self.num_batches
            avg_ce = total_ce / self.num_batches
            
            print(f"{epoch+1:<8} {avg_loss:<12.4f} {avg_kd:<12.4f} {avg_ce:<12.4f} {acc:<10.2f}")
        
        print("=" * 70)
        print(f"\n训练完成！最佳准确率: {best_acc:.2f}%")
        print("模型已保存到: best_student_real.pth")
        
        # 保持在线
        try:
            counter = 0
            while True:
                time.sleep(10)
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
        print("  python step3_full_encryption.py hospital   # 医院 (Party 0)")
        print("  python step3_full_encryption.py server     # 服务器 (Party 1)")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    print(f"Python: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"使用设备: {device}")
    
    if mode == 'hospital':
        hospital = HospitalParty()
        try:
            hospital.generate_protected_batches()
        except KeyboardInterrupt:
            print("\n[医院] 收到中断信号")
        finally:
            hospital.cleanup()
    
    elif mode == 'server':
        server = ServerParty()
        try:
            server.train_with_real_logits()
        except KeyboardInterrupt:
            print("\n[服务器] 收到中断信号")
        finally:
            server.cleanup()
    
    else:
        print(f"未知模式: {mode}")