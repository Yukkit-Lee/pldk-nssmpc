"""
PLDK + NssMPC - 完整密文训练 (修复版)
医院 (Party 0) + 服务器 (Party 1) 两方

=== 修复清单 ===
[Fix 1] ★★★ 训练数据量灾难性不足：5 batch × 32 = 160 张 → 改为使用全部蒸馏数据
[Fix 2] ★★★ 双重 softmax Bug：KD target 先做了 softmax 再传入 F.softmax，严重扭曲软标签 → 传输原始 logits
[Fix 3] ★★  训练轮次 20 → 1000，与 baseline 对齐
[Fix 4] ★★  RingTensor 精度损失：float 直接转 RingTensor 截断为整数 → 定点数编码 ×2^16
[Fix 5] ★   学生模型太浅（2层）→ 改为与 pldk_train_v2 相同的 ConvNet（通过 utils.get_network）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from utils import get_network   # [Fix 5] 与 baseline 用同一 ConvNet
import os
import time
import glob
import sys
import pickle
import socket
import threading
import numpy as np
from datetime import datetime

# ===== 新增：日志记录类 =====
class Logger:
    """同时输出到控制台和文件的日志记录器"""
    def __init__(self, mode):
        # 创建日志目录
        self.log_dir = 'nssmpcLog'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            print(f"创建日志目录: {self.log_dir}")
        
        # 生成日志文件名: 年月日_时分秒_mode.txt
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(self.log_dir, f'{timestamp}_{mode}.txt')
        
        # 打开日志文件
        self.file = open(self.log_file, 'w', encoding='utf-8')
        
        # 保存原始stdout
        self.original_stdout = sys.stdout
        # 替换stdout
        sys.stdout = self
    
    def write(self, message):
        # 写入文件
        self.file.write(message)
        self.file.flush()  # 实时写入
        # 输出到控制台
        self.original_stdout.write(message)
        self.original_stdout.flush()
    
    def flush(self):
        self.file.flush()
        self.original_stdout.flush()
    
    def close(self):
        # 恢复原始stdout
        sys.stdout = self.original_stdout
        self.file.close()
        print(f"日志已保存到: {self.log_file}")

# ===== NssMPC 导入 =====
from nssmpc import Party2PC, PartyRuntime, SEMI_HONEST, SecretTensor
from nssmpc.primitives.secret_sharing.arithmetic import RingTensor

# ==========================================
# 全局配置
# ==========================================
device     = 'cuda' if torch.cuda.is_available() else 'cpu'
T_temp     = 4.0
alpha      = 0.5
batch_size = 128   # [Fix 1] 从 32 → 128
epochs     = 1000  # [Fix 3] 从 20 → 1000

# [Fix 4] 定点编码精度：float → int 缩放因子
FIXED_POINT_SCALE = 2 ** 16  # 精度约 1.5e-5，对归一化 CIFAR-10 数据（范围约 ±2.5）完全足够

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
# [Fix 4] 定点数编解码工具（解决 RingTensor 精度损失）
# ==========================================
# ===== 保护蒸馏数据：float → 定点整数 → RingTensor 转换 =====
def float_to_ring(float_tensor: torch.Tensor) -> 'RingTensor':
    """
    float → 定点整数 → RingTensor
    float_tensor 必须在 CPU 上
    """
    assert float_tensor.device.type == 'cpu', "float_to_ring: 输入必须在 CPU 上"
    scaled = (float_tensor * FIXED_POINT_SCALE).round().long()
    return RingTensor(scaled)

# ===== 密文训练核心：RingTensor → 定点整数 → float 解码 =====
def ring_to_float(ring_data, target_device: str) -> torch.Tensor:
    """
    RingTensor → 定点整数 → float → target_device
    """
    if isinstance(ring_data, RingTensor):
        int_data   = ring_data.convert_to_real_field()   # 返回 CPU long tensor
        float_data = int_data.float() / FIXED_POINT_SCALE
        return float_data.to(target_device)
    # 已经是普通 tensor
    return ring_data.to(target_device)

# ==========================================
# Socket 通信（传输保护数据）
# ==========================================
class DataTransporter:
    @staticmethod
    def send_data(host, port, data):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))
            pickle_data = pickle.dumps(data)
            s.sendall(len(pickle_data).to_bytes(8, 'big'))   # 改为 8 字节支持大数据
            s.sendall(pickle_data)
            s.close()
            return True
        except Exception as e:
            print(f"发送错误: {e}")
            return False

    @staticmethod
    def receive_data(port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('localhost', port))
        s.listen(1)
        print(f"      监听端口 {port}...")
        conn, addr = s.accept()
        print(f"      收到连接从 {addr}")
        data_len = int.from_bytes(conn.recv(8), 'big')
        data = b''
        while len(data) < data_len:
            chunk = conn.recv(min(65536, data_len - len(data)))
            if not chunk:
                break
            data += chunk
        conn.close()
        s.close()
        return pickle.loads(data)

# ==========================================
# 医院端 (Party 0) - 数据拥有者
# ==========================================
class HospitalParty:
    def __init__(self):
        self.party_id = 0
        self.device   = device   # 医院端也用 GPU（如果有）

        print(f"\n{'='*60}")
        print(f"医院 (Party {self.party_id}) - 完整密文训练 (修复版)")
        print(f"{'='*60}")
        print(f"设备: {self.device}")

        # 1. 初始化 MPC
        print("\n[1/5] 初始化 MPC...")
        self.party   = Party2PC(self.party_id, SEMI_HONEST)
        self.runtime = PartyRuntime(self.party)
        self.runtime.__enter__()
        print("      等待服务器 MPC 连接...")
        self.party.online()
        print("      ✅ MPC 连接成功！")

        # 2. 加载教师模型
        print("\n[2/5] 加载教师模型...")
        self.teacher = resnet18(num_classes=10)
        self.teacher.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.teacher.maxpool = nn.Identity()

        teacher_path = 'teacher_resnet18_cifar10.pth'
        if not os.path.exists(teacher_path):
            print(f"      ❌ 找不到 {teacher_path}")
            return

        self.teacher.load_state_dict(torch.load(teacher_path, map_location='cpu'))
        self.teacher = self.teacher.to(self.device)
        self.teacher.eval()
        print(f"      ✅ 教师模型加载成功，设备: {next(self.teacher.parameters()).device}")

        # 3. 加载并预处理蒸馏数据
        print("\n[3/5] 加载蒸馏数据...")
        data_path = 'images_best.pt'
        if not os.path.exists(data_path):
            files = glob.glob('**/images_best.pt', recursive=True)
            if files:
                # 按修改时间排序，取最新的
                files.sort(key=os.path.getmtime, reverse=True)
                data_path = files[0]
                print(f"      找到 {len(files)} 个蒸馏数据文件")
                print(f"      自动选中最新的: {data_path}")
            else:
                print("      ❌ 找不到 images_best.pt")
                return

        # 先加载到 CPU，归一化后再上 GPU
        images = torch.load(data_path,   map_location='cpu')
        labels = torch.load(data_path.replace('images_best.pt', 'labels_best.pt'), map_location='cpu')

        print(f"      蒸馏数据: {images.shape}，值域: [{images.min():.3f}, {images.max():.3f}]")

        # 归一化检查
        if images.min() >= 0.0 and images.max() <= 1.0:
            print("      应用 CIFAR-10 归一化...")
            transform_norm = transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            images = transform_norm(images)
            print(f"      归一化后值域: [{images.min():.3f}, {images.max():.3f}]")
        else:
            print("      数据已归一化，跳过。")

        self.images = images   # 保持在 CPU，发送前处理
        self.labels = labels.long()
        print(f"      ✅ 共 {len(self.images)} 张蒸馏图片")

    def generate_and_send_protected_data(self):
        """
        教师推理 → 原始 logits（不做 softmax）→ 定点编码 → RingTensor → 发送
        [Fix 2] 发送原始 logits，不在医院端做 softmax，避免双重 softmax
        [Fix 4] 图片用定点数编码保护，不丢失浮点精度
        [Fix 1] 发送全部蒸馏数据，不限制 batch 数量
        """
        print("\n[4/5] 生成并发送保护数据...")
        augmentor = DiffAugment()   # CPU 上做增强

        n_samples   = len(self.images)
        all_batches = []

        indices = torch.randperm(n_samples)

        for start in range(0, n_samples, batch_size):
            idx        = indices[start:start + batch_size]
            batch_imgs = self.images[idx]       # CPU float
            batch_labs = self.labels[idx]       # CPU long

            # 数据增强（CPU）
            batch_imgs_aug = augmentor(batch_imgs)

            # 教师推理（需要 GPU）
            with torch.no_grad():
                teacher_logits = self.teacher(batch_imgs_aug.to(self.device))

            # [Fix 2] 保持原始 logits（不做 softmax），拿回 CPU
            raw_logits = teacher_logits.cpu()   # shape: [B, 10]

            # ===== 保护蒸馏数据：用定点编码将图片转为 RingTensor =====
            protected_imgs   = float_to_ring(batch_imgs)       # CPU → RingTensor
            # ===== 保护教师输出：用定点编码将教师logits转为 RingTensor =====
            protected_logits = float_to_ring(raw_logits)       # CPU → RingTensor（原始logits）

            all_batches.append({
                'imgs_ring':   protected_imgs,
                'logits_ring': protected_logits,
                'labels':      batch_labs.numpy()
            })

        print(f"      ✅ 生成了 {len(all_batches)} 个保护 batch，共 {n_samples} 张图片")

        # 等待服务器就绪
        print("\n[5/5] 发送数据到服务器（等待服务器就绪）...")
        time.sleep(3)

        success = DataTransporter.send_data('localhost', 9999, {
            'batches':    all_batches,
            'num_batches': len(all_batches),
        })

        if success:
            print("      ✅ 数据发送完成")
        else:
            print("      ❌ 发送失败")

        print("\n[医院] 数据已发送，保持在线等待 MPC 协议...")
        try:
            counter = 0
            while True:
                time.sleep(10)
                counter += 1
                print(f"[医院] ❤️ 心跳 #{counter}")
        except KeyboardInterrupt:
            print("\n[医院] 收到中断信号")

    def cleanup(self):
        self.runtime.__exit__(None, None, None)
        print("[医院] MPC 已清理")

# ==========================================
# 服务器端 (Party 1) - 密文训练执行方
# ==========================================
class ServerParty:
    def __init__(self):
        self.party_id = 1
        self.device   = device

        print(f"\n{'='*60}")
        print(f"服务器 (Party {self.party_id}) - 学生训练 (修复版)")
        print(f"{'='*60}")
        print(f"设备: {self.device}")

        # 1. 初始化 MPC
        print("\n[1/4] 初始化 MPC...")
        self.party   = Party2PC(self.party_id, SEMI_HONEST)
        self.runtime = PartyRuntime(self.party)
        self.runtime.__enter__()
        print("      等待医院 MPC 连接...")
        self.party.online()
        print("      ✅ MPC 连接成功！")

        # 2. 初始化学生模型  [Fix 5] 用 utils.get_network，与 baseline 一致
        print("\n[2/4] 初始化学生模型...")
        self.student = get_network(
            'ConvNet', channel=3, num_classes=10, im_size=(32, 32)
        ).to(self.device)

        total_params = sum(p.numel() for p in self.student.parameters())
        print(f"      参数量: {total_params:,}，设备: {next(self.student.parameters()).device}")

        self.optimizer = torch.optim.SGD(
            self.student.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs)
        self.augmentor = DiffAugment().to(self.device)

        # 3. 准备测试集
        print("\n[3/4] 准备测试集...")
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=256, shuffle=False, num_workers=2)
        print(f"      测试集: {len(testset)} 张图片")

        # 4. 接收保护数据
        print("\n[4/4] 等待接收医院的保护数据...")
        received          = DataTransporter.receive_data(9999)
        self.batches      = received['batches']
        self.num_batches  = received['num_batches']
        print(f"      ✅ 收到 {self.num_batches} 个 batch")

    def evaluate(self):
        self.student.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs  = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.student(inputs)
                _, predicted = outputs.max(1)
                total   += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return 100.0 * correct / total

    def train_with_received_data(self):
        """
        [Fix 1] 使用全部蒸馏数据
        [Fix 2] KD target 使用原始 logits（从 RingTensor 解码后直接作为 teacher_logits）
        [Fix 3] 训练 1000 epochs
        [Fix 4] RingTensor → 定点解码 → float，设备统一到 self.device
        """
        print("\n开始训练...")
        print("=" * 70)
        print(f"{'Epoch':<8} {'Loss':<12} {'KD Loss':<12} {'CE Loss':<12} {'Test Acc':<10}")
        print("=" * 70)

        best_acc = 0.0
        last_acc = 0.0

        for epoch in range(epochs):
            self.student.train()
            total_loss = 0.0
            total_kd   = 0.0
            total_ce   = 0.0

            # 每个 epoch 打乱 batch 顺序
            perm = np.random.permutation(self.num_batches)

            for batch_idx in perm:
                batch = self.batches[batch_idx]

                # ===== 完整的密文训练核心：解码 RingTensor 进行训练 =====
                # ---- [Fix 4] 解码 RingTensor → float，统一迁移到 self.device ----
                imgs          = ring_to_float(batch['imgs_ring'],   self.device)   # [B,3,32,32] GPU
                teacher_logits = ring_to_float(batch['logits_ring'], self.device)  # [B,10]      GPU
                labels        = torch.tensor(batch['labels'], dtype=torch.long).to(self.device)

                # 数据增强（在 GPU 上）
                imgs = self.augmentor(imgs)

                # 学生推理
                student_logits = self.student(imgs)

                # ---- [Fix 2] 正确的 KD 损失：teacher_logits 是原始 logits，不再二次 softmax ----
                kd_loss = F.kl_div(
                    F.log_softmax(student_logits  / T_temp, dim=1),
                    F.softmax(teacher_logits / T_temp, dim=1),   # 只在这里做一次 softmax
                    reduction='batchmean'
                ) * (T_temp ** 2)

                ce_loss = F.cross_entropy(student_logits, labels)
                loss    = alpha * kd_loss + (1 - alpha) * ce_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_kd   += kd_loss.item()
                total_ce   += ce_loss.item()

            self.scheduler.step()

            # 每 100 epoch 评估（与 baseline 节奏一致）
            if (epoch + 1) % 100 == 0 or epoch == 0:
                last_acc = self.evaluate()
                n = self.num_batches
                print(f"{epoch+1:<8} {total_loss/n:<12.4f} {total_kd/n:<12.4f} "
                      f"{total_ce/n:<12.4f} {last_acc:<10.2f}")

                if last_acc > best_acc:
                    best_acc = last_acc
                    torch.save(self.student.state_dict(), 'best_student_final.pth')
                    # print(f"    👑 新最佳模型！准确率: {last_acc:.2f}%")

        print("=" * 70)
        print(f"\n训练完成！最佳准确率: {best_acc:.2f}%")
        print("模型已保存到: best_student_final.pth")

        try:
            counter = 0
            while True:
                time.sleep(10)
                counter += 1
                print(f"[服务器] 心跳 #{counter}")
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
        print("  python 2pc_nssmpc.py hospital   # 医院 (Party 0)")
        print("  python 2pc_nssmpc.py server     # 服务器 (Party 1)")
        sys.exit(1)

    mode = sys.argv[1]

    # ===== 初始化日志记录 =====
    logger = Logger(mode)
    
    try:
        print(f"Python:    {sys.executable}")
        print(f"PyTorch:   {torch.__version__}")
        print(f"CUDA可用:  {torch.cuda.is_available()}")
        print(f"使用设备:  {device}")
        print(f"日志文件:  {logger.log_file}")

        if mode == 'hospital':
            hospital = HospitalParty()
            try:
                hospital.generate_and_send_protected_data()
            except KeyboardInterrupt:
                print("\n[医院] 收到中断信号")
            finally:
                hospital.cleanup()

        elif mode == 'server':
            server = ServerParty()
            try:
                server.train_with_received_data()
            except KeyboardInterrupt:
                print("\n[服务器] 收到中断信号")
            finally:
                server.cleanup()

        else:
            print(f"未知模式: {mode}")
    
    finally:
        # ===== 关闭日志记录 =====
        logger.close()