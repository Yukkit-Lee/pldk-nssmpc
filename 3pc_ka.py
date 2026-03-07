"""
PLDK + NssMPC - 3方完整密文训练 + Knowledge Alignment
=======================================================================
参与方:
  Party 0 - 医院A : 用自己的数据集 data_A 本地训练 Teacher_A
                    蒸馏得到 images01_best.pt，加密发给服务器
  Party 1 - 医院B : 用自己的数据集 data_B 本地训练 Teacher_B
                    蒸馏得到 images02_best.pt，加密发给服务器
  Party 2 - 服务器: MPC解密重构 -> Logits_A + Logits_B
                    -> Knowledge Alignment -> AlignedLogits
                    -> Student Training -> Global Model

  注意: 医院A 和 医院B 的训练数据集不同，因此教师模型也不同：
        teacher_a_resnet18.pth  (由医院A用 data_A 训练)
        teacher_b_resnet18.pth  (由医院B用 data_B 训练)
        两者权重不同，logits 分布存在尺度/置信度偏差，这正是 KA 要解决的问题。

运行方式（三个终端，建议先起服务器）:
  python 3pc_ka.py server      # 服务器 (Party 2)
  python 3pc_ka.py hospital_a  # 医院A  (Party 0)
  python 3pc_ka.py hospital_b  # 医院B  (Party 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from utils import get_network   # [Fix 5]
import os
import sys
import time
import glob
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
        self.log_dir = 'nssmpcLog_3pc_KA'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            print(f"创建日志目录: {self.log_dir}")
        
        # 生成日志文件名: 年月日_时分秒_mode.txt
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 根据 mode 生成不同的文件名
        if mode == 'server':
            log_name = f'{timestamp}_server.txt'
        elif mode == 'hospital_a':
            log_name = f'{timestamp}_hospitalA.txt'
        elif mode == 'hospital_b':
            log_name = f'{timestamp}_hospitalB.txt'
        else:
            log_name = f'{timestamp}_{mode}.txt'
            
        self.log_file = os.path.join(self.log_dir, log_name)
        
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

# ============================================================
# 全局配置
# ============================================================
device     = 'cuda' if torch.cuda.is_available() else 'cpu'
T_temp     = 4.0
alpha      = 0.5
batch_size = 128
epochs     = 1000   # [Fix 3]

FIXED_POINT_SCALE = 2 ** 16   # [Fix 4] 精度约 1.5e-5

# 通信端口
PORT_A_TO_SERVER = 9991   # 医院A -> 服务器
PORT_B_TO_SERVER = 9992   # 医院B -> 服务器

# ============================================================
# 数据增强
# ============================================================
class DiffAugment(nn.Module):
    def __init__(self):
        super().__init__()
        self.aug = nn.Sequential(
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        )

    def forward(self, x):
        return self.aug(x)

# ============================================================
# 定点数编解码  [Fix 4]
# ============================================================
def float_to_ring(float_tensor: torch.Tensor) -> RingTensor:
    """float (CPU) -> 定点整数 -> RingTensor"""
    assert float_tensor.device.type == 'cpu', \
        "[float_to_ring] 输入必须在 CPU，当前: " + str(float_tensor.device)
    scaled = (float_tensor * FIXED_POINT_SCALE).round().long()
    return RingTensor(scaled)


def ring_to_float(ring_data, target_device: str) -> torch.Tensor:
    """RingTensor -> 定点整数 -> float -> target_device  [Fix 6]"""
    if isinstance(ring_data, RingTensor):
        int_data   = ring_data.convert_to_real_field()          # CPU long tensor
        float_data = int_data.float() / FIXED_POINT_SCALE
        return float_data.to(target_device)
    return ring_data.to(target_device)

# ============================================================
# Socket 通信（支持大体积数据）
# ============================================================
class DataTransporter:
    HEADER = 8   # 8 字节长度头，支持超大数据包

    @staticmethod
    def send_data(host: str, port: int, obj) -> bool:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(300)
            s.connect((host, port))
            data = pickle.dumps(obj)
            s.sendall(len(data).to_bytes(DataTransporter.HEADER, 'big'))
            s.sendall(data)
            s.close()
            return True
        except Exception as e:
            print(f"  [Socket 发送失败] {e}")
            return False

    @staticmethod
    def receive_data(port: int):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.settimeout(600)
        s.bind(('0.0.0.0', port))
        s.listen(1)
        print(f"      监听端口 {port}...")
        conn, addr = s.accept()
        print(f"      收到连接来自 {addr}")
        length = int.from_bytes(conn.recv(DataTransporter.HEADER), 'big')
        buf = b''
        while len(buf) < length:
            chunk = conn.recv(min(65536, length - len(buf)))
            if not chunk:
                break
            buf += chunk
        conn.close()
        s.close()
        return pickle.loads(buf)

# ============================================================
# ★ Knowledge Alignment 模块（核心新增）
# ============================================================
class KnowledgeAligner:
    """
    解决多教师冲突的 Knowledge Alignment 模块。

    问题背景：
      医院A 用 data_A 训练 Teacher_A，医院B 用 data_B 训练 Teacher_B。
      两个教师在不同数据上学到了不同的特征偏好，导致：
        - 尺度偏差：Teacher_A 的 logits 整体可能比 Teacher_B 大/小
        - 置信度偏差：Teacher_A 对某些类别置信度高，Teacher_B 对另一些类别置信度高
        - 语义冲突：同一张图两个教师给出截然不同的软标签
      直接混合训练会让学生收到矛盾信号，精度下降。

    三步对齐策略：
      Step 1  统计对齐（Statistical Alignment）
              对每方 logits 做 Z-score 归一化，映射到全局 canonical 空间。
              消除 Teacher_A 和 Teacher_B 因训练数据不同导致的均值/方差差异。
              aligned = (logits - mean_src) / std_src * std_global + mean_global

      Step 2  置信度加权融合（Confidence-Weighted Blending）
              计算每个教师在每个类别上的平均置信度。
              低置信类别的 logits 噪声多，向全局均值收缩以减弱其影响。
              conf_weight[c] = conf_src[c] / (conf_src[c] + conf_other[c])
              aligned = conf_weight * aligned + (1 - conf_weight) * mean_global

      Step 3  标签条件修正（Label-Conditional Correction）
              对 ground-truth 类别的 logit 轻微上调（+0.5σ），
              防止对齐后出现软标签与硬标签语义倒置。

    fit()   训练前一次性调用，开销集中在此处，训练循环中无额外开销。
    align() 每个 batch 调用，仅矩阵加减乘除，耗时可忽略不计。
    """

    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
        self.fitted      = False
        # 统计量全部在 target_device 上 [Fix 6]
        self.mean_a = self.std_a = None
        self.mean_b = self.std_b = None
        self.mean_global = self.std_global = None
        self.conf_a = self.conf_b = None

    @torch.no_grad()
    def fit(self, batches_a: list, batches_b: list, dev: str):
        """
        从两方全量 RingTensor logits 中计算对齐统计量，仅调用一次。

        参数：
          batches_a : 医院A 所有 batch（含 logits_ring，来自 Teacher_A）
          batches_b : 医院B 所有 batch（含 logits_ring，来自 Teacher_B）
          dev       : 统计量存放设备
        """
        print("\n  [KA] 计算 Knowledge Alignment 统计量...")
        print("       Teacher_A 由医院A 用 data_A 独立训练")
        print("       Teacher_B 由医院B 用 data_B 独立训练")
        print("       两者权重不同，logits 分布存在偏差，正在统计...")

        # ---- 解码两方全量 logits（CPU 操作，节省显存）----
        pool_a = [ring_to_float(b['logits_ring'], 'cpu') for b in batches_a]
        pool_b = [ring_to_float(b['logits_ring'], 'cpu') for b in batches_b]

        logits_a   = torch.cat(pool_a, dim=0)                       # [N_a, C]
        logits_b   = torch.cat(pool_b, dim=0)                       # [N_b, C]
        logits_all = torch.cat([logits_a, logits_b], dim=0)         # [N_a+N_b, C]

        # ---- 各方统计量 ----
        mean_a = logits_a.mean(dim=0);   std_a = logits_a.std(dim=0).clamp(min=1e-6)
        mean_b = logits_b.mean(dim=0);   std_b = logits_b.std(dim=0).clamp(min=1e-6)
        mean_global = logits_all.mean(dim=0)
        std_global  = logits_all.std(dim=0).clamp(min=1e-6)

        # ---- 各类别平均置信度 ----
        conf_a = F.softmax(logits_a / T_temp, dim=1).mean(dim=0)   # [C]
        conf_b = F.softmax(logits_b / T_temp, dim=1).mean(dim=0)   # [C]

        # ---- 迁移到目标设备 [Fix 6] ----
        self.mean_a      = mean_a.to(dev)
        self.std_a       = std_a.to(dev)
        self.mean_b      = mean_b.to(dev)
        self.std_b       = std_b.to(dev)
        self.mean_global = mean_global.to(dev)
        self.std_global  = std_global.to(dev)
        self.conf_a      = conf_a.to(dev)
        self.conf_b      = conf_b.to(dev)
        self.fitted      = True

        # ---- 打印对齐报告 ----
        print(f"  [KA] 统计量计算完成：")
        print(f"       Teacher_A（data_A 训练）logits — "
              f"均值:{logits_a.mean():.4f}  标准差:{logits_a.std():.4f}")
        print(f"       Teacher_B（data_B 训练）logits — "
              f"均值:{logits_b.mean():.4f}  标准差:{logits_b.std():.4f}")
        print(f"       全局空间                       — "
              f"均值:{logits_all.mean():.4f}  标准差:{logits_all.std():.4f}")
        print(f"  [KA] 各类别置信度对比（Teacher_A / Teacher_B）：")
        for c in range(self.num_classes):
            diff   = abs(conf_a[c].item() - conf_b[c].item())
            marker = "  <- 冲突" if diff > 0.05 else ""
            print(f"       Class {c:2d}: "
                  f"A={conf_a[c]:.4f}  B={conf_b[c]:.4f}{marker}")

    def align(self,
              logits: torch.Tensor,
              source: str,
              labels: torch.Tensor) -> torch.Tensor:
        """
        对单个 batch 的 logits 执行三步对齐，返回 aligned_logits。

        参数：
          logits  : [B, C]  原始教师 logits，已在 target_device 上
          source  : 'a' 表示来自 Teacher_A，'b' 表示来自 Teacher_B
          labels  : [B]     ground-truth 标签，已在 target_device 上

        返回：
          aligned_logits : [B, C]，与输入同设备
        """
        assert self.fitted, "请先调用 aligner.fit() 再调用 align()"
        assert logits.device == labels.device, (
            f"[KA] logits({logits.device}) 与 labels({labels.device}) 设备不一致")

        # ---- Step 1：统计对齐 ----
        # 把 Teacher_A 或 Teacher_B 的 logits 归一化到全局 canonical 空间
        # 消除两教师因训练数据不同导致的尺度和均值偏差
        if source == 'a':
            src_mean, src_std = self.mean_a, self.std_a
            src_conf, oth_conf = self.conf_a, self.conf_b
        else:
            src_mean, src_std = self.mean_b, self.std_b
            src_conf, oth_conf = self.conf_b, self.conf_a

        aligned = (logits - src_mean) / src_std * self.std_global + self.mean_global

        # ---- Step 2：置信度加权融合 ----
        # 该教师擅长的类别（高置信度）：保留对齐后的 logits
        # 该教师不擅长的类别（低置信度）：向全局均值收缩，减少噪声传播
        conf_w = src_conf / (src_conf + oth_conf + 1e-8)   # [C]
        aligned = (conf_w.unsqueeze(0) * aligned
                   + (1.0 - conf_w).unsqueeze(0) * self.mean_global.unsqueeze(0))

        # ---- Step 3：标签条件修正 ----
        # 对 ground-truth 类别的 logit 轻微上调（+0.5σ）
        # 防止对齐操作后软标签与硬标签语义方向倒置
        for i in range(logits.size(0)):
            c = labels[i].item()
            aligned[i, c] = aligned[i, c] + 0.5 * self.std_global[c]

        return aligned   # [B, C]，与输入 logits 同设备


# ============================================================
# 医院A (Party 0)  —— images01_best.pt
# ============================================================
class HospitalA:
    def __init__(self):
        self.party_id = 0
        self.dev      = device

        print(f"\n{'='*60}")
        print(f"医院A (Party {self.party_id}) - images01_best.pt")
        print(f"{'='*60}")
        print(f"设备: {self.dev}")

        # ---- MPC 初始化 ----
        print("\n[1/5] 初始化 MPC...")
        self.party   = Party2PC(self.party_id, SEMI_HONEST)
        self.runtime = PartyRuntime(self.party)
        self.runtime.__enter__()
        print("      等待对端 MPC 连接...")
        self.party.online()
        print("      MPC 连接成功！")

        # ---- 加载教师模型 ----
        print("\n[2/5] 加载教师模型 Teacher_A（由医院A 用 data_A 本地独立训练）...")
        self.teacher = resnet18(num_classes=10)
        self.teacher.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.teacher.maxpool = nn.Identity()

        # 医院A 专属教师模型，权重文件由医院A 用自己的数据 data_A 训练产生
        # 与医院B 的 teacher_b_resnet18.pth 完全独立，权重不同
        teacher_path = 'teacher_a_resnet18.pth'
        if not os.path.exists(teacher_path):
            print(f"      找不到 {teacher_path}")
            print(f"      请先运行: python train_teacher.py --party a  生成该文件")
            sys.exit(1)

        # map_location='cpu' 加载，再显式上 GPU [Fix 6]
        self.teacher.load_state_dict(torch.load(teacher_path, map_location='cpu'))
        self.teacher = self.teacher.to(self.dev)
        self.teacher.eval()
        print(f"      Teacher_A 加载成功，设备: {next(self.teacher.parameters()).device}")

        # ---- 加载蒸馏数据 ----
        print("\n[3/5] 加载蒸馏数据 images01_best.pt（由 data_A 蒸馏产生）...")
        data_path = 'images01_best.pt'
        if not os.path.exists(data_path):
            files = glob.glob('**/images01_best.pt', recursive=True)
            if files:
                data_path = files[0]
                print(f"      自动找到: {data_path}")
            else:
                print("      找不到 images01_best.pt")
                sys.exit(1)

        # 全程 CPU；推理时才上 GPU，推完立刻拿回 CPU [Fix 6]
        images = torch.load(data_path, map_location='cpu')
        labels = torch.load(
            data_path.replace('images01_best.pt', 'labels01_best.pt'),
            map_location='cpu')

        print(f"      数据形状: {images.shape}，值域: [{images.min():.3f}, {images.max():.3f}]")

        if images.min() >= 0.0 and images.max() <= 1.0:
            print("      应用 CIFAR-10 归一化...")
            images = transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(images)
            print(f"      归一化后值域: [{images.min():.3f}, {images.max():.3f}]")
        else:
            print("      数据已归一化，跳过。")

        self.images = images          # CPU float
        self.labels = labels.long()   # CPU long
        print(f"      共 {len(self.images)} 张蒸馏图片")

    def generate_and_send(self):
        """
        Teacher 推理 -> 原始 logits [Fix 2]
        -> 定点编码 -> RingTensor [Fix 4]
        -> 全量发送 [Fix 1]
        """
        print("\n[4/5] 生成并加密训练数据...")
        augmentor   = DiffAugment()   # CPU 增强
        n           = len(self.images)
        perm        = torch.randperm(n)
        all_batches = []

        for start in range(0, n, batch_size):
            idx    = perm[start:start + batch_size]
            b_imgs = self.images[idx]   # CPU
            b_labs = self.labels[idx]   # CPU

            # 增强在 CPU，推理时上 GPU，结果立刻拿回 CPU [Fix 6]
            b_imgs_aug = augmentor(b_imgs)
            with torch.no_grad():
                raw_logits = self.teacher(b_imgs_aug.to(self.dev)).cpu()  # [B,10] CPU

            # [Fix 4] 定点编码：保护蒸馏数据 & 保护教师输出
            all_batches.append({
                'imgs_ring':   float_to_ring(b_imgs),      # 保护蒸馏数据
                'logits_ring': float_to_ring(raw_logits),  # 保护教师输出（原始 logits）[Fix 2]
                'labels':      b_labs.numpy(),
            })

        print(f"      生成 {len(all_batches)} 个加密 batch（共 {n} 张）")

        print(f"\n[5/5] 发送到服务器（端口 {PORT_A_TO_SERVER}）...")
        time.sleep(2)   # 等服务器监听就绪
        ok = DataTransporter.send_data('localhost', PORT_A_TO_SERVER, {
            'batches':     all_batches,
            'num_batches': len(all_batches),
            'source':      'hospital_a',
        })
        print("      发送完成" if ok else "      发送失败，请确认服务器已启动")

        print("\n[医院A] 数据已发送，保持在线（MPC 协议需要）...")
        try:
            cnt = 0
            while True:
                time.sleep(10)
                cnt += 1
                print(f"[医院A] 心跳 #{cnt}")
        except KeyboardInterrupt:
            print("\n[医院A] 收到中断信号")

    def cleanup(self):
        self.runtime.__exit__(None, None, None)
        print("[医院A] MPC 已清理")

# ============================================================
# 医院B (Party 1)  —— images02_best.pt
# ============================================================
class HospitalB:
    def __init__(self):
        self.party_id = 1
        self.dev      = device

        print(f"\n{'='*60}")
        print(f"医院B (Party {self.party_id}) - images02_best.pt")
        print(f"{'='*60}")
        print(f"设备: {self.dev}")

        # ---- MPC 初始化 ----
        print("\n[1/5] 初始化 MPC...")
        self.party   = Party2PC(self.party_id, SEMI_HONEST)
        self.runtime = PartyRuntime(self.party)
        self.runtime.__enter__()
        print("      等待对端 MPC 连接...")
        self.party.online()
        print("      MPC 连接成功！")

        # ---- 加载教师模型 ----
        print("\n[2/5] 加载教师模型 Teacher_B（由医院B 用 data_B 本地独立训练）...")
        self.teacher = resnet18(num_classes=10)
        self.teacher.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.teacher.maxpool = nn.Identity()

        # 医院B 专属教师模型，权重文件由医院B 用自己的数据 data_B 训练产生
        # 与医院A 的 teacher_a_resnet18.pth 完全独立，权重不同，logits 分布不同
        teacher_path = 'teacher_b_resnet18.pth'
        if not os.path.exists(teacher_path):
            print(f"      找不到 {teacher_path}")
            print(f"      请先运行: python train_teacher.py --party b  生成该文件")
            sys.exit(1)

        self.teacher.load_state_dict(torch.load(teacher_path, map_location='cpu'))
        self.teacher = self.teacher.to(self.dev)
        self.teacher.eval()
        print(f"      Teacher_B 加载成功，设备: {next(self.teacher.parameters()).device}")

        # ---- 加载蒸馏数据 ----
        print("\n[3/5] 加载蒸馏数据 images02_best.pt（由 data_B 蒸馏产生）...")
        data_path = 'images02_best.pt'
        if not os.path.exists(data_path):
            files = glob.glob('**/images02_best.pt', recursive=True)
            if files:
                data_path = files[0]
                print(f"      自动找到: {data_path}")
            else:
                print("      找不到 images02_best.pt")
                sys.exit(1)

        images = torch.load(data_path, map_location='cpu')
        labels = torch.load(
            data_path.replace('images02_best.pt', 'labels02_best.pt'),
            map_location='cpu')

        print(f"      数据形状: {images.shape}，值域: [{images.min():.3f}, {images.max():.3f}]")

        if images.min() >= 0.0 and images.max() <= 1.0:
            print("      应用 CIFAR-10 归一化...")
            images = transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(images)
            print(f"      归一化后值域: [{images.min():.3f}, {images.max():.3f}]")
        else:
            print("      数据已归一化，跳过。")

        self.images = images
        self.labels = labels.long()
        print(f"      共 {len(self.images)} 张蒸馏图片")

    def generate_and_send(self):
        print("\n[4/5] 生成并加密训练数据...")
        augmentor   = DiffAugment()
        n           = len(self.images)
        perm        = torch.randperm(n)
        all_batches = []

        for start in range(0, n, batch_size):
            idx    = perm[start:start + batch_size]
            b_imgs = self.images[idx]
            b_labs = self.labels[idx]

            b_imgs_aug = augmentor(b_imgs)
            with torch.no_grad():
                raw_logits = self.teacher(b_imgs_aug.to(self.dev)).cpu()

            all_batches.append({
                'imgs_ring':   float_to_ring(b_imgs),
                'logits_ring': float_to_ring(raw_logits),
                'labels':      b_labs.numpy(),
            })

        print(f"      生成 {len(all_batches)} 个加密 batch（共 {n} 张）")

        print(f"\n[5/5] 发送到服务器（端口 {PORT_B_TO_SERVER}）...")
        time.sleep(2)
        ok = DataTransporter.send_data('localhost', PORT_B_TO_SERVER, {
            'batches':     all_batches,
            'num_batches': len(all_batches),
            'source':      'hospital_b',
        })
        print("      发送完成" if ok else "      发送失败，请确认服务器已启动")

        print("\n[医院B] 数据已发送，保持在线（MPC 协议需要）...")
        try:
            cnt = 0
            while True:
                time.sleep(10)
                cnt += 1
                print(f"[医院B] 心跳 #{cnt}")
        except KeyboardInterrupt:
            print("\n[医院B] 收到中断信号")

    def cleanup(self):
        self.runtime.__exit__(None, None, None)
        print("[医院B] MPC 已清理")

# ============================================================
# 服务器 (Party 2) —— 聚合两方数据，训练 Student
# ============================================================
class ServerParty:
    def __init__(self):
        self.party_id = 2
        self.dev      = device

        print(f"\n{'='*60}")
        print(f"服务器 (Party {self.party_id}) - 3方联合训练 + Knowledge Alignment")
        print(f"{'='*60}")
        print(f"设备: {self.dev}")

        # ---- [1/5] Student 模型 ----
        print("\n[1/5] 初始化 Student 模型...")
        self.student = get_network(
            'ConvNet', channel=3, num_classes=10, im_size=(32, 32)
        ).to(self.dev)
        n_params = sum(p.numel() for p in self.student.parameters())
        print(f"      参数量: {n_params:,}，设备: {next(self.student.parameters()).device}")

        self.optimizer = torch.optim.SGD(
            self.student.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs)
        self.augmentor = DiffAugment().to(self.dev)   # [Fix 6] augmentor 在 GPU

        # ---- 测试集 ----
        print("\n[2/5] 准备 CIFAR-10 测试集...")
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=256, shuffle=False, num_workers=2)
        print(f"      测试集: {len(testset)} 张图片")

        # ---- 并行接收两家医院的数据（MPC 解密重构）[Fix 7] ----
        print("\n[3/5] MPC 协同解密，并行接收两方加密数据...")
        print(f"      医院A（Teacher_A，data_A 蒸馏）-> 端口 {PORT_A_TO_SERVER}")
        print(f"      医院B（Teacher_B，data_B 蒸馏）-> 端口 {PORT_B_TO_SERVER}")

        recv_result = {}

        def recv_one(port, key):
            try:
                data = DataTransporter.receive_data(port)
                # [KA-3] 给每个 batch 打上来源标签，供 align() 识别是哪个教师的 logits
                for b in data['batches']:
                    b['source'] = key   # 'a' -> Teacher_A logits；'b' -> Teacher_B logits
                recv_result[key] = data['batches']
                print(f"      收到 {key} 数据：{len(data['batches'])} 个 batch")
            except Exception as e:
                print(f"      接收 {key} 失败: {e}")
                recv_result[key] = []

        t_a = threading.Thread(target=recv_one, args=(PORT_A_TO_SERVER, 'a'), daemon=True)
        t_b = threading.Thread(target=recv_one, args=(PORT_B_TO_SERVER, 'b'), daemon=True)
        t_a.start()
        t_b.start()
        t_a.join()
        t_b.join()

        self.batches_a = recv_result.get('a', [])
        self.batches_b = recv_result.get('b', [])

        # [Fix 7] 合并后统一打乱训练
        self.all_batches = self.batches_a + self.batches_b
        self.num_batches = len(self.all_batches)

        n_a = sum(len(b['labels']) for b in self.batches_a)
        n_b = sum(len(b['labels']) for b in self.batches_b)
        print(f"\n      医院A（Teacher_A）: {len(self.batches_a)} batch（{n_a} 张）")
        print(f"      医院B（Teacher_B）: {len(self.batches_b)} batch（{n_b} 张）")
        print(f"      合计:              {self.num_batches} batch（{n_a + n_b} 张）")

        # ---- [KA-1] Knowledge Alignment 预统计（训练前一次性完成）----
        # 此处消除 Teacher_A（来自 data_A）与 Teacher_B（来自 data_B）的分布冲突
        print("\n[4/5] 执行 Knowledge Alignment 预统计...")
        self.aligner = KnowledgeAligner(num_classes=10)
        self.aligner.fit(self.batches_a, self.batches_b, self.dev)
        print("      Knowledge Alignment 就绪")

        print("\n[5/5] 初始化完成，准备开始训练。")

    # ---- 评估 ----
    def evaluate(self) -> float:
        self.student.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs  = inputs.to(self.dev)    # [Fix 6]
                targets = targets.to(self.dev)
                preds   = self.student(inputs).argmax(dim=1)
                correct += preds.eq(targets).sum().item()
                total   += targets.size(0)
        return 100.0 * correct / total

    # ---- 训练主循环 ----
    def train_with_received_data(self):
        """
        数据流（与设计图完全对应）：
          MPC 解密重构 ring_to_float()
            -> Logits_A（来自 Teacher_A，data_A 训练）
               Logits_B（来自 Teacher_B，data_B 训练）
            -> [KA-2] KnowledgeAligner.align()
            -> AlignedLogits（消除了两教师因不同数据导致的分布冲突）
            -> KD Loss + CE Loss
            -> Student Training
            -> Global Model
        """
        print("\n开始联合训练（含 Knowledge Alignment）...")
        print("=" * 80)
        print(f"{'Epoch':<8} {'Loss':<12} {'KD_Loss':<12} {'CE_Loss':<12} {'Test_Acc':<10}")
        print("=" * 80)

        best_acc = 0.0

        for epoch in range(epochs):
            self.student.train()
            total_loss = total_kd = total_ce = 0.0

            # 每 epoch 打乱全部 batch（两方数据混合）[Fix 7]
            perm = np.random.permutation(self.num_batches)

            for bi in perm:
                batch = self.all_batches[bi]

                # ===== Step 1：MPC 解密重构 RingTensor -> float -> GPU =====
                # [Fix 4 + Fix 6] 解码并统一迁移到 self.dev
                imgs       = ring_to_float(batch['imgs_ring'],   self.dev)   # [B,3,32,32] GPU
                raw_logits = ring_to_float(batch['logits_ring'], self.dev)   # [B,10]      GPU
                labels     = torch.tensor(
                    batch['labels'], dtype=torch.long).to(self.dev)          # [B]         GPU

                # ===== Step 2：Knowledge Alignment =====
                # [KA-2] 根据 source 标签（'a'=Teacher_A / 'b'=Teacher_B）
                #        对解密后的 logits 进行三步对齐，消除多教师冲突
                #        aligned_logits 与 raw_logits 形状相同 [B,10]，已在 GPU 上
                aligned_logits = self.aligner.align(
                    raw_logits, batch['source'], labels)

                # 数据增强（GPU）
                imgs = self.augmentor(imgs)

                # 学生前向
                student_logits = self.student(imgs)

                # ===== Step 3：KD Loss（使用 AlignedLogits）=====
                # [Fix 2] aligned_logits 保持 logits 尺度，只在这里做唯一一次 softmax
                kd_loss = F.kl_div(
                    F.log_softmax(student_logits  / T_temp, dim=1),
                    F.softmax(aligned_logits / T_temp, dim=1),
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

            # 每 100  epoch 评估一次
            if (epoch + 1) % 100 == 0 or epoch == 0:
                acc = self.evaluate()
                n   = self.num_batches
                print(f"{epoch+1:<8} {total_loss/n:<12.4f} {total_kd/n:<12.4f} "
                      f"{total_ce/n:<12.4f} {acc:<10.2f}")

                if acc > best_acc:
                    best_acc = acc
                    torch.save(self.student.state_dict(), 'best_student_3pc_ka.pth')
                    # print(f"    新最佳模型！准确率: {acc:.2f}%")

        print("=" * 80)
        print(f"\n联合训练完成！最佳准确率: {best_acc:.2f}%")
        print("模型已保存到: best_student_3pc_ka.pth")

        if os.path.exists('best_student_3pc_ka.pth'):
            self.student.load_state_dict(
                torch.load('best_student_3pc_ka.pth', map_location=self.dev))
            final_acc = self.evaluate()
            print(f"最终准确率（最佳 ckpt）: {final_acc:.2f}%")

        try:
            cnt = 0
            while True:
                time.sleep(10)
                cnt += 1
                print(f"[服务器] 心跳 #{cnt}")
        except KeyboardInterrupt:
            print("\n[服务器] 收到中断信号")

    def cleanup(self):
        print("[服务器] 清理完成")

# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('server', 'hospital_a', 'hospital_b'):
        print("用法（三个终端分别执行）:")
        print("  python 3pc_ka.py server      # 第1步：先启动服务器")
        print("  python 3pc_ka.py hospital_a  # 第2步：启动医院A（Party 0）")
        print("  python 3pc_ka.py hospital_b  # 第3步：启动医院B（Party 1）")
        sys.exit(1)

    mode = sys.argv[1]
    
    # ===== 初始化日志记录 =====
    logger = Logger(mode)
    
    try:
        print(f"Python:   {sys.executable}")
        print(f"PyTorch:  {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        print(f"使用设备: {device}")
        print(f"日志文件: {logger.log_file}")

        if mode == 'server':
            srv = ServerParty()
            try:
                srv.train_with_received_data()
            except KeyboardInterrupt:
                print("\n[服务器] 收到中断信号")
            finally:
                srv.cleanup()

        elif mode == 'hospital_a':
            ha = HospitalA()
            try:
                ha.generate_and_send()
            except KeyboardInterrupt:
                print("\n[医院A] 收到中断信号")
            finally:
                ha.cleanup()

        elif mode == 'hospital_b':
            hb = HospitalB()
            try:
                hb.generate_and_send()
            except KeyboardInterrupt:
                print("\n[医院B] 收到中断信号")
            finally:
                hb.cleanup()

    finally:
        # ===== 关闭日志记录 =====
        logger.close()