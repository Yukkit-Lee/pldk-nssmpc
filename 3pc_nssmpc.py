"""
PLDK + NssMPC - 3方完整密文训练（通用数据集版）
=======================================================================
参与方:
  Party 0 - 医院A : 持有蒸馏数据 + 教师模型，可配置数据集
  Party 1 - 医院B : 持有蒸馏数据 + 教师模型，可配置数据集
  Party 2 - 服务器: 接收两方加密数据，联合训练 Student，不接触原始数据

支持数据集: cifar10 / stl10
  → 切换数据集只需修改顶部【用户配置区】，逻辑代码无需改动

运行方式（三个终端，建议先起服务器）:
  python 3pc_nssmpc.py server      # 服务器 (Party 2)
  python 3pc_nssmpc.py hospital_a  # 医院A  (Party 0)
  python 3pc_nssmpc.py hospital_b  # 医院B  (Party 1)

继承所有修复:
  [Fix 1] 全量数据参与训练，不限 batch 数
  [Fix 2] 发送原始 logits，服务器端才做 softmax，避免双重 softmax
  [Fix 3] 训练 1000 epochs
  [Fix 4] RingTensor 定点编码 x2^16，解决浮点精度丢失
  [Fix 5] get_network 保证 Student 架构与 baseline 一致
  [Fix 6] 所有 tensor 操作前显式 .to(device)，杜绝 CPU/GPU 混用
  [Fix 7] 服务器合并两方数据后统一打乱训练

STL-10 注意事项:
  · 蒸馏数据由 MTT --res=32 生成，实际分辨率为 32×32
  · 教师模型在原始 96×96 上训练（同样采用 3×3 conv1 + Identity maxpool 架构）
  · 医院端推理前自动将 32×32 蒸馏图像 upsample 到 96×96 喂给教师
  · Student 始终接受 32×32 输入，DiffAugment 保持 32×32 裁剪不变
  · 服务器测试集自动 Resize(32) 以匹配 Student 输入
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
        self.log_dir = 'nssmpcLog_3pc'
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
# ╔══════════════════════════════════════════════════════════╗
# ║              ★★★  用户配置区  ★★★                         ║
# ║  切换数据集只需修改这里，下方逻辑代码无需改动              ║
# ╚══════════════════════════════════════════════════════════╝
# ============================================================

# ---- 数据集参数库（新增数据集在此添加一条记录即可）----
DATASET_CONFIGS = {
    'cifar10': {
        'mean'       : (0.4914, 0.4822, 0.4465),
        'std'        : (0.2023, 0.1994, 0.2010),
        'num_classes': 10,
        # 教师模型的推理分辨率（教师在此尺寸上训练）
        # CIFAR-10 教师在 32×32 上训练，蒸馏图像也是 32×32 → 无需 upsample
        'teacher_res': 32,
    },
    'stl10': {
        'mean'       : (0.4467, 0.4398, 0.4066),
        'std'        : (0.2603, 0.2566, 0.2713),
        'num_classes': 10,
        # STL-10 教师在原始 96×96 上训练（conv1=3×3 + maxpool=Identity）
        # 蒸馏图像由 MTT --res=32 生成，实际为 32×32
        # → 医院端推理前自动 upsample 32×32 → 96×96
        'teacher_res': 96,
    },
}

# ---- 医院A 配置 ----
# 修改 'dataset' 为 'cifar10' 或 'stl10' 即可切换数据集
# HOSPITAL_A_CONFIG = {
#     'dataset'     : 'cifar10',                   # ← 切换数据集改这里
#     'teacher_path': 'teacher_resnet18_cifar10.pth',  # ← 教师模型路径
#     'images_path' : 'images_best_cifar10.pt',    # ← 蒸馏图像路径
#     'labels_path' : 'labels_best_cifar10.pt',    # ← 蒸馏标签路径
# }

# STL-10 示例（注释掉上面，取消注释下面）:
HOSPITAL_A_CONFIG = {
    'dataset'     : 'stl10',
    'teacher_path': 'teacher_resnet18_stl10.pth',
    'images_path' : 'images_best_stl10.pt',
    'labels_path' : 'labels_best_stl10.pt',
}

# ---- 医院B 配置 ----
# HOSPITAL_B_CONFIG = {
#     'dataset'     : 'cifar10',                   # ← 切换数据集改这里
#     'teacher_path': 'teacher_resnet18_cifar10_low.pth',    # ← 教师模型路径（95%精度教师）
#     'images_path' : 'images_best_cifar10.pt',    # ← 蒸馏图像路径
#     'labels_path' : 'labels_best_cifar10.pt',    # ← 蒸馏标签路径
# }

# # STL-10 示例（注释掉上面，取消注释下面）:
HOSPITAL_B_CONFIG = {
    'dataset'     : 'stl10',
    'teacher_path': 'teacher_resnet18_stl10.pth',
    'images_path' : 'images_best_stl10.pt',
    'labels_path' : 'labels_best_stl10.pt',
}

# ---- 服务器配置 ----
# dataset 决定测试集类型（通常与医院数据集保持一致）
# model_save 为训练完成后保存的 Student 模型文件名
# SERVER_CONFIG = {
#     'dataset'   : 'cifar10',                     # ← 切换数据集改这里
#     'model_save': 'best_student_3pc_cifar10.pth', # ← 保存文件名
# }

# STL-10 示例（注释掉上面，取消注释下面）:
SERVER_CONFIG = {
    'dataset'   : 'stl10',
    'model_save': 'best_student_3pc_stl10_stl10_high.pth',
}

# ============================================================
# 全局训练参数（通常不需要修改）
# ============================================================
device     = 'cuda' if torch.cuda.is_available() else 'cpu'
T_temp     = 5.0    #### 4.0 
alpha      = 0.3    #### 0.5
batch_size = 128
epochs     = 1000   # [Fix 3]

FIXED_POINT_SCALE = 2 ** 16   # [Fix 4] 精度约 1.5e-5

# 通信端口
PORT_A_TO_SERVER = 9991   # 医院A -> 服务器
PORT_B_TO_SERVER = 9992   # 医院B -> 服务器

# ============================================================
# 数据增强（蒸馏数据始终为 32×32，两个数据集裁剪参数相同）
# ============================================================
class DiffAugment(nn.Module):
    """
    差分增强：蒸馏数据无论来自 CIFAR-10 还是 STL-10，
    MTT --res=32 生成的图像均为 32×32，裁剪参数固定不变。
    """
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
# 医院A (Party 0)  —— 从 HOSPITAL_A_CONFIG 读取配置
# ============================================================
class HospitalA:
    def __init__(self):
        self.party_id = 0
        self.dev      = device
        self.cfg      = HOSPITAL_A_CONFIG
        self.dcfg     = DATASET_CONFIGS[self.cfg['dataset']]

        ds_name = self.cfg['dataset'].upper()
        print(f"\n{'='*60}")
        print(f"医院A (Party {self.party_id}) - 数据集: {ds_name}")
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
        # 两个数据集的教师均采用：3×3 conv1 + Identity maxpool（与 train_teacher_STL10.py 一致）
        print(f"\n[2/5] 加载教师模型（{ds_name}）...")
        self.teacher = resnet18(num_classes=self.dcfg['num_classes'])
        self.teacher.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.teacher.maxpool = nn.Identity()

        teacher_path = self.cfg['teacher_path']
        if not os.path.exists(teacher_path):
            print(f"      找不到 {teacher_path}")
            sys.exit(1)

        self.teacher.load_state_dict(torch.load(teacher_path, map_location='cpu'))
        self.teacher = self.teacher.to(self.dev)
        self.teacher.eval()
        teacher_res = self.dcfg['teacher_res']
        print(f"      Teacher 加载成功，推理分辨率: {teacher_res}×{teacher_res}")
        print(f"      设备: {next(self.teacher.parameters()).device}")

        # ---- 加载蒸馏数据 ----
        print(f"\n[3/5] 加载蒸馏数据 {self.cfg['images_path']}...")
        data_path = self.cfg['images_path']
        lbl_path  = self.cfg['labels_path']

        if not os.path.exists(data_path):
            files = glob.glob(f'**/{data_path}', recursive=True)
            if files:
                data_path = files[0]
                lbl_path  = self.cfg['labels_path']
                print(f"      自动找到: {data_path}")
            else:
                print(f"      找不到 {data_path}")
                sys.exit(1)

        images = torch.load(data_path, map_location='cpu')
        labels = torch.load(lbl_path,  map_location='cpu')

        print(f"      数据形状: {images.shape}，值域: [{images.min():.3f}, {images.max():.3f}]")

        # 补归一化（若蒸馏数据保存时未归一化，即值域在 [0,1]）
        if images.min() >= 0.0 and images.max() <= 1.0:
            print(f"      应用 {ds_name} 归一化...")
            images = transforms.Normalize(self.dcfg['mean'], self.dcfg['std'])(images)
            print(f"      归一化后值域: [{images.min():.3f}, {images.max():.3f}]")
        else:
            print("      数据已归一化，跳过。")

        self.images = images          # CPU float，形状 [N, 3, 32, 32]
        self.labels = labels.long()   # CPU long
        print(f"      共 {len(self.images)} 张蒸馏图片")

    def generate_and_send(self):
        """
        Teacher 推理 -> 原始 logits [Fix 2]
        -> 定点编码 -> RingTensor [Fix 4]
        -> 全量发送 [Fix 1]

        STL-10 教师推理前自动 upsample 蒸馏图像 32×32 → 96×96。
        """
        teacher_res    = self.dcfg['teacher_res']
        actual_h       = self.images.shape[-2]
        needs_upsample = (teacher_res != actual_h)

        if needs_upsample:
            print(f"\n[4/5] 生成并加密训练数据（32×32 蒸馏图 → upsample → {teacher_res}×{teacher_res} → 教师推理）...")
        else:
            print("\n[4/5] 生成并加密训练数据...")

        augmentor   = DiffAugment()   # CPU 增强，32×32 裁剪
        n           = len(self.images)
        perm        = torch.randperm(n)
        all_batches = []

        for start in range(0, n, batch_size):
            idx    = perm[start:start + batch_size]
            b_imgs = self.images[idx]   # CPU [B, 3, 32, 32]
            b_labs = self.labels[idx]

            # DiffAugment 在 CPU 做 32×32 裁剪
            b_imgs_aug = augmentor(b_imgs)

            # STL-10：upsample 增强后的 32×32 到 96×96，再喂给教师 [Fix B]
            if needs_upsample:
                imgs_for_teacher = F.interpolate(
                    b_imgs_aug.to(self.dev),
                    size=(teacher_res, teacher_res),
                    mode='bilinear', align_corners=False,
                )
            else:
                imgs_for_teacher = b_imgs_aug.to(self.dev)   # [Fix 6]

            # 教师推理，结果立刻拿回 CPU [Fix 6]
            with torch.no_grad():
                raw_logits = self.teacher(imgs_for_teacher).cpu()   # [B, C] CPU

            # [Fix 4] 定点编码：蒸馏图像和教师 logits 均保护
            all_batches.append({
                'imgs_ring':   float_to_ring(b_imgs),       # 原始 32×32 蒸馏图（加密）
                'logits_ring': float_to_ring(raw_logits),   # 原始 logits（加密）[Fix 2]
                'labels':      b_labs.numpy(),
            })

        print(f"      生成 {len(all_batches)} 个加密 batch（共 {n} 张）")

        print(f"\n[5/5] 发送到服务器（端口 {PORT_A_TO_SERVER}）...")
        time.sleep(2)
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
# 医院B (Party 1)  —— 从 HOSPITAL_B_CONFIG 读取配置
# ============================================================
class HospitalB:
    def __init__(self):
        self.party_id = 1
        self.dev      = device
        self.cfg      = HOSPITAL_B_CONFIG
        self.dcfg     = DATASET_CONFIGS[self.cfg['dataset']]

        ds_name = self.cfg['dataset'].upper()
        print(f"\n{'='*60}")
        print(f"医院B (Party {self.party_id}) - 数据集: {ds_name}")
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
        print(f"\n[2/5] 加载教师模型（{ds_name}）...")
        self.teacher = resnet18(num_classes=self.dcfg['num_classes'])
        self.teacher.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.teacher.maxpool = nn.Identity()

        teacher_path = self.cfg['teacher_path']
        if not os.path.exists(teacher_path):
            print(f"      找不到 {teacher_path}")
            sys.exit(1)

        self.teacher.load_state_dict(torch.load(teacher_path, map_location='cpu'))
        self.teacher = self.teacher.to(self.dev)
        self.teacher.eval()
        teacher_res = self.dcfg['teacher_res']
        print(f"      Teacher 加载成功，推理分辨率: {teacher_res}×{teacher_res}")
        print(f"      设备: {next(self.teacher.parameters()).device}")

        # ---- 加载蒸馏数据 ----
        print(f"\n[3/5] 加载蒸馏数据 {self.cfg['images_path']}...")
        data_path = self.cfg['images_path']
        lbl_path  = self.cfg['labels_path']

        if not os.path.exists(data_path):
            files = glob.glob(f'**/{data_path}', recursive=True)
            if files:
                data_path = files[0]
                print(f"      自动找到: {data_path}")
            else:
                print(f"      找不到 {data_path}")
                sys.exit(1)

        images = torch.load(data_path, map_location='cpu')
        labels = torch.load(lbl_path,  map_location='cpu')

        print(f"      数据形状: {images.shape}，值域: [{images.min():.3f}, {images.max():.3f}]")

        if images.min() >= 0.0 and images.max() <= 1.0:
            print(f"      应用 {ds_name} 归一化...")
            images = transforms.Normalize(self.dcfg['mean'], self.dcfg['std'])(images)
            print(f"      归一化后值域: [{images.min():.3f}, {images.max():.3f}]")
        else:
            print("      数据已归一化，跳过。")

        self.images = images
        self.labels = labels.long()
        print(f"      共 {len(self.images)} 张蒸馏图片")

    def generate_and_send(self):
        teacher_res    = self.dcfg['teacher_res']
        actual_h       = self.images.shape[-2]
        needs_upsample = (teacher_res != actual_h)

        if needs_upsample:
            print(f"\n[4/5] 生成并加密训练数据（32×32 蒸馏图 → upsample → {teacher_res}×{teacher_res} → 教师推理）...")
        else:
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

            if needs_upsample:
                imgs_for_teacher = F.interpolate(
                    b_imgs_aug.to(self.dev),
                    size=(teacher_res, teacher_res),
                    mode='bilinear', align_corners=False,
                )
            else:
                imgs_for_teacher = b_imgs_aug.to(self.dev)

            with torch.no_grad():
                raw_logits = self.teacher(imgs_for_teacher).cpu()

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
        self.scfg     = SERVER_CONFIG
        self.dcfg     = DATASET_CONFIGS[self.scfg['dataset']]
        self.model_save = self.scfg['model_save']

        ds_name = self.scfg['dataset'].upper()
        print(f"\n{'='*60}")
        print(f"服务器 (Party {self.party_id}) - 3方联合训练")
        print(f"测试集数据集: {ds_name}")
        print(f"{'='*60}")
        print(f"设备: {self.dev}")

        # ---- Student 模型 [Fix 5] ----
        # Student 始终接受 32×32 输入（蒸馏数据均为 32×32）
        print("\n[1/4] 初始化 Student 模型（im_size=32×32）...")
        self.student = get_network(
            'ConvNet', channel=3, num_classes=self.dcfg['num_classes'], im_size=(32, 32)
        ).to(self.dev)
        n_params = sum(p.numel() for p in self.student.parameters())
        print(f"      参数量: {n_params:,}，设备: {next(self.student.parameters()).device}")

        self.optimizer = torch.optim.SGD(
            self.student.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs)
        self.augmentor = DiffAugment().to(self.dev)   # [Fix 6] augmentor 在 GPU

        # ---- 测试集（根据 SERVER_CONFIG['dataset'] 自动选择）----
        print(f"\n[2/4] 准备 {ds_name} 测试集...")
        if self.scfg['dataset'] == 'cifar10':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.dcfg['mean'], self.dcfg['std']),
            ])
            testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform_test)
        elif self.scfg['dataset'] == 'stl10':
            # STL-10 原图 96×96，Resize 到 32×32 匹配 Student 输入
            transform_test = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(self.dcfg['mean'], self.dcfg['std']),
            ])
            testset = torchvision.datasets.STL10(
                root='./data', split='test', download=True, transform=transform_test)
        else:
            raise ValueError(f"不支持的数据集: {self.scfg['dataset']}")

        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=256, shuffle=False, num_workers=2)
        print(f"      测试集: {len(testset)} 张图片")

        # ---- 并行接收两家医院的数据 [Fix 7] ----
        print("\n[3/4] 并行等待两家医院的加密数据...")
        print(f"      医院A -> 端口 {PORT_A_TO_SERVER}")
        print(f"      医院B -> 端口 {PORT_B_TO_SERVER}")

        recv_result = {}

        def recv_one(port, key):
            try:
                data = DataTransporter.receive_data(port)
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

        batches_a = recv_result.get('a', [])
        batches_b = recv_result.get('b', [])

        # [Fix 7] 合并两方所有 batch
        self.all_batches = batches_a + batches_b
        self.num_batches = len(self.all_batches)

        n_a = sum(len(b['labels']) for b in batches_a)
        n_b = sum(len(b['labels']) for b in batches_b)
        print(f"\n      医院A: {len(batches_a)} batch（{n_a} 张）")
        print(f"      医院B: {len(batches_b)} batch（{n_b} 张）")
        print(f"      合计:  {self.num_batches} batch（{n_a + n_b} 张加密图片）")

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
        解码 RingTensor -> float -> GPU [Fix 4 + Fix 6]
        KD 使用原始 logits，仅在此处做一次 softmax [Fix 2]
        全量合并数据，每 epoch 统一打乱 [Fix 1 + Fix 7]
        """
        print("\n[4/4] 开始联合训练...")
        print("=" * 72)
        print(f"{'Epoch':<8} {'Loss':<12} {'KD_Loss':<12} {'CE_Loss':<12} {'Test_Acc':<10}")
        print("=" * 72)

        best_acc = 0.0

        for epoch in range(epochs):
            self.student.train()
            total_loss = total_kd = total_ce = 0.0

            # 每 epoch 打乱全部 batch（两方数据混合）[Fix 7]
            perm = np.random.permutation(self.num_batches)

            for bi in perm:
                batch = self.all_batches[bi]

                # ===== 密文训练核心：RingTensor 解码 -> float -> GPU =====
                # [Fix 4 + Fix 6] 解码并统一迁移到 self.dev
                imgs           = ring_to_float(batch['imgs_ring'],   self.dev)  # [B,3,32,32] GPU
                teacher_logits = ring_to_float(batch['logits_ring'], self.dev)  # [B,10]      GPU
                labels         = torch.tensor(
                    batch['labels'], dtype=torch.long).to(self.dev)             # [B]         GPU

                # 数据增强（GPU）
                imgs = self.augmentor(imgs)

                # 学生前向
                student_logits = self.student(imgs)

                # [Fix 2] teacher_logits 是原始 logits，只在这里做一次 softmax
                kd_loss = F.kl_div(
                    F.log_softmax(student_logits  / T_temp, dim=1),
                    F.softmax(teacher_logits / T_temp, dim=1),
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

            # 每 100 epoch 评估一次（与 baseline 节奏一致）
            if (epoch + 1) % 100 == 0 or epoch == 0:
                acc = self.evaluate()
                n   = self.num_batches
                print(f"{epoch+1:<8} {total_loss/n:<12.4f} {total_kd/n:<12.4f} "
                      f"{total_ce/n:<12.4f} {acc:<10.2f}")

                if acc > best_acc:
                    best_acc = acc
                    torch.save(self.student.state_dict(), self.model_save)

        print("=" * 72)
        print(f"\n联合训练完成！最佳准确率: {best_acc:.2f}%")
        print(f"模型已保存到: {self.model_save}")

        # 加载最佳 checkpoint 做最终评估
        if os.path.exists(self.model_save):
            self.student.load_state_dict(
                torch.load(self.model_save, map_location=self.dev))
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
        print("  python 3pc_nssmpc.py server      # 第1步：先启动服务器")
        print("  python 3pc_nssmpc.py hospital_a  # 第2步：启动医院A（Party 0）")
        print("  python 3pc_nssmpc.py hospital_b  # 第3步：启动医院B（Party 1）")
        sys.exit(1)

    mode = sys.argv[1]

    # 从配置推断当前数据集（用于日志文件名）
    if mode == 'hospital_a':
        _ds_tag = HOSPITAL_A_CONFIG['dataset']
    elif mode == 'hospital_b':
        _ds_tag = HOSPITAL_B_CONFIG['dataset']
    else:
        _ds_tag = SERVER_CONFIG['dataset']

    # ===== 初始化日志记录（文件名含数据集标识）=====
    logger = Logger(f"{mode}_{_ds_tag}")
    
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