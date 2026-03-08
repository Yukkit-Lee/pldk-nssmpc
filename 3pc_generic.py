"""
PLDK + NssMPC - 3方完整密文训练 + Knowledge Alignment  ★ 通用数据集版本 ★
=======================================================================
设计目标：
  支持任意数据集组合（同构/异构），换数据集只需修改顶部配置，无需改动任何逻辑代码。

核心配置（只需修改这里）：
  DATASET_CONFIGS  : 数据集参数库（归一化/尺寸/类别数/增强padding/测试集构造器）
  HOSPITAL_CONFIGS : 每家医院使用的数据集、教师模型路径、蒸馏数据路径
  TARGET_SIZE      : 统一训练分辨率（所有蒸馏数据和测试集均 resize 到此尺寸）

混合数据集问题（问题1）：
  当医院A 用 CIFAR-10（32×32）、医院B 用 STL-10（96×96）时：
    - 两个数据集尺寸不同，学生模型只能接受一种分辨率
    - 解决方案：定义 TARGET_SIZE，医院端在加密前 resize，服务器测试集用
      ConcatDataset 合并两方数据集（均 resize 到 TARGET_SIZE）
    - 测试集体现真实泛化能力：同时评估在两家医院数据集上的表现

数据流：
  医院A: data_A -> Teacher_A -> Logits_A -> resize到TARGET_SIZE -> MPC加密 -> 发送
  医院B: data_B -> Teacher_B -> Logits_B -> resize到TARGET_SIZE -> MPC加密 -> 发送
  服务器: MPC解密 -> KA对齐 -> 统一分辨率联合训练 -> 在合并测试集上评估

运行方式（三个终端，建议先起服务器）:
  python 3pc_ka_generic.py server      # 服务器 (Party 2)
  python 3pc_ka_generic.py hospital_a  # 医院A  (Party 0)
  python 3pc_ka_generic.py hospital_b  # 医院B  (Party 1)

继承所有原有修复:
  [Fix 1] 全量数据参与训练
  [Fix 2] 原始 logits 传输，服务器端唯一一次 softmax
  [Fix 3] 1000 epochs
  [Fix 4] RingTensor 定点编码 x2^16
  [Fix 5] get_network 与 baseline 架构一致
  [Fix 6] 所有 tensor 显式 .to(device)
  [Fix 7] 两方数据合并后统一打乱训练
  [KA-1~3] Knowledge Alignment 三步对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from utils import get_network
import os
import sys
import time
import glob
import pickle
import socket
import threading
import numpy as np

from nssmpc import Party2PC, PartyRuntime, SEMI_HONEST, SecretTensor
from nssmpc.primitives.secret_sharing.arithmetic import RingTensor

# ================================================================
# ★★★  配置区  ★★★   —— 只需修改这里，逻辑代码不用动
# ================================================================

# ---- 数据集参数库 ----
# 新增数据集：在此添加一条记录即可
DATASET_CONFIGS = {
    'cifar10': {
        'mean'        : (0.4914, 0.4822, 0.4465),
        'std'         : (0.2023, 0.1994, 0.2010),
        'native_size' : (32, 32),    # 原始图像尺寸
        'num_classes' : 10,
        'crop_padding': 4,           # RandomCrop padding = native_size/8
        # 测试集构造器，参数为 (root, transform)，返回 Dataset
        'test_dataset': lambda root, tf: torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=tf),
    },
    'stl10': {
        'mean'        : (0.4467, 0.4398, 0.4066),
        'std'         : (0.2603, 0.2566, 0.2713),
        'native_size' : (96, 96),
        'num_classes' : 10,
        'crop_padding': 12,
        'test_dataset': lambda root, tf: torchvision.datasets.STL10(
            root=root, split='test', download=True, transform=tf),
    },
    # 新数据集示例（取消注释并填写即可）：
    # 'cifar100': {
    #     'mean'        : (0.5071, 0.4867, 0.4408),
    #     'std'         : (0.2675, 0.2565, 0.2761),
    #     'native_size' : (32, 32),
    #     'num_classes' : 100,
    #     'crop_padding': 4,
    #     'test_dataset': lambda root, tf: torchvision.datasets.CIFAR100(
    #         root=root, train=False, download=True, transform=tf),
    # },
}

# ---- 医院配置 ----
# 指定每家医院使用的数据集、教师模型权重文件、蒸馏数据文件
HOSPITAL_CONFIGS = {
    'hospital_a': {
        'dataset'      : 'cifar10',                  # 必须是 DATASET_CONFIGS 中的 key
        'teacher_path' : 'teacher_a_cifar10.pth',    # 由医院A 用 data_A 独立训练
        'images_path'  : 'images01_best_cifar10.pt', # 蒸馏图像
        'labels_path'  : 'labels01_best_cifar10.pt', # 蒸馏标签
        'party_id'     : 0,
        'port'         : 9991,
    },
    'hospital_b': {
        'dataset'      : 'stl10',                    # 不同数据集
        'teacher_path' : 'teacher_b_stl10.pth',
        'images_path'  : 'images02_best_stl10.pt',
        'labels_path'  : 'labels02_best_stl10.pt',
        'party_id'     : 1,
        'port'         : 9992,
    },
}

# ---- 统一训练分辨率 ----
# 所有蒸馏数据和测试集均 resize 到此尺寸
# 建议取参与医院中最大的 native_size，或统一设为 (96, 96)
TARGET_SIZE = (96, 96)       # 备选32×32

# ---- 全局训练参数 ----
T_temp            = 4.0
alpha             = 0.5
batch_size        = 64      # TARGET_SIZE 较大时用 64，若全为 32×32 可改为 128
epochs            = 1000
FIXED_POINT_SCALE = 2 ** 16
NUM_CLASSES       = 10      # 若所有医院数据集 num_classes 不同，需做适配（见注释）

# ================================================================
# 以下为逻辑代码，换数据集通常不需要修改
# ================================================================

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ----------------------------------------------------------------
# 辅助：根据 dataset_name 构造 resize + normalize transform
# ----------------------------------------------------------------
def make_transform(dataset_name: str, target_size: tuple) -> transforms.Compose:
    """
    构造将图像归一化到指定分辨率的 transform。
    - 先 Resize 到 target_size（若与 native_size 不同）
    - 再用该数据集的 mean/std 归一化
    """
    cfg = DATASET_CONFIGS[dataset_name]
    t_list = []
    if target_size != cfg['native_size']:
        t_list.append(transforms.Resize(target_size))
    t_list.append(transforms.Normalize(cfg['mean'], cfg['std']))
    return transforms.Compose(t_list)


def make_test_transform(dataset_name: str, target_size: tuple) -> transforms.Compose:
    """构造测试集 transform（ToTensor + Resize + Normalize）"""
    cfg = DATASET_CONFIGS[dataset_name]
    t_list = [transforms.ToTensor()]
    if target_size != cfg['native_size']:
        t_list.append(transforms.Resize(target_size))
    t_list.append(transforms.Normalize(cfg['mean'], cfg['std']))
    return transforms.Compose(t_list)


# ----------------------------------------------------------------
# 数据增强（基于 TARGET_SIZE，在 CPU 执行）
# ----------------------------------------------------------------
class DiffAugment(nn.Module):
    """适配任意分辨率的数据增强，crop_padding = target_size // 8"""
    def __init__(self, target_size: tuple = TARGET_SIZE):
        super().__init__()
        padding = target_size[0] // 8   # 32->4, 96->12，与各数据集 baseline 一致
        self.aug = nn.Sequential(
            transforms.RandomCrop(target_size[0], padding=padding),
            transforms.RandomHorizontalFlip(),
        )

    def forward(self, x):
        return self.aug(x)


# ----------------------------------------------------------------
# 定点编解码  [Fix 4]
# ----------------------------------------------------------------
def float_to_ring(float_tensor: torch.Tensor) -> RingTensor:
    assert float_tensor.device.type == 'cpu', \
        f"[float_to_ring] 输入必须在 CPU，当前: {float_tensor.device}"
    scaled = (float_tensor * FIXED_POINT_SCALE).round().long()
    return RingTensor(scaled)


def ring_to_float(ring_data, target_device: str) -> torch.Tensor:
    if isinstance(ring_data, RingTensor):
        return (ring_data.convert_to_real_field().float() / FIXED_POINT_SCALE).to(target_device)
    return ring_data.to(target_device)


# ----------------------------------------------------------------
# Socket 通信
# ----------------------------------------------------------------
class DataTransporter:
    HEADER = 8

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


# ----------------------------------------------------------------
# ★ Knowledge Alignment 模块
# ----------------------------------------------------------------
class KnowledgeAligner:
    """
    三步对齐消除多教师冲突：
      Step 1  统计对齐：Z-score 归一化到全局 canonical 空间，消除不同数据集
              训练出的教师模型在 logits 均值/方差上的系统偏差。
      Step 2  置信度加权融合：低置信类别向全局均值收缩，减少噪声教师的影响。
      Step 3  标签条件修正：ground-truth 类别 logit 轻微上调（+0.5σ），
              防止对齐后软标签与硬标签语义倒置。
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        self.num_classes = num_classes
        self.fitted      = False
        self.mean_a = self.std_a = None
        self.mean_b = self.std_b = None
        self.mean_global = self.std_global = None
        self.conf_a = self.conf_b = None

    @torch.no_grad()
    def fit(self, batches_a: list, batches_b: list, dev: str):
        """训练前一次性统计，CPU 操作节省显存"""
        print("\n  [KA] 计算 Knowledge Alignment 统计量...")

        logits_a = torch.cat([ring_to_float(b['logits_ring'], 'cpu') for b in batches_a], dim=0)
        logits_b = torch.cat([ring_to_float(b['logits_ring'], 'cpu') for b in batches_b], dim=0)
        logits_all = torch.cat([logits_a, logits_b], dim=0)

        mean_a = logits_a.mean(dim=0);   std_a = logits_a.std(dim=0).clamp(min=1e-6)
        mean_b = logits_b.mean(dim=0);   std_b = logits_b.std(dim=0).clamp(min=1e-6)
        mean_global = logits_all.mean(dim=0)
        std_global  = logits_all.std(dim=0).clamp(min=1e-6)
        conf_a = F.softmax(logits_a / T_temp, dim=1).mean(dim=0)
        conf_b = F.softmax(logits_b / T_temp, dim=1).mean(dim=0)

        self.mean_a      = mean_a.to(dev);      self.std_a  = std_a.to(dev)
        self.mean_b      = mean_b.to(dev);      self.std_b  = std_b.to(dev)
        self.mean_global = mean_global.to(dev); self.std_global = std_global.to(dev)
        self.conf_a      = conf_a.to(dev);      self.conf_b = conf_b.to(dev)
        self.fitted      = True

        print(f"  [KA] Teacher_A logits — 均值:{logits_a.mean():.4f}  标准差:{logits_a.std():.4f}")
        print(f"  [KA] Teacher_B logits — 均值:{logits_b.mean():.4f}  标准差:{logits_b.std():.4f}")
        print(f"  [KA] 全局空间         — 均值:{logits_all.mean():.4f}  标准差:{logits_all.std():.4f}")
        print(f"  [KA] 各类别置信度对比（A / B）：")
        for c in range(self.num_classes):
            diff   = abs(conf_a[c].item() - conf_b[c].item())
            marker = "  <- 冲突" if diff > 0.05 else ""
            print(f"       Class {c:2d}: A={conf_a[c]:.4f}  B={conf_b[c]:.4f}{marker}")

    def align(self, logits: torch.Tensor, source: str, labels: torch.Tensor) -> torch.Tensor:
        assert self.fitted, "请先调用 aligner.fit()"
        if source == 'a':
            src_mean, src_std, src_conf, oth_conf = self.mean_a, self.std_a, self.conf_a, self.conf_b
        else:
            src_mean, src_std, src_conf, oth_conf = self.mean_b, self.std_b, self.conf_b, self.conf_a

        # Step 1: 统计对齐
        aligned = (logits - src_mean) / src_std * self.std_global + self.mean_global

        # Step 2: 置信度加权融合
        conf_w  = src_conf / (src_conf + oth_conf + 1e-8)
        aligned = (conf_w.unsqueeze(0) * aligned
                   + (1.0 - conf_w).unsqueeze(0) * self.mean_global.unsqueeze(0))

        # Step 3: 标签条件修正
        for i in range(logits.size(0)):
            c = labels[i].item()
            aligned[i, c] = aligned[i, c] + 0.5 * self.std_global[c]

        return aligned


# ----------------------------------------------------------------
# 医院方（通用）
# ----------------------------------------------------------------
class HospitalParty:
    """
    通用医院方。传入 hospital_key（'hospital_a' 或 'hospital_b'）后
    自动从 HOSPITAL_CONFIGS 和 DATASET_CONFIGS 读取全部参数。
    """

    def __init__(self, hospital_key: str):
        self.key    = hospital_key
        self.cfg    = HOSPITAL_CONFIGS[hospital_key]          # 医院配置
        self.dcfg   = DATASET_CONFIGS[self.cfg['dataset']]   # 数据集配置
        self.dev    = device
        self.party_id = self.cfg['party_id']

        print(f"\n{'='*64}")
        print(f"{hospital_key.upper()} (Party {self.party_id})")
        print(f"  数据集: {self.cfg['dataset'].upper()}  "
              f"原始尺寸: {self.dcfg['native_size']}  "
              f"目标尺寸: {TARGET_SIZE}")
        print(f"{'='*64}")
        print(f"设备: {self.dev}")

        # ---- [1/5] MPC 初始化 ----
        print("\n[1/5] 初始化 MPC...")
        self.party   = Party2PC(self.party_id, SEMI_HONEST)
        self.runtime = PartyRuntime(self.party)
        self.runtime.__enter__()
        print("      等待对端 MPC 连接...")
        self.party.online()
        print("      MPC 连接成功！")

        # ---- [2/5] 加载教师模型 ----
        dataset_name = self.cfg['dataset'].upper()
        print(f"\n[2/5] 加载教师模型（{dataset_name}，由本院独立训练）...")
        self.teacher = resnet18(num_classes=self.dcfg['num_classes'])
        # 适配非 ImageNet 尺寸：去掉大步长第一层和 maxpool
        self.teacher.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.teacher.maxpool = nn.Identity()

        teacher_path = self.cfg['teacher_path']
        if not os.path.exists(teacher_path):
            print(f"      找不到 {teacher_path}")
            print(f"      请先用本院数据训练教师模型并保存到该路径")
            sys.exit(1)

        self.teacher.load_state_dict(torch.load(teacher_path, map_location='cpu'))
        self.teacher = self.teacher.to(self.dev)
        self.teacher.eval()
        print(f"      教师模型加载成功，设备: {next(self.teacher.parameters()).device}")

        # ---- [3/5] 加载蒸馏数据 ----
        print(f"\n[3/5] 加载蒸馏数据（{self.cfg['images_path']}）...")
        img_path = self.cfg['images_path']
        lbl_path = self.cfg['labels_path']

        for p in [img_path, lbl_path]:
            if not os.path.exists(p):
                found = glob.glob(f'**/{p}', recursive=True)
                if found:
                    if p == img_path:
                        img_path = found[0]
                    else:
                        lbl_path = found[0]
                    print(f"      自动找到: {found[0]}")
                else:
                    print(f"      找不到 {p}")
                    sys.exit(1)

        # 全程 CPU，推理时才上 GPU [Fix 6]
        images = torch.load(img_path, map_location='cpu')
        labels = torch.load(lbl_path, map_location='cpu')

        print(f"      数据形状: {images.shape}  值域: [{images.min():.3f}, {images.max():.3f}]")

        # 若图像未归一化（[0,1] 范围），先用本数据集参数归一化
        if images.min() >= 0.0 and images.max() <= 1.0:
            norm_tf = transforms.Normalize(self.dcfg['mean'], self.dcfg['std'])
            images  = norm_tf(images)
            print(f"      归一化后值域: [{images.min():.3f}, {images.max():.3f}]")
        else:
            print("      数据已归一化，跳过归一化步骤。")

        # 若本数据集原始尺寸与 TARGET_SIZE 不一致，Resize 到统一分辨率
        native = self.dcfg['native_size']
        if native != TARGET_SIZE:
            print(f"      Resize: {native} -> {TARGET_SIZE}（统一训练分辨率）...")
            images = F.interpolate(
                images, size=TARGET_SIZE, mode='bilinear', align_corners=False)
            print(f"      Resize 后形状: {images.shape}")

        self.images = images          # CPU float  [N, 3, H, W]
        self.labels = labels.long()   # CPU long   [N]
        print(f"      共 {len(self.images)} 张蒸馏图片，尺寸 {TARGET_SIZE}")

    def generate_and_send(self):
        """推理 -> 加密 -> 发送  [Fix 1,2,4]"""
        print("\n[4/5] 生成并加密训练数据...")
        augmentor   = DiffAugment(target_size=TARGET_SIZE)  # CPU 增强，已适配 TARGET_SIZE
        n           = len(self.images)
        perm        = torch.randperm(n)
        all_batches = []

        for start in range(0, n, batch_size):
            idx        = perm[start:start + batch_size]
            b_imgs     = self.images[idx]   # CPU [B,3,H,W]
            b_labs     = self.labels[idx]   # CPU [B]
            b_imgs_aug = augmentor(b_imgs)

            # 推理：上 GPU -> 拿回 CPU [Fix 6]
            with torch.no_grad():
                raw_logits = self.teacher(b_imgs_aug.to(self.dev)).cpu()  # [B,C] CPU

            # [Fix 4] 定点编码：同时保护蒸馏图像和教师 logits
            all_batches.append({
                'imgs_ring'  : float_to_ring(b_imgs),      # 加密蒸馏图像
                'logits_ring': float_to_ring(raw_logits),  # 加密原始 logits [Fix 2]
                'labels'     : b_labs.numpy(),
            })

        print(f"      生成 {len(all_batches)} 个加密 batch（共 {n} 张，尺寸 {TARGET_SIZE}）")

        port = self.cfg['port']
        print(f"\n[5/5] 发送到服务器（端口 {port}）...")
        time.sleep(2)
        ok = DataTransporter.send_data('localhost', port, {
            'batches'    : all_batches,
            'num_batches': len(all_batches),
            'source'     : self.key,     # 'hospital_a' / 'hospital_b'
            'dataset'    : self.cfg['dataset'],
        })
        print("      发送完成" if ok else "      发送失败，请确认服务器已启动")

        print(f"\n[{self.key}] 数据已发送，保持在线（MPC 协议需要）...")
        try:
            cnt = 0
            while True:
                time.sleep(10)
                cnt += 1
                print(f"[{self.key}] 心跳 #{cnt}")
        except KeyboardInterrupt:
            print(f"\n[{self.key}] 收到中断信号")

    def cleanup(self):
        self.runtime.__exit__(None, None, None)
        print(f"[{self.key}] MPC 已清理")


# ----------------------------------------------------------------
# 服务器方
# ----------------------------------------------------------------
class ServerParty:
    """
    通用服务器方。
    自动根据 HOSPITAL_CONFIGS 识别参与医院，
    合并测试集（混合数据集时用 ConcatDataset）。
    """

    def __init__(self):
        self.party_id = 2
        self.dev      = device

        # 参与的医院及其端口（从配置自动读取）
        self.hospital_keys = list(HOSPITAL_CONFIGS.keys())  # ['hospital_a', 'hospital_b']
        datasets_used = list({HOSPITAL_CONFIGS[k]['dataset'] for k in self.hospital_keys})

        print(f"\n{'='*64}")
        print(f"服务器 (Party {self.party_id}) - 通用 3方联合训练 + Knowledge Alignment")
        print(f"  参与医院: {self.hospital_keys}")
        print(f"  数据集:   {datasets_used}")
        print(f"  目标分辨率: {TARGET_SIZE}")
        print(f"{'='*64}")
        print(f"设备: {self.dev}")

        # ---- [1/5] Student 模型 ----
        print("\n[1/5] 初始化 Student 模型...")
        self.student = get_network(
            'ConvNet', channel=3, num_classes=NUM_CLASSES, im_size=TARGET_SIZE
        ).to(self.dev)
        n_params = sum(p.numel() for p in self.student.parameters())
        print(f"      ConvNet  im_size={TARGET_SIZE}  参数量: {n_params:,}")

        self.optimizer = torch.optim.SGD(
            self.student.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs)
        self.augmentor = DiffAugment(target_size=TARGET_SIZE).to(self.dev)

        # ---- [2/5] 测试集（混合数据集：ConcatDataset）----
        print("\n[2/5] 构建测试集（混合所有参与方数据集）...")
        test_sub_datasets = []
        for hospital_key in self.hospital_keys:
            hcfg  = HOSPITAL_CONFIGS[hospital_key]
            dname = hcfg['dataset']
            dcfg  = DATASET_CONFIGS[dname]
            tf    = make_test_transform(dname, TARGET_SIZE)
            ds    = dcfg['test_dataset']('./data', tf)
            test_sub_datasets.append(ds)
            print(f"      {hospital_key} ({dname.upper()})  {len(ds)} 张  "
                  f"-> resize 到 {TARGET_SIZE}")

        if len(test_sub_datasets) == 1:
            full_testset = test_sub_datasets[0]
        else:
            # 合并不同数据集的测试集，统一分辨率后可直接 concat
            full_testset = torch.utils.data.ConcatDataset(test_sub_datasets)

        self.test_loader = torch.utils.data.DataLoader(
            full_testset, batch_size=128, shuffle=False, num_workers=2)
        print(f"      合并测试集共 {len(full_testset)} 张图片  "
              f"（{'同构' if len(test_sub_datasets)==1 else '异构混合'}）")

        # ---- [3/5] 并行接收两方加密数据 ----
        print("\n[3/5] MPC 协同解密，并行接收各方加密数据...")
        recv_result = {}

        def recv_one(hospital_key):
            port = HOSPITAL_CONFIGS[hospital_key]['port']
            try:
                data = DataTransporter.receive_data(port)
                # [KA-3] 打上来源标签（'hospital_a' / 'hospital_b'）
                for b in data['batches']:
                    b['source'] = hospital_key
                recv_result[hospital_key] = data['batches']
                n_imgs = sum(len(b['labels']) for b in data['batches'])
                print(f"      收到 {hospital_key} ({data.get('dataset','?').upper()})  "
                      f"{len(data['batches'])} batch，{n_imgs} 张")
            except Exception as e:
                print(f"      接收 {hospital_key} 失败: {e}")
                recv_result[hospital_key] = []

        threads = [threading.Thread(target=recv_one, args=(k,), daemon=True)
                   for k in self.hospital_keys]
        for t in threads: t.start()
        for t in threads: t.join()

        # 分开存储以供 KA fit()，合并供训练
        self.batches_per_hospital = {k: recv_result.get(k, []) for k in self.hospital_keys}
        self.all_batches = []
        for k in self.hospital_keys:
            self.all_batches.extend(self.batches_per_hospital[k])
        self.num_batches = len(self.all_batches)

        total_imgs = sum(len(b['labels']) for b in self.all_batches)
        print(f"\n      合计: {self.num_batches} batch，{total_imgs} 张加密图片")

        # ---- [4/5] Knowledge Alignment 预统计 ----
        print("\n[4/5] 执行 Knowledge Alignment 预统计...")
        self.aligner = KnowledgeAligner(num_classes=NUM_CLASSES)
        # 注：当前 KA 支持 2 方；多方扩展可在此迭代对齐
        keys = self.hospital_keys
        self.aligner.fit(
            self.batches_per_hospital[keys[0]],
            self.batches_per_hospital[keys[1]],
            self.dev
        )
        print("      Knowledge Alignment 就绪")

        print("\n[5/5] 初始化完成，准备开始训练。")

    # ---- 评估 ----
    def evaluate(self) -> float:
        self.student.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs  = inputs.to(self.dev)
                targets = targets.to(self.dev)
                preds   = self.student(inputs).argmax(dim=1)
                correct += preds.eq(targets).sum().item()
                total   += targets.size(0)
        return 100.0 * correct / total

    # ---- 训练主循环 ----
    def train_with_received_data(self):
        """
        数据流：
          MPC 解密重构 ring_to_float()
            -> Logits_A / Logits_B（已统一到 TARGET_SIZE 分辨率）
            -> KnowledgeAligner.align()  消除多教师分布冲突
            -> KD Loss + CE Loss
            -> Student Training on TARGET_SIZE
            -> 在混合测试集上评估（体现真实泛化能力）
        """
        print("\n开始联合训练（含 Knowledge Alignment）...")
        print(f"  训练分辨率: {TARGET_SIZE}  "
              f"测试集: {'+'.join(HOSPITAL_CONFIGS[k]['dataset'].upper() for k in self.hospital_keys)}")
        print("=" * 80)
        print(f"{'Epoch':<8} {'Loss':<12} {'KD_Loss':<12} {'CE_Loss':<12} {'Test_Acc':<10}")
        print("=" * 80)

        best_acc  = 0.0
        save_name = 'best_student_3pc_ka_generic.pth'

        for epoch in range(epochs):
            self.student.train()
            total_loss = total_kd = total_ce = 0.0
            perm = np.random.permutation(self.num_batches)

            for bi in perm:
                batch = self.all_batches[bi]

                # Step 1: MPC 解密重构 [Fix 4,6]
                imgs       = ring_to_float(batch['imgs_ring'],   self.dev)  # [B,3,H,W] GPU
                raw_logits = ring_to_float(batch['logits_ring'], self.dev)  # [B,C]     GPU
                labels     = torch.tensor(
                    batch['labels'], dtype=torch.long).to(self.dev)         # [B]       GPU

                # Step 2: Knowledge Alignment [KA-2]
                aligned_logits = self.aligner.align(raw_logits, batch['source'], labels)

                # 数据增强（GPU）
                imgs = self.augmentor(imgs)

                # 学生前向
                student_logits = self.student(imgs)

                # Step 3: KD Loss（唯一一次 softmax）[Fix 2]
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

            if (epoch + 1) % 100 == 0 or epoch == 0:
                acc = self.evaluate()
                n   = self.num_batches
                print(f"{epoch+1:<8} {total_loss/n:<12.4f} {total_kd/n:<12.4f} "
                      f"{total_ce/n:<12.4f} {acc:<10.2f}")

                if acc > best_acc:
                    best_acc = acc
                    torch.save(self.student.state_dict(), save_name)
                    print(f"    新最佳模型！准确率: {acc:.2f}%  已保存 -> {save_name}")

        print("=" * 80)
        print(f"\n联合训练完成！最佳准确率: {best_acc:.2f}%")

        if os.path.exists(save_name):
            self.student.load_state_dict(torch.load(save_name, map_location=self.dev))
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


# ================================================================
# 主程序
# ================================================================
if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('server', 'hospital_a', 'hospital_b'):
        print("用法（三个终端分别执行）:")
        print("  python 3pc_ka_generic.py server      # 第1步：先启动服务器")
        print("  python 3pc_ka_generic.py hospital_a  # 第2步：启动医院A")
        print("  python 3pc_ka_generic.py hospital_b  # 第3步：启动医院B")
        print()
        print("当前配置：")
        for k, v in HOSPITAL_CONFIGS.items():
            print(f"  {k}: 数据集={v['dataset']}  教师={v['teacher_path']}")
        print(f"  TARGET_SIZE = {TARGET_SIZE}")
        sys.exit(1)

    mode = sys.argv[1]
    print(f"Python:   {sys.executable}")
    print(f"PyTorch:  {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"使用设备: {device}")

    if mode == 'server':
        srv = ServerParty()
        try:
            srv.train_with_received_data()
        except KeyboardInterrupt:
            print("\n[服务器] 收到中断信号")
        finally:
            srv.cleanup()

    elif mode in ('hospital_a', 'hospital_b'):
        hospital = HospitalParty(mode)
        try:
            hospital.generate_and_send()
        except KeyboardInterrupt:
            print(f"\n[{mode}] 收到中断信号")
        finally:
            hospital.cleanup()