"""
2pc_nssmpc.py
=============
PLDK + NssMPC 完整密文训练（修复版）
医院 (Party 0) + 服务器 (Party 1) 两方

支持数据集：cifar10 / stl10（通过命令行第三个参数切换）

运行方式：
  终端1：python 2pc_nssmpc.py server   cifar10
  终端2：python 2pc_nssmpc.py hospital cifar10

  终端1：python 2pc_nssmpc.py server   stl10
  终端2：python 2pc_nssmpc.py hospital stl10

=== 修复清单 ===
[Fix 1] ★★★ 训练数据量灾难性不足：5 batch × 32 = 160 张 → 使用全部蒸馏数据
[Fix 2] ★★★ 双重 softmax Bug：传输原始 logits，服务器端唯一一次 softmax
[Fix 3] ★★  训练轮次 20 → 1000，与 baseline 对齐
[Fix 4] ★★  RingTensor 精度损失 → 定点数编码 ×2^16
[Fix 5] ★   学生模型 → 与 pldk_train_v2 相同的 ConvNet（utils.get_network）
[STL10 A] 测试集使用 STL10(split='test') + Resize(32)
[STL10 B] 教师推理前将蒸馏 32×32 图像 upsample 到 96×96
[STL10 C] 数据增强和 im_size 均使用 (32, 32)
[STL10 D] 模型保存名称：best_2pc_{dataset}.pth
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
import numpy as np
from datetime import datetime

# ===== 日志记录类（保持原版不变）=====
class Logger:
    """同时输出到控制台和文件的日志记录器"""
    def __init__(self, mode, dataset):
        # 创建日志目录
        self.log_dir = 'nssmpcLog'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            print(f"创建日志目录: {self.log_dir}")

        # 生成日志文件名：年月日_时分秒_mode_dataset.txt
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(
            self.log_dir, f'{timestamp}_{mode}_{dataset}.txt')

        # 打开日志文件
        self.file = open(self.log_file, 'w', encoding='utf-8')

        # 保存并替换 stdout
        self.original_stdout = sys.stdout
        sys.stdout = self

    def write(self, message):
        self.file.write(message)
        self.file.flush()
        self.original_stdout.write(message)
        self.original_stdout.flush()

    def flush(self):
        self.file.flush()
        self.original_stdout.flush()

    def close(self):
        sys.stdout = self.original_stdout
        self.file.close()
        print(f"日志已保存到: {self.log_file}")


# ===== NssMPC 导入 =====
from nssmpc import Party2PC, PartyRuntime, SEMI_HONEST, SecretTensor
from nssmpc.primitives.secret_sharing.arithmetic import RingTensor


# ==========================================
# 数据集配置字典
# 切换数据集只需修改命令行参数，无需改代码
# ==========================================
DATASET_CONFIGS = {
    'cifar10': {
        # 归一化参数（与蒸馏流程一致）
        'mean'          : (0.4914, 0.4822, 0.4465),
        'std'           : (0.2023, 0.1994, 0.2010),
        # 教师模型路径
        'teacher_path'  : 'teacher_resnet18_cifar10.pth',
        # 蒸馏数据路径
        'images_path'   : 'images_best_cifar10.pt',
        'labels_path'   : 'labels_best_cifar10.pt',
        # Student im_size 与蒸馏分辨率一致
        'im_size'       : (32, 32),
        # 教师模型的原始训练分辨率（推理时需 upsample 到此尺寸）
        'teacher_res'   : 32,
        # DiffAugment 裁剪尺寸
        'crop_size'     : 32,
        'crop_padding'  : 4,
        # 类别数
        'num_classes'   : 10,
        # 测试集加载函数标识
        'test_dataset'  : 'cifar10',
        # 输出模型文件名
        'model_save'    : 'best_2pc_cifar10.pth',
    },
    'stl10': {
        # STL-10 归一化参数
        'mean'          : (0.4467, 0.4398, 0.4066),
        'std'           : (0.2603, 0.2566, 0.2713),
        # 教师模型在原始 96×96 STL-10 上训练
        'teacher_path'  : 'teacher_resnet18_stl10.pth',
        # 蒸馏数据：MTT --res=32 生成的 32×32 蒸馏图像
        'images_path'   : 'images_best_stl10.pt',
        'labels_path'   : 'labels_best_stl10.pt',
        # Student 接受 32×32 输入（与蒸馏分辨率一致）
        'im_size'       : (32, 32),
        # [STL10 B] 教师模型训练分辨率为 96×96
        # 推理前需将 32×32 蒸馏图像 upsample 到 96×96
        'teacher_res'   : 96,
        # DiffAugment：蒸馏图像是 32×32，保持 32 裁剪
        'crop_size'     : 32,
        'crop_padding'  : 4,
        'num_classes'   : 10,
        'test_dataset'  : 'stl10',
        # [STL10 D] 输出模型文件名
        'model_save'    : 'best_2pc_stl10.pth',
    },
}


# ==========================================
# 全局配置（在 main() 中根据命令行参数设置）
# ==========================================
device     = 'cuda' if torch.cuda.is_available() else 'cpu'
T_temp     = 4.0
alpha      = 0.5
batch_size = 128   # [Fix 1]
epochs     = 1000  # [Fix 3]

# [Fix 4] 定点编码精度：float → int 缩放因子
FIXED_POINT_SCALE = 2 ** 16  # 精度约 1.5e-5，对归一化数据（±3 范围）足够


# ==========================================
# 数据增强（通用 32×32，两个数据集均适用）
# ==========================================
class DiffAugment(nn.Module):
    """
    差分增强模块，蒸馏数据均为 32×32，
    两个数据集用相同的增强策略。
    """
    def __init__(self, crop_size=32, padding=4):
        super().__init__()
        self.aug = nn.Sequential(
            transforms.RandomCrop(crop_size, padding=padding),
            transforms.RandomHorizontalFlip(),
        )

    def forward(self, x):
        return self.aug(x)


# ==========================================
# [Fix 4] 定点数编解码工具
# ==========================================
def float_to_ring(float_tensor: torch.Tensor) -> 'RingTensor':
    """
    float → 定点整数 → RingTensor
    输入必须在 CPU 上，否则 RingTensor 初始化会失败。
    """
    assert float_tensor.device.type == 'cpu', \
        "float_to_ring: 输入必须在 CPU 上"
    scaled = (float_tensor * FIXED_POINT_SCALE).round().long()
    return RingTensor(scaled)


def ring_to_float(ring_data, target_device: str) -> torch.Tensor:
    """
    RingTensor → 定点整数 → float → target_device
    兼容 RingTensor 和普通 tensor 两种输入。
    """
    if isinstance(ring_data, RingTensor):
        int_data   = ring_data.convert_to_real_field()   # CPU long tensor
        float_data = int_data.float() / FIXED_POINT_SCALE
        return float_data.to(target_device)
    return ring_data.to(target_device)


# ==========================================
# Socket 通信（保持原版不变）
# ==========================================
class DataTransporter:
    @staticmethod
    def send_data(host, port, data):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((host, port))
            pickle_data = pickle.dumps(data)
            s.sendall(len(pickle_data).to_bytes(8, 'big'))
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
    def __init__(self, cfg: dict):
        self.party_id = 0
        self.device   = device
        self.cfg      = cfg

        print(f"\n{'='*60}")
        print(f"医院 (Party {self.party_id}) - 完整密文训练 (修复版)")
        print(f"数据集: {cfg.get('_name', 'unknown').upper()}")
        print(f"{'='*60}")
        print(f"设备: {self.device}")

        # [1] 初始化 MPC
        print("\n[1/5] 初始化 MPC...")
        self.party   = Party2PC(self.party_id, SEMI_HONEST)
        self.runtime = PartyRuntime(self.party)
        self.runtime.__enter__()
        print("      等待服务器 MPC 连接...")
        self.party.online()
        print("      MPC 连接成功！")

        # [2] 加载教师模型
        print("\n[2/5] 加载教师模型...")
        self.teacher = self._build_teacher()
        print(f"      设备: {next(self.teacher.parameters()).device}")

        # [3] 加载蒸馏数据
        print("\n[3/5] 加载蒸馏数据...")
        self.images, self.labels = self._load_distilled_data()

    def _build_teacher(self) -> nn.Module:
        """
        构建教师模型（ResNet-18）。

        CIFAR-10 和 STL-10 的教师均采用相同的修改版架构：
          · conv1 改为 3×3 stride=1（去掉标准 7×7 的大步长降采样）
          · maxpool 替换为 Identity（保留更多空间信息）
        这与 train_teacher_STL10.py 第 76-77 行完全一致。

        教师推理使用的分辨率（teacher_res）与架构无关：
          · CIFAR-10 教师在 32×32 上推理（直接用蒸馏图像）
          · STL-10  教师在 96×96 上推理（蒸馏图像需 upsample）
          → upsample 逻辑在 generate_and_send_protected_data() 中处理
        """
        teacher_res  = self.cfg['teacher_res']
        teacher_path = self.cfg['teacher_path']
        num_classes  = self.cfg['num_classes']

        model = resnet18(num_classes=num_classes)

        # ★ 两个数据集的教师均使用相同的修改版架构 ★
        # 与 train_teacher_STL10.py 第 76-77 行及 train_teacher_cifar10.py 保持一致
        model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        print(f"      教师架构: ResNet-18（3×3 conv1 + Identity maxpool，推理分辨率 {teacher_res}×{teacher_res}）")

        if not os.path.exists(teacher_path):
            raise FileNotFoundError(
                f"找不到教师模型: {teacher_path}\n"
                f"请先运行对应的 train_teacher 脚本。"
            )
        model.load_state_dict(torch.load(teacher_path, map_location='cpu'))
        model = model.to(self.device)
        model.eval()
        print(f"      加载成功: {teacher_path}")
        return model

    def _load_distilled_data(self):
        """
        加载蒸馏图像，自动补归一化（若数据在 [0,1] 区间）。
        返回：images (CPU float32), labels (CPU long)
        """
        cfg       = self.cfg
        img_path  = cfg['images_path']
        lbl_path  = cfg['labels_path']

        # 自动搜索
        if not os.path.exists(img_path):
            found = glob.glob(f'**/{img_path}', recursive=True)
            if found:
                found.sort(key=os.path.getmtime, reverse=True)
                img_path = found[0]
                lbl_path = img_path.replace(
                    cfg['images_path'], cfg['labels_path'])
                print(f"      自动找到蒸馏数据: {img_path}")
            else:
                raise FileNotFoundError(f"找不到蒸馏数据文件: {img_path}")

        images = torch.load(img_path, map_location='cpu')
        labels = torch.load(lbl_path, map_location='cpu')
        if images.dtype != torch.float32:
            images = images.float()

        print(f"      数据形状: {images.shape}  "
              f"值域: [{images.min():.3f}, {images.max():.3f}]")

        # 补归一化（若数据在 [0,1]）
        if images.min() >= 0.0 and images.max() <= 1.0:
            print(f"      应用数据集归一化 mean={cfg['mean']} std={cfg['std']}...")
            transform_norm = transforms.Normalize(cfg['mean'], cfg['std'])
            images = transform_norm(images)
            print(f"      归一化后值域: [{images.min():.3f}, {images.max():.3f}]")
        else:
            print("      数据已归一化，跳过。")

        labels = labels.long()
        print(f"      共 {len(images)} 张蒸馏图片")
        return images, labels

    def generate_and_send_protected_data(self):
        """
        [STL10 B] 教师推理前将蒸馏 32×32 图像 upsample 到教师训练分辨率
        [Fix 2]   传输原始 logits（不做 softmax）
        [Fix 4]   定点编码后封装为 RingTensor 传输
        """
        teacher_res  = self.cfg['teacher_res']
        im_size      = self.cfg['im_size']   # 蒸馏图像实际尺寸
        needs_upsample = (teacher_res != im_size[0])

        if needs_upsample:
            print(f"\n[4/5] 教师推理（蒸馏 {im_size[0]}×{im_size[0]} "
                  f"→ upsample → {teacher_res}×{teacher_res} → logits）...")
        else:
            print(f"\n[4/5] 教师推理（直接 {im_size[0]}×{im_size[0]} → logits）...")

        n_samples  = len(self.images)
        all_batches = []

        for start in range(0, n_samples, batch_size):
            end       = min(start + batch_size, n_samples)
            b_imgs    = self.images[start:end]   # [B, 3, 32, 32]  CPU
            b_labs    = self.labels[start:end]

            # [STL10 B] STL-10 教师需要 96×96 输入，upsample 后再推理
            if needs_upsample:
                imgs_for_teacher = F.interpolate(
                    b_imgs.to(self.device),
                    size=(teacher_res, teacher_res),
                    mode='bilinear',
                    align_corners=False,
                )
            else:
                imgs_for_teacher = b_imgs.to(self.device)

            # 教师推理：不做 softmax，输出原始 logits
            with torch.no_grad():
                teacher_logits = self.teacher(imgs_for_teacher).cpu()  # 拿回 CPU

            # [Fix 4] 定点编码蒸馏图像和 logits → RingTensor
            protected_imgs   = float_to_ring(b_imgs)          # 蒸馏图（32×32）
            protected_logits = float_to_ring(teacher_logits)  # 教师 logits

            all_batches.append({
                'imgs_ring'  : protected_imgs,
                'logits_ring': protected_logits,
                'labels'     : b_labs.numpy(),
            })

            if (start // batch_size + 1) % 5 == 0 or end == n_samples:
                print(f"      进度: {end}/{n_samples} 张")

        print(f"      生成了 {len(all_batches)} 个保护 batch，共 {n_samples} 张")

        # 等待服务器就绪后发送
        print("\n[5/5] 发送数据到服务器（等待服务器就绪）...")
        time.sleep(3)
        success = DataTransporter.send_data('localhost', 9999, {
            'batches'    : all_batches,
            'num_batches': len(all_batches),
        })
        if success:
            print("      数据发送完成")
        else:
            print("      发送失败")

        print("\n[医院] 数据已发送，保持在线等待 MPC 协议...")
        try:
            counter = 0
            while True:
                time.sleep(10)
                counter += 1
                print(f"[医院] 心跳 #{counter}")
        except KeyboardInterrupt:
            print("\n[医院] 收到中断信号")

    def cleanup(self):
        self.runtime.__exit__(None, None, None)
        print("[医院] MPC 已清理")


# ==========================================
# 服务器端 (Party 1) - 密文训练执行方
# ==========================================
class ServerParty:
    def __init__(self, cfg: dict):
        self.party_id = 1
        self.device   = device
        self.cfg      = cfg

        print(f"\n{'='*60}")
        print(f"服务器 (Party {self.party_id}) - 学生训练 (修复版)")
        print(f"数据集: {cfg.get('_name', 'unknown').upper()}")
        print(f"{'='*60}")
        print(f"设备: {self.device}")

        # [1] 初始化 MPC
        print("\n[1/4] 初始化 MPC...")
        self.party   = Party2PC(self.party_id, SEMI_HONEST)
        self.runtime = PartyRuntime(self.party)
        self.runtime.__enter__()
        print("      等待医院 MPC 连接...")
        self.party.online()
        print("      MPC 连接成功！")

        # [2] 初始化学生模型（im_size 统一为蒸馏图像尺寸 32×32）
        print("\n[2/4] 初始化学生模型...")
        im_size      = cfg['im_size']
        num_classes  = cfg['num_classes']
        self.student = get_network(
            'ConvNet', channel=3, num_classes=num_classes, im_size=im_size
        ).to(self.device)
        total_params = sum(p.numel() for p in self.student.parameters())
        print(f"      im_size={im_size}  参数量: {total_params:,}")
        print(f"      设备: {next(self.student.parameters()).device}")

        self.optimizer = torch.optim.SGD(
            self.student.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs)

        # DiffAugment：蒸馏图像均为 32×32
        crop_size    = cfg['crop_size']
        crop_padding = cfg['crop_padding']
        self.augmentor = DiffAugment(
            crop_size=crop_size, padding=crop_padding).to(self.device)
        print(f"      增强: RandomCrop({crop_size}, padding={crop_padding}) + HFlip")

        # [3] 准备测试集
        print("\n[3/4] 准备测试集...")
        self.test_loader = self._build_test_loader()

        # [4] 接收保护数据
        print("\n[4/4] 等待接收医院的保护数据...")
        received         = DataTransporter.receive_data(9999)
        self.batches     = received['batches']
        self.num_batches = received['num_batches']
        print(f"      收到 {self.num_batches} 个 batch")

    def _build_test_loader(self):
        """
        根据数据集类型构建测试集 DataLoader。
        [STL10 A] STL-10 测试集需要 Resize(32) 适配 Student 输入尺寸。
        """
        cfg          = self.cfg
        test_dataset = cfg['test_dataset']
        mean, std    = cfg['mean'], cfg['std']

        if test_dataset == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            ds = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform)
            print(f"      CIFAR-10 测试集: {len(ds)} 张")

        elif test_dataset == 'stl10':
            # [STL10 A] STL-10 原图 96×96，Student 接受 32×32 → 先 Resize
            transform = transforms.Compose([
                transforms.Resize((32, 32)),   # 适配蒸馏分辨率
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            ds = torchvision.datasets.STL10(
                root='./data', split='test', download=True, transform=transform)
            print(f"      STL-10 测试集: {len(ds)} 张（已 Resize 到 32×32）")

        else:
            raise ValueError(f"不支持的 test_dataset 类型: {test_dataset}")

        return torch.utils.data.DataLoader(
            ds, batch_size=256, shuffle=False, num_workers=2)

    def evaluate(self) -> float:
        """在测试集上评估 Student 的分类准确率"""
        self.student.eval()
        correct = total = 0
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
        [Fix 2] KD target 使用原始 logits（RingTensor 解码后不再做 softmax）
        [Fix 3] 训练 1000 epochs
        [Fix 4] RingTensor → 定点解码 → float，统一迁移到 self.device
        """
        model_save = self.cfg['model_save']   # [STL10 D] 数据集相关的保存名

        print("\n开始训练...")
        print("=" * 70)
        print(f"{'Epoch':<8} {'Loss':<12} {'KD Loss':<12} {'CE Loss':<12} {'Test Acc':<10}")
        print("=" * 70)

        best_acc = 0.0
        last_acc = 0.0

        for epoch in range(epochs):
            self.student.train()
            total_loss = total_kd = total_ce = 0.0

            # 每个 epoch 打乱 batch 顺序
            perm = np.random.permutation(self.num_batches)

            for batch_idx in perm:
                batch = self.batches[batch_idx]

                # [Fix 4] 解码 RingTensor → float → GPU
                imgs           = ring_to_float(batch['imgs_ring'],   self.device)
                teacher_logits = ring_to_float(batch['logits_ring'], self.device)
                labels         = torch.tensor(
                    batch['labels'], dtype=torch.long).to(self.device)

                # 数据增强（GPU 上）
                imgs = self.augmentor(imgs)

                # 学生推理
                student_logits = self.student(imgs)

                # [Fix 2] 正确 KD 损失：teacher_logits 是原始 logits，这里才做 softmax
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

            # 每 100 epoch 评估
            if (epoch + 1) % 100 == 0 or epoch == 0:
                last_acc = self.evaluate()
                n = self.num_batches
                print(f"{epoch+1:<8} {total_loss/n:<12.4f} {total_kd/n:<12.4f} "
                      f"{total_ce/n:<12.4f} {last_acc:<10.2f}")

                # [STL10 D] 保存最佳模型，名称含数据集信息
                if last_acc > best_acc:
                    best_acc = last_acc
                    torch.save(self.student.state_dict(), model_save)

        print("=" * 70)
        print(f"\n训练完成！最佳准确率: {best_acc:.2f}%")
        print(f"模型已保存到: {model_save}")

        # 保持在线，等待 MPC 协议清理
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
    # 命令行参数解析
    # 用法: python 2pc_nssmpc.py <mode> <dataset>
    #       mode:    hospital | server
    #       dataset: cifar10  | stl10
    if len(sys.argv) not in (2, 3):
        print("用法:")
        print("  python 2pc_nssmpc.py hospital cifar10   # 医院 (CIFAR-10)")
        print("  python 2pc_nssmpc.py server   cifar10   # 服务器 (CIFAR-10)")
        print("  python 2pc_nssmpc.py hospital stl10     # 医院 (STL-10)")
        print("  python 2pc_nssmpc.py server   stl10     # 服务器 (STL-10)")
        sys.exit(1)

    mode    = sys.argv[1]
    dataset = sys.argv[2] if len(sys.argv) == 3 else 'cifar10'  # 默认 cifar10

    if dataset not in DATASET_CONFIGS:
        print(f"不支持的数据集: {dataset}，可选: {list(DATASET_CONFIGS.keys())}")
        sys.exit(1)

    if mode not in ('hospital', 'server'):
        print(f"不支持的模式: {mode}，可选: hospital / server")
        sys.exit(1)

    # 将数据集名注入配置（用于打印）
    cfg          = dict(DATASET_CONFIGS[dataset])
    cfg['_name'] = dataset

    # 初始化日志（含数据集名，便于区分日志文件）
    logger = Logger(mode, dataset)

    try:
        print(f"Python:    {sys.executable}")
        print(f"PyTorch:   {torch.__version__}")
        print(f"CUDA可用:  {torch.cuda.is_available()}")
        print(f"使用设备:  {device}")
        print(f"数据集:    {dataset.upper()}")
        print(f"日志文件:  {logger.log_file}")

        if mode == 'hospital':
            hospital = HospitalParty(cfg)
            try:
                hospital.generate_and_send_protected_data()
            except KeyboardInterrupt:
                print("\n[医院] 收到中断信号")
            finally:
                hospital.cleanup()

        elif mode == 'server':
            server = ServerParty(cfg)
            try:
                server.train_with_received_data()
            except KeyboardInterrupt:
                print("\n[服务器] 收到中断信号")
            finally:
                server.cleanup()

    finally:
        # 关闭日志记录
        logger.close()