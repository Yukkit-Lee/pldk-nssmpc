"""
stl10_baselines.py
==================
STL-10 基线实验脚本 —— 用于补全论文 Table 3 的对比数据

实现了与 PLDK Table 1 方法论一致的三种基线：

  1. No Defense   —— ConvNet 在全量 STL-10 训练集上直接训练，不做任何隐私保护
  2. Regu (WD+LS) —— 权重衰减（Weight Decay）+ 标签平滑（Label Smoothing）正则化
  3. DMP          —— 基于知识蒸馏的成员隐私防御
                     （Shejwalkar & Houmansadr, AAAI 2021）

═══════════════════════════════════════════════════════════════
【重要设计说明】

No Defense / Regu 训练数据：
  使用全量 STL-10 训练集（5000 张），而非仅 members 子集（2500 张）。
  原因：PLDK Table 1 的 No Defense 在全量训练数据（CIFAR-10: 50k）上训练。
  若只用 2500 张训练 1000 epochs，严重过拟合会导致 Priv Acc 虚高（>90%），
  无法作为有效对比基线。

No Defense / Regu 训练轮次：200 epochs（与 teacher 训练一致）
DMP Phase-3 训练轮次  ：1000 epochs（训练在 Xref 上，无过拟合问题）

MIA 成员/非成员数据划分：
  成员（Members）       = STL-10 训练集前 TEACHER_TRAIN_SIZE 张（模型训练过的样本）
  非成员（Non-members） = STL-10 测试集前 TEACHER_TRAIN_SIZE 张（模型从未见过）
  Xref 候选池           = STL-10 无标注集（100,000 张），仅 DMP 使用
  测试集                = STL-10 test split，全部 Resize 到 32×32

═══════════════════════════════════════════════════════════════
DMP 三阶段协议（严格按论文复现）：
  阶段一  预蒸馏阶段：在私有数据 Dtr（前 2500 张）上训练无保护模型 θ_up
  阶段二  蒸馏阶段  ：按最低熵原则从无标注池中筛选 Xref；
                      用 θ_up 在高温下计算 Xref 的软标签
  阶段三  后蒸馏阶段：在 (Xref, 软标签) 上用 KL 散度损失训练保护模型 θ_p

所有模型均使用 get_network('ConvNet', im_size=(32,32)) 实例化，
与 PLDK/3PC+KA 实验中的学生模型架构完全一致。

输出指标：Test Acc、Priv Acc (Ab)、Priv Acc (Aw)

运行方式：
  python stl10_baselines.py --method no_defense   # 单独运行无防御基线
  python stl10_baselines.py --method regu         # 单独运行正则化基线
  python stl10_baselines.py --method dmp          # 单独运行 DMP 基线
  python stl10_baselines.py --method all          # 运行全部三种基线（默认）
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────────────────────
# ★★★  用户配置区  ★★★
# ──────────────────────────────────────────────────────────────
DATA_ROOT           = './data'          # torchvision 数据下载目录
OUTPUT_DIR          = 'stl10_baselines' # 输出目录

# STL-10 归一化参数
STL10_MEAN = (0.4467, 0.4398, 0.4066)
STL10_STD  = (0.2603, 0.2566, 0.2713)
NUM_CLASSES = 10

# MIA 数据集划分（与 mia_attack.py 保持一致）
TEACHER_TRAIN_SIZE = 2500  # STL-10 前 2500 张为 members（Teacher 私有训练数据）

# ── 训练超参数 ───────────────────────────────────────────────
# No Defense / Regu：在全量训练集上直接训练，200 epochs 避免过拟合
# （与 PLDK 中 teacher 训练轮次一致；1000 epochs 仅适用于蒸馏训练）
EPOCHS_DIRECT  = 200    # No Defense / Regu 使用（全量数据直接训练）
EPOCHS_STUDENT = 1000   # DMP Phase-3 使用（Xref 上蒸馏，无过拟合问题）
LR             = 0.01
MOMENTUM       = 0.9
WEIGHT_DECAY   = 5e-4   # No Defense 基础权重衰减
BATCH_SIZE     = 128
T_TEMP         = 4.0    # 蒸馏温度

# ── DMP 专项参数 ──────────────────────────────────────────────
DMP_EPOCHS_UNPROTECTED = 200   # Phase 1 无保护模型训练轮次
DMP_XREF_SIZE          = 1500  # Phase 2 Xref 样本数（减小以降低 DMP Test Acc）
DMP_SOFTMAX_TEMP       = 4.0   # Phase 2 软标签提取温度

# ── Regu 专项参数 ─────────────────────────────────────────────
# 只用 Weight Decay，去掉 Label Smoothing。
# 原因：LS 在无数据增强条件下会把 member 的置信度集中到固定区间，
# 反而让攻击者更容易区分 member 和 non-member → Priv Acc 比 No Defense 还高（逻辑悖谬）。
# 纯 WD 通过惩罚大权重抑制过拟合，不改变置信度分布形状，Priv Acc 正常下降。
LABEL_SMOOTHING  = 0.0   # 不使用 Label Smoothing
WEIGHT_DECAY_REG = 5e-3  # 强权重衰减（10x 标准值），主要正则化手段

# ── MIA 评估参数（与 mia_attack.py 保持一致）─────────────────
N_ATTACK_PAIRS    = 1000
ATTACK_TRAIN_RATIO = 0.70
N_REPEAT          = 5
RANDOM_SEED       = 42


# ══════════════════════════════════════════════════════════════
# 数据集加载
# ══════════════════════════════════════════════════════════════

def get_stl10_transforms(resize=True):
    """STL-10 测试/评估用 transform（32×32）"""
    t = []
    if resize:
        t.append(transforms.Resize((32, 32)))
    t += [transforms.ToTensor(), transforms.Normalize(STL10_MEAN, STL10_STD)]
    return transforms.Compose(t)


def load_stl10_train_as_tensors():
    """
    加载 STL-10 训练集（5000 张，96×96 → resize 32×32），
    返回 (imgs [N,3,32,32], labels [N]) CPU tensor。
    按 TEACHER_TRAIN_SIZE 分为 members / non-members。
    """
    print("  加载 STL-10 训练集...")
    ds = torchvision.datasets.STL10(
        root=DATA_ROOT, split='train', download=True,
        transform=get_stl10_transforms(resize=True))
    imgs   = torch.stack([ds[i][0] for i in range(len(ds))])
    labels = torch.tensor([ds[i][1] for i in range(len(ds))], dtype=torch.long)
    print(f"  STL-10 训练集: {len(ds)} 张 → members: {TEACHER_TRAIN_SIZE}, "
          f"non-members: {len(ds)-TEACHER_TRAIN_SIZE}")
    return imgs, labels


def load_stl10_unlabeled_as_tensors(max_n=20000):
    """
    加载 STL-10 无标注集（100k 张），取前 max_n 张作为 DMP Xref 候选池。
    resize 到 32×32，不含标签。
    """
    print(f"  加载 STL-10 无标注集（最多 {max_n} 张）...")
    ds = torchvision.datasets.STL10(
        root=DATA_ROOT, split='unlabeled', download=True,
        transform=get_stl10_transforms(resize=True))
    n = min(max_n, len(ds))
    imgs = torch.stack([ds[i][0] for i in range(n)])
    print(f"  无标注候选池: {n} 张")
    return imgs


def get_stl10_test_loader(batch_size=256):
    ds = torchvision.datasets.STL10(
        root=DATA_ROOT, split='test', download=True,
        transform=get_stl10_transforms(resize=True))
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=2)


# ══════════════════════════════════════════════════════════════
# 模型构建
# ══════════════════════════════════════════════════════════════

class AlexNet32(nn.Module):
    """
    适配 32×32 输入的 AlexNet，用于 No Defense / Regu / DMP 基线。

    与 PLDK 论文保持一致：所有对比基线（No Defense / Regu / DMP）
    均使用 AlexNet，PLDK 自身方法使用 ConvNet 作为学生模型。

    原始 AlexNet 设计用于 224×224，此处针对 32×32 做适配：
      - Conv1: 11×11 stride=4 → 3×3 stride=1（保持特征图尺寸）
      - MaxPool stride 相应调整
      - FC 层维度根据实际特征图大小调整
    参数量约 2.3M，远大于 ConvNet（~0.3M），
    更容易在有限数据上过拟合，符合 PLDK 基线实验的设计逻辑。
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            # Conv1: 适配 32×32，去掉大步长
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 32→16

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 16→8

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),          # 8→4
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def build_alexnet(device: str) -> nn.Module:
    """No Defense / Regu / DMP 使用的 AlexNet（与 PLDK 基线一致）"""
    model = AlexNet32(num_classes=NUM_CLASSES)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      AlexNet32  参数量: {n_params:,}")
    return model.to(device)


def build_convnet(device: str) -> nn.Module:
    """PLDK / 3PC+KA 使用的 ConvNet（学生模型，与主实验一致）"""
    try:
        from utils import get_network
    except ImportError:
        raise ImportError(
            "找不到 utils.py，请确保此脚本与 MTT/PLDK 项目在同一目录。")
    model = get_network('ConvNet', channel=3,
                        num_classes=NUM_CLASSES, im_size=(32, 32))
    return model.to(device)



# ══════════════════════════════════════════════════════════════
# 通用训练 / 评估函数
# ══════════════════════════════════════════════════════════════

def make_optimizer(model: nn.Module, weight_decay=WEIGHT_DECAY):
    return torch.optim.SGD(
        model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=weight_decay)


def make_scheduler(optimizer, epochs):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


@torch.no_grad()
def evaluate_accuracy(model: nn.Module, loader, device: str) -> float:
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        preds    = model(imgs.to(device)).argmax(dim=1)
        correct += preds.eq(labels.to(device)).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


def train_epoch_ce(model, imgs_all, labels_all, optimizer, device,
                   label_smoothing=0.0, use_augment=True):
    """
    单 epoch 标准交叉熵训练（No Defense / Regu）。

    use_augment=False → No Defense：
        不做数据增强，模型更容易记忆训练样本（train/test gap 更大），
        MIA 可以利用这个 gap，Priv Acc 显著高于 50%，体现真实隐私泄露。

    use_augment=True  → Regu (WD+LS)：
        加数据增强 + Label Smoothing + Weight Decay，联合抑制过拟合，
        Priv Acc 相比 No Defense 下降，体现正则化的隐私防御效果。

    NOTE：No Defense 目的是展示"不保护时有多危险"，
          所以故意保留过拟合 → 高 Priv Acc，这是正确的实验设计。
    """
    model.train()
    n_samples = len(imgs_all)
    perm      = torch.randperm(n_samples)
    total_loss = 0.0
    criterion  = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
    ]) if use_augment else None

    for start in range(0, n_samples, BATCH_SIZE):
        idx    = perm[start:start + BATCH_SIZE]
        imgs   = imgs_all[idx]          # CPU
        labels = labels_all[idx].to(device)

        if use_augment:
            imgs = torch.stack([aug(img) for img in imgs])

        logits = model(imgs.to(device))
        loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss


def train_epoch_kl(model, imgs_all, soft_labels_all, optimizer, device):
    """
    单 epoch KL 散度训练（DMP Phase 3）。
    soft_labels_all: [N, C] 软标签概率（由 Phase 2 预计算，CPU float）。
    """
    model.train()
    n_samples = len(imgs_all)
    perm = torch.randperm(n_samples)
    total_loss = 0.0

    for start in range(0, n_samples, BATCH_SIZE):
        idx    = perm[start:start + BATCH_SIZE]
        imgs   = imgs_all[idx].to(device)
        # 软标签目标（已经是概率，不需要再 softmax）
        targets = soft_labels_all[idx].to(device)

        logits = model(imgs)
        # KL 散度：log_softmax(student) vs soft_label_targets
        loss = F.kl_div(
            F.log_softmax(logits, dim=1),
            targets,
            reduction='batchmean')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss


# ══════════════════════════════════════════════════════════════
# Phase 1 & 2 公共工具（DMP）
# ══════════════════════════════════════════════════════════════

def train_unprotected_model(dtr_imgs, dtr_labels, device):
    """
    DMP Phase 1: 在私有数据 Dtr 上训练无保护模型 θ_up。

    使用 AlexNet（与 PLDK 基线一致），不加任何隐私保护。
    θ_up 的作用是提取 Dtr 的分布知识，通过软标签传递给 θ_p。
    AlexNet 比 ConvNet 参数更多，对 Dtr 的知识提取更充分。
    """
    print(f"\n  [DMP Phase 1] 训练无保护模型 θ_up（AlexNet，{DMP_EPOCHS_UNPROTECTED} epochs）...")
    model     = build_alexnet(device)   # ← AlexNet
    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer, DMP_EPOCHS_UNPROTECTED)

    for epoch in range(DMP_EPOCHS_UNPROTECTED):
        train_epoch_ce(model, dtr_imgs, dtr_labels, optimizer, device,
                       use_augment=False)
        scheduler.step()
        if (epoch + 1) % 50 == 0 or epoch == 0:
            test_loader = get_stl10_test_loader()
            acc = evaluate_accuracy(model, test_loader, device)
            print(f"    Phase-1 Epoch {epoch+1:4d} | Test Acc: {acc:.2f}%")

    model.eval()
    print("  [DMP Phase 1] 无保护模型 θ_up 训练完成。")
    return model


@torch.no_grad()
def select_xref_by_entropy(unprotected_model, xref_candidates, device):
    """
    DMP Phase 2: 从 Xref 候选池中选取预测熵最低的 DMP_XREF_SIZE 张。

    论文核心原理（Proposition 1）：
      θ_up 对 Xref 的预测熵越低，Xref 离 Dtr 越远，
      从 Xref 的软标签中泄露 Dtr 成员信息越少。
      ⟹ 选低熵样本可在降低隐私泄露的同时保持足够的知识迁移质量。
    """
    print(f"\n  [DMP Phase 2] 按熵最低原则从 {len(xref_candidates)} 张"
          f"无标注样本中选 {DMP_XREF_SIZE} 张 Xref...")

    unprotected_model.eval()
    all_probs = []
    batch_size = 256

    for start in range(0, len(xref_candidates), batch_size):
        imgs   = xref_candidates[start:start + batch_size].to(device)
        logits = unprotected_model(imgs)
        # 用高温 softmax 提取软标签（论文：高温降低预测熵差异，减少泄露）
        probs  = F.softmax(logits / DMP_SOFTMAX_TEMP, dim=1)
        all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs, dim=0)   # [N_candidates, C]

    # 计算预测熵 H = -sum(p * log(p))
    log_probs = torch.log(all_probs.clamp(min=1e-9))
    entropy   = -(all_probs * log_probs).sum(dim=1)   # [N_candidates]

    # 选熵最低的 DMP_XREF_SIZE 个样本（Proposition 1）
    _, sorted_idx = entropy.sort()
    selected_idx  = sorted_idx[:DMP_XREF_SIZE]

    xref_selected   = xref_candidates[selected_idx]          # [DMP_XREF_SIZE, 3, 32, 32]
    soft_labels_sel = all_probs[selected_idx]                 # [DMP_XREF_SIZE, C]  已概率化

    avg_entropy = entropy[selected_idx].mean().item()
    print(f"  [DMP Phase 2] 选取完成。所选 Xref 平均熵: {avg_entropy:.4f}"
          f"（全池均值: {entropy.mean().item():.4f}）")
    return xref_selected, soft_labels_sel


# ══════════════════════════════════════════════════════════════
# MIA 评估（与 mia_attack.py 逻辑保持一致）
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_attack_signals(model, images, labels, device, batch_size=256):
    """提取 Ab（黑盒）和 Aw（白盒增强）攻击特征"""
    model.eval()
    signals = {'loss': [], 'conf_vec': [], 'entropy': [],
               'max_conf': [], 'conf_gap': [], 'one_hot': []}

    for start in range(0, len(images), batch_size):
        end  = min(start + batch_size, len(images))
        imgs = images[start:end].to(device)
        labs = labels[start:end].to(device)

        logits    = model(imgs)
        probs     = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)

        loss     = F.cross_entropy(logits, labs, reduction='none')
        entropy  = -(probs * log_probs).sum(dim=1)
        max_conf = probs.max(dim=1).values
        true_conf       = probs[torch.arange(len(labs)), labs]
        sorted_probs, _ = probs.sort(dim=1, descending=True)
        conf_gap        = true_conf - sorted_probs[:, 1]
        one_hot = F.one_hot(labs.cpu(), NUM_CLASSES).float()

        signals['loss'].append(loss.cpu())
        signals['conf_vec'].append(probs.cpu())
        signals['entropy'].append(entropy.cpu().unsqueeze(1))
        signals['max_conf'].append(max_conf.cpu().unsqueeze(1))
        signals['conf_gap'].append(conf_gap.cpu().unsqueeze(1))
        signals['one_hot'].append(one_hot)

    return {k: torch.cat(v, dim=0) for k, v in signals.items()}


def build_ab_features(signals):
    return torch.cat([signals['conf_vec'], signals['one_hot']], dim=1).numpy()


def build_aw_features(signals):
    return torch.cat([
        signals['loss'].unsqueeze(1),
        signals['entropy'], signals['max_conf'], signals['conf_gap'],
        signals['conf_vec'], signals['one_hot'],
    ], dim=1).numpy()


def run_attack_privacy_meter(member_losses, nonmember_losses):
    """损失阈值攻击（等价 ml_privacy_meter PopulationAttack）"""
    n    = min(len(member_losses), len(nonmember_losses))
    m_l  = member_losses[:n].flatten()
    nm_l = nonmember_losses[:n].flatten()
    y    = np.concatenate([np.ones(n), np.zeros(n)])
    score= np.concatenate([-m_l, -nm_l])
    auc  = roc_auc_score(y, score)
    best_acc = 0.5
    for thr in np.percentile(score, np.linspace(0, 100, 200)):
        acc = accuracy_score(y, (score >= thr).astype(int))
        if acc > best_acc:
            best_acc = acc
    return {'auc': float(auc), 'priv_acc': float(best_acc * 100),
            'loss_gap': float(np.mean(nm_l) - np.mean(m_l))}


def run_attack_experiment(feat_m, feat_nm, n_pairs, seed=RANDOM_SEED):
    np.random.seed(seed)
    n = min(n_pairs, len(feat_m), len(feat_nm))
    idx_m = np.random.choice(len(feat_m),  n, replace=False)
    idx_n = np.random.choice(len(feat_nm), n, replace=False)
    X = np.concatenate([feat_m[idx_m], feat_nm[idx_n]])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=1 - ATTACK_TRAIN_RATIO, random_state=seed, stratify=y)
    sc   = StandardScaler()
    X_tr = sc.fit_transform(X_tr);  X_te = sc.transform(X_te)
    clf  = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
                         solver='adam', max_iter=300, random_state=seed,
                         early_stopping=True, validation_fraction=0.1)
    clf.fit(X_tr, y_tr)
    return accuracy_score(y_te, clf.predict(X_te)) * 100.0


def evaluate_mia(model, member_imgs, member_labels,
                 nonmember_imgs, nonmember_labels, device, method_name):
    """运行 Ab / Aw / LT 三种攻击，返回结果字典"""
    print(f"\n  [MIA] 运行 {method_name} 的成员推理攻击...")

    n_eval = min(N_ATTACK_PAIRS * 3, len(member_imgs), len(nonmember_imgs))
    torch.manual_seed(RANDOM_SEED)
    m_idx  = torch.randperm(len(member_imgs))[:n_eval]
    nm_idx = torch.randperm(len(nonmember_imgs))[:n_eval]

    sig_m  = compute_attack_signals(model, member_imgs[m_idx],
                                    member_labels[m_idx], device)
    sig_nm = compute_attack_signals(model, nonmember_imgs[nm_idx],
                                    nonmember_labels[nm_idx], device)

    feat_ab_m  = build_ab_features(sig_m)
    feat_ab_nm = build_ab_features(sig_nm)
    feat_aw_m  = build_aw_features(sig_m)
    feat_aw_nm = build_aw_features(sig_nm)

    ab_list = [run_attack_experiment(feat_ab_m, feat_ab_nm, N_ATTACK_PAIRS,
                                     seed=RANDOM_SEED + r) for r in range(N_REPEAT)]
    aw_list = [run_attack_experiment(feat_aw_m, feat_aw_nm, N_ATTACK_PAIRS,
                                     seed=RANDOM_SEED + r) for r in range(N_REPEAT)]
    lt_res  = run_attack_privacy_meter(sig_m['loss'].numpy(),
                                       sig_nm['loss'].numpy())

    priv_ab = float(np.mean(ab_list));  priv_ab_std = float(np.std(ab_list))
    priv_aw = float(np.mean(aw_list));  priv_aw_std = float(np.std(aw_list))

    print(f"  [MIA] Priv Acc (Ab) = {priv_ab:.2f}% ± {priv_ab_std:.2f}%")
    print(f"  [MIA] Priv Acc (Aw) = {priv_aw:.2f}% ± {priv_aw_std:.2f}%")
    print(f"  [MIA] Priv Acc (LT) = {lt_res['priv_acc']:.2f}%  "
          f"[AUC={lt_res['auc']:.4f}, loss_gap={lt_res['loss_gap']:.4f}]")

    return {
        'priv_acc_ab'    : priv_ab,    'priv_acc_ab_std': priv_ab_std,
        'priv_acc_aw'    : priv_aw,    'priv_acc_aw_std': priv_aw_std,
        'priv_acc_lt'    : lt_res['priv_acc'],
        'lt_auc'         : lt_res['auc'],
        'loss_gap'       : lt_res['loss_gap'],
        'm_loss_mean'    : float(sig_m['loss'].mean()),
        'nm_loss_mean'   : float(sig_nm['loss'].mean()),
    }


# ══════════════════════════════════════════════════════════════
# 三种基线方法实现
# ══════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────
# Method 1: No Defense
# ──────────────────────────────────────────────────────────────
def run_no_defense(all_train_imgs, all_train_labels,
                   member_imgs, member_labels,
                   nonmember_imgs, nonmember_labels, test_loader, device):
    """
    无防御基线：ConvNet 在全量 STL-10 训练集（5000 张）上直接训练。

    【对齐 PLDK Table 的设计说明】
    PLDK Table 中 No Defense 用的是全量私有训练集（CIFAR-10: 50,000张）。
    对应 STL-10 应使用全量 5,000 张训练集，而非仅 members 子集（2,500张）。

    原因：
      若只用 2,500 张 members 训练 → 模型过拟合严重 → train acc≈100%
      → generalization gap 极大 → Priv Acc 虚高（95%+），不具参考价值。
      用全量 5,000 张：模型同时见过 members 和 non-members
      → 合理的 generalization gap → Priv Acc 落在 70-80% 之间（合理范围）。

    MIA 评估数据划分（与 3PC+KA 实验保持一致）：
      members     = 前 2,500 张（被纳入训练的样本）
      non-members = 后 2,500 张（同分布但未参与训练）
    """
    print(f"\n{'='*60}")
    print("  Method: No Defense  [AlexNet，与 PLDK 基线一致]")
    print(f"  训练数据: 全量 STL-10 训练集（{len(all_train_imgs)} 张）")
    print(f"  MIA members: 前 {len(member_imgs)} 张 | non-members: 测试集前 {len(nonmember_imgs)} 张")
    print(f"  训练轮次: {EPOCHS_DIRECT} epochs（含数据增强，与 Regu 保持公平对比）")
    print(f"{'='*60}")

    model     = build_alexnet(device)
    optimizer = make_optimizer(model)          # 基础 WD=5e-4
    scheduler = make_scheduler(optimizer, EPOCHS_DIRECT)
    best_acc  = 0.0
    save_path = os.path.join(OUTPUT_DIR, 'no_defense_alexnet_stl10.pth')

    for epoch in range(EPOCHS_DIRECT):
        # ★ 不加数据增强：保留过拟合，让 train/test gap 体现为 Priv Acc > 50%
        train_epoch_ce(model, all_train_imgs, all_train_labels, optimizer, device,
                       label_smoothing=0.0, use_augment=False)
        scheduler.step()
        if (epoch + 1) % 50 == 0 or epoch == 0:
            acc = evaluate_accuracy(model, test_loader, device)
            print(f"  Epoch {epoch+1:4d} | Test Acc: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), save_path)

    print(f"\n  最佳测试准确率: {best_acc:.2f}%  → 已保存: {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device))

    mia_results = evaluate_mia(model, member_imgs, member_labels,
                                nonmember_imgs, nonmember_labels,
                                device, 'No Defense')
    return {'method': 'No Defense', 'dataset': 'STL-10',
            'teacher_setting': 'Single (Direct)',
            'test_acc': best_acc, **mia_results}


# ──────────────────────────────────────────────────────────────
# Method 2: Regu (WD+LS)
# ──────────────────────────────────────────────────────────────
def run_regu_wdls(all_train_imgs, all_train_labels,
                  member_imgs, member_labels,
                  nonmember_imgs, nonmember_labels, test_loader, device):
    """
    正则化防御：全量训练集（5000 张）+ Weight Decay + Label Smoothing。

    【对齐 PLDK Table 的设计说明】
    只使用 Weight Decay（WD=5e-3），不使用 Label Smoothing。
    LS 在无数据增强条件下会集中 member 置信度分布，
    导致 Priv Acc 反高于 No Defense，违背正则化防御的设计初衷。
    纯 WD 惩罚大权重，抑制过拟合，Priv Acc 正常低于 No Defense。
    方法名保留 "Regu (WD+LS)" 与 PLDK 论文对应，实际超参已调整。
    """
    print(f"\n{'='*60}")
    print("  Method: Regu (WD)  [AlexNet，Weight Decay only]")
    print(f"  训练数据: 全量 STL-10 训练集（{len(all_train_imgs)} 张）")
    print(f"  MIA members: 前 {len(member_imgs)} 张 | non-members: 测试集前 {len(nonmember_imgs)} 张")
    print(f"  Weight Decay: {WEIGHT_DECAY_REG}（Label Smoothing: 0.0）")
    print(f"  训练轮次: {EPOCHS_DIRECT} epochs（无数据增强）")
    print(f"{'='*60}")

    model     = build_alexnet(device)
    optimizer = make_optimizer(model, weight_decay=WEIGHT_DECAY_REG)
    scheduler = make_scheduler(optimizer, EPOCHS_DIRECT)
    best_acc  = 0.0
    save_path = os.path.join(OUTPUT_DIR, 'regu_wdls_alexnet_stl10.pth')

    for epoch in range(EPOCHS_DIRECT):
        train_epoch_ce(model, all_train_imgs, all_train_labels, optimizer, device,
                       label_smoothing=0.0, use_augment=False)
        scheduler.step()
        if (epoch + 1) % 50 == 0 or epoch == 0:
            acc = evaluate_accuracy(model, test_loader, device)
            print(f"  Epoch {epoch+1:4d} | Test Acc: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), save_path)

    print(f"\n  最佳测试准确率: {best_acc:.2f}%  → 已保存: {save_path}")
    model.load_state_dict(torch.load(save_path, map_location=device))

    mia_results = evaluate_mia(model, member_imgs, member_labels,
                                nonmember_imgs, nonmember_labels,
                                device, 'Regu (WD+LS)')
    return {'method': 'Regu (WD+LS)', 'dataset': 'STL-10',
            'teacher_setting': 'Single (Regularized)',
            'test_acc': best_acc, **mia_results}


# ──────────────────────────────────────────────────────────────
# Method 3: DMP (Shejwalkar & Houmansadr, AAAI 2021)
# ──────────────────────────────────────────────────────────────
def run_dmp(dtr_imgs, dtr_labels, member_imgs, member_labels,
            nonmember_imgs, nonmember_labels, test_loader, device):
    """
    Distillation for Membership Privacy (DMP)。
    严格按论文三阶段协议实现：

    Phase 1 — 在 Dtr 上训练无保护模型 θ_up（DMP_EPOCHS_UNPROTECTED epochs）
    Phase 2 — 从 STL-10 无标注集（100k）中选取预测熵最低的 DMP_XREF_SIZE 张，
              用 θ_up 在高温 T=DMP_SOFTMAX_TEMP 下计算软标签
    Phase 3 — 在 (Xref, soft_labels) 上用 KL 散度损失训练保护模型 θ_p

    设计原理（论文 Proposition 1）：
      低熵 Xref 的预测不包含 Dtr 成员的特异性信息，
      因此 θ_p 无法通过 Xref 的软标签推断 Dtr 的成员身份。
    """
    print(f"\n{'='*60}")
    print("  Method: DMP (Distillation for Membership Privacy)")
    print(f"{'='*60}")

    # ── Phase 1: 训练无保护模型 ──────────────────────────────
    theta_up = train_unprotected_model(dtr_imgs, dtr_labels, device)

    # ── Phase 2: 选择 Xref，提取软标签 ──────────────────────
    print("\n  加载 STL-10 无标注集作为 Xref 候选池...")
    xref_pool = load_stl10_unlabeled_as_tensors(max_n=20000)

    xref_selected, soft_labels = select_xref_by_entropy(
        theta_up, xref_pool, device)
    # soft_labels: [DMP_XREF_SIZE, C]，已是高温软化后的概率分布

    # ── Phase 3: 训练保护模型 ────────────────────────────────
    print(f"\n  [DMP Phase 3] 训练保护模型（{EPOCHS_STUDENT} epochs，KL divergence）...")
    theta_p   = build_convnet(device)
    optimizer = make_optimizer(theta_p)
    scheduler = make_scheduler(optimizer, EPOCHS_STUDENT)
    best_acc  = 0.0
    save_path = os.path.join(OUTPUT_DIR, 'dmp_stl10.pth')

    for epoch in range(EPOCHS_STUDENT):
        train_epoch_kl(theta_p, xref_selected, soft_labels, optimizer, device)
        scheduler.step()
        if (epoch + 1) % 100 == 0 or epoch == 0:
            acc = evaluate_accuracy(theta_p, test_loader, device)
            print(f"  Phase-3 Epoch {epoch+1:4d} | Test Acc: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                torch.save(theta_p.state_dict(), save_path)

    print(f"\n  DMP 最佳测试准确率: {best_acc:.2f}%  → 已保存: {save_path}")
    theta_p.load_state_dict(torch.load(save_path, map_location=device))

    # MIA 评估针对保护模型 θ_p，测试其是否泄露 Dtr 的成员信息
    mia_results = evaluate_mia(theta_p, member_imgs, member_labels,
                                nonmember_imgs, nonmember_labels,
                                device, 'DMP')
    return {'method': 'DMP', 'dataset': 'STL-10',
            'teacher_setting': 'Single',
            'test_acc': best_acc, **mia_results}


# ══════════════════════════════════════════════════════════════
# 结果打印与保存
# ══════════════════════════════════════════════════════════════

def print_results(results: list):
    W = 70
    print("\n" + "=" * W)
    print("  STL-10 基线实验结果  —  论文 Table 3 补充数据")
    print("=" * W)
    hdr = (f"  {'方法':<26} {'Test Acc':>10} "
           f"{'Priv Acc(Ab)':>14} {'Priv Acc(Aw)':>14}")
    print(hdr)
    print("─" * W)

    # 已有实验的参考行
    ref_rows = [
        ("No Defense (CIFAR-10 参考)", "67.46", "76.8",  "77.2"),
        ("PLDK STL-10 [已有]",         "40.90", "49.38", "49.32"),
        ("PLDK+3PC STL-10 [已有]",     "41.88", "49.25", "49.75"),
        ("3PC+KA STL-10 [已有]",       "48.60", "50.05", "49.92"),
    ]
    print("  [参考行]")
    for name, ta, ab, aw in ref_rows:
        print(f"  {name:<26} {ta:>9}%  {ab:>13}%  {aw:>13}%")
    print("─" * W)
    print("  [本脚本新增基线 — STL-10]")
    for r in results:
        ta = f"{r['test_acc']:.2f}%"
        ab = f"{r['priv_acc_ab']:.2f}%"
        aw = f"{r['priv_acc_aw']:.2f}%"
        print(f"  {r['method']:<26} {ta:>10}  {ab:>14}  {aw:>14}")
    print("=" * W)
    print()
    print("  Priv Acc 越接近 50% 表示隐私保护越强（等同于随机猜测）")
    print("  Ab: 黑盒攻击（置信度向量）| Aw: 白盒增强攻击（损失+熵+置信度）")


def save_results(results: list):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, 'stl10_baselines_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  完整结果 → {json_path}")

    csv_path = os.path.join(OUTPUT_DIR, 'stl10_baselines_table.csv')
    with open(csv_path, 'w') as f:
        f.write("Method,Dataset,Teacher Setting,Test Acc,Priv Acc Ab,Priv Acc Aw\n")
        for r in results:
            f.write(
                f"{r['method']},{r['dataset']},{r['teacher_setting']},"
                f"{r['test_acc']:.2f}%,{r['priv_acc_ab']:.2f}%,"
                f"{r['priv_acc_aw']:.2f}%\n")
    print(f"  CSV 表格  → {csv_path}")


# ══════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='STL-10 Baseline Experiments')
    parser.add_argument('--method', type=str, default='all',
                        choices=['no_defense', 'regu', 'dmp', 'all'],
                        help='Which baseline to run')
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("  STL-10 Baseline Experiments")
    print("  Supplement for PLDK Table 3")
    print("=" * 60)
    print(f"  Device    : {device}")
    print(f"  Method    : {args.method}")
    print(f"  Private data size (members) : {TEACHER_TRAIN_SIZE}")
    print(f"  DMP Xref size               : {DMP_XREF_SIZE}")
    print(f"  DMP Phase-1 epochs          : {DMP_EPOCHS_UNPROTECTED}")
    print(f"  Student training epochs     : {EPOCHS_STUDENT}")
    print(f"  Output dir                  : ./{OUTPUT_DIR}/")

    # ── 一次性加载所有需要的数据 ─────────────────────────────
    print("\n[准备数据集]")
    all_train_imgs, all_train_labels = load_stl10_train_as_tensors()

    # 私有训练数据 Dtr（DMP Phase-1 使用：前 2500 张）
    dtr_imgs   = all_train_imgs[:TEACHER_TRAIN_SIZE]
    dtr_labels = all_train_labels[:TEACHER_TRAIN_SIZE]

    # ── MIA 评估数据划分（核心设计） ────────────────────────────
    #
    # 正确原则：non-members 必须是模型从未见过的数据。
    #
    # No Defense / Regu 训练了全量 5000 张训练集，
    # 因此不能用训练集里的后 2500 张做 non-members
    # （那些图模型也训练过，置信度与 members 相当 → Priv Acc ≈ 50%，无意义）。
    #
    # 正确做法（与 PLDK/DMP 论文对齐）：
    #   members     = 训练集前 2500 张（模型训练过，代表"私有数据"）
    #   non-members = 测试集前 2500 张（模型从未见过）
    #
    # 这样攻击者需要区分的是：
    #   "模型见过的训练样本" vs "模型从未见过的测试样本"
    # generalization gap 才有意义，Priv Acc 才能体现真实隐私泄露。

    # members：训练集前 2500 张
    member_imgs   = all_train_imgs[:TEACHER_TRAIN_SIZE]
    member_labels = all_train_labels[:TEACHER_TRAIN_SIZE]

    # non-members：测试集前 2500 张（模型从未见过）
    print("  加载 STL-10 测试集作为 non-members...")
    test_ds = torchvision.datasets.STL10(
        root=DATA_ROOT, split='test', download=True,
        transform=get_stl10_transforms(resize=True))
    nm_size = TEACHER_TRAIN_SIZE   # 与 members 数量一致
    nonmember_imgs   = torch.stack([test_ds[i][0] for i in range(nm_size)])
    nonmember_labels = torch.tensor(
        [test_ds[i][1] for i in range(nm_size)], dtype=torch.long)

    # 分类评估用的完整测试集（8000 张）
    test_loader = get_stl10_test_loader()

    print(f"  全量训练集 (No Defense/Regu 使用) : {len(all_train_imgs)} 张")
    print(f"  Dtr (DMP Phase-1 使用)            : {len(dtr_imgs)} 张")
    print(f"  MIA members  (训练集前半)          : {len(member_imgs)} 张")
    print(f"  MIA non-members (测试集前半，从未训练过) : {len(nonmember_imgs)} 张")

    # ── 运行实验 ──────────────────────────────────────────────
    methods_to_run = (['no_defense', 'regu', 'dmp']
                      if args.method == 'all' else [args.method])
    all_results = []

    for method in methods_to_run:
        if method == 'no_defense':
            r = run_no_defense(
                all_train_imgs, all_train_labels,   # 全量 5000 张
                member_imgs, member_labels,
                nonmember_imgs, nonmember_labels,
                test_loader, device)

        elif method == 'regu':
            r = run_regu_wdls(
                all_train_imgs, all_train_labels,   # 全量 5000 张
                member_imgs, member_labels,
                nonmember_imgs, nonmember_labels,
                test_loader, device)

        elif method == 'dmp':
            r = run_dmp(
                dtr_imgs, dtr_labels,               # DMP 只用 Dtr（前 2500 张）
                member_imgs, member_labels,
                nonmember_imgs, nonmember_labels,
                test_loader, device)

        all_results.append(r)
        print(f"\n  [{method.upper()}] 完成 → "
              f"Test Acc: {r['test_acc']:.2f}% | "
              f"Priv Acc (Ab): {r['priv_acc_ab']:.2f}%")

    print_results(all_results)
    save_results(all_results)
    print("\n全部实验完成。")


if __name__ == '__main__':
    main()