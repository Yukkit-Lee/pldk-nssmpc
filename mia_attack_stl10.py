"""
mia_attack.py
=============
成员推理攻击（MIA）实验，复现 PLDK 论文 Table 1 评估体系。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【本版本扩展】

  模型 1 & 2（PLDK 单教师 / 多教师无KA）：
    数据集换为 STL-10
    members    = STL-10 训练集前 TEACHER_TRAIN_SIZE_STL 张（Teacher 私有数据）
    non-members= STL-10 训练集后半段（留出集）
    test_acc   = STL-10 测试集（图像 Resize 到 32×32）

  模型 3（PLDK+NssMPC+KA，跨域）：
    Hospital A 的数据集 = CIFAR-10；Hospital B 的数据集 = STL-10
    members    = CIFAR-10 训练集前 TEACHER_TRAIN_SIZE_C10 张
                 + STL-10 训练集前 TEACHER_TRAIN_SIZE_STL 张
    non-members= CIFAR-10 训练集后半段（留出集）
                 + STL-10 训练集后半段（留出集）
    test_acc   = 分别报告 CIFAR-10 测试集 和 STL-10 测试集准确率
    查询 Student 时：CIFAR-10 图像用 C10 归一化；STL-10 用 STL 归一化 + Resize(32)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【成员/非成员设计原则（所有数据集通用）】

  "成员"  = Teacher 直接训练过的私有原始图像
  "非成员"= Teacher 和 Student 均未直接训练的留出图像
  Student 只在蒸馏合成数据上训练，对两者置信度相近 → priv_acc ≈ 50%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
攻击类型：
  Ab  黑盒：置信度向量 + 真实标签
  Aw  白盒增强：损失 + 熵 + 最大置信度 + margin + 置信度向量 + 标签
  LT  损失阈值：等价 ml_privacy_meter PopulationAttack

运行：python mia_attack.py
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ╔══════════════════════════════════════════════════════════════╗
# ║                  ★★★  用户配置区  ★★★                        ║
# ╚══════════════════════════════════════════════════════════════╝

# ── 待评估模型列表 ──────────────────────────────────────────────
# dataset 字段决定使用哪套成员/非成员数据及测试集
#   'stl10'  → 模型 1 & 2：STL-10 单域评估
#   'cross'  → 模型 3：CIFAR-10 + STL-10 跨域评估
MODEL_CONFIGS = {
    'PLDK (Single Teacher, STL-10)': {
        'checkpoint' : 'best_student_stl10.pth',
        'description': 'PLDK baseline, single teacher, STL-10, no encryption',
        'dataset'    : 'stl10',
    },
    'PLDK+NssMPC (Multi, no KA, STL-10)': {
        'checkpoint' : 'best_2pc_stl10.pth',
        'description': '3PC multi-teacher, NssMPC, STL-10, no Knowledge Alignment',
        'dataset'    : 'stl10',
    },
    'PLDK+NssMPC+KA (Cross-Domain)': {
        'checkpoint' : 'best_student_3pc_ka_generic_stl10_cifar10.pth',
        'description': '3PC multi-teacher, NssMPC + KA, Hospital-A=CIFAR-10, Hospital-B=STL-10',
        'dataset'    : 'cross',
    },
}

# ── 数据集参数 ──────────────────────────────────────────────────
CIFAR10_MEAN  = (0.4914, 0.4822, 0.4465)
CIFAR10_STD   = (0.2023, 0.1994, 0.2010)

STL10_MEAN    = (0.4467, 0.4398, 0.4066)
STL10_STD     = (0.2603, 0.2566, 0.2713)

DATA_ROOT     = './data'
NUM_CLASSES   = 10

# ── MIA 数据划分参数 ────────────────────────────────────────────
# CIFAR-10：共 50000 训练图像，论文用前 25000 作 Teacher 私有训练集
TEACHER_TRAIN_SIZE_C10  = 25000

# STL-10：共 5000 训练图像（500×10类），前 2500 作 Teacher 私有训练集
TEACHER_TRAIN_SIZE_STL  = 2500

# 每次攻击使用的成员/非成员对数
N_ATTACK_PAIRS    = 1000   # STL-10 总量较小，设为 1000

ATTACK_TRAIN_RATIO = 0.70  # 攻击模型训练集比例
N_REPEAT           = 5     # 重复次数（取均值）
RANDOM_SEED        = 42

# ── 输出目录 ────────────────────────────────────────────────────
OUTPUT_DIR = 'mia_results_stl10'


# ══════════════════════════════════════════════════════════════
# Student 模型加载
# ══════════════════════════════════════════════════════════════

def build_student(checkpoint_path: str, device: str) -> nn.Module:
    """加载训练好的 Student（ConvNet），im_size=(32,32)适用于所有数据集"""
    try:
        from utils import get_network
    except ImportError:
        raise ImportError(
            "找不到 utils.py（包含 get_network）。\n"
            "请确保 mia_attack.py 与 MTT/PLDK 项目文件在同一目录。"
        )
    model = get_network('ConvNet', channel=3,
                        num_classes=NUM_CLASSES, im_size=(32, 32))
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"找不到模型检查点: {checkpoint_path}\n"
            f"请先完成训练并在 MODEL_CONFIGS 中设置正确路径。"
        )
    state = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════
# 数据集加载（STL-10 / CIFAR-10 / 跨域）
# ══════════════════════════════════════════════════════════════

def _load_full_trainset(dataset_name: str):
    """
    加载完整训练集为 tensor，返回 (imgs, labels)。
    STL-10 图像会 Resize 到 32×32（与 Student 输入匹配）。
    """
    if dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        ds = torchvision.datasets.CIFAR10(
            root=DATA_ROOT, train=True, download=True, transform=transform)
    elif dataset_name == 'stl10':
        # STL-10 原图 96×96，Resize 到 32×32 匹配 Student 输入
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(STL10_MEAN, STL10_STD),
        ])
        ds = torchvision.datasets.STL10(
            root=DATA_ROOT, split='train', download=True, transform=transform)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    print(f"      加载 {dataset_name.upper()} 训练集（{len(ds)} 张）...")
    imgs   = torch.stack([ds[i][0] for i in range(len(ds))])
    labels = torch.tensor([ds[i][1] for i in range(len(ds))], dtype=torch.long)
    return imgs, labels


def get_stl10_split():
    """
    STL-10 成员/非成员划分（模型 1 & 2 使用）：
      members    : STL-10 训练集前 TEACHER_TRAIN_SIZE_STL 张（Teacher 私有数据）
      non-members: STL-10 训练集后半段（Teacher 和 Student 均未直接训练）
    所有图像 Resize 到 32×32，使用 STL-10 归一化。
    """
    imgs, labels = _load_full_trainset('stl10')
    n_member = TEACHER_TRAIN_SIZE_STL
    member_imgs   = imgs[:n_member]
    member_labels = labels[:n_member]
    non_member_imgs   = imgs[n_member:]
    non_member_labels = labels[n_member:]
    print(f"      成员   : STL-10 训练集前 {n_member} 张（Teacher 私有数据）")
    print(f"      非成员 : STL-10 训练集后 {len(non_member_imgs)} 张（留出集）")
    return (member_imgs, member_labels), (non_member_imgs, non_member_labels)


def get_cross_domain_split():
    """
    跨域成员/非成员划分（模型 3 使用）：
      Hospital A 使用 CIFAR-10 → CIFAR-10 训练集前半段为成员
      Hospital B 使用 STL-10  → STL-10  训练集前半段为成员
      非成员来自两个数据集的留出集

    注意：两个数据集均用各自的归一化参数，Resize 统一为 32×32，
    因此可以直接 cat 拼接后输入同一个 Student 模型。
    """
    print("      [跨域] 加载 CIFAR-10 训练集...")
    c10_imgs, c10_labels = _load_full_trainset('cifar10')
    print("      [跨域] 加载 STL-10 训练集...")
    stl_imgs, stl_labels = _load_full_trainset('stl10')

    # 成员：两个 Teacher 的私有训练集拼接
    nc = TEACHER_TRAIN_SIZE_C10
    ns = TEACHER_TRAIN_SIZE_STL
    member_imgs   = torch.cat([c10_imgs[:nc],  stl_imgs[:ns]],  dim=0)
    member_labels = torch.cat([c10_labels[:nc], stl_labels[:ns]], dim=0)

    # 非成员：两个数据集的留出集拼接
    non_member_imgs   = torch.cat([c10_imgs[nc:],  stl_imgs[ns:]],  dim=0)
    non_member_labels = torch.cat([c10_labels[nc:], stl_labels[ns:]], dim=0)

    print(f"      成员   : CIFAR-10 前{nc}张 + STL-10 前{ns}张 = {len(member_imgs)} 张")
    print(f"      非成员 : CIFAR-10 后{len(c10_imgs)-nc}张 + STL-10 后{len(stl_imgs)-ns}张"
          f" = {len(non_member_imgs)} 张")
    return (member_imgs, member_labels), (non_member_imgs, non_member_labels)


# ══════════════════════════════════════════════════════════════
# 测试集准确率评估
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def _eval_on_loader(model: nn.Module, loader, device: str) -> float:
    """在给定 DataLoader 上评估分类准确率"""
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        preds    = model(imgs.to(device)).argmax(dim=1)
        correct += preds.eq(labels.to(device)).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


def evaluate_test_acc(model: nn.Module, device: str, dataset: str) -> dict:
    """
    根据 dataset 字段选择评估集：
      'stl10' → STL-10 测试集（Resize 到 32×32）
      'cross' → CIFAR-10 测试集 + STL-10 测试集，分别报告
    返回 dict：{'cifar10': acc} 或 {'stl10': acc} 或 {'cifar10': acc, 'stl10': acc}
    """
    results = {}

    if dataset in ('stl10', 'cross'):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(STL10_MEAN, STL10_STD),
        ])
        stl_test  = torchvision.datasets.STL10(
            root=DATA_ROOT, split='test', download=True, transform=transform)
        stl_loader = torch.utils.data.DataLoader(
            stl_test, batch_size=256, shuffle=False, num_workers=2)
        results['stl10'] = _eval_on_loader(model, stl_loader, device)

    if dataset in ('cifar10', 'cross'):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        c10_test  = torchvision.datasets.CIFAR10(
            root=DATA_ROOT, train=False, download=True, transform=transform)
        c10_loader = torch.utils.data.DataLoader(
            c10_test, batch_size=256, shuffle=False, num_workers=2)
        results['cifar10'] = _eval_on_loader(model, c10_loader, device)

    return results


# ══════════════════════════════════════════════════════════════
# 攻击信号（特征）提取
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_attack_signals(model: nn.Module,
                            images: torch.Tensor,
                            labels: torch.Tensor,
                            device: str,
                            batch_size: int = 256) -> dict:
    """
    提取多种攻击信号（所有数据集通用）：

    所有输入图像应已完成：
      · 对应数据集的归一化（CIFAR-10 / STL-10 各自的 mean/std）
      · Resize 到 32×32（统一 Student 输入尺寸）
    因此跨域数据可以直接混合查询。

    黑盒信号（Ab）：softmax 置信度向量（10D）+ one-hot 标签（10D）→ 20D
    白盒增强（Aw）：损失(1) + 熵(1) + 最大置信度(1) + margin(1) + 置信度(10) + one-hot(10) → 24D
    损失阈值（LT）：交叉熵损失（1D），等价 ml_privacy_meter PopulationAttack
    """
    model.eval()
    all_signals = {
        'loss'    : [],
        'conf_vec': [],
        'entropy' : [],
        'max_conf': [],
        'conf_gap': [],
        'one_hot' : [],
    }

    for start in range(0, len(images), batch_size):
        end  = min(start + batch_size, len(images))
        imgs = images[start:end].to(device)
        labs = labels[start:end].to(device)

        logits    = model(imgs)
        probs     = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)

        # 逐样本交叉熵损失（成员损失通常偏低）
        loss = F.cross_entropy(logits, labs, reduction='none')

        # 预测熵（成员通常更低）
        entropy = -(probs * log_probs).sum(dim=1)

        # 最大置信度（成员通常更高）
        max_conf = probs.max(dim=1).values

        # 真实类别 vs 次高类别的 margin（成员通常更大）
        true_conf        = probs[torch.arange(len(labs)), labs]
        sorted_probs, _  = probs.sort(dim=1, descending=True)
        conf_gap         = true_conf - sorted_probs[:, 1]

        # one-hot 标签
        one_hot = F.one_hot(labs.cpu(), NUM_CLASSES).float()

        all_signals['loss'].append(loss.cpu())
        all_signals['conf_vec'].append(probs.cpu())
        all_signals['entropy'].append(entropy.cpu().unsqueeze(1))
        all_signals['max_conf'].append(max_conf.cpu().unsqueeze(1))
        all_signals['conf_gap'].append(conf_gap.cpu().unsqueeze(1))
        all_signals['one_hot'].append(one_hot)

    return {k: torch.cat(v, dim=0) for k, v in all_signals.items()}


def build_ab_features(signals: dict) -> np.ndarray:
    """黑盒攻击（Ab）特征：置信度向量(10) + one-hot(10) = 20 维"""
    return torch.cat([signals['conf_vec'], signals['one_hot']], dim=1).numpy()


def build_aw_features(signals: dict) -> np.ndarray:
    """
    白盒增强攻击（Aw）特征：
    损失(1) + 熵(1) + 最大置信度(1) + margin(1) + 置信度向量(10) + one-hot(10) = 24 维
    """
    return torch.cat([
        signals['loss'].unsqueeze(1),
        signals['entropy'],
        signals['max_conf'],
        signals['conf_gap'],
        signals['conf_vec'],
        signals['one_hot'],
    ], dim=1).numpy()


# ══════════════════════════════════════════════════════════════
# ml_privacy_meter v2 等价实现（损失阈值攻击 LT）
# ══════════════════════════════════════════════════════════════

def run_privacy_meter_loss_attack(member_losses: np.ndarray,
                                   nonmember_losses: np.ndarray) -> dict:
    """
    损失阈值攻击（等价 ml_privacy_meter v2 PopulationAttack）：
    · 成员损失通常 < 非成员损失
    · 扫描阈值，选最优攻击准确率
    · 输出 AUC 和最优阈值准确率
    """
    n    = min(len(member_losses), len(nonmember_losses))
    m_l  = member_losses[:n].flatten()
    nm_l = nonmember_losses[:n].flatten()

    y     = np.concatenate([np.ones(n), np.zeros(n)])
    score = np.concatenate([-m_l, -nm_l])   # 损失取负：低损失=高分=成员

    auc      = roc_auc_score(y, score)
    best_acc = 0.5
    for thr in np.percentile(score, np.linspace(0, 100, 200)):
        pred     = (score >= thr).astype(int)
        acc      = accuracy_score(y, pred)
        if acc > best_acc:
            best_acc = acc

    loss_gap = float(np.mean(nm_l) - np.mean(m_l))
    return {
        'auc'      : float(auc),
        'priv_acc' : float(best_acc * 100),
        'loss_gap' : loss_gap,
    }


def try_import_privacy_meter():
    """尝试导入 ml_privacy_meter v2（v2 为配置文件驱动，仅做版本检测）"""
    try:
        import importlib
        pm      = importlib.import_module('privacy_meter')
        version = getattr(pm, '__version__', 'unknown')
        print(f"  ml_privacy_meter : 已检测到 v{version}（v2 配置文件驱动）")
        print("                     本脚本使用等价的直接实现，结果相同")
        return True
    except ImportError:
        print("  ml_privacy_meter : 未安装（使用内置等价实现，结果相同）")
        return False


# ══════════════════════════════════════════════════════════════
# 攻击模型
# ══════════════════════════════════════════════════════════════

def build_attack_clf():
    """
    二分类攻击 MLP（对标论文 Nasr et al. 2019 的攻击模型结构）。
    输入：信号特征向量，输出：成员/非成员概率。
    """
    return MLPClassifier(
        hidden_layer_sizes  = (128, 64),
        activation          = 'relu',
        solver              = 'adam',
        max_iter            = 300,
        random_state        = RANDOM_SEED,
        early_stopping      = True,
        validation_fraction = 0.1,
        n_iter_no_change    = 15,
    )


def run_attack_experiment(feat_member: np.ndarray,
                          feat_nonmember: np.ndarray,
                          n_pairs: int,
                          seed: int = RANDOM_SEED) -> float:
    """
    执行单次攻击实验，返回攻击准确率（%）。
    流程：平衡采样 → 拼接标签 → 拆分 → StandardScaler → 训练 MLP → 评估
    """
    np.random.seed(seed)
    n = min(n_pairs, len(feat_member), len(feat_nonmember))

    idx_m = np.random.choice(len(feat_member),    n, replace=False)
    idx_n = np.random.choice(len(feat_nonmember), n, replace=False)

    X = np.concatenate([feat_member[idx_m], feat_nonmember[idx_n]], axis=0)
    y = np.concatenate([np.ones(n), np.zeros(n)])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=1 - ATTACK_TRAIN_RATIO,
        random_state=seed, stratify=y)

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)

    clf = build_attack_clf()
    clf.fit(X_tr, y_tr)
    return accuracy_score(y_te, clf.predict(X_te)) * 100.0


# ══════════════════════════════════════════════════════════════
# 单模型完整评估
# ══════════════════════════════════════════════════════════════

def evaluate_one_model(model_name: str, cfg: dict, device: str,
                       member_data: tuple, nonmember_data: tuple) -> dict:
    """
    对单个 Student 完整执行 MIA 评估：
      1. 加载模型 + 评估 Test Acc（按 dataset 类型选择测试集）
      2. 提取成员/非成员攻击信号
      3. Ab / Aw / LT 三种攻击（各 N_REPEAT 次取均值）
    """
    print(f"\n{'─'*64}")
    print(f"  评估: {model_name}")
    print(f"  数据集模式: {cfg['dataset'].upper()}")
    print(f"{'─'*64}")

    # [1] 加载模型
    print(f"  [1/4] 加载检查点: {cfg['checkpoint']}")
    model = build_student(cfg['checkpoint'], device)

    # [2] 评估 Test Accuracy
    print("  [2/4] 评估分类准确率...")
    acc_dict = evaluate_test_acc(model, device, cfg['dataset'])
    for ds_name, acc in acc_dict.items():
        print(f"         Test Acc [{ds_name.upper()}] = {acc:.2f}%")
    # 取第一个（或平均）作为主要指标
    test_acc_main = float(np.mean(list(acc_dict.values())))

    # [3] 提取攻击信号
    print("  [3/4] 提取攻击信号...")
    member_imgs, member_labels       = member_data
    nonmember_imgs, nonmember_labels = nonmember_data

    # 取平衡子集，保持类别均衡
    n_eval = min(N_ATTACK_PAIRS * 3,
                 len(member_imgs), len(nonmember_imgs))
    torch.manual_seed(RANDOM_SEED)
    m_idx  = torch.randperm(len(member_imgs))[:n_eval]
    nm_idx = torch.randperm(len(nonmember_imgs))[:n_eval]

    print(f"         成员信号（{n_eval} 张）...")
    sig_m  = compute_attack_signals(
        model, member_imgs[m_idx], member_labels[m_idx], device)

    print(f"         非成员信号（{n_eval} 张）...")
    sig_nm = compute_attack_signals(
        model, nonmember_imgs[nm_idx], nonmember_labels[nm_idx], device)

    feat_ab_m  = build_ab_features(sig_m)
    feat_ab_nm = build_ab_features(sig_nm)
    feat_aw_m  = build_aw_features(sig_m)
    feat_aw_nm = build_aw_features(sig_nm)

    print(f"         Ab 特征维度: {feat_ab_m.shape[1]}")
    print(f"         Aw 特征维度: {feat_aw_m.shape[1]}")

    # [4] 运行攻击
    print(f"  [4/4] 运行攻击（重复 {N_REPEAT} 次）...")

    # 黑盒 Ab
    ab_list = [
        run_attack_experiment(feat_ab_m, feat_ab_nm,
                              N_ATTACK_PAIRS, seed=RANDOM_SEED + r)
        for r in range(N_REPEAT)
    ]
    priv_ab     = float(np.mean(ab_list))
    priv_ab_std = float(np.std(ab_list))
    print(f"         Priv Acc (Ab) = {priv_ab:.2f}% ± {priv_ab_std:.2f}%")

    # 白盒增强 Aw
    aw_list = [
        run_attack_experiment(feat_aw_m, feat_aw_nm,
                              N_ATTACK_PAIRS, seed=RANDOM_SEED + r)
        for r in range(N_REPEAT)
    ]
    priv_aw     = float(np.mean(aw_list))
    priv_aw_std = float(np.std(aw_list))
    print(f"         Priv Acc (Aw) = {priv_aw:.2f}% ± {priv_aw_std:.2f}%")

    # 损失阈值 LT
    lt_result = run_privacy_meter_loss_attack(
        sig_m['loss'].numpy(), sig_nm['loss'].numpy())
    print(f"         Priv Acc (LT) = {lt_result['priv_acc']:.2f}%  "
          f"[AUC={lt_result['auc']:.4f}, loss_gap={lt_result['loss_gap']:.4f}]")

    # 诊断信息
    m_loss_mean  = float(sig_m['loss'].mean())
    nm_loss_mean = float(sig_nm['loss'].mean())
    print(f"         Loss 均值 — 成员: {m_loss_mean:.4f} | 非成员: {nm_loss_mean:.4f}")
    print(f"         差值接近 0 → Student 无法区分成员/非成员 → 隐私保护有效")

    return {
        'model_name'     : model_name,
        'description'    : cfg['description'],
        'dataset'        : cfg['dataset'],
        'test_acc_detail': acc_dict,
        'test_acc'       : test_acc_main,
        'priv_acc_ab'    : priv_ab,
        'priv_acc_ab_std': priv_ab_std,
        'priv_acc_aw'    : priv_aw,
        'priv_acc_aw_std': priv_aw_std,
        'priv_acc_lt'    : lt_result['priv_acc'],
        'lt_auc'         : lt_result['auc'],
        'loss_gap'       : lt_result['loss_gap'],
        'm_loss_mean'    : m_loss_mean,
        'nm_loss_mean'   : nm_loss_mean,
    }


# ══════════════════════════════════════════════════════════════
# 结果展示（对标 Table 1 格式）
# ══════════════════════════════════════════════════════════════

def print_table(results: list):
    """以 PLDK 论文 Table 1 格式打印对比结果"""
    W = 80
    print("\n" + "=" * W)
    print("  MIA Evaluation Results  —  STL-10 / Cross-Domain  (Student: ConvNet)")
    print("  Reference: PLDK Table 1, Nasr et al. 2019 attack settings")
    print("=" * W)

    # 论文参考值（CIFAR-10，仅作参考量级对比）
    paper_rows = [
        ("No defense (CNN) [CIFAR-10 ref]",    "CNN", "67.46", "76.8",  "77.2",  "—"),
        ("PLDK (s=500) [CIFAR-10 paper]",       "CNN", "69.30", "50.21", "50.28", "—"),
        ("PLDK (pretrain 10k) [CIFAR-10 paper]","CNN", "71.80", "51.25", "51.80", "—"),
    ]
    hdr = (f"  {'Algorithm':<40} {'Dataset':<10} {'Test':>7} "
           f"{'Ab':>8} {'Aw':>8} {'LT':>8}")
    print(hdr)
    print("─" * W)
    print("  [Paper Reference — PLDK Table 1 (CIFAR-10)]")
    for row in paper_rows:
        name, mdl, tacc, ab, aw, lt = row
        print(f"  {name:<40} {'C-10':<10} {tacc:>6}%  {ab:>7}%  {aw:>7}%  {lt:>7}")
    print("─" * W)
    print("  [Ours — Experimental Results]")
    for r in results:
        # 测试集标注
        acc_detail = r.get('test_acc_detail', {})
        if 'cifar10' in acc_detail and 'stl10' in acc_detail:
            ds_tag  = 'C10+STL'
            tacc_str = (f"C10:{acc_detail['cifar10']:.1f}%/"
                        f"STL:{acc_detail['stl10']:.1f}%")
        elif 'stl10' in acc_detail:
            ds_tag   = 'STL-10'
            tacc_str = f"{acc_detail['stl10']:.2f}%"
        else:
            ds_tag   = 'C10'
            tacc_str = f"{acc_detail.get('cifar10', r['test_acc']):.2f}%"

        ab_str = f"{r['priv_acc_ab']:.2f}%"
        aw_str = f"{r['priv_acc_aw']:.2f}%"
        lt_str = f"{r['priv_acc_lt']:.2f}%"
        print(f"  {r['model_name']:<40} {ds_tag:<10} {tacc_str:>10}  "
              f"{ab_str:>7}  {aw_str:>7}  {lt_str:>7}")

    print("=" * W)
    print()
    print("  Ab  Black-box: confidence vector + label (20-dim)")
    print("  Aw  White-box enhanced: loss + entropy + margin + conf + label (24-dim)")
    print("  LT  Loss-threshold attack (≡ ml_privacy_meter PopulationAttack)")
    print("  Priv Acc closer to 50.0% = stronger membership privacy")
    print()


# ══════════════════════════════════════════════════════════════
# 结果保存
# ══════════════════════════════════════════════════════════════

def save_results(results: list):
    """保存完整 JSON + 论文格式 CSV"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    json_path = os.path.join(OUTPUT_DIR, 'mia_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  完整结果 → {json_path}")

    csv_path = os.path.join(OUTPUT_DIR, 'mia_table.csv')
    with open(csv_path, 'w') as f:
        f.write("Algorithm,Dataset,Test Acc,Priv Acc Ab,Priv Acc Aw,Priv Acc LT,LT AUC\n")
        for r in results:
            if r.get('test_acc') is None:
                continue
            # 测试集准确率详细描述
            acc_detail = r.get('test_acc_detail', {})
            if 'cifar10' in acc_detail and 'stl10' in acc_detail:
                tacc_csv = (f"C10:{acc_detail['cifar10']:.2f}%/"
                            f"STL:{acc_detail['stl10']:.2f}%")
            elif 'stl10' in acc_detail:
                tacc_csv = f"{acc_detail['stl10']:.2f}%"
            else:
                tacc_csv = f"{r['test_acc']:.2f}%"

            f.write(
                f"{r['model_name']},{r['dataset'].upper()},"
                f"{tacc_csv},"
                f"{r['priv_acc_ab']:.2f}%,"
                f"{r['priv_acc_aw']:.2f}%,"
                f"{r['priv_acc_lt']:.2f}%,"
                f"{r['lt_auc']:.4f}\n"
            )
    print(f"  CSV 表格  → {csv_path}")


# ══════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 64)
    print("  PLDK MIA Evaluation  —  STL-10 / Cross-Domain")
    print("=" * 64)
    print(f"  使用设备        : {device}")
    print(f"  STL-10 Teacher  : 训练集前 {TEACHER_TRAIN_SIZE_STL} 张（成员）")
    print(f"  CIFAR-10 Teacher: 训练集前 {TEACHER_TRAIN_SIZE_C10} 张（成员，跨域模型）")
    print(f"  攻击对数        : {N_ATTACK_PAIRS}")
    print(f"  重复次数        : {N_REPEAT}")
    print(f"  输出目录        : ./{OUTPUT_DIR}/")

    try_import_privacy_meter()

    # 按数据集类型分组加载数据，避免重复加载
    print("\n[准备攻击数据集]")
    data_cache = {}   # key: 'stl10' or 'cross'

    for model_name, cfg in MODEL_CONFIGS.items():
        ds = cfg['dataset']
        if ds not in data_cache:
            print(f"\n  加载 {ds.upper()} 成员/非成员数据...")
            if ds == 'stl10':
                data_cache[ds] = get_stl10_split()
            elif ds == 'cross':
                data_cache[ds] = get_cross_domain_split()
            elif ds == 'cifar10':
                # 保留原始 CIFAR-10 评估能力
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
                ])
                ds_obj = torchvision.datasets.CIFAR10(
                    root=DATA_ROOT, train=True, download=True, transform=transform)
                imgs   = torch.stack([ds_obj[i][0] for i in range(len(ds_obj))])
                labels = torch.tensor(
                    [ds_obj[i][1] for i in range(len(ds_obj))], dtype=torch.long)
                n = TEACHER_TRAIN_SIZE_C10
                data_cache[ds] = (
                    (imgs[:n], labels[:n]),
                    (imgs[n:], labels[n:])
                )

    all_results = []
    for model_name, cfg in MODEL_CONFIGS.items():
        ds = cfg['dataset']
        member_data, nonmember_data = data_cache[ds]
        try:
            result = evaluate_one_model(
                model_name, cfg, device, member_data, nonmember_data)
            all_results.append(result)
        except FileNotFoundError as e:
            print(f"\n  [跳过] {model_name}")
            print(f"         {e}")
            all_results.append({
                'model_name' : model_name,
                'description': cfg['description'],
                'dataset'    : ds,
                'test_acc'   : None,
                'error'      : str(e),
            })

    valid = [r for r in all_results if r.get('test_acc') is not None]
    if valid:
        print_table(valid)
        save_results(all_results)
    else:
        print("\n[警告] 没有成功评估任何模型，请检查检查点路径配置。")

    print("实验完成。")


if __name__ == '__main__':
    main()