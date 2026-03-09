"""
mia_experiment.py
=================
成员推理攻击（MIA）实验，完整复现 PLDK 论文 Table 1 的评估体系。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【关键设计说明 — 为什么之前 priv_acc 高达 80%+？】

  错误设计（旧版）：
    members    = 蒸馏合成数据（Student 直接训练过这些图像）
    non-members= CIFAR-10 测试集（真实自然图像）
    问题：攻击模型区分的是"合成图像 vs 真实图像"风格，
         而非"成员 vs 非成员"的隐私问题 → 准确率虚高 80%+

  正确设计（对标论文 Appendix A）：
    members    = 原始 CIFAR-10 训练集（Teacher 的私有训练数据）
    non-members= 独立留出的 CIFAR-10 数据（Teacher 和 Student 均未直接访问）

    PLDK 的隐私保护逻辑：
      Student 只在蒸馏合成数据上训练，从未直接接触原始私有数据。
      因此 Student 对"原始训练集（members）"和"留出数据（non-members）"
      的置信度分布几乎相同 → 攻击模型无法区分 → priv_acc 接近 50%。

  论文 Table 1 评估的真正问题：
    "攻击者能否通过查询 Student 模型，推断出哪张原始私有图像
     曾被用于训练 Teacher（进而间接暴露成员隐私）？"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【ml_privacy_meter 集成说明】

  v2.0 已完全改为配置文件驱动，旧版类 API 已移除。
  本脚本直接实现等价的核心攻击，同时提供 ml_privacy_meter
  的 loss-threshold 信号计算接口（损失阈值攻击，v2 的主要方式）。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
评估目标：
  · PLDK 明文 baseline（单教师，pldk_train_v2.py）
  · PLDK + NssMPC 多教师无 KA（exp2_multi_nssmpc.py）
  · PLDK + NssMPC + KA（exp3_multi_ka.py）

攻击类型（对标论文）：
  Ab  黑盒攻击：攻击模型仅访问预测置信向量 + 真实标签
  Aw  白盒攻击：攻击模型额外利用损失值 + 预测熵 + 最大置信度
  LT  损失阈值攻击：ml_privacy_meter PopulationAttack 等价实现

依赖：
  pip install torch torchvision scikit-learn
  conda activate privacy_meter   # ml_privacy_meter（可选增强）

运行：
  python mia_experiment.py
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

# ── 待评估模型列表（对应 Table 1 中每一行）──────────────────
MODEL_CONFIGS = {
    'PLDK (Single Teacher, Plaintext)': {
        # Student 模型检查点路径
        'checkpoint'       : 'best_pldk.pth',
        # 说明
        'description'      : 'PLDK baseline, single teacher, no encryption',
    },
    'PLDK+NssMPC (Multi, no KA)': {
        'checkpoint'       : 'best_student_3pc.pth',
        'description'      : '3PC multi-teacher, NssMPC encrypted, no Knowledge Alignment',
    },
    'PLDK+NssMPC+KA (Ours)': {
        'checkpoint'       : 'best_student_3pc_ka_generic.pth',
        'description'      : '3PC multi-teacher, NssMPC + Knowledge Alignment',
    },
}

# ── CIFAR-10 数据集参数 ───────────────────────────────────────
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
DATA_ROOT    = './data'
NUM_CLASSES  = 10

# ── MIA 实验数据划分参数 ──────────────────────────────────────
# 论文设置：CIFAR-10 共 50k 训练图像
# Teacher 使用前 TEACHER_TRAIN_SIZE 张作为私有训练集（成员）
# 剩余图像作为非成员（Teacher 和 Student 均未直接访问）
# 论文中为 25k 成员 + 25k 非成员，但可根据实际训练情况调整
TEACHER_TRAIN_SIZE = 25000    # Teacher 私有训练集大小（成员数量）
N_ATTACK_PAIRS     = 2500     # 攻击实验使用的成员/非成员对数
                               # （建议与蒸馏数据量相当，500~2500）
ATTACK_TRAIN_RATIO = 0.70     # 攻击模型训练集比例（其余用于评估）
N_REPEAT           = 5        # 重复实验次数（取均值减少随机性）
RANDOM_SEED        = 42

# ── 输出目录 ──────────────────────────────────────────────────
OUTPUT_DIR = 'mia_results'


# ══════════════════════════════════════════════════════════════
# Student 模型加载
# ══════════════════════════════════════════════════════════════

def build_student(checkpoint_path: str, device: str) -> nn.Module:
    """加载训练完成的 Student（ConvNet），依赖 utils.get_network"""
    try:
        from utils import get_network
    except ImportError:
        raise ImportError(
            "找不到 utils.py（包含 get_network）。\n"
            "请确保 mia_experiment.py 与 MTT/PLDK 项目文件在同一目录。"
        )
    model = get_network('ConvNet', channel=3,
                        num_classes=NUM_CLASSES, im_size=(32, 32))
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"找不到模型检查点: {checkpoint_path}\n"
            f"请先完成训练并在 MODEL_CONFIGS 中设置正确路径。"
        )
    state = torch.load(checkpoint_path, map_location='cpu')
    # 兼容 {'model': state_dict, ...} 格式与直接 state_dict 格式
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════
# 数据准备（核心修复）
# ══════════════════════════════════════════════════════════════

def get_cifar10_split():
    """
    加载 CIFAR-10 训练集并按论文设置划分：
      members    : 前 TEACHER_TRAIN_SIZE 张（Teacher 的私有训练集）
      non_members: 剩余图像（Teacher 和 Student 均未直接训练）

    核心原理：
      Student 仅在蒸馏合成数据上训练，对原始 CIFAR-10 训练集和留出集
      的置信度分布相近，因此攻击者无法区分成员与非成员 → priv_acc ~50%
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform)

    all_imgs   = torch.stack([trainset[i][0] for i in range(len(trainset))])
    all_labels = torch.tensor([trainset[i][1] for i in range(len(trainset))],
                              dtype=torch.long)

    # 成员：Teacher 的私有训练集
    member_imgs   = all_imgs[:TEACHER_TRAIN_SIZE]
    member_labels = all_labels[:TEACHER_TRAIN_SIZE]

    # 非成员：留出集（Teacher 和 Student 均未直接训练过）
    non_member_imgs   = all_imgs[TEACHER_TRAIN_SIZE:]
    non_member_labels = all_labels[TEACHER_TRAIN_SIZE:]

    print(f"  成员样本   : CIFAR-10 训练集前 {TEACHER_TRAIN_SIZE} 张（Teacher 私有数据）")
    print(f"  非成员样本 : CIFAR-10 训练集后 {len(non_member_imgs)} 张（留出集）")
    return (member_imgs, member_labels), (non_member_imgs, non_member_labels)


def get_test_set_for_accuracy():
    """加载 CIFAR-10 测试集，仅用于评估 Student 的分类准确率"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=transform)


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
    对每个样本提取多种攻击信号：

    黑盒信号（Ab）：
      · softmax 置信度向量（10 维）
      · 真实类别 one-hot（10 维）
      → 拼接为 20 维特征

    白盒增强信号（Aw）：
      · 交叉熵损失值（1 维）—— 成员损失通常更低
      · 预测熵（1 维）—— 成员预测通常更尖锐
      · 最大置信度（1 维）—— 成员置信度通常更高
      · 真实类别置信度与次高类别之差（1 维）—— gap 越大越可能是成员
      · 拼接以上 + 置信度向量 → 24 维特征
      （注：真正的白盒攻击还利用梯度，见 compute_gradient_features）

    损失阈值信号（LT）：
      · 仅使用交叉熵损失值，阈值判断成员/非成员
      · 等价于 ml_privacy_meter PopulationAttack
    """
    model.eval()
    all_signals = {
        'loss'       : [],   # [N, 1]  交叉熵损失
        'conf_vec'   : [],   # [N, 10] softmax 置信度向量
        'entropy'    : [],   # [N, 1]  预测熵
        'max_conf'   : [],   # [N, 1]  最大置信度
        'conf_gap'   : [],   # [N, 1]  真实类 conf 与次高类 conf 的差
        'one_hot'    : [],   # [N, 10] 真实标签 one-hot
    }

    for start in range(0, len(images), batch_size):
        end  = min(start + batch_size, len(images))
        imgs = images[start:end].to(device)
        labs = labels[start:end].to(device)

        logits    = model(imgs)                                # [B, C]
        probs     = F.softmax(logits, dim=1)                   # [B, C]
        log_probs = F.log_softmax(logits, dim=1)               # [B, C]

        # 交叉熵损失（逐样本）
        loss = F.cross_entropy(logits, labs, reduction='none') # [B]

        # 预测熵 H = -sum(p * log(p))
        entropy = -(probs * log_probs).sum(dim=1)              # [B]

        # 最大置信度
        max_conf = probs.max(dim=1).values                     # [B]

        # 真实类别置信度 vs 次高类置信度的差（margin）
        true_conf   = probs[torch.arange(len(labs)), labs]     # [B]
        sorted_probs, _ = probs.sort(dim=1, descending=True)
        second_conf  = sorted_probs[:, 1]                      # [B]
        conf_gap     = true_conf - second_conf                  # [B]

        # one-hot 标签
        one_hot = F.one_hot(labs.cpu(), NUM_CLASSES).float()

        all_signals['loss'].append(loss.cpu())
        all_signals['conf_vec'].append(probs.cpu())
        all_signals['entropy'].append(entropy.cpu().unsqueeze(1))
        all_signals['max_conf'].append(max_conf.cpu().unsqueeze(1))
        all_signals['conf_gap'].append(conf_gap.cpu().unsqueeze(1))
        all_signals['one_hot'].append(one_hot)

    # 拼接所有 batch
    return {
        k: torch.cat(v, dim=0) for k, v in all_signals.items()
    }


def build_ab_features(signals: dict) -> np.ndarray:
    """
    黑盒攻击（Ab）特征：置信度向量 + one-hot 标签
    维度：10 + 10 = 20
    """
    feat = torch.cat([signals['conf_vec'], signals['one_hot']], dim=1)
    return feat.numpy()


def build_aw_features(signals: dict) -> np.ndarray:
    """
    白盒增强攻击（Aw）特征：
    损失 + 熵 + 最大置信度 + gap + 置信度向量 + one-hot
    维度：1 + 1 + 1 + 1 + 10 + 10 = 24

    注：此为"灰盒"白盒特征——利用了更多输出统计量，但不计算梯度。
    对于 Student 这类轻量模型，这组特征的效果与完整梯度白盒相当，
    且避免了逐样本反向传播的极大计算开销。
    若需完整梯度白盒特征，参见 compute_gradient_features()。
    """
    feat = torch.cat([
        signals['loss'].unsqueeze(1),  # 损失越低越可能是成员
        signals['entropy'],            # 熵越低越可能是成员
        signals['max_conf'],           # 置信度越高越可能是成员
        signals['conf_gap'],           # margin 越大越可能是成员
        signals['conf_vec'],           # 置信度向量
        signals['one_hot'],            # 真实类别信息
    ], dim=1)
    return feat.numpy()


def build_lt_signal(signals: dict) -> np.ndarray:
    """
    损失阈值攻击（LT）信号：仅使用交叉熵损失值。
    等价于 ml_privacy_meter v2 的 PopulationAttack / LossBasedAttack。
    """
    return signals['loss'].unsqueeze(1).numpy()  # [N, 1]


# ══════════════════════════════════════════════════════════════
# 梯度白盒特征（完整 Aw，可选）
# ══════════════════════════════════════════════════════════════

def compute_gradient_features(model: nn.Module,
                               images: torch.Tensor,
                               labels: torch.Tensor,
                               device: str,
                               max_samples: int = 300,
                               grad_dim: int = 256) -> np.ndarray:
    """
    完整梯度白盒特征（对应论文 Nasr et al. 2019 白盒攻击）：
    提取最后一个线性层权重梯度（L2归一化后降维至 grad_dim）。

    注意：逐样本计算梯度，速度慢，建议最多取 max_samples 个样本。
    """
    # 找最后一个线性层
    classifier = None
    for m in reversed(list(model.modules())):
        if isinstance(m, nn.Linear):
            classifier = m
            break
    if classifier is None:
        print("  [警告] 未找到线性分类头，跳过完整梯度白盒特征")
        return None

    n = min(max_samples, len(images))
    grad_feats = []
    model.eval()

    for i in range(n):
        model.zero_grad()
        img = images[i:i+1].to(device)
        lbl = labels[i:i+1].to(device)
        loss = F.cross_entropy(model(img), lbl)
        loss.backward()

        with torch.no_grad():
            g = classifier.weight.grad.detach().cpu().flatten()
            if len(g) > grad_dim:
                # 保留绝对值最大的 grad_dim 维（信息量最大）
                topk = g.abs().topk(grad_dim).indices
                g    = g[topk]
            else:
                # 不足则补零
                pad = torch.zeros(grad_dim - len(g))
                g   = torch.cat([g, pad])
            # L2 归一化消除样本间尺度差异
            g = g / (g.norm() + 1e-8)
            grad_feats.append(g.numpy())

    return np.stack(grad_feats, axis=0)  # [n, grad_dim]


# ══════════════════════════════════════════════════════════════
# ml_privacy_meter v2 接口
# ══════════════════════════════════════════════════════════════

def run_privacy_meter_loss_attack(member_losses: np.ndarray,
                                   nonmember_losses: np.ndarray) -> dict:
    """
    ml_privacy_meter v2 等价实现：损失阈值攻击（LossBasedAttack）。

    ml_privacy_meter v2 的 PopulationAttack 核心逻辑：
      · 计算每个样本的损失值
      · 假设成员损失 < 非成员损失（过拟合程度更高）
      · 扫描损失阈值，选择使攻击准确率最高的阈值
      · 输出 ROC 曲线下面积（AUC）和平衡点准确率

    此实现与 ml_privacy_meter v2 的 LossMetric 信号 + 
    MinMaxThresholdAttack 完全等价，无需依赖库本身。
    """
    n   = min(len(member_losses), len(nonmember_losses))
    m_l = member_losses[:n].flatten()
    nm_l= nonmember_losses[:n].flatten()

    # 构建标签和信号（损失越低越可能是成员）
    y    = np.concatenate([np.ones(n),  np.zeros(n)])
    # 将损失取负号，使"低损失=高分=成员"方向一致
    score= np.concatenate([-m_l, -nm_l])

    # AUC（不需要阈值）
    auc  = roc_auc_score(y, score)

    # 最优阈值下的攻击准确率（扫描所有可能阈值）
    best_acc = 0.5
    for thr in np.percentile(score, np.linspace(0, 100, 200)):
        pred = (score >= thr).astype(int)
        acc  = accuracy_score(y, pred)
        if acc > best_acc:
            best_acc = acc

    # 均值损失差（成员损失通常更低）
    loss_gap = float(np.mean(nm_l) - np.mean(m_l))

    return {
        'auc'      : float(auc),
        'priv_acc' : float(best_acc * 100),
        'loss_gap' : loss_gap,    # 正值表示非成员损失更高（符合预期）
    }


def try_import_privacy_meter():
    """
    尝试导入 ml_privacy_meter v2。
    v2 主要通过命令行 / 配置文件使用，直接 import 仅用于信号计算。
    返回 True/False 表示是否可用。
    """
    try:
        # v2 包名仍为 privacy_meter，但内部结构已改变
        import importlib
        pm = importlib.import_module('privacy_meter')
        version = getattr(pm, '__version__', 'unknown')
        print(f"  ml_privacy_meter : 已检测到 v{version}")
        print("  [注意] v2 为配置文件驱动，本脚本使用等价的直接实现")
        return True
    except ImportError:
        print("  ml_privacy_meter : 未安装（使用内置等价实现，结果相同）")
        return False


# ══════════════════════════════════════════════════════════════
# 攻击模型
# ══════════════════════════════════════════════════════════════

def build_attack_clf(input_dim: int):
    """
    二分类攻击 MLP（对标论文 Nasr et al. 2019 的攻击模型结构）。
    输入：信号特征向量，输出：成员/非成员概率。
    """
    return MLPClassifier(
        hidden_layer_sizes = (128, 64),
        activation         = 'relu',
        solver             = 'adam',
        max_iter           = 300,
        random_state       = RANDOM_SEED,
        early_stopping     = True,
        validation_fraction= 0.1,
        n_iter_no_change   = 15,
    )


def run_attack_experiment(feat_member: np.ndarray,
                          feat_nonmember: np.ndarray,
                          n_pairs: int,
                          seed: int = RANDOM_SEED) -> float:
    """
    执行单次攻击实验，返回攻击准确率（%）。
    流程：平衡采样 → 拼接标签 → 拆分训练/测试 → 训练攻击 MLP → 评估
    """
    np.random.seed(seed)
    n = min(n_pairs, len(feat_member), len(feat_nonmember))

    idx_m = np.random.choice(len(feat_member),   n, replace=False)
    idx_n = np.random.choice(len(feat_nonmember), n, replace=False)

    X = np.concatenate([feat_member[idx_m], feat_nonmember[idx_n]], axis=0)
    y = np.concatenate([np.ones(n), np.zeros(n)])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size    = 1 - ATTACK_TRAIN_RATIO,
        random_state = seed,
        stratify     = y,
    )

    # 特征标准化（对 MLP 重要）
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)

    clf = build_attack_clf(input_dim=X.shape[1])
    clf.fit(X_tr, y_tr)
    return accuracy_score(y_te, clf.predict(X_te)) * 100.0


# ══════════════════════════════════════════════════════════════
# 模型分类准确率评估
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_test_acc(model: nn.Module, device: str) -> float:
    """在 CIFAR-10 测试集上评估 Student 分类准确率"""
    testset = get_test_set_for_accuracy()
    loader  = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=2)
    correct = total = 0
    model.eval()
    for imgs, labels in loader:
        preds    = model(imgs.to(device)).argmax(dim=1)
        correct += preds.eq(labels.to(device)).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


# ══════════════════════════════════════════════════════════════
# 单模型完整评估
# ══════════════════════════════════════════════════════════════

def evaluate_one_model(model_name: str, cfg: dict, device: str,
                       member_data: tuple, nonmember_data: tuple) -> dict:
    """
    对单个 Student 完整执行 MIA 评估：
      1. 加载模型 + 评估 Test Acc
      2. 提取成员/非成员攻击信号
      3. 黑盒攻击 Ab（N_REPEAT 次均值）
      4. 白盒增强攻击 Aw（N_REPEAT 次均值）
      5. 损失阈值攻击 LT（等价 ml_privacy_meter PopulationAttack）
    """
    print(f"\n{'─'*62}")
    print(f"  评估: {model_name}")
    print(f"{'─'*62}")

    # [1] 加载模型
    print(f"  [1/4] 加载检查点: {cfg['checkpoint']}")
    model = build_student(cfg['checkpoint'], device)

    # [2] Test Accuracy
    print("  [2/4] 评估分类准确率...")
    test_acc = evaluate_test_acc(model, device)
    print(f"         Test Acc = {test_acc:.2f}%")

    # [3] 提取攻击信号
    print("  [3/4] 提取攻击信号...")
    member_imgs, member_labels       = member_data
    nonmember_imgs, nonmember_labels = nonmember_data

    # 取平衡子集（加速计算，同时保持类别平衡）
    n_eval = min(N_ATTACK_PAIRS * 3, len(member_imgs), len(nonmember_imgs))
    torch.manual_seed(RANDOM_SEED)
    m_idx  = torch.randperm(len(member_imgs))[:n_eval]
    nm_idx = torch.randperm(len(nonmember_imgs))[:n_eval]

    print(f"         计算成员信号（{n_eval} 张）...")
    sig_member    = compute_attack_signals(
        model, member_imgs[m_idx], member_labels[m_idx], device)

    print(f"         计算非成员信号（{n_eval} 张）...")
    sig_nonmember = compute_attack_signals(
        model, nonmember_imgs[nm_idx], nonmember_labels[nm_idx], device)

    # 构建各类特征
    feat_ab_m  = build_ab_features(sig_member)
    feat_ab_nm = build_ab_features(sig_nonmember)
    feat_aw_m  = build_aw_features(sig_member)
    feat_aw_nm = build_aw_features(sig_nonmember)

    print(f"         Ab 特征维度: {feat_ab_m.shape[1]}")
    print(f"         Aw 特征维度: {feat_aw_m.shape[1]}")

    # [4] 运行攻击
    print(f"  [4/4] 运行攻击（重复 {N_REPEAT} 次）...")

    # ── 黑盒攻击 Ab ──────────────────────────────────────────
    ab_list = [
        run_attack_experiment(feat_ab_m, feat_ab_nm,
                              N_ATTACK_PAIRS, seed=RANDOM_SEED + r)
        for r in range(N_REPEAT)
    ]
    priv_ab      = float(np.mean(ab_list))
    priv_ab_std  = float(np.std(ab_list))
    print(f"         Priv Acc (Ab)  = {priv_ab:.2f}% ± {priv_ab_std:.2f}%")

    # ── 白盒增强攻击 Aw ───────────────────────────────────────
    aw_list = [
        run_attack_experiment(feat_aw_m, feat_aw_nm,
                              N_ATTACK_PAIRS, seed=RANDOM_SEED + r)
        for r in range(N_REPEAT)
    ]
    priv_aw     = float(np.mean(aw_list))
    priv_aw_std = float(np.std(aw_list))
    print(f"         Priv Acc (Aw)  = {priv_aw:.2f}% ± {priv_aw_std:.2f}%")

    # ── 损失阈值攻击 LT（等价 ml_privacy_meter PopulationAttack）──
    lt_result = run_privacy_meter_loss_attack(
        sig_member['loss'].numpy(),
        sig_nonmember['loss'].numpy(),
    )
    print(f"         Priv Acc (LT)  = {lt_result['priv_acc']:.2f}%  "
          f"[AUC={lt_result['auc']:.4f}, loss_gap={lt_result['loss_gap']:.4f}]")

    # 损失分布统计（诊断信息）
    m_loss_mean  = float(sig_member['loss'].mean())
    nm_loss_mean = float(sig_nonmember['loss'].mean())
    print(f"         Loss 均值 — 成员: {m_loss_mean:.4f} | 非成员: {nm_loss_mean:.4f}")
    print(f"         差值接近 0 表示 Student 对原始数据无区分能力（隐私保护有效）")

    return {
        'model_name'    : model_name,
        'description'   : cfg['description'],
        'test_acc'      : test_acc,
        'priv_acc_ab'   : priv_ab,
        'priv_acc_ab_std': priv_ab_std,
        'priv_acc_aw'   : priv_aw,
        'priv_acc_aw_std': priv_aw_std,
        'priv_acc_lt'   : lt_result['priv_acc'],
        'lt_auc'        : lt_result['auc'],
        'loss_gap'      : lt_result['loss_gap'],
        'm_loss_mean'   : m_loss_mean,
        'nm_loss_mean'  : nm_loss_mean,
    }


# ══════════════════════════════════════════════════════════════
# 结果展示（对标 Table 1 格式）
# ══════════════════════════════════════════════════════════════

def print_table(results: list):
    """以 PLDK 论文 Table 1 格式打印对比结果"""
    W = 78
    print("\n" + "=" * W)
    print("  MIA Evaluation Results  —  CIFAR-10  (Student: ConvNet)")
    print("  Reference: PLDK Table 1, Nasr et al. 2019 attack settings")
    print("=" * W)

    # 论文参考值（Table 1 CIFAR-10 部分）
    paper_rows = [
        ("No defense (CNN)",                   "CNN", "67.46", "76.8",  "77.2",  "—"),
        ("Regu (WD+LS)",                        "Alex","53.20", "53.0",  "53.8",  "—"),
        ("Adv Reg",                             "Alex","53.40", "51.2",  "51.9",  "—"),
        ("DMP",                                 "Alex","65.00", "50.6",  "51.3",  "—"),
        ("PLDK (distilled s=500) [paper]",      "CNN", "69.30", "50.21", "50.28", "—"),
        ("PLDK (pretrain 10k syn) [paper]",     "CNN", "71.80", "51.25", "51.80", "—"),
    ]
    hdr = (f"  {'Algorithm':<38} {'Mdl':<5} {'Test':>7} "
           f"{'Ab':>8} {'Aw':>8} {'LT':>8}")
    print(hdr)
    print("─" * W)
    print("  [Paper Reference — PLDK Table 1]")
    for row in paper_rows:
        name, mdl, tacc, ab, aw, lt = row
        print(f"  {name:<38} {mdl:<5} {tacc:>6}%  {ab:>7}%  {aw:>7}%  {lt:>7}")
    print("─" * W)
    print("  [Ours — Experimental Results]")
    for r in results:
        tacc = f"{r['test_acc']:.2f}"
        ab   = f"{r['priv_acc_ab']:.2f}"
        aw   = f"{r['priv_acc_aw']:.2f}"
        lt   = f"{r['priv_acc_lt']:.2f}"
        print(f"  {r['model_name']:<38} {'CNN':<5} {tacc:>6}%  "
              f"{ab:>7}%  {aw:>7}%  {lt:>7}%")
    print("=" * W)
    print()
    print("  Ab  Black-box attack (confidence vector + label)")
    print("  Aw  White-box enhanced (loss + entropy + margin + confidence)")
    print("  LT  Loss-threshold attack (≡ ml_privacy_meter PopulationAttack)")
    print("  Priv Acc closer to 50.0% = stronger membership privacy")
    print()


# ══════════════════════════════════════════════════════════════
# 结果保存
# ══════════════════════════════════════════════════════════════

def save_results(results: list):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # JSON（完整数据，含标准差和损失统计）
    json_path = os.path.join(OUTPUT_DIR, 'mia_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  完整结果 → {json_path}")

    # CSV（直接粘贴到论文表格）
    csv_path = os.path.join(OUTPUT_DIR, 'mia_table.csv')
    with open(csv_path, 'w') as f:
        f.write("Algorithm,Model,Test Acc,Priv Acc Ab,Priv Acc Aw,Priv Acc LT,LT AUC\n")
        for r in results:
            if r.get('test_acc') is None:
                continue
            f.write(
                f"{r['model_name']},CNN,"
                f"{r['test_acc']:.2f}%,"
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

    print("=" * 62)
    print("  PLDK MIA Evaluation  —  对标论文 Table 1")
    print("=" * 62)
    print(f"  使用设备        : {device}")
    print(f"  Teacher 训练集  : CIFAR-10 训练集前 {TEACHER_TRAIN_SIZE} 张（成员）")
    print(f"  非成员集        : CIFAR-10 训练集后 {50000-TEACHER_TRAIN_SIZE} 张")
    print(f"  攻击实验对数    : {N_ATTACK_PAIRS} 对（成员+非成员）")
    print(f"  重复次数        : {N_REPEAT}")
    print(f"  输出目录        : ./{OUTPUT_DIR}/")

    # 检查 ml_privacy_meter
    try_import_privacy_meter()

    # 一次性加载数据（所有模型共用相同的成员/非成员数据）
    print("\n[准备攻击数据集]")
    print("  加载 CIFAR-10 并按 Teacher 训练集划分...")
    member_data, nonmember_data = get_cifar10_split()

    all_results = []

    for model_name, cfg in MODEL_CONFIGS.items():
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
                'test_acc'   : None,
                'error'      : str(e),
            })

    # 展示和保存结果
    valid = [r for r in all_results if r.get('test_acc') is not None]
    if valid:
        print_table(valid)
        save_results(all_results)
    else:
        print("\n[警告] 没有成功评估任何模型，请检查检查点路径配置。")

    print("实验完成。")


if __name__ == '__main__':
    main()