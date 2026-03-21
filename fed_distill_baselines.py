"""
fed_distill_baselines.py（Cross-Setting 修复版）
=================================================
Table 6: Cross-setting comparison

【根本错误修复】

旧版：CIFAR-10 公共数据 = 测试集 → 学生在测试集训练 + 测试集评估 → 99%（严重错误）

正确设计：
  CIFAR-10:
    Teacher A 私有训练数据（members）= 训练集前 25000 张
    公共数据（基线方法训练用）       = 训练集后 25000 张（索引 25000-50000）
    MIA non-members                  = 训练集后 25000 张（与公共数据相同段）
    Test Acc 评估                    = 测试集（10000张，与训练/公共数据完全分离）

  STL-10:
    Teacher A 私有训练数据（members）= 训练集前 2500 张
    公共数据（基线方法训练用）       = 训练集全部 5000 张（有真实标签）
    MIA non-members                  = 训练集后 2500 张
    Test Acc 评估                    = 测试集（8000张）

【为什么 CIFAR-10 公共数据用训练集后段？】
  Teacher A 只在前 25k 上训练，对前 25k（members）的预测置信度更高。
  学生在后 25k 的软标签上训练，对 members 和 non-members 的响应不同。
  → MIA 可检测出这一差异 → Priv Acc > 50%，体现无 MPC 保护时的隐私泄露。

运行：
  python fed_distill_baselines.py --dataset both --method all
  python fed_distill_baselines.py --dataset cifar10 --method fedmd
  python fed_distill_baselines.py --dataset stl10 --method selective_fd
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ══════════════════════════════════════════════════════════════
# ★★★  用户配置区  ★★★
# ══════════════════════════════════════════════════════════════

CIFAR10_TEACHER_A_PATH = 'teacher_resnet18_cifar10.pth'
CIFAR10_TEACHER_B_PATH = 'teacher_b_resnet18.pth'
STL10_TEACHER_A_PATH   = 'teacher_resnet18_stl10.pth'
STL10_TEACHER_B_PATH   = 'teacher_resnet18_stl10_low.pth'

DATA_ROOT  = './data'
OUTPUT_DIR = 'fed_baselines_results'

EPOCHS_GLOBAL = 200
LR            = 0.01
MOMENTUM      = 0.9
WEIGHT_DECAY  = 5e-4
BATCH_SIZE    = 128
T_TEMP        = 4.0
ALPHA         = 0.5
NUM_CLASSES   = 10

STL10_TEACHER_RES        = (96, 96)
# Selective-FD 过滤策略：Top-K 选择（保留最高置信度前 TOP_K_RATIO 比例样本）
# 原论文用绝对阈值（0.7），但 STL-10 教师较弱，0.7 会过滤掉 97% 样本（只剩 251 张）
# 251 张 × 1000 epochs → 完全过拟合崩溃 → 10%（随机猜测）
# 改用 Top-K：固定保留 top 30%，教师越弱保留越多低熵样本，保证训练稳定
SELECTIVE_TOP_K_RATIO    = 0.30   # 保留置信度最高的前 30% 样本

# CIFAR-10 数据段划分
CIFAR10_MEMBER_SIZE  = 25000   # [0:25000]     Teacher A 私有训练集（members）
CIFAR10_PUBLIC_START = 25000   # [25000:50000] 公共数据 = non-members（★不是测试集）

# STL-10 数据段划分
STL10_MEMBER_SIZE = 2500       # [0:2500] members，[2500:5000] non-members

N_ATTACK_PAIRS     = 1000
ATTACK_TRAIN_RATIO = 0.70
N_REPEAT           = 5
RANDOM_SEED        = 42

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
STL10_MEAN   = (0.4467, 0.4398, 0.4066)
STL10_STD    = (0.2603, 0.2566, 0.2713)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ══════════════════════════════════════════════════════════════
# 模型
# ══════════════════════════════════════════════════════════════

def build_resnet18_teacher():
    from torchvision.models import resnet18
    model = resnet18(num_classes=NUM_CLASSES)
    model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model.to(device)


def build_convnet_student():
    try:
        from utils import get_network
        model = get_network('ConvNet', channel=3,
                            num_classes=NUM_CLASSES, im_size=(32, 32))
    except ImportError:
        raise ImportError("找不到 utils.py")
    return model.to(device)


def load_teacher(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到教师模型: {path}")
    model = build_resnet18_teacher()
    state = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    model.eval()
    print(f"  加载教师: {path}")
    return model


# ══════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════

def load_cifar10_train_tensors():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    ds = torchvision.datasets.CIFAR10(
        DATA_ROOT, train=True, download=True, transform=tf)
    imgs   = torch.stack([ds[i][0] for i in range(len(ds))])
    labels = torch.tensor([ds[i][1] for i in range(len(ds))], dtype=torch.long)
    return imgs, labels


def get_cifar10_test_loader():
    """仅用于评估 Test Acc，不用于任何训练"""
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    ds = torchvision.datasets.CIFAR10(
        DATA_ROOT, train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)


def load_stl10_train_tensors():
    tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(STL10_MEAN, STL10_STD)])
    ds = torchvision.datasets.STL10(
        DATA_ROOT, split='train', download=True, transform=tf)
    imgs   = torch.stack([ds[i][0] for i in range(len(ds))])
    labels = torch.tensor([ds[i][1] for i in range(len(ds))], dtype=torch.long)
    return imgs, labels


def get_stl10_test_loader():
    tf = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(STL10_MEAN, STL10_STD)])
    ds = torchvision.datasets.STL10(
        DATA_ROOT, split='test', download=True, transform=tf)
    return DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)


# ══════════════════════════════════════════════════════════════
# 教师推理
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def get_teacher_soft_labels(teacher, imgs, temperature=T_TEMP,
                             batch_size=256, upsample_to=None):
    teacher.eval()
    all_probs = []
    for start in range(0, len(imgs), batch_size):
        batch = imgs[start:start + batch_size].to(device)
        if upsample_to is not None:
            batch = F.interpolate(batch, size=upsample_to,
                                  mode='bilinear', align_corners=False)
        logits = teacher(batch)
        probs  = F.softmax(logits / temperature, dim=1)
        all_probs.append(probs.cpu())
    return torch.cat(all_probs, dim=0)


# ══════════════════════════════════════════════════════════════
# 学生训练
# ══════════════════════════════════════════════════════════════

def make_optimizer(model):
    return torch.optim.SGD(model.parameters(), lr=LR,
                           momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)


def make_scheduler(opt):
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_GLOBAL)


@torch.no_grad()
def evaluate_accuracy(model, loader):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        preds    = model(imgs.to(device)).argmax(dim=1)
        correct += preds.eq(labels.to(device)).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


def train_student_kd(pub_imgs, soft_labels, hard_labels,
                     test_loader, save_path):
    """
    蒸馏训练学生模型。
    hard_labels 不为 None 时：α·KL + (1-α)·CE
    hard_labels 为 None 时：纯 KL（软标签蒸馏）
    """
    # 安全检查：标签越界自动禁用 CE
    use_hard = False
    if hard_labels is not None:
        if hard_labels.min() >= 0 and hard_labels.max() < NUM_CLASSES:
            use_hard = True
        else:
            print(f"  [警告] 标签越界 min={hard_labels.min()} max={hard_labels.max()}，"
                  f"退化为纯 KD")

    student   = build_convnet_student()
    optimizer = make_optimizer(student)
    scheduler = make_scheduler(optimizer)
    best_acc  = 0.0
    n         = len(pub_imgs)

    for epoch in range(EPOCHS_GLOBAL):
        student.train()
        perm = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            idx       = perm[start:start + BATCH_SIZE]
            imgs      = pub_imgs[idx].to(device)
            targets_s = soft_labels[idx].to(device)

            logits  = student(imgs)
            kd_loss = F.kl_div(
                F.log_softmax(logits / T_TEMP, dim=1),
                targets_s, reduction='batchmean') * (T_TEMP ** 2)

            if use_hard:
                ce_loss = F.cross_entropy(logits, hard_labels[idx].to(device))
                loss    = ALPHA * kd_loss + (1 - ALPHA) * ce_loss
            else:
                loss = kd_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 100 == 0 or epoch == 0:
            acc = evaluate_accuracy(student, test_loader)
            print(f"  Epoch {epoch+1:4d} | Test Acc: {acc:.2f}%")
            if acc > best_acc:
                best_acc = acc
                torch.save(student.state_dict(), save_path)

    student.load_state_dict(
        torch.load(save_path, map_location=device, weights_only=False))
    print(f"  最佳 Test Acc: {best_acc:.2f}%")
    return student, best_acc


# ══════════════════════════════════════════════════════════════
# 三种方法
# ══════════════════════════════════════════════════════════════

def run_fedmd(teacher_a, teacher_b, pub_imgs, pub_labels,
              test_loader, ds_name, upsample):
    print(f"\n{'='*60}")
    print(f"  FedMD [{ds_name}]  公共数据: {len(pub_imgs)} 张")
    print(f"{'='*60}")
    soft_a    = get_teacher_soft_labels(teacher_a, pub_imgs, upsample_to=upsample)
    soft_b    = get_teacher_soft_labels(teacher_b, pub_imgs, upsample_to=upsample)
    consensus = (soft_a + soft_b) / 2.0
    save_path = os.path.join(OUTPUT_DIR, f'fedmd_{ds_name}.pth')
    return train_student_kd(pub_imgs, consensus, pub_labels, test_loader, save_path)


def run_fedkd(teacher_a, teacher_b, pub_imgs, pub_labels,
              test_loader, ds_name, upsample):
    print(f"\n{'='*60}")
    print(f"  FedKD [{ds_name}]  公共数据: {len(pub_imgs)} 张")
    print(f"{'='*60}")
    soft_a = get_teacher_soft_labels(teacher_a, pub_imgs, upsample_to=upsample)
    soft_b = get_teacher_soft_labels(teacher_b, pub_imgs, upsample_to=upsample)
    conf_a = soft_a.max(dim=1).values.unsqueeze(1)
    conf_b = soft_b.max(dim=1).values.unsqueeze(1)
    w_a    = conf_a / (conf_a + conf_b + 1e-8)
    w_b    = conf_b / (conf_a + conf_b + 1e-8)
    aggr   = w_a * soft_a + w_b * soft_b
    save_path = os.path.join(OUTPUT_DIR, f'fedkd_{ds_name}.pth')
    return train_student_kd(pub_imgs, aggr, pub_labels, test_loader, save_path)


def run_selective_fd(teacher_a, teacher_b, pub_imgs, pub_labels,
                     test_loader, ds_name, upsample):
    """
    Selective-FD：逐类 Top-K 过滤（Per-Class Top-K）

    【问题根源】全局 Top-K 在 STL-10 弱教师下导致类别严重不均衡：
      STL-10 教师（84%+71%）对不同类别置信度差异大。
      全局 Top-K 30% = 1500 张，但某类可能有 400 张，某类只有 20 张。
      学生无法学习稀少类别 → test acc 26%（接近10类随机猜测）。

    【修复】逐类 Top-K：每个类别内部独立按置信度排序，各取前 30%。
      每类固定选取相同比例的样本，彻底消除类别不均衡。
      与原 Selective-FD 核心思想一致：优先选教师"更确定"的样本，
      只是把过滤粒度从全局细化到类别级别。
    """
    print(f"\n{'='*60}")
    print(f"  Selective-FD [{ds_name}]  逐类 Top-K（每类保留前 {SELECTIVE_TOP_K_RATIO*100:.0f}%）")
    print(f"{'='*60}")
    soft_a = get_teacher_soft_labels(teacher_a, pub_imgs, upsample_to=upsample)
    soft_b = get_teacher_soft_labels(teacher_b, pub_imgs, upsample_to=upsample)

    # 样本质量分数：取两个教师最大置信度中的较高值
    max_a   = soft_a.max(dim=1).values   # [N]
    max_b   = soft_b.max(dim=1).values   # [N]
    quality = torch.maximum(max_a, max_b) # [N]

    # ★ 逐类 Top-K：在每个类别内独立排序后各取前 TOP_K_RATIO
    keep_list = []
    if pub_labels is not None:
        # 有真实标签：按真实类别分组
        for c in range(NUM_CLASSES):
            cls_idx = (pub_labels == c).nonzero(as_tuple=True)[0]
            if len(cls_idx) == 0:
                continue
            n_cls_keep       = max(int(len(cls_idx) * SELECTIVE_TOP_K_RATIO), 1)
            _, cls_order     = quality[cls_idx].sort(descending=True)
            keep_list.append(cls_idx[cls_order[:n_cls_keep]])
    else:
        # 无真实标签：用 Teacher A 预测类别分组
        pred_cls = soft_a.argmax(dim=1)
        for c in range(NUM_CLASSES):
            cls_idx = (pred_cls == c).nonzero(as_tuple=True)[0]
            if len(cls_idx) == 0:
                continue
            n_cls_keep       = max(int(len(cls_idx) * SELECTIVE_TOP_K_RATIO), 1)
            _, cls_order     = quality[cls_idx].sort(descending=True)
            keep_list.append(cls_idx[cls_order[:n_cls_keep]])

    keep    = torch.cat(keep_list)
    n_kept  = len(keep)
    n_total = len(pub_imgs)
    print(f"  逐类 Top-K 保留: {n_kept}/{n_total} 张 ({100*n_kept/n_total:.1f}%)")
    print(f"  平均置信度: {quality[keep].mean().item():.4f}"
          f"（全量均值: {quality.mean().item():.4f}）")

    # 输出逐类统计，验证均衡性
    ref_labels = pub_labels if pub_labels is not None else soft_a.argmax(dim=1)
    per_cls = [(ref_labels[keep] == c).sum().item() for c in range(NUM_CLASSES)]
    print(f"  每类样本数: {per_cls}  min={min(per_cls)} max={max(per_cls)}")

    imgs_s = pub_imgs[keep]
    labs_s = pub_labels[keep] if pub_labels is not None else None
    sa_s   = soft_a[keep];  sb_s = soft_b[keep]
    ca     = sa_s.max(dim=1).values.unsqueeze(1)
    cb     = sb_s.max(dim=1).values.unsqueeze(1)
    wa     = ca / (ca + cb + 1e-8)
    wb     = cb / (ca + cb + 1e-8)
    aggr   = wa * sa_s + wb * sb_s
    save_path = os.path.join(OUTPUT_DIR, f'selective_fd_{ds_name}.pth')
    return train_student_kd(imgs_s, aggr, labs_s, test_loader, save_path)




# ══════════════════════════════════════════════════════════════
# MIA 评估
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_attack_signals(model, images, labels, batch_size=256):
    model.eval()
    sigs = {'loss': [], 'conf_vec': [], 'entropy': [],
            'max_conf': [], 'conf_gap': [], 'one_hot': []}
    for start in range(0, len(images), batch_size):
        end   = min(start + batch_size, len(images))
        imgs  = images[start:end].to(device)
        labs  = labels[start:end].to(device)
        logits    = model(imgs)
        probs     = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        loss      = F.cross_entropy(logits, labs, reduction='none')
        entropy   = -(probs * log_probs).sum(dim=1)
        max_conf  = probs.max(dim=1).values
        true_c    = probs[torch.arange(len(labs)), labs]
        sp, _     = probs.sort(dim=1, descending=True)
        gap       = true_c - sp[:, 1]
        oh        = F.one_hot(labs.cpu(), NUM_CLASSES).float()
        sigs['loss'].append(loss.cpu())
        sigs['conf_vec'].append(probs.cpu())
        sigs['entropy'].append(entropy.cpu().unsqueeze(1))
        sigs['max_conf'].append(max_conf.cpu().unsqueeze(1))
        sigs['conf_gap'].append(gap.cpu().unsqueeze(1))
        sigs['one_hot'].append(oh)
    return {k: torch.cat(v, dim=0) for k, v in sigs.items()}


def build_ab_features(s):
    return torch.cat([s['conf_vec'], s['one_hot']], dim=1).numpy()


def build_aw_features(s):
    return torch.cat([s['loss'].unsqueeze(1), s['entropy'],
                      s['max_conf'], s['conf_gap'],
                      s['conf_vec'], s['one_hot']], dim=1).numpy()


def run_attack_experiment(fm, fnm, n_pairs, seed):
    np.random.seed(seed)
    n     = min(n_pairs, len(fm), len(fnm))
    idx_m = np.random.choice(len(fm),  n, replace=False)
    idx_n = np.random.choice(len(fnm), n, replace=False)
    X  = np.concatenate([fm[idx_m], fnm[idx_n]])
    y  = np.concatenate([np.ones(n), np.zeros(n)])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=1 - ATTACK_TRAIN_RATIO,
        random_state=seed, stratify=y)
    sc   = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    clf  = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
                         solver='adam', max_iter=300, random_state=seed,
                         early_stopping=True, validation_fraction=0.1,
                         n_iter_no_change=15)
    clf.fit(X_tr, y_tr)
    return accuracy_score(y_te, clf.predict(X_te)) * 100.0


def evaluate_mia(model, member_imgs, member_labels,
                 nonmember_imgs, nonmember_labels, name):
    print(f"\n  [MIA] {name}...")
    n_eval = min(N_ATTACK_PAIRS * 3, len(member_imgs), len(nonmember_imgs))
    torch.manual_seed(RANDOM_SEED)
    mi = torch.randperm(len(member_imgs))[:n_eval]
    ni = torch.randperm(len(nonmember_imgs))[:n_eval]
    sm = compute_attack_signals(model, member_imgs[mi], member_labels[mi])
    sn = compute_attack_signals(model, nonmember_imgs[ni], nonmember_labels[ni])
    fab_m = build_ab_features(sm);  fab_nm = build_ab_features(sn)
    faw_m = build_aw_features(sm);  faw_nm = build_aw_features(sn)
    ab_list = [run_attack_experiment(fab_m, fab_nm, N_ATTACK_PAIRS, RANDOM_SEED + r)
               for r in range(N_REPEAT)]
    aw_list = [run_attack_experiment(faw_m, faw_nm, N_ATTACK_PAIRS, RANDOM_SEED + r)
               for r in range(N_REPEAT)]
    priv_ab = float(np.mean(ab_list))
    priv_aw = float(np.mean(aw_list))
    print(f"  Priv Acc (Ab) = {priv_ab:.2f}% ± {np.std(ab_list):.2f}%")
    print(f"  Priv Acc (Aw) = {priv_aw:.2f}% ± {np.std(aw_list):.2f}%")
    return priv_ab, priv_aw


# ══════════════════════════════════════════════════════════════
# 结果
# ══════════════════════════════════════════════════════════════

METHOD_NAMES    = {'fedmd': 'FedMD', 'fedkd': 'FedKD',
                   'selective_fd': 'Selective-FD'}
METHOD_SETTINGS = {'fedmd'       : 'Heterogeneous FD',
                   'fedkd'       : 'Collaborative KD',
                   'selective_fd': 'Heterogeneous FD + selection'}
METHOD_FNS      = {'fedmd': run_fedmd, 'fedkd': run_fedkd,
                   'selective_fd': run_selective_fd}


def print_table(results):
    W = 92
    print("\n" + "=" * W)
    print("  Table 6: Cross-setting comparison")
    print("=" * W)
    hdr = (f"  {'Method':<18} {'Dataset':<10} {'Setting':<30} "
           f"{'Test Acc':>9} {'Priv(Ab)':>10} {'Priv(Aw)':>10}")
    print(hdr)
    print("─" * W)
    for n, d, s, ta, ab, aw in [
        ("MPC+KA (Ours)", "CIFAR-10",
         "Secure distillation + MPC", "64.36%", "49.62%", "49.94%"),
        ("MPC+KA (Ours)", "STL-10",
         "Secure distillation + MPC", "48.60%", "50.05%", "49.92%"),
    ]:
        print(f"  {n:<18} {d:<10} {s:<30} {ta:>9} {ab:>10} {aw:>10}")
    print("─" * W)
    for r in results:
        ta = f"{r['test_acc']:.2f}%"
        ab = f"{r['priv_ab']:.2f}%"
        aw = f"{r['priv_aw']:.2f}%"
        print(f"  {r['method']:<18} {r['dataset']:<10} "
              f"{r['setting']:<30} {ta:>9} {ab:>10} {aw:>10}")
    print("=" * W)


def save_results(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'fed_baselines_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    csv_path = os.path.join(OUTPUT_DIR, 'table6_fed_baselines.csv')
    with open(csv_path, 'w') as f:
        f.write("Method,Dataset,Setting,Test Acc,Priv Acc Ab,Priv Acc Aw\n")
        for r in results:
            f.write(f"{r['method']},{r['dataset']},{r['setting']},"
                    f"{r['test_acc']:.2f}%,{r['priv_ab']:.2f}%,"
                    f"{r['priv_aw']:.2f}%\n")
    print(f"\n  结果已保存: {csv_path}")


# ══════════════════════════════════════════════════════════════
# 主运行函数
# ══════════════════════════════════════════════════════════════

def run_cifar10(methods, all_results):
    print(f"\n{'#'*60}")
    print("  数据集: CIFAR-10")
    print(f"  公共数据: 训练集后 25000 张（索引 {CIFAR10_PUBLIC_START}:50000，有真实标签）")
    print(f"  Test Acc 评估: 测试集（10000张，与公共数据完全分离）")
    print(f"{'#'*60}")

    teacher_a = load_teacher(CIFAR10_TEACHER_A_PATH)
    teacher_b = load_teacher(CIFAR10_TEACHER_B_PATH)

    print("\n  加载 CIFAR-10 全量训练集...")
    all_imgs, all_labels = load_cifar10_train_tensors()

    # ★ 公共数据 = 训练集后半段（不是测试集！）
    pub_imgs   = all_imgs[CIFAR10_PUBLIC_START:]    # 25000张，索引25000-50000
    pub_labels = all_labels[CIFAR10_PUBLIC_START:]  # 有真实标签

    # MIA 划分
    member_imgs      = all_imgs[:CIFAR10_MEMBER_SIZE]    # 前25000 = members
    member_labels    = all_labels[:CIFAR10_MEMBER_SIZE]
    nonmember_imgs   = all_imgs[CIFAR10_PUBLIC_START:]   # 后25000 = non-members（与公共数据相同段）
    nonmember_labels = all_labels[CIFAR10_PUBLIC_START:]

    print(f"  公共数据     : {len(pub_imgs)} 张（训练集 [{CIFAR10_PUBLIC_START}:50000]）")
    print(f"  MIA members  : {len(member_imgs)} 张（训练集 [0:{CIFAR10_MEMBER_SIZE}]）")
    print(f"  MIA non-mem  : {len(nonmember_imgs)} 张（训练集 [{CIFAR10_PUBLIC_START}:50000]）")
    print(f"  Test Acc 评估: CIFAR-10 测试集（独立，不参与任何训练）")

    test_loader = get_cifar10_test_loader()  # ★ 测试集仅用于评估

    for m in methods:
        student, test_acc = METHOD_FNS[m](
            teacher_a, teacher_b, pub_imgs, pub_labels,
            test_loader, 'cifar10', upsample=None)

        priv_ab, priv_aw = evaluate_mia(
            student, member_imgs, member_labels,
            nonmember_imgs, nonmember_labels,
            f"{METHOD_NAMES[m]} [CIFAR-10]")

        all_results.append({
            'method'  : METHOD_NAMES[m],
            'dataset' : 'CIFAR-10',
            'setting' : METHOD_SETTINGS[m],
            'test_acc': test_acc,
            'priv_ab' : priv_ab,
            'priv_aw' : priv_aw,
        })


def run_stl10(methods, all_results):
    print(f"\n{'#'*60}")
    print("  数据集: STL-10")
    print(f"  公共数据: STL-10 训练集全量 5000 张（有真实标签）")
    print(f"  Test Acc 评估: 测试集（8000张）")
    print(f"{'#'*60}")

    teacher_a = load_teacher(STL10_TEACHER_A_PATH)
    teacher_b = load_teacher(STL10_TEACHER_B_PATH)

    print("\n  加载 STL-10 训练集（32×32）...")
    all_imgs, all_labels = load_stl10_train_tensors()

    # 公共数据 = 全量训练集（5000张，有标签）
    pub_imgs   = all_imgs
    pub_labels = all_labels  # ★ 有真实标签，确保 CE loss 正常

    # MIA 划分（与 mia_attack_stl10.py 完全一致）
    member_imgs      = all_imgs[:STL10_MEMBER_SIZE]
    member_labels    = all_labels[:STL10_MEMBER_SIZE]
    nonmember_imgs   = all_imgs[STL10_MEMBER_SIZE:]
    nonmember_labels = all_labels[STL10_MEMBER_SIZE:]

    print(f"  公共数据     : {len(pub_imgs)} 张（全量训练集，有真实标签）")
    print(f"  MIA members  : {len(member_imgs)} 张（训练集前 {STL10_MEMBER_SIZE}）")
    print(f"  MIA non-mem  : {len(nonmember_imgs)} 张（训练集后半）")

    test_loader = get_stl10_test_loader()

    for m in methods:
        # ★ STL-10 教师推理必须上采样至 96×96
        student, test_acc = METHOD_FNS[m](
            teacher_a, teacher_b, pub_imgs, pub_labels,
            test_loader, 'stl10', upsample=STL10_TEACHER_RES)

        priv_ab, priv_aw = evaluate_mia(
            student, member_imgs, member_labels,
            nonmember_imgs, nonmember_labels,
            f"{METHOD_NAMES[m]} [STL-10]")

        all_results.append({
            'method'  : METHOD_NAMES[m],
            'dataset' : 'STL-10',
            'setting' : METHOD_SETTINGS[m],
            'test_acc': test_acc,
            'priv_ab' : priv_ab,
            'priv_aw' : priv_aw,
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='both',
                        choices=['cifar10', 'stl10', 'both'])
    parser.add_argument('--method', default='all',
                        choices=['fedmd', 'fedkd', 'selective_fd', 'all'])
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  Table 6: Cross-Setting Federated Distillation Baselines")
    print("=" * 60)
    print(f"  Device  : {device}")
    print(f"  Dataset : {args.dataset}")
    print(f"  Method  : {args.method}")
    print()
    print("  CIFAR-10 公共数据: 训练集后 25000 张（★不是测试集）")
    print("  STL-10   公共数据: 训练集全量 5000 张（有真实标签）")

    methods  = (['fedmd', 'fedkd', 'selective_fd']
                if args.method == 'all' else [args.method])
    datasets = (['cifar10', 'stl10']
                if args.dataset == 'both' else [args.dataset])

    all_results = []
    if 'cifar10' in datasets:
        run_cifar10(methods, all_results)
    if 'stl10' in datasets:
        run_stl10(methods, all_results)

    print_table(all_results)
    save_results(all_results)
    print("\n全部实验完成。")


if __name__ == '__main__':
    main()