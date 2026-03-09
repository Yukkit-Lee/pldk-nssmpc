"""
visualize_mpc.py
================
可视化 PLDK + NssMPC 数据管线的三个阶段：

  图1  蒸馏数据网格（10类 × 10张，对标论文 Fig.3(b)）
  图2  秘密分享视角（原图 / Party0 / Party1，模拟两家医院的视图）
  图3  MPC重构对比（原图 vs 重构图，两行）

只需修改 ACTIVE_DATASET 即可切换数据集。
依赖：torch, torchvision, matplotlib, numpy
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms

# ╔══════════════════════════════════════════════════════════╗
# ║               用户配置区（只需改这里）                    ║
# ╚══════════════════════════════════════════════════════════╝

# 切换数据集：'cifar10' 或 'stl10'
ACTIVE_DATASET = 'cifar10'

# 数据集参数库
DATASET_CONFIGS = {
    'cifar10': {
        'mean'        : (0.4914, 0.4822, 0.4465),
        'std'         : (0.2023, 0.1994, 0.2010),
        'images_path' : 'images_best_cifar10.pt',
        'labels_path' : None,          # None 则根据 MTT 排列推断
        'ipc'         : 50,            # 每类蒸馏图片数
        'num_classes' : 10,
        'distill_res' : 32,
        'class_names' : [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck',
        ],
        'tag'         : 'CIFAR-10',
    },
    'stl10': {
        'mean'        : (0.4467, 0.4398, 0.4066),
        'std'         : (0.2603, 0.2566, 0.2713),
        'images_path' : 'images_best_stl10.pt',
        'labels_path' : None,
        'ipc'         : 50,
        'num_classes' : 10,
        'distill_res' : 32,
        'class_names' : [
            'airplane', 'bird', 'car', 'deer', 'dog',
            'horse', 'monkey', 'ship', 'truck', 'frog',
        ],
        'tag'         : 'STL-10',
    },
}

# 可视化全局参数
COLS_PER_CLASS    = 10      # 图1每类显示列数
N_MPC_SHOW        = 8       # 图2、图3显示的图片数量
FIXED_POINT_SCALE = 2**16   # 定点编码精度（与 3pc_generic.py 保持一致）
OUTPUT_DIR        = 'comparisonFig'   # 输出目录
DPI               = 180               # 输出分辨率


# ══════════════════════════════════════════════════════════
# MPC 辅助函数（完全复现 3pc_generic.py 的加密/解密逻辑）
# ══════════════════════════════════════════════════════════

def float_to_ring(t: torch.Tensor) -> torch.Tensor:
    """浮点数 → 定点整数编码（×2^16，取整，转 int64）"""
    return (t * FIXED_POINT_SCALE).round().to(torch.int64)


def additive_secret_share(ring: torch.Tensor):
    """
    加法秘密共享（2-out-of-2，Z_{2^64} 整数环）
    share0：均匀随机 int64
    share1 = ring - share0（mod 2^64，int64 自动 wrapping）
    单独一份在信息论意义上与均匀随机无区别。
    """
    share0 = torch.randint(
        low  = -(2**63),
        high =   2**63 - 1,
        size = ring.shape,
        dtype= torch.int64,
    )
    share1 = ring - share0   # int64 自动 mod 2^64
    return share0, share1


def ring_to_float(s0: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
    """两份 share 相加后除以 2^16，还原浮点数（对应 3pc_generic ring_to_float）"""
    return (s0 + s1).to(torch.float32) / FIXED_POINT_SCALE


# ══════════════════════════════════════════════════════════
# 图像处理工具
# ══════════════════════════════════════════════════════════

def denormalize(imgs: torch.Tensor, cfg: dict) -> torch.Tensor:
    """反归一化：normalized tensor → [0, 1] 像素范围"""
    mean = torch.tensor(cfg['mean'], dtype=torch.float32).view(3, 1, 1)
    std  = torch.tensor(cfg['std'],  dtype=torch.float32).view(3, 1, 1)
    return (imgs * std + mean).clamp(0.0, 1.0)


def to_hwc(t: torch.Tensor) -> np.ndarray:
    """[3, H, W] float tensor → (H, W, 3) numpy 数组"""
    return t.permute(1, 2, 0).numpy().astype(np.float32)


def share_to_display(share: torch.Tensor) -> torch.Tensor:
    """
    将 int64 份额映射到 [0,1] 供显示。
    逐图 min-max 归一化，展现份额的噪声纹理。
    """
    s  = share.to(torch.float64)
    mn = s.flatten(1).min(1).values.view(-1, 1, 1, 1)
    mx = s.flatten(1).max(1).values.view(-1, 1, 1, 1)
    return ((s - mn) / (mx - mn + 1.0)).clamp(0.0, 1.0).to(torch.float32)


# ══════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════

def load_distilled_data(cfg: dict):
    """
    加载蒸馏图像文件，必要时补归一化，并推断/加载类别标签。
    返回：images_norm [N,3,H,W]，labels [N]（均在 CPU）
    """
    path = cfg['images_path']
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到蒸馏数据文件: '{path}'\n"
            f"请在 DATASET_CONFIGS['{ACTIVE_DATASET}']['images_path'] 中设置正确路径。"
        )

    imgs = torch.load(path, map_location='cpu')
    if imgs.dtype != torch.float32:
        imgs = imgs.float()

    print(f"  加载文件 : {path}")
    print(f"  数据形状 : {imgs.shape}")
    print(f"  值域     : [{imgs.min():.4f}, {imgs.max():.4f}]")

    # 若数据在 [0,1] 范围内则尚未归一化
    if imgs.min() >= 0.0 and imgs.max() <= 1.0:
        print("  检测到未归一化数据，应用数据集归一化...")
        imgs = transforms.Normalize(cfg['mean'], cfg['std'])(imgs)
        print(f"  归一化后值域 : [{imgs.min():.4f}, {imgs.max():.4f}]")
    else:
        print("  数据已归一化，跳过。")

    # 加载或推断标签
    lbl_path = cfg.get('labels_path')
    if lbl_path and os.path.exists(lbl_path):
        labels = torch.load(lbl_path, map_location='cpu').long()
        print(f"  标签来源 : {lbl_path}")
    else:
        # MTT 默认布局：[class0×ipc, class1×ipc, ...]
        ipc = cfg['ipc']
        nc  = cfg['num_classes']
        labels = torch.arange(nc).repeat_interleave(ipc)
        print(f"  标签推断 : MTT 默认排列（ipc={ipc}, classes={nc}）")

    return imgs, labels


# ══════════════════════════════════════════════════════════
# 图1：蒸馏数据网格（对标论文 Fig.3(b)）
# ══════════════════════════════════════════════════════════

def make_figure1(images_norm, labels, cfg):
    """
    绘制 num_classes × COLS_PER_CLASS 的蒸馏数据网格。
    每行一类，左侧显示类名（不加粗），图片间距紧凑，
    对标 PLDK 论文 Fig.3(b) 风格。
    """
    nc    = cfg['num_classes']
    cols  = COLS_PER_CLASS
    names = cfg['class_names']
    tag   = cfg['tag']
    res   = images_norm.shape[-1]

    # 按类别收集 cols 张图片
    grid_imgs = []
    for c in range(nc):
        idx  = (labels == c).nonzero(as_tuple=True)[0]
        sel_idx = torch.randperm(len(idx))[:cols] if len(idx) >= cols \
                  else torch.arange(len(idx))
        sel  = denormalize(images_norm[idx[sel_idx]], cfg)
        row  = [to_hwc(sel[i]) for i in range(len(sel))]
        # 图片不足时用灰色填充
        blank = np.full((res, res, 3), 0.88, dtype=np.float32)
        row  += [blank] * (cols - len(row))
        grid_imgs.append(row)

    # ── 布局参数（紧凑风格，与 Fig.3(b) 一致）──
    LABEL_W  = 0.80    # 类名列宽度（英寸）
    CELL     = 0.33    # 每个图片单元格的尺寸（英寸）
    TOP_PAD  = 0.40    # 标题区高度（英寸）
    BOT_PAD  = 0.12
    IMG_GAP  = 0.012   # 图片间空隙（单元格比例）

    fig_w = LABEL_W + cols * CELL + 0.12
    fig_h = TOP_PAD + nc * CELL + BOT_PAD

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')

    # 标题（仅保留数据集名称，无副标题）
    fig.text(
        0.5, 1.0 - 0.02 / fig_h,
        f"Distilled Dataset — {tag}",
        ha='center', va='top',
        fontsize=11, fontweight='bold', color='#1A1A2E',
    )

    # 计算坐标系比例
    lf = LABEL_W / fig_w          # 图片区域左边界（比例）
    tf = TOP_PAD / fig_h          # 标题区占用高度（比例）
    plot_h = 1.0 - tf - BOT_PAD / fig_h
    row_h  = plot_h / nc
    col_w  = (1.0 - lf - 0.01) / cols

    for row in range(nc):
        # 左侧类名标签（不加粗）
        y_center = 1.0 - tf - (row + 0.5) * row_h
        fig.text(
            lf - 0.008, y_center,
            names[row],
            ha='right', va='center',
            fontsize=7.5, color='#222222', fontweight='normal',   # 不加粗
        )

        for col in range(cols):
            # 计算每个 axes 的位置（紧凑间距）
            l = lf + col * col_w + col_w * IMG_GAP / 2
            b = 1.0 - tf - (row + 1) * row_h + row_h * IMG_GAP / 2
            w = col_w * (1 - IMG_GAP)
            h = row_h * (1 - IMG_GAP)

            ax = fig.add_axes([l, b, w, h])
            ax.imshow(grid_imgs[row][col], interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            # 细边框增加整洁感
            for sp in ax.spines.values():
                sp.set_edgecolor('#BBBBBB'); sp.set_linewidth(0.3)

    # 类名列与图片区分隔线
    # fig.add_artist(plt.Line2D(
    #     [lf - 0.004, lf - 0.004],
    #     [BOT_PAD / fig_h, 1.0 - tf + 0.01],
    #     transform=fig.transFigure,
    #     color='#CCCCCC', linewidth=0.6,
    # ))

    # 创建输出目录并保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fname = f'figure1_distilled_{tag.lower().replace("-","")}.png'
    out   = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  已保存 → {out}")


# ══════════════════════════════════════════════════════════
# 图2：秘密分享视角（原图 / Party 0 / Party 1）
# ══════════════════════════════════════════════════════════

def make_figure2(images_norm, share0, share1, cfg):
    """
    三行布局：
      第1行 原始蒸馏图像
      第2行 Hospital A (Party 0) 持有的份额 → 呈随机噪声
      第3行 Hospital B (Party 1) 持有的份额 → 呈随机噪声
    不显示任何坐标轴刻度/表格，不显示直方图面板。
    """
    tag = cfg['tag']
    n   = min(N_MPC_SHOW, len(images_norm))

    # 准备三行数据
    orig_01   = denormalize(images_norm[:n], cfg)  # 反归一化到 [0,1]
    share0_01 = share_to_display(share0[:n])        # Party 0 份额可视化
    share1_01 = share_to_display(share1[:n])        # Party 1 份额可视化

    row_data   = [orig_01, share0_01, share1_01]
    row_labels = [
        'Original',
        'Client A (Party 0)',
        'Client B (Party 1)',
    ]

    # 每个格子尺寸紧凑
    CELL     = 1.05    # 英寸
    LABEL_W  = 1.45    # 左侧行标签列宽度
    TOP_PAD  = 0.55    # 标题区高度
    BOT_PAD  = 0.15

    fig_w = LABEL_W + n * CELL + 0.15
    fig_h = TOP_PAD + 3 * CELL + BOT_PAD

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')

    # 标题
    fig.text(
        0.5, 1.0 - 0.02 / fig_h,
        f"Figure 2 — Additive Secret Share: Client A & B Views  [{tag}]",
        ha='center', va='top',
        fontsize=10, fontweight='bold', color='#1A1A2E',
    )

    # 布局比例
    lf     = LABEL_W / fig_w
    tf     = TOP_PAD / fig_h
    row_h  = (1.0 - tf - BOT_PAD / fig_h) / 3
    col_w  = (1.0 - lf - 0.01) / n
    GAP    = 0.012

    for row in range(3):
        # 行标签（左侧）
        y_center = 1.0 - tf - (row + 0.5) * row_h
        fig.text(
            lf - 0.010, y_center,
            row_labels[row],
            ha='right', va='center',
            fontsize=8.5, color='#1A1A2E',
        )

        for col in range(n):
            l = lf + col * col_w + col_w * GAP / 2
            b = 1.0 - tf - (row + 1) * row_h + row_h * GAP / 2
            w = col_w * (1 - GAP)
            h = row_h * (1 - GAP)

            ax = fig.add_axes([l, b, w, h])
            ax.imshow(to_hwc(row_data[row][col]), interpolation='nearest')
            # 完全移除坐标轴刻度和边框表格
            ax.set_xticks([]); ax.set_yticks([])
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for sp in ax.spines.values():
                sp.set_edgecolor('#BBBBBB'); sp.set_linewidth(0.3)

    # # 注释文字
    # fig.text(
    #     0.5, BOT_PAD / fig_h - 0.01,
    #     "Each share is uniformly random — "
    #     "information-theoretically secure without the other share",
    #     ha='center', va='bottom',
    #     fontsize=7.5, color='#666666', style='italic',
    # )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fname = f'figure2_secret_share_{tag.lower().replace("-","")}.png'
    out   = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  已保存 → {out}")


# ══════════════════════════════════════════════════════════
# 图3：MPC 重构对比（原图 vs 重构图，两行）
# ══════════════════════════════════════════════════════════

def make_figure3(images_norm, recon_norm, cfg, max_err):
    """
    两行对比：
      第1行 原始蒸馏图像（反归一化）
      第2行 MPC 重构后的图像（反归一化）
    不显示坐标轴刻度/表格，不显示误差行。
    """
    tag = cfg['tag']
    n   = min(N_MPC_SHOW, len(images_norm))

    orig_01  = denormalize(images_norm[:n], cfg)
    recon_01 = denormalize(recon_norm[:n],  cfg)

    row_data   = [orig_01, recon_01]
    row_labels = ['Original', 'Reconstructed']

    CELL    = 1.05
    LABEL_W = 1.20
    TOP_PAD = 0.58
    BOT_PAD = 0.15

    fig_w = LABEL_W + n * CELL + 0.15
    fig_h = TOP_PAD + 2 * CELL + BOT_PAD

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')

    # 标题带误差统计
    fig.text(
        0.5, 1.0 - 0.02 / fig_h,
        f"Figure 3 — MPC Reconstruction Fidelity  [{tag}]",
        ha='center', va='top',
        fontsize=10, fontweight='bold', color='#1A1A2E',
    )
    # fig.text(
    #     0.5, 1.0 - 0.32 / fig_h,
    #     f"Fixed-point scale $2^{{16}}$  |  "
    #     f"Max pixel error = {max_err:.2e}  |  "
    #     f"Theoretical bound = {1/FIXED_POINT_SCALE:.2e}",
    #     ha='center', va='top',
    #     fontsize=8, color='#555555',
    # )

    # 布局比例
    lf    = LABEL_W / fig_w
    tf    = TOP_PAD / fig_h
    row_h = (1.0 - tf - BOT_PAD / fig_h) / 2
    col_w = (1.0 - lf - 0.01) / n
    GAP   = 0.012

    for row in range(2):
        # 行标签
        y_center = 1.0 - tf - (row + 0.5) * row_h
        fig.text(
            lf - 0.010, y_center,
            row_labels[row],
            ha='right', va='center',
            fontsize=8.5, color='#1A1A2E',
        )

        for col in range(n):
            l = lf + col * col_w + col_w * GAP / 2
            b = 1.0 - tf - (row + 1) * row_h + row_h * GAP / 2
            w = col_w * (1 - GAP)
            h = row_h * (1 - GAP)

            ax = fig.add_axes([l, b, w, h])
            ax.imshow(to_hwc(row_data[row][col]), interpolation='nearest')
            # 完全移除坐标轴刻度和边框表格
            ax.set_xticks([]); ax.set_yticks([])
            ax.tick_params(left=False, bottom=False,
                           labelleft=False, labelbottom=False)
            for sp in ax.spines.values():
                sp.set_edgecolor('#BBBBBB'); sp.set_linewidth(0.3)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fname = f'figure3_reconstructed_{tag.lower().replace("-","")}.png'
    out   = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(out, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  已保存 → {out}")


# ══════════════════════════════════════════════════════════
# 主程序入口
# ══════════════════════════════════════════════════════════

def main():
    # 验证数据集配置键
    if ACTIVE_DATASET not in DATASET_CONFIGS:
        raise ValueError(
            f"ACTIVE_DATASET='{ACTIVE_DATASET}' 不在配置中。\n"
            f"可用选项: {list(DATASET_CONFIGS.keys())}"
        )
    cfg  = DATASET_CONFIGS[ACTIVE_DATASET]
    tag  = cfg['tag']
    ftag = tag.lower().replace('-', '')

    print("=" * 60)
    print(f"  PLDK + NssMPC  ——  数据可视化管线")
    print(f"  数据集     : {tag}")
    print(f"  IPC        : {cfg['ipc']}   类别数 : {cfg['num_classes']}")
    print(f"  蒸馏分辨率 : {cfg['distill_res']}×{cfg['distill_res']}")
    print(f"  定点精度   : 2^16 = {FIXED_POINT_SCALE}")
    print(f"  输出目录   : {OUTPUT_DIR}/")
    print("=" * 60)

    # [1] 加载蒸馏数据
    print("\n[1/5] 加载蒸馏数据...")
    images_norm, labels = load_distilled_data(cfg)

    # [2] 模拟 MPC 加密/重构流程
    print("\n[2/5] 模拟 MPC 加密/重构流程...")
    show = images_norm[:N_MPC_SHOW].clone()

    ring           = float_to_ring(show)          # 定点编码
    share0, share1 = additive_secret_share(ring)  # 加法秘密共享
    recon          = ring_to_float(share0, share1) # 重构浮点数

    max_err = (recon - show).abs().max().item()
    print(f"  int64 值域   : [{ring.min().item()}, {ring.max().item()}]")
    print(f"  最大重构误差 : {max_err:.2e}（理论上限 {1/FIXED_POINT_SCALE:.2e}）")
    assert max_err <= 1.0 / FIXED_POINT_SCALE + 1e-9, \
        "错误：重构误差超出定点精度上限，请检查 FIXED_POINT_SCALE！"
    print("  ✓ 重构精度验证通过")

    # [3] 图1：蒸馏数据网格
    print(f"\n[3/5] 生成图1：蒸馏数据网格（{cfg['num_classes']}×{COLS_PER_CLASS}）...")
    make_figure1(images_norm, labels, cfg)

    # [4] 图2：秘密分享视角
    print("\n[4/5] 生成图2：秘密分享视角（原图 / Hospital A / Hospital B）...")
    make_figure2(images_norm, share0, share1, cfg)

    # [5] 图3：MPC 重构对比
    print("\n[5/5] 生成图3：MPC 重构对比（原图 vs 重构图）...")
    make_figure3(images_norm, recon, cfg, max_err)

    # 输出摘要
    print("\n" + "=" * 60)
    print(f"  全部图片已保存至 ./{OUTPUT_DIR}/")
    print(f"    figure1_distilled_{ftag}.png")
    print(f"    figure2_secret_share_{ftag}.png")
    print(f"    figure3_reconstructed_{ftag}.png")
    print(f"\n  最大重构误差 : {max_err:.2e}")
    print(f"  定点精度上限 : {1/FIXED_POINT_SCALE:.2e}")
    print("=" * 60)


if __name__ == '__main__':
    main()