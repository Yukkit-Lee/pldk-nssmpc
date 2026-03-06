import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import torchvision.transforms as transforms

# ================= 配置区域 =================
# CIFAR-10 类别名称
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# CIFAR-10 的标准均值和标准差 (用于反归一化)
# 这些值来自你生成数据代码中使用的 get_dataset 默认设定
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]

# 设置要展示的列数（每类展示多少张图）
# 你有 IPC=50，这里展示前 10 张，刚好凑成 10x10 的方格
SAMPLES_PER_CLASS_TO_SHOW = 10 
# ===========================================


def load_distilled_data():
    """自动查找并加载 images_best.pt"""
    print("Searching for 'images_best.pt'...")
    files = glob.glob("**/images_best.pt", recursive=True)
    if not files:
        raise FileNotFoundError("Could not find 'images_best.pt' in current directory or subdirectories.")
    
    path = files[0]
    print(f"Found file at: {path}")
    
    # 加载到 CPU
    images = torch.load(path, map_location=torch.device('cpu'))
    print(f"Data shape loaded: {images.shape}") # 预期应该是 [500, 3, 32, 32]
    
    # 简单验证
    total_expected = 10 * 50 # 10类 * IPC50
    if images.shape[0] != total_expected:
        print(f"Warning: Expected {total_expected} images, but found {images.shape[0]}.")
        
    return images

def denormalize(tensor):
    """
    将标准化的 Tensor 反转回可视化的 [0, 1] 范围。
    x_norm = (x - mean) / std  =>  x = x_norm * std + mean
    """
    # 克隆 tensor 防止修改原始数据
    img = tensor.clone()
    
    # 定义反归一化变换
    # 注意：这里是先乘标准差，再加均值
    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)],
        std=[1/s for s in CIFAR10_STD]
    )
    img = inv_normalize(img)
    
    # 将数据限制在 [0, 1] 范围内，防止噪点过曝
    img = torch.clamp(img, 0, 1)
    return img

def show_images_grid(images, rows=10, cols=10, ipc_total=50):
    """
    创建一个网格图来展示图片。
    rows: 类别数 (CIFAR10 是 10)
    cols: 每类展示几张
    ipc_total: 数据中实际的 IPC 数 (你的是 50)
    """
    fig, axes = plt.subplots(rows, cols, figsize=(cols, rows * 1.2))
    fig.suptitle(f'Distilled Images (IPC={ipc_total}, Showing top {cols})', fontsize=16, y=0.95)

    # 进行反归一化处理
    print("Denormalizing images for visualization...")
    images_denorm = denormalize(images)

    for c in range(rows): # 遍历 10 个类别
        for i in range(cols): # 遍历每类的前 cols 张图
            ax = axes[c, i]
            
            # 计算图像在 huge tensor 中的全局索引
            # 数据是按顺序排列的：前50张是类0，接下来50张是类1...
            idx = c * ipc_total + i
            
            img_tensor = images_denorm[idx]
            
            # 将 PyTorch [C, H, W] 转换为 Matplotlib 需要的 [H, W, C] 格式
            img_np = img_tensor.permute(1, 2, 0).numpy()
            
            ax.imshow(img_np)
            ax.axis('off') # 关闭坐标轴
            
            # 在每行的第一列添加类别名称标签
            if i == 0:
                ax.set_title(CLASS_NAMES[c], x=-0.2, y=0.4, ha='right', va='center', fontsize=12)

    plt.tight_layout(rect=[0.02, 0, 1, 0.93]) # 调整布局留出标题空间
    
    # 保存图像
    save_path = "distilled_images_grid.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSuccessfully saved visualization to: {save_path}")
    print("Done! Please check the generated PNG file.")
    
    # 如果在支持图形界面的环境下，可以取消下面这行的注释直接显示
    # plt.show()

if __name__ == "__main__":
    try:
        images_tensor = load_distilled_data()
        # 我们展示 10个类别，每类展示前 10 张，你的实际 IPC 是 50
        show_images_grid(images_tensor, rows=10, cols=SAMPLES_PER_CLASS_TO_SHOW, ipc_total=50)
    except Exception as e:
        print(f"An error occurred: {e}")