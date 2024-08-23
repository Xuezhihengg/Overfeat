import torch
import matplotlib.pyplot as plt


def plot_single_image(image_tensor, title='Single Image'):
    """
    绘制单个图像张量。

    Args:
        image_tensor (Tensor): 图像张量，形状为 (3, 256, 256)。
        title (str): 绘图的标题。
    """
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("image_tensor must be a torch.Tensor")

    if len(image_tensor.shape) != 3 or image_tensor.shape[0] != 3:
        raise ValueError("image_tensor must have shape (3, 256, 256)")

    # 确保图像张量在第一个通道上是C-contiguous
    image_tensor = image_tensor.contiguous()

    # 创建图形和子图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 绘制每个颜色通道的图像
    for i, ax in enumerate(axes):
        ax.imshow(image_tensor[i].cpu().numpy(), cmap='gray')  # 使用灰度色图显示
        ax.axis('off')  # 不显示坐标轴
        ax.set_title(f'Channel {i + 1}')  # 设置通道标题

    plt.tight_layout()
    plt.suptitle(title)  # 设置主标题
    plt.subplots_adjust(top=0.9)  # 调整标题位置
    plt.show()