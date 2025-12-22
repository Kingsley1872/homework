"""
UIFace-Plus: 身份插值实验

本脚本实现了在不同身份嵌入之间的插值，生成平滑过渡的人脸序列。
主要功能：
1. 支持球面线性插值 (SLERP) 和线性插值 (LERP)
2. 在多个身份之间生成插值序列
3. 可视化插值结果

技术要点：
- SLERP 在高维空间中保持更好的几何性质
- 适用于探索潜在空间的连续性和平滑性

作者：复旦大学 生成模型课程
日期：2024年12月
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# 添加路径
generation_dir = os.path.join(os.path.dirname(__file__), '..', 'generation')
uiface_dir = os.path.join(generation_dir, 'uiface')
sys.path.insert(0, uiface_dir)

from generate_with_visualization import UIFacePlusGenerator


class IdentityInterpolator:
    """身份插值器"""

    def __init__(self, generator):
        self.generator = generator

    @staticmethod
    def slerp(v0, v1, t):
        """
        球面线性插值 (Spherical Linear Interpolation)
        比线性插值更适合在高维空间中插值
        """
        v0 = v0 / np.linalg.norm(v0)
        v1 = v1 / np.linalg.norm(v1)

        dot = np.sum(v0 * v1)
        dot = np.clip(dot, -1.0, 1.0)

        theta = np.arccos(dot)
        sin_theta = np.sin(theta)

        if sin_theta < 1e-6:
            # 如果两个向量几乎相同，使用线性插值
            return (1.0 - t) * v0 + t * v1

        s0 = np.sin((1.0 - t) * theta) / sin_theta
        s1 = np.sin(t * theta) / sin_theta

        return s0 * v0 + s1 * v1

    def interpolate_identities(self, identity1, identity2, num_steps=10, use_slerp=True):
        """
        在两个身份之间插值

        Args:
            identity1: 第一个身份嵌入 (512维向量)
            identity2: 第二个身份嵌入 (512维向量)
            num_steps: 插值步数
            use_slerp: 是否使用球面插值

        Returns:
            interpolated_images: 插值后的图像列表
        """
        interpolated_images = []

        print(f"Interpolating between two identities with {num_steps} steps...")

        for i in range(num_steps):
            t = i / (num_steps - 1)

            # 插值
            if use_slerp:
                interpolated_embedding = self.slerp(identity1, identity2, t)
            else:
                # 线性插值
                interpolated_embedding = (1 - t) * identity1 + t * identity2

            # 确保数据类型为 float32（修复 Double/Float 类型不匹配问题）
            interpolated_embedding = interpolated_embedding.astype(np.float32)

            # 生成人脸
            final_image, _ = self.generator.generate_with_steps(
                identity_embedding=interpolated_embedding,
                num_steps=50,
                save_intermediate=False,
                cfg_scale=1.5
            )

            interpolated_images.append(final_image)
            print(f"  Generated image {i+1}/{num_steps} (t={t:.2f})")

        return interpolated_images

    def visualize_interpolation(self, images, save_path, title="Identity Interpolation"):
        """可视化插值结果"""
        num_images = len(images)
        cols = min(num_images, 10)
        rows = (num_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))

        if rows == 1:
            axes = axes.reshape(1, -1)

        for i, img in enumerate(images):
            row = i // cols
            col = i % cols
            axes[row, col].imshow(img)
            axes[row, col].set_title(f't={i/(num_images-1):.2f}', fontsize=10)
            axes[row, col].axis('off')

        # 隐藏多余的子图
        for i in range(num_images, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')

        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Interpolation visualization saved to {save_path}")

    def create_interpolation_grid(self, num_identities=4, num_steps=8):
        """
        创建多个身份之间的插值网格

        Args:
            num_identities: 身份数量
            num_steps: 每对身份之间的插值步数
        """
        # 生成随机身份嵌入（确保是 float32 类型）
        identities = [np.random.randn(512).astype(np.float32) for _ in range(num_identities)]

        output_dir = Path('outputs/interpolation')
        output_dir.mkdir(parents=True, exist_ok=True)

        # 对每对相邻的身份进行插值
        for i in range(num_identities - 1):
            print(f"\n=== Interpolating between identity {i+1} and {i+2} ===")

            interpolated_images = self.interpolate_identities(
                identities[i],
                identities[i+1],
                num_steps=num_steps,
                use_slerp=True
            )

            # 保存可视化
            self.visualize_interpolation(
                interpolated_images,
                output_dir / f'interpolation_{i+1}_to_{i+2}.png',
                title=f'Identity {i+1} → Identity {i+2}'
            )

            # 保存单独的图像
            for j, img in enumerate(interpolated_images):
                img.save(output_dir / f'id{i+1}_to_id{i+2}_step{j}.png')

        print(f"\n✅ Interpolation completed!")
        print(f"📁 Check outputs in: {output_dir.absolute()}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Identity Interpolation Experiment')
    parser.add_argument('--config', type=str,
                        default='./generation/uiface/configs/model/unet_cond_ca_cpd25_uncond20.yaml',
                        help='Model config path')
    parser.add_argument('--checkpoint', type=str,
                        default='UIFace-Plus/models/ema_averaged_model_250000.ckpt',
                        help='UIFace checkpoint path')
    parser.add_argument('--vq_encoder', type=str,
                        default='UIFace-Plus/models/first_stage_encoder_state_dict.pt',
                        help='VQ encoder path')
    parser.add_argument('--vq_decoder', type=str,
                        default='UIFace-Plus/models/first_stage_decoder_state_dict.pt',
                        help='VQ decoder path')
    parser.add_argument('--num_identities', type=int, default=4,
                        help='Number of identities to generate')
    parser.add_argument('--num_steps', type=int, default=8,
                        help='Interpolation steps between each pair')

    args = parser.parse_args()

    # 创建生成器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    try:
        generator = UIFacePlusGenerator(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            vq_encoder_path=args.vq_encoder,
            vq_decoder_path=args.vq_decoder,
            device=device
        )

        # 创建插值器
        interpolator = IdentityInterpolator(generator)

        # 执行插值实验
        interpolator.create_interpolation_grid(
            num_identities=args.num_identities,
            num_steps=args.num_steps
        )

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
